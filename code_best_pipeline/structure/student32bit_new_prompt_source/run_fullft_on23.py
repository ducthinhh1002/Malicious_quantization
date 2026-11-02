"""Receive verified runtime/cache, train full PixArt and return final model for evaluation."""
from pathlib import Path
import json,os,sys,time,hashlib,urllib.request,tarfile,subprocess,http.client,socket,concurrent.futures,threading,zlib
ROOT=Path('/var/lib/structure32_dit_sweep_20261101');ROOT.mkdir(exist_ok=True)
M=json.loads((ROOT/'transport.json').read_text());P=ROOT/'pixart512_fullft'
def status(stage,**kw):(ROOT/'remote_status.json').write_text(json.dumps({'stage':stage,'updated':time.time(),'pid':os.getpid(),**kw}))
try:
    status('download_verified_runtime_model_and_cache')
    bundle=ROOT/'incoming.tar'
    if not (ROOT/'bundle_extracted.json').exists():
        start=bundle.stat().st_size if bundle.exists() else 0
        assert start<=M['bytes']
        done=start;lock=threading.Lock();fd=os.open(bundle,os.O_WRONLY|os.O_CREAT,0o600)
        def fetch(a):
            global done
            b=min(M['bytes'],a+64*1024*1024)-1
            for attempt in range(5):
                try:
                    req=urllib.request.Request(f"http://10.1.21.21:{M['port']}/{M['token']}/bundle",headers={'Range':f'bytes={a}-{b}','X-Compress':'zlib'})
                    with urllib.request.urlopen(req,timeout=120) as resp:
                        assert resp.status==206 and resp.headers.get('X-Compress')=='zlib'
                        content=zlib.decompress(resp.read());assert len(content)==b-a+1
                        offset=a
                        for i in range(0,len(content),4*1024*1024):
                            chunk=content[i:i+4*1024*1024]
                            pos=0
                            while pos<len(chunk):pos+=os.pwrite(fd,chunk[pos:],offset+pos)
                            offset+=len(chunk)
                        assert offset==b+1
                    with lock:
                        done+=b-a+1;status('download_verified_runtime_model_and_cache',downloaded=done,total=M['bytes'],streams=8)
                    return
                except Exception:
                    if attempt==4:raise
                    time.sleep(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch,range(start,M['bytes'],64*1024*1024)))
        os.close(fd)
        with bundle.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert done==M['bytes'] and digest==M['sha256']
        status('extract_exact_runtime_and_cache')
        with tarfile.open(bundle) as t:
            for m in t.getmembers():
                target=(Path('/')/m.name).resolve()
                allowed=[Path('/var/lib/vllm/env'),ROOT,Path('/var/lib/structure32_newprompts_rank32')]
                assert any(target==base or base in target.parents for base in allowed),str(target)
            t.extractall('/',filter='data')
        (ROOT/'bundle_extracted.json').write_text(json.dumps({'sha256':M['sha256']}));bundle.unlink()
    # Same interpreter, packages and CUDA build as .21; no global environment changes.
    py='/var/lib/vllm/env/bin/python'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(P),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    status('fullft_smoke_then_train20000',gpu=0)
    with (ROOT/'fullft_supervisor.log').open('a') as log:subprocess.run([py,'-u',str(P/'ft_worker.py')],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('return_final_full_model_to_21')
    out=ROOT/'final_result.tar'
    with tarfile.open(out,'w') as t:
        for folder in [P/'transformer']:
            for f in folder.rglob('*'):
                if f.is_file():t.add(f,arcname=str(f.relative_to(P)))
        for n in ['trained.json','smoke.json','experiment_identity.json','losses.jsonl','trainable_scope.json']:
            t.add(P/n,arcname=n)
    with out.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    c=http.client.HTTPConnection('10.1.21.21',M['port'],timeout=600)
    c.putrequest('POST','/'+M['token']+'/result');c.putheader('Content-Length',str(out.stat().st_size));c.putheader('X-Content-SHA256',digest);c.endheaders()
    with out.open('rb') as f:
        while chunk:=f.read(4*1024*1024):c.send(chunk)
    response=c.getresponse();assert response.status==200,response.read();c.close();out.unlink()
    status('training_completed_final_model_returned_to_21',full_state_checkpoints_retained_on_23=True)
except Exception as exc:status('failed',error=repr(exc));raise
