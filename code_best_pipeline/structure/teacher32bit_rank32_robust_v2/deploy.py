"""Upload this isolated code bundle and launch after read-only resource checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
REMOTE='/var/lib/structure32_teacher_robust_v2'
EXPECTED='2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8'

def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='10.1.21.21')
    args=p.parse_args()
    if args.host not in ['10.1.21.21','10.1.21.23']:raise ValueError('Unsupported target')
    options=['-i','./tla-vpqh.pem','-o','BatchMode=yes','-o','ConnectTimeout=15',
             '-o','ProxyCommand=ssh -i ./tla-vpqh.pem -o BatchMode=yes -o ConnectTimeout=15 -W %h:%p root@171.253.169.209']
    def remote(script):
        r=subprocess.run(['ssh',*options,'root@'+args.host,'python3 -u -'],
                         input=script,text=True,cwd=WORKSPACE,capture_output=True,timeout=90)
        if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout)
        return r.stdout
    probe=remote('''import json,hashlib,subprocess,shutil
from pathlib import Path
p=Path('/var/lib/structure32_student12500/runs/structure32')
f=p/'teacher_fp/refined/adapter_model.safetensors'
assert hashlib.file_digest(f.open('rb'),'sha256').hexdigest()=='''+repr(EXPECTED)+'''
assert Path('/var/lib/vllm/env/bin/python').exists()
assert Path('/var/lib/vllm/harness/semantic_teacher.py').exists()
assert shutil.disk_usage('/var/lib').free>30*2**30
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus=[list(map(int,line.split(','))) for line in rows.strip().splitlines()]
candidates=[g for g in gpus if g[1]>=71680]
assert candidates, 'No GPU has 70GiB free; existing jobs will not be stopped'
gpu=max(candidates,key=lambda g:g[1])[0]
print(json.dumps({'gpu':gpu,'resources':gpus,'source':str(p),'checkpoint_verified':True}))
''')
    info=json.loads(probe)
    print('PROBE',json.dumps(info),flush=True)
    output=WORKSPACE/'output_artifacts/deployments/teacher32bit_robust_v2'
    output.mkdir(parents=True,exist_ok=True)
    bundle=output/'code.tar'
    with tarfile.open(bundle,'w') as tar:
        for file in sorted(ROOT.iterdir()):
            if file.is_file() and file.suffix in ['.py','.json','.md']:
                tar.add(file,arcname=file.name)
    digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
    remote("from pathlib import Path\np=Path("+repr(REMOTE)+")\np.mkdir(exist_ok=True)\nassert not (p/'code_manifest.json').exists(), 'Already initialized; use the recorded run, do not redeploy over it'\n")
    subprocess.run(['scp',*options,str(bundle),'root@'+args.host+':'+REMOTE+'/code.tar'],
                   cwd=WORKSPACE,check=True,timeout=120)
    launch=remote('''import os,json,time,hashlib,tarfile,subprocess
from pathlib import Path
p=Path('''+repr(REMOTE)+''')
assert hashlib.file_digest((p/'code.tar').open('rb'),'sha256').hexdigest()=='''+repr(digest)+'''
with tarfile.open(p/'code.tar') as tar:
 for m in tar.getmembers():
  assert m.isfile() and '/' not in m.name and '\\\\' not in m.name and m.name not in ['.','..']
  with tar.extractfile(m) as src:(p/m.name).write_bytes(src.read())
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='''+repr(str(info['gpu']))+''',PYTHONPATH=str(p),
 WM_HARNESS='/var/lib/vllm/harness',WM_SOURCE='''+repr(info['source'])+''',
 OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
py='/var/lib/vllm/env/bin/python'
subprocess.run([py,'-m','compileall','-q',str(p)],env=env,check=True)
state=p/'deployment';state.mkdir(exist_ok=True)
with (state/'supervisor.log').open('ab') as log:
 child=subprocess.Popen([py,str(p/'run_teacher.py')],cwd=p,env=env,
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':child.pid,'gpu':int(env['CUDA_VISIBLE_DEVICES']),'time':time.time(),
 'root':str(p),'source':env['WM_SOURCE'],'bundle_sha256':'''+repr(digest)+''',
 'display_name':'VLLM::Enginecore','status':'launched; real-model smoke pending'}
(state/'launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
''')
    record={'host':args.host,**json.loads(launch)}
    (output/'launch.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
