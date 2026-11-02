"""Resumable local-to-.23 deployment. Never contacts .21."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,tarfile,time
ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
STATE=WORKSPACE/'output_artifacts/deployments/structure8bit_rank4_refine600_23'
STATE.mkdir(parents=True,exist_ok=True)
KEY=WORKSPACE/'tla-vpqh.pem'
REMOTE='/var/lib/structure8bit_rank4_refine600'
OPTIONS=['-i',str(KEY),'-o','BatchMode=yes','-o','ConnectTimeout=30','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o',f'ProxyCommand=ssh -i "{KEY}" -o BatchMode=yes -W %h:%p root@171.253.169.209']
def status(stage,**kw):
    p=STATE/'status.json';p.write_text(json.dumps({'stage':stage,'updated':time.time(),**kw},indent=2)+'\n')
def ssh(cmd):subprocess.run(['ssh',*OPTIONS,'root@10.1.21.23',cmd],check=True)
def upload(p):
    for attempt in range(20):
        try:
            probe=subprocess.run(['ssh',*OPTIONS,'root@10.1.21.23',f'test -f {REMOTE}/incoming/{p.name}.partial'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if probe.returncode not in (0,1):raise subprocess.CalledProcessError(probe.returncode,probe.args)
            operation='reput' if probe.returncode==0 else 'put'
            subprocess.run(['sftp',*OPTIONS,'-b','-','root@10.1.21.23'],input=f'{operation} "{p.as_posix()}" "{REMOTE}/incoming/{p.name}.partial"\n',text=True,check=True)
            with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            ssh(f"echo '{digest}  {REMOTE}/incoming/{p.name}.partial' | sha256sum -c - && mv {REMOTE}/incoming/{p.name}.partial {REMOTE}/incoming/{p.name} && tar -xf {REMOTE}/incoming/{p.name} -C {REMOTE}")
            return
        except subprocess.CalledProcessError:
            if attempt==19:raise
            time.sleep(5)
def image_bundle(name):
    p=STATE/(name+'.tar');marker=STATE/(name+'.uploaded.json')
    if marker.exists():return
    src=WORKSPACE/'output_artifacts/runs/structure/student7500/run/images'/name
    if not p.exists():
        with tarfile.open(p.with_suffix('.partial'),'w') as t:t.add(src,arcname='source/images/'+name)
        p.with_suffix('.partial').replace(p)
    upload(p);marker.write_text(json.dumps({'completed':time.time(),'bytes':p.stat().st_size}))
def main():
    ssh(f'mkdir -p {REMOTE}/incoming {REMOTE}/deployment')
    p=STATE/'code_teacher.tar'
    with tarfile.open(p,'w') as t:
        for f in ROOT.iterdir():
            if f.name in ['__pycache__','env','runtime','deployment','runs']:continue
            t.add(f,arcname=f.name,filter=lambda x:None if '__pycache__' in x.name else x)
    status('uploading_code_and_refine600_teacher');upload(p)
    ssh(f"cd {REMOTE} && nohup python3 -u prepare_server.py > deployment/setup.log 2>&1 < /dev/null &")
    status('uploading_structure_images',parallel_transfers=3)
    names=['teacher_wm_train','teacher_wm_validation','teacher_wm_test','teacher_clean_validation','teacher_clean_test','unadapted']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(image_bundle,names))
    ssh(f"echo '{{\"source\":\"local Structure7500 original images\",\"teacher_refinement_steps\":600}}' > {REMOTE}/DATA_READY.json")
    status('deployment_complete_remote_pipeline_autostart',remote=REMOTE,server='10.1.21.23')
if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=repr(e));raise
