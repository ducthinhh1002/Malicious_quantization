"""Prepare an isolated environment and launch the Structure refine600/rank4 run."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
STATE=ROOT/'deployment';STATE.mkdir(exist_ok=True)
def status(stage,**kw):
    p=STATE/'setup_status.json';t=p.with_suffix('.partial')
    t.write_text(json.dumps({'stage':stage,'updated':time.time(),**kw},indent=2));t.replace(p)
def check(args):subprocess.run(list(map(str,args)),check=True,cwd=ROOT)
def main():
    env=ROOT/'env';py=env/'bin/python'
    if not py.exists():check([sys.executable,'-m','venv','--without-pip',env])
    if subprocess.run([str(py),'-m','pip','--version'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        check(['/opt/jupyterlab-os/venv/bin/python','-m','pip','--python',py,'install','pip'])
    status('installing_environment')
    check([py,'-m','pip','install','-r',ROOT/'harness/requirements-semantic.txt','setproctitle==1.3.7','matplotlib'])
    os.environ['PYTHONPATH']=str(ROOT)
    status('downloading_pinned_base_models')
    check([py,ROOT/'prepare_server.py','--models'])
    status('waiting_for_local_structure_images')
    while not (ROOT/'DATA_READY.json').exists():time.sleep(15)
    priority=Path('/var/lib/structure8bit_rank32_aug/deployment/status.json')
    while not (priority.exists() and json.loads(priority.read_text()).get('stage')=='completed'):
        status('waiting_for_rank32_priority');time.sleep(20)
    status('verifying_teacher_and_training_images')
    source=ROOT/'source';cp=source/'teacher_fp/refined'
    trained=json.loads((cp/'trained.json').read_text())
    for name,digest in trained['files'].items():
        assert hashlib.sha256((cp/name).read_bytes()).hexdigest()==digest,name
    choice=json.loads((source/'teacher_choice.json').read_text())
    assert choice['teacher_refinement_steps']==600 and choice['num_bits']==8
    assert json.loads((source/'carrier/key.json').read_text())['signs'].__len__()==8
    count=0
    for folder in (source/'images').iterdir():
        if not folder.is_dir():continue
        for row in json.loads((folder/'image_hashes.json').read_text()):
            assert hashlib.sha256((folder/row['file_name']).read_bytes()).hexdigest()==row['sha256'],str(folder/row['file_name'])
            count+=1
    status('initializing_student',verified_images=count)
    check([py,ROOT/'campaign.py','--initialize'])
    envvars=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',
                 BITMARK_RUNTIME_ROOT=str(ROOT/'runtime'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    status('running_student_pipeline',teacher_refinement_steps=600,student_rank=4,num_bits=8,student_steps=12500)
    subprocess.run([str(py),str(ROOT/'campaign.py'),'--supervise'],env=envvars,cwd=ROOT,check=True)
    status('completed')
def models():
    from huggingface_hub import hf_hub_download
    source=ROOT/'source';p=source/'assets.json';a=json.loads(p.read_text())
    specs=[('teacher_clean','Manojb/stable-diffusion-2-1-base',a['teacher_source']['revision'],'teacher_clean_hashes'),
           ('student','CompVis/stable-diffusion-v1-4',a['student_revision'],'student_hashes')]
    for key,repo,rev,hashes in specs:
        folder=ROOT/'runtime/models'/key
        for name,digest in a[hashes].items():
            f=Path(hf_hub_download(repo_id=repo,revision=rev,filename=name,local_dir=folder))
            assert hashlib.sha256(f.read_bytes()).hexdigest()==digest,repo+'/'+name
        a[key]=str(folder)
    a['teacher']=a['teacher_clean'];p.write_text(json.dumps(a,indent=2)+'\n')
if __name__=='__main__':
    try:
        if '--models' in sys.argv:models()
        else:main()
    except Exception as e:status('failed',error=repr(e));raise
