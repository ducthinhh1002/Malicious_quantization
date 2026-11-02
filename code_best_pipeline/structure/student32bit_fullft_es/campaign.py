"""Full student fine-tuning on unchanged Structure32 teacher images; no watermark loss."""
import bootstrap
from bootstrap import ROOT,HARNESS,read_json,freeze,write_json,sha256
from pathlib import Path
import os,subprocess,sys,time,json,fcntl,shutil
SOURCE=Path('/var/lib/structure32_student12500/runs/structure32')
RUN=ROOT/'run';STATE=ROOT/'deployment';BRANCH='full_wm'
def status(stage,**kw):write_json(STATE/'status.json',{'stage':stage,'pid':os.getpid(),'updated':time.time(),**kw})
def initialize():
    RUN.mkdir(exist_ok=True);(RUN/'result').mkdir(exist_ok=True)
    for name in ['assets.json','prompts.json','carrier','images','scores']:
        dst=RUN/name;src=SOURCE/name
        if name in ['images','scores']:
            dst.mkdir(exist_ok=True)
            labels=['teacher_wm_train','teacher_wm_validation','teacher_wm_test','teacher_clean_validation','teacher_clean_test','unadapted'] if name=='images' else ['unadapted.json']
            for label in labels:
                if not (dst/label).exists():(dst/label).symlink_to(src/label,target_is_directory=(src/label).is_dir())
        elif not dst.exists():dst.symlink_to(src,target_is_directory=src.is_dir())
    for p in (SOURCE/'result').glob('threshold*.json'):
        if not (RUN/'result'/p.name).exists():(RUN/'result'/p.name).symlink_to(p)
    c=read_json(SOURCE/'config.json');c.update(ranks=[],lora_only=False,paper_full_ema=False,train_clean_controls=False,
        max_train_steps=12500,checkpoint_steps=[12500],full_learning_rate=1e-5,batch_size=4,precision='bf16')
    freeze(RUN/'config.json',c)
    assert c['num_bits']==32
    for row in read_json(RUN/'images/teacher_wm_train/image_hashes.json'):
        assert sha256(RUN/'images/teacher_wm_train'/row['file_name'])==row['sha256']
    freeze(ROOT/'identity.json',{'trainer_sha256':sha256(HARNESS/'vendor/diffusers/train_text_to_image.py'),
        'assets_sha256':sha256(RUN/'assets.json'),'data_sha256':sha256(RUN/'images/teacher_wm_train/image_hashes.json'),
        'teacher_adapter_sha256':sha256(SOURCE/'teacher_fp/refined/adapter_model.safetensors'),
        'code':{p.name:sha256(p) for p in ROOT.glob('*.py')},'steps':12500,'batch':4,'scope':'all UNet parameters; VAE/text encoder frozen',
        'loss':'ordinary diffusion denoising MSE; no watermark-specific objective','lr':'1e-5 until7500; cosine to2e-6 at12500',
        'acceptance_threshold':24,'payload_bits':32,'fpr_theory':0.0035001833457499743})
def command(smoke=False):
    from student import training_command
    c=read_json(RUN/'config.json');a=read_json(RUN/'assets.json')
    if smoke:c.update(max_train_steps=4,checkpoint_steps=[])
    cmd=training_command(c,a,RUN,{'name':BRANCH,'kind':'full','condition':'wm'})
    cmd[cmd.index('--checkpointing_steps')+1]='1250'
    cmd+=['--checkpoints_total_limit','2']
    if smoke:cmd[cmd.index('--output_dir')+1]=str(RUN/'technical_smoke')
    else:
        out=RUN/'students'/BRANCH
        checkpoints=sorted([p for p in out.glob('validation-state-*') if (p/'VALIDATION_COMPLETE.json').exists()]+[p for p in out.glob('checkpoint-*') if (p/'scheduler.bin').exists()],key=lambda p:int(p.name.split('-')[-1]))
        if checkpoints:cmd+=['--resume_from_checkpoint',str(checkpoints[-1])]
    return [sys.executable,str(ROOT/'driver.py'),'--identity',str(ROOT/'identity.json'),*cmd[2:]]
def train(smoke=False):
    from student import checkpoint_hashes
    out=RUN/'students'/BRANCH
    if not smoke and (out/'complete.json').exists():return
    subprocess.run(command(smoke),cwd=ROOT,check=True)
    if not smoke:
        control=read_json(RUN/'early_stopping.json');best=Path(control['best_checkpoint'])
        hashes=checkpoint_hashes(best)
        freeze(out/'complete.json',{'checkpoints':{str((best/k).relative_to(out)):v for k,v in hashes.items()},'full_unet':True,'stopping':control})
        freeze(RUN/'result/selection/full_wm.json',{'selected_step':control['best_step'],'checkpoint_hashes':hashes,'checkpoint_dir':str(best),'rule':'minimum held-out denoising loss; no test selection'})
def generate():
    from student import load_pipeline,generate_images
    best=Path(read_json(RUN/'early_stopping.json')['best_checkpoint'])
    pipe,cp=load_pipeline(RUN,BRANCH,checkpoint_dir=best)
    generate_images(RUN,BRANCH,pipe,cp)

def summary():
    from key_summary import summary as report
    report(RUN,BRANCH)
    from key_metrics import aggregate_metrics
    import numpy as np,math
    rows=read_json(RUN/'scores/full_wm.json');q=np.array([r['coordinates'] for r in rows]);key=np.array(read_json(RUN/'carrier/key.json')['signs']);result=[]
    for size in [1,32,256]:
        n=len(q)//size;rng=np.random.default_rng(20261029+size);yes=total=0
        for _ in range(1 if size==1 else 100):
            ids=rng.permutation(len(q))[:n*size].reshape(n,size);k=(np.where(q[ids].mean(1)>=0,1,-1)==key).sum(1)
            yes+=int((k>=24).sum());total+=n
        result.append({'group_size':size,'accepted':yes,'groups':total,'acceptance_rate':yes/total})
    freeze(RUN/'result/key_acceptance_24of32.json',{'threshold':24,'bits':32,'fpr_theoretical':sum(math.comb(32,k) for k in range(24,33))/2**32,'null':'independent fair random bits; not empirical FPR','rows':result})
def stage(name,args):
    marker=STATE/(name+'.complete.json')
    if marker.exists():return
    status(name)
    with (STATE/(name+'.log')).open('a') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    freeze(marker,{'completed':time.time()})
def main():
    if len(sys.argv)>1:
        {'smoke':lambda:train(True),'train':train,'summary':summary,'generate':generate}[sys.argv[1]]();return
    STATE.mkdir(exist_ok=True)
    with (STATE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (ROOT/'identity.json').exists():initialize()
        assert read_json(ROOT/'identity.json')['code']=={p.name:sha256(p) for p in ROOT.glob('*.py')},'Code changed after freeze'
        py=sys.executable
        # Migration resumes validated full-state1250; original4-update smoke already passed.
        stage('02_full_unet12500',[py,str(ROOT/'campaign.py'),'train'])
        stage('03_generate',[py,str(ROOT/'campaign.py'),'generate'])
        stage('04_score',[py,str(HARNESS/'semantic_teacher.py'),'--run',str(RUN),'--stage','score','--directories',BRANCH])
        stage('05_quality',[py,str(HARNESS/'utility.py'),'--run',str(RUN),'--labels',BRANCH])
        stage('06_summary',[py,str(ROOT/'campaign.py'),'summary'])
        status('completed')
if __name__=='__main__':
    try:main()
    except Exception as e:STATE.mkdir(exist_ok=True);status('failed',error=repr(e));raise
