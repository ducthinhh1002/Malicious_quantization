"""Independent continuation preserving the original12500 artifacts and identity."""
from pathlib import Path
import os,sys,json,time,subprocess,shutil,fcntl
ROOT=Path('/var/lib/structure32_student_manipulations')
NAME=sys.argv[1]
assert NAME in ['brightness07','blur15','flip','rotation15','crop75','drop96','gaussian005']
B=ROOT/'branches'/NAME
sys.path.insert(0,str(B))
import campaign
from common import read_json,write_json,freeze,sha256
OLD=campaign.RUN
BASE=ROOT/'continuation20k'/NAME
RUN=BASE/'run'

def status(stage,**kw):write_json(BASE/'status.json',{'stage':stage,'updated':time.time(),'pid':os.getpid(),**kw})
def stage(name,args):
    marker=BASE/(name+'.complete.json')
    if marker.exists():return
    status(name)
    with (BASE/(name+'.log')).open('a') as log:
        subprocess.run(args,cwd=B,check=True,stdout=log,stderr=subprocess.STDOUT)
    freeze(marker,{'completed':time.time()})

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OLD/'complete.json').exists():
            f=campaign.QUEUE/'status.json'
            if f.exists() and read_json(f).get('state')=='failed':
                raise RuntimeError('Original branch failed: '+str(read_json(f)))
            status('waiting_for_original12500_evaluation');time.sleep(20)
        campaign.verify()
        RUN.mkdir(exist_ok=True)
        c=read_json(OLD/'config.json');c.update(max_train_steps=20000,checkpoint_steps=[20000])
        freeze(RUN/'config.json',c)
        for name in ['assets.json','prompts.json','carrier','teacher_choice.json']:
            dest=RUN/name
            if not dest.exists():dest.symlink_to((OLD/name).resolve(),target_is_directory=(OLD/name).is_dir())
        for name in ['teacher_wm_train','teacher_wm_validation','teacher_wm_test','teacher_clean_validation','teacher_clean_test','unadapted']:
            dest=RUN/'images'/name;dest.parent.mkdir(exist_ok=True)
            if not dest.exists():dest.symlink_to((OLD/'images'/name).resolve(),target_is_directory=True)
        (RUN/'result').mkdir(exist_ok=True)
        out=RUN/'students'/campaign.BRANCH;out.mkdir(parents=True,exist_ok=True)
        src=OLD/'students'/campaign.BRANCH/'checkpoint-12500';dst=out/'checkpoint-12500'
        from resume_state import verify_checkpoint
        identity=read_json(campaign.QUEUE/'resume_identity.json')
        verify_checkpoint(src,identity)
        if not dst.exists():shutil.copytree(src,dst,copy_function=os.link)
        freeze(BASE/'plan.json',{'source_checkpoint':str(src),'source_identity':identity,
            'teacher':'original Structure32 refine600','rank':32,'bits':32,'start':12500,'end':20000,
            'new_steps':7500,'lr':2e-5,'optimizer_rng_restored':True,'original12500_preserved':True})
        cmd=campaign.command(False)
        cmd[cmd.index('--output_dir')+1]=str(out)
        cmd[cmd.index('--max_train_steps')+1]='20000'
        cmd[cmd.index('--snapshot_steps')+1]='20000'
        states=sorted([p for p in out.glob('checkpoint-*') if (p/'resume_manifest.json').exists()],key=lambda p:int(p.name.split('-')[-1]))
        latest=states[-1];proof=verify_checkpoint(latest,identity)
        if proof['step']<20000:
            cmd+=['--resume_from_checkpoint',str(latest)]
            stage('01_train_to20000',cmd)
        from student import checkpoint_hashes,load_pipeline,generate_images
        rows=[json.loads(s) for s in (out/'updates.jsonl').read_text().splitlines()]
        assert rows[-1]['step']==20000 and rows[-1]['applied']+rows[-1]['skipped']==20000
        assert rows[-1]['skipped']<=1, 'Unexpected repeated AMP overflow; investigate'
        import torch
        from safetensors.torch import load_file
        assert all(torch.isfinite(t).all() for t in load_file(str(out/'adapter_model.safetensors')).values())
        freeze(BASE/'optimizer_audit.json',{'attempted_steps':20000,'applied_updates':rows[-1]['applied'],'amp_skipped_updates':rows[-1]['skipped'],'note':'Report actual update count; AMP overflow skips are not counted as applied updates'})
        verify_checkpoint(out/'checkpoint-20000',identity)
        freeze(out/'complete.json',{'checkpoints':checkpoint_hashes(out),'optimizer_update_counts':rows[-1]})
        cp=out/'snapshot-20000'
        freeze(RUN/'result/selection'/f'{campaign.BRANCH}.json',{'selected_step':20000,'checkpoint_hashes':checkpoint_hashes(cp),'rule':'fixed20000 by user','test_used_for_selection':False})
        if not (BASE/'02_generate.complete.json').exists():
            status('02_generate')
            pipe,weights=load_pipeline(RUN,campaign.BRANCH,cp)
            generate_images(RUN,campaign.BRANCH,pipe,weights)
            del pipe
            import gc,torch;gc.collect();torch.cuda.empty_cache()
            freeze(BASE/'02_generate.complete.json',{'completed':time.time()})
        stage('03_score',[sys.executable,str(campaign.HARNESS/'semantic_teacher.py'),'--run',str(RUN),'--stage','score','--directories',campaign.BRANCH])
        from acceptance_metrics import aggregate_metrics
        scores=read_json(RUN/'scores'/f'{campaign.BRANCH}.json')
        freeze(RUN/'result/student_acceptance32.json',{'student_step':20000,
            'training_manipulation':read_json(B/'manipulation.json'),
            'metrics':aggregate_metrics([r['coordinates'] for r in scores],read_json(RUN/'carrier/key.json')['signs'])})
        stage('04_quality',[sys.executable,str(campaign.HARNESS/'utility.py'),'--run',str(RUN),'--labels',campaign.BRANCH])
        import torch,lpips
        from semantic_teacher import pixels
        from common import image_name
        model=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
        distances=[]
        for start in range(0,1000,8):
            names=[image_name(i) for i in range(start,min(start+8,1000))]
            with torch.inference_mode():distances.extend(model(pixels([RUN/'images'/campaign.BRANCH/n for n in names]),pixels([RUN/'images/unadapted'/n for n in names])).flatten().cpu().tolist())
        freeze(RUN/'result/paired_lpips.json',{'n':1000,'paired_lpips':sum(distances)/1000,'per_image':distances})
        status('completed')

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
