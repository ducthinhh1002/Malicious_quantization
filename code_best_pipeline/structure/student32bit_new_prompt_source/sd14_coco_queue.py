"""Queue fresh SD1.4 rank32 on the exact shared Sana/PixArt COCO image set."""
from pathlib import Path
import os,sys,json,time,shutil,subprocess,fcntl
PARENT=Path(__file__).resolve().parent
ROOT=PARENT/'sd14_coco_disjoint'
SOURCE=PARENT/'coco_disjoint_source'
HARNESS=Path('/var/lib/structure32_student12500/harness')
sys.path.insert(0,str(HARNESS))
from common import read_json,write_json,freeze,sha256

def status(stage,**kw):
    write_json(ROOT/'status.json',{'stage':stage,'pid':os.getpid(),'updated':time.time(),**kw})

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
    assert dst.resolve()==src.resolve()

def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze(ROOT/'queue_plan.json',{'student':'clean SD1.4; new LoRA rank32 attention_conv',
            'source':str(SOURCE),'train':5000,'validation':1000,'test':1000,
            'teacher':'unchanged Structure32 refine600','bits':32,'batch_size':4,'seed':0,'steps':20000,
            'data_reuse':'same read-only prompt JSON and image directories as Sana/PixArt; no regeneration',
            'run_after':'sana600m releases GPU0 after completion or technical failure','full_state_every':1250,'test_selection':False})
        status('queued_waiting_for_coco_data_and_sana_completion')
        while True:
            ready=(PARENT/'coco_data_complete.json').exists()
            f=PARENT/'cross_arch/sana600m/status.json'
            state=read_json(f).get('stage') if f.exists() else None
            if ready and state in ['completed','failed']:break
            if state=='failed':status('queued_sana_failed_needs_attention')
            time.sleep(30)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
            BITMARK_RUNTIME_ROOT='/var/lib/vllm/runtime',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        status('score_shared_teacher_and_sd14_control')
        if not (ROOT/'source_scores.complete.json').exists():
            with (ROOT/'source_scores.log').open('a') as log:
                subprocess.run([sys.executable,str(HARNESS/'semantic_teacher.py'),'--run',str(SOURCE),
                    '--stage','score','--directories','unadapted','teacher_wm_test'],env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
            freeze(ROOT/'source_scores.complete.json',{'completed':time.time()})
        freeze(SOURCE/'complete.json',{'role':'shared disjoint COCO source ready','prompt_sha256':sha256(SOURCE/'prompts.json')})
        link(SOURCE,ROOT/'source');link(HARNESS,ROOT/'harness')
        freeze(ROOT/'plan.json',{'dataset':'COCO2014 disjoint from all original teacher splits',
            'new_counts':{'train':5000,'validation':1000,'test':1000},'source':str(SOURCE),
            'prompt_sha256':sha256(SOURCE/'prompts.json'),
            'training_image_manifest_sha256':sha256(SOURCE/'images/teacher_wm_train/image_hashes.json'),
            'teacher_sha256':'2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8',
            'same_data_as':['sana600m','pixart512'],'selection':'fixed20000; no validation/test checkpoint selection'})
        dest=ROOT/'student';dest.mkdir(exist_ok=True)
        files=['bootstrap.py','schedule.py','resume_state.py','driver.py','campaign.py','tests.py',
            'key_metrics.py','key_summary.py','key_select.py','fixed_endpoint.py','acceptance.py','acceptance_metrics.py']
        for name in files:
            if not (dest/name).exists():shutil.copy2(PARENT/'student'/name,dest/name)
        shutil.copy2(PARENT/'sitecustomize.py',dest/'sitecustomize.py')
        env['PYTHONPATH']=str(dest)
        with (ROOT/'student_supervisor.log').open('a') as log:
            subprocess.run([sys.executable,str(dest/'campaign.py'),'--initialize'],cwd=dest,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
            status('student_smoke_then_train20000')
            subprocess.run([sys.executable,str(dest/'campaign.py'),'--supervise'],cwd=dest,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        status('completed')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        ROOT.mkdir(exist_ok=True);status('failed',error=repr(exc));raise
