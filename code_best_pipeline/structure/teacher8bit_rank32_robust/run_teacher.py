"""Teacher-only EOT refinement, followed by direct clean/manipulated image evaluation."""
import bootstrap
from bootstrap import ROOT,SHARED,HARNESS
from pathlib import Path
import json,os,time,subprocess,hashlib,shutil,fcntl
from common import read_json,freeze,sha256,write_json
RUN=ROOT/'run';STATE=ROOT/'deployment';STATE.mkdir(exist_ok=True)
def status(stage,**kw):write_json(STATE/'status.json',{'stage':stage,'pid':os.getpid(),'updated':time.time(),**kw})
def share(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src,target_is_directory=src.is_dir())
def initialize():
    RUN.mkdir(exist_ok=True);source=SHARED/'source'
    cp=source/'teacher_fp/refined'
    assert sha256(cp/'adapter_model.safetensors')=='e4cdc8feefe33480266f53191592030f6dcffa40c05238985a1726ac97a73d6f'
    assert len(read_json(source/'carrier/key.json')['signs'])==8
    for name in ['assets.json','prompts.json','carrier','scores']:
        # Metadata copied; no evaluation file from this new run overwrites old records.
        src=source/name;dst=RUN/name
        if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
        else:shutil.copy2(src,dst)
    (RUN/'result').mkdir(exist_ok=True)
    for p in (source/'result').glob('threshold*.json'):shutil.copy2(p,RUN/'result'/p.name)
    for label in ['teacher_clean_train','teacher_clean_validation','teacher_clean_test','unadapted']:
        share(source/'images'/label,RUN/'images'/label)
    share(source/'images/teacher_wm_train',RUN/'images/baseline_wm_train')
    share(source/'images/teacher_wm_test',RUN/'images/teacher_refine600')
    (RUN/'teacher_fp').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'calibration_target.json',RUN/'teacher_fp/calibration_target.json')
    c=read_json(source/'config.json');c.update(teacher_lr=5e-6,teacher_train_steps=600,teacher_checkpoint_every=200)
    freeze(RUN/'config.json',c)
    spec=read_json(ROOT/'original_refine_spec.json');spec.update(teacher_lr=5e-6,teacher_steps=600,
        native_micro_batch=1,bit_margin=.1,robust_weight=1.)
    freeze(RUN/'refine_spec.json',spec)
    freeze(RUN/'teacher_choice.json',{'method':'teacher_only_eot_per_bit','source_checkpoint':str(cp),
        'source_adapter_sha256':sha256(cp/'adapter_model.safetensors'),'source_refinement_steps':600,
        'additional_refinement_steps':600,'num_bits':8,'rank':32,'fixed_key_carrier':True,
        'student_training':False,'test_used_for_selection':False})
    from attacks import CONFIG
    freeze(RUN/'attack_config.json',CONFIG)
    freeze(ROOT/'code_manifest.json',{p.name:sha256(p) for p in ROOT.glob('*.py')})
def stage(name,args):
    marker=STATE/(name+'.complete.json')
    if marker.exists():return
    status(name)
    with (STATE/(name+'.log')).open('a') as f:subprocess.run(args,cwd=ROOT,check=True,stdout=f,stderr=subprocess.STDOUT)
    freeze(marker,{'completed':time.time()})
def generate():
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import load_native,generate_set
    model=merge_teacher(RUN,RUN/'teacher_fp/refined');pipe=load_native(model)
    for split in ['validation','test']:generate_set(RUN,model,split,'teacher_wm_'+split,pipe)
def quality():
    from semantic_report import teacher_quality
    freeze(RUN/'result/teacher_quality.json',teacher_quality(RUN,'teacher_wm_validation'))
def main():
    import sys
    if '--generate' in sys.argv:generate();return
    if '--quality' in sys.argv:quality();return
    with (STATE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not ((SHARED/'DATA_READY.json').exists() and (SHARED/'CLEAN_TRAIN_READY.json').exists()):
            status('waiting_for_teacher_training_images');time.sleep(20)
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=45000:break
            status('waiting_for_gpu1',free_mib=free);time.sleep(30)
        if not (ROOT/'code_manifest.json').exists():initialize()
        py=sys.executable
        stage('01_actual_teacher_smoke',[py,str(ROOT/'teacher_refine.py'),'--run',str(RUN),'--smoke'])
        stage('02_teacher_eot600',[py,str(ROOT/'teacher_refine.py'),'--run',str(RUN)])
        stage('03_generate_teacher_panel',[py,str(ROOT/'run_teacher.py'),'--generate'])
        stage('04_teacher_scores',[py,str(HARNESS/'semantic_teacher.py'),'--run',str(RUN),'--stage','score','--directories','teacher_wm_validation','teacher_wm_test'])
        stage('05_teacher_quality',[py,str(ROOT/'run_teacher.py'),'--quality'])
        stage('06_teacher_manipulations',[py,str(ROOT/'robustness_eval.py'),'--run',str(RUN)])
        status('completed')
if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=repr(e));raise
