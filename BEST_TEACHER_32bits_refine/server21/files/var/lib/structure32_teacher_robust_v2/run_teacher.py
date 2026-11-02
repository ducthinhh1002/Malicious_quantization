"""Isolated, resumable teacher32 refinement; no existing run is overwritten."""
import bootstrap
from bootstrap import ROOT, HARNESS
import os,sys,json,time,shutil,subprocess,fcntl
from pathlib import Path
from common import read_json,write_json,freeze,sha256
from attacks import CONFIG

RUN=ROOT/'run'
STATE=ROOT/'deployment'
SOURCE=Path(os.environ.get('WM_SOURCE','/var/lib/structure32_student12500/runs/structure32'))
EXPECTED='2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8'

def status(stage,**kw):
    write_json(STATE/'status.json',{'stage':stage,'pid':os.getpid(),'updated':time.time(),**kw})

def link(src,dst):
    src=src.resolve(strict=True)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src,target_is_directory=src.is_dir())
    elif dst.resolve()!=src:raise RuntimeError('Unexpected existing destination '+str(dst))

def initialize():
    cp=SOURCE/'teacher_fp/refined'
    assert sha256(cp/'adapter_model.safetensors')==EXPECTED,'Wrong source teacher'
    assert len(read_json(SOURCE/'carrier/key.json')['signs'])==32
    assert read_json(cp/'adapter_config.json')['r']==32
    c=read_json(SOURCE/'config.json')
    assert c['inference_steps']==50 and c['teacher_batch']==4
    assert len(read_json(SOURCE/'prompts.json')['train'])==5000
    for split in ['validation','test']:assert len(read_json(SOURCE/'prompts.json')[split])==1000
    RUN.mkdir(exist_ok=True)
    for name in ['assets.json','prompts.json','carrier']:
        src=SOURCE/name;dst=RUN/name
        if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
        else:shutil.copy2(src,dst)
    for split in ['train','validation','test']:
        clean=SOURCE/'images'/f'teacher_clean_{split}'
        if clean.exists():link(clean,RUN/'images'/f'teacher_clean_{split}')
        link(SOURCE/'images'/f'teacher_wm_{split}',RUN/'images'/f'before_{split}')
    link(SOURCE/'images/teacher_wm_train',RUN/'images/baseline_wm_train')
    (RUN/'teacher_fp').mkdir(exist_ok=True)
    shutil.copy2(SOURCE/'teacher_fp/calibration_target.json',RUN/'teacher_fp/calibration_target.json')
    c.update(teacher_lr=5e-6,teacher_train_steps=1200,teacher_checkpoint_every=200)
    freeze(RUN/'config.json',c)
    spec=read_json(ROOT/'original_refine_spec.json')
    spec.update(teacher_lr=5e-6,teacher_steps=1200,native_micro_batch=2,
                bit_margin=.1,robust_fraction=1.,curriculum_steps=200,
                native_every=2,checkpoint_every=200)
    freeze(RUN/'refine_spec.json',spec)
    freeze(RUN/'teacher_choice.json',{'source_checkpoint':str(cp),'source_adapter_sha256':EXPECTED,
        'method':'two_view_image_pcgrad_weakbit_teacher32','source_refinement_steps':600,
        'additional_refinement_steps':1200,'num_bits':32,'rank':32,
        'key_carrier_sampler_unchanged':True,'test_used_for_selection':False})
    freeze(RUN/'attack_config.json',CONFIG)
    freeze(ROOT/'code_manifest.json',{p.name:sha256(p) for p in ROOT.glob('*.py')})

def stage(name,args):
    marker=STATE/(name+'.complete.json')
    if marker.exists():return
    status(name)
    with (STATE/(name+'.log')).open('a') as log:
        subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    freeze(marker,{'completed':time.time()})

def generate():
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import load_native,generate_set
    merged=merge_teacher(RUN,RUN/'teacher_fp/refined')
    pipe=load_native(merged)
    for split in ['validation','test']:
        generate_set(RUN,merged,split,'teacher_wm_'+split,pipe)

def quality():
    # LPIPS always has a separate saved result. CLIP/cache failure cannot erase
    # this result or block the already completed manipulation evaluation.
    import torch,lpips
    from semantic_teacher import pixels
    from common import image_name
    n=len(read_json(RUN/'prompts.json')['validation'])
    net=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    results={}
    for label in ['before_validation','teacher_wm_validation']:
        distances=[]
        for start in range(0,n,8):
            ids=range(start,min(n,start+8))
            with torch.inference_mode():
                d=net(pixels([RUN/'images/teacher_clean_validation'/image_name(i) for i in ids]),
                      pixels([RUN/'images'/label/image_name(i) for i in ids])).flatten()
            distances.extend(d.cpu().tolist())
        results[label]={'n':n,'paired_lpips':sum(distances)/n,'per_image':distances}
    freeze(RUN/'result/paired_lpips.json',results)
    from semantic_report import teacher_quality
    for label in results:
        freeze(RUN/'result'/f'quality_{label}.json',teacher_quality(RUN,label))

def main():
    STATE.mkdir(exist_ok=True)
    if '--generate' in sys.argv:generate();return
    if '--quality' in sys.argv:quality();return
    with (STATE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (ROOT/'code_manifest.json').exists():initialize()
        else:
            for name,digest in read_json(ROOT/'code_manifest.json').items():
                if sha256(ROOT/name)!=digest:raise RuntimeError('Code changed since initialization: '+name)
        py=sys.executable
        stage('00_technical_tests',[py,str(ROOT/'test_protocol.py')])
        stage('01_actual_teacher_smoke',[py,str(ROOT/'teacher_refine.py'),'--run',str(RUN),'--smoke'])
        stage('02_teacher_robust1200',[py,str(ROOT/'teacher_refine.py'),'--run',str(RUN)])
        stage('03_generate',[py,str(ROOT/'run_teacher.py'),'--generate'])
        stage('04_test_manipulations',[py,str(ROOT/'evaluate.py'),'--run',str(RUN)])
        stage('05_validation_manipulations',[py,str(ROOT/'evaluate.py'),'--run',str(RUN),'--split','validation'])
        try:stage('06_quality',[py,str(ROOT/'run_teacher.py'),'--quality'])
        except subprocess.CalledProcessError as e:
            status('manipulations_completed_quality_pending',error=str(e));return
        status('completed')

if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=repr(e));raise
