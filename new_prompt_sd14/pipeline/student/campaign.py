"""Independent Structure teacher -> ordinary SD1.4 student, 20000 updates."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import bootstrap
from bootstrap import ROOT,BASE,HARNESS,read_json,write_json,freeze,sha256

SOURCE=BASE/'source'
QUEUE=ROOT/'runs/structure32_student20000_seed0'
RUN=QUEUE/'student20000'
BRANCH='lora_r32_conv_wm'
FILES=['bootstrap.py','schedule.py','resume_state.py','driver.py','campaign.py','tests.py','key_metrics.py','key_summary.py','key_select.py','fixed_endpoint.py','acceptance.py','acceptance_metrics.py']
TRAINER=HARNESS/'vendor/diffusers/train_text_to_image_lora.py'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def share(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source.resolve(),target_is_directory=source.is_dir())


def initialize():
    if not (SOURCE/'complete.json').exists():raise RuntimeError('Structure source is incomplete')
    RUN.mkdir(parents=True,exist_ok=True)
    c=read_json(SOURCE/'config.json')
    c.update(max_train_steps=20000,checkpoint_steps=[20000])
    freeze(RUN/'config.json',c)
    for rel in ['assets.json','prompts.json','carrier','result/threshold.json',
        'result/threshold_student.json','result/threshold_teacher.json',
        'images/teacher_wm_train','images/teacher_wm_validation','images/teacher_wm_test',
        'images/teacher_clean_validation','images/teacher_clean_test','images/unadapted',
        'scores/unadapted.json']:
        share(SOURCE/rel,RUN/rel)
    freeze(RUN/'teacher_choice.json',{**read_json(SOURCE/'teacher_choice.json'),
        'method':'full_structure32_student20000','teacher_is_identical_to':str(SOURCE),
        'student_initialization':'clean pinned SD1.4; new LoRA; no prior student checkpoint',
        'student_steps':20000,'student_trainer':'ordinary denoising MSE, rank32 attention_conv, batch4',
        'lr_schedule':'1e-4 through7500; cosine to2e-5 at12500; fixed2e-5 thereafter',
        'selection':'fixed20000 by user; no validation checkpoint evaluation',
        'quality_stopping_gates':False,'test_used_for_selection':False})
    manifest={name:sha256(ROOT/name) for name in FILES}
    freeze(QUEUE/'code_manifest.json',manifest)
    provenance={rel:sha256(SOURCE/rel) for rel in ['assets.json','prompts.json','config.json',
        'images/teacher_wm_train/image_hashes.json','images/teacher_wm_validation/image_hashes.json',
        'carrier/complete.json','result/threshold_student.json','teacher_fp/refined/adapter_model.safetensors']}
    freeze(QUEUE/'plan.json',{'created':(read_json(QUEUE/'plan.json')['created'] if (QUEUE/'plan.json').exists() else now()),'source':str(SOURCE),'source_provenance':provenance,
        'initialization':'clean SD1.4, train from scratch with unchanged Structure teacher images',
        'primary_metrics':['bit_accuracy','full_key_recovery_rate'],'new_student_updates':20000,
        'schedule':'constant1e-4 for7500; cosine to2e-5 by12500',
        'full_state_every':1250,'retention':'all16 complete state checkpoints; no deletion',
        'state_contents':['LoRA parameters','AdamW moments/step','LR scheduler','FP16 scaler',
            'Python/NumPy/Torch CPU/CUDA RNG','cumulative optimizer update counts','epoch/seed data-order identity'],
        'sampler':'SeedableRandomSampler data_seed0, explicit set_epoch; resume only full epoch boundaries',
        'sampler_caveat':'Deterministic sampler differs from earlier unseeded RandomSampler; this is a fresh trial, not bitwise replay',
        'selection':'fixed20000 by user; no validation checkpoint evaluation',
        'scientific_gates':False,'project':'vllm',
        'training_gpu_cap_gib':44,'limitations':['single seed','existing exploratory test','32bit exploratory single-seed experiment','LPIPS is paired deviation, not anatomy']})
    freeze(QUEUE/'resume_identity.json',{'format':1,'trainer_sha256':sha256(TRAINER),
        'student_assets_sha256':sha256(RUN/'assets.json'),
        'training_images_manifest_sha256':sha256(RUN/'images/teacher_wm_train/image_hashes.json'),
        'prompts_sha256':sha256(RUN/'prompts.json'),'seed':0,'batch_size':4,'steps_per_epoch':1250,
        'precision':'fp16','rank':32,'scope':'attention_conv',
        'schedule':'absolute7500to12500cosine_floor0.2','sampler':'seedable explicit epochs',
        'training_code':{n:manifest[n] for n in ['driver.py','resume_state.py','schedule.py']}})
    verify()


def verify():
    from common import verify_vendor
    verify_vendor()
    for name,h in read_json(QUEUE/'code_manifest.json').items():
        if sha256(ROOT/name)!=h:raise RuntimeError('Frozen experiment code changed: '+name)
    for rel,h in read_json(QUEUE/'plan.json')['source_provenance'].items():
        if sha256(SOURCE/rel)!=h:raise RuntimeError('Source changed: '+rel)
    if sha256(TRAINER)!=read_json(QUEUE/'resume_identity.json')['trainer_sha256']:
        raise RuntimeError('Student trainer changed')


def command(smoke=False):
    from student import training_command,branches
    c=read_json(RUN/'config.json');assets=read_json(RUN/'assets.json')
    if smoke:c.update(max_train_steps=4,checkpoint_steps=[])
    branch=next(b for b in branches(c) if b['name']==BRANCH)
    cmd=training_command(c,assets,RUN,branch)
    cmd[cmd.index('--checkpointing_steps')+1]='1250'
    if smoke:cmd[cmd.index('--output_dir')+1]=str(RUN/'technical_smoke')
    return [sys.executable,str(ROOT/'driver.py'),'--trainer',str(TRAINER),
        '--identity-file',str(QUEUE/'resume_identity.json'),
        *(['--technical-smoke'] if smoke else []),*cmd[2:]]


def train(smoke=False):
    from student import checkpoint_hashes
    from resume_state import verify_checkpoint
    import torch
    from safetensors.torch import load_file
    verify()
    out=RUN/('technical_smoke' if smoke else 'students/'+BRANCH)
    if (out/'complete.json').exists():
        if read_json(out/'complete.json')['checkpoints']!=checkpoint_hashes(out):
            raise RuntimeError('Completed weights changed')
        return
    entries=read_json(RUN/'images/teacher_wm_train/image_hashes.json')
    if len(entries)!=5000 or any(sha256(RUN/'images/teacher_wm_train'/v['file_name'])!=v['sha256'] for v in entries):
        raise RuntimeError('Training image integrity failed')
    cmd=command(smoke)
    if not smoke:
        state=sorted([p for p in out.glob('checkpoint-*') if p.name[11:].isdigit() and (p/'resume_manifest.json').exists()],
                     key=lambda p:int(p.name[11:]))
        if state:
            proof=verify_checkpoint(state[-1],read_json(QUEUE/'resume_identity.json'))
            if proof['step']<20000:cmd+=['--resume_from_checkpoint',str(state[-1])]
            elif not (out/'adapter_model.safetensors').exists():
                from student import load_adapter
                raise RuntimeError('Final state exists but final inference artifact missing; recover before evaluation')
            else:cmd=None
    if cmd:
        freeze(out/'protocol.json',{'initial_command':command(smoke),'data_manifest_sha256':sha256(RUN/'images/teacher_wm_train/image_hashes.json'),
            'initialization':'clean SD1.4','resume_identity_sha256':sha256(QUEUE/'resume_identity.json')})
        subprocess.run(cmd,cwd=HARNESS,check=True)
    weights=load_file(str(out/'adapter_model.safetensors'))
    if not all(torch.isfinite(v).all() for v in weights.values()):raise RuntimeError('Nonfinite weights')
    if not any(v.abs().max()>0 for n,v in weights.items() if 'lora_B' in n):raise RuntimeError('LoRA did not learn')
    rows=[__import__('json').loads(s) for s in (out/'updates.jsonl').read_text().splitlines()]
    expected=4 if smoke else 20000
    if rows[-1]['step']!=expected or rows[-1]['applied']+rows[-1]['skipped']!=expected:
        raise RuntimeError('Unexpected optimizer update count')
    if not smoke:
        for step in range(1250,20001,1250):verify_checkpoint(out/f'checkpoint-{step}',read_json(QUEUE/'resume_identity.json'))
    freeze(out/'complete.json',{'checkpoints':checkpoint_hashes(out),'finite_weights':True,
        'full_state_verified':not smoke,'optimizer_update_counts':rows[-1]})
    if smoke:freeze(RUN/'smoke.json',{'passed':True,'actual_updates':4,'scientific_result':False})


def wait_resources(stage):
    required=50000 if stage in ['00_smoke','01_train'] else 55000
    while True:
        free=int(subprocess.check_output(['nvidia-smi','-i',os.environ.get('CUDA_VISIBLE_DEVICES','1').split(',')[0],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        disk=shutil.disk_usage(ROOT).free/2**30
        if free>=required and disk>=12:return
        write_json(QUEUE/'status.json',{'state':'waiting_for_resources','stage':stage,'free_gpu_mib':free,
            'required_gpu_mib':required,'disk_free_gib':disk,'pid':os.getpid(),'updated':now()})
        time.sleep(30)


def stage(label,script,*args):
    verify();marker=RUN/'stages'/(label+'.json')
    if marker.exists():return
    wait_resources(label)
    write_json(QUEUE/'status.json',{'state':'running','stage':label,'pid':os.getpid(),'updated':now()})
    log=RUN/'logs'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        subprocess.run([sys.executable,str(script),'--run',str(RUN),*map(str,args)],cwd=ROOT,
            check=True,stdout=f,stderr=subprocess.STDOUT)
    freeze(marker,{'completed':now(),'log':str(log)})


def endpoint():
    from student import load_pipeline,generate_images
    pipe,cp=load_pipeline(RUN,BRANCH,RUN/'students'/BRANCH/'snapshot-20000')
    generate_images(RUN,BRANCH,pipe,cp,label='endpoint20000')


def summary():
    from key_summary import summary as key_report
    key_report(RUN,BRANCH)


def supervise():
    import fcntl
    with (QUEUE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            stage('00_smoke',Path(__file__),'--operation','smoke')
            stage('01_train',Path(__file__),'--operation','train')
            stage('05_fixed20000',ROOT/'fixed_endpoint.py','--branch',BRANCH)
            stage('06_test',HARNESS/'student.py','--stage','generate','--branch',BRANCH)
            stage('08_score',HARNESS/'semantic_teacher.py','--stage','score','--directories',BRANCH)
            stage('09_quality',HARNESS/'utility.py','--labels',BRANCH)
            stage('10_summary',Path(__file__),'--operation','summary')
            stage('11_acceptance32',ROOT/'acceptance.py','--branch',BRANCH)
            write_json(QUEUE/'status.json',{'state':'completed','updated':now()})
        except Exception as e:
            write_json(QUEUE/'status.json',{'state':'failed','error':str(e),'updated':now()});raise


def launch():
    verify()
    proof=read_json(QUEUE/'tests.json')
    if not proof['passed'] or proof['manifest_sha256']!=sha256(QUEUE/'code_manifest.json'):
        raise RuntimeError('Tests must cover frozen code')
    p=QUEUE/'launcher.json'
    if p.exists():
        pid=read_json(p)['pid'];cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and str(ROOT/'campaign.py').encode() in cmd.read_bytes():
            print('ALREADY RUNNING',pid);return
    env=dict(os.environ,BITMARK_RUNTIME_ROOT=str(BASE/'runtime'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','1'),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with (QUEUE/'supervisor.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__)),'--supervise'],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(p,{'pid':proc.pid,'started':now(),'project':'vllm'});print('LAUNCHED',proc.pid)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--launch',action='store_true')
    p.add_argument('--supervise',action='store_true');p.add_argument('--run',type=Path)
    p.add_argument('--operation',choices=['smoke','train','endpoint','summary']);a=p.parse_args()
    if a.initialize:initialize();print('INITIALIZED')
    elif a.launch:launch()
    elif a.supervise:supervise()
    elif a.operation:
        if a.run.resolve()!=RUN.resolve():raise RuntimeError('Unexpected run path')
        verify()
        if a.operation in ['smoke','train']:train(a.operation=='smoke')
        elif a.operation=='endpoint':endpoint()
        else:summary()
    else:p.error('Choose operation')
