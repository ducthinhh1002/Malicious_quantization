"""Prioritize a continuous10000-update student, then restart balanced_native."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import shutil
import time
import transfer_campaign as shared
from common import read_json,write_json,freeze,sha256,verify_vendor

ROOT=shared.ROOT
PRIOR=ROOT/'pareto_quality_20261025'
SOURCE=PRIOR/'projected_native'
QUEUE=ROOT/'projected_student10000_20261025'
RUN=QUEUE/'student10000'
BRANCH=shared.q.BRANCH
FILES=['student10000_campaign.py','student10000_replay_driver.py']
now=shared.now


def verify():
    shared.base.verify_code();shared.q.verify_plan();verify_vendor()
    plan=read_json(QUEUE/'plan.json')
    for name,digest in read_json(QUEUE/'code_manifest.json').items():
        if sha256(ROOT/name)!=digest:raise RuntimeError('New campaign code changed: '+name)
    for rel,digest in plan['source_provenance'].items():
        if sha256(SOURCE/rel)!=digest:raise RuntimeError('Source artifact changed: '+rel)
    for name,digest in read_json(PRIOR/'code_manifest.json').items():
        if sha256(ROOT/name)!=digest:raise RuntimeError('Original queue code changed: '+name)


def initialize():
    if not (SOURCE/'complete.json').exists():raise RuntimeError('Source student evaluation incomplete')
    QUEUE.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    config=read_json(SOURCE/'config.json')
    config.update(max_train_steps=10000,checkpoint_steps=[10000])
    freeze(RUN/'config.json',config)
    reference='students/'+BRANCH+'/snapshot-7500/adapter_model.safetensors'
    provenance={rel:sha256(SOURCE/rel) for rel in ['config.json','prompts.json','assets.json',
        'carrier/complete.json','result/threshold_student.json','result/threshold_teacher.json',
        'result/student_summary.json','images/teacher_wm_train/image_hashes.json',reference]}
    freeze(QUEUE/'plan.json',{'order':['student10000','resume_original_balanced_native_queue'],
        'source_provenance':provenance,'source':str(SOURCE),'source_reference_adapter':reference,
        'request':'Train projected_native student through10000, then balanced_native',
        'training':'same frozen upstream trainer, clean SD1.4 initialization, identical seed/data/LR/batch; uninterrupted optimizer to10000',
        'reason_for_replay':'Source inference snapshots contain no optimizer/RNG; exact state resume unavailable',
        'teacher_unchanged':True,'final_student_objective_unchanged':True,
        'evaluation':'fixed10000 endpoint, existing paired1000-image validation/test; compare original7500 separately',
        'full_state_checkpoints':[2500,5000,7500,10000],
        'selection':'fixed10000 requested exposure budget; no selection on test/detector',
        'limitations':'Adaptive extension after seeing7500 test; one training seed. Replay equality audited at7500.'})
    freeze(QUEUE/'code_manifest.json',{name:sha256(ROOT/name) for name in FILES})
    for rel in ['assets.json','prompts.json','carrier','result/threshold.json',
        'result/threshold_teacher.json','result/threshold_student.json',
        'images/teacher_wm_train','images/teacher_wm_validation','images/teacher_wm_test',
        'images/teacher_clean_validation','images/teacher_clean_test','images/unadapted',
        'scores/unadapted.json','validation_cache']:
        shared.q.share(SOURCE/rel,RUN/rel)
    freeze(RUN/'teacher_choice.json',{**read_json(SOURCE/'teacher_choice.json'),
        'method':'projected_native_student10000','student_steps':10000,
        'teacher_is_identical_to':str(SOURCE),'refinement_steps':600,
        'student_trainer':'unchanged ordinary rank32 attention_conv,10000updates,batch4,LR1e-4',
        'student_initialization':'same clean SD1.4/seed; replay because original optimizer was not saved',
        'test_used_for_selection':False,
        'adaptive_followup':'10000 budget requested after viewing7500 test; confirm on fresh holdout later'})
    verify()


def train(run):
    from student import training_command,branches,checkpoint_hashes
    import torch
    from safetensors.torch import load_file
    verify();config=read_json(run/'config.json');assets=read_json(run/'assets.json')
    branch=next(b for b in branches(config) if b['name']==BRANCH)
    out=run/'students'/BRANCH;out.mkdir(parents=True,exist_ok=True)
    expected=read_json(run/'images/teacher_wm_train/image_hashes.json')
    if len(expected)!=5000 or any(sha256(run/'images/teacher_wm_train'/v['file_name'])!=v['sha256'] for v in expected):
        raise RuntimeError('Training dataset integrity failure')
    original=training_command(config,assets,run,branch)
    # Snapshot7500 audits the replay; evaluation remains fixed at10000.
    original[original.index('--snapshot_steps')+1:]=['7500','10000']
    original[original.index('--checkpointing_steps')+1]='2500'
    command=[sys.executable,str(ROOT/'student10000_replay_driver.py'),
        '--trainer',original[1],'--reference',str(SOURCE/read_json(QUEUE/'plan.json')['source_reference_adapter']),
        '--proof',str(run/'replay_7500.json'),*original[2:]]
    freeze(out/'protocol.json',{'command':command,'branch':branch,'training_images':expected,
        'selection':'fixed10000 requested budget; original7500 comparison retained separately',
        'state_checkpoints':'accelerator.save_state every2500, including model/optimizer/scaler/RNG'})
    if (out/'complete.json').exists():
        if read_json(out/'complete.json')['checkpoints']!=checkpoint_hashes(out):raise RuntimeError('Trained weights changed')
        return
    if any(out.glob('checkpoint-*')):
        raise RuntimeError('Interrupted replay has saved state; inspect and explicitly resume at an epoch boundary rather than overwrite it')
    subprocess.run(command,cwd=shared.q.HARNESS,check=True)
    weights=load_file(str(out/'adapter_model.safetensors'))
    if any(not torch.isfinite(t).all() for t in weights.values()):raise RuntimeError('Nonfinite student weights')
    audit=read_json(out/'optimization_audit.json')
    if audit['applied']!=10000 or audit['skipped']!=0:raise RuntimeError('Student did not apply10000updates')
    if not (run/'replay_7500.json').exists():raise RuntimeError('Missing replay audit')
    for step in [2500,5000,7500,10000]:
        folder=out/f'checkpoint-{step}'
        if not folder.exists() or not any(folder.glob('optimizer*')):raise RuntimeError('Full optimizer checkpoint missing')
    write_json(out/'complete.json',{'checkpoints':checkpoint_hashes(out),'finite_weights':True,
        'weights_changed_verified':True,'optimizer_update_counts':audit})


def wait_resources(stage):
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        disk=shutil.disk_usage(ROOT).free/2**30
        if free>=80000 and disk>=14:return
        write_json(QUEUE/'status.json',{'state':'waiting_for_resources','stage':stage,
            'free_gpu_mib':free,'required_gpu_mib':80000,'free_disk_gib':disk,'required_disk_gib':14,'updated':now()})
        time.sleep(30)


def call(label,script,*args):
    verify();marker=RUN/'stages'/(label+'.json')
    if marker.exists():return
    wait_resources(label)
    write_json(QUEUE/'status.json',{'state':'running','stage':label,'pid':os.getpid(),'updated':now()})
    log=RUN/'queue_logs'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable,str(script),'--run',str(RUN),*map(str,args)],cwd=ROOT,
            check=True,stdout=stream,stderr=subprocess.STDOUT)
    freeze(marker,{'completed':now(),'log':str(log)})


def supervise():
    from assets import fcntl
    with (QUEUE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify()
            pid=read_json(PRIOR/'launcher.json')['pid'];p=Path(f'/proc/{pid}/cmdline')
            if p.exists() and b'pareto_quality_campaign.py' in p.read_bytes():raise RuntimeError('Original queue must be stopped before priority extension')
            call('01_train',Path(__file__),'--operation','train')
            h=shared.q.HARNESS
            call('02_validation',h/'validation.py','--stage','evaluate','--branch',BRANCH)
            label='validation_'+BRANCH+'_step10000'
            call('03_validation_score',h/'semantic_teacher.py','--stage','score','--directories',label)
            call('04_validation_quality',h/'utility.py','--split','validation','--labels',label)
            call('05_select',h/'validation.py','--stage','select','--branch',BRANCH)
            call('06_test',h/'student.py','--stage','generate','--branch',BRANCH)
            call('07_score',h/'semantic_teacher.py','--stage','score','--directories',BRANCH)
            call('08_quality',h/'utility.py','--labels',BRANCH)
            call('09_summary',Path(__file__),'--operation','summary')
            call('10_visual_audit',ROOT/'native_quality_audit.py')
            a=read_json(SOURCE/'result/student_summary.json');b=read_json(RUN/'result/student_summary.json')
            freeze(QUEUE/'comparison.json',{'source7500':{'detection':a['metrics']['detection_rate'],'lpips':a['lpips'],
                **{k:v for k,v in a['utility'].items() if not isinstance(v,(dict,list))}},
                'new10000':{'detection':b['metrics']['detection_rate'],'lpips':b['lpips'],
                **{k:v for k,v in b['utility'].items() if not isinstance(v,(dict,list))}},
                'replay7500':read_json(RUN/'replay_7500.json'),'test_used_for_checkpoint_selection':False,
                'limitation':'adaptive follow-up, reused test, one seed'})
            # Its original completed projected job is skipped, then balanced runs.
            subprocess.run([sys.executable,str(ROOT/'pareto_quality_campaign.py'),'--launch'],cwd=ROOT,check=True)
            write_json(QUEUE/'status.json',{'state':'completed','balanced_queue_restarted':True,'updated':now()})
        except Exception as error:
            write_json(QUEUE/'status.json',{'state':'failed','error':str(error),'balanced_queue_restarted':False,'updated':now()})
            raise


def main():
    p=argparse.ArgumentParser()
    for flag in ['initialize','launch','supervise']:p.add_argument('--'+flag,action='store_true')
    p.add_argument('--operation',choices=['train','summary']);p.add_argument('--run',type=Path)
    args=p.parse_args()
    if args.initialize:initialize();print('INITIALIZED');return
    verify()
    if args.launch:
        path=QUEUE/'launcher.json'
        if path.exists():
            pid=read_json(path)['pid'];proc=Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'student10000_campaign.py' in proc.read_bytes():print('Already launched',pid);return
        env=dict(os.environ,BITMARK_RUNTIME_ROOT='/root/logs/runtime',CUDA_VISIBLE_DEVICES='0',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
        with (QUEUE/'supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__)),'--supervise'],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(path,{'pid':child.pid,'queued':now()});print('LAUNCHED',child.pid);return
    if args.supervise:supervise();return
    if args.operation=='train':train(args.run)
    elif args.operation=='summary':shared.base.student_summary(args.run)
    else:p.error('Choose operation')


if __name__=='__main__':main()
