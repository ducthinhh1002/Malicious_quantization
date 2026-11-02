"""Clean control, full-trajectory improvement, and matched continuation control."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import transfer_campaign as shared
from common import read_json, write_json, freeze, sha256

ROOT, SOURCE = shared.ROOT, shared.SOURCE
QUEUE = ROOT / 'native_quality_20261025'
ORDER = ['clean_control', 'native_full', 'continuation']
COMMON = {**shared.COMMON, 'transfer':False, 'aux_gradient_fraction':.25,
          'native_every':5, 'native_lpips_weight':1., 'native_pixel_weight':.1,
          'native_semantic_weight':.8, 'native_gradient':'full original PNDM trajectory',
          'native_rgb':'RGB8 forward with explicitly declared STE backward'}
SPECS = {name:{**COMMON,'native':name=='native_full'} for name in ORDER}
FILES = ['native_quality_campaign.py','native_quality_refine.py','native_quality_tests.py',
         'native_quality_audit.py','transfer_campaign.py','transfer_refine.py','transfer_proxy.py']
shared.QUEUE, shared.ORDER, shared.SPECS = QUEUE, ORDER, SPECS
now, verify = shared.now, shared.verify


def prepare(name):
    run=QUEUE/name;run.mkdir(parents=True,exist_ok=True)
    spec=SPECS[name];config=read_json(SOURCE/'config.json')
    if name!='clean_control':
        config.update(teacher_train_steps=spec['teacher_steps'],teacher_lr=spec['teacher_lr'],
                      teacher_checkpoint_every=spec['checkpoint_every'])
    freeze(run/'config.json',config);freeze(run/'refine_spec.json',spec)
    for rel in ['assets.json','prompts.json','carrier','result/threshold_teacher.json',
                'result/threshold_student.json','result/threshold.json','teacher_fp/calibration_target.json',
                'images/teacher_clean_train','images/teacher_clean_validation','images/teacher_clean_test',
                'images/unadapted','scores/unadapted.json']:
        shared.q.share(SOURCE/rel,run/rel)
    source=shared.base.checkpoint(SOURCE)
    freeze(run/'teacher_choice.json',{
        'method':name,'source_checkpoint':str(source),
        'source_adapter_sha256':sha256(source/'adapter_model.safetensors'),
        'teacher_kappa':1.,'initial_effective_scale':1.,'refinement_steps':0 if name=='clean_control' else 600,
        'teacher_selection':'fixed final step','test_used_for_selection':False,'quality_stopping_gates':False,
        'dataset_condition':'clean' if name=='clean_control' else 'watermarked',
        'student_branch_name_note':'clean control uses wm branch label only to keep identical upstream training command',
        'student_trainer':'unchanged ordinary rank32 attention_conv / 7500 updates / batch4 / 5000 training images',
        'method_definition':('full-trajectory PNDM50 LPIPS + pixel + original coherent calibrated RGB shift; '
                            'original threshold_cfg_v1 gradient retained' if name=='native_full' else name),
        'limitations':'One seed and reused exploratory test. Cached clean reference has errors. '
                      'Mixed precision unmerged training vs FP16 merged production has rounding differences. '
                      'Full native loss preserves owner detection, not proof of transfer to student.'})
    if name=='clean_control':
        for split in ['train','validation','test']:
            shared.q.share(SOURCE/'images'/f'teacher_clean_{split}',run/'images'/f'teacher_wm_{split}')
        for stage in ['01_teacher','02_data']:
            freeze(run/'stages'/f'{stage}.json',{'completed':True,'operation':'reuse frozen clean teacher images; no watermark teacher'})


def initialize():
    QUEUE.mkdir(parents=True,exist_ok=True)
    freeze(QUEUE/'code_manifest.json',{name:sha256(ROOT/name) for name in FILES})
    freeze(QUEUE/'specs.json',SPECS)
    freeze(QUEUE/'plan.json',{
        'order':ORDER,'source_run':str(SOURCE),
        'student_threshold_sha256':sha256(SOURCE/'result/threshold_student.json'),
        'calibration_target_sha256':sha256(SOURCE/'teacher_fp/calibration_target.json'),
        'prior_campaign_status_sha256':sha256(ROOT/'minhnqn_refine_20261024/status.json'),
        'original_teacher_objective_preserved':True,'final_student_objective_unchanged':True,
        'sampler':'original PNDMScheduler, 50 requested steps, CFG7.5, same generation seeds',
        'selection':'teacher fixed last step; student minimum validation MSE; never test',
        'gates':'numerical/integrity/storage only; no scientific outcome early stopping',
        'clean_control':'same wm-labelled branch trained on linked clean images; report labels condition explicitly',
        'new_native_every':5,'native_full_gradient':True,
        'controls':'clean student, same-budget original continuation, prior quality/gradient refinements',
        'storage':'share existing clean/unadapted images; only derived model exports removed by existing verified helper',
        'evaluation':'all 1000 images, fixed detector; FID/KID/CLIP/LPIPS plus pixel diagnostics and blind human-review panel'})
    verify()
    for name in ORDER:prepare(name)


def wait_resources(job,stage):
    need=90000 if stage in ['00_smoke','01_teacher'] else 80000
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
                         '--format=csv,noheader,nounits'],text=True).strip())
        disk=shutil.disk_usage(ROOT).free/2**30
        minimum=12 if stage=='02_data' else 3
        if free>=need and disk>=minimum:return
        write_json(QUEUE/'status.json',{'state':'waiting_for_resources','job':job,'stage':stage,
            'free_gpu_mib':free,'required_gpu_mib':need,'free_disk_gib':disk,
            'required_disk_gib':minimum,'pid':os.getpid(),'updated':now()})
        time.sleep(30)


def call(run,label,script,*args):
    if (run/'stages'/f'{label}.json').exists():return
    wait_resources(run.name,label)
    return shared.call(run,label,script,*args)


def job(name):
    run=QUEUE/name
    if (run/'stages/15_visual_audit.json').exists():return
    if name!='clean_control':
        call(run,'00_smoke',ROOT/'native_quality_refine.py','--smoke')
        if not read_json(run/'smoke.json')['passed']:raise RuntimeError('Actual-model smoke did not pass')
    old=shared.call
    def route(run,label,script,*args):
        if (run/'stages'/f'{label}.json').exists():return
        wait_resources(run.name,label)
        if Path(script).name=='transfer_campaign.py':script=Path(__file__)
        if label=='01_teacher':script=ROOT/'native_quality_refine.py'
        return old(run,label,script,*args)
    shared.call=route
    try:shared.job(name)
    finally:shared.call=old
    call(run,'15_visual_audit',ROOT/'native_quality_audit.py')


def supervise():
    from assets import fcntl
    with (QUEUE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify()
            for name in ORDER:job(name)
            shared.comparison()
            write_json(QUEUE/'status.json',{'state':'completed','updated':now()})
        except Exception as error:
            write_json(QUEUE/'status.json',{'state':'failed','error':str(error),'updated':now()})
            raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--initialize',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--supervise',action='store_true');parser.add_argument('--run',type=Path)
    parser.add_argument('--operation',choices=['data','teacher_report','summary'])
    args=parser.parse_args()
    if args.initialize:initialize();print('INITIALIZED');return
    verify()
    if args.launch:
        proof=read_json(QUEUE/'unit_tests.json')
        if not proof['passed'] or proof['manifest_sha256']!=sha256(QUEUE/'code_manifest.json'):
            raise RuntimeError('Tests must cover this frozen manifest')
        previous=QUEUE/'launcher.json'
        if previous.exists():
            pid=read_json(previous)['pid'];proc=Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'native_quality_campaign.py' in proc.read_bytes():
                print('Already queued',pid);return
        env=dict(os.environ,BITMARK_RUNTIME_ROOT='/root/logs/runtime',CUDA_VISIBLE_DEVICES='0',
            PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        with (QUEUE/'supervisor.log').open('a') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__)),'--supervise'],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(previous,{'pid':process.pid,'queued':now()});print('QUEUED',process.pid);return
    if args.supervise:supervise();return
    if args.operation=='data':
        from semantic_teacher import numerical_policy
        numerical_policy();shared.data(args.run)
    elif args.operation=='teacher_report':shared.teacher_report(args.run)
    elif args.operation=='summary':shared.base.student_summary(args.run)
    else:parser.error('Choose an operation')


if __name__=='__main__':main()
