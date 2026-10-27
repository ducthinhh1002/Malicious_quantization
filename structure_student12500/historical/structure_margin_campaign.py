"""Full native structure/margin experiment, then a structure-loss ablation."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import transfer_campaign as shared
from common import read_json, write_json, freeze, sha256

ROOT,SOURCE=shared.ROOT,shared.SOURCE
QUEUE=ROOT/'structure_margin_20261026'
DEPENDENCY=ROOT/'pareto_quality_20261025'
BASELINE=ROOT/'kappa1_manojb_5k/students/k1_evolution'
ORDER=['structure_margin_native','margin_native']
COMMON={**shared.COMMON,'transfer':False,'native_every':2,'backward_calls':10,
    'quality_fraction':1.,'score_fraction':1.,'native_batch':4,'native_micro_batch':2,'max_gradient_gain':100.,'pixel_weight':.1,
    'score_reference':'max(source k1 paired native RGB score, fixed tau + sigma)',
    'quality_stopping_gates':False}
SPECS={name:{**COMMON,'project_step':True,'structure_weight':.25 if name=='structure_margin_native' else 0.} for name in ORDER}
FILES=['structure_margin_campaign.py','structure_margin_refine.py','structure_margin_objectives.py',
       'structure_margin_tests.py','structure_margin_audit.py','pareto_quality_refine.py','pareto_step.py','pareto_quality_tests.py',
       'transfer_campaign.py','transfer_refine.py','transfer_proxy.py',
       'native_quality_refine.py','native_quality_audit.py']
shared.QUEUE,shared.ORDER,shared.SPECS=QUEUE,ORDER,SPECS
now=shared.now


def verify():
    shared.verify()
    plan=read_json(QUEUE/'plan.json')
    for rel,digest in plan['baseline_provenance'].items():
        if sha256(BASELINE/rel)!=digest:raise RuntimeError('Baseline provenance changed: '+rel)
    source=shared.base.checkpoint(SOURCE)
    if sha256(source/'adapter_model.safetensors')!=plan['source_adapter_sha256']:
        raise RuntimeError('Source teacher adapter changed')


def initialize():
    choice=read_json(BASELINE/'teacher_choice.json')
    if choice['teacher_kappa']!=1. or choice['vector']!=[1.]*6:
        raise RuntimeError('Expected exact kappa1, scale1 source baseline')
    source=shared.base.checkpoint(SOURCE)
    if Path(choice['source_checkpoint'])!=source:raise RuntimeError('Baseline lineage mismatch')
    prompts=read_json(SOURCE/'prompts.json')['train'];config=read_json(SOURCE/'config.json')
    generation=read_json(BASELINE/'images/teacher_wm_train/generation.json')
    if generation['prompts_sha256']!=sha256(SOURCE/'prompts.json'):
        raise RuntimeError('Baseline training prompts differ')
    if generation['seeds']!=[config['seed']*1000000+p['image_id'] for p in prompts]:
        raise RuntimeError('Baseline training seeds differ')
    QUEUE.mkdir(parents=True,exist_ok=True)
    freeze(QUEUE/'code_manifest.json',{name:sha256(ROOT/name) for name in FILES})
    freeze(QUEUE/'specs.json',SPECS)
    freeze(QUEUE/'plan.json',{'order':ORDER,'dependency':str(DEPENDENCY),'source_run':str(SOURCE),
        'source_adapter_sha256':sha256(source/'adapter_model.safetensors'),
        'student_threshold_sha256':sha256(SOURCE/'result/threshold_student.json'),
        'calibration_target_sha256':sha256(SOURCE/'teacher_fp/calibration_target.json'),
        'baseline_provenance':{rel:sha256(BASELINE/rel) for rel in [
            'teacher_choice.json','images/teacher_wm_train/generation.json','images/teacher_wm_train/image_hashes.json']},
        'original_teacher_loss_unchanged':True,'final_student_unchanged':True,
        'sampler':'original PNDM50 CFG7.5, 51 forward calls; backward last10 calls only',
        'precision':'BF16 training denoiser, FP32 sampler; original FP16 merged production',
        'gradient_method':'batch4 quality and squared per-image score deficits; separately norm-balanced; project actual AdamW displacement',
        'constraints':'original loss, batch mean squared score deficit, LPIPS+.1pixel+structure_weight*(1-MS-SSIM)',
        'limitations':'Truncated gradients, first-order approximation, per-prompt teacher constraints do not guarantee student detection. Adam moments retain proposal history.',
        'selection':'teacher fixed final600; student min validation MSE at2500/5000/7500',
        'gates':'numerical/integrity/resources only; never stop due to poor scientific metrics',
        'student':'ordinary SD1.4 rank32 attention_conv, 5000train/1000val/1000test,7500updates,batch4',
        'reuse':'hard-linked clean/unadapted/baseline references; verified merged teacher export in bounded /dev/shm staging, persistent adapters and image hashes',
        'evaluation':'same fixed threshold; detection, paired LPIPS, FID/KID/CLIP, visual diagnostic/panel',
        'test_caveat':'existing repeatedly viewed exploratory test; fresh holdout/multiple seeds needed for final claims'})
    verify()
    for name in ORDER:
        run=QUEUE/name;run.mkdir(parents=True,exist_ok=True);spec=SPECS[name]
        config=read_json(SOURCE/'config.json')
        config.update(teacher_train_steps=spec['teacher_steps'],teacher_lr=spec['teacher_lr'],teacher_checkpoint_every=spec['checkpoint_every'])
        freeze(run/'config.json',config);freeze(run/'refine_spec.json',spec)
        for rel in ['assets.json','prompts.json','carrier','result/threshold_teacher.json',
            'result/threshold_student.json','result/threshold.json','teacher_fp/calibration_target.json',
            'images/teacher_clean_train','images/teacher_clean_validation','images/teacher_clean_test',
            'images/unadapted','scores/unadapted.json']:
            shared.q.share(SOURCE/rel,run/rel)
        shared.q.share(BASELINE/'images/teacher_wm_train',run/'images/baseline_wm_train')
        freeze(run/'teacher_choice.json',{'method':name,'source_checkpoint':str(source),
            'source_adapter_sha256':sha256(source/'adapter_model.safetensors'),'teacher_kappa':1.,
            'initial_effective_scale':1.,'refinement_steps':600,'teacher_selection':'fixed final step',
            'test_used_for_selection':False,'quality_stopping_gates':False,'dataset_condition':'watermarked',
            'backward_calls':10,'student_trainer':'unchanged ordinary rank32 attention_conv / 7500 updates / batch4',
            'limitations':'Local truncated-gradient descent constraints are not a downstream detection guarantee'})


def wait_dependency():
    while True:
        status=read_json(DEPENDENCY/'status.json')
        if status['state']=='completed':return
        if status['state']=='failed':raise RuntimeError('Prior native campaign failed; inspect before resuming')
        pid=read_json(DEPENDENCY/'launcher.json')['pid'];proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or b'pareto_quality_campaign.py' not in proc.read_bytes():
            raise RuntimeError('Prior campaign supervisor absent before completion')
        write_json(QUEUE/'status.json',{'state':'waiting_for_dependency','dependency':str(DEPENDENCY),
            'dependency_job':status.get('job'),'dependency_stage':status.get('stage'),
            'pid':os.getpid(),'updated':now()})
        time.sleep(30)


def wait_resources(name,stage):
    need=100000 if stage in ['00_smoke','01_teacher'] else 80000
    minimum=6 if stage=='02_data' else (8 if stage=='00_smoke' else 3)
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        disk=shutil.disk_usage(ROOT).free/2**30
        ram_ok=True
        if stage=='02_data':
            memory={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
            ram_ok=shutil.disk_usage('/dev/shm').free>=10*2**30 and memory['MemAvailable']>=24*2**30
        if free>=need and disk>=minimum and ram_ok:return
        write_json(QUEUE/'status.json',{'state':'waiting_for_resources','job':name,'stage':stage,
            'free_gpu_mib':free,'required_gpu_mib':need,'free_disk_gib':disk,
            'required_disk_gib':minimum,'temporary_merge_ram_ready':ram_ok,'pid':os.getpid(),'updated':now()})
        time.sleep(30)


def job(name):
    run=QUEUE/name
    if (run/'stages/16_structure_audit.json').exists():return
    old=shared.call
    def route(run,label,script,*args):
        verify()
        if (run/'stages'/f'{label}.json').exists():return
        wait_resources(run.name,label)
        if Path(script).name=='transfer_campaign.py':script=Path(__file__)
        if label=='01_teacher':script=ROOT/'structure_margin_refine.py'
        return old(run,label,script,*args)
    route(run,'00_smoke',ROOT/'structure_margin_refine.py','--smoke')
    if not read_json(run/'smoke.json')['passed']:raise RuntimeError('Actual-model smoke failed')
    shared.call=route
    try:shared.job(name)
    finally:shared.call=old
    route(run,'15_visual_audit',ROOT/'native_quality_audit.py')
    route(run,'16_structure_audit',ROOT/'structure_margin_audit.py')


def supervise():
    from assets import fcntl
    with (QUEUE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify();wait_dependency()
            for name in ORDER:job(name)
            shared.comparison()
            write_json(QUEUE/'status.json',{'state':'completed','updated':now()})
        except Exception as error:
            write_json(QUEUE/'status.json',{'state':'failed','error':str(error),'updated':now()});raise


def data(run):
    # The upstream merge verification and original production generation are
    # unchanged. Only the disposable serialized export is placed in tmpfs;
    # no checkpoint/image/result is stored only in RAM.
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import generate_set, load_native
    run = run.resolve()
    if run.parent != QUEUE.resolve() or run.name not in ORDER:
        raise RuntimeError('Unexpected run for temporary merge')
    if (run/'data_complete.json').exists():
        expected=read_json(run/'data_complete.json')['adapter_sha256']
        if sha256(run/'teacher_fp/refined/adapter_model.safetensors')!=expected:
            raise RuntimeError('Completed generation adapter changed')
        return
    tmp_root=Path('/dev/shm/structure_margin_20261026')
    stage=tmp_root/run.name
    if tmp_root.is_symlink() or stage.is_symlink() or stage.resolve().parent!=tmp_root.resolve():
        raise RuntimeError('Unsafe temporary merge path')
    stage.mkdir(parents=True,exist_ok=True)
    folder=run/'teacher_fp/refined'
    for name in ['adapter_model.safetensors','adapter_config.json']:
        source,target=folder/name,stage/name
        if target.exists() and sha256(target)!=sha256(source):
            raise RuntimeError('Stale temporary adapter')
        if not target.exists():shutil.copy2(source,target)
    model=merge_teacher(run,stage)
    freeze(folder/'merge_audit.json',read_json(stage/'merge_test.json'))
    pipe=load_native(model)
    for split in ('train','validation','test'):
        generate_set(run,model,split,f'teacher_wm_{split}',pipe)
    del pipe
    freeze(run/'data_complete.json',{'adapter_sha256':sha256(folder/'adapter_model.safetensors'),
        'temporary_export':str(stage),'merge_audit_sha256':sha256(folder/'merge_audit.json')})
    # Delete only this newly generated reproducible tmpfs export, after all
    # permanent images and the merge proof have been saved successfully.
    if stage.resolve().parent!=Path('/dev/shm/structure_margin_20261026').resolve() or stage.is_symlink():
        raise RuntimeError('Unsafe temporary export cleanup')
    shutil.rmtree(stage)


def main():
    parser=argparse.ArgumentParser()
    for flag in ['initialize','launch','supervise']:parser.add_argument('--'+flag,action='store_true')
    parser.add_argument('--run',type=Path);parser.add_argument('--operation',choices=['data','teacher_report','summary'])
    args=parser.parse_args()
    if args.initialize:initialize();print('INITIALIZED');return
    verify()
    if args.launch:
        proof=read_json(QUEUE/'unit_tests.json')
        if not proof['passed'] or proof['manifest_sha256']!=sha256(QUEUE/'code_manifest.json'):
            raise RuntimeError('Unit tests must cover frozen code manifest')
        previous=QUEUE/'launcher.json'
        if previous.exists():
            pid=read_json(previous)['pid'];proc=Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'structure_margin_campaign.py' in proc.read_bytes():print('Already queued',pid);return
        env=dict(os.environ,BITMARK_RUNTIME_ROOT='/root/logs/runtime',CUDA_VISIBLE_DEVICES='0',
            PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        with (QUEUE/'supervisor.log').open('a') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__)),'--supervise'],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(previous,{'pid':process.pid,'queued':now()});print('QUEUED',process.pid);return
    if args.supervise:supervise();return
    if args.operation=='data':
        from semantic_teacher import numerical_policy
        numerical_policy();data(args.run)
    elif args.operation=='teacher_report':shared.teacher_report(args.run)
    elif args.operation=='summary':shared.base.student_summary(args.run)
    else:parser.error('Choose operation')


if __name__=='__main__':main()
