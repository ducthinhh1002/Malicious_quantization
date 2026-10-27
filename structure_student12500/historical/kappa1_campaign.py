"""Kappa-1 controlled campaign, reusing audited queue machinery and native images."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import research_queue as q
from common import freeze,read_json,write_json,sha256

q.QUEUE=q.ROOT/'kappa1_campaign'
SOURCE=q.ROOT/'run'
original_prepare=q.prepare_run
original_data=q.generate_student_data


def checkpoint(source):
    p=source/'teacher_fp/candidates/kappa_1.0/checkpoint-3000'
    proof=read_json(p/'checkpoint.json')
    assert proof['step']==3000
    for name in ('adapter_model.safetensors','adapter_config.json'):
        if sha256(p/name)!=proof['files'][name]:raise RuntimeError('Changed kappa-1 checkpoint')
    return p
q.checkpoint=checkpoint


def prepare(run,pilot_n=None,shared_student=False):
    result=original_prepare(run,pilot_n,False)
    if shared_student:
        for rel in ('result/threshold_student.json','result/threshold.json','images/teacher_clean_test',
                    'images/unadapted','scores/unadapted.json'):
            q.share(SOURCE/rel,run/rel)
        if run.name=='k1_scale100':
            control=SOURCE;label=q.BASE_BRANCH;kappa=2.
        else:
            control=q.QUEUE/'students/k1_scale100';label=q.BRANCH;kappa=1.
            if not (control/'complete.json').exists():raise RuntimeError('Kappa-1 baseline must finish first')
        q.share(control/'images'/label,run/'images/baseline_student')
        q.share(control/'scores'/f'{label}.json',run/'scores/baseline_student.json')
        freeze(run/'baseline_lineage.json',{'source_run':str(control),'teacher_kappa':kappa,'teacher_scale':1.,
               'selection':read_json(control/'result/selection'/f'{label}.json'),
               'training_protocol_sha256':sha256(control/'students'/label/'protocol.json')})
    return result
q.prepare_run=prepare


def own(run,label,operation,vector=None):
    args=[Path(__file__),'--operation',operation,'--run',run]
    if vector is not None:args+=['--vector',','.join(map(str,vector))]
    q.call(run,label,args)
q.own=own


def data(run,vector):
    """Reuse only identical teacher adapters and paired generation metadata."""
    from semantic_teacher import numerical_policy,generate_set,load_native
    from semantic_assets import snapshot_identity
    from safetensors.torch import load_file
    import torch
    numerical_policy()
    scale=vector[0] if len(set(vector))==1 else None
    if scale not in (1.,.85):return original_data(run,vector)
    tag='kappa_1.0' if scale==1. else 'scale085_k1'
    folder=SOURCE/'teacher_fp/candidates'/tag
    if not (folder/'merge_test.json').exists():return original_data(run,vector)
    model=folder/'merged_pipeline'
    proof=read_json(folder/'merge_test.json')
    if snapshot_identity(model)!=proof['model_hashes']:raise RuntimeError('Reused merged teacher changed')
    expected=q.scale_state(load_file(str(checkpoint(SOURCE)/'adapter_model.safetensors')),
                           read_json(q.QUEUE/'plan.json')['groups'],vector)
    actual=load_file(str(folder/'adapter_model.safetensors'))
    if set(expected)!=set(actual) or any(not torch.equal(expected[k],actual[k]) for k in expected):
        raise RuntimeError('Reused adapter is not requested kappa-1 vector')
    old=SOURCE/'images'/f'candidate_{tag}_validation'
    if old.exists():q.share(old,run/'images/teacher_wm_validation')
    pipe=load_native(model)
    for split in ('train','validation','test'):
        # generate_set checks prompt/seed/sampler/model identities and PNG hashes.
        generate_set(run,model,split,f'teacher_wm_{split}',pipe)
    freeze(run/'reuse.json',{'model':str(model),'validation_source':str(old),
           'adapter_sha256':sha256(folder/'adapter_model.safetensors'),'training_images_reused':False})
q.generate_student_data=data


def specs():
    groups=read_json(q.QUEUE/'plan.json')['groups']
    return [('k1_scale100',[1.]*len(groups)),('k1_scale095',[.95]*len(groups)),
            ('k1_scale085',[.85]*len(groups)),
            ('k1_selective',[.85 if g in ('down_attention','down_conv','up_attention') else 1. for g in groups])]


def initialize():
    if (q.QUEUE/'plan.json').exists():return q.verify_plan()
    plan=q.initialize(SOURCE)
    plan.update(source_kappa=1.,fixed_scales=[1.,.95,.85],
        jobs=['k1_scale100','k1_scale095_and_085','k1_selective'],
        methods='Fixed kappa-1 checkpoint; modest global attenuation and conservative group attenuation',
        historical_baseline_reproduced=False,
        historical_teacher_lpips=.488,current_teacher_lpips=.6762673923512921,
        historical_student_target={'detection':.997,'fid':62.676,'kid':.0069348,'clip':.79334},
        control='k1_scale100; its first report additionally contrasts the previous VM4 kappa-2 baseline')
    write_json(q.QUEUE/'plan.json',plan)
    manifest=read_json(q.QUEUE/'code_manifest.json');manifest[Path(__file__).name]=sha256(Path(__file__))
    write_json(q.QUEUE/'code_manifest.json',manifest)
    freeze(q.QUEUE/'specs.json',dict(specs()))
    return plan


def supervise():
    import fcntl
    with (q.QUEUE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);q.verify_plan()
        jobs=specs()
        def job(item):
            name,vector=item
            own(q.QUEUE,name,'train_'+name,vector)
        try:
            write_json(q.QUEUE/'status.json',{'state':'running','job':jobs[0][0],'pid':os.getpid(),'updated':q.now()})
            job(jobs[0])
            write_json(q.QUEUE/'status.json',{'state':'running','job':'scale095_and_085','pid':os.getpid(),'updated':q.now()})
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(job,jobs[1:3]))
            write_json(q.QUEUE/'status.json',{'state':'running','job':'selective','pid':os.getpid(),'updated':q.now()})
            job(jobs[3])
            write_json(q.QUEUE/'status.json',{'state':'completed','updated':q.now()})
        except Exception as e:
            write_json(q.QUEUE/'status.json',{'state':'failed','error':str(e),'updated':q.now()});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--supervise',action='store_true')
    p.add_argument('--operation');p.add_argument('--run',type=Path);p.add_argument('--vector')
    a=p.parse_args()
    if a.launch:
        initialize()
        previous=q.QUEUE/'launcher.json'
        if previous.exists():
            pid=read_json(previous)['pid'];cmd=Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and b'kappa1_campaign.py' in cmd.read_bytes():
                print('Already running',pid);return
        env=dict(os.environ,BITMARK_RUNTIME_ROOT='/root/logs/runtime',CUDA_VISIBLE_DEVICES='0',
                 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (q.QUEUE/'supervisor.log').open('a') as f:
            proc=subprocess.Popen([sys.executable,str(Path(__file__)),'--supervise'],cwd=q.ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(previous,{'pid':proc.pid,'started':q.now()});print('LAUNCHED',proc.pid);return
    if a.supervise:supervise();return
    q.verify_plan()
    vector=[float(v) for v in a.vector.split(',')] if a.vector else None
    if a.operation.startswith('train_'):
        name=a.operation[len('train_'):]
        if dict(specs()).get(name)!=vector:raise RuntimeError('Unregistered vector')
        q.train_student(name,vector,{'teacher_kappa':1.,'method':name,'source_checkpoint':str(checkpoint(SOURCE))})
    else:q.worker(a.operation,a.run,vector)


if __name__=='__main__':main()
