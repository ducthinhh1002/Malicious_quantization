"""Read-only checkpoints; paired baseline/0.85 validation beside an active run."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
HARNESS=ROOT/'Malicious_quantization/bitmark_lora'
sys.path.insert(0,str(HARNESS))
from common import read_json,freeze,sha256


def prepare(source,step):
    checkpoint=source/'teacher_fp/candidates/kappa_2.0'/f'checkpoint-{step}'
    proof=read_json(checkpoint/'checkpoint.json')
    if proof['step']!=step:raise RuntimeError('Checkpoint step mismatch')
    for name,digest in proof['files'].items():
        if sha256(checkpoint/name)!=digest:raise RuntimeError('Checkpoint hash mismatch')
    run=ROOT/'parallel_scale085'/f'step_{step}'
    run.mkdir(parents=True,exist_ok=True)
    c=read_json(source/'config.json')
    # Both worker variants use identical execution batches; optimizer is unused.
    c.update(generation_batch=4,score_batch=4)
    freeze(run/'config.json',c)
    def link(src,dst):
        src,dst=Path(src),Path(dst)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha256(src)!=sha256(dst):raise RuntimeError('Diagnostic input changed')
            return
        os.link(src,dst)
    for relative in ['assets.json','prompts.json','carrier','result/threshold_teacher.json','images/teacher_clean_validation']:
        src=source/relative;dst=run/relative
        if src.is_dir():shutil.copytree(src,dst,copy_function=link,dirs_exist_ok=True)
        else:link(src,dst)
    adapter=run/'adapters/source'
    for name in ['adapter_config.json','adapter_model.safetensors']:link(checkpoint/name,adapter/name)
    freeze(run/'lineage.json',{'source_run':str(source),'checkpoint':str(checkpoint),
           'checkpoint_manifest_sha256':sha256(checkpoint/'checkpoint.json'),
           'source_config_sha256':sha256(source/'config.json'),
           'execution_overrides':{'generation_batch':4,'score_batch':4},
           'use':'validation diagnostic only; no change to main teacher/student selection','test_used':False})
    return run


def worker(run,scale):
    from semantic_teacher_train import merge_teacher
    from threshold_quality import build_adapter
    from semantic_teacher import generate_set,score,numerical_policy
    from semantic_report import detection_metrics,teacher_quality
    # Standalone workers must inherit the original run's strict FP32 policy
    # before merge equivalence; do not silently use CUDA's TF32 defaults.
    numerical_policy()
    c=read_json(run/'config.json');source=run/'adapters/source'
    tag='baseline' if scale==1. else f'scale{scale:.2f}'.replace('.','')
    folder=run/'adapters'/tag
    if scale==.85:build_adapter(run,folder,[source],[1.])
    elif scale!=1.:
        from safetensors.torch import load_file,save_file
        import torch
        folder.mkdir(parents=True,exist_ok=True)
        state=load_file(str(source/'adapter_model.safetensors'))
        if not any('.lora_B.' in k for k in state):raise RuntimeError('No B factors')
        result={k:(v.float()*scale if '.lora_B.' in k else v.float()).contiguous() for k,v in state.items()}
        if not all(torch.isfinite(v).all() for v in result.values()):raise RuntimeError('Nonfinite scale')
        target=folder/'adapter_model.safetensors'
        if not target.exists():save_file(result,str(target))
        else:
            existing=load_file(str(target))
            if set(existing)!=set(result) or any(not torch.equal(existing[k],v) for k,v in result.items()):raise RuntimeError('Scaled weights changed')
        freeze(folder/'adapter_config.json',read_json(source/'adapter_config.json'))
        freeze(folder/'scale_provenance.json',{'scale':scale,'source_sha256':sha256(source/'adapter_model.safetensors'),'test_used':False})
    else:
        folder.mkdir(parents=True,exist_ok=True)
        for name in ['adapter_config.json','adapter_model.safetensors']:
            target=folder/name
            if not target.exists():os.link(source/name,target)
            if sha256(target)!=sha256(source/name):raise RuntimeError('Baseline adapter mismatch')
    model=merge_teacher(run,folder)
    label=f'candidate_parallel_{tag}_validation'
    generate_set(run,model,'validation',label);score(run,[label])
    threshold=read_json(run/'result/threshold_teacher.json')['threshold']
    result={'scale':scale,'label':label,'metrics':detection_metrics(run,label,threshold,c['validation_n']),
            'quality':teacher_quality(run,label),'test_used':False}
    freeze(run/'result'/f'{tag}.json',result)
    print('DONE',tag,result['metrics']['detection_rate'],result['quality']['paired_lpips'],flush=True)


def compare(run):
    from semantic_report import score_rows
    from threshold_quality import paired_gate
    b=read_json(run/'result/baseline.json');v=read_json(run/'result/scale085.json')
    n=read_json(run/'config.json')['validation_n']
    threshold=read_json(run/'result/threshold_teacher.json')['threshold']
    gate=paired_gate(b['quality'],v['quality'],[r['score'] for r in score_rows(run,b['label'],n)],
                     [r['score'] for r in score_rows(run,v['label'],n)],threshold)
    gain=1-v['quality']['paired_lpips']/b['quality']['paired_lpips']
    summary={'baseline':b,'scale085':v,'paired_gate':gate,'relative_lpips_improvement':gain,
             'meets_15pct_lpips_goal':gain>=.15,'test_used':False,'final_acceptance':False,
             'note':'Validation diagnostic; main frozen campaign and student acceptance are unchanged.'}
    freeze(run/'result/comparison.json',summary)
    print('COMPARISON',run.name,'LPIPS improvement',gain,'gate',gate,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--worker',type=Path)
    p.add_argument('--scale',type=float,choices=[1.,.85,.75,.65]);a=p.parse_args()
    if a.worker:
        if a.scale is None:p.error('--scale required')
        worker(a.worker,a.scale);return
    if a.source is None:p.error('--source required')
    destination=ROOT/'parallel_scale085';destination.mkdir(exist_ok=True)
    # A separate lock prevents duplicate supervisors or writes to the same models.
    import fcntl
    with (destination/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for step in (2500,3000):
            checkpoint=a.source/'teacher_fp/candidates/kappa_2.0'/f'checkpoint-{step}/checkpoint.json'
            while not checkpoint.exists():
                print('Waiting for completed checkpoint',step,flush=True);time.sleep(10)
            run=prepare(a.source,step)
            processes=[];streams=[]
            try:
                for scale in (1.,.85):
                    stream=(run/f'worker_{scale}.log').open('a');streams.append(stream)
                    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(run),'--scale',str(scale)],
                                           stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
                    processes.append(child)
                    print('STARTED',step,scale,'pid',child.pid,flush=True)
                codes=[child.wait() for child in processes]
                if any(codes):raise RuntimeError(f'Parallel workers failed: {codes}; see {run}')
            finally:
                for child in processes:
                    if child.poll() is None:child.terminate();child.wait()
                for stream in streams:stream.close()
            compare(run)
        freeze(destination/'complete.json',{'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[2500,3000]})


if __name__=='__main__':main()
