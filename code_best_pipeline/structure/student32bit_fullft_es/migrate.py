"""Migrate at the next completed periodic checkpoint; retain optimizer and RNG."""
from pathlib import Path
import os,json,re,signal,time,subprocess,shutil,math
ROOT=Path(__file__).resolve().parent
OLD=Path('/var/lib/structure32_student_fullft')
STATE=ROOT/'deployment';STATE.mkdir(exist_ok=True)
def update(stage,**kw):(STATE/'migration.json').write_text(json.dumps({'stage':stage,'updated':time.time(),**kw},indent=2))
def main():
    log=OLD/'deployment/02_full_unet12500.log'
    steps=re.findall(r'(\d+)/12500',log.read_text(errors='replace'));step=int(steps[-1]);target=min(12500,math.ceil(step/1250)*1250)
    checkpoint=OLD/'run/students/full_wm'/f'checkpoint-{target}'
    update('waiting_for_safe_checkpoint',observed_step=step,target_step=target)
    while f'Saved state to {checkpoint}' not in log.read_text(errors='replace'):time.sleep(3)
    assert all((checkpoint/f).exists() for f in ['optimizer.bin','scheduler.bin','random_states_0.pkl'])
    assert (checkpoint/'unet/diffusion_pytorch_model.safetensors').exists()
    # Stop supervisor and trainer after full state is durable. Process titles may be changed.
    targets=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
            if str(OLD).encode() in env.get(b'PYTHONPATH',b'').split(b':'):targets.append(int(p.name))
        except OSError:pass
    for pid in targets:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    time.sleep(3)
    for pid in targets:
        try:
            if 'State:\tZ' not in Path('/proc',str(pid),'status').read_text():os.kill(pid,signal.SIGKILL)
        except (OSError,ProcessLookupError):pass
    (OLD/'deployment/status.json').write_text(json.dumps({'stage':'migrated_to_early_stopping','resume_step':target,'new_root':str(ROOT)}))
    run=ROOT/'run';run.mkdir(exist_ok=True)
    for name in ['assets.json','prompts.json','carrier','images','scores']:
        p=run/name
        if not p.exists():p.symlink_to(OLD/'run'/name,target_is_directory=(OLD/'run'/name).is_dir())
    # scores must be writable independently when generating new results.
    if (run/'scores').is_symlink():
        (run/'scores').unlink();(run/'scores').mkdir()
        (run/'scores/unadapted.json').symlink_to(OLD/'run/scores/unadapted.json')
    # images share only original input/reference folders, not generated full_wm output.
    if (run/'images').is_symlink():
        (run/'images').unlink();(run/'images').mkdir()
        for p in (OLD/'run/images').iterdir():
            if p.name!='full_wm':(run/'images'/p.name).symlink_to(p,target_is_directory=p.is_dir())
    dest=run/'students/full_wm'/checkpoint.name
    if not dest.exists():shutil.copytree(checkpoint,dest,copy_function=os.link)
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',BITMARK_RUNTIME_ROOT='/var/lib/vllm/runtime')
    update('resuming_with_validation',resume_step=target,old_identity=str(OLD/'identity.json'))
    with (STATE/'supervisor.log').open('a') as f:
        subprocess.run(['/var/lib/vllm/env/bin/python','-u',str(ROOT/'campaign.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    update('completed')
if __name__=='__main__':
    try:main()
    except Exception as e:update('failed',error=repr(e));raise
