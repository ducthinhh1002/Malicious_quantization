"""Reuse local-uploaded Structure data/env; wait for GPU memory and serialize with rank4."""
from pathlib import Path
import json,os,subprocess,time,fcntl,hashlib
ROOT=Path(__file__).resolve().parent
SHARED=Path('/var/lib/structure8bit_rank4_refine600')
STATE=ROOT/'deployment';STATE.mkdir(exist_ok=True)
def status(stage,**kw):
    p=STATE/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'stage':stage,'updated':time.time(),'pid':os.getpid(),**kw},indent=2));t.replace(p)
def main():
    with (STATE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        py=SHARED/'env/bin/python'
        while not (SHARED/'DATA_READY.json').exists():status('waiting_for_shared_structure_images');time.sleep(20)
        while True:
            p=SHARED/'deployment/setup_status.json'
            data=json.loads(p.read_text()) if p.exists() else {}
            if data.get('stage')=='waiting_for_rank32_priority':break
            status('waiting_for_shared_environment',dependency_stage=data.get('stage'));time.sleep(20)
        while not (ROOT/'CONTROL_READY.json').exists():status('waiting_for_original_rank32_checkpoint');time.sleep(20)
        source=SHARED/'source'
        cp=source/'teacher_fp/refined/adapter_model.safetensors'
        assert hashlib.sha256(cp.read_bytes()).hexdigest()=='e4cdc8feefe33480266f53191592030f6dcffa40c05238985a1726ac97a73d6f'
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',
                 BITMARK_RUNTIME_ROOT=str(SHARED/'runtime'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        status('initialize_rank32_aug')
        subprocess.run([str(py),str(ROOT/'campaign.py'),'--initialize'],env=env,cwd=ROOT,check=True)
        status('running_rank32_aug')
        subprocess.run([str(py),str(ROOT/'campaign.py'),'--supervise'],env=env,cwd=ROOT,check=True)
        status('completed')
if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=repr(e));raise
