"""Bounded automatic recovery for the project-owned overnight queue."""
import json,os,subprocess,time,signal
from pathlib import Path
r=Path('/root/workspace/serum'); os.chdir(r)
env=os.environ.copy(); env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
attempt=0
while attempt<3 and not (r/'STOP_CORE_QUEUE').exists():
    attempt+=1
    log=open(r/'logs/core_queue.log','a')
    process=subprocess.Popen(['.venv/bin/python','scripts/queue_core_seed0.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while process.poll() is None:
        record={'state':'running','attempt':attempt,'queue_pid':process.pid,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        (r/'metrics/core_supervisor_status.json').write_text(json.dumps(record,indent=2))
        if (r/'STOP_CORE_QUEUE').exists():
            os.killpg(process.pid,signal.SIGINT)
            process.wait()
            break
        time.sleep(20)
    code=process.returncode; log.close()
    if code==0:
        state='completed'
        break
    if (r/'STOP_CORE_QUEUE').exists():
        state='stopped'
        break
    print(f'Queue exited {code}; recovery attempt {attempt}/3',flush=True)
    state='retries_exhausted'
    time.sleep(20)
else:
    state='stopped' if (r/'STOP_CORE_QUEUE').exists() else 'retries_exhausted'
(r/'metrics/core_supervisor_status.json').write_text(json.dumps({'state':state,'attempts':attempt,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},indent=2))
