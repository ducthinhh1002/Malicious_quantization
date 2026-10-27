"""Detach the reviewed queue exactly once; all experiments remain resumable."""
import os
from pathlib import Path
import subprocess
import sys
import research_queue as rq
from common import read_json,write_json

rq.verify_plan()
record=rq.QUEUE/'launcher.json'
if record.exists():
    prior=read_json(record); cmdline=Path(f"/proc/{prior['pid']}/cmdline")
    if cmdline.exists() and b'research_queue.py' in cmdline.read_bytes() and b'--supervise' in cmdline.read_bytes():
        print(f"Queue already running: PID {prior['pid']}")
        raise SystemExit(0)
env=dict(os.environ, BITMARK_RUNTIME_ROOT='/root/logs/runtime',CUDA_VISIBLE_DEVICES='0',
         PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',
         HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (rq.QUEUE/'supervisor.log').open('a') as stream:
    process=subprocess.Popen([sys.executable,str(rq.ROOT/'research_queue.py'),'--supervise'],
        cwd=rq.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
write_json(record,{'pid':process.pid,'started':rq.now(),'plan':str(rq.QUEUE/'plan.json')})
print(f'Queue launched: PID {process.pid}')
