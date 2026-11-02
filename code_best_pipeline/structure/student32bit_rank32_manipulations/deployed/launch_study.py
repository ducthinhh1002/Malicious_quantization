"""Four bounded slots; all seven single-manipulation students run to12500."""
from pathlib import Path
import subprocess,os,json,time,fcntl
ROOT=Path(__file__).resolve().parent
PY='/var/lib/vllm/env/bin/python'
ORDER=['rotation15','crop75','drop96','gaussian005','brightness07','blur15','flip']

def main():
    with (ROOT/'study.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (ROOT/'prepared.json').exists()
        pending=[]
        for name in ORDER:
            q=ROOT/'branches'/name/'runs/structure32_student12500_seed0'
            status=q/'status.json'
            if status.exists() and json.loads(status.read_text()).get('state')=='completed':continue
            pending.append(name)
        slots=[{'gpu':0},{'gpu':1},{'gpu':0},{'gpu':1}]
        finished={}
        while pending or any('child' in s for s in slots):
            for slot in slots:
                if 'child' in slot and slot['child'].poll() is not None:
                    finished[slot['name']]={'returncode':slot['child'].returncode,'gpu':slot['gpu']}
                    slot['log'].close();del slot['child'];del slot['log'];del slot['name']
                if 'child' not in slot and pending:
                    name=pending.pop(0);branch=ROOT/'branches'/name
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(slot['gpu']),PYTHONPATH=str(branch),
                        BITMARK_RUNTIME_ROOT='/var/lib/vllm/runtime',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                    log=(branch/'supervisor.log').open('ab')
                    child=subprocess.Popen([PY,str(branch/'campaign.py'),'--supervise'],cwd=branch,env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    slot.update(child=child,name=name,log=log)
                    print('LAUNCHED',name,'gpu',slot['gpu'],'pid',child.pid,flush=True)
            state={'updated':time.time(),'running':[{'name':s['name'],'gpu':s['gpu'],'pid':s['child'].pid} for s in slots if 'child' in s],
                   'queued':pending,'finished':finished}
            (ROOT/'study_status.json').write_text(json.dumps(state,indent=2))
            if state['running']:time.sleep(15)
        print('STUDY_FINISHED',json.dumps(finished),flush=True)

if __name__=='__main__':main()
