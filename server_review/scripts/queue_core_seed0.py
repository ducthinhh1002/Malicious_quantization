"""User-authorized exploratory core run, retaining the weak baseline status."""
import csv,json,os,subprocess,time,shutil
from pathlib import Path
r=Path('/root/workspace/serum'); os.chdir(r)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
def status(state,**kwargs):
    record={'state':state,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kwargs}
    (r/'metrics/core_queue_status.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record),flush=True)
def run(command,gpu=None,log=None):
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
    if log:
        with open(r/'logs'/log,'a') as f: subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    else: subprocess.run(command,env=env,check=True)
def worker(kind,cond,gpu):
    for step in ['0.25','1.00']:
        name=f'{cond}_step{step}_seed0'
        if not (r/f'metrics/{name}_scores.npy.result.json').exists():
            run(['bash','scripts/run_core_condition.sh',kind,cond,step,str(gpu)],gpu,log=f'core_{name}.log')
        assert json.loads((r/f'metrics/{name}_scores.npy.result.json').read_text())['num_images']==5000
try:
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--worker',choices=['clean','wm']); a=p.parse_args()
    if a.worker:
        worker(a.worker,'C01' if a.worker=='clean' else 'C11',0 if a.worker=='clean' else 1)
    else:
        run(['.venv/bin/python','scripts/check_data_ready.py'])
        selection=json.loads((r/'metrics/pilot_selection.json').read_text())
        assert selection['status']=='selected' and selection['criterion']=='image preservation only'
        assert selection['mild']==.25 and selection['strong']==1.
        assert shutil.disk_usage(r).free>100*1024**3,'Less than 100 GiB free; inspect storage before launching'
        # Reuse exact pilot train outputs, matching source names and generation seeds.
        for kind in ['clean','wm']:
            for step in ['0.25','1.00']:
                out=r/f'data/laundered/{kind}_step_{step}_train'; out.mkdir(parents=True,exist_ok=True)
                manifest=out/'laundering_manifest.csv'
                if not manifest.exists():
                    pilot=r/f'data/pilot/{kind}_{step}'
                    with (pilot/'laundering_manifest.csv').open() as f: old=list(csv.DictReader(f))
                    assert len(old)==500
                    rows=[]
                    for i,src in enumerate(sorted((r/f'data/raw/{kind}_train').glob('*.png'))[:500]):
                        saved=pilot/src.name; dst=out/src.name
                        assert Path(old[i]['output']).resolve()==saved.resolve()
                        assert src.samefile(Path(old[i]['source'])), 'Pilot source does not match the paired raw image'
                        assert int(old[i]['seed'])==10000+i and float(old[i]['step'])==float(step)
                        if not dst.exists(): os.link(saved,dst)
                        rows.append({'index':i,'source':str(src),'output':str(dst),'step':step,'seed':10000+i})
                    with manifest.open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=['index','source','output','step','seed']); w.writeheader(); w.writerows(rows)
        status('core_seed0_running',baseline='weak; not auto-approved',strengths=[.25,1.])
        processes=[]
        # Start the watermark branch while GPU 0 creates the shared clean teacher reference.
        logfile=open(r/'logs/core_wm_worker.log','a')
        processes.append((subprocess.Popen(['.venv/bin/python','scripts/queue_core_seed0.py','--worker','wm'],stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True),logfile))
        run(['.venv-student/bin/python','scripts/generate_student.py','--unadapted','--base-model',str(r/'models/sd21-base'),'--prompts','prompts/student_eval.txt','--out','data/utility_teacher_clean','--base-seed','60000','--resume'],0,'utility_teacher_reference.log')
        logfile=open(r/'logs/core_clean_worker.log','a')
        processes.append((subprocess.Popen(['.venv/bin/python','scripts/queue_core_seed0.py','--worker','clean'],stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True),logfile))
        while any(process.poll() is None for process,logfile in processes):
            failures=[str(logfile.name) for process,logfile in processes if process.poll() not in (None,0)]
            if failures: raise RuntimeError(f'Core worker failed: {failures}')
            status('core_seed0_running',baseline='weak; not auto-approved',
                   active_worker_pids=[p.pid for p,f in processes if p.poll() is None])
            time.sleep(20)
        failures=[str(f.name) for p,f in processes if p.returncode!=0]
        for process,logfile in processes: logfile.close()
        if failures: raise RuntimeError(f'Core worker failed: {failures}')
        status('utility_running')
        names=['C00_seed0','C10_seed0']+[f'{c}_step{s}_seed0' for s in ['0.25','1.00'] for c in ['C01','C11']]
        # Evaluate serially against a single shared reference feature cache.
        for name in names:
            if not (r/f'metrics/{name}_utility.json').exists():
                run(['.venv-student/bin/python','scripts/student_utility.py','--student',name],0,f'utility_{name}.log')
        run(['.venv/bin/python','scripts/summarize_core_seed0.py'])
        status('completed_seed0',next_decision=str(r/'metrics/core_followup_decision.json'))
except BaseException as exc:
    # Children have their own process groups. Remove owned descendants on failure
    # so a supervisor never restarts into duplicate GPU jobs.
    if 'processes' in locals():
        import signal
        for process,logfile in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
    if not ('a' in locals() and a.worker): status('failed',error=str(exc))
    raise
