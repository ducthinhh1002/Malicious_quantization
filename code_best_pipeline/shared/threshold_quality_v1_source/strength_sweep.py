"""Predeclared 0.75/0.65 diagnostics; no mutation of the running experiment."""
from pathlib import Path
import subprocess,sys,json
import parallel_scale_eval as pe
from common import read_json,freeze
root=Path(__file__).resolve().parent
run=pe.prepare(root/'run',3000)
jobs=[]
for s in [.75,.65]:
    f=(run/f'worker_{s}.log').open('a')
    p=subprocess.Popen([sys.executable,str(root/'parallel_scale_eval.py'),'--worker',str(run),'--scale',str(s)],stdout=f,stderr=subprocess.STDOUT)
    jobs.append((p,f));print('STARTED',s,p.pid,flush=True)
codes=[]
for p,f in jobs:codes.append(p.wait());f.close()
if any(codes):raise RuntimeError(f'Strength workers failed {codes}')
from semantic_report import score_rows
from threshold_quality import paired_gate
b=read_json(run/'result/baseline.json');n=read_json(run/'config.json')['validation_n']
threshold=read_json(run/'result/threshold_teacher.json')['threshold']
bs=[r['score'] for r in score_rows(run,b['label'],n)]
rows=[]
for tag in ['scale085','scale075','scale065']:
    v=read_json(run/'result'/f'{tag}.json')
    cs=[r['score'] for r in score_rows(run,v['label'],n)]
    rows.append({'scale':v['scale'],'metrics':v['metrics'],'quality':v['quality'],
                 'relative_lpips_gain':1-v['quality']['paired_lpips']/b['quality']['paired_lpips'],
                 'paired_diagnostics':paired_gate(b['quality'],v['quality'],bs,cs,threshold,comparisons=7)})
freeze(run/'result/strength_sweep.json',{'baseline':b,'candidates':rows,'quality_gates_enforced':False,
       'test_used':False,'student_transfer_verified':False,'purpose':'diagnostic development strength curve; original selection unchanged'})
print('STRENGTH SWEEP COMPLETE',flush=True)
