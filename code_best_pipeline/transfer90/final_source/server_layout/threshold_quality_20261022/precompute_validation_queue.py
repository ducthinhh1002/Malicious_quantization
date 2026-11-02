"""Precompute queued validation measurements; no student/test or source edits."""
import fcntl,json,os
from pathlib import Path
import research_queue as rq
from common import write_json,read_json

def main():
 with (rq.QUEUE/'validation_precompute.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=rq.verify_plan()
  pilot=rq.prepare_run(rq.QUEUE/'pilot',pilot_n=plan['pilot_n'])
  vectors=[[.85 if b else .65 for b in mask] for mask in rq.coalitions(len(plan['groups']))]
  progress=rq.QUEUE/'validation_precompute.json'
  for start in range(0,len(vectors),2):
   status=read_json(rq.QUEUE/'status.json')
   if status.get('job') in ('attribution','pareto'):
    write_json(progress,{'state':'handed_to_supervisor','pid':os.getpid(),'completed':start,'total':len(vectors),'time':rq.now()});return
   write_json(progress,{'state':'running','pid':os.getpid(),'completed':start,'total':len(vectors),'time':rq.now(),'test_used':False,'source_main_kept_running':True})
   rows=rq.eval_many(pilot,vectors[start:start+2])
   print('VALIDATION PRECOMPUTE',start+len(rows),'/',len(vectors),flush=True)
  write_json(progress,{'state':'complete','pid':os.getpid(),'completed':len(vectors),'total':len(vectors),'time':rq.now(),'test_used':False,'next':'Existing supervisor will reuse verified measurements for KernelSHAP; no student/test started here'})
if __name__=='__main__':main()
