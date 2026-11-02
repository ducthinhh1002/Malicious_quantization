"""Append rank32 marked students after the existing coordinator releases its lock."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

SOURCE=Path('/root/logs/signature_author_sd21_to_sd14_20261021')
RUN=Path('/root/logs/results/signature_author_sd21_to_sd14_20261021_074620Z')
sys.path.insert(0,str(SOURCE))
from new_watermark.learning_channel import signature_transfer as st
from new_watermark.learning_channel.run import source_manifest
from new_watermark.learning_channel.signature_metrics import match_counts, statistics, empirical_threshold

def write(path,data):
    tmp=path.with_suffix('.rank32.tmp'); tmp.write_text(json.dumps(data,indent=2)); tmp.replace(path)

def main():
    own=(RUN/'rank32_queue.lock').open('a')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=st.read_json(RUN/'status.json')
    parent=status['coordinator_pid']
    env=dict(x.split('=',1) for x in Path(f'/proc/{parent}/environ').read_bytes().decode().split('\0') if '=' in x)
    assert source_manifest()==st.read_json(RUN/'source_manifest.json')
    queue=dict(status='waiting',pid=os.getpid(),after_coordinator_pid=parent,
        added_at_utc=datetime.now(timezone.utc).isoformat(),
        branches=['r32_attention_marked','r32_attention_conv_marked'],
        training='same 5000 steps, lr1e-5, batch2 accumulation2, seed2310; marked data only',
        evaluation='1000 query images; no matched rank32 clean controls',frozen_source_unchanged=True)
    write(RUN/'rank32_queue.json',queue)
    print('Queued',queue,flush=True)
    lock=(RUN/'marked_only_scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if st.read_json(RUN/'status.json')['status']!='complete':
        raise RuntimeError('Previous coordinator did not finish successfully; refusing to start')
    try:
        for scope in ['attention','attention_conv']:
            name=st.branch_name(32,scope,'marked')
            st.prepare_branch(RUN,32,scope,'marked')
            for action in ['train','student_images']:
                label=action+'_'+name
                job=RUN/('job_'+label+'.json')
                st.freeze(job,dict(action=action,run=str(RUN),name=name))
                with (RUN/(label+'.log')).open('x') as log:
                    p=subprocess.Popen([sys.executable,'-u','-m',st.MODULE,'--job',str(job)],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
                    write(RUN/'status.json',dict(status='running',stage=label,active_stages=[dict(stage=label,pid=p.pid)],coordinator_pid=os.getpid(),scheduling='rank32_extension'))
                    write(RUN/'rank32_queue.json',dict(queue,status='running',stage=label,worker_pid=p.pid))
                    print('Started',label,p.pid,flush=True)
                    if p.wait()!=0: raise RuntimeError(label+' failed; see worker log')
            scores=match_counts(st.read_json(RUN/name/'query.json')['logits'])
            cal=match_counts(st.read_json(RUN/'r4_attention_clean'/'calibration.json')['logits'])
            threshold=empirical_threshold(cal)
            st.freeze(RUN/f'result_r32_{scope}.json',dict(rank=32,scope=scope,evaluation='marked_only',
                binomial_nominal_1pct=dict(marked=statistics(scores,33)),
                binomial_nominal_1e_6=dict(marked=statistics(scores,41)),
                reference_r4_attention_empirical=dict(marked=statistics(scores,threshold),calibration=statistics(cal,threshold),reference_branch='r4_attention_clean'),
                limitation='No matched rank32 clean control; reference threshold is rank4 attention only.'))
            summary=st.read_json(RUN/'results.json'); summary['rows']=[st.read_json(p) for p in sorted(RUN.glob('result_r*.json'))]
            summary.update(completed_configurations=len(summary['rows']),interpretation='One paired rank4 control; remaining configurations marked-only, including requested rank32 extension')
            write(RUN/'results.json',summary)
        write(RUN/'rank32_queue.json',dict(queue,status='complete'))
        write(RUN/'status.json',dict(status='complete',outcome='requested_rank32_extension_complete',completed_marked_configurations=6,completed_pairs=1))
    except BaseException as e:
        write(RUN/'rank32_queue.json',dict(queue,status='failed',error=str(e)))
        raise

if __name__=='__main__': main()
