"""Queue rank16 then rank32 after the current final-teacher rank4 probe."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import paramiko

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from new_watermark.learning_channel.run import source_manifest


def main():
    previous=json.loads((ROOT/'server_review/lalt_step20000_key221_probe/deployment.json').read_text())
    tag='lalt_rank_queue_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    snapshot='/root/logs/'+tag
    reports='/root/logs/results/'+tag
    tasks=[dict(rank=rank,tag=tag+'_r'+str(rank),reports=reports+'_r'+str(rank),
        artifacts='/root/logs/runtime/lalt_probe_artifacts/'+tag+'_r'+str(rank),
        log=snapshot+'_r'+str(rank)+'.log') for rank in (16,32)]
    plan=dict(after=previous['reports'],runs=tasks)
    c=paramiko.SSHClient()
    c.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('171.232.252.207',username='root',key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
    def command(body):
        _,out,err=c.exec_command(body,timeout=60)
        result,error=out.read().decode(),err.read().decode()
        if out.channel.recv_exit_status(): raise RuntimeError(result+error)
        return result
    try:
        command('test ! -e '+shlex.quote(snapshot)+' && cp -a '+shlex.quote(previous['folder'])+' '+shlex.quote(snapshot))
        hashes=source_manifest()
        with c.open_sftp() as s:
            for name,digest in hashes.items():
                s.put(str(ROOT/name),snapshot+'/'+name)
                with s.open(snapshot+'/'+name,'rb') as f:
                    if hashlib.sha256(f.read()).hexdigest()!=digest: raise RuntimeError('Upload mismatch '+name)
        script=f'''
import json, shutil
from pathlib import Path
plan=json.loads({json.dumps(plan)!r})
queue=Path({reports!r}); queue.mkdir(parents=True,exist_ok=False)
(queue/'plan.json').write_text(json.dumps(plan,indent=2))
reference=Path(plan['after'])
pinned=json.loads((reference/'frozen_owner.json').read_text())
for task in plan['runs']:
    run=Path(task['reports']);run.mkdir(parents=True,exist_ok=False)
    Path(task['artifacts']).mkdir(parents=True,exist_ok=False)
    config=dict(pinned,tag=task['tag'],reports=task['reports'],artifacts=task['artifacts'],
        probe_rank=task['rank'],comparison_reference_run=str(reference))
    (run/'frozen_owner.json').write_text(json.dumps(config,indent=2))
    for name in ['owner_config.json','owner_prompts.json','owner_assets.json']:
        shutil.copyfile(reference/name,run/name)
print('Queue prepared')
'''
        inp,out,err=c.exec_command('/root/logs/gate_env/bin/python -',timeout=60)
        inp.write(script);inp.channel.shutdown_write()
        result,error=out.read().decode(),err.read().decode()
        if out.channel.recv_exit_status(): raise RuntimeError(result+error)
        body=f'''#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {shlex.quote(snapshot)}
exec > {shlex.quote(snapshot+'.log')} 2>&1
/root/logs/gate_env/bin/python -m unittest new_watermark.learning_channel.test_learning_channel new_watermark.learning_channel.test_reader_stability -q
exec /root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.probe_queue --plan {shlex.quote(reports+'/plan.json')}
'''
        with c.open_sftp() as s:
            with s.open(snapshot+'.sh','w') as f:f.write(body)
        command('bash -n '+shlex.quote(snapshot+'.sh'))
        command('tmux new-session -d -s '+shlex.quote(tag)+' bash '+shlex.quote(snapshot+'.sh'))
        record=dict(session=tag,snapshot=snapshot,reports=reports,plan=plan,
            log=snapshot+'.log',source_hashes=hashes,existing_rank4_unchanged=True)
        local=ROOT/'server_review/lalt_rank_queue'
        local.mkdir(parents=True,exist_ok=True)
        (local/'deployment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'},indent=2))
    finally:
        c.close()


if __name__=='__main__':main()
