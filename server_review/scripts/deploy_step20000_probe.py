"""Pin final LALT teacher/key 221 and launch ordinary-LoRA development evaluation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import paramiko

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from new_watermark.learning_channel.run import source_manifest


def main():
    tag = 'lalt_step20000_key221_probe_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    snapshot = '/root/logs/' + tag
    output = '/root/logs/results/' + tag
    artifacts = '/root/logs/runtime/lalt_probe_artifacts/' + tag
    parent = '/root/logs/results/lalt_stable_reader_20261018_111112Z'
    record = dict(tag=tag, session=tag, reports=output, artifacts=artifacts, folder=snapshot,
        log=snapshot+'.log', owner_step=20000, probe_key=221, probe_wrong_key=34)
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('171.232.252.207',username='root',
        key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
    def command(body):
        _, out, err = client.exec_command(body, timeout=60)
        result, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(result+error)
        return result
    try:
        command(f'test ! -e {shlex.quote(snapshot)} && cp -a /root/logs/lalt_stable_reader_20261018_111112Z {shlex.quote(snapshot)}')
        files = source_manifest()
        with client.open_sftp() as sftp:
            for name,digest in files.items():
                sftp.put(str(ROOT/name), snapshot+'/'+name)
                with sftp.open(snapshot+'/'+name,'rb') as stream:
                    if hashlib.sha256(stream.read()).hexdigest()!=digest:
                        raise RuntimeError('Upload mismatch: '+name)
        script = f'''
import json, hashlib, shutil
from pathlib import Path
import torch
parent=Path({parent!r})
config=json.loads((parent/'config.json').read_text())
source=Path(json.loads((parent/'storage.json').read_text())['artifacts'])/'teacher_last.pt'
raw=source.read_bytes()
import io
state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
if state['step']!=20000 or state['config']!=config: raise RuntimeError('Wrong final teacher')
reports=Path({output!r}); reports.mkdir(parents=True,exist_ok=False)
artifacts=Path({artifacts!r}); artifacts.mkdir(parents=True,exist_ok=False)
target=artifacts/'owner_step20000.pt'
with target.open('xb') as f: f.write(raw)
digest=hashlib.sha256(raw).hexdigest()
pinned=dict(tag={tag!r},reports=str(reports),artifacts=str(artifacts),checkpoint=str(target),
    sha256=digest,source_run=str(parent),owner_step=20000,step=20000,probe_key=221,probe_wrong_key=34,
    cache_run='/root/logs/results/lalt_20261018_053126Z_diagnostic',
    key_selection='User selected key221 with 83.59375% teacher validation TPR at step20000; development selection, not final test')
(reports/'frozen_owner.json').write_text(json.dumps(pinned,indent=2))
for name in ['config','prompts','assets']:
    shutil.copyfile(parent/(name+'.json'),reports/('owner_'+name+'.json'))
print(json.dumps(pinned))
'''
        inp,out,err=client.exec_command('/root/logs/gate_env/bin/python -',timeout=60)
        inp.write(script);inp.channel.shutdown_write()
        result,error=out.read().decode(),err.read().decode()
        if out.channel.recv_exit_status(): raise RuntimeError(result+error)
        pinned=json.loads(result)
        body=f'''#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {shlex.quote(snapshot)}
exec > {shlex.quote(record['log'])} 2>&1
/root/logs/gate_env/bin/python -m unittest new_watermark.learning_channel.test_learning_channel new_watermark.learning_channel.test_reader_stability -q
exec /root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.midpoint_probe --run {shlex.quote(output)}
'''
        with client.open_sftp() as sftp:
            with sftp.open(snapshot+'.sh','w') as f: f.write(body)
        command('bash -n '+shlex.quote(snapshot+'.sh'))
        command('tmux new-session -d -s '+shlex.quote(tag)+' bash '+shlex.quote(snapshot+'.sh'))
        record.update(source_hashes=files,checkpoint_sha256=pinned['sha256'], launched_at_utc=datetime.now(timezone.utc).isoformat())
        local=ROOT/'server_review/lalt_step20000_key221_probe'
        local.mkdir(parents=True,exist_ok=True)
        (local/'deployment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        (local/'frozen_owner.json').write_text(json.dumps(pinned,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'},indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
