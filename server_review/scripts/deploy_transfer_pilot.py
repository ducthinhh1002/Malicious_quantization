"""Launch an isolated owner learning-channel diagnostic, leaving rank32 intact."""
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
    tag='lalt_transfer_pilot_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    snapshot='/root/logs/'+tag
    reports='/root/logs/results/'+tag
    parent='/root/logs/results/lalt_stable_reader_20261018_111112Z'
    cache='/root/logs/results/lalt_20261018_053126Z_diagnostic'
    control='/root/logs/results/lalt_step20000_key221_probe_20261018_231633Z'
    c=paramiko.SSHClient()
    c.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('171.232.252.207',username='root',key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
    def command(body):
        _,out,err=c.exec_command(body,timeout=60)
        result,error=out.read().decode(),err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(result+error)
        return result
    try:
        command('test ! -e '+shlex.quote(snapshot)+' && cp -a /root/logs/lalt_stable_reader_20261018_111112Z '+shlex.quote(snapshot))
        hashes=source_manifest()
        with c.open_sftp() as s:
            for name,digest in hashes.items():
                s.put(str(ROOT/name),snapshot+'/'+name)
                with s.open(snapshot+'/'+name,'rb') as f:
                    if hashlib.sha256(f.read()).hexdigest()!=digest:raise RuntimeError('Upload mismatch: '+name)
            body=f'''#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {shlex.quote(snapshot)}
exec > {shlex.quote(snapshot+'.log')} 2>&1
/root/logs/gate_env/bin/python -m unittest new_watermark.learning_channel.test_learning_channel new_watermark.learning_channel.test_reader_stability new_watermark.learning_channel.test_transfer_diagnostic -q
/root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.transfer_pilot --action prepare --parent {shlex.quote(parent)} --cache-run {shlex.quote(cache)} --clean-reference {shlex.quote(control)} --output {shlex.quote(reports)}
exec /root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.transfer_pilot --action run --output {shlex.quote(reports)}
'''
            with s.open(snapshot+'.sh','w') as f:f.write(body)
        command('bash -n '+shlex.quote(snapshot+'.sh'))
        command('tmux new-session -d -s '+shlex.quote(tag)+' bash '+shlex.quote(snapshot+'.sh'))
        record=dict(session=tag,snapshot=snapshot,reports=reports,log=snapshot+'.log',
            source_hashes=hashes,outer_steps=64,starting_teacher_step=20000,
            clean_control_source=control,cache_source=cache,rank32_unchanged=True)
        folder=ROOT/'server_review/lalt_transfer_pilot'
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'deployment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'},indent=2))
    finally:c.close()


if __name__=='__main__':main()
