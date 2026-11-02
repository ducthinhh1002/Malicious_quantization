"""Checksummed Stable Signature transfer suite; smoke then four matched pairs."""
from datetime import datetime,timezone
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
    tag='signature_transfer_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    snapshot='/root/logs/'+tag; reports='/root/logs/results/'+tag
    c=paramiko.SSHClient(); c.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('171.232.252.207',username='root',key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
    def command(body,timeout=180):
        _,out,err=c.exec_command(body,timeout=timeout)
        text,error=out.read().decode(),err.read().decode()
        if out.channel.recv_exit_status(): raise RuntimeError(text+error)
        return text
    try:
        if int(command('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits').strip())<49000:
            raise RuntimeError('Need 48 GiB free; no existing job interrupted')
        if int(command('df --output=avail /root/logs | tail -1').strip())<20*1024**2:
            raise RuntimeError('Need 20 GiB disk; no output deleted')
        command('test ! -e '+shlex.quote(snapshot)+' && cp -a /root/logs/lalt_stable_reader_20261018_111112Z '+shlex.quote(snapshot))
        hashes=source_manifest()
        with c.open_sftp() as s:
            for name,digest in hashes.items():
                s.put(str(ROOT/name),snapshot+'/'+name)
                with s.open(snapshot+'/'+name,'rb') as f:
                    if hashlib.sha256(f.read()).hexdigest()!=digest: raise RuntimeError('Upload mismatch: '+name)
            body=f'''#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {shlex.quote(snapshot)}
exec > {shlex.quote(snapshot+'.log')} 2>&1
/root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.signature_transfer --output {shlex.quote(reports+'_smoke')} --smoke
exec /root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.signature_transfer --output {shlex.quote(reports)}
'''
            with s.open(snapshot+'.sh','w') as f: f.write(body)
        command('bash -n '+shlex.quote(snapshot+'.sh'))
        tests=command('cd '+shlex.quote(snapshot)+" && OMP_NUM_THREADS=4 USE_TF=0 USE_FLAX=0 CUDA_VISIBLE_DEVICES='' /root/logs/gate_env/bin/python -m unittest "
            'new_watermark.learning_channel.test_learning_channel new_watermark.learning_channel.test_reader_stability '
            'new_watermark.learning_channel.test_transfer_diagnostic new_watermark.learning_channel.test_transfer_repair '
            'new_watermark.learning_channel.test_contrast_replay new_watermark.learning_channel.test_transfer_readout_audit '
            'new_watermark.learning_channel.test_spectral_transport new_watermark.learning_channel.test_semantic_relation '
            'new_watermark.learning_channel.test_signature_transfer -q 2>&1')
        folder=ROOT/'server_review/signature_transfer'/tag; folder.mkdir(parents=True,exist_ok=False)
        record=dict(session=tag,snapshot=snapshot,reports=reports,smoke_reports=reports+'_smoke',log=snapshot+'.log',
            source_hashes=hashes,server_tests=tests,student_steps=5000,training_images=2048,
            ranks=[4,16],scopes=['attention','attention_conv'],conditions=['clean','marked'],
            backbone='SD2.1 clean for student; public marked VAE for teacher only')
        (folder/'deployment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        command('tmux new-session -d -s '+shlex.quote(tag)+' bash '+shlex.quote(snapshot+'.sh'))
        print(tests)
        print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes','server_tests')},indent=2))
    finally: c.close()


if __name__=='__main__': main()
