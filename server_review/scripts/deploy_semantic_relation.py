"""Launch an immutable native-fingerprint smoke test followed by the pilot."""
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
    tag='semantic_relation_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    snapshot='/root/logs/'+tag
    reports='/root/logs/results/'+tag
    client=paramiko.SSHClient()
    client.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('171.232.252.207',username='root',key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
    def command(body,timeout=180):
        _,out,err=client.exec_command(body,timeout=timeout)
        result,error=out.read().decode(),err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(result+error)
        return result
    try:
        free=int(command('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits').strip())
        if free<49000:
            raise RuntimeError('Need 48 GiB free; no existing process interrupted')
        if int(command('df --output=avail /root/logs | tail -1').strip())<30*1024**2:
            raise RuntimeError('Need 30 GiB disk; no existing output deleted')
        command('test ! -e '+shlex.quote(snapshot)+' && cp -a /root/logs/lalt_stable_reader_20261018_111112Z '+shlex.quote(snapshot))
        hashes=source_manifest()
        with client.open_sftp() as sftp:
            for name,digest in hashes.items():
                sftp.put(str(ROOT/name),snapshot+'/'+name)
                with sftp.open(snapshot+'/'+name,'rb') as stream:
                    if hashlib.sha256(stream.read()).hexdigest()!=digest:
                        raise RuntimeError('Upload checksum mismatch: '+name)
            script=f'''#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {shlex.quote(snapshot)}
exec > {shlex.quote(snapshot+'.log')} 2>&1
/root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.semantic_experiment --output {shlex.quote(reports+'_smoke')} --smoke
exec /root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.semantic_experiment --output {shlex.quote(reports)}
'''
            with sftp.open(snapshot+'.sh','w') as stream:
                stream.write(script)
        command('bash -n '+shlex.quote(snapshot+'.sh'))
        tests=command('cd '+shlex.quote(snapshot)+" && OMP_NUM_THREADS=4 USE_TF=0 USE_FLAX=0 CUDA_VISIBLE_DEVICES='' /root/logs/gate_env/bin/python -m unittest "
            'new_watermark.learning_channel.test_learning_channel '
            'new_watermark.learning_channel.test_reader_stability '
            'new_watermark.learning_channel.test_transfer_diagnostic '
            'new_watermark.learning_channel.test_transfer_repair '
            'new_watermark.learning_channel.test_contrast_replay '
            'new_watermark.learning_channel.test_transfer_readout_audit '
            'new_watermark.learning_channel.test_spectral_transport '
            'new_watermark.learning_channel.test_semantic_relation -q 2>&1')
        local=ROOT/'server_review/semantic_relation'/tag
        local.mkdir(parents=True,exist_ok=False)
        record=dict(session=tag,snapshot=snapshot,reports=reports,smoke_reports=reports+'_smoke',
            log=snapshot+'.log',source_hashes=hashes,server_tests=tests,
            owner_steps=500,betas=[1.5,3.],signs=[1,-1],student_steps=2500,rank=4,
            branches=['clean','positive','negative','mixed'],native_gate_required=True)
        (local/'deployment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        command('tmux new-session -d -s '+shlex.quote(tag)+' bash '+shlex.quote(snapshot+'.sh'))
        print(tests)
        print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes','server_tests')},indent=2))
    finally:
        client.close()


if __name__=='__main__':
    main()
