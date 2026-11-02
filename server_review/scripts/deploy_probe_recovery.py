"""Queue a source-isolated recovery after the existing suite releases the GPU."""
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
    parent = 'lalt_repair_20261020_135310Z'
    tag = 'lalt_probe_recovery_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    snapshot = '/root/logs/' + tag
    reports = '/root/logs/results/' + tag
    source = '/root/logs/results/' + parent + '/adaptive_ordinary_r4'
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / 'output_artifacts/vm4_known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('171.232.252.207', username='root',
                   key_filename=str(Path.home() / '.ssh/id_ed25519_vm4'), timeout=20)

    def command(body, timeout=120):
        _, out, err = client.exec_command(body, timeout=timeout)
        result, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(result + error)
        return result

    try:
        command('test ! -e ' + shlex.quote(snapshot) + ' && cp -a /root/logs/' + parent + ' ' + shlex.quote(snapshot))
        hashes = source_manifest()
        with client.open_sftp() as sftp:
            for name, digest in hashes.items():
                sftp.put(str(ROOT / name), snapshot + '/' + name)
                with sftp.open(snapshot + '/' + name, 'rb') as stream:
                    if hashlib.sha256(stream.read()).hexdigest() != digest:
                        raise RuntimeError('Upload mismatch: ' + name)
            script = f'''#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {shlex.quote(snapshot)}
exec > {shlex.quote(snapshot + '.log')} 2>&1
echo 'Queued: waiting for original suite to release GPU'
while tmux has-session -t {shlex.quote(parent)} 2>/dev/null; do sleep 20; done
exec /root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.recover_probe --source {shlex.quote(source)} --output {shlex.quote(reports)}
'''
            with sftp.open(snapshot + '.sh', 'w') as stream:
                stream.write(script)
        tests = command('cd ' + shlex.quote(snapshot) + " && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 USE_TF=0 USE_FLAX=0 /root/logs/gate_env/bin/python -m unittest new_watermark.learning_channel.test_learning_channel new_watermark.learning_channel.test_reader_stability new_watermark.learning_channel.test_transfer_diagnostic new_watermark.learning_channel.test_transfer_repair -q 2>&1", 180)
        print(tests)
        # Verify the fix against the real historical manifests before launching any GPU work.
        verify = '''from pathlib import Path
from new_watermark.learning_channel.runtime import read_json
from new_watermark.learning_channel.midpoint_probe import reuse_training_rows
source=Path(SOURCE)
pin=read_json(source/'frozen_owner.json')
old=read_json(Path(pin['reuse_controls_run'])/'prompts.json')
prompts=read_json(source/'owner_prompts.json')
cache=Path(read_json(Path(pin['cache_run'])/'storage.json')['artifacts'])/'prefix_cache'
available=[]
for path in cache.glob('*.json'):
 if path.name=='identity.json': continue
 item=read_json(path)
 if 'source_image_id' in item.get('row',{}) and path.with_suffix('.safetensors').exists():
  available.append(dict(row=item['row'],path=str(path.with_suffix('.safetensors')),sha256=item['sha256']))
rows=reuse_training_rows(prompts,available,old,old['calibration'],old['query'])
print('Historical dataset pinned and validated:',len(rows),'rows')
'''.replace('SOURCE', repr(source))
        with client.open_sftp() as sftp:
            with sftp.open(snapshot + '/verify_reuse.py', 'w') as stream:
                stream.write(verify)
        verification = command('cd ' + shlex.quote(snapshot) + ' && USE_TF=0 USE_FLAX=0 /root/logs/gate_env/bin/python verify_reuse.py 2>&1')
        print(verification)
        command('bash -n ' + shlex.quote(snapshot + '.sh'))
        folder = ROOT / 'server_review/lalt_repair' / parent
        record = dict(session=tag, snapshot=snapshot, reports=reports, log=snapshot+'.log',
            wait_for_session=parent, source_probe=source, source_hashes=hashes,
            tests=tests, verification=verification)
        (folder / 'recovery_deployment.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        command('tmux new-session -d -s ' + shlex.quote(tag) + ' bash ' + shlex.quote(snapshot + '.sh'))
        print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'}, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
