"""Deploy a parallel, memory-capped ordinary-LoRA probe of pinned LALT step 2000."""
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
    folder = ROOT / 'server_review/lalt_step2000_probe'
    pinned = json.loads((folder / 'frozen_owner.json').read_text())
    tag = pinned['tag']; remote = '/root/logs/' + tag
    c = paramiko.SSHClient()
    c.load_host_keys(str(ROOT / 'output_artifacts/vm4_known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('171.232.252.207', username='root', key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'), timeout=15)

    def command(text):
        _, out, err = c.exec_command(text, timeout=60)
        value, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(value + error)
        return value

    try:
        command('test ! -e ' + shlex.quote(remote) + ' && cp -a /root/logs/lalt_20261018_053126Z ' + shlex.quote(remote))
        hashes = source_manifest()
        with c.open_sftp() as s:
            for name, digest in hashes.items():
                s.put(str(ROOT/name), remote + '/' + name)
                with s.open(remote + '/' + name, 'rb') as f:
                    if hashlib.sha256(f.read()).hexdigest() != digest:
                        raise RuntimeError('Source upload mismatch: ' + name)
            body = f'''#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {shlex.quote(remote)}
exec > {shlex.quote(remote+'.log')} 2>&1
/root/logs/gate_env/bin/python -m unittest new_watermark.learning_channel.test_learning_channel -q
exec /root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.midpoint_probe --run {shlex.quote(pinned['reports'])}
'''
            with s.open(remote + '.sh', 'w') as f:
                f.write(body)
        command('bash -n ' + shlex.quote(remote + '.sh'))
        command('tmux new-session -d -s ' + shlex.quote(tag) + ' bash ' + shlex.quote(remote + '.sh'))
        record = dict(**pinned, folder=remote, session=tag, log=remote+'.log', source_hashes=hashes,
            main_teacher_unchanged=True, parallel_probe_cap_gib=32)
        (folder / 'deployment.json').write_text(json.dumps(record, indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'}, indent=2))
    finally:
        c.close()


if __name__ == '__main__':
    main()
