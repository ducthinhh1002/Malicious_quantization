"""Deploy isolated reader repair, after the frozen-reader prompt audit completes."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shlex
import sys

import paramiko

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from new_watermark.learning_channel.run import source_manifest


def main():
    audit = json.loads((ROOT / 'server_review/lcf_reader_deployment/deployment.json').read_text())
    tag = 'lcf_reader_repair_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    remote = '/root/logs/' + tag
    output = '/root/logs/results/' + tag
    c = paramiko.SSHClient()
    c.load_host_keys(str(ROOT / 'output_artifacts/vm4_known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('171.232.252.207', username='root',
              key_filename=str(Path.home() / '.ssh/id_ed25519_vm4'), timeout=15)

    def command(text):
        _, out, err = c.exec_command(text, timeout=60)
        result, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(result + error)
        return result

    try:
        command('test ! -e ' + shlex.quote(remote) + ' && cp -a /root/logs/lcf_gain_20261018_021834Z ' + shlex.quote(remote))
        files = source_manifest()
        with c.open_sftp() as s:
            for name, digest in files.items():
                s.put(str(ROOT / name), remote + '/' + name)
                with s.open(remote + '/' + name, 'rb') as f:
                    if hashlib.sha256(f.read()).hexdigest() != digest:
                        raise RuntimeError('Upload checksum mismatch: ' + name)
            # Wait only for our own audit. Do not interrupt any other GPU processes.
            body = f'''#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8
cd {shlex.quote(remote)}
exec > {shlex.quote(remote + '.log')} 2>&1
/root/logs/gate_env/bin/python -m unittest new_watermark.learning_channel.test_learning_channel -v
for ((i=0; i<180; i++)); do
  if ! tmux has-session -t {shlex.quote(audit['session'])} 2>/dev/null; then break; fi
  sleep 10
done
/root/logs/gate_env/bin/python -c {shlex.quote('import json; assert json.load(open(' + repr(audit['output'] + '/status.json') + '))["status"] == "complete", "Frozen-reader audit has not completed successfully"')}
/root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.reader_repair --run /root/logs/results/lcf_gain_20261018_021834Z_diagnostic --output {shlex.quote(output)} --artifact-root /root/logs/runtime/lcf_artifacts --steps 800
'''
            with s.open(remote + '.sh', 'w') as f:
                f.write(body)
        command('bash -n ' + shlex.quote(remote + '.sh'))
        command('tmux new-session -d -s ' + shlex.quote(tag) + ' bash ' + shlex.quote(remote + '.sh'))
        record = dict(folder=remote, output=output, session=tag, log=remote + '.log', hashes=files,
            source_run='/root/logs/results/lcf_gain_20261018_021834Z_diagnostic',
            frozen_audit=audit['output'], launched_at_utc=datetime.now(timezone.utc).isoformat())
        dst = ROOT / 'server_review/lcf_reader_deployment/repair_deployment.json'
        dst.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in record.items() if k != 'hashes'}, indent=2))
    finally:
        c.close()


if __name__ == '__main__':
    main()
