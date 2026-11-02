"""Upload immutable reader-stability continuation and launch GPU verification only."""
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
    tag = 'lalt_stable_reader_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    snapshot = '/root/logs/' + tag
    parent = '/root/logs/results/lalt_20261018_053126Z_diagnostic'
    checkpoint = '/root/logs/runtime/lalt_probe_artifacts/lalt_step2000_probe_20261018_102531Z/owner_step2000.pt'
    output = '/root/logs/results/' + tag
    record = dict(session=tag, snapshot=snapshot, output=output, parent=parent,
        checkpoint=checkpoint, log=snapshot+'.log', check_log=snapshot+'_check.log',
        check_session=tag+'_check', script=snapshot+'.sh')
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
        command(f'test ! -e {shlex.quote(snapshot)} && cp -a /root/logs/lalt_20261018_053126Z {shlex.quote(snapshot)}')
        files = source_manifest()
        with client.open_sftp() as sftp:
            for name,digest in files.items():
                sftp.put(str(ROOT/name), snapshot+'/'+name)
                with sftp.open(snapshot+'/'+name,'rb') as stream:
                    if hashlib.sha256(stream.read()).hexdigest()!=digest:
                        raise RuntimeError('Upload mismatch: '+name)
            common = f'#!/usr/bin/env bash\nset -Eeuo pipefail\nexport OMP_NUM_THREADS=8\ncd {shlex.quote(snapshot)}\n'
            python = '/root/logs/gate_env/bin/python'
            module = '-u -m new_watermark.learning_channel.stable_reader_run'
            check = common + f'''exec > {shlex.quote(record['check_log'])} 2>&1
{python} -m unittest new_watermark.learning_channel.test_reader_stability new_watermark.learning_channel.test_learning_channel -q
{python} {module} --action prepare --parent {shlex.quote(parent)} --checkpoint {shlex.quote(checkpoint)} --output {shlex.quote(output)}
{python} {module} --action check --output {shlex.quote(output)}
'''
            run = common + f'''exec > {shlex.quote(record['log'])} 2>&1
exec {python} {module} --action run --output {shlex.quote(output)}
'''
            for path,body in [(snapshot+'_check.sh',check),(record['script'],run)]:
                with sftp.open(path,'w') as stream:
                    stream.write(body)
                sftp.chmod(path,0o700)
        command('bash -n '+shlex.quote(snapshot+'_check.sh')+' '+shlex.quote(record['script']))
        command('tmux new-session -d -s '+shlex.quote(record['check_session'])+' bash '+shlex.quote(snapshot+'_check.sh'))
        record.update(source_hashes=files, launched_at_utc=datetime.now(timezone.utc).isoformat(),
            phase='gpu_verification_only', ordinary_probe_unchanged=True)
        local=ROOT/'server_review/lalt_stable_reader'
        local.mkdir(parents=True,exist_ok=True)
        (local/'deployment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'},indent=2))
    finally:
        client.close()


if __name__=='__main__':
    main()
