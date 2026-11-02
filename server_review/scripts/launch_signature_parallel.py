"""Deploy the external scheduler without changing the frozen server snapshot."""
import argparse
import json
from pathlib import Path
import shlex

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/logs/runtime/signature_parallel_coordinator_20261021.py'
SOURCE = '/root/logs/signature_author_sd21_to_sd14_20261021'
RUN = '/root/logs/results/signature_author_sd21_to_sd14_20261021_074620Z'
LAUNCH = '/root/logs/results/signature_author_sd21_to_sd14_launch.json'
PYTHON = '/root/logs/gate_env/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    c = paramiko.SSHClient()
    c.load_host_keys(str(ROOT / 'output_artifacts/vm4_known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('171.232.252.207', username='root', key_filename=str(Path.home() / '.ssh/id_ed25519_vm4'), timeout=20)
    try:
        with c.open_sftp() as s:
            s.put(str(Path(__file__).with_name('signature_parallel_coordinator.py')), REMOTE)
        command = [PYTHON, REMOTE, '--source', SOURCE, '--run', RUN, '--launch', LAUNCH,
                   '--parent', '2893854', '--clean-pid', '3041812']
        _, out, err = c.exec_command(shlex.join(command + ['--validate-only']))
        print(out.read().decode(), end='')
        errors = err.read().decode()
        if errors:
            print(errors)
        if out.channel.recv_exit_status() != 0:
            raise RuntimeError('Preflight failed; scheduler unchanged')
        if args.launch:
            code = ('import subprocess,json; '
                    f'log=open({RUN + "/parallel_coordinator.log"!r},"x"); '
                    f'p=subprocess.Popen({command!r},cwd={SOURCE!r},stdin=subprocess.DEVNULL,'
                    'stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
                    'print(json.dumps({"coordinator_pid":p.pid}))')
            _, out, err = c.exec_command(shlex.join([PYTHON, '-c', code]))
            result = out.read().decode()
            print(result, end='')
            print(err.read().decode(), end='')
            if out.channel.recv_exit_status() != 0:
                raise RuntimeError('Launch failed')
            local = ROOT / 'server_review/signature_author_transfer' / Path(RUN).name
            (local / 'parallel_launch_request.json').write_text(result, encoding='utf-8')
    finally:
        c.close()


if __name__ == '__main__':
    main()
