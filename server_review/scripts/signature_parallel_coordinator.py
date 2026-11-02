"""External scheduler: adopt an active clean trainer and overlap rank-4 marked.

Runs outside the frozen experiment source; model/data/training code is untouched.
Only the old scheduler PID is replaced. Its training subprocess is preserved.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def identity(pid):
    try:
        tail = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if tail[0] == 'Z' else tail[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--launch', type=Path, required=True)
    p.add_argument('--parent', type=int, required=True)
    p.add_argument('--clean-pid', type=int, required=True)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(args.source))
    from new_watermark.learning_channel import signature_transfer as st
    from new_watermark.learning_channel.run import source_manifest

    run = args.run
    original_launch = st.read_json(args.launch)
    assert original_launch['pid'] == args.parent
    assert original_launch['run'] == str(run)
    assert source_manifest() == st.read_json(run / 'source_manifest.json')
    assert st.read_json(run / 'status.json') == dict(status='running', stage='train_r4_attention_clean')
    parent_cmd = Path(f'/proc/{args.parent}/cmdline').read_bytes()
    clean_cmd = Path(f'/proc/{args.clean_pid}/cmdline').read_bytes()
    assert b'new_watermark.learning_channel.signature_transfer' in parent_cmd
    assert str(run).encode() in parent_cmd
    assert b'job_train_r4_attention_clean.json' in clean_cmd
    assert Path(f'/proc/{args.parent}/cwd').resolve() == args.source
    assert int(Path(f'/proc/{args.clean_pid}/stat').read_text().rsplit(')', 1)[1].split()[1]) == args.parent
    clean_identity = identity(args.clean_pid)
    assert clean_identity is not None
    assert not (run / 'r4_attention_marked').exists()
    assert st.read_json(run / 'config.json')['matrix'][0] == dict(rank=4, scope='attention')
    if args.validate_only:
        print('Validated source identity, active clean child, and unused marked branch.', flush=True)
        return

    lock = (run / 'parallel_coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Preserve runtime settings, without writing environment values into reports.
    environment = dict(x.split('=', 1) for x in Path(f'/proc/{args.parent}/environ').read_bytes().decode().split('\0') if '=' in x)
    active = {}
    takeover = False
    stopped = False

    def publish():
        rows = [dict(stage=k, pid=v['pid']) for k, v in active.items()]
        write(run / 'status.json', dict(status='running', stage=rows[0]['stage'] if rows else 'coordinator',
              active_stages=rows, coordinator_pid=os.getpid(), scheduling='rank4_clean_marked_parallel'))

    def launch(action, name):
        label = action + '_' + name
        job = run / ('job_' + label + '.json')
        st.freeze(job, dict(action=action, run=str(run), name=name))
        with (run / (label + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', st.MODULE, '--job', str(job)],
                                     cwd=args.source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        active[label] = dict(pid=child.pid, child=child, action=action, name=name)
        publish()
        print(f'Started {label}: PID {child.pid}', flush=True)

    def validate_result(task):
        branch = run / task['name']
        if task['action'] == 'train':
            record = st.read_json(branch / 'student_marked.json')
            assert st.sha256(record['weights']) == record['sha256']
            audit = st.read_json(branch / 'adapter_audit.json')
            cfg = st.read_json(branch / 'config.json')
            assert audit['rank'] == cfg['probe_rank'] and audit['scope'] == cfg['probe_scope']
        else:
            assert (branch / 'query.json').is_file()
            if task['name'].endswith('_clean'):
                assert (branch / 'calibration.json').is_file()

    def wait_active():
        while active:
            for label, task in list(active.items()):
                child = task.get('child')
                if child is None:
                    if identity(task['pid']) == clean_identity:
                        continue
                else:
                    code = child.poll()
                    if code is None:
                        continue
                    if code != 0:
                        raise RuntimeError(f'{label} exited {code}; see its log')
                validate_result(task)
                del active[label]
                print(f'Completed {label}', flush=True)
                if task['action'] == 'train':
                    launch('student_images', task['name'])
                publish()
            if active:
                time.sleep(10)

    try:
        os.kill(args.parent, signal.SIGSTOP)
        stopped = True
        active['train_r4_attention_clean'] = dict(pid=args.clean_pid, child=None,
                                                 action='train', name='r4_attention_clean')
        st.prepare_branch(run, 4, 'attention', 'marked')
        launch('train', 'r4_attention_marked')
        record = dict(changed_at_utc=datetime.now(timezone.utc).isoformat(),
                      reason='User requested parallel rank-4 marked training',
                      old_scheduler_pid=args.parent, adopted_clean_pid=args.clean_pid,
                      coordinator_pid=os.getpid(), marked_pid=active['train_r4_attention_marked']['pid'],
                      scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      frozen_experiment_source_unchanged=True, training_configuration_unchanged=True,
                      original_launch=original_launch)
        st.freeze(run / 'parallel_takeover.json', record)
        write(args.launch, dict(original_launch, pid=os.getpid(), previous_pid=args.parent,
                               log=str(run / 'parallel_coordinator.log'), scheduling='rank4_clean_marked_parallel'))
        # SIGTERM only the sleeping scheduler, never the process group or trainer.
        os.kill(args.parent, signal.SIGTERM)
        os.kill(args.parent, signal.SIGCONT)
        stopped = False
        takeover = True
        wait_active()
        st.pair_report(run, 4, 'attention')
        # Preserve the original order for the remaining requested matrix.
        matrix = st.read_json(run / 'config.json')['matrix']
        for spec in matrix[1:]:
            for condition in ('clean', 'marked'):
                st.prepare_branch(run, spec['rank'], spec['scope'], condition)
                launch('train', st.branch_name(spec['rank'], spec['scope'], condition))
                wait_active()
            st.pair_report(run, spec['rank'], spec['scope'])
        write(run / 'status.json', dict(status='complete', outcome='rank_scope_matrix_complete',
              completed_pairs=len(matrix), student_steps=st.read_json(run / 'config.json')['probe_steps']))
    except BaseException as error:
        write(run / 'status.json', dict(status='failed', message=str(error), coordinator_pid=os.getpid(),
              active_stages=[dict(stage=k, pid=v['pid']) for k, v in active.items()], takeover=takeover))
        raise
    finally:
        if stopped:
            os.kill(args.parent, signal.SIGCONT)


if __name__ == '__main__':
    main()
