"""Adopt running workers; skip the three remaining clean training branches."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

SOURCE = Path('/root/logs/signature_author_sd21_to_sd14_20261021')
RUN = Path('/root/logs/results/signature_author_sd21_to_sd14_20261021_074620Z')
LAUNCH = Path('/root/logs/results/signature_author_sd21_to_sd14_launch.json')
sys.path.insert(0, str(SOURCE))
from new_watermark.learning_channel import signature_transfer as st
from new_watermark.learning_channel.run import source_manifest
from new_watermark.learning_channel.signature_metrics import match_counts, statistics, empirical_threshold


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    lock = (RUN / 'marked_only_scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = st.read_json(LAUNCH)
    parent = old['pid']
    assert parent == 3075202
    assert source_manifest() == st.read_json(RUN / 'source_manifest.json')
    assert b'signature_parallel_coordinator' in Path(f'/proc/{parent}/cmdline').read_bytes()
    active = {}
    stopped = False

    def publish():
        rows = [dict(stage=k, pid=v['pid']) for k, v in active.items()]
        write(RUN / 'status.json', dict(status='running', stage=rows[0]['stage'] if rows else 'coordinator',
              active_stages=rows, coordinator_pid=os.getpid(), scheduling='remaining_marked_only'))

    def launch(action, name):
        label = action + '_' + name
        job = RUN / ('job_' + label + '.json')
        st.freeze(job, dict(action=action, run=str(RUN), name=name))
        with (RUN / (label + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', st.MODULE, '--job', str(job)],
                                     cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, env=environment)
        active[label] = dict(pid=child.pid, child=child, identity=identity(child.pid), action=action, name=name)
        publish()
        print('Started', label, child.pid, flush=True)

    def wait_active():
        while active:
            for label, task in list(active.items()):
                if 'child' in task:
                    code = task['child'].poll()
                    if code is None:
                        continue
                    if code != 0:
                        raise RuntimeError(f'{label} exited {code}')
                elif identity(task['pid']) == task['identity']:
                    continue
                branch = RUN / task['name']
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
                del active[label]
                print('Completed', label, flush=True)
                if task['action'] == 'train':
                    launch('student_images', task['name'])
                publish()
            if active:
                time.sleep(10)

    try:
        os.kill(parent, signal.SIGSTOP)
        stopped = True
        status = st.read_json(RUN / 'status.json')
        for row in status['active_stages']:
            label, pid = row['stage'], row['pid']
            action = 'student_images' if label.startswith('student_images_') else 'train'
            name = label[len(action) + 1:]
            assert name in ('r4_attention_clean', 'r4_attention_marked')
            assert identity(pid) is not None
            assert (RUN / ('job_' + label + '.json')).as_posix().encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
            active[label] = dict(pid=pid, identity=identity(pid), action=action, name=name)
        environment = dict(x.split('=', 1) for x in Path(f'/proc/{parent}/environ').read_bytes().decode().split('\0') if '=' in x)
        skipped = ['r4_attention_conv_clean', 'r16_attention_clean', 'r16_attention_conv_clean']
        assert not any((RUN / name).exists() for name in skipped)
        st.freeze(RUN / 'marked_only_schedule.json', dict(changed_at_utc=datetime.now(timezone.utc).isoformat(),
            reason='User explicitly removed remaining clean branches', skipped_branches=skipped,
            retained_workers=status['active_stages'], old_scheduler_pid=parent, coordinator_pid=os.getpid(),
            training_configuration_unchanged=True, frozen_source_unchanged=True,
            evaluation='Rank4 attention remains paired; remaining marked branches have no matched clean control'))
        write(LAUNCH, dict(old, previous_pid=parent, pid=os.getpid(), scheduling='remaining_marked_only',
                          log=str(RUN / 'marked_only_coordinator.log')))
        publish()
        os.kill(parent, signal.SIGTERM)
        os.kill(parent, signal.SIGCONT)
        stopped = False
        wait_active()
        st.pair_report(RUN, 4, 'attention')
        for spec in st.read_json(RUN / 'config.json')['matrix'][1:]:
            name = st.branch_name(spec['rank'], spec['scope'], 'marked')
            st.prepare_branch(RUN, spec['rank'], spec['scope'], 'marked')
            launch('train', name)
            wait_active()
            scores = match_counts(st.read_json(RUN / name / 'query.json')['logits'])
            cal = match_counts(st.read_json(RUN / 'r4_attention_clean' / 'calibration.json')['logits'])
            threshold = empirical_threshold(cal)
            result = dict(rank=spec['rank'], scope=spec['scope'], evaluation='marked_only',
                binomial_nominal_1pct=dict(marked=statistics(scores, 33)),
                binomial_nominal_1e_6=dict(marked=statistics(scores, 41)),
                reference_r4_attention_empirical=dict(marked=statistics(scores, threshold),
                    calibration=statistics(cal, threshold), reference_branch='r4_attention_clean'),
                limitation='No matched clean student for this configuration; FPR, paired gain and AUROC unavailable. Reference threshold is from rank4 attention only.')
            st.freeze(RUN / f"result_r{spec['rank']}_{spec['scope']}.json", result)
            rows = [st.read_json(p) for p in sorted(RUN.glob('result_r*.json'))]
            write(RUN / 'results.json', dict(completed_configurations=len(rows), completed_pairs=1, rows=rows,
                  skipped_clean_branches=skipped, interpretation='One paired control; three marked-only configurations by user request'))
        write(RUN / 'status.json', dict(status='complete', outcome='requested_marked_matrix_complete',
            completed_pairs=1, completed_marked_configurations=4, skipped_clean_branches=skipped))
    except BaseException as error:
        write(RUN / 'marked_only_scheduler_error.json', dict(error=str(error), pid=os.getpid()))
        raise
    finally:
        if stopped:
            os.kill(parent, signal.SIGCONT)


if __name__ == '__main__':
    main()
