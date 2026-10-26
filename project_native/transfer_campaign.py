"""Queue perceptual and paired-learner refinement after the active recovery run."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import manojb_5k_campaign as base
from common import read_json, write_json, freeze, sha256

q = base.q
ROOT = q.ROOT
SOURCE = ROOT / 'kappa1_manojb_5k/source'
QUEUE = ROOT / 'transfer_refine_20261024'
DEPENDENCY = ROOT / 'scale085_recovery_20261024'
ORDER = ['quality', 'quality_transfer']
COMMON = {'teacher_steps': 600, 'checkpoint_every': 200, 'teacher_lr': 1e-5,
          'quality_weight': .05, 'transfer_weight': .1, 'aux_gradient_fraction': .25,
          'transfer_every': 10, 'proxy_rank': 8, 'inner_steps': 2, 'query_steps': 8}
SPECS = {name: {**COMMON, 'transfer': name == 'quality_transfer'} for name in ORDER}
FILES = ['transfer_campaign.py', 'transfer_refine.py', 'transfer_proxy.py']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify():
    base.verify_code(); q.verify_plan()
    for name, digest in read_json(QUEUE / 'code_manifest.json').items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError('Refinement code changed: ' + name)
    plan = read_json(QUEUE / 'plan.json')
    if sha256(SOURCE / 'result/threshold_student.json') != plan['student_threshold_sha256']:
        raise RuntimeError('Student threshold changed')
    if sha256(SOURCE / 'teacher_fp/calibration_target.json') != plan['calibration_target_sha256']:
        raise RuntimeError('Teacher calibration target changed')


def prepare(run, name):
    run.mkdir(parents=True, exist_ok=True)
    spec = SPECS[name]
    c = read_json(SOURCE / 'config.json')
    c.update(teacher_train_steps=spec['teacher_steps'], teacher_lr=spec['teacher_lr'],
             teacher_checkpoint_every=spec['checkpoint_every'])
    freeze(run / 'config.json', c)
    freeze(run / 'refine_spec.json', spec)
    for rel in ['assets.json', 'prompts.json', 'carrier', 'result/threshold_teacher.json',
                'result/threshold_student.json', 'result/threshold.json', 'teacher_fp/calibration_target.json',
                'images/teacher_clean_train', 'images/teacher_clean_validation', 'images/teacher_clean_test',
                'images/unadapted', 'scores/unadapted.json']:
        q.share(SOURCE / rel, run / rel)
    cp = base.checkpoint(SOURCE)
    freeze(run / 'teacher_choice.json', {'method': name, 'source_checkpoint': str(cp),
           'source_adapter_sha256': sha256(cp / 'adapter_model.safetensors'),
           'teacher_kappa': 1., 'initial_effective_scale': 1., 'source_step': 3000,
           'refinement_steps': spec['teacher_steps'], 'teacher_selection': 'fixed final step',
           'test_used_for_selection': False, 'quality_stopping_gates': False,
           'student_trainer': 'unchanged ordinary rank32 attention_conv / 7500 updates / batch4',
           'proxy': 'owner only; rank8 attention_conv, 2 fresh AdamW updates, full 8-step DDIM query',
           'limitations': 'Short low-rank proxy is not final learner. FP32 proxy, RGB8 STE, AdamW sqrt floor 1e-24. '
                          'Additional teacher compute versus unrefined control; no matched continuation-only ablation.'})
    return run


def initialize():
    QUEUE.mkdir(parents=True, exist_ok=True)
    freeze(QUEUE / 'code_manifest.json', {name: sha256(ROOT / name) for name in FILES})
    freeze(QUEUE / 'specs.json', SPECS)
    freeze(QUEUE / 'plan.json', {'order': ORDER, 'dependency': str(DEPENDENCY),
          'source_run': str(SOURCE), 'source_code_manifest_sha256': sha256(ROOT / 'kappa1_manojb_5k/code_manifest.json'),
          'student_threshold_sha256': sha256(SOURCE / 'result/threshold_student.json'),
          'calibration_target_sha256': sha256(SOURCE / 'teacher_fp/calibration_target.json'),
          'original_teacher_objective_preserved': True, 'final_student_objective_unchanged': True,
          'gates': 'numerical/integrity/resource checks only; finish all scientific outcomes',
          'test_protocol': 'existing exploratory test panel, no checkpoint selection on test',
          'quality_metrics': ['LPIPS', 'FID', 'KID', 'CLIP'],
          'reuse': 'all clean images, prompts, assets, carrier, thresholds, unadapted reference; hard links',
          'storage_policy': 'retain teachers/checkpoints/results/images; remove only rebuildable merged exports',
          'sequential_heavy_jobs': True, 'initial_free_space_min_gib': 17})
    verify()
    for name in ORDER:
        prepare(QUEUE / name, name)


def disk_wait(gib, stage):
    while shutil.disk_usage(ROOT).free < gib * 2**30:
        write_json(QUEUE / 'status.json', {'state': 'waiting_for_disk', 'stage': stage,
                   'required_free_gib': gib, 'free_gib': shutil.disk_usage(ROOT).free / 2**30, 'updated': now()})
        print('WAIT disk', stage, gib, flush=True)
        time.sleep(30)


def call(run, label, script, *args):
    verify()
    marker = run / 'stages' / (label + '.json')
    if marker.exists():
        return
    disk_wait(2.5, str(run.name) + '/' + label)
    write_json(QUEUE / 'status.json', {'state': 'running', 'job': run.name, 'stage': label, 'updated': now()})
    log = run / 'queue_logs' / (label + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
    print('STAGE', run.name, label, flush=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable, str(script), '--run', str(run), *map(str, args)],
                        cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    freeze(marker, {'completed': now(), 'log': str(log)})


def data(run):
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import generate_set, load_native
    folder = run / 'teacher_fp/refined'
    if not (run / 'data_complete.json').exists():
        disk_wait(7., run.name + '/data_merge_generation')
        model = merge_teacher(run, folder)
        pipe = load_native(model)
        for split in ('train', 'validation', 'test'):
            generate_set(run, model, split, f'teacher_wm_{split}', pipe)
        del pipe
        freeze(run / 'data_complete.json', {'adapter_sha256': sha256(folder / 'adapter_model.safetensors')})
    # Only this run's reproducible model export is removed, after all images are complete.
    target = (folder / 'merged_pipeline').resolve()
    if QUEUE.resolve() not in target.parents or target.name != 'merged_pipeline' or target.is_symlink():
        raise RuntimeError('Unsafe derived-export path')
    proof = folder / 'merge_test.json'
    if proof.exists():
        freeze(folder / 'merge_audit.json', read_json(proof))
        if target.exists():
            shutil.rmtree(target)
        proof.unlink()


def teacher_report(run):
    from semantic_report import teacher_quality, detection_metrics
    c = read_json(run / 'config.json')
    result = {'quality': teacher_quality(run, 'teacher_wm_validation'),
              'metrics': detection_metrics(run, 'teacher_wm_validation',
                    read_json(run / 'result/threshold_teacher.json')['threshold'], c['validation_n']),
              'quality_gate_enforced': False}
    freeze(run / 'result/refined_teacher.json', result)


def job(name):
    run = QUEUE / name
    if (run / 'complete.json').exists():
        return
    call(run, '01_teacher', ROOT / 'transfer_refine.py')
    call(run, '02_data', Path(__file__), '--operation', 'data')
    call(run, '03_teacher_score', q.HARNESS / 'semantic_teacher.py', '--stage', 'score', '--directories', 'teacher_wm_validation')
    call(run, '04_teacher_quality', Path(__file__), '--operation', 'teacher_report')
    call(run, '05_cache', q.HARNESS / 'validation.py', '--stage', 'cache')
    call(run, '06_train', q.HARNESS / 'student.py', '--stage', 'train', '--branch', q.BRANCH)
    call(run, '07_validation', q.HARNESS / 'validation.py', '--stage', 'evaluate', '--branch', q.BRANCH)
    labels = [f'validation_{q.BRANCH}_step{s}' for s in read_json(run / 'config.json')['checkpoint_steps']]
    call(run, '08_validation_score', q.HARNESS / 'semantic_teacher.py', '--stage', 'score', '--directories', *labels)
    call(run, '09_validation_quality', q.HARNESS / 'utility.py', '--split', 'validation', '--labels', *labels)
    call(run, '10_select', q.HARNESS / 'validation.py', '--stage', 'select', '--branch', q.BRANCH)
    call(run, '11_test', q.HARNESS / 'student.py', '--stage', 'generate', '--branch', q.BRANCH)
    call(run, '12_score', q.HARNESS / 'semantic_teacher.py', '--stage', 'score', '--directories', q.BRANCH, 'teacher_wm_test')
    call(run, '13_quality', q.HARNESS / 'utility.py', '--labels', 'teacher_clean_test', 'teacher_wm_test', 'unadapted', q.BRANCH)
    call(run, '14_summary', Path(__file__), '--operation', 'summary')


def comparison():
    paths = {'unrefined_scale100': ROOT / 'kappa1_manojb_5k/students/k1_evolution',
             'unrefined_scale085': ROOT / 'kappa1_manojb_5k/students/k1_scale085',
             **{name: QUEUE / name for name in ORDER}}
    rows = {}
    for name, path in paths.items():
        row = read_json(path / 'result/student_summary.json')
        rows[name] = {'detection': row['metrics']['detection_rate'], 'lpips': row['lpips'],
                     'selected_step': row['selected_checkpoint']['selected_step'],
                     **{k: row['utility'][k] for k in ('frechet_inception_distance', 'kernel_inception_distance_mean', 'clip_score')}}
    freeze(QUEUE / 'comparison.json', {'results': rows, 'test_used_for_selection': False,
            'limitation': 'One seed, adaptive exploratory test reuse, extra teacher compute vs unrefined controls'})


def supervise():
    from assets import fcntl
    with (QUEUE / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify()
            for name in ORDER:
                smoke = read_json(QUEUE / name / 'smoke.json')
                if not smoke['passed']:
                    raise RuntimeError('Missing successful technical gradient smoke')
            while not all((DEPENDENCY / 'students' / n / 'complete.json').exists()
                          for n in ['scale085_long22500', 'scale085_conv_restore']):
                status = read_json(DEPENDENCY / 'status.json')
                write_json(QUEUE / 'status.json', {'state': 'waiting_for_recovery', 'dependency': str(DEPENDENCY),
                           'dependency_state': status['state'], 'pid': os.getpid(), 'updated': now()})
                if status['state'] == 'failed':
                    raise RuntimeError('Recovery dependency failed; inspect it before resuming queue')
                pid = read_json(DEPENDENCY / 'launcher.json')['pid']
                proc = Path(f'/proc/{pid}/cmdline')
                if not proc.exists() or b'scale085_recovery.py' not in proc.read_bytes():
                    raise RuntimeError('Recovery supervisor exited before both jobs completed')
                time.sleep(30)
            disk_wait(17., 'campaign_start')
            for name in ORDER:
                job(name)
            comparison()
            write_json(QUEUE / 'status.json', {'state': 'completed', 'updated': now()})
        except Exception as exc:
            write_json(QUEUE / 'status.json', {'state': 'failed', 'error': str(exc), 'updated': now()})
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initialize', action='store_true'); p.add_argument('--launch', action='store_true')
    p.add_argument('--supervise', action='store_true'); p.add_argument('--run', type=Path)
    p.add_argument('--operation', choices=['data', 'teacher_report', 'summary'])
    args = p.parse_args()
    if args.initialize:
        initialize(); print('INITIALIZED'); return
    if args.launch:
        verify()
        previous = QUEUE / 'launcher.json'
        if previous.exists():
            pid = read_json(previous)['pid']; proc = Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'transfer_campaign.py' in proc.read_bytes():
                print('Already queued', pid); return
        # Refuse to launch an untested objective.
        for name in ORDER:
            if not read_json(QUEUE / name / 'smoke.json')['passed']:
                raise RuntimeError('Technical smoke not passed: ' + name)
        env = dict(os.environ, BITMARK_RUNTIME_ROOT='/root/logs/runtime', CUDA_VISIBLE_DEVICES='0',
                   PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        with (QUEUE / 'supervisor.log').open('a') as f:
            proc = subprocess.Popen([sys.executable, str(Path(__file__)), '--supervise'], cwd=ROOT,
                        env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(previous, {'pid': proc.pid, 'queued': now()})
        print('QUEUED', proc.pid); return
    if args.supervise:
        supervise(); return
    verify()
    if args.operation == 'data':
        data(args.run)
    elif args.operation == 'teacher_report':
        teacher_report(args.run)
    elif args.operation == 'summary':
        base.student_summary(args.run)
    else:
        p.error('Choose an operation')


if __name__ == '__main__':
    main()
