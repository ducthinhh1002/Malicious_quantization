"""Two controlled recovery trials: convolution restoration and exposure budget.

Retains the original student trainer, frozen detector and paired image protocol.
Does not resume the cancelled .95 scheduler or modify its frozen code manifest.
"""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time
import manojb_5k_campaign as base
from common import freeze, read_json, write_json, sha256

q = base.q
ORIGINAL = q.ROOT / 'kappa1_manojb_5k'
SOURCE = ORIGINAL / 'source'
q.QUEUE = q.ROOT / 'scale085_recovery_20261024'
SPECS = {
    'scale085_long22500': {'vector': [.85] * 6, 'steps': 22500, 'snapshots': [7500, 15000, 22500],
                          'reuse_085_data': True, 'factor': 'student exposure only; 18 passes instead of 6'},
    'scale085_conv_restore': {'vector': [.85, 1., .85, .85, .85, .9], 'steps': 7500,
                            'snapshots': [2500, 5000, 7500], 'reuse_085_data': False,
                            'factor': 'restore down_conv to 1 and up_conv to .9; other groups stay .85'},
}


def own(run, label, operation, vector=None):
    args = [Path(__file__), '--operation', operation, '--run', run]
    if vector is not None:
        args += ['--vector', ','.join(map(str, vector))]
    q.call(run, label, args)
q.own = own


def initialize():
    q.QUEUE.mkdir(parents=True, exist_ok=True)
    plan = read_json(ORIGINAL / 'plan.json')
    plan.update(order=list(SPECS), jobs=list(SPECS),
                purpose='Recover scale .85 transfer; distinguish teacher attenuation from student exposure',
                quality_gates_enforced=False, cancelled_scale095=True,
                inherited_detector=True, final_test_reused_for_comparison=True,
                limitation='Adaptive follow-up; existing test is not a new untouched confirmatory holdout')
    freeze(q.QUEUE / 'plan.json', plan)
    freeze(q.QUEUE / 'specs.json', SPECS)
    manifest = read_json(ORIGINAL / 'code_manifest.json')
    manifest[Path(__file__).name] = sha256(Path(__file__))
    freeze(q.QUEUE / 'code_manifest.json', manifest)
    q.verify_plan()


def prepare(name):
    spec = SPECS[name]; run = q.QUEUE / 'students' / name
    run.mkdir(parents=True, exist_ok=True)
    c = read_json(SOURCE / 'config.json')
    c.update(max_train_steps=spec['steps'], checkpoint_steps=spec['snapshots'])
    freeze(run / 'config.json', c)
    for rel in ('prompts.json', 'assets.json', 'carrier', 'result/threshold.json',
                'result/threshold_teacher.json', 'result/threshold_student.json',
                'images/teacher_clean_validation', 'images/teacher_clean_test',
                'images/unadapted', 'scores/unadapted.json'):
        q.share(SOURCE / rel, run / rel)
    freeze(run / 'teacher_choice.json', {'teacher_kappa': 1., 'vector': spec['vector'],
           'method': name, 'factor': spec['factor'], 'source_checkpoint': str(q.checkpoint(SOURCE)),
           'selection_source': 'fixed hypothesis / existing teacher validation, not test ranking',
           'test_used_for_checkpoint_selection': False})
    if spec['reuse_085_data']:
        prior = ORIGINAL / 'students/k1_scale085'
        for rel in ('images/teacher_wm_train', 'images/teacher_wm_validation', 'images/teacher_wm_test', 'validation_cache'):
            q.share(prior / rel, run / rel)
        freeze(run / 'reuse.json', {'source': str(prior), 'images_and_validation_cache_only': True,
               'optimizer_resume': False,
               'reason': 'Original snapshots contain adapters but no optimizer/RNG state; train from the same clean student seed'})
    return run


def train(name):
    spec = SPECS[name]; run = prepare(name)
    if (run / 'complete.json').exists():
        return
    if not spec['reuse_085_data']:
        own(run, '01_data', 'student_data', spec['vector'])
    q.harness(run, '02_cache', 'validation.py', '--stage', 'cache')
    q.harness(run, '03_train', 'student.py', '--stage', 'train', '--branch', q.BRANCH)
    q.harness(run, '04_validation', 'validation.py', '--stage', 'evaluate', '--branch', q.BRANCH)
    labels = [f'validation_{q.BRANCH}_step{s}' for s in spec['snapshots']]
    q.harness(run, '05_validation_scores', 'semantic_teacher.py', '--stage', 'score', '--directories', *labels)
    q.harness(run, '06_validation_quality', 'utility.py', '--split', 'validation', '--labels', *labels)
    q.harness(run, '07_select', 'validation.py', '--stage', 'select', '--branch', q.BRANCH)
    q.harness(run, '08_test', 'student.py', '--stage', 'generate', '--branch', q.BRANCH)
    q.harness(run, '09_scores', 'semantic_teacher.py', '--stage', 'score', '--directories', q.BRANCH, 'teacher_wm_test')
    q.harness(run, '10_quality', 'utility.py', '--labels', 'teacher_clean_test', 'teacher_wm_test', 'unadapted', q.BRANCH)
    own(run, '11_summary', 'summary')


def comparison():
    import numpy as np
    controls = {'scale085_7500': ORIGINAL / 'students/k1_scale085',
                'scale100_7500': ORIGINAL / 'students/k1_evolution'}
    paths = {**controls, **{name: q.QUEUE / 'students' / name for name in SPECS}}
    rows = {k: read_json(p / 'result/student_summary.json') for k, p in paths.items()}
    threshold = read_json(SOURCE / 'result/threshold_student.json')['threshold']
    reference = rows['scale085_7500']; result = []
    for name, row in rows.items():
        delta = np.array(reference['lpips_per_image']) - np.array(row['lpips_per_image'])
        result.append({'name': name, 'tpr': row['metrics']['detection_rate'], 'lpips': row['lpips'],
                       'lpips_gain_vs085': float(delta.mean()),
                       'lost_085_positives': sum(b >= threshold and s < threshold for b, s in zip(reference['scores'], row['scores'])),
                       'utility': {k:v for k,v in row['utility'].items() if k not in ('clip_per_image', 'image_hashes')},
                       'selected_step': row['selected_checkpoint']['selected_step']})
    freeze(q.QUEUE / 'comparison.json', {'results': result, 'test_used_for_checkpoint_selection': False,
           'limitation': 'Same test panel in an adaptive follow-up; verify success on a fresh holdout and multiple training seeds'})


def control_report():
    """Publish direct .85-versus-1 results as soon as the retained evaluation finishes."""
    a = read_json(ORIGINAL / 'students/k1_scale085/result/student_summary.json')
    b = read_json(ORIGINAL / 'students/k1_evolution/result/student_summary.json')
    keys = ('frechet_inception_distance', 'kernel_inception_distance_mean', 'clip_score')
    freeze(q.QUEUE / 'control_comparison.json', {
        'scale085': {'tpr': a['metrics']['detection_rate'], 'lpips': a['lpips'], **{k:a['utility'][k] for k in keys}},
        'scale100': {'tpr': b['metrics']['detection_rate'], 'lpips': b['lpips'], **{k:b['utility'][k] for k in keys}},
        'paired_test_n': 1000, 'detector_unchanged': True})


def supervise():
    from assets import fcntl
    with (q.QUEUE / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            q.verify_plan()
            write_json(q.QUEUE / 'status.json', {'state': 'running', 'pid': os.getpid(),
                       'jobs': list(SPECS), 'conv_restore': 'waiting_for_scale1_evaluation', 'updated': q.now()})
            def job(name):
                if name == 'scale085_conv_restore':
                    control = ORIGINAL / 'students/k1_evolution/complete.json'
                    while not control.exists():
                        pid = read_json(ORIGINAL / 'cancelled_jobs.json')['retained_evaluation_pid']
                        process = Path(f'/proc/{pid}/cmdline')
                        if not process.exists() or b'manojb_5k_campaign.py' not in process.read_bytes():
                            raise RuntimeError('Scale-1 evaluation stopped before completion; inspect original log')
                        time.sleep(30)
                    control_report()
                    write_json(q.QUEUE / 'control_ready.json', {'completed': True, 'time': q.now()})
                    write_json(ORIGINAL / 'status.json', {'state': 'completed_without_scale095',
                               'cancelled_jobs': ['k1_scale095'], 'updated': q.now()})
                own(q.QUEUE, name, 'train_' + name)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(job, SPECS))
            comparison()
            write_json(q.QUEUE / 'status.json', {'state': 'completed', 'updated': q.now()})
        except Exception as e:
            write_json(q.QUEUE / 'status.json', {'state': 'failed', 'error': str(e), 'updated': q.now()})
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true'); parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--operation'); parser.add_argument('--run', type=Path); parser.add_argument('--vector')
    args = parser.parse_args()
    if args.launch:
        initialize()
        previous = q.QUEUE / 'launcher.json'
        if previous.exists():
            pid = read_json(previous)['pid']; cmd = Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and b'scale085_recovery.py' in cmd.read_bytes():
                print('Already running', pid); return
        env = dict(os.environ, BITMARK_RUNTIME_ROOT='/root/logs/runtime', CUDA_VISIBLE_DEVICES='0',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (q.QUEUE / 'supervisor.log').open('a') as f:
            proc = subprocess.Popen([sys.executable, str(Path(__file__)), '--supervise'], cwd=q.ROOT, env=env,
                                    stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(previous, {'pid': proc.pid, 'started': q.now()}); print('LAUNCHED', proc.pid); return
    if args.supervise:
        supervise(); return
    q.verify_plan()
    if args.operation.startswith('train_'):
        train(args.operation[len('train_'):])
    elif args.operation == 'summary':
        base.student_summary(args.run)
    elif args.operation == 'student_data':
        vector = [float(v) for v in args.vector.split(',')]
        if vector != SPECS[args.run.name]['vector']:
            raise RuntimeError('Unregistered vector')
        q.generate_student_data(args.run, vector)
    else:
        raise ValueError(args.operation)


if __name__ == '__main__':
    main()
