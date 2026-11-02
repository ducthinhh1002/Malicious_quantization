"""Append two owner-only ablations after transfer_refine, without editing it."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import transfer_campaign as shared
from common import freeze, read_json, write_json, sha256

ROOT = shared.ROOT
DEPENDENCY = ROOT / 'transfer_refine_20261024'
QUEUE = ROOT / 'minhnqn_refine_20261024'
ORDER = ['quality_gradient', 'quality_cka']
COMMON = {**shared.COMMON, 'transfer': False, 'sketch_width': 512, 'cka_grid': 8}
SPECS = {
    'quality_gradient': {**COMMON, 'method': 'gradient', 'method_every': 10, 'method_weight': .05},
    'quality_cka': {**COMMON, 'method': 'cka', 'method_every': 1, 'method_weight': .05},
}
FILES = ['minhnqn_campaign.py', 'minhnqn_refine.py', 'minhnqn_objectives.py',
         'test_minhnqn.py', 'transfer_campaign.py', 'transfer_refine.py', 'transfer_proxy.py']
# Reuse helpers only in this isolated process. No files in the earlier queue change.
shared.QUEUE, shared.ORDER, shared.SPECS = QUEUE, ORDER, SPECS
now, verify, call = shared.now, shared.verify, shared.call


def prepare(name):
    run = QUEUE / name
    run.mkdir(parents=True, exist_ok=True)
    spec = SPECS[name]
    config = read_json(shared.SOURCE / 'config.json')
    config.update(teacher_train_steps=spec['teacher_steps'], teacher_lr=spec['teacher_lr'],
                  teacher_checkpoint_every=spec['checkpoint_every'])
    freeze(run / 'config.json', config)
    freeze(run / 'refine_spec.json', spec)
    for rel in ['assets.json', 'prompts.json', 'carrier', 'result/threshold_teacher.json',
                'result/threshold_student.json', 'result/threshold.json', 'teacher_fp/calibration_target.json',
                'images/teacher_clean_train', 'images/teacher_clean_validation', 'images/teacher_clean_test',
                'images/unadapted', 'scores/unadapted.json']:
        shared.q.share(shared.SOURCE / rel, run / rel)
    source = shared.base.checkpoint(shared.SOURCE)
    freeze(run / 'teacher_choice.json', {
        'method': name, 'source_checkpoint': str(source),
        'source_adapter_sha256': sha256(source / 'adapter_model.safetensors'),
        'teacher_kappa': 1., 'initial_effective_scale': 1., 'source_step': 3000,
        'refinement_steps': spec['teacher_steps'], 'teacher_selection': 'fixed final step',
        'student_trainer': 'unchanged ordinary rank32 attention_conv / 7500 updates / batch4',
        'quality_stopping_gates': False, 'test_used_for_selection': False,
        'reference': 'fixed source teacher for gradient; frozen clean for LPIPS and CKA',
        'method_definition': ('signed CountSketch delta-denoising-gradient preservation; same SD1.4 rank32 conv probe; '
                              'one detached clean warmup; source-minus-clean reference; no student distillation'
                              if spec['method'] == 'gradient' else
                              'centered spatial patch CKA on frozen LPIPS AlexNet layers 1,2; 8x8 patch grid'),
        'limitations': 'Low-noise predicted-x0 view is a proxy for native sampling; one exploratory seed. '
                       'Gradient matching is a local MSE-gradient proxy, not final AdamW trajectory or detection proof.'})


def initialize():
    QUEUE.mkdir(parents=True, exist_ok=True)
    freeze(QUEUE / 'code_manifest.json', {name: sha256(ROOT / name) for name in FILES})
    freeze(QUEUE / 'specs.json', SPECS)
    freeze(QUEUE / 'plan.json', {
        'order': ORDER, 'dependency': str(DEPENDENCY),
        'source_run': str(shared.SOURCE),
        'source_code_manifest_sha256': sha256(ROOT / 'kappa1_manojb_5k/code_manifest.json'),
        'dependency_code_manifest_sha256': sha256(DEPENDENCY / 'code_manifest.json'),
        'student_threshold_sha256': sha256(shared.SOURCE / 'result/threshold_student.json'),
        'calibration_target_sha256': sha256(shared.SOURCE / 'teacher_fp/calibration_target.json'),
        'original_teacher_objective_preserved': True, 'final_student_objective_unchanged': True,
        'gates': 'integrity/numerical/resource only; no quality/detection stopping gate',
        'execution': 'dependency completion -> free GPU check -> actual GPU smoke -> full job, sequential',
        'quality_metrics': ['LPIPS', 'FID', 'KID', 'CLIP'],
        'reuse': 'hard-link all existing clean images, carrier, prompts and thresholds',
        'test_protocol': 'exploratory reused test; never used to select checkpoint',
        'controls': ['transfer_refine_20261024/quality', 'transfer_refine_20261024/quality_transfer',
                     'kappa1_manojb_5k/students/k1_evolution'],
        'gpu_free_min_mib': 115000, 'disk_free_min_gib': 24})
    verify()
    for name in ORDER:
        prepare(name)


def job(name):
    run = QUEUE / name
    if (run / 'complete.json').exists():
        return
    call(run, '00_smoke', ROOT / 'minhnqn_refine.py', '--smoke')
    result = read_json(run / 'smoke.json')
    if not result['passed'] or result.get('method_verified') != SPECS[name]['method']:
        raise RuntimeError('New objective did not pass actual-model GPU smoke')
    call(run, '01_teacher', ROOT / 'minhnqn_refine.py')
    # Existing job starts at 01; marker above skips it, all downstream stages are identical.
    # Route child operations to this entry point, preserving this queue's integrity context.
    original_call = shared.call
    def dispatch(run, label, script, *args):
        if Path(script).name == 'transfer_campaign.py':
            script = Path(__file__)
        return original_call(run, label, script, *args)
    shared.call = dispatch
    try:
        shared.job(name)
    finally:
        shared.call = original_call


def resources_ready():
    free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
    return free >= 115000, free


def supervise():
    from assets import fcntl
    with (QUEUE / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify()
            while True:
                if sha256(DEPENDENCY / 'code_manifest.json') != read_json(QUEUE / 'plan.json')['dependency_code_manifest_sha256']:
                    raise RuntimeError('Dependency manifest changed')
                state = read_json(DEPENDENCY / 'status.json')
                completed = state['state'] == 'completed' and all(
                    (DEPENDENCY / n / 'complete.json').exists() for n in ['quality', 'quality_transfer'])
                if completed:
                    break
                pid = read_json(DEPENDENCY / 'launcher.json')['pid']
                proc = Path(f'/proc/{pid}/cmdline')
                alive = proc.exists() and b'transfer_campaign.py' in proc.read_bytes()
                write_json(QUEUE / 'status.json', {
                    'state': 'waiting_for_dependency' if alive and state['state'] != 'failed' else 'dependency_needs_attention',
                    'dependency': str(DEPENDENCY), 'dependency_state': state['state'],
                    'dependency_alive': bool(alive), 'order': ORDER,
                    'gpu_smoke': 'pending until dependency completes', 'pid': os.getpid(), 'updated': now()})
                time.sleep(30)
            for name in ORDER:
                while True:
                    ready, free = resources_ready()
                    if ready:
                        break
                    write_json(QUEUE / 'status.json', {'state': 'waiting_for_gpu', 'free_mib': free,
                               'job': name, 'pid': os.getpid(), 'updated': now()})
                    time.sleep(30)
                shared.disk_wait(24., name + '/start')
                job(name)
            shared.comparison()
            # Include the matched LPIPS-only control and prior transfer refinement in final table.
            rows = read_json(QUEUE / 'comparison.json')
            rows['preceding_campaign'] = read_json(DEPENDENCY / 'comparison.json')
            freeze(QUEUE / 'comparison_with_controls.json', rows)
            write_json(QUEUE / 'status.json', {'state': 'completed', 'updated': now()})
        except Exception as exc:
            write_json(QUEUE / 'status.json', {'state': 'failed', 'error': str(exc), 'updated': now()})
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--operation', choices=['data', 'teacher_report', 'summary'])
    args = parser.parse_args()
    if args.initialize:
        initialize(); print('INITIALIZED'); return
    verify()
    if args.launch:
        test = read_json(QUEUE / 'unit_tests.json')
        if not test['passed'] or test['code_manifest_sha256'] != sha256(QUEUE / 'code_manifest.json'):
            raise RuntimeError('Unit tests do not cover the frozen manifest')
        previous = QUEUE / 'launcher.json'
        if previous.exists():
            pid = read_json(previous)['pid']; proc = Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'minhnqn_campaign.py' in proc.read_bytes():
                print('Already queued', pid); return
        env = dict(os.environ, BITMARK_RUNTIME_ROOT='/root/logs/runtime', CUDA_VISIBLE_DEVICES='0',
                   PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        with (QUEUE / 'supervisor.log').open('a') as stream:
            proc = subprocess.Popen([sys.executable, str(Path(__file__)), '--supervise'], cwd=ROOT,
                    env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(previous, {'pid': proc.pid, 'queued': now()})
        print('QUEUED', proc.pid); return
    if args.supervise:
        supervise(); return
    if args.operation == 'data':
        shared.data(args.run)
    elif args.operation == 'teacher_report':
        shared.teacher_report(args.run)
    elif args.operation == 'summary':
        shared.base.student_summary(args.run)
    else:
        parser.error('Choose an operation')


if __name__ == '__main__':
    main()
