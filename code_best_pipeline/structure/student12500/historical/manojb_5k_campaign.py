"""Pinned Manojb, original 5000/1000 splits, kappa=1 and ordered student trials.

MPaGE-inspired parameter evolution, not its LLM heuristic-synthesis algorithm.
The original teacher objective and student trainer are imported unchanged.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import research_queue as q
from common import freeze, read_json, write_json, sha256, image_name
from group_search import grid_elites, pareto_front, shortlist

q.QUEUE = q.ROOT / 'kappa1_manojb_5k'
SOURCE = q.QUEUE / 'source'
OLD = q.ROOT / 'run'
ORDER = ('k1_scale085', 'k1_evolution', 'k1_scale100', 'k1_scale095')
LEVELS = (.80, .85, .90, .95, 1.)
REVISION = '0094d483a120f3f33dafbd187ea4aa60d10de75c'
PREPARE = q.prepare_run


def checkpoint(source):
    folder = source / 'teacher_fp/candidates/kappa_1.0/checkpoint-3000'
    proof = read_json(folder / 'checkpoint.json')
    if proof['step'] != 3000:
        raise RuntimeError('Wrong teacher checkpoint')
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        if sha256(folder / name) != proof['files'][name]:
            raise RuntimeError('Teacher checkpoint changed')
    return folder
q.checkpoint = checkpoint


def own(run, label, operation, vector=None):
    args = [Path(__file__), '--operation', operation, '--run', run]
    if vector is not None:
        args += ['--vector', ','.join(map(str, vector))]
    q.call(run, label, args)
q.own = own


def prepare(run, pilot_n=None, shared_student=False):
    PREPARE(run, pilot_n, False)
    if shared_student:
        for rel in ('result/threshold_student.json', 'result/threshold.json',
                    'images/teacher_clean_test', 'images/unadapted', 'scores/unadapted.json'):
            q.share(SOURCE / rel, run / rel)
    return run
q.prepare_run = prepare


def propose(rows, seen, seed, random=False):
    """Grid diversity, cross-cell recombination and bounded discrete mutation."""
    import numpy as np
    rng = np.random.default_rng(seed)
    cells = grid_elites(rows)
    n = len(rows[0]['vector'])
    for attempt in range(2000):
        if random or attempt > 100:
            vector = rng.choice(LEVELS, n).tolist()
        else:
            a = int(rng.integers(len(cells)))
            others = [i for i in range(len(cells)) if i != a] or [a]
            b = int(rng.choice(others))
            left = cells[a][int(rng.integers(len(cells[a])))]['vector']
            right = cells[b][int(rng.integers(len(cells[b])))]['vector']
            vector = [left[i] if rng.random() < .5 else right[i] for i in range(n)]
            j = int(rng.integers(n))
            vector[j] = float(rng.choice([v for v in LEVELS if v != vector[j]]))
        if tuple(vector) not in seen:
            return vector
    raise RuntimeError('Search space exhausted')


def choose_evolution(rows, baseline, threshold):
    """Validation-only decision; failed detection is reported, never hidden."""
    annotated = []
    margin_floor = .9 * max(0., baseline['margin_p10'])
    for row in rows:
        lost = sum(b >= threshold and s < threshold for b, s in zip(baseline['scores'], row['scores']))
        annotated.append({**row, 'lost_positive_images': lost,
                          'detection_preserved': lost == 0 and row['margin_p10'] >= margin_floor})
    eligible = [r for r in annotated if r['detection_preserved']]
    if eligible:
        selected = min(eligible, key=lambda r: (r['lpips'], r['clip_drop'], -r['margin_p10'], r['tag']))
    else:
        selected = min(annotated, key=lambda r: (r['lost_positive_images'], -r['margin_p10'], r['lpips'], r['tag']))
    return {'selected': selected, 'candidates': annotated, 'margin_floor': margin_floor,
            'test_used': False, 'quality_stopping_gate': False,
            'rule': 'No lost scale-1 positives and >=90% positive p10 margin, then LPIPS/CLIP; '
                    'if none qualify, train best detection candidate and mark constraint failure'}


def verify_code():
    for rel, digest in read_json(q.QUEUE / 'code_manifest.json').items():
        if sha256(q.ROOT / rel) != digest:
            raise RuntimeError('Frozen code changed: ' + rel)


def initialize():
    q.QUEUE.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(exist_ok=True)
    c = read_json(OLD / 'config.json')
    model = Path(c['cache']) / 'Manojb--stable-diffusion-2-1-base' / REVISION
    c.update(train_n=5000, validation_n=1000, validation_generation_n=1000,
             teacher_snapshot=str(model), teacher_kappas=[1.], teacher_micro_batch=4,
             threshold_quality=False, train_clean_controls=False, ranks=[32],
             adapter_scopes=['attention_conv'], lora_only=True, dora=False,
             teacher_objective='threshold_cfg_v1', teacher_train_steps=3000,
             max_train_steps=7500, checkpoint_steps=[2500, 5000, 7500])
    freeze(SOURCE / 'config.json', c)
    files = [Path(__file__), q.ROOT / 'research_queue.py', q.ROOT / 'group_search.py']
    files += list(q.HARNESS.glob('*.py')) + [q.HARNESS / 'vendor/provenance.json']
    files += list((q.HARNESS / 'vendor/diffusers').glob('*.py'))
    freeze(q.QUEUE / 'code_manifest.json', {str(p.relative_to(q.ROOT)): sha256(p) for p in files})
    freeze(q.QUEUE / 'campaign.json', {
        'order': list(ORDER), 'teacher_repository': 'Manojb/stable-diffusion-2-1-base',
        'teacher_revision': REVISION, 'source_config_sha256': sha256(SOURCE / 'config.json'),
        'teacher_kappa': 1., 'teacher_from_clean': True,
        'splits': {'train': 5000, 'validation': 1000, 'calibration': 5000, 'test': 1000},
        'old_carrier_and_thresholds_reused': False,
        'reason': 'Original split offsets move with train size; recompute PCA/null to avoid overlap',
        'same_key_seed': c['key_seed'], 'same_sampler_steps': c['inference_steps'],
        'quality_stopping_gates': False, 'levels': list(LEVELS),
        'pilot_n': 64, 'new_evaluations_per_policy': 16, 'search_seed': 20261024,
        'full_validation_shortlist': 3,
        'research': 'https://arxiv.org/abs/2507.20923',
        'method': 'MPaGE-inspired Pareto-grid parameter evolution; no LLM synthesis',
        'test_used_for_search': False})


def assets():
    from semantic_assets import snapshot_identity, verified_student_snapshot
    from assets import prepare_prompts
    c = read_json(SOURCE / 'config.json')
    teacher = Path(c['teacher_snapshot'])
    verified = read_json(teacher / 'verified.json')
    if verified['repository'] != 'Manojb/stable-diffusion-2-1-base' or verified['revision'] != REVISION:
        raise RuntimeError('Wrong Manojb provenance')
    identity = snapshot_identity(teacher)
    if identity != {r['file']: r['sha256'] for r in verified['files']}:
        raise RuntimeError('Pinned Manojb files changed')
    a = prepare_prompts(c, SOURCE, Path(c['cache']))
    student = verified_student_snapshot(c['student_snapshot'])
    a.update(method='semantic_fp', teacher_clean=str(teacher), teacher=str(teacher), student=student,
             teacher_source={'repository': verified['repository'], 'revision': REVISION},
             student_revision='133a221b8aa7292a167afc5127cb63fb5005638b',
             teacher_clean_hashes=identity, student_hashes=snapshot_identity(student),
             prompt_sha256=sha256(SOURCE / 'prompts.json'))
    freeze(SOURCE / 'assets.json', a)
    # Reuse only complete batches with identical prompts, order, model and seeds.
    # This avoids assuming batch-composition invariance in floating point kernels.
    old_c = read_json(OLD / 'config.json'); old_p = read_json(OLD / 'prompts.json')
    prompts = read_json(SOURCE / 'prompts.json'); batch = c['generation_batch']
    records = []
    if old_c['seed'] == c['seed']:
        bank = {}
        for split in ('train', 'validation', 'calibration', 'test'):
            directory = OLD / 'images' / f'teacher_clean_{split}'
            if not (directory / 'image_hashes.json').exists():
                continue
            g = read_json(directory / 'generation.json')
            if (g['model_hashes'] != identity or g['batch_size'] != batch or
                    g['steps'] != c['inference_steps'] or g['cfg'] != 7.5 or
                    g['dtype'] != 'float16' or g['resolution'] != 512):
                continue
            entries = read_json(directory / 'image_hashes.json')
            for start in range(0, len(entries), batch):
                group = old_p[split][start:start + batch]
                key = tuple((r['image_id'], r['text']) for r in group)
                bank[key] = (directory, entries[start:start + batch])
        for split, rows in prompts.items():
            for start in range(0, len(rows), batch):
                key = tuple((r['image_id'], r['text']) for r in rows[start:start + batch])
                if key not in bank:
                    continue
                directory, entries = bank[key]
                for j, entry in enumerate(entries):
                    src = directory / entry['file_name']
                    row = rows[start + j]
                    if (sha256(src) != entry['sha256'] or entry['text'] != row['text'] or
                            entry['seed'] != c['seed'] * 1000000 + row['image_id']):
                        raise RuntimeError('Old image reuse mismatch')
                    dst = SOURCE / 'images' / f'teacher_clean_{split}' / image_name(start + j)
                    q.link_file(src, dst)
                    records.append({'source': str(src), 'destination': str(dst), 'sha256': entry['sha256']})
    freeze(SOURCE / 'reused_clean_images.json', {'count': len(records), 'files': records})
    print('Verified and reused clean images:', len(records), flush=True)


def stage(label, file, *args):
    done = SOURCE / 'stages' / (label + '.json')
    if done.exists():
        return
    q.harness(SOURCE, label, file, *args)
    freeze(done, {'completed': True, 'code_manifest_sha256': sha256(q.QUEUE / 'code_manifest.json')})


def prepare_source():
    if not (SOURCE / 'stages/assets.json').exists():
        assets()
        freeze(SOURCE / 'stages/assets.json', {'complete': True})
    stage('01_clean', 'semantic_teacher.py', '--stage', 'clean')
    stage('02_carrier', 'semantic_teacher.py', '--stage', 'carrier')
    stage('03_teacher_scores', 'semantic_teacher.py', '--stage', 'score', '--directories', 'teacher_clean_calibration')
    stage('04_teacher_null', 'semantic_report.py', '--teacher-null-only')
    # Call a single candidate directly: record quality, never use the old quality selector.
    stage('05_teacher_k1', 'semantic_teacher.py', '--stage', 'candidate', '--kappa', '1')
    stage('06_student_null_images', 'student.py', '--stage', 'generate', '--branch', 'unadapted', '--split', 'calibration')
    stage('07_student_null_scores', 'semantic_teacher.py', '--stage', 'score', '--directories', 'unadapted_calibration')
    stage('08_thresholds', 'report.py', '--calibrate-only')
    if not (SOURCE / 'stages/clean_test.json').exists():
        own(SOURCE, '09_clean_test', 'clean_test')
        freeze(SOURCE / 'stages/clean_test.json', {'complete': True})
    stage('10_unadapted_test', 'student.py', '--stage', 'generate', '--branch', 'unadapted')
    stage('11_unadapted_scores', 'semantic_teacher.py', '--stage', 'score', '--directories', 'unadapted')
    from safetensors import safe_open
    from group_search import group_for_key
    cp = checkpoint(SOURCE)
    with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt') as f:
        groups = sorted({group_for_key(k) for k in f.keys()})
    if len(groups) != 6:
        raise RuntimeError('Unexpected adapter groups')
    plan = {**read_json(q.QUEUE / 'campaign.json'), 'source_run': str(SOURCE), 'source_checkpoint': str(cp),
            'source_config_sha256': sha256(SOURCE / 'config.json'),
            'source_adapter_sha256': sha256(cp / 'adapter_model.safetensors'),
            'carrier_sha256': sha256(SOURCE / 'carrier/complete.json'),
            'threshold_teacher_sha256': sha256(SOURCE / 'result/threshold_teacher.json'),
            'prompts_sha256': sha256(SOURCE / 'prompts.json'), 'groups': groups}
    freeze(q.QUEUE / 'plan.json', plan)
    q.verify_plan()


def student_summary(run):
    import numpy as np
    import torch
    import lpips
    from semantic_teacher import pixels, numerical_policy
    from semantic_report import detection_metrics, score_rows
    numerical_policy()
    c = read_json(run / 'config.json'); n = c['test_n']
    threshold = read_json(run / 'result/threshold_student.json')['threshold']
    left = read_json(run / 'images/unadapted/generation.json')
    right = read_json(run / 'images' / q.BRANCH / 'generation.json')
    if {k:v for k,v in left.items() if k != 'checkpoint'} != {k:v for k,v in right.items() if k != 'checkpoint'}:
        raise RuntimeError('Unpaired student LPIPS')
    model = lpips.LPIPS(net='alex', version='0.1').cuda().eval().requires_grad_(False)
    distances = []
    for start in range(0, n, c['score_batch']):
        names = [image_name(i) for i in range(start, min(start + c['score_batch'], n))]
        with torch.inference_mode():
            distances += model(pixels([run / 'images/unadapted' / f for f in names]),
                               pixels([run / 'images' / q.BRANCH / f for f in names])).flatten().cpu().tolist()
    result = {'metrics': detection_metrics(run, q.BRANCH, threshold, n),
              'scores': [r['score'] for r in score_rows(run, q.BRANCH, n)],
              'lpips': float(np.mean(distances)), 'lpips_per_image': distances,
              'utility': read_json(run / 'result/utility.json')['branches'][q.BRANCH],
              'choice': read_json(run / 'teacher_choice.json'),
              'selected_checkpoint': read_json(run / 'result/selection' / (q.BRANCH + '.json')),
              'test_used_for_selection': False, 'quality_gates_enforced': False}
    freeze(run / 'result/student_summary.json', result)
    freeze(run / 'complete.json', {'completed': True, 'report': 'result/student_summary.json'})


def train_student(name, vector):
    run = prepare(q.QUEUE / 'students' / name, shared_student=True)
    freeze(run / 'teacher_choice.json', {'teacher_kappa': 1., 'vector': vector, 'method': name,
                                        'source_checkpoint': str(checkpoint(SOURCE)), 'test_used': False})
    if (run / 'complete.json').exists():
        return
    # An evolutionary fallback may equal a fixed scale. Reuse that exact seed,
    # dataset and checkpoint instead of spending another 7500 identical updates.
    for prior in sorted((q.QUEUE / 'students').glob('*')):
        if prior == run or not (prior / 'complete.json').exists():
            continue
        if read_json(prior / 'teacher_choice.json')['vector'] != vector:
            continue
        if sha256(prior / 'config.json') != sha256(run / 'config.json'):
            raise RuntimeError('Cannot reuse student with different protocol')
        for rel in ('images', 'students', 'scores', 'validation_cache'):
            q.share(prior / rel, run / rel)
        for artifact in (prior / 'result').iterdir():
            if artifact.name != 'student_summary.json':
                q.share(artifact, run / 'result' / artifact.name)
        summary = read_json(prior / 'result/student_summary.json')
        summary.update(choice=read_json(run / 'teacher_choice.json'), reused_exact_student=str(prior))
        freeze(run / 'result/student_summary.json', summary)
        freeze(run / 'reuse.json', {'source': str(prior), 'reason': 'Identical vector, dataset, seed and protocol'})
        freeze(run / 'complete.json', {'completed': True, 'report': 'result/student_summary.json'})
        return
    own(run, '01_teacher_data', 'student_data', vector)
    q.harness(run, '02_cache', 'validation.py', '--stage', 'cache')
    q.harness(run, '03_train', 'student.py', '--stage', 'train', '--branch', q.BRANCH)
    q.harness(run, '04_validation', 'validation.py', '--stage', 'evaluate', '--branch', q.BRANCH)
    labels = [f'validation_{q.BRANCH}_step{s}' for s in (2500, 5000, 7500)]
    q.harness(run, '05_validation_scores', 'semantic_teacher.py', '--stage', 'score', '--directories', *labels)
    q.harness(run, '06_validation_quality', 'utility.py', '--split', 'validation', '--labels', *labels)
    q.harness(run, '07_select', 'validation.py', '--stage', 'select', '--branch', q.BRANCH)
    q.harness(run, '08_test', 'student.py', '--stage', 'generate', '--branch', q.BRANCH)
    q.harness(run, '09_scores', 'semantic_teacher.py', '--stage', 'score', '--directories', q.BRANCH, 'teacher_wm_test')
    q.harness(run, '10_quality', 'utility.py', '--labels', 'teacher_clean_test', 'teacher_wm_test', 'unadapted', q.BRANCH)
    own(run, '11_summary', 'student_summary')


def evolution():
    plan = read_json(q.QUEUE / 'plan.json'); n = len(plan['groups'])
    pilot = prepare(q.QUEUE / 'evolution/pilot', pilot_n=plan['pilot_n'])
    seeds = [[.85] * n, [1.] * n] + [[1. if i == j else .85 for i in range(n)] for j in range(n)]
    initial = q.eval_many(pilot, seeds)
    archives = {mode: list(initial) for mode in ('pareto', 'random')}
    for i in range(plan['new_evaluations_per_policy']):
        vectors = []
        for mode in ('pareto', 'random'):
            rows = archives[mode]
            seed = plan['search_seed'] + i + (10000 if mode == 'random' else 0)
            vector = propose(rows, {tuple(r['vector']) for r in rows}, seed, random=mode == 'random')
            freeze(q.QUEUE / 'evolution/proposals' / f'{mode}_{i:02d}.json', {'vector': vector, 'seed': seed})
            vectors.append(vector)
        for mode, row in zip(('pareto', 'random'), q.eval_many(pilot, vectors)):
            archives[mode].append(row)
        write_json(q.QUEUE / 'evolution/progress.json', {'round': i + 1, 'budget': 16})
    freeze(q.QUEUE / 'evolution/archive.json', {'archives': archives,
           'fronts': {k: pareto_front(v) for k, v in archives.items()}, 'test_used': False})
    candidates = shortlist(archives['pareto'], count=3)
    full = prepare(q.QUEUE / 'evolution/full_validation')
    vectors = list(dict.fromkeys(tuple(r['vector']) for r in candidates + initial[:2]))
    rows = q.eval_many(full, [list(v) for v in vectors])
    baseline = next(r for r in rows if r['vector'] == [1.] * n)
    # Include .85 and 1.0 controls as fallbacks; never manufacture an improvement.
    decision = choose_evolution(rows, baseline, read_json(full / 'result/threshold_teacher.json')['threshold'])
    freeze(q.QUEUE / 'evolution/selection.json', decision)
    return decision['selected']['vector']


def finish_comparisons():
    import numpy as np
    base = read_json(q.QUEUE / 'students/k1_scale100/result/student_summary.json')
    threshold = read_json(SOURCE / 'result/threshold_student.json')['threshold']
    rows = []
    for name in ORDER:
        r = read_json(q.QUEUE / 'students' / name / 'result/student_summary.json')
        delta = np.array(base['lpips_per_image']) - np.array(r['lpips_per_image'])
        rng = np.random.default_rng(20261024)
        bootstrap = np.concatenate([delta[rng.integers(0, len(delta), (200, len(delta)))].mean(1) for _ in range(10)])
        rows.append({'name': name, 'tpr': r['metrics']['detection_rate'], 'lpips': r['lpips'],
                     'lpips_gain': float(delta.mean()), 'lpips_gain_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
                     'lost_baseline_positives': sum(b >= threshold and s < threshold for b, s in zip(base['scores'], r['scores'])),
                     'utility': r['utility'], 'vector': r['choice']['vector']})
    freeze(q.QUEUE / 'comparison.json', {'baseline': 'k1_scale100', 'results': rows, 'quality_gates_enforced': False,
                                       'note': 'One training seed; image bootstrap does not measure training-seed variance'})


def supervise():
    from assets import fcntl
    with (q.QUEUE / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify_code()
            status = lambda job: write_json(q.QUEUE / 'status.json', {'state': 'running', 'job': job, 'pid': os.getpid(), 'updated': q.now()})
            status('prepare_5000_1000_and_train_k1_teacher')
            prepare_source()
            vectors = {'k1_scale085': [.85] * 6, 'k1_scale100': [1.] * 6, 'k1_scale095': [.95] * 6}
            for name in ORDER:
                status(name)
                if name == 'k1_evolution':
                    vector = evolution()
                else:
                    vector = vectors[name]
                own(q.QUEUE, name, 'train_' + name, vector)
            finish_comparisons()
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
        initialize(); verify_code()
        previous = q.QUEUE / 'launcher.json'
        if previous.exists():
            pid = read_json(previous)['pid']; cmd = Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and b'manojb_5k_campaign.py' in cmd.read_bytes():
                print('Already running', pid); return
        env = dict(os.environ, BITMARK_RUNTIME_ROOT='/root/logs/runtime', CUDA_VISIBLE_DEVICES='0',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (q.QUEUE / 'supervisor.log').open('a') as f:
            proc = subprocess.Popen([sys.executable, str(Path(__file__)), '--supervise'], cwd=q.ROOT, env=env,
                                    stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(previous, {'pid': proc.pid, 'started': q.now()}); print('LAUNCHED', proc.pid); return
    if args.supervise:
        supervise(); return
    verify_code()
    if args.operation == 'clean_test':
        from semantic_teacher import generate_set
        generate_set(SOURCE, read_json(SOURCE / 'assets.json')['teacher_clean'], 'test', 'teacher_clean_test')
        return
    q.verify_plan()
    vector = [float(v) for v in args.vector.split(',')] if args.vector else None
    if args.operation.startswith('train_'):
        name = args.operation[len('train_'):]
        expected = {'k1_scale085': [.85] * 6, 'k1_scale100': [1.] * 6, 'k1_scale095': [.95] * 6}
        if name == 'k1_evolution':
            expected[name] = read_json(q.QUEUE / 'evolution/selection.json')['selected']['vector']
        if expected.get(name) != vector:
            raise RuntimeError('Unregistered student candidate')
        train_student(name, vector)
    elif args.operation == 'student_summary':
        student_summary(args.run)
    else:
        q.worker(args.operation, args.run, vector)


if __name__ == '__main__':
    main()
