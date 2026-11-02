"""Wait for existing Stage 0 controls, then run the explicitly authorized pilot."""
import json
import os
import subprocess
import time
from pathlib import Path

root = Path('/root/workspace/serum')
os.chdir(root)
status_path = root/'metrics/stage12_queue_status.json'
def status(state, **kwargs):
    record = {'state': state, 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs}
    status_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
try:
    while True:
        missing = []
        for name, session in [('SD14_unadapted', 'quant-sd14-control'), ('C00_seed0', 'quant-c00')]:
            result = root/f'metrics/{name}_scores.npy.result.json'
            if not result.exists():
                missing.append(name)
                if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode:
                    raise RuntimeError(f'{name} did not finish and its tmux session is absent; inspect its log')
            else:
                assert json.loads(result.read_text())['num_images'] == 5000
        if not missing:
            break
        status('waiting_stage0_controls', missing=missing)
        time.sleep(30)
    subprocess.run(['.venv/bin/python', 'scripts/check_data_ready.py'], check=True)
    subprocess.run(['.venv/bin/python', 'scripts/summarize_baseline_controls.py'], check=True)
    while not (root/'metrics/stage12_preflight.json').exists():
        if subprocess.run(['tmux', 'has-session', '-t', 'quant-stage12-preflight'], capture_output=True).returncode:
            raise RuntimeError('Stage 1 preflight failed; inspect logs/stage12_preflight.log')
        status('waiting_stage1_preflight')
        time.sleep(30)
    assert json.loads((root/'metrics/stage12_preflight.json').read_text())['ok']
    # Authorization is limited to the image pilot. Do not create a baseline pass gate.
    status('stage1_running', baseline_gate='not auto-approved')
    pilot = root/'data/pilot'
    pilot.mkdir(parents=True, exist_ok=True)
    for kind in ('clean', 'wm'):
        out = pilot/kind
        out.mkdir(exist_ok=True)
        source = sorted((root/f'data/raw/{kind}_train').glob('*.png'))[:500]
        assert len(source) == 500
        for src in source:
            dst = out/src.name
            if not dst.exists():
                os.link(src, dst)
        assert sorted(p.name for p in out.glob('*.png')) == [p.name for p in source]
    prompts = (root/'prompts/adapt_train.txt').read_text().splitlines()[:500]
    assert len(prompts) == 500
    (pilot/'prompts.txt').write_text('\n'.join(prompts)+'\n')
    workers = []
    for kind, gpu in [('clean', '0'), ('wm', '1')]:
        log = open(root/f'logs/pilot_{kind}.log', 'a', buffering=1)
        workers.append((subprocess.Popen(['bash', 'scripts/run_pilot_worker.sh', kind, gpu], stdout=log, stderr=subprocess.STDOUT), log))
    failures = []
    for worker, log in workers:
        if worker.wait():
            failures.append(str(log.name))
        log.close()
    if failures:
        raise RuntimeError(f'Pilot worker failed; inspect {failures}')
    status('stage2_running')
    subprocess.run(['.venv/bin/python', 'scripts/select_pilot_strengths.py'], check=True)
    selection = json.loads((root/'metrics/pilot_selection.json').read_text())
    status('completed' if selection['status'] == 'selected' else 'needs_preservation_review', selection=selection['status'])
except Exception as exc:
    status('failed', error=str(exc))
    raise
