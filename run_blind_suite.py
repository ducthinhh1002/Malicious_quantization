"""Run predeclared blind experiments, retain failures, aggregate without test selection."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def configuration(profile):
    # Same batch, step counts and preservation across natural W4/W8 ablations.
    # GAN has extra discriminator compute; report it, never call costs equal.
    common = ["--bits", "8", "4", "--quality-policy", "report",
              "--optimizer", "adamw", "--weight-decay", "0", "--lr-schedule", "warmup_cosine",
              "--natural-preservation", "lowpass", "--preserve-weight", "2",
              "--qat-semantic-preserve-weight", "2", "--warmup-steps", "20"]
    if profile == "pilot":
        return common + ["--steps", "200", "--qat-steps", "200", "--ft-steps", "200",
            "--train-batch-size", "1", "--natural-train-n", "256", "--natural-search-n", "64",
            "--eval-every", "50", "--ft-lr", ".0005"]
    return common + ["--steps", "1000", "--qat-steps", "1000", "--ft-steps", "1000",
        "--train-batch-size", "4", "--natural-train-n", "4000", "--natural-search-n", "256",
        "--natural-resolution", "256", "--eval-every", "100", "--ft-lr", ".0005"]


def collect(run):
    owner_path = run / "owner_evaluation.json"
    if not owner_path.exists():
        return []
    owner = json.loads(owner_path.read_text())
    report = json.loads((run / "report.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    result = []
    for row in owner['rows']:
        method = row['method']
        selected = report['selections'].get(method, {})
        quality = report['branch_quality'].get(method, {})
        fields = ('method', 'comparison_group', 'role', 'bit_accuracy', 'tpr', 'n', 'psnr', 'ssim', 'lpips',
                  'test_quality_valid', 'search_quality_valid', 'reference_valid',
                  'evasion_on_originally_detected', 'tpr_drop_percentage_points',
                  'tpr_drop_ci95_low_pp', 'tpr_drop_ci95_high_pp')
        result.append({"run": str(run), "seed": manifest['args']['seed'],
            **{key: row.get(key) for key in fields},
            "quality_violation_fraction": quality.get('quality_violation_fraction'),
            "valid_updates": selected.get('valid_updates'),
            "train_image_forwards": selected.get('train_image_forwards'),
            "discriminator_image_forwards": selected.get('discriminator_image_forwards', 0),
            "branch_seconds": report.get('branch_compute', {}).get(method, {}).get('search_and_export_seconds_including_profile')})
    return result


def run_suite(root, seeds, profile, extra, execute, runner=subprocess.run):
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve().parent / "run_blind_quantization.sh"
    # Prevent overrides that would redirect a run and invalidate its summary.
    if any(x.split('=')[0] in ('--output', '--seed', '--image-output', '--artifact-output') for x in extra):
        raise ValueError("Suite owns output/seed/artifact paths; use --root and --seeds")
    commands = [{"seed": seed, "output": str(root / f"{root.name}_seed_{seed}"),
        "command": ["bash", str(script), *configuration(profile), "--seed", str(seed),
                    "--output", str(root / f"{root.name}_seed_{seed}"), *extra], "status": "pending"} for seed in seeds]
    protocol = {"profile": profile, "runs": commands,
        "execution": "simulated low-bit weights, FP32 VAE; no native INT4 speed claims",
        "owner_feedback_for_selection": False,
        "surrogate": "not_run_requires_independent_surrogate_assets",
        "paper_reproductions": False,
        "interpretation": "All runs reported; no best seed/method selected using owner test metrics. Repeated prompts across seeds are not independent prompt samples."}
    write_json(root / "suite_plan.json", protocol)
    rows = []
    for item in commands:
        if not execute:
            continue
        item['status'] = 'running'
        write_json(root / "suite_status.json", protocol)
        log = root / f"seed_{item['seed']}.log"
        print(f"Running seed {item['seed']}; log: {log}", flush=True)
        start = time.monotonic()
        try:
            with log.open('w') as stream:
                process = runner(item['command'], stdout=stream, stderr=subprocess.STDOUT, check=False)
            item['returncode'] = process.returncode
            item['status'] = 'completed' if process.returncode == 0 else 'failed'
            rows.extend(collect(Path(item['output'])))
            report_path = Path(item['output']) / 'report.json'
            if report_path.exists():
                failures = json.loads(report_path.read_text()).get('branch_failures', [])
                item['branch_failures'] = failures
                if failures and process.returncode == 0:
                    item['status'] = 'completed_with_branch_failures'
            if process.returncode == 0 and not (Path(item['output']) / 'owner_evaluation.json').exists():
                item['status'] = 'incomplete'
        except Exception as error:
            item.update(status='failed', error=f"{type(error).__name__}: {error}")
        item['seconds'] = time.monotonic() - start
        write_json(root / "suite_status.json", protocol)
        if rows:
            with (root / "suite_results.csv").open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        print(f"Seed {item['seed']}: {item['status']}", flush=True)
    write_json(root / "suite_status.json", protocol)
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[3407, 4407, 5407])
    p.add_argument('--profile', choices=['pilot', 'full'], default='full')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('extra', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error('Seeds must be distinct and nonnegative')
    root = args.root or Path(__file__).resolve().parent / 'output_attack' / (
        'suite_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    protocol = run_suite(root.resolve(), args.seeds, args.profile, extra, not args.plan_only)
    print(f"Suite plan/status/results: {root.resolve()}")
    return int(any(x['status'] in ('failed', 'incomplete', 'completed_with_branch_failures') for x in protocol['runs']))


if __name__ == '__main__':
    sys.exit(main())
