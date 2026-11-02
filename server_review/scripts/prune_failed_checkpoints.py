"""Audited one-file-at-a-time pruning for explicitly reviewed historical vm4 runs.

Run on vm4. The default writes a plan, --apply consumes that immutable plan.
No recursive deletes, images, reports, prefix caches, environments or hub assets.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat

ROOT = Path('/root/logs')
WEIGHTS = {'.pt', '.pth', '.safetensors', '.ckpt', '.bin'}
PRESERVE = [ROOT / 'runtime/cache', ROOT / 'torch_shared', ROOT / 'gate_env',
            ROOT / 'bitgreen_integration/models']


def targets():
    specs = []
    for name in ('bitgreen_sd21_20261017_101804', 'bitgreen_constrained_20261017_100852',
                 'bitgreen_stablegrad_1017_acc4', 'bitgreen_transport_1017'):
        evidence = ROOT / 'results' / name / 'result/teacher_gate.json'
        report = json.loads(evidence.read_text())
        for arm in ('plain', 'robust'):
            row = report['arms'][arm]
            if row['eligible'] or row['tpr'] > .02 or report['run_students']:
                raise RuntimeError('Run no longer meets reviewed failure criterion: ' + name)
            specs.append((ROOT / 'runtime/bitgreen_artifacts' / name / 'models' / arm,
                str(evidence), dict(reason='failed_teacher_near_clean_fpr', arm=arm, **row)))
    # Retain robust acc1 as a representative negative checkpoint; retain quality
    # robust as the best weak-signal diagnostic, and the earlier strong-signal pair.
    for name, arm, maximum in [('bitgreen_stablegrad_1017_acc1', 'plain', .02),
                                ('bitgreen_quality_20261017_082439_942427', 'plain', .05)]:
        evidence = ROOT / 'results' / name / 'result/teacher_gate.json'
        report = json.loads(evidence.read_text()); row = report['arms'][arm]
        if row['eligible'] or row['tpr'] > maximum or report['run_students']:
            raise RuntimeError('Not a reviewed failed/superseded arm')
        specs.append((ROOT / 'runtime/bitgreen_artifacts' / name / 'models' / arm,
                      str(evidence), dict(reason='failed_arm_with_representative_retained', **row)))
    for name in ('distribution_gate_20261017_001814', 'centered_teacher_20261017_060553'):
        evidence = ROOT / 'results' / name / 'result/teacher_summary.csv'
        rows = list(csv.DictReader(evidence.open()))
        for arm in ('control', 'clip', 'unet'):
            steps = (250, 500) if name.startswith('distribution_gate') else (250, 1000)
            for step in steps:
                metrics = [r for r in rows if r['arm'] == arm and int(r['step']) == step]
                if not metrics or any(float(r['image_tpr']) > .025 or r['inheritance_tested'] != 'False' for r in metrics):
                    raise RuntimeError('Distribution checkpoint not supported by failure evidence')
                specs.append((ROOT / 'runtime/distribution_artifacts' / name / 'arms' / arm / 'models' / f'teacher_{step}',
                    str(evidence), dict(reason='near_fpr_teacher; centered_step500_matched_triplet_retained',
                                        arm=arm, step=step, metrics=metrics)))
    selections = [('results/teacher_gate_improved/proxy', .05),
                  ('results/gate_20261016_144609_3785966/without_c3', .04)]
    for name, maximum in selections:
        evidence = ROOT / name / 'result/teacher_selection.json'
        report = json.loads(evidence.read_text())
        if report['valid_design_winner'] or any(r['teacher_tpr'] > maximum for r in report['measurements']):
            raise RuntimeError('Teacher unexpectedly has a valid/strong signal')
        specs.append((ROOT / name / 'models' / 'teacher_250', str(evidence),
                      dict(reason='invalid_weak_teacher', measurements=report['measurements'])))
    # Completed, non-scientific technical smoke exports. Keep LALT smoke intact.
    for name in ('distribution_smoke/arms/clip/models/teacher_1',
                 'distribution_smoke/arms/control/models/teacher_1',
                 'runtime/bitgreen_artifacts/bitgreen_quality_smoke/models/plain',
                 'runtime/native_transfer_artifacts/native_transfer_20261017_174732Z_smoke_38fe2fce53/models/smoke'):
        specs.append((ROOT / name, 'completed historical technical smoke export',
                      dict(reason='technical_smoke_export_no_scientific_checkpoint_value')))
    return specs


def safe(path, allowed):
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(ROOT.resolve()) or not resolved.is_relative_to(allowed.resolve()):
        raise RuntimeError('Path escaped the reviewed checkpoint folder: ' + str(path))
    if any(resolved.is_relative_to(p.resolve()) for p in PRESERVE):
        raise RuntimeError('Protected shared assets')
    if any('lalt_' in part or part == 'prefix_cache' for part in resolved.parts):
        raise RuntimeError('Active LALT or cached data is protected')
    return resolved


def open_files():
    result = set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for fd in (entry / 'fd').iterdir():
                try:
                    result.add(fd.resolve(strict=True))
                except (OSError, RuntimeError):
                    pass
        except OSError:
            pass
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    if not args.apply:
        rows = []
        opened = open_files()
        for folder, evidence, metrics in targets():
            if not folder.exists():
                continue
            for base, dirs, names in os.walk(folder, followlinks=False):
                dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
                for name in sorted(names):
                    f = Path(base) / name
                    if f.suffix.lower() not in WEIGHTS or f.is_symlink():
                        continue
                    resolved = safe(f, folder)
                    if resolved in opened:
                        raise RuntimeError('A process still has the candidate open: ' + str(f))
                    st = f.stat()
                    if not stat.S_ISREG(st.st_mode):
                        raise RuntimeError('Not a regular checkpoint file')
                    rows.append(dict(path=str(resolved), allowed_root=str(folder.resolve()),
                        bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino,
                        evidence=evidence, metrics=metrics))
        report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), candidates=rows,
            bytes=sum(r['bytes'] for r in rows), preserved_current_run='lalt_20261018_053126Z',
            preserve_policy='all reports/images/code/cache; positive/representative checkpoints; every active LALT artifact')
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open('x') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(dict(plan=str(args.plan), files=len(rows), logical_gib=report['bytes'] / 1024**3)))
        return
    plan = json.loads(args.plan.read_text())
    log = args.plan.with_suffix('.deleted.jsonl')
    if log.exists():
        raise RuntimeError('Refusing to replay a partially applied deletion log')
    # Revalidate report eligibility immediately before touching files.
    allowed_now = {str(folder.resolve()) for folder, _, _ in targets()}
    opened = open_files()
    for row in plan['candidates']:
        f, allowed = Path(row['path']), Path(row['allowed_root'])
        if str(allowed) not in allowed_now:
            raise RuntimeError('Folder is not in the reviewed allowlist')
        resolved = safe(f, allowed)
        st = resolved.stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino) != (row['bytes'], row['mtime_ns'], row['inode']):
            raise RuntimeError('Checkpoint changed since planning')
        if resolved in opened:
            raise RuntimeError('Checkpoint is open in a live process')
    before = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    count, total = 0, 0
    with log.open('x') as stream:
        for row in plan['candidates']:
            f = safe(Path(row['path']), Path(row['allowed_root']))
            st = f.stat()
            if (st.st_size, st.st_mtime_ns, st.st_ino) != (row['bytes'], row['mtime_ns'], row['inode']):
                raise RuntimeError('Checkpoint changed before its individual deletion')
            f.unlink()  # Exactly one reviewed regular weight file; no recursive delete.
            count += 1; total += row['bytes']
            stream.write(json.dumps(dict(deleted_at_utc=datetime.now(timezone.utc).isoformat(), **row)) + '\n')
            stream.flush(); os.fsync(stream.fileno())
            print(f'Deleted {count}: {row["bytes"] / 1024**3:.3f} GiB {f}', flush=True)
    after = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    summary = dict(deleted_files=count, logical_bytes=total, available_before=before,
                   available_after=after, observed_free_change=after-before, deletion_log=str(log))
    args.plan.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
