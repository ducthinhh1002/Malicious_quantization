"""Retain useful image sets and deterministic paired examples of failed old runs.

vm4-only allowlist. No recursive removal; only numbered generated image files.
Full reports/manifests stay, and partial sets receive an explicit retention note.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path('/root/logs')
PATTERN = re.compile(r'^sample_\d+\.(png|jpg|jpeg|webp)$', re.I)


def policies():
    result = []
    weak = ('bitgreen_sd21_20261017_101804', 'bitgreen_constrained_20261017_100852',
            'bitgreen_stablegrad_1017_acc4', 'bitgreen_transport_1017')
    for name in weak:
        evidence = ROOT / 'results' / name / 'result/teacher_gate.json'
        report = json.loads(evidence.read_text())
        if report['run_students'] or any(report['arms'][a]['eligible'] or report['arms'][a]['tpr'] > .02
                                        for a in ('plain', 'robust')):
            raise RuntimeError('Not an established failed historical run: ' + name)
        result.append((ROOT / 'runtime/bitgreen_artifacts' / name / 'images', 16, 'all', str(evidence)))
    for name in ('bitgreen_stablegrad_1017_acc1', 'bitgreen_quality_20261017_082439_942427'):
        evidence = ROOT / 'results' / name / 'result/teacher_gate.json'
        report = json.loads(evidence.read_text())
        if report['arms']['plain']['eligible'] or report['run_students']:
            raise RuntimeError('Plain arm is not eligible for pruning')
        result.append((ROOT / 'runtime/bitgreen_artifacts' / name / 'images', 16, 'plain_only', str(evidence)))
    result.append((ROOT / 'runtime/distribution_artifacts/distribution_gate_20261017_001814/images',
                   16, 'all', '/root/logs/results/distribution_gate_20261017_001814/result/teacher_summary.csv'))
    result.append((ROOT / 'runtime/distribution_artifacts/centered_teacher_20261017_060553/images',
                   16, 'old_steps', '/root/logs/results/centered_teacher_20261017_060553/result/teacher_summary.csv'))
    for name in ('results/teacher_gate_improved/proxy', 'results/teacher_gate_improved/rollout',
                 'results/gate_20261016_144609_3785966/without_c3'):
        evidence = ROOT / name / 'result/teacher_selection.json'
        report = json.loads(evidence.read_text())
        if report['valid_design_winner']:
            raise RuntimeError('A selected teacher is valid; retain images')
        result.append((ROOT / name / 'images', 16, 'all', str(evidence)))
    result.append((ROOT / 'output_image/native_transfer_20261017_174732Z_pilot_cb512e6605',
                   8, 'native_redundant', '/root/logs/results/native_transfer_20261017_174732Z_pilot/decision.json'))
    for name in ('distribution_smoke/images',
                 'output_image/native_transfer_20261017_174732Z_smoke_38fe2fce53'):
        result.append((ROOT / name, 2, 'all', 'completed historical technical smoke; keep two examples per set'))
    return result


def safe(path, allowed):
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(ROOT.resolve()) or not resolved.is_relative_to(allowed.resolve()):
        raise RuntimeError('Path escaped the reviewed image root')
    if any('lalt_' in part or part in {'cache', 'prefix_cache', 'gate_env', 'torch_shared'} for part in resolved.parts):
        raise RuntimeError('Protected current LALT/assets/cache path')
    if not PATTERN.fullmatch(path.name):
        raise RuntimeError('Not an explicitly numbered generated image')
    return resolved


def selected_folder(folder, mode):
    if mode == 'plain_only':
        return 'plain' in folder.name
    if mode == 'old_steps':
        return folder.name.endswith(('_250', '_1000'))
    if mode == 'native_redundant':
        return folder.name == 'clean_calibration' or re.fullmatch(r'd\d+_a\d+_(plus|minus)', folder.name)
    return True


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
        candidates, sets = [], []
        opened = open_files()
        for allowed, retain, mode, evidence in policies():
            if not allowed.exists():
                continue
            for base, dirs, names in os.walk(allowed, followlinks=False):
                folder = Path(base)
                dirs[:] = [d for d in dirs if not (folder / d).is_symlink()]
                if not selected_folder(folder, mode):
                    continue
                images = sorted((folder / n for n in names if PATTERN.fullmatch(n)), key=lambda f: f.name)
                count = 32 if mode == 'native_redundant' and folder.name == 'clean_calibration' else retain
                if len(images) <= count:
                    continue
                keep, remove = images[:count], images[count:]
                sets.append(dict(folder=str(folder), allowed_root=str(allowed), original_images=len(images),
                    retained_images=[f.name for f in keep], removed_images=len(remove), evidence=evidence,
                    sampling_rule='first sorted filenames; paired sets retain same indices; no detector/quality selection',
                    complete_dataset_after_cleanup=False, report_scores_refer_to_original_full_dataset=True))
                for f in remove:
                    resolved = safe(f, allowed); s = f.stat()
                    if resolved in opened or not stat.S_ISREG(s.st_mode):
                        raise RuntimeError('Image is open or not a regular file')
                    candidates.append(dict(path=str(resolved), allowed_root=str(allowed),
                        bytes=s.st_size, inode=s.st_ino, mtime_ns=s.st_mtime_ns))
        report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), sets=sets, candidates=candidates,
            bytes=sum(r['bytes'] for r in candidates),
            preserve_policy='all LALT; all reports/figures; strong BitGreen full sets; quality/acc1 robust and matched clean; centered step500 triplet; native fit/gate images')
        with args.plan.open('x') as f:
            json.dump(report, f, indent=2)
        print(json.dumps(dict(plan=str(args.plan), files=len(candidates), sets=len(sets), gib=report['bytes']/1024**3)))
        return
    plan = json.loads(args.plan.read_text())
    log = args.plan.with_suffix('.deleted.jsonl')
    if log.exists():
        raise RuntimeError('Do not replay a deletion journal')
    allowed_now = {str(r[0].resolve()) for r in policies()}
    opened = open_files()
    for row in plan['candidates']:
        allowed = Path(row['allowed_root'])
        if str(allowed.resolve()) not in allowed_now:
            raise RuntimeError('Not in the current reviewed allowlist')
        f = safe(Path(row['path']), allowed); s = f.stat()
        if (s.st_size, s.st_ino, s.st_mtime_ns) != (row['bytes'], row['inode'], row['mtime_ns']) or f in opened:
            raise RuntimeError('Image changed since planning or is open')
    # Preserve a note before deleting so interrupted cleanup cannot masquerade
    # as a complete image dataset. Original manifests remain as historical evidence.
    for info in plan['sets']:
        note = Path(info['folder']) / 'IMAGE_RETENTION.json'
        with note.open('x') as stream:
            json.dump(dict(**info, cleanup_plan=str(args.plan), state='planned_or_in_progress'), stream, indent=2)
    before = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    count = 0
    with log.open('x') as stream:
        for row in plan['candidates']:
            f = safe(Path(row['path']), Path(row['allowed_root'])); s = f.stat()
            if (s.st_size, s.st_ino, s.st_mtime_ns) != (row['bytes'], row['inode'], row['mtime_ns']):
                raise RuntimeError('Image changed before its individual deletion')
            f.unlink()
            count += 1
            stream.write(json.dumps(dict(deleted_at_utc=datetime.now(timezone.utc).isoformat(), **row)) + '\n')
            if count % 100 == 0:
                stream.flush()
            if count % 1000 == 0:
                print(f'Deleted {count}/{len(plan["candidates"])} images', flush=True)
        stream.flush(); os.fsync(stream.fileno())
    for info in plan['sets']:
        folder = Path(info['folder'])
        remaining = sorted(f.name for f in folder.iterdir() if PATTERN.fullmatch(f.name))
        if remaining != info['retained_images']:
            raise RuntimeError('Retained image set verification failed')
        (folder / 'IMAGE_RETENTION.json').write_text(json.dumps(dict(**info, cleanup_plan=str(args.plan), state='complete'), indent=2))
    after = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    summary = dict(deleted_images=count, logical_bytes=plan['bytes'], retained_examples=sum(len(s['retained_images']) for s in plan['sets']),
        pruned_sets=len(plan['sets']), available_before=before, available_after=after, observed_free_change=after-before)
    args.plan.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
