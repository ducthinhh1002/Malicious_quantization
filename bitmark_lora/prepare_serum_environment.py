"""Prepare the selected SERUM interpreter without installing FlashAttention.

No model downloads or CUDA allocations. Installs into sys.executable's existing
environment only; use SERUM_PYTHON to select a dedicated environment if needed.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REQUIREMENTS = HERE / 'requirements.txt'


def pinned_versions(path=REQUIREMENTS):
    pins = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line or line.startswith('--'):
            continue
        name, separator, version = line.partition('==')
        if not separator or not name or not version:
            raise ValueError(f'Expected an exact shared dependency pin: {raw}')
        pins[name] = version
    return pins


def mismatches(pins):
    differences = []
    for name, expected in pins.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = 'NOT INSTALLED'
        if actual != expected:
            differences.append(f'{name}: {actual} -> {expected}')
    return differences


def audit_imports():
    # A separate process frees all imported ML libraries before the real run.
    code = """
import runpy
import common
from serum_teacher import check_imports
check_imports()
import datasets, pyarrow, numpy, scipy, torch, torchvision, pandas
from torch_fidelity import calculate_metrics
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import HfApi, hf_hub_download
for trainer in ('train_text_to_image_lora.py', 'train_text_to_image.py'):
    runpy.run_path(str(common.HERE/'vendor/diffusers'/trainer), run_name='dependency_import_audit')
torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), .5)
print('SERUM dependency imports OK; no models downloaded or GPU memory allocated.', flush=True)
"""
    subprocess.run([sys.executable, '-c', code], cwd=HERE, check=True)


def prepare(check_only=False):
    pins = pinned_versions()
    changes = mismatches(pins)
    if changes:
        print('SERUM dependency differences:\n  ' + '\n  '.join(changes), flush=True)
        if check_only:
            raise RuntimeError('Dependencies missing or mismatched. Run bash run_serum.sh with '
                               'SERUM_AUTO_INSTALL=1, or install bitmark_lora/requirements.txt '
                               'using the selected interpreter.')
        if subprocess.run([sys.executable, '-m', 'pip', '--version'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'], check=True)
        print(f'Installing pinned SERUM dependencies into {sys.executable}', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-user',
                        '--disable-pip-version-check', '-r', str(REQUIREMENTS)], check=True)
        importlib.invalidate_caches()
        # Pip ran in another process; inspect metadata afresh before starting.
        remaining = mismatches(pins)
        if remaining:
            raise RuntimeError('Dependency versions still differ after install: ' + '; '.join(remaining))
    else:
        print('SERUM pinned dependency versions already match; skipping pip.', flush=True)
    audit_imports()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('The pinned CUDA environment targets Linux x86_64.')
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError('SERUM pinned dependencies require Python 3.10, 3.11, or 3.12. '
                           'Select a compatible interpreter using SERUM_PYTHON.')
    # Serialize setup for launchers sharing an interpreter, including two GPUs.
    import fcntl
    lock_id = hashlib.sha256(str(Path(sys.prefix).resolve()).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f'serum-setup-{os.getuid()}-{lock_id}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(args.check_only)


if __name__ == '__main__':
    main()
