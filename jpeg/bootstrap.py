"""Isolated, attack-free teacher refinement; original reader and key are frozen."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'core'), str(ROOT / 'harness')]
os.environ.update(USE_TORCH='1', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import setproctitle
setproctitle.setproctitle('VLLM::Enginecore')
setproctitle.setthreadtitle('VLLM::Enginecore')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze(path, data):
    if Path(path).exists():
        if read(path) != data:
            raise RuntimeError('Frozen input changed: ' + str(path))
    else:
        write(path, data)
