"""Use the historical harness read-only; experiment code/results live here."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_base = os.environ.get('PN90_BASE_ROOT', str(HERE.parent / 'threshold_quality_20261022'))
if _base == '/root/logs' or _base.startswith('/root/logs/'):
    _base = str(HERE.parent) + _base[len('/root/logs'):]
ROOT = Path(_base).resolve()
if not ROOT.is_dir():
    ROOT = HERE.parent / 'project_native'
HARNESS = ROOT / 'src'
if not HARNESS.is_dir():
    HARNESS = ROOT / 'Malicious_quantization' / 'bitmark_lora'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HARNESS))
# Keep new modules ahead of historical modules, without replacing their files.
sys.path.insert(0, str(HERE))
from common import read_json, write_json, freeze, sha256


def approved_relocation(path, expected, actual):
    """Accept only an explicitly audited old->new code hash, never data changes."""
    import json
    from common import relocate_deployment_paths
    receipt = HERE / 'relocation_allowlist.json'
    if not receipt.exists():
        return False
    entry = json.loads(receipt.read_text())['files'].get(str(Path(relocate_deployment_paths(str(path))).resolve()))
    return bool(entry and entry['original_sha256'] == expected and entry['relocated_sha256'] == actual)
