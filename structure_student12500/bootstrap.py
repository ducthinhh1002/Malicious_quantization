"""Load bundled historical code; configure experiment artifacts separately."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
HISTORICAL = ROOT/'historical'
BASE = Path(os.environ.get('STRUCTURE12500_DATA_ROOT', str(ROOT/'data/threshold_quality_20261022')))
HARNESS = HISTORICAL/'Malicious_quantization/bitmark_lora'
sys.path[:0] = [str(HARNESS), str(HISTORICAL)]
from common import read_json, write_json, freeze, sha256

