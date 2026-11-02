"""Portable deployment of the original 12500-step student wrapper."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
HARNESS = BASE / 'harness'
sys.path[:0] = [str(HARNESS), str(BASE)]
from common import read_json, write_json, freeze, sha256
