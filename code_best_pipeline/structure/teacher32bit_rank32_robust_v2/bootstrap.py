import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARNESS = Path(os.environ.get('WM_HARNESS', '/var/lib/vllm/harness'))
sys.path.insert(0, str(HARNESS))

