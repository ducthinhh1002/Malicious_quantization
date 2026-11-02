from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
HARNESS=Path('/var/lib/vllm/harness')
sys.path.insert(0,str(HARNESS))
from common import read_json,freeze,write_json,sha256
