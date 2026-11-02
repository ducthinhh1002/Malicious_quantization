from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
HARNESS=Path('/var/lib/structure32_student12500/harness')
sys.path[:0]=[str(HARNESS),str(BASE)]
from common import read_json,write_json,freeze,sha256
