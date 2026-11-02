from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
BASE=ROOT
SHARED=Path('/var/lib/structure8bit_rank4_refine600')
HARNESS=SHARED/'harness'
sys.path.insert(0,str(HARNESS))
from common import read_json,write_json,freeze,sha256
