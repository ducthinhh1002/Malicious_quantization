from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
BASE=ROOT
HARNESS=ROOT/'harness'
sys.path.insert(0,str(HARNESS))
from common import read_json,write_json,freeze,sha256
