import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/var/lib/vllm')
HARNESS=ROOT/'harness'
sys.path[:0]=[str(ROOT/'core'),str(HARNESS)]
from common import read_json,write_json,freeze,sha256
SOURCE=ROOT/'runs/k1_32bit'
RUN=ROOT/'runs/structure32'
STATE=ROOT/'deployment'

def share(src,dst):
    src,dst=Path(src),Path(dst)
    if src.is_dir():
        for p in src.rglob('*'):
            if p.is_file():share(p,dst/p.relative_to(src))
    else:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha256(src)!=sha256(dst):raise RuntimeError('Reuse hash mismatch: '+str(dst))
        else:os.link(src,dst)
