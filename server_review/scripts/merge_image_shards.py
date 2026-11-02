#!/usr/bin/env python3
import argparse,os
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--inputs',nargs='+',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--prefix',required=True); a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
i=0
for d in a.inputs:
 for src in sorted(d.glob('*.png')):
  dst=a.out/f'{a.prefix}_{i:06d}.png'
  if dst.exists(): dst.unlink()
  os.link(src,dst); i+=1
print('merged',i,'images into',a.out)