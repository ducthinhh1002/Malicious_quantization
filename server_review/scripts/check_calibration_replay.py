import os
from pathlib import Path
import numpy as np
from serum_scores import score
ROOT=Path('/root/workspace/serum')
subset=ROOT/'data/calibration_replay64'
subset.mkdir(exist_ok=True)
for src in sorted((ROOT/'data/raw/calib_clean').glob('*.png'))[:64]:
    dst=subset/src.name
    if not dst.exists(): os.link(src,dst)
s=score(subset,ROOT/'metrics/calibration_replay64.npy')
original=np.load(ROOT/'metrics/serum_calib_clean_scores_v2.npy')[:64]
assert np.array_equal(s,original), f'Max scoring drift: {np.max(np.abs(s-original))}'
print('Seeded scoring replay64: exact match',flush=True)
