import json
from pathlib import Path
r=Path('/root/workspace/serum')
for s in ('raw','0.25','0.50','0.75','1.00'):
    row={'strength':s}
    for k in ('clean','wm'):
        directory=r/'data/pilot'/(k if s=='raw' else f'{k}_{s}')
        entry={'images':len(list(directory.glob('*.png')))}
        detector=r/f'metrics/pilot/{k}_{s}_scores.npy.result.json'
        if detector.exists(): entry['detector']=json.loads(detector.read_text())
        preservation=r/f'metrics/pilot/{k}_{s}.json'
        if preservation.exists(): entry['preservation']=json.loads(preservation.read_text())['means']
        row[k]=entry
    print(json.dumps(row),flush=True)
