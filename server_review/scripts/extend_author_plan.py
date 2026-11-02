import json
from pathlib import Path
p=Path('/root/workspace/serum/manifests/serum_author_plan.json')
plan=json.loads(p.read_text())
plan['conditions']=['C00_seed0_author','C10_seed0_author','C01_step1.00_seed0_author','C11_step1.00_seed0_author']
plan['raw_controls_only']=False
plan['execution_roles']={'GPU0':'quant-author-queue: resumed C00 then fresh C10',
    'GPU1':'quant-author-strong: reused-clean-data C01 then author-WM-data C11; waits for teacher calibration'}
plan['next_gate']='User authorized strong conditions without waiting for baseline review. Review baseline validity when interpreting results; extra seeds/defenses remain conditional.'
plan['strong_plan']='/root/workspace/serum/manifests/serum_author_strong_plan.json'
tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(plan,indent=2));tmp.replace(p)
print('Active plan expanded to all four strong-core conditions.')
