import json
from pathlib import Path
root=Path('/root/workspace/serum')
assert json.loads((root/'manifests/teacher_integrity_v2.json').read_text())['ok']
assert json.loads((root/'metrics/serum_threshold_1pct.json').read_text())['model']==json.loads((root/'manifests/serum_frozen.json').read_text())
print('Data audit and frozen calibration ready')
