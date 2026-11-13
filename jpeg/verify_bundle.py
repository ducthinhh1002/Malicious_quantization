"""CPU-only bundle verification, no model download or inference."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parent
expected='2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8'
assert hashlib.sha256((ROOT/'inputs/refine600/adapter_model.safetensors').read_bytes()).hexdigest()==expected
vendor=ROOT/'harness/vendor'
for name,digest in json.loads((vendor/'provenance.json').read_text())['files'].items():
    assert hashlib.sha256((vendor/name).read_bytes()).hexdigest()==digest,name
for name,digest in json.loads((ROOT/'inputs/original/carrier/complete.json').read_text()).items():
    assert hashlib.sha256((ROOT/'inputs/original/carrier'/name).read_bytes()).hexdigest()==digest,name
for folder in [ROOT,ROOT.parent/'resize']:
    for p in folder.rglob('*.py'):
        if any(x in p.parts for x in ['.venv','cache','source','run','model','deps','__pycache__']):continue
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
rows=json.loads((ROOT.parent/'resize/inputs/prompts.json').read_text())
assert all(len(rows[s])==n for s,n in [('train',5000),('validation',1000),('test',1000)])
original=json.loads((ROOT/'inputs/original/prompts.json').read_text())
for split in ['train','validation','test']:
    assert not {r['text'] for r in original['train']}.intersection(r['text'] for r in rows[split])
print('Verified warm adapter, original carrier, vendored hashes, Python syntax and disjoint student prompts.')
