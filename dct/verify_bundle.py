"""CPU-only artifact/source validation for an exported DCT bundle."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text())
for name,proof in manifest.items():
    path=ROOT/name
    assert path.is_file(),str(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256'],name
prompt=ROOT/'benchmark/shared/data/prompts.json'
assert hashlib.sha256(prompt.read_bytes()).hexdigest()=='5cf63b020893205a1865fa9f238bc81d63d1cd50bdb3d0a0958b007f4cdd84c2'
rows=json.loads(prompt.read_text())
assert all(len(rows[s])==n for s,n in [('train',5000),('validation',1000),('test',1000)])
for left,right in [('train','validation'),('train','test'),('validation','test')]:
    assert not {r['image_id'] for r in rows[left]}.intersection(r['image_id'] for r in rows[right])
    assert not {r['text'] for r in rows[left]}.intersection(r['text'] for r in rows[right])
for path in ROOT.rglob('*.py'):
    if any(x in path.parts for x in ['.venv','.cache','run','__pycache__']):continue
    ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
print('DCT bundle hashes, Python syntax and prompt split isolation verified.')
