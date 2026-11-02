"""Read-only validation of HF training pairing and cross-version text embeddings."""
import json
import hashlib
from pathlib import Path
import numpy as np

root=Path('/root/workspace/serum')
prompts=(root/'prompts/adapt_train.txt').read_text().splitlines()
record={'pairing':{}}
for condition,kind in [('C00','clean'),('C10','wm')]:
    rows=[json.loads(line) for line in (root/f'data/hf/{condition}_train/metadata.jsonl').read_text().splitlines()]
    images=sorted((root/f'data/raw/{kind}_train').glob('*.png'))
    assert len(rows)==len(images)==len(prompts)==5000
    mismatches=[]
    for i,(row,image,prompt) in enumerate(zip(rows,images,prompts)):
        linked=root/f'data/hf/{condition}_train'/row['file_name']
        if row['text'] != prompt or hashlib.sha256(linked.read_bytes()).digest()!=hashlib.sha256(image.read_bytes()).digest():
            mismatches.append(i)
    record['pairing'][condition]={'count':5000,'mismatches':mismatches}
old=np.load(root/'metrics/text_probe_serum.npz')
new=np.load(root/'metrics/text_probe_student.npz')
diff=new['embeddings']-old['embeddings']
record['text_encoder_cross_version']={
    'token_ids_identical': bool(np.array_equal(new['input_ids'],old['input_ids'])),
    'embedding_max_abs_difference':float(np.max(np.abs(diff))),
    'embedding_relative_l2_difference':float(np.linalg.norm(diff)/np.linalg.norm(old['embeddings'])),
    'serum_environment':json.loads((root/'metrics/text_probe_serum.npz.json').read_text()),
    'student_environment':json.loads((root/'metrics/text_probe_student.npz.json').read_text()),
}
(root/'metrics/radioactivity_setup_audit.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
