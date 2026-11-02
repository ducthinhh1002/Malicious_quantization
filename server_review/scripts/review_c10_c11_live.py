import csv, hashlib, itertools, json, os, time
from pathlib import Path
R=Path('/root/workspace/serum')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads((R/p).read_text())
out={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
paths=['scripts/train_full_sd14.py','scripts/queue_full_author.py','scripts/select_full_checkpoint.py','scripts/select_lora_checkpoint.py','scripts/launder_ctrlregen.py','scripts/author_serum_stage0.py','scripts/generate_full_student.py','scripts/generate_student.py','scripts/audit_full_checkpoint.py','scripts/student_utility.py','third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py']
out['source_sha256']={p:sha(R/p) for p in paths}
prompts={p.stem:p.read_text().splitlines() for p in (R/'prompts').glob('*.txt')}
norm=lambda x:' '.join(x.casefold().split())
out['prompts']={k:{'n':len(v),'unique_normalized':len(set(map(norm,v)))} for k,v in prompts.items()}
out['prompt_intersections']={a+'__'+b:len(set(map(norm,prompts[a]))&set(map(norm,prompts[b]))) for a,b in itertools.combinations(prompts,2)}
out['data']={}
for split in ['train','val']:
    source=R/f'data/serum_author/raw/wm_{split}'
    regen=R/f'data/serum_author/laundered/wm_step_1.00_{split}'
    inventory=read(f'metrics/serum_author/wm_strong_{split}_inventory.json')
    rows=list(csv.DictReader((regen/'laundering_manifest.csv').open()))
    assert len(inventory)==len(rows)==len(prompts[f'adapt_{split}'])
    hashes=[]; equal=0
    for i,(inv,row) in enumerate(zip(inventory,rows)):
        src=source/f'{i:05d}.png';dst=regen/src.name
        h1,h2=sha(src),sha(dst); hashes.append(h1)
        assert h1==inv['source_sha256'] and h2==inv['output_sha256']
        assert Path(row['source']).resolve()==src.resolve() and Path(row['output']).resolve()==dst.resolve()
        assert int(row['index'])==i and float(row['step'])==1
        equal+=h1==h2
        if split=='train':
            for kind,orig in [('C10',src),('C11_step1.00',dst)]:
                hf=R/f'data/serum_author/hf/{kind}_train/{i:06d}.png'
                assert os.path.samefile(hf,orig) or sha(hf)==sha(orig)
    out['data'][split]={'n':len(rows),'all_inventory_hashes_match':True,'identical_raw_regen_images':equal,'raw_unique_hashes':len(set(hashes))}
    out['data'][split]['_hashes']=hashes
out['raw_train_val_exact_image_overlap']=len(set(out['data']['train'].pop('_hashes'))&set(out['data']['val'].pop('_hashes')))
for kind in ['C10','C11_step1.00']:
    rows=[json.loads(x) for x in (R/f'data/serum_author/hf/{kind}_train/metadata.jsonl').read_text().splitlines()]
    assert [x['text'] for x in rows]==prompts['adapt_train']
    assert [x['file_name'] for x in rows]==[f'{i:06d}.png' for i in range(5000)]
out['all_10000_training_captions_and_files_match']=True
out['validation']={}
for name in ['C10_full_seed0_author','C11_full_step1.00_seed0_author','C10_seed0_author','C11_step1.00_seed0_author']:
    v=read(f'students/{name}/validation_selection.json')
    first=[x for x in v['results'] if Path(x['checkpoint']).name.startswith('checkpoint-') and int(Path(x['checkpoint']).name.split('-')[-1])<=5000]
    out['validation'][name]={'best':v['best'],'early':first,'best_within_5000':min(first,key=lambda x:x['mse'])}
out['runtime']={}
for c,n in [('C10','C10_full_step3000_seed0_author'),('C11','C11_full_step1.00_step3000_seed0_author')]:
    out['runtime'][c]={'status':read(f'metrics/serum_author/full_step3000_{c}_status.json'),'images':len(list((R/f'student_outputs/{n}').glob('*.png')))}
(R/'metrics/serum_author/c10_c11_code_review_evidence_20261012.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
