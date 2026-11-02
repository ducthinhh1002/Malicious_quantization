"""Audit the full reused core dataset and pin a common clipfix protocol."""
import csv
import hashlib
import itertools
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import try_to_load_from_cache
import numpy as np

R=Path('/root/workspace/serum')
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def inspect_image(path):
    with Image.open(path) as im:
        im.load();assert im.mode=='RGB' and im.size==(512,512),str(path)
    return digest(path)
def main():
    import os
    os.chdir(R)
    expected={'adapt_train.txt':5000,'adapt_val.txt':1000,'student_eval.txt':5000,'detector_calib.txt':5000}
    prompts={name:(R/'prompts'/name).read_text().splitlines() for name in expected}
    for name,n in expected.items():
        assert len(prompts[name])==n and len(set(prompts[name]))==n and all(x.strip() for x in prompts[name]),name
    for a,b in itertools.combinations(prompts,2):assert not set(prompts[a])&set(prompts[b]),(a,b)
    c10=json.loads((R/'manifests/C10_seed0_clipfix.json').read_text())
    for name in prompts:assert digest(R/'prompts'/name)==c10['prompt_hashes'][name]
    assert digest(R/'metrics/serum_threshold_1pct.json')==c10['threshold_sha256']
    frozen=json.loads((R/'manifests/serum_frozen.json').read_text())
    for val in frozen.values():assert digest(val['path'])==val['sha256']
    trainer=R/'third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py'
    assert digest(trainer)==c10['trainer_sha256']
    assert 'lora_layers = [p for p in unet.parameters() if p.requires_grad]' in trainer.read_text()
    selection=json.loads((R/'metrics/pilot_selection.json').read_text())
    assert selection['status']=='selected' and selection['criterion']=='image preservation only'
    assert (selection['mild'],selection['strong'])==(.25,1.)
    cache=Path(try_to_load_from_cache('CompVis/stable-diffusion-v1-4','model_index.json'))
    assert cache.parent.name=='133a221b8aa7292a167afc5127cb63fb5005638b'
    records={}; inventories={}
    for kind in ['clean','wm']:
        for strength in [None,'0.25','1.00']:
            base=('C00' if kind=='clean' else 'C10') if strength is None else ('C01' if kind=='clean' else 'C11')+f'_step{strength}'
            condition={}
            for split,n in [('train',5000),('val',1000)]:
                folder=R/(f'data/raw/{kind}_{split}' if strength is None else f'data/laundered/{kind}_step_{strength}_{split}')
                files=sorted(folder.glob('*.png'))
                raw=sorted((R/f'data/raw/{kind}_{split}').glob('*.png'))
                assert len(files)==n and [x.name for x in files]==[x.name for x in raw]
                if strength is not None:
                    with (folder/'laundering_manifest.csv').open() as f:rows=list(csv.DictReader(f))
                    assert len(rows)==n
                    for i,(row,src,dst) in enumerate(zip(rows,raw,files)):
                        assert Path(row['source']).resolve()==src.resolve() and Path(row['output']).resolve()==dst.resolve()
                        assert int(row['seed'])==(10000 if split=='train' else 20000)+i and float(row['step'])==float(strength)
                with ThreadPoolExecutor(max_workers=4) as pool: hashes=list(pool.map(inspect_image,files))
                inventory=[{'file':str(p.relative_to(R)),'sha256':h} for p,h in zip(files,hashes)]
                inventories[f'{base}_{split}']=inventory
                condition[split]={'directory':str(folder),'count':n,'decoded_all':True,'inventory_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()}
                if split=='train':
                    hf=R/f'data/hf/{base}_train'
                    rows=[json.loads(s) for s in (hf/'metadata.jsonl').read_text().splitlines()]
                    assert len(rows)==len(list(hf.glob('*.png')))==5000
                    for i,(row,src,h) in enumerate(zip(rows,files,hashes)):
                        assert row['file_name']==f'{i:06d}.png' and row['text']==prompts['adapt_train.txt'][i]
                        dst=hf/row['file_name'];assert src.samefile(dst) or digest(dst)==h
                    condition['hf_directory']=str(hf)
                    condition['metadata_sha256']=digest(hf/'metadata.jsonl')
                print('AUDITED',base,split,n,flush=True)
            records[base]=condition
    assert shutil.disk_usage(R).free>100*1024**3
    scripts=['select_lora_checkpoint.py','generate_student.py','serum_scores.py','student_utility.py','find_resume_checkpoint.py']
    frozen_files={str(trainer.relative_to(R)):digest(trainer)}
    frozen_files.update({f'scripts/{s}':digest(R/'scripts'/s) for s in scripts})
    frozen_files.update({f'prompts/{p}':digest(R/'prompts'/p) for p in prompts})
    for file in ['metrics/serum_threshold_1pct.json','manifests/serum_frozen.json','metrics/pilot_selection.json']:
        frozen_files[file]=digest(R/file)
    report={'status':'passed','conditions':records,'prompt_counts':expected,'splits_disjoint':True,'frozen_files':frozen_files,'base_model_revision':cache.parent.name,'base_model_index_sha256':digest(cache),'threshold':json.loads((R/'metrics/serum_threshold_1pct.json').read_text())['threshold'],'training':{'rank':4,'learning_rate':1e-5,'batch_size':8,'steps':20000,'seed':0,'mixed_precision':'fp16'},'baseline_gate':'pending corrected C10; user-authorized exploratory full rerun, no automatic pass','visual_audit':'contact sheets generated separately; requires inspection'}
    save(R/'manifests/clipfix_core_inventory.json',inventories)
    save(R/'manifests/clipfix_core_protocol.json',report)
    # 50 deterministic random prompt/source rows, each with both regenerated variants.
    directory=R/'manifests/clipfix_visual_audit';directory.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261011)
    samples=[(kind,int(i)) for kind in ['clean','wm'] for i in rng.choice(5000,25,replace=False)]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    import textwrap
    for page in range(10):
        sheet=Image.new('RGB',(1120,1040),'white');draw=ImageDraw.Draw(sheet)
        draw.text((10,8),f'Pairing review {page+1}/10: prompt | raw | mild 0.25 | strong 1.00',fill='black',font=font)
        for j,(kind,idx) in enumerate(samples[page*5:page*5+5]):
            y=45+j*195
            caption=f'{kind} #{idx}: '+prompts['adapt_train.txt'][idx]
            draw.multiline_text((10,y),'\n'.join(textwrap.wrap(caption,49)[:10]),fill='black',font=font,spacing=2)
            for col,strength in enumerate([None,'0.25','1.00']):
                folder=R/(f'data/raw/{kind}_train' if strength is None else f'data/laundered/{kind}_step_{strength}_train')
                path=sorted(folder.glob('*.png'))[idx]
                with Image.open(path) as im:sheet.paste(im.resize((184,184)),(440+col*222,y))
        sheet.save(directory/f'page_{page+1:02d}.jpg',quality=90)
    save(directory/'samples.json',{'seed':20261011,'rows':samples,'scope':'25 clean + 25 watermarked source/prompt rows, with mild and strong variants; not a quality metric'})
    print('PREFLIGHT PASSED; 36,000 source images decoded and paired',flush=True)
if __name__=='__main__':main()
