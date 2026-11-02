import hashlib,itertools,json
from pathlib import Path
import numpy as np
from transformers import CLIPTokenizer
R=Path('/root/workspace/serum');out={}
p={x.stem:x.read_text().splitlines() for x in (R/'prompts').glob('*.txt')}
tok=CLIPTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='tokenizer',local_files_only=True)
reps={'exact':p,'normalized':{k:[' '.join(x.casefold().split()) for x in v] for k,v in p.items()},'sd14_tokens':{k:[tuple(x) for x in tok(v,padding='max_length',max_length=tok.model_max_length,truncation=True)['input_ids']] for k,v in p.items()}}
exclusions=set()
for rep,sets in reps.items():
    results=[]
    for a,b in itertools.combinations(sets,2):
        lookup={}
        for j,x in enumerate(sets[b]):lookup.setdefault(x,[]).append(j)
        for i,x in enumerate(sets[a]):
            if x in lookup:
                for j in lookup[x]:
                    results.append({'left':a,'left_index':i,'right':b,'right_index':j})
                    if a=='student_eval':exclusions.add(i)
                    if b=='student_eval':exclusions.add(j)
    out[rep]=results
out['excluded_eval_indices']=sorted(exclusions)
mask=np.ones(5000,dtype=bool);mask[list(exclusions)]=False
threshold=json.loads((R/'metrics/serum_author/threshold_1pct.json').read_text())['threshold']
out['sensitivity']={}
for file in sorted((R/'metrics/serum_author').glob('*seed0_author_scores.npy')):
    v=np.load(file)
    if v.shape!=(5000,):continue
    out['sensitivity'][file.stem]={'all_n':len(v),'all_rate':float((v>=threshold).mean()),'disjoint_n':int(mask.sum()),'disjoint_rate':float((v[mask]>=threshold).mean())}
out['threshold_unchanged']=threshold
(R/'metrics/serum_author/c10_c11_prompt_review_20261012.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
n=json.loads((R/'third_party/CtrlRegen/ctrlregen_plus_demo.ipynb').read_text())
print('OFFICIAL_NOTEBOOK_CODE')
for cell in n['cells']:
    if cell['cell_type']=='code':
        text=''.join(cell['source'])
        if any(x in text for x in ['strength=', 'color_match(', 'UniPCMultistepScheduler']):print(text)
