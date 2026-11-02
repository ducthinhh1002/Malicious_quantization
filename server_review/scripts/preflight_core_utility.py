import json,os
from pathlib import Path
import torch
import torch_fidelity
from transformers import CLIPModel,CLIPProcessor
from PIL import Image

torch.set_num_threads(4)
r=Path('/root/workspace/serum')
for kind in ['clean','wm']:
    out=r/f'data/utility_preflight/{kind}'; out.mkdir(parents=True,exist_ok=True)
    for src in sorted((r/f'data/pilot/{kind}').glob('*.png'))[:16]:
        dst=out/src.name
        if not dst.exists(): os.link(src,dst)
metrics=torch_fidelity.calculate_metrics(input1=str(r/'data/utility_preflight/clean'),input2=str(r/'data/utility_preflight/wm'),
    cuda=True,kid=True,fid=False,isc=False,kid_subset_size=8,kid_subsets=2,batch_size=8,
    rng_seed=42000,input1_cache_name='preflight_clean',input2_cache_name='preflight_wm',cache_root=str(r/'data/utility_feature_cache'))
clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval().to('cuda')
cp=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
src=sorted((r/'data/pilot/clean').glob('*.png'))[0]
with Image.open(src) as im: inputs=cp(images=[im.convert('RGB')],return_tensors='pt').to('cuda')
text=cp(text=[(r/'prompts/student_eval.txt').read_text().splitlines()[0]],padding=True,truncation=True,max_length=77,return_tensors='pt').to('cuda')
with torch.no_grad():
    vi=clip.get_image_features(**inputs); vt=clip.get_text_features(**text)
    if not isinstance(vi,torch.Tensor): vi=vi.pooler_output
    if not isinstance(vt,torch.Tensor): vt=vt.pooler_output
    assert vi.shape==vt.shape==(1,512) and torch.isfinite(vi).all() and torch.isfinite(vt).all()
report={'ok':True,'scope':'Dependency and feature extraction smoke check only; not research metrics',
    'torch_fidelity':torch_fidelity.__version__,'kid_extraction_completed':True,'clip_features_shape':list(vi.shape)}
(r/'metrics/core_utility_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
