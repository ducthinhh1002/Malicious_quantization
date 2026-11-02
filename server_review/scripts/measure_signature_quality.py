"""Post-hoc quality audit of saved PNGs; never loads a diffusion model."""
import json, os, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import lpips
from skimage.metrics import structural_similarity
from transformers import CLIPModel, CLIPProcessor

RUN = Path('/root/logs/results/signature_author_sd21_to_sd14_20261021_074620Z')
OUT = RUN / 'quality_audit'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
device = 'cuda'
torch.cuda.set_per_process_memory_fraction(0.07)
read = lambda p: json.loads(p.read_text())
def save(name, obj):
    p = OUT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)
def load(record, i):
    entry = record['entries'][i]
    path = Path(record['folder']) / entry['file_name']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError('Image hash mismatch: ' + str(path))
    return Image.open(path).convert('RGB')
records = {name: read(RUN / ('teacher_' + name[8:] + '_query.json'))
           for name in ('teacher_clean', 'teacher_marked')}
for branch in sorted(RUN.glob('r*_*/query.json')):
    records[branch.parent.name] = read(branch)
clip_path = next(Path('/root/logs/runtime/cache/openai--clip-vit-base-patch32').glob('*'))
clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device).eval()
processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
report = {'protocol': {'n': 1000, 'clip': 'openai/clip-vit-base-patch32',
    'clip_score': '100 * max(cosine(image,text),0)',
    'pair_metrics': 'RGB PNG at original resolution; LPIPS AlexNet [-1,1]; SSIM gaussian sigma1.5 population covariance; PSNR data range1',
    'limits': 'Pair distances measure output changes, not absolute quality. CNN versus attention clean is unmatched scope. No matched rank16 clean controls. CLIP measures caption alignment, not aesthetics.'}, 'clip': {}, 'pairs': {}}
with torch.inference_mode():
    for name, rec in records.items():
        values=[]
        for start in range(0,len(rec['entries']),16):
            indices=range(start,min(start+16,len(rec['entries'])))
            images=[load(rec,i) for i in indices]
            captions=[rec['entries'][i]['text'] for i in indices]
            inputs=processor(text=captions,images=images,return_tensors='pt',padding=True,truncation=True).to(device)
            out=clip(**inputs)
            values.extend((100*(out.image_embeds*out.text_embeds).sum(-1).clamp_min(0)).cpu().tolist())
        report['clip'][name]={'n':len(values),'mean':float(np.mean(values)), 'values':values}
        save('quality.json',report)
        print(name,'CLIP',np.mean(values),flush=True)
del clip
torch.cuda.empty_cache()
metric=lpips.LPIPS(net='alex').to(device).eval()
pairs=[('teacher_marked','teacher_clean'),('r4_attention_marked','r4_attention_clean'),
       ('r4_attention_conv_marked','r4_attention_clean')]
with torch.inference_mode():
    for left,right in pairs:
        if left not in records or right not in records: continue
        a,b=records[left],records[right]
        if a['source_ids'] != b['source_ids']: raise RuntimeError('Source ID mismatch')
        rows=[]
        for i in range(len(a['entries'])):
            if a['entries'][i]['text']!=b['entries'][i]['text']: raise RuntimeError('Caption mismatch')
            x=np.asarray(load(a,i),dtype=np.float32)/255
            y=np.asarray(load(b,i),dtype=np.float32)/255
            tx=torch.from_numpy(x.copy()).permute(2,0,1)[None].to(device)*2-1
            ty=torch.from_numpy(y.copy()).permute(2,0,1)[None].to(device)*2-1
            rows.append({'psnr':float(-10*np.log10(max(float(np.mean((x-y)**2)),1e-12))),
                'ssim':float(structural_similarity(x,y,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),
                'lpips':float(metric(tx,ty).item())})
        result={'n':len(rows),'matched_scope':left!='r4_attention_conv_marked',
            **{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},'values':rows}
        report['pairs'][left+'__'+right]=result
        save('quality.json',report)
        print(left,right,{k:v for k,v in result.items() if k!='values'},flush=True)
names=[n for n in ('teacher_clean','teacher_marked','r4_attention_clean','r4_attention_marked','r4_attention_conv_marked') if n in records]
canvas=Image.new('RGB',(256*len(names),290*8),'white'); draw=ImageDraw.Draw(canvas)
for row,i in enumerate([0,1,2,4,6,10,20,40]):
    for col,name in enumerate(names):
        canvas.paste(load(records[name],i).resize((256,256)),(col*256,row*290+30))
        draw.text((col*256+3,row*290+2),name,fill='black')
        draw.text((col*256+3,row*290+15),'query '+str(i),fill='black')
canvas.save(OUT/'comparison.jpg')
save('complete.json',{'status':'complete','branches':list(records)})
