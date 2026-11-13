"""Frozen-reader clean/JPEG25 comparison, no checkpoint/threshold selection."""
import bootstrap
from bootstrap import ROOT, read, write, sha, now
import io
import numpy as np
from PIL import Image
import torch
from semantic_teacher import extractor
from semantic_carrier import SemanticCarrier
run=ROOT/'run'
out=run/'evaluation_jpeg25';out.mkdir(exist_ok=True)
rows=read(run/'prompts.json')['test']
folder=run/'images/teacher_jpeg600_test'
entries=read(folder/'image_hashes.json')
assert len(entries)==len(rows)==1000
ext=extractor(run);carrier=SemanticCarrier.load(run/'carrier','cuda')
signs=carrier.signs.cpu().numpy()
results={}
for condition in ['original','jpeg25']:
 target=out/(condition+'_coordinates.json')
 if target.exists():coords=np.asarray(read(target),dtype=np.float32)
 else:
  values=[]
  for start in range(0,1000,4):
   images=[]
   for entry in entries[start:start+4]:
    path=folder/entry['file_name'];assert sha(path)==entry['sha256']
    with Image.open(path) as im:
     im=im.convert('RGB')
     if condition=='jpeg25':
      buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=25,subsampling=2,optimize=False,progressive=False);buffer.seek(0)
      with Image.open(buffer) as decoded:array=np.array(decoded.convert('RGB'),copy=True)
     else:array=np.array(im,copy=True)
    images.append(torch.from_numpy(array).permute(2,0,1).float()/127.5-1)
   with torch.inference_mode():q=carrier.coordinates(ext.encode_image(torch.stack(images).cuda(),[r['text'] for r in rows[start:start+4]]))
   assert torch.isfinite(q).all()
   values.extend(q.cpu().tolist())
   write(out/'status.json',dict(state='running',condition=condition,n=len(values),updated=now()))
   print(condition,len(values),flush=True)
  write(target,values);coords=np.asarray(values,dtype=np.float32)
 metrics=[]
 for size in [1,32,256]:
  if size==1:grouped=coords
  else:
   rng=np.random.default_rng(0)
   grouped=np.concatenate([coords[rng.permutation(1000)[:(1000//size)*size]].reshape(-1,size,32).mean(1) for _ in range(100)])
  matched=((grouped>0)==(signs>0)).sum(1)
  metrics.append(dict(images_per_group=size,n_groups=len(matched),bit_accuracy=float(matched.mean()/32),key_acceptance_ge24_rate=float((matched>=24).mean()),exact_key_rate=float((matched==32).mean())))
 results[condition]=metrics
 write(out/'results.json',dict(checkpoint=str(run/'teacher_fp/refined/checkpoint-1200'),jpeg_quality=25,subsampling=2,reader_and_key='JPEG-stable PCA subset; original32 signs; newly calibrated carrier',aggregation='mean coordinates then signs;100 partitions for groups>1, overlapping across partitions',conditions=results))
 print('RESULT',condition,metrics,flush=True)
write(out/'status.json',dict(state='completed',updated=now()))
