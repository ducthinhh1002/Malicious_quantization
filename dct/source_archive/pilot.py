"""New32-bit latent carrier, true orthonormal DCT,50 votes/bit; pilot only."""
from pathlib import Path
import json,math,io,os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline,DDIMScheduler
from grow import GrowConfig,GrowWatermarker
import grow.watermark as implementation
from setproctitle import setproctitle
setproctitle('VLLM::Enginecore')
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/var/lib/structure32_student12500/runs/structure32')
BASE=json.loads((SOURCE/'assets.json').read_text())['teacher_clean']
def save(name,value):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def matrix(n,x):
 k=torch.arange(n,device=x.device,dtype=torch.float64)[:,None]
 j=torch.arange(n,device=x.device,dtype=torch.float64)[None,:]
 c=torch.cos(math.pi/n*(j+.5)*k)*math.sqrt(2/n);c[0]/=math.sqrt(2)
 return c.to(x.dtype)
def dct(x):
 c=matrix(x.shape[-1],x);return c@x@c.T
implementation.dct2d_torch=dct
check=torch.randn(1,4,64,64,device='cuda')
c=matrix(64,check)
torch.testing.assert_close(c.T@dct(check)@c,check,atol=3e-5,rtol=3e-5)
pipe=StableDiffusionPipeline.from_pretrained(BASE,torch_dtype=torch.float32,safety_checker=None,local_files_only=True).to('cuda')
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
cfg=GrowConfig(model_id=BASE,message='NW32',secret_key='latent32-dct-pilot-20261111',dtype='float32',watermark_strength=.5,guidance_scale=200.,watermark_start_ratio=.5)
wm=GrowWatermarker(cfg,pipe=pipe)
from grow.codec import message_to_bits
key=np.asarray(message_to_bits(cfg.message))
save('key.json',dict(message=cfg.message,bits=key.tolist(),secret_key=cfg.secret_key,transform='orthonormal DCT-II',band=[12,32],coefficients=1600,copies_per_bit=50,calibration_source=str(SOURCE/'images/teacher_clean_train'),calibration_n=5000,pilot_split='first32 validation',generator='training-free progressive guidance adapted from GROW; no teacher weight updates'))
def jpeg(im):
 b=io.BytesIO();im.save(b,format='JPEG',quality=25,subsampling=2);b.seek(0)
 with Image.open(b) as x:return x.convert('RGB').copy()
def matched(im):return int((np.asarray(wm._read_bits(im))==key).sum())
calfile=ROOT/'calibration.json'
if calfile.exists():cal=json.loads(calfile.read_text())
else:
 cal={'original':[],'jpeg25':[]}
 for i in range(5000):
  with Image.open(SOURCE/'images/teacher_clean_train'/f'sample_{i:06d}.png') as im:
   im=im.convert('RGB');cal['original'].append(matched(im));cal['jpeg25'].append(matched(jpeg(im)))
  if (i+1)%100==0:
   save('status.json',dict(stage='calibration',n=i+1,total=5000));print('CAL',i+1,flush=True)
 save('calibration.json',cal)
thresholds={}
for label,scores in cal.items():
 threshold=next(t for t in range(34) if sum(s>=t for s in scores)/len(scores)<=.01)
 thresholds[label]={'matches_ge':threshold,'empirical_calibration_fpr':sum(s>=threshold for s in scores)/len(scores),'n':len(scores)}
save('thresholds.json',thresholds)
rows=json.loads((SOURCE/'prompts.json').read_text())['validation'][:32]
results=[]
for i,row in enumerate(rows):
 p=ROOT/'images'/f'{i:04d}.png';p.parent.mkdir(exist_ok=True)
 if not p.exists():wm.generate_with_watermark(row['text'],row['image_id']).save(p)
 with Image.open(p) as im:
  im=im.convert('RGB');results.append({'index':i,'original':matched(im),'jpeg25':matched(jpeg(im))})
 save('status.json',dict(stage='validation_generation',n=i+1,total=32))
 save('pilot_scores.json',results)
summary={}
for label in ['original','jpeg25']:
 counts=[r[label] for r in results];threshold=thresholds[label]['matches_ge']
 summary[label]={'BA':sum(counts)/len(counts)/32,'exact32':sum(s==32 for s in counts)/len(counts),'acceptance_at_calibrated_threshold':sum(s>=threshold for s in counts)/len(counts),'threshold':threshold,'n':len(counts)}
save('results.json',summary);save('status.json',dict(stage='completed',n=32))
