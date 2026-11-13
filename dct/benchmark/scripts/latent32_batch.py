"""Per-image-equivalent guided generation with partitioned multi-GPU workers."""
import gc,json,math,os,sys,time,io,fcntl
from pathlib import Path
import numpy as np
import torch
from PIL import Image
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P),str(P/'benchmark/methods/GROW/official/GROW')]
from benchmark.shared.process_title import configure
configure()
from diffusers import StableDiffusionPipeline,DDIMScheduler
from grow import GrowConfig,GrowWatermarker
from grow.utils import tensor_to_image
import grow.watermark as implementation
from common import save_images
W=P/'run'
cache={}
def dct(x):
 n=x.shape[-1];key=(n,str(x.device),x.dtype)
 if key not in cache:
  k=torch.arange(n,device=x.device,dtype=torch.float64)[:,None];j=torch.arange(n,device=x.device,dtype=torch.float64)[None,:]
  c=torch.cos(math.pi/n*(j+.5)*k)*math.sqrt(2/n);c[0]/=math.sqrt(2);cache[key]=c.to(x.dtype)
 c=cache[key];return c@x@c.T
implementation.dct2d_torch=dct
def generate(wm,rows):
 pipe=wm.pipe;cfg=wm.cfg;n=len(rows)
 tokens=pipe.tokenizer(['']*n+[r['text'] for r in rows],padding='max_length',max_length=77,truncation=True,return_tensors='pt').input_ids.cuda()
 with torch.no_grad():hidden=pipe.text_encoder(tokens)[0]
 z=torch.cat([torch.randn((1,4,64,64),device='cuda',dtype=torch.float32,generator=torch.Generator('cuda').manual_seed(r['image_id'])) for r in rows])
 z=z*pipe.scheduler.init_noise_sigma
 pipe.scheduler.set_timesteps(50)
 target,mask=wm.build_target_signal(64,64)
 for step,t in enumerate(pipe.scheduler.timesteps):
  x=pipe.scheduler.scale_model_input(torch.cat([z,z]),t)
  with torch.no_grad():u,c=pipe.unet(x,t,encoder_hidden_states=hidden).sample.chunk(2)
  if step>=25:
   a=pipe.scheduler.alphas_cumprod[t];b=1-a
   clean=((x[n:]-b.sqrt()*c)/a.sqrt()).detach().requires_grad_(True)
   # Sum batch losses to preserve the original per-image gradient magnitude.
   loss=(dct(clean)-target).square()[:,mask[0]].sum()/mask.sum()
   g=torch.autograd.grad(loss,clean)[0]
   with torch.no_grad():c=(x[n:]-a.sqrt()*(clean-200*g))/b.sqrt()
  with torch.no_grad():z=pipe.scheduler.step(u+7.5*(c-u),t,z).prev_sample
 with torch.no_grad():rgb=pipe.vae.decode(z/pipe.vae.config.scaling_factor).sample/2+.5
 return [tensor_to_image(x) for x in rgb]
def main():
 worker=int(sys.argv[1]);batch=int(sys.argv[2]);assets=json.loads((P/'assets.json').read_text());prompts=json.loads((W/'prompts.json').read_text())
 pipe=StableDiffusionPipeline.from_pretrained(assets['teacher_path'],torch_dtype=torch.float32,safety_checker=None,local_files_only=True).to('cuda');pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config);pipe.set_progress_bar_config(disable=True)
 cfg=GrowConfig(message='NW32',secret_key='latent32-dct-pilot-20261111',dtype='float32',watermark_strength=.5,guidance_scale=200.,watermark_start_ratio=.5)
 wm=GrowWatermarker(cfg,pipe=pipe)
 if os.environ.get('BATCH_CHECK')=='1':
  rows=prompts['test'][:2];start=time.monotonic();a=generate(wm,rows);elapsed=time.monotonic()-start
  refs=[wm.generate_with_watermark(r['text'],r['image_id']) for r in rows]
  report={'batch_seconds':elapsed,'max_pixel_differences':[int(np.abs(np.asarray(x).astype(int)-np.asarray(y).astype(int)).max()) for x,y in zip(a,refs)],'batch_bits':[wm._read_bits(x) for x in a],'reference_bits':[wm._read_bits(x) for x in refs]}
  def jpeg_bits(image):
   buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=25,subsampling=2);buffer.seek(0)
   with Image.open(buffer) as decoded:return wm._read_bits(decoded.convert('RGB'))
  report['batch_jpeg25_bits']=[jpeg_bits(x) for x in a]
  report['reference_jpeg25_bits']=[jpeg_bits(x) for x in refs]
  assert report['batch_bits']==report['reference_bits']
  (W/'batch_check.json').write_text(json.dumps(report,indent=2));print('CHECK_PASSED',flush=True);return
 for split,rows in prompts.items():
  if split not in ['train','test']:continue
  folder=W/'latent32/images'/('teacher_wm_'+split);folder.mkdir(parents=True,exist_ok=True)
  completed=0
  while True:
   with (W/'generation_queue.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    queue=json.loads((W/'generation_queue.json').read_text())
    chosen=queue[split][:batch];queue[split]=queue[split][batch:]
    tmp=W/'generation_queue.tmp';tmp.write_text(json.dumps(queue));tmp.replace(W/'generation_queue.json')
   if not chosen:break
   began=time.monotonic();images=generate(wm,[rows[i] for i in chosen]);save_images(images,folder,chosen);completed+=len(chosen)
   status=dict(stage='teacher_generation',split=split,worker=worker,batch=len(chosen),seconds=time.monotonic()-began,completed_by_worker=completed,pid=os.getpid())
   path=W/f'worker_gpu{worker}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(status));tmp.replace(path);print(json.dumps(status),flush=True)
if __name__=='__main__':main()
