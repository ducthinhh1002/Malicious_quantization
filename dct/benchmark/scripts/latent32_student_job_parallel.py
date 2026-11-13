"""Progressive32-bit DCT teacher images -> reused SD1.4 r4 12500-step distillation."""
from pathlib import Path
import gc,io,json,math,os,subprocess,sys
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline,DDIMScheduler
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P),str(P/'benchmark/methods/GROW/official/GROW')]
from benchmark.shared.process_title import configure
configure()
from benchmark.scripts import ood_clean_20261111 as reused
from benchmark.scripts.additional_clean import verify_images
from common import save_images,manifest
from grow import GrowConfig,GrowWatermarker
from grow.codec import message_to_bits
import grow.watermark as implementation
WORK=P/'run'
import fcntl
WORK.mkdir(parents=True,exist_ok=True)
job_lock=(WORK/'parallel_job.lock').open('a')
fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
reused.WORK=WORK
method='latent32';folder=WORK/method
assets=reused.read(P/'assets.json')
prompts=reused.read(P/'benchmark/shared/data/prompts.json')
reused.write(WORK/'prompts.json',prompts)
def dct(x):
 n=x.shape[-1];k=torch.arange(n,device=x.device,dtype=torch.float64)[:,None];j=torch.arange(n,device=x.device,dtype=torch.float64)[None,:]
 c=torch.cos(math.pi/n*(j+.5)*k)*math.sqrt(2/n);c[0]/=math.sqrt(2);c=c.to(x.dtype)
 return c@x@c.T
implementation.dct2d_torch=dct
cfg=GrowConfig(model_id=assets['teacher_path'],message='NW32',secret_key='latent32-dct-pilot-20261111',dtype='float32',watermark_strength=.5,guidance_scale=200.,watermark_start_ratio=.5)
def reader():
 pipe=StableDiffusionPipeline.from_pretrained(assets['teacher_path'],torch_dtype=torch.float32,safety_checker=None,local_files_only=True).to('cuda')
 pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config);pipe.set_progress_bar_config(disable=True)
 return GrowWatermarker(cfg,pipe=pipe)
worker_spec=os.environ.get('DCT_WORKERS',os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]+':2')
worker_settings=[tuple(map(int,item.split(':'))) for item in worker_spec.split(',')]
reused.write(folder/'status.json',dict(stage='teacher_generation_parallel',pid=os.getpid(),workers=dict(worker_settings)))
queue={split:[i for i in range(len(prompts[split])) if not (folder/'images'/('teacher_wm_'+split)/f'sample_{i:05d}.png').exists()] for split in ['train','test']}
reused.write(WORK/'generation_queue.json',queue)
workers=[]
for gpu,batch in worker_settings:
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
 env.pop('BATCH_CHECK',None)
 log=(WORK/f'worker_gpu{gpu}.log').open('a')
 workers.append(subprocess.Popen([sys.executable,'-u','-m','benchmark.scripts.latent32_batch',str(gpu),str(batch)],env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT))
returncodes=[worker.wait() for worker in workers]
if any(returncodes):raise RuntimeError('Generation worker failed; inspect worker logs')
for split in ['train','test']:
 images=folder/'images'/('teacher_wm_'+split)
 manifest(images,prompts[split])
 assert verify_images(images,prompts[split])
reused.write(folder/'status.json',dict(stage='student_training',steps=12500,pid=os.getpid()))
reused.train(method)
from benchmark.shared.evaluation.student_images import generate
job=folder/'student';out=job/'evaluation/12500';link=out/'students/sd14_r4';link.parent.mkdir(parents=True,exist_ok=True)
if not link.exists():link.symlink_to(job/'students/sd14_r4/checkpoint-12500',target_is_directory=True)
images=out/'test_images'
if not verify_images(images,prompts['test']):generate('sd14','r4',assets['student_path'],out,prompts['test'],images)
wm=reader();key=np.array(message_to_bits(cfg.message));results={}
for role,directory in [('teacher',folder/'images/teacher_wm_test'),('student',images)]:
 scores={'original':[],'jpeg25':[]}
 for path in sorted(directory.glob('sample_*.png')):
  with Image.open(path) as im:
   im=im.convert('RGB');scores['original'].append(int((np.array(wm._read_bits(im))==key).sum()))
   b=io.BytesIO();im.save(b,format='JPEG',quality=25,subsampling=2);b.seek(0)
   with Image.open(b) as decoded:scores['jpeg25'].append(int((np.array(wm._read_bits(decoded.convert('RGB')))==key).sum()))
 reused.write(out/(role+'_scores.json'),scores)
 results[role]={label:dict(n=len(v),BA=sum(v)/len(v)/32,exact32=sum(s==32 for s in v)/len(v),accept_ge24=sum(s>=24 for s in v)/len(v)) for label,v in scores.items()}
 reused.write(out/'results.json',results)
reused.write(folder/'status.json',dict(stage='completed'))
