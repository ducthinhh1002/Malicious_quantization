import gc, inspect, json, time
from pathlib import Path
import torch
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline
R=Path('/root/workspace/serum')
result={}
base='SG161222/Realistic_Vision_V4.0_noVAE'
s=UniPCMultistepScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True)
s.set_timesteps(50)
t=int(s.timesteps[0]);a=float(s.alphas_cumprod[t])
result['regen_scheduler']={'class':type(s).__name__,'t0':t,'alpha_bar':a,'signal_coefficient':a**.5,'noise_coefficient':(1-a)**.5,'config':dict(s.config)}
result['prepare_latents_source']=inspect.getsource(StableDiffusionControlNetImg2ImgPipeline.prepare_latents)
result['get_timesteps_source']=inspect.getsource(StableDiffusionControlNetImg2ImgPipeline.get_timesteps)
result['versions']={'torch':torch.__version__,'diffusers':__import__('diffusers').__version__}
(R/'metrics/serum_author/c10_c11_regen_numerics_20261012.json').write_text(json.dumps(result,indent=2,default=str))
print(json.dumps(result,indent=2,default=str))
