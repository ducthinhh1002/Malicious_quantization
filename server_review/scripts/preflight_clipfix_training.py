"""Exercise actual fp16 trainer, clipping, checkpoint save and inference load."""
import json,os,runpy,sys
from pathlib import Path
import torch
from accelerate import Accelerator
from safetensors.torch import load_file

r=Path('/root/workspace/serum');os.chdir(r)
dest=r/'data/clipfix_training_preflight'
assert not dest.exists(),'Use a new diagnostic destination'
original=Accelerator.clip_grad_norm_
calls=[]
def checked(self,parameters,max_norm,*args,**kwargs):
    params=list(parameters)
    assert len(params)==256 and all(p.requires_grad for p in params)
    optimizer_ids={id(p) for opt in self._optimizers for group in opt.param_groups for p in group['params']}
    assert {id(p) for p in params}==optimizer_ids
    before=original(self,params,max_norm,*args,**kwargs)
    gradients=[p.grad.float().norm() for p in params if p.grad is not None]
    norm_after=torch.stack(gradients).norm().item()
    if torch.isfinite(before):assert norm_after<=float(max_norm)+1e-4
    calls.append({'parameter_count':len(params),'max_norm':float(max_norm),'norm_before':float(before) if torch.isfinite(before) else 'nonfinite scaler warmup','norm_after':norm_after if torch.isfinite(torch.tensor(norm_after)) else 'nonfinite scaler warmup'})
    return before
Accelerator.clip_grad_norm_=checked
sys.argv=['train_text_to_image_lora.py','--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4','--train_data_dir=data/hf/C00_train','--image_column=image','--caption_column=text','--resolution=512','--train_batch_size=8','--gradient_accumulation_steps=1','--learning_rate=1e-5','--rank=4','--max_train_steps=4','--checkpointing_steps=2','--mixed_precision=fp16','--seed=0',f'--output_dir={dest}']
runpy.run_path(str(r/'third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py'),run_name='__main__')
assert len(calls)==4 and any(isinstance(c['norm_before'],float) for c in calls)
weights=load_file(str(dest/'checkpoint-4/pytorch_lora_weights.safetensors'))
assert all(torch.isfinite(v).all() for v in weights.values())
assert any(torch.count_nonzero(v)>0 for k,v in weights.items() if 'lora_B' in k)
state=torch.load(dest/'checkpoint-4/optimizer.bin',map_location='cpu',weights_only=False)
assert min(float(v['step']) for v in state['state'].values())>=1
from diffusers import StableDiffusionPipeline
pipe=StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',local_files_only=True,torch_dtype=torch.float16,safety_checker=None).to('cuda')
pipe.load_lora_weights(str(dest/'checkpoint-4'))
im=pipe('a red apple on a wooden table',generator=torch.Generator(device='cuda').manual_seed(50000),num_inference_steps=2,height=512,width=512).images[0]
assert im.size==(512,512)
result={'status':'passed','actual_fp16_training_steps':4,'clipping_calls':calls,'optimizer_updated':True,'checkpoint_save_and_inference_reload':True,'scope':'diagnostic smoke test; no research results overwritten'}
(r/'metrics/clipfix_training_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
