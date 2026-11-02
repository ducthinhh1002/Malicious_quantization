import json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
r=Path('/root/workspace/serum')
weights=load_file(str(r/'data/overnight_training_smoke/pytorch_lora_weights.safetensors'))
assert weights and all(torch.isfinite(t).all() for t in weights.values())
pipe=StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',torch_dtype=torch.float16,safety_checker=None,local_files_only=True).to('cuda')
pipe.load_lora_weights(str(r/'data/overnight_training_smoke'))
image=pipe('a landscape',num_inference_steps=2,height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(42)).images[0]
assert image.size==(512,512) and image.mode=='RGB'
(r/'metrics/overnight_training_preflight.json').write_text(json.dumps({'ok':True,'scope':'One-step training, Accelerate checkpoint save and offline LoRA inference smoke test only; not a research run'},indent=2))
print('Offline training + adapter save/load + generation passed',flush=True)
