"""C10 diagnostics; preserves all original training and evaluation artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

root = Path('/root/workspace/serum')
out = {}
# Reproduce the iterator lifecycle used by the pinned training script.
p = torch.nn.Parameter(torch.ones(1))
params = filter(lambda x: x.requires_grad, [p])
optimizer = torch.optim.AdamW(params)
p.grad = torch.tensor([100.])
norm = torch.nn.utils.clip_grad_norm_(params, 1.)
out['gradient_clipping_reproduction'] = {'returned_norm': float(norm), 'gradient_after': float(p.grad), 'optimizer_parameter_count': len(optimizer.param_groups[0]['params'])}
assert float(p.grad) == 100.
fixed = [p]
fixed_norm = torch.nn.utils.clip_grad_norm_(fixed, 1.)
out['gradient_clipping_list_fix'] = {'returned_norm': float(fixed_norm), 'gradient_after': float(p.grad)}
assert float(p.grad) <= 1.00001
selection = json.loads((root/'students/C10_seed0/validation_selection.json').read_text())
best = Path(selection['best']['checkpoint'])
state = torch.load(best/'optimizer.bin', map_location='cpu', weights_only=False)
steps = [float(x['step']) for x in state['state'].values()]
out['optimizer'] = {'state_count': len(steps), 'min_step': min(steps), 'max_step': max(steps), 'groups': [{k:v for k,v in g.items() if k!='params'} for g in state['param_groups']]}
weights = load_file(str(best/'pytorch_lora_weights.safetensors'))
out['adapter'] = {'tensors':len(weights), 'all_finite':all(bool(torch.isfinite(x).all()) for x in weights.values()), 'sha256':hashlib.sha256((best/'pytorch_lora_weights.safetensors').read_bytes()).hexdigest()}
del state, weights
pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16, safety_checker=None, local_files_only=True).to('cuda')
pipe.load_lora_weights(str(best))
pipe.set_progress_bar_config(disable=True)
prompts = (root/'prompts/student_eval.txt').read_text().splitlines()[:4]
def generate():
    return pipe(prompts, negative_prompt=['']*4, generator=[torch.Generator(device='cuda').manual_seed(50000+i) for i in range(4)], num_inference_steps=50, guidance_scale=7.5, height=512, width=512).images
with torch.no_grad():
    images = generate()
    pipe.disable_lora()
    base_images = generate()
out['generation_replay'] = {'scheduler':type(pipe.scheduler).__name__, 'selected_checkpoint':str(best), 'images':[]}
for i,(im,base) in enumerate(zip(images,base_images)):
    old = np.array(Image.open(root/f'student_outputs/C10_seed0/sample_{i:06d}.png').convert('RGB')).astype(float)
    fresh = np.array(im).astype(float)
    bare = np.array(base).astype(float)
    out['generation_replay']['images'].append({'index':i, 'replay_exact':bool(np.array_equal(old,fresh)), 'replay_mae':float(np.abs(old-fresh).mean()), 'base_vs_original_mae':float(np.abs(old-bare).mean())})
path=root/'metrics/c10_deep_audit_2026-10-11.json'
path.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
