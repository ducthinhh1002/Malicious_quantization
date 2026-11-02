"""Read-only checkpoint and four-image generation replay for original C00."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

r=Path('/root/workspace/serum')
selection=json.loads((r/'students/C00_seed0/validation_selection.json').read_text())
assert Path(selection['images']).name=='clean_val'
best=Path(selection['best']['checkpoint'])
weights=load_file(str(best/'pytorch_lora_weights.safetensors'))
state=torch.load(best/'optimizer.bin',map_location='cpu',weights_only=False)
steps=[float(v['step']) for v in state['state'].values()]
result={'condition':'C00_seed0','validation_images':selection['images'],'validation_count':selection['num_images'],'best':selection['best'],'adapter_tensors':len(weights),'all_finite':all(bool(torch.isfinite(v).all()) for v in weights.values()),'optimizer_step_range':[min(steps),max(steps)],'historical_clipping':'Affected by the same pre-fix iterator bug as C10','replay':[]}
del weights,state
pipe=StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',torch_dtype=torch.float16,safety_checker=None,local_files_only=True).to('cuda')
pipe.load_lora_weights(str(best));pipe.set_progress_bar_config(disable=True)
prompts=(r/'prompts/student_eval.txt').read_text().splitlines()[:4]
def generate():
    return pipe(prompts,negative_prompt=['']*4,generator=[torch.Generator(device='cuda').manual_seed(50000+i) for i in range(4)],num_inference_steps=50,guidance_scale=7.5,height=512,width=512).images
with torch.no_grad():
    adapted=generate();pipe.disable_lora();base=generate()
for i,(a,b) in enumerate(zip(adapted,base)):
    original=np.asarray(Image.open(r/f'student_outputs/C00_seed0/sample_{i:06d}.png').convert('RGB')).astype(float)
    result['replay'].append({'index':i,'exact':bool(np.array_equal(original,np.asarray(a))),'mae':float(np.abs(original-np.asarray(a)).mean()),'unadapted_mae':float(np.abs(original-np.asarray(b)).mean())})
(r/'metrics/c00_runtime_audit_2026-10-11.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
