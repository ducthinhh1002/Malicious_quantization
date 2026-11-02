"""Read training progress and verify that the selected C10 adapter changes predictions."""
import json
import re
from pathlib import Path
import torch
import numpy as np
from diffusers import StableDiffusionPipeline

root = Path('/root/workspace/serum')
report = {'training': {}}
for name, log in [('C10_seed0', 'c10_baseline.log'), ('C00_seed0', 'c00_baseline.log')]:
    text = (root/'logs'/log).read_text(errors='replace')
    values = {}
    for match in re.finditer(r'(\d+)/20000[^\r\n]*?step_loss=([\d.eE+-]+)', text):
        values[int(match[1])] = float(match[2])
    steps = sorted(values)
    report['training'][name] = {
        'last_logged_step': max(steps) if steps else None,
        'first_1000_mean_step_loss': float(np.mean([values[s] for s in steps if 0 < s <= 1000])) if steps else None,
        'last_1000_mean_step_loss': float(np.mean([values[s] for s in steps[-1000:]])) if steps else None,
        'nonfinite_logged_loss': bool(re.search(r'step_loss=(nan|inf)', text)),
        'note': 'Training batch losses vary with random noise and timestep; not the checkpoint selection criterion.',
    }

selection = json.loads((root/'students/C10_seed0/validation_selection.json').read_text())
adapter = selection['best']['checkpoint']
pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32, safety_checker=None).to('cuda')
pipe.unet.eval().requires_grad_(False)
pipe.text_encoder.eval().requires_grad_(False)
prompts = (root/'prompts/student_eval.txt').read_text().splitlines()[:2]
tokens = pipe.tokenizer(prompts, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
with torch.no_grad():
    enc = pipe.text_encoder(tokens.input_ids.to('cuda'))[0]
    noise = torch.randn((2, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(98765), device='cuda')
    timestep = torch.tensor([300, 700], device='cuda')
    before = pipe.unet(noise, timestep, enc).sample
    pipe.load_lora_weights(adapter)
    after = pipe.unet(noise, timestep, enc).sample
    pipe.disable_lora()
    disabled = pipe.unet(noise, timestep, enc).sample
diff = (after-before).float()
report['adapter_runtime_check'] = {
    'checkpoint': adapter,
    'active_adapters': pipe.get_active_adapters(),
    'all_predictions_finite': bool(torch.isfinite(after).all()),
    'mean_absolute_prediction_change': diff.abs().mean().item(),
    'max_absolute_prediction_change': diff.abs().max().item(),
    'relative_l2_prediction_change': (diff.norm()/before.norm()).item(),
    'disabled_vs_base_max_abs': (disabled-before).abs().max().item(),
    'scope': 'Two fixed UNet inputs; verifies adapter execution, does not measure radioactivity or image utility.',
}
assert torch.isfinite(after).all() and diff.abs().max() > 0, 'Adapter has no finite effect'
(root/'metrics/sd14_training_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
