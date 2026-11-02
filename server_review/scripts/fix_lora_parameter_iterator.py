"""Apply and record the verified gradient-clipping fix without modifying run artifacts."""
import difflib
import hashlib
import json
import py_compile
from pathlib import Path
import torch
from accelerate import Accelerator

root=Path('/root/workspace/serum')
path=root/'third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py'
old=path.read_text()
needle='    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())'
replacement='    lora_layers = [p for p in unet.parameters() if p.requires_grad]'
assert old.count(needle)==1, 'Unexpected source; refusing automatic patch'
new=old.replace(needle,replacement)
patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/examples/text_to_image/train_text_to_image_lora.py',tofile='b/examples/text_to_image/train_text_to_image_lora.py'))
(root/'manifests/lora_gradient_clipping_fix.patch').write_text(patch)
path.write_text(new)
py_compile.compile(str(path),doraise=True)
accelerator=Accelerator(cpu=True)
p=torch.nn.Parameter(torch.ones(1))
params=[p]
optimizer=torch.optim.AdamW(params,lr=1e-5)
observed=[]
for _ in range(2):
    p.grad=torch.tensor([100.])
    norm=accelerator.clip_grad_norm_(params,1.)
    observed.append({'norm_before':float(norm),'gradient_after':float(p.grad)})
    assert float(p.grad)<=1.00001 and float(norm)==100.
    optimizer.step()
    optimizer.zero_grad()
record={'source':str(path),'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest(),'change':'Materialize trainable LoRA parameters so optimizer construction does not exhaust gradient clipping input','accelerate_clipping_two_steps':observed,'existing_runs':'Unmodified; all six seed-0 students were trained before this fix','causal_effect_on_tpr':'Unknown; requires controlled rerun'}
(root/'manifests/lora_gradient_clipping_fix.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
