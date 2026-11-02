import gc,json
from pathlib import Path
import torch
R=Path('/root/workspace/serum'); out={}
torch.set_num_threads(2)
for name in ['C10_full_seed0_author','C11_full_step1.00_seed0_author','C10_seed0_author','C11_step1.00_seed0_author']:
    p=R/f'students/{name}/checkpoint-20000/optimizer.bin'
    if not p.exists():out[name]={'optimizer_missing':True};continue
    state=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    steps=sorted({float(v['step']) for v in state['state'].values() if 'step' in v})
    groups=[{k:v for k,v in x.items() if k!='params'} for x in state['param_groups']]
    out[name]={'optimizer_steps':steps,'param_groups':groups,'parameter_count':len(state['state'])}
    del state;gc.collect()
    print(name,out[name],flush=True)
# Exercise exact iterator/list patterns with two successive clipping calls.
from accelerate import Accelerator
a=Accelerator(cpu=True)
for kind in ['full','lora']:
    model=torch.nn.Linear(1,1,bias=False);params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(model.parameters() if kind=='full' else params,lr=5e-6)
    model,opt=a.prepare(model,opt);checks=[]
    for _ in range(2):
        opt.zero_grad();a.backward((model(torch.tensor([[100.0]]))-100)**2)
        pre=float(a.clip_grad_norm_(model.parameters() if kind=='full' else params,1.))
        post=float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()])))
        checks.append({'before':pre,'after':post});assert pre>1 and post<=1.00001
        opt.step()
    out['clipping_'+kind]=checks
(R/'metrics/serum_author/c10_c11_optimizer_review_20261012.json').write_text(json.dumps(out,indent=2))
