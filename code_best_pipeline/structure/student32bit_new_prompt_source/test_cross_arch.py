"""Small numerical checks before queuing expensive model downloads/training."""
import torch,numpy as np,tempfile,random
from pathlib import Path
from diffusers import DDPMScheduler,FlowMatchEulerDiscreteScheduler,PixArtTransformer2DModel,SanaTransformer2DModel
from peft import LoraConfig,get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import save_file,load_file

def main():
    torch.set_num_threads(2)
    x=torch.randn(4,4,8,8);noise=torch.randn_like(x);sigma=torch.tensor([0.,.25,.75,1.]).reshape(4,1,1,1)
    y=(1-sigma)*x+sigma*noise
    assert torch.equal(y[0],x[0]) and torch.equal(y[-1],noise[-1])
    target=noise-x
    torch.testing.assert_close(y-sigma*target,x)
    scheduler=DDPMScheduler(num_train_timesteps=1000,prediction_type='epsilon')
    t=torch.tensor([0,200,500,999]);alpha=scheduler.alphas_cumprod[t].reshape(4,1,1,1)
    torch.testing.assert_close(scheduler.add_noise(x,noise,t),alpha.sqrt()*x+(1-alpha).sqrt()*noise)
    # Verify PEFT adapter-only save/load and trainable-name full-state roundtrip.
    from copy import deepcopy
    base=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.SiLU(),torch.nn.Linear(8,4))
    from peft import get_peft_model
    cfg=LoraConfig(r=32,lora_alpha=32,target_modules=['0','2'],init_lora_weights='gaussian')
    m=get_peft_model(deepcopy(base),cfg);opt=torch.optim.AdamW([v for v in m.parameters() if v.requires_grad],lr=1e-4)
    inp=torch.randn(3,4);m(inp).square().mean().backward();opt.step()
    expected=m(inp).detach();state=get_peft_model_state_dict(m)
    n=get_peft_model(deepcopy(base),cfg);set_peft_model_state_dict(n,state)
    torch.testing.assert_close(n(inp),expected)
    with tempfile.TemporaryDirectory() as folder:
        file=Path(folder)/'adapter.safetensors';save_file({k:v.detach().clone().contiguous() for k,v in state.items()},str(file))
        restored=load_file(str(file));assert restored.keys()==state.keys()
        for k in restored:torch.testing.assert_close(restored[k],state[k],atol=0,rtol=0)
    print('PASS flow targets, DDPM epsilon path, adapter-only save/load parity')

if __name__=='__main__':main()
