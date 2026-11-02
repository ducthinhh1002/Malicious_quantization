"""Frozen ordinary LoRA trainer + declared LR schedule and compact state hooks.

No carrier, key, extractor, surrogate or watermark loss is imported by training.
"""
import bootstrap
import argparse
from pathlib import Path
import runpy
import sys
from objectives import learning_rate_factor


def accelerator_class():
    import torch
    from accelerate import Accelerator
    from safetensors.torch import save_file, load_file

    class CompactAccelerator(Accelerator):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.num_processes!=1:
                raise RuntimeError('Single GPU protocol only')
            self.register_save_state_pre_hook(self.save_adapters)
            self.register_load_state_pre_hook(self.load_adapters)
        def save_adapters(self,models,weights,directory):
            if len(models)!=1: raise RuntimeError('Expected only student UNet')
            state={n:p.detach().cpu().contiguous() for n,p in self.unwrap_model(models[0]).named_parameters() if p.requires_grad}
            if not state or any('lora_' not in n or not torch.isfinite(t).all() for n,t in state.items()):
                raise RuntimeError('Unexpected/nonfinite trainable student parameters')
            save_file(state,str(Path(directory)/'trainable.safetensors'))
            weights.clear()  # Frozen base UNet is already pinned; keep optimizer/RNG/scaler/scheduler.
        def load_adapters(self,models,directory):
            if len(models)!=1: raise RuntimeError('Expected only student UNet')
            state=load_file(str(Path(directory)/'trainable.safetensors'))
            params={n:p for n,p in self.unwrap_model(models[0]).named_parameters() if p.requires_grad}
            if params.keys()!=state.keys():raise RuntimeError('Trainable state mismatch')
            with torch.no_grad():
                for name,param in params.items():
                    if param.shape!=state[name].shape or not torch.isfinite(state[name]).all():
                        raise RuntimeError('Invalid saved adapter')
                    param.copy_(state[name])
            models.clear()
    return CompactAccelerator


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trainer',type=Path,required=True)
    p.add_argument('--trainer-sha256',required=True)
    a,arguments=p.parse_known_args()
    if bootstrap.sha256(a.trainer)!=a.trainer_sha256:
        raise RuntimeError('Pinned upstream student trainer changed')
    if '--resume_from_checkpoint' in arguments:
        raise RuntimeError('This pilot starts from clean initialization; inspect saved state before any explicit continuation')
    import torch
    CompactAccelerator=accelerator_class()
    namespace=runpy.run_path(str(a.trainer),run_name='pn90_frozen_student_trainer')
    globals_=namespace['main'].__globals__
    globals_['Accelerator']=CompactAccelerator
    def scheduler(name,optimizer,**kwargs):
        total=kwargs['num_training_steps']
        if name!='constant' or total!=15000 or kwargs['num_warmup_steps']!=0:
            raise RuntimeError('Unexpected training schedule arguments')
        return torch.optim.lr_scheduler.LambdaLR(optimizer,learning_rate_factor)
    globals_['get_scheduler']=scheduler
    sys.argv=[str(a.trainer),*arguments]
    namespace['main']()


if __name__=='__main__':main()
