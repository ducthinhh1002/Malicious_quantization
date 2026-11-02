"""Pinned full UNet trainer; deterministic sampler and absolute LR cooldown."""
import bootstrap
from bootstrap import ROOT,HARNESS,read_json,sha256
import argparse,sys,json
def patch(source):
    marker='    for epoch in range(first_epoch, args.num_train_epochs):\n'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'        train_dataloader.set_epoch(epoch)\n')
    marker='                global_step += 1\n';assert source.count(marker)==1
    source=source.replace(marker,"                if accelerator.optimizer_step_was_skipped:raise RuntimeError('Skipped optimizer update; inspect scaler before continuation')\n"+marker)
    marker='    text_encoder.requires_grad_(False)\n';assert source.count(marker)==1
    source=source.replace(marker,marker+"    assert all(p.requires_grad for p in unet.parameters())\n    assert not any(p.requires_grad for p in vae.parameters())\n    assert not any(p.requires_grad for p in text_encoder.parameters())\n    assert not any('lora' in n.lower() for n,p in unet.named_parameters())\n    print('FULL_UNET_TRAINABLE_PARAMETERS',sum(p.numel() for p in unet.parameters()),flush=True)\n")
    return source
def main():
    import torch
    from accelerate import Accelerator,DataLoaderConfiguration
    from schedule import learning_rate_factor
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);a,args=p.parse_known_args()
    identity=read_json(a.identity);trainer=HARNESS/'vendor/diffusers/train_text_to_image.py'
    assert sha256(trainer)==identity['trainer_sha256']
    class DeterministicAccelerator(Accelerator):
        def __init__(self,*args,**kw):
            kw['dataloader_config']=DataLoaderConfiguration(use_seedable_sampler=True,data_seed=0)
            super().__init__(*args,**kw)
            assert self.num_processes==1 and self.gradient_accumulation_steps==1
    ns={'__name__':'full_student_trainer','__file__':str(trainer)}
    exec(compile(patch(trainer.read_text()),str(trainer),'exec'),ns)
    ns['Accelerator']=DeterministicAccelerator
    def scheduler(name,optimizer,**kw):
        assert name=='constant' and kw['num_warmup_steps']==0
        return torch.optim.lr_scheduler.LambdaLR(optimizer,learning_rate_factor)
    ns['get_scheduler']=scheduler;sys.argv=[str(trainer),*args];ns['main']()
if __name__=='__main__':main()
