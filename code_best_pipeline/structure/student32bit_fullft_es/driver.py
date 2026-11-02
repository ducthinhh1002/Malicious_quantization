"""Pinned full UNet trainer; deterministic sampler and absolute LR cooldown."""
import bootstrap
from bootstrap import ROOT,HARNESS,read_json,sha256
import argparse,sys,json
from pathlib import Path
def patch(source):
    marker='    for epoch in range(first_epoch, args.num_train_epochs):\n'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'        train_dataloader.set_epoch(epoch)\n')
    marker='                global_step += 1\n';assert source.count(marker)==1
    source=source.replace(marker,"                if accelerator.optimizer_step_was_skipped:raise RuntimeError('Skipped optimizer update; inspect scaler before continuation')\n"+marker)
    marker='    text_encoder.requires_grad_(False)\n';assert source.count(marker)==1
    source=source.replace(marker,marker+"    assert all(p.requires_grad for p in unet.parameters())\n    assert not any(p.requires_grad for p in vae.parameters())\n    assert not any(p.requires_grad for p in text_encoder.parameters())\n    assert not any('lora' in n.lower() for n,p in unet.named_parameters())\n    print('FULL_UNET_TRAINABLE_PARAMETERS',sum(p.numel() for p in unet.parameters()),flush=True)\n")
    marker='                accelerator.log({"train_loss": train_loss}, step=global_step)\n'
    assert source.count(marker)==1
    source=source.replace(marker,"                append_metrics(Path(args.output_dir).parents[1]/'metrics/train_loss.jsonl', {'step':global_step,'loss':train_loss,'learning_rate':lr_scheduler.get_last_lr()[0]})\n"+marker+"                if global_step % 500 == 0:\n                    if validation_control(unet,vae,text_encoder,tokenizer,noise_scheduler,accelerator,global_step,args.output_dir):args.max_train_steps=global_step\n")
    marker='    progress_bar = tqdm(\n';assert source.count(marker)==1
    source=source.replace(marker,"    if global_step>0:validation_control(unet,vae,text_encoder,tokenizer,noise_scheduler,accelerator,global_step,args.output_dir)\n"+marker)
    marker='        if accelerator.is_main_process:\n            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:'
    assert source.count(marker)==1
    source=source.replace(marker,'        if global_step >= args.max_train_steps:break\n'+marker)
    marker='        for step, batch in enumerate(train_dataloader):\n';assert source.count(marker)==1
    source=source.replace(marker,"        resume_skip=initial_global_step % num_update_steps_per_epoch if epoch==first_epoch else 0\n        active_loader=accelerator.skip_first_batches(train_dataloader,resume_skip) if resume_skip else train_dataloader\n        for step, batch in enumerate(active_loader):\n")
    source=source.replace('global_step = int(path.split("-")[1])','global_step = int(path.split("-")[-1])')
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
    from validation_control import ValidationController,append
    ns={'Path':Path,'validation_control':ValidationController(),'append_metrics':append,'__name__':'full_student_trainer','__file__':str(trainer)}
    exec(compile(patch(trainer.read_text()),str(trainer),'exec'),ns)
    ns['Accelerator']=DeterministicAccelerator
    def scheduler(name,optimizer,**kw):
        assert name=='constant' and kw['num_warmup_steps']==0
        return torch.optim.lr_scheduler.LambdaLR(optimizer,learning_rate_factor)
    ns['get_scheduler']=scheduler;sys.argv=[str(trainer),*args];ns['main']()
if __name__=='__main__':main()
