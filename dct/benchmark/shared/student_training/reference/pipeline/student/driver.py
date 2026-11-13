"""Isolated wrapper of the pinned ordinary student trainer.

Only LR schedule, deterministic epoch sampler, checkpoint storage and audit
are changed. No watermark objective, reader, key or owner score in training.
"""
import argparse
import sys
from pathlib import Path
import bootstrap
from bootstrap import read_json,sha256
from schedule import learning_rate_factor
from resume_state import accelerator_class,verify_checkpoint


def patched_source(source):
    marker='    for epoch in range(first_epoch, args.num_train_epochs):\n'
    if source.count(marker)!=1:raise RuntimeError('Pinned trainer loop changed')
    return source.replace(marker,marker+'        train_dataloader.set_epoch(epoch)\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trainer',type=Path,required=True)
    p.add_argument('--identity-file',type=Path,required=True)
    p.add_argument('--technical-smoke',action='store_true')
    a,arguments=p.parse_known_args()
    identity=read_json(a.identity_file)
    if sha256(a.trainer)!=identity['trainer_sha256']:raise RuntimeError('Trainer hash changed')
    def arg(name):return arguments[arguments.index(name)+1]
    total=int(arg('--max_train_steps'));output=Path(arg('--output_dir'))
    if ((not a.technical_smoke and (total<12500 or total%1250))
            or (a.technical_smoke and total!=4) or int(arg('--checkpointing_steps'))!=1250):
        raise RuntimeError('1250-step epoch boundaries and >=12500 total steps required')
    if int(arg('--train_batch_size'))!=4 or int(arg('--seed'))!=0 or int(arg('--dataloader_num_workers'))!=0:
        raise RuntimeError('Unexpected deterministic data configuration')
    if '--resume_from_checkpoint' in arguments:
        directory=output/Path(arg('--resume_from_checkpoint')).name
        proof=verify_checkpoint(directory,identity)
        if proof['step']>=total:raise RuntimeError('Resume target must be later than saved step')
    import torch
    torch.cuda.set_per_process_memory_fraction(min(1.,44*2**30/torch.cuda.get_device_properties(0).total_memory))
    source=patched_source(a.trainer.read_text())
    ns={'__name__':'structure12500_pinned_trainer','__file__':str(a.trainer)}
    exec(compile(source,str(a.trainer),'exec'),ns)
    ns['Accelerator']=accelerator_class(identity,0,1250,output)
    def scheduler(name,optimizer,**kwargs):
        if name!='constant' or kwargs['num_warmup_steps']!=0:
            raise RuntimeError('Unexpected scheduler arguments')
        return torch.optim.lr_scheduler.LambdaLR(optimizer,learning_rate_factor)
    ns['get_scheduler']=scheduler
    sys.argv=[str(a.trainer),*arguments]
    ns['main']()


if __name__=='__main__':main()
