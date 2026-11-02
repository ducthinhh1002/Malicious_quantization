import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from schedule import learning_rate_factor
from resume_state import accelerator_class,verify_checkpoint


def worker(root,total,resume,device):
    from accelerate.utils import set_seed
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    set_seed(31)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.lora_weight=torch.nn.Parameter(torch.randn(3))
        def forward(self,x):
            return (x*self.lora_weight).sum(-1)
    identity={'test':'continuous vs full-state resume','device':device}
    cls=accelerator_class(identity,17,4,root)
    acc=cls(cpu=device=='cpu',mixed_precision='fp16' if device=='cuda' else 'no')
    model=Model()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:learning_rate_factor(7498+n*1000))
    data=torch.arange(24,dtype=torch.float32).reshape(8,3)/20
    loader=torch.utils.data.DataLoader(data,batch_size=2,shuffle=True,num_workers=0)
    model,opt,loader,sch=acc.prepare(model,opt,loader,sch)
    if resume:acc.load_state(root/'checkpoint-4')
    start=acc.progress.step
    seen=[]
    for epoch in range(start//4,2):
        loader.set_epoch(epoch)
        for batch in loader:
            with acc.accumulate(model):
                noise=torch.randn_like(batch)*.01
                factor=random.random()+float(np.random.rand())
                with acc.autocast():loss=(model(batch+noise)-factor).square().mean()
                acc.backward(loss);acc.clip_grad_norm_(model.parameters(),1.)
                opt.step();sch.step();opt.zero_grad()
            step=acc.progress.step+1
            acc.log({'train_loss':loss.item()},step=step)
            seen.append(batch.cpu().tolist())
            if step%4==0:acc.save_state(root/f'checkpoint-{step}')
            if step>=total:break
        if acc.progress.step>=total:break
    state={'params':{n:p.detach().cpu() for n,p in acc.unwrap_model(model).named_parameters()},
        'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'progress':acc.progress.state_dict(),
        'next_torch':torch.rand(4),'next_python':random.random(),'next_numpy':np.random.rand(),
        'seen':seen}
    if torch.cuda.is_available() and device=='cuda':state['next_cuda']=torch.rand(4,device='cuda').cpu()
    torch.save(state,root/f'final-{total}.pt')


class ResumeTest(unittest.TestCase):
    def test_schedule_absolute_steps_and_extension(self):
        self.assertEqual(learning_rate_factor(7500),1.)
        self.assertAlmostEqual(learning_rate_factor(10000),.6)
        self.assertAlmostEqual(learning_rate_factor(12500),.2)
        self.assertAlmostEqual(learning_rate_factor(20000),.2)
        self.assertTrue(all(learning_rate_factor(i)>=learning_rate_factor(i+1) for i in range(7500,12500)))

    def compare_resume(self,device):
        with tempfile.TemporaryDirectory(prefix='student12500_resume_') as tmp:
            root=Path(tmp);a=root/'full';b=root/'split'
            for folder,total,resume in [(a,8,False),(b,4,False),(b,8,True)]:
                cmd=[sys.executable,__file__,'--worker',str(folder),'--total',str(total),'--device',device]
                if resume:cmd.append('--resume')
                p=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stdout+'\n'+p.stderr)
            x=torch.load(a/'final-8.pt',weights_only=False);y=torch.load(b/'final-8.pt',weights_only=False)
            def compare(left,right):
                if torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=0,atol=0)
                elif isinstance(left,dict):
                    self.assertEqual(left.keys(),right.keys())
                    for k in left:compare(left[k],right[k])
                elif isinstance(left,(tuple,list)):
                    self.assertEqual(len(left),len(right))
                    for aa,bb in zip(left,right):compare(aa,bb)
                else:self.assertEqual(left,right)
            for key in ['params','optimizer','scheduler','progress','next_torch','next_python','next_numpy']:
                compare(x[key],y[key])
            compare(x['seen'][4:],y['seen'])
            if device=='cuda':compare(x['next_cuda'],y['next_cuda'])
            proof=verify_checkpoint(b/'checkpoint-8',{'test':'continuous vs full-state resume','device':device})
            self.assertEqual(proof['step'],8)
            bad=b/'checkpoint-8/optimizer.bin'
            with bad.open('ab') as f:f.write(b'tamper')
            with self.assertRaises(RuntimeError):verify_checkpoint(b/'checkpoint-8',proof['identity'])

    def test_cpu_resume_preserves_updates_data_rng_and_optimizer(self):
        self.compare_resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_gpu_fp16_resume_preserves_scaler_and_next_update(self):
        self.compare_resume('cuda')

    def test_pinned_trainer_patch_only_adds_epoch_setting(self):
        from driver import patched_source
        text='start\n    for epoch in range(first_epoch, args.num_train_epochs):\n        unet.train()\n'
        changed=patched_source(text)
        self.assertEqual(changed.replace('        train_dataloader.set_epoch(epoch)\n',''),text)
        with self.assertRaises(RuntimeError):patched_source('wrong source')


if __name__=='__main__':
    if '--worker' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--total',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--device')
        a=p.parse_args();worker(a.worker,a.total,a.resume,a.device)
    else:unittest.main()
