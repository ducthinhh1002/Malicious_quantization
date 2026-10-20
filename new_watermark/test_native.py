"""Native trajectory, fixed reader and calibration checks without downloads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paths
import unittest
import tempfile
from unittest.mock import patch


class Native(unittest.TestCase):
    def test_training_loop_prior_cycle_snapshot_and_report_on_cpu(self):
        import contextlib
        import io
        import json
        import torch
        from types import SimpleNamespace
        from diffusers import UNet2DConditionModel,DDIMScheduler,DDPMScheduler
        from safetensors.torch import save_file
        from new_watermark.native_unet import train_native,TransportReader
        from new_watermark.launch import arguments,config
        from new_watermark.paths import write_json,read_json
        torch.set_num_threads(2)
        unet=UNet2DConditionModel(sample_size=4,in_channels=4,out_channels=4,
             down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),
             block_out_channels=(8,),layers_per_block=1,norm_num_groups=4,cross_attention_dim=8,attention_head_dim=2)
        class VAE(torch.nn.Module):
            config=SimpleNamespace(scaling_factor=1.)
            def decode(self,z): return SimpleNamespace(sample=z[:,:3].tanh())
            def encode(self,x):
                z=torch.cat((x,x[:,:1]),1)
                return SimpleNamespace(latent_dist=SimpleNamespace(mean=z,std=torch.full_like(z,.01)))
        class Perceptual(torch.nn.Module):
            def forward(self,x,y): return (x-y).square().mean()
        reader=TransportReader(width=4,input_size=4).eval().requires_grad_(False)
        c=config(arguments(['--owner-checkpoint','/unused','--owner-sha256','a'*64,
                            '--teacher-rank','2','--teacher-checkpoints','2','4','--teacher-tail-steps','0',
                            '--teacher-warmup-steps','1','--inference-steps','2','--native-prior-weight','1',
                            '--native-bit-tail-weight','1','--meta-weight','0']))
        pipe=SimpleNamespace(unet=unet,vae=VAE(),text_encoder=torch.nn.Identity(),
                             scheduler=DDIMScheduler(num_train_timesteps=20,clip_sample=False))
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp); write_json(run/'config.json',c)
            write_json(run/'assets.json',{'student':'/unused'})
            write_json(run/'result/detector.json',{'keys':[221,34],'thresholds':[-2.,-2.]})
            (run/'teacher_cache').mkdir()
            name='teacher_train_000000.safetensors'
            save_file({'latent':torch.randn(1,4,4,4),'text':torch.randn(2,3,8)},str(run/'teacher_cache'/name))
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules,{'lpips':SimpleNamespace(LPIPS=lambda **kw:Perceptual())}))
                stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self))
                stack.enter_context(patch.object(torch.nn.Module,'cuda',lambda self,*a,**k:self))
                stack.enter_context(patch('torch.cuda.max_memory_allocated',return_value=0))
                stack.enter_context(patch('new_watermark.data.load_base',return_value=pipe))
                stack.enter_context(patch('new_watermark.teacher.verify_cache',return_value={'files':{name:'unused'}}))
                stack.enter_context(patch('new_watermark.native_unet.load_reader',return_value=reader))
                stack.enter_context(patch.object(DDPMScheduler,'from_pretrained',return_value=DDPMScheduler(num_train_timesteps=20)))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                train_native(run)
            audit=read_json(run/'result/teacher_training_audit.json')
            self.assertEqual((audit['applied'],audit['skipped']),(4,0))
            self.assertEqual(set(read_json(run/'teacher/complete.json')['snapshots']),{'2','4'})
            records=[json.loads(line) for line in (run/'teacher/training.jsonl').read_text().splitlines()]
            self.assertIn('cycle',records[-1]['terms'])
            self.assertGreater(records[-1]['terms']['prior_relative_mse'],0.)
            self.assertIn('weighted_total_loss',records[-1]['terms'])
            self.assertTrue(all(p.grad is None for p in reader.parameters()))

    def test_union_excludes_old_reader_and_new_development_splits(self):
        import json
        from new_watermark.data import prior_prompt_exclusion
        from new_watermark.launch import arguments,config
        from new_watermark.paths import write_json
        with tempfile.TemporaryDirectory() as temp:
            first=Path(temp)/'first.json'; second=Path(temp)/'second.json'
            write_json(first,{'train':[{'image_id':1}],'test':[{'image_id':2}]})
            write_json(second,{'train':[{'image_id':2}],'test':[{'image_id':3}]})
            c=config(arguments(['--exclude-prior-prompts',str(first),'--exclude-additional-prompts',str(second)]))
            json.dumps(c)
            ids,provenance=prior_prompt_exclusion(c)
            self.assertEqual(ids,{1,2,3})
            self.assertEqual(len(provenance['sources']),2)

    def test_bit_tail_focus_and_reader_diagnostics(self):
        import torch
        from new_watermark.native_unet import bit_objective,reader_diagnostics
        x=torch.tensor([[-2.,3.,3.,3.,3.,3.,3.,3.]],requires_grad=True)
        mean,_,_=bit_objective(x,torch.ones_like(x),0.)
        average_gradient=torch.autograd.grad(mean,x)[0]
        focused,_,_=bit_objective(x,torch.ones_like(x),1.)
        gradient=torch.autograd.grad(focused,x)[0]
        self.assertGreater((gradient[0,0]/average_gradient[0,0]).item(),
                           (gradient[0,1]/average_gradient[0,1]).item())
        d=reader_diagnostics([[5.,1.,-1.,1.,1.,1.,-1.,1.,1.],
                              [5.,-1.,-1.,1.,1.,1.,-1.,1.,1.]],[221,34])
        self.assertEqual(d['exact_message_accuracy'],.5)
        self.assertEqual(d['per_bit_accuracy'][0],.5)
        self.assertEqual(d['mean_presence_logit'],5.)

    def test_full_trajectory_gradient_matches_finite_difference(self):
        import torch
        from types import SimpleNamespace
        from diffusers import DDIMScheduler
        from new_watermark.native_unet import trajectory
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor(.08,dtype=torch.float64))
            def forward(self,x,t,h):
                return SimpleNamespace(sample=self.weight*x.tanh())
        model=Tiny()
        scheduler=DDIMScheduler(num_train_timesteps=20,clip_sample=False)
        noise=torch.tensor([[[[.2,.7],[-.4,.1]]]],dtype=torch.float64)
        text=torch.zeros(2,1,1,dtype=torch.float64)
        def objective(grad,tail):
            return trajectory(model,scheduler.config,noise,text,4,tail,grad).square().sum()
        objective(True,0).backward()
        full=model.weight.grad.item()
        model.weight.grad=None
        objective(True,1).backward()
        truncated=model.weight.grad.item()
        epsilon=1e-5
        with torch.no_grad():
            model.weight.add_(epsilon); plus=objective(False,0).item()
            model.weight.sub_(2*epsilon); minus=objective(False,0).item()
            model.weight.add_(epsilon)
        self.assertAlmostEqual(full,(plus-minus)/(2*epsilon),places=6)
        self.assertGreater(abs(full-truncated),1e-4)

    def test_prior_zero_at_base_and_gradients_restore_drift(self):
        import torch
        from types import SimpleNamespace
        from diffusers import DDPMScheduler
        from new_watermark.native_unet import prior_loss,learning_rate_factor
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.delta=torch.nn.Parameter(torch.tensor(0.))
                self.enabled=True
            def disable_adapters(self): self.enabled=False
            def enable_adapters(self): self.enabled=True
            def forward(self,x,t,h):
                return SimpleNamespace(sample=x*.4+(self.delta if self.enabled else 0))
        model=Tiny(); scheduler=DDPMScheduler(num_train_timesteps=20)
        args=(scheduler,torch.ones(1,1,2,2),torch.zeros(2,1,1),torch.zeros(1,1,2,2),torch.tensor([3]),.01)
        loss,_=prior_loss(model,*args)
        self.assertEqual(loss.item(),0)
        self.assertTrue(model.enabled)
        with torch.no_grad(): model.delta.fill_(.1)
        loss,terms=prior_loss(model,*args); loss.backward()
        self.assertGreater(model.delta.grad.item(),0)
        self.assertGreater(terms['prior_relative_mse'].item(),0)
        self.assertTrue(model.enabled)
        self.assertAlmostEqual(learning_rate_factor(1,2000,.1,100),.01)
        self.assertAlmostEqual(learning_rate_factor(100,2000,.1,100),1.)
        self.assertAlmostEqual(learning_rate_factor(2000,2000,.1,100),.1)

    def test_audit_reports_quality_failures_without_aborting(self):
        import json
        from new_watermark.training_audit import audit_training
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'log.jsonl'
            rows=[{'step':1,'applied':1,'skipped':0,'gradient_norm':20.,
                   'terms':{'mse':.02,'lpips':.3},'meta':{'proxy_gain':-.01}},
                  {'step':2,'applied':1,'skipped':1,'gradient_norm':None,'terms':{},'meta':{}}]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            report=audit_training(path,{'teacher_pixel_budget':.001,'teacher_lpips_budget':.1,'teacher_tail_steps':4})
            self.assertEqual(report['applied'],1)
            self.assertEqual(report['skipped'],1)
            self.assertEqual(report['meta_proxy']['nonpositive_count'],1)
            self.assertEqual(report['clipped_fraction'],1.)
            self.assertGreaterEqual(len(report['warnings']),4)
            json.dumps(report,allow_nan=False)

    def test_reader_score_matches_numpy_and_loss_has_image_gradient(self):
        import numpy as np
        import torch
        from new_watermark.native_unet import TransportReader,reader_features,keyed_scores,detection_loss
        from new_watermark.method import score_features
        torch.set_num_threads(2); torch.manual_seed(71)
        reader=TransportReader(width=4,input_size=16).eval().requires_grad_(False)
        x=torch.rand(2,3,32,32,requires_grad=True)
        d={'type':'frozen_lalt_reader_8bit','keys':[221,34],'thresholds':[-2.,-2.]}
        f=reader_features(reader,x)
        np.testing.assert_allclose(keyed_scores(f,d['keys']).detach().numpy(),
            score_features(f.detach().numpy(),d),rtol=1e-6)
        loss,_=detection_loss(reader,x,d,.25)
        loss.backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in reader.parameters()))

    def test_current_adapter_is_used_before_tail_and_tail_has_gradient(self):
        import torch
        from diffusers import UNet2DConditionModel,DDIMScheduler
        from diffusers.models.attention_processor import AttnProcessor
        from peft import LoraConfig
        from new_watermark.native_unet import trajectory
        torch.set_num_threads(2); torch.manual_seed(72)
        model=UNet2DConditionModel(sample_size=4,in_channels=4,out_channels=4,
             down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),
             block_out_channels=(8,),layers_per_block=1,norm_num_groups=4,cross_attention_dim=8,attention_head_dim=2)
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=2,lora_alpha=2,target_modules=['to_q','to_v','conv_out']))
        model.set_attn_processor(AttnProcessor()); model.eval()
        with torch.no_grad():
            for n,p in model.named_parameters():
                if '.lora_B.' in n: p.normal_(std=.01)
        sched=DDIMScheduler(num_train_timesteps=20,clip_sample=False)
        noise=torch.randn(1,4,4,4); text=torch.randn(2,3,8)
        events=[]
        hook=model.register_forward_pre_hook(lambda m,args:events.append((torch.is_grad_enabled(),m.active_adapters())))
        z=trajectory(model,sched.config,noise,text,4,2,True)
        self.assertEqual([e[0] for e in events[:4]],[False,False,True,True])
        self.assertTrue(all(e[1]==['default'] for e in events[:4]))
        hook.remove(); z.square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)
        with torch.no_grad(): expected=trajectory(model,sched.config,noise,text,4,2,False)
        torch.testing.assert_close(z,expected)

    def test_two_key_calibration_and_ties(self):
        import numpy as np
        from new_watermark.native_unet import calibrate_native
        from new_watermark.paths import write_json,read_json
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            write_json(run/'config.json',{'owner_key':221,'owner_wrong_key':34,'owner_sha256':'a'*64,'fpr':.01})
            write_json(run/'features/clean_calibration.json',{})
            cal={'features':np.zeros((1000,9)).tolist()}
            calibrate_native(run,cal)
            d=read_json(run/'result/detector.json')
            from new_watermark.method import score_features
            self.assertFalse((score_features(cal['features'],d)>=d['thresholds']).any())
            self.assertEqual(d['nominal_per_key_fpr'],.005)


if __name__=='__main__': unittest.main()
