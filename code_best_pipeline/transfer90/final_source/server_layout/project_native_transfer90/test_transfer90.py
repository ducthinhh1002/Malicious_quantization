"""CPU tests covering objective gradients, frozen verification and checkpoint state."""
import bootstrap
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from objectives import robust_margin,bit_margin,learning_rate_factor
from verification import summarize,validated_rows,paired_difference,wilson
from bootstrap import write_json,sha256


class Objectives(unittest.TestCase):
    def test_weak_image_not_cancelled(self):
        x=torch.tensor([[.1,.2,.3],[100.,100.,100.]],requires_grad=True)
        loss=robust_margin(x,torch.ones_like(x),.35).mean();loss.backward()
        self.assertGreater(float(loss),0);self.assertTrue((x.grad[0]<0).all());self.assertTrue((x.grad[1]==0).all())
    def test_worst_view_gets_extra_gradient(self):
        x=torch.tensor([[0.,.5,.5]],requires_grad=True)
        robust_margin(x,torch.ones_like(x),1).sum().backward()
        self.assertGreater(float(x.grad[0,0].abs()),float(x.grad[0,1].abs()))
    def test_satisfied_views_zero(self):
        x=torch.full((2,3),2.,requires_grad=True)
        robust_margin(x,torch.ones_like(x),1).sum().backward()
        self.assertEqual(float(x.grad.abs().sum()),0)
    def test_invalid_inputs(self):
        for sigma in [0,-1]:
            with self.assertRaises(ValueError):robust_margin(torch.zeros(1,3),torch.ones(1,3),sigma)
        with self.assertRaises(ValueError):robust_margin(torch.zeros(1,3),torch.ones(1,2),1)
        with self.assertRaises(ValueError):robust_margin(torch.full((1,3),float('nan')),torch.ones(1,3),1)
    def test_margin_gradcheck(self):
        x=torch.tensor([[.1,.4,.8]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:robust_margin(z,torch.ones_like(z),.35),x))
    def test_bit_gradient(self):
        c=torch.tensor([[0.,0.]],requires_grad=True);s=torch.tensor([1.,-1.])
        bit_margin(c,s).sum().backward()
        self.assertLess(float(c.grad[0,0]),0);self.assertGreater(float(c.grad[0,1]),0)
    def test_lr(self):
        self.assertEqual(learning_rate_factor(0),1);self.assertEqual(learning_rate_factor(10000),1)
        self.assertAlmostEqual(learning_rate_factor(15000),.2)
        self.assertAlmostEqual(learning_rate_factor(12500),.6)
        self.assertTrue(all(learning_rate_factor(s)>=learning_rate_factor(s+1) for s in range(10000,15000)))
    def test_microbatch_objective_gradient(self):
        w=torch.tensor(.3,requires_grad=True);a=torch.arange(1,5.)[:,None]*torch.tensor([[1.,2.,3.]])
        loss=robust_margin(w*a,torch.ones_like(a),.35).mean();g=torch.autograd.grad(loss,w)[0]
        expected=0.
        for row in a:
            term=robust_margin((w*row)[None,:],torch.ones(1,3),.35).sum()/4
            expected+=torch.autograd.grad(term,w)[0]
        torch.testing.assert_close(g,expected)


class Verification(unittest.TestCase):
    def test_zero_not_matching_bits(self):
        r=summarize(np.zeros((10,8)),np.zeros(10),np.ones(8),.85)
        self.assertEqual(r['per_image_bit_accuracy'],0);self.assertEqual(r['aggregate_matched_bits'],0)
    def test_aggregate_not_per_image(self):
        x=np.r_[np.full((1,8),100.),np.full((9,8),-1.)]
        r=summarize(x,x.mean(1),np.ones(8),.85)
        self.assertEqual(r['aggregate_bit_accuracy'],1);self.assertEqual(r['tpr'],.1)
        self.assertFalse(r['target90_point_estimate_met'])
    def test_point_and_ci_distinct(self):
        s=np.r_[np.ones(900),np.zeros(100)]
        r=summarize(np.repeat(s[:,None],8,axis=1),s,np.ones(8),.85)
        self.assertTrue(r['target90_point_estimate_met']);self.assertFalse(r['target90_ci95_lower_met'])
    def test_paired_no_change(self):
        r=paired_difference(np.ones(10),np.ones(10),repeats=20)
        self.assertEqual(r['tpr_difference_pp'],0);self.assertEqual(r['paired_bootstrap_ci95_pp'],[0,0])
    def test_wilson_boundaries(self):
        self.assertAlmostEqual(wilson(0,100)[0],0);self.assertAlmostEqual(wilson(100,100)[1],1)
        with self.assertRaises(ValueError):wilson(0,0)
    def test_hash_score_and_alignment_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'images'/'test';folder.mkdir(parents=True)
            file=folder/'sample_000000.png';file.write_bytes(b'fixture content')
            row=dict(file=file.name,sha256=sha256(file),coordinates=[1.]*8,score=1.)
            target=root/'scores/test.json';write_json(target,[row]);key={'signs':[1]*8}
            validated_rows(root,'test',1,key)
            bad=copy.deepcopy(row);bad['score']=2;write_json(target,[bad])
            with self.assertRaises(ValueError):validated_rows(root,'test',1,key)
            write_json(target,[row]);file.write_bytes(b'changed')
            with self.assertRaises(ValueError):validated_rows(root,'test',1,key)
    def test_stage_failure_has_no_complete_marker(self):
        import campaign
        with tempfile.TemporaryDirectory() as temp:
            q=Path(temp)
            with patch.object(campaign,'verify'),patch.object(campaign,'wait_resources'),\
                 patch.object(campaign.subprocess,'run',side_effect=RuntimeError('technical failure')):
                with self.assertRaises(RuntimeError):campaign.call(q,'example',Path('fake.py'))
            self.assertFalse((q/'transfer90/stages/example.json').exists())
    def test_checkpoint_selection_is_mse(self):
        from validation import choose_checkpoint
        self.assertEqual(choose_checkpoint([dict(step=10,denoising_mse=.2,tpr=1.),
            dict(step=20,denoising_mse=.1,tpr=0.)])['step'],20)
    def test_structure_identity_gradient(self):
        from structure_metrics import ms_ssim
        x=torch.rand(1,3,192,192,requires_grad=True)
        score=ms_ssim(x,x.detach());self.assertAlmostEqual(float(score),1.,places=4)
        score.sum().backward();self.assertTrue(torch.isfinite(x.grad).all())


class Checkpoint(unittest.TestCase):
    def test_compact_state_restores_optimizer_scheduler_and_rng(self):
        from student_driver import accelerator_class
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.base=torch.nn.Parameter(torch.ones(2),requires_grad=False)
                self.lora_weight=torch.nn.Parameter(torch.tensor([.2,.4]))
            def forward(self,x):return ((self.base+self.lora_weight)*x).sum()
        accelerator=accelerator_class()(cpu=True)
        model=Toy();optimizer=torch.optim.AdamW([model.lora_weight],lr=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,learning_rate_factor)
        model,optimizer,scheduler=accelerator.prepare(model,optimizer,scheduler)
        with tempfile.TemporaryDirectory() as temp:
            loss=model(torch.ones(2));accelerator.backward(loss);optimizer.step();scheduler.step();optimizer.zero_grad()
            expected=model.lora_weight.detach().clone();moment=optimizer.state[model.lora_weight]['exp_avg'].clone()
            count=scheduler.state_dict()['last_epoch'];rng=torch.get_rng_state().clone()
            accelerator.save_state(temp)
            self.assertTrue((Path(temp)/'trainable.safetensors').exists())
            self.assertFalse((Path(temp)/'model.safetensors').exists())
            self.assertTrue(list(Path(temp).glob('optimizer*')));self.assertTrue(list(Path(temp).glob('random_states*')))
            with torch.no_grad():model.lora_weight.fill_(99.)
            optimizer.state[model.lora_weight]['exp_avg'].zero_();torch.rand(20);scheduler.step()
            accelerator.load_state(temp)
            torch.testing.assert_close(model.lora_weight,expected)
            torch.testing.assert_close(optimizer.state[model.lora_weight]['exp_avg'],moment)
            self.assertEqual(scheduler.state_dict()['last_epoch'],count)
            # Accelerate restores RNG as well; verify the actual tensor state.
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        accelerator.free_memory()


if __name__=='__main__':unittest.main()
