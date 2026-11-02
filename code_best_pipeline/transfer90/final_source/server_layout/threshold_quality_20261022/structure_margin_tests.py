import copy
import unittest
import numpy as np
import torch
from structure_margin_objectives import ssim_components, ms_ssim, margin_penalty
from pareto_step import ProtectedAdamW


class ObjectivesTest(unittest.TestCase):
    def test_ssim_matches_independent_skimage(self):
        from skimage.metrics import structural_similarity
        rng=np.random.default_rng(731)
        a=rng.uniform(size=(2,3,64,64))
        b=np.clip(a+rng.normal(0,.15,a.shape),0,1)
        measured,_=ssim_components(torch.tensor(a),torch.tensor(b))
        expected=np.array([[structural_similarity(a[i,j],b[i,j],data_range=1.,
            gaussian_weights=True,sigma=1.5,use_sample_covariance=False) for j in range(3)] for i in range(2)])
        np.testing.assert_allclose(measured.numpy(),expected,atol=1e-12)

    def test_multiscale_identity_finite_gradient_and_geometric_damage(self):
        torch.manual_seed(17)
        x=torch.rand(2,3,192,192)
        torch.testing.assert_close(ms_ssim(x,x),torch.ones(2),atol=1e-6,rtol=1e-6)
        y=(.8*x+.1).requires_grad_()
        score=ms_ssim(y,x)
        gradient=torch.autograd.grad(score.sum(),y)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max(),0)
        self.assertTrue((ms_ssim(x.flip(-1),x)<score.detach()).all())
        flat=torch.zeros(1,3,192,192,requires_grad=True)
        g=torch.autograd.grad(ms_ssim(flat,torch.ones_like(flat)).sum(),flat)[0]
        self.assertTrue(torch.isfinite(g).all())

    def test_individual_shortfalls_do_not_cancel(self):
        scores=torch.tensor([2.,0.,.5,1.],requires_grad=True)
        targets=torch.ones(4)
        losses=margin_penalty(scores,targets,.5)
        torch.testing.assert_close(losses,torch.tensor([0.,4.,1.,0.]))
        grad=torch.autograd.grad(losses.mean(),scores)[0]
        torch.testing.assert_close(grad,torch.tensor([0.,-2.,-1.,0.]))
        self.assertGreater(float(losses.mean()),0)

    def test_microbatch_accumulation_matches_full_gradient(self):
        theta=torch.tensor([.1,.4],requires_grad=True)
        x=torch.tensor([[1.,2.],[-1.,3.],[2.,1.],[3.,-.5]])
        targets=torch.ones(4)
        full=margin_penalty(x@theta,targets,.4).mean()+(x@theta).square().mean()
        gf=torch.autograd.grad(full,theta)[0]
        accumulated=torch.zeros_like(theta)
        for start in [0,2]:
            value=x[start:start+2]@theta
            local=(margin_penalty(value,targets[start:start+2],.4).sum()+value.square().sum())/4
            accumulated+=torch.autograd.grad(local,theta)[0]
        torch.testing.assert_close(gf,accumulated)

    def test_satisfied_margin_is_zero_and_does_not_block_projection(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        opt=ProtectedAdamW([p],lr=.01)
        p.grad=torch.ones_like(p)
        opt.constraints=[[torch.ones_like(p)],[torch.zeros_like(p)],[torch.ones_like(p)]]
        opt.step()
        self.assertGreater(opt.last_projection['realized_norm'],0)

    def test_ablation_and_protocol_are_frozen(self):
        from structure_margin_campaign import SPECS,ORDER
        a,b=[SPECS[n] for n in ORDER]
        self.assertEqual({k:v for k,v in a.items() if k!='structure_weight'},
                         {k:v for k,v in b.items() if k!='structure_weight'})
        self.assertEqual((a['structure_weight'],b['structure_weight']),(.25,0.))
        self.assertEqual(a['native_batch'],4)
        self.assertEqual(a['teacher_steps'],600)
        self.assertEqual(a['backward_calls'],10)

    def test_batched_native_sampler_preserves_cfg_prompt_pairing(self):
        from types import SimpleNamespace
        from diffusers import PNDMScheduler
        from pareto_quality_refine import sample_tail
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor(.02))
            def forward(self,x,t,encoder_hidden_states):
                return SimpleNamespace(sample=x.tanh()*self.w+encoder_hidden_states[:,0,0,None,None,None])
        def scheduler():
            return PNDMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,
                beta_schedule='scaled_linear',skip_prk_steps=True,steps_offset=1,set_alpha_to_one=False)
        torch.manual_seed(14);noise=torch.randn(2,4,4,4)
        hidden=torch.tensor([0.,0.,.001,-.002]).reshape(4,1,1)
        model=Model()
        batch=sample_tail(model,scheduler(),noise,hidden,checkpointed=False)
        individual=torch.cat([sample_tail(model,scheduler(),noise[i:i+1],hidden[[i,i+2]],
            checkpointed=False) for i in range(2)])
        torch.testing.assert_close(batch,individual,rtol=1e-6,atol=1e-6)
        g1=torch.autograd.grad(batch.square().mean(),model.w)[0]
        g2=torch.autograd.grad(individual.square().mean(),model.w)[0]
        torch.testing.assert_close(g1,g2,rtol=1e-5,atol=1e-6)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
