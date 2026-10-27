import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from pareto_step import project_displacement, ProtectedAdamW, dot


class ProjectionTest(unittest.TestCase):
    def test_exact_known_projection_and_opposed_dependent_constraints(self):
        d=[torch.tensor([2.,3.,-4.],dtype=torch.float64)]
        gs=[[torch.tensor(v,dtype=torch.float64)] for v in [[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.]]]
        actual,_=project_displacement(d,gs)
        torch.testing.assert_close(actual[0],torch.tensor([0.,0.,-4.],dtype=torch.float64))

    def test_random_qp_agrees_with_independent_scipy_solver(self):
        from scipy.optimize import minimize
        rng=np.random.default_rng(42)
        for _ in range(35):
            proposed=rng.normal(size=7);a=rng.normal(size=(3,7))
            projected,_=project_displacement([torch.tensor(proposed)],[[torch.tensor(row)] for row in a])
            reference=minimize(lambda x:.5*np.square(x-proposed).sum(),np.zeros(7),
                jac=lambda x:x-proposed,constraints=[{'type':'ineq','fun':lambda x:-a@x,'jac':lambda x:-a}],
                method='SLSQP',options={'ftol':1e-12,'maxiter':200})
            self.assertTrue(reference.success)
            np.testing.assert_allclose(projected[0].numpy(),reference.x,atol=1e-6)

    def test_zero_nonfinite_and_feasible_update(self):
        d=[torch.tensor([-1.,-2.])];g=[[torch.tensor([1.,0.])],[torch.zeros(2)]]
        result,_=project_displacement(d,g);torch.testing.assert_close(result[0],d[0])
        with self.assertRaises(RuntimeError):project_displacement([torch.tensor([float('nan')])],[])
        with self.assertRaises(RuntimeError):project_displacement([torch.ones(1)],[[torch.tensor([float('inf')])]])

    def test_projects_actual_adam_momentum_and_weight_decay_step(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
        optimizer=ProtectedAdamW([p],lr=.1,weight_decay=.1)
        # Populate moments so protecting just the current gradient is insufficient.
        p.grad=torch.tensor([-10.,-1.],dtype=torch.float64);optimizer.step()
        before=p.detach().clone();g=[torch.tensor([1.,0.],dtype=torch.float64)]
        p.grad=torch.tensor([.01,.01],dtype=torch.float64);optimizer.constraints=[g]
        optimizer.step()
        self.assertGreater(optimizer.last_projection['proposal_dots'][0],0.)
        self.assertLessEqual(float(dot(g,[p-before])),1e-10)

    def test_optimizer_resume_preserves_next_projected_step(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
        opt=ProtectedAdamW([p],lr=.01)
        p.grad=torch.tensor([1.,-2.],dtype=torch.float64);opt.step()
        state=copy.deepcopy(opt.state_dict());q=torch.nn.Parameter(p.detach().clone())
        restored=ProtectedAdamW([q],lr=.01);restored.load_state_dict(state)
        for param,optim in [(p,opt),(q,restored)]:
            param.grad=torch.tensor([-.3,2.],dtype=torch.float64)
            optim.constraints=[[torch.tensor([1.,0.],dtype=torch.float64)]];optim.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)


class SamplerTest(unittest.TestCase):
    def test_tail_preserves_full_forward_and_pndm_history(self):
        from diffusers import PNDMScheduler
        from pareto_quality_refine import sample_tail
        from native_quality_refine import sample_native
        class Denoiser(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor(.02));self.modes=[]
            def forward(self,x,t,encoder_hidden_states):
                self.modes.append(torch.is_grad_enabled())
                return SimpleNamespace(sample=torch.tanh(x)*self.w+encoder_hidden_states[:,0,0,None,None,None])
        def scheduler():return PNDMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,
            beta_schedule='scaled_linear',skip_prk_steps=True,steps_offset=1,set_alpha_to_one=False)
        model=Denoiser();torch.manual_seed(8);noise=torch.randn(1,4,4,4);text=torch.zeros(2,1,1);text[1]=.001
        full=sample_native(model,scheduler(),noise,text,checkpointed=False)
        model.modes=[];tail=sample_tail(model,scheduler(),noise,text,backward_calls=10,checkpointed=False)
        self.assertEqual(model.modes,[False]*41+[True]*10)
        torch.testing.assert_close(full,tail,rtol=0,atol=0)
        grad=torch.autograd.grad(tail.square().mean(),model.w)[0]
        saved=sample_tail(model,scheduler(),noise,text,backward_calls=10,checkpointed=True)
        torch.testing.assert_close(torch.autograd.grad(saved.square().mean(),model.w)[0],grad)
        self.assertTrue(torch.isfinite(grad));self.assertGreater(float(grad.abs()),0.)
        all_calls=sample_tail(model,scheduler(),noise,text,backward_calls=51,checkpointed=False)
        torch.testing.assert_close(torch.autograd.grad(all_calls.square().mean(),model.w)[0],
                                   torch.autograd.grad(full.square().mean(),model.w)[0])

    def test_balance_caps_tiny_gradients_and_ablation_is_single_switch(self):
        from pareto_quality_refine import balanced
        from pareto_quality_campaign import SPECS,ORDER
        scale,stats=balanced([torch.tensor([1e-8])],1.,1.,100.)
        self.assertEqual(scale,100.);self.assertLess(stats['added_norm'],1.)
        self.assertEqual(balanced([torch.zeros(2)],1.,1.,100.)[0],0.)
        a,b=[SPECS[n] for n in ORDER]
        self.assertEqual({k:v for k,v in a.items() if k!='project_step'},
                         {k:v for k,v in b.items() if k!='project_step'})
        self.assertTrue(a['project_step']);self.assertFalse(b['project_step'])
        self.assertEqual(a['native_every'],2);self.assertEqual(a['teacher_steps'],600)


if __name__=='__main__':unittest.main()
