import unittest
from types import SimpleNamespace
import torch
from native_quality_refine import sample_native


class SamplerTest(unittest.TestCase):
    def test_original_pndm_forward_and_full_gradient_checkpoint_equivalence(self):
        from diffusers import PNDMScheduler
        class Denoiser(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.02))
            def forward(self,x,t,encoder_hidden_states):
                # All steps depend on trainable weight and the actual latent.
                return SimpleNamespace(sample=torch.tanh(x)*self.weight+encoder_hidden_states[:,0,0,None,None,None])
        def scheduler():
            return PNDMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,
                beta_schedule='scaled_linear',skip_prk_steps=True,steps_offset=1,set_alpha_to_one=False)
        torch.manual_seed(4);noise=torch.randn(1,4,4,4);hidden=torch.zeros(2,1,1)
        hidden[1]=.001
        model=Denoiser()
        full=sample_native(model,scheduler(),noise,hidden,steps=50,checkpointed=False)
        expected_gradient=torch.autograd.grad(full.square().mean(),model.weight)[0]
        saved=sample_native(model,scheduler(),noise,hidden,steps=50,checkpointed=True)
        actual_gradient=torch.autograd.grad(saved.square().mean(),model.weight)[0]
        torch.testing.assert_close(saved,full,rtol=0,atol=0)
        torch.testing.assert_close(actual_gradient,expected_gradient,rtol=1e-5,atol=1e-5)
        self.assertTrue(torch.isfinite(actual_gradient));self.assertGreater(float(actual_gradient.abs()),0.)
        # Independent ordinary sampler loop for forward correctness.
        native=scheduler();native.set_timesteps(50);z=noise*native.init_noise_sigma
        for t in native.timesteps:
            u,c=model(native.scale_model_input(torch.cat([z,z]),t),t,hidden).sample.chunk(2)
            z=native.step(u+7.5*(c-u),t,z).prev_sample
        torch.testing.assert_close(saved,z,rtol=0,atol=0)
        self.assertEqual(len(native.timesteps),51)

    def test_scheduler_state_not_reused_and_repeated_sample_reproducible(self):
        from diffusers import PNDMScheduler
        class Denoiser(torch.nn.Module):
            def forward(self,x,t,encoder_hidden_states):return SimpleNamespace(sample=x*.01)
        noise=torch.ones(1,4,4,4);text=torch.zeros(2,1,1)
        values=[sample_native(Denoiser(),PNDMScheduler(skip_prk_steps=True),noise,text,50,False) for _ in range(2)]
        torch.testing.assert_close(*values,rtol=0,atol=0)

    def test_half_noise_promoted_to_float_scheduler_state(self):
        from diffusers import PNDMScheduler
        class Denoiser(torch.nn.Module):
            def forward(self,x,t,encoder_hidden_states):return SimpleNamespace(sample=x*.01)
        output=sample_native(Denoiser(),PNDMScheduler(skip_prk_steps=True),
            torch.ones(1,4,4,4,dtype=torch.float16),torch.zeros(2,1,1),50,False)
        self.assertEqual(output.dtype,torch.float32)
        self.assertTrue(torch.isfinite(output).all())

    def test_ablation_specs_keep_original_student_and_native_sampler(self):
        from native_quality_campaign import SPECS,ORDER
        self.assertEqual(ORDER,['clean_control','native_full','continuation'])
        a,b=SPECS['native_full'],SPECS['continuation']
        self.assertEqual({k:v for k,v in a.items() if k!='native'},
                         {k:v for k,v in b.items() if k!='native'})
        self.assertTrue(a['native']);self.assertFalse(b['native'])
        self.assertEqual(a['teacher_steps'],600);self.assertEqual(a['aux_gradient_fraction'],.25)
        self.assertFalse(a['transfer'])

    def test_color_diagnostic_does_not_claim_quality_score(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from native_quality_audit import pixel_statistics
        with tempfile.TemporaryDirectory() as folder:
            Image.new('RGB',(4,4),(255,0,0)).save(Path(folder)/'sample_000000.png')
            result=pixel_statistics(Path(folder),1)
        self.assertEqual(result['clipped_channel_fraction_mean'],1.)
        self.assertEqual(result['mean_hsv_saturation'],1.)
        self.assertIn('do not measure',result['interpretation'])


if __name__=='__main__':unittest.main()
