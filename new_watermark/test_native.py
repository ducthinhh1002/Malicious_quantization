"""Native trajectory, fixed reader and calibration checks without downloads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paths
import unittest
import tempfile
from unittest.mock import patch


class Native(unittest.TestCase):
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
