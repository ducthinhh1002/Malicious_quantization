import unittest
import torch
from threshold_quality import combine_states, paired_gate


class QualityTests(unittest.TestCase):
    def test_quality_failures_do_not_block_detection_preserving_candidate(self):
        from threshold_quality import detection_eligible
        row={'paired_gate':{'passes':False,'checks':{'baseline_tpr_at_least_99pct':True,'no_lost_positive_images':True,
             'clip_drop_at_most_001':False,'lpips_improves_at_least_2pct':False}}}
        self.assertTrue(detection_eligible(row))
        row['paired_gate']['checks']['no_lost_positive_images']=False
        self.assertFalse(detection_eligible(row))
    def test_linear_delta_is_exact_085_not_squared(self):
        torch.manual_seed(4)
        a=torch.randn(3,5);b=torch.randn(7,3)
        state={'layer.lora_A.weight':a,'layer.lora_B.weight':b}
        merged=combine_states([state],[1.])
        torch.testing.assert_close(merged['layer.lora_A.weight'],a)
        torch.testing.assert_close(merged['layer.lora_B.weight']@merged['layer.lora_A.weight'],.85*(b@a))

    def test_conv_mixture_exact_in_forward_and_peft_merge(self):
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.conv=torch.nn.Conv2d(4,6,3,padding=1,bias=False)
            def forward(self,x):return self.conv(x)
        torch.manual_seed(7)
        states=[]
        for _ in range(2):
            states.append({'base_model.model.conv.lora_A.weight':torch.randn(2,4,3,3),
                           'base_model.model.conv.lora_B.weight':torch.randn(6,2,1,1)})
        mixed=combine_states(states,[.5,.5]);x=torch.randn(2,4,9,9)
        net=Net();clean=net(x).detach()
        net=get_peft_model(net,LoraConfig(r=4,lora_alpha=4,target_modules=['conv']))
        set_peft_model_state_dict(net,mixed)
        expected=clean
        for s in states:
            expected=expected+.425*torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,s['base_model.model.conv.lora_A.weight'],padding=1),s['base_model.model.conv.lora_B.weight'])
        torch.testing.assert_close(net(x),expected,atol=1e-5,rtol=1e-5)
        net=net.merge_and_unload()
        torch.testing.assert_close(net(x),expected,atol=1e-5,rtol=1e-5)

    def quality(self,lp=.5,clip=.009):
        return {'lpips_per_image':[lp]*256,'paired_lpips':lp,'clip_drop':clip}

    def test_gate_accepts_joint_improvement(self):
        self.assertTrue(paired_gate(self.quality(),self.quality(.4,.008),[2.]*256,[1.8]*256,1.)['passes'])

    def test_gate_rejects_detection_loss_and_equal_tpr_swap(self):
        base=[2.]*256;base[0]=0.
        cand=[2.]*256;cand[1]=0.
        gate=paired_gate(self.quality(),self.quality(.4,.008),base,cand,1.)
        self.assertFalse(gate['passes']);self.assertEqual(gate['lost_positives'],1)

    def test_gate_rejects_quality_regressions_and_weak_baseline(self):
        for q,bs in [(self.quality(),[2.]*256),(self.quality(.4,.0095),[2.]*256),(self.quality(.4,.008),[0.]*256)]:
            self.assertFalse(paired_gate(self.quality(),q,bs,[2.]*256,1.)['passes'])

    def test_nonfinite_and_target_mismatch_fail_closed(self):
        with self.assertRaises(ValueError):paired_gate(self.quality(),self.quality(float('nan')),[2.]*256,[2.]*256,1.)
        with self.assertRaises(ValueError):combine_states([{'weight':torch.ones(2)}],[1.])

    def test_matched_student_conditions(self):
        from student import branches,training_conditions
        c={'threshold_quality':True,'ranks':[32],'adapter_scopes':['attention_conv'],'lora_only':True,'paper_full_ema':False}
        self.assertEqual(training_conditions(c),('baseline','wm'))
        self.assertEqual([b['name'] for b in branches(c)],['lora_r32_conv_baseline','lora_r32_conv_wm'])

if __name__=='__main__':unittest.main()
