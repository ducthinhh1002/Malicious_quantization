import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import scale085_recovery as recovery


class Recovery(unittest.TestCase):
    def test_original_trainer_changes_only_budget_for_exposure_trial(self):
        from student import training_command, branches
        c = dict(threshold_quality=False, watermark_method='semantic_fp', train_clean_controls=False,
                 adapter_scopes=['attention_conv'], ranks=[32], lora_only=True, paper_full_ema=False,
                 batch_size=4, learning_rate=1e-4, precision='fp16', seed=0, max_train_steps=7500,
                 checkpoint_steps=[2500, 5000, 7500])
        b = branches(c)[0]
        before = training_command(c, {'student': '/sd14'}, Path('/run'), b)
        after = training_command({**c, 'max_train_steps': 22500, 'checkpoint_steps': [7500,15000,22500]},
                                 {'student': '/sd14'}, Path('/run'), b)
        before[before.index('--max_train_steps')+1] = '22500'
        before[before.index('--snapshot_steps')+1:] = ['7500','15000','22500']
        self.assertEqual(before, after)

    def test_fixed_teacher_trial_keeps_four_groups_at085(self):
        from group_search import scale_state
        import torch
        groups = ['down_attention','down_conv','mid_attention','mid_conv','up_attention','up_conv']
        s = recovery.SPECS['scale085_conv_restore']
        self.assertEqual(dict(zip(groups,s['vector'])), dict(down_attention=.85, down_conv=1.,
                         mid_attention=.85, mid_conv=.85, up_attention=.85, up_conv=.9))
        self.assertEqual(s['steps'],7500)
        state = {'down_blocks.0.resnets.0.conv1.lora_A.weight':torch.ones(2,3),
                 'down_blocks.0.resnets.0.conv1.lora_B.weight':torch.ones(3,2)}
        out = scale_state(state, ['down_conv'], [1.])
        for key in state:
            self.assertTrue(torch.equal(state[key], out[key]))

    def test_reused_data_skips_generation_and_selection_precedes_test(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(recovery,'prepare',return_value=Path(folder)), \
                 patch.object(recovery.q,'harness',side_effect=lambda run,label,*a:calls.append(label)), \
                 patch.object(recovery,'own',side_effect=lambda run,label,*a:calls.append(label)):
                recovery.train('scale085_long22500')
        self.assertNotIn('01_data', calls)
        self.assertLess(calls.index('07_select'),calls.index('08_test'))
        self.assertIn('11_summary',calls)


if __name__ == '__main__':
    unittest.main()
