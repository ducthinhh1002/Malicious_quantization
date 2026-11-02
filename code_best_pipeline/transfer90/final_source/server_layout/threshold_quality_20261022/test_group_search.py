import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import group_search as gs
import research_queue as rq


class Algorithms(unittest.TestCase):
    def test_full_coalition_shapley_with_interaction(self):
        masks=gs.coalitions(3)
        values=[7+2*a+3*b-c+6*a*b for a,b,c in masks]
        fit=gs.kernel_shap(masks,values)
        np.testing.assert_allclose(fit['phi'],[5,6,-1],atol=1e-10)
        self.assertAlmostEqual(sum(fit['phi']),fit['full']-fit['empty'])

    def test_missing_coalition_rejected(self):
        with self.assertRaises(ValueError): gs.kernel_shap([[0,0],[1,1]],[0,1])

    def test_uniform_scale_matches_exact_delta_linear_and_conv(self):
        state={}
        for module,a_shape,b_shape in [
            ('down_blocks.0.attentions.0.to_q',(2,4),(4,2)),
            ('up_blocks.0.resnets.0.conv1',(2,4,3,3),(4,2,1,1))]:
            state[module+'.lora_A.weight']=torch.randn(a_shape)
            state[module+'.lora_B.weight']=torch.randn(b_shape)
        groups=sorted({gs.group_for_key(k) for k in state})
        out=gs.scale_state(state,groups,[.65]*len(groups))
        for k,v in state.items():
            if '.lora_A.' in k: self.assertTrue(torch.equal(out[k],v))
            else: torch.testing.assert_close(out[k],.65*v)
        for module in ('down_blocks.0.attentions.0.to_q','up_blocks.0.resnets.0.conv1'):
            a,b=state[module+'.lora_A.weight'],state[module+'.lora_B.weight']
            scaled=out[module+'.lora_B.weight'].reshape(4,2)@out[module+'.lora_A.weight'].reshape(2,-1)
            torch.testing.assert_close(scaled,.65*(b.reshape(4,2)@a.reshape(2,-1)))

    def test_unmapped_targets_and_nonfinite_are_rejected(self):
        self.assertEqual(gs.group_for_key('conv_in.lora_A.weight'),'down_conv')
        self.assertEqual(gs.group_for_key('conv_out.lora_B.weight'),'up_conv')
        with self.assertRaises(ValueError): gs.group_for_key('other.lora_A.weight')
        s={'mid_block.conv1.lora_A.weight':torch.ones(1,1),
           'mid_block.conv1.lora_B.weight':torch.tensor([[float('nan')]])}
        with self.assertRaises(ValueError): gs.scale_state(s,['mid_conv'],[.85])

    def test_pareto_preserves_quality_margin_tradeoff(self):
        def r(tag,lp,margin):return {'tag':tag,'lpips':lp,'clip_drop':0.,'margin_p10':margin,'vector':[.65,.85]}
        rows=[r('a',.1,.1),r('b',.2,.3),r('dominated',.3,0.)]
        self.assertEqual({r['tag'] for r in gs.pareto_front(rows)},{'a','b'})
        self.assertEqual(len(gs.grid_elites(rows)),2)

    def test_constant_grid_and_reproducible_unique_search(self):
        row={'tag':'a','lpips':1.,'clip_drop':0.,'margin_p10':.2,'vector':[.65,.65]}
        self.assertEqual(len(gs.grid_elites([row])),1)
        for mode in ('pareto','random'):
            seen={(.65,.65)}
            v=gs.proposal([row],seen,2,42,mode)
            self.assertEqual(v,gs.proposal([row],seen,2,42,mode))
            self.assertNotIn(tuple(v),seen)
            self.assertTrue(set(v)<=set([.65,.70,.75,.85,1.]))


class QueueIntegrity(unittest.TestCase):
    def test_only_owned_derived_exports_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'queue'; folder=root/'pilot/derived/x/merged_pipeline'
            folder.mkdir(parents=True)
            with patch.object(rq,'QUEUE',root):
                with self.assertRaises(ValueError): rq.safe_remove_derived(root/'pilot')
                with self.assertRaises(ValueError): rq.safe_remove_derived(Path(tmp)/'merged_pipeline')
                rq.safe_remove_derived(folder)
            self.assertFalse(folder.exists())

    def test_hardlink_preserves_source_and_checks_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=root/'a'; b=root/'nested/b'; a.write_bytes(b'checkpoint')
            rq.link_file(a,b); self.assertEqual(b.read_bytes(),b'checkpoint')
            b.unlink(); b.write_bytes(b'changed')
            with self.assertRaises(RuntimeError): rq.link_file(a,b)
            self.assertEqual(a.read_bytes(),b'checkpoint')

    def test_student_keeps_original_training_protocol(self):
        from student import training_command,branches
        cfg={'threshold_quality':False,'watermark_method':'semantic_fp','train_clean_controls':False,
             'adapter_scopes':['attention_conv'],'ranks':[32],'lora_only':True,'paper_full_ema':False,
             'batch_size':4,'learning_rate':1e-4,'precision':'fp16','seed':0,'max_train_steps':7500,
             'checkpoint_steps':[2500,5000,7500]}
        self.assertEqual([b['name'] for b in branches(cfg)],[rq.BRANCH])
        command=training_command(cfg,{'student':'/fixed/sd14'},Path('/run'),branches(cfg)[0])
        for key,value in [('--max_train_steps','7500'),('--rank','32'),('--adapter_scope','attention_conv'),
                          ('--train_batch_size','4'),('--mixed_precision','fp16')]:
            self.assertEqual(command[command.index(key)+1],value)
        self.assertEqual(command[command.index('--snapshot_steps')+1:],['2500','5000','7500'])

    def test_student_pipeline_selects_before_test_and_measures_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); calls=[]
            def own(*args): calls.append(args[2])
            def harness(*args): calls.append(args[1])
            with patch.object(rq,'prepare_run',return_value=run),patch.object(rq,'freeze'),\
                 patch.object(rq,'own',side_effect=own),patch.object(rq,'harness',side_effect=harness),\
                 patch.object(rq,'read_json',return_value={'checkpoint_steps':[2500,5000,7500]}):
                rq.train_student('scale065',[.65]*6,{'method':'fixed'})
            self.assertLess(calls.index('07_select_mse'),calls.index('08_student_test'))
            self.assertIn('10_test_quality',calls)
            self.assertIn('student_report',calls)


if __name__=='__main__': unittest.main()
