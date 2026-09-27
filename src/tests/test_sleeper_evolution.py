import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from wmq_evolution import fronts, ranked, search
from wmq_sleeper_evolution import GeneticWeight, split_records, tail_diagnostics
from wmq_grouped_quant import GroupedW4
from wmq_sleepermark import parser, train_branch, selected_weights, state_hash


class EvolutionTests(unittest.TestCase):
    def test_pareto_layers_and_tradeoff_extremes(self):
        values = [[0, 3], [1, 2], [3, 0], [4, 4], [1, 2]]
        self.assertEqual(fronts(values), [[0, 1, 2, 4], [3]])
        self.assertEqual(set(ranked(values)[:2]), {0, 2})

    def test_reproducible_budget_bounds_and_matched_initial_population(self):
        def objective(x):
            return [np.square(x-1).sum(), np.square(x+2).sum()]
        a, pareto = search(objective, 6, 8, 4, seed=17)
        b, _ = search(objective, 6, 8, 4, seed=17)
        control, _ = search(objective, 6, 8, 4, seed=17, random=True)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 40)
        self.assertEqual(len({tuple(r['gene']) for r in a}), 40)
        self.assertEqual(len({tuple(r['gene']) for r in control}), 40)
        self.assertEqual(a[:8], control[:8])
        self.assertNotEqual(a[8:], control[8:])
        self.assertTrue(all(-3 <= g <= 3 for row in a for g in row['gene']))
        self.assertEqual(pareto, fronts([row['fitness'] for row in a])[0])
        with self.assertRaises(ValueError):
            search(lambda x: [float('nan'), 0], 2, 4, 1)

    def test_unique_budget_near_exhaustion(self):
        records, _ = search(lambda x: [float((x*x).sum()), float(x.sum())], 2, 8, 5)
        self.assertEqual(len({tuple(r['gene']) for r in records}), 48)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            search(lambda x: [0, 0], 1, 4, 1)

    def test_tail_guard_detects_damage_hidden_by_mean(self):
        reference = np.ones(16)
        damaged = np.zeros(16)
        damaged[0] = 8  # mean .5 looks better than RTN, but one row is much worse.
        self.assertLess(damaged.mean(), reference.mean())
        report = tail_diagnostics(damaged, reference, 2.)
        self.assertEqual(report['ordinary_ratio_max'], 8.)
        self.assertEqual(report['ordinary_tail_violation_fraction'], 1/16)
        self.assertEqual(tail_diagnostics(reference, reference, 2.)['ordinary_ratio_max'], 1.)
        self.assertTrue(np.isfinite(list(tail_diagnostics([0, 1e-15], [0, 0], 2.).values())).all())

    def test_grid_anchor_and_changed_genome_stay_integer_w4(self):
        torch.manual_seed(3)
        weight = torch.randn(7, 13)
        adapter = GeneticWeight(weight, 8, 'mse')
        adapter.enabled = True
        adapter.set_gene(weight, [0, 0])
        torch.testing.assert_close(adapter(weight), GroupedW4(weight, 8, 'mse')(False), rtol=0, atol=0)
        for gene in ([3, -3], [-3, 3]):
            adapter.set_gene(weight, gene)
            result = adapter(weight).reshape(7, -1)
            scales = (adapter.base_scale*float(np.exp(.04*gene[0]))).expand(-1, -1, 8).reshape(7, -1)[:, :13]
            codes = result/scales
            torch.testing.assert_close(codes, codes.round(), rtol=1e-6, atol=1e-6)
            self.assertTrue(bool(((codes >= -8.000001) & (codes <= 7.000001)).all()))

    def test_prompt_disjoint_bank(self):
        data = [(None, None, f'p{i}', t) for i in range(6) for t in (1, 3, 9)]
        fit, select = split_records(data, 8, 1)
        self.assertFalse({data[i][2] for i in fit} & {data[i][2] for i in select})
        self.assertEqual([fit, select], split_records(data, 8, 1))
        self.assertEqual({data[i][3] for i in fit}, {1, 3, 9})

    def test_real_diffusers_export_restoration_and_failure_cleanup(self):
        from diffusers import UNet2DConditionModel, DDPMScheduler
        torch.set_num_threads(2)
        torch.manual_seed(7)
        unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
            up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), block_out_channels=(16, 32),
            layers_per_block=1, norm_num_groups=8, cross_attention_dim=8, attention_head_dim=4).requires_grad_(False)
        names = selected_weights(unet, 'up_attentions')
        before = state_hash(unet)
        data = [(torch.randn(1, 4, 16, 16), torch.randn(1, 3, 8), f'prompt{i}', 3) for i in range(4)]
        args = parser().parse_args(['--evolution-population', '4', '--evolution-generations', '1',
                                   '--evolution-records', '2', '--spatial-shift', '1'])
        scheduler = DDPMScheduler(num_train_timesteps=10)
        with tempfile.TemporaryDirectory() as tmp, patch('wmq_sleepermark.encode_text',
                side_effect=lambda pipe, prompts: torch.zeros(len(prompts), 3, 8)):
            for method in ('genetic_quantizer_w4', 'random_quantizer_w4'):
                result = train_branch(SimpleNamespace(unet=unet), scheduler, data, names, method, args, Path(tmp))
                self.assertEqual(before, state_hash(unet))
                self.assertEqual(set(result[method]), set(names))
                self.assertTrue(all(torch.isfinite(v).all() for v in result[method].values()))
                report = json.loads((Path(tmp)/(method+'_search.json')).read_text())
                self.assertEqual(len(report['records']), 8)
                self.assertFalse(set(report['fit_prompt_names']) & set(report['select_prompt_names']))
                self.assertFalse(report['test_used_for_selection'])
                self.assertLessEqual(report['selected']['ordinary_loss'], report['ordinary_loss_limit'])
                self.assertLessEqual(report['selected']['ordinary_ratio_max'], report['per_record_ratio_limit'])
                self.assertEqual(report['unique_genomes_evaluated'], 8)
            with patch('wmq_sleeper_evolution.search', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    train_branch(SimpleNamespace(unet=unet), scheduler, data, names,
                                 'genetic_quantizer_w4', args, Path(tmp))
            self.assertEqual(before, state_hash(unet))


if __name__ == '__main__':
    unittest.main()
