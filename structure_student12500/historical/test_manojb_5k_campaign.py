import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import manojb_5k_campaign as m
from common import write_json, read_json


def row(tag, lpips, scores, margin):
    return {'tag': tag, 'lpips': lpips, 'clip_drop': .01, 'scores': scores,
            'margin_p10': margin, 'vector': [.85] * 6}


class Campaign(unittest.TestCase):
    def test_better_quality_cannot_hide_lost_detections_or_margin(self):
        base = row('base', .6, [1., 1., .2], .3)
        lost = row('lost', .1, [.8, .3, .2], .3)
        weak = row('weak', .2, [.8, .8, .2], .1)
        good = row('good', .4, [.9, .9, .2], .28)
        decision = m.choose_evolution([lost, weak, good], base, .5)
        self.assertEqual(decision['selected']['tag'], 'good')
        self.assertFalse(decision['candidates'][0]['detection_preserved'])
        self.assertFalse(decision['candidates'][1]['detection_preserved'])

    def test_failed_constraint_is_explicit_and_does_not_abort(self):
        base = row('base', .6, [1., 1.], .3)
        candidate = row('bad', .4, [1., .1], .1)
        selected = m.choose_evolution([candidate], base, .5)['selected']
        self.assertFalse(selected['detection_preserved'])
        self.assertEqual(selected['lost_positive_images'], 1)

    def test_proposals_reproducible_unique_and_conservative(self):
        rows = [row('a', .5, [1.], .2), row('b', .6, [1.], .4)]
        rows[1]['vector'] = [1.] * 6
        seen = {tuple(r['vector']) for r in rows}
        for mode in (False, True):
            for seed in range(40):
                v = m.propose(rows, seen, seed, random=mode)
                self.assertEqual(v, m.propose(rows, seen, seed, random=mode))
                self.assertNotIn(tuple(v), seen)
                self.assertTrue(set(v) <= set(m.LEVELS))

    def test_original_split_disjoint_and_old_calibration_overlap_detectable(self):
        from assets import split_prompts
        annotations = [{'image_id': i, 'id': i, 'caption': f'prompt {i}'} for i in range(13000)]
        old = split_prompts(annotations, 0, {'train': 1000, 'calibration': 5000, 'test': 1000, 'validation': 256})
        new = split_prompts(annotations, 0, {'train': 5000, 'calibration': 5000, 'test': 1000, 'validation': 1000})
        ids = [{r['image_id'] for r in v} for v in new.values()]
        self.assertEqual(len(set.union(*ids)), 12000)
        self.assertTrue({r['image_id'] for r in old['calibration']} & ids[0])

    def test_order_has_no_early_scale1_dependency(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(m.q, 'QUEUE', root), patch.object(m, 'verify_code'), patch.object(m, 'prepare_source'), \
                 patch.object(m, 'evolution', return_value=[.9] * 6), patch.object(m, 'finish_comparisons'), \
                 patch.object(m, 'own', side_effect=lambda run, label, operation, vector: calls.append((label, vector))):
                m.supervise()
            self.assertEqual([r[0] for r in calls], list(m.ORDER))
            self.assertEqual(read_json(root / 'status.json')['state'], 'completed')


if __name__ == '__main__':
    unittest.main()
