import json
import unittest

from wmq_llm_proposals import fit_context, parse_proposals


class LLMProposalTests(unittest.TestCase):
    def test_context_whitelists_fit_fields(self):
        archive = [{'gene': [0, 0], 'fitness': [1., 2.], 'secret': 'must-not-leak'},
                   {'gene': [1, -1], 'fitness': [2., 1.], 'owner_score': .5}]
        context = fit_context(['g'], archive, 2, 3)
        serialized = json.dumps(context)
        self.assertNotIn('secret', serialized)
        self.assertNotIn('owner_score', serialized)
        self.assertEqual(set(context['fit_archive'][0]), {'id', 'gene', 'fitness'})

    def test_strict_json_and_bounds(self):
        self.assertEqual(parse_proposals('{"genes":[[0,1],[-3,3]]}', 2, 3), [[0, 1], [-3, 3]])
        self.assertEqual(parse_proposals('```json\n{"genes":[[1,2]]}\n```', 2, 3), [[1, 2]])
        for invalid in ('{"genes":[[4,0]]}', '{"genes":[[1.0,0]]}',
                        '{"genes":[[0,0]],"code":"x"}', '{"genes":[]}'):
            with self.assertRaises(ValueError):
                parse_proposals(invalid, 2, 3)


if __name__ == '__main__':
    unittest.main()
