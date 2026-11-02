import unittest
from channel_transfer_audit import paired_summary, validation_indices


class AuditTest(unittest.TestCase):
    def test_fixed_validation_subset(self):
        ids = validation_indices(1000, 64, 20261024)
        self.assertEqual(ids, validation_indices(1000, 64, 20261024))
        self.assertEqual(len(set(ids)), 64)
        self.assertEqual(ids, sorted(ids))
        with self.assertRaises(ValueError):
            validation_indices(10, 11, 0)

    def test_paired_signal_not_raw_marked_mean(self):
        result = paired_summary([1., 2., 3.], [1.5, 2.5, 3.5], 2.)
        self.assertAlmostEqual(result['paired_signal'], .5)
        self.assertEqual(result['paired_signal_ci95'], [.5, .5])
        self.assertAlmostEqual(result['clean_fraction_above_frozen_threshold'], 2 / 3)
        with self.assertRaises(ValueError):
            paired_summary([1.], [1., 2.], 0.)
        with self.assertRaises(ValueError):
            paired_summary([1.], [float('nan')], 0.)


if __name__ == '__main__':
    unittest.main()
