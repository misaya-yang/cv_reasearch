"""Scientific accounting contracts, independent of models and external data."""
import unittest
import read_sh_evidence as reader


class AccountingTests(unittest.TestCase):
    def test_pool_before_average_and_keep_fold_weighting_distinct(self):
        rows = [dict(fold=f, cls=c, gt=g, A=[i, u, p]) for f, c, g, i, u, p in
                [(0, 0, 1, 1, 2, 2), (0, 0, 1, 1, 8, 8),
                 (0, 1, 8, 8, 10, 10), (1, 0, 1, 1, 1, 1)]]
        self.assertAlmostEqual(reader.miou(rows, 'A'), 200/3)
        self.assertEqual(reader.miou(rows, 'A', official=True), 75)

    def test_invalid_integer_ledger_is_rejected(self):
        for triplet in ([3, 4, 2], [1, 9, 2], [1.0, 3, 2], [-1, 3, 2]):
            with self.subTest(triplet=triplet), self.assertRaises(ValueError):
                reader.counts(dict(gt=2, A=triplet), 'A')

    def test_duplicate_and_mismatched_episodes_are_rejected(self):
        row = dict(index='x', gt=2)
        with self.assertRaises(ValueError):
            reader.indexed([row, row])
        with self.assertRaises(ValueError):
            reader.matching(row, dict(index='x', gt=3), ('gt',))

    def test_marginal_errors_do_not_cancel_in_area_balance(self):
        rows = [dict(gt=10, A=[4, 16, 10]), dict(gt=10, A=[2, 9+10-2, 9])]
        result = reader.error_accounting(rows, 'A')
        self.assertEqual(result['FP'], 13)
        self.assertEqual(result['FN'], 14)
        self.assertEqual(result['over_area'], 0)
        self.assertEqual(result['under_area'], 1)

    def test_empty_cohort_is_explicit(self):
        with self.assertRaises(ValueError):
            reader.miou([], 'A')
        self.assertIsNone(reader.error_accounting([], 'A')['median_predicted_over_GT'])

    def test_report_difference_is_not_silently_accepted(self):
        reader.require_close(1.0, 1.0+1e-12)
        with self.assertRaises(ValueError):
            reader.require_close(1.0, 1.001)


if __name__ == '__main__':
    unittest.main()
