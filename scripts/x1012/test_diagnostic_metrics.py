"""Regression checks for W45/W49's oracle: independent exhaustive thresholds."""
import unittest
import numpy as np
from diagnostic_metrics import exact_best_iou, classify_diagnostic


def brute(field, gt, valid=None):
    f, g = np.asarray(field), np.asarray(gt, bool)
    v = np.ones(g.shape, bool) if valid is None else np.asarray(valid, bool)
    return max([0.] + [np.count_nonzero((f > t) & g & v) /
                      max(np.count_nonzero(((f > t) | g) & v), 1)
                      for t in np.r_[-np.inf, np.unique(f[v]), np.inf]])


class ExactCutTests(unittest.TestCase):
    def test_matches_independent_brute_force(self):
        rng = np.random.default_rng(58)
        for _ in range(100):
            f = rng.integers(-3, 4, (7, 5)).astype(float)
            g, v = rng.random(f.shape) < .2, rng.random(f.shape) < .8
            self.assertAlmostEqual(exact_best_iou(f, g, v), brute(f, g, v), places=14)

    def test_small_foreground_between_quantiles(self):
        f = np.linspace(0, 1, 10000)
        g = f > .996
        old = max(np.count_nonzero((f > t) & g) / max(np.count_nonzero((f > t) | g), 1)
                  for t in np.quantile(f, np.linspace(0, .999, 120)))
        self.assertLess(old, .5)
        self.assertEqual(exact_best_iou(f, g), 1.)

    def test_ties_cannot_be_split(self):
        self.assertEqual(exact_best_iou(np.ones(4), np.array([1, 0, 0, 0])), .25)

    def test_empty_and_constant_cases(self):
        self.assertEqual(exact_best_iou(np.ones(4), np.zeros(4)), 0.)
        self.assertEqual(exact_best_iou(np.ones(4), np.ones(4)), 1.)
        self.assertEqual(exact_best_iou(np.ones(4), np.ones(4), np.zeros(4)), 0.)
        self.assertEqual(exact_best_iou(np.array([]), np.array([])), 0.)

    def test_ignore_region(self):
        f, g, v = np.array([.8, .7, .6, .1]), np.array([0, 0, 1, 0]), np.array([0, 0, 1, 1])
        self.assertEqual(exact_best_iou(f, g, v), 1.)
        self.assertLess(exact_best_iou(f, g), 1.)

    def test_best_is_at_least_every_fixed_threshold(self):
        rng = np.random.default_rng(59)
        f, g = rng.normal(size=300), rng.random(300) < .03
        optimum = exact_best_iou(f, g)
        for t in np.r_[np.linspace(-4, 4, 50), .5]:
            m = f > t
            self.assertGreaterEqual(optimum + 1e-14, (m & g).sum() / max((m | g).sum(), 1))

    def test_positive_witness_precedes_probe_failure(self):
        self.assertEqual(classify_diagnostic({'cut_F': .68, 'cut_PQ': .001, 'cut_PR': .002}), 'threshold_recoverable')

    def test_bad_shapes_and_nonfinite_scores(self):
        with self.assertRaises(ValueError): exact_best_iou(np.zeros(2), np.zeros(3))
        with self.assertRaises(ValueError): exact_best_iou(np.zeros(2), np.zeros(2), np.zeros(3))
        with self.assertRaises(ValueError): exact_best_iou(np.array([np.nan, 1]), np.ones(2))


if __name__ == '__main__':
    unittest.main()
