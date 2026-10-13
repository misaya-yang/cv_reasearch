"""Small correctness checks for the scientific contracts, without experiments."""
from __future__ import annotations

import unittest
import numpy as np
import common as c
import score
from diagnostics import chain_prediction


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c.initialize()

    def test_oracle_does_not_split_equal_scores(self):
        gt = np.indices((1024, 1024)).sum(0) % 2 == 0
        valid = np.ones_like(gt)
        result = score.strict_oracle(np.zeros((64, 64), np.float32), gt, valid)
        self.assertEqual(result, [524288, 1048576, 1048576])

    def test_oracle_excludes_ignored_pixels(self):
        gt = np.zeros((1024, 1024), bool)
        gt[:512] = True
        valid = gt.copy()
        result = score.strict_oracle(np.ones((64, 64), np.float32), gt, valid)
        self.assertEqual(result, [524288, 524288, 524288])
        self.assertEqual(score.strict_oracle(np.ones((64, 64), np.float32), gt,
                                            np.zeros_like(valid)), [0, 0, 0])

    def test_absolute_and_minmax_readouts_remain_distinct(self):
        field = np.full((64, 64), .6, np.float32)
        self.assertTrue(c.readout(field, absolute=True).all())
        self.assertFalse(c.readout(field).any())

    def test_split_is_by_photo_not_episode(self):
        one = {'index': 1, 'row': {'query_photo_id': 'photo-A', 'query_path': 'ignored-A'}}
        two = {'index': 2, 'row': {'query_photo_id': 'photo-A', 'query_path': 'ignored-B'}}
        self.assertEqual(c.half(one), c.half(two))

    def test_bootstrap_pairs_all_episodes_of_each_photo(self):
        records = [{'photo': photo, 'fold': 0, 'cls': cls,
                    'a': [i, u, 1], 'b': [i, u, 1]}
                   for photo, cls, i, u in [('same', 0, 1, 2), ('same', 1, 7, 9),
                                          ('other', 0, 4, 5)]]
        self.assertEqual(score.paired_ci(records, 'a', 'b', np.random.default_rng(c.SEED)), [0., 0.])

    def test_empty_chain_matches_frozen_candidate_in_ten_domains(self):
        bindings = c.rows()
        for domain in c.DOMAINS:
            binding = next(b for b in bindings if b['dataset'] == domain)
            with c.rc.pc.inference_only(binding):
                actual = chain_prediction(binding, [])
                with np.load(c.prediction_path('T0', binding), allow_pickle=False) as archive:
                    np.testing.assert_array_equal(actual, archive['candidate'])


if __name__ == '__main__':
    unittest.main()
