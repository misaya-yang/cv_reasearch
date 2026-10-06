"""Local mathematical checks and synthetic CPU cost; no real segmentation evaluation."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from ics.methods.reference_adjacency import (
    Config, centered_reward, decode, grid_edges, motif_reference, permutation_pairs,
    predict, region_counts,
)
from ics.methods.reference_occupancy import spatial_tree, unit


def members(children):
    n = len(children) + 1
    result = [set([i]) for i in range(n)]
    for a, b in children:
        result.append(result[a] | result[b])
    return result


def energy(mask, unary, reward, children, sets):
    value = float(mask @ unary)
    stack = [len(sets) - 1]
    n = len(mask)
    while stack:
        node = stack.pop()
        if mask[list(sets[node])].all():
            value += reward[node]
        elif node >= n:
            stack.extend(children[node - n])
    return value


def main():
    rng = np.random.RandomState(20261006)
    children = np.array([[0, 1], [2, 3], [6, 4], [7, 5], [8, 9]])
    sets = members(children)
    masks = np.array(list(itertools.product([False, True], repeat=6)))
    max_error = 0.0
    for repeat in range(32):
        unary, reward = rng.normal(size=6), rng.normal(size=11)
        if repeat == 0:
            unary[:] = 0
            reward[:] = 0
        mask, margin, optimum = decode(unary, reward, children)
        values = np.array([energy(m, unary, reward, children, sets) for m in masks])
        expected = np.array([values[masks[:, i]].max() - values[~masks[:, i]].max()
                             for i in range(6)])
        max_error = max(max_error, float(np.max(np.abs(margin - expected))))
        np.testing.assert_allclose(margin, expected, atol=1e-12)
        np.testing.assert_allclose(optimum, values.max(), atol=1e-12)
        np.testing.assert_allclose(energy(mask, unary, reward, children, sets), optimum, atol=1e-12)
    # Every tree node's internal edge histogram is checked by enumerating its pixels.
    shape, k = (4, 4), 3
    q = unit(rng.normal(size=(16, 8)))
    tree = spatial_tree(q, shape)
    labels = rng.randint(k, size=16)
    vertices, edges = region_counts(labels, shape, tree, k)
    left, right = grid_edges(shape)
    for node, group in enumerate(members(tree)):
        expected = np.zeros(k * (k + 1) // 2, dtype=int)
        for a, b in zip(left, right):
            if int(a) in group and int(b) in group:
                lo, hi = sorted([int(labels[a]), int(labels[b])])
                expected[lo * k - lo * (lo - 1) // 2 + hi - lo] += 1
        np.testing.assert_array_equal(edges[node], expected)
        np.testing.assert_array_equal(vertices[node], np.bincount(labels[list(group)], minlength=k))
    # Equal label histograms, different arrangement: information absent from occupancy.
    shape = (8, 8)
    stripes = np.tile(np.r_[np.zeros(4, dtype=int), np.ones(4, dtype=int)], (8, 1)).ravel()
    checker = (np.indices(shape).sum(0) % 2).ravel()
    potential, ref_info = motif_reference(stripes, np.ones(64, dtype=bool), shape, 2, Config())
    left, right = grid_edges(shape)
    scores = {}
    for name, label in [('same_arrangement', stripes), ('different_arrangement', checker)]:
        pair = np.minimum(label[left], label[right]) * 2 + np.maximum(label[left], label[right])
        counts = np.bincount(pair, minlength=4)[[0, 1, 3]][None]
        vertex = np.bincount(label, minlength=2)[None]
        reward, residual = centered_reward(vertex, counts, potential, .25)
        scores[name] = dict(token_counts=vertex[0].tolist(), edge_counts=counts[0].tolist(),
                           residual=float(residual[0]), reward=float(reward[0]))
    assert scores['same_arrangement']['residual'] > 0
    assert scores['different_arrangement']['residual'] < 0
    np.testing.assert_array_equal(scores['same_arrangement']['token_counts'],
                                  scores['different_arrangement']['token_counts'])
    # Under uniform label permutation, expected centered score is exactly zero.
    null = permutation_pairs(np.array([[32, 32]]))[0]
    null_error = float((null @ potential) - (permutation_pairs(np.array([32, 32])) @ potential))
    assert abs(null_error) < 1e-15
    # Feature-to-field preparation at actual cache dimensions, synthetic values only.
    q, r = unit(rng.normal(size=(4096, 1024))), unit(rng.normal(size=(4096, 1024)))
    cov = np.zeros((64, 64)); cov[16:48, 16:48] = 1
    base = rng.uniform(.1, .9, size=(64, 64))
    start = time.perf_counter()
    output = predict(q, r, cov, base)
    wall = time.perf_counter() - start
    assert output['field'].shape == (64, 64) and np.isfinite(output['field']).all()
    zero = predict(q, r, cov, base, Config(strength=0))
    np.testing.assert_allclose(zero['field'], base, atol=1e-12)
    np.testing.assert_array_equal(zero['token_mask'], base > .5)
    report = dict(kind='mathematical_and_synthetic_checks_only', real_episodes=0,
                  exhaustive_masks_per_objective=64, objectives=32,
                  maximum_marginal_error=max_error, internal_edge_histograms_checked=31,
                  equal_histogram_witness=scores, permutation_null_error=null_error,
                  synthetic_grid=[64,64], feature_dimension=1024,
                  synthetic_candidate_wall_seconds=wall,
                  synthetic_info=output['info'], real_runtime='unmeasured',
                  real_miou='unmeasured', no_gpu=True, no_server=True)
    path = Path(__file__).with_name('structural_check.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('maximum_marginal_error', 'equal_histogram_witness',
                                                  'synthetic_candidate_wall_seconds', 'real_episodes')}))


if __name__ == '__main__':
    main()
