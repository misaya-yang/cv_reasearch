import itertools

import numpy as np

from ics.methods.reference_occupancy import (
    Config, aggregate_tree, apply, max_marginals, occupancy_debt, prepare,
)


def test_coverage_debt_detects_repeated_use_without_absolute_score_dependence():
    cfg = Config()
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    a = np.array([[.5, .5], [1.0, 0.0], [0.0, 1.0]])
    b = np.array([.5, .5])
    debt, info = occupancy_debt(a, b, cost, cfg)
    assert abs(debt[0]) < 1e-8
    assert debt[1] > .4 and debt[2] > .4
    assert info['maximum_marginal_l1'] < cfg.sinkhorn_tolerance
    shifted, _ = occupancy_debt(a, b, cost + np.array([[.7], [-.2]]), cfg)
    np.testing.assert_allclose(shifted, debt, atol=1e-9)
    permuted, _ = occupancy_debt(a[:, ::-1], b[::-1], cost[::-1, ::-1], cfg)
    np.testing.assert_allclose(permuted, debt, atol=1e-9)


def _energy(mask, unary, penalty, children):
    n = len(mask)
    all_on = aggregate_tree(mask.astype(int), children)
    sizes = aggregate_tree(np.ones(n, dtype=int), children)
    total = float(unary @ mask)
    stack = [2 * n - 2]
    while stack:
        node = stack.pop()
        if all_on[node] == sizes[node]:
            total -= penalty[node]
        elif node >= n:
            stack.extend(children[node - n])
    return total


def test_tree_max_marginals_match_exhaustive_masks():
    children = np.array([[0, 1], [2, 3], [6, 4], [7, 5], [8, 9]])
    rng = np.random.RandomState(19)
    masks = np.array(list(itertools.product([False, True], repeat=6)))
    for _ in range(5):
        unary = rng.normal(size=6)
        penalty = rng.uniform(size=11)
        marginal, optimum = max_marginals(unary, penalty, children)
        energy = np.array([_energy(m, unary, penalty, children) for m in masks])
        expected = np.array([energy[masks[:, i]].max() - energy[~masks[:, i]].max()
                             for i in range(6)])
        np.testing.assert_allclose(marginal, expected, atol=1e-12)
        np.testing.assert_allclose(optimum, energy.max(), atol=1e-12)
        zero, _ = max_marginals(unary, np.zeros(11), children)
        np.testing.assert_allclose(zero, unary, atol=1e-12)


def test_joint_coverage_can_recover_a_low_score_piece():
    children = np.array([[0, 1]])
    # Each isolated piece incurs a debt; together they cover the reference.
    marginal, optimum = max_marginals(np.array([.2, -.02]), np.array([.3, .3, 0]), children)
    assert (marginal > 0).all()
    np.testing.assert_allclose(optimum, .18)
    # A region that cannot pay for the missing piece is removed instead.
    marginal, optimum = max_marginals(np.array([.2, -.4]), np.array([.3, .3, 0]), children)
    assert (marginal < 0).all()
    np.testing.assert_allclose(optimum, 0)


def test_complete_feature_to_field_path_has_exact_zero_strength_control():
    rng = np.random.RandomState(7)
    q, r = rng.normal(size=(64, 12)), rng.normal(size=(64, 12))
    cov = np.zeros((8, 8), dtype=np.float32)
    cov[2:6, 2:6] = 1
    base = rng.uniform(-.1, 1.1, size=(8, 8)).astype(np.float32)
    cfg = Config(reference_modes=3, query_modes=6, lloyd_steps=2)
    prepared = prepare(q, r, cov, base.shape, cfg)
    assert prepared['sizes'][-1] == 64
    assert (prepared['debt'] >= 0).all()
    unchanged, _ = apply(base, prepared, 0)
    np.testing.assert_array_equal(unchanged, base)
    for mode in ['regional', 'pointwise']:
        field, _ = apply(base, prepared, .5, mode)
        assert field.shape == base.shape and np.isfinite(field).all()
    assert prepared['info']['query_gt_used'] is False
    assert prepared['info']['encoder_forwards'] == 0
