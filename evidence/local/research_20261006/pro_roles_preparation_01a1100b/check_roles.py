"""Small CPU contract checks, not real segmentation experiments."""
from pathlib import Path
import inspect
import itertools
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from ics.methods import pro_role_prediction as m


def naive_ward(q, shape):
    n = len(q)
    groups = {i: {i} for i in range(n)}
    grid_edges = {(i, i + 1) for i in range(n) if i % shape[1] + 1 < shape[1]}
    grid_edges |= {(i, i + shape[1]) for i in range(n) if i + shape[1] < n}
    result = []
    for v in range(n, 2 * n - 1):
        proposals = []
        for a, b in itertools.combinations(groups, 2):
            ga, gb = groups[a], groups[b]
            if not any((x in ga and y in gb) or (x in gb and y in ga) for x, y in grid_edges):
                continue
            if min(ga) > min(gb):
                a, b, ga, gb = b, a, gb, ga
            diff = q[sorted(ga)].mean(0) - q[sorted(gb)].mean(0)
            cost = len(ga) * len(gb) / (len(ga) + len(gb)) * (diff @ diff)
            proposals.append((float(cost), min(ga), min(gb), a, b))
        _, _, _, a, b = min(proposals)
        result.append((a, b))
        groups[v] = groups.pop(a) | groups.pop(b)
    return np.asarray(result)


def enumerate_tree_value(tree, gains):
    n = len(tree) + 1
    def choices(v):
        out = [(np.zeros(n, bool), 0.)]
        whole = np.zeros(n, bool)
        stack = [v]
        while stack:
            x = stack.pop()
            if x < n:
                whole[x] = True
            else:
                stack.extend(tree[x - n])
        out.append((whole, gains[v]))
        if v >= n:
            for a, b in itertools.product(choices(tree[v - n, 0]), choices(tree[v - n, 1])):
                out.append((a[0] | b[0], a[1] + b[1]))
        return out
    return max(value for _, value in choices(2 * n - 2))


def main():
    rng = np.random.RandomState(11)
    maximum_gap = 0.
    for _ in range(24):
        q = m.unit(rng.randn(6, 7))
        tree, size, minimum, _, _ = m.ward_tree(q, (2, 3))
        assert np.array_equal(tree, naive_ward(q, (2, 3)))
        gains = rng.randn(11)
        _, result = m.decode_tree(tree, gains)
        gap = abs(result - enumerate_tree_value(tree, gains))
        maximum_gap = max(gap, maximum_gap)
        assert gap < 1e-12
        assert np.array_equal(np.sort(m.node_subregions(10, tree, size, minimum, limit=6)), np.arange(6))
    # Exact all-equal tie picks ordered smallest leaf pair, dynamically.
    tree, *_ = m.ward_tree(np.ones((4, 3)), (2, 2))
    assert np.array_equal(tree, [[0, 1], [4, 2], [5, 3]])
    cfg = m.Config()
    counter = dict(hungarian_calls=0, degenerate_directions=0)
    assigned = m.assignment(np.empty((4, 0)), 1., counter)
    assert np.array_equal(assigned, [-1, -1, -1, -1])
    rd = 1 - np.eye(4)
    qd = 1 - np.eye(3)
    missing, residual = m.predict_remaining(rd, qd, np.array([0, 2]), np.array([1, 3]),
                                           np.array([0, -1]), cfg, counter)
    assert missing is None and residual is None
    # Prediction API cannot receive held-out cross-image scores.
    assert 'cross' not in str(inspect.signature(m.predict_remaining))
    fit = np.array([0, 2])
    prediction, residual = m.predict_remaining(rd, qd, np.array([0, 2]), np.array([1, 3]), fit, cfg, counter)
    assert set(prediction[prediction >= 0]).isdisjoint({0, 2})
    assert len(set(prediction[prediction >= 0])) == int((prediction >= 0).sum())
    # Tiny-leaf bias is retained with the full4096-node kappa.
    _, value = m.decode_tree(np.empty((0, 2), dtype=np.int32), np.array([4 - np.log(4097)]))
    assert value == 0
    # Active end-to-end tiny input, no native fallback and complete four fields.
    r = np.eye(8)
    q = r[[0, 1, 2, 3, 4, 5, 6, 7]]
    cov = np.array([[1., 1., 1., 1.], [0., 0., 0., 0.]])
    result = m.predict(q, r, cov, np.array([[.8, .7, .6, .8], [.1, .2, .3, .1]]))
    assert set(result['fields']) == set(m.ARMS)
    assert result['info']['template_count'] == 1
    assert all(v.shape == (2, 4) and np.isfinite(v).all() for v in result['fields'].values())
    # Input-condition fallback must read full native, never a threshold proxy.
    fallback = np.zeros((1024, 1024), dtype=bool)
    fallback[1, 2] = True
    used = []
    def native():
        used.append(True)
        return fallback
    tiny = m.predict(q, r, np.array([[1., 0., 0., 0.], [0., 0., 0., 0.]]), np.ones((2, 4)), native_fallback=native)
    assert used and all(np.array_equal(v, fallback) for v in tiny['masks'].values())
    report = dict(status='passed_small_CPU_contract_checks', ward_cases=25, dp_cases=24,
                  maximum_dp_value_gap=maximum_gap, independent_dummy_check=True,
                  heldout_prediction_has_no_cross_cost_argument=True,
                  insufficient_real_anchors_check=True, full_native_fallback_check=True,
                  single_token_bias_preserved=True, active_complete_fields_check=True,
                  real_episodes=0, actual_cohort_cost='unmeasured')
    (Path(__file__).parent / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
