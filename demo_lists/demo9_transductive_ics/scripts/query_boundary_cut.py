"""Reference-score level sets, query-affinity stopping; no training or labels.

Only the candidate family is reference-conditioned. This tests the user's
proposed stopping mechanism, not an established new/size-unbiased method.
"""
import numpy as np


def grid_edges(height, width):
    ids = np.arange(height*width).reshape(height, width)
    a = np.concatenate([ids[:, :-1].ravel(), ids[:-1].ravel()])
    b = np.concatenate([ids[:, 1:].ravel(), ids[1:].ravel()])
    return a, b


def minimum_mean_boundary_cut(score, affinity, *, reference_threshold=.5):
    """All distinct strict level sets; equal-valued nodes enter simultaneously.

Choose minimum mean four-neighbour crossing affinity. Exact objective ties
choose the normalized threshold closest to the EXISTING source .5 decision.
No area cap, component selector, query truth, or tuned affinity scale.
Small frontiers have more variable mean estimates: no size-unbiased claim.
"""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 2 or not score.size or not np.isfinite(score).all():
        raise ValueError('One finite scalar score grid required')
    a, b = grid_edges(*score.shape)
    affinity = np.asarray(affinity, dtype=np.float64)
    if affinity.shape != a.shape or not np.isfinite(affinity).all():
        raise ValueError('Finite affinity in fixed horizontal-then-vertical edge order required')
    low, high = float(score.min()), float(score.max())
    span = max(high-low, 1e-6)  # EXACT original FoRIS normalization floor
    native_cut = low+reference_threshold*span
    values = np.unique(score)
    if len(values) < 2 or not len(a):
        return dict(threshold=native_cut, normalized_threshold=reference_threshold,
                    mask=score > native_cut, boundary_mean=None, boundary_edges=0,
                    state='DEGENERATE_NATIVE_FALLBACK', candidates=0)
    flat = score.ravel()
    neighbours = [[] for _ in flat]
    for i, j, weight in zip(a, b, affinity):
        neighbours[i].append((j, weight))
        neighbours[j].append((i, weight))
    order = np.argsort(-flat, kind='stable')
    chosen = np.zeros(len(flat), dtype=bool)
    boundary_sum, boundary_size = 0., 0
    candidates = []
    cursor = 0
    while cursor < len(order):
        stop = cursor+1
        while stop < len(order) and flat[order[stop]] == flat[order[cursor]]:
            stop += 1
        for node in order[cursor:stop]:
            chosen[node] = True
            for adjacent, weight in neighbours[node]:
                if chosen[adjacent]:
                    boundary_sum -= weight
                    boundary_size -= 1
                else:
                    boundary_sum += weight
                    boundary_size += 1
        if stop < len(order) and boundary_size > 0:
            threshold = (float(flat[order[cursor]])+float(flat[order[stop]]))/2
            normalized = (threshold-low)/span
            candidates.append((boundary_sum/boundary_size, abs(normalized-reference_threshold),
                               threshold, boundary_size))
        cursor = stop
    if not candidates:
        raise RuntimeError('Nonconstant connected grid has no valid level-set boundary')
    floor = min(x[0] for x in candidates)
    # Only compensate accumulated FLOAT64 roundoff, not a tuned semantic gate.
    arithmetic_tolerance = 16*np.finfo(np.float64).eps*max(1., np.abs(affinity).sum())
    ties = [x for x in candidates if x[0] <= floor+arithmetic_tolerance]
    best = min(ties, key=lambda x: (x[1], x[2]))
    return dict(threshold=best[2], normalized_threshold=(best[2]-low)/span,
                mask=score > best[2], boundary_mean=best[0], boundary_edges=best[3],
                state='QUERY_AFFINITY_CUT', candidates=len(candidates),
                target_size_or_identity_guarantee=False)


def cosine_edge_affinity(features):
    """Native query [C,H,W] fields, not globally debiased identity fields."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3 or not np.isfinite(features).all():
        raise ValueError('Finite query feature grid required')
    norm = np.linalg.norm(features, axis=0, keepdims=True)
    unit = np.divide(features, norm, out=np.zeros_like(features), where=norm > 0)
    a, b = grid_edges(*features.shape[1:])
    tokens = unit.reshape(features.shape[0], -1)
    return (tokens[:, a]*tokens[:, b]).sum(axis=0)


def cpu_check():
    # Strong conditional premise: correct target is a level set AND its
    # exterior affinity is strictly below all target/background interior edges.
    cases = []
    for i in range(10):
        score = np.linspace(0., .25, 48).reshape(6, 8)
        truth = np.zeros((6, 8), bool)
        truth[1:5, 2:6] = True
        score[truth] = np.linspace(.4+i*.001, 1., truth.sum())
        f = np.stack([truth.astype(float), (~truth).astype(float)])
        weights = cosine_edge_affinity(f)
        answer = minimum_mean_boundary_cut(score, weights)
        assert np.array_equal(answer['mask'], truth)
        assert answer['boundary_mean'] == 0
        # Every candidate's objective is independently recomputed on its edges.
        a, b = grid_edges(6, 8)
        candidates = []
        values = np.unique(score)
        for left, right in zip(values[:-1], values[1:]):
            t = (left+right)/2
            m = (score > t).ravel()
            boundary = m[a] != m[b]
            if boundary.any():
                candidates.append(float(weights[boundary].mean()))
        assert abs(answer['boundary_mean']-min(candidates)) < 1e-12
        mapped = minimum_mean_boundary_cut(score**3, weights)
        assert np.array_equal(mapped['mask'], truth)  # grid-family, NOT subpixel interpolation claim
        cases.append(dict(case=i, conditional_target_recovered=True, brute_force_objective_exact=True))
    constant = minimum_mean_boundary_cut(np.ones((2, 2)), np.ones(4))
    assert constant['state'] == 'DEGENERATE_NATIVE_FALLBACK'
    # Concrete failure: a weak internal texture seam beats the true exterior.
    score = np.array([[.9, .8, .4, .3], [.9, .8, .4, .3]])
    a, b = grid_edges(2, 4)
    weights = np.ones(len(a))
    columns = np.arange(8).reshape(2, 4).ravel() % 4
    weights[(columns[a] == 0) & (columns[b] == 1)] = 0.
    weights[(columns[a] == 1) & (columns[b] == 2)] = .2
    failure = minimum_mean_boundary_cut(score, weights)
    expected = np.zeros_like(score, bool)
    expected[:, :2] = True
    assert not np.array_equal(failure['mask'], expected)
    return dict(state='CPU_QUERY_BOUNDARY_CUT_CHECKED', cases=cases,
                exact_internal_seam_counterexample=True, constant_field_native_fallback=True,
                real_task_gain_measured=False, size_unbiased_claim=False)


if __name__ == '__main__':
    import json
    print(json.dumps(cpu_check()))
