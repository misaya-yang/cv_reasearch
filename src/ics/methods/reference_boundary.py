"""Reference-conditioned boundary evidence and exact binary joint inference.

Reference labels estimate an edge likelihood ratio. Query labels never enter
this module. The native reference-contrast field is an explicit unary prior;
the native query aggregation stages and final masks are not method inputs.
"""
from __future__ import annotations

import numpy as np

CONFIG = {
    "name": "reference_boundary_joint_v1",
    "unary": "minmax native Part2 reference foreground/background contrast",
    "edges": "four-connected query grid",
    "edge_statistic": "cosine distance / geometric mean incident distance",
    "reference_density": "16 reference-quantile bins, add-one counts, equal boundary/same priors",
    "pairwise": "max(0, log density_same/density_cut)/4",
    "solver": "integer s-t minimum cut, scale 16384",
    "finalizer": "bilinear binary-token field to 1024, threshold 0.5",
    "query_labels_in_inference": False,
    "extra_encoder_calls": 0,
}


def grid_edges(shape):
    ids = np.arange(np.prod(shape)).reshape(shape)
    return (np.concatenate([ids[:, :-1].ravel(), ids[:-1].ravel()]),
            np.concatenate([ids[:, 1:].ravel(), ids[1:].ravel()]))


def edge_statistic(features, edge):
    x = np.asarray(features, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    a, b = edge
    dist = np.maximum(1 - (x[a] * x[b]).sum(1), 1e-6)
    count = np.bincount(np.r_[a, b], minlength=len(x))
    local = np.bincount(np.r_[a, b], weights=np.r_[dist, dist], minlength=len(x)) / count
    return dist / np.sqrt(local[a] * local[b])


def reference_edge_model(r, coverage):
    c = np.asarray(coverage).ravel()
    edge = grid_edges(np.shape(coverage))
    stat = edge_statistic(r, edge)
    a, b = edge
    # Mixed boundary tokens are excluded from reference estimation, not queries.
    pure = (c <= .1) | (c >= .9)
    valid = pure[a] & pure[b]
    cut = (c[a] >= .5) != (c[b] >= .5)
    bins = np.unique(np.quantile(stat, np.linspace(0, 1, 17)[1:-1]))
    idx = np.searchsorted(bins, stat)
    n = len(bins) + 1
    same = np.bincount(idx[valid & ~cut], minlength=n).astype(float) + 1
    diff = np.bincount(idx[valid & cut], minlength=n).astype(float) + 1
    logratio = np.log(same / same.sum()) - np.log(diff / diff.sum())
    return bins, logratio, {"reference_same_edges": int((valid & ~cut).sum()),
                            "reference_cut_edges": int((valid & cut).sum())}


def minimum_cut(probability, edge, weights):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow
    p = np.asarray(probability, dtype=float).ravel().clip(1e-5, 1 - 1e-5)
    n, scale = len(p), 16384
    a, b = edge
    foreground_cost, background_cost = -np.log(p), -np.log1p(-p)
    rows = np.r_[np.full(n, n), np.arange(n), a, b]
    cols = np.r_[np.arange(n), np.full(n, n + 1), b, a]
    values = np.r_[background_cost, foreground_cost, weights, weights]
    capacities = csr_matrix((np.rint(values * scale).astype(np.int64), (rows, cols)),
                            shape=(n + 2, n + 2))
    capacities.eliminate_zeros()
    flow = maximum_flow(capacities, n, n + 1)
    residual = capacities - flow.flow
    residual.data = (residual.data > 0).astype(np.int64)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    selected = np.zeros(n + 2, dtype=bool)
    selected[reachable] = True
    y = selected[:n]
    energy = np.where(y, foreground_cost, background_cost).sum() + (weights * (y[a] != y[b])).sum()
    return y.reshape(np.shape(probability)), {"energy": float(energy), "quantized_optimum": float(flow.flow_value / scale)}


def predict(q, r, coverage, reference_contrast):
    s = np.asarray(reference_contrast, dtype=np.float64)
    p = (s - s.min()) / max(float(np.ptp(s)), 1e-8)
    edge = grid_edges(s.shape)
    stat = edge_statistic(q, edge)
    bins, ratio, info = reference_edge_model(r, coverage)
    evidence = ratio[np.searchsorted(bins, stat)]
    learned = np.maximum(evidence, 0) / 4
    # The simple control uses the same graph and total pairwise mass.
    affinity = np.exp(-stat)
    affinity *= learned.sum() / max(float(affinity.sum()), 1e-12)
    primary, primary_info = minimum_cut(p, edge, learned)
    control, control_info = minimum_cut(p, edge, affinity)
    masks = {"boundary_joint": primary, "affinity.control": control,
             "reference_unary.control": p > .5}
    info.update(primary=primary_info, control=control_info,
                pairwise_mass=float(learned.sum()),
                primary_changed_tokens=int((primary != (p > .5)).sum()))
    return masks, info, {"boundary_evidence": evidence.astype(np.float32),
                         "local_distance": stat.astype(np.float32)}


def selfcheck():
    """Exhaustive small-graph check, independent of the cut implementation."""
    rng = np.random.RandomState(6)
    edge = grid_edges((2, 3))
    for _ in range(20):
        p = rng.uniform(.05, .95, (2, 3))
        w = rng.uniform(0, 1, len(edge[0]))
        y, info = minimum_cut(p, edge, w)
        energies = []
        for code in range(64):
            z = ((code >> np.arange(6)) & 1).astype(bool)
            energies.append(float(np.where(z, -np.log(p.ravel()), -np.log1p(-p.ravel())).sum()
                                  + (w * (z[edge[0]] != z[edge[1]])).sum()))
        assert info["energy"] <= min(energies) + 1e-3
        plain, _ = minimum_cut(p, edge, np.zeros_like(w))
        assert np.array_equal(plain, p > .5)
    return "20 exhaustive six-node objectives and zero-edge identity passed"
