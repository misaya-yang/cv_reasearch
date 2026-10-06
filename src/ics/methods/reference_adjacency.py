"""CPU candidate: reference appearance adjacency beyond mode proportions.

This is a specified hypothesis, not a measured segmentation improvement.
Uses cached features, the reference mask and a complete continuous base field.
No query truth, class ID, new encoder invocation or fitted development labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from .reference_occupancy import aggregate_tree, cluster, spatial_tree, unit


@dataclass(frozen=True)
class Config:
    reference_modes: int = 8
    lloyd_steps: int = 5
    purity: float = 0.9
    prior_edges: float = 1.0
    log_odds_clip: float = 2.0
    strength: float = 0.25


def grid_edges(shape):
    h, w = shape
    ids = np.arange(h * w).reshape(h, w)
    return (np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel())),
            np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel())))


def pair_index(left, right, k):
    lo, hi = np.minimum(left, right), np.maximum(left, right)
    return lo * k - lo * (lo - 1) // 2 + hi - lo


def permutation_pairs(counts):
    """Exact unordered-pair probabilities after permuting vertex labels."""
    counts = np.asarray(counts, dtype=np.float64)
    k = counts.shape[-1]
    left, right = np.triu_indices(k)
    n = counts.sum(-1)
    numerator = counts[..., left] * counts[..., right]
    numerator[..., left == right] -= counts[..., left[left == right]]
    numerator[..., left != right] *= 2
    denominator = n * (n - 1)
    return np.divide(numerator, denominator[..., None],
                     out=np.zeros_like(numerator), where=denominator[..., None] > 0)


def edge_lca(children, left, right):
    """Vectorized binary lifting: each edge belongs to its lowest containing node."""
    n = len(children) + 1
    total = 2 * n - 1
    parent = np.arange(total)
    parent[children.ravel()] = np.repeat(np.arange(n, total), 2)
    depth = np.zeros(total, dtype=np.int32)
    for offset in range(n - 2, -1, -1):
        depth[children[offset]] = depth[n + offset] + 1
    levels = max(1, int(depth.max()).bit_length())
    up = np.empty((levels, total), dtype=np.int32)
    up[0] = parent
    for level in range(1, levels):
        up[level] = up[level - 1, up[level - 1]]
    a, b = np.array(left, copy=True), np.array(right, copy=True)
    swap = depth[a] < depth[b]
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    delta = depth[a] - depth[b]
    for level in range(levels):
        move = ((delta >> level) & 1).astype(bool)
        a[move] = up[level, a[move]]
    for level in range(levels - 1, -1, -1):
        move = up[level, a] != up[level, b]
        a[move], b[move] = up[level, a[move]], up[level, b[move]]
    return np.where(a == b, a, parent[a])


def region_counts(labels, shape, children, k):
    n = len(labels)
    vertex = aggregate_tree(np.eye(k, dtype=np.int32)[labels], children)
    left, right = grid_edges(shape)
    owner = edge_lca(children, left, right)
    pairs = k * (k + 1) // 2
    edges = np.zeros((2 * n - 1, pairs), dtype=np.int32)
    np.add.at(edges, (owner, pair_index(labels[left], labels[right], k)), 1)
    for offset, (a, b) in enumerate(children):
        edges[n + offset] += edges[a] + edges[b]
    return vertex, edges


def motif_reference(labels, foreground, shape, k, cfg):
    counts = np.bincount(labels[foreground], minlength=k)
    null = permutation_pairs(counts)
    left, right = grid_edges(shape)
    keep = foreground[left] & foreground[right]
    observed = np.bincount(pair_index(labels[left[keep]], labels[right[keep]], k),
                           minlength=len(null)).astype(np.float64)
    potential = np.zeros_like(null)
    if observed.sum() and cfg.prior_edges > 0:
        probability = (observed + cfg.prior_edges * null) / (observed.sum() + cfg.prior_edges)
        valid = null > 0
        potential[valid] = np.clip(np.log(probability[valid] / null[valid]),
                                   -cfg.log_odds_clip, cfg.log_odds_clip)
    return potential, dict(foreground_tokens=int(counts.sum()),
                          foreground_edges=int(observed.sum()),
                          token_counts=counts.tolist(), edge_counts=observed.astype(int).tolist())


def centered_reward(vertex, edges, potential, strength):
    sizes = vertex.sum(1)
    num_edges = edges.sum(1)
    observed = np.einsum('np,p->n', edges, potential, optimize=False)
    expected = np.einsum('np,p->n', permutation_pairs(vertex), potential, optimize=False)
    mean = np.divide(observed, num_edges, out=np.zeros(len(vertex)), where=num_edges > 0)
    residual = np.where(num_edges > 0, mean - expected, 0.0)
    return strength * sizes * residual, residual


def decode(unary, reward, children):
    """Exact MAP and max-marginals for signed maximal-FG-subtree rewards.

F(node) forces every descendant FG; G(node) requires at least one BG.
The MAP traceback supplies a globally consistent mask, including objective ties.
"""
    unary = np.asarray(unary, dtype=np.float64).ravel()
    n = len(unary)
    reward = np.asarray(reward, dtype=np.float64)
    if children.shape != (n - 1, 2) or reward.shape != (2 * n - 1,):
        raise ValueError('Tree/reward shape mismatch')
    if not np.isfinite(unary).all() or not np.isfinite(reward).all():
        raise ValueError('Nonfinite objective')
    full = aggregate_tree(unary, children) + reward
    partial = np.zeros(2 * n - 1)
    choice = np.zeros(n - 1, dtype=np.int8)
    for offset, (a, b) in enumerate(children):
        options = (partial[a] + partial[b], full[a] + partial[b], partial[a] + full[b])
        choice[offset] = int(np.argmax(options))
        partial[n + offset] = options[choice[offset]]
    state = np.zeros(2 * n - 1, dtype=bool)
    state[-1] = full[-1] > partial[-1]
    for offset in range(n - 2, -1, -1):
        a, b = children[offset]
        if state[n + offset]:
            state[a] = state[b] = True
        else:
            state[a], state[b] = choice[offset] == 1, choice[offset] == 2
    outer_full = np.full(2 * n - 1, -np.inf)
    outer_partial = np.full(2 * n - 1, -np.inf)
    ancestor_full = np.full(2 * n - 1, -np.inf)
    outer_full[-1] = outer_partial[-1] = 0
    for offset in range(n - 2, -1, -1):
        node = n + offset
        a, b = children[offset]
        above = max(ancestor_full[node], outer_full[node] + full[node])
        for child, sibling in ((a, b), (b, a)):
            ancestor_full[child] = above
            outer_full[child] = outer_partial[node] + partial[sibling]
            outer_partial[child] = outer_partial[node] + max(full[sibling], partial[sibling])
    margin = np.maximum(ancestor_full[:n], outer_full[:n] + full[:n]) - outer_partial[:n]
    return state[:n], margin, float(max(full[-1], partial[-1]))


def predict(q, r, coverage, base_field, cfg=Config()):
    started = time.perf_counter()
    if (cfg.reference_modes < 1 or cfg.lloyd_steps < 1 or cfg.prior_edges <= 0
            or cfg.strength < 0 or cfg.log_odds_clip <= 0 or not 0 < cfg.purity <= 1):
        raise ValueError('Invalid configuration')
    q, r = unit(q), unit(r)
    base = np.asarray(base_field, dtype=np.float64)
    cov = np.asarray(coverage, dtype=np.float64)
    if (base.ndim != 2 or cov.shape != base.shape or len(q) != base.size or len(r) != base.size
            or q.shape[1] != r.shape[1] or not np.isfinite(base).all()
            or not np.isfinite(cov).all() or cov.min() < 0 or cov.max() > 1 or cov.max() <= 0):
        raise ValueError('Require aligned square/rectangular reference/query grids and valid mask')
    foreground = cov.ravel() >= cfg.purity
    if not foreground.any():
        foreground = cov.ravel() == cov.max()
    centers, _, _ = cluster(r[foreground], cfg.reference_modes, cfg.lloyd_steps)
    ref_labels = np.einsum('nd,kd->nk', r, centers, optimize=False).argmax(1)
    labels = np.einsum('nd,kd->nk', q, centers, optimize=False).argmax(1)
    k = len(centers)
    potential, reference_info = motif_reference(ref_labels, foreground, base.shape, k, cfg)
    children = spatial_tree(q, base.shape)
    vertex, edges = region_counts(labels, base.shape, children, k)
    reward, residual = centered_reward(vertex, edges, potential, cfg.strength)
    mask, margin, energy = decode(base.ravel() - 0.5, reward, children)
    return dict(token_mask=mask.reshape(base.shape), field=(0.5 + margin).reshape(base.shape),
                info=dict(config=asdict(cfg), reference=reference_info, modes=k,
                          regions=len(reward), potential=potential.tolist(), maximum_energy=energy,
                          residual_min=float(residual.min()), residual_max=float(residual.max()),
                          added_tokens=int((mask & (base.ravel() <= .5)).sum()),
                          deleted_tokens=int((~mask & (base.ravel() > .5)).sum()),
                          wall_seconds=time.perf_counter() - started,
                          query_gt_used=False, new_encoder_forwards=0,
                          segmentation_benefit='unmeasured', minute_budget='unmeasured'))
