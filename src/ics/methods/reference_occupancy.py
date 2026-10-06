"""Experimental regional reference-occupancy penalty on frozen features.

Hypothesis, not an established result: a false foreground region can repeatedly
match one reference appearance mode. The cost of imposing reference-mode
occupancy, after subtracting independent matching cost, measures that reliance.
A complete foreground region can reduce its penalty by including missing modes.

All variables are inference inputs. Query labels and class/fold IDs are absent.
Optimal transport and spatial hierarchies are existing tools; novelty of this
particular objective and real segmentation benefit remain unverified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np


@dataclass(frozen=True)
class Config:
    reference_modes: int = 8
    query_modes: int = 32
    lloyd_steps: int = 5
    temperature: float = 0.07
    purity: float = 0.9
    sinkhorn_tolerance: float = 1e-7
    sinkhorn_iterations: int = 2048
    region_batch: int = 256


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite token matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Zero-norm feature')
    return np.ascontiguousarray(x / norm)


def cluster(x, k, steps, sample_weight=None):
    """Deterministic spherical Lloyd compression, no target labels or fitting set."""
    x = unit(x)
    weight = (np.ones(len(x), dtype=np.float64) if sample_weight is None
              else np.asarray(sample_weight, dtype=np.float64))
    if weight.shape != (len(x),) or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError('Positive finite token weights required')
    mean = np.average(x, axis=0, weights=weight)
    first = int(np.argmax(np.einsum('nd,d->n', x, mean, optimize=False)))
    seeds = [first]
    nearest = np.einsum('nd,d->n', x, x[first], optimize=False)
    for _ in range(1, min(k, len(x))):
        nxt = int(np.argmin(nearest))
        if float(nearest[nxt]) >= 1 - 1e-6:
            break
        seeds.append(nxt)
        nearest = np.maximum(nearest, np.einsum('nd,d->n', x, x[nxt], optimize=False))
    centers = x[seeds].copy()
    for _ in range(steps):
        labels = np.einsum('nd,kd->nk', x, centers, optimize=False).argmax(1)
        mass = np.bincount(labels, weights=weight, minlength=len(centers))
        sums = np.stack([(x[labels == j] * weight[labels == j, None]).sum(0)
                         for j in range(len(centers))])
        valid = mass > 0
        centers = unit(sums[valid])
    labels = np.einsum('nd,kd->nk', x, centers, optimize=False).argmax(1)
    mass = np.bincount(labels, weights=weight, minlength=len(centers))
    valid = mass > 0
    if not valid.all():
        remap = np.cumsum(valid) - 1
        labels = remap[labels]
        centers, mass = centers[valid], mass[valid]
    return centers, labels.astype(np.int32), mass / mass.sum()


def spatial_tree(q, shape):
    """All nodes of the 4-neighbor cosine single-linkage tree, including pixels.

    This is a fixed proposal geometry, not a claim that each node is an object.
    Edge order and tie resolution are deterministic. No score/GT selects edges.
    """
    h, w = shape
    n = h * w
    if n != len(q) or n < 2:
        raise ValueError('A query grid with at least two tokens is required')
    ids = np.arange(n).reshape(h, w)
    a = np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel()))
    b = np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel()))
    distance = 1 - np.einsum('ij,ij->i', q[a], q[b])
    order = np.argsort(distance, kind='stable')
    parent = np.arange(n)
    roots = np.arange(n)
    children = np.empty((n - 1, 2), dtype=np.int32)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    made = 0
    for edge in order:
        u, v = find(int(a[edge])), find(int(b[edge]))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        children[made] = roots[u], roots[v]
        parent[v] = u
        roots[u] = n + made
        made += 1
        if made == n - 1:
            break
    if made != n - 1:
        raise RuntimeError('Disconnected grid tree')
    return children


def aggregate_tree(leaves, children):
    leaves = np.asarray(leaves)
    n = len(leaves)
    if children.shape != (n - 1, 2):
        raise ValueError('Expected a full binary tree')
    out = np.empty((2 * n - 1,) + leaves.shape[1:], dtype=leaves.dtype)
    out[:n] = leaves
    for offset, (left, right) in enumerate(children):
        node = n + offset
        if left < 0 or right < 0 or left >= node or right >= node or left == right:
            raise ValueError('Invalid child order')
        out[node] = out[left] + out[right]
    return out


def occupancy_debt(a, b, cost, cfg):
    """Balanced minus row-only entropic transport, with inspectable convergence.

    Each a is a query-region probability histogram; b is the reference prior.
    Both objectives use KL(T || a b^T). The difference is >=0, is invariant to
    arbitrary rowwise cost offsets, and is zero when independent matching already
    has the required reference marginal. It does not certify object identity.
    """
    a, b, cost = (np.asarray(v, dtype=np.float64) for v in (a, b, cost))
    if (a.ndim != 2 or b.ndim != 1 or cost.shape != (a.shape[1], len(b))
            or np.any(a < 0) or np.any(b <= 0)
            or not all(np.isfinite(v).all() for v in (a, b, cost))
            or not np.allclose(a.sum(1), 1) or not np.isclose(b.sum(), 1)):
        raise ValueError('Invalid probability histograms or cost matrix')
    if cfg.temperature <= 0 or cfg.sinkhorn_tolerance <= 0:
        raise ValueError('Positive temperature and solver tolerance required')
    if len(b) == 1:
        return np.zeros(len(a)), dict(iterations=0, maximum_marginal_l1=0.0,
                                      maximum_duality_gap=0.0)
    # Removing row minima changes both transport objectives by the same amount.
    shifted = cost - cost.min(1, keepdims=True)
    kernel = np.exp(-shifted / cfg.temperature)
    if not np.all(kernel > 0):
        raise ValueError('Transport kernel underflow; numerical recipe unsupported')
    v = np.broadcast_to(b, (len(a), len(b))).copy()
    worst = float('inf')
    for iteration in range(1, cfg.sinkhorn_iterations + 1):
        u = a / np.einsum('bk,ik->bi', v, kernel, optimize=False)
        v = b / np.einsum('bi,ik->bk', u, kernel, optimize=False)
        # Gauge normalization prevents drifting scales without changing T.
        v /= np.exp(np.mean(np.log(v), axis=1, keepdims=True))
        if iteration % 8 == 0 or iteration == cfg.sinkhorn_iterations:
            kv = np.einsum('bk,ik->bi', v, kernel, optimize=False)
            u = a / kv
            column = v * np.einsum('bi,ik->bk', u, kernel, optimize=False)
            worst = float(np.abs(column - b).sum(1).max(initial=0))
            if worst <= cfg.sinkhorn_tolerance:
                break
    if not np.isfinite(worst) or worst > cfg.sinkhorn_tolerance:
        raise RuntimeError(f'Transport not converged: marginal_l1={worst:.3g}')
    log_ratio = np.log(v / b)
    # T has exact row marginals; its approximate column marginal is retained
    # in the primal objective rather than silently replacing it by b.
    primal = cfg.temperature * (-(a * np.log(kv)).sum(1) + (column * log_ratio).sum(1))
    dual = cfg.temperature * (-(a * np.log(kv)).sum(1) + (b * log_ratio).sum(1))
    free = -cfg.temperature * (a * np.log(np.einsum('ik,k->i', kernel, b, optimize=False))).sum(1)
    debt = primal - free
    gap = float(np.abs(primal - dual).max(initial=0))
    if debt.min(initial=0) < -1e-6 or not np.isfinite(debt).all():
        raise RuntimeError('Invalid negative/nonfinite occupancy debt')
    return np.maximum(debt, 0), dict(iterations=iteration, maximum_marginal_l1=worst,
                                    maximum_duality_gap=gap)


def max_marginals(unary, penalty, children):
    """Exact max-marginals for maximal foreground-subtree penalties.

    Foreground regions form the unique maximal all-FG subtrees of a binary mask.
    F[node] means every pixel is FG; G[node] means at least one is BG. This avoids
    letting a mask evade its regional penalty by arbitrarily splitting its nodes.
    """
    unary = np.asarray(unary, dtype=np.float64).reshape(-1)
    n = len(unary)
    penalty = np.asarray(penalty, dtype=np.float64)
    if penalty.shape != (2 * n - 1,) or not np.isfinite(penalty).all() or np.any(penalty < 0):
        raise ValueError('Finite nonnegative region penalties required')
    full = aggregate_tree(unary, children) - penalty
    partial = np.zeros(2 * n - 1, dtype=np.float64)
    for offset, (left, right) in enumerate(children):
        partial[n + offset] = max(partial[left] + partial[right],
                                  full[left] + partial[right], partial[left] + full[right])
    outer_full = np.full(2 * n - 1, -np.inf)
    outer_partial = np.full(2 * n - 1, -np.inf)
    ancestor_full = np.full(2 * n - 1, -np.inf)
    outer_full[-1] = outer_partial[-1] = 0
    for offset in range(n - 2, -1, -1):
        node = n + offset
        left, right = children[offset]
        above = max(ancestor_full[node], outer_full[node] + full[node])
        for child, sibling in ((left, right), (right, left)):
            ancestor_full[child] = above
            outer_full[child] = outer_partial[node] + partial[sibling]
            outer_partial[child] = outer_partial[node] + max(full[sibling], partial[sibling])
    on = np.maximum(ancestor_full[:n], outer_full[:n] + full[:n])
    off = outer_partial[:n]
    return on - off, float(max(full[-1], partial[-1]))


def prepare(q, r, coverage, shape, cfg=Config()):
    started = time.perf_counter()
    q, r = unit(q), unit(r)
    coverage = np.asarray(coverage, dtype=np.float32).ravel()
    if (len(q) != int(np.prod(shape)) or len(r) != len(coverage)
            or q.shape[1] != r.shape[1] or not np.isfinite(coverage).all()
            or np.any(coverage < 0) or np.any(coverage > 1)):
        raise ValueError('Feature/grid/reference-mask mismatch')
    if coverage.max(initial=0) <= 0:
        raise ValueError('Reference foreground is empty')
    fg = coverage >= cfg.purity
    if not fg.any():
        fg = coverage == coverage.max()
    ref, _, b = cluster(r[fg], cfg.reference_modes, cfg.lloyd_steps, coverage[fg])
    query, labels, _ = cluster(q, cfg.query_modes, cfg.lloyd_steps)
    children = spatial_tree(q, shape)
    hist = aggregate_tree(np.eye(len(query), dtype=np.int32)[labels], children)
    sizes = hist.sum(1)
    # Histograms with the same proportions share exactly the same transport solve.
    gcd = np.gcd.reduce(hist, axis=1)
    canonical = hist // gcd[:, None]
    unique, inverse = np.unique(canonical, axis=0, return_inverse=True)
    proportions = unique / unique.sum(1, keepdims=True)
    cost = 1 - np.clip(np.einsum('ud,vd->uv', query, ref, optimize=False), -1, 1)
    unique_debt = np.empty(len(unique), dtype=np.float64)
    checks = []
    for start in range(0, len(unique), cfg.region_batch):
        stop = min(start + cfg.region_batch, len(unique))
        unique_debt[start:stop], info = occupancy_debt(proportions[start:stop], b, cost, cfg)
        checks.append(info)
    debt = unique_debt[inverse]
    return dict(children=children, sizes=sizes, debt=debt, shape=tuple(shape),
                info=dict(config=asdict(cfg), query_gt_used=False, encoder_forwards=0,
                          reference_modes=len(ref), query_modes=len(query), regions=len(hist),
                          distinct_region_histograms=len(unique), solver=checks,
                          debt_mean=float(debt.mean()), debt_max=float(debt.max()),
                          preparation_seconds=time.perf_counter() - started))


def apply(base_field, prepared, strength=1.0, mode='regional'):
    base = np.asarray(base_field, dtype=np.float32)
    if base.shape != prepared['shape'] or not np.isfinite(base).all() or strength < 0:
        raise ValueError('Finite matching base field and nonnegative strength required')
    if strength == 0:
        return base.copy(), dict(mode=mode, strength=0, maximum_field_change=0.0)
    if mode == 'regional':
        penalty = strength * prepared['sizes'] * prepared['debt']
        margin, energy = max_marginals(base.ravel().astype(np.float64) - .5,
                                       penalty, prepared['children'])
        field = (.5 + margin).reshape(base.shape).astype(np.float32)
    elif mode == 'pointwise':
        # Same features, reference dictionary, OT and coefficient. Removing region
        # coupling leaves only each pixel's source-occupancy penalty (cannot add FG).
        field = base - (strength * prepared['debt'][:base.size]).reshape(base.shape).astype(np.float32)
        energy = None
    else:
        raise ValueError('Unknown occupancy mode')
    if not np.isfinite(field).all():
        raise RuntimeError('Nonfinite decoded field')
    return field, dict(mode=mode, strength=float(strength), maximum_energy=energy,
                       maximum_field_change=float(np.abs(field - base).max()),
                       token_added=int(((field > .5) & (base <= .5)).sum()),
                       token_deleted=int(((field <= .5) & (base > .5)).sum()))
