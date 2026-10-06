"""Reference-anchor absorbing walk on a compressed unlabeled query manifold.

FG/BG reference anchors are exact absorbing boundary values. Query modes have
no base-field fidelity in the graph solve. The complete method adds harmonic
first-hit value minus direct reference-kernel evidence to the existing base;
it does not replace the base with a global reference classifier. This is an
unmeasured candidate, not a novelty claim for graph label propagation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy import sparse

from .reference_occupancy import cluster, unit


METHOD_ID = 'reference_anchor_absorption_residual_v1'


@dataclass(frozen=True)
class Config:
    foreground_anchors: int = 16
    background_anchors: int = 16
    query_modes: int = 64
    reference_lloyd_steps: int = 5
    query_lloyd_steps: int = 3
    purity: float = .9
    minimum_reference_samples: int = 8
    neighbors: int = 8
    cosine_temperature: float = .07


def feature_graph(nodes, neighbors):
    """Mutual angular kNN with local bandwidths, fixed feature geometry only."""
    nodes = np.asarray(nodes, dtype=np.float64)
    n = len(nodes)
    distance = np.maximum(0, 1 - np.einsum('id,jd->ij', nodes, nodes, optimize=False))
    np.fill_diagonal(distance, np.inf)
    k = min(neighbors, n - 1)
    order = np.argsort(distance, axis=1, kind='stable')[:, :k]
    directed = np.zeros((n, n), dtype=bool)
    directed[np.arange(n)[:, None], order] = True
    bandwidth = np.maximum(distance[np.arange(n), order[:, -1]], 1e-6)
    denom = np.sqrt(bandwidth[:, None] * bandwidth[None, :])
    weight = np.exp(-distance / denom)
    weight *= directed & directed.T
    np.fill_diagonal(weight, 0)
    return weight


def absorbing_values(weight, labels):
    """Exact finite Dirichlet solve, on components containing both classes.

    labels are 0/1 for labeled reference nodes, NaN for query modes. Components
    with either class absent abstain; they cannot supply two-class evidence.
    The returned raw values there are neutral .5, not semantic probabilities.
    """
    weight = np.asarray(weight, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(labels)
    if (weight.shape != (n, n) or not n or not np.isfinite(weight).all()
            or (weight < 0).any() or not np.allclose(weight, weight.T, atol=1e-13)
            or np.any(np.diag(weight) != 0)
            or not np.all(np.isnan(labels) | (labels == 0) | (labels == 1))):
        raise ValueError('Symmetric nonnegative graph and FG/BG/NaN labels required')
    count, component = connected_components(sparse.csr_matrix(weight), directed=False)
    values = np.full(n, .5, dtype=np.float64)
    active = np.zeros(n, dtype=bool)
    fixed = np.isfinite(labels)
    values[fixed] = labels[fixed]
    residuals = []
    for c in range(count):
        member = component == c
        boundary = np.flatnonzero(member & fixed)
        unknown = np.flatnonzero(member & ~fixed)
        if not len(unknown) or not (np.any(labels[boundary] == 0) and np.any(labels[boundary] == 1)):
            continue
        local = weight[np.ix_(unknown, unknown)]
        matrix = np.diag(weight[unknown].sum(1)) - local
        rhs = np.einsum('ij,j->i', weight[np.ix_(unknown, boundary)], labels[boundary], optimize=False)
        result = np.linalg.solve(matrix, rhs)
        if not np.isfinite(result).all() or result.min() < -1e-8 or result.max() > 1 + 1e-8:
            raise FloatingPointError('Invalid Dirichlet solution')
        residuals.append(float(np.max(np.abs(np.einsum('ij,j->i', matrix, result, optimize=False) - rhs))))
        values[unknown] = np.clip(result, 0, 1)
        active[unknown] = True
    return values, active, dict(components=count, maximum_linear_residual=max(residuals, default=0.0),
                                active_query_nodes=int(active.sum()))


def predict(q, r, coverage, base, cfg=Config()):
    started = time.perf_counter()
    if (cfg.foreground_anchors < 1 or cfg.background_anchors < 1
            or cfg.foreground_anchors + cfg.background_anchors > 32 or cfg.query_modes < 1
            or cfg.query_modes > 64 or cfg.reference_lloyd_steps < 1 or cfg.query_lloyd_steps < 1
            or not .5 < cfg.purity <= 1 or cfg.minimum_reference_samples < 2
            or cfg.neighbors < 1 or cfg.cosine_temperature <= 0):
        raise ValueError('Require <=32 reference anchors, <=64 query modes, positive fixed controls')
    q, r = unit(q), unit(r)
    cov, base = np.asarray(coverage, dtype=np.float64), np.asarray(base, dtype=np.float64)
    if (base.ndim != 2 or base.shape != cov.shape or base.size != len(q) or len(r) != len(q)
            or q.shape[1] != r.shape[1] or not np.isfinite(base).all() or not np.isfinite(cov).all()
            or cov.min() < 0 or cov.max() > 1):
        raise ValueError('Require aligned finite q/r/cov/base')
    keys = ('field', 'nearest_control', 'kernel_control', 'one_hop_control',
            'one_step_control', 'component_control', 'full_harmonic_control')
    fields = {key: base.copy() for key in keys}
    info = dict(method_id=METHOD_ID, config=asdict(cfg), abstention=True,
                query_gt_used=False, new_encoder_forwards=0,
                real_gain='unmeasured', real_runtime='unmeasured')
    fg = cov.ravel() >= cfg.purity
    bg = cov.ravel() <= 1 - cfg.purity
    if min(int(fg.sum()), int(bg.sum())) < cfg.minimum_reference_samples:
        info['reason'] = 'missing_pure_reference_class'
    else:
        foreground, _, fg_mass = cluster(r[fg], cfg.foreground_anchors, cfg.reference_lloyd_steps, cov.ravel()[fg])
        background, _, bg_mass = cluster(r[bg], cfg.background_anchors, cfg.reference_lloyd_steps, 1 - cov.ravel()[bg])
        anchors = np.concatenate((foreground, background))
        nf, na = len(foreground), len(anchors)
        modes, assignment, _ = cluster(q, cfg.query_modes, cfg.query_lloyd_steps)
        nq = len(modes)
        nodes = np.concatenate((anchors, modes))
        labels = np.r_[np.ones(nf), np.zeros(na - nf), np.full(nq, np.nan)]
        weight = feature_graph(nodes, cfg.neighbors)
        values, active, solve_info = absorbing_values(weight, labels)
        # All controls share the exact query compression and hard mode readout.
        response = np.einsum('id,jd->ij', modes, anchors, optimize=False).astype(np.float64)
        nearest = np.clip(.5 + .5 * (response[:, :nf].max(1) - response[:, nf:].max(1)), 0, 1)
        scaled = response / cfg.cosine_temperature
        scaled -= scaled.max(1, keepdims=True)
        kernel = np.exp(scaled)
        positive = np.einsum('ij,j->i', kernel[:, :nf], fg_mass, optimize=False)
        negative = np.einsum('ij,j->i', kernel[:, nf:], bg_mass, optimize=False)
        direct = positive / (positive + negative)
        terminal_edges = weight[na:, :na]
        terminal_degree = terminal_edges.sum(1)
        one_hop = np.divide(terminal_edges[:, :nf].sum(1), terminal_degree,
                            out=np.full(nq, .5), where=terminal_degree > 0)
        degree = weight[na:].sum(1)
        one_step = np.divide(np.einsum('ij,j->i', weight[na:], np.r_[labels[:na], direct], optimize=False), degree,
                             out=direct.copy(), where=degree > 0)
        _, component = connected_components(sparse.csr_matrix(weight), directed=False)
        component_prior = np.full(nq, .5)
        for c in np.unique(component[na:]):
            boundary = (component[:na] == c)
            if boundary.any():
                component_prior[component[na:] == c] = labels[:na][boundary].mean()
        query_values = values[na:]
        query_active = active[na:]
        token_active = query_active[assignment]
        flat = base.ravel().copy()
        residual = query_values - direct
        flat[token_active] += residual[assignment[token_active]]
        fields['field'] = flat.reshape(base.shape)
        for key, value in (('nearest_control', nearest), ('kernel_control', direct),
                           ('one_hop_control', one_hop), ('one_step_control', one_step),
                           ('component_control', component_prior), ('full_harmonic_control', query_values)):
            control = base.ravel().copy()
            if key in ('one_hop_control', 'one_step_control', 'component_control'):
                control[token_active] += (value - direct)[assignment[token_active]]
            else:
                control[token_active] = value[assignment[token_active]]
            fields[key] = control.reshape(base.shape)
        info.update(abstention=not bool(token_active.any()), reference_anchors=na,
                    foreground_anchors=nf, background_anchors=na-nf, query_modes=nq,
                    graph_nodes=len(nodes), graph_edges=int(np.count_nonzero(np.triu(weight, 1))),
                    active_tokens=int(token_active.sum()), abstained_tokens=int((~token_active).sum()),
                    solve=solve_info, added_tokens=int(((fields['field'] > .5) & (base <= .5)).sum()),
                    deleted_tokens=int(((fields['field'] <= .5) & (base > .5)).sum()),
                    maximum_context_residual=float(np.max(np.abs(residual), initial=0)),
                    field_semantics='base + first-hit - direct reference-kernel; no clamp or extra minmax')
    info['wall_seconds'] = time.perf_counter() - started
    return dict(**fields, info=info)
