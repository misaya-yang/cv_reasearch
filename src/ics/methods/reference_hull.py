"""Reference foreground/background convex-hull distances; unvalidated transfer.

Frank-Wolfe gaps bound optimization error in exact arithmetic. They do not
bound segmentation error or validate a semantic mixture model for DINO features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from .reference_occupancy import cluster, unit


@dataclass(frozen=True)
class Config:
    foreground_modes: int = 16
    background_modes: int = 16
    lloyd_steps: int = 5
    purity: float = .9
    minimum_reference_samples: int = 8
    maximum_iterations: int = 256
    gap_tolerance: float = 1e-8
    decision_guard: float = 1e-7
    subspace_relative_tolerance: float = 1e-6


def validate_config(cfg):
    integers = (cfg.foreground_modes, cfg.background_modes, cfg.lloyd_steps,
                cfg.minimum_reference_samples, cfg.maximum_iterations)
    real = (cfg.purity, cfg.gap_tolerance, cfg.decision_guard,
            cfg.subspace_relative_tolerance)
    if (any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in integers)
            or not all(np.isfinite(v) for v in real)
            or not 1 <= cfg.foreground_modes <= 16 or not 1 <= cfg.background_modes <= 16
            or cfg.lloyd_steps < 1 or not .5 < cfg.purity <= 1
            or cfg.minimum_reference_samples < 2 or cfg.maximum_iterations < 1
            or cfg.gap_tolerance <= 0 or cfg.decision_guard <= 0
            or not 0 < cfg.subspace_relative_tolerance < 1):
        raise ValueError('Invalid reference hull configuration')


def hull_distance(query, anchors, cfg=Config()):
    """Squared distance bounds using feasible simplex Frank-Wolfe iterates.

    With f(w)=.5*||Aw-q||^2, gap=<w-s, grad f>; hence squared
    distance lies in [max(0,cost-2*gap), cost] in exact arithmetic.
    No convergence is required to use this optimization bound. Floating-point
    guards are nominal safeguards, not outward-rounded interval arithmetic.
    """
    validate_config(cfg)
    query = np.asarray(query, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    if (query.ndim != 2 or anchors.ndim != 2 or not len(query) or not len(anchors)
            or query.shape[1] != anchors.shape[1] or query.shape[1] == 0
            or not np.isfinite(query).all() or not np.isfinite(anchors).all()):
        raise ValueError('Nonempty finite compatible query and anchor matrices required')
    gram = np.einsum('id,jd->ij', anchors, anchors, optimize=False)
    response = np.einsum('nd,kd->nk', query, anchors, optimize=False)
    norm2 = np.einsum('nd,nd->n', query, query, optimize=False)
    rows = np.arange(len(query))
    index = np.argmin(np.diag(gram)[None, :] - 2*response, axis=1)
    weight = np.zeros_like(response)
    weight[rows, index] = 1
    for iteration in range(cfg.maximum_iterations + 1):
        fitted = np.einsum('nk,kj->nj', weight, gram, optimize=False)
        gradient = fitted - response
        index = gradient.argmin(axis=1)
        gap = np.maximum(0, np.sum(weight*gradient, axis=1) - gradient[rows, index])
        if np.max(gap) <= cfg.gap_tolerance or iteration == cfg.maximum_iterations:
            break
        curvature = np.maximum(0, np.diag(gram)[index] - 2*fitted[rows, index] + np.sum(weight*fitted, axis=1))
        step = np.divide(gap, curvature, out=np.ones_like(gap), where=curvature > 1e-14)
        step[gap == 0] = 0
        step = np.clip(step, 0, 1)
        weight *= 1 - step[:, None]
        weight[rows, index] += step
    cost = np.maximum(0, norm2 - 2*np.sum(weight*response, axis=1) + np.sum(weight*fitted, axis=1))
    lower = np.maximum(0, cost - 2*gap)
    return dict(cost=cost, lower=lower, gap=gap,
                info=dict(iterations=iteration, maximum_gap=float(gap.max()),
                          unconverged_rows=int((gap > cfg.gap_tolerance).sum()),
                          converged=bool(np.all(gap <= cfg.gap_tolerance)),
                          maximum_simplex_sum_error=float(np.max(np.abs(weight.sum(axis=1)-1))),
                          minimum_coefficient=float(weight.min())))


def basis(anchors, tolerance):
    if len(anchors) == 0:
        return np.empty((0, anchors.shape[1]))
    _, values, vectors = np.linalg.svd(anchors, full_matrices=False)
    return vectors[values > tolerance*values[0]] if values[0] > 0 else vectors[:0]


def span_distance(query, anchors, tolerance, affine=False):
    origin = anchors[0] if affine else np.zeros(anchors.shape[1])
    vectors = anchors[1:] - origin if affine else anchors
    directions = basis(vectors, tolerance)
    shifted = query - origin
    projection = np.einsum('nd,kd->nk', shifted, directions, optimize=False)
    return np.maximum(0, np.sum(shifted**2, axis=1) - np.sum(projection**2, axis=1))


def predict(q, r, coverage, base, cfg=Config()):
    started = time.perf_counter()
    validate_config(cfg)
    q, r = unit(q).astype(np.float64), unit(r)
    coverage = np.asarray(coverage, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if (base.ndim != 2 or not base.size or not q.shape[1]
            or base.shape != coverage.shape or base.size != len(q) or len(q) != len(r)
            or q.shape[1] != r.shape[1] or not np.isfinite(base).all() or not np.isfinite(coverage).all()
            or coverage.min() < 0 or coverage.max() > 1):
        raise ValueError('Aligned finite features, reference coverage and base required')
    fields = {key: base.copy() for key in ('field', 'nearest_control', 'centroid_control',
                                         'subspace_control', 'affine_control')}
    info = dict(config=asdict(cfg), abstention=True, query_gt_used=False, new_encoder_forwards=0,
                real_gain='unmeasured', complete_dataset_minutes='unmeasured')
    bounds = dict(lower_difference=np.zeros(base.shape), upper_difference=np.zeros(base.shape),
                  resolved=np.zeros(base.shape, dtype=bool))
    fg, bg = coverage.ravel() >= cfg.purity, (1-coverage.ravel()) >= cfg.purity
    if min(int(fg.sum()), int(bg.sum())) < cfg.minimum_reference_samples:
        info['reason'] = 'missing_reference_foreground_or_background_samples'
    else:
        foreground, _, _ = cluster(r[fg], cfg.foreground_modes, cfg.lloyd_steps, coverage.ravel()[fg])
        background, _, _ = cluster(r[bg], cfg.background_modes, cfg.lloyd_steps, 1-coverage.ravel()[bg])
        foreground, background = foreground.astype(np.float64), background.astype(np.float64)
        f, b = hull_distance(q, foreground, cfg), hull_distance(q, background, cfg)
        lower_delta = b['lower'] - f['cost']
        upper_delta = b['cost'] - f['lower']
        resolved = (lower_delta > cfg.decision_guard) | (upper_delta < -cfg.decision_guard)
        delta = b['cost'] - f['cost']
        value = np.clip(.5 + delta/8, 0, 1)
        fields['field'].ravel()[resolved] = value[resolved]
        bounds.update(lower_difference=lower_delta.reshape(base.shape),
                      upper_difference=upper_delta.reshape(base.shape),
                      resolved=resolved.reshape(base.shape))

        def control(score, divisor):
            out = base.ravel().copy()
            take = np.abs(score) > cfg.decision_guard
            out[take] = np.clip(.5 + score[take]/divisor, 0, 1)
            return out.reshape(base.shape)

        response_f = np.einsum('nd,kd->nk', q, foreground, optimize=False)
        response_b = np.einsum('nd,kd->nk', q, background, optimize=False)
        fields['nearest_control'] = control(response_f.max(axis=1)-response_b.max(axis=1), 4)
        # Original pure-class means; compression must not alter this control.
        centroid_f = np.average(r[fg].astype(np.float64), axis=0, weights=coverage.ravel()[fg])
        centroid_b = np.average(r[bg].astype(np.float64), axis=0, weights=1-coverage.ravel()[bg])
        fields['centroid_control'] = control(np.einsum('nd,d->n', q, centroid_f-centroid_b, optimize=False), 4)
        for key, affine in (('subspace_control', False), ('affine_control', True)):
            df = span_distance(q, foreground, cfg.subspace_relative_tolerance, affine)
            db = span_distance(q, background, cfg.subspace_relative_tolerance, affine)
            fields[key] = control(db-df, 8)
        info.update(abstention=False, foreground_modes=len(foreground), background_modes=len(background),
                    foreground_solver=f['info'], background_solver=b['info'],
                    resolved_rows=int(resolved.sum()), fallback_rows=int((~resolved).sum()),
                    decision='signed BG-minus-FG squared-distance interval excludes zero beyond guard',
                    score_semantics='clipped .5 + (BG cost - FG cost)/8; not a calibrated probability',
                    bound_scope='Frank-Wolfe objective gap in exact arithmetic, with nominal floating guard; not a semantic confidence bound')
    info['wall_seconds'] = time.perf_counter() - started
    return dict(**fields, bounds=bounds, info=info)
