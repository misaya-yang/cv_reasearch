"""Density-line revision: bounded full-feature Gaussian class-density evidence.

This is a substantive cached variant of the failed scalar prior-shift line,
not an extra independent method count. It neither estimates query prevalence
nor requires a query component to reproduce a reference object's composition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from .reference_occupancy import cluster, unit
from .reference_quadratic import solve_kernel


METHOD_ID = 'reference_density_full_feature_bounded_v2'


@dataclass(frozen=True)
class Config:
    foreground_modes: int = 16
    background_modes: int = 16
    lloyd_steps: int = 5
    purity: float = .9
    minimum_reference_samples: int = 8
    gaussian_temperature: float = .07
    maximum_correction: float = .125
    quadratic_control_ridge: float = .01


def _cosine(left, right):
    # Ignore stale Accelerate FP status flags only; validate actual bounded output.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        result = left @ right.T
    if not np.isfinite(result).all() or np.max(np.abs(result), initial=0) > 1.001:
        raise FloatingPointError('Invalid normalized feature product')
    return np.clip(result, -1., 1.).astype(np.float64)


def balanced_contrast(foreground, background):
    foreground, background = np.asarray(foreground), np.asarray(background)
    if (not np.isfinite(foreground).all() or not np.isfinite(background).all()
            or np.any(foreground < 0) or np.any(background < 0)):
        raise FloatingPointError('Finite nonnegative class kernel densities required')
    denominator = foreground + background
    return np.divide(foreground-background, denominator,
                     out=np.zeros_like(denominator, dtype=np.float64), where=denominator > 0)


def predict(q, r, coverage, base, cfg=Config()):
    started = time.perf_counter()
    if asdict(cfg) != asdict(Config()):
        raise ValueError('This preparation fixes one recipe; no parameter search')
    q, r = unit(q), unit(r)
    coverage, base = np.asarray(coverage, dtype=np.float64), np.asarray(base, dtype=np.float64)
    if (coverage.ndim != 2 or base.ndim != 2 or coverage.size != len(r) or base.size != len(q)
            or q.shape[1] != r.shape[1] or not np.isfinite(coverage).all() or not np.isfinite(base).all()
            or np.any((coverage < 0) | (coverage > 1))):
        raise ValueError('Finite aligned feature/reference-mask/MEAN-field packet required')
    keys = ('field', 'polynomial_control', 'quadratic_ridge_control', 'uniform_control',
            'nearest_control', 'centroid_control', 'standalone_control')
    fields = {key: base.copy() for key in keys}
    info = dict(config=asdict(cfg), method_id=METHOD_ID,
                revised_family='reference_score_density_query_prior_v1', independent_method_increment=0,
                abstention=True, query_gt_used=False, new_encoder_forwards=0,
                real_segmentation_gain='unmeasured', real_complete_runtime='unmeasured')
    if not coverage.any():
        fields = {key: np.zeros_like(base) for key in keys}
        info['reason'] = 'empty_reference'
    else:
        fg = coverage.ravel() >= cfg.purity
        bg = (1-coverage.ravel()) >= cfg.purity
        if min(int(fg.sum()), int(bg.sum())) < cfg.minimum_reference_samples:
            info['reason'] = 'missing_pure_reference_foreground_or_background_samples'
        else:
            foreground, _, f_mass = cluster(r[fg], cfg.foreground_modes, cfg.lloyd_steps,
                                             coverage.ravel()[fg])
            background, _, b_mass = cluster(r[bg], cfg.background_modes, cfg.lloyd_steps,
                                             1-coverage.ravel()[bg])
            anchors = np.concatenate((foreground, background))
            nf = len(foreground)
            response = _cosine(q, anchors)
            gaussian = np.exp((response-1)/cfg.gaussian_temperature)
            f = np.einsum('nk,k->n', gaussian[:, :nf], f_mass, optimize=False)
            b = np.einsum('nk,k->n', gaussian[:, nf:], b_mass, optimize=False)
            contrast = balanced_contrast(f, b)

            def corrected(value):
                return base + cfg.maximum_correction * value.reshape(base.shape)

            fields['field'] = corrected(contrast)
            fields['standalone_control'] = (.5+.5*contrast).reshape(base.shape)
            polynomial = ((1+response)*.5)**2
            fields['polynomial_control'] = corrected(balanced_contrast(
                np.einsum('nk,k->n', polynomial[:, :nf], f_mass, optimize=False),
                np.einsum('nk,k->n', polynomial[:, nf:], b_mass, optimize=False)))
            fields['uniform_control'] = corrected(balanced_contrast(
                gaussian[:, :nf].mean(1), gaussian[:, nf:].mean(1)))
            fields['nearest_control'] = corrected(np.tanh(
                (response[:, :nf].max(1)-response[:, nf:].max(1))/cfg.gaussian_temperature))
            centroid_f = np.average(r[fg].astype(np.float64), axis=0, weights=coverage.ravel()[fg])
            centroid_b = np.average(r[bg].astype(np.float64), axis=0, weights=1-coverage.ravel()[bg])
            fields['centroid_control'] = corrected(np.tanh(
                np.einsum('nd,d->n', q, centroid_f-centroid_b, optimize=False)/cfg.gaussian_temperature))
            gram = _cosine(anchors, anchors)
            weights = .5*np.r_[f_mass, b_mass]
            labels = np.r_[np.ones(nf), -np.ones(len(background))]
            coefficients, solver = solve_kernel(((1+gram)*.5)**2, labels, weights,
                                                 cfg.quadratic_control_ridge)
            fields['quadratic_ridge_control'] = corrected(np.clip(
                np.einsum('nk,k->n', polynomial, coefficients, optimize=False), -1, 1))
            info.update(abstention=False, foreground_modes=nf, background_modes=len(background),
                        reference_foreground_samples=int(fg.sum()), reference_background_samples=int(bg.sum()),
                        foreground_mass=f_mass.tolist(), background_mass=b_mass.tolist(),
                        quadratic_control_solver=solver,
                        minimum_query_gaussian_density=float(min(f.min(), b.min())),
                        maximum_absolute_correction=float(np.max(np.abs(fields['field']-base))),
                        added_tokens=int(((fields['field'] > .5) & (base <= .5)).sum()),
                        deleted_tokens=int(((fields['field'] <= .5) & (base > .5)).sum()),
                        score_semantics='class-balanced Gaussian density contrast; no query-prior fitting or probability calibration')
    info['wall_seconds'] = time.perf_counter()-started
    return dict(**fields, info=info)
