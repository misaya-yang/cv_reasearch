"""Balanced reference-supervised degree-two kernel classifier; unvalidated.

The kernel compares individual query tokens with reference mode anchors. This
is distinct from matching covariance of entire query connected components.
Its regression output is a score, not a calibrated semantic probability.
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
    ridge: float = .01
    subspace_relative_tolerance: float = 1e-6


def solve_kernel(gram, labels, weights, ridge):
    gram = np.asarray(gram, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (gram.shape != (len(labels), len(labels)) or weights.shape != labels.shape
            or not np.isfinite(gram).all() or not np.isfinite(labels).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0) or ridge <= 0):
        raise ValueError('Finite Gram, labels and positive weights/ridge required')
    matrix = gram + np.diag(ridge / weights)
    coefficient = np.linalg.solve(matrix, labels)
    reconstructed = np.einsum('ij,j->i', matrix, coefficient, optimize=False)
    residual = np.linalg.norm(reconstructed - labels) / max(np.linalg.norm(labels), 1e-12)
    return coefficient, dict(relative_linear_residual=float(residual),
                             minimum_regularized_eigenvalue=float(np.linalg.eigvalsh(matrix).min()))


def subspace(anchors, tolerance):
    _, values, vectors = np.linalg.svd(anchors.astype(np.float64), full_matrices=False)
    keep = values > tolerance * values[0]
    return vectors[keep]


def predict(q, r, coverage, base, cfg=Config()):
    started = time.perf_counter()
    if (cfg.foreground_modes < 1 or cfg.background_modes < 1 or cfg.lloyd_steps < 1
            or not .5 < cfg.purity <= 1 or cfg.minimum_reference_samples < 2
            or cfg.ridge <= 0 or not 0 < cfg.subspace_relative_tolerance < 1):
        raise ValueError('Invalid quadratic reference configuration')
    q, r = unit(q), unit(r)
    base = np.asarray(base, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=np.float64)
    if (base.ndim != 2 or base.shape != coverage.shape or base.size != len(q) or len(q) != len(r)
            or q.shape[1] != r.shape[1] or not np.isfinite(base).all() or not np.isfinite(coverage).all()
            or coverage.min() < 0 or coverage.max() > 1):
        raise ValueError('Aligned finite query/reference/mask/base required')
    fields = {key: base.copy() for key in ('field', 'linear_control', 'homogeneous_control',
                                         'kernel_mean_control', 'nearest_control', 'subspace_control')}
    info = dict(config=asdict(cfg), abstention=True, query_gt_used=False, new_encoder_forwards=0,
                real_gain='unmeasured', complete_dataset_minutes='unmeasured')
    fg = coverage.ravel() >= cfg.purity
    bg = (1 - coverage.ravel()) >= cfg.purity
    if min(int(fg.sum()), int(bg.sum())) < cfg.minimum_reference_samples:
        info['reason'] = 'missing_reference_foreground_or_background_samples'
    else:
        foreground, _, fg_mass = cluster(r[fg], cfg.foreground_modes, cfg.lloyd_steps, coverage.ravel()[fg])
        background, _, bg_mass = cluster(r[bg], cfg.background_modes, cfg.lloyd_steps, 1 - coverage.ravel()[bg])
        anchors = np.concatenate((foreground, background)).astype(np.float64)
        nf = len(foreground)
        weights = .5 * np.r_[fg_mass, bg_mass]
        labels = np.r_[np.ones(nf), -np.ones(len(background))]
        cosine_gram = np.einsum('id,jd->ij', anchors, anchors, optimize=False)
        quadratic, quadratic_info = solve_kernel(((1 + cosine_gram)*.5)**2, labels, weights, cfg.ridge)
        homogeneous, homogeneous_info = solve_kernel(cosine_gram**2, labels, weights, cfg.ridge)
        linear, linear_info = solve_kernel(cosine_gram, labels, weights, cfg.ridge)
        response = np.einsum('nd,kd->nk', q, anchors, optimize=False)

        def field(score):
            return np.clip(.5 + .5 * score.reshape(base.shape), 0, 1)

        fields['field'] = field(np.einsum('nk,k->n', ((1 + response)*.5)**2, quadratic, optimize=False))
        fields['homogeneous_control'] = field(np.einsum('nk,k->n', response**2, homogeneous, optimize=False))
        fields['linear_control'] = field(np.einsum('nk,k->n', response, linear, optimize=False))
        fields['kernel_mean_control'] = field(np.einsum('nk,k->n', ((1 + response)*.5)**2, weights * labels, optimize=False))
        fields['nearest_control'] = field(response[:, :nf].max(1) - response[:, nf:].max(1))
        bases = [subspace(bank, cfg.subspace_relative_tolerance) for bank in (foreground, background)]
        energy = []
        for basis in bases:
            projection = np.einsum('nd,kd->nk', q, basis, optimize=False)
            energy.append(np.sum(projection**2, axis=1))
        fields['subspace_control'] = field(energy[0] - energy[1])
        info.update(abstention=False, foreground_modes=nf, background_modes=len(background),
                    quadratic_solver=quadratic_info, linear_solver=linear_info,
                    homogeneous_control_solver=homogeneous_info,
                    subspace_dimensions=[len(basis) for basis in bases],
                    score_semantics='signed kernel regression; clipping is a readout, not probability calibration')
    info['wall_seconds'] = time.perf_counter() - started
    return dict(**fields, info=info)
