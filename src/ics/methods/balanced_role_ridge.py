"""Fixed paired ridge controls: coverage target versus balanced binary roles.

Inputs are the caller's existing CPU reference representation and continuous
coverage. This module does no normalization, feature extraction, query readout
or role selection beyond the historical 128-quantile-per-role sampling recipe.
The balanced target is an experiment hypothesis, not a segmentation gain.
"""
from __future__ import annotations

import numpy as np
import torch

PER_ROLE_SAMPLES = 128
RIDGE_STRENGTH = .01


def _cpu_double(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('Require CPU reference/features; no device transfer')
        result = value.detach().double()
    else:
        result = torch.from_numpy(np.asarray(value).copy()).double()
    if not torch.isfinite(result).all():
        raise ValueError('Require finite CPU features')
    return result


@torch.inference_mode()
def weighted_ridge(x: torch.Tensor, target: torch.Tensor,
                   weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Historical FP64 dual ridge, lambda=.01, unregularized intercept.

    Minimize sum_j a_j (x_j w+b-y_j)^2 + .01 ||w||^2. Center using the
    supplied weights and solve xc xc^T + diag(.01/a). Every target fit is
    explicitly evaluated; no coefficient/target substitution is performed.
    """
    x, y, a = (_cpu_double(v) for v in (x, target, weights))
    if x.ndim != 2 or y.shape != (len(x),) or a.shape != y.shape or len(x) == 0:
        raise ValueError('Require matching nonempty x[N,D], target[N], weights[N]')
    if x.shape[1] == 0 or (a <= 0).any():
        raise ValueError('Require positive channel count and strictly positive weights')
    mx = (a[:, None] * x).sum(0) / a.sum()
    my = (a * y).sum() / a.sum()
    xc, yc = x - mx, y - my
    matrix = xc @ xc.T + torch.diag(RIDGE_STRENGTH / a.clamp_min(1e-15))
    coefficient = xc.T @ torch.linalg.solve(matrix, yc)
    bias = my - mx @ coefficient
    if not torch.isfinite(coefficient).all() or not torch.isfinite(bias):
        raise RuntimeError('Nonfinite fixed ridge solution')
    return dict(coefficient=coefficient, bias=bias)


@torch.inference_mode()
def fit_pair(reference: torch.Tensor | np.ndarray,
             coverage: np.ndarray) -> dict:
    """Fit both objectives with identical historical ids and sample weights.

    C/B are selected-sample masses after quantile-id deduplication, not full-R
    masses. The source coverage dtype is retained for historical cumsum/id
    parity; role weights and both independent fits are evaluated in FP64.
    """
    if isinstance(reference, torch.Tensor) and reference.device.type != 'cpu':
        raise ValueError('Require CPU reference; no device transfer')
    c_full = np.asarray(coverage).reshape(-1)
    if not len(c_full) or not np.isfinite(c_full).all() or (c_full < 0).any() or (c_full > 1).any():
        raise ValueError('Require finite nonempty continuous reference coverage [0,1]')
    x_full = _cpu_double(reference)
    if x_full.ndim != 2 or len(x_full) != len(c_full) or x_full.shape[1] == 0:
        raise ValueError('Require matching reference[N,D] and coverage[N]')
    if not c_full.sum(dtype=np.float64) > 0:
        raise ValueError('Missing foreground role in full reference')
    if not (1-c_full).sum(dtype=np.float64) > 0:
        raise ValueError('Missing background role in full reference')
    ids = []
    for role in (c_full, 1-c_full):
        cumulative = np.cumsum(role)
        ids.extend(np.searchsorted(cumulative, (np.arange(PER_ROLE_SAMPLES)+.5)
                                   * cumulative[-1]/PER_ROLE_SAMPLES, side='left'))
    ids = np.unique(ids)
    if (ids < 0).any() or (ids >= len(c_full)).any():
        raise ValueError('Historical quantile sampling produced invalid ids')
    x = x_full[ids]
    c = torch.from_numpy(c_full[ids].copy()).double()
    C, B = c.sum(), (1-c).sum()
    if not C > 0 or not B > 0:
        raise ValueError('Missing foreground/background role in selected reference')
    fg_role, bg_role = c/C, (1-c)/B
    weights = .5*fg_role + .5*bg_role
    y_old = 2*c-1
    y_star = (fg_role-bg_role)/(fg_role+bg_role)
    old_fit = weighted_ridge(x, y_old, weights)
    balanced_fit = weighted_ridge(x, y_star, weights)
    return dict(fits=dict(old_coverage=old_fit, balanced_role=balanced_fit),
        ids=ids, weights=weights, c_selected=c, y_old=y_old, y_star=y_star,
        FG_mass=float(C), BG_mass=float(B), selected_pi=float(C/len(c)),
        role_weights_FG=fg_role, role_weights_BG=bg_role,
        collapse_constant=float((weights*(1-y_star*y_star)).sum()),
        old_target_weighted_mean=float((weights*y_old).sum()/weights.sum()),
        balanced_target_weighted_mean=float((weights*y_star).sum()/weights.sum()),
        selection_coverage_dtype=c_full.dtype.str, fit_tokens=len(ids),
        per_role_samples=PER_ROLE_SAMPLES, ridge_strength=RIDGE_STRENGTH,
        weight_clamp_active=bool((weights < 1e-15).any()), intercept_regularized=False,
        input_preprocessing='caller-preserved representation; no implicit L2/APD/other transform',
        target_mass_scope='same selected quantile-id-deduplicated sample for both targets',
        role_column_order='foreground/background named arrays, no unnamed packed role columns')
