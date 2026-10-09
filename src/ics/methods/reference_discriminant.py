"""Class-balanced full-background ridge reference discriminant.

The FP32 role means/delta are the same frozen erasure raw-control statistics.
Only their direction is changed by a classical FP64 covariance solve. This is
not new relational evidence, an optimal-LDA claim, or an established IoU gain.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .reference_erasure_guide import _role_means


@torch.inference_mode()
def solve_direction(covariance: torch.Tensor, delta: torch.Tensor):
    """Solve (Sigma+trace(Sigma)/C I)v=delta, no epsilon-based event choices."""
    if covariance.dtype != torch.float64 or delta.dtype != torch.float64:
        raise ValueError('Direction solve requires FP64 covariance and delta')
    c = len(delta)
    if covariance.shape != (c, c) or not torch.isfinite(covariance).all() or not torch.isfinite(delta).all():
        raise ValueError('Invalid finite covariance/delta')
    ridge = float(covariance.diagonal().sum() / c)
    if ridge < 0:
        raise ValueError('Class within-covariance has negative trace')
    if torch.count_nonzero(delta) == 0:
        return torch.zeros_like(delta), dict(ridge=ridge, fallback='zero_delta_reuse_raw_guide', residual=None)
    if ridge == 0:
        return torch.zeros_like(delta), dict(ridge=ridge, fallback='zero_trace_reuse_raw_guide', residual=None)
    system = covariance + ridge * torch.eye(c, dtype=torch.float64)
    vector = torch.linalg.solve(system, delta)
    residual = float((system @ vector - delta).norm() / delta.norm())
    if not torch.isfinite(vector).all() or vector.norm() == 0:
        raise RuntimeError('Invalid nonzero discriminant solution')
    return vector, dict(ridge=ridge, fallback=None, residual=residual)


@torch.inference_mode()
def predict(q_parent: torch.Tensor, reference: torch.Tensor, cov: np.ndarray,
            parent_guide: np.ndarray):
    """Full roles, original FP32 centers, FP64 class-balanced covariance."""
    if any(t.dtype != torch.float32 or t.device.type != 'cpu' or t.ndim != 2 or not torch.isfinite(t).all()
           for t in (q_parent, reference)):
        raise ValueError('Use preserved finite CPU FP32 q/reference')
    if q_parent.shape[1] != reference.shape[1] or (q_parent.norm(dim=1)-1).abs().max() > 1e-3:
        raise ValueError('Require actual unit-normalized parent query')
    coverage = np.asarray(cov, dtype=np.float32)
    if coverage.ndim != 2 or coverage.size != len(reference) or not np.isfinite(coverage).all() or coverage.min() < 0 or coverage.max() > 1:
        raise ValueError('Invalid complete reference coverage')
    weights = torch.from_numpy(coverage.ravel().copy())
    fg_mass = float(weights.sum(dtype=torch.float64))
    bg_weights = 1-weights
    bg_mass = float(bg_weights.sum(dtype=torch.float64))
    c = reference.shape[1]
    info = dict(fg_mass=fg_mass, bg_mass=bg_mass, channels=c,
                covariance_center='original FP32 erasure means converted to FP64, not recomputed means',
                class_balance=[.5, .5], ridge_formula='trace(Sigma)/C',
                encoder_calls=0, query_GT_in_inference=False)
    old = np.asarray(parent_guide, dtype=np.float32).reshape(-1)
    if old.size != len(q_parent) or not np.isfinite(old).all():
        raise ValueError('Require original parent guide for explicit empty/zero-role fallback')
    if fg_mass == 0 or bg_mass == 0:
        mean_fg = mean_bg = delta = torch.zeros(c, dtype=torch.float32)
        covariance = torch.zeros(c, c, dtype=torch.float64)
        raw_guide = old.copy()
        vector = torch.zeros(c, dtype=torch.float64)
        unit = vector.clone()
        guide = old.copy()
        solve_info = dict(ridge=0., fallback='missing_reference_role_reuse_parent_guide', residual=None)
    else:
        mean_fg, mean_bg, delta = _role_means(reference, weights, fg_mass, bg_mass)
        raw_guide = old.copy() if torch.count_nonzero(delta) == 0 else (q_parent @ F.normalize(delta, dim=0)).numpy()
        x = reference.double()
        fg_centered, bg_centered = x-mean_fg.double(), x-mean_bg.double()
        covariance = .5 * ((fg_centered.T @ (fg_centered * weights.double()[:, None])) / fg_mass
                         + (bg_centered.T @ (bg_centered * bg_weights.double()[:, None])) / bg_mass)
        vector, solve_info = solve_direction(covariance, delta.double())
        if solve_info['fallback'] is not None:
            unit = torch.zeros_like(vector)
            guide = raw_guide.copy()
        else:
            unit = vector/vector.norm()
            guide = (q_parent @ unit.float()).numpy()
    info.update(solve_info, trace=float(covariance.diagonal().sum()),
                mean_fg_norm=float(mean_fg.norm()), mean_bg_norm=float(mean_bg.norm()),
                delta_FP32_norm=float(delta.norm()), solution_FP64_norm=float(vector.norm()),
                covariance_symmetry_max=float((covariance-covariance.T).abs().max()))
    return dict(raw_guide=np.asarray(raw_guide, dtype=np.float32), guide=np.asarray(guide, dtype=np.float32),
                mean_fg=mean_fg.numpy(), mean_bg=mean_bg.numpy(), delta=delta.numpy(),
                covariance=covariance.numpy(), ridge=np.asarray(solve_info['ridge'], dtype=np.float64),
                vector=vector.numpy(), unit_vector_FP64=unit.numpy(), unit_vector_FP32=unit.float().numpy(),
                diagnostics=info)
