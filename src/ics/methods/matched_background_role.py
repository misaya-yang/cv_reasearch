"""Frozen full-cloud and matched-background kernel role readouts.

Classical nonparametric hard-negative statistics, not new relationship evidence.
The inputs retain the parent MEAN's CPU FP32 processing and unit normalization.
Reference hard-negative weights exclude same-position foreground matches and
use one common reference-kernel scale, so relative background mass is retained.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


TAU = 0.07


def _validate(q: torch.Tensor, r: torch.Tensor, cov: np.ndarray):
    if not isinstance(q, torch.Tensor) or not isinstance(r, torch.Tensor):
        raise TypeError('q/r must be the parent normalized torch tensors')
    if q.dtype != torch.float32 or r.dtype != torch.float32:
        raise TypeError('q/r must retain parent FP32 processing')
    if q.device.type != 'cpu' or r.device.type != 'cpu':
        raise ValueError('The locked input processing is on CPU')
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError('Expected q[N,C] and r[M,C] with matching channels')
    n, m = q.shape[0], r.shape[0]
    side = math.isqrt(n)
    if not n or not m or not q.shape[1] or side * side != n:
        raise ValueError('A nonempty square query grid is required')
    if not torch.isfinite(q).all() or not torch.isfinite(r).all():
        raise ValueError('Nonfinite features')
    if (q.norm(dim=1) - 1).abs().max() > 1e-3 or (r.norm(dim=1) - 1).abs().max() > 1e-3:
        raise ValueError('Inputs must already be unit normalized; no renormalization occurs')
    coverage = np.asarray(cov)
    if coverage.ndim != 2 or coverage.size != m:
        raise ValueError('Coverage must match the two-dimensional reference grid')
    if not np.isfinite(coverage).all() or coverage.min() < 0 or coverage.max() > 1:
        raise ValueError('Coverage must be finite and in [0,1]')
    return np.asarray(coverage, dtype=np.float32).ravel(), (side, side)


@torch.inference_mode()
def predict(q: torch.Tensor, r: torch.Tensor, cov: np.ndarray) -> dict[str, Any]:
    """Return 64-square posterior fields and hard-background diagnostics.

    Smaller square query grids are supported for independent algebra checks.
    There is no query label, encoder, parameter search, graph, or mask readout.
    """
    coverage, query_shape = _validate(q, r, cov)
    reference_shape = np.asarray(cov).shape
    fg_mass = float(coverage.sum(dtype=np.float64))
    bg_mass = float((1.0 - coverage).sum(dtype=np.float64))
    info: dict[str, Any] = {
        'tau': TAU, 'fg_mass': fg_mass, 'bg_mass': bg_mass,
        'self_excluded': True,
        'reference_kernel_scaling': 'one global maximum before exponentiation',
        'query_kernel_scaling': 'one maximum per query; common to both roles',
        'query_GT_in_inference': False, 'encoder_calls': 0,
        'source_re_normalized': False,
        'reference_tokens': int(r.shape[0]),
        'omega_mass': 0.0, 'omega_nonzero_tokens': 0,
        'omega_effective_tokens': 0.0, 'omega_mixed_weight_fraction': 0.0,
        'global_fallback': None, 'matched_fallback': None,
    }
    if fg_mass == 0 or bg_mass == 0:
        info.update(global_fallback='missing_reference_role_constant_half',
                    matched_fallback='missing_reference_role_constant_half',
                    reference_kernel_computed=False)
        p = np.full(query_shape, 0.5, dtype=np.float32)
        return {'p_global': p, 'p_matched': p.copy(),
                'omega': np.zeros(reference_shape, dtype=np.float32),
                'diagnostics': info}

    c = torch.from_numpy(coverage.copy())
    kernel_qr = q @ r.T
    kernel_qr.sub_(kernel_qr.max(dim=1, keepdim=True).values).div_(TAU).exp_()
    density_fg = (kernel_qr @ c) / fg_mass
    density_bg_global = (kernel_qr @ (1.0 - c)) / bg_mass
    p_global = density_fg / (density_fg + density_bg_global)

    # The scale here must be common across every background position. A
    # per-row maximum would change the relative hard-negative weights.
    kernel_rr = r @ r.T
    global_max = float(kernel_rr.max())
    kernel_rr.sub_(global_max).div_(TAU).exp_()
    kernel_rr.fill_diagonal_(0.0)
    omega = (1.0 - c) * ((kernel_rr @ c) / fg_mass)
    del kernel_rr
    omega_np = omega.numpy()
    omega_mass = float(omega_np.sum(dtype=np.float64))
    info.update(reference_kernel_computed=True, reference_kernel_global_max=global_max,
                omega_mass=omega_mass, omega_nonzero_tokens=int((omega_np > 0).sum()))
    if omega_mass == 0:
        # No off-diagonal negative evidence: preserve the complete-cloud readout.
        p_matched = p_global.clone()
        info['matched_fallback'] = 'no_offdiagonal_background_mass_reuse_global'
    else:
        density_bg_matched = (kernel_qr @ omega) / omega_mass
        p_matched = density_fg / (density_fg + density_bg_matched)
        omega64 = omega_np.astype(np.float64)
        info['omega_effective_tokens'] = float(omega_mass ** 2 / np.dot(omega64, omega64))
        info['omega_max_normalized_weight'] = float(omega64.max() / omega_mass)
        mixed = (coverage > 0) & (coverage < 1)
        info['omega_mixed_weight_fraction'] = float(omega64[mixed].sum() / omega_mass)

    if not torch.isfinite(p_global).all() or not torch.isfinite(p_matched).all():
        raise RuntimeError('Nonfinite kernel posterior')
    if not torch.isfinite(omega).all() or (omega < 0).any():
        raise RuntimeError('Invalid matched-background weights')
    pg = p_global.numpy().reshape(query_shape).astype(np.float32)
    pm = p_matched.numpy().reshape(query_shape).astype(np.float32)
    info.update(p_global_min=float(pg.min()), p_global_max=float(pg.max()),
                p_matched_min=float(pm.min()), p_matched_max=float(pm.max()))
    return {'p_global': pg, 'p_matched': pm,
            'omega': omega_np.reshape(reference_shape).astype(np.float32),
            'diagnostics': info}
