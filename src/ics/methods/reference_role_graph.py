"""Reference FG/BG conditional edge control for the frozen MEAN graph.

This is an exploratory regularizer, not a claim of new relational information.
It keeps the caller's unary, confidence, graph normalization, and solver settings.
Production inputs are 4096 query/reference tokens of dimension 1024; smaller
square grids are accepted so the same numerical path has independent checks.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import cg


TAU = 0.07
LAMBDA = 16.0


def _validate(q, r, cov, w, a, y):
    if not isinstance(q, torch.Tensor) or not isinstance(r, torch.Tensor):
        raise TypeError('q and r must be the already normalized torch tensors')
    if q.dtype != torch.float32 or r.dtype != torch.float32:
        raise TypeError('q and r must retain the parent FP32 processing')
    if q.device.type != 'cpu' or r.device.type != 'cpu':
        raise ValueError('The locked parent MEAN processing is on CPU')
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError('Expected q[N,C] and r[M,C] with matching channels')
    n, m = q.shape[0], r.shape[0]
    side = math.isqrt(n)
    if not n or not m or not q.shape[1] or side * side != n:
        raise ValueError('A nonempty square query grid is required')
    cov = np.asarray(cov)
    if cov.ndim != 2 or cov.size != m:
        raise ValueError('Reference coverage must match its two-dimensional grid')
    if not np.isfinite(cov).all() or cov.min() < 0 or cov.max() > 1:
        raise ValueError('Reference coverage must be finite and in [0,1]')
    if not torch.isfinite(q).all() or not torch.isfinite(r).all():
        raise ValueError('Nonfinite reference/query features')
    # No re-normalization: preserve exactly the caller's processed features.
    if (q.norm(dim=1) - 1).abs().max() > 1e-3 or (r.norm(dim=1) - 1).abs().max() > 1e-3:
        raise ValueError('q/r must already be unit normalized by parent MEAN')
    if not sparse.isspmatrix_csr(w) or w.shape != (n, n):
        raise ValueError('w must be the parent CSR graph of shape [N,N]')
    if w.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise TypeError('w must preserve a floating-point graph dtype')
    if not np.isfinite(w.data).all() or (w.data < 0).any():
        raise ValueError('Graph weights must be finite and nonnegative')
    symmetry = w - w.T
    if symmetry.nnz and np.max(np.abs(symmetry.data)) > 1e-7:
        raise ValueError('Parent reciprocal graph must be symmetric')
    a, y = np.asarray(a), np.asarray(y)
    if a.shape != (n,) or y.shape != (n,) or a.dtype != np.float64 or y.dtype != np.float64:
        raise ValueError('a and y must be the parent float64 vectors [N]')
    if not np.isfinite(a).all() or not np.isfinite(y).all() or (a <= 0).any():
        raise ValueError('Finite unary and strictly positive confidence are required')
    return cov, a, y, (side, side)


@torch.inference_mode()
def _role_posterior(q: torch.Tensor, r: torch.Tensor, cov: np.ndarray):
    coverage = np.asarray(cov, dtype=np.float32).ravel()
    fg_mass = float(coverage.sum(dtype=np.float64))
    bg_mass = float((1.0 - coverage).sum(dtype=np.float64))
    if fg_mass == 0 or bg_mass == 0:
        return np.full(q.shape[0], 0.5, dtype=np.float32), {
            'fg_mass': fg_mass, 'bg_mass': bg_mass,
            'role_fallback': 'missing_reference_role_constant_half',
        }
    kernel = q @ r.T
    kernel.sub_(kernel.max(dim=1, keepdim=True).values).div_(TAU).exp_()
    weights = torch.from_numpy(coverage.copy())
    density_fg = (kernel @ weights) / fg_mass
    density_bg = (kernel @ (1.0 - weights)) / bg_mass
    p = (density_fg / (density_fg + density_bg)).numpy()
    if not np.isfinite(p).all():
        raise RuntimeError('Nonfinite reference role density posterior')
    return p, {'fg_mass': fg_mass, 'bg_mass': bg_mass, 'role_fallback': None}


def _solve(w: sparse.csr_matrix, a: np.ndarray, y: np.ndarray):
    # Intentionally follow locked MEAN's degree dtype and algebraic order.
    degree = np.asarray(w.sum(1)).ravel()
    h = sparse.diags(a) + LAMBDA * (sparse.diags(degree) - w)
    rhs = a * y
    iterations = [0]

    def callback(_):
        iterations[0] += 1

    z, status = cg(h, rhs, x0=y, rtol=1e-7, atol=1e-9,
                   maxiter=300, callback=callback)
    if status:
        raise RuntimeError('Reference role graph CG failure: ' + str(status))
    if not np.isfinite(z).all():
        raise RuntimeError('Nonfinite reference role graph solution')
    return z, {
        'cg_iterations': iterations[0],
        'cg_relative_residual': float(np.linalg.norm(h @ z - rhs) / max(np.linalg.norm(rhs), 1e-12)),
    }


def _refine(w: sparse.csr_matrix, a: np.ndarray, y: np.ndarray,
            p: np.ndarray, grid_shape: tuple[int, int]):
    if np.ptp(p) == 0:
        # This bypass preserves original graph dtype/order for identity checks.
        role_w, uniform_w, ratio = w, w, 1.0
    elif not w.nnz or float(w.sum()) == 0:
        role_w, uniform_w, ratio = w, w, 1.0
    else:
        role_w = w.copy()
        rows = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
        factors = (1.0 - np.abs(p[rows] - p[w.indices])).astype(w.dtype)
        role_w.data *= factors
        # Do not normalize the attenuated graph's mean degree back to one.
        ratio = float(role_w.sum()) / float(w.sum())
        uniform_w = w * np.asarray(ratio, dtype=w.dtype)
    z_role, role_info = _solve(role_w, a, y)
    z_uniform, uniform_info = _solve(uniform_w, a, y)
    parent_sum = float(w.sum())
    info = {
        'parent_weight_sum': parent_sum,
        'role_weight_sum': float(role_w.sum()),
        'uniform_weight_sum': float(uniform_w.sum()),
        'uniform_weight_ratio': ratio,
        'graph_undirected_edges': int(w.nnz // 2),
        'role_graph': role_info, 'uniform_graph': uniform_info,
        'p_min': float(p.min()), 'p_max': float(p.max()),
        'p_mean': float(p.mean()), 'p_std': float(p.std()),
        'constant_role_bypass': bool(np.ptp(p) == 0),
    }
    return {
        'role_graph': z_role.reshape(grid_shape).astype(np.float32),
        'uniform_graph': z_uniform.reshape(grid_shape).astype(np.float32),
        'p': p.reshape(grid_shape).astype(np.float32),
        'info': info,
    }


@torch.inference_mode()
def predict(q: torch.Tensor, r: torch.Tensor, cov: np.ndarray,
            w: sparse.csr_matrix, a: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Return role/uniform fields, saved role probabilities, and diagnostics.

    No query labels, area estimates, learned parameters, encoder, min-max, or
    mask finalizer occur here. The caller must use the locked final readout.
    """
    cov, a, y, grid_shape = _validate(q, r, cov, w, a, y)
    p, density_info = _role_posterior(q, r, cov)
    result = _refine(w, a, y, p, grid_shape)
    result['info'].update(density_info)
    result['info'].update(tau=TAU, lambda_value=LAMBDA,
                          query_GT_in_inference=False, encoder_calls=0,
                          graph_renormalized_after_gating=False)
    return result
