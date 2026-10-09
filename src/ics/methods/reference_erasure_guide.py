"""Mask-defined reference erasure guides with a fixed parent MEAN readout.

The views alter reference pixels before encoding. They are not token masking,
zero-cost reuse, independent relationship evidence, or a segmentation method
with an established gain. This module neither opens labels nor builds models.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def build_reference_views(image: torch.Tensor, mask: torch.Tensor):
    """Erase the true and fixed half-period shifted masks in normalized space."""
    if image.dtype != torch.float32 or image.device.type != 'cpu':
        raise ValueError('Views require the original CPU FP32 normalized image')
    if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
        raise ValueError('Expected a finite CHW RGB image')
    if mask.dtype != torch.bool or mask.device.type != 'cpu' or mask.shape != image.shape[1:]:
        raise ValueError('Expected the original nearest-resized Boolean reference mask')
    h, w = mask.shape
    shifted = torch.roll(mask, (h // 2, w // 2), dims=(0, 1))
    views = torch.stack((image.masked_fill(mask[None], 0),
                         image.masked_fill(shifted[None], 0)))
    area = int(mask.sum())
    overlap = int((mask & shifted).sum())
    assert int(shifted.sum()) == area
    info = dict(view_order=['true_fg_erasure', 'shifted_mask_erasure'],
                erased_value=0.0, erased_space='original normalized FP32 RGB tensor',
                shift_hw=[h // 2, w // 2], mask_area=area,
                shifted_mask_area=int(shifted.sum()), shifted_overlap_fg_pixels=overlap,
                shifted_overlap_fg_fraction=None if not area else overlap / area,
                shifted_is_guaranteed_background=False, query_GT_in_inference=False)
    return views, shifted, info


@torch.inference_mode()
def process_reference(raw_tokens: torch.Tensor | np.ndarray, basis: torch.Tensor,
                      apd_applied: bool, grid=(64, 64)) -> torch.Tensor:
    """Preserve source NCHW channel reductions and explicit P_perp matmul.

The return is FP32. The caller must not round full/view outputs to FP16 before
their subtraction. Query processing and the original APD decision are untouched.
"""
    raw = torch.as_tensor(raw_tokens)
    h, w = grid
    if raw.dtype != torch.float32 or raw.device.type != 'cpu' or raw.ndim != 2:
        raise ValueError('Require unmodified CPU FP32 raw [N,C] outputs')
    if raw.shape[0] != h * w or not torch.isfinite(raw).all():
        raise ValueError('Raw tokens do not match a finite grid')
    c = raw.shape[1]
    if basis.dtype != torch.float32 or basis.device.type != 'cpu' or basis.shape[0] != c:
        raise ValueError('Use the preserved native FP32 basis for these channels')
    maps = raw.reshape(h, w, c).permute(2, 0, 1).contiguous()[None, None]
    maps = F.normalize(maps, p=2, dim=2)
    if apd_applied:
        p_perp = torch.eye(c, dtype=maps.dtype) - basis @ basis.T
        maps = torch.matmul(p_perp[None], maps.reshape(1, c, h * w)).reshape(1, 1, c, h, w)
        maps = F.normalize(maps, p=2, dim=2)
    # The old runner normalizes the observed Part1 maps a second time before
    # flattening and FP16 storage. Keep that reduction but retain FP32 here.
    maps = F.normalize(maps[0].float(), dim=1)
    return maps[0].flatten(1).T.contiguous()


def _role_means(tokens: torch.Tensor, coverage: torch.Tensor,
                fg_mass: float, bg_mass: float):
    foreground = (tokens * coverage[:, None]).sum(0) / fg_mass
    background = (tokens * (1 - coverage)[:, None]).sum(0) / bg_mass
    return foreground, background, foreground - background


@torch.inference_mode()
def build_guides(q_parent: torch.Tensor, full: torch.Tensor, erased: torch.Tensor,
                 shifted: torch.Tensor, cov: np.ndarray,
                 parent_guide: np.ndarray) -> dict[str, Any]:
    """Three matched role contrasts; all reference differences occur in FP32."""
    tensors = (q_parent, full, erased, shifted)
    if any(t.dtype != torch.float32 or t.device.type != 'cpu' or t.ndim != 2
           or not torch.isfinite(t).all() for t in tensors):
        raise ValueError('All processed tokens must be finite CPU FP32 [N,C]')
    if full.shape != erased.shape or full.shape != shifted.shape or full.shape[1] != q_parent.shape[1]:
        raise ValueError('Reference views must retain the same grid/channels')
    if (q_parent.norm(dim=1) - 1).abs().max() > 1e-3:
        raise ValueError('Use the actual already normalized parent query')
    c_np = np.asarray(cov, dtype=np.float32)
    if c_np.ndim != 2 or c_np.size != full.shape[0] or not np.isfinite(c_np).all() or c_np.min() < 0 or c_np.max() > 1:
        raise ValueError('Coverage must be the complete valid reference grid')
    old = np.asarray(parent_guide, dtype=np.float32).reshape(-1)
    if old.size != q_parent.shape[0] or not np.isfinite(old).all():
        raise ValueError('Require the original MEAN guide for explicit role fallbacks')
    coverage = torch.from_numpy(c_np.ravel().copy())
    fg_mass = float(coverage.sum(dtype=torch.float64))
    bg_mass = float((1 - coverage).sum(dtype=torch.float64))
    delta_true = full - erased
    delta_shift = full - shifted
    arrays = {'coverage': c_np.copy()}
    info = dict(fg_mass=fg_mass, bg_mass=bg_mass,
                subtraction_dtype='float32; no pre-subtraction FP16 rounding',
                query_parent_preserved=True, original_APD_policy_preserved=True,
                query_GT_in_inference=False, guides={})
    samples = {'raw': full, 'true': delta_true, 'shift': delta_shift}
    raw_guide = old.copy()
    for name, sample in samples.items():
        fallback = None
        if fg_mass == 0 or bg_mass == 0:
            foreground = background = vector = torch.zeros(full.shape[1], dtype=torch.float32)
            unit = vector.clone()
            guide = old.copy()
            fallback = 'missing_reference_role_reuse_parent_MEAN_guide'
        else:
            foreground, background, vector = _role_means(sample, coverage, fg_mass, bg_mass)
            norm = float(vector.norm())
            unit = F.normalize(vector, dim=0)
            if name != 'raw' and torch.count_nonzero(sample) == 0:
                guide = raw_guide.copy()
                fallback = 'exactly_zero_delta_reuse_raw_contrast_guide'
            elif norm == 0:
                guide = old.copy() if name == 'raw' else raw_guide.copy()
                fallback = 'zero_contrast_reuse_parent_guide' if name == 'raw' else 'zero_contrast_reuse_raw_guide'
            else:
                guide = (q_parent @ unit).numpy().astype(np.float32)
        if name == 'raw':
            raw_guide = guide.copy()
        arrays['guide_' + name] = guide
        arrays['foreground_mean_' + name] = foreground.numpy()
        arrays['background_mean_' + name] = background.numpy()
        arrays['contrast_' + name] = vector.numpy()
        arrays['unit_contrast_' + name] = unit.numpy()
        info['guides'][name] = dict(fallback=fallback, delta_all_zero=bool(torch.count_nonzero(sample) == 0),
                                  foreground_norm=float(foreground.norm()),
                                  background_norm=float(background.norm()),
                                  contrast_norm=float(vector.norm()), unit_contrast_norm=float(unit.norm()),
                                  input_token_norm_mean=float(sample.norm(dim=1).mean()))
    return dict(arrays=arrays, diagnostics=info, delta_true=delta_true, delta_shift=delta_shift)


def parent_graph_readout(guide: np.ndarray, s: np.ndarray, a: np.ndarray, h):
    """Use the actual parent H/A/s and its fixed rank/CG equations unchanged."""
    from scipy.sparse.linalg import cg
    from .rcg import rank
    guide = np.asarray(guide, dtype=np.float32).ravel()
    s = np.asarray(s)
    a = np.asarray(a)
    if s.dtype != np.float32 or a.dtype != np.float64 or guide.shape != s.shape or a.shape != s.shape:
        raise ValueError('Require parent float32 s/guide and float64 A')
    if h.shape != (len(s), len(s)) or not np.isfinite(guide).all():
        raise ValueError('Invalid actual parent system')
    y = (s + .25 * (rank(guide) - rank(s))).astype(np.float64)
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    z, status = cg(h, a * y, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
    if status:
        raise RuntimeError('Reference erasure readout CG failed: ' + str(status))
    info = dict(alpha=.25, lambda_value=16., cg_rtol=1e-7, cg_atol=1e-9, cg_maxiter=300,
                cg_iterations=iterations[0],
                cg_relative_residual=float(np.linalg.norm(h @ z - a * y) / max(np.linalg.norm(a * y), 1e-12)),
                query_GT_in_inference=False, graph='same actual parent H/A; no new graph')
    return z.astype(np.float32), y, info
