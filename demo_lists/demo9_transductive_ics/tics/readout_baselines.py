"""Portable, independently implemented RSRM/FROST mathematical components.

Source contracts audited 2026-10-02:
RSRM: https://github.com/Sparkling-Water/RSRM/tree/4173c9fd38b4156b9d9200dd6ef941d88eca91bd
  model/rsrm.py; dinov3/dinov3/{layers/block,models/vision_transformer}.py
FROST: https://github.com/jhpark-ai/FROST/tree/b9ece69d7495a698c298e7cc3d16efacd4497a43
  frost/model.py, frost/density.py (official implementation: Apache-2.0).
Equations are implemented independently, without copying third-party source.

This is an adapted same-backbone component control, NOT a full reproduction:
no backbone loading, downloads, RSRM ASE/HPM, or implicit positional basis.
The caller supplies already normalized raw-attention maps and lawful support
masks. No query label argument exists. Native and counterfactual comparisons
must share support weights and each image's NATIVE scales.
Statistics use float32, matching the source hub feature path. Converting a
stored FP16 trace to float32 does not undo its serialization rounding.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class SAFRState:
    weights: Tensor                    # [B,L]
    valid: Tensor                      # [B], explicit safe guard for source degeneracy
    position_variance_rates: tuple     # support views, each [B,L]
    semantic_layers: tuple             # support views, each [L], source batch-any


def raw_attention_readout(projected_update: Tensor, layer_scale, final_norm) -> Tensor:
    """Post-attention projection/dropout -> LayerScale -> encoder final norm.

    Input [B,N,C] is ALREADY the actual attention module output. The caller
    merges heads/applies that module's output norm, projection and dropout in
    its native order. Do not add the residual or pass through the block MLP.
    Returns a side output only; never mutates the native trajectory.
    """
    return final_norm(layer_scale(projected_update.clone()))


def native_layer_scales(raw: Tensor) -> Tensor:
    """RSRM exact statistic: [B,L,C,H,W] -> std [B,L,1,1,1], correction=1."""
    if raw.ndim != 5:
        raise ValueError("raw must be [B,L,C,H,W]")
    return raw.float().std(dim=(2, 3, 4), keepdim=True, correction=1)


def fuse_raw_layers(raw: Tensor, weights: Tensor, scales: Tensor) -> Tensor:
    """No centering and no automatic final L2 normalization, as in RSRM."""
    if raw.shape[:2] != weights.shape or scales.shape != (*raw.shape[:2], 1, 1, 1):
        raise ValueError("Expected raw[B,L,C,H,W], weights[B,L], scales[B,L,1,1,1]")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Degenerate native scales: caller must preserve host fallback")
    return (raw * weights[..., None, None, None] / scales).sum(dim=1)


def _semantic_layers(rate: Tensor) -> Tensor:
    # Match source: no interval before the first position layer; batch union.
    keep = torch.zeros_like(rate, dtype=torch.bool)
    for b in range(rate.shape[0]):
        position = torch.where(rate[b] > rate[b].mean())[0].tolist()
        for i, layer in enumerate(position):
            end = position[i + 1] if i + 1 < len(position) else rate.shape[1]
            if end > layer + 1:
                chosen = layer + 1 + int(rate[b, layer + 1:end].argmin())
                keep[b, chosen] = True
    return keep.any(dim=0)


def _fisher_ratio(feature: Tensor, mask: Tensor) -> Tensor:
    # Equivalent to trace(cov_fg+cov_bg+1e-8), using unbiased variances.
    values = []
    for sample, labels in zip(feature, mask):
        points = sample.flatten(1).T
        fg, bg = points[labels.flatten() == 1], points[labels.flatten() == 0]
        if min(len(fg), len(bg)) < 2:
            values.append(points.new_tensor(float("nan")))
            continue
        between = (fg.mean(0) - bg.mean(0)).square().sum()
        within = fg.var(0, correction=1).sum() + bg.var(0, correction=1).sum()
        values.append(between / (within + points.shape[1] * 1e-8))
    return torch.stack(values)


@torch.no_grad()
def safr_support_weights(raw_maps: Sequence[Tensor], final_maps: Sequence[Tensor],
                         masks: Sequence[Tensor]) -> SAFRState:
    """RSRM support-only layer recipe; supports arbitrary actual depth/width.

    Inputs per view: raw [B,L,C,H,W], final [B,C,H,W], mask [B,H,W].
    Mask resampling is the caller's explicit protocol: RSRM uses bilinear,
    align_corners=True, then >.5. FROST uses its own grid sampler instead.
    For invalid FG/BG or zero weight sum, valid=False and weights=0; unlike
    official unguarded NaNs, the caller can take the specified host fallback.
    """
    if not raw_maps or len(raw_maps) != len(final_maps) or len(raw_maps) != len(masks):
        raise ValueError("One raw/final/mask entry is required per support view")
    rates, selected, per_view = [], [], []
    for raw, final, mask in zip(raw_maps, final_maps, masks):
        raw, final = raw.float(), final.float()
        batch, layers, channels, height, width = raw.shape
        if final.shape != (batch, channels, height, width) or mask.shape != (batch, height, width):
            raise ValueError("Support raw/final/mask grids differ")
        count = round(channels / 200)
        if count < 1:
            raise ValueError("Source SAFR requires round(C/200)>=1")
        indices = final.var(dim=(-1, -2), correction=1).topk(count, dim=1).indices
        variance = raw.var(dim=(-1, -2), correction=1)
        selected_sum = variance.gather(2, indices[:, None].expand(batch, layers, count)).sum(-1)
        total = variance.sum(-1)
        rate = selected_sum / (total + 1e-8)
        keep = _semantic_layers(rate)
        fisher = raw.new_zeros(batch, layers)
        for layer in torch.where(keep)[0].tolist():
            fisher[:, layer] = _fisher_ratio(raw[:, layer], mask)
        per_view.append(fisher * total / (selected_sum + 1e-8))
        rates.append(rate)
        selected.append(keep)
    unnormalized = torch.stack(per_view).mean(0)
    sums = unnormalized.sum(1, keepdim=True)
    valid = torch.isfinite(unnormalized).all(1) & torch.isfinite(sums[:, 0]) & (sums[:, 0] > 0)
    weights = torch.where(valid[:, None], unnormalized / sums, torch.zeros_like(unnormalized))
    return SAFRState(weights, valid, tuple(rates), tuple(selected))


@torch.no_grad()
def frost_whitening(foreground: Tensor, background: Tensor, shrinkage: float = .95) -> Tensor:
    """Support-pooled within-class scatter, eigenvalue floor 1e-6; [C,C]."""
    if min(len(foreground), len(background)) == 0:
        raise ValueError("Whitening requires both support classes")
    fg, bg = foreground.float(), background.float()
    centered = torch.cat((fg - fg.mean(0), bg - bg.mean(0)), dim=0)
    covariance = centered.T @ centered / max(len(fg) + len(bg) - 2, 1)
    channels = covariance.shape[0]
    covariance = ((1 - shrinkage) * covariance + shrinkage * covariance.diagonal().mean()
                  * torch.eye(channels, dtype=covariance.dtype, device=covariance.device))
    values, directions = torch.linalg.eigh(covariance)
    return (directions * values.clamp_min(1e-6).rsqrt()) @ directions.T


def _density(points: Tensor, anchors: Tensor, sigma: float, exclude_self=False,
             chunk_rows=256) -> Tensor:
    count = len(anchors) - int(exclude_self)
    if count < 1:
        raise ValueError("Insufficient anchors for density")
    output = []
    for start in range(0, len(points), chunk_rows):
        scores = (points[start:start + chunk_rows] @ anchors.T - 1) / max(sigma, 1e-6)
        if exclude_self:
            rows = torch.arange(len(scores), device=scores.device)
            scores[rows, start + rows] = -torch.inf
        output.append(scores.logsumexp(1) - math.log(count))
    return torch.cat(output)


def frost_kde_ratio(query: Tensor, foreground: Tensor, background: Tensor, sigma: float) -> Tensor:
    """All inputs are already L2-normalized; no anchor pooling/subsampling."""
    return _density(query, foreground, sigma) - _density(query, background, sigma)


def frost_bandwidth(foreground: Tensor, background: Tensor,
                    grid=(.05, .10, .20, .50, 1.00)) -> tuple[float, float]:
    """Source LOO mean margin, not accuracy/IoU; ties keep first grid value."""
    best = (float(grid[0]), -math.inf)
    for sigma in grid:
        if min(len(foreground), len(background)) < 2:
            margin = 0.0
        else:
            fg_margin = (_density(foreground, foreground, sigma, True)
                         - _density(foreground, background, sigma)).mean()
            bg_margin = (_density(background, foreground, sigma)
                         - _density(background, background, sigma, True)).mean()
            margin = float(fg_margin) - float(bg_margin)
        if margin > best[1]:
            best = (float(sigma), margin)
    return best


def frost_candidate_gate(query_white: Tensor, support_white: Sequence[Tensor],
                          support_masks: Sequence[Tensor]) -> Tensor:
    """Non-unit whitened tokens, source dot-product gate; returns [N] bool.

    Per-view top3 strict majority; across views source uses >=ceil(S/2),
    including exact half for an even number of views. No query GT input.
    """
    prototypes = [x[m.bool()].mean(0) for x, m in zip(support_white, support_masks) if m.bool().any()]
    if not prototypes:
        return torch.zeros(len(query_white), dtype=torch.bool, device=query_white.device)
    prototype = F.normalize(torch.stack(prototypes).mean(0), dim=0)
    forward = query_white @ prototype
    positive = forward > 0
    if not positive.any():
        positive = forward > forward.quantile(.9)
    votes = torch.zeros(len(query_white), device=query_white.device)
    for features, mask in zip(support_white, support_masks):
        k = min(3, len(features))
        indices = (query_white @ features.T).topk(k, dim=1).indices
        votes += mask.bool()[indices].sum(1) >= (k // 2 + 1)
    return positive & (votes >= math.ceil(len(support_white) / 2))


def frost_bilateral(margin: Tensor, geometry: Tensor, rgb: Tensor, shape: tuple[int, int]) -> Tensor:
    """Source defaults; geometry MUST be raw-normalized native query tokens."""
    height, width = shape
    if len(margin) != height * width:
        raise ValueError("Margin/grid mismatch")
    yy, xx = torch.meshgrid(torch.arange(height, device=margin.device),
                           torch.arange(width, device=margin.device), indexing="ij")
    coords = torch.stack((yy.flatten(), xx.flatten()), 1).float()
    standardized = margin / (margin.std(correction=1) + 1e-6)
    weights = torch.exp((geometry @ geometry.T) / .20)
    weights *= torch.exp(-(rgb[:, None] - rgb[None]).square().sum(-1) / .05)
    weights *= (coords[:, None] - coords[None]).square().sum(-1).sqrt() <= 16
    weights *= torch.exp(-(standardized[:, None] - standardized[None]).square())
    weights.fill_diagonal_(0)
    transition = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
    result = margin.clone()
    for _ in range(10):
        result = .30 * margin + .70 * (transition @ result)
    return result


def frost_finalize(margin_grid: Tensor, candidate_grid: Tensor, support_foreground_fraction: float,
                   model_shape: tuple[int, int], original_shape: tuple[int, int] | None = None) -> Tensor:
    """Continuous upsample to MODEL size, threshold, gate; optional binary resize.

    Official FROST first thresholds/intersects at the encoder input size, then
    bilinearly resizes the BINARY result to original HxW. Do not directly resize
    the continuous score to original HxW and call that the official protocol.
    """
    radius = round(4 * support_foreground_fraction)
    candidate = candidate_grid.float()[None, None]
    if radius > 0:
        candidate = F.max_pool2d(candidate, 2 * radius + 1, stride=1, padding=radius)
    score = F.interpolate(margin_grid[None, None].float(), model_shape,
                          mode="bilinear", align_corners=False)[0, 0]
    allowed = F.interpolate(candidate, model_shape, mode="bilinear", align_corners=False)[0, 0] > .5
    prediction = (score > 0) & allowed
    if original_shape is not None:
        prediction = F.interpolate(prediction[None, None].float(), original_shape,
                                   mode="bilinear", align_corners=False)[0, 0] > .5
    return prediction


@torch.no_grad()
def frost_style_dense_readout(support: Sequence[Tensor], masks: Sequence[Tensor], query: Tensor,
                              geometry_query: Tensor, rgb_grid: Tensor,
                              support_foreground_fraction: float,
                              model_shape: tuple[int, int],
                              original_shape: tuple[int, int] | None = None) -> tuple[Tensor, dict]:
    """Compose audited feature-to-mask components, deliberately WITHOUT APD.

    support/query/geometry are maps [C,H,W], masks [H,W], RGB [H,W,3] in
    [0,1]. Caller supplies original-resolution support foreground fraction
    and identity+flip views under an explicit shared input/encoding contract.
    FROST-style means no last-layer positional basis is transplanted into
    an attention-update fusion space; full native FROST remains a separate row.
    """
    if len(support) != len(masks) or not support:
        raise ValueError("One support mask per feature map is required")
    if query.ndim != 3 or geometry_query.shape != query.shape or any(
            x.ndim != 3 or x.shape[0] != query.shape[0] or y.shape != x.shape[-2:]
            for x, y in zip(support, masks)):
        raise ValueError("Expected support/query/geometry [C,H,W], masks [H,W]")
    height, width = query.shape[-2:]
    references = [F.normalize(x.float().flatten(1).T, dim=1) for x in support]
    labels = [x.bool().flatten() for x in masks]
    foreground = torch.cat([x[y] for x, y in zip(references, labels)])
    background = torch.cat([x[~y] for x, y in zip(references, labels)])
    transform = frost_whitening(foreground, background)
    white_references = [x @ transform for x in references]
    white_query = F.normalize(query.float().flatten(1).T, dim=1) @ transform
    candidate = frost_candidate_gate(white_query, white_references, labels).reshape(height, width)
    if not candidate.any():
        empty_margin = query.new_full((height, width), -1)
        prediction = frost_finalize(empty_margin, candidate, support_foreground_fraction,
                                     model_shape, original_shape)
        return prediction, dict(sigma=None, loo_margin=None, candidate_grid=candidate,
                                ell_smooth=None, state="EMPTY_GATE", scope="adapted no-APD")
    fg = F.normalize(torch.cat([x[y] for x, y in zip(white_references, labels)]), dim=1)
    bg = F.normalize(torch.cat([x[~y] for x, y in zip(white_references, labels)]), dim=1)
    sigma, loo = frost_bandwidth(fg, bg)
    ratio = frost_kde_ratio(F.normalize(white_query, dim=1), fg, bg, sigma)
    geometry = F.normalize(geometry_query.float().flatten(1).T, dim=1)
    smoothed = frost_bilateral(ratio, geometry, rgb_grid.reshape(-1, 3), (height, width))
    prediction = frost_finalize(smoothed.reshape(height, width), candidate,
                                 support_foreground_fraction, model_shape, original_shape)
    return prediction, dict(sigma=sigma, loo_margin=loo, candidate_grid=candidate,
                            ell_smooth=smoothed.reshape(height, width),
                            scope="adapted raw-fusion FROST-style; no transplanted APD")
