"""Reference ranking at the original FoRIS pre-CRF foreground pixel count.

This is a fixed-cardinality functional test, not an independent extent model or
an originality claim. The caller supplies shared unit features and the lawful
FoRIS pre-CRF mask. CRF and original-frame rendering belong to the runner.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn.functional as F

from .autonomous_context_transport import (
    PER_ROLE_SAMPLES, RIDGE_STRENGTH, _features, _grid, _ridge, _role_sample,
)


class Prediction(TypedDict):
    fields64: dict[str, np.ndarray]
    masks1024: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


def _baseline(value: Any) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != (1024, 1024) or mask.dtype != np.bool_:
        raise ValueError('foris_precrf_mask must be a bool [1024,1024] array')
    return mask


def topk_at_baseline_count(pixel_score: Any, baseline_mask: Any
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact K largest values; baseline pixels then row-major resolve ties.

    For a constant score this returns the baseline bit for bit. The baseline
    membership has no influence between unequal scores. No score perturbation,
    annotation, threshold fitting or foreground-size estimate is introduced.
    """
    baseline = _baseline(baseline_mask)
    score = np.asarray(pixel_score)
    if score.shape != baseline.shape or not np.issubdtype(score.dtype, np.floating):
        raise ValueError('pixel_score must be a floating [1024,1024] array')
    if not np.isfinite(score).all():
        raise ValueError('pixel_score must be finite')
    s, old = score.ravel(), baseline.ravel()
    k, n = int(old.sum()), old.size
    info: dict[str, Any] = dict(k=k, pixels=n, constant_field=bool(s.min() == s.max()),
        tie_order='baseline membership then row-major; only at equal scores',
        score_min=float(s.min()), score_max=float(s.max()))
    if k in (0, n):
        return baseline.copy(), dict(info, cutoff=None, cutoff_ties=0,
            chosen_from_cutoff_ties=0, added_pixels=0, removed_pixels=0,
            fallback='empty_or_full_baseline')
    cutoff = float(np.partition(s, n-k)[n-k])
    chosen = s > cutoff
    need = k-int(chosen.sum())
    tied = np.flatnonzero(s == cutoff)
    prior = tied[old[tied]]
    other = tied[~old[tied]]
    take_prior = min(need, len(prior))
    chosen[prior[:take_prior]] = True
    chosen[other[:need-take_prior]] = True
    added = int(np.count_nonzero(chosen & ~old))
    removed = int(np.count_nonzero(old & ~chosen))
    if int(chosen.sum()) != k or added != removed:
        raise RuntimeError('Fixed-cardinality readout did not preserve K')
    return chosen.reshape(baseline.shape), dict(info, cutoff=cutoff,
        cutoff_ties=len(tied), chosen_from_cutoff_ties=need,
        added_pixels=added, removed_pixels=removed,
        fallback='constant_field' if info['constant_field'] else None)


def render_topk(field: Any, baseline_mask: Any
                ) -> tuple[np.ndarray, dict[str, Any]]:
    """Promote the finite coarse field to FP64 before CPU bilinear1024."""
    coarse = torch.as_tensor(field)
    if (coarse.device.type != 'cpu' or coarse.ndim != 2
            or not coarse.is_floating_point() or not bool(torch.isfinite(coarse).all())):
        raise ValueError('field must be a finite CPU floating 2D array')
    pixel = F.interpolate(coarse.double()[None, None], (1024, 1024),
                          mode='bilinear', align_corners=False)[0, 0].numpy()
    mask, info = topk_at_baseline_count(pixel, baseline_mask)
    return mask, dict(info, interpolation='CPU FP64 bilinear; align_corners=False')


@torch.inference_mode()
def fit_predict(reference_features: Any, reference_coverage: Any,
                query_features: Any, foris_precrf_mask: Any,
                query_grid_hw: tuple[int, int] = (64, 64)) -> Prediction:
    """Fit fixed balanced-role ridge and matched mean-difference control.

    Inputs are CPU FP32 unit vectors after the caller's same shared APD branch.
    Ridge reuses the frozen source-only helpers: 128 role-mass quantiles per role
    are deduplicated, then selected continuous masses define balanced binary
    risk, with lambda .01 and an unpenalized bias. Prototype means use the same
    selected IDs and normalized positive/negative masses. Their contrast is a
    distribution-matched head comparison, not isolated covariance causality.
    Both readouts preserve the supplied pre-CRF foreground count exactly.
    """
    started = time.perf_counter()
    baseline = _baseline(foris_precrf_mask)
    reference = _features(reference_features, 'reference_features')
    query = _features(query_features, 'query_features')
    if reference.shape[1] != query.shape[1]:
        raise ValueError('Reference and query feature channels must match')
    hw = _grid(query_grid_hw, len(query))
    if hw != (64, 64):
        raise ValueError('This frozen experiment requires the whole64 query grid')
    c = np.asarray(reference_coverage).reshape(-1)
    if (c.size != len(reference) or not np.isfinite(c).all()
            or (c < 0).any() or (c > 1).any()):
        raise ValueError('reference_coverage must match tokens with values in [0,1]')
    positive_mass, negative_mass = float(c.sum(dtype=np.float64)), float((1-c).sum(dtype=np.float64))
    diagnostics: dict[str, Any] = dict(method='reference_rank_at_foris_precrf_count',
        ridge_strength=RIDGE_STRENGTH, per_role_quantile_count=PER_ROLE_SAMPLES,
        shared_feature_preprocessing='caller unit FP32 and shared APD; no implicit transform',
        source_foreground_mass=positive_mass, source_background_mass=negative_mass,
        query_labels_used=False, new_encoder_calls=0,
        baseline_use='exact pre-CRF K; baseline pixels resolve exact score ties',
        finalizer='runner applies identical original CRF; final K may change',
        prototype_scope='same selected IDs and normalized role masses as ridge')
    k = int(baseline.sum())
    fallback = ('empty_or_full_baseline' if k in (0, baseline.size)
                else 'missing_reference_role' if min(positive_mass, negative_mass) <= 0 else None)
    if fallback:
        fields = {name: np.zeros(hw, np.float64) for name in ('ridge', 'prototype')}
        diagnostics['fit_skipped'] = fallback
    else:
        sample = _role_sample(c)
        fitted = _ridge(reference[sample['ids']], sample)
        r, q = reference[sample['ids']].double(), query.double()
        foreground = sample['positive'] @ r
        background = sample['negative'] @ r
        delta = foreground-background
        ridge = q @ fitted['coefficient']+fitted['bias']
        prototype = q @ delta
        # Numerical nulls have no transferable direction. Keep their fields
        # constant so floating cancellation cannot manufacture a ranking.
        if float(fitted['coefficient'].norm()) <= 1e-12:
            ridge = torch.full_like(ridge, float(fitted['bias']))
        if float(delta.norm()) <= 1e-12:
            prototype = torch.zeros_like(prototype)
        fields = dict(ridge=ridge.numpy().reshape(hw), prototype=prototype.numpy().reshape(hw))
        diagnostics.update(sample=sample['diagnostics'], fit=fitted['diagnostics'],
            prototype_contrast_norm=float(delta.norm()),
            ridge_prototype_direction_cosine=float(fitted['coefficient'] @ delta /
                (fitted['coefficient'].norm()*delta.norm()).clamp_min(1e-30)))
    masks, readouts = {}, {}
    for name, field in fields.items():
        masks[name], readouts[name] = render_topk(field, baseline)
    diagnostics.update(readouts=readouts, seconds=time.perf_counter()-started)
    return dict(fields64=fields, masks1024=masks, diagnostics=diagnostics)
