"""One-sided background admission for the frozen FoRIS/ridge composition.

The caller supplies lawful, already-computed host and reference fields.  This
module has no dataset names, ground truth, encoder, fitting, file I/O or CRF.
Its interval is an observable consistency constraint, not a probability model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _resize(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float64))[None, None]
    return F.interpolate(tensor, shape, mode="bilinear", align_corners=False)[0, 0].numpy()


def _mask(value: np.ndarray, name: str, shape=None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or not result.size or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} has an invalid shape")
    if not np.all((result == 0) | (result == 1)):
        raise ValueError(f"{name} must be binary")
    return result.astype(bool, copy=False)


def _field(value: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _lower_weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    values, weights = values.ravel(), weights.ravel()
    keep = weights > 0
    if not keep.any():
        return None
    selected, mass = values[keep], weights[keep].astype(np.float64)
    order = np.argsort(selected, kind="stable")
    selected, mass = selected[order], mass[order]
    at = np.searchsorted(np.cumsum(mass), .5*mass.sum(), side="left")
    return float(selected[at])


def _readout(score: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return (score > 0) | ((score == 0) & baseline)


def compose_background_admission(
    baseline_original: np.ndarray,
    baseline_cli1024: np.ndarray,
    source_score64: np.ndarray,
    ridge128: np.ndarray,
    reference_threshold: float,
    reference_separation: float,
) -> dict:
    """Return the frozen complete composition and its same-area control.

    ``parent`` is the unchanged reference-ridge composition reconstructed by
    the same FP64 clip-at128 and bilinear readout. The caller must verify it
    against the sealed parent before using this function on that input binding.
    """
    baseline = _mask(baseline_original, "baseline_original")
    cli = _mask(baseline_cli1024, "baseline_cli1024", (1024, 1024))
    source = _field(source_score64, "source_score64", (64, 64))
    ridge = _field(ridge128, "ridge128", (128, 128))
    threshold, separation = float(reference_threshold), float(reference_separation)
    if not np.isfinite(threshold):
        raise ValueError("The reference threshold must be finite")

    span = float(source.max()-source.min())
    normalized = (source-source.min())/max(span, 1e-6)
    if span:
        confidence_cli = np.abs(2*_resize(normalized, (1024, 1024))-1)
        confidence = np.abs(2*_resize(normalized, baseline.shape)-1)
    else:
        confidence_cli = np.zeros((1024, 1024))
        confidence = np.zeros(baseline.shape)
    background_core = (~cli) & (confidence_cli > .5)
    weights = background_core.reshape(128, 8, 128, 8).sum((1, 3))
    floor = _lower_weighted_median(ridge, weights)
    admission_threshold = max(threshold, floor) if floor is not None else threshold

    valid_source = np.isfinite(separation) and separation > np.finfo(float).eps
    if valid_source:
        reference_evidence = np.clip((ridge-threshold)/separation, -1, 1)
        admission_evidence = np.clip((ridge-admission_threshold)/separation, -1, 1)
    else:
        reference_evidence = np.zeros_like(ridge)
        admission_evidence = reference_evidence
    host = (2*baseline.astype(float)-1)*confidence
    parent = _readout(host+.5*_resize(reference_evidence, baseline.shape), baseline)
    if admission_threshold == threshold or not valid_source:
        admitted = parent.copy()
    else:
        admitted = _readout(host+.5*_resize(admission_evidence, baseline.shape), baseline)
    complete = parent & (baseline | admitted)

    # The control keeps exactly as much old added foreground, ranked by the
    # original host score. Its budget is lawful; no labels choose the pixels.
    added_ids = np.flatnonzero((parent & ~baseline).ravel())
    retained = int((complete & ~baseline).sum())
    control = parent & baseline
    tie_count = 0
    if retained == len(added_ids):
        control = parent.copy()
    elif retained:
        rank_score = _resize(source, baseline.shape).ravel()[added_ids]
        order = np.lexsort((added_ids, -rank_score))
        chosen = added_ids[order[:retained]]
        control.ravel()[chosen] = True
        cutoff = rank_score[order[retained-1]]
        at_cutoff = int(np.count_nonzero(rank_score == cutoff))
        selected_at_cutoff = int(np.count_nonzero(rank_score[order[:retained]] == cutoff))
        tie_count = at_cutoff if selected_at_cutoff < at_cutoff else 0

    # These are contract checks, not evidence of segmentation quality.
    assert np.array_equal(complete & baseline, parent & baseline)
    assert not np.any(complete & ~parent)
    assert not np.any((complete != baseline) & (confidence > .5))
    assert np.array_equal(control & baseline, parent & baseline)
    assert int((control & ~baseline).sum()) == retained
    return dict(
        prediction=complete,
        same_area_source_order=control,
        parent=parent,
        diagnostics=dict(
            reference_threshold=threshold,
            reference_separation=separation if np.isfinite(separation) else None,
            background_core_pixels=int(background_core.sum()),
            background_core_cells=int((weights > 0).sum()),
            background_median=floor,
            admission_threshold=admission_threshold,
            constraint_activated=admission_threshold > threshold,
            source_zero_evidence_fallback=not valid_source,
            old_added_pixels=len(added_ids),
            retained_added_pixels=retained,
            removed_old_added_pixels=len(added_ids)-retained,
            original_foreground_deletions=int((baseline & ~parent).sum()),
            matched_control_cutoff_tie_pixels=tie_count,
        ),
    )
