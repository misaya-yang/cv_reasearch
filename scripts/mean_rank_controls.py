"""Prepare fixed MEAN unary controls; no encoders, masks, graph changes or scoring.

Inputs are the existing normalized FP32 source score and reference guide.
Every output is rounded in FP32, then promoted to the locked CG RHS precision.
The caller must reuse the same A/W, lambda16, interpolation and threshold.
"""
import numpy as np


def average_rank_cdf(values):
    x = np.asarray(values, dtype=np.float32).ravel()
    if not x.size or not np.isfinite(x).all():
        raise ValueError('Require finite nonempty values')
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    lower = np.cumsum(counts)-counts
    return ((lower[inverse]+0.5*counts[inverse])/x.size).astype(np.float32)


def quantile_transfer(source, guide):
    """Assign sorted source magnitudes to guide order; average whole tie groups."""
    s = np.asarray(source, dtype=np.float32).ravel()
    g = np.asarray(guide, dtype=np.float32).ravel()
    if s.size != g.size or not s.size or not np.isfinite(s).all() or not np.isfinite(g).all():
        raise ValueError('Source/guide must have matching finite nonempty shapes')
    ordered_source = np.sort(s)
    order = np.argsort(g, kind='stable')
    sorted_guide = g[order]
    starts = np.r_[0, np.flatnonzero(sorted_guide[1:] != sorted_guide[:-1])+1]
    ends = np.r_[starts[1:], s.size]
    result = np.empty_like(s)
    for start, end in zip(starts, ends):
        result[order[start:end]] = np.mean(ordered_source[start:end], dtype=np.float64)
    return result


def fixed_unaries(source, guide):
    s = np.asarray(source, dtype=np.float32).ravel()
    g = np.asarray(guide, dtype=np.float32).ravel()
    if s.size != g.size or not s.size or not np.isfinite(s).all() or not np.isfinite(g).all():
        raise ValueError('Source/guide must have matching finite nonempty shapes')
    if s.min() < 0 or s.max() > 1:
        raise ValueError('Supply the existing source min-max score, not a new normalization')
    rs, rg = average_rank_cdf(s), average_rank_cdf(g)
    transferred = quantile_transfer(s, g)
    return {name: value.astype(np.float64) for name, value in dict(
        graph_only=s,
        flat=s+0.25*(np.float32(0.5)-rs),
        ordinary=0.75*s+0.25*transferred,
        mean=s+0.25*(rg-rs),
    ).items()}


def ranking_violation(source, guide):
    """Exact real-form loss for diagnostics, including symmetric guide ties."""
    s = np.asarray(source, dtype=np.float64).ravel()
    g = np.asarray(guide, dtype=np.float64).ravel()
    if s.size != g.size or not s.size or not np.isfinite(s).all() or not np.isfinite(g).all():
        raise ValueError('Source/guide must have matching finite nonempty shapes')
    # Compute average ranks in FP64 for the theoretical identity, not the unary.
    _, inverse, counts = np.unique(g, return_inverse=True, return_counts=True)
    lower = np.cumsum(counts)-counts
    rank = (lower[inverse]+0.5*counts[inverse])/g.size
    coefficients = 2*np.arange(s.size)-s.size+1
    dispersion = np.dot(coefficients, np.sort(s))/(2*s.size)
    return float(dispersion-np.dot(rank-0.5, s))
