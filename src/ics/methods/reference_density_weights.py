"""Shared reference repeat-degree balancing for continuous task roles.

This module has no image, model, cache or query dependency. Degree is computed
once over physical reference patches, not separately over duplicated roles.
It is a hypothesis/control implementation, not an accepted segmentation gain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DensityRoleWeights:
    degree: np.ndarray
    role_weight: np.ndarray  # [N, 2], foreground then background
    log_role_weight: np.ndarray  # support outside coverage is -inf
    role_mass: np.ndarray  # unnormalized inverse-degree role mass
    valid: bool
    reason: str | None


def density_degree(cosine: np.ndarray, tau: float = .07,
                   block_rows: int = 128) -> np.ndarray:
    """d_r=sum_t exp((cos(R_r,R_t)-1)/tau), including each self patch.

    Actual input cosine is retained without clipping. FP64 exp/sum is bounded
    in memory by block_rows*N; no persistent dense kernel is made.
    """
    gram = np.asarray(cosine)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not len(gram):
        raise ValueError('Require a nonempty square reference cosine matrix')
    if not np.isfinite(tau) or tau <= 0 or block_rows <= 0:
        raise ValueError('Require finite positive tau and block_rows')
    degree = np.empty(len(gram), np.float64)
    for start in range(0, len(gram), block_rows):
        g = np.asarray(gram[start:start + block_rows], np.float64)
        if not np.isfinite(g).all():
            raise ValueError('Nonfinite reference cosine')
        with np.errstate(over='raise', invalid='raise'):
            degree[start:start + block_rows] = np.exp((g - 1.) / tau).sum(1)
    if not np.isfinite(degree).all() or (degree <= 0).any():
        raise ValueError('Require finite positive shared reference degree')
    return degree


def inverse_density_role_weights(coverage: np.ndarray,
                                 degree: np.ndarray) -> DensityRoleWeights:
    """Normalize c/d and (1-c)/d independently, preserving soft supports.

    Missing foreground/background is explicitly invalid. No smoothing, role
    quota, semantic-purity assertion or fabricated support is introduced.
    """
    c = np.asarray(coverage, np.float64).reshape(-1)
    d = np.asarray(degree, np.float64).reshape(-1)
    if not len(c) or d.shape != c.shape:
        raise ValueError('Coverage and degree must have equal nonempty lengths')
    if not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
        raise ValueError('Require finite continuous coverage in [0,1]')
    if not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError('Require finite positive shared reference degree')
    unnormalized = np.stack((c / d, (1. - c) / d), axis=1)
    mass = unnormalized.sum(0)
    if not np.isfinite(mass).all():
        raise ValueError('Nonfinite inverse-density role mass')
    weights = np.zeros_like(unnormalized)
    present = mass > 0
    weights[:, present] = unnormalized[:, present] / mass[present]
    log_weights = np.full_like(weights, -np.inf)
    positive = weights > 0
    log_weights[positive] = np.log(weights[positive])
    valid = bool(present.all())
    missing = ['foreground', 'background'][int(present[0])] if not valid else None
    return DensityRoleWeights(d.copy(), weights, log_weights, mass,
                              valid, None if valid else 'missing_' + missing)


def classify_reference(cosine: np.ndarray, role_weight: np.ndarray,
                       tau: float = .07, block_rows: int = 128) -> dict:
    """Full/leave-self-out reference role logits for fixed full-R weights.

    Leave-self-out deletes that kernel contribution in both roles and divides
    by the remaining role weight. Degree/weights stay fixed from full R; this
    is not leave-one-out estimation of the KDE itself.
    """
    gram = np.asarray(cosine)
    weights = np.asarray(role_weight, np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or weights.shape != (len(gram), 2):
        raise ValueError('Require square Gram and [N,2] role weights')
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isfinite(tau) or tau <= 0:
        raise ValueError('Require nonnegative finite weights and positive tau')
    mass = weights.sum(0)
    if np.any((mass > 0) & ~np.isclose(mass, 1., rtol=0., atol=1e-12)):
        raise ValueError('Each present role must be normalized to one')
    n = len(gram)
    full = np.full(n, np.nan, np.float64)
    loo = full.copy()
    valid_full = np.full(n, bool((mass > 0).all()))
    remaining = mass[None, :] - weights
    valid_loo = (remaining > 0).all(1)
    if not valid_full.any():
        return dict(lse_full=full, lse_loo=loo, valid_full=valid_full, valid_loo=valid_loo)
    for start in range(0, n, block_rows):
        end = min(n, start + block_rows)
        g = np.asarray(gram[start:end], np.float64)
        if not np.isfinite(g).all():
            raise ValueError('Nonfinite reference cosine')
        kernel = np.exp((g - g.max(1, keepdims=True)) / tau)
        evidence = kernel @ weights
        full[start:end] = np.log(evidence[:, 0]) - np.log(evidence[:, 1])
        kernel[np.arange(end - start), np.arange(start, end)] = 0.
        evidence = kernel @ weights
        ok = valid_loo[start:end]
        indices = np.arange(start, end)[ok]
        if len(indices):
            evidence = evidence[ok] / remaining[indices]
            loo[indices] = np.log(evidence[:, 0]) - np.log(evidence[:, 1])
    return dict(lse_full=full, lse_loo=loo, valid_full=valid_full, valid_loo=valid_loo)
