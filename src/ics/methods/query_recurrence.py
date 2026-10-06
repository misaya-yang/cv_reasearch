"""Unvalidated query-instance recurrence cue from disjoint trusted components.

No extra training example: this adapts appearance using the current query only.
Spatially separate components are not statistically independent confirmations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy import ndimage

from .reference_occupancy import unit


@dataclass(frozen=True)
class Config:
    seed_threshold: float = .75
    minimum_seed_pixels: int = 8
    maximum_seeds: int = 32
    reference_gate: float = .5
    peer_cosine: float = .9
    minimum_peers: int = 2
    correction_weight: float = .5
    purity: float = .9


def normalized_mean(rows, weight=None):
    vector = np.mean(rows, axis=0) if weight is None else np.average(rows, axis=0, weights=weight)
    norm = np.linalg.norm(vector)
    return None if norm < 1e-8 else vector / norm


def choose_star(descriptors, reference_cosines, confidence, cfg):
    """Largest reference-gated cosine star; no maximum-clique/optimality claim."""
    eligible = reference_cosines >= cfg.reference_gate
    similarity = np.einsum('id,jd->ij', descriptors, descriptors, optimize=False)
    np.fill_diagonal(similarity, 1.0)
    stars = []
    for root in np.flatnonzero(eligible):
        members = np.flatnonzero(eligible & (similarity[root] >= cfg.peer_cosine))
        if len(members) < cfg.minimum_peers:
            continue
        # Reference and confidence break equal-count ties without query labels.
        key = (len(members), float(reference_cosines[members].mean()), float(confidence[members].mean()), -int(root))
        stars.append((key, int(root), members))
    if not stars:
        return None, None, similarity
    _, root, members = max(stars, key=lambda item: item[0])
    return root, members, similarity


def predict(q, r, coverage, base, cfg=Config()):
    started = time.perf_counter()
    if (not .5 < cfg.seed_threshold <= 1 or cfg.minimum_seed_pixels < 1 or cfg.maximum_seeds < 2
            or not -1 <= cfg.reference_gate <= 1 or not 0 < cfg.peer_cosine <= 1
            or not 2 <= cfg.minimum_peers <= cfg.maximum_seeds or cfg.correction_weight < 0
            or not 0 < cfg.purity <= 1):
        raise ValueError('Invalid query recurrence configuration')
    q, r = unit(q), unit(r)
    base = np.asarray(base, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=np.float64)
    if (base.ndim != 2 or base.shape != coverage.shape or base.size != len(q) or len(q) != len(r)
            or q.shape[1] != r.shape[1] or not np.isfinite(base).all() or not np.isfinite(coverage).all()
            or coverage.min() < 0 or coverage.max() > 1 or coverage.max() <= 0):
        raise ValueError('Aligned features, finite base and nonempty reference coverage required')
    fields = dict(field=base.copy(), all_seed_control=base.copy(), single_seed_control=base.copy())
    info = dict(config=asdict(cfg), query_gt_used=False, new_encoder_forwards=0,
                seed_components=0, eligible_seeds=0, selected_seed_labels=[], abstention=True,
                real_gain='unmeasured', complete_dataset_minutes='unmeasured')
    pure = coverage.ravel() >= cfg.purity
    if not pure.any():
        pure = coverage.ravel() == coverage.max()
    reference = normalized_mean(r[pure], coverage.ravel()[pure])
    if reference is None:
        info['reason'] = 'degenerate_reference_centroid'
    else:
        labels, _ = ndimage.label(base >= cfg.seed_threshold)
        counts = np.bincount(labels.ravel())
        confidence_sum = np.bincount(labels.ravel(), weights=base.ravel(), minlength=len(counts))
        available = np.flatnonzero(counts >= cfg.minimum_seed_pixels)
        available = available[available > 0]
        # Rank scalar component statistics before any D-dimensional descriptors;
        # even a fragmented query constructs at most maximum_seeds descriptors.
        candidates = sorted(available, key=lambda label: (-confidence_sum[label] / counts[label],
                                                         -int(counts[label]), int(label)))[:cfg.maximum_seeds]
        info['available_seed_components'] = int(len(available))
        seeds = []
        for label in candidates:
            take = labels.ravel() == label
            count = int(counts[label])
            descriptor = normalized_mean(q[take])
            if descriptor is not None:
                seeds.append((float(confidence_sum[label] / count), count, int(label), descriptor))
        info['seed_components'] = len(seeds)
        if seeds:
            descriptors = np.stack([item[3] for item in seeds])
            confidence = np.array([item[0] for item in seeds])
            reference_cosines = np.einsum('id,d->i', descriptors, reference, optimize=False)
            eligible = reference_cosines >= cfg.reference_gate
            info['eligible_seeds'] = int(eligible.sum())
            reference_response = np.einsum('nd,d->n', q, reference, optimize=False).reshape(base.shape)

            def corrected(prototype):
                if prototype is None or cfg.correction_weight == 0:
                    return base.copy()
                response = np.einsum('nd,d->n', q, prototype, optimize=False).reshape(base.shape)
                return np.clip(base + cfg.correction_weight * (response - reference_response), 0, 1)

            if eligible.any():
                fields['all_seed_control'] = corrected(normalized_mean(descriptors[eligible]))
                candidates = np.flatnonzero(eligible)
                best = candidates[np.argmax(reference_cosines[candidates])]
                fields['single_seed_control'] = corrected(descriptors[best])
            root, members, similarity = choose_star(descriptors, reference_cosines, confidence, cfg)
            if members is not None:
                prototype = normalized_mean(descriptors[members])
                if prototype is not None and cfg.correction_weight > 0:
                    fields['field'] = corrected(prototype)
                    info.update(abstention=False, selected_seed_labels=[int(seeds[i][2]) for i in members],
                                root_seed_label=int(seeds[root][2]), selected_reference_cosines=reference_cosines[members].tolist(),
                                minimum_member_pair_cosine=float(similarity[np.ix_(members, members)].min()))
            if info['abstention']:
                info['reason'] = 'no_supported_star_or_zero_weight'
        else:
            info['reason'] = 'no_trusted_component'
    info['wall_seconds'] = time.perf_counter() - started
    info['added_tokens'] = int(((fields['field'] > .5) & ~(base > .5)).sum())
    info['deleted_tokens'] = int((~(fields['field'] > .5) & (base > .5)).sum())
    return dict(**fields, info=info)
