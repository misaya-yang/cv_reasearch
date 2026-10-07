"""CPU reference-to-query SPD minimum transport, low-rank CORAL-family variant.

Reads frozen q/r and the complete reference coverage only. No FoRIS score,
MEAN/native prediction, query labels, class IDs, or extra model is accepted.
It changes the covariance-alignment gauge, not the available marginal moments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from .reference_occupancy import cluster


METHOD_ID = 'reference_spd_minimum_transport_cached_v0'
FIELD_KEYS = ('field', 'prototype_control', 'mean_only_control', 'diagonal_control',
              'coral_control', 'zca_control', 'pooled_zca_control')


@dataclass(frozen=True)
class Config:
    maximum_rank: int = 32
    foreground_modes: int = 16
    background_modes: int = 16
    lloyd_steps: int = 5
    pooled_trace_ridge: float = 1e-3
    trace_floor: float = 1e-12
    relative_basis_tolerance: float = 1e-10


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Finite nonempty token matrices required')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def _dot(left, right):
    # Validate actual products; macOS Accelerate can expose stale floating flags.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        value = left @ right
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite feature product')
    return value


def discriminant_basis(r_centered, q_centered, delta, cfg=Config()):
    """Keep R-label FG-BG mean difference, then unlabeled max residual directions.

    Each new direction removes its explained energy from every centered R/Q row.
    No DxD matrix, PCA/SVD on NxD, query labels, or random seed is required.
    """
    x = np.concatenate((r_centered, q_centered))
    residual = np.sum(x*x, axis=1)
    reference_scale = float(np.max(residual, initial=0))
    vectors = []
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > cfg.relative_basis_tolerance:
        vector = delta/delta_norm
        vectors.append(vector)
        residual = np.maximum(0, residual-_dot(x, vector)**2)
    while len(vectors) < min(cfg.maximum_rank, x.shape[1]):
        index = int(np.argmax(residual))  # Lowest combined token ID on ties.
        if residual[index] <= cfg.relative_basis_tolerance**2*max(reference_scale, cfg.trace_floor):
            break
        vector = x[index].copy()
        if vectors:
            u = np.column_stack(vectors)
            for _ in range(2):
                vector -= _dot(u, _dot(u.T, vector))
        norm = float(np.linalg.norm(vector))
        if norm <= cfg.relative_basis_tolerance*np.sqrt(max(reference_scale, cfg.trace_floor)):
            # Selected residual was roundoff, not another mode; recompute exact
            # residual norms before deciding whether the basis is complete.
            u = np.column_stack(vectors) if vectors else np.empty((x.shape[1], 0))
            reconstructed = _dot(_dot(x, u), u.T)
            residual = np.sum((x-reconstructed)**2, axis=1)
            if float(residual.max()) <= cfg.relative_basis_tolerance**2*max(reference_scale, cfg.trace_floor):
                break
            continue
        vector /= norm
        vectors.append(vector)
        residual = np.maximum(0, residual-_dot(x, vector)**2)
    u = np.column_stack(vectors) if vectors else np.empty((x.shape[1], 0))
    error = float(np.linalg.norm(u.T@u-np.eye(len(vectors))))
    preserved = float(np.linalg.norm(delta-u@(u.T@delta)))
    return u, dict(actual_rank=len(vectors), orthogonality_error=error,
                   reference_discriminant_norm=delta_norm, discriminant_projection_error=preserved,
                   basis_selection='reference-label delta first, then unlabeled centered R/Q largest residual row')


def spd_power(matrix, power):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('Finite square SPD matrix required')
    matrix = (matrix+matrix.T)*.5
    values, vectors = np.linalg.eigh(matrix)
    if len(values) and values.min() <= 0:
        raise ValueError('Strictly positive eigenvalues required; no silent spectral repair')
    return (vectors*values**power)@vectors.T


def minimum_transport(source_scatter, query_scatter):
    """Unique SPD A with A Sr A=Sq and minimum Gaussian squared displacement."""
    sr = np.asarray(source_scatter, dtype=np.float64)
    sq = np.asarray(query_scatter, dtype=np.float64)
    root, inverse = spd_power(sr, .5), spd_power(sr, -.5)
    inner = root@sq@root
    operator = inverse@spd_power(inner, .5)@inverse
    operator = (operator+operator.T)*.5
    residual = float(np.linalg.norm(operator@sr@operator.T-sq)/max(np.linalg.norm(sq), 1e-300))
    if not np.isfinite(operator).all() or residual > 1e-7:
        raise RuntimeError(f'SPD transport covariance identity failed: {residual}')
    cost = float(np.trace(sr)+np.trace(sq)-2*np.trace(operator@sr))
    return operator, dict(covariance_transport_relative_residual=residual,
                         operator_minimum_eigenvalue=float(np.linalg.eigvalsh(operator).min()),
                         regularized_gaussian_displacement=cost)


def prototype_bank(r, coverage, cfg=Config()):
    banks, reports = [], []
    for weights, maximum in ((coverage, cfg.foreground_modes), (1-coverage, cfg.background_modes)):
        selected = weights > 0
        _, labels, mass = cluster(r[selected], maximum, cfg.lloyd_steps, weights[selected])
        # Affine transport acts on raw weighted mode means, not re-normalized means.
        means = np.stack([np.average(r[selected][labels == j], axis=0,
                                    weights=weights[selected][labels == j]) for j in range(len(mass))])
        banks.append(means)
        reports.append(dict(modes=len(means), positive_weight_tokens=int(selected.sum()), mass=mass.tolist()))
    return np.concatenate(banks), len(banks[0]), reports


def distance_field(distance, foreground_modes, query_shape):
    distance = np.maximum(0, np.asarray(distance, dtype=np.float64))
    df, db = distance[:, :foreground_modes].min(1), distance[:, foreground_modes:].min(1)
    g = np.divide(db-df, db+df, out=np.zeros_like(db), where=db+df > 0)
    return (.5+.5*g).reshape(query_shape)


def render(field, original_shape):
    import torch
    import torch.nn.functional as functional
    tensor = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))[None, None]
    work = functional.interpolate(tensor, (1024, 1024), mode='bilinear', align_corners=False) > .5
    original = functional.interpolate(work.float(), tuple(original_shape), mode='bilinear', align_corners=False) > .5
    return work[0, 0].numpy(), original[0, 0].numpy()


def predict(q, r, coverage, *, query_shape=None, original_shape=(1024, 1024), cfg=Config()):
    started = time.perf_counter()
    if asdict(cfg) != asdict(Config()):
        raise ValueError('One fixed domain-alignment recipe; no parameter search')
    q, r = _unit(q), _unit(r)
    cov = np.asarray(coverage, dtype=np.float64)
    query_shape = tuple(cov.shape) if query_shape is None else tuple(query_shape)
    if (cov.ndim != 2 or cov.size != len(r) or int(np.prod(query_shape)) != len(q)
            or len(query_shape) != 2 or q.shape[1] != r.shape[1] or not np.isfinite(cov).all()
            or np.any((cov < 0) | (cov > 1))):
        raise ValueError('Aligned finite q/r/reference coverage and query grid required')
    if len(original_shape) != 2 or any(int(v) != v or v < 1 for v in original_shape):
        raise ValueError('Positive integer query original H/W required')
    original_shape = tuple(map(int, original_shape))
    info = dict(method_id=METHOD_ID, config=asdict(cfg), independent_method_increment=0,
                family='CORAL/second-moment-domain-alignment operator revision',
                query_gt_used=False, score_or_mask_host_used=False, new_encoder_forwards=0,
                empirical_gain='unmeasured', natural_SPD_style_assumption='unverified',
                reference_labels_used='mode banks and explicit FG-BG mean-difference basis direction',
                query_labels_used=False)
    fields = {key: np.zeros(query_shape) for key in FIELD_KEYS}
    if not cov.any():
        info['abstention'] = 'empty_reference'
    elif float((1-cov).sum()) == 0:
        foreground = r.mean(0); norm = float(np.linalg.norm(foreground))
        score = _dot(q, foreground/norm) if norm else np.zeros(len(q))
        fields = {key: (.5+.5*score).reshape(query_shape) for key in FIELD_KEYS}
        info['abstention'] = 'no_reference_background; common raw foreground-cosine fallback'
    else:
        m_r, m_q = r.mean(0), q.mean(0)
        weights = cov.ravel()
        delta = np.average(r, axis=0, weights=weights)-np.average(r, axis=0, weights=1-weights)
        u, basis_info = discriminant_basis(r-m_r, q-m_q, delta, cfg)
        rank = u.shape[1]
        p, nf, bank_info = prototype_bank(r, weights, cfg)
        base_response = _dot(q, p.T)
        q_norm, p_norm = np.sum(q*q, 1), np.sum(p*p, 1)
        raw_distance = q_norm[:, None]+p_norm[None, :]-2*base_response
        fields['prototype_control'] = distance_field(raw_distance, nf, query_shape)
        shift = m_q-m_r
        info.update(basis=basis_info, prototype_banks=bank_info, actual_dimension=q.shape[1],
                    means_shift_norm=float(np.linalg.norm(shift)))
        projected_q, projected_p = _dot(q, u), _dot(p-m_r, u)

        def mapped_field(operator):
            correction = _dot(projected_p, (operator-np.eye(rank)).T)
            mapped = p+shift+_dot(correction, u.T)
            response = base_response+_dot(q, shift)[:, None]+_dot(projected_q, correction.T)
            distance = q_norm[:, None]+np.sum(mapped*mapped, 1)[None, :]-2*response
            return distance_field(distance, nf, query_shape)

        fields['mean_only_control'] = mapped_field(np.eye(rank))
        if not rank:
            for key in ('field', 'diagonal_control', 'coral_control', 'zca_control', 'pooled_zca_control'):
                fields[key] = fields['mean_only_control'].copy()
            info['abstention'] = 'zero_selected_rank; mean-shift fallback'
        else:
            rc, qc = _dot(r-m_r, u), _dot(q-m_q, u)
            sr, sq = _dot(rc.T, rc)/len(r), _dot(qc.T, qc)/len(q)
            ridge = cfg.pooled_trace_ridge*max(float((np.trace(sr)+np.trace(sq))/(2*rank)), cfg.trace_floor)
            sr, sq = sr+ridge*np.eye(rank), sq+ridge*np.eye(rank)
            operator, transport_info = minimum_transport(sr, sq)
            qr, ri = spd_power(sq, .5), spd_power(sr, -.5)
            coral = qr@ri
            fields['field'] = mapped_field(operator)
            fields['coral_control'] = mapped_field(coral)
            fields['diagonal_control'] = mapped_field(np.diag(np.sqrt(np.diag(sq)/np.diag(sr))))
            centered_response = (base_response-_dot(q, m_r)[:, None]
                                 -_dot(p, m_q)[None, :]+m_q@m_r)
            centered_distance = (np.sum((q-m_q)**2, 1)[:, None]
                                 +np.sum((p-m_r)**2, 1)[None, :]-2*centered_response)
            latent_distance = np.sum(qc*qc, 1)[:, None]+np.sum(projected_p*projected_p, 1)[None, :]-2*_dot(qc, projected_p.T)
            outside = np.maximum(0, centered_distance-latent_distance)

            def whitened_field(query_root, reference_root):
                x, y = _dot(qc, query_root), _dot(projected_p, reference_root)
                distance = outside+np.sum(x*x, 1)[:, None]+np.sum(y*y, 1)[None, :]-2*_dot(x, y.T)
                return distance_field(distance, nf, query_shape)

            fields['zca_control'] = whitened_field(spd_power(sq, -.5), ri)
            pooled = spd_power((sr+sq)*.5, -.5)
            fields['pooled_zca_control'] = whitened_field(pooled, pooled)
            info.update(abstention=False, ridge=ridge, transport=transport_info,
                        coral_operator_difference=float(np.linalg.norm(operator-coral)),
                        scatter_commutator_norm=float(np.linalg.norm(sr@sq-sq@sr)),
                        source_scatter_condition=float(np.linalg.cond(sr)), query_scatter_condition=float(np.linalg.cond(sq)),
                        covariance_dimension=rank, whitening_scope='full selected <=32D space; untouched orthogonal remainder')
    masks_work, masks_original = {}, {}
    for key, field in fields.items():
        masks_work[key], masks_original[key] = render(field, original_shape)
    info['predict_with_all_renderers_seconds'] = time.perf_counter()-started
    return dict(**fields, masks_work=masks_work, masks_original=masks_original, info=info)
