"""Seven fixed CPU-only representation rules; none claims measured quality.

Inputs are frozen final-layer patches and the known reference mask.  Control
rows are deliberately separate from METHODS and do not count as methods.
"""
from __future__ import annotations

import numpy as np

from .common import Episode, Result, prototype_margin, unit, validate
from ics.methods.direct_dino_features import resize


def _mm(a, b):
    """Deterministic non-BLAS dot, avoiding macOS BLAS stale FP flags."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 2 and b.ndim == 2:
        return np.einsum("ij,jk->ik", a, b, optimize=False)
    if a.ndim == 2 and b.ndim == 1:
        return np.einsum("ij,j->i", a, b, optimize=False)
    if a.ndim == 1 and b.ndim == 2:
        return np.einsum("i,ij->j", a, b, optimize=False)
    return np.einsum("i,i->", a, b, optimize=False)


def _result(ep, margin, **info):
    margin = np.asarray(margin, dtype=np.float64).reshape(ep.q_hw).copy()
    margin.ravel()[ep.q_valid <= 0] = -1.0
    if not np.isfinite(margin).all():
        raise ValueError("Non-finite complete signed margin")
    return Result(margin, dict(quality_evidence="unmeasured", query_GT_read=False,
                               new_encoder_calls=0, **info))


def _degenerate(ep):
    validate(ep)
    if ep.wf.sum() <= 0 or ep.wb.sum() <= 0:
        return _result(ep, prototype_margin(ep), state="empty_reference_role_fallback")
    return None


def _margin(q, r, wf, wb):
    if np.sum(wf) <= 0:
        return np.full(len(q), -1.0)
    if np.sum(wb) <= 0:
        return np.full(len(q), 1.0)
    fg = unit(np.sum(unit(r) * np.asarray(wf)[:, None], axis=0))
    bg = unit(np.sum(unit(r) * np.asarray(wb)[:, None], axis=0))
    return _mm(unit(q), fg - bg)


def direct_control(ep):
    validate(ep)
    return _result(ep, prototype_margin(ep), control="direct_prototype")


def adversarial_channel_support(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    v = unit(np.sum(ep.r * ep.wf[:, None], axis=0)) - unit(np.sum(ep.r * ep.wb[:, None], axis=0))
    k = min(len(v), max(1, int(np.ceil(np.sqrt(len(v))))))
    margin = np.empty(len(ep.q))
    penalties = np.empty(len(ep.q))
    for start in range(0, len(ep.q), 128):
        c = ep.q[start:start + 128] * v
        positives = np.maximum(c, 0)
        top = np.partition(positives, positives.shape[1] - k, axis=1)[:, -k:]
        penalties[start:start + len(c)] = np.sum(top, axis=1)
        margin[start:start + len(c)] = np.sum(c, axis=1) - np.sum(top, axis=1)
    return _result(ep, margin, deleted_coordinates=k,
                   median_penalty=float(np.median(penalties[ep.q_valid > 0])),
                   feature_basis_dependent=True)


def adversarial_constant_control(ep):
    """Same median erosion magnitude, without token-dependent concentration."""
    out = adversarial_channel_support(ep)
    penalty = out.info.get("median_penalty", 0.0)
    return _result(ep, prototype_margin(ep) - penalty,
                   control="constant_median_channel_erosion", penalty=penalty)


def _weighted_coordinate_quantile(x, weights, quantile):
    keep = np.asarray(weights) > 0
    x, weights = np.asarray(x)[keep], np.asarray(weights)[keep]
    if len(x) == 0:
        raise ValueError("Positive reference mass required for quantiles")
    out = np.empty(x.shape[1])
    for first in range(0, x.shape[1], 64):
        block = x[:, first:first + 64]
        order = np.argsort(block, axis=0, kind="stable")
        ordered_x = np.take_along_axis(block, order, axis=0)
        ordered_w = np.asarray(weights)[order]
        cumulative = np.cumsum(ordered_w, axis=0)
        threshold = quantile * np.sum(weights)
        ids = np.argmax(cumulative >= threshold, axis=0)
        columns = np.arange(block.shape[1])
        values = ordered_x[ids, columns].copy()
        # A weighted median can be an interval.  Use its midpoint when mass
        # exactly straddles the boundary, avoiding an arbitrary lower-tail
        # collapse of a balanced two-mode reference and its MAD.
        ties = np.isclose(cumulative[ids, columns], threshold, rtol=0, atol=1e-12)
        ties &= ids + 1 < len(x)
        values[ties] = .5 * (values[ties] + ordered_x[ids[ties] + 1, columns[ties]])
        out[first:first + block.shape[1]] = values
    return out


def reference_mad_winsor(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    center = _weighted_coordinate_quantile(ep.r, ep.wvalid, .5)
    mad = _weighted_coordinate_quantile(np.abs(ep.r - center), ep.wvalid, .5)
    radius = np.maximum(3 * 1.4826 * mad, 1e-4)
    r = unit(np.clip(ep.r, center - radius, center + radius))
    q = unit(np.clip(ep.q, center - radius, center + radius))
    return _result(ep, _margin(q, r, ep.wf, ep.wb),
                   feature_basis_dependent=True, box_strength=3,
                   clipped_query_coordinates=int(np.count_nonzero((ep.q < center - radius) | (ep.q > center + radius))))


def reference_center_control(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    center = _weighted_coordinate_quantile(ep.r, ep.wvalid, .5)
    return _result(ep, _margin(unit(ep.q - center), unit(ep.r - center), ep.wf, ep.wb),
                   control="reference_median_center_only")


def _pool(x, valid, hw, scale):
    h, w = hw
    oh, ow = (h + scale - 1) // scale, (w + scale - 1) // scale
    pooled = np.zeros((oh, ow, x.shape[-1]))
    mass = np.zeros((oh, ow))
    x = x.reshape(h, w, -1)
    valid = valid.reshape(hw)
    for y in range(oh):
        for z in range(ow):
            ys, xs = slice(y * scale, min(h, (y + 1) * scale)), slice(z * scale, min(w, (z + 1) * scale))
            weight = valid[ys, xs]
            mass[y, z] = np.sum(weight)
            if mass[y, z] > 0:
                pooled[y, z] = np.sum(x[ys, xs] * weight[..., None], axis=(0, 1)) / mass[y, z]
    return pooled, mass


def _pool_weight(weight, hw, scale):
    h, w = hw
    out = np.zeros(((h + scale - 1) // scale, (w + scale - 1) // scale))
    weight = weight.reshape(hw)
    for y in range(len(out)):
        for x in range(out.shape[1]):
            out[y, x] = np.sum(weight[y * scale:min(h, (y + 1) * scale), x * scale:min(w, (x + 1) * scale)])
    return out.ravel()


def multiscale_feature_consensus(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    fields = []
    scales = [s for s in (1, 2, 4) if s <= min(*ep.q_hw, *ep.r_hw)]
    for scale in scales:
        q, qmass = _pool(ep.q, ep.q_valid, ep.q_hw, scale)
        r, _ = _pool(ep.r, ep.wvalid, ep.r_hw, scale)
        wf, wb = _pool_weight(ep.wf, ep.r_hw, scale), _pool_weight(ep.wb, ep.r_hw, scale)
        field = _margin(q.reshape(-1, q.shape[-1]), r.reshape(-1, r.shape[-1]), wf, wb).reshape(qmass.shape)
        field[qmass <= 0] = -1
        fields.append(resize(field, ep.q_hw))
    return _result(ep, np.median(fields, axis=0), scales=scales,
                   statistic="semantic_match_after_feature_pooling")


def multiscale_score_control(ep):
    validate(ep)
    fields = []
    direct = prototype_margin(ep).ravel()[:, None]
    scales = [s for s in (1, 2, 4) if s <= min(*ep.q_hw, *ep.r_hw)]
    for scale in scales:
        pooled, mass = _pool(direct, ep.q_valid, ep.q_hw, scale)
        field = pooled[..., 0]
        field[mass <= 0] = -1
        fields.append(resize(field, ep.q_hw))
    return _result(ep, np.median(fields, axis=0), control="same_scale_scalar_pooling", scales=scales)


def _reflect(indices, length):
    if length <= 1:
        return np.zeros_like(indices)
    period = 2 * (length - 1)
    ids = np.mod(indices, period)
    return np.where(ids < length, ids, period - ids)


def _local_ids(hw, centers, exclude_center=False):
    ys, xs = np.divmod(centers, hw[1])
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if not exclude_center or dy or dx]
    return np.stack([_reflect(ys + dy, hw[0]) * hw[1] + _reflect(xs + dx, hw[1])
                     for dy, dx in offsets], axis=1)


def _spatial_geomedian(x, valid, hw, mode="geomedian"):
    out = np.empty_like(x)
    for start in range(0, len(x), 64):
        centers = np.arange(start, min(start + 64, len(x)))
        ids = _local_ids(hw, centers)
        values, weights = x[ids], valid[ids]
        mass = np.sum(weights, axis=1, keepdims=True)
        center = np.divide(np.sum(values * weights[..., None], axis=1), mass,
                           out=np.zeros((len(centers), x.shape[1])), where=mass > 0)
        if mode == "geomedian":
            for _ in range(8):
                distance = np.linalg.norm(values - center[:, None, :], axis=2)
                iw = weights / np.maximum(distance, 1e-6)
                imass = np.sum(iw, axis=1, keepdims=True)
                center = np.divide(np.sum(values * iw[..., None], axis=1), imass,
                                   out=center.copy(), where=imass > 0)
        out[centers] = unit(center)
    return out


def spatial_geomedian(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    q = _spatial_geomedian(ep.q, ep.q_valid, ep.q_hw)
    r = _spatial_geomedian(ep.r, ep.wvalid, ep.r_hw)
    return _result(ep, _margin(q, r, ep.wf, ep.wb), iterations=8,
                   query_change=float(np.max(np.abs(q - ep.q))),
                   reference_change=float(np.max(np.abs(r - ep.r))))


def spatial_mean_control(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    q = _spatial_geomedian(ep.q, ep.q_valid, ep.q_hw, mode="mean")
    r = _spatial_geomedian(ep.r, ep.wvalid, ep.r_hw, mode="mean")
    return _result(ep, _margin(q, r, ep.wf, ep.wb), control="same_neighborhood_feature_mean")


def spatial_scalar_median_control(ep):
    validate(ep)
    direct = prototype_margin(ep).ravel()
    out = np.empty(len(ep.q))
    for start in range(0, len(ep.q), 64):
        centers = np.arange(start, min(start + 64, len(ep.q)))
        ids = _local_ids(ep.q_hw, centers)
        for center, row in zip(centers, ids):
            values, weights = direct[row], ep.q_valid[row]
            keep = weights > 0
            if not np.any(keep):
                out[center] = -1
            else:
                order = np.argsort(values[keep], kind="stable")
                v, w = values[keep][order], weights[keep][order]
                out[center] = v[np.searchsorted(np.cumsum(w), .5 * np.sum(w), side="left")]
    return _result(ep, out, control="same_neighborhood_scalar_median")


def _affine_reconstruct(x, valid, hw, mode="affine", include_center=False):
    out = np.empty_like(x)
    for center in range(len(x)):
        cy, cx = divmod(center, hw[1])
        coords = [(y, z) for y in range(max(0, cy - 2), min(hw[0], cy + 3))
                  for z in range(max(0, cx - 2), min(hw[1], cx + 3))
                  if (include_center or y != cy or z != cx) and valid[y * hw[1] + z] > 0]
        if not coords:
            out[center] = x[center]
            continue
        ids = np.array([y * hw[1] + z for y, z in coords])
        weight = valid[ids]
        if mode == "mean":
            predicted = np.sum(x[ids] * weight[:, None], axis=0) / np.sum(weight)
        else:
            design = np.array([[1, y - cy, z - cx] for y, z in coords], dtype=float)
            gram = _mm(design.T, weight[:, None] * design)
            if np.linalg.matrix_rank(gram, tol=1e-10) < 3:
                out[center] = x[center]
                continue
            coefficients = _mm(np.linalg.solve(gram, np.array([1., 0., 0.])), design.T) * weight
            predicted = _mm(coefficients, x[ids])
        out[center] = unit(predicted)
    return out


def local_affine_reconstruction(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    q = _affine_reconstruct(ep.q, ep.q_valid, ep.q_hw)
    r = _affine_reconstruct(ep.r, ep.wvalid, ep.r_hw)
    return _result(ep, _margin(q, r, ep.wf, ep.wb),
                   model="approximate_spatial_affine_unit_feature_field",
                   center_in_fit=False)


def affine_mean_control(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    q = _affine_reconstruct(ep.q, ep.q_valid, ep.q_hw, mode="mean")
    r = _affine_reconstruct(ep.r, ep.wvalid, ep.r_hw, mode="mean")
    return _result(ep, _margin(q, r, ep.wf, ep.wb), control="same_neighbor_leave_center_mean")


def affine_include_center_control(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    q = _affine_reconstruct(ep.q, ep.q_valid, ep.q_hw, include_center=True)
    r = _affine_reconstruct(ep.r, ep.wvalid, ep.r_hw, include_center=True)
    return _result(ep, _margin(q, r, ep.wf, ep.wb), control="same_affine_model_including_center")


def _lowrank_reconstruct(x, valid, hw, include_center=False):
    out = np.empty_like(x)
    for center in range(len(x)):
        cy, cx = divmod(center, hw[1])
        coords = [(y, z) for y in range(max(0, cy - 1), min(hw[0], cy + 2))
                  for z in range(max(0, cx - 1), min(hw[1], cx + 2))
                  if (include_center or y != cy or z != cx) and valid[y * hw[1] + z] > 0]
        ids = np.array([y * hw[1] + z for y, z in coords], dtype=int)
        if len(ids) < 2:
            out[center] = x[center]
            continue
        weights = valid[ids] / np.sum(valid[ids])
        mean = np.sum(x[ids] * weights[:, None], axis=0)
        a = (x[ids] - mean) * np.sqrt(weights[:, None])
        values, vectors = np.linalg.eigh(_mm(a, a.T))
        order = np.argsort(values)[::-1][:2]
        order = order[values[order] > 1e-12]
        if len(order):
            basis = _mm(a.T, vectors[:, order]) / np.sqrt(values[order])[None, :]
            predicted = mean + _mm(basis, _mm(basis.T, x[center] - mean))
        else:
            predicted = mean
        out[center] = unit(predicted)
    return out


def local_lowrank_reconstruction(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    q = _lowrank_reconstruct(ep.q, ep.q_valid, ep.q_hw)
    r = _lowrank_reconstruct(ep.r, ep.wvalid, ep.r_hw)
    return _result(ep, _margin(q, r, ep.wf, ep.wb), local_rank=2,
                   covariance_inverse_scaling=False, center_in_fit=False)


def lowrank_mean_control(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    # Same 3x3 in-bounds leave-center neighbor domain, rank zero.
    q, r = [], []
    for x, weights, hw, target in ((ep.q, ep.q_valid, ep.q_hw, q), (ep.r, ep.wvalid, ep.r_hw, r)):
        for center in range(len(x)):
            cy, cx = divmod(center, hw[1])
            ids = [y * hw[1] + z for y in range(max(0, cy - 1), min(hw[0], cy + 2))
                   for z in range(max(0, cx - 1), min(hw[1], cx + 2))
                   if (y != cy or z != cx) and weights[y * hw[1] + z] > 0]
            target.append(unit(np.sum(x[ids] * weights[ids, None], axis=0) / np.sum(weights[ids])) if ids else x[center])
    return _result(ep, _margin(np.asarray(q), np.asarray(r), ep.wf, ep.wb), control="same_neighbors_rank_zero")


def lowrank_include_center_control(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    q = _lowrank_reconstruct(ep.q, ep.q_valid, ep.q_hw, include_center=True)
    r = _lowrank_reconstruct(ep.r, ep.wvalid, ep.r_hw, include_center=True)
    return _result(ep, _margin(q, r, ep.wf, ep.wb), control="same_local_rank_center_included")


def _fit_sample(ep):
    ids = []
    for weights in (ep.wf, ep.wb):
        cumulative = np.cumsum(weights)
        targets = (np.arange(128) + .5) * cumulative[-1] / 128
        ids.extend(np.searchsorted(cumulative, targets, side="left"))
    ids = np.unique(ids)
    ids = ids[ep.wvalid[ids] > 0]
    x = ep.r[ids]
    y = 2 * ep.wf[ids] / ep.wvalid[ids] - 1
    a = .5 * ep.wf[ids] / np.sum(ep.wf[ids]) + .5 * ep.wb[ids] / np.sum(ep.wb[ids])
    return x, y, a


def _weighted_ridge(x, y, weights, strength=.01):
    mass = np.sum(weights)
    mx, my = np.sum(x * weights[:, None], axis=0) / mass, np.sum(y * weights) / mass
    xc, yc = x - mx, y - my
    matrix = _mm(xc, xc.T) + np.diag(strength / np.maximum(weights, 1e-15))
    alpha = np.linalg.solve(matrix, yc)
    coefficient = _mm(xc.T, alpha)
    return coefficient, float(my - _mm(mx, coefficient))


def _huber_objective(x, y, weights, coefficient, bias):
    residual = _mm(x, coefficient) + bias - y
    magnitude = np.abs(residual)
    rho = np.where(magnitude <= .5, .5 * residual**2, .5 * (magnitude - .25))
    return float(_mm(weights, rho) + .005 * _mm(coefficient, coefficient))


def huber_reference_readout(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    x, y, weights = _fit_sample(ep)
    coefficient, bias = _weighted_ridge(x, y, weights)
    objectives = [_huber_objective(x, y, weights, coefficient, bias)]
    for _ in range(12):
        residual = _mm(x, coefficient) + bias - y
        robust_weights = weights * np.minimum(1, .5 / np.maximum(np.abs(residual), 1e-12))
        coefficient, bias = _weighted_ridge(x, y, robust_weights)
        objectives.append(_huber_objective(x, y, weights, coefficient, bias))
    margin = np.empty(len(ep.q))
    for first in range(0, len(ep.q), 256):
        margin[first:first + 256] = _mm(ep.q[first:first + 256], coefficient) + bias
    return _result(ep, margin, fit_tokens=len(x), huber_delta=.5, ridge_strength=.01,
                   iterations=12, objective_history=objectives,
                   optimization_data="known_reference_labels_only")


def huber_ridge_control(ep):
    fallback = _degenerate(ep)
    if fallback:
        return fallback
    x, y, weights = _fit_sample(ep)
    coefficient, bias = _weighted_ridge(x, y, weights)
    return _result(ep, _mm(ep.q, coefficient) + bias, control="same_subset_balanced_ridge", fit_tokens=len(x))


METHODS = {
    "inv_adversarial_channel_support": adversarial_channel_support,
    "inv_reference_mad_winsor": reference_mad_winsor,
    "inv_multiscale_feature_consensus": multiscale_feature_consensus,
    "inv_spatial_geomedian": spatial_geomedian,
    "inv_local_affine_reconstruction": local_affine_reconstruction,
    "inv_local_lowrank_reconstruction": local_lowrank_reconstruction,
    "inv_huber_reference_readout": huber_reference_readout,
}

CONTROLS = {
    "inv_direct_prototype": direct_control,
    "inv_adversarial_constant": adversarial_constant_control,
    "inv_reference_center_only": reference_center_control,
    "inv_multiscale_score_pool": multiscale_score_control,
    "inv_spatial_feature_mean": spatial_mean_control,
    "inv_spatial_score_median": spatial_scalar_median_control,
    "inv_affine_mean": affine_mean_control,
    "inv_affine_include_center": affine_include_center_control,
    "inv_lowrank_mean": lowrank_mean_control,
    "inv_lowrank_include_center": lowrank_include_center_control,
    "inv_huber_ridge": huber_ridge_control,
}
