"""Reference-conditioned CPU decisions; no query labels or host field inputs.

Each output is a complete signed native query margin. Experimental claims and
the known-reference assumptions live in cards/decision_risk.json; these are
not calibrated query posteriors or guarantees of segmentation improvement.
"""
from __future__ import annotations

import numpy as np

from .common import Episode, Result, neighbors, prototype_margin, unit, validate


def _base(ep):
    return prototype_margin(ep).ravel()


def _finish(ep, margin, method, **info):
    margin = np.asarray(margin, float).copy().reshape(-1)
    margin[ep.q_valid <= 0] = -1.
    return Result(margin.reshape(ep.q_hw), dict(method_id=method,
        query_GT_read=False, host_scores_used=False, calibrated_probability=False,
        extra_encoder_forwards=0, **info))


def _degenerate(ep, method):
    validate(ep)
    if ep.wf.sum() <= 0 or ep.wb.sum() <= 0:
        return _finish(ep, _base(ep), method, state="empty_reference_class_fallback")
    return None


def _spatial_groups(ep):
    y, x = np.indices(ep.r_hw)
    return (2 * np.minimum(1, 2 * y // ep.r_hw[0])
            + np.minimum(1, 2 * x // ep.r_hw[1])).ravel()


def _sample_class(r, weights, cap):
    """Fixed weighted-quantile reference compression, never Q-dependent."""
    ids = np.flatnonzero(weights > 0)
    if len(ids) <= cap:
        return r[ids], weights[ids] / weights[ids].sum()
    cumulative = np.cumsum(weights[ids]) / weights[ids].sum()
    picked = ids[np.searchsorted(cumulative, (np.arange(cap) + .5) / cap)]
    return r[picked], np.full(cap, 1 / cap)


def _class_data(ep, cap=64):
    f, wf = _sample_class(ep.r, ep.wf, cap)
    b, wb = _sample_class(ep.r, ep.wb, cap)
    return np.vstack((f, b)), np.r_[np.ones(len(f)), -np.ones(len(b))], np.r_[.5 * wf, .5 * wb]


def _logistic_step(x, y, weights, w, bias, rate=.1):
    logits = x @ w + bias
    wrong = 1 / (1 + np.exp(np.clip(y * logits, -50, 50)))
    coeff = -weights * y * wrong
    return w - rate * (x.T @ coeff + .1 * w), bias - rate * coeff.sum()


def average_logistic(ep):
    fallback = _degenerate(ep, "DR_control_average_logistic")
    if fallback:
        return fallback
    x, y, weights = _class_data(ep)
    w, bias = np.zeros(ep.r.shape[1]), 0.
    for _ in range(200):
        w, bias = _logistic_step(x, y, weights, w, bias)
    return _finish(ep, ep.q @ w + bias, "DR_control_average_logistic", steps=200)


def coverage_affine(ep):
    fallback = _degenerate(ep, "DR01")
    if fallback:
        return fallback
    c = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf), where=ep.wvalid > 0)
    a = np.c_[c, 1 - c]
    gram = a.T @ (ep.wvalid[:, None] * a)
    eigen = np.linalg.eigvalsh(gram)
    if eigen[0] <= 1e-6 * max(eigen[-1], 1e-12):
        return _finish(ep, _base(ep), "DR01", state="ill_conditioned_coverage_fallback")
    endpoints = np.linalg.solve(gram, a.T @ (ep.wvalid[:, None] * ep.r))
    residual = float(np.sum(ep.wvalid[:, None] * (ep.r - a @ endpoints) ** 2) / ep.wvalid.sum())
    fg, bg = unit(endpoints)
    return _finish(ep, ep.q @ (fg - bg), "DR01", coverage_gram_eigen=eigen.tolist(),
                   unnormalized_endpoint_norms=np.linalg.norm(endpoints, axis=1).tolist(),
                   unit_reference_affine_residual=residual, exact_DINO_demixing_claim=False)


def _boundary_direction(ep, local=True):
    c = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf), where=ep.wvalid > 0)
    if local:
        a, b = neighbors(ep.r_hw)
        forward = (c[a] >= .75) & (c[b] <= .25)
        reverse = (c[b] >= .75) & (c[a] <= .25)
        f, b = np.r_[a[forward], b[reverse]], np.r_[b[forward], a[reverse]]
    else:
        fids, bids = np.flatnonzero(c >= .75), np.flatnonzero((c <= .25) & (ep.wvalid > 0))
        if not len(fids) or not len(bids):
            return None
        count = len(fids) * len(bids)
        index = np.arange(count) if count <= 64 else np.linspace(0, count - 1, 64, dtype=int)
        f, b = fids[index // len(bids)], bids[index % len(bids)]
    good = (ep.wvalid[f] > 0) & (ep.wvalid[b] > 0)
    f, b = f[good], b[good]
    if not len(f):
        return None
    if len(f) > 64:
        select = np.linspace(0, len(f) - 1, 64, dtype=int)
        f, b = f[select], b[select]
    weight = ep.wvalid[f] * ep.wvalid[b] * (c[f] - c[b])
    d = unit(ep.r[f] - ep.r[b])
    direction = np.sum(weight[:, None] * d, axis=0) / weight.sum()
    return direction, len(f)


def boundary_pairs(ep):
    fallback = _degenerate(ep, "DR02")
    if fallback:
        return fallback
    found = _boundary_direction(ep)
    if found is None:
        return _finish(ep, _base(ep), "DR02", state="no_local_pure_pair_fallback")
    direction, count = found
    return _finish(ep, ep.q @ direction, "DR02", pair_count=count,
                   midpoint_term_identically_zero_for_unit_reference=True)


def global_pairs(ep):
    fallback = _degenerate(ep, "DR_control_global_pairs")
    if fallback:
        return fallback
    found = _boundary_direction(ep, local=False)
    if found is None:
        return _finish(ep, _base(ep), "DR_control_global_pairs", state="no_pure_pair_fallback")
    direction, count = found
    return _finish(ep, ep.q @ direction, "DR_control_global_pairs", pair_count=count)


def _worst_group_risk(ep, cells, method):
    fallback = _degenerate(ep, method)
    if fallback:
        return fallback
    groups = []
    for cell in range(4):
        for y, weight in ((1., ep.wf), (-1., ep.wb)):
            masked = weight * (cells == cell)
            if masked.sum() > 0:
                x, a = _sample_class(ep.r, masked, 32)
                groups.append((x, np.full(len(x), y), a))
    w, bias = np.zeros(ep.r.shape[1]), 0.
    p = np.full(len(groups), 1 / len(groups))
    for _ in range(200):
        losses, dw, db = [], np.zeros_like(w), 0.
        for pg, (x, y, a) in zip(p, groups):
            z = x @ w + bias
            losses.append(float(np.sum(a * np.logaddexp(0, -y * z))))
            coeff = -a * y / (1 + np.exp(np.clip(y * z, -50, 50)))
            dw += pg * (x.T @ coeff)
            db += pg * coeff.sum()
        w -= .1 * (dw + .1 * w)
        bias -= .1 * db
        losses = np.asarray(losses)
        p *= np.exp(.1 * (losses - losses.max()))
        p /= p.sum()
    final_losses = [float(np.sum(a * np.logaddexp(0, -y * (x @ w + bias)))) for x, y, a in groups]
    return _finish(ep, ep.q @ w + bias, method, steps=200, group_count=len(groups),
                   group_losses=final_losses, adversary_weights=p.tolist(),
                   primal_objective=max(final_losses) + .05 * float(w @ w),
                   duality_gap="unknown; finite primal-dual iteration budget, no convergence guarantee")


def worst_spatial_risk(ep):
    return _worst_group_risk(ep, _spatial_groups(ep), "DR03")


def shuffled_group_risk(ep):
    cells = _spatial_groups(ep).copy()
    np.random.default_rng(0).shuffle(cells)
    return _worst_group_risk(ep, cells, "DR_control_shuffled_group_risk")


def trimmed_risk(ep):
    fallback = _degenerate(ep, "DR04")
    if fallback:
        return fallback
    x, y, weights = _class_data(ep)
    fg = unit(np.sum(ep.r * ep.wf[:, None], axis=0))
    bg = unit(np.sum(ep.r * ep.wb[:, None], axis=0))
    w, bias = fg - bg, 0.
    retained = []
    for _ in range(5):
        loss = np.logaddexp(0, -y * (x @ w + bias))
        active = np.zeros_like(weights)
        for label in (1, -1):
            ids = np.flatnonzero(y == label)
            order = ids[np.argsort(loss[ids], kind="stable")]
            budget = .75 * weights[ids].sum()
            before = np.r_[0., np.cumsum(weights[order])[:-1]]
            active[order] = np.minimum(weights[order], np.maximum(0, budget - before)) / .75
        retained.append(np.flatnonzero(active > 0).tolist())
        for _ in range(40):
            w, bias = _logistic_step(x, y, active, w, bias)
    return _finish(ep, ep.q @ w + bias, "DR04", steps=200, retained_sample_indices=retained,
                   contamination_fraction_assumption=.25, global_optimum_claim=False)


def _jackknife(ep):
    cells, margins = _spatial_groups(ep), []
    for cell in range(4):
        wf, wb = ep.wf * (cells != cell), ep.wb * (cells != cell)
        if wf.sum() > 0 and wb.sum() > 0:
            f = unit(np.sum(ep.r * wf[:, None], axis=0))
            b = unit(np.sum(ep.r * wb[:, None], axis=0))
            margins.append(ep.q @ (f - b))
    return np.asarray(margins)


def spatial_jackknife(ep):
    fallback = _degenerate(ep, "DR05")
    if fallback:
        return fallback
    m, loo = _base(ep), _jackknife(ep)
    if len(loo) < 3:
        return _finish(ep, m, "DR05", state="insufficient_loo_classes_fallback")
    k, mean = len(loo), loo.mean(axis=0)
    corrected = k * m - (k - 1) * mean
    sd = np.sqrt((k - 1) / k * np.sum((loo - mean) ** 2, axis=0))
    guard = (corrected - 1.96 * sd > 0) | (corrected + 1.96 * sd < 0)
    result = np.where(guard, corrected, m)
    return _finish(ep, result, "DR05", loo_count=k,
                   guarded_tokens=int(np.count_nonzero(guard & (ep.q_valid > 0))),
                   changed_native_decisions=int(np.count_nonzero((result > 0) != (m > 0))),
                   nominal_confidence_coverage_claim=False)


def unguarded_jackknife(ep):
    fallback = _degenerate(ep, "DR_control_unguarded_jackknife")
    if fallback:
        return fallback
    m, loo = _base(ep), _jackknife(ep)
    return _finish(ep, len(loo) * m - (len(loo) - 1) * loo.mean(axis=0) if len(loo) >= 3 else m,
                   "DR_control_unguarded_jackknife")


def pair_rank(ep):
    fallback = _degenerate(ep, "DR06")
    if fallback:
        return fallback
    # Keep original reference weights; common real probes contain only 64 tokens.
    ids = np.flatnonzero(ep.wvalid > 0)
    if len(ids) > 128:
        ids = ids[np.linspace(0, len(ids) - 1, 128, dtype=int)]
    r, wf, wb = ep.r[ids], ep.wf[ids], ep.wb[ids]
    if wf.sum() <= 0 or wb.sum() <= 0:
        return _finish(ep, _base(ep), "DR06", state="compression_removed_class_fallback")
    result = np.empty(len(ep.q))
    for start in range(0, len(ep.q), 64):
        for offset, sim in enumerate(ep.q[start:start + 64] @ r.T):
            order = np.argsort(sim, kind="stable")
            _, first, count = np.unique(sim[order], return_index=True, return_counts=True)
            fg = np.add.reduceat(wf[order], first)
            bg = np.add.reduceat(wb[order], first)
            below = np.r_[0., np.cumsum(bg)[:-1]]
            p = float(fg @ (below + .5 * bg) / (wf.sum() * wb.sum()))
            result[start + offset] = 2 * p - 1
    return _finish(ep, result, "DR06", reference_samples=len(ids),
                   U_statistic_not_query_posterior=True)


def median_similarity(ep):
    fallback = _degenerate(ep, "DR_control_median_similarity")
    if fallback:
        return fallback
    ids = np.flatnonzero(ep.wvalid > 0)
    if len(ids) > 128:
        ids = ids[np.linspace(0, len(ids) - 1, 128, dtype=int)]
    result = np.empty(len(ep.q))
    for start in range(0, len(ep.q), 64):
        for offset, sim in enumerate(ep.q[start:start + 64] @ ep.r[ids].T):
            order = np.argsort(sim, kind="stable")
            values = sim[order]
            medians = []
            for weights in (ep.wf[ids][order], ep.wb[ids][order]):
                if weights.sum() <= 0:
                    medians.append(0.)
                else:
                    index = np.searchsorted(np.cumsum(weights), .5 * weights.sum())
                    medians.append(values[min(index, len(values) - 1)])
            result[start + offset] = medians[0] - medians[1]
    return _finish(ep, result, "DR_control_median_similarity")


def _components(mask):
    h, w = mask.shape
    seen, pieces = np.zeros((h, w), bool), []
    for seed in np.flatnonzero(mask.ravel()):
        yy, xx = divmod(int(seed), w)
        if seen[yy, xx]:
            continue
        seen[yy, xx], stack, ids = True, [(yy, xx)], []
        while stack:
            y, x = stack.pop()
            ids.append(y * w + x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        pieces.append(np.asarray(ids, int))
    return pieces


def _component_prototypes(ep, foreground):
    c = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf), where=ep.wvalid > 0)
    mask = ((c > .5) if foreground else (c <= .5)) & (ep.wvalid > 0)
    weight = ep.wf if foreground else ep.wb
    pieces = _components(mask.reshape(ep.r_hw))
    pieces.sort(key=lambda ids: -float(weight[ids].sum()))
    if len(pieces) > 8:
        pieces = pieces[:8] + [np.concatenate(pieces[8:])]
    prototypes = [unit(np.sum(ep.r[ids] * weight[ids, None], axis=0)) for ids in pieces if weight[ids].sum() > 0]
    return np.asarray(prototypes)


def reference_components(ep):
    fallback = _degenerate(ep, "DR07")
    if fallback:
        return fallback
    f, b = _component_prototypes(ep, True), _component_prototypes(ep, False)
    if not len(f) or not len(b):
        return _finish(ep, _base(ep), "DR07", state="coarse_component_class_missing_fallback")
    margin = (ep.q @ f.T).max(axis=1) - (ep.q @ b.T).max(axis=1)
    return _finish(ep, margin, "DR07", foreground_components=len(f), background_components=len(b),
                   true_instance_annotation_claim=False)


def component_kmeans(ep):
    """Same number of weighted prototypes; remove only the mask connectivity."""
    fallback = _degenerate(ep, "DR_control_component_kmeans")
    if fallback:
        return fallback
    prototypes = []
    for foreground, weights in ((True, ep.wf), (False, ep.wb)):
        k = max(1, len(_component_prototypes(ep, foreground)))
        ids = np.flatnonzero(weights > 0)
        x, a = ep.r[ids], weights[ids]
        k = min(k, len(x))
        chosen = [int(np.argmax(a))]
        while len(chosen) < k:
            distance = 1 - (x @ x[chosen].T).max(axis=1)
            distance[chosen] = -1
            chosen.append(int(np.argmax(distance)))
        centers = x[chosen].copy()
        for _ in range(12):
            label = np.argmax(x @ centers.T, axis=1)
            for group in range(k):
                keep = label == group
                if keep.any():
                    centers[group] = unit(np.sum(x[keep] * a[keep, None], axis=0))
        prototypes.append(centers)
    f, b = prototypes
    margin = (ep.q @ f.T).max(axis=1) - (ep.q @ b.T).max(axis=1)
    return _finish(ep, margin, "DR_control_component_kmeans", foreground_centers=len(f), background_centers=len(b))


def _pure_reference(ep):
    c = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf), where=ep.wvalid > 0)
    fi = np.flatnonzero((c >= .75) & (ep.wvalid > 0))
    bi = np.flatnonzero((c <= .25) & (ep.wvalid > 0))
    fi = fi if len(fi) <= 64 else fi[np.linspace(0, len(fi) - 1, 64, dtype=int)]
    bi = bi if len(bi) <= 64 else bi[np.linspace(0, len(bi) - 1, 64, dtype=int)]
    return fi, bi


def pure_prototype(ep):
    fallback = _degenerate(ep, "DR_control_pure_prototype")
    if fallback:
        return fallback
    fi, bi = _pure_reference(ep)
    if not len(fi) or not len(bi):
        return _finish(ep, _base(ep), "DR_control_pure_prototype", state="missing_pure_class_fallback")
    f = unit(np.sum(ep.r[fi] * ep.wf[fi, None], axis=0))
    b = unit(np.sum(ep.r[bi] * ep.wb[bi, None], axis=0))
    return _finish(ep, ep.q @ (f - b), "DR_control_pure_prototype")


def hull_margin(ep):
    fallback = _degenerate(ep, "DR08")
    if fallback:
        return fallback
    fi, bi = _pure_reference(ep)
    if not len(fi) or not len(bi):
        return _finish(ep, _base(ep), "DR08", state="missing_pure_class_fallback")
    ff, bb = ep.r[fi], ep.r[bi]
    f = np.average(ff, axis=0, weights=ep.wf[fi])
    b = np.average(bb, axis=0, weights=ep.wb[bi])
    steps = 0
    for steps in range(1, 251):
        d = f - b
        fv, bv = ff[np.argmin(ff @ d)], bb[np.argmax(bb @ d)]
        delta = (fv - bv) - d
        gap = max(0., float(-2 * d @ delta))
        norm = float(delta @ delta)
        if gap < 1e-10 or norm < 1e-20:
            break
        rate = np.clip(-float(d @ delta) / norm, 0., 1.)
        f, b = f + rate * (fv - f), b + rate * (bv - b)
    d = f - b
    v = ff[np.argmin(ff @ d)] - bb[np.argmax(bb @ d)]
    gap = max(0., float(2 * d @ (d - v)))
    if np.linalg.norm(d) < 1e-8:
        return _finish(ep, _base(ep), "DR08", state="reference_hulls_overlap_fallback", fw_gap=gap)
    direction = unit(d)
    margin = ep.q @ direction - float((f + b) @ direction / 2)
    return _finish(ep, margin, "DR08", frank_wolfe_steps=steps, fw_objective=float(d @ d),
                   fw_gap=gap, reference_maximum_margin_exact_claim=gap < 1e-8,
                   query_generalization_guarantee=False)


METHODS = {"DR01": coverage_affine, "DR02": boundary_pairs,
           "DR03": worst_spatial_risk, "DR04": trimmed_risk,
           "DR05": spatial_jackknife, "DR06": pair_rank,
           "DR07": reference_components, "DR08": hull_margin}

CONTROLS = {"DR_control_average_logistic": average_logistic,
            "DR_control_pure_prototype": pure_prototype,
            "DR_control_global_pairs": global_pairs,
            "DR_control_unguarded_jackknife": unguarded_jackknife,
            "DR_control_shuffled_group_risk": shuffled_group_risk,
            "DR_control_median_similarity": median_similarity,
            "DR_control_component_kmeans": component_kmeans}

# Later accepted families remain in a separate source so the first definitions
# are stable. Historical frozen runs retain their original source snapshot.
from .decision_risk_batch2 import METHODS as BATCH2_METHODS, CONTROLS as BATCH2_CONTROLS
METHODS.update(BATCH2_METHODS)
CONTROLS.update(BATCH2_CONTROLS)
