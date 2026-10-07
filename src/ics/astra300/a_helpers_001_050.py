"""A-card finite reference-only calibration and exact geometry helpers.

No query masks, host scores, or remote resources are read. Unspecified numerical
choices are recipes, not constants attributed to the supplied cards.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import ndimage
from scipy.special import logsumexp

from .common import Episode, Result, validate, ArtifactUnavailable
from ics.methods.direct_dino_features import sample_grid

EPS = 1e-10


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class ObservationUnavailable(ArtifactUnavailable):
    """The card needs an actual observation which the caller has not supplied."""


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def sqdist(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.empty((len(a), len(b)), float)
    for start in range(0, len(a), 128):
        aa = a[start:start + 128]
        out[start:start + 128] = np.maximum(np.sum(aa * aa, 1)[:, None]
            + np.sum(b * b, 1)[None, :] - 2 * aa @ b.T, 0)
    return out


def fps(x, ids, cap=64):
    ids = np.asarray(sorted(set(map(int, ids))), int)
    if not len(ids):
        return ids
    chosen = [int(ids[0])]
    distance = sqdist(x[ids], x[chosen])[:, 0]
    while len(chosen) < min(cap, len(ids)):
        order = np.lexsort((ids, -distance))
        index = next(int(k) for k in order if int(ids[k]) not in chosen)
        chosen.append(int(ids[index]))
        distance = np.minimum(distance, sqdist(x[ids], x[chosen[-1]:chosen[-1] + 1])[:, 0])
    return np.array(chosen, int)


def spatial_blocks(hw):
    y, x = np.indices(hw)
    return ((y >= hw[0] / 2) * 2 + (x >= hw[1] / 2)).ravel().astype(int)


@dataclass
class Frame:
    ep: Episode
    train: np.ndarray
    x: np.ndarray
    wf: np.ndarray
    wb: np.ndarray
    valid: np.ndarray
    c: np.ndarray
    fids: np.ndarray
    bids: np.ndarray
    blocks: np.ndarray

    @classmethod
    def make(cls, ep, train=None):
        train = ep.wvalid > 0 if train is None else np.asarray(train, bool) & (ep.wvalid > 0)
        # Unknown/held-out pixels are removed from BOTH roles, never relabeled BG.
        valid = ep.wvalid * train
        wf, wb = ep.wf * train, ep.wb * train
        c = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf, float), where=ep.wvalid > 0)
        fids = np.flatnonzero(train & (c >= .9))
        bids = np.flatnonzero(train & (c <= .1))
        return cls(ep, train, np.asarray(ep.r, float), wf, wb, valid, c,
                   fids, bids, spatial_blocks(ep.r_hw))

    def banks(self, cap=64):
        banks, ids, synthesized = [], [], []
        for role_ids, weights in ((self.fids, self.wf), (self.bids, self.wb)):
            selected = fps(self.x, role_ids, cap)
            if len(selected):
                banks.append(self.x[selected]); ids.append(selected); synthesized.append(False)
            elif weights.sum() > 0:
                banks.append(unit(np.sum(self.x * weights[:, None], 0))[None]); ids.append(np.array([-1])); synthesized.append(True)
            else:
                banks.append(np.empty((0, self.x.shape[1]))); ids.append(np.array([], int)); synthesized.append(False)
        return (*banks, *ids, synthesized)


def nearest_mean_distance(x, bank, k=5):
    if not len(bank):
        return np.full(len(x), np.inf)
    d = sqdist(x, bank)
    k = min(k, len(bank))
    return np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)


def b0(frame, q):
    f, b, _, _, synthetic = frame.banks()
    info = {"anchor_counts": [len(f), len(b)], "weighted_mean_roles": synthetic}
    if frame.wf.sum() <= 0:
        return np.full(len(q), -1.), dict(info, degeneration="empty_reference_mask")
    if frame.wb.sum() <= 0:
        ids = frame.fids
        distances = []
        for block in range(4):
            source = ids[frame.blocks[ids] != block]
            held = ids[frame.blocks[ids] == block]
            if len(source) and len(held):
                bank = frame.x[fps(frame.x, source, 64)]
                distances.extend(nearest_mean_distance(frame.x[held], bank, 1).tolist())
        if not distances:
            return np.full(len(q), -1.), dict(info, degeneration="single_role_no_cross_block_support")
        radius = float(np.quantile(distances, .95))
        return (radius - nearest_mean_distance(q, f, 1)) / 2, dict(info,
            degeneration="single_role_support_rejection", single_role_radius2=radius)
    return (nearest_mean_distance(q, b) - nearest_mean_distance(q, f)) / 2, info


def threshold(scores, wf, wb, fixed_positive=None):
    """Exact balanced soft-coverage absolute-error minimizer; strict score>t."""
    scores = np.asarray(scores, float)
    wf, wb = np.asarray(wf, float), np.asarray(wb, float)
    if wf.sum() <= 0 or wb.sum() <= 0 or not len(scores):
        return 0., None
    # Literal source |binary_prediction - soft_coverage|. A B0 point has its
    # fixed cut0 prediction even during reference threshold search; it is never
    # allowed to change labels as t_R is scanned.
    weights = .5 * wf / wf.sum() + .5 * wb / wb.sum()
    coverage = np.divide(wf, wf + wb, out=np.zeros(len(wf)), where=(wf + wb) > 0)
    fixed = np.full(len(scores),-1,int) if fixed_positive is None else np.asarray(fixed_positive,int)
    if fixed.shape!=scores.shape or not np.isin(fixed,(-1,0,1)).all():
        raise ValueError("Fixed fallback predictions must be aligned -1/0/1")
    active=scores[fixed<0]
    if not len(active):
        return 0.,float(np.sum(weights*np.abs(fixed-coverage)))
    values=np.unique(np.r_[active,0.,np.nextafter(active.min(),-np.inf)])
    # Direct canonical arithmetic matches the public literal C_R, including
    # exact tie behavior. The finite budget is the observed source cut values.
    losses=np.array([np.sum(weights*np.abs(np.where(fixed>=0,fixed,(scores>value).astype(int))-coverage))
                     for value in values])
    best = min(range(len(values)), key=lambda k: (float(losses[k]), abs(float(values[k])), float(values[k])))
    return float(values[best]), float(losses[best])


def fallback_scope(activity,fit_info,n,*,alpha_contract=False):
    activity=dict(activity)
    explicit=activity.pop('_fallback_mask',None)
    whole=bool(activity.pop('_fallback_all',False) or fit_info.get('mechanism')=='A_B0_5NN')
    mask=np.full(n,whole,bool) if explicit is None else np.asarray(explicit,bool)
    if mask.shape!=(n,):raise ValueError('Fallback scope must align predicted points')
    # The raw-alpha source cards already specify alpha=(s_B0+2)/4 and cut.5;
    # their positive affine mapping must stay in the continuous alpha field.
    if alpha_contract:mask[:]=False
    activity['fallback_cut0_points']=int(mask.sum())
    return mask,activity


def infer_a(ep, method_id, fit, assumptions=(), fixed_threshold=None):
    validate(ep)
    start = time.perf_counter()
    full = Frame.make(ep)
    folds, scores, weights_f, weights_b, fixed_predictions = [], [], [], [], []
    # Exactly one joint configuration is used unless the card's wrapper says otherwise.
    if fixed_threshold is None and full.wf.sum() > 0 and full.wb.sum() > 0:
        for block in range(4):
            held = (full.blocks == block) & (ep.wvalid > 0)
            train = ~held
            frame = Frame.make(ep, train)
            if not held.any() or frame.wf.sum() <= 0 or frame.wb.sum() <= 0:
                folds.append({"block": block, "effective": False}); continue
            predict, fit_info = fit(frame)
            s, activity = predict(frame.x[held], np.flatnonzero(held), "r")
            fallback,activity=fallback_scope(activity,fit_info,len(s))
            if not np.isfinite(s).all():
                raise ValueError(f"{method_id} produced non-finite reference scores")
            scores.extend(s.tolist()); weights_f.extend(ep.wf[held].tolist()); weights_b.extend(ep.wb[held].tolist())
            fixed_predictions.extend(np.where(fallback,(s>0).astype(int),-1).tolist())
            folds.append({"block": block, "effective": True, "fit": fit_info, "activity": activity})
    if fixed_threshold is not None:
        cut, loss = float(fixed_threshold), None
    elif sum(x["effective"] for x in folds) >= 2:
        cut, loss = threshold(scores, weights_f, weights_b,fixed_predictions)
    else:
        cut, loss = 0., None
    predict, fit_info = fit(full)
    valid = ep.q_valid > 0
    margin = np.full(len(ep.q), -1., float)
    score, activity = predict(np.asarray(ep.q[valid], float), np.flatnonzero(valid), "q")
    fallback,activity=fallback_scope(activity,fit_info,len(score),alpha_contract=fixed_threshold is not None)
    if not np.isfinite(score).all():
        raise ValueError(f"{method_id} produced non-finite complete query scores")
    margin[valid] = score - cut
    margin[np.flatnonzero(valid)[fallback]]=score[fallback]
    if fallback.any():
        native_fallback=np.zeros(len(ep.q),bool);native_fallback[np.flatnonzero(valid)[fallback]]=True
        activity['fallback_native_bitmask_hex']=np.packbits(native_fallback,bitorder='little').tobytes().hex()
        activity['fallback_native_bitmask_length']=len(ep.q)
    return Result(margin.reshape(ep.q_hw), info=jsonable({"method_id": method_id,
        "renderer": "A_U_continuous_original_then_strict_positive", "threshold": cut,
        "calibration": {"configurations": 1, "four_spatial_blocks": folds,
                        "balanced_coverage_error": loss, "effective_folds": sum(x["effective"] for x in folds)},
        "fit": fit_info, "activity": activity,
        "implementation_assumption": list(assumptions),
        "cpu_seconds": time.perf_counter() - start,
        "new_encoder_forwards": 0, "query_gt_access": False}))


def render_u(ep, result):
    """A exact U: direct continuous field into original RGB pixel centers."""
    g = ep.query_geometry
    oh, ow = ep.original_shape
    if g:
        sh, sw = g["resized_hw"]; oy, ox = g["padding_top_left"]; side = g["view_side"]
        y = (oy + (np.arange(oh) + .5) * sh / oh) * ep.q_hw[0] / side - .5
        x = (ox + (np.arange(ow) + .5) * sw / ow) * ep.q_hw[1] / side - .5
    else:
        y = (np.arange(oh) + .5) * ep.q_hw[0] / oh - .5
        x = (np.arange(ow) + .5) * ep.q_hw[1] / ow - .5
    xx, yy = np.meshgrid(x, y)
    field = sample_grid(np.asarray(result.margin, float).reshape(ep.q_hw), yy, xx)
    return {"margin_original": field, "original": field > 0}


def rank_basis(matrix, cap=None):
    matrix = np.asarray(matrix, float)
    if not matrix.size:
        return np.empty((matrix.shape[-1], 0))
    _, s, v = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(s > max(matrix.shape) * np.finfo(float).eps * (s[0] if len(s) else 1)))
    if cap is not None:
        rank = min(rank, cap)
    return v[:rank].T


def local_difference_basis(bank, k=8):
    if len(bank) < 3:
        return np.empty((bank.shape[1], 0))
    distance = sqdist(bank, bank)
    np.fill_diagonal(distance, np.inf)
    nn = np.argsort(distance, axis=1, kind="stable")[:, :min(k, len(bank) - 1)]
    return rank_basis((bank[nn] - bank[:, None, :]).reshape(-1, bank.shape[1]))


def spherical_modes(x, ids, cap=8):
    chosen = fps(x, ids, cap)
    if not len(chosen):
        return np.empty((0, x.shape[1])), np.array([], int), []
    centers = x[chosen].copy()
    xx = x[ids]
    assignment = np.zeros(len(ids), int)
    for _ in range(20):
        assignment = np.argmax(xx @ centers.T, axis=1)
        active = [k for k in range(len(centers)) if np.any(assignment == k)]
        new = unit(np.array([xx[assignment == k].mean(0) for k in active]))
        if new.shape == centers.shape and np.max(np.abs(new - centers)) < 1e-8:
            centers = new; break
        centers = new
    assignment = np.argmax(xx @ centers.T, axis=1)
    groups = [np.asarray(ids)[assignment == k] for k in range(len(centers))]
    return centers, assignment, groups


def fit_ridge(frame, descriptors=None, strength=1.):
    x = frame.x if descriptors is None else np.asarray(descriptors, float)
    ids = np.flatnonzero(frame.train)
    weights = .5 * frame.wf[ids] / max(frame.wf.sum(), EPS) + .5 * frame.wb[ids] / max(frame.wb.sum(), EPS)
    y = np.divide(frame.wf[ids] / max(frame.wf.sum(), EPS) - frame.wb[ids] / max(frame.wb.sum(), EPS),
                  frame.wf[ids] / max(frame.wf.sum(), EPS) + frame.wb[ids] / max(frame.wb.sum(), EPS),
                  out=np.zeros(len(ids)), where=weights > 0)
    a = np.c_[x[ids], np.ones(len(ids))]
    # Exact primal or dual solve, whichever has fewer variables; no truncated PCA.
    z, yy = a * np.sqrt(weights[:, None]), y * np.sqrt(weights)
    if len(ids) < a.shape[1]:
        coef = z.T @ np.linalg.solve(z @ z.T + strength * np.eye(len(ids)), yy)
    else:
        coef = np.linalg.solve(z.T @ z + strength * np.eye(a.shape[1]), z.T @ yy)
    return coef


def fit_rbf(frame, descriptors=None, cap=64):
    descriptors = frame.x if descriptors is None else descriptors
    fids, bids = fps(frame.x, frame.fids, cap), fps(frame.x, frame.bids, cap)
    ids = np.r_[fids, bids]
    if not len(fids) or not len(bids):
        return None
    z = descriptors[ids]
    d = sqdist(z, z)
    width = max(float(np.median(d[np.triu_indices(len(d), 1)])), EPS)
    kernel = np.exp(-d / width)
    coef = np.linalg.solve(kernel + np.eye(len(ids)), np.r_[np.ones(len(fids)), -np.ones(len(bids))])
    return z, coef, width


def triangles(bank):
    if len(bank) < 3:
        return []
    d = sqdist(bank, bank); np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1, kind="stable")[:, :2]
    return [bank[np.r_[i, nn[i]]] for i in range(len(bank))]


def triangle_distance(q, tri):
    """Exact closest point in closed triangle including all degenerate edges."""
    a, b, c = tri
    u, v = b - a, c - a
    delta = q - a
    gram = np.array([[u @ u, u @ v], [u @ v, v @ v]])
    rhs = np.c_[delta @ u, delta @ v]
    uv = rhs @ np.linalg.pinv(gram)
    inside = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv.sum(1) <= 1)
    distance = np.full(len(q), np.inf)
    projected = a + uv[:, 0, None] * u + uv[:, 1, None] * v
    distance[inside] = np.sum((q[inside] - projected[inside]) ** 2, 1)
    for p, z in ((a, b), (a, c), (b, c)):
        edge = z - p
        t = np.clip((q - p) @ edge / max(edge @ edge, EPS), 0, 1)
        dd = np.sum((q - p - t[:, None] * edge) ** 2, 1)
        distance = np.minimum(distance, dd)
    return distance


def triangle_bank_distance(q, pieces, original):
    if not pieces:
        return nearest_mean_distance(q, original, 1)
    result = np.full(len(q), np.inf)
    for start in range(0, len(q), 128):
        z = q[start:start + 128]
        result[start:start + 128] = np.min([triangle_distance(z, t) for t in pieces], axis=0)
    return result
