"""Query dictionaries named only by the complete annotated reference.

QP01 is inverse class-balanced voting. QP02 jointly names dictionary entries
by a finite reference classification risk. Neither receives query seeds/labels.
"""
from __future__ import annotations

import math

import numpy as np

from .common import Episode, Result, prototype_margin, unit, validate


def _degenerate(ep: Episode):
    validate(ep)
    if not np.any(ep.q_valid > 0) or ep.wf.sum() <= 0:
        return Result(np.full(ep.q_hw, -1.), {"status": "no_query_or_reference_foreground"})
    if ep.wb.sum() <= 0:
        out = np.where(ep.q_valid > 0, 1., -1.)
        return Result(out.reshape(ep.q_hw), {"status": "no_reference_background"})
    return None


def dictionary(ep: Episode):
    """Fixed query-only spherical dictionary, with coverage-weighted centers."""
    q = unit(ep.q)
    ids = np.flatnonzero(ep.q_valid > 0)
    if not len(ids):
        return np.empty((0, q.shape[1])), np.full(len(q), -1, int)
    x, weight = q[ids], ep.q_valid[ids]
    k = min(8, max(1, math.isqrt(len(ids))))
    picks = [0]
    best = x @ x[0]
    for _ in range(1, k):
        pick = int(np.argmin(best))
        if best[pick] >= 1 - 1e-12:
            break
        picks.append(pick)
        best = np.maximum(best, x @ x[pick])
    centers = x[picks].copy()
    previous = None
    for _ in range(12):
        labels = np.argmax(x @ centers.T, axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        for j in range(len(centers)):
            chosen = labels == j
            if np.any(chosen):
                total = np.sum(x[chosen] * weight[chosen, None], axis=0)
                centers[j] = unit(total) if np.linalg.norm(total) > 1e-12 else x[np.flatnonzero(chosen)[0]]
    labels = np.argmax(x @ centers.T, axis=1)
    full = np.full(len(q), -1, int)
    full[ids] = labels
    return centers, full


def _result(ep, labels, scores, info):
    margin = np.full(len(ep.q), -1.)
    good = labels >= 0
    margin[good] = scores[labels[good]]
    return Result(margin.reshape(ep.q_hw), info)


def _voting_scores(ep, reference_labels, k):
    fg = np.bincount(reference_labels, weights=ep.wf, minlength=k) / ep.wf.sum()
    bg = np.bincount(reference_labels, weights=ep.wb, minlength=k) / ep.wb.sum()
    supported = fg + bg > 0
    score = np.full(k, -1.)
    score[supported] = (fg[supported] - bg[supported]) / (fg[supported] + bg[supported])
    return score, fg, bg


def qp01(ep: Episode) -> Result:
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    centers, labels = dictionary(ep)
    ref_label = np.argmax(unit(ep.r) @ centers.T, axis=1)
    score, fg, bg = _voting_scores(ep, ref_label, len(centers))
    return _result(ep, labels, score, {"method": "QP01", "centers": len(centers),
                                    "foreground_support": fg.tolist(), "background_support": bg.tolist(),
                                    "evidence": "reference_label_transfer_not_query_identity_proof"})


def _risk(sim, assignment, wf, wb):
    """Balanced logistic risk, fixed temperature and no query-label target."""
    margin = np.max(sim[:, assignment], axis=1) - np.max(sim[:, ~assignment], axis=1)
    return float(.5 * (wf @ np.logaddexp(0., -margin / .07)
                       + wb @ np.logaddexp(0., margin / .07)))


def qp02(ep: Episode) -> Result:
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    centers, labels = dictionary(ep)
    k = len(centers)
    if k < 2:
        margin = prototype_margin(ep).copy()
        margin.ravel()[ep.q_valid == 0] = -1.
        return Result(margin, {"method": "QP02", "centers": k,
                      "status": "one_dictionary_entry_prototype_fallback", "best_reference_risk": None})
    sim = unit(ep.r) @ centers.T
    wf, wb = ep.wf / ep.wf.sum(), ep.wb / ep.wb.sum()
    best_risk, best_mask = np.inf, 0
    for mask in range(1, (1 << k) - 1):
        assignment = np.asarray([(mask >> j) & 1 for j in range(k)], bool)
        value = _risk(sim, assignment, wf, wb)
        if value < best_risk - 1e-12:
            best_risk, best_mask = value, mask
    score = np.asarray([1. if (best_mask >> j) & 1 else -1. for j in range(k)])
    return _result(ep, labels, score, {"method": "QP02", "centers": k,
                   "selected_bitmask": best_mask, "candidates": (1 << k) - 2,
                   "best_reference_risk": best_risk,
                   "evidence": "reference_risk_optimum_is_not_query_quality_guarantee"})


def pointwise_kde(ep: Episode) -> Result:
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    q, r = unit(ep.q), unit(ep.r)
    logf = np.log(np.maximum(ep.wf / ep.wf.sum(), 1e-300))
    logb = np.log(np.maximum(ep.wb / ep.wb.sum(), 1e-300))
    logf[ep.wf == 0] = -np.inf
    logb[ep.wb == 0] = -np.inf
    margin = np.full(len(q), -1.)
    good = np.flatnonzero(ep.q_valid > 0)
    for start in range(0, len(good), 128):
        rows = good[start:start + 128]
        sim = q[rows] @ r.T / .07
        f, b = sim + logf, sim + logb
        mf, mb = np.max(f, axis=1), np.max(b, axis=1)
        margin[rows] = mf + np.log(np.exp(f - mf[:, None]).sum(axis=1)) - mb - np.log(np.exp(b - mb[:, None]).sum(axis=1))
    return Result(margin.reshape(ep.q_hw), {"control": "pointwise_balanced_KDE", "temperature": .07})


def pooled_kde(ep: Episode) -> Result:
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    centers, labels = dictionary(ep)
    pointwise = pointwise_kde(ep).margin.ravel()
    good = labels >= 0
    total = np.bincount(labels[good], weights=ep.q_valid[good] * pointwise[good], minlength=len(centers))
    mass = np.bincount(labels[good], weights=ep.q_valid[good], minlength=len(centers))
    score = np.divide(total, mass, out=np.full_like(total, -1.), where=mass > 0)
    return _result(ep, labels, score, {"control": "same_dictionary_mean_KDE"})


def fine_inverse_vote(ep: Episode) -> Result:
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    centers, labels = dictionary(ep)
    q_ids = np.flatnonzero(ep.q_valid > 0)
    q, r = unit(ep.q[q_ids]), unit(ep.r)
    nearest = np.empty(len(r), int)
    for start in range(0, len(r), 128):
        end = min(start + 128, len(r))
        nearest[start:end] = q_ids[np.argmax(r[start:end] @ q.T, axis=1)]
    score, fg, bg = _voting_scores(ep, labels[nearest], len(centers))
    return _result(ep, labels, score, {"control": "nearest_query_token_then_same_dictionary_vote",
                                    "foreground_support": fg.tolist(), "background_support": bg.tolist()})


def center_prototype(ep: Episode) -> Result:
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    centers, labels = dictionary(ep)
    fg = unit(np.sum(unit(ep.r) * ep.wf[:, None], axis=0))
    bg = unit(np.sum(unit(ep.r) * ep.wb[:, None], axis=0))
    return _result(ep, labels, centers @ (fg - bg), {"control": "same_dictionary_center_prototype"})


METHODS = {"QP01": qp01, "QP02": qp02}
CONTROLS = {"QP_pointwise_KDE": pointwise_kde, "QP_pooled_KDE": pooled_kde,
            "QP_fine_inverse_vote": fine_inverse_vote, "QP_center_prototype": center_prototype}


def _weighted_representatives(x, weight, maximum=128):
    ids = np.flatnonzero(weight > 0)
    mass = weight[ids] / weight[ids].sum()
    count = min(maximum, len(ids))
    selected = np.searchsorted(np.cumsum(mass), (np.arange(count) + .5) / count,
                               side="left")
    return unit(x[ids[np.minimum(selected, len(ids) - 1)]])


def _kernel(a, b):
    return np.exp((np.clip(a @ b.T, -1., 1.) - 1.) / .25)


def _effective_n(weight):
    return float(weight.sum() ** 2 / max(1e-300, weight @ weight))


def _mmd_calibration(ep):
    fg = _weighted_representatives(ep.r, ep.wf)
    bg = _weighted_representatives(ep.r, ep.wb)
    kff = _kernel(fg, fg)
    t_ref = 0.
    for fold in range(4):
        a = np.arange(len(fg)) % 4 == fold
        b = ~a
        if np.any(a) and np.any(b):
            discrepancy = (kff[np.ix_(a, a)].mean() + kff[np.ix_(b, b)].mean()
                           - 2 * kff[np.ix_(a, b)].mean())
            t_ref = max(t_ref, float(discrepancy))
    return fg, bg, float(kff.mean()), float(_kernel(bg, bg).mean()), t_ref


def _window_indices(hw, side=4):
    grid = np.arange(np.prod(hw)).reshape(hw)
    for y in range(0, hw[0], side):
        for x in range(0, hw[1], side):
            yield grid[y:y + side, x:x + side].ravel()


def _distribution_gate(ep, variant="absolute_region"):
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    q = unit(ep.q)
    fg, bg, ff, bb, calibration = _mmd_calibration(ep)
    # At most Nq x 128 cross-kernel storage; no whole-query Gram matrix.
    qf, qb = _kernel(q, fg).mean(axis=1), _kernel(q, bg).mean(axis=1)
    out = np.full(len(q), -1.)
    stats = []
    for ids in _window_indices(ep.q_hw):
        ids = ids[ep.q_valid[ids] > 0]
        if not len(ids):
            continue
        weight = ep.q_valid[ids] / ep.q_valid[ids].sum()
        qq = float(weight @ _kernel(q[ids], q[ids]) @ weight)
        cf, cb = float(weight @ qf[ids]), float(weight @ qb[ids])
        df, db = max(0., qq - 2 * cf + ff), max(0., qq - 2 * cb + bb)
        threshold = calibration + 1 / _effective_n(ep.wf) + 1 / _effective_n(ep.q_valid[ids])
        relative = db - df
        if variant == "relative_only":
            score = relative
        elif variant == "absolute_pointwise_pooled":
            diagonal = np.exp((np.clip(np.sum(q[ids] ** 2, axis=1), 0., 1.) - 1.) / .25)
            point_df = diagonal - 2 * qf[ids] + ff
            point_db = diagonal - 2 * qb[ids] + bb
            score = float(weight @ np.minimum(point_db - point_df, threshold - point_df))
        else:
            score = min(relative, threshold - df)
        out[ids] = score
        stats.append({"dFG": df, "dBG": db, "threshold": threshold, "query_self_kernel": qq})
    return Result(out.reshape(ep.q_hw), {"method": "QP04" if variant == "absolute_region" else variant,
                  "reference_representatives": [len(fg), len(bg)], "reference_calibration": calibration,
                  "windows": stats, "threshold_is_heuristic_not_confidence_bound": True})


def qp04(ep):
    return _distribution_gate(ep)


def relative_mmd(ep):
    return _distribution_gate(ep, "relative_only")


def absolute_pointwise_pooled(ep):
    return _distribution_gate(ep, "absolute_pointwise_pooled")


def window_kde(ep):
    point = pointwise_kde(ep)
    out = np.full(len(ep.q), -1.)
    for ids in _window_indices(ep.q_hw):
        ids = ids[ep.q_valid[ids] > 0]
        if len(ids):
            out[ids] = np.average(point.margin.ravel()[ids], weights=ep.q_valid[ids])
    return Result(out.reshape(ep.q_hw), {"control": "fixed_window_mean_KDE"})


def qp06(ep: Episode) -> Result:
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    unary = pointwise_kde(ep).margin
    valid = ep.q_valid.reshape(ep.q_hw)
    f_cost = valid * np.logaddexp(0., -unary)
    b_cost = valid * np.logaddexp(0., unary)
    out = np.full(ep.q_hw, -1.)
    leaves = [0]

    def solve(y0, y1, x0, x1):
        fg = float(f_cost[y0:y1, x0:x1].sum())
        bg = float(b_cost[y0:y1, x0:x1].sum())
        # Ties are background, then foreground, then a subdivided description.
        leaf_cost, kind = (bg, -1.) if bg <= fg else (fg, 1.)
        children = []
        if y1 - y0 > 1 or x1 - x0 > 1:
            ys = (y0, (y0 + y1) // 2, y1) if y1 - y0 > 1 else (y0, y1)
            xs = (x0, (x0 + x1) // 2, x1) if x1 - x0 > 1 else (x0, x1)
            for ya, yb in zip(ys[:-1], ys[1:]):
                for xa, xb in zip(xs[:-1], xs[1:]):
                    children.append(solve(ya, yb, xa, xb))
            split_cost = sum(child[0] for child in children) + math.log(2)
            if split_cost < leaf_cost - 1e-12:
                return split_cost, ("split", children)
        return leaf_cost, ("leaf", y0, y1, x0, x1, kind)

    def emit(node):
        if node[0] == "split":
            for _, child in node[1]:
                emit(child)
        else:
            _, y0, y1, x0, x1, kind = node
            out[y0:y1, x0:x1] = kind
            leaves[0] += 1

    optimum, tree = solve(0, ep.q_hw[0], 0, ep.q_hw[1])
    emit(tree)
    out[valid == 0] = -1.
    return Result(out, {"method": "QP06", "description_objective": optimum,
                       "leaf_blocks": leaves[0], "code_per_split": math.log(2),
                       "identity_source": "reference_KDE_not_tree_coherence"})


def _direction_reference_risk(ep, directions, offsets=None):
    margin = unit(ep.r) @ directions.T
    if offsets is not None:
        margin -= np.asarray(offsets)[None, :]
    wf, wb = ep.wf / ep.wf.sum(), ep.wb / ep.wb.sum()
    return .5 * (wf @ np.logaddexp(0., -margin / .07)
                 + wb @ np.logaddexp(0., margin / .07))


def qp07(ep: Episode) -> Result:
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    centers, _ = dictionary(ep)
    directions, pairs = [], []
    for a in range(len(centers)):
        for b in range(len(centers)):
            if a != b and np.linalg.norm(centers[a] - centers[b]) > 1e-12:
                directions.append(unit(centers[a] - centers[b]))
                pairs.append((a, b))
    if not directions:
        margin = prototype_margin(ep).copy()
        margin.ravel()[ep.q_valid == 0] = -1.
        return Result(margin, {"method": "QP07", "status": "one_center_prototype_fallback"})
    directions = np.asarray(directions)
    offsets = np.asarray([float(.5 * (centers[a] + centers[b]) @ direction)
                          for (a, b), direction in zip(pairs, directions)])
    risk = _direction_reference_risk(ep, directions, offsets)
    chosen = int(np.argmin(risk))
    a, b = pairs[chosen]
    direction = directions[chosen]
    offset = float(offsets[chosen])
    margin = unit(ep.q) @ direction - offset
    margin[ep.q_valid == 0] = -1.
    return Result(margin.reshape(ep.q_hw), {"method": "QP07", "pair": pairs[chosen],
                  "centers": len(centers), "candidate_directions": len(pairs),
                  "reference_risk": float(risk[chosen]), "offset": offset,
                  "reference_fit_is_not_query_quality_guarantee": True})


def reference_pair_direction(ep):
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    r = unit(ep.r)
    fg = _weighted_representatives(r, ep.wf, 8)
    bg = _weighted_representatives(r, ep.wb, 8)
    directions = np.asarray([unit(f - b) for f in fg for b in bg
                             if np.linalg.norm(f - b) > 1e-12])
    if not len(directions):
        return Result(prototype_margin(ep), {"control": "reference_pair_direction", "status": "fallback"})
    risks = _direction_reference_risk(ep, directions)
    selected = int(np.argmin(risks))
    margin = unit(ep.q) @ directions[selected]
    margin[ep.q_valid == 0] = -1.
    return Result(margin.reshape(ep.q_hw), {"control": "reference_pair_direction",
                  "reference_risk": float(risks[selected]), "candidate_directions": len(directions)})


METHODS.update({"QP04": qp04, "QP06": qp06, "QP07": qp07})
CONTROLS.update({"QP_relative_MMD": relative_mmd, "QP_absolute_pointwise_pooled": absolute_pointwise_pooled,
                 "QP_window_KDE": window_kde, "QP_reference_pair_direction": reference_pair_direction})


def _fit_condition_tree(values, fg_weight, bg_weight, depth=3):
    """CART on legal reference weights; no query values determine cuts."""
    def impurity(f, b):
        return np.divide(2 * f * b, f + b,
                         out=np.zeros_like(np.asarray(f, float)), where=f + b > 0)

    def build(ids, remaining):
        fg, bg = fg_weight[ids].sum(), bg_weight[ids].sum()
        leaf = {"leaf": float((fg - bg) / (fg + bg))}
        baseline = float(impurity(fg, bg))
        if remaining == 0 or len(ids) < 2 or baseline <= 1e-12:
            return leaf
        best_loss, best = baseline, None
        for coordinate in range(values.shape[1]):
            order = np.argsort(values[ids, coordinate], kind="stable")
            sorted_ids = ids[order]
            x = values[sorted_ids, coordinate]
            split = x[:-1] < x[1:]
            if not np.any(split):
                continue
            prefix_f, prefix_b = np.cumsum(fg_weight[sorted_ids])[:-1], np.cumsum(bg_weight[sorted_ids])[:-1]
            loss = impurity(prefix_f, prefix_b) + impurity(fg - prefix_f, bg - prefix_b)
            loss[~split] = np.inf
            at = int(np.argmin(loss))
            if loss[at] < best_loss - 1e-12:
                best_loss = float(loss[at])
                threshold = float(x[at] + (x[at + 1] - x[at]) / 2)
                best = coordinate, threshold, sorted_ids[:at + 1], sorted_ids[at + 1:]
        if best is None:
            return leaf
        coordinate, threshold, left, right = best
        return {"coordinate": coordinate, "threshold": threshold,
                "left": build(left, remaining - 1), "right": build(right, remaining - 1)}

    return build(np.arange(len(values)), depth)


def _route_condition_tree(values, tree):
    margin = np.empty(len(values))

    def route(ids, node):
        if "leaf" in node:
            margin[ids] = node["leaf"]
            return
        left = values[ids, node["coordinate"]] <= node["threshold"]
        route(ids[left], node["left"])
        route(ids[~left], node["right"])

    route(np.arange(len(values)), tree)
    return margin


def _conditional_landmark_tree(ep, axis_source="query", depth=3):
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    q, r = unit(ep.q), unit(ep.r)
    r_ids = np.flatnonzero(ep.wvalid > 0)
    if axis_source == "query":
        centers, _ = dictionary(ep)
    elif axis_source == "reference":
        # Only the two fields consumed by dictionary; no fake labels or GT.
        from types import SimpleNamespace
        centers, _ = dictionary(SimpleNamespace(q=r, q_valid=ep.wvalid))
    else:
        balanced = .5 * (ep.wf / ep.wf.sum() + ep.wb / ep.wb.sum())
        mean = balanced @ r
        variance = balanced @ (r - mean) ** 2
        selected = np.argsort(-variance, kind="stable")[:min(16, r.shape[1])]
        centers = np.eye(r.shape[1])[selected]
    if len(centers) < 2:
        margin = prototype_margin(ep).copy()
        margin.ravel()[ep.q_valid == 0] = -1.
        return Result(margin, {"method": "QP08", "status": "one_landmark_prototype_fallback"})
    ref_values = r[r_ids] @ centers.T
    fg = ep.wf[r_ids] / ep.wf.sum()
    bg = ep.wb[r_ids] / ep.wb.sum()
    tree = _fit_condition_tree(ref_values, fg, bg, depth)
    margin = _route_condition_tree(q @ centers.T, tree)
    margin[ep.q_valid == 0] = -1.
    return Result(margin.reshape(ep.q_hw), {"method": "QP08" if axis_source == "query" and depth == 3 else "tree_control",
                  "axis_source": axis_source, "axes": len(centers), "maximum_depth": depth,
                  "tree": tree, "reference_fit_not_query_transfer_proof": True})


def qp08(ep):
    return _conditional_landmark_tree(ep)


def reference_landmark_tree(ep):
    return _conditional_landmark_tree(ep, "reference")


def query_landmark_stump(ep):
    return _conditional_landmark_tree(ep, "query", 1)


def fixed_coordinate_tree(ep):
    return _conditional_landmark_tree(ep, "fixed_DINO_coordinates")


def query_landmark_linear(ep):
    degenerate = _degenerate(ep)
    if degenerate is not None:
        return degenerate
    centers, _ = dictionary(ep)
    ref_values = unit(ep.r) @ centers.T
    mean_f = ep.wf @ ref_values / ep.wf.sum()
    mean_b = ep.wb @ ref_values / ep.wb.sum()
    margin = unit(ep.q) @ centers.T @ (mean_f - mean_b)
    margin[ep.q_valid == 0] = -1.
    return Result(margin.reshape(ep.q_hw), {"control": "same_query_landmark_linear_mean"})


METHODS["QP08"] = qp08
CONTROLS.update({"QP08_reference_landmark_tree": reference_landmark_tree,
                 "QP08_stump": query_landmark_stump, "QP08_linear": query_landmark_linear,
                 "QP08_fixed_coordinate_tree": fixed_coordinate_tree})
