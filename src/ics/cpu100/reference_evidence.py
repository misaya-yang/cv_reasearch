"""Three reference-evidence CPU decisions; real segmentation benefit unverified.

Only legal frozen native tokens and complete reference weights are read.  The
coordinate-based methods deliberately assume the fixed DINO feature basis.
They are not probabilities, target-area estimators, or FoRIS-field corrections.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np

from .common import Episode, Result, prototype_margin, unit, validate

_EPS = 1e-12


def _dot(a, b):
    # Accelerate can leave stale FP status flags; inspect the actual product.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        value = a @ b
    if not np.isfinite(value).all():
        raise FloatingPointError("Nonfinite bounded feature product")
    return value


def _output(ep, margin, **info):
    margin = np.asarray(margin, float).reshape(ep.q_hw).copy()
    if not np.isfinite(margin).all():
        raise FloatingPointError("Nonfinite complete query margin")
    margin.ravel()[ep.q_valid <= 0] = 0
    return Result(margin, {"query_gt_used": False, "new_encoder_forwards": 0,
                           "real_segmentation_gain": "unknown", "fixed_cut": ">0",
                           "source_id": ep.source_id, **info})


def _degenerate(ep):
    if ep.wf.sum() <= 0:
        return _output(ep, -np.ones(ep.q_hw), inactive_reason="empty_reference_foreground")
    if ep.wb.sum() <= 0:
        return _output(ep, np.ones(ep.q_hw), inactive_reason="no_reference_background")
    return None


def _weighted_mean(x, w):
    return np.einsum("nd,n->d", x, w, optimize=False) / w.sum()


def _cost_contrast(fg, bg):
    fg, bg = np.asarray(fg), np.asarray(bg)
    ff, fb = np.isfinite(fg), np.isfinite(bg)
    out = np.zeros_like(fg, dtype=float)
    out[ff & ~fb] = 1
    out[fb & ~ff] = -1
    both = ff & fb
    den = fg[both] + bg[both]
    out[both] = np.divide(bg[both] - fg[both], den, out=np.zeros_like(den), where=den > _EPS)
    return out


def _weighted_midrank(x, w):
    """Separate image CDF; zero-validity padding contributes no mass."""
    out = np.full_like(x, .5, dtype=float)
    take = np.flatnonzero(w > 0)
    if not len(take):
        return out
    for d in range(x.shape[1]):
        order = take[np.argsort(x[take, d], kind="stable")]
        _, start, counts = np.unique(x[order, d], return_index=True, return_counts=True)
        mass = np.add.reduceat(w[order], start)
        mid = (np.cumsum(mass) - .5 * mass) / mass.sum()
        out[order, d] = np.repeat(mid, counts)
    return out


def _centroid_cost_margin(q, r, wf, wb):
    fg, bg = _weighted_mean(r, wf), _weighted_mean(r, wb)
    df = np.sum((q - fg) ** 2, axis=1)
    db = np.sum((q - bg) ** 2, axis=1)
    return _cost_contrast(df, db)


def ordinal_copula(ep: Episode) -> Result:
    validate(ep)
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    r = _weighted_midrank(ep.r, ep.wvalid)
    q = _weighted_midrank(ep.q, ep.q_valid)
    margin = _centroid_cost_margin(q, r, ep.wf, ep.wb)
    return _output(ep, margin, method_id="ref_ordinal_copula",
                   representation="per-image per-channel weighted midrank",
                   coordinate_dependent=True, class_composition_invariance=False,
                   margin_semantics="BG-minus-FG ordinal centroid cost divided by sum")


def _zscore(ep):
    def transform(x, w):
        mu = _weighted_mean(x, w)
        sd = np.sqrt(_weighted_mean((x - mu) ** 2, w))
        return np.divide(x - mu, sd, out=np.zeros_like(x), where=sd > _EPS)
    if ep.q_valid.sum() <= 0:
        return _output(ep, np.zeros(ep.q_hw), control="empty_query_validity")
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    return _output(ep, _centroid_cost_margin(transform(ep.q, ep.q_valid),
                   transform(ep.r, ep.wvalid), ep.wf, ep.wb),
                   control="per-image coordinate z-score centroid")


def _prototype(ep):
    return _output(ep, prototype_margin(ep), control="raw weighted unit FG-BG prototype")


def _squared_distance(a, b):
    return np.maximum(0, np.sum(a*a, axis=1)[:, None]
                      + np.sum(b*b, axis=1)[None, :] - 2*_dot(a, b.T))


def _subsample(ids, maximum=256):
    if len(ids) <= maximum:
        return ids
    return ids[np.linspace(0, len(ids)-1, maximum, dtype=int)]


def _support_packet(ep):
    ids = [_subsample(np.flatnonzero((ep.wf >= .9) & (ep.wvalid >= .9))),
           _subsample(np.flatnonzero((ep.wf <= .1) & (ep.wb >= .9) & (ep.wvalid >= .9)))]
    if min(map(len, ids)) < 3:
        return None, {"inactive_reason": "fewer_than3_unambiguous_reference_rows_per_role",
                      "role_counts": list(map(len, ids)), "fallback": "raw FG-BG prototype"}
    banks, radii, weights = [], [], []
    for selected, w in zip(ids, (ep.wf, ep.wb)):
        bank = ep.r[selected]
        distance = _squared_distance(bank, bank)
        np.fill_diagonal(distance, np.inf)
        k = min(len(bank)-1, int(np.ceil(np.sqrt(len(bank)))))
        radius = np.partition(distance, k-1, axis=1)[:, k-1]
        positive = radius[radius > _EPS]
        if len(positive):
            radius = np.where(radius > _EPS, radius, positive.min())
        else:
            radius = np.zeros_like(radius)
        banks.append(bank)
        radii.append(radius)
        weights.append(w[selected] / w[selected].sum())
    info = {"role_counts": list(map(len, ids)), "reference_rows_maximum_per_role": 256,
            "reference_sampling": "deterministic uniform positions among selected spatial row IDs",
            "role_k": [min(len(b)-1, int(np.ceil(np.sqrt(len(b))))) for b in banks],
            "radius_min_median_max": [[float(h.min()), float(np.median(h)), float(h.max())] for h in radii]}
    return (banks, radii, weights), info


def _support_margins(ep):
    packet, info = _support_packet(ep)
    if packet is None:
        raw = prototype_margin(ep).ravel()
        return {name: raw.copy() for name in ("local", "nearest", "global", "kernel")}, info
    banks, radii, weights = packet
    outputs = {name: np.empty(len(ep.q)) for name in ("local", "nearest", "global", "kernel")}
    for start in range(0, len(ep.q), 32):
        q = ep.q[start:start+32]
        class_costs = {name: [] for name in ("local", "nearest", "global")}
        densities = []
        for bank, h, w in zip(banks, radii, weights):
            d = _squared_distance(q, bank)
            if h.max() <= _EPS:
                normalized = np.where(d <= _EPS, 0., np.inf)
                global_cost = np.where(d.min(axis=1) <= _EPS, 0., np.inf)
            else:
                normalized = d/h[None, :]
                global_cost = d.min(axis=1)/np.median(h)
            class_costs["local"].append(normalized.min(axis=1))
            class_costs["nearest"].append(d.min(axis=1))
            class_costs["global"].append(global_cost)
            # Same original Gaussian-density temperature, identical anchors/weights.
            densities.append(np.einsum("nk,k->n", np.exp(-d/.14), w, optimize=False))
        for key in class_costs:
            outputs[key][start:start+len(q)] = _cost_contrast(*class_costs[key])
        den = densities[0]+densities[1]
        outputs["kernel"][start:start+len(q)] = np.divide(
            densities[0]-densities[1], den, out=np.zeros_like(den), where=den > _EPS)
    return outputs, info


def local_support_radius(ep: Episode) -> Result:
    validate(ep)
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    margins, info = _support_margins(ep)
    return _output(ep, margins["local"], method_id="ref_local_support_radius", **info)


def _support_control(ep, name):
    validate(ep)
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    margins, info = _support_margins(ep)
    return _output(ep, margins[name], control="matched_support_"+name, **info)


def _weighted_median(x, w):
    order = np.argsort(x, kind="stable")
    return float(x[order[np.searchsorted(np.cumsum(w[order]), .5*w.sum(), side="left")]])


def _posterior_ll(p, wf, wb):
    return .5*np.dot(wf, np.log(np.clip(p, _EPS, 1))) + .5*np.dot(wb, np.log(np.clip(1-p, _EPS, 1)))


def _channel_packet(ep):
    wf, wb = ep.wf/ep.wf.sum(), ep.wb/ep.wb.sum()
    mu = _weighted_mean(ep.r, ep.wvalid)
    sd = np.sqrt(_weighted_mean((ep.r-mu)**2, ep.wvalid))
    effect = np.divide(np.abs(_weighted_mean(ep.r, ep.wf)-_weighted_mean(ep.r, ep.wb)),
                       sd, out=np.zeros_like(sd), where=sd > _EPS)
    channels = np.argsort(-effect, kind="stable")[:min(8, ep.r.shape[1])]
    if len(channels) < 3:
        return None
    thresholds = np.array([_weighted_median(ep.r[:, c], ep.wvalid) for c in channels])
    rb, qb = ep.r[:, channels] > thresholds, ep.q[:, channels] > thresholds
    best = None
    for triple in combinations(range(len(channels)), 3):
        tr = list(triple)
        code = rb[:, tr] @ np.array((1, 2, 4))
        f = (np.bincount(code, weights=wf, minlength=8)+1)/9
        b = (np.bincount(code, weights=wb, minlength=8)+1)/9
        post = f[code]/(f[code]+b[code])
        uni_f, uni_b = np.ones(len(ep.r)), np.ones(len(ep.r))
        uni_qf, uni_qb = np.ones(len(ep.q)), np.ones(len(ep.q))
        for j in tr:
            cf = (np.bincount(rb[:, j].astype(int), weights=wf, minlength=2)+1)/3
            cb = (np.bincount(rb[:, j].astype(int), weights=wb, minlength=2)+1)/3
            uni_f *= cf[rb[:, j].astype(int)]
            uni_b *= cb[rb[:, j].astype(int)]
            uni_qf *= cf[qb[:, j].astype(int)]
            uni_qb *= cb[qb[:, j].astype(int)]
        gain = _posterior_ll(post, wf, wb)-_posterior_ll(uni_f/(uni_f+uni_b), wf, wb)
        if best is None or gain > best[0]+1e-15:
            qcode = qb[:, tr] @ np.array((1, 2, 4))
            best = (gain, tr, f, b, qcode, uni_qf, uni_qb, rb[:, tr], qb[:, tr])
    gain, tr, f, b, qcode, uqf, uqb, selected_r, selected_q = best
    return {"gain": float(gain), "channels": channels[tr].tolist(),
            "thresholds": thresholds[tr].tolist(), "r_bits": selected_r,
            "q_bits": selected_q, "joint": np.log(f[qcode]/b[qcode]),
            "univariate": np.log(uqf/uqb), "wf": wf, "wb": wb,
            "q_cell_reference_mass": np.bincount(
                rb[:, tr] @ np.array((1, 2, 4)), weights=ep.wvalid, minlength=8)[qcode]}


def _bit_features(bits, degree):
    signs = 2*bits.astype(float)-1
    columns = [np.ones(len(bits))]
    for order in range(1, degree+1):
        for group in combinations(range(3), order):
            columns.append(np.prod(signs[:, list(group)], axis=1))
    return np.column_stack(columns)


def _code_margins(ep):
    packet = _channel_packet(ep)
    if packet is None:
        raw = prototype_margin(ep).ravel()
        return {name: raw.copy() for name in ("joint", "univariate", "pairwise", "degree3")}, {"inactive_reason": "fewer_than3_feature_channels"}
    margins = {"joint": packet["joint"], "univariate": packet["univariate"]}
    for degree, name in ((2, "pairwise"), (3, "degree3")):
        xr = _bit_features(packet["r_bits"], degree)
        xq = _bit_features(packet["q_bits"], degree)
        w = .5*(packet["wf"]+packet["wb"])
        signed_w = .5*(packet["wf"]-packet["wb"])
        gram = _dot(xr.T, w[:, None]*xr)
        rhs = _dot(xr.T, signed_w)
        coef = np.linalg.solve(gram+.01*np.eye(len(gram)), rhs)
        margins[name] = _dot(xq, coef)
    info = {"selected_channels": packet["channels"], "selected_thresholds": packet["thresholds"],
            "known_reference_balanced_ll_gain_over_univariate": packet["gain"],
            "selection_candidates": 56 if ep.r.shape[1] >= 8 else len(list(combinations(range(ep.r.shape[1]), 3))),
            "query_unseen_selected_cells": int(np.sum((packet["q_cell_reference_mass"] == 0)&(ep.q_valid > 0))),
            "coordinate_dependent": True, "reference_selection_may_overfit": True,
            "joint_score_semantics": "role-balanced finite-cell log density ratio, not calibrated probability"}
    return margins, info


def joint_channel_code(ep: Episode) -> Result:
    validate(ep)
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    margins, info = _code_margins(ep)
    return _output(ep, margins["joint"], method_id="ref_joint_channel_code", **info)


def _code_control(ep, name):
    validate(ep)
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    margins, info = _code_margins(ep)
    return _output(ep, margins[name], control="matched_channel_"+name, **info)


METHODS = {"ref_ordinal_copula": ordinal_copula,
           "ref_local_support_radius": local_support_radius,
           "ref_joint_channel_code": joint_channel_code}
CONTROLS = {"ref_raw_prototype_control": _prototype,
            "ref_ordinal_zscore_control": _zscore,
            "ref_support_nearest_control": lambda ep: _support_control(ep, "nearest"),
            "ref_support_global_radius_control": lambda ep: _support_control(ep, "global"),
            "ref_support_fixed_kernel_control": lambda ep: _support_control(ep, "kernel"),
            "ref_code_univariate_control": lambda ep: _code_control(ep, "univariate"),
            "ref_code_pairwise_control": lambda ep: _code_control(ep, "pairwise"),
            "ref_code_degree3_control": lambda ep: _code_control(ep, "degree3")}


# Batch2 only adds new methods; the frozen first3 functions remain unchanged.

def _tree_packet(ep):
    selected = np.unique(np.r_[
        _subsample(np.flatnonzero(ep.wf > 0), 128),
        _subsample(np.flatnonzero(ep.wb > 0), 128)])
    r, wf, wb = ep.r[selected], ep.wf[selected], ep.wb[selected]
    if not len(r) or wf.sum() <= 0 or wb.sum() <= 0:
        return None
    wf, wb = wf/wf.sum(), wb/wb.sum()
    w = .5*(wf+wb)
    mu = _weighted_mean(r, w)
    variance = _weighted_mean((r-mu)**2, w)
    channels = np.argsort(-variance, kind="stable")[:min(16, r.shape[1])]
    return r[:, channels], wf, wb, channels, selected


def _fit_tree(x, wf, wb, maximum_depth):
    f, b = .5*wf, .5*wb

    def fit(ids, depth):
        fm, bm = float(f[ids].sum()), float(b[ids].sum())
        total = fm+bm
        leaf = {"margin": (fm-bm)/total if total else 0.,
                "fg_mass": fm, "bg_mass": bm, "source_rows": len(ids), "depth": depth}
        if depth >= maximum_depth or len(ids) < 2 or min(fm, bm) <= 0:
            return leaf
        cost = 2*fm*bm/total
        best = None
        for j in range(x.shape[1]):
            order = ids[np.argsort(x[ids, j], kind="stable")]
            xx = x[order, j]
            lf, lb = np.cumsum(f[order])[:-1], np.cumsum(b[order])[:-1]
            rf, rb = fm-lf, bm-lb
            lt, rt = lf+lb, rf+rb
            valid = (xx[:-1] < xx[1:]) & (lt >= .02) & (rt >= .02)
            child = np.full(len(xx)-1, np.inf)
            child[valid] = 2*lf[valid]*lb[valid]/lt[valid] + 2*rf[valid]*rb[valid]/rt[valid]
            k = int(np.argmin(child))
            gain = cost-float(child[k])
            if gain > 1e-12 and (best is None or gain > best[0]+1e-15):
                threshold = float((xx[k]+xx[k+1])*.5)
                best = gain, j, threshold, order[:k+1], order[k+1:]
        if best is None:
            return leaf
        gain, channel, threshold, left, right = best
        return {**leaf, "channel": channel, "threshold": threshold, "gini_gain": gain,
                "left": fit(left, depth+1), "right": fit(right, depth+1)}

    return fit(np.arange(len(x)), 0)


def _tree_predict(x, tree):
    out = np.empty(len(x))
    def visit(ids, node):
        if "channel" not in node:
            out[ids] = node["margin"]
            return
        left = x[ids, node["channel"]] <= node["threshold"]
        visit(ids[left], node["left"])
        visit(ids[~left], node["right"])
    visit(np.arange(len(x)), tree)
    return out


def _tree_decision(ep, depth):
    validate(ep)
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    packet = _tree_packet(ep)
    if packet is None:
        return _output(ep, prototype_margin(ep), inactive_reason="role_missing_after_fixed_tree_sampling")
    x, wf, wb, channels, selected = packet
    tree = _fit_tree(x, wf, wb, depth)
    stump = _fit_tree(x, wf, wb, 1)
    margin = _tree_predict(ep.q[:, channels], tree)
    train_margin = _tree_predict(x, tree)
    stump_margin = _tree_predict(x, stump)
    gain = _posterior_ll((1+train_margin)/2, wf, wb) - _posterior_ll((1+stump_margin)/2, wf, wb)
    return _output(ep, margin, method_id="ref_conditional_tree" if depth==3 else "ref_tree_stump_control",
                   maximum_depth=depth, selected_channels=channels.tolist(), reference_rows=len(selected),
                   reference_sampling="uniform<=128 per positive known role; merged spatial IDs<=256",
                   minimum_child_balanced_mass=.02, tree=tree,
                   known_reference_balanced_ll_gain_over_stump=float(gain),
                   source_new_branch_value_observed=bool(gain > 1e-12),
                   coordinate_dependent=True, score_semantics="source-balanced leaf role margin, not query posterior")


def conditional_tree(ep: Episode) -> Result:
    return _tree_decision(ep, 3)


def _weighted_quantiles(x, weights, levels):
    ids = np.flatnonzero(weights > 0)
    if not len(ids):
        raise ValueError("Role quantiles need positive mass")
    out = np.empty((len(levels), x.shape[1]))
    for d in range(x.shape[1]):
        order = ids[np.argsort(x[ids, d], kind="stable")]
        cumulative = np.cumsum(weights[order])
        positions = np.searchsorted(cumulative, np.asarray(levels)*cumulative[-1], side="left")
        out[:, d] = x[order[np.minimum(positions, len(order)-1)], d]
    return out


def _box_costs(ep):
    f = _weighted_quantiles(ep.r, ep.wf, (.05, .95))
    b = _weighted_quantiles(ep.r, ep.wb, (.05, .95))
    full = _weighted_quantiles(ep.r, ep.wvalid, (.05, .95))
    scale = full[1]-full[0]
    costs = {key: [np.zeros(len(ep.q)), np.zeros(len(ep.q))] for key in ("maximum", "mean", "squared")}
    worst_channels = [np.full(len(ep.q), -1), np.full(len(ep.q), -1)]
    for start in range(0, ep.r.shape[1], 64):
        stop = min(ep.r.shape[1], start+64)
        q = ep.q[:, start:stop]
        span = scale[start:stop]
        for role, bounds in enumerate((f, b)):
            violation = np.maximum(0, np.maximum(bounds[0, start:stop]-q, q-bounds[1, start:stop]))
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized = np.divide(violation, span, out=np.zeros_like(violation), where=span > _EPS)
            normalized[:, span <= _EPS] = (violation[:, span <= _EPS] > _EPS).astype(float)
            maximum = normalized.max(axis=1)
            improve = maximum > costs["maximum"][role]
            worst_channels[role][improve] = start+normalized[improve].argmax(axis=1)
            costs["maximum"][role] = np.maximum(costs["maximum"][role], maximum)
            costs["mean"][role] += normalized.sum(axis=1)
            costs["squared"][role] += np.sum(normalized*normalized, axis=1)
    for key in ("mean", "squared"):
        costs[key] = [v/ep.r.shape[1] for v in costs[key]]
    valid = ep.q_valid > 0
    info = {"role_interval_quantiles": [.05, .95], "channel_scale": "whole-reference valid-weight5%-95% span",
            "coordinate_dependent": True, "constant_reference_channels": int((scale <= _EPS).sum()),
            "query_inside_fg_box": int(((costs["maximum"][0] <= _EPS)&valid).sum()),
            "query_inside_bg_box": int(((costs["maximum"][1] <= _EPS)&valid).sum()),
            "query_inside_both_boxes": int(((costs["maximum"][0] <= _EPS)&(costs["maximum"][1] <= _EPS)&valid).sum()),
            "fg_worst_channel_counts": np.bincount(worst_channels[0][valid & (worst_channels[0] >= 0)], minlength=ep.r.shape[1]).tolist(),
            "bg_worst_channel_counts": np.bincount(worst_channels[1][valid & (worst_channels[1] >= 0)], minlength=ep.r.shape[1]).tolist()}
    return costs, info


def _box_decision(ep, name):
    validate(ep)
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    costs, info = _box_costs(ep)
    return _output(ep, _cost_contrast(*costs[name]),
                   method_id="ref_role_support_box" if name=="maximum" else "ref_box_"+name+"_control",
                   score_semantics="BG-minus-FG support violation; not probability or certified coverage", **info)


def role_support_box(ep: Episode) -> Result:
    return _box_decision(ep, "maximum")


def _energy_banks(ep):
    banks, weights, selected = [], [], []
    for w in (ep.wf, ep.wb):
        ids = _subsample(np.flatnonzero(w > 0), 256)
        banks.append(ep.r[ids])
        weights.append(w[ids]/w[ids].sum())
        selected.append(ids)
    return banks, weights, selected


def _energy_margins(ep):
    banks, weights, selected = _energy_banks(ep)
    self_term = []
    self_squared = []
    for bank, w in zip(banks, weights):
        distance2 = _squared_distance(bank, bank)
        ww = w[:, None]*w[None, :]
        self_term.append(.5*float(np.sum(np.sqrt(distance2)*ww)))
        self_squared.append(.5*float(np.sum(distance2*ww)))
    margins = {key: np.empty(len(ep.q)) for key in ("energy", "squared", "no_self", "kernel", "centroid")}
    for start in range(0, len(ep.q), 32):
        q = ep.q[start:start+32]
        cost, squared, no_self, density, centroid = [], [], [], [], []
        for role, (bank, w) in enumerate(zip(banks, weights)):
            d2 = _squared_distance(q, bank)
            ed = np.einsum("nk,k->n", np.sqrt(d2), w, optimize=False)
            cost.append(ed-self_term[role])
            no_self.append(ed)
            squared.append(np.einsum("nk,k->n", d2, w, optimize=False)-self_squared[role])
            density.append(np.einsum("nk,k->n", np.exp(-d2/.14), w, optimize=False))
            mu = _weighted_mean(bank, w)
            centroid.append(np.sum((q-mu)**2, axis=1))
        sl = slice(start, start+len(q))
        margins["energy"][sl] = cost[1]-cost[0]
        margins["no_self"][sl] = no_self[1]-no_self[0]
        margins["squared"][sl] = squared[1]-squared[0]
        margins["centroid"][sl] = centroid[1]-centroid[0]
        den = density[0]+density[1]
        margins["kernel"][sl] = np.divide(density[0]-density[1], den, out=np.zeros_like(den), where=den > _EPS)
    info = {"selected_role_rows": [len(ids) for ids in selected],
            "reference_sampling": "uniform<=256 spatial positive-mass rows per role; exact same controls",
            "class_unsquared_self_dispersion_half": self_term,
            "squared_energy_centroid_max_error": float(np.max(np.abs(margins["squared"]-margins["centroid"]))),
            "proper_score_assumption": "same class distribution in R and Q; unverified, not pointwise identity",
            "score_semantics": "BG-minus-FG unsquared energy score; not posterior or IoU estimate"}
    return margins, info


def _energy_decision(ep, name):
    validate(ep)
    inactive = _degenerate(ep)
    if inactive is not None:
        return inactive
    margins, info = _energy_margins(ep)
    return _output(ep, margins[name],
                   method_id="ref_distribution_energy" if name=="energy" else "ref_energy_"+name+"_control", **info)


def distribution_energy(ep: Episode) -> Result:
    return _energy_decision(ep, "energy")


METHODS.update({"ref_conditional_tree": conditional_tree,
                "ref_role_support_box": role_support_box,
                "ref_distribution_energy": distribution_energy})
CONTROLS.update({"ref_tree_stump_control": lambda ep: _tree_decision(ep, 1),
                 "ref_box_mean_control": lambda ep: _box_decision(ep, "mean"),
                 "ref_box_squared_control": lambda ep: _box_decision(ep, "squared"),
                 "ref_energy_squared_control": lambda ep: _energy_decision(ep, "squared"),
                 "ref_energy_no_self_control": lambda ep: _energy_decision(ep, "no_self"),
                 "ref_energy_fixed_kernel_control": lambda ep: _energy_decision(ep, "kernel"),
                 "ref_energy_centroid_control": lambda ep: _energy_decision(ep, "centroid")})
