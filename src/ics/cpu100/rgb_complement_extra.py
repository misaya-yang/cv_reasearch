"""Additional reviewed RGB observations; frozen first batch remains untouched.

Control-only entries are not independent methods. Missing physical RGB retains
the same explicit RGBUnavailable behavior as the frozen first batch.
"""
from __future__ import annotations

import time
import numpy as np
from functools import partial

from .rgb_complement import (_prepared, patch_descriptor, _nearest_distance, _result,
                             _require, _canvas, _token_means, RGB_SIDE, radiometric_color)
from .common import prototype_margin, neighbors, Result
from ics.methods.direct_dino_features import unit


def all_allowed_patch_dictionary(ep):
    """Strong count-zero control: no farthest-sample dictionary compression."""
    started = time.perf_counter()
    rw, qw, fg, bg, _, _ = _prepared(ep, width=8)
    r, q = patch_descriptor(rw), patch_descriptor(qw)
    info = dict(R_FG_windows=int(fg.sum()), R_BG_windows=int(bg.sum()),
                role="same exact RGB descriptor and labels, all allowed R exemplars",
                complexity="O(Nq*Nr*195), Q-block128 distance storage; no full pairwise matrix")
    if not fg.any() or not bg.any():
        h = np.zeros(len(q))
        info["abstention"] = "missing_pure_R_class_windows"
    else:
        df, db = _nearest_distance(q, r[fg]), _nearest_distance(q, r[bg])
        h = np.divide(db - df, db + df, out=np.zeros(len(q)), where=db + df > 1e-12)
        info.update(FG_dictionary_size=int(fg.sum()), BG_dictionary_size=int(bg.sum()))
    return _result(ep, h, info, "RGB04.all_allowed_dictionary", started)


def _rgb_grid(ep, role, *, shading_quotient=False):
    image = ep.r_rgb if role == "r" else ep.q_rgb
    geometry = ep.reference_geometry if role == "r" else ep.query_geometry
    hw = ep.r_hw if role == "r" else ep.q_hw
    canvas = _canvas(image, geometry)
    if not shading_quotient:
        return _token_means(canvas, hw)
    values = unit(canvas.astype(float) / 255.)
    h, w = hw
    if RGB_SIDE % h or RGB_SIDE % w:
        raise ValueError("Exact native RGB cell alignment required")
    return values.reshape(h, RGB_SIDE // h, w, RGB_SIDE // w, 3).mean((1, 3)).reshape(-1, 3)


def shading_quotient(ep, arm="primary"):
    started = time.perf_counter()
    _require(ep)
    rr = _rgb_grid(ep, "r", shading_quotient=arm not in ("raw", "gray"))
    qr = _rgb_grid(ep, "q", shading_quotient=arm not in ("raw", "gray"))
    fg = unit((rr * ep.wf[:, None]).sum(0))
    bg = unit((rr * ep.wb[:, None]).sum(0))
    if arm in ("raw", "gray"):
        # Euclidean raw-intensity prototype keeps class magnitude information.
        mf = (rr * ep.wf[:, None]).sum(0) / max(ep.wf.sum(), 1e-12)
        mb = (rr * ep.wb[:, None]).sum(0) / max(ep.wb.sum(), 1e-12)
        if arm == "gray":
            qr, mf, mb = qr.mean(1, keepdims=True), mf.mean(keepdims=True), mb.mean(keepdims=True)
        df, db = ((qr - mf) ** 2).sum(1), ((qr - mb) ** 2).sum(1)
        h = np.divide(db - df, db + df, out=np.zeros(len(qr)), where=db + df > 1e-12)
    else:
        h = np.einsum("ij,j->i", unit(qr), fg - bg, optimize=False) / 2
    info = dict(invariance_scope="positive scalar at a non-clipped observed RGB pixel or constant-color patch; resize-before-unit need not commute with arbitrary shading",
                empty_black_Q_cells=int((np.linalg.norm(qr, axis=1) < 1e-12).sum()),
                discarded_information="pixel brightness magnitude; achromatic class evidence may be lost")
    return _result(ep, h, info, "RGB07." + arm, started, rgb_only=arm == "rgb_only")


def _edge_observations(features, colors, edges):
    a, b = edges
    dino = np.maximum(1 - np.einsum("ij,ij->i", features[a], features[b], optimize=False), 0)
    rgb = ((colors[a] - colors[b]) ** 2).mean(1)
    return np.column_stack((dino, rgb))


def _density_ratios(observed, same, cut):
    """Same/cut class priors are equal; add-one bin counts are explicit."""
    thresholds = [np.unique(np.quantile(observed[:, j], (.25, .5, .75))) for j in range(2)]
    sizes = [len(t) + 1 for t in thresholds]
    labels = np.column_stack([np.searchsorted(t, observed[:, j], side="left")
                              for j, t in enumerate(thresholds)])
    result = {}
    for arm in ("joint", "dino", "rgb"):
        if arm == "joint":
            idx, n = labels[:, 0] * sizes[1] + labels[:, 1], sizes[0] * sizes[1]
        else:
            j = 0 if arm == "dino" else 1
            idx, n = labels[:, j], sizes[j]
        hs = np.bincount(idx[same], minlength=n).astype(float) + 1
        hc = np.bincount(idx[cut], minlength=n).astype(float) + 1
        ratio = np.log(hs / hs.sum()) - np.log(hc / hc.sum())
        result[arm] = dict(logratio=ratio, same_counts=(hs - 1).tolist(), cut_counts=(hc - 1).tolist())
    return thresholds, sizes, result


def _lookup(observed, thresholds, sizes, model, arm):
    labels = np.column_stack([np.searchsorted(t, observed[:, j], side="left")
                              for j, t in enumerate(thresholds)])
    index = (labels[:, 0] * sizes[1] + labels[:, 1] if arm == "joint"
             else labels[:, 0 if arm == "dino" else 1])
    return np.maximum(model[arm]["logratio"][index], 0) / 4, labels


def _signed_cut(margin, edges, weights):
    """Exact min cut of one fixed integer-quantized signed-unary objective."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order
    m = np.asarray(margin, float).ravel()
    a, b = edges
    n, scale = len(m), 1_000_000
    rows = np.r_[np.full(n, n), np.arange(n), a, b]
    cols = np.r_[np.arange(n), np.full(n, n + 1), b, a]
    cost = np.r_[np.maximum(m, 0), np.maximum(-m, 0), weights, weights]
    quantized = np.rint(cost * scale).astype(np.int64)
    if (quantized < 0).any() or quantized.max(initial=0) >= 2 ** 31:
        raise ValueError("Quantized capacity outside conservative SciPy per-edge contract")
    graph = csr_matrix((quantized, (rows, cols)), shape=(n + 2, n + 2))
    graph.eliminate_zeros()
    flow = maximum_flow(graph, n, n + 1)
    residual = graph - flow.flow
    residual.data = (residual.data > 0).astype(np.int64)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    selected = np.zeros(n + 2, bool)
    selected[reachable] = True
    y = selected[:n]
    energy = float(-m[y].sum() + (weights * (y[a] != y[b])).sum())
    # Subtract the same foreground/background unary offset from flow value.
    quantized_energy = float(flow.flow_value / scale - np.rint(np.maximum(m, 0) * scale).sum() / scale)
    return y, dict(energy=energy, quantized_energy=quantized_energy, capacity_scale=scale,
                   quantized_objective_exact=True, unquantized_optimality_claim=False)


def joint_boundary(ep, arm="primary"):
    started = time.perf_counter()
    _require(ep)
    base = prototype_margin(ep)
    re = neighbors(ep.r_hw)
    a, b = re
    pure = ((ep.wf >= .95) | (ep.wb >= .95)) & (ep.wvalid >= .95)
    valid = pure[a] & pure[b]
    cut = (ep.wf[a] >= .95) != (ep.wf[b] >= .95)
    info = dict(reference_cut_edges=int((valid & cut).sum()), reference_same_edges=int((valid & ~cut).sum()))
    if not (valid & cut).any() or not (valid & ~cut).any():
        info.update(abstention="missing_known_R_cut_or_same_edges", method="RGB06." + arm,
                    query_GT_used=False, natural_segmentation_benefit="unmeasured")
        return Result(base, info)
    ro = _edge_observations(ep.r, _rgb_grid(ep, "r"), re)
    thresholds, sizes, model = _density_ratios(ro[valid], ~cut[valid], cut[valid])
    qe0 = neighbors(ep.q_hw)
    keep = (ep.q_valid[qe0[0]] > 0) & (ep.q_valid[qe0[1]] > 0)
    qe = qe0[0][keep], qe0[1][keep]
    qo = _edge_observations(ep.q, _rgb_grid(ep, "q"), qe)
    weights, labels = _lookup(qo, thresholds, sizes, model, "joint")
    primary_mass = float(weights.sum())
    if arm in ("dino", "rgb"):
        weights, labels = _lookup(qo, thresholds, sizes, model, arm)
        mass = float(weights.sum())
        weights = (weights * primary_mass / mass if mass > 1e-12
                   else np.full(len(weights), primary_mass / max(len(weights), 1)))
        info["zero_evidence_same_mass_uniform_fallback"] = mass <= 1e-12
    elif arm == "uniform":
        weights = np.full(len(weights), primary_mass / max(len(weights), 1))
    elif arm == "zero":
        weights[:] = 0
    m = base.ravel().copy()
    m[ep.q_valid == 0] = -2
    y, certificate = _signed_cut(m, qe, weights)
    margin = np.where(y, 1., -1.).reshape(ep.q_hw)
    if ep.wf.sum() == 0:
        margin[:] = -1
    elif ep.wb.sum() == 0:
        margin[:] = 1
    info.update(method="RGB06." + arm, query_GT_used=False, new_encoder_calls=0,
                natural_segmentation_benefit="unmeasured", DINO_producer=ep.producer,
                reference_thresholds=[x.tolist() for x in thresholds], reference_bin_sizes=sizes,
                reference_likelihoods={k: {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                                           for kk, vv in v.items()} for k, v in model.items()},
                query_edge_bins=labels.tolist(), pairwise_mass=float(weights.sum()),
                primary_pairwise_mass=primary_mass, cut_certificate=certificate,
                wall_seconds=time.perf_counter() - started,
                input_type="actual RGB and known R edge labels plus direct DINO signed unary")
    return Result(margin, info)


def boundary_side_role(ep, arm="primary"):
    started = time.perf_counter()
    _require(ep)
    r, q = _rgb_grid(ep, "r"), _rgb_grid(ep, "q")
    a, b = neighbors(ep.r_hw)
    pure = ((ep.wf >= .95) | (ep.wb >= .95)) & (ep.wvalid >= .95)
    valid = pure[a] & pure[b]
    cut = valid & ((ep.wf[a] >= .95) != (ep.wf[b] >= .95))
    info = dict(R_directed_cut_edges=int(cut.sum()))
    if not cut.any():
        return _result(ep, np.zeros(len(q)), dict(info, abstention="no_pure_R_cut_edges"), "RGB09." + arm, started)
    difference = r[a[cut]] - r[b[cut]]
    difference *= np.where(ep.wf[a[cut]] >= .95, 1., -1.)[:, None]
    direction = unit(unit(difference).mean(0))
    a, b = neighbors(ep.q_hw)
    keep = (ep.q_valid[a] > 0) & (ep.q_valid[b] > 0)
    a, b = a[keep], b[keep]
    distance = np.linalg.norm(q[a] - q[b], axis=1)
    ra, rb = neighbors(ep.r_hw)
    within = pure[ra] & pure[rb] & ((ep.wf[ra] >= .95) == (ep.wf[rb] >= .95))
    scale = max(float(np.median(np.linalg.norm(r[ra[within]] - r[rb[within]], axis=1))) if within.any() else 0., 1 / 255)
    # Union-find components use only observed RGB and the known-R within scale.
    parent = np.arange(len(q))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    if arm != "pointwise":
        for aa, bb in zip(a[distance <= scale], b[distance <= scale]):
            x, y = find(int(aa)), find(int(bb))
            if x != y:
                parent[max(x, y)] = min(x, y)
    component = np.asarray([find(i) for i in range(len(q))])
    boundary = component[a] != component[b]
    aa, bb, dist = a[boundary], b[boundary], distance[boundary]
    vote = np.einsum("ij,j->i", unit(q[aa] - q[bb]), direction, optimize=False) * dist
    totals = np.bincount(np.r_[component[aa], component[bb]], weights=np.r_[vote, -vote], minlength=len(q))
    mass = np.bincount(np.r_[component[aa], component[bb]], weights=np.r_[dist, dist], minlength=len(q))
    h = np.divide(totals, mass, out=np.zeros(len(q)), where=mass > 1e-12)[component]
    info.update(reference_polarity_direction=direction.tolist(), within_R_RGB_scale=scale,
                Q_components=int(len(np.unique(component[ep.q_valid > 0]))),
                assumption="foreground-side color contrast transfers despite absolute-color change; not a bright-target prior")
    return _result(ep, h, info, "RGB09." + arm, started)


def symmetric_rgb_cut(ep):
    """Same signed unary/grid/renderer, conventional Q-only symmetric Potts."""
    started = time.perf_counter()
    _require(ep)
    q = _rgb_grid(ep, "q")
    a, b = neighbors(ep.q_hw)
    keep = (ep.q_valid[a] > 0) & (ep.q_valid[b] > 0)
    a, b = a[keep], b[keep]
    distance = ((q[a] - q[b]) ** 2).mean(1)
    sigma = max(float(np.median(distance)) if len(distance) else 0., 1e-4)
    affinity = np.exp(-distance / sigma)
    degree = max(float(2 * affinity.sum() / max((ep.q_valid > 0).sum(), 1)), 1e-8)
    weights = .25 * affinity / degree
    m = prototype_margin(ep).ravel().copy()
    m[ep.q_valid == 0] = -2
    y, certificate = _signed_cut(m, (a, b), weights)
    return Result(np.where(y, 1., -1.).reshape(ep.q_hw),
                  dict(method="RGB09.symmetric_RGB_Potts_control", query_GT_used=False,
                       input_type="same actual Q RGB/native signed DINO unary/grid/readout",
                       pairwise_mass=float(weights.sum()), cut_certificate=certificate,
                       wall_seconds=time.perf_counter() - started))


METHODS = {"RGB06": joint_boundary, "RGB07": shading_quotient, "RGB09": boundary_side_role}
CONTROLS = {
    "RGB04.all_allowed_dictionary": all_allowed_patch_dictionary,
    "RGB06.dino": partial(joint_boundary, arm="dino"),
    "RGB06.rgb": partial(joint_boundary, arm="rgb"),
    "RGB06.uniform": partial(joint_boundary, arm="uniform"),
    "RGB06.zero": partial(joint_boundary, arm="zero"),
    "RGB07.raw": partial(shading_quotient, arm="raw"),
    "RGB07.gray": partial(shading_quotient, arm="gray"),
    "RGB07.global_affine": radiometric_color,
    "RGB07.rgb_only": partial(shading_quotient, arm="rgb_only"),
    "RGB09.pointwise": partial(boundary_side_role, arm="pointwise"),
    "RGB09.symmetric_Potts": symmetric_rgb_cut,
}
