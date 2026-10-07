"""Supplied A001--A050/B051--B075 methods, preserved by stable source ID.

Each registered kernel implements its card. Missing real observations raise an
explicit dependency error; neither stubs nor B0 wrappers are registered methods.
"""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.special import logsumexp

from .a_helpers_001_050 import (EPS, Frame, ObservationUnavailable, b0, fit_rbf,
    fit_ridge, fps, infer_a, local_difference_basis, nearest_mean_distance,
    rank_basis, render_u, spatial_blocks, spherical_modes, sqdist, threshold,
    triangle_bank_distance, triangle_distance, triangles, unit)

METHODS = {}
CONTROLS = {}
REQUIREMENTS = {}
RECIPES = {}


def register(mid, fit, assumptions, controls=()):
    RECIPES[mid] = {"implementation_assumption": list(assumptions), "configurations": 1,
                    "calibration": "four-quadrant, label-state rebuilt, exact balanced coverage threshold",
                    "renderer": "A_U_continuous_original_then_threshold"}
    METHODS[mid] = partial(infer_a, method_id=mid, fit=fit, assumptions=assumptions)
    for name, control_fit in controls:
        cid = f"{mid}__{name}"
        CONTROLS[cid] = partial(infer_a, method_id=cid, fit=control_fit,
            assumptions=assumptions + ("control; not an additional method",))


def _plain(frame):
    return lambda q, ids, role: b0(frame, q), {"mechanism": "A_B0_5NN"}


def _ridge(frame):
    if frame.wf.sum() <= 0 or frame.wb.sum() <= 0:
        return _plain(frame)
    coef = fit_ridge(frame)
    return lambda q, ids, role: (q @ coef[:-1] + coef[-1], {}), {"mechanism": "balanced_full_linear_ridge", "L2": 1.}


def _diag_metric(frame):
    if frame.wf.sum() <= 0 or frame.wb.sum() <= 0:
        return _plain(frame)
    fg = np.sum(frame.x * frame.wf[:, None], 0) / frame.wf.sum()
    bg = np.sum(frame.x * frame.wb[:, None], 0) / frame.wb.sum()
    variance = np.sum((frame.x - fg) ** 2 * frame.wf[:, None] + (frame.x - bg) ** 2 * frame.wb[:, None], 0) / frame.valid.sum()
    metric = (fg - bg) ** 2 / (variance + 1e-4)
    metric /= max(float(metric.mean()), EPS)
    f, b, *_ = frame.banks()
    def predict(q, ids, role):
        return (nearest_mean_distance(q * np.sqrt(metric), b * np.sqrt(metric))
                - nearest_mean_distance(q * np.sqrt(metric), f * np.sqrt(metric))) / 2, {}
    return predict, {"mechanism": "supervised_Fisher_diagonal_metric", "ridge": 1e-4}


def _rbf(frame):
    model = fit_rbf(frame)
    if model is None:
        return _plain(frame)
    bank, coef, width = model
    return lambda q, ids, role: (np.exp(-sqdist(q, bank) / width) @ coef, {}), {"mechanism": "same_anchor_RBF_ridge", "width2": width, "L2": 1.}


def _a001(frame, mode="protected"):
    f, b, *_ = frame.banks()
    if len(f) < 3 or len(b) < 3 or frame.wf.sum() <= 0 or frame.wb.sum() <= 0:
        return _plain(frame)
    uf, ub = local_difference_basis(f), local_difference_basis(b)
    left, cosine, _ = np.linalg.svd(uf.T @ ub, full_matrices=False)
    shared = uf @ left[:, cosine >= 1 - 1e-8]
    if mode == "protected" and shared.shape[1]:
        cross = sqdist(f, b)
        delta = np.r_[f - b[np.argmin(cross, 1)], f[np.argmin(cross, 0)] - b]
        # Intersection with protected differences' orthogonal complement, rather
        # than subtracting components and leaving the shared subspace.
        overlap = delta @ shared
        _, s, v = np.linalg.svd(overlap, full_matrices=True)
        rank = int(np.sum(s > 1e-8 * max(1., s[0] if len(s) else 1)))
        u = shared @ v[rank:].T
    elif mode == "random" and shared.shape[1]:
        u = rank_basis(np.random.default_rng(0).standard_normal((shared.shape[1], frame.x.shape[1])))
    else:
        u = shared
    def transform(q):
        return q @ u @ u.T if mode == "keep" else q - q @ u @ u.T
    if not u.shape[1]:
        transform = lambda q: q
    ff, bb = transform(f), transform(b)
    def predict(q, ids, role):
        qq = transform(q)
        return (nearest_mean_distance(qq, bb) - nearest_mean_distance(qq, ff)) / 2, {"projected": bool(u.shape[1])}
    return predict, {"shared_rank": shared.shape[1], "removed_rank": u.shape[1],
        "shared_principal_cosines": cosine.tolist(), "mode": mode,
        "identity_degeneracy": not bool(u.shape[1])}


register("A001", _a001, (
    "Local differences use up to8 neighbors among64 FPS anchors per role; thin SVD numerical rank uses standard machine epsilon.",
    "Shared subspace is exact within principal-cosine tolerance1e-8; protected intersection nullspace tolerance1e-8.",
    "Projected Euclidean5NN squared-distance difference divided by2; do not unit-normalize away the projection's norm changes.",
    "Purity is conditional foreground coverage wf/wvalid>=.9 or<=.1; padding support excluded.",),
    (("full_ridge", _ridge), ("diagonal_metric", _diag_metric),
     ("random_shared_rank", partial(_a001, mode="random")), ("keep_shared", partial(_a001, mode="keep"))))


def _ellipsoid_projection(t, axes):
    axes = np.maximum(np.asarray(axes), 1e-8)
    hit = np.sum((t / axes) ** 2, 1) > 1
    out = t.copy()
    if np.any(hit):
        z = t[hit]
        lo = np.zeros(len(z)); hi = np.maximum(np.linalg.norm(z, axis=1) * axes.max(), 1.)
        for _ in range(40):
            mid = (lo + hi) / 2
            ratio = np.sum((z * axes / (axes * axes + mid[:, None])) ** 2, 1)
            lo = np.where(ratio > 1, mid, lo); hi = np.where(ratio > 1, hi, mid)
        out[hit] = z * axes * axes / (axes * axes + hi[:, None])
    return out


def _atlas(frame):
    f, b, fi, bi, _ = frame.banks()
    atlas = []
    if len(f) < 3 or len(b) == 0 or np.any(fi < 0):
        return f, b, atlas
    dd = sqdist(f, f)
    for k, center in enumerate(f):
        near = np.argsort(dd[k], kind="stable")[:min(16, len(f))]
        u = rank_basis(f[near] - center, 4)
        if not u.shape[1]:
            continue
        source = frame.fids[frame.blocks[frame.fids] != frame.blocks[fi[k]]]
        if not len(source):
            continue
        # Genuine training-fold cross-block extrapolation observations. Only
        # nearest16 held-out-block points are used for this local atlas piece.
        source = source[np.argsort(sqdist(frame.x[source], center[None])[:, 0], kind="stable")[:16]]
        delta = frame.x[source] - center
        tangent = delta @ u
        axes = np.maximum(np.quantile(np.abs(tangent), .95, axis=0), 1e-6)
        axes *= max(float(np.quantile(np.sqrt(np.sum((tangent / axes) ** 2, 1)), .95)), 1.)
        residual = delta - tangent @ u.T
        normal = max(float(np.quantile(np.linalg.norm(residual, axis=1), .95)), 1e-6)
        selected = np.argsort(sqdist(b, center[None])[:, 0], kind="stable")[:8]
        planes = []
        for j in selected:
            difference = b[j] - center; distance = np.linalg.norm(difference)
            if distance <= EPS:
                continue
            local = np.sort(sqdist(b[j:j + 1], b)[0])
            radius = float(np.sqrt(np.quantile(local[1:min(9, len(b))], .9))) if len(b) > 1 else 0.
            n = difference / distance
            planes.append((n, float(n @ center + max(distance - radius, 0.) / 2)))
        atlas.append((center, u, axes, normal, planes))
    return f, b, atlas


def _piece_projection(q, piece, bounded=True, cut=True):
    center, u, axes, radius, planes = piece
    if not bounded:
        target = center + (q - center) @ u @ u.T
        return np.sum((q - target) ** 2, 1), np.ones(len(q), bool), 0
    planes = planes if cut else []
    # Dykstra computes projection onto the convex intersection, not independent
    # clipping of a single affine fit. Each sweep visits every set.
    sets = 2 + len(planes)
    correction = np.zeros((sets, len(q), q.shape[1]))
    z = q.copy(); previous = z.copy()
    good = np.zeros(len(q), bool)
    for iteration in range(30):
        for j in range(sets):
            v = z + correction[j]
            if j == 0:
                t = (v - center) @ u
                out = v + (_ellipsoid_projection(t, axes) - t) @ u.T
            elif j == 1:
                delta = v - center
                orth = delta - delta @ u @ u.T
                norm = np.linalg.norm(orth, axis=1)
                out = v - orth * np.maximum(1 - radius / np.maximum(norm, EPS), 0)[:, None]
            else:
                n, limit = planes[j - 2]
                out = v - np.maximum(v @ n - limit, 0)[:, None] * n
            correction[j] = v - out
            z = out
        shift = np.linalg.norm(z - previous, axis=1)
        t = (z - center) @ u
        feasibility = np.maximum(np.sqrt(np.sum((t / axes) ** 2, 1)) - 1, 0)
        orth = z - center - t @ u.T
        feasibility = np.maximum(feasibility, np.maximum(np.linalg.norm(orth, axis=1) - radius, 0))
        for n, limit in planes:
            feasibility = np.maximum(feasibility, np.maximum(z @ n - limit, 0))
        good = (shift <= 1e-5) & (feasibility <= 1e-5)
        if good.all():
            break
        previous = z.copy()
    distance = np.sum((q - z) ** 2, 1)
    distance[~good] = np.sum((q[~good] - center) ** 2, 1)
    return distance, good, iteration + 1


def _a002(frame, bounded=True, cut=True):
    f, b, atlas = _atlas(frame)
    if not atlas or frame.wf.sum() <= 0 or frame.wb.sum() <= 0:
        return _plain(frame)
    def predict(q, ids, role):
        dist = np.full(len(q), np.inf); failures = 0; total = 0; max_steps = 0
        for start in range(0, len(q), 64):
            z = q[start:start + 64]
            part = np.full(len(z), np.inf)
            for piece in atlas:
                dd, good, steps = _piece_projection(z, piece, bounded, cut)
                part = np.minimum(part, dd)
                failures += int((~good).sum()); total += len(good); max_steps = max(max_steps, steps)
            dist[start:start + 64] = part
        return (nearest_mean_distance(q, b, 1) - dist) / 2, {
            "piece_projection_failures": failures, "piece_projection_trials": total,
            "failed_projection_uses_original_piece_anchor": True, "max_sweeps": max_steps}
    return predict, {"pieces": len(atlas), "tangent_ranks": [p[1].shape[1] for p in atlas],
                     "normal_tolerances": [p[3] for p in atlas], "bounded": bounded, "BG_cut": cut}


register("A002", _a002, (
    "Each64-FPS FG anchor uses nearest16 anchor PCA rank<=4; widths/tolerance use nearest16 FG from other training spatial blocks,95th percentiles.",
    "Each piece is tangent ellipsoid intersect independent orthogonal ball and up to8 BG-ball bisector halfspaces; BG radius local8-neighbor90th percentile.",
    "Dykstra<=30 sweeps; shift and feasibility<=1e-5; un-converged point-piece uses original anchor distance. Ellipsoid scalar projection40 bisections.",
    "No atlas if cross-block support absent; this source-specified degeneracy is reported, not an executed atlas.",),
    (("unbounded_local_space", partial(_a002, bounded=False)), ("uncut_ellipsoid", partial(_a002, cut=False)),
     ("same_anchor_5NN", _plain), ("full_affinity_RBF", _rbf)))


def _boundary_pairs(frame):
    f, b, fi, bi, _ = frame.banks()
    if np.any(fi < 0) or np.any(bi < 0) or not len(fi) or not len(bi):
        return []
    y, x = np.indices(frame.ep.r_hw)
    coords = np.c_[y.ravel(), x.ravel()]
    # Actual soft-mask boundary sides; held-out cells cannot form either endpoint.
    fg = np.zeros(len(frame.x), bool); fg[frame.fids] = True
    bg = np.zeros(len(frame.x), bool); bg[frame.bids] = True
    fg_border = fg & ndimage.binary_dilation(bg.reshape(frame.ep.r_hw), iterations=2).ravel()
    bg_border = bg & ndimage.binary_dilation(fg.reshape(frame.ep.r_hw), iterations=2).ravel()
    ff = fps(frame.x, np.flatnonzero(fg_border), 64)
    bb = fps(frame.x, np.flatnonzero(bg_border), 64)
    pairs = []
    for i in ff:
        legal = bb[np.sum((coords[bb] - coords[i]) ** 2, axis=1) <= 16]
        if not len(legal):
            continue
        j = int(legal[np.argmin(sqdist(frame.x[i:i + 1], frame.x[legal])[0])])
        local = frame.fids[np.sum((coords[frame.fids] - coords[i]) ** 2, axis=1) <= 4]
        local = local[local != i]
        if not len(local):
            continue
        radius2 = float(np.quantile(sqdist(frame.x[i:i + 1], frame.x[local])[0], .9))
        pairs.append((i, j, radius2))
    return pairs


def _a003(frame, domains=True, permute=False):
    pairs = _boundary_pairs(frame)
    if not pairs:
        return _plain(frame)
    p = frame.x[[v[0] for v in pairs]]
    b = frame.x[[v[1] for v in pairs]]
    if permute:
        b = np.roll(b, 1, axis=0)
    radius = np.array([v[2] for v in pairs])
    fbank, *_ = frame.banks()
    def predict(q, ids, role):
        base, info = b0(frame, q)
        hit = sqdist(q, p) <= radius[None, :] if domains else np.ones((len(q), len(p)), bool)
        diff = q @ (p - b).T
        hit_any = hit.any(1)
        # Signed contrast multiplied by nonnegative ordinary FG cosine support.
        support = np.maximum(1 - nearest_mean_distance(q, fbank) / 2, 0)
        for i in np.flatnonzero(hit_any):
            base[i] = float(np.median(diff[i, hit[i]])) * support[i]
        return base, dict(info, domain_hit_points=int(hit_any.sum()), total_points=len(q))
    return predict, {"boundary_pairs": pairs, "domain": domains, "permuted_pairs": permute,
                     "paired_normal_is_algebraic_distance_difference": True}


def _a003_endpoint_control(frame,k=5):
    pairs=_boundary_pairs(frame)
    if not pairs:return _plain(frame)
    fi=np.unique([v[0] for v in pairs]);bi=np.unique([v[1] for v in pairs])
    f,b=frame.x[fi],frame.x[bi]
    def predict(q,ids,role):
        return (nearest_mean_distance(q,b,k)-nearest_mean_distance(q,f,k))/2,{}
    return predict,{"mechanism":"same_boundary_pair_endpoints_NN","unique_FG_endpoint_ids":fi.tolist(),
        "unique_BG_endpoint_ids":bi.tolist(),"neighbors":k}


register("A003", _a003, (
    "Boundary side candidates lie within2 four-neighbor mask steps; paired endpoints at most4patch Euclidean apart (two boundary-side radii).",
    "FG local radius uses all pureFG within Euclidean2patch neighborhood,90th percentile squared distance.",
    "Ordinary FG support=max(mean5NN cosine,0); contrast median retains its sign.",),
    (("no_domains", partial(_a003, domains=False)), ("permuted_pairs", partial(_a003, permute=True)),
     ("same_endpoint_5NN", _a003_endpoint_control),
     ("same_endpoint_nearest", partial(_a003_endpoint_control,k=1))))


def _maximum_margin(x, y):
    """Finite exact convex SVM dual on the source's coverage anchors."""
    # C=1 per sample, balanced by role count. Unlike revision0's total hinge
    # weight1, this allows inactive margins and actual nonlinear pairwise fits.
    bound = np.where(y > 0, len(y)/(2*max((y > 0).sum(),1)), len(y)/(2*max((y < 0).sum(),1)))
    kernel = (x @ x.T) * y[:, None] * y[None, :]
    def objective(alpha):
        return .5 * alpha @ kernel @ alpha - alpha.sum(), kernel @ alpha - 1
    out = minimize(objective, np.zeros(len(x)), jac=True, method="SLSQP",
        bounds=list(zip(np.zeros(len(x)), bound)),
        constraints={"type":"eq", "fun":lambda a:a@y, "jac":lambda a:y},
        options={"maxiter":100,"ftol":1e-10})
    w = (out.x * y) @ x
    interior = (out.x > 1e-7) & (out.x < bound - 1e-7)
    if interior.any():
        intercept = float(np.mean(y[interior] - x[interior] @ w))
    else:
        # A dual optimum without interior multipliers has an interval of valid
        # intercepts. This midpoint obeys the actual soft-margin KKT bounds.
        lower=[];upper=[]
        residual=y-x@w
        for i in range(len(y)):
            if (y[i]>0 and out.x[i]<=1e-7) or (y[i]<0 and out.x[i]>=bound[i]-1e-7):lower.append(residual[i])
            if (y[i]<0 and out.x[i]<=1e-7) or (y[i]>0 and out.x[i]>=bound[i]-1e-7):upper.append(residual[i])
        lo=max(lower,default=-1.);hi=min(upper,default=1.)
        intercept=float((lo+hi)/2)
    return np.r_[w,intercept], bool(out.success), int(out.nit)


def _a004(frame):
    if not len(frame.fids) or not len(frame.bids):
        return _plain(frame)
    fc, _, fg = spherical_modes(frame.x, frame.fids, 4)
    bc, _, bg = spherical_modes(frame.x, frame.bids, 4)
    coef = np.empty((len(fc), len(bc), frame.x.shape[1] + 1)); failures = 0; iterations = []
    for a, fa in enumerate(fg):
        for b, ba in enumerate(bg):
            # Source compression is explicit: up to16 coverage points per
            # cluster, total<=64 role support points for four clusters.
            fa = fps(frame.x, fa, 16); ba = fps(frame.x, ba, 16)
            coef[a, b], good, nit = _maximum_margin(frame.x[np.r_[fa, ba]], np.r_[np.ones(len(fa)), -np.ones(len(ba))])
            failures += int(not good); iterations.append(nit)
    if failures:
        predict, info = _plain(frame)
        return predict, dict(info, svm_failures=failures, solver_iterations=iterations)
    corners = coef - coef[:, :1] - coef[:1, :] + coef[:1, :1]
    equivalent = bool(np.max(np.abs(corners)) <= 1e-8)
    radii = np.array([max(float(np.quantile(sqdist(frame.x[ids], fc[a:a + 1])[:, 0], .95)), EPS)
                      for a, ids in enumerate(fg)])
    def predict(q, ids, role):
        if equivalent:
            # Zero four-corner implies separable LEARNED potentials; centroid
            # cosine equivalence does not follow. Keep their actual offsets.
            positive = q @ coef[:,0,:-1].T + coef[:,0,-1]
            negative_coef = coef[0,0] - coef[0]
            negative = q @ negative_coef[:,:-1].T + negative_coef[:,-1]
            return positive.max(1)-negative.max(1), {"equivalent_separable_four_corner": True,
                "centroid_cosine_equivalence_not_implied":True}
        out = np.full(len(q), -np.inf)
        for a in range(len(fc)):
            boundaries = q @ coef[a, :, :-1].T + coef[a, :, -1]
            score = boundaries.min(1)
            score = np.minimum(score, (radii[a] - sqdist(q, fc[a:a + 1])[:, 0]) / max(radii[a], EPS))
            out = np.maximum(out, score)
        return out, {"equivalent_separable_four_corner": False}
    return predict, {"FG_modes": len(fc), "BG_modes": len(bc), "four_corner_maxabs": float(np.abs(corners).max()),
        "independent_mechanism": not equivalent, "solver_iterations": iterations, "solver_failures": failures}


def _a004_same_support_rbf(frame):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    fc,_,fg=spherical_modes(frame.x,frame.fids,4)
    bc,_,bg=spherical_modes(frame.x,frame.bids,4)
    fi=np.unique(np.concatenate([fps(frame.x,group,16) for group in fg]))
    bi=np.unique(np.concatenate([fps(frame.x,group,16) for group in bg]))
    bank=frame.x[np.r_[fi,bi]];y=np.r_[np.ones(len(fi)),-np.ones(len(bi))]
    distance=sqdist(bank,bank)
    width=max(float(np.median(distance[np.triu_indices(len(bank),1)])),EPS)
    kernel=np.exp(-distance/width)
    coef=np.linalg.solve(kernel+np.eye(len(bank)),y)
    return lambda q,ids,role:(np.exp(-sqdist(q,bank)/width)@coef,{}),{
        "mechanism":"RBF_same_A004_cluster_support_anchors","FG_support_ids":fi.tolist(),
        "BG_support_ids":bi.tolist(),"width2":width,"L2":1.,"FG_modes":len(fc),"BG_modes":len(bc)}


register("A004", _a004, (
    "Spherical FG/BG clustering K=min(4,pure_count), FPS initialization,<=20 iterations; source permitsK<=16.",
    "Every cluster selects<=16FPS support anchors,total<=64perrole; balanced soft-margin SVM dual C=1persample,SLSQP<=100, freeintercept fromKKT; failed fitreportsB0.",
    "Each FG outreach radius is training-cluster distance95th percentile; intersect via minimum normalized radius margin.",
    "Four-corner coefficient/intercept maxabs<=1e-8 implies max learnedFG−max learnedBG potentials including offsets, not centroidcosine; source's prototype-equivalence wording is underdefined and explicitly repaired.",),
    (("same_anchor_RBF", _a004_same_support_rbf), ("maxFG_minus_maxBG", lambda f: _mode_margin(f))))


def _mode_margin(frame):
    if not len(frame.fids) or not len(frame.bids):
        return _plain(frame)
    fc, *_ = spherical_modes(frame.x, frame.fids, 4)
    bc, *_ = spherical_modes(frame.x, frame.bids, 4)
    return lambda q, ids, role: (np.max(q @ fc.T, 1) - np.max(q @ bc.T, 1), {}), {"FG_modes": len(fc), "BG_modes": len(bc)}


def _a005(frame, trim=True, random=False):
    f, b, fi, bi, _ = frame.banks()
    if not len(f) or not len(b):
        return _plain(frame)
    ft, bt = triangles(f), triangles(b)
    if random:
        rng = np.random.default_rng(0)
        ft = [f[rng.choice(len(f), 3, replace=False)] for _ in ft]
        bt = [b[rng.choice(len(b), 3, replace=False)] for _ in bt]
    cross = []
    if np.all(fi >= 0) and np.all(bi >= 0):
        for block in range(4):
            fa = frame.fids[frame.blocks[frame.fids] == block]
            bb = bi[frame.blocks[bi] != block]
            if len(fa) and len(bb):
                cross.extend(nearest_mean_distance(frame.x[fa], frame.x[bb], 1).tolist())
    cut = float(np.quantile(cross, .1)) if cross else 0.
    before = len(ft)
    if trim:
        ft = [t for t in ft if float(triangle_distance(b, t).min()) >= cut]
    def predict(q, ids, role):
        df = triangle_bank_distance(q, ft, f)
        db = triangle_bank_distance(q, bt, b)
        return (db - df) / 2, {"FG_pieces": len(ft), "BG_pieces": len(bt),
                              "FG_role_anchor_fallback": not bool(ft), "BG_role_anchor_fallback": not bool(bt)}
    return predict, {"FG_pieces_before": before, "FG_pieces_after": len(ft),
        "BG_pieces": len(bt), "cross_block_opposite_distance_q10": cut,
        "safety_measure": "exact_closed_triangle_to_all_BG_anchors", "random_pieces": random}


register("A005", _a005, (
    "One closed triangle per up-to64 FPS role anchor and its nearest2 same-role anchors; exact interior plus edge projection, including degeneracy.",
    "Only FG pieces are source-specified BG-trimmed; BG pieces are the same untrimmed local construction.",
    "Safety cutoff is q.1 of FG training-block to other-block BG-anchor nearest distances; no validcrossblock observations gives0 cutoff and explicit info.",),
    (("no_BG_trim", partial(_a005, trim=False)), ("same_count_random_triangles", partial(_a005, random=True)),
     ("same_vertex_NN", _plain), ("same_vertex_RBF", _rbf)))


def _mask_components(frame):
    ep = frame.ep
    if ep.reference_mask is None:
        raise ObservationUnavailable("A006 needs the complete original reference mask; patch purity is not a substitute")
    mask = np.asarray(ep.reference_mask, bool).copy()
    h, w = mask.shape
    if ep.reference_geometry:
        g = ep.reference_geometry
        sh, sw = g["resized_hw"]; oy, ox = g["padding_top_left"]; side = g["view_side"]
        yy = np.minimum(ep.r_hw[0] - 1, ((oy + (np.arange(h) + .5) * sh / h) * ep.r_hw[0] / side).astype(int))
        xx = np.minimum(ep.r_hw[1] - 1, ((ox + (np.arange(w) + .5) * sw / w) * ep.r_hw[1] / side).astype(int))
    elif mask.shape == ep.r_hw:
        yy, xx = np.arange(h), np.arange(w)
    else:
        raise ObservationUnavailable("A006 needs bound reference geometry to map original mask components to patches")
    native_id = yy[:, None] * ep.r_hw[1] + xx[None, :]
    mask &= frame.train[native_id]
    labels, count = ndimage.label(mask)
    groups = []
    all_foreground = np.bincount(native_id[mask], minlength=len(frame.x))
    for k in range(1, count + 1):
        support = np.bincount(native_id[labels == k], minlength=len(frame.x))
        ids = np.flatnonzero(frame.train & (frame.wf > 0) & (support > 0))
        # Actual complete-mask connectivity is measured at original pixels.
        # Patch mass>1 is evaluated by component's fractional per-patch area.
        pixels = np.bincount(native_id.ravel(), minlength=len(frame.x))
        mass = np.sum(np.divide(support, pixels, out=np.zeros_like(support, float), where=pixels > 0))
        if mass > 1 and len(ids):
            component_wf = frame.wf[ids] * np.divide(support[ids], all_foreground[ids],
                out=np.zeros(len(ids)), where=all_foreground[ids] > 0)
            groups.append((k, ids, mass, component_wf))
    groups.sort(key=lambda z: (-z[2], z[0]))
    return groups[:8]


def _a006(frame, mode="equal"):
    groups = _mask_components(frame)
    if len(groups) <= 1 or frame.wb.sum() <= 0:
        return _plain(frame)
    _, b, *_ = frame.banks()
    allocation = min(16, max(1, 64 // len(groups)))
    def component_bank(ids, weights):
        pure = np.intersect1d(ids, frame.fids)
        if len(pure):
            return frame.x[fps(frame.x, pure, allocation)], False
        return unit(np.sum(frame.x[ids] * weights[:, None], 0))[None], True
    bank_packets = [component_bank(ids, weights) for _, ids, mass, weights in groups]
    banks = [v[0] for v in bank_packets]
    scales = []
    for _, ids, _, component_wf in groups:
        observed = []
        for block in range(4):
            source = ids[frame.blocks[ids] != block]; held = ids[frame.blocks[ids] == block]
            if len(source) and len(held):
                weights = component_wf[np.isin(ids, source)]
                bank, _ = component_bank(source, weights)
                observed.extend(((nearest_mean_distance(frame.x[held], b) - nearest_mean_distance(frame.x[held], bank)) / 2).tolist())
        scales.append(max(float(np.quantile(observed, .75) - np.quantile(observed, .25)) if observed else 1., 1e-6))
    weights = np.ones(len(groups)) if mode == "equal" else np.array([len(g[1]) if mode == "token" else g[2] for g in groups], float)
    weights /= weights.sum()
    def predict(q, ids, role):
        db = nearest_mean_distance(q, b)
        score = np.array([(db - nearest_mean_distance(q, bank)) / (2 * scale) for bank, scale in zip(banks, scales)]).T
        out = np.max(score, 1) if mode == "max" else logsumexp(score + np.log(weights), axis=1)
        return out, {"active_components": len(groups)}
    return predict, {"original_mask_components": [g[0] for g in groups], "patch_masses": [g[2] for g in groups],
                     "FG_anchor_count": sum(map(len, banks)), "BG_anchor_count": len(b), "component_scales": scales, "mode": mode,
                     "component_weighted_mean_fallback": [v[1] for v in bank_packets]}


register("A006", _a006, (
    "Original-mask4-connectivity rebuilt after heldout native-cell footprints removed; component mass in nativepatch equivalents must>1.",
    "At most8 largest validcomponents, stable originalID ties; equalpercomponent anchor cap=min(16,floor(64/componentcount)).",
    "Every real>1patch component retained evenwith no pureFGpatch; thenuseunit coverage-weightedFGmean fromcomponent footprint, allocatingsharedpatch mass bycomponentforegroundpixel proportion.",
    "Component score scale is own trainingcrossblock IQR; no effective IQR=>1, minimum1e-6. Equal logmeanexp temperature1.",),
    (("token_weight", partial(_a006, mode="token")), ("area_weight", partial(_a006, mode="area")),
     ("component_max", partial(_a006, mode="max")), ("allFG_5NN", _plain)))
REQUIREMENTS["A006"] = ["complete_original_reference_mask", "reference_geometry"]


def _fingerprints(frame):
    f, b, fi, bi, _ = frame.banks()
    if not len(f) or not len(b) or np.any(fi < 0) or np.any(bi < 0):
        return None
    cross = sqdist(f, b)
    candidates = [(i, int(np.argmin(cross[i]))) for i in range(len(f))]
    records = []
    for i, j in candidates:
        delta = f[i] - b[j]; eligible = []; weights = []
        for block in range(4):
            held = frame.train & (frame.blocks == block)
            # Pair endpoints are absent from the validation block: no self-bit
            # reliability credit. Outer-fold label state already removed.
            if block in (frame.blocks[fi[i]], frame.blocks[bi[j]]):
                continue
            wf, wb = frame.wf[held], frame.wb[held]
            if wf.sum() <= 0 or wb.sum() <= 0:
                continue
            positive = frame.x[held] @ delta > 0
            error = .5 * (wf[~positive].sum() / wf.sum() + wb[positive].sum() / wb.sum())
            eligible.append(float(error)); weights.append(max(0., 1 - 2 * error))
        if eligible and max(eligible) < .5:
            records.append((i, j, min(weights)))
    if not records:
        return None
    records.sort(key=lambda z: (-z[2], int(fi[z[0]]), int(bi[z[1]])))
    records = records[:64]
    delta = np.array([f[i] - b[j] for i, j, _ in records])
    weight = np.array([w for _, _, w in records]); weight /= weight.sum()
    return f, b, delta, weight, records


def _a007(frame, continuous=False):
    packet = _fingerprints(frame)
    if packet is None:
        return _plain(frame)
    f, b, delta, weight, records = packet
    # Endpoint/bit budget does not restrict the complete legal codeword bank.
    cf, cb = frame.x[frame.fids] @ delta.T > 0, frame.x[frame.bids] @ delta.T > 0
    if continuous:
        reference = frame.x @ delta.T
        model = fit_rbf(frame, reference)
        if model is None:
            return _plain(frame)
        bank, coef, width = model
        return lambda q, ids, role: (np.exp(-sqdist(q @ delta.T, bank) / width) @ coef, {}), {"bits": len(delta), "continuous": True}
    def predict(q, ids, role):
        codes = q @ delta.T > 0
        df = np.array([np.min(np.sum((row[None] != cf) * weight, 1)) for row in codes])
        db = np.array([np.min(np.sum((row[None] != cb) * weight, 1)) for row in codes])
        score = db - df; tied = np.abs(score) <= 1e-12
        base, _ = b0(frame, q[tied]); score[tied] = base
        return score, {"Hamming_ties_B0": int(tied.sum()), "points": len(q)}
    return predict, {"bits": len(delta), "pair_reliabilities": [r[2] for r in records], "continuous": False}


register("A007", _a007, (
    "Candidatebits are64FPS FG anchors each paired nearestBG anchor; must have balancederror<.5 in EVERY eligibleother spatialblock.",
    "Blocks containing either pair endpoint give no reliability credit. Bitweight=min across eligibleblocks(1−2balancederror), normalized.",
    "Reference codeword banks contain ALL pure-role legalRpatches;64cap applies onlyto bitendpoints. Minimum weightedHamming perrole, exactscoreties use sourceB0.",),
    (("same_pairs_continuous_RBF", partial(_a007, continuous=True)), ("original_5NN", _plain)))


def _a008(frame, domain=True, permute=False, expert_cap=8):
    if not len(frame.fids) or not len(frame.bids):
        return _plain(frame)
    centers, _, groups = spherical_modes(frame.x, frame.fids, 8)
    f, b, *_ = frame.banks()
    experts = []
    for k, center in enumerate(centers):
        bb = b[np.argsort(sqdist(b, center[None])[:, 0], kind="stable")[:expert_cap]]
        ff = frame.x[fps(frame.x, groups[k], max(1, 64 // len(groups)))]
        radius = max(float(np.quantile(sqdist(frame.x[groups[k]], center[None])[:, 0], .95)), EPS)
        response = frame.x[groups[k]] @ bb.T
        experts.append((ff, bb, radius, np.quantile(response, [.05, .95], axis=0)))
    if permute:
        negatives = [e[1] for e in experts]
        experts = [(e[0], negatives[(i + 1) % len(experts)], e[2], e[3]) for i, e in enumerate(experts)]
    def predict(q, ids, role):
        distance = sqdist(q, centers)
        logweight = -distance / .07
        weights = np.exp(logweight - logsumexp(logweight, axis=1, keepdims=True))
        base, info = b0(frame, q)
        out, active = np.zeros(len(q)), np.zeros(len(q))
        for k, (ff, bb, radius, responses) in enumerate(experts):
            hit = distance[:, k] <= radius if domain else np.ones(len(q), bool)
            score = (nearest_mean_distance(q, bb) - nearest_mean_distance(q, ff)) / 2
            out += weights[:, k] * np.where(hit, score, base)
            active += hit * weights[:, k]
        return out, dict(info, local_expert_weight_mean=float(active.mean()) if len(q) else 0,
                         any_domain_hit_points=int((active > 0).sum()))
    return predict, {"FG_modes": len(centers), "expert_BG_counts": [len(e[1]) for e in experts],
        "FG_support_radius2": [e[2] for e in experts], "FG_to_BG_response_ranges": [e[3].tolist() for e in experts],
        "domain": domain, "BG_binding_permuted": permute}


register("A008", _a008, (
    "FG sphericalK=min(8,pureFGcount), sourceFPS20iterations; each modeFG anchors cap=floor(64/K), commonBG64 anchors, nearest8BG perexpert.",
    "FG mode support radius squared Euclideandistance95th percentile, storedFG-to-BGresponse range5/95; softmode weights exp(−distance2/.07).",
    "Each mode contributes FG5NN−itsBG5NN inside radius, originalB0 outside; weights normalized overALL modes, noquerypseudo labels.",),
    (("no_domain", partial(_a008, domain=False)), ("permuted_BG_experts", partial(_a008, permute=True)),
     ("nearest_single_BG", partial(_a008, expert_cap=1)), ("all_BG_5NN", _plain), ("local_RBF", _rbf)))


from .a_algorithms_009_020 import install as _install_009_020
_install_009_020(register, REQUIREMENTS)
from .a_algorithms_023_040 import install as _install_023_040
_install_023_040(register, REQUIREMENTS)

from .a_algorithms_009_020 import full_profile_rbf as _full_profile_rbf
for _mid in ("A009","A011","A012","A013","A014","A015","A016","A017","A018","A019","A020","A023","A024","A025"):
    CONTROLS[_mid+"__complete_reference_profile_RBF"] = partial(infer_a,
        method_id=_mid+"__complete_reference_profile_RBF",fit=_full_profile_rbf,
        assumptions=("Controlonly; complete legalR profiles includingALL trainingreference columns,64FPS supervisedhead pointsperrole, medianwidth,L2=1.",))
