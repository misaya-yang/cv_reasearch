"""Fixed600 query task graph around saved strong S; no file, label or GT IO."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg


ARMS = ("query_task_graph_complete", "query_feature_graph_complete",
        "query_uniform_geometry_complete", "same_edit_budget_foris_margin_from_S")
FIELDS = ("full", "feature_only", "geometry_only")
NATIVE_HW = (64, 64)
NODE_COUNT = 4096
CG_RTOL = 1e-7


def _edges():
    """Each undirected eight-neighbour edge exactly once, row-major."""
    grid = np.arange(NODE_COUNT, dtype=np.int64).reshape(NATIVE_HW)
    first, second = [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ys = slice(0, 64-dy)
        xs = slice(max(0, -dx), min(64, 64-dx))
        first.append(grid[ys, xs].ravel())
        second.append(grid[slice(dy, 64), slice(max(0, dx), min(64, 64+dx))].ravel())
    i, j = np.concatenate(first), np.concatenate(second)
    assert len(i) == len(j) == 16002
    return i, j


def _seed_arrays(foreground_seed, background_seed):
    f, b = (np.asarray(v, dtype=np.float64).reshape(NODE_COUNT) for v in (foreground_seed, background_seed))
    if (not np.isfinite(f).all() or not np.isfinite(b).all()
            or np.any(f < 0) or np.any(b < 0) or np.any(f+b > 1)):
        raise ValueError("Seed weights must be disjoint pixel-area fractions in [0,1]")
    return f, b


def _solve(i, j, weights, f, b):
    """CPU64 anchored Laplacian; unsupported components remain unavailable."""
    started = time.monotonic()
    stats = {"reason": None, "CG_rtol": CG_RTOL, "CG_atol": 0.0,
             "CG_maxiter": NODE_COUNT, "iterations": 0}
    if not np.isfinite(weights).all() or np.any(weights < 0):
        stats["reason"] = "invalid_edge_weights"
        return None, None, stats
    nonzero = weights > 0
    ii, jj, ww = i[nonzero], j[nonzero], weights[nonzero]
    adjacency = sparse.csr_matrix((np.concatenate((ww, ww)),
                                 (np.concatenate((ii, jj)), np.concatenate((jj, ii)))),
                                shape=(NODE_COUNT, NODE_COUNT), dtype=np.float64)
    adjacency.eliminate_zeros()
    n_components, component = connected_components(adjacency, directed=False)
    mass = np.bincount(component, weights=f+b, minlength=n_components)
    supported = mass[component] > 0
    stats.update(positive_edges=int(nonzero.sum()), underflow_zero_edges=int((~nonzero).sum()),
                 graph_components=int(n_components), unanchored_components=int(np.count_nonzero(mass == 0)),
                 unanchored_nodes=int(np.count_nonzero(~supported)))
    # True exp underflow can disconnect the graph. No epsilon edge or BG label
    # is invented: solve only anchored components, keep unknown components at
    # the neutral placeholder .5 and explicitly suppress their output edits.
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    matrix = sparse.diags(degree+f+b, format="csr")-adjacency
    ids = np.flatnonzero(supported)
    if not len(ids):
        stats["reason"] = "no_anchored_graph_component"
        return None, None, stats
    matrix = matrix[ids][:, ids].tocsr()
    diagonal = matrix.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        stats["reason"] = "invalid_matrix_diagonal"
        return None, None, stats
    with np.errstate(over="ignore", divide="ignore"):
        inverse = 1.0/diagonal
    if not np.isfinite(inverse).all():
        stats["reason"] = "numerically_unrepresentable_diagonal_preconditioner"
        return None, None, stats
    rhs = f[ids]
    preconditioner = sparse.diags(inverse, format="csr")

    def iteration(_):
        stats["iterations"] += 1

    try:
        solution, info = cg(matrix, rhs, rtol=CG_RTOL, atol=0.0,
                            maxiter=NODE_COUNT, M=preconditioner, callback=iteration)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        stats.update(reason="CG_runtime_failure", error=str(error), wall_seconds=time.monotonic()-started)
        return None, None, stats
    residual = float(np.linalg.norm(matrix @ solution-rhs))
    norm_rhs = float(np.linalg.norm(rhs))
    stats.update(CG_info=int(info), residual_l2=residual, rhs_l2=norm_rhs,
                 relative_residual=residual/norm_rhs if norm_rhs else residual,
                 wall_seconds=time.monotonic()-started)
    if (info != 0 or not np.isfinite(solution).all() or not math.isfinite(residual)
            or residual > CG_RTOL*norm_rhs):
        stats["reason"] = "CG_convergence_or_residual_failure"
        return None, None, stats
    stats.update(min_before_roundoff_clip=float(solution.min()), max_before_roundoff_clip=float(solution.max()))
    # Exact maximum principle gives [0,1]. Clamp only the completed numerical
    # solution; retain the original extrema and residual as audit scalars.
    field = np.full(NODE_COUNT, .5, dtype=np.float64)
    field[ids] = np.clip(solution, 0, 1)
    return field.reshape(NATIVE_HW), supported.reshape(NATIVE_HW), stats


def compute(raw_q, rgb64, foreground_seed, background_seed, *, device="mps"):
    """Shared cached Q geometry; three fixed complete-field controls.

    rgb64 is the mean RGB/255 of each actual encoded-query 16x16 patch.
    Seed fractions are area(S&B&G&R0) and area(~S&~B&~G&~R0),
    in the same CLI. All other pixel memberships are unknown.
    MPS computes unit features/adjacent squared distances; sparseCG is CPU64.
    """
    f, b = _seed_arrays(foreground_seed, background_seed)
    stats = {"FG_seed_mass": float(f.sum()), "BG_seed_mass": float(b.sum()),
             "mixed_role_nodes": int(np.count_nonzero((f > 0) & (b > 0))),
             "unknown_area_mass": float((1-f-b).sum()), "device": str(device),
             "edge_count": 16002, "fields": {}}
    result = {"fields": dict.fromkeys(FIELDS), "support": dict.fromkeys(FIELDS), "stats": stats}
    if f.sum() <= 0 or b.sum() <= 0:
        stats["reason"] = "both_query_seed_roles_required"
        for name in FIELDS:
            stats["fields"][name] = {"reason": "both_query_seed_roles_required"}
        return result
    rgb = np.asarray(rgb64, dtype=np.float64)
    if rgb.shape != (64, 64, 3) or not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError("rgb64 must be finite64x64x3 actual-query patch RGB in[0,1]")
    raw = np.asarray(raw_q)
    if raw.shape != (4096, 1024) or not np.isfinite(raw).all():
        raise ValueError("Invalid frozen query O24")
    i, j = _edges()
    q = torch.as_tensor(raw, dtype=torch.float32, device=device)
    norms = torch.linalg.vector_norm(q, dim=1)
    if not bool((torch.isfinite(norms).all() & (norms > 0).all()).item()):
        raise ValueError("Zero/nonfinite raw query token is an input error")
    q = q/norms[:, None]
    feature_d2 = []
    for begin in range(0, len(i), 1024):
        left = torch.as_tensor(i[begin:begin+1024], device=device)
        right = torch.as_tensor(j[begin:begin+1024], device=device)
        difference = q[left]-q[right]
        feature_d2.append((difference*difference).sum(dim=1).detach().to("cpu").to(torch.float64).numpy())
    df = np.concatenate(feature_d2)
    colors = rgb.reshape(NODE_COUNT, 3)
    dc = np.square(colors[i]-colors[j]).sum(axis=1)
    del q, norms, difference
    if not np.isfinite(df).all() or not np.isfinite(dc).all():
        raise ValueError("Nonfinite adjacent-query distance")
    median_f, median_c = float(np.median(df)), float(np.median(dc))
    feature_term = df/median_f if median_f > 0 else np.zeros_like(df)
    color_term = dc/median_c if median_c > 0 else np.zeros_like(dc)
    stats.update(feature_distance_median=median_f, RGB_distance_median=median_c,
                 feature_zero_scale_omitted=median_f == 0, RGB_zero_scale_omitted=median_c == 0)
    weights = {"full": np.exp(-feature_term-color_term),
               "feature_only": np.exp(-feature_term), "geometry_only": np.ones_like(df)}
    for name in FIELDS:
        field, support, diagnostic = _solve(i, j, weights[name], f, b)
        result["fields"][name], result["support"][name] = field, support
        stats["fields"][name] = diagnostic
    return result


def _render(field, shape, mode="bilinear"):
    import torch.nn.functional as F
    tensor = torch.as_tensor(np.ascontiguousarray(field), dtype=torch.float64)[None, None]
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(tensor, size=tuple(shape), mode=mode, **kwargs)[0, 0].numpy()


def compose_frame(masks, score64, dense):
    """Strict .5 graph decisions only in S/whole disagreement; fallback S."""
    strong, b, g, r, n = (np.asarray(masks[k], dtype=bool) for k in ("S", "B", "G", "R0", "N"))
    assert strong.ndim == 2 and all(v.shape == strong.shape for v in (b, g, r, n))
    union = b | g | r
    assert not np.any(n & ~union)
    intersection = b & g & r
    da, dd = ~strong & union, strong & ~intersection
    fixed_FG, fixed_BG = strong & intersection, ~strong & ~union
    outputs, certificates = {}, {}
    for arm, name in zip(ARMS[:3], FIELDS):
        field = dense["fields"][name]
        if field is None:
            prediction = strong.copy()
            add, delete = np.zeros(strong.shape, bool), np.zeros(strong.shape, bool)
        else:
            u = _render(np.asarray(field, dtype=np.float64).reshape(NATIVE_HW), strong.shape)
            support = dense.get("support", {}).get(name)
            valid = np.ones(strong.shape, bool) if support is None else _render(support, strong.shape, "nearest") > .5
            assert np.isfinite(u).all()
            add, delete = da & valid & (u > .5), dd & valid & (u < .5)
            prediction = (strong & ~delete) | add
            alternate = strong.copy()
            alternate[da & valid & (u > .5)] = True
            alternate[dd & valid & (u < .5)] = False
            assert np.array_equal(prediction, alternate)
        assert np.array_equal(prediction[fixed_FG | fixed_BG], strong[fixed_FG | fixed_BG])
        outputs[arm] = prediction
        certificates[arm] = {"add": int(add.sum()), "delete": int(delete.sum())}
    main = outputs[ARMS[0]]
    ka, kd = int((main & ~strong).sum()), int((strong & ~main).sum())
    score = np.asarray(score64, dtype=np.float64).reshape(NATIVE_HW)
    if not np.isfinite(score).all():
        raise ValueError("Invalid frozen FoRIS score64")
    span = float(np.ptp(score))
    margin = 2*_render((score-score.min())/max(span, 1e-6), strong.shape)-1
    budget, alternate = strong.copy(), strong.copy()
    for domain, count, descending, value in ((da, ka, True, True), (dd, kd, False, False)):
        ids = np.flatnonzero(domain.ravel())
        assert count <= len(ids)
        rank = np.lexsort((ids, -margin.ravel()[ids] if descending else margin.ravel()[ids]))
        budget.ravel()[ids[rank[:count]]] = value
        stable = np.argsort(-margin.ravel()[ids] if descending else margin.ravel()[ids], kind="stable")
        alternate.ravel()[ids[stable[:count]]] = value
    assert np.array_equal(budget, alternate)
    assert int((budget & ~strong).sum()) == ka and int((strong & ~budget).sum()) == kd
    assert np.array_equal(budget[fixed_FG | fixed_BG], strong[fixed_FG | fixed_BG])
    outputs[ARMS[-1]] = budget
    certificates[ARMS[-1]] = {"add": ka, "delete": kd}
    return outputs, {"edit_counts": certificates, "edit_counts_relative_host": "S",
                     "D_add": int(da.sum()), "D_delete": int(dd.sum()),
                     "protected_FG": int(fixed_FG.sum()), "protected_BG": int(fixed_BG.sum()),
                     "strict_half_tie_retains_S": True, "no_confidence_or_alpha_gate": True,
                     "sameframe_S_budget_positions_verified": True}
