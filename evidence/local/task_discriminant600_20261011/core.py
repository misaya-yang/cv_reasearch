"""Fixed600 complete task discriminant; legal R and predicted Q roles only.

Pure per-episode mathematics. No file IO, encoder, query GT or dataset selector.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve


ARMS = ("joint_task_discriminant_complete", "reference_task_discriminant_complete",
        "query_task_discriminant_complete", "same_edit_budget_foris_margin_from_S")
FIELDS = ("joint", "reference_only", "query_only")
ROLE_COUNT = 128
RIDGE_LAMBDA = .01
NATIVE_HW = (64, 64)
NODE_COUNT = 4096


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _midpoints(weights):
    w = np.asarray(weights, dtype=np.float64).reshape(NODE_COUNT)
    mass = float(w.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("Unavailable role")
    return np.searchsorted(np.cumsum(w), (np.arange(ROLE_COUNT)+.5)*mass/ROLE_COUNT,
                           side="right").astype(np.int64)


def _roles(fg, bg):
    """Collapse repeated native IDs within an image, including F/B collisions."""
    mf, mb = float(fg.sum()), float(bg.sum())
    stats = {"FG_mass": mf, "BG_mass": mb, "available": mf > 0 and mb > 0}
    if not stats["available"]:
        stats["reason"] = "both_image_roles_required"
        return None, stats
    fi, bi = _midpoints(fg), _midpoints(bg)
    nf = np.bincount(fi, minlength=NODE_COUNT)
    nb = np.bincount(bi, minlength=NODE_COUNT)
    ids = np.flatnonzero(nf+nb)
    count = nf[ids]+nb[ids]
    y = (nf[ids]-nb[ids]).astype(np.float64)/count
    a = count.astype(np.float64)/(2*ROLE_COUNT)
    # Dropped within-ID variance is a constant in the fit objective. Restore
    # it when reporting original balanced +/-1 role-label risks.
    variance_offset = float(np.sum(a*(1-y*y)))
    assert int(nf.sum()) == int(nb.sum()) == ROLE_COUNT and float(a.sum()) == 1.0
    stats.update(unique_native_ids=int(len(ids)), FG_index_sha256=_hash(fi), BG_index_sha256=_hash(bi),
                 collapsed_index_sha256=_hash(ids), target_sha256=_hash(y), base_weight_sha256=_hash(a),
                 mixed_role_ids=int(np.count_nonzero((nf[ids] > 0) & (nb[ids] > 0))),
                 neutral_target_ids=int(np.count_nonzero(y == 0)), within_ID_role_variance=variance_offset,
                 role_samples_each=ROLE_COUNT, base_weight_sum=float(a.sum()), reason=None)
    return {"ids": ids, "y": y, "a": a, "variance_offset": variance_offset}, stats


def _unit(raw, device):
    values = np.asarray(raw)
    if values.shape != (NODE_COUNT, 1024) or not np.isfinite(values).all():
        raise ValueError("Invalid frozen raw O24 input")
    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    norm = torch.linalg.vector_norm(x, dim=1)
    if not bool((torch.isfinite(norm).all() & (norm > 0).all()).item()):
        raise ValueError("Zero/nonfinite raw token is an input error")
    return x/norm[:, None]


def _cpu64(tensor):
    return tensor.detach().to("cpu").to(torch.float64).numpy()


def _system(x, y, a, gram):
    """CPU64 weighted centering and small dual solve, unpenalized intercept."""
    mx, my = a @ x, float(a @ y)
    xc, yc = x-mx, y-my
    matrix = .5*(gram+gram.T)+np.diag(RIDGE_LAMBDA/a)
    dual = cho_solve(cho_factor(matrix, lower=True, check_finite=True), yc, check_finite=True)
    w = xc.T @ dual
    bias = float(my-mx @ w)
    residual = float(np.linalg.norm(matrix @ dual-yc))
    norm_rhs = float(np.linalg.norm(yc))
    if not np.isfinite(w).all() or not math.isfinite(bias) or not math.isfinite(residual):
        raise FloatingPointError("Nonfinite centered ridge solution")
    return w, bias, {"solveResidual": residual,
                    "solve_relative_residual": residual/norm_rhs if norm_rhs else residual,
                    "label_mean": my, "weight_sum": float(a.sum()),
                    "feature_center_norm": float(np.linalg.norm(mx)),
                    "coefficient_norm": float(np.linalg.norm(w)), "intercept": bias,
                    "regularizer": RIDGE_LAMBDA*float(w @ w)}


def _head(images, mixture, unit_q, device, validation_q=None):
    begun = time.monotonic()
    costs, diagnostics = {}, {"reason": None, "image_risk_weights": mixture, "lambda": RIDGE_LAMBDA}
    x = np.concatenate([images[name]["x"] for name in mixture])
    y = np.concatenate([images[name]["roles"]["y"] for name in mixture])
    a = np.concatenate([mixture[name]*images[name]["roles"]["a"] for name in mixture])
    assert float(a.sum()) == 1.0 and np.all(a > 0)
    mx = a @ x
    xc = x-mx
    costs["CPU64_weights_center_seconds"] = time.monotonic()-begun
    tick = time.monotonic()
    centered = torch.as_tensor(xc, dtype=torch.float32, device=device)
    gram = _cpu64(centered @ centered.T)
    if str(device) == "mps":
        torch.mps.synchronize()
    costs["MPS32_Gram_seconds"] = time.monotonic()-tick
    del centered
    tick = time.monotonic()
    try:
        w, bias, solution_stats = _system(x, y, a, gram)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
        diagnostics.update(reason="small_ridge_solve_failure", error=str(error), localfitcost=costs)
        return None, diagnostics
    diagnostics.update(solution_stats)
    costs["CPU64_dual_solve_seconds"] = time.monotonic()-tick
    tick = time.monotonic()
    coefficient = torch.as_tensor(w, dtype=torch.float32, device=device)
    logits = _cpu64(unit_q @ coefficient+np.float32(bias)).reshape(NATIVE_HW)
    if str(device) == "mps":
        torch.mps.synchronize()
    costs["MPS32_Q_readout_seconds"] = time.monotonic()-tick
    if not np.isfinite(logits).all():
        diagnostics.update(reason="nonfinite_query_logit", localfitcost=costs)
        return None, diagnostics
    tick = time.monotonic()
    risks = {}
    for name, image in images.items():
        if image is None:
            risks[name] = None
            continue
        roles = image["roles"]
        prediction = image["x"] @ w+bias
        collapsed = float(np.sum(roles["a"]*(prediction-roles["y"])**2))
        risks[name] = {"collapsed_balanced_MSE": collapsed,
                       "original_role_balanced_MSE": collapsed+roles["variance_offset"],
                       "labels": "legal_R" if name == "R" else "Q_prediction_seeds_not_GT"}
    diagnostics["RPseudoQV_risk"] = risks
    costs["CPU64_legal_R_pseudoQ_risk_seconds"] = time.monotonic()-tick
    if validation_q is not None:
        tick = time.monotonic()
        vx = np.concatenate([images[name]["validation_x"] for name in mixture])
        vc = vx-a @ vx
        vgram = vc @ vc.T
        vw, vb, vs = _system(vx, y, a, vgram)
        reference = (validation_q @ vw+vb).reshape(NATIVE_HW)
        delta = logits-reference
        diagnostics["truth_free_CPU64_reference"] = {
            "max_abs_logit": float(np.max(np.abs(delta))), "RMS_logit": float(np.sqrt(np.mean(delta*delta))),
            "native_sign_difference": int(np.count_nonzero((logits > 0) != (reference > 0))),
            "CPU64_solveResidual": vs["solveResidual"], "seconds": time.monotonic()-tick,
            "prediction_precision_selection": False, "extra_prediction_arm": False}
    costs["total_local_head_seconds"] = time.monotonic()-begun
    diagnostics["localfitcost"] = costs
    diagnostics["training_row_count"] = len(a)
    return logits, diagnostics


def compute(raw_r, raw_q, p, f, b, *, device="mps", validate=False):
    """Joint .5R+.5Q, R-only and Q-only, with independently required roles.

    p is legal reference foreground softcoverage. f/b are Q four-way1111/0000
    pixel-area fractions. Unknown Q pixels receive no labels. validate=True
    records a CPU64 numerical reference only; production logits remain MPS32.
    """
    p, f, b = (np.asarray(v, dtype=np.float64).reshape(NODE_COUNT) for v in (p, f, b))
    if (not all(np.isfinite(v).all() for v in (p, f, b)) or np.any((p < 0) | (p > 1))
            or np.any(f < 0) or np.any(b < 0) or np.any(f+b > 1)):
        raise ValueError("Invalid reference/query role coverage")
    rr, rs = _roles(p, 1-p)
    qr, qs = _roles(f, b)
    stats = {"lambda": RIDGE_LAMBDA, "role128": ROLE_COUNT, "device": str(device),
             "roles": {"R": rs, "Q": qs}, "heads": {}, "validation_requested": bool(validate),
             "label_interpretation": "role-balanced sample labels, not foreground area probability"}
    result = {"fields": dict.fromkeys(FIELDS), "stats": stats}
    unit_q = _unit(raw_q, device)
    unit_r = _unit(raw_r, device) if rr is not None else None
    images = {}
    for name, roles, unit, raw in (("R", rr, unit_r, raw_r), ("Q", qr, unit_q, raw_q)):
        if roles is None:
            images[name] = None
            continue
        indices = torch.as_tensor(roles["ids"], device=device)
        images[name] = {"roles": roles, "x": _cpu64(unit[indices])}
        if validate:
            sampled = np.asarray(raw, dtype=np.float64)[roles["ids"]]
            images[name]["validation_x"] = sampled/np.linalg.norm(sampled, axis=1, keepdims=True)
    validation_q = None
    if validate:
        vq = np.asarray(raw_q, dtype=np.float64)
        validation_q = vq/np.linalg.norm(vq, axis=1, keepdims=True)
    mixtures = {"joint": {"R": .5, "Q": .5}, "reference_only": {"R": 1.0}, "query_only": {"Q": 1.0}}
    for field, mixture in mixtures.items():
        missing = [name for name in mixture if images[name] is None]
        if missing:
            stats["heads"][field] = {"reason": "required_image_roles_unavailable", "missing_images": missing}
            continue
        logits, head_stats = _head(images, mixture, unit_q, device, validation_q)
        result["fields"][field], stats["heads"][field] = logits, head_stats
    return result


def _render(field, shape):
    import torch.nn.functional as F
    tensor = torch.as_tensor(np.ascontiguousarray(field), dtype=torch.float64)[None, None]
    return F.interpolate(tensor, size=tuple(shape), mode="bilinear", align_corners=False)[0, 0].numpy()


def compose_frame(masks, score64, dense):
    """Unscaled logit strict0 readout only in four-way disagreement."""
    strong, bb, gg, rr, nn = (np.asarray(masks[k], dtype=bool) for k in ("S", "B", "G", "R0", "N"))
    assert strong.ndim == 2 and all(v.shape == strong.shape for v in (bb, gg, rr, nn))
    union, intersection = bb | gg | rr, bb & gg & rr
    assert not np.any(nn & ~union)
    da, dd = ~strong & union, strong & ~intersection
    protected = (strong & intersection) | (~strong & ~union)
    outputs, counts = {}, {}
    for arm, field in zip(ARMS[:3], FIELDS):
        native = dense["fields"][field]
        if native is None:
            prediction = strong.copy()
            add, delete = np.zeros(strong.shape, bool), np.zeros(strong.shape, bool)
        else:
            z = _render(np.asarray(native, dtype=np.float64).reshape(NATIVE_HW), strong.shape)
            assert np.isfinite(z).all()
            add, delete = da & (z > 0), dd & (z < 0)
            prediction = (strong & ~delete) | add
            rebuilt = strong.copy()
            rebuilt[da & (z > 0)] = True
            rebuilt[dd & (z < 0)] = False
            assert np.array_equal(prediction, rebuilt)
        assert np.array_equal(prediction[protected], strong[protected])
        outputs[arm] = prediction
        counts[arm] = {"add": int(add.sum()), "delete": int(delete.sum())}
    main = outputs[ARMS[0]]
    ka, kd = int((main & ~strong).sum()), int((strong & ~main).sum())
    score = np.asarray(score64, dtype=np.float64).reshape(NATIVE_HW)
    assert np.isfinite(score).all()
    span = float(np.ptp(score))
    margin = 2*_render((score-score.min())/max(span, 1e-6), strong.shape)-1
    budget, rebuilt = strong.copy(), strong.copy()
    for domain, count, descending, value in ((da, ka, True, True), (dd, kd, False, False)):
        ids = np.flatnonzero(domain.ravel())
        assert 0 <= count <= len(ids)
        rank = np.lexsort((ids, -margin.ravel()[ids] if descending else margin.ravel()[ids]))
        budget.ravel()[ids[rank[:count]]] = value
        stable = np.argsort(-margin.ravel()[ids] if descending else margin.ravel()[ids], kind="stable")
        rebuilt.ravel()[ids[stable[:count]]] = value
    assert np.array_equal(budget, rebuilt)
    assert int((budget & ~strong).sum()) == ka and int((strong & ~budget).sum()) == kd
    assert np.array_equal(budget[protected], strong[protected])
    outputs[ARMS[-1]] = budget
    counts[ARMS[-1]] = {"add": ka, "delete": kd}
    return outputs, {"edit_counts": counts, "edit_counts_relative_host": "S",
                     "D_add": int(da.sum()), "D_delete": int(dd.sum()),
                     "protected_four_way_pixels": int(protected.sum()), "strict_zero_tie_retains_S": True,
                     "no_minmax_tanh_confidence_or_alpha_readout": True,
                     "sameframe_S_budget_positions_verified": True}
