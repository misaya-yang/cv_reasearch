"""Frozen D unary with query graph and reference-cross-edge extent inference.

The source labels are fixed, so cross-image energy is exactly a degree-weighted
reference-label unary. This is not iterated correspondence or latent source
inference. Its complete output requires ``finalize`` with the source FoRIS CRF.
The module loads no model, image, query annotation, or prediction mask.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
import torch
import torch.nn.functional as F

from . import rcg


CONFIG = {
    "name": "D_unary_reference_graph_extent_v1",
    "score_origin": "multilayer.transition.part4.raw",
    "graph_feature_space": "existing_last_layer_debiased_cache",
    "query_k": 20,
    "cross_image_k": 10,
    "source_purity": 0.9,
    "query_graph_lambda": 16.0,
    "reference_unary_strength": 1.0,
    "confidence_floor": 0.1,
    "normalization_epsilon": 1e-6,
    "cg_rtol": 1e-7,
    "cg_atol": 1e-9,
    "cg_maxiter": 300,
    "new_encoder_forwards": 0,
    "query_gt_used": False,
    "finalizer": "bilinear 1024 >0.5, no re-minmax; source FoRIS CRF",
    "parameter_provenance": "RCG graph/solver constants; new source coefficient=1 before outcomes",
}


def _unary(score, extras):
    if extras is None or extras.get("score_origin") != CONFIG["score_origin"]:
        raise ValueError("Require explicit frozen D raw part-4 score provenance")
    raw = np.asarray(score, dtype=np.float32)
    if raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError("Require a finite two-dimensional raw D score")
    return (raw - raw.min()) / max(float(raw.max() - raw.min()),
                                  CONFIG["normalization_epsilon"])


def _inputs(q, r, cov, shape, device):
    q = torch.as_tensor(q, device=device, dtype=torch.float32)
    r = torch.as_tensor(r, device=device, dtype=torch.float32)
    c = np.asarray(cov, dtype=np.float32).reshape(-1)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("Matching query/reference feature channels required")
    if len(q) != int(np.prod(shape)) or len(r) != len(c) or min(len(q), len(r)) < 2:
        raise ValueError("Feature tokens must agree with score/coverage grids")
    if not torch.isfinite(q).all() or not torch.isfinite(r).all() or not np.isfinite(c).all():
        raise ValueError("Nonfinite cache input")
    if (q.norm(dim=1) == 0).any() or (r.norm(dim=1) == 0).any():
        raise ValueError("Zero-norm cache token")
    if c.min() < 0 or c.max() > 1 or c.sum() <= 0 or (1 - c).sum() <= 0:
        raise ValueError("Coverage in [0,1] with both source roles is required")
    return F.normalize(q, dim=1), F.normalize(r, dim=1), c


def _adaptive(values):
    distance = (1 - values).clamp_min(0)
    return torch.exp(-distance / distance[:, -1:].clamp_min(1e-6))


def _query_graph(q):
    n = len(q)
    k = min(CONFIG["query_k"], n - 1)
    affinity = q @ q.T
    affinity.fill_diagonal_(-2)
    values, index = affinity.topk(k, dim=1)
    del affinity
    weights = _adaptive(values).cpu().numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(n), k),
                                    index.cpu().numpy().ravel())), shape=(n, n))
    w = w.multiply(w.T).tocsr()
    w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(axis=1)).ravel()
    w = w / max(float(degree.mean()), 1e-8)
    return w.astype(np.float64).tocsr()


def _cross_graph(q, r, c):
    purity = CONFIG["source_purity"]
    fg, bg = c >= purity, c <= 1 - purity
    fallback = {"foreground": not bool(fg.any()), "background": not bool(bg.any())}
    if fallback["foreground"]:
        fg[np.argmax(c)] = True
    if fallback["background"]:
        bg[np.argmin(c)] = True
    selected = np.flatnonzero(fg | bg)
    labels = c[selected].astype(np.float64)
    ref = r[torch.as_tensor(selected, device=r.device)]
    affinity = q @ ref.T
    kq = min(CONFIG["cross_image_k"], len(ref))
    kr = min(CONFIG["cross_image_k"], len(q))
    vq, iq = affinity.topk(kq, dim=1)
    vr, ir = affinity.T.topk(kr, dim=1)
    del affinity, ref
    wq = sparse.csr_matrix((_adaptive(vq).cpu().numpy().ravel(),
                           (np.repeat(np.arange(len(q)), kq), iq.cpu().numpy().ravel())),
                          shape=(len(q), len(selected)))
    wr = sparse.csr_matrix((_adaptive(vr).cpu().numpy().ravel(),
                           (np.repeat(np.arange(len(selected)), kr), ir.cpu().numpy().ravel())),
                          shape=(len(selected), len(q)))
    cross = wq.multiply(wr.T).tocsr()
    cross.data = np.sqrt(cross.data)
    cross.eliminate_zeros()
    degree = np.asarray(cross.sum(axis=1)).ravel()
    normalizer = float(degree.mean())
    if normalizer > 0:
        cross = cross / normalizer
    cross = cross.astype(np.float64).tocsr()
    return cross, labels, {
        "reference_selected_tokens": len(selected),
        "reference_selected_fg_tokens": int(fg[selected].sum()),
        "reference_selected_bg_tokens": int(bg[selected].sum()),
        "reference_purity_fallback": fallback,
        "cross_edges": int(cross.nnz),
        "cross_unconnected_queries": int((degree == 0).sum()),
        "cross_unscaled_mean_query_degree": normalizer,
    }


def _solve(s, w, cross, labels, mode):
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    degree = np.asarray(w.sum(axis=1)).ravel()
    a = CONFIG["confidence_floor"] + np.abs(2 * s - 1)
    a = a / a.mean()
    cross_degree = np.asarray(cross.sum(axis=1)).ravel()
    cross_target = np.asarray(cross @ labels).ravel()
    total_mass = float(cross_degree.sum())
    global_mean = float(cross_target.sum() / total_mass) if total_mass else float(labels.mean())
    if mode == "no_reference":
        cross_degree = np.zeros_like(s)
        cross_target = np.zeros_like(s)
    elif mode == "global_mean_reference":
        # One global scalar preserves total cross-edge label mass. This is not
        # each row's weighted label mean, which would equal the primary exactly.
        cross_target = cross_degree * global_mean
    elif mode != "reference":
        raise ValueError(f"Unknown composition mode: {mode}")
    coefficient = CONFIG["reference_unary_strength"]
    h = (sparse.diags(a + coefficient * cross_degree)
         + CONFIG["query_graph_lambda"] * (sparse.diags(degree) - w)).tocsr()
    rhs = a * s + coefficient * cross_target
    iterations = [0]

    def callback(_):
        iterations[0] += 1

    z, status = cg(h, rhs, x0=s, rtol=CONFIG["cg_rtol"], atol=CONFIG["cg_atol"],
                   maxiter=CONFIG["cg_maxiter"], callback=callback)
    if status != 0:
        raise RuntimeError(f"Composition CG failed: status={status}")
    residual = float(np.linalg.norm(h @ z - rhs) / max(np.linalg.norm(rhs), 1e-12))
    if not np.isfinite(z).all():
        raise RuntimeError("Nonfinite composition solution")
    info = {
        "cg_iterations": iterations[0], "cg_relative_residual": residual,
        "query_graph_undirected_edges": int(w.nnz // 2),
        "query_graph_isolated_tokens": int((degree == 0).sum()),
        "source_global_cross_edge_weighted_mean": global_mean,
        "source_term_equivalence": "fixed labels reduce cross energy to degree-weighted unary",
        "maximum_principle_error": float(max(0, z.max() - 1, -z.min())),
        "unary_mean": float(s.mean()), "output_mean": float(z.mean()),
        "unary_foreground_token_fraction": float((s > 0.5).mean()),
        "output_foreground_token_fraction": float((z > 0.5).mean()),
        "source_cross_degree_quantiles": np.quantile(cross_degree, [0, .25, .5, .75, 1]).tolist(),
        "soft_add_mass": float(np.maximum(z - s, 0).sum()),
        "soft_delete_mass": float(np.maximum(s - z, 0).sum()),
    }
    return z, info


@torch.inference_mode()
def _run(q, r, cov, score, *, device, extras, mode):
    started = time.perf_counter()
    s = _unary(score, extras)
    q, r, c = _inputs(q, r, cov, s.shape, device)
    w = _query_graph(q)
    cross, labels, graph_info = _cross_graph(q, r, c)
    z, info = _solve(s, w, cross, labels, mode)
    info.update(graph_info)
    info.update({"config": dict(CONFIG), "mode": mode,
                 "query_gt_used": False, "extra_encoder_forwards": 0,
                 "native_finalization_pending": True,
                 "elapsed_seconds": time.perf_counter() - started})
    return z.reshape(s.shape).astype(np.float32), info


def predict(q, r, cov, score, *, device="cpu", extras=None):
    return _run(q, r, cov, score, device=device, extras=extras, mode="reference")


def control(q, r, cov, score, *, device="cpu", extras=None):
    """Identical D unary/query graph, without reference-cross-edge energy."""
    return _run(q, r, cov, score, device=device, extras=extras, mode="no_reference")


def mean_reference_control(q, r, cov, score, *, device="cpu", extras=None):
    return _run(q, r, cov, score, device=device, extras=extras, mode="global_mean_reference")


def unary_control(q, r, cov, score, *, device="cpu", extras=None):
    return _unary(score, extras), {
        "config": dict(CONFIG), "mode": "D_unary_native_CRF_replay",
        "query_gt_used": False, "native_finalization_pending": True,
    }


def rcg_control(q, r, cov, score, *, device="cpu", extras=None):
    _unary(score, extras)  # Require the same D-score provenance for every arm.
    field, info = rcg.predict(q, r, cov, score, device=device, extras=extras)
    info.update({"composition_mode": "unmodified_RCG_on_D_raw_score",
                 "score_origin": CONFIG["score_origin"],
                 "native_finalization_pending": True,
                 "composition_finalizer": CONFIG["finalizer"]})
    return field, info


@torch.inference_mode()
def finalize(field, extras):
    """Render the solved probability once, then use the existing source CRF."""
    if extras is None or "host" not in extras or "query_tensor" not in extras:
        raise ValueError("Source FoRIS host and actual transformed query RGB required")
    host = extras["host"]
    if host.mask_refiner != "crf" or host.resize_to_orig_size:
        raise ValueError("Require the unchanged source CRF at native 1024 resolution")
    query = extras["query_tensor"]
    if query.ndim == 3:
        query = query[None]
    if tuple(query.shape) != (1, 3, 1024, 1024):
        raise ValueError("Require one actual transformed 1024 query image")
    z = torch.as_tensor(field, dtype=torch.float32, device=host.device)
    if z.ndim != 2 or not torch.isfinite(z).all():
        raise ValueError("Finite probability field required")
    # Do not call _binarize_response: it would renormalize a probability field.
    binary = F.interpolate(z[None, None], (1024, 1024), mode="bilinear",
                           align_corners=False)[0, 0] > 0.5
    return host._finalize_mask(binary, query).bool().cpu().numpy()


additional_controls = {
    "mean_reference": mean_reference_control,
    "rcg": rcg_control,
    "unary": unary_control,
}
