"""Simple complete score: reference FG mean guidance and reciprocal query graph.

This is an intentionally simple candidate/control family, not a novelty claim or
a reproduction claim for the screenshot's score. It reuses the established RCG
graph and fidelity equations, replacing complex reference matching by one mean.
The FoRIS continuous score is an ordinary input cue; no FoRIS hard mask constrains
the output. Final rendering can both add and delete relative to raw DINO matching.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg
from .rcg import minmax, rank

CONFIG = dict(name="simple_reference_mean_graph", alpha=.25, graph_lambda=16.0,
              query_k=20, reference_purity=.9, fidelity_floor=.1,
              source_mean="unit(mean(unit(reference pure-FG tokens)))",
              cue="FoRIS continuous response, minmax once", finalizer="bilinear1024 then >0.5; no re-minmax or CRF",
              parameters="fixed existing RCG/mean settings before this DEV241 comparison", extra_forwards=0,
              novelty="simple candidate and ablations, not an independent novelty claim")


@torch.inference_mode()
def _run(q, r, cov, score, *, device, alpha, lam):
    q = torch.as_tensor(q, device=device, dtype=torch.float32)
    r = torch.as_tensor(r, device=device, dtype=torch.float32)
    cov, score = np.asarray(cov), np.asarray(score)
    if q.shape != (4096, 1024) or r.shape != q.shape or cov.shape != (64, 64) or score.shape != cov.shape:
        raise ValueError("Require the aligned DINO cache and 64x64 fields")
    if not bool(torch.isfinite(q).all() and torch.isfinite(r).all()) or not np.isfinite(cov).all() or not np.isfinite(score).all():
        raise ValueError("Nonfinite input")
    if cov.min() < 0 or cov.max() > 1 or cov.max() == 0 or bool((q.norm(dim=1) == 0).any() or (r.norm(dim=1) == 0).any()):
        raise ValueError("Invalid reference coverage or zero-norm tokens")
    q, r = F.normalize(q, dim=1), F.normalize(r, dim=1)
    fi = np.flatnonzero(cov.ravel() >= CONFIG["reference_purity"])
    if not len(fi): fi = np.flatnonzero(cov.ravel() == cov.max())
    guide = (q @ F.normalize(r[fi].mean(0), dim=0)).cpu().numpy()
    s = minmax(score).ravel()
    y = (s + alpha * (rank(guide) - rank(s))).astype(np.float64)
    info = dict(config=CONFIG, alpha=alpha, graph_lambda=lam, reference_tokens=len(fi), query_gt_used=False,
                cue_identity="FoRIS continuous score; edit origin is defined separately", finalizer=CONFIG["finalizer"])
    if lam == 0:
        return y.reshape(64, 64).astype(np.float32), info
    sim = q @ q.T; sim.fill_diagonal_(-2)
    values, indices = sim.topk(CONFIG["query_k"], dim=1)
    dist = (1 - values).clamp_min(0)
    weights = torch.exp(-dist / dist[:, -1:].clamp_min(1e-6)).cpu().numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), CONFIG["query_k"]), indices.cpu().numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T); w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel(); w = w / max(float(degree.mean()), 1e-8)
    degree = np.asarray(w.sum(1)).ravel()
    fidelity = CONFIG["fidelity_floor"] + np.abs(2*s - 1); fidelity = (fidelity / fidelity.mean()).astype(np.float64)
    matrix = sparse.diags(fidelity) + lam * (sparse.diags(degree) - w)
    rhs = fidelity*y
    z, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300)
    if status: raise RuntimeError(f"Graph solve failed: {status}")
    info.update(graph_undirected_edges=int(w.nnz//2), relative_residual=float(np.linalg.norm(matrix@z-rhs)/max(np.linalg.norm(rhs),1e-12)))
    return z.reshape(64, 64).astype(np.float32), info


def predict(q, r, cov, score, *, device="cpu", extras=None):
    return _run(q, r, cov, score, device=device, alpha=CONFIG["alpha"], lam=CONFIG["graph_lambda"])


def control(q, r, cov, score, *, device="cpu", extras=None):
    """Same graph and fidelity, remove the reference-mean correction."""
    return _run(q, r, cov, score, device=device, alpha=0., lam=CONFIG["graph_lambda"])


def unary_control(q, r, cov, score, *, device="cpu", extras=None):
    """Same reference mean and score, remove graph propagation."""
    return _run(q, r, cov, score, device=device, alpha=CONFIG["alpha"], lam=0.)


additional_controls = {"unary": unary_control}
