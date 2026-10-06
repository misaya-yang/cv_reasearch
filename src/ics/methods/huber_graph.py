"""CPU candidate: Huber graph fidelity with a computable primal-dual gap.

Uses the frozen reference-guided pre-graph target and the same cached query graph.
Replaces the quadratic edge energy, preserving its curvature at small contrasts.
Empirical segmentation benefit and novelty are unverified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np


@dataclass(frozen=True)
class Config:
    graph_lambda: float = 16.0
    transition: float = 0.05
    gap_per_token: float = 1e-8
    maximum_iterations: int = 20000
    check_every: int = 25


def make_mean_inputs(q, r, coverage, score):
    """Reconstruct the existing locked MEAN pre-graph inputs on CPU, without GT."""
    started=time.perf_counter()
    import torch
    import torch.nn.functional as F
    from scipy import sparse
    from .rcg import minmax, rank
    q, r = torch.as_tensor(q,dtype=torch.float32,device='cpu'), torch.as_tensor(r,dtype=torch.float32,device='cpu')
    cov, score = np.asarray(coverage), np.asarray(score)
    if (q.shape!=(4096,1024) or r.shape!=q.shape or cov.shape!=(64,64) or score.shape!=cov.shape
            or not bool(torch.isfinite(q).all() and torch.isfinite(r).all())
            or not np.isfinite(cov).all() or not np.isfinite(score).all()
            or cov.min()<0 or cov.max()>1 or cov.max()<=0
            or bool((q.norm(dim=1)==0).any() or (r.norm(dim=1)==0).any())):
        raise ValueError('Require the aligned frozen 64x64x1024 cache and valid reference mask')
    if float(np.ptp(score))<1e-9:
        empty=np.array([],dtype=np.int64)
        return (np.zeros_like(score,dtype=np.float64),np.ones_like(score,dtype=np.float64),
                empty,empty,np.array([],dtype=np.float64)),dict(source_constant_empty_rule=True)
    with torch.inference_mode():
        q,r=F.normalize(q,dim=1),F.normalize(r,dim=1)
        fi=np.flatnonzero(cov.ravel()>=.9)
        if not len(fi):
            fi=np.flatnonzero(cov.ravel()==cov.max())
        guide=(q@F.normalize(r[fi].mean(0),dim=0)).numpy()
        s=minmax(score).ravel()
        y=(s+.25*(rank(guide)-rank(s))).astype(np.float64)
        sim=q@q.T;sim.fill_diagonal_(-2)
        values,indices=sim.topk(20,dim=1)
        dist=(1-values).clamp_min(0)
        weight=torch.exp(-dist/dist[:,-1:].clamp_min(1e-6)).numpy().ravel()
        graph=sparse.csr_matrix((weight,(np.repeat(np.arange(4096),20),indices.numpy().ravel())),shape=(4096,4096))
        graph=graph.multiply(graph.T);graph.data=np.sqrt(graph.data)
        degree=np.asarray(graph.sum(1)).ravel()
        graph=graph/max(float(degree.mean()),1e-8)
        upper=sparse.triu(graph,k=1).tocoo()
        a=.1+np.abs(2*s-1);a=(a/a.mean()).astype(np.float64)
    inputs=(y.reshape(64,64),a.reshape(64,64),upper.row.astype(np.int64),
            upper.col.astype(np.int64),upper.data.astype(np.float64))
    return inputs,dict(source='locked_MEAN_a0.25_l16_pregraph_reconstruction_cpu',
                       edges=int(upper.nnz),query_gt_used=False,new_encoder_forwards=0,
                       pregraph_build_seconds=time.perf_counter()-started)


def prepare(target, fidelity, left, right, weights):
    target = np.asarray(target, dtype=np.float64)
    fidelity = np.asarray(fidelity, dtype=np.float64)
    raw_left, raw_right = np.asarray(left), np.asarray(right)
    if raw_left.dtype.kind not in 'iu' or raw_right.dtype.kind not in 'iu':
        raise ValueError('Integer edge endpoints required')
    left, right = raw_left.astype(np.int64), raw_right.astype(np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if (target.ndim != 2 or target.size == 0 or fidelity.shape != target.shape
            or not np.isfinite(target).all() or not np.isfinite(fidelity).all()
            or np.any(fidelity <= 0) or left.ndim != 1 or right.shape != left.shape
            or weights.shape != left.shape or not np.isfinite(weights).all()
            or np.any(weights <= 0) or np.any(left < 0) or np.any(right >= target.size)
            or np.any(left >= right)):
        raise ValueError('Invalid target/fidelity/canonical positive-weight graph')
    if len(np.unique(left * target.size + right)) != len(left):
        raise ValueError('Duplicate graph edges')
    return target, fidelity, left, right, weights


def divergence(p, left, right, n):
    return np.bincount(left, weights=p, minlength=n) - np.bincount(right, weights=p, minlength=n)


def energy(z, target, fidelity, left, right, costs, transition):
    difference = np.abs(z[left] - z[right])
    edge = np.where(difference <= transition, .5 * difference**2,
                    transition * (difference - .5 * transition))
    return float(.5 * np.sum(fidelity * (z - target)**2) + np.dot(costs, edge))


def certificate(z, p, target, fidelity, left, right, costs, transition):
    div = divergence(p, left, right, len(z))
    dual_minimizer = np.clip(target - div / fidelity, 0, 1)
    dual = (.5 * np.sum(fidelity * (dual_minimizer - target)**2)
            + np.dot(div, dual_minimizer) - .5 * np.sum(p**2 / costs))
    primal = energy(z, target, fidelity, left, right, costs, transition)
    gap = float(primal - dual)
    if gap < -1e-8 * max(1.0, abs(primal), abs(dual)):
        raise RuntimeError('Invalid negative primal-dual gap')
    return primal, float(dual), max(0.0, gap)


def predict(target, fidelity, left, right, weights, cfg=Config()):
    started = time.perf_counter()
    if (cfg.graph_lambda < 0 or cfg.transition <= 0 or cfg.gap_per_token <= 0
            or cfg.maximum_iterations < 1 or cfg.check_every < 1):
        raise ValueError('Invalid optimization configuration')
    target, fidelity, left, right, weights = prepare(target, fidelity, left, right, weights)
    shape = target.shape
    y, a = target.ravel(), fidelity.ravel()
    n = len(y)
    z = np.clip(y, 0, 1)
    if len(left) == 0 or cfg.graph_lambda == 0:
        return z.reshape(shape), dict(config=asdict(cfg), iterations=0, gap=0.0,
                                      query_gt_used=False, new_encoder_forwards=0,
                                      maximum_field_error_bound=0.0,
                                      wall_seconds=time.perf_counter() - started)
    costs = cfg.graph_lambda * weights
    cap = costs * cfg.transition
    degree = np.bincount(np.r_[left, right], minlength=n)
    # ||incidence||^2 <= 2*maximum unweighted degree; tau*sigma*||K||^2<1.
    step = .99 / np.sqrt(2.0 * degree.max())
    p = np.zeros(len(left), dtype=np.float64)
    extrapolated = z.copy()
    gap = float('inf')
    for iteration in range(1, cfg.maximum_iterations + 1):
        p = np.clip((p + step * (extrapolated[left] - extrapolated[right]))
                    / (1 + step / costs), -cap, cap)
        div = divergence(p, left, right, n)
        previous = z
        z = np.clip((z - step * div + step * a * y) / (1 + step * a), 0, 1)
        extrapolated = 2 * z - previous
        if iteration % cfg.check_every == 0 or iteration == cfg.maximum_iterations:
            primal, dual, gap = certificate(z, p, y, a, left, right, costs, cfg.transition)
            if gap / n <= cfg.gap_per_token:
                break
    if gap / n > cfg.gap_per_token or not np.isfinite(z).all():
        raise RuntimeError(f'Huber solver not certified: iterations={iteration}, gap/token={gap/n:.6g}')
    # Strong convexity implies a_i*(z_i-z*_i)^2/2 <= total primal-dual gap.
    coordinate_bound = np.sqrt(2 * gap / a)
    info = dict(config=asdict(cfg), iterations=iteration, edges=len(left), step=float(step),
                primal=primal, dual=dual, gap=gap, gap_per_token=gap/n,
                maximum_field_error_bound=float(coordinate_bound.max()),
                certified_token_threshold_fraction=float((np.abs(z-.5) > coordinate_bound).mean()),
                edge_force_cap_max=float(cap.max()), isolated_tokens=int((degree == 0).sum()),
                query_gt_used=False, new_encoder_forwards=0,
                wall_seconds=time.perf_counter() - started,
                segmentation_benefit='unmeasured', complete_dataset_minutes='unmeasured')
    return z.reshape(shape), info


def quadratic_control(target, fidelity, left, right, weights, graph_lambda=16.0):
    """Same boxed fidelity and graph; bounded quadratic comparison, not another method."""
    from scipy import sparse
    from scipy.optimize import minimize
    target, fidelity, left, right, weights = prepare(target, fidelity, left, right, weights)
    shape = target.shape
    y, a = target.ravel(), fidelity.ravel()
    costs = graph_lambda * weights
    n = len(y)
    if graph_lambda < 0:
        raise ValueError('Nonnegative graph coefficient required')
    degree = np.bincount(np.r_[left, right], weights=np.r_[costs, costs], minlength=n)
    matrix = sparse.coo_matrix((np.r_[degree+a, -costs, -costs],
                               (np.r_[np.arange(n), left, right], np.r_[np.arange(n), right, left])),
                              shape=(n,n)).tocsr()
    def objective(z):
        delta = z - y
        edge = z[left] - z[right]
        value = .5 * np.dot(a * delta, delta) + .5 * np.dot(costs, edge**2)
        return value, matrix @ z - a*y
    result = minimize(objective, np.clip(y,0,1), method='L-BFGS-B', jac=True,
                      bounds=[(0,1)]*n, options=dict(gtol=1e-10, ftol=1e-14, maxiter=20000))
    if not result.success:
        raise RuntimeError(f'Quadratic control failed: {result.message}')
    return result.x.reshape(shape), dict(objective=float(result.fun), iterations=int(result.nit),
                                        contract='boxed quadratic, same target/fidelity/edges/weights')
