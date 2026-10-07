"""Exact common C101--C150 protocol; legal reference mask is the only label.

No query labels, external descriptors, candidate pruning, or prototype replacement
of complete reference profiles. Candidate/source cost is deliberately not capped.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import cg
from scipy.special import expit, logsumexp


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def dot(a, b):
    """Bounded exact dot products, chunked only along rows (not prototypes)."""
    out = np.empty((len(a), len(b)), dtype=np.float64)
    for k in range(0, len(a), 256):
        out[k:k + 256] = np.einsum('id,jd->ij', a[k:k + 256], b, optimize=False)
    return out


def blocks(hw, side=4):
    yy, xx = np.indices(hw)
    return (np.minimum(side - 1, yy * side // hw[0]) * side +
            np.minimum(side - 1, xx * side // hw[1])).ravel()


def edges4(hw, valid=None):
    a = np.arange(np.prod(hw), dtype=np.int32).reshape(hw)
    i = np.concatenate((a[:-1].ravel(), a[:, :-1].ravel()))
    j = np.concatenate((a[1:].ravel(), a[:, 1:].ravel()))
    if valid is not None:
        keep = (np.asarray(valid)[i] > 0) & (np.asarray(valid)[j] > 0)
        i, j = i[keep], j[keep]
    return i, j


def graph4(x, hw, valid=None):
    i, j = edges4(hw, valid)
    w = np.maximum(np.einsum('id,id->i', x[i], x[j]), 0) ** 4
    graph = sparse.csr_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])),
                             shape=(len(x), len(x)))
    graph.eliminate_zeros()
    return graph


def mutual_graph(x, k=20, profile=False):
    if len(x) <= 1:
        return sparse.csr_matrix((len(x), len(x)))
    affinity = dot(x, x)
    if profile:
        norm = np.sum(x * x, axis=1)
        distance = np.maximum(norm[:, None] + norm[None] - 2 * affinity, 0)
        order = np.argsort(distance, axis=1, kind='stable')
    else:
        order = np.argsort(-affinity, axis=1, kind='stable')
    rows, cols = [], []
    for i in range(len(x)):
        nn = order[i][order[i] != i][:min(k, len(x) - 1)]
        rows.extend([i] * len(nn)); cols.extend(nn.tolist())
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=affinity.shape)
    adjacency = adjacency.multiply(adjacency.T)
    ii, jj = adjacency.nonzero()
    weight = (np.exp(-distance[ii, jj]) if profile else
              np.maximum(affinity[ii, jj], 0) ** 4)
    graph = sparse.csr_matrix((weight, (ii, jj)), shape=affinity.shape)
    graph.eliminate_zeros()
    return graph


def spherical(x, k, valid=None, iterations=20):
    ids = np.flatnonzero(np.ones(len(x), dtype=bool) if valid is None else np.asarray(valid) > 0)
    if not len(ids):
        return np.empty((0, x.shape[1])), np.full(len(x), -1, dtype=np.int32)
    k = min(k, len(ids))
    chosen = [int(ids[0])]
    distance = np.full(len(x), np.inf)
    for _ in range(1, k):
        distance = np.minimum(distance, 1 - dot(x, x[chosen[-1]:chosen[-1] + 1])[:, 0])
        distance[chosen] = -np.inf
        best = ids[np.argmax(distance[ids])]
        chosen.append(int(best))
    centers = x[chosen].copy()
    label = np.full(len(x), -1, dtype=np.int32)
    for _ in range(iterations):
        label[ids] = np.argmax(dot(x[ids], centers), axis=1)
        for c in range(k):
            members = ids[label[ids] == c]
            if len(members):
                centers[c] = unit(x[members].mean(axis=0))
    label[ids] = np.argmax(dot(x[ids], centers), axis=1)
    return centers, label


def split_atoms(label, hw, valid):
    i, j = edges4(hw, valid)
    keep = (label[i] == label[j]) & (label[i] >= 0)
    n = len(label)
    g = sparse.csr_matrix((np.ones(2 * keep.sum()),
                           (np.r_[i[keep], j[keep]], np.r_[j[keep], i[keep]])), shape=(n, n))
    _, cc = connected_components(g, directed=False)
    groups = [np.flatnonzero((cc == c) & (valid > 0)).astype(np.int32) for c in np.unique(cc)]
    return tuple(a for a in groups if len(a))


def candidates(x, hw, valid, u):
    """All maximum-affinity merge nodes, all max-tree nodes, all leaves."""
    n = len(x); valid_ids = np.flatnonzero(valid > 0)
    groups = []; seen = set()
    def add(a):
        a = np.sort(np.asarray(a, dtype=np.int32))
        key = a.tobytes()
        if len(a) and key not in seen:
            seen.add(key); groups.append(a)
    for i in valid_ids:
        add([i])
    ii, jj = edges4(hw, valid)
    weight = np.maximum(np.einsum('id,id->i', x[ii], x[jj]), 0) ** 4
    parent = np.arange(n); members = {int(i): np.array([i], dtype=np.int32) for i in valid_ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = int(parent[i])
        return i
    for edge in np.lexsort((jj, ii, -weight)):
        a, b = find(int(ii[edge])), find(int(jj[edge]))
        if a == b:
            continue
        if a > b: a, b = b, a
        merged = np.sort(np.r_[members.pop(a), members.pop(b)])
        parent[b] = a; members[a] = merged; add(merged)
    # Build exact upper-level-set components. Equal-value plateaus enter jointly.
    parent = np.arange(n); active = np.zeros(n, dtype=bool); members = {}
    adj = [[] for _ in range(n)]
    for a, b in zip(ii, jj):
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    order = valid_ids[np.argsort(-u[valid_ids], kind='stable')]
    pos = 0
    while pos < len(order):
        stop = pos + 1
        while stop < len(order) and u[order[stop]] == u[order[pos]]: stop += 1
        new = order[pos:stop]
        for i in new:
            active[i] = True; members[int(i)] = np.array([i], dtype=np.int32)
        for i in new:
            for j in adj[int(i)]:
                if not active[j]: continue
                a, b = find(int(i)), find(j)
                if a == b: continue
                if a > b: a, b = b, a
                merged = np.sort(np.r_[members.pop(a), members.pop(b)])
                parent[b] = a; members[a] = merged
        for a in sorted({find(int(i)) for i in new}): add(members[a])
        pos = stop
    return tuple(groups)


def gate(scores, coverage, weights, precision=.9):
    keep = weights > 0
    if not keep.any() or np.sum(coverage[keep] * weights[keep]) <= 0:
        return math.inf
    best = math.inf; recall = -1.
    total = np.sum(coverage[keep] * weights[keep])
    for t in np.unique(scores[keep]):
        selected = keep & (scores >= t)
        tp = np.sum(weights[selected] * coverage[selected])
        p = tp / max(np.sum(weights[selected]), 1e-12)
        r = tp / total
        if p >= precision and (r > recall or (r == recall and t > best)):
            best, recall = float(t), float(r)
    return best


@dataclass
class Context:
    x: np.ndarray
    hw: tuple
    valid: np.ndarray
    bank: np.ndarray
    coverage: np.ndarray
    bank_weights: np.ndarray
    bank_ids: np.ndarray
    profile: np.ndarray
    max_f: np.ndarray
    max_b: np.ndarray
    u: np.ndarray
    fg: np.ndarray
    bg: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    weak_threshold: float
    W: sparse.csr_matrix
    atoms: tuple
    regions: tuple
    broad: np.ndarray
    scale: float
    source: bool = False
    fold: int | None = None

    def base_descriptors(self):
        return np.array([[np.mean(self.u[c]), np.median(self.u[c]), np.min(self.u[c]),
                          np.max(self.u[c]), np.mean(self.max_f[c]), np.mean(self.max_b[c])]
                         for c in self.regions], dtype=float)

    def role_prototypes(self, k=16):
        return tuple(spherical(self.bank[ids], min(k, len(ids)))[0]
                     if len(ids) else np.empty((0, self.x.shape[1])) for ids in (self.fg, self.bg))


class Prepared:
    def __init__(self, ep):
        self.ep = ep; self.r = unit(ep.r); self.q = unit(ep.q)
        self.rv = np.asarray(ep.wvalid, dtype=float); self.qv = np.asarray(ep.q_valid, dtype=float)
        self.c = np.divide(ep.wf, self.rv, out=np.zeros_like(self.rv), where=self.rv > 0)
        self.rb = blocks(ep.r_hw); self.rr = dot(self.r, self.r)
        self.oof_m = np.full(len(self.r), np.nan)
        for f in range(16):
            qids = np.flatnonzero((self.rb == f) & (self.rv > 0))
            bank = np.flatnonzero((self.rb != f) & (self.rv > 0))
            fi, bi = bank[self.c[bank] >= .9], bank[self.c[bank] <= .1]
            if len(qids) and len(fi) and len(bi):
                self.oof_m[qids] = self.rr[np.ix_(qids, fi)].max(axis=1) - self.rr[np.ix_(qids, bi)].max(axis=1)
        observed = self.oof_m[np.isfinite(self.oof_m)]
        self.scale = max(float(np.subtract(*np.percentile(observed, [75, 25]))) if len(observed) else 0, .01)
        oof_u = np.clip(np.nan_to_num(self.oof_m) / self.scale, -4, 4)
        calibrated = np.isfinite(self.oof_m)
        weight = self.rv * calibrated
        self.tp = gate(oof_u, self.c, weight); self.tn = gate(-oof_u, 1-self.c, weight)
        fg_mass = self.c * weight
        self.tw = math.inf
        if fg_mass.sum()>0:
            for threshold in np.unique(oof_u[calibrated])[::-1]:
                if fg_mass[oof_u>=threshold].sum() >= .95*fg_mass.sum():
                    self.tw=float(threshold);break
        self.r_atoms = split_atoms(spherical(self.r, 64, self.rv)[1], ep.r_hw, self.rv)
        self.q_atoms = split_atoms(spherical(self.q, 64, self.qv)[1], ep.q_hw, self.qv)
        self.k0 = None
        self.fold_settings={}
        self.fold_k0={}

    def settings(self,excluded_fold=None):
        if excluded_fold is None:return self.scale,self.tp,self.tn,self.tw
        if excluded_fold in self.fold_settings:return self.fold_settings[excluded_fold]
        oof=np.full(len(self.r),np.nan)
        excluded=(excluded_fold,)if isinstance(excluded_fold,(int,np.integer))else tuple(excluded_fold)
        eligible=(self.rv>0)&~np.isin(self.rb,excluded)
        for inner in range(16):
            held=np.flatnonzero(eligible&(self.rb==inner))
            bank=np.flatnonzero(eligible&(self.rb!=inner))
            fg=bank[self.c[bank]>=.9];bg=bank[self.c[bank]<=.1]
            if len(held)and len(fg)and len(bg):
                oof[held]=self.rr[np.ix_(held,fg)].max(axis=1)-self.rr[np.ix_(held,bg)].max(axis=1)
        observed=oof[np.isfinite(oof)]
        scale=max(float(np.subtract(*np.percentile(observed,[75,25])))if len(observed)else 0,.01)
        score=np.clip(np.nan_to_num(oof)/scale,-4,4);weights=self.rv*np.isfinite(oof)
        tp=gate(score,self.c,weights);tn=gate(-score,1-self.c,weights)
        mass=self.c*weights;tw=math.inf
        if mass.sum()>0:
            for threshold in np.unique(score[np.isfinite(oof)])[::-1]:
                if mass[score>=threshold].sum()>=.95*mass.sum():tw=float(threshold);break
        result=(scale,tp,tn,tw);self.fold_settings[excluded_fold]=result
        return result

    def context(self, source=False, fold=None):
        excluded=()if fold is None else((fold,)if isinstance(fold,(int,np.integer))else tuple(fold))
        ids = np.flatnonzero((self.rv > 0) & ~np.isin(self.rb,excluded))
        x, hw, valid = ((self.r, self.ep.r_hw, self.rv) if source else
                        (self.q, self.ep.q_hw, self.qv))
        profile = self.rr[:, ids] if source else dot(x, self.r[ids])
        coverage = self.c[ids]; f = np.flatnonzero(coverage >= .9); b = np.flatnonzero(coverage <= .1)
        mf = profile[:, f].max(axis=1) if len(f) else np.full(len(x), np.nan)
        mb = profile[:, b].max(axis=1) if len(b) else np.full(len(x), np.nan)
        scale,tp,tn,tw=self.settings(fold)
        u = np.clip(np.nan_to_num(mf - mb) / scale, -4, 4)
        pos = (u >= tp) & (valid > 0) if np.isfinite(tp) else np.zeros(len(x), bool)
        neg = (-u >= tn) & (valid > 0) if np.isfinite(tn) else np.zeros(len(x), bool)
        conflict = pos & neg; pos[conflict] = False; neg[conflict] = False
        atoms = self.r_atoms if source else self.q_atoms
        regions = candidates(x, hw, valid, u)
        broad = u > 0
        for a in atoms:
            k = max(1, int(np.ceil(len(a)/4)))
            if np.sort(u[a])[-k:].mean() > tw: broad[a] = True
        return Context(x, hw, valid, self.r[ids], coverage, self.rv[ids], ids, profile, mf, mb,
                       u, f, b, pos, neg, tw, graph4(x, hw, valid), atoms, regions,
                       broad, scale, source, fold)

    def degenerate(self):
        ids = np.flatnonzero(self.rv > 0)
        if np.sum(self.ep.wf) == 0:
            return np.full(len(self.q), -1.), 'empty_reference_mask'
        if not np.any(self.c[ids] >= .9) or not np.any(self.c[ids] <= .1):
            sim = dot(self.q, self.r[ids]); vote = self.c[ids[np.argmax(sim, axis=1)]]
            return vote - .5, 'missing_pure_role_complete_reference_nearest_coverage'
        return None

    def k0_for(self,ctx):
        if ctx.fold is None:return self.k0
        if ctx.fold not in self.fold_k0:
            excluded=(ctx.fold,)if isinstance(ctx.fold,(int,np.integer))else tuple(ctx.fold)
            self.fold_k0[ctx.fold]=train_kernel(self,None,base=True,excluded_folds=excluded)
        return self.fold_k0[ctx.fold]


class Kernel:
    def __init__(self, descriptors, coverage, weights=None):
        d = np.asarray(descriptors, dtype=float)
        if d.ndim != 2: raise ValueError('fixed two-dimensional descriptor array required')
        self.n = d.shape[1]; self.active = len(d) > 0
        self.coverage = np.asarray(coverage, dtype=float)
        self.weights = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float)
        self.active &= np.sum(self.weights*self.coverage) > 0 and np.sum(self.weights*(1-self.coverage)) > 0
        self.median = np.zeros(self.n); self.iqr = np.ones(self.n)
        for j in range(self.n):
            known = d[np.isfinite(d[:, j]), j]
            if len(known):
                self.median[j] = np.median(known); self.iqr[j] = max(np.subtract(*np.percentile(known, [75, 25])), .1)
        self.d = self.standardize(d)
        # Exact median over distinct vectors. No approximate sampling/prototype cap.
        distinct = np.unique(self.d, axis=0)
        if len(distinct) <= 1: self.sigma = .1
        else:
            from scipy.spatial.distance import pdist
            self.sigma = max(float(np.median(pdist(distinct))), .1)

    def standardize(self, d):
        d = np.asarray(d, dtype=float)
        missing = ~np.isfinite(d)
        z = np.where(missing, 0, (d-self.median)/self.iqr)
        return np.concatenate((z, missing.astype(float)), axis=1)

    def __call__(self, descriptors):
        d = np.asarray(descriptors, dtype=float)
        if d.ndim == 1: d = d[None]
        if d.shape[1] != self.n: raise ValueError('descriptor dimensions differ from source')
        if not self.active: return np.zeros(len(d))
        lf, lb = self.log_densities(d)
        return np.clip(lf-lb, -4, 4)

    def log_densities(self, descriptors):
        d = np.atleast_2d(np.asarray(descriptors, dtype=float))
        if not self.active: return np.zeros(len(d)), np.zeros(len(d))
        z = self.standardize(d); out = np.empty(len(z))
        out_b = np.empty(len(z))
        fw = self.weights*self.coverage; bw = self.weights*(1-self.coverage)
        lf = np.log(fw, where=fw>0, out=np.full(len(fw), -np.inf))
        lb = np.log(bw, where=bw>0, out=np.full(len(bw), -np.inf))
        for k in range(0, len(z), 128):
            delta = z[k:k+128, None] - self.d[None]
            logk = -np.sum(delta*delta, axis=2)/(2*self.sigma**2)
            out[k:k+128] = logsumexp(logk+lf, axis=1)-np.log(fw.sum())
            out_b[k:k+128] = logsumexp(logk+lb,axis=1)-np.log(bw.sum())
        return out, out_b


def rp(ctx, values, regions=None):
    regions = ctx.regions if regions is None else regions
    total = np.zeros(len(ctx.x)); norm = np.zeros(len(ctx.x))
    for c, h in zip(regions, values):
        total[c] += np.asarray(h) / len(c); norm[c] += 1/len(c)
    return np.divide(total, norm, out=np.zeros_like(total), where=norm > 0)


def G(ctx, W=None, positive=None, negative=None, hard=False):
    W = ctx.W if W is None else sparse.csr_matrix(W)
    degree = np.asarray(W.sum(axis=1)).ravel(); nz = degree[degree > 0]
    if len(nz): W = W / np.median(nz)
    degree = np.asarray(W.sum(axis=1)).ravel()
    A = sparse.eye(len(ctx.x), format='csr') + sparse.diags(degree) - W
    pos = ctx.positive if positive is None else np.asarray(positive, bool)
    neg = ctx.negative if negative is None else np.asarray(negative, bool)
    if np.any(pos & neg): raise ValueError('conflicting anchors must be removed')
    rhs = ctx.u.copy(); rhs[pos] = 4.; rhs[neg] = -4.
    if hard:
        fixed = pos | neg; free = ~fixed; z = rhs.copy()
        if free.any():
            z[free], info = cg(A[free][:, free], rhs[free]-A[free][:, fixed]@rhs[fixed], rtol=1e-6, atol=0, maxiter=2000)
        else: info = 0
    else: z, info = cg(A, rhs, rtol=1e-6, atol=0, maxiter=2000)
    return (z if info == 0 and np.isfinite(z).all() else ctx.u.copy()), int(info)


def region_coverage(ctx, c):
    return float(np.sum(ctx.coverage[c]*ctx.bank_weights[c])/max(np.sum(ctx.bank_weights[c]),1e-12))


def train_kernel(prepared, descriptor_fn, mode='region', base=False,excluded_folds=()):
    descriptors, labels, weights = [], [], []
    for f in range(16):
        if f in excluded_folds:continue
        if not np.any((prepared.rb == f) & (prepared.rv > 0)): continue
        ctx = prepared.context(True, (f,)+tuple(excluded_folds) if excluded_folds else f)
        if not len(ctx.fg) or not len(ctx.bg): continue
        d = ctx.base_descriptors() if base else descriptor_fn(ctx, prepared)
        if mode == 'region':
            if len(d) != len(ctx.regions): raise ValueError('all candidates must return a descriptor')
            for c, row in zip(ctx.regions, d):
                if not np.isfinite(row).any(): continue
                center = c[len(c)//2]
                if prepared.rb[center] != f: continue
                label_support=c[~np.isin(prepared.rb[c],excluded_folds)]if excluded_folds else c
                if not len(label_support):continue
                descriptors.append(row)
                labels.append(float(np.sum(prepared.ep.wf[label_support])/max(np.sum(prepared.rv[label_support]),1e-12)))
                weights.append(float(np.mean(prepared.rv[label_support])))
        elif mode == 'point':
            if len(d) != len(ctx.x): raise ValueError('complete point descriptors required')
            ids = np.flatnonzero((prepared.rb == f) & (prepared.rv > 0))
            ids = ids[np.isfinite(d[ids]).any(axis=1)]
            descriptors.extend(d[ids]); labels.extend(prepared.c[ids]); weights.extend(prepared.rv[ids])
        else: raise ValueError(mode)
    if not descriptors:
        # Dimensionality is still that of the exact card's descriptor.
        probe = prepared.context(True)
        d = probe.base_descriptors() if base else descriptor_fn(probe, prepared)
        return Kernel(np.empty((0, np.asarray(d).shape[1])), np.empty(0))
    return Kernel(np.asarray(descriptors), np.asarray(labels), np.asarray(weights))


_PREPARED = None
def prepare(ep):
    global _PREPARED
    if _PREPARED is None or _PREPARED.ep is not ep:
        _PREPARED = Prepared(ep)
    if _PREPARED.k0 is None:
        _PREPARED.k0 = train_kernel(_PREPARED, None, base=True)
    return _PREPARED


def region_run(ep, descriptor_fn, inactive_fn=None):
    p = prepare(ep); degenerate = p.degenerate()
    if degenerate is not None: return degenerate[0], {'degenerate':degenerate[1]}
    k = train_kernel(p, descriptor_fn)
    ctx = p.context(); d = descriptor_fn(ctx, p); h = k(d)
    h[~np.isfinite(d).any(axis=1)] = 0
    if inactive_fn is not None: h[~np.asarray(inactive_fn(ctx,p), bool)] = 0
    z = ctx.u + rp(ctx, h); z[ctx.valid <= 0] = -4
    return z, {'kernel_active':bool(k.active), 'source_sigma':k.sigma,
               'source_sample_count':len(k.d), 'candidate_count':len(ctx.regions),
               'candidate_memberships':sum(map(len,ctx.regions)), 'source_oof_blocks':16,
               'source_folds_share_encoder_context':True, 'scale':p.scale,
               'broad_union_full_query':bool(np.all(ctx.broad[ctx.valid>0]))}


def point_run(ep, descriptor_fn, additive=False, inactive_fn=None):
    p = prepare(ep); degenerate = p.degenerate()
    if degenerate is not None: return degenerate[0], {'degenerate':degenerate[1]}
    k = train_kernel(p, descriptor_fn, mode='point'); ctx = p.context(); d = descriptor_fn(ctx,p)
    prediction = k(d)
    if additive:
        if inactive_fn is not None: prediction[~inactive_fn(ctx,p)] = 0
        z = ctx.u + prediction
    else:
        z = prediction if k.active else ctx.u.copy()
        if inactive_fn is not None: z[~inactive_fn(ctx,p)] = ctx.u[~inactive_fn(ctx,p)]
    z[~np.isfinite(d).any(axis=1)] = ctx.u[~np.isfinite(d).any(axis=1)]
    z[ctx.valid<=0] = -4
    return z, {'kernel_active':bool(k.active), 'source_sigma':k.sigma,
               'source_sample_count':len(k.d), 'source_oof_blocks':16,
               'source_folds_share_encoder_context':True, 'scale':p.scale}


def dynamic_region_run(ep,sample_fn):
    """Same source K/RP for cards whose original algorithm defines its regions."""
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return deg[0],{'degenerate':deg[1]}
    data=[];labels=[];weights=[];dimension=None
    for f in range(16):
        if not np.any((p.rb==f)&(p.rv>0)):continue
        ctx=p.context(True,f)
        if not len(ctx.fg)or not len(ctx.bg):continue
        regions,d=sample_fn(ctx,p);d=np.asarray(d,float)
        if d.ndim!=2:raise ValueError('dynamic descriptors must have fixed dimensions')
        dimension=d.shape[1]
        for c,row in zip(regions,d):
            if not len(c)or not np.isfinite(row).any()or p.rb[c[len(c)//2]]!=f:continue
            data.append(row);labels.append(ep.wf[c].sum()/max(ep.wvalid[c].sum(),1e-12));weights.append(ep.wvalid[c].mean())
    ctx=p.context();regions,d=sample_fn(ctx,p);d=np.asarray(d,float)
    dimension=d.shape[1]if dimension is None else dimension
    model=Kernel(np.asarray(data).reshape(-1,dimension),np.asarray(labels),np.asarray(weights))
    h=model(d);h[~np.isfinite(d).any(axis=1)]=0
    z=ctx.u+rp(ctx,h,regions);z[ctx.valid<=0]=-4
    return z,{'kernel_active':bool(model.active),'source_samples':len(data),'regions':len(regions),'memberships':sum(map(len,regions))}


def region_point_run(ep,sample_fn,additive=False):
    """Apply each candidate's point K before RP, preserving nonlinear order.

    This retains every source candidate/point observation. Its source bandwidth
    cost may be quadratic in total memberships; no pruning is substituted.
    """
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return deg[0],{'degenerate':deg[1]}
    data=[];labels=[];weights=[];dimension=None
    for fold in range(16):
        if not np.any((p.rb==fold)&(p.rv>0)):continue
        ctx=p.context(True,fold)
        if not len(ctx.fg)or not len(ctx.bg):continue
        regions,fields=sample_fn(ctx,p)
        for c,d in zip(regions,fields):
            d=np.asarray(d,float);dimension=d.shape[1]
            if d.shape[0]!=len(c):raise ValueError('candidate point descriptors must match all members')
            chosen=(p.rb[c]==fold)&np.isfinite(d).any(axis=1)
            ids=c[chosen];data.extend(d[chosen]);labels.extend(p.c[ids]);weights.extend(p.rv[ids])
    ctx=p.context();regions,descriptors=sample_fn(ctx,p)
    if dimension is None:
        dimension=descriptors[0].shape[1]if len(descriptors)else 1
    model=Kernel(np.asarray(data).reshape(-1,dimension),np.asarray(labels),np.asarray(weights))
    if not model.active:
        z=ctx.u.copy();z[ctx.valid<=0]=-4
        return z,{'kernel_active':False,'source_point_candidate_samples':len(data),'increment_inactive':'no two-class source kernel support','nonlinear_order':'per-candidate K, then RP'}
    fields=[]
    for c,d in zip(regions,descriptors):
        d=np.asarray(d,float)
        if model.active:field=model(d)
        else:field=np.zeros(len(c))if additive else ctx.u[c].copy()
        missing=~np.isfinite(d).any(axis=1)
        field[missing]=0 if additive else ctx.u[c][missing]
        fields.append(field)
    z=rp(ctx,fields,regions)
    covered=np.zeros(len(ctx.x),bool)
    for c in regions:covered[c]=True
    if additive:z=ctx.u+z
    else:z[~covered]=ctx.u[~covered]
    z[ctx.valid<=0]=-4
    return z,{'kernel_active':bool(model.active),'source_point_candidate_samples':len(data),
              'regions':len(regions),'memberships':sum(map(len,regions)),'nonlinear_order':'per-candidate K, then RP',
              'source_kernel_exact_distinct_bandwidth':'no membership cap or prototype approximation'}
