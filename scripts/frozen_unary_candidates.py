"""Prepared fixed C1 operator and C2 minimum scorer; not enabled candidates.

No query labels enter these functions. C1 keeps the parent's graph and solver;
C2 uses one reference role pair, one shared anchor, one whole-query match and
no null. Full C2 and C3 remain conditional on their measured prerequisites.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg

CONTEXT_TEMPERATURE = .07


def parent_circle(mask):
    x = torch.as_tensor(np.asarray(mask, dtype=np.float32))[None, None]
    return (F.interpolate(x, (64, 64), mode='area')[0, 0].numpy() >= .5)


def blend_unary(parent_y, probability, confidence):
    y = np.asarray(parent_y, dtype=np.float64)
    p, c = (np.asarray(v, dtype=np.float64) for v in (probability, confidence))
    if y.shape != p.shape or y.shape != c.shape:
        raise ValueError('Unary, probability and confidence shapes differ')
    if not all(np.isfinite(v).all() for v in (y, p, c)):
        raise ValueError('Nonfinite unary inputs')
    if c.min() < 0 or c.max() > .5 or p.min() < -1e-6 or p.max() > 1+1e-6:
        raise ValueError('Fixed probability/confidence range violated')
    if not np.any(c):
        return y.copy()
    return (1-c)*y+c*p


def c1_unary(parent_y, margin, final_mask, reference_valid=True):
    y = np.asarray(parent_y, dtype=np.float64)
    d = np.asarray(margin, dtype=np.float32).reshape(y.shape)
    if y.size != 4096 or not np.isfinite(d).all() or np.abs(d).max() > 2+1e-6:
        raise ValueError('Require a finite unit-feature margin on the 64 grid')
    probability = .5+d/4
    confidence = parent_circle(final_mask).reshape(y.shape)*np.minimum(.5, np.abs(d)/2)
    if not reference_valid:
        confidence.fill(0)
    return blend_unary(y, probability, confidence), probability, confidence


def solve_parent_system(matrix, diagonal_a, unary):
    """Use the inherited (A+16L), confidence and exact locked CG settings."""
    y, a = (np.asarray(x, dtype=np.float64).ravel() for x in (unary, diagonal_a))
    if matrix.shape != (len(y), len(y)) or a.shape != y.shape:
        raise ValueError('Parent graph/confidence geometry differs')
    z, status = cg(matrix, a*y, x0=y, rtol=1e-7, atol=1e-9, maxiter=300)
    if status:
        raise RuntimeError('Inherited CG did not converge: '+str(status))
    return z.astype(np.float32).reshape(np.asarray(unary).shape)


def reference_walk(reference_app, grid=(64, 64)):
    """B0 distance kernel on valid spatial 8-neighbors, row-normalized."""
    r = F.normalize(torch.as_tensor(reference_app).float().cpu(), dim=-1)
    h, w = grid
    if len(r) != h*w or h*w < 3 or not torch.isfinite(r).all():
        raise ValueError('Invalid reference grid/features')
    y, x = np.indices(grid)
    source, dest = [], []
    for dy, dx in [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]:
        valid = (y+dy >= 0) & (y+dy < h) & (x+dx >= 0) & (x+dx < w)
        source.extend((y[valid]*w+x[valid]).tolist())
        dest.extend(((y[valid]+dy)*w+x[valid]+dx).tolist())
    source, dest = np.asarray(source), np.asarray(dest)
    distance = (1-(r[source]*r[dest]).sum(-1)).clamp_min(0).numpy()
    scale = np.zeros(h*w, dtype=np.float32)
    np.maximum.at(scale, source, distance)
    weight = np.exp(-distance/np.maximum(scale[source], 1e-6))
    graph = sparse.csr_matrix((weight, (source, dest)), shape=(h*w, h*w))
    total = np.asarray(graph.sum(1)).ravel()
    return sparse.diags(1/total)@graph


def first_context_roles(reference_app, foreground, hard_background, coverage, grid=(64, 64)):
    """First FG representative, mutual-walk BG/t and first shared anchor."""
    fg, bg = (np.sort(np.unique(np.asarray(v, dtype=int))) for v in (foreground, hard_background))
    if not len(fg) or not len(bg):
        return None
    if np.intersect1d(fg, bg).size:
        raise ValueError('Reference FG/BG memberships overlap')
    cov = np.asarray(coverage).ravel()
    if len(cov) != len(reference_app) or not np.isfinite(cov).all():
        raise ValueError('Reference coverage differs')
    f = int(fg[np.argmax(cov[fg])])  # Sorted rows resolve maximum-coverage ties.
    p = reference_walk(reference_app, grid)
    power = p@p
    best = None
    for t in (2, 4, 8):
        if t != 2:
            power = power@power
        forward, backward = power[f, bg].toarray().ravel(), power[bg, f].toarray().ravel()
        reach = np.sqrt(np.maximum(forward*backward, 0))
        b = int(bg[np.argmax(reach)])
        value = float(reach.max())
        if best is None or value > best[0] or value == best[0] and b < best[1]:
            best = (value, b, t, power.copy())
    value, b, t, power = best
    shared = np.minimum(power.getrow(f).toarray().ravel(), power.getrow(b).toarray().ravel())
    shared[[f, b]] = -1
    anchor = int(np.argmax(shared))
    return dict(foreground=f, background=b, anchor=anchor, walk_steps=t, mutual_reach=value)


def relations(branches, anchor, num_heads=16):
    """Bidirectional Q/K head cosines at fixed layers16/24, before RoPE."""
    values = []
    for layer in (16, 24):
        q, k = (torch.as_tensor(branches[f'{kind}/{layer}']).float() for kind in ('Q', 'K'))
        if q.ndim != 2 or q.shape != k.shape or q.shape[-1] % num_heads:
            raise ValueError('Q/K head geometry differs')
        q = F.normalize(q.reshape(len(q), num_heads, -1), dim=-1)
        k = F.normalize(k.reshape(len(k), num_heads, -1), dim=-1)
        values.extend([(q*k[anchor]).sum(-1), (q[anchor]*k).sum(-1)])
    return torch.cat(values, dim=-1)


def minimal_context(reference_app, query_app, reference_qk, query_qk, roles, num_heads=16):
    """Scores/controls only; uses the same matched anchor for FG and BG."""
    r, q = (F.normalize(torch.as_tensor(v).float(), dim=-1) for v in (reference_app, query_app))
    f, b, anchor = (int(roles[k]) for k in ('foreground', 'background', 'anchor'))
    if len({f, b, anchor}) != 3 or r.ndim != 2 or q.shape != r.shape:
        raise ValueError('Invalid role pair/shared anchor')
    matched = int((q@r[anchor]).argmax())  # Lowest row wins ties; no self shortcut.
    wrong = (matched+len(q)//2)%len(q)     # +2048 on the actual4096 grid.
    ref_relation = relations(reference_qk, anchor, num_heads)
    query_relation = relations(query_qk, matched, num_heads)
    wrong_relation = relations(query_qk, wrong, num_heads)
    app_f = (1-(q@r[f]).clamp(-1, 1))/2
    app_b = (1-(q@r[b]).clamp(-1, 1))/2

    def energy(relation):
        ef = .5*app_f+.5*(relation-ref_relation[f]).square().mean(-1)/4
        eb = .5*app_b+.5*(relation-ref_relation[b]).square().mean(-1)/4
        return ef, eb
    ef, eb = energy(query_relation)
    wf, wb = energy(wrong_relation)
    p = torch.sigmoid((eb-ef)/CONTEXT_TEMPERATURE)
    direct = torch.sigmoid(.5*(app_b-app_f)/CONTEXT_TEMPERATURE)
    mismatch = torch.sigmoid((wb-wf)/CONTEXT_TEMPERATURE)
    # Deleting a root-as-anchor correspondence leaves no legal hypothesis.
    valid = torch.ones(len(q), dtype=torch.bool, device=q.device);valid[matched]=False
    for score in (p, direct, mismatch):
        score[~valid] = .5
    return dict(probability=p, direct_probability=direct, wrong_probability=mismatch,
                energy_fg=ef, energy_bg=eb, query_anchor=matched, wrong_anchor=wrong, valid=valid)
