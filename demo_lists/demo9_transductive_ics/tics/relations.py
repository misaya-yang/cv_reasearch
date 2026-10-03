"""Relation maps between a query, a reference and the FoRIS score. Nothing in them names a class: every channel is
a similarity, a rank or a difference between neighbours, so a read-out fitted on them can be moved to unseen classes."""
import torch
import torch.nn.functional as F

NAMES = ("sn", "s2", "s3", "rank", "fg", "bg", "nn_label", "cycle", "grp_core", "grp_out", "grp_core5", "grp_out5",
         "proto_core", "proto_out", "edge_r", "edge_d")


def unit_range(x):
    x = x - x.min()
    return x / x.max().clamp_min(1e-6)


def near_mask(n, device, radius=2):
    """Patches within `radius` of each other on the grid: a patch never votes for itself or its close neighbours."""
    side = int(round(n ** 0.5))
    y, x = torch.arange(n, device=device) // side, torch.arange(n, device=device) % side
    return ((y[:, None] - y[None]).abs() <= radius) & ((x[:, None] - x[None]).abs() <= radius)


def seeds(sn, least=16):
    """What FoRIS is sure of: the top and the bottom of its normalised score."""
    order = sn.argsort()
    core, out = sn >= 0.8, sn <= 0.2
    if int(core.sum()) < least:
        core = torch.zeros_like(core).index_fill(0, order[-least:], True)
    if int(out.sum()) < least:
        out = torch.zeros_like(out).index_fill(0, order[:least], True)
    return core, out


def relation_maps(q, r, cov, score, s2, s3, near):
    """q, r: [N, C] unit features of query and reference; cov: reference mask coverage per patch; score, s2, s3:
    FoRIS's final and stage scores on the query grid. Returns [len(NAMES), N]."""
    n = q.shape[0]
    side = int(round(n ** 0.5))
    pos = torch.arange(n, device=q.device)
    sn = unit_range(score.flatten().float())
    m = dict(sn=sn, s2=unit_range(s2.flatten().float()), s3=unit_range(s3.flatten().float()))
    m["rank"] = sn.argsort().argsort().float() / (n - 1)
    cov = cov.flatten().float()
    fgm = cov >= 0.5
    sim = q @ r.T
    zero = torch.zeros(n, device=q.device)
    m["fg"] = sim[:, fgm].amax(1) if fgm.any() else zero
    m["bg"] = sim[:, ~fgm].amax(1) if (~fgm).any() else zero
    fwd, back = sim.argmax(1), sim.argmax(0)
    m["nn_label"] = cov[fwd]
    there = back[fwd]
    far = torch.hypot((there // side - pos // side).float(), (there % side - pos % side).float())
    m["cycle"] = m["nn_label"] * (far <= 2).float()
    core, out = seeds(sn)
    G = (q @ q.T).masked_fill(near, -1.0)
    top = lambda cols, k: G[:, cols].topk(min(k, int(cols.sum())), dim=1).values.mean(1)
    m["grp_core"], m["grp_out"], m["grp_core5"], m["grp_out5"] = top(core, 1), top(out, 1), top(core, 5), top(out, 5)
    m["proto_core"] = q @ F.normalize(q[core].mean(0), dim=0)
    m["proto_out"] = q @ F.normalize(q[out].mean(0), dim=0)
    g = q.view(side, side, -1)
    m["edge_r"] = F.pad(1 - (g[:, :-1] * g[:, 1:]).sum(-1), (0, 1)).flatten()
    m["edge_d"] = F.pad(1 - (g[:-1] * g[1:]).sum(-1), (0, 0, 0, 1)).flatten()
    return torch.stack([m[k] for k in NAMES])
