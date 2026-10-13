"""Reference-labelled shared semantic hierarchy; frozen features only.

The hierarchy is unsupervised in R/Q feature space. Reference labels determine
a stop/split coding mixture, rather than labelling one fixed query partition.
The coding machinery is classical model averaging, not an originality claim.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betaln, expit
import torch
import torch.nn.functional as F

DEPTH = 7
LLOYD_STEPS = 10
PRIOR_MASS = .5
TAU = .07


@torch.inference_mode()
def shared_tree(reference, query, depth=DEPTH, steps=LLOYD_STEPS):
    """Complete binary bisecting spherical k-means, with deterministic ties.

    Both images contribute all their native unit patches. Every level splits
    within the preceding parent, with the farthest-from-mean point followed by
    its farthest peer as initialization. Empty children remain empty. No mask,
    image coordinates, query labels or previously predicted masks enter here.
    """
    x = torch.cat((reference, query), dim=0)
    assert x.ndim == 2 and torch.isfinite(x).all()
    parent = torch.zeros(len(x), dtype=torch.long, device=x.device)
    paths = [parent.cpu().numpy()]
    records = []
    row = torch.arange(len(x), device=x.device)
    for level in range(depth):
        n = 2**level
        membership = F.one_hot(parent, n).to(x.dtype)
        means = F.normalize(membership.T @ x, dim=-1)
        belongs = membership.bool()
        distance = x @ means.T
        first = torch.where(belongs, distance, torch.inf).argmin(dim=0)
        seeds = x[first]
        distance = x @ seeds.T
        second = torch.where(belongs, distance, torch.inf).argmin(dim=0)
        centers = torch.stack((seeds, x[second]), dim=1).reshape(2*n, -1)
        child = 2*parent
        for _ in range(steps):
            cosine = x @ centers.T
            left = cosine[row, 2*parent]
            right = cosine[row, 2*parent+1]
            child = 2*parent+(right > left).long()
            membership = F.one_hot(child, 2*n).to(x.dtype)
            sums = membership.T @ x
            updated = F.normalize(sums, dim=-1)
            # An empty child cannot acquire points from a different parent.
            # Keeping its previous center lets a later Lloyd step revive it.
            centers = torch.where((membership.sum(0)>0)[:, None], updated, centers)
        parent = child
        path = parent.cpu().numpy()
        paths.append(path)
        records.append({"level": level+1, "nonempty_nodes": int(np.unique(path).size)})
    paths = np.stack(paths).astype(np.int16)
    assert np.array_equal(paths[1:]//2, paths[:-1])
    return paths, records


def label_tree(paths, coverage, n_reference):
    """Fractional-mask Beta coding mixture on a fixed shared hierarchy.

    The prior mean is reference foreground area. Empty-reference nodes back
    off to the parent's estimate. The Beta score treats patch masses as coding
    counts; correlated patches do not justify calibrated statistical certainty.
    """
    c = np.asarray(coverage, dtype=np.float64).reshape(-1)
    assert len(c) == n_reference and np.isfinite(c).all()
    assert c.min() >= 0 and c.max() <= 1
    pi = float(c.mean())
    depth = len(paths)-1
    if pi <= 0 or pi >= 1:
        constant = np.full(paths.shape[1]-n_reference, pi-.5, np.float32)
        return {"adaptive": constant, "flat": constant.copy()}, {
            "fallback": "missing_reference_role", "pi": pi,
            "effective_query_depth": 0., "source_entropy": 0.}, {}
    a, b = PRIOR_MASS*pi, PRIOR_MASS*(1-pi)
    fg, bg, probability, local, evidence, stop = [], [], [], [], [], []
    for level, assignment in enumerate(paths):
        n = 2**level
        f = np.bincount(assignment[:n_reference], weights=c, minlength=n)
        g = np.bincount(assignment[:n_reference], weights=1-c, minlength=n)
        p = (f+a)/(f+g+PRIOR_MASS)
        if level:
            empty = f+g == 0
            p[empty] = probability[-1][np.arange(n)[empty]//2]
        fg.append(f); bg.append(g); probability.append(p)
        local.append(betaln(f+a, g+b)-betaln(a, b))
    evidence = [None]*(depth+1)
    stop = [None]*(depth+1)
    evidence[-1] = local[-1]
    stop[-1] = np.ones_like(local[-1])
    for level in range(depth-1, -1, -1):
        split = evidence[level+1].reshape(-1, 2).sum(1)
        evidence[level] = np.logaddexp(local[level], split)-np.log(2.)
        stop[level] = np.exp(local[level]-np.log(2.)-evidence[level])
        assert np.isfinite(stop[level]).all() and stop[level].min() >= 0
        assert stop[level].max() <= 1+1e-12
    query_path = paths[:, n_reference:]
    mixed = probability[-1][query_path[-1]].copy()
    effective_depth = np.full(len(mixed), depth, dtype=np.float64)
    for level in range(depth-1, -1, -1):
        w = stop[level][query_path[level]]
        mixed = w*probability[level][query_path[level]]+(1-w)*mixed
        effective_depth = w*level+(1-w)*effective_depth
    def balanced(p):
        denominator = p*(1-pi)+(1-p)*pi
        return (p*(1-pi)/denominator-.5).astype(np.float32)
    fields = {"adaptive": balanced(mixed),
              "flat": balanced(probability[-1][query_path[-1]])}
    arrays = {}
    for level in range(depth+1):
        for name, values in (("fg", fg[level]), ("bg", bg[level]),
                             ("p", probability[level]), ("stop", stop[level]),
                             ("local", local[level]), ("evidence", evidence[level])):
            arrays[f"tree/{level}/{name}"] = values
    info = {"fallback": None, "pi": pi,
            "effective_query_depth": float(effective_depth.mean()),
            "root_stop_weight": float(stop[0][0]),
            "empty_reference_query_leaf_fraction": float(
                np.mean((fg[-1]+bg[-1])[query_path[-1]] == 0)),
            "source_entropy": float(-pi*np.log(pi)-(1-pi)*np.log1p(-pi))}
    return fields, info, arrays


@torch.inference_mode()
def role_density(reference, query, coverage, chunk=256):
    mass = torch.as_tensor(coverage, dtype=query.dtype, device=query.device).reshape(-1)
    total = float(mass.sum())
    if total <= 0 or total >= len(mass):
        return np.full(len(query), -.5 if total <= 0 else .5, np.float32)
    logf = torch.where(mass > 0, mass.log()-np.log(total), -torch.inf)
    logb = torch.where(mass < 1, (1-mass).log()-np.log(len(mass)-total), -torch.inf)
    values = []
    for left in range(0, len(query), chunk):
        cosine = query[left:left+chunk] @ reference.T/TAU
        margin = torch.logsumexp(cosine+logf, -1)-torch.logsumexp(cosine+logb, -1)
        values.append((margin.sigmoid()-.5).cpu().numpy())
    return np.concatenate(values)


def check_fixtures():
    # Two shared semantic groups, each present in both images. No network.
    paths = np.array([[0, 0, 0, 0], [0, 1, 0, 1]], dtype=np.int16)
    fields, _, _ = label_tree(paths, [1., 0.], 2)
    opposite, _, _ = label_tree(paths, [0., 1.], 2)
    constant, _, _ = label_tree(paths, [.5, .5], 2)
    assert fields["adaptive"][0] > 0 > fields["adaptive"][1]
    assert np.max(np.abs(fields["adaptive"]+opposite["adaptive"])) < 1e-7
    assert not np.any(constant["adaptive"])
    # An unsupported query child inherits its parent, without a new FG prior.
    empty = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int16)
    unknown, _, _ = label_tree(empty, [.8, .2], 2)
    assert abs(float(unknown["adaptive"][0])) < 1e-7
    return {"shared_roles": True, "complement_symmetry": True,
            "constant_mask_neutral": True, "unseen_child_backoff": True}
