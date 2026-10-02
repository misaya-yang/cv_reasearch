"""Transductive propagation over an ImageSet: unlabeled images of the concept become references with predicted masks.

Vocabulary
  P            dict image -> current mask; P[0] is the true mask of the labelled reference
  X[y][x]      cross prediction: mask of image x when (y, P[y]) is the only reference
  reliability  one number per unlabeled image y, how far (y, P[y]) can be trusted as a reference
     roundtrip   IoU(X[y][0], true reference mask): can y re-segment the labelled reference?  (gold question)
     agree       mean over other unlabeled x of IoU(X[y][x], P[x]): does y reproduce what the pool believes?
     spectral    principal eigenvector of the symmetrised agreement matrix (rank-1 "annotator quality" model)
     gold+agree  geometric mean of roundtrip and agree
  choose       which images act as references: all | tophalf | bottomhalf | randhalf | thr:<t> | top:<m>
"""
import numpy as np, torch
from .imageset import ImageSet


def one_shot(s, J):
    g0 = s.gt64[0]; P = {0: g0}; P.update({j: s.predict(j, [0], [g0]) for j in J}); return P


def cross_predictions(s, P, sources, targets):
    return {y: {x: s.predict(x, [y], [P[y]]) for x in targets if x != y} for y in sources if P[y].any()}


def reliability(s, P, J, kind, X=None):
    if kind == "none": return {y: 1.0 for y in J}
    need_rt = kind in ("roundtrip", "gold+agree"); need_pool = kind in ("agree", "spectral", "gold+agree")
    if X is None: X = cross_predictions(s, P, J, ([0] if need_rt else []) + (J if need_pool else []))
    rt = {y: (s.iou(X[y][0], P[0]) if y in X else 0.0) for y in J} if need_rt else None
    if kind == "roundtrip": return rt
    A = np.zeros((len(J), len(J)))
    for a, y in enumerate(J):
        for b, x in enumerate(J):
            if x != y and y in X: A[a, b] = s.iou(X[y][x], P[x])
    ag = {y: float(A[a].sum() / max(len(J) - 1, 1)) for a, y in enumerate(J)}
    if kind == "agree": return ag
    if kind == "gold+agree": return {y: float(np.sqrt(ag[y] * rt[y])) for y in J}
    if kind == "spectral":
        S = (A + A.T) / 2; v = np.abs(np.linalg.eigh(S)[1][:, -1]) if len(J) > 1 else np.ones(1)
        return {y: float(v[a] / max(v.max(), 1e-9)) for a, y in enumerate(J)}
    raise ValueError(kind)


def choose(r, rule, rng=None):
    ys = list(r); vals = np.array([r[y] for y in ys])
    if rule == "all": keep = np.ones(len(ys), bool)
    elif rule == "tophalf": keep = vals >= np.median(vals)
    elif rule == "bottomhalf": keep = vals <= np.median(vals)
    elif rule == "randhalf": keep = np.zeros(len(ys), bool); keep[rng.permutation(len(ys))[: (len(ys) + 1) // 2]] = True
    elif rule.startswith("thr:"): keep = vals > float(rule[4:])
    elif rule.startswith("top:"): keep = np.zeros(len(ys), bool); keep[np.argsort(-vals)[: int(rule[4:])]] = True
    else: raise ValueError(rule)
    return {y: bool(k) for y, k in zip(ys, keep)}


def propagate(s: ImageSet, J, rounds=4, k=5, trust="agree", rule="tophalf", weighted=False, backward="pooled", rng=None, P1=None, track=None):
    """Returns the list of mask dicts after rounds 1..rounds. J: unlabeled images that take part (query, pool, distractors).
    weighted: use the reliabilities of the chosen references as vote / prototype weights (the labelled reference has weight 1).
    track: optional dict that receives the reliabilities and choices of every round (for diagnostics)."""
    P = dict(P1) if P1 is not None else one_shot(s, J); out = [P]
    for r in range(2, rounds + 1):
        rel = reliability(s, P, J, trust); ok = choose(rel, rule, rng); Pn = dict(P)
        if track is not None: track.setdefault("rel", []).append(rel); track.setdefault("ok", []).append(ok)
        top = max(max(rel.values()), 1e-9)
        for j in J:
            o = [x for x in J if x != j and ok[x] and P[x].any()]
            w = [1.0] + [rel[x] / top for x in o] if weighted else None
            Pn[j] = s.predict(j, [0] + o, [P[0]] + [P[x] for x in o], backward=backward, k=k, weights=w)
        P = Pn; out.append(P)
    return out
