#!/usr/bin/env python3
"""Diagnostic: perfect reliability selection is worth +5 (probe_errors.py: pooled vote with only the "easy" pool images 62.1,
with all of them 57.0), the label-free selection in use about +1. Which label-free signal tells easy from hard pool images,
and can hard images be repaired from easy ones?

Per pool image y (one-shot mask P1[y]; easy = its IoU with the truth >= 0.5), label-free features:
  link   IoU of the gold mask re-predicted from (y, P1[y]) with the true gold mask
  agree  mean IoU of the other pool images' masks re-predicted from (y, P1[y]) with their own one-shot masks
  fmsg   mean similarity of the gold's foreground patches to their nearest patch in y
  fmsb   mean similarity of P1[y]'s patches to their nearest patch in the gold image
  marg   mean over P1[y]'s patches of (similarity to nearest gold foreground patch - to nearest gold background patch)
  area   log(|P1[y]| / |gold mask|)
Rows (pooled vote for the query): pool, pool-easy (oracle), keep-<feature> (the 62% of the pool with the highest feature),
  link>=0.5, two-stage-oracle (hard images re-predicted from gold + easy images, then all used), two-stage-link (same with link).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import numpy as np, torch
from tics import ImageSet, one_shot

ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--M", type=int, default=15); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=""); ap.add_argument("--every", type=int, default=100); ap.add_argument("--keep", type=float, default=0.62)
a = ap.parse_args(); D = torch.load(a.file, weights_only=False); cls = np.array(D["cls"]); names = D["names"]; nE = len(names) // 2
rng = np.random.default_rng(0); t0 = time.time(); recs = []; FEATS = ["link", "agree", "fmsg", "fmsb", "marg", "area"]


def miou(key):
    acc = {}
    for r in recs:
        if key in r["iu"]: s_ = acc.setdefault(r["c"], [0.0, 0.0]); s_[0] += r["iu"][key][0]; s_[1] += r["iu"][key][1]
    return 100 * float(np.mean([i / max(u, 1.0) for i, u in acc.values()])) if acc else float("nan")


for e in range(nE if not a.limit else min(a.limit, nE)):
    c = int(cls[2 * e]); ref, qry = 2 * e, 2 * e + 1
    cand = [2 * o_ + 1 for o_ in range(nE) if o_ != e and cls[2 * o_] == c and names[2 * o_ + 1] not in (names[ref], names[qry])]
    seen = set(); cand = [x for x in cand if not (names[x] in seen or seen.add(names[x]))]
    pool = [int(x) for x in rng.permutation(cand)[:a.M]]; idx = [ref, qry] + pool
    s = ImageSet(dict(c=c, names=[names[i] for i in idx], fq=D["fq"][idx], lab=D["lab"][idx], Po=[D["Po"][i] for i in idx], gt64=D["gt64"][idx], gt_bits=D["gt_bits"][idx], S=D["S"]))
    J = list(range(1, s.n)); q = 1; g0 = s.gt64[0]; o = [x for x in J if x != q]; P1 = one_shot(s, J)
    r = dict(e=e, c=c, iu={}); put = lambda k, m: r["iu"].__setitem__(k, s.iu(m, q)); pv = lambda ys, M: s.predict(q, [0] + ys, [g0] + [M[y] for y in ys], backward="pooled", k=5)
    put("1shot", P1[q]); put("pool", pv(o, P1))
    if len(o) >= 4:
        f = {k: {} for k in FEATS}; qual = {}
        for y in o:
            m = P1[y]; qual[y] = s.iou(m, s.gt64[y])
            if not m.any():
                for k in FEATS: f[k][y] = -1.0 if k != "area" else -9.0
                continue
            f["link"][y] = s.iou(s.predict(0, [y], [m]), g0)
            f["agree"][y] = float(np.mean([s.iou(s.predict(x, [y], [m]), P1[x]) for x in J if x != y]))
            f["fmsg"][y] = float(s.nnv(0, y)[0][g0].mean()) if g0.any() else 0.0; f["fmsb"][y] = float(s.nnv(y, 0)[0][m].mean())
            S_ = s.fd[y][m] @ s.fd[0].T; f["marg"][y] = float((S_[:, g0].max(1).values - S_[:, ~g0].max(1).values).mean()) if g0.any() else 0.0
            f["area"][y] = float(np.log(max(float(m.sum()), 1.0) / max(float(g0.sum()), 1.0)))
        r["pool"] = dict(qual=[qual[y] for y in o], **{k: [f[k][y] for y in o] for k in FEATS}); nk = max(1, int(round(a.keep * len(o))))
        easy = [y for y in o if qual[y] >= 0.5]; put("pool-easy", pv(easy, P1))
        for k in FEATS[:5]: put("keep-" + k, pv(sorted(o, key=lambda y: -f[k][y])[:nk], P1))
        z = {k: (np.array([f[k][y] for y in o]) - np.mean([f[k][y] for y in o])) / (np.std([f[k][y] for y in o]) + 1e-6) for k in ("link", "agree", "marg")}
        zs = z["link"] + z["agree"] + z["marg"]; put("keep-link+agree+marg", pv([o[i] for i in np.argsort(-zs)[:nk]], P1))
        put("link>=0.5", pv([y for y in o if f["link"][y] >= 0.5], P1))
        for tag, good in (("oracle", easy), ("link", sorted(o, key=lambda y: -f["link"][y])[:nk])):
            M2 = dict(P1)
            for y in o:
                if y not in good: M2[y] = s.predict(y, [0] + good, [g0] + [P1[x] for x in good], backward="pooled", k=5)
            put("two-stage-" + tag, pv(o, M2)); r["fixed_" + tag] = [float(np.mean([qual[y] for y in o if y not in good] or [0])), float(np.mean([s.iou(M2[y], s.gt64[y]) for y in o if y not in good] or [0]))]
    recs.append(r); del s
    if (e + 1) % a.every == 0:
        torch.cuda.empty_cache(); print(e + 1, f"{time.time() - t0:.0f}s", " | ".join(f"{k} {miou(k):.1f}" for k in ("1shot", "pool", "pool-easy", "keep-link", "keep-agree", "two-stage-oracle")), flush=True)

keys = [k for k in recs[0]["iu"]]; full = [r for r in recs if "pool-easy" in r["iu"]]; recs_all = recs; recs = full
res = dict(file=a.file, args=vars(a), episodes=len(recs), miou={k: miou(k) for k in recs[0]["iu"]}, records=recs)
print(f"\nmIoU over the {len(recs)} episodes with a pool of at least 4:", json.dumps({k: round(v, 1) for k, v in res["miou"].items()}))
X = {k: np.concatenate([r["pool"][k] for r in recs]) for k in FEATS}; yq = np.concatenate([r["pool"]["qual"] for r in recs]); ye = yq >= 0.5
grp = np.concatenate([[r["c"]] * len(r["pool"]["qual"]) for r in recs])
def auc(sc, y):
    o_ = np.argsort(sc); rk = np.empty(len(sc)); rk[o_] = np.arange(1, len(sc) + 1); n1 = y.sum(); n0 = len(y) - n1; return float((rk[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
print(f"pool images {len(yq)}, easy share {ye.mean():.2f}. AUC for easy / rank correlation with the mask's true IoU:")
rank = lambda v: np.argsort(np.argsort(v)).astype(float)
for k in FEATS: print(f"  {k:6s} AUC {auc(X[k], ye):.3f}   rho {np.corrcoef(rank(X[k]), rank(yq))[0, 1]:+.2f}")
Z = np.stack([X[k] for k in FEATS], 1); Z = np.concatenate([Z, Z[:, [5]] ** 2], 1); p = np.zeros(len(yq))
for c_ in np.unique(grp):                                           # logistic regression, leave one class out
    tr = grp != c_; mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6; A = np.c_[(Z[tr] - mu) / sd, np.ones(tr.sum())]; w = np.zeros(A.shape[1])
    for _ in range(300): pr_ = 1 / (1 + np.exp(-A @ w)); w -= 0.5 * (A.T @ (pr_ - ye[tr]) / len(pr_) + 1e-3 * w)
    p[~tr] = np.c_[(Z[~tr] - mu) / sd, np.ones((~tr).sum())] @ w
print(f"  logistic regression on all features, leave-one-class-out: AUC {auc(p, ye):.3f}   rho {np.corrcoef(rank(p), rank(yq))[0, 1]:+.2f}")
for tag in ("oracle", "link"):
    v = np.array([r["fixed_" + tag] for r in recs]); print(f"two-stage-{tag}: mean true IoU of the re-predicted (not kept) pool images before {v[:, 0].mean():.3f} -> after {v[:, 1].mean():.3f}")
if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
