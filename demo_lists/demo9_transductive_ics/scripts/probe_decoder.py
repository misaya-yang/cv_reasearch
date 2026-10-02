#!/usr/bin/env python3
"""Diagnostic (uses true masks in some rows): what can a pool of images do that one (reference, query) pair cannot?

H_dec  With a pool, a patch classifier fitted in closed form on the pool (ridge regression on the debiased DINOv3 features,
       foreground against background, classes balanced) uses the references better than nearest-neighbour voting.
       Measured at the ceiling first: pool with TRUE masks, ridge against INSID3's pooled vote ("true").
H_cal  The labelled reference can calibrate a pool-fitted classifier: fit on the pool's pseudo masks only, choose the decision
       threshold that best re-segments the reference, apply it to the query.
H_id   Round-trip precision / recall on the labelled reference tells whether a concept's pseudo masks miss or over-cover:
       correlation over episodes with the true precision / recall of the one-shot pool masks.

Rows (only the episode's query is scored; pool = up to --M other queries of the same class):
  1shot, naive, true                          as in run_episodes.py
  ours                                        agree:tophalf, 4 rounds (the current method)
  ridge-true-patch / -cluster  [diagnostic]   classifier fitted on reference + pool with true masks; per patch / averaged in INSID3 clusters
  ridge-ref                                   classifier fitted on the reference image alone
  ridge-p1, ridge-p1 r3                       fitted on reference + one-shot pseudo masks of query and pool; 1 pass / 3 self-training passes
  ridge-ours                                  fitted on reference + the masks of "ours"
  ridge-p1-cal                                fitted on pseudo masks without the reference, threshold calibrated on the reference
  python scripts/probe_decoder.py --file $DEMO9_CACHE/episodes_f0_n400.pt --limit 200 --out results/probe_decoder_f0.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import numpy as np, torch, torch.nn.functional as F
from tics import ImageSet, one_shot, propagate

ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--M", type=int, default=15); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--alpha", type=float, default=0.1, help="ridge strength relative to the mean eigenvalue of the weighted covariance")
ap.add_argument("--alphas", default="0.01,0.1,1", help="ridge strengths reported for the true-mask ceiling")
ap.add_argument("--out", default=""); ap.add_argument("--every", type=int, default=50); ap.add_argument("--skip-ours", action="store_true")
a = ap.parse_args(); D = torch.load(a.file, weights_only=False); cls = np.array(D["cls"]); names = D["names"]; nE = len(names) // 2
rng = np.random.default_rng(0); t0 = time.time(); recs = []; DEV = "cuda"


def fit(s, imgs, masks, alpha, img_w=None):
    """balanced ridge on patches of `imgs` with labels `masks`; returns (beta, mu) with score(x) = (x - mu) @ beta"""
    X = torch.cat([s.fd[i] for i in imgs]); y = torch.cat([m.float() for m in masks]) * 2 - 1
    w = torch.cat([torch.full((s.P,), 1.0 if img_w is None else float(img_w[k]), device=DEV) for k in range(len(imgs))])
    pos = y > 0
    if pos.sum() == 0 or (~pos).sum() == 0: return None
    w = torch.where(pos, w / w[pos].sum(), w / w[~pos].sum()) * 0.5
    mu = (w[:, None] * X).sum(0); Xc = X - mu; C = Xc.T @ (w[:, None] * Xc); lam = alpha * torch.trace(C) / C.shape[0]
    beta = torch.linalg.solve(C + lam * torch.eye(C.shape[0], device=DEV), Xc.T @ (w * y)); return beta, mu


def score(s, t, model): return (s.fd[t] - model[1]) @ model[0]
def cl_mean(s, t, sc): return ((F.one_hot(s.lab[t], s.K[t]).float().T @ sc) / s.area[t])[s.lab[t]]
def pr(p, g):
    tp = float((p & g).sum()); return tp / max(float(p.sum()), 1.0), tp / max(float(g.sum()), 1.0)


def miou(key):
    acc = {}
    for r in recs:
        if key in r["iu"]: s_ = acc.setdefault(r["c"], [0.0, 0.0]); s_[0] += r["iu"][key][0]; s_[1] += r["iu"][key][1]
    return 100 * float(np.mean([i / max(u, 1.0) for i, u in acc.values()])) if acc else float("nan")


alphas = [float(x) for x in a.alphas.split(",")]
for e in range(nE if not a.limit else min(a.limit, nE)):
    c = int(cls[2 * e]); ref, qry = 2 * e, 2 * e + 1
    cand = [2 * o + 1 for o in range(nE) if o != e and cls[2 * o] == c and names[2 * o + 1] not in (names[ref], names[qry])]
    seen = set(); cand = [x for x in cand if not (names[x] in seen or seen.add(names[x]))]
    pool = [int(x) for x in rng.permutation(cand)[:a.M]]; idx = [ref, qry] + pool
    s = ImageSet(dict(c=c, names=[names[i] for i in idx], fq=D["fq"][idx], lab=D["lab"][idx], Po=[D["Po"][i] for i in idx], gt64=D["gt64"][idx], gt_bits=D["gt_bits"][idx], S=D["S"]))
    J = list(range(1, s.n)); q = 1; g0 = s.gt64[0]; o = [x for x in J if x != q]; P1 = one_shot(s, J)
    r = dict(e=e, c=c, pool=len(pool), iu={}); put = lambda k, m: r["iu"].__setitem__(k, s.iu(m, q))
    put("1shot", P1[q]); put("naive", s.predict(q, [0] + o, [g0] + [P1[x] for x in o], backward="majority"))
    put("true", s.predict(q, [0] + o, [g0] + [s.gt64[x] for x in o], backward="pooled", k=5))
    if not a.skip_ours:
        ours = propagate(s, J, rounds=4, k=5, trust="agree", rule="tophalf", P1=P1)[-1]; put("ours", ours[q])
    # ceiling of discriminative decoding: reference + pool with true masks
    for al in alphas:
        m = fit(s, [0] + o, [g0] + [s.gt64[x] for x in o], al)
        if m is not None:
            sc = score(s, q, m); put(f"ridge-true-patch a{al:g}", sc > 0); put(f"ridge-true-cluster a{al:g}", cl_mean(s, q, sc) > 0)
    m = fit(s, [0], [g0], a.alpha)
    if m is not None: put("ridge-ref", cl_mean(s, q, score(s, q, m)) > 0)
    # pseudo labels: one-shot masks of query and pool, then self-training
    lab = {j: P1[j] for j in J}
    for it in (1, 2, 3):
        use = [j for j in J if lab[j].any()]; m = fit(s, [0] + use, [g0] + [lab[j] for j in use], a.alpha)
        if m is None: break
        lab = {j: cl_mean(s, j, score(s, j, m)) > 0 for j in J}
        if it == 1: put("ridge-p1", lab[q])
        if it == 3: put("ridge-p1 r3", lab[q])
    if not a.skip_ours:
        use = [j for j in J if ours[j].any()]; m = fit(s, [0] + use, [g0] + [ours[j] for j in use], a.alpha)
        if m is not None: put("ridge-ours", cl_mean(s, q, score(s, q, m)) > 0)
    # gold calibration: fit without the reference, pick the threshold that best re-segments the reference
    use = [j for j in J if P1[j].any()]; m = fit(s, use, [P1[j] for j in use], a.alpha) if use else None
    if m is not None:
        s0 = cl_mean(s, 0, score(s, 0, m)); cands = torch.quantile(s0, torch.linspace(0.02, 0.98, 49, device=DEV))
        best = max(cands.tolist(), key=lambda t: s.iou(s0 > t, g0)); put("ridge-p1-cal", cl_mean(s, q, score(s, q, m)) > best)
        r["cal_thr"] = float(best); r["cal_ref_iou"] = s.iou(s0 > best, g0); r["nocal_ref_iou"] = s.iou(s0 > 0, g0)
    # H_id: round-trip precision / recall on the reference against the true precision / recall of the pool's pseudo masks
    tp_, tr_, gp_, gr_ = [], [], [], []
    for y in o:
        if not P1[y].any(): continue
        p, rc = pr(P1[y], s.gt64[y]); tp_.append(p); tr_.append(rc)
        p, rc = pr(s.predict(0, [y], [P1[y]]), g0); gp_.append(p); gr_.append(rc)
    if tp_: r["id"] = [float(np.mean(v)) for v in (tp_, tr_, gp_, gr_)]
    recs.append(r); del s
    if (e + 1) % a.every == 0:
        torch.cuda.empty_cache(); keys = [k for k in recs[0]["iu"]]
        print(e + 1, f"{time.time() - t0:.0f}s", " | ".join(f"{k} {miou(k):.1f}" for k in keys), flush=True)

keys = [k for k in recs[0]["iu"] if all(k in r["iu"] for r in recs)]
res = dict(file=a.file, args=vars(a), episodes=len(recs), mean_pool=float(np.mean([r["pool"] for r in recs])), miou={k: miou(k) for k in keys}, records=recs)
ids = np.array([r["id"] for r in recs if "id" in r])
if len(ids) > 10:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    res["id_corr"] = dict(precision=float(np.corrcoef(rank(ids[:, 0]), rank(ids[:, 2]))[0, 1]), recall=float(np.corrcoef(rank(ids[:, 1]), rank(ids[:, 3]))[0, 1]),
                          means=dict(true_precision=float(ids[:, 0].mean()), true_recall=float(ids[:, 1].mean()), gold_precision=float(ids[:, 2].mean()), gold_recall=float(ids[:, 3].mean())))
    print("H_id rank correlation over episodes (true pool pseudo-mask P/R against round-trip P/R on the reference):", json.dumps(res["id_corr"]))
print(json.dumps({k: round(v, 2) for k, v in res["miou"].items()}))
if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
