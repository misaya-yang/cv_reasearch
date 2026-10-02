#!/usr/bin/env python3
"""Diagnostic: the best single reference beats pooling (probe_errors.py, fold 0: one oracle-chosen pool image with its true mask 74.8,
all 15 true masks pooled 67.7; oracle choice among the one-shot result and the 15 single-PSEUDO-reference results 70.7).
Can the best hypothesis for the query be found without labels?

Candidates for the query q: the direct one-shot mask (gold -> q) and, for every pool image y, the mask of q predicted from y alone
with y's one-shot mask (gold -> y -> q). Label-free scores of a candidate m:
  rtg    long round trip: IoU of the gold mask re-predicted from (q, m) with the true gold mask
  cyc    pairwise cycle: IoU of y re-predicted from (q, m) with the mask of y that produced m   (for the direct candidate: rtg)
  link   round trip of y: IoU of the gold mask re-predicted from (y, mask of y) with the true gold mask   (direct candidate: 1)
  fms    forward match strength: mean similarity of the reference's foreground patches to their nearest patch in q
Rows: sel-<score> (candidate with the best score), sel-rtg-cons (replace the one-shot only when rtg improves by 0.05),
      pool-top3<score> (gold + the three best pool images by score, pooled vote), best1-pseudo (oracle).
The same with TRUE masks on the pool (diagnostic of the selector alone): selT-<score>, true-top3<score>, best1-true, true.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import numpy as np, torch
from tics import ImageSet, one_shot

ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--M", type=int, default=15); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=""); ap.add_argument("--every", type=int, default=100)
a = ap.parse_args(); D = torch.load(a.file, weights_only=False); cls = np.array(D["cls"]); names = D["names"]; nE = len(names) // 2
rng = np.random.default_rng(0); t0 = time.time(); recs = []


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
    J = list(range(1, s.n)); q = 1; g0 = s.gt64[0]; gq = s.gt64[q]; o = [x for x in J if x != q]; P1 = one_shot(s, J)
    r = dict(e=e, c=c, iu={}); put = lambda k, m: r["iu"].__setitem__(k, s.iu(m, q))
    rtg = lambda m: s.iou(s.predict(0, [q], [m]), g0) if m.any() else 0.0
    def table(masks_of):                                           # masks_of: reference mask of every pool image
        ys = [y for y in o if masks_of[y].any()]; M = [P1[q]] + [s.predict(q, [y], [masks_of[y]]) for y in ys]
        sc = dict(rtg=np.array([rtg(m) for m in M]))
        sc["cyc"] = np.array([sc["rtg"][0]] + [s.iou(s.predict(y, [q], [m]), masks_of[y]) if m.any() else 0.0 for y, m in zip(ys, M[1:])])
        sc["link"] = np.array([1.0] + [s.iou(s.predict(0, [y], [masks_of[y]]), g0) for y in ys])
        sc["fms"] = np.array([float(s.nnv(0, q)[0][g0].mean())] + [float(s.nnv(y, q)[0][masks_of[y]].mean()) for y in ys])
        sc["rtg*cyc"] = sc["rtg"] * sc["cyc"]; sc["rtg*link"] = sc["rtg"] * sc["link"]; sc["cyc*link"] = sc["cyc"] * sc["link"]; sc["rtg*cyc*link"] = sc["rtg"] * sc["cyc"] * sc["link"]
        return ys, M, sc, np.array([s.iou(m, gq) for m in M])
    put("1shot", P1[q]); put("pool", s.predict(q, [0] + o, [g0] + [P1[x] for x in o], backward="pooled", k=5))
    put("true", s.predict(q, [0] + o, [g0] + [s.gt64[x] for x in o], backward="pooled", k=5))
    for tag, masks_of, pre in (("pseudo", P1, "sel-"), ("true", {y: s.gt64[y] for y in o}, "selT-")):
        ys, M, sc, t = table(masks_of); put("best1-" + tag, M[int(t.argmax())]); r[tag] = dict(t=t.tolist(), **{k: v.tolist() for k, v in sc.items()})
        for k, v in sc.items():
            put(pre + k, M[int(v.argmax())])
            if len(ys) >= 3 and k in ("rtg", "fms", "rtg*cyc", "rtg*cyc*link"):
                top = [ys[i - 1] for i in np.argsort(-v[1:])[:3] + 1]
                put(("pool" if tag == "pseudo" else "true") + "-top3" + k, s.predict(q, [0] + top, [g0] + [masks_of[y] for y in top], backward="pooled", k=5))
        i = int(sc["rtg"].argmax()); put(pre + "rtg-cons", M[i] if sc["rtg"][i] > sc["rtg"][0] + 0.05 else M[0])
    recs.append(r); del s
    if (e + 1) % a.every == 0:
        torch.cuda.empty_cache(); print(e + 1, f"{time.time() - t0:.0f}s", " | ".join(f"{k} {miou(k):.1f}" for k in ("1shot", "pool", "true", "best1-pseudo", "sel-rtg", "sel-rtg*cyc", "selT-rtg", "best1-true")), flush=True)

keys = [k for k in recs[0]["iu"] if all(k in r["iu"] for r in recs)]; res = dict(file=a.file, args=vars(a), episodes=len(recs), miou={k: miou(k) for k in keys}, records=recs)
print("\nmIoU:", json.dumps({k: round(v, 1) for k, v in res["miou"].items()}))
rank = lambda v: np.argsort(np.argsort(v)).astype(float)
for tag in ("pseudo", "true"):
    print(f"\n{tag} masks on the pool: mean within-episode rank correlation of each score with the candidates' true IoU; mean true IoU of the chosen candidate; share of episodes where the choice is within 0.05 of the best")
    rows = [r[tag] for r in recs if len(r[tag]["t"]) >= 4]
    best = np.mean([max(x["t"]) for x in rows]); first = np.mean([x["t"][0] for x in rows]); print(f"  oracle best {best:.3f}   direct one-shot {first:.3f}   mean candidate {np.mean([np.mean(x['t']) for x in rows]):.3f}   direct candidate is the best in {np.mean([int(np.argmax(x['t'])) == 0 for x in rows]):.2f}")
    for k in ("rtg", "cyc", "link", "fms", "rtg*cyc", "rtg*link", "cyc*link", "rtg*cyc*link"):
        rho = [np.corrcoef(rank(np.array(x[k])), rank(np.array(x["t"])))[0, 1] for x in rows if np.std(x[k]) > 0 and np.std(x["t"]) > 0]
        ch = [x["t"][int(np.argmax(x[k]))] for x in rows]; print(f"  {k:13s} rho {np.mean(rho):+.2f}   chosen {np.mean(ch):.3f}   within 0.05 of best {np.mean([max(x['t']) - c_ <= 0.05 for x, c_ in zip(rows, ch)]):.2f}")
if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
