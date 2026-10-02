"""Transductive in-context segmentation on the STANDARD COCO-20i episodes (same references and queries as the benchmark).
For episode e the method additionally sees up to --M unlabeled images of the same class: the queries of other episodes.
Only the episode's own query is scored; mIoU is the benchmark's (per class sum of intersections / sum of unions, mean over classes).
  1shot      INSID3 on (reference, query)                       = the benchmark number on these episodes
  naive      INSID3's own multi-reference rule with its one-shot masks of the pool images as extra references
  v0         pooled vote (k), pseudo references filtered by round-trip score > rt, --rounds rounds
  agree      like v0 but pseudo references are trusted by their agreement with the other pool masks (top half)
  true       pool images with their true masks, pooled vote (ceiling)
"""
import argparse, os, sys, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV
ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--M", type=int, default=15); ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--k", type=int, default=5)
ap.add_argument("--rt", type=float, default=0.5); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="")
a = ap.parse_args(); D = torch.load(a.file, weights_only=False); cls = np.array(D["cls"]); names = D["names"]; nE = len(names) // 2
rng = np.random.default_rng(0); acc = {}; t0 = time.time(); pools = []
iou = lambda p, q: float((p & q).sum()) / max(float((p | q).sum()), 1.0)
def add(name, c, cs, j, m):
    i, u = cs.iu(m, j); s = acc.setdefault(name, {}).setdefault(int(c), [0.0, 0.0]); s[0] += i; s[1] += u
E = range(nE if not a.limit else a.limit)
for e in E:
    c = cls[2 * e]; ref, qry = 2 * e, 2 * e + 1
    cand = [2 * o + 1 for o in range(nE) if o != e and cls[2 * o] == c and names[2 * o + 1] not in (names[ref], names[qry])]
    seen = set(); cand = [x for x in cand if not (names[x] in seen or seen.add(names[x]))]
    pool = [int(x) for x in rng.permutation(cand)[:a.M]]; idx = [ref, qry] + pool; pools.append(len(pool))
    cs = ClassSet(dict(c=int(c), names=[names[i] for i in idx], fq=D["fq"][idx], lab=D["lab"][idx], Po=[D["Po"][i] for i in idx], gt64=D["gt64"][idx], gt_bits=D["gt_bits"][idx], S=D["S"]))
    g0 = cs.gt64[0]; J = list(range(1, cs.n)); q = 1
    P1 = {0: g0}; P1.update({j: cs.predict(j, [0], [g0]) for j in J}); add("1shot", c, cs, q, P1[q])
    o = [x for x in J if x != q]
    add("naive", c, cs, q, cs.predict(q, [0] + o, [g0] + [P1[x] for x in o]))
    add("true", c, cs, q, cs.predict(q, [0] + o, [g0] + [cs.gt64[x] for x in o], backward="pooled", k=a.k))
    add("true, INSID3 vote", c, cs, q, cs.predict(q, [0] + o, [g0] + [cs.gt64[x] for x in o]))
    for mode in ("v0", "agree"):
        P = dict(P1)
        for r in range(2, a.rounds + 1):
            if mode == "v0": trust = {y: (iou(cs.predict(0, [y], [P[y]]), g0) > a.rt if P[y].any() else False) for y in J}
            else:
                X = {y: {x: cs.predict(x, [y], [P[y]]) for x in J if x != y} for y in J if P[y].any()}
                ag = {y: (np.mean([iou(X[y][x], P[x]) for x in J if x != y]) if y in X and len(J) > 1 else 0.0) for y in J}; med = np.median(list(ag.values())); trust = {y: ag[y] >= med and y in X for y in J}
            Pn = dict(P)
            for j in J:
                oo = [x for x in J if x != j and trust[x]]; Pn[j] = cs.predict(j, [0] + oo, [g0] + [P[x] for x in oo], backward="pooled", k=a.k)
            P = Pn
            if r in (2, a.rounds): add(f"{mode} round{r}", c, cs, q, P[q])
    del cs
    if (e + 1) % 50 == 0:
        torch.cuda.empty_cache()
        print(e + 1, f"{time.time() - t0:.0f}s", " ".join(f"{k} {100 * np.mean([i / max(u, 1) for i, u in v.values()]):.1f}" for k, v in acc.items()), f"| mean pool {np.mean(pools):.1f}", flush=True)
res = dict(file=a.file, episodes=len(E), mean_pool=float(np.mean(pools)), miou={k: 100 * float(np.mean([i / max(u, 1) for i, u in v.values()])) for k, v in acc.items()}, per_class={k: {c: 100 * i / max(u, 1) for c, (i, u) in v.items()} for k, v in acc.items()})
print(json.dumps({k: v for k, v in res.items() if k != "per_class"}))
if a.out: json.dump(res, open(a.out, "w"))
