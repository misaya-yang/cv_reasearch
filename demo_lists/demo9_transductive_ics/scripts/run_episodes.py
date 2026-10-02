#!/usr/bin/env python3
"""GPU, seconds per episode: transductive variants on cached standard episodes. Only the episode's own query is scored.

Pool of episode e: up to --M queries of OTHER episodes of the same class (unlabeled), plus --D distractors (queries of other
classes whose image does not contain the class). Always reported: 1shot (INSID3), naive (INSID3's multi-reference vote with its
own one-shot masks of the pool), true (pool with true masks, pooled vote = ceiling).
--variants: comma list of  <trust>:<rule>[:w]   trust in none|roundtrip|agree|spectral|gold+agree,
            rule in all|tophalf|bottomhalf|randhalf|thr:<t>|top:<m>, ":w" = reliability-weighted vote.  See tics/propagate.py.
--diag adds rows that use true masks of pool images (label+m, pseudo-noFP, pseudo-noFN); they are diagnostics, not methods.
Per-episode intersections/unions are written to --out for paired statistics (scripts/stats.py).
  python scripts/run_episodes.py --file $DEMO9_CACHE/episodes_f0_n400.pt --out results/e1_f0.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import numpy as np, torch
from tics import ImageSet, one_shot, propagate

ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--M", type=int, default=15); ap.add_argument("--D", type=int, default=0)
ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--k", type=int, default=5); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--pool-seed", type=int, default=0)
ap.add_argument("--variants", default="agree:tophalf,agree:randhalf,agree:bottomhalf,agree:all:w,agree:tophalf:w,spectral:tophalf,gold+agree:tophalf,roundtrip:thr:0.5,none:all")
ap.add_argument("--out", default=""); ap.add_argument("--every", type=int, default=50)
ap.add_argument("--diag", action="store_true", help="add diagnostic rows that use the TRUE masks of pool images: label+m (m extra labelled references, INSID3's own vote),\n"
                "pseudo-noFP / pseudo-noFN (one-shot pool masks with their false positives removed / their misses filled, pooled vote)")
a = ap.parse_args(); D = torch.load(a.file, weights_only=False); cls = np.array(D["cls"]); names = D["names"]; nE = len(names) // 2
present = D.get("present"); rng = np.random.default_rng(a.pool_seed); rng_d = np.random.default_rng(a.pool_seed + 1); t0 = time.time(); recs = []
def parse(v):
    t = v.split(":"); w = t[-1] == "w"; t = t[:-1] if w else t; return t[0], ":".join(t[1:]), w
variants = [(v, *parse(v)) for v in a.variants.split(",") if v]
def miou(key):
    acc = {}
    for r in recs:
        if key in r["iu"]: s_ = acc.setdefault(r["c"], [0.0, 0.0]); s_[0] += r["iu"][key][0]; s_[1] += r["iu"][key][1]
    return 100 * float(np.mean([i / max(u, 1.0) for i, u in acc.values()])) if acc else float("nan")
for e in range(nE if not a.limit else min(a.limit, nE)):
    c = int(cls[2 * e]); ref, qry = 2 * e, 2 * e + 1
    cand = [2 * o + 1 for o in range(nE) if o != e and cls[2 * o] == c and names[2 * o + 1] not in (names[ref], names[qry])]
    seen = set(); cand = [x for x in cand if not (names[x] in seen or seen.add(names[x]))]
    pool = [int(x) for x in rng.permutation(cand)[:a.M]]; dis = []
    if a.D:
        dc = [2 * o + 1 for o in range(nE) if cls[2 * o] != c and c not in present[2 * o + 1] and names[2 * o + 1] not in [names[i] for i in [ref, qry] + pool]]
        dis = [int(x) for x in rng_d.permutation(dc)[:a.D]]
    idx = [ref, qry] + pool + dis; gt64 = D["gt64"][idx].clone(); gb = D["gt_bits"][idx].copy()
    if dis: gt64[-len(dis):] = False; gb[-len(dis):] = 0                      # distractors do not contain the class
    s = ImageSet(dict(c=c, names=[names[i] for i in idx], fq=D["fq"][idx], lab=D["lab"][idx], Po=[D["Po"][i] for i in idx], gt64=gt64, gt_bits=gb, S=D["S"]))
    J = list(range(1, s.n)); q = 1; g0 = s.gt64[0]; P1 = one_shot(s, J); o = [x for x in J if x != q]; real = [x for x in o if x < 2 + len(pool)]
    r = dict(e=e, c=c, pool=len(pool), dis=len(dis), iu={})
    r["iu"]["1shot"] = s.iu(P1[q], q)
    r["iu"]["naive"] = s.iu(s.predict(q, [0] + o, [g0] + [P1[x] for x in o], backward="majority"), q)
    r["iu"]["true"] = s.iu(s.predict(q, [0] + real, [g0] + [s.gt64[x] for x in real], backward="pooled", k=a.k), q)
    if a.diag:
        for m in (1, 2, 4):
            if len(real) >= m: r["iu"][f"label+{m}"] = s.iu(s.predict(q, [0] + real[:m], [g0] + [s.gt64[x] for x in real[:m]], backward="majority"), q)
        r["iu"]["pseudo-noFP"] = s.iu(s.predict(q, [0] + real, [g0] + [P1[x] & s.gt64[x] for x in real], backward="pooled", k=a.k), q)
        r["iu"]["pseudo-noFN"] = s.iu(s.predict(q, [0] + real, [g0] + [P1[x] | s.gt64[x] for x in real], backward="pooled", k=a.k), q)
        r["iu"]["pseudo-all"] = s.iu(s.predict(q, [0] + real, [g0] + [P1[x] for x in real], backward="pooled", k=a.k), q)
    for name, trust, rule, w in variants:
        track = {}; out = propagate(s, J, rounds=a.rounds, k=a.k, trust=trust, rule=rule, weighted=w, rng=np.random.default_rng(1000 + e), P1=P1, track=track)
        r["iu"][name] = s.iu(out[-1][q], q); r["iu"][name + " r2"] = s.iu(out[1][q], q)
        if dis: r.setdefault("dis_used", {})[name] = float(np.mean([track["ok"][-1][x] for x in J if x >= 2 + len(pool)]))
    recs.append(r); del s
    if (e + 1) % a.every == 0:
        torch.cuda.empty_cache(); print(e + 1, f"{time.time() - t0:.0f}s", " | ".join(f"{k} {miou(k):.1f}" for k in ["1shot", "naive", "true"] + [v[0] for v in variants]), flush=True)
keys = [k for k in recs[0]["iu"] if all(k in r["iu"] for r in recs)]; res = dict(file=a.file, args=vars(a), episodes=len(recs), mean_pool=float(np.mean([r["pool"] for r in recs])), miou={k: miou(k) for k in keys}, records=recs)
if a.D: res["distractors_used"] = {v[0]: float(np.mean([r["dis_used"][v[0]] for r in recs if "dis_used" in r])) for v in variants}; print("share of distractors used as references (last round):", res["distractors_used"])
print(json.dumps(res["miou"]))
if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
