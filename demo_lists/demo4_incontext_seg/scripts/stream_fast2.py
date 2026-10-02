"""Stream sweep on cached features: pooled-kNN backward rule, iterative rounds, round-trip filter. Prints class means only
(and per-class for --show). Same data and metric as stream_fast.py."""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/demo4_cache/stream/f0_s0_N16"); ap.add_argument("--show", default="")
a = ap.parse_args(); U = torch.load(f"{a.dir}/U.pt").to(DEV)
files = sorted(glob.glob(f"{a.dir}/class*.pt"), key=lambda p: int(os.path.basename(p)[5:-3])); rows = {}; t0 = time.time()
for f in files:
    cs = ClassSet(f, U); n = cs.n; J = list(range(1, n)); g0 = cs.gt64[0]; acc = {}
    def add(name, j, m):
        i, u = cs.iu(m, j); s = acc.setdefault(name, [0.0, 0.0]); s[0] += i; s[1] += u
    def rtrip(P): return np.array([1.0] + [cs.iou64(cs.predict(0, [j], [P[j]]), 0) if P[j].any() else 0.0 for j in J])
    P1 = [g0] + [cs.predict(j, [0], [g0]) for j in J]
    for j in J: add("1shot", j, P1[j])
    for k in (3, 5, 7):
        kw = dict(backward="pooled", k=k)
        for j in J:
            o = [x for x in J if x != j]; add(f"gt:all k{k}", j, cs.predict(j, [0] + o, [g0] + [cs.gt64[x] for x in o], **kw))
        for filt in ("none", "rt0.5"):
            P = P1
            for r in (2, 3, 4):
                rt = rtrip(P) if filt != "none" else np.ones(n); Pn = [g0]
                for j in J:
                    o = [x for x in J if x != j and rt[x] > 0.5]; Pn.append(cs.predict(j, [0] + o, [g0] + [P[x] for x in o], **kw))
                P = Pn
                for j in J: add(f"pseudo k{k} {filt} round{r}", j, P[j])
    # INSID3's own majority vote, iterated
    P = P1
    for r in (2, 3, 4):
        Pn = [g0]
        for j in J:
            o = [x for x in J if x != j]; Pn.append(cs.predict(j, [0] + o, [g0] + [P[x] for x in o]))
        P = Pn
        for j in J: add(f"pseudo majority none round{r}", j, P[j])
    rows[cs.c] = {k: 100 * v[0] / max(v[1], 1.0) for k, v in acc.items()}; del cs; torch.cuda.empty_cache()
names = list(next(iter(rows.values())))
print(f"{len(files)} classes, {time.time() - t0:.0f}s")
for k in names: print(f"   {k:34s} {np.mean([r[k] for r in rows.values()]):6.1f}   wins/losses vs 1shot by >2 pts: {sum(r[k] > r['1shot'] + 2 for r in rows.values())}/{sum(r[k] < r['1shot'] - 2 for r in rows.values())}")
for k in [x for x in a.show.split(",") if x]:
    print(k, " ".join(f"{c}:{r[k]:.0f}" for c, r in rows.items()))
