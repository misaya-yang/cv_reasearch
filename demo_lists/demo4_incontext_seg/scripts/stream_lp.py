"""Label propagation over the image set on cached features.

State: a soft label in [-1, 1] per patch of every image; the labelled reference is clamped to its true mask (+1 / -1).
Start: INSID3's one-shot masks (+1 / -1). One round: every unlabeled image re-estimates its patches from the other
images (ClassSet.transfer), averages inside its clusters, and (optionally) is binarised. Read-out: score > 0.
Ceilings use the TRUE masks of the other images as their labels (what the same read-out gives with perfect labels).
"""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/demo4_cache/stream/f0_s0_N16"); ap.add_argument("--show", default="")
a = ap.parse_args(); U = torch.load(f"{a.dir}/U.pt").to(DEV)
files = sorted(glob.glob(f"{a.dir}/class*.pt"), key=lambda p: int(os.path.basename(p)[5:-3])); rows = {}; t0 = time.time()
pm = lambda m: m.float() * 2 - 1
for f in files:
    cs = ClassSet(f, U); n = cs.n; J = list(range(1, n)); g0 = cs.gt64[0]; acc = {}
    def add(name, j, m):
        i, u = cs.iu(m, j); s = acc.setdefault(name, [0.0, 0.0]); s[0] += i; s[1] += u
    P1 = [g0] + [cs.predict(j, [0], [g0]) for j in J]
    for j in J:
        add("1shot INSID3", j, P1[j])
        add("1shot transfer, patch", j, cs.transfer(j, [0], [pm(g0)]) > 0)
        add("1shot transfer, cluster", j, cs.cluster_mean(j, cs.transfer(j, [0], [pm(g0)])) > 0)
    for k in (1, 3, 5, 16):
        for j in J:
            o = [x for x in J if x != j]; sc = cs.transfer(j, [0] + o, [pm(g0)] + [pm(cs.gt64[x]) for x in o], k=k)
            add(f"ceiling k{k} patch", j, sc > 0); add(f"ceiling k{k} cluster", j, cs.cluster_mean(j, sc) > 0)
    for k, hard, w0 in ((3, True, 1.0), (5, True, 1.0), (5, False, 1.0), (16, False, 1.0), (5, False, 3.0), (5, True, 3.0)):
        Sx = [pm(g0)] + [pm(P1[j]) for j in J]
        for r in range(2, 6):
            Sn = [pm(g0)]
            for j in J:
                o = [x for x in J if x != j]; sc = cs.cluster_mean(j, cs.transfer(j, [0] + o, [Sx[0]] + [Sx[x] for x in o], k=k, w0=w0))
                Sn.append(pm(sc > 0) if hard else sc.clamp(-1, 1))
            Sx = Sn
            if r in (2, 3, 5):
                for j in J: add(f"LP k{k} {'hard' if hard else 'soft'} w0={w0} round{r}", j, Sx[j] > 0)
    rows[cs.c] = {k: 100 * v[0] / max(v[1], 1.0) for k, v in acc.items()}; del cs; torch.cuda.empty_cache()
names = list(next(iter(rows.values())))
print(f"{len(files)} classes, {time.time() - t0:.0f}s")
for k in names: print(f"   {k:34s} {np.mean([r[k] for r in rows.values()]):6.1f}   wins/losses vs INSID3 1shot by >2 pts: {sum(r[k] > r['1shot INSID3'] + 2 for r in rows.values())}/{sum(r[k] < r['1shot INSID3'] - 2 for r in rows.values())}")
for k in [x for x in a.show.split(",") if x]:
    print(k, " ".join(f"{c}:{r[k]:.0f}" for c, r in rows.items()))
