"""Stream test: a pseudo label is trusted only where two decisions agree.
For every patch of an unlabeled image: region decision = INSID3's predicted mask; patch decision = its nearest patch in the
LABELLED reference is foreground. Agreement -> trusted foreground / trusted background; disagreement -> unknown (the patch is
not used as evidence for other images).   variants: both | fg (only purify foreground) | bg (only abstain on background) | none
"""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/demo4_cache/stream/f0_s0_N16"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--k", type=int, default=3); ap.add_argument("--show", default="")
a = ap.parse_args(); U = torch.load(f"{a.dir}/U.pt").to(DEV)
files = sorted(glob.glob(f"{a.dir}/class*.pt"), key=lambda p: int(os.path.basename(p)[5:-3])); rows = {}; t0 = time.time()
for f in files:
    cs = ClassSet(f, U); n = cs.n; J = list(range(1, n)); g0 = cs.gt64[0]; acc = {}; allk = torch.ones(cs.P, dtype=torch.bool, device=DEV)
    def add(name, j, m):
        i, u = cs.iu(m, j); s = acc.setdefault(name, [0.0, 0.0]); s[0] += i; s[1] += u
    b0 = [g0] + [g0[cs.nn(j, 0)] for j in J]                                   # patch decision from the labelled reference
    P1 = [g0] + [cs.predict(j, [0], [g0]) for j in J]
    for j in J:
        o = [x for x in J if x != j]; add("1shot", j, P1[j]); add("true masks", j, cs.predict(j, [0] + o, [g0] + [cs.gt64[x] for x in o], backward="pooled", k=a.k))
    for mode in ("none", "both", "fg", "bg"):
        P = P1
        for r in range(2, a.rounds + 1):
            kn = [allk]
            for x in J:
                agree_fg = P[x] & b0[x]; agree_bg = ~P[x] & ~b0[x]
                kn.append({"none": allk, "both": agree_fg | agree_bg, "fg": agree_fg | ~P[x], "bg": P[x] | agree_bg}[mode])
            Pn = [g0]
            for j in J:
                o = [x for x in J if x != j]; Pn.append(cs.predict(j, [0] + o, [g0] + [P[x] for x in o], backward="tri", k=a.k, known=[allk] + [kn[x] for x in o]))
            P = Pn
            for j in J: add(f"{mode} round{r}", j, P[j])
    rows[cs.c] = {k: 100 * v[0] / max(v[1], 1.0) for k, v in acc.items()}; del cs; torch.cuda.empty_cache()
names = list(next(iter(rows.values())))
print(f"{len(files)} classes, {time.time() - t0:.0f}s  ({a.dir})")
for k in names: print(f"   {k:16s} {np.mean([r[k] for r in rows.values()]):6.1f}   better/worse than 1shot by >2: {sum(r[k] > r['1shot'] + 2 for r in rows.values())}/{sum(r[k] < r['1shot'] - 2 for r in rows.values())}")
for k in [x for x in a.show.split(";") if x]: print(f"{k:16s}", " ".join(f"{c}:{r[k]:.0f}" for c, r in rows.items()))
