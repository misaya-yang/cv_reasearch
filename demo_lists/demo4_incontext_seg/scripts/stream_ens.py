"""Stream test: cross-prediction matrix. X[y][x] = INSID3's mask of image x when image y (with its current mask) is the only
reference. From it:
  agreement A[y, x] = IoU(X[y][x], current mask of x): how well reference y reproduces what the pool believes about x
  ensemble   V_x     = share of trusted references whose single-reference prediction marks a patch of x as foreground
Variants (all use the pooled rule, k=5, for the final prediction):
  rt            trusted references = round-trip score > 0.5 (version 0)
  agree         trusted = mean agreement with the other images' masks in the top half
  rt+tri        trusted by round-trip; a patch of a pseudo reference is used as evidence only if its mask label agrees with the
                ensemble vote (foreground: V >= 0.5, background: V <= 0.2); other patches are unknown
  ens           the ensemble vote itself as the prediction (V >= 0.5)
"""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/demo4_cache/stream/f0_s0_N16"); ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--k", type=int, default=5); ap.add_argument("--N", type=int, default=16)
a = ap.parse_args(); U = torch.load(f"{a.dir}/U.pt").to(DEV)
files = sorted(glob.glob(f"{a.dir}/class*.pt"), key=lambda p: int(os.path.basename(p)[5:-3])); rows = {}; t0 = time.time()
for f in files:
    cs = ClassSet(f, U); J = list(range(1, min(a.N, cs.n_real - 1) + 1)); g0 = cs.gt64[0]; acc = {}; allk = torch.ones(cs.P, dtype=torch.bool, device=DEV)
    def add(name, j, m):
        i, u = cs.iu(m, j); s = acc.setdefault(name, [0.0, 0.0]); s[0] += i; s[1] += u
    iou = lambda p, q: float((p & q).sum()) / max(float((p | q).sum()), 1.0)
    P1 = {0: g0}; P1.update({j: cs.predict(j, [0], [g0]) for j in J})
    for j in J:
        o = [x for x in J if x != j]; add("1shot", j, P1[j]); add("true masks", j, cs.predict(j, [0] + o, [g0] + [cs.gt64[x] for x in o], backward="pooled", k=a.k))
    for mode in ("rt", "agree", "rt+tri", "ens"):
        P = dict(P1)
        for r in range(2, a.rounds + 1):
            X = {y: {x: cs.predict(x, [y], [P[y]]) for x in [0] + J if x != y} for y in J if P[y].any()}    # cross predictions
            rt = {y: (iou(X[y][0], g0) if y in X else 0.0) for y in J}
            if mode == "agree":
                ag = {y: (np.mean([iou(X[y][x], P[x]) for x in J if x != y]) if y in X else 0.0) for y in J}; med = np.median(list(ag.values())); trust = {y: ag[y] >= med for y in J}
            else: trust = {y: rt[y] > 0.5 for y in J}
            Pn = dict(P)
            for j in J:
                o = [x for x in J if x != j and trust[x]]
                if mode == "ens":
                    votes = [P1[j].float()] + [X[y][j].float() for y in o if y in X]; Pn[j] = torch.stack(votes).mean(0) >= 0.5
                elif mode == "rt+tri":
                    kn = [allk]
                    for x in o:
                        srcs = [y for y in J if y != x and trust[y] and y in X]; V = torch.stack([P1[x].float()] + [X[y][x].float() for y in srcs]).mean(0)
                        kn.append((P[x] & (V >= 0.5)) | (~P[x] & (V <= 0.2)))
                    Pn[j] = cs.predict(j, [0] + o, [g0] + [P[x] for x in o], backward="tri", k=a.k, known=kn)
                else:
                    Pn[j] = cs.predict(j, [0] + o, [g0] + [P[x] for x in o], backward="pooled", k=a.k)
            P = Pn
            for j in J: add(f"{mode} round{r}", j, P[j])
    rows[cs.c] = {k: 100 * v[0] / max(v[1], 1.0) for k, v in acc.items()}; del cs; torch.cuda.empty_cache()
    print(f"class {f.split('class')[-1][:-3]} {time.time() - t0:.0f}s", flush=True)
names = list(next(iter(rows.values())))
for k in names: print(f"   {k:20s} {np.mean([r[k] for r in rows.values()]):6.1f}   better/worse than 1shot by >2: {sum(r[k] > r['1shot'] + 2 for r in rows.values())}/{sum(r[k] < r['1shot'] - 2 for r in rows.values())}")
for k in ("1shot", "rt round4", "rt+tri round4", "ens round4", "true masks"): print(f"{k:16s}", " ".join(f"{c}:{r[k]:.0f}" for c, r in rows.items()))
