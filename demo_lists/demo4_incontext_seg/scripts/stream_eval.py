"""Transductive in-context segmentation, version 0, on cached stream sets (all folds given by --dirs).

  round 1   every unlabeled image is segmented from the labelled reference alone (INSID3)
  round r   image j is segmented from the labelled reference plus the other pool images whose round-trip score exceeds
            --rt (their current masks are the reference masks); INSID3's backward majority vote over reference images is
            replaced by the vote of the --k references whose nearest patch is most similar ("pooled")
  round-trip score of image x: IoU between the true mask of the labelled reference and the mask obtained on the labelled
  reference when (x, current mask of x) is the only reference. It uses no label of x.
--pool N: the pool is the first N unlabeled images; the first --test of them are scored (so pools of different size are
compared on the same test images).  Output: mean over classes of sum-IoU, per fold and averaged.
"""
import argparse, glob, os, sys, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV
ap = argparse.ArgumentParser(); ap.add_argument("--dirs", default=",".join(f"/root/demo4_cache/stream/f{f}_s0_N16" for f in range(4)))
ap.add_argument("--pools", default="16"); ap.add_argument("--test", type=int, default=0); ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--k", type=int, default=5); ap.add_argument("--rt", type=float, default=0.5)
ap.add_argument("--out", default=""); ap.add_argument("--distract", default="0", help="numbers of distractor images added to the pool, e.g. 0,16")
a = ap.parse_args(); res = {}
for d in a.dirs.split(","):
    U = torch.load(f"{d}/U.pt").to(DEV); files = sorted(glob.glob(f"{d}/class*.pt"), key=lambda p: int(os.path.basename(p)[5:-3])); rows = {}; t0 = time.time()
    for f in files:
        cs = ClassSet(f, U); g0 = cs.gt64[0]; acc = {}
        def add(name, j, m):
            i, u = cs.iu(m, j); s = acc.setdefault(name, [0.0, 0.0]); s[0] += i; s[1] += u
        P1 = [g0] + [cs.predict(j, [0], [g0]) for j in range(1, cs.n)]; kept = {}
        for N, Dn in [(int(x), int(y)) for x in a.pools.split(",") for y in a.distract.split(",")]:
            real = list(range(1, min(N, cs.n_real - 1) + 1)); T = real[:a.test] if a.test else real
            J = real + list(range(cs.n_real, min(cs.n_real + Dn, cs.n))); N = f"{N}+{Dn}d" if Dn else N
            for j in T:
                o = [x for x in J if x != j]; add(f"N={N} 1shot", j, P1[j])
                add(f"N={N} naive (INSID3 vote, all, 2 rounds)", j, cs.predict(j, [0] + o, [g0] + [P1[x] for x in o]))
                add(f"N={N} true masks, INSID3 vote", j, cs.predict(j, [0] + o, [g0] + [cs.gt64[x] for x in o]))
                add(f"N={N} true masks, pooled", j, cs.predict(j, [0] + o, [g0] + [cs.gt64[x] for x in o], backward="pooled", k=a.k))
            P = list(P1)
            for r in range(2, a.rounds + 1):
                rt = {x: (cs.iou64(cs.predict(0, [x], [P[x]]), 0) if P[x].any() else 0.0) for x in J}; Pn = list(P)
                for j in J:
                    o = [x for x in J if x != j and rt[x] > a.rt]; Pn[j] = cs.predict(j, [0] + o, [g0] + [P[x] for x in o], backward="pooled", k=a.k)
                P = Pn
                if r == a.rounds and len(J) > len(real):
                    s2 = acc.setdefault(f"N={N} share of distractors passing the round-trip filter", [0.0, 0.0]); s2[0] += sum(rt[x] > a.rt for x in J if x >= cs.n_real) / 100; s2[1] += len(J) - len(real)
                    s3 = acc.setdefault(f"N={N} share of real images passing the round-trip filter", [0.0, 0.0]); s3[0] += sum(rt[x] > a.rt for x in real) / 100; s3[1] += len(real)
                for j in T: add(f"N={N} ours round{r}", j, P[j])
        rows[cs.c] = {k: 100 * v[0] / max(v[1], 1.0) for k, v in acc.items()}; del cs; torch.cuda.empty_cache()
    res[d] = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))}; res[d + "/classes"] = rows
    print(f"{os.path.basename(d)}: {time.time() - t0:.0f}s", flush=True)
ds = a.dirs.split(","); names = list(res[ds[0]])
print(f"{'':44s}" + "".join(f"{os.path.basename(d)[:8]:>10s}" for d in ds) + f"{'mean':>10s}")
for k in names: print(f"{k:44s}" + "".join(f"{res[d][k]:10.1f}" for d in ds) + f"{np.mean([res[d][k] for d in ds]):10.1f}")
if a.out: json.dump({k: v for k, v in res.items()}, open(a.out, "w"))
