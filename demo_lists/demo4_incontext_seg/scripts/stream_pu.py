"""Stream sweep: treat pseudo masks as positive-unlabeled (their complement is not trusted as background).
  majority need=x : INSID3's vote, but a patch needs only x foreground votes instead of half of the references
  pu delta=d      : nearest foreground patch over all references vs nearest background patch; background patches of pseudo
                    references are discounted by d in similarity (d=0: plain pooled nearest neighbour; large d: only the labelled
                    reference supplies background evidence)
Each variant is iterated for a few rounds (round r uses the masks of round r-1 for all other images)."""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/demo4_cache/stream/f0_s0_N16"); ap.add_argument("--show", default=""); ap.add_argument("--rounds", type=int, default=4)
a = ap.parse_args(); U = torch.load(f"{a.dir}/U.pt").to(DEV)
files = sorted(glob.glob(f"{a.dir}/class*.pt"), key=lambda p: int(os.path.basename(p)[5:-3])); rows = {}; t0 = time.time()
VAR = {"majority need=half": dict(), "majority need=2": dict(need=2), "majority need=3": dict(need=3), "majority need=4": dict(need=4),
       "pu delta=0": dict(backward="pu", delta=0.0), "pu delta=0.05": dict(backward="pu", delta=0.05), "pu delta=0.1": dict(backward="pu", delta=0.1),
       "pu delta=0.2": dict(backward="pu", delta=0.2), "pu delta=9": dict(backward="pu", delta=9.0)}
for f in files:
    cs = ClassSet(f, U); n = cs.n; J = list(range(1, n)); g0 = cs.gt64[0]; acc = {}
    def add(name, j, m):
        i, u = cs.iu(m, j); s = acc.setdefault(name, [0.0, 0.0]); s[0] += i; s[1] += u
    P1 = [g0] + [cs.predict(j, [0], [g0]) for j in J]
    for j in J: add("1shot", j, P1[j])
    for name, kw in VAR.items():
        for j in J:
            o = [x for x in J if x != j]; add(f"{name} | true masks", j, cs.predict(j, [0] + o, [g0] + [cs.gt64[x] for x in o], **kw))
        P = P1
        for r in range(2, a.rounds + 1):
            Pn = [g0]
            for j in J:
                o = [x for x in J if x != j]; Pn.append(cs.predict(j, [0] + o, [g0] + [P[x] for x in o], **kw))
            P = Pn
            for j in J: add(f"{name} | round{r}", j, P[j])
    rows[cs.c] = {k: 100 * v[0] / max(v[1], 1.0) for k, v in acc.items()}; del cs; torch.cuda.empty_cache()
print(f"{len(files)} classes, {time.time() - t0:.0f}s;   mean IoU: with true masks of the other images | rounds 2..{a.rounds} with pseudo masks | classes better/worse than 1shot by >2 (last round)")
print(f"   {'1shot':22s} {np.mean([r['1shot'] for r in rows.values()]):6.1f}")
for name in VAR:
    last = f"{name} | round{a.rounds}"
    print(f"   {name:22s} {np.mean([r[name + ' | true masks'] for r in rows.values()]):6.1f} | " + " ".join(f"{np.mean([r[f'{name} | round{k}'] for r in rows.values()]):6.1f}" for k in range(2, a.rounds + 1))
          + f" | {sum(r[last] > r['1shot'] + 2 for r in rows.values())}/{sum(r[last] < r['1shot'] - 2 for r in rows.values())}")
for k in [x for x in a.show.split(";") if x]:
    print(k, " ".join(f"{c}:{r[k]:.0f}" for c, r in rows.items()))
