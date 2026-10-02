"""Per class: what is wrong with the round-1 pseudo masks, and which repair of them would matter for the pooled rule?
Columns: 1shot | pseudo (pooled k=3, round 2) | same with false positives removed from the pseudo masks | with misses filled | true masks
then precision and recall of the round-1 masks (patch level, pooled over the unlabeled images)."""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/demo4_cache/stream/f0_s0_N16"); a = ap.parse_args(); U = torch.load(f"{a.dir}/U.pt").to(DEV)
files = sorted(glob.glob(f"{a.dir}/class*.pt"), key=lambda p: int(os.path.basename(p)[5:-3])); kw = dict(backward="pooled", k=3); rows = []
for f in files:
    cs = ClassSet(f, U); n = cs.n; J = list(range(1, n)); g0 = cs.gt64[0]; acc = {}
    def add(name, j, m):
        i, u = cs.iu(m, j); s = acc.setdefault(name, [0.0, 0.0]); s[0] += i; s[1] += u
    P1 = [g0] + [cs.predict(j, [0], [g0]) for j in J]
    tp = sum(float((P1[j] & cs.gt64[j]).sum()) for j in J); pp = sum(float(P1[j].sum()) for j in J); gp = sum(float(cs.gt64[j].sum()) for j in J)
    for j in J:
        o = [x for x in J if x != j]; add("1shot", j, P1[j])
        add("pseudo", j, cs.predict(j, [0] + o, [g0] + [P1[x] for x in o], **kw))
        add("noFP", j, cs.predict(j, [0] + o, [g0] + [P1[x] & cs.gt64[x] for x in o], **kw))
        add("noFN", j, cs.predict(j, [0] + o, [g0] + [P1[x] | cs.gt64[x] for x in o], **kw))
        add("true", j, cs.predict(j, [0] + o, [g0] + [cs.gt64[x] for x in o], **kw))
    r = {k: 100 * v[0] / max(v[1], 1.0) for k, v in acc.items()}; r.update(c=cs.c, prec=tp / max(pp, 1), rec=tp / max(gp, 1), fgshare=gp / (len(J) * cs.P), ninst=0); rows.append(r); del cs; torch.cuda.empty_cache()
print("class   1shot  pseudo   noFP   noFN   true | round-1 precision recall | fg share of image")
for r in rows: print(f"{r['c']:4d}  {r['1shot']:6.1f} {r['pseudo']:6.1f} {r['noFP']:6.1f} {r['noFN']:6.1f} {r['true']:6.1f} |   {r['prec']:.2f}   {r['rec']:.2f} | {r['fgshare']:.3f}")
print("mean  " + " ".join(f"{np.mean([r[k] for r in rows]):6.1f}" for k in ["1shot", "pseudo", "noFP", "noFN", "true"]) + f" |   {np.mean([r['prec'] for r in rows]):.2f}   {np.mean([r['rec'] for r in rows]):.2f}")
