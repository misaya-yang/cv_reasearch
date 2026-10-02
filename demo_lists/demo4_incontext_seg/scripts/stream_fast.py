"""Stream experiments on cached features (seconds per variant). See icx/fast.py and scripts/stream_cache.py.

Image 0 of every class is the labelled reference; images 1..N are unlabeled images of the same class and are the test images.
For every test image j the pool is all other unlabeled images. Per-class IoU = sum of intersections / sum of unions over the
test images of the class (few-shot segmentation convention); the last row is the mean over classes.
"""
import argparse, glob, os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/demo4_cache/stream/f0_s0_N16"); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--m", type=int, default=4)
a = ap.parse_args(); U = torch.load(f"{a.dir}/U.pt").to(DEV)
files = sorted(glob.glob(f"{a.dir}/class*.pt"), key=lambda p: int(os.path.basename(p)[5:-3]))
rows = {}; t0 = time.time()
COCO = "person bicycle car motorcycle airplane bus train truck boat trafficlight hydrant stopsign parkingmeter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sportsball kite bat glove skateboard surfboard racket bottle wineglass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hotdog pizza donut cake chair couch plant bed table toilet tv laptop mouse remote keyboard phone microwave oven toaster sink fridge book clock vase scissors teddy drier toothbrush".split()
for f in files:
    cs = ClassSet(f, U); n = cs.n; J = list(range(1, n)); g0 = cs.gt64[0]; acc = {}
    def add(name, j, m):
        i, u = cs.iu(m, j); s = acc.setdefault(name, [0.0, 0.0]); s[0] += i; s[1] += u
    P1 = [g0] + [cs.predict(j, [0], [g0]) for j in J]
    true = np.array([1.0] + [cs.iou64(P1[j], j) for j in J])
    rt = np.array([1.0] + [cs.iou64(cs.predict(0, [j], [P1[j]]), 0) if P1[j].any() else 0.0 for j in J])
    for j in J:
        o = np.array([k for k in J if k != j])
        add("1shot", j, P1[j])
        add("gt:4", j, cs.predict(j, [0] + list(o[:a.m]), [g0] + [cs.gt64[k] for k in o[:a.m]]))
        add("gt:all", j, cs.predict(j, [0] + list(o), [g0] + [cs.gt64[k] for k in o]))
        add("pseudo:all", j, cs.predict(j, [0] + list(o), [g0] + [P1[k] for k in o]))
        ks = [k for k in o if rt[k] > a.thr]; add("pseudo:rt", j, cs.predict(j, [0] + ks, [g0] + [P1[k] for k in ks]))
        ks = [k for k in o if true[k] > 0.5]; add("pseudo:oracle-filter", j, cs.predict(j, [0] + ks, [g0] + [P1[k] for k in ks]))
        for bk, kw in (("pooled1", dict(backward="pooled", k=1)), ("pooled3", dict(backward="pooled", k=3))):
            add(f"gt:4 {bk}", j, cs.predict(j, [0] + list(o[:a.m]), [g0] + [cs.gt64[k] for k in o[:a.m]], **kw))
            add(f"gt:all {bk}", j, cs.predict(j, [0] + list(o), [g0] + [cs.gt64[k] for k in o], **kw))
            add(f"pseudo:all {bk}", j, cs.predict(j, [0] + list(o), [g0] + [P1[k] for k in o], **kw))
            ks = [k for k in o if rt[k] > a.thr]; add(f"pseudo:rt {bk}", j, cs.predict(j, [0] + ks, [g0] + [P1[k] for k in ks], **kw))
    rows[cs.c] = {k: 100 * v[0] / max(v[1], 1.0) for k, v in acc.items()}
    rows[cs.c]["_corr"] = float(np.corrcoef(rt[1:], true[1:])[0, 1]) if rt[1:].std() > 0 else float("nan"); rows[cs.c]["_true"] = float(true[1:].mean())
    del cs; torch.cuda.empty_cache()
names = [k for k in next(iter(rows.values())) if not k.startswith("_")]
print(f"{len(files)} classes, {time.time() - t0:.0f}s;  columns: " + " | ".join(f"({i}) {k}" for i, k in enumerate(names)))
print(f"{'class':>16s}" + "".join(f"{'(' + str(i) + ')':>8s}" for i in range(len(names))) + "   corr(rt,true)  mean round-1 IoU")
for c, r in rows.items():
    print(f"{c:3d} {COCO[c]:>12s}" + "".join(f"{r[k]:8.1f}" for k in names) + f"   {r['_corr']:8.2f}   {r['_true']:8.2f}")
print(f"{'mean':>16s}" + "".join(f"{np.mean([r[k] for r in rows.values()]):8.1f}" for k in names))
