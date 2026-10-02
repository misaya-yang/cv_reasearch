"""What goes wrong in INSID3's cluster selection?  (CPU only, from budget.py tables)  python scripts/failures.py results/budget_coco_f0.tables.pt"""
import sys, torch, numpy as np
T = [t for p in sys.argv[1:] for t in torch.load(p)]
def ep_iou(t, s):
    a, p = t["area"].float(), t["pur"].float(); fg = a * p; s = s.float(); i = float((s * fg).sum()); u = float((s * a).sum() + ((1 - s) * fg).sum()); return i, u
def miou(sel_fn, keep=None):
    d = {}
    for n, t in enumerate(T):
        i, u = ep_iou(t, sel_fn(n, t)); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()]))
base = lambda n, t: t["combined"].float() > 0.2; orac = lambda n, t: t["pur"].float() > 0.5
print(f"{len(T)} episodes   INSID3 {miou(base):.2f}   oracle selection {miou(orac):.2f}")
# episode categories
cat = []
for t in T:
    s = base(0, t); p = t["pur"].float(); a = t["area"].float(); i, u = ep_iou(t, s); iou = i / max(u, 1e-6); sp = float(p[t["seed"]]) if t["seed"] >= 0 else -1
    fg = float((a * p).sum() / a.sum()); o_i, o_u = ep_iou(t, p > 0.5)
    if s.sum() == 0: c = "empty"
    elif sp < 0.1: c = "seed on background (purity < 0.1)"
    elif sp < 0.5: c = "seed impure (0.1-0.5)"
    else:
        fn = float((a * p * (~s) * (p > 0.5)).sum()); fp = float((a * (1 - p) * s * (p <= 0.5)).sum())
        c = "seed right, IoU >= 0.7" if iou >= 0.7 else ("seed right, mostly misses foreground" if fn >= fp else "seed right, mostly adds background")
    cat.append((c, iou, fg, o_i / max(o_u, 1e-6)))
print("\nepisode category: share | mean IoU | mean oracle IoU | mean foreground share | mIoU if only these episodes were replaced by the oracle selection")
for c in sorted(set(x[0] for x in cat)):
    idx = [n for n, x in enumerate(cat) if x[0] == c]; S = set(idx)
    print(f"  {c:44s} {len(idx) / len(T) * 100:5.1f}% | {np.mean([cat[n][1] for n in idx]):.2f} | {np.mean([cat[n][3] for n in idx]):.2f} | {np.mean([cat[n][2] for n in idx]) * 100:5.1f}% | {miou(lambda n, t: orac(n, t) if n in S else base(n, t)):.2f}")
# by foreground size
print("\nby foreground share of the target image: episodes | INSID3 mean IoU | oracle mean IoU | seed right")
for lo, hi in ((0, .01), (.01, .03), (.03, .1), (.1, .3), (.3, 1.01)):
    idx = [n for n, x in enumerate(cat) if lo <= x[2] < hi]
    if idx: print(f"  {lo * 100:4.0f}%-{hi * 100:4.0f}%  {len(idx):4d} | {np.mean([cat[n][1] for n in idx]):.2f} | {np.mean([cat[n][3] for n in idx]):.2f} | {np.mean([float(T[n]['pur'][T[n]['seed']]) > 0.5 if T[n]['seed'] >= 0 else 0 for n in idx]):.2f}")
# cluster level: what do missed foreground clusters look like?
rowsk = {"FG kept": [], "FG missed": [], "BG kept": [], "BG dropped": []}
for t in T:
    if t["seed"] < 0 or float(t["pur"][t["seed"]]) <= 0.5: continue          # only episodes with a correct seed
    s = base(0, t); p = t["pur"].float() > 0.5
    for k in range(t["K"]):
        if k == t["seed"]: continue
        key = ("FG " if p[k] else "BG ") + (("kept" if s[k] else "missed") if p[k] else ("kept" if s[k] else "dropped"))
        rowsk[key].append([float(t[x][k]) for x in ("area", "cross", "intra", "aw", "combined", "back", "fgmax", "bgmax")])
print("\nclusters other than the seed, in episodes with a correct seed (area-weighted means)\n   group        n   area share |  cross  intra  aw     combined  back   fgmax  bgmax")
tot = sum(sum(r[0] for r in v) for v in rowsk.values())
for k, v in rowsk.items():
    v = torch.tensor(v); w = v[:, 0] / v[:, 0].sum()
    print(f"   {k:10s} {len(v):6d}  {float(v[:, 0].sum() / tot) * 100:5.1f}%     | " + "  ".join(f"{float((v[:, j] * w).sum()):.3f}" for j in range(1, 8)))
