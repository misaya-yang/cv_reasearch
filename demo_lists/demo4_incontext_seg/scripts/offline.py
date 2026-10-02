"""Cluster-selection rules evaluated offline from the per-cluster tables written by budget.py (CPU only).

  python scripts/offline.py results/budget_coco_f0.tables.pt [more tables ...]
IoU here is computed at feature resolution with soft ground truth (cluster area x foreground share), so numbers are a little
higher than the 1024-pixel evaluation; compare rules with each other, not with the paper.
"""
import sys, torch, numpy as np
T = [t for p in sys.argv[1:] for t in torch.load(p)]
print(len(T), "episodes; mean clusters", np.mean([t["K"] for t in T]))

def miou(sel_fn):
    d = {}
    for t in T:
        a, p = t["area"].float(), t["pur"].float(); s = sel_fn(t).float(); fg = a * p
        i = float((s * fg).sum()); u = float((s * a).sum() + ((1 - s) * fg).sum()); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()]))

def auc(key_fn):
    """Area-weighted AUC of a per-cluster score for 'cluster is mostly foreground', pooled over episodes."""
    s = torch.cat([key_fn(t).float() for t in T]); y = torch.cat([(t["pur"].float() > 0.5) for t in T]); w = torch.cat([t["area"].float() for t in T])
    o = s.argsort(descending=True); y, w = y[o], w[o]; tp = (w * y).cumsum(0) / (w * y).sum(); fp = (w * ~y).cumsum(0) / (w * ~y).sum()
    return float(torch.trapz(tp, fp))

f = lambda k: (lambda t: t[k].float())
print("oracle: clusters by ground-truth majority   %.2f" % miou(lambda t: t["pur"].float() > 0.5))
print("INSID3 (combined > 0.2)                     %.2f" % miou(lambda t: t["combined"].float() > 0.2))
print("seed only                                   %.2f" % miou(lambda t: torch.arange(t["K"]) == t["seed"]))
scores = {"combined (INSID3)": f("combined"), "cross (sim to prototype)": f("cross"), "back (share of patches whose nearest reference patch is foreground)": f("back"),
          "fgmax": f("fgmax"), "fgmax - bgmax": lambda t: t["fgmax"].float() - t["bgmax"].float(), "fgmax_o - bgmax_o": lambda t: t["fgmax_o"].float() - t["bgmax_o"].float(),
          "cross * back": lambda t: t["cross"].float() * t["back"].float()}
print("\nscore: AUC | best global threshold -> mIoU")
for n, fn in scores.items():
    allv = torch.cat([fn(t) for t in T]); qs = torch.quantile(allv, torch.linspace(0.02, 0.98, 49)); best = max(((miou(lambda t, q=q: fn(t) > q)), float(q)) for q in qs)
    print(f"  {n:70s} {auc(fn):.3f} | thr {best[1]:+.3f} -> {best[0]:.2f}")
