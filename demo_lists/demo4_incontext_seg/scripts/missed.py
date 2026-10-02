"""Which foreground does INSID3 miss when its seed is right: other parts of the seed's object, or other instances?  (CPU, budget.py tables)"""
import sys, torch, numpy as np
from scipy import ndimage
T = [t for p in sys.argv[1:] for t in torch.load(p)]
tot = dict(same=0.0, other=0.0, kept_same=0.0, kept_other=0.0); ncomp = []; rows = []
def stat(t, ks):
    a = t["area"].float()[ks]; w = a / a.sum().clamp(min=1); return [float((t[x].float()[ks] * w).sum()) for x in ("cross", "intra", "back", "fgmax", "bgmax")] + [float(a.sum())]
S = {k: [] for k in ("same object, kept", "same object, missed", "other instance, kept", "other instance, missed")}
for t in T:
    if t["seed"] < 0 or float(t["pur"][t["seed"]]) <= 0.5: continue
    lab = t["lab"].long().numpy(); g = t["g"].float().numpy() / 255 > 0.5; K = t["K"]; fgk = t["pur"].float() > 0.5; sel = t["combined"].float() > 0.2
    cc, n = ndimage.label(g); ncomp.append(n)
    # component of each foreground cluster = the component holding most of its foreground patches; seed's component = "same object"
    comp = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        v = cc[(lab == k) & g]; comp[k] = int(np.bincount(v).argmax()) if len(v) else 0
    same = fgk & (comp == comp[t["seed"]]); other = fgk & ~same; same[t["seed"]] = False
    for name, m in (("same object, kept", same & sel), ("same object, missed", same & ~sel), ("other instance, kept", other & sel), ("other instance, missed", other & ~sel)):
        if m.any(): S[name].append(stat(t, m.nonzero()[:, 0]))
print(f"{len(ncomp)} episodes with a correct seed; foreground connected components per target: mean {np.mean(ncomp):.2f}, more than one in {np.mean(np.array(ncomp) > 1) * 100:.0f}%")
A = sum(sum(r[-1] for r in v) for v in S.values())
print("foreground clusters other than the seed (area-weighted)\n   group                     area share |  cross  intra  back   fgmax  bgmax")
for k, v in S.items():
    v = torch.tensor(v); w = v[:, -1] / v[:, -1].sum(); print(f"   {k:26s} {float(v[:, -1].sum() / A) * 100:5.1f}%    | " + "  ".join(f"{float((v[:, j] * w).sum()):.3f}" for j in range(5)))
