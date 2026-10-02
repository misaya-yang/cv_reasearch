"""Does the tree itself say which nodes are objects?  Persistence of a node = merge height of its parent - its own merge height.
  python scripts/rules2.py results/leaves_f0_300.leaves.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from icx.offline import *
T = load(sys.argv[1:]); print(len(T), "episodes;  INSID3 (same soft metric) %.2f" % baseline(T))
e = 1e-9; f1 = lambda p, r: 2 * p * r / (p + r + e); A = lambda t: t["N"]["area"]; root = lambda t, k: t["N"][k][-1]
for t in T:
    n = t["n"]; h = np.concatenate([np.zeros(n), t["dist"].astype(np.float32)]); par = t["parent"]; death = np.where(par >= 0, h[np.maximum(par, 0)], 1.0)
    t["birth"], t["death"], t["pers"] = h, death, death - h
iou = lambda t: t["N"]["fg"] / (A(t) + t["G"] - t["N"]["fg"] + e)
F = lambda t: f1(t["N"]["nn_back"] / A(t), t["N"]["nn_fwd"] / (root(t, "nn_fwd") + e))
# 1. where does the oracle node sit?
b = [int(iou(t).argmax()) for t in T]
print("oracle node: birth %.3f  death %.3f  persistence %.3f   (INSID3 cuts at height 0.4)" % tuple(np.mean([t[k][i] for t, i in zip(T, b)]) for k in ("birth", "death", "pers")))
print("  persistence rank of the oracle node among nodes with area >= 4 (median): %.0f of %.0f" % (np.median([(t["pers"][A(t) >= 4] > t["pers"][i]).sum() + 1 for t, i in zip(T, b)]), np.median([(A(t) >= 4).sum() for t in T])))
print("  share of episodes where birth <= 0.4 < death (the node exists as one flat cluster at INSID3's cut): %.0f%%" % (100 * np.mean([t["birth"][i] <= 0.4 < t["death"][i] for t, i in zip(T, b)])))
print("  oracle node born above the cut (INSID3 over-segments it): %.0f%%;  dead below the cut (INSID3 has merged it with something else): %.0f%%" % (100 * np.mean([t["birth"][i] > 0.4 for t, i in zip(T, b)]), 100 * np.mean([t["death"][i] <= 0.4 for t, i in zip(T, b)])))
show(T, "oracle: best node", iou)
show(T, "F1 (hard)", F)
for gam in (0.25, 0.5, 1.0): show(T, f"F1 * persistence^{gam}", lambda t, gam=gam: F(t) * np.clip(t["pers"], 0, None) ** gam)
for m in (10, 20, 50):
    def top(t, m=m):
        s = np.where(A(t) >= 4, t["pers"], -1); keep = np.argsort(-s)[:m]; out = np.full(len(s), -1.0); out[keep] = F(t)[keep]; return out
    show(T, f"best F1 among the {m} most persistent nodes (area >= 4)", top)
    def topo(t, m=m):
        s = np.where(A(t) >= 4, t["pers"], -1); keep = np.argsort(-s)[:m]; out = np.full(len(s), -1.0); out[keep] = iou(t)[keep]; return out
    show(T, f"   oracle among those {m}", topo)
for lo in (0.3, 0.4, 0.5):
    show(T, f"F1 among nodes alive at height {lo} (flat cut)", lambda t, lo=lo: np.where((t["birth"] <= lo) & (t["death"] > lo), F(t), -1))
    show(T, f"   oracle among those", lambda t, lo=lo: np.where((t["birth"] <= lo) & (t["death"] > lo), iou(t), -1))
