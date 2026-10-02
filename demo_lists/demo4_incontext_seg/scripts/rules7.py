"""F1 node selection with evidence from different layers.  python scripts/rules7.py results/layers_f0_300.layers.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from icx.offline2 import accumulate, f1
T = torch.load(sys.argv[1], weights_only=False); names = sorted({k.split("/")[0] for k in T[0]["ev"]})
for t in T:
    n = len(t["g"]); ch = t["children"].astype(np.int64); keys = list(t["ev"]); X = accumulate(ch, np.stack([np.ones(n, np.float32), t["g"].astype(np.float32)] + [t["ev"][k].astype(np.float32) for k in keys]))
    t["area"], t["fg"] = X[0], X[1]; t["G"] = float(X[1, -1]); t["N"] = {k: X[j + 2] for j, k in enumerate(keys)}
    t["F"] = {v: f1(t["N"][v + "/back"] / t["area"], t["N"][v + "/fwd"] / max(t["n_reffg"], 1)) for v in names}; t["iou"] = t["fg"] / (t["area"] + t["G"] - t["fg"] + 1e-9)
def ev(pick):
    d = {}; io = []
    for t in T:
        k = pick(t); i = float(t["fg"][k]); u = float(t["area"][k] + t["G"] - i); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u; io.append(i / max(u, 1e-6))
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()])), np.array(io)
print(len(T), "episodes; oracle best node %.2f" % ev(lambda t: int(t["iou"].argmax()))[0])
per = {}
for v in names:
    m, io = ev(lambda t, v=v: int(t["F"][v].argmax())); per[v] = io; print(f"  F1 with {v:8s} {m:6.2f} | mean IoU {io.mean():.3f} | IoU<0.1 {np.mean(io < 0.1) * 100:3.0f}%")
ls = [v for v in names if v.startswith("L")]
print("  oracle choice of layer per episode (mean IoU): %.3f  vs best single %.3f" % (np.max(np.stack([per[v] for v in ls]), 0).mean(), max(per[v].mean() for v in ls)))
for grp, nm in (([v for v in ls if v.endswith("deb")], "sum of F1 over debiased layers"), ([v for v in ls if v.endswith("raw")], "sum of F1 over raw layers"), (ls, "sum of F1 over all")):
    m, io = ev(lambda t, grp=grp: int(sum(t["F"][v] for v in grp).argmax())); print(f"  {nm:34s} {m:6.2f} | mean IoU {io.mean():.3f} | IoU<0.1 {np.mean(io < 0.1) * 100:3.0f}%")
    m, io = ev(lambda t, grp=grp: int(np.prod([t["F"][v] + 1e-3 for v in grp], 0).argmax())); print(f"  product instead of sum             {m:6.2f} | mean IoU {io.mean():.3f}")
