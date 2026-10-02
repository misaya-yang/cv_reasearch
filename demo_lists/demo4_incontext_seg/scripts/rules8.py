"""F1 node selection under different evidence variants (tables from ev2.py).  python scripts/rules8.py results/ev2_f0_300.ev2.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from icx.offline2 import accumulate, f1
T = torch.load(sys.argv[1], weights_only=False); names = [k[:-5] for k in T[0]["ev"] if k.endswith("/back")]
for t in T:
    n = len(t["g"]); ch = t["children"].astype(np.int64); keys = [k for k in t["ev"] if not k.endswith("/nfg")]
    X = accumulate(ch, np.stack([np.ones(n, np.float32), t["g"].astype(np.float32)] + [t["ev"][k].astype(np.float32) for k in keys])); t["area"], t["fg"] = X[0], X[1]; t["G"] = float(X[1, -1]); N = {k: X[j + 2] for j, k in enumerate(keys)}
    t["F"] = {v: f1(N[v + "/back"] / t["area"], N[v + "/fwd"] / max(t["ev"][v + "/nfg"], 1)) for v in names}; t["iou"] = t["fg"] / (t["area"] + t["G"] - t["fg"] + 1e-9)
    b = int(t["iou"].argmax()); t["PR"] = {v: (N[v + "/back"][b] / t["area"][b], N[v + "/fwd"][b] / max(t["ev"][v + "/nfg"], 1)) for v in names}
def ev(pick):
    d = {}; io = []
    for t in T:
        k = pick(t); i = float(t["fg"][k]); u = float(t["area"][k] + t["G"] - i); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u; io.append(i / max(u, 1e-6))
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()])), np.array(io)
print(len(T), "episodes; oracle best node %.2f" % ev(lambda t: int(t["iou"].argmax()))[0]); print("  variant            mIoU | mean IoU | IoU<0.1 | precision, recall at the oracle node")
for v in names:
    m, io = ev(lambda t, v=v: int(t["F"][v].argmax())); pr = np.array([t["PR"][v] for t in T]).mean(0); print(f"  {v:16s} {m:6.2f} | {io.mean():.3f} | {np.mean(io < 0.1) * 100:3.0f}% | {pr[0]:.2f}, {pr[1]:.2f}")
