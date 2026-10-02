"""Re-rank the top candidates with second-look similarities.  python scripts/rules9.py results/look_f0_300.look.pt"""
import sys, numpy as np, torch
T = torch.load(sys.argv[1], weights_only=False)
def ev(pick, K=None):
    d = {}; io = []
    for t in T:
        j = pick(t); i = float(t["fg"][j]); u = float(t["area"][j] + t["G"] - i); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u; io.append(i / max(u, 1e-6))
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()])), np.array(io)
def sp(x, y): rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else 0.0
K = len(T[0]["top"]); print(len(T), "episodes, candidates per episode:", K)
print("  F1 (first candidate)            %.2f | mean IoU %.3f" % (lambda r: (r[0], r[1].mean()))(ev(lambda t: 0)))
print("  oracle among the candidates     %.2f | mean IoU %.3f" % (lambda r: (r[0], r[1].mean()))(ev(lambda t: int(t["iou"].argmax()))))
print("  rank correlation with true IoU among candidates: F1 %+.2f" % np.mean([sp(t["F1"], t["iou"]) for t in T]))
for k in [x for x in T[0] if x.split("_")[0] in ("plain", "grey")]:
    a_ = ev(lambda t, k=k: int(t[k].argmax())); b_ = ev(lambda t, k=k: int((t[k] * t["F1"]).argmax())); c_ = ev(lambda t, k=k: int((np.clip(t[k], 0, None) ** 4 * t["F1"]).argmax()))
    print(f"  {k:12s} alone {a_[0]:6.2f} ({a_[1].mean():.3f}) | x F1 {b_[0]:6.2f} ({b_[1].mean():.3f}) | ^4 x F1 {c_[0]:6.2f} ({c_[1].mean():.3f}) | rank corr {np.mean([sp(t[k], t['iou']) for t in T]):+.2f}")
