"""Tests on leaves2.py tables: (1) does the reference tell the granularity? (2) distribution matching, (3) cycle verification.
  python scripts/rules4.py results/l2_f0_300.l2.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.offline2 import *
T = load(sys.argv[1:]); print(len(T), "episodes;  INSID3 (same soft metric) %.2f" % baseline(T))
show(T, "oracle: best node", iou); show(T, "F1 (bidirectional, hard nearest neighbours)", F1)
# (1) granularity transfer
bt = [int(iou(t).argmax()) for t in T]; br = [int(t["r_iou"].argmax()) for t in T]
def sp(x, y): rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
tb = np.array([t["birth"][i] for t, i in zip(T, bt)]); td = np.array([t["death"][i] for t, i in zip(T, bt)]); rb = np.array([t["r_birth"][i] for t, i in zip(T, br)]); rd = np.array([t["r_death"][i] for t, i in zip(T, br)])
print(f"\n(1) reference mask in the reference tree: best node IoU {np.mean([t['r_iou'][i] for t, i in zip(T, br)]):.3f}; birth {rb.mean():.3f}, death {rd.mean():.3f};  target oracle node: birth {tb.mean():.3f}, death {td.mean():.3f}")
print(f"    rank correlation target vs reference:  birth {sp(tb, rb):+.2f}   death {sp(td, rd):+.2f}   mid-level {sp((tb + td) / 2, (rb + rd) / 2):+.2f}")
for t, j in zip(T, br): t["lvl"] = (t["r_birth"][j] + t["r_death"][j]) / 2; t["rb"], t["rd"] = t["r_birth"][j], t["r_death"][j]
show(T, "F1 among target nodes alive at the reference's mid level", lambda t: np.where((t["birth"] <= t["lvl"]) & (t["death"] > t["lvl"]), F1(t), -1))
show(T, "   oracle among those", lambda t: np.where((t["birth"] <= t["lvl"]) & (t["death"] > t["lvl"]), iou(t), -1))
show(T, "F1 among nodes whose life overlaps the reference node's life", lambda t: np.where((t["birth"] < t["rd"]) & (t["death"] > t["rb"]), F1(t), -1))
show(T, "   oracle among those", lambda t: np.where((t["birth"] < t["rd"]) & (t["death"] > t["rb"]), iou(t), -1))
# (2) distribution matching
print("\n(2) distribution distance between node patches and reference foreground patches (smaller = better)")
for k in ("mmd_lin_d", "mmd_lin_o", "mmd_rff10", "mmd_rff30"):
    show(T, f"argmin {k}", lambda t, k=k: -t[k].astype(np.float32))
    show(T, f"argmin {k}, area >= 8", lambda t, k=k: np.where(area(t) >= 8, -t[k].astype(np.float32), -9))
    show(T, f"F1 * exp(-{k} / median)", lambda t, k=k: F1(t) * np.exp(-t[k].astype(np.float32) / np.median(t[k].astype(np.float32))))
# (3) cycle verification of the top candidates
print("\n(3) swap roles: use the candidate as the prompt, re-predict the reference mask, score by IoU with the true reference mask")
for K in (5, 10, 20):
    def pick(t, K=K, mode="cycle"):
        s = F1(t); top = np.argsort(-s)[:K]; cyc = np.array([reverse_best(t, k)[0] for k in top])
        return int(top[np.argmax(cyc if mode == "cycle" else cyc * s[top])])
    show(T, f"top-{K} by F1, re-ranked by cycle IoU", pick=pick)
    show(T, f"top-{K} by F1, re-ranked by F1 * cycle IoU", pick=lambda t, K=K: pick(t, K, "prod"))
    show(T, f"   oracle among the top-{K}", pick=lambda t, K=K: int(np.argsort(-F1(t))[:K][np.argmax(iou(t)[np.argsort(-F1(t))[:K]])]))
