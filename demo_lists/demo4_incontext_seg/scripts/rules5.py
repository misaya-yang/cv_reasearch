"""The reference tree as labelled regions: does a transferred IoU estimate pick the right target node?
  python scripts/rules5.py results/l3_f0_300.l3.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.offline2 import *
T = load(sys.argv[1:]); print(len(T), "episodes;  INSID3 (same soft metric) %.2f" % baseline(T))
g = lambda k: (lambda t: t[k].astype(np.float32))
show(T, "oracle: best node", iou); show(T, "F1 (bidirectional, hard nearest neighbours)", F1)
for k in ("tr1", "tr5", "krr10", "krr30"):
    show(T, f"{k}: transferred IoU", g(k)); show(T, f"{k}, area >= 4", lambda t, k=k: np.where(area(t) >= 4, g(k)(t), -9))
    show(T, f"{k} * F1", lambda t, k=k: np.clip(g(k)(t), 0, None) * F1(t)); show(T, f"sqrt({k}) * F1", lambda t, k=k: np.sqrt(np.clip(g(k)(t), 0, None)) * F1(t))
    show(T, f"{k} * similarity to its reference node", lambda t, k=k: np.clip(g(k)(t), 0, None) * np.clip(g("trs")(t), 0, None))
# how good is the estimate as a predictor of the true IoU of a node?
def sp(x, y): rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
for k in ("tr1", "tr5", "krr10", "krr30"):
    cs = [sp(g(k)(t)[area(t) >= 4], iou(t)[area(t) >= 4]) for t in T]; print(f"  rank correlation between {k} and the true IoU over nodes (area >= 4), per episode: mean {np.mean(cs):+.2f}")
cs = [sp(F1(t)[area(t) >= 4], iou(t)[area(t) >= 4]) for t in T]; print(f"  same for F1: mean {np.mean(cs):+.2f}")
b = [int(iou(t).argmax()) for t in T]; print("at the oracle node: tr1 %.2f  tr5 %.2f  krr10 %.2f  sim %.2f | true IoU %.2f | best reference node IoU %.2f" % (*[np.mean([g(k)(t)[i] for t, i in zip(T, b)]) for k in ("tr1", "tr5", "krr10", "trs")], np.mean([iou(t)[i] for t, i in zip(T, b)]), np.mean([t["r_iou_max"] for t in T])))
