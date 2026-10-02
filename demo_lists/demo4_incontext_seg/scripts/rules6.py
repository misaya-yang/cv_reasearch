"""Decision rules on top of the bidirectional F1 score of tree nodes (CPU; leaves2/3 tables).
  mbr      minimum-Bayes-risk choice: the node with the highest F1-weighted mean IoU to the other high-scoring nodes
  plateau  the largest ancestor of the best node whose F1 is within a tolerance of the best
  python scripts/rules6.py results/l3_f0_300.l3.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.offline2 import *
T = load(sys.argv[1:]); print(len(T), "episodes;  INSID3 (same soft metric) %.2f" % baseline(T))
show(T, "oracle: best node", iou); show(T, "F1", F1)
def anc_matrix(t, idx):
    """A[a, b] = True if idx[a] is an ancestor of (or equal to) idx[b]."""
    par = t["parent"]; pos = {int(v): j for j, v in enumerate(idx)}; A = np.zeros((len(idx), len(idx)), bool)
    for j, v in enumerate(idx):
        u = int(v)
        while u != -1:
            if u in pos: A[pos[u], j] = True
            u = par[u]
    return A
def mbr(t, K=50, temp=0.05, minarea=1):
    s = np.where(area(t) >= minarea, F1(t), -1); idx = np.argsort(-s)[:K]; w = np.exp((s[idx] - s[idx].max()) / temp); a = area(t)[idx]; A = anc_matrix(t, idx)
    rel = A | A.T; I = np.where(rel, np.minimum(a[:, None], a[None]) / np.maximum(a[:, None], a[None]), 0.0)      # nested nodes: IoU = small / large; disjoint: 0
    return int(idx[np.argmax(I @ w)])
for K in (20, 50, 200):
    for temp in (0.02, 0.05, 0.1, 0.2): show(T, f"MBR over top-{K} nodes, temperature {temp}", pick=lambda t, K=K, temp=temp: mbr(t, K, temp))
def plateau(t, tol):
    s = F1(t); k = int(s.argmax()); best = k; u = t["parent"][k]
    while u != -1:
        if s[u] >= (1 - tol) * s[k]: best = u
        u = t["parent"][u]
    return int(best)
for tol in (0.02, 0.05, 0.1, 0.2): show(T, f"largest ancestor with F1 within {tol * 100:.0f}% of the best", pick=lambda t, tol=tol: plateau(t, tol))
