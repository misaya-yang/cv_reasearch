"""Combine INSID3's own output with the F1-selected tree node (CPU; l3b tables).  python scripts/rules10.py results/l3b_f*.l3.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.offline2 import *
def run(T):
    acc = {}
    def add(name, t, s):
        g = t["g"].astype(np.float32); i = float((s * g).sum()); u = float(s.sum() + t["G"] - i); x = acc.setdefault(name, {}).setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u
    for t in T:
        n = t["n"]; s0 = t["insid3"].astype(np.float32); k = int(F1(t).argmax()); s1 = np.zeros(n, np.float32); s1[leaves_of(t["ch"], n, k)] = 1
        m = t["r_mask"]; back = m[t["nn_t"].astype(np.int64)].astype(np.float32); nnr = t["nn_r"].astype(np.int64)
        def score(s):
            if s.sum() == 0: return 0.0
            P = float((s * back).sum() / s.sum()); R = float(s[nnr[m]].mean()) if m.any() else 0.0; return 2 * P * R / (P + R + 1e-9)
        add("INSID3", t, s0); add("F1 node", t, s1); add("union", t, np.maximum(s0, s1)); add("intersection", t, s0 * s1)
        add("whichever has the higher F1", t, s0 if score(s0) >= score(s1) else s1)
        u = np.maximum(s0, s1); cands = [s0, s1, u, s0 * s1]; add("best of {INSID3, node, union, intersection} by F1", t, cands[int(np.argmax([score(c) for c in cands]))])
        g = t["g"].astype(np.float32)
        def tiou(s): i = float((s * g).sum()); return i / max(float(s.sum() + t["G"] - i), 1e-6)
        add("oracle choice between INSID3 and node", t, s0 if tiou(s0) >= tiou(s1) else s1)
    return {k: 100 * float(np.mean([i / max(u, 1e-6) for i, u in v.values()])) for k, v in acc.items()}
R = [run(load([p])) for p in sys.argv[1:]]
for k in R[0]: print(f"  {k:52s} " + "  ".join(f"{r[k]:6.2f}" for r in R) + f"   mean {np.mean([r[k] for r in R]):6.2f}")
