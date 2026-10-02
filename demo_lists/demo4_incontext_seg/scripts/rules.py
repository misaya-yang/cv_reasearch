"""First batch of node-scoring rules on leaves.py tables.  python scripts/rules.py results/leaves_f0_300.leaves.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from icx.offline import *
T = load(sys.argv[1:]); print(len(T), "episodes;  INSID3 (same soft metric) %.2f" % baseline(T))
e = 1e-9; f1 = lambda p, r: 2 * p * r / (p + r + e); A = lambda t: t["N"]["area"]; root = lambda t, k: t["N"][k][-1]
show(T, "oracle: best node", lambda t: t["N"]["fg"] / (A(t) + t["G"] - t["N"]["fg"] + e))
print(" hard nearest neighbours")
for s in ("", "_o"):
    show(T, f"F1(P = back share, R = share of ref-FG landing in node){s}", lambda t: f1(t["N"]["nn_back" + s] / A(t), t["N"]["nn_fwd" + s] / (root(t, "nn_fwd" + s) + e)))
for k in ("30", "100"):
    print(f" soft, kappa = {k}")
    q = lambda t: t["N"]["q" + k]; mF = lambda t: t["N"]["mF" + k]; mB = lambda t: t["N"]["mB" + k]; dF = lambda t: t["N"]["dF" + k]; dB = lambda t: t["N"]["dB" + k]; oF = lambda t: t["N"]["oF" + k]
    show(T, "F1(P = mean q, R = share of ref-FG mass in node)", lambda t: f1(q(t) / A(t), mF(t) / (root(t, "mF" + k) + e)))
    show(T, "expected IoU from q:  Q / (area + Q_all - Q)", lambda t: q(t) / (A(t) + root(t, "q" + k) - q(t) + e))
    show(T, "mutual: F1(P = dF / (dF + dB), R = dF share)", lambda t: f1(dF(t) / (dF(t) + dB(t) + e), dF(t) / (root(t, "dF" + k) + e)))
    show(T, "mutual: F1(P = mean q, R = dF share)", lambda t: f1(q(t) / A(t), dF(t) / (root(t, "dF" + k) + e)))
    show(T, "OT: expected IoU  oF / (area + oF_all - oF)", lambda t: oF(t) / (A(t) + root(t, "oF" + k) - oF(t) + e))
    show(T, "OT: F1(P = oF / area, R = oF share)", lambda t: f1(oF(t) / A(t), oF(t) / (root(t, "oF" + k) + e)))
    show(T, "F1(P = hard back share, R = OT share)", lambda t: f1(t["N"]["nn_back"] / A(t), oF(t) / (root(t, "oF" + k) + e)))
    show(T, "F1(P = OT, R = hard fwd share)", lambda t: f1(oF(t) / A(t), t["N"]["nn_fwd"] / (root(t, "nn_fwd") + e)))
