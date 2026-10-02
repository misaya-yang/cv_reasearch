"""How is the F1 pick related to the oracle node (same branch, wrong level / different branch)?  And: seed patch + level choice.
  python scripts/rules3.py results/leaves_f0_300.leaves.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from icx.offline import *
T = load(sys.argv[1:]); e = 1e-9; f1 = lambda p, r: 2 * p * r / (p + r + e); A = lambda t: t["N"]["area"]; root = lambda t, k: t["N"][k][-1]
iou = lambda t: t["N"]["fg"] / (A(t) + t["G"] - t["N"]["fg"] + e); F = lambda t: f1(t["N"]["nn_back"] / A(t), t["N"]["nn_fwd"] / (root(t, "nn_fwd") + e))
def chain(t, v):
    out = []
    while v != -1: out.append(v); v = t["parent"][v]
    return out
rel = {"same": [], "pick is inside the oracle node": [], "pick contains the oracle node": [], "different branch": []}
for t in T:
    k = int(F(t).argmax()); b = int(iou(t).argmax()); io = iou(t)
    r = "same" if k == b else ("pick is inside the oracle node" if b in chain(t, k) else ("pick contains the oracle node" if k in chain(t, b) else "different branch"))
    rel[r].append((io[k], io[b]))
print("F1 pick vs oracle node: share | picked IoU | oracle IoU")
for r, v in rel.items():
    v = np.array(v); print(f"  {r:32s} {len(v) / len(T) * 100:5.1f}% | {v[:, 0].mean() if len(v) else 0:.2f} | {v[:, 1].mean() if len(v) else 0:.2f}")
# seed leaf + best level on its path (oracle level): how much does committing to a seed patch cost?
def path_oracle(seedfn, name):
    d = {}; hit = []
    for t in T:
        s = seedfn(t); ch = chain(t, s); io = iou(t)[ch]; k = ch[int(io.argmax())]; i, u = soft_iou(t, k); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u; hit.append(t["g"][s] > 0.5)
    print(f"  seed = {name:46s} seed patch is foreground {np.mean(hit) * 100:3.0f}% | oracle level on its path -> {100 * np.mean([i / max(u, 1e-6) for i, u in d.values()]):.2f}")
print("\nseed patch, then the best level on the path from that patch to the root (level chosen by the oracle)")
n = T[0]["n"]
path_oracle(lambda t: int(t["ev"]["sim"].argmax()), "max similarity to the reference prototype")
path_oracle(lambda t: int(t["ev"]["fgmax"].argmax()), "max similarity to any reference FG patch")
path_oracle(lambda t: int((t["ev"]["fgmax"].astype(np.float32) - t["ev"]["bgmax"].astype(np.float32)).argmax()), "max (FG - BG) similarity margin")
path_oracle(lambda t: int(t["ev"]["dF100"].argmax()), "max mutual FG mass (dual softmax, 100)")
path_oracle(lambda t: int(t["ev"]["dF30"].argmax()), "max mutual FG mass (dual softmax, 30)")
path_oracle(lambda t: int((t["ev"]["dF30"].astype(np.float32) - t["ev"]["dB30"].astype(np.float32)).argmax()), "max mutual FG - BG mass (30)")
path_oracle(lambda t: int(t["g"].argmax()), "a true foreground patch (oracle)")
# level choice on the path of a given seed with the F1 rule
def path_rule(seedfn, name, score):
    return show(T, name, lambda t: (lambda ch: (lambda out: (out.__setitem__(ch, score(t)[ch]), out)[1])(np.full(len(A(t)), -1.0)))(chain(t, seedfn(t))))
print("\nlevel chosen by a rule")
path_rule(lambda t: int(t["ev"]["dF30"].argmax()), "seed = max mutual FG mass (30), level by F1", F)
path_rule(lambda t: int(t["g"].argmax()), "seed = a true foreground patch (oracle), level by F1", F)
