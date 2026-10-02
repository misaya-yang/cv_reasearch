"""Offline tools for the tables written by scripts/leaves2.py: both trees and nearest-neighbour indices are available, so any
hard-nearest-neighbour statistic of any region can be computed here (CPU only)."""
import numpy as np, torch

def accumulate(children, leaf):
    """leaf: (M, n) per-leaf values -> (M, 2n-1) per-node sums over the subtree."""
    M, n = leaf.shape; X = np.zeros((M, 2 * n - 1), np.float32); X[:, :n] = leaf
    for i, (x, y) in enumerate(children): X[:, n + i] = X[:, x] + X[:, y]
    return X

def parents(children, n):
    par = np.full(2 * n - 1, -1, np.int64); par[children[:, 0]] = n + np.arange(n - 1); par[children[:, 1]] = n + np.arange(n - 1); return par

def leaves_of(children, n, node):
    out, st = [], [int(node)]
    while st:
        v = st.pop()
        if v < n: out.append(v)
        else: st.extend(children[v - n])
    return np.array(out)

def load(paths):
    T = [t for p in paths for t in torch.load(p, weights_only=False)]
    for t in T:
        n = len(t["g"]); t["n"] = n; ch = t["children"].astype(np.int64); rch = t["r_children"].astype(np.int64); t["ch"], t["rch"] = ch, rch
        m = t["r_mask"]; nn_t, nn_r = t["nn_t"].astype(np.int64), t["nn_r"].astype(np.int64)
        back = m[nn_t].astype(np.float32); fwd = np.bincount(nn_r[m], minlength=n).astype(np.float32); bfwd = np.bincount(nn_r[~m], minlength=n).astype(np.float32)
        X = accumulate(ch, np.stack([np.ones(n, np.float32), t["g"].astype(np.float32), back, fwd, bfwd, t["sim"].astype(np.float32)]))
        t["N"] = dict(area=X[0], fg=X[1], back=X[2], fwd=X[3], bfwd=X[4], sim=X[5]); t["G"] = float(X[1, -1]); t["n_reffg"] = int(m.sum()); t["n_refbg"] = int((~m).sum())
        t["parent"] = parents(ch, n); t["r_parent"] = parents(rch, n)
        h = np.concatenate([np.zeros(n), t["dist"].astype(np.float32)]); t["birth"] = h; t["death"] = np.where(t["parent"] >= 0, h[np.maximum(t["parent"], 0)], 1.0)
        R = accumulate(rch, np.stack([np.ones(n, np.float32), t["r_g"].astype(np.float32)])); t["RN"] = dict(area=R[0], fg=R[1]); t["RG"] = float(R[1, -1])
        rh = np.concatenate([np.zeros(n), t["r_dist"].astype(np.float32)]); t["r_birth"] = rh; t["r_death"] = np.where(t["r_parent"] >= 0, rh[np.maximum(t["r_parent"], 0)], 1.0)
        t["r_iou"] = R[1] / (R[0] + t["RG"] - R[1] + 1e-9)
    return T

e_ = 1e-9
f1 = lambda p, r: 2 * p * r / (p + r + e_)
area = lambda t: t["N"]["area"]
iou = lambda t: t["N"]["fg"] / (t["N"]["area"] + t["G"] - t["N"]["fg"] + e_)
F1 = lambda t: f1(t["N"]["back"] / t["N"]["area"], t["N"]["fwd"] / max(t["n_reffg"], 1))

def evaluate(T, pick):
    d, info = {}, []
    for t in T:
        k = pick(t); i = float(t["N"]["fg"][k]); u = float(t["N"]["area"][k] + t["G"] - i); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u; info.append((t["N"]["area"][k] / max(t["G"], 1e-6), i / max(u, 1e-6)))
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()])), np.array(info)

def show(T, name, score=None, pick=None):
    m, info = evaluate(T, pick or (lambda t: int(np.argmax(score(t))))); r = info[:, 0]
    print(f"  {name:66s} {m:6.2f} | mean IoU {info[:, 1].mean():.3f} | too small {np.mean(r < 0.5) * 100:3.0f}%  too big {np.mean(r > 2) * 100:3.0f}% | IoU<0.1 {np.mean(info[:, 1] < 0.1) * 100:3.0f}%")
    return m

def baseline(T):
    d = {}
    for t in T:
        s = t["insid3"].astype(np.float32); g = t["g"].astype(np.float32); i = float((s * g).sum()); u = float(s.sum() + t["G"] - i); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()]))

def reverse_best(t, k):
    """Swap roles: target node k is the prompt. Returns (cycle IoU with the true reference mask, index of the chosen reference node).
    Reference nodes are scored with the same bidirectional F1."""
    n = t["n"]; L = leaves_of(t["ch"], n, k); inN = np.zeros(n, bool); inN[L] = True
    a = inN[t["nn_r"].astype(np.int64)].astype(np.float32)                       # reference patch whose nearest target patch lies in the node
    c = np.bincount(t["nn_t"].astype(np.int64)[L], minlength=n).astype(np.float32)   # how many node patches choose this reference patch
    X = accumulate(t["rch"], np.stack([a, c])); s = f1(X[0] / t["RN"]["area"], X[1] / max(len(L), 1)); j = int(s.argmax())
    return float(t["r_iou"][j]), j
