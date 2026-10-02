"""Offline evaluation of region-scoring rules on the tables written by scripts/leaves.py (CPU only)."""
import numpy as np, torch

def load(paths):
    T = [t for p in paths for t in torch.load(p, weights_only=False)]
    for t in T:
        n = len(t["g"]); ch = t["children"].astype(np.int64); keys = ["area", "fg"] + list(t["ev"])
        X = np.zeros((len(keys), 2 * n - 1), np.float32); X[0, :n] = 1; X[1, :n] = t["g"]
        for j, k in enumerate(t["ev"]): X[j + 2, :n] = t["ev"][k]
        for i, (x, y) in enumerate(ch): X[:, n + i] = X[:, x] + X[:, y]
        t["N"] = {k: X[j] for j, k in enumerate(keys)}; t["G"] = float(X[1, -1]); t["n"] = n
        par = np.full(2 * n - 1, -1, np.int64); par[ch[:, 0]] = n + np.arange(n - 1); par[ch[:, 1]] = n + np.arange(n - 1); t["parent"] = par
    return T

def soft_iou(t, k): N = t["N"]; return float(N["fg"][k]), float(N["area"][k] + t["G"] - N["fg"][k])

def evaluate(T, pick):
    """pick(t) -> node index. Returns mIoU (class-wise sums, then mean) and per-episode (area ratio, IoU)."""
    d, info = {}, []
    for t in T:
        k = pick(t); i, u = soft_iou(t, k); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u; info.append((t["N"]["area"][k] / max(t["G"], 1e-6), i / max(u, 1e-6)))
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()])), np.array(info)

def baseline(T):
    d = {}
    for t in T:
        s = t["insid3"].astype(np.float32); g = t["g"].astype(np.float32); i = float((s * g).sum()); u = float(s.sum() + t["G"] - i); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()]))

def show(T, name, score):
    m, info = evaluate(T, lambda t: int(np.argmax(score(t)))); r = info[:, 0]
    print(f"  {name:64s} {m:6.2f} | mean IoU {info[:, 1].mean():.3f} | too small {np.mean(r < 0.5) * 100:3.0f}%  too big {np.mean(r > 2) * 100:3.0f}% | IoU<0.1 {np.mean(info[:, 1] < 0.1) * 100:3.0f}%")
    return m
