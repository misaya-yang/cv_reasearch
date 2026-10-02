"""More node-scoring rules + where the bidirectional rule goes wrong (CPU; tables from hier.py)."""
import sys, torch, numpy as np
T = [t for p in sys.argv[1:] for t in torch.load(p, weights_only=False)]
A = lambda t: t["area"]; iou = lambda t: t["fg"] / (t["area"] + t["G"] - t["fg"] + 1e-9)
P = lambda t: t["back"] / t["area"]; R = lambda t: t["fwd"] / max(t["n_reffg"], 1); Rb = lambda t: t["bfwd"] / max(t["n_refbg"], 1)
IoUr = lambda t: t["fwd"] / (t["n_reffg"] + t["bfwd"] + 1e-9)                      # push the node onto the reference, compare with the reference mask
IoUt = lambda t: t["back"] / (A(t) + t["back"][-1] - t["back"] + 1e-9)             # pull the reference mask onto the target, compare with the node
Pr = lambda t: t["fwd"] / (t["fwd"] + t["bfwd"] + 1e-9)                            # reference patches landing in the node: share that is foreground
F1 = lambda t: 2 * P(t) * R(t) / (P(t) + R(t) + 1e-9)
def run(name, score, verbose=True):
    d = {}; info = []
    for t in T:
        k = int(score(t).argmax()); i = float(t["fg"][k]); u = float(t["area"][k] + t["G"] - t["fg"][k]); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u
        info.append((t["area"][k] / max(t["G"], 1e-6), i / max(u, 1e-6), float(iou(t).max())))
    info = np.array(info); m = 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()])); r = info[:, 0]
    if verbose: print(f"  {name:52s} {m:6.2f} | mean IoU {info[:, 1].mean():.3f} | area ratio <0.5: {np.mean(r < 0.5) * 100:.0f}%  >2: {np.mean(r > 2) * 100:.0f}% | IoU<0.1: {np.mean(info[:, 1] < 0.1) * 100:.0f}%")
    return m, info
print(len(T), "episodes")
run("oracle best node", iou)
run("F1(P, R)", F1)
run("reference-side IoU (cycle)", IoUr)
run("target-side IoU", IoUt)
run("IoU_r * IoU_t", lambda t: IoUr(t) * IoUt(t))
run("IoU_r * P", lambda t: IoUr(t) * P(t))
run("harmonic(IoU_r, P)", lambda t: 2 * IoUr(t) * P(t) / (IoUr(t) + P(t) + 1e-9))
run("harmonic(P, R, P_ref)", lambda t: 3 / (1 / (P(t) + 1e-6) + 1 / (R(t) + 1e-6) + 1 / (Pr(t) + 1e-6)))
run("P * R * P_ref", lambda t: P(t) * R(t) * Pr(t))
run("F1 * cos", lambda t: F1(t) * np.clip(t["cos"].astype(np.float32), 0, None))
run("P * R * P_ref * cos", lambda t: P(t) * R(t) * Pr(t) * np.clip(t["cos"].astype(np.float32), 0, None))
run("geometric: (P R P_ref cos)^(1/4) with area >= 4", lambda t: np.where(A(t) >= 4, P(t) * R(t) * Pr(t) * np.clip(t["cos"].astype(np.float32), 0, None), 0))
# diagnosis of the F1 rule
m, info = run("F1(P, R)", F1, verbose=False); rank = []; top = {1: [], 3: [], 10: [], 30: []}
for t in T:
    s = F1(t); o = np.argsort(-s); io = iou(t); b = io.max()
    for k in top: top[k].append(io[o[:k]].max())
print("\nF1 rule: best IoU among its top-k nodes (mean):  " + "  ".join(f"top-{k}: {np.mean(v):.3f}" for k, v in top.items()) + f"   oracle {np.mean([iou(t).max() for t in T]):.3f}")
bad = info[:, 1] < 0.1; print(f"episodes where the pick has IoU < 0.1: {bad.mean() * 100:.0f}%; in those, mean oracle IoU {info[bad, 2].mean():.2f}, mean true foreground share {np.mean([t['G'] / 4096 for t, b in zip(T, bad) if b]) * 100:.1f}% (all episodes {np.mean([t['G'] / 4096 for t in T]) * 100:.1f}%)")
