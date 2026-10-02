"""Where does the bidirectional F1 node rule fail?  By target foreground size and by reference foreground size. (CPU; hier.py tables)"""
import sys, torch, numpy as np
T = [t for p in sys.argv[1:] for t in torch.load(p, weights_only=False)]
A = lambda t: t["area"]; iou = lambda t: t["fg"] / (t["area"] + t["G"] - t["fg"] + 1e-9)
P = lambda t: t["back"] / t["area"]; R = lambda t: t["fwd"] / max(t["n_reffg"], 1); F1 = lambda t: 2 * P(t) * R(t) / (P(t) + R(t) + 1e-9)
rows = []
for t in T:
    k = int(F1(t).argmax()); io = iou(t); n = (len(t["area"]) + 1) // 2; b = int(io.argmax())
    rows.append((t["G"] / n, t["n_reffg"] / n, io[k], io.max(), t["area"][k] / max(t["G"], 1e-6), F1(t)[k], F1(t)[b], P(t)[b], R(t)[b], P(t)[k], R(t)[k]))
r = np.array(rows)
def tab(col, edges, title):
    print(title + ": episodes | picked IoU | oracle IoU | IoU<0.1 | pick too small (<0.5x) | too big (>2x) | F1 of pick vs F1 of oracle node | P,R at oracle node")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r[:, col] >= lo) & (r[:, col] < hi)
        if m.sum(): print(f"  {lo * 100:5.1f}%-{hi * 100:5.1f}%  {m.sum():4d} | {r[m, 2].mean():.2f} | {r[m, 3].mean():.2f} | {np.mean(r[m, 2] < 0.1) * 100:3.0f}% | {np.mean(r[m, 4] < 0.5) * 100:3.0f}% | {np.mean(r[m, 4] > 2) * 100:3.0f}% | {r[m, 5].mean():.2f} vs {r[m, 6].mean():.2f} | {r[m, 7].mean():.2f}, {r[m, 8].mean():.2f}")
tab(0, [0, .01, .03, .1, .3, 1.01], "by target foreground share")
tab(1, [0, .01, .03, .1, .3, 1.01], "by reference foreground share")
print("share of total union mass by target foreground share:", " ".join(f"{((r[:, 0] >= lo) & (r[:, 0] < hi)).sum()}" for lo, hi in zip([0, .01, .03, .1, .3], [.01, .03, .1, .3, 1.01])))
