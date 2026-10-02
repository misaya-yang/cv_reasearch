"""Node-scoring rules over the full cluster tree, evaluated offline (CPU) from the tables written by hier.py.
  python scripts/nodes_offline.py results/hier_f0_300.nodes.pt
IoU is computed at feature resolution with soft ground truth. Every rule picks ONE tree node per episode."""
import sys, torch, numpy as np
T = [t for p in sys.argv[1:] for t in torch.load(p, weights_only=False)]
def miou(pick):
    d = {}; info = []
    for t in T:
        k = pick(t); i = float(t["fg"][k]); u = float(t["area"][k] + t["G"] - t["fg"][k]); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u; info.append((t["area"][k] / max(t["G"], 1e-6), i / max(u, 1e-6)))
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()])), np.array(info)
def show(name, pick):
    m, info = miou(pick); r = info[:, 0]
    print(f"  {name:58s} {m:6.2f} | mean IoU {info[:, 1].mean():.3f} | picked/true area: median {np.median(r):.2f}, <0.5 in {np.mean(r < 0.5) * 100:.0f}%, >2 in {np.mean(r > 2) * 100:.0f}% | IoU<0.1 in {np.mean(info[:, 1] < 0.1) * 100:.0f}%")
A = lambda t: t["area"]; iou = lambda t: t["fg"] / (t["area"] + t["G"] - t["fg"] + 1e-9)
P = lambda t: t["back"] / t["area"]; R = lambda t: t["fwd"] / max(t["n_reffg"], 1); Rb = lambda t: t["bfwd"] / max(t["n_refbg"], 1)
print(len(T), "episodes")
show("oracle: best node", lambda t: int(iou(t).argmax()))
show("pooled cosine to reference foreground", lambda t: int(t["cos"].astype(np.float32).argmax()))
show("pooled cosine, area >= 8", lambda t: int(np.where(A(t) >= 8, t["cos"].astype(np.float32), -9).argmax()))
show("mean patch similarity to prototype (cross), area >= 8", lambda t: int(np.where(A(t) >= 8, t["sim"] / A(t), -9).argmax()))
show("overlap with 'nearest reference patch is foreground' map", lambda t: int((t["back"] / (A(t) + t["back"][-1] - t["back"] + 1e-9)).argmax()))
show("overlap with INSID3's candidate map", lambda t: int((t["cand"] / (A(t) + t["cand"][-1] - t["cand"] + 1e-9)).argmax()))
show("F1 of precision (back share) and recall (ref-FG coverage)", lambda t: int((2 * P(t) * R(t) / (P(t) + R(t) + 1e-9)).argmax()))
show("precision * recall", lambda t: int((P(t) * R(t)).argmax()))
show("recall - ref-BG coverage (Youden on the reference side)", lambda t: int((R(t) - Rb(t)).argmax()))
show("P * (R - Rb)", lambda t: int((P(t) * (R(t) - Rb(t))).argmax()))
for rho in (0.2, 0.33, 0.5): show(f"ML under independent noise: back count - {rho} * area", lambda t, rho=rho: int((t["back"] - rho * A(t)).argmax()))
show("sum of (fgmax - bgmax) over the node", lambda t: int((t["fgmax"] - t["bgmax"]).argmax()))
show("cos * F1", lambda t: int((np.clip(t["cos"].astype(np.float32), 0, None) * 2 * P(t) * R(t) / (P(t) + R(t) + 1e-9)).argmax()))
# how do the label-free quantities look at the best node, at its parent, and at its larger child?
rows = []
for t in T:
    b = int(iou(t).argmax()); par = int(t["parent"][b]); n = (len(t["area"]) + 1) // 2
    if par < 0: continue
    rows.append([[f(t)[x] for f in (iou, lambda t: t["cos"].astype(np.float32), P, R, Rb, lambda t: t["coh_d"].astype(np.float32), lambda t: A(t) / max(t["G"], 1e-6))] for x in (b, par)])
rows = np.array(rows); print("\nbest node vs its parent (means): IoU, pooled cosine, precision, recall, ref-BG coverage, coherence, area/true area")
for j, nm in enumerate(("best node", "its parent")): print(f"  {nm:10s} " + "  ".join(f"{rows[:, j, q].mean():.3f}" for q in range(7)))
