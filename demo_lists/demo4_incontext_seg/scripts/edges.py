"""Can the target's own structure tell which clusters belong to the same object?  (CPU, from budget.py tables)
For pairs of clusters in the target: similarity of their prototypes, for pairs inside the foreground vs pairs that cross its boundary."""
import sys, torch, numpy as np
T = [t for p in sys.argv[1:] for t in torch.load(p)]
def auc(s, y):
    o = s.argsort(descending=True); y = y[o].float(); tp = y.cumsum(0) / y.sum(); fp = (1 - y).cumsum(0) / (1 - y).sum(); return float(torch.trapz(tp, fp))
R = {k: [] for k in ("adj_o", "adj_d", "all_o", "all_d")}; Y = {k: [] for k in R}; miss_adj = []; bg_adj = []
for t in T:
    K = t["K"]; lab = t["lab"].long(); fg = t["pur"].float() > 0.5
    if K < 2 or fg.sum() == 0: continue
    A = torch.zeros(K, K, dtype=torch.bool)
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1], lab[1:])):
        m = a != b; A[a[m], b[m]] = True; A[b[m], a[m]] = True
    So = t["Po"].float() @ t["Po"].float().T; Sd = t["Pd"].float() @ t["Pd"].float().T
    iu = torch.triu(torch.ones(K, K, dtype=torch.bool), 1); ff = fg[:, None] & fg[None]; fb = fg[:, None] ^ fg[None]
    for name, S_, mask in (("adj_o", So, A & iu), ("adj_d", Sd, A & iu), ("all_o", So, iu), ("all_d", Sd, iu)):
        m = mask & (ff | fb); R[name].append(S_[m]); Y[name].append(ff[m])
    sel = t["combined"].float() > 0.2
    if t["seed"] >= 0 and fg[t["seed"]]:
        touch = (A & sel[None]).any(1)                           # cluster touches the selected region
        miss = fg & ~sel; bgd = ~fg & ~sel
        miss_adj += touch[miss].tolist(); bg_adj += touch[bgd].tolist()
for k in R:
    s, y = torch.cat(R[k]), torch.cat(Y[k]); print(f"{k}: {len(s)} pairs, {float(y.float().mean()) * 100:.0f}% inside the foreground; mean similarity inside {float(s[y].mean()):.3f} vs across {float(s[~y].mean()):.3f}; AUC {auc(s, y):.3f}")
print(f"missed foreground clusters that touch the selected region: {np.mean(miss_adj) * 100:.0f}% (of {len(miss_adj)});  dropped background clusters that touch it: {np.mean(bg_adj) * 100:.0f}% (of {len(bg_adj)})")
