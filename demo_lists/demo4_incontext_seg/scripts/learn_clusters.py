"""Measuring device 2: can a learned classifier of INSID3's flat clusters close the selection gap?  (CPU; budget.py tables, 4 folds)

Per cluster: the label-free statistics INSID3 and our rules use (+ ranks within the episode); optionally the pooled debiased feature and
the reference prototype (--feat). Trained on three folds (class-disjoint), tested on the fourth. Selection = clusters with predicted
foreground probability above a threshold; the threshold is 0.5 or the one maximising expected IoU for the episode.

  python scripts/learn_clusters.py results/budget_coco_f0.tables.pt results/budget_coco_f1.tables.pt results/budget_coco_f2.tables.pt results/budget_coco_f3.tables.pt [--feat]
"""
import sys, torch, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
paths = [p for p in sys.argv[1:] if not p.startswith("--")]; FEAT = "--feat" in sys.argv
folds = [torch.load(p) for p in paths]
KEYS = ("cross", "intra", "aw", "combined", "back", "fgmax", "bgmax", "fgmax_o", "bgmax_o")
def feats(t):
    K = t["K"]; a = t["area"].float(); cols = [t[k].float() for k in KEYS] + [torch.log(a / a.sum()), t["fgmax"].float() - t["bgmax"].float(), (torch.arange(K) == t["seed"]).float(), torch.full((K,), float(K))]
    base = torch.stack(cols, 1); rk = torch.stack([(c[None] > c[:, None]).float().sum(1) for c in cols[:9]], 1)        # rank of each statistic within the episode (0 = highest)
    mx = torch.stack([c.max().expand(K) for c in cols[:9]], 1); X = torch.cat([base, rk, base[:, :9] - mx], 1)
    if FEAT:
        Pd = t["Pd"].float(); pr = t["proto"].float(); X = torch.cat([X, Pd, pr[None].expand(K, -1), Pd * pr[None]], 1)
    return X.numpy()
for T in folds:
    for t in T: t["X"] = feats(t); t["y"] = (t["pur"].float() > 0.5).numpy(); t["w"] = (t["area"].float() / t["area"].float().sum()).numpy()
def miou(T, sel):
    d = {}
    for t in T:
        a, p = t["area"].float().numpy(), t["pur"].float().numpy(); s = sel(t).astype(np.float32); fg = a * p; i = float((s * fg).sum()); u = float((s * a).sum() + ((1 - s) * fg).sum()); x = d.setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u
    return 100 * float(np.mean([i / max(u, 1e-6) for i, u in d.values()]))
def eiou_sel(t):
    """Expected-IoU-optimal subset given per-cluster probabilities: sort by probability, take the prefix with the best ratio."""
    a = t["area"].float().numpy(); q = t["prob"]; o = np.argsort(-q); inter = np.cumsum((a * q)[o]); iou = inter / (np.cumsum(a[o]) + (a * q).sum() - inter + 1e-9); k = int(iou.argmax()); s = np.zeros(len(a), bool); s[o[:k + 1]] = True; return s
res = []
for f, T in enumerate(folds):
    tr = [t for g, Tg in enumerate(folds) if g != f for t in Tg]; X = np.concatenate([t["X"] for t in tr]); y = np.concatenate([t["y"] for t in tr]); w = np.concatenate([t["w"] for t in tr])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=w)
    for t in T: t["prob"] = m.predict_proba(t["X"])[:, 1]
    r = (miou(T, lambda t: t["combined"].float().numpy() > 0.2), miou(T, lambda t: t["prob"] > 0.5), miou(T, lambda t: t["prob"] > 0.35), miou(T, eiou_sel), miou(T, lambda t: t["pur"].float().numpy() > 0.5)); res.append(r)
    print(f"fold {f}: INSID3 {r[0]:.2f} | learned, p > 0.5: {r[1]:.2f} | p > 0.35: {r[2]:.2f} | expected-IoU-optimal subset: {r[3]:.2f} | oracle {r[4]:.2f}", flush=True)
print("mean:   INSID3 %.2f | p > 0.5: %.2f | p > 0.35: %.2f | expected-IoU-optimal: %.2f | oracle %.2f" % tuple(np.mean(res, 0)))
