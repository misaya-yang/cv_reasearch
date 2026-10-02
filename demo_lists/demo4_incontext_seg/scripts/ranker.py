"""How much could be gained by *combining* the available evidence?  A learned node scorer as a measuring device.

Gradient-boosted regression of a node's IoU from label-free node features; trained on episodes of three COCO-20i folds
(classes disjoint from the test fold; validation images), tested on the fourth. This is not a training-free method; it
tells whether the information needed to pick the right node is present in the evidence at all, and which evidence matters.

  python scripts/ranker.py results/l3b_f0.l3.pt results/l3b_f1.l3.pt results/l3b_f2.l3.pt results/l3b_f3.l3.pt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.offline2 import *
from sklearn.ensemble import HistGradientBoostingRegressor
folds = [load([p]) for p in sys.argv[1:]]; rng = np.random.default_rng(0)
NAMES = None
def feats(t):
    global NAMES
    n = t["n"]; N = t["N"]; A = N["area"]; ev = accumulate(t["ch"], np.stack([t["ev"][k].astype(np.float32) for k in ("q30", "mF30", "mB30", "dF30", "dB30", "fgmax", "bgmax")]))
    nf, nb = max(t["n_reffg"], 1), max(t["n_refbg"], 1); P = N["back"] / A; R = N["fwd"] / nf; Rb = N["bfwd"] / nb; Pref = N["fwd"] / (N["fwd"] + N["bfwd"] + e_); Fs = f1(P, R)
    br = int(t["r_iou"].argmax()); par = t["parent"]; pp = np.where(par >= 0, par, np.arange(len(par)))
    cols = dict(P=P, R=R, Rb=Rb, Pref=Pref, F1=Fs, q=ev[0] / A, mF=ev[1] / (ev[1, -1] + e_), mB=ev[2] / (ev[2, -1] + e_), dF=ev[3] / (ev[3, -1] + e_), dP=ev[3] / (ev[3] + ev[4] + e_),
                sim=N["sim"] / A, fgmax=ev[5] / A, bgmax=ev[6] / A, cos=t["cos_d"].astype(np.float32), coh=t["coh_d"].astype(np.float32), tr1=t["tr1"].astype(np.float32), tr5=t["tr5"].astype(np.float32),
                trs=t["trs"].astype(np.float32), tr1_area=t["tr1_area"].astype(np.float32), larea=np.log(A / n), birth=t["birth"], death=t["death"], pers=t["death"] - t["birth"],
                ref_frac=np.full(len(A), nf / n, np.float32), r_birth=np.full(len(A), t["r_birth"][br], np.float32), r_death=np.full(len(A), t["r_death"][br], np.float32), r_iou=np.full(len(A), t["r_iou"][br], np.float32),
                F1_rel=Fs / (Fs.max() + e_))
    for k in ("P", "R", "F1", "q", "larea"): cols["par_" + k] = cols[k][pp]; cols["d_" + k] = cols[k][pp] - cols[k]
    cF = np.zeros(len(A), np.float32); cF[n:] = np.maximum(Fs[t["ch"][:, 0]], Fs[t["ch"][:, 1]]); cF[:n] = Fs[:n]; cols["child_F1"] = cF
    NAMES = list(cols); return np.stack([cols[k] for k in NAMES], 1).astype(np.float32)
for T in folds:
    for t in T: t["X"] = feats(t); t["y"] = iou(t).astype(np.float32); t["ok"] = area(t) >= 2
def rows(T, top=150, rand=150):
    X, y = [], []
    for t in T:
        ok = np.nonzero(t["ok"])[0]; s = t["X"][ok, NAMES.index("F1")]; idx = np.unique(np.concatenate([ok[np.argsort(-s)[:top]], rng.choice(ok, min(rand, len(ok)), replace=False), [int(t["y"].argmax())]]))
        X.append(t["X"][idx]); y.append(t["y"][idx])
    return np.concatenate(X), np.concatenate(y)
res = []
for f, T in enumerate(folds):
    tr = [t for g, Tg in enumerate(folds) if g != f for t in Tg]; X, y = rows(tr)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(X, y)
    for t in T: t["pred"] = np.where(t["ok"], m.predict(t["X"]), -1)
    print(f"fold {f}: trained on {len(X)} nodes from {len(tr)} episodes;  INSID3 {baseline(T):.2f}")
    a_ = show(T, "  F1 rule", F1); b_ = show(T, "  learned scorer", lambda t: t["pred"]); c_ = show(T, "  oracle node", iou); res.append((baseline(T), a_, b_, c_))
    if f == 0:
        from sklearn.inspection import permutation_importance
        Xv, yv = rows(T, 100, 100); pi = permutation_importance(m, Xv, yv, n_repeats=3, random_state=0, n_jobs=4)
        print("   most important features:", ", ".join(f"{NAMES[i]} {pi.importances_mean[i]:.3f}" for i in np.argsort(-pi.importances_mean)[:12]))
r = np.array(res); print("\nmean over folds:  INSID3 %.2f | F1 rule %.2f | learned scorer %.2f | oracle node %.2f" % tuple(r.mean(0)))
