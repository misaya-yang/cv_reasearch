"""Non-linear supervised bound with episode context (diagnostic only, uses query labels to fit).

Gradient-boosted trees on: every per-token cue, its within-episode percentile rank, and
episode-level scalars (area estimates, reference area, cue quantiles).  This can express
"top-k with k = g(episode statistics)" and any per-token interaction.  If this cannot pass
FoRIS on held-out episodes / a held-out domain, the information is not in these fields.
"""
import sys
from pathlib import Path
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, str(Path(__file__).parent))
import lab
import importlib.util

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
A_ = lab.ASSETS / "a/x1012"
rows = [b for b in lab.subset(300) if (A_ / "fields300" / f"{b['index']:06d}.npz").exists()]
dom = np.array([b["dataset"] for b in rows])
half = np.zeros(len(rows), bool)
for d in lab.DOMAINS:
    half[np.where(dom == d)[0][::2]] = True
rng = np.random.default_rng(0)
mm = lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-6)


def rank(v):
    r = np.empty(len(v)); r[np.argsort(v)] = np.linspace(0, 1, len(v)); return r


def feats(z, bd, g, use_foris):
    s, f5, f2 = z["sq"].astype(np.float64), bd["fwdF0.05"].astype(np.float64), bd["fwdF0.02"].astype(np.float64)
    bF, bB = bd["backF0.05"].astype(np.float64), bd["backB0.05"].astype(np.float64)
    a = z["assign"].astype(int)
    cnt = np.maximum(np.bincount(a, minlength=64), 1)
    reg = lambda v: (np.bincount(a, weights=v, minlength=64) / cnt)[a]
    tok = [mm(s), f5, f2, np.log(4096 * bF + 1e-4), np.log(4096 * bF + 1e-4) - np.log(4096 * bB + 1e-4), mm(reg(s)), reg(f5),
           g["vote"].astype(np.float64), g["coh"].astype(np.float64), g["vote"] * g["coh"], g["fs"].astype(np.float64)]
    if use_foris:
        tok.append(mm(z["foris_score"].astype(np.float64).reshape(-1)))
    tok = tok + [rank(t) for t in tok]
    p = bF / bF.sum()
    epi = [np.log(z["y"].mean() + 1e-4), np.log((f5 > 0.5).mean() + 1e-4), np.log(f5.mean() + 1e-4), np.log((f2 > 0.5).mean() + 1e-4),
           -(p * np.log(p + 1e-30)).sum(), np.log((mm(s) > 0.7).mean() + 1e-4), np.log((mm(s) > 0.5).mean() + 1e-4),
           np.quantile(s, 0.5), np.quantile(s, 0.9), np.quantile(s, 0.99), s.max(), s.std(), g["fs"].mean(), g["coh"].mean()]
    if use_foris:
        fz = mm(z["foris_score"].astype(np.float64).reshape(-1))
        epi += [np.log((fz > 0.5).mean() + 1e-4), np.log((fz > 0.3).mean() + 1e-4), np.log((fz > 0.7).mean() + 1e-4), fz.mean()]
    return np.concatenate([np.stack(tok, 1), np.tile(np.array(epi), (4096, 1))], 1).astype(np.float32)


def load(use_foris):
    D = []
    for b in rows:
        k = f"{b['index']:06d}.npz"
        z, bd, g = np.load(A_ / "fields300" / k), np.load(A_ / "bidir300" / k), np.load(A_ / "geo300" / k)
        gt = torch.from_numpy(np.unpackbits(z["gt"])[: 1024 * 1024].reshape(1024, 1024).astype(bool))
        D.append((feats(z, bd, g, use_foris), lab.coverage64(gt).flatten().numpy() > 0.5, gt, (int(z["foris_iu"][0]), int(z["foris_iu"][1]))))
    return D


def run(D, tr, te):
    X = np.concatenate([D[e][0][rng.choice(4096, 384, replace=False)] for e in np.where(tr)[0]])
    rng2 = np.random.default_rng(0)
    y = np.concatenate([D[e][1][rng2.choice(4096, 384, replace=False)] for e in np.where(tr)[0]])
    return X, y


def fit_eval(D, tr, te):
    Xs, ys = [], []
    for e in np.where(tr)[0]:
        idx = rng.choice(4096, 384, replace=False)
        Xs.append(D[e][0][idx]); ys.append(D[e][1][idx])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    clf.fit(np.concatenate(Xs), np.concatenate(ys))
    M = lab.Meter()
    for e in np.where(te)[0]:
        p = torch.from_numpy(clf.predict_proba(D[e][0])[:, 1].astype(np.float32))
        M.add(rows[e], *lab.iu(lab.up(p.reshape(64, 64)) > 0.5, D[e][2]))
    return M


def line(tag, v):
    print(f"{tag:52s}" + "".join(f"{x:8.1f}" for x in v) + f"{np.mean(v):8.1f}")


print("class-pooled mIoU (fused rows: no CRF); columns: " + " ".join(d[:7] for d in lab.DOMAINS) + " mean")
for use_foris in (False, True):
    D = load(use_foris)
    if not use_foris:
        Fm = lab.Meter()
        for e in np.where(~half)[0]:
            Fm.add(rows[e], *D[e][3])
        r = Fm.row(lab.DOMAINS); line("FoRIS complete (CRF) [half B]", [r[d][0] for d in lab.DOMAINS])
    tag = "with FoRIS score" if use_foris else "without FoRIS score"
    r = fit_eval(D, half, ~half).row(lab.DOMAINS)
    line(f"GBDT A -> B, {tag}", [r[d][0] for d in lab.DOMAINS])
    out = [fit_eval(D, half & (dom != d), ~half & (dom == d)).row((d,))[d][0] for d in lab.DOMAINS]
    line(f"GBDT leave-one-domain-out, {tag}", out)
