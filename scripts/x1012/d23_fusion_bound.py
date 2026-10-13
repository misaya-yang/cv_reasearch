"""Offline bound: a few-weight logistic fusion of scale-free token cues, with transfer tests.

Diagnostic of what a fixed set of constants could reach (FoRIS itself carries ~15 hand-set
constants).  Weights are fit on tokens of a training split with query labels and then frozen:
  (a) pooled half A -> half B;   (b) COCO only -> the four other domains.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
F300 = lab.ASSETS / "a/x1012/fields300"
B300 = lab.ASSETS / "a/x1012/bidir300"
rows = [b for b in lab.subset(300) if (F300 / f"{b['index']:06d}.npz").exists()]
dom = np.array([b["dataset"] for b in rows])
half = np.zeros(len(rows), bool)
for d in lab.DOMAINS:
    idx = np.where(dom == d)[0]
    half[idx[::2]] = True
rng = np.random.default_rng(0)


def mm(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-6)


def cues(z, bd):
    s, f5, f2 = z["sq"].astype(np.float64), bd["fwdF0.05"].astype(np.float64), bd["fwdF0.02"].astype(np.float64)
    gF, gB = np.log(4096 * bd["backF0.05"] + 1e-4), np.log(4096 * bd["backB0.05"] + 1e-4)
    a = z["assign"].astype(int)
    cnt = np.maximum(np.bincount(a, minlength=64), 1)
    reg = lambda v: (np.bincount(a, weights=v, minlength=64) / cnt)[a]
    order = np.argsort(-s)
    pa = np.cumsum(f5[order]) / np.arange(1, 4097)
    pc = np.cumsum(bd["backF0.05"][order])
    J = pa * pc / np.maximum(pa + pc - pa * pc, 1e-9)
    k = int(np.argmax(J[4:])) + 4
    inJ = np.zeros(4096)
    inJ[order[: k + 1]] = 1
    X = np.stack([mm(s), f5, mm(f5), (f2 > 0.5).astype(float), gF, gF - gB, mm(reg(s)), reg(f5), mm(reg(f5)), inJ,
                  np.full(4096, np.log(z["y"].mean() + 1e-4))], 1)
    return X


NAMES = ["proto_mm", "f", "f_mm", "nn_vote", "logliftF", "liftF-B", "reg(proto)_mm", "reg(f)", "reg(f)_mm", "in_J_prefix", "log_ref_area"]
D = []
for b in rows:
    z = np.load(F300 / f"{b['index']:06d}.npz")
    bd = np.load(B300 / f"{b['index']:06d}.npz")
    gt = torch.from_numpy(np.unpackbits(z["gt"])[: 1024 * 1024].reshape(1024, 1024).astype(bool))
    D.append((cues(z, bd), lab.coverage64(gt).flatten().numpy(), gt, (int(z["foris_iu"][0]), int(z["foris_iu"][1]))))


def fit(sel, cols):
    Xs, ys = [], []
    for e in np.where(sel)[0]:
        X, c, _, _ = D[e]
        idx = rng.choice(4096, 256, replace=False)
        Xs.append(X[idx][:, cols]); ys.append(c[idx])
    X, y = torch.tensor(np.concatenate(Xs)).float(), torch.tensor(np.concatenate(ys)).float()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    w = torch.zeros(X.shape[1] + 1, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=200)
    Xn = (X - mu) / sd
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xn @ w[:-1] + w[-1], y) + 1e-4 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), mu, sd


def evaluate(model, sel, cols, rule="0.5"):
    w, mu, sd = model
    M = lab.Meter()
    for e in np.where(sel)[0]:
        X, _, gt, _ = D[e]
        p = torch.sigmoid(((torch.tensor(X[:, cols]).float() - mu) / sd) @ w[:-1] + w[-1])
        S = lab.up(p.reshape(64, 64))
        M.add(rows[e], *lab.iu((S > 0.5) if rule == "0.5" else (lab.minmax(S) > 0.5), gt))
    return M


def line(tag, M, doms=lab.DOMAINS):
    r = M.row(doms)
    v = [r[d][0] for d in doms]
    print(f"{tag:44s}" + "".join(f"{x:8.1f}" for x in v) + f"{np.mean(v):8.1f}")


F = lab.Meter()
for e in np.where(~half)[0]:
    F.add(rows[e], *D[e][3])
print(f"class-pooled mIoU, no CRF for fused rows;  columns: " + " ".join(d[:7] for d in lab.DOMAINS) + " mean")
line("FoRIS complete (CRF) [half B]", F)
ALLC = list(range(len(NAMES)))
sets = {"all cues": ALLC, "no region cues": [0, 1, 2, 3, 4, 5, 9, 10], "proto_mm + f + nn_vote": [0, 1, 3],
        "proto_mm only": [0], "f only": [1]}
for tag, cols in sets.items():
    m = fit(half, cols)
    line(f"pooled A->B | {tag} | p>0.5", evaluate(m, ~half, cols))
    if tag == "all cues":
        line(f"pooled A->B | {tag} | minmax .5", evaluate(m, ~half, cols, "mm"))
        print("   weights (standardised): " + ", ".join(f"{n}={float(x):+.2f}" for n, x in zip(NAMES, m[0][:-1])))
for src in lab.DOMAINS:
    m = fit(dom == src, ALLC)
    Mx = evaluate(m, (dom != src) & ~half, ALLC)
    Fx = lab.Meter()
    for e in np.where((dom != src) & ~half)[0]:
        Fx.add(rows[e], *D[e][3])
    others = [d for d in lab.DOMAINS if d != src]
    r, f = Mx.row(others), Fx.row(others)
    print(f"fit on {src:12s} -> others | all cues: " + "  ".join(f"{d[:6]} {r[d][0]:5.1f} (FoRIS {f[d][0]:5.1f})" for d in others))
