"""Supervised information bound for per-token cues (fit with query labels; diagnostic only).

A few-weight logistic fusion is fit on half A of the dev episodes and applied, frozen, to half B.
Cue groups: appearance (prototype, forward/backward transfer, region means), FoRIS's own score,
and spatial-arrangement cues of the dense correspondence.  If a group adds nothing to the
bound, no fixed-constant rule built from it can add anything either.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab

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


def box(v):
    t = torch.from_numpy(v.astype(np.float32)).reshape(1, 1, 64, 64)
    return torch.nn.functional.avg_pool2d(t, 3, 1, 1, count_include_pad=False).flatten().numpy()


def cues(z, bd, g):
    s, f5, f2 = z["sq"].astype(np.float64), bd["fwdF0.05"].astype(np.float64), bd["fwdF0.02"].astype(np.float64)
    gF, gB = np.log(4096 * bd["backF0.05"] + 1e-4), np.log(4096 * bd["backB0.05"] + 1e-4)
    a = z["assign"].astype(int)
    cnt = np.maximum(np.bincount(a, minlength=64), 1)
    reg = lambda v: (np.bincount(a, weights=v, minlength=64) / cnt)[a]
    app = [mm(s), f5, mm(f5), (f2 > 0.5).astype(float), gF, gF - gB, mm(reg(s)), reg(f5), mm(reg(f5)),
           np.full(4096, np.log(z["y"].mean() + 1e-4))]
    foris = [mm(z["foris_score"].astype(np.float64).reshape(-1))]
    v, c = g["vote"].astype(np.float64), g["coh"].astype(np.float64)
    geo = [v, c, g["coh_lab"], v * c, g["mnn"], v * g["mnn"], np.log1p(g["hitF"]) - np.log1p(g["hitB"]), box(v * c), box(v), reg(v * c)]
    return np.stack(app + foris + geo, 1)


APP = ["proto_mm", "f", "f_mm", "nn_vote_soft", "logliftF", "liftF-B", "reg(proto)_mm", "reg(f)", "reg(f)_mm", "log_ref_area"]
GEO = ["vote", "coh", "coh_lab", "vote*coh", "mnn", "vote*mnn", "hitF-hitB", "box(vote*coh)", "box(vote)", "reg(vote*coh)"]
NAMES = APP + ["foris_mm"] + GEO
iA, iF, iG = list(range(10)), [10], list(range(11, 21))
D = []
for b in rows:
    k = f"{b['index']:06d}.npz"
    z, bd, g = np.load(A_ / "fields300" / k), np.load(A_ / "bidir300" / k), np.load(A_ / "geo300" / k)
    gt = torch.from_numpy(np.unpackbits(z["gt"])[: 1024 * 1024].reshape(1024, 1024).astype(bool))
    D.append((cues(z, bd, g), lab.coverage64(gt).flatten().numpy(), gt, (int(z["foris_iu"][0]), int(z["foris_iu"][1]))))


def fit(sel, cols):
    Xs, ys = [], []
    for e in np.where(sel)[0]:
        idx = rng.choice(4096, 256, replace=False)
        Xs.append(D[e][0][idx][:, cols]); ys.append(D[e][1][idx])
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


def evaluate(model, sel, cols):
    w, mu, sd = model
    M = lab.Meter()
    for e in np.where(sel)[0]:
        p = torch.sigmoid(((torch.tensor(D[e][0][:, cols]).float() - mu) / sd) @ w[:-1] + w[-1])
        M.add(rows[e], *lab.iu(lab.up(p.reshape(64, 64)) > 0.5, D[e][2]))
    return M


def line(tag, M):
    r = M.row(lab.DOMAINS)
    v = [r[d][0] for d in lab.DOMAINS]
    print(f"{tag:46s}" + "".join(f"{x:8.1f}" for x in v) + f"{np.mean(v):8.1f}")


Fm = lab.Meter()
for e in np.where(~half)[0]:
    Fm.add(rows[e], *D[e][3])
print("class-pooled mIoU on half B (fused rows: no CRF); columns: " + " ".join(d[:7] for d in lab.DOMAINS) + " mean")
line("FoRIS complete (CRF)", Fm)
for tag, cols in {"FoRIS score": iF, "appearance": iA, "geometry": iG, "appearance + geometry": iA + iG,
                  "FoRIS score + appearance": iA + iF, "FoRIS score + geometry": iF + iG, "all three": iA + iF + iG}.items():
    m = fit(half, cols)
    line("fit A -> B | " + tag, evaluate(m, ~half, cols))
    if tag == "all three":
        print("   weights: " + ", ".join(f"{NAMES[c]}={float(x):+.2f}" for c, x in zip(cols, m[0][:-1])))
print("\nleave-one-domain-out (fit on the 4 other domains' half A, test on the held-out domain's half B), all three groups:")
out, ref = [], []
for d in lab.DOMAINS:
    m = fit(half & (dom != d), iA + iF + iG)
    out.append(evaluate(m, ~half & (dom == d), iA + iF + iG).row((d,))[d][0]); ref.append(Fm.row((d,))[d][0])
print("  fused " + " ".join(f"{x:6.1f}" for x in out) + f"  mean {np.mean(out):.1f}   | FoRIS " + " ".join(f"{x:6.1f}" for x in ref) + f"  mean {np.mean(ref):.1f}")
