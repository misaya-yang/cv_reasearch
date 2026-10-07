"""Does an object-size estimate read from the query image alone (colour and edges around the score peak) pick the
cut level? fresh600 token record; same harness as score_size_threshold.py (level per quartile of the estimate, fitted
on the other folds, token-level class mIoU against the fixed 0.5 cut of the RCG field). CPU only, scipy only.

    PYTHONPATH=/root/demo4_cache/env python scripts/score_query_size.py
"""
import json, numpy as np
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage

D = "/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/claude_order_fresh600/"; IMG = "/root/demo4_cache/data/COCO2014/"
Z = np.load(D + "tokens.npz"); rows = json.load(open(D + "rows.json")); fold = np.array([r["fold"] for r in rows]); N = len(rows)
t, r = Z["truth"].astype(np.float32), Z["rcg"].astype(np.float32)
S = 256; UP = S // 64


def lab(rgb):
    x = rgb / 255.; x = np.where(x > .04045, ((x + .055) / 1.055) ** 2.4, x / 12.92)
    xyz = x @ np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]]).T / np.array([.9505, 1., 1.089])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def one(i):
    im = np.asarray(Image.open(IMG + rows[i]["query"]).convert("RGB").resize((S, S), Image.BILINEAR), np.float32)
    L = ndimage.gaussian_filter(lab(im), (1, 1, 0)); g = np.sqrt(sum(ndimage.sobel(L[..., c], a) ** 2 for c in range(3) for a in (0, 1)))
    f = r[i].reshape(64, 64); py, px = np.unravel_index(f.argmax(), f.shape); seed = np.zeros((S, S), bool); seed[py * UP:(py + 1) * UP, px * UP:(px + 1) * UP] = True
    up = lambda m: np.kron(m, np.ones((UP, UP), bool))
    out = {}
    for tau in (10, 20):  # colour flood: connected pixels within tau of the seed colour
        m = np.linalg.norm(L - L[seed].mean(0), axis=-1) < tau; lb, _ = ndimage.label(m | seed); out[f"colour flood dE<{tau}"] = (lb == lb[seed][0]).sum()
    for q in (60, 75):  # edge-bounded flood: connected pixels below the q-th gradient percentile
        m = g < np.percentile(g, q); lb, _ = ndimage.label(m | seed); out[f"edge-bounded flood p{q}"] = (lb == lb[seed][0]).sum()
    g16 = np.minimum(g / (g.max() + 1e-9) * 65535, 65535).astype(np.uint16)
    for name, fg in (("peak", seed), ("rcg>0.9", up(f > .9) | seed)):  # watershed on colour edges between score seeds
        bg = up(f < .1) & ~fg
        if not bg.any(): bg = np.zeros((S, S), bool); bg[0], bg[-1], bg[:, 0], bg[:, -1] = True, True, True, True; bg &= ~fg
        mk = np.zeros((S, S), np.int16); mk[bg] = 2; mk[fg] = 1; out[f"watershed {name} vs rcg<0.1"] = (ndimage.watershed_ift(g16, mk) == 1).sum()
    return out


if __name__ == "__main__":
    with Pool(28) as p: E = p.map(one, range(N), chunksize=8)
    Lg = lambda x: np.log(np.maximum(x, .5) / 4096); y = Lg(t.sum(1)); P = {k: Lg(np.array([e[k] for e in E]) / UP ** 2) for k in E[0]}
    area = Lg((r > .5).sum(1)); P = {"area rcg>0.5 (score-derived control)": area, **P}
    cls = np.array([x["c"] for x in rows]); C = np.unique(cls); lev = np.linspace(.2, .9, 29); k0 = int(np.argmin(abs(lev - .5))); tb = t > .5; a = np.arange(N)
    I = np.stack([((r > l) & tb).sum(1) for l in lev], 1).astype(float); U = np.stack([((r > l) | tb).sum(1) for l in lev], 1).astype(float)
    miou = lambda i, u: np.mean([i[cls == c].sum() / max(u[cls == c].sum(), 1e-9) for c in C]) * 100
    base = miou(I[:, k0], U[:, k0]); J = base / 100; V = (I - J * U) - (I[:, [k0]] - J * U[:, [k0]]); found = (I / np.maximum(U, 1)).max(1) >= .5

    def held(X):
        o = np.zeros(N)
        for k in np.unique(fold):
            A = np.c_[X[fold != k], np.ones((fold != k).sum())]; o[fold == k] = np.c_[X[fold == k], np.ones((fold == k).sum())] @ np.linalg.lstsq(A, y[fold != k], rcond=None)[0]
        return o

    def gain(est, nb=4):
        pick = np.full(N, k0)
        for k in np.unique(fold):
            fit, hd = fold != k, fold == k; b = np.digitize(est, np.quantile(est[fit], np.linspace(0, 1, nb + 1)[1:-1]))
            for q in range(nb):
                if (fit & (b == q)).sum() > 15: pick[hd & (b == q)] = V[fit & (b == q)].sum(0).argmax()
        return miou(I[a, pick], U[a, pick]) - base

    for k in list(P)[1:]: P[k + " + area, held-fold fit"] = held(np.c_[P[k], area])
    P["all six + area, held-fold fit"] = held(np.stack([P[k] for k in list(E[0])] + [area], 1))
    print(f"token-level mIoU at 0.5: {base:.2f}, n={N}, found {found.sum()};  TRUE size (diagnostic): {gain(y):+.2f}")
    print(f"{'estimate':52s} gain   corr(true)  within 2.7x  corr of its error with the area error")
    ea = held(area[:, None]) - y
    for k, v in P.items():
        h = held(v[:, None]) if v.ndim == 1 else v; e = h - y
        print(f"{k:52s} {gain(v):+.2f}   {np.corrcoef(v, y)[0, 1]:.2f}        {np.mean(abs(e) < 1):.2f}         {np.corrcoef(e, ea)[0, 1]:.2f}")
