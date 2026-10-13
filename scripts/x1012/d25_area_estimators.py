"""Area estimators that do not come from the level of the ranking itself.

Each estimator returns a predicted target area fraction of the query; it is judged by
its log2 error against the true area and by the IoU of the top-k cut (k = estimate)
on the centred-prototype ranking and on the FoRIS score ranking.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
F300, B300 = lab.ASSETS / "a/x1012/fields300", lab.ASSETS / "a/x1012/bidir300"
rows = [b for b in lab.subset(300) if (F300 / f"{b['index']:06d}.npz").exists()]
dom = np.array([b["dataset"] for b in rows])
half = np.zeros(len(rows), bool)
for d in lab.DOMAINS:
    half[np.where(dom == d)[0][::2]] = True


def topk(S, frac):
    k = int(min(max(frac * S.numel(), 1), S.numel()))
    return S >= torch.kthvalue(S.flatten(), S.numel() - k + 1).values


def mm(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-6)


E, T, R = [], [], []
for b in rows:
    z, bd = np.load(F300 / f"{b['index']:06d}.npz"), np.load(B300 / f"{b['index']:06d}.npz")
    gt = torch.from_numpy(np.unpackbits(z["gt"])[: 1024 * 1024].reshape(1024, 1024).astype(bool))
    est = {"ref area": float(z["y"].mean())}
    for t in ("0.02", "0.05"):
        f, bF, bB = bd["fwdF" + t].astype(np.float64), bd["backF" + t].astype(np.float64), bd["backB" + t].astype(np.float64)
        p = bF / bF.sum()
        est[f"|f>0.5| t{t}"] = float((f > 0.5).mean())
        est[f"sum f t{t}"] = float(f.mean())
        est[f"perplexity(backF) t{t}"] = float(np.exp(-(p * np.log(p + 1e-30)).sum()) / 4096)
        est[f"1/sum backF^2 t{t}"] = float(1 / (p ** 2).sum() / 4096)
        est[f"|backF>backB| t{t}"] = float((bF * (1 - z["y"].mean()) > bB * z["y"].mean()).mean()) if False else float((bF / max(bF.sum(), 1e-30) > bB / max(bB.sum(), 1e-30)).mean())
    est["foris minmax .5 area"] = float((lab.up(torch.from_numpy(mm(z["foris_score"]).astype(np.float32))) > 0.5).float().mean())
    est["proto minmax .7 area"] = float((lab.up(torch.from_numpy(mm(z["sq"]).reshape(64, 64).astype(np.float32))) > 0.7).float().mean())
    E.append(est); T.append(float(gt.float().mean()))
    R.append({n: lab.up(torch.from_numpy(s.astype(np.float32)).reshape(64, 64)) for n, s in (("proto", z["sq"]), ("foris", z["foris_score"].reshape(-1)))} | {"gt": gt, "fi": (int(z["foris_iu"][0]), int(z["foris_iu"][1]))})
names = list(E[0])
X = np.log(np.maximum(np.array([[e[n] for n in names] for e in E]), 1e-5))
y = np.log(np.maximum(np.array(T), 1e-6))

# pooled A->B log-linear combination of all estimators (bound for constants)
A = np.c_[X[half], np.ones(half.sum())]
w = np.linalg.lstsq(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y[half], rcond=None)[0]
fit_all = np.c_[X, np.ones(len(X))] @ w
print("log2 error of area estimators [half B]: median |err| per domain, then pooled median |err| and median signed err")
cols = {n: X[:, i] for i, n in enumerate(names)} | {"FIT all estimators (A->B)": fit_all}
for n, v in cols.items():
    err = (v - y) / np.log(2)
    print(f"{n:30s}" + "".join(f"{np.median(np.abs(err[(dom == d) & ~half])):7.2f}" for d in lab.DOMAINS) + f"   | {np.median(np.abs(err[~half])):5.2f} {np.median(err[~half]):+5.2f}")

print(f"\nclass-pooled mIoU of top-k cut with k = estimate [half B]; columns " + " ".join(d[:6] for d in lab.DOMAINS) + " mean")
for rk in ("proto", "foris"):
    for n, v in cols.items():
        M = lab.Meter()
        for e in np.where(~half)[0]:
            M.add(rows[e], *lab.iu(topk(R[e][rk], float(np.exp(v[e]))), R[e]["gt"]))
        r = M.row(lab.DOMAINS)
        print(f"{rk} | k = {n:30s}" + "".join(f"{r[d][0]:7.1f}" for d in lab.DOMAINS) + f"{np.mean([r[d][0] for d in lab.DOMAINS]):7.1f}")
M = lab.Meter()
for e in np.where(~half)[0]:
    M.add(rows[e], *R[e]["fi"])
r = M.row(lab.DOMAINS)
print(f"{'FoRIS complete (CRF)':42s}" + "".join(f"{r[d][0]:7.1f}" for d in lab.DOMAINS) + f"{np.mean([r[d][0] for d in lab.DOMAINS]):7.1f}")
