"""Offline: extent from bidirectional attention mass (a label-free precision/recall surrogate).

For a candidate query set S (level set of a ranking):
  a(S) = mean_{j in S} sum_i P(i|j) y_i        forward purity  (falls when S is too large)
  c(S) = sum_{j in S} backF(j)                 backward capture of the reference foreground's
                                               attention (falls when S is too small)
Pick the level maximising the IoU-form combination a c / (a + c - a c) (or F1).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lab

F300 = lab.ASSETS / "a/x1012/fields300"
B300 = lab.ASSETS / "a/x1012/bidir300"
rows = [b for b in lab.subset(300) if (F300 / f"{b['index']:06d}.npz").exists()]
T = dict(np.load(F300 / "_tab.npz"))
ALPHA = np.arange(0.04, 0.97, 0.02)
dom = np.array([b["dataset"] for b in rows])
cls = np.array([f"{b['row']['fold']}_{b['row']['global_class_id']}" for b in rows])
half = np.zeros(len(rows), bool)
for d in lab.DOMAINS:
    idx = np.where(dom == d)[0]
    half[idx[::2]] = True
n = np.arange(len(rows))


def miou(I, U, sel):
    out = []
    for d in lab.DOMAINS:
        m = sel & (dom == d)
        out.append(np.mean([100 * I[m & (cls == c)].sum() / max(U[m & (cls == c)].sum(), 1) for c in np.unique(cls[m])]))
    return np.array(out)


def show(tag, v):
    print(f"{tag:46s}" + "".join(f"{x:8.1f}" for x in v) + f"{v.mean():8.1f}")


print(f"class-pooled mIoU on report half (n={int((~half).sum())}); no CRF")
print(f"{'':46s}" + "".join(f"{d[:7]:>8s}" for d in lab.DOMAINS) + "    mean")
show("FoRIS complete output (CRF)", miou(T["fI"], T["fU"], ~half))
for k, name in ((0, "proto"), (2, "foris")):
    show(f"{name} | oracle level", miou(T["oI"][:, k], T["oU"][:, k], ~half))
    best = max(range(len(ALPHA)), key=lambda j: miou(T["I"][n, k, j], T["U"][n, k, j], half).mean())
    show(f"{name} | const alpha={ALPHA[best]:.2f}", miou(T["I"][n, k, best], T["U"][n, k, best], ~half))
    picks, corr = {}, lab.defaultdict(list)
    for e, b in enumerate(rows):
        z = np.load(F300 / f"{b['index']:06d}.npz")
        bd = np.load(B300 / f"{b['index']:06d}.npz")
        s = z["sq"] if name == "proto" else z["foris_score"].reshape(-1)
        sn = (s - s.min()) / max(s.max() - s.min(), 1e-6)
        S = sn[None, :] > ALPHA[:, None]                                   # (levels, tokens)
        size = np.maximum(S.sum(1), 1)
        iou = T["I"][e, k] / np.maximum(T["U"][e, k], 1)
        for tau in (0.02, 0.05):
            a = (S * bd[f"fwdF{tau}"][None]).sum(1) / size
            c = (S * bd[f"backF{tau}"][None]).sum(1)
            cb = (S * bd[f"backB{tau}"][None]).sum(1)
            crit = {
                f"IoU-form t{tau}": a * c / np.maximum(a + c - a * c, 1e-9),
                f"F1 t{tau}": 2 * a * c / np.maximum(a + c, 1e-9),
                f"capture-leak t{tau}": c - cb,                             # FG capture minus BG capture
                f"IoU-form(c, 1-cb) t{tau}": c * (1 - cb) / np.maximum(c + (1 - cb) - c * (1 - cb), 1e-9),
            }
            for cn, v in crit.items():
                picks.setdefault(cn, []).append(int(np.argmax(v)))
                if np.std(v) > 0 and np.std(iou) > 0:
                    corr[cn].append(np.corrcoef(v, iou)[0, 1])
    for cn, p in picks.items():
        p = np.array(p)
        show(f"{name} | {cn}  (r={np.mean(corr[cn]):+.2f})", miou(T["I"][n, k, p], T["U"][n, k, p], ~half))
