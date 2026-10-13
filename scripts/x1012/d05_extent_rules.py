"""Extent rules on a fixed score field (cached FoRIS pre-binarise score64), CPU only.

Question: with ranking held fixed, how much of the GT-optimal-threshold headroom do
label-free level-set rules recover?  All rules read only the query score field.
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rows = lab.subset(N)


def otsu(s, bins=128):
    h = torch.histc(s, bins=bins, min=float(s.min()), max=float(s.max()))
    p = h / h.sum()
    c = torch.linspace(float(s.min()), float(s.max()), bins)
    w0 = torch.cumsum(p, 0)
    m0 = torch.cumsum(p * c, 0)
    mt = m0[-1]
    var = (mt * w0 - m0) ** 2 / (w0 * (1 - w0)).clamp_min(1e-9)
    return float(c[var.argmax()])


def otsu_multi(s, bins=128):
    """Three-class Otsu; return the upper threshold (foreground = top class)."""
    h = torch.histc(s, bins=bins, min=float(s.min()), max=float(s.max())).double()
    p = h / h.sum()
    c = torch.linspace(float(s.min()), float(s.max()), bins).double()
    P = torch.cumsum(p, 0)
    S = torch.cumsum(p * c, 0)
    best, arg = -1.0, (0, 0)
    for i in range(4, bins - 8, 2):
        j = torch.arange(i + 2, bins - 2, 2)
        w0, w1, w2 = P[i], P[j] - P[i], 1 - P[j]
        m0, m1, m2 = S[i] / w0.clamp_min(1e-9), (S[j] - S[i]) / w1.clamp_min(1e-9), (S[-1] - S[j]) / w2.clamp_min(1e-9)
        v = w0 * m0 ** 2 + w1 * m1 ** 2 + w2 * m2 ** 2
        k = int(v.argmax())
        if float(v[k]) > best:
            best, arg = float(v[k]), (i, int(j[k]))
    return float(c[arg[0]]), float(c[arg[1]])


def stable(s, lo=0.25, hi=0.8, n=56):
    """MSER-style: level in [lo, hi] of the robust range where relative area changes least."""
    a, b = torch.quantile(s, 0.01), torch.quantile(s, 0.999)
    ts = a + (b - a) * torch.linspace(lo, hi, n)
    area = torch.stack([(s > t).float().mean() for t in ts]).clamp_min(1e-6)
    d = (area[:-4] - area[4:]) / area[2:-2]
    return float(ts[2:-2][d.argmin()])


names = ["foris_out", "minmax.5", "robust.5", "otsu", "otsu3_hi", "otsu3_lo", "stable", "oracle"]
M = {k: lab.Meter() for k in names}
for b in rows:
    gt = lab.mask1024(b["row"]["query_mask_path"])
    s64 = torch.from_numpy(np.load(b["score64"]["path"])["score"]).float()
    s = lab.up(s64)
    f = s.flatten()[::16]
    M["foris_out"].add(b, *lab.iu(lab.foris_cli(b), gt))
    M["minmax.5"].add(b, *lab.iu(s > 0.5 * (s.min() + s.max()), gt))
    a, c = torch.quantile(f, 0.01), torch.quantile(f, 0.995)
    M["robust.5"].add(b, *lab.iu(s > 0.5 * (a + c), gt))
    M["otsu"].add(b, *lab.iu(s > otsu(f), gt))
    t1, t2 = otsu_multi(f)
    M["otsu3_hi"].add(b, *lab.iu(s > t2, gt))
    M["otsu3_lo"].add(b, *lab.iu(s > t1, gt))
    M["stable"].add(b, *lab.iu(s > stable(f), gt))
    i, u, _ = lab.best_threshold_iou(s, gt, n=48)
    M["oracle"].add(b, i, u)

print(f"class-pooled mIoU, CLI-1024, no CRF except foris_out; n={N}/domain")
print(f"{'rule':12s}" + "".join(f"{d:>13s}" for d in lab.DOMAINS))
for k in names:
    r = M[k].row()
    print(f"{k:12s}" + "".join(f"{r[d][0]:13.2f}" for d in lab.DOMAINS))
