"""Offline report for the whole-image pyramid dump."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
OUT = Path(sys.argv[1])
ext = len(sys.argv) > 3
rows = lab.ext_subset(int(sys.argv[2]), tuple(sys.argv[3].split(","))) if ext else lab.subset(int(sys.argv[2]), offset=1)
key = lambda b: b["index"] if isinstance(b["index"], str) else format(b["index"], "06d")
rows = [b for b in rows if (OUT / f"{key(b)}.npz").exists()]
DOMS = tuple(dict.fromkeys(b["dataset"] for b in rows))
S = (256, 512, 768, 1024)
print("episodes", len(rows))


def up(x):
    g = int(round(x.size ** 0.5))
    return F.interpolate(torch.from_numpy(x).reshape(1, 1, g, g).float(), size=(1024, 1024), mode="bilinear", align_corners=False)[0, 0]


M, A, by = {}, {}, lab.defaultdict(list)
for b in rows:
    d = b["dataset"]
    z = np.load(OUT / f"{key(b)}.npz")
    gt = lab.mask1024(b["row"]["query_mask_path"])
    ig = lab.ignore1024(b)
    m = {f"r{a}_q{q}": up(z[f"r{a}_q{q}"]) for a in S for q in S}
    fused = {
        "diag mean (256..1024)": sum(m[f"r{s}_q{s}"] for s in S) / 4,
        "diag mean (512..1024)": sum(m[f"r{s}_q{s}"] for s in S[1:]) / 3,
        "diag mean (512,1024)": (m["r512_q512"] + m["r1024_q1024"]) / 2,
        "all-pairs mean": sum(m.values()) / 16,
        "q-mean @ r1024": sum(m[f"r1024_q{s}"] for s in S) / 4,
        "r-mean @ q1024": sum(m[f"r{s}_q1024"] for s in S) / 4,
        "diag max (512..1024)": torch.stack([m[f"r{s}_q{s}"] for s in S[1:]]).max(0).values,
    }
    show = {f"r{s}_q{s}": m[f"r{s}_q{s}"] for s in S}
    show.update({"r1024_q512": m["r1024_q512"], "r512_q1024": m["r512_q1024"], "r256_q1024": m["r256_q1024"], "r1024_q256": m["r1024_q256"]})
    show.update(fused)
    aq = gt.float().mean().item()
    rec = {}
    for name, Sx in show.items():
        g = gt
        if ig is not None:
            Sx = torch.where(ig, Sx.min(), Sx)
            g = gt & ~ig
        i, u, _ = lab.best_threshold_iou(Sx, g)
        M.setdefault(name + "|oracle", lab.Meter()).add(b, i, u)
        sn = (Sx - Sx.min()) / (Sx.max() - Sx.min()).clamp_min(1e-6)
        for a in (0.5, 0.6, 0.7):
            M.setdefault(name + f"|mm{a}", lab.Meter()).add(b, *lab.iu(sn > a, g))
        A.setdefault(name, lab.defaultdict(list))[d].append(lab.auroc(Sx, g, stride=5))
        rec[name] = 100 * i / max(u, 1)
    by[d].append((aq, rec))
    p = lab.foris_cli(b)
    g = gt
    if ig is not None:
        p, g = p & ~ig, gt & ~ig
    M.setdefault("FoRIS_crf_output", lab.Meter()).add(b, *lab.iu(p, g))

print(f"class-pooled mIoU (then AUROC per domain for oracle rows)")
print(f"{'map|rule':30s}" + "".join(f"{d[:11]:>12s}" for d in DOMS) + "   mean")
for name in M:
    r = M[name].row(DOMS)
    base = name.split("|")[0]
    extra = "".join(f" {np.nanmean(A[base][d]):.3f}" for d in DOMS) if name.endswith("oracle") else ""
    if name.endswith("mm0.5") and not name.startswith("r1024_q1024"):
        continue
    print(f"{name:30s}" + "".join(f"{r[d][0]:12.1f}" for d in DOMS) + f"{np.mean([r[d][0] for d in DOMS]):7.1f}  {extra}")

print("\nper-episode oracle-level IoU by query target area: r1024_q1024 | r512_q512 | r256_q256 | diag mean(512..1024)  (n)")
for d in DOMS:
    line = f"{d:12s}"
    for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.01)):
        sel = [r for (aq, r) in by[d] if lo <= aq < hi]
        if sel:
            line += f"  <{hi:.2f} " + " ".join(f"{np.mean([r[k] for r in sel]):4.0f}" for k in ("r1024_q1024", "r512_q512", "r256_q256", "diag mean (512..1024)")) + f" ({len(sel)})"
    print(line)
