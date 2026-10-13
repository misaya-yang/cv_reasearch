"""Offline report for the two-scale dump: ranking and constant-level IoU of each view pairing."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
OUT = Path(sys.argv[1])
rows = [b for b in lab.subset(int(sys.argv[2]), offset=2) if (OUT / f"{b['index']:06d}.npz").exists()]
print("episodes", len(rows))


def up(x, size=1024):
    g = int(round(x.numel() ** 0.5))
    return F.interpolate(x.reshape(1, 1, g, g).float(), size=(size, size), mode="bilinear", align_corners=False)[0, 0]


def z(x):
    return (x - x.mean()) / x.std().clamp_min(1e-6)


M, A, by = {}, {}, lab.defaultdict(list)
for b in rows:
    d = b["dataset"]
    f = np.load(OUT / f"{b['index']:06d}.npz")
    gt = lab.mask1024(b["row"]["query_mask_path"])
    m = {k: up(torch.from_numpy(f[k])) for k in ("gg", "fg", "ff", "gf", "ff_own")}
    m["mean(gg,fg)"] = 0.5 * (m["gg"] + m["fg"])
    m["max(gg,fg)"] = torch.maximum(m["gg"], m["fg"])
    m["mean(gg,gf)"] = 0.5 * (m["gg"] + m["gf"])
    m["mean(all4)"] = 0.25 * (m["gg"] + m["fg"] + m["ff"] + m["gf"])
    m["max(all4)"] = torch.stack([m["gg"], m["fg"], m["ff"], m["gf"]]).max(0).values
    m["mean(gg,fg,ff)"] = (m["gg"] + m["fg"] + m["ff"]) / 3
    aq, ar = gt.float().mean().item(), float(f["y64"].mean())
    rec = {}
    for name, S in m.items():
        i, u, _ = lab.best_threshold_iou(S, gt)
        M.setdefault(name + "|oracle", lab.Meter()).add(b, i, u)
        sn = (S - S.min()) / (S.max() - S.min()).clamp_min(1e-6)
        for a in (0.5, 0.6, 0.7):
            M.setdefault(name + f"|mm{a}", lab.Meter()).add(b, *lab.iu(sn > a, gt))
        au = lab.auroc(S, gt, stride=5)
        A.setdefault(name, lab.defaultdict(list))[d].append(au)
        rec[name] = (au, 100 * i / max(u, 1))
    by[d].append((np.log2(max(aq, 1e-5) / max(ar, 1e-5)), aq, rec))
    M.setdefault("res_ceiling64", lab.Meter()).add(b, *lab.iu(up(lab.coverage64(gt).flatten()) > 0.5, gt))
    M.setdefault("res_ceiling128", lab.Meter()).add(b, *lab.iu(up(F.avg_pool2d(gt[None, None].float(), 8).flatten()) > 0.5, gt))
    M.setdefault("FoRIS_crf_output", lab.Meter()).add(b, *lab.iu(lab.foris_cli(b), gt))

print(f"class-pooled mIoU; AUROC for |oracle rows; n={len(rows)}")
print(f"{'map|rule':24s}" + "".join(f"{d:>14s}" for d in lab.DOMAINS) + "   mean")
for name in M:
    r = M[name].row()
    base = name.split("|")[0]
    extra = "".join(f" {np.nanmean(A[base][d]):.3f}" for d in lab.DOMAINS) if name.endswith("oracle") else ""
    print(f"{name:24s}" + "".join(f"{r[d][0]:14.1f}" for d in lab.DOMAINS) + f"{np.mean([r[d][0] for d in lab.DOMAINS]):7.1f}  {extra}")

print("\nper-episode oracle-level IoU by scale relation log2(query area / reference area): gg | fg | gf | ff   (n)")
for d in lab.DOMAINS:
    line = f"{d:12s}"
    for lo, hi in ((-99, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 99)):
        sel = [r for (mis, aq, r) in by[d] if lo <= mis < hi]
        if sel:
            line += f"  [{lo:+.1f},{hi:+.1f}) " + " ".join(f"{np.mean([r[k][1] for r in sel]):4.0f}" for k in ("gg", "fg", "gf", "ff")) + f" ({len(sel)})"
    print(line)
print("\nper-episode oracle-level IoU by query target area: gg | fg | mean(gg,fg) | ff   (n)")
for d in lab.DOMAINS:
    line = f"{d:12s}"
    for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.01)):
        sel = [r for (mis, aq, r) in by[d] if lo <= aq < hi]
        if sel:
            line += f"  <{hi:.2f} " + " ".join(f"{np.mean([r[k][1] for r in sel]):4.0f}" for k in ("gg", "fg", "mean(gg,fg)", "ff")) + f" ({len(sel)})"
    print(line)
