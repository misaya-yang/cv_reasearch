"""W05: is reference/query scale mismatch a cause of the ranking failures FoRIS and the candidate share?

Reuses the d21 pyramid dump (300 dev episodes, COCO/LVIS/PASCAL/PACO/SUIM): contrast-prototype
score maps for every (reference encoding size, query encoding size).  The query stays at 1024;
only the reference encoding size changes.  Scale ratio = log2(reference target width / query
target width) from W03 geometry (query width uses GT: diagnostic).  No encoder call.

Pre-stated: P1 when the reference target is >=2x wider than the query target, a down-scaled
reference (512 or 256) ranks the query better than the 1024 reference; P2 at matched scale the
1024 reference is best.  If P1 fails, scale mismatch is a correlate, not a usable cause.

    python scripts/x1012/w05_scale_mismatch.py ../cv_data/a/x1012/pyramid60
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab
from w03_reference_regime import geom

lab.DEV = torch.device("cpu")
torch.set_num_threads(6)
OUT = Path(sys.argv[1])
key = lambda b: b["index"] if isinstance(b["index"], str) else format(b["index"], "06d")
rows = [b for b in lab.subset(60, offset=1) if (OUT / f"{key(b)}.npz").exists()]
S = (256, 512, 768, 1024)
up = lambda x: F.interpolate(torch.from_numpy(x).reshape(1, 1, 64, 64).float(), size=(1024, 1024), mode="bilinear", align_corners=False)[0, 0]
recs = []
for b in rows:
    z = np.load(OUT / f"{key(b)}.npz")
    gt = lab.mask1024(b["row"]["query_mask_path"])
    rw = geom(lab.mask1024(b["row"]["reference_mask_path"]).numpy())[1]
    qa, qw, _ = geom(gt.numpy())
    if qw <= 0 or rw <= 0:
        continue
    m = {s: up(z[f"r{s}_q1024"]) for s in S}
    m["max"] = torch.stack([m[s] for s in S]).max(0).values
    m["max512+"] = torch.stack([m[s] for s in S[1:]]).max(0).values
    m["mean"] = torch.stack([m[s] for s in S]).mean(0)
    r = {"d": b["dataset"], "fold": b["row"]["fold"], "cls": b["row"]["global_class_id"], "sr": float(np.log2(rw / qw)), "qw": qw, "rw": rw}
    for k, x in m.items():
        i, u, _ = lab.best_threshold_iou(x, gt)
        r[f"o{k}"] = (int(i), int(u))
        r[f"a{k}"] = float(lab.auroc(x, gt, stride=5))
        r[f"n{k}"] = tuple(int(v) for v in lab.iu(lab.minmax(x) > 0.6, gt))
    recs.append(r)
print("episodes", len(recs))
K = list(S) + ["max512+", "max", "mean"]
BINS = [("ref >=4x wider", 2, 99), ("ref 2-4x wider", 1, 2), ("ref 1.4-2x wider", .5, 1), ("matched", -.5, .5), ("ref 1.4-2x narrower", -1, -.5), ("ref >=2x narrower", -99, -1), ("ALL", -99, 99)]
pool = lambda L, f: 100 * sum(r[f][0] for r in L) / max(sum(r[f][1] for r in L), 1)
ep = lambda L, f: np.mean([100 * r[f][0] / max(r[f][1], 1) for r in L])
for title, fn in (("AUROC (mean over episodes)", lambda L, k: np.mean([r[f"a{k}"] for r in L])),
                  ("GT-optimal-level IoU, per-episode mean (diagnostic)", lambda L, k: ep(L, f"o{k}")),
                  ("natural min-max>.6 IoU, per-episode mean", lambda L, k: ep(L, f"n{k}"))):
    print(f"\n=== {title}; columns = reference encoding size (query fixed at 1024)")
    print(f"{'scale ratio':22s}{'n':>5s}" + "".join(f"{str(k):>9s}" for k in K) + "   best single")
    for lab_, lo, hi in BINS:
        L = [r for r in recs if lo <= r["sr"] < hi]
        if len(L) < 8:
            continue
        v = [fn(L, k) for k in K]
        f = "{:9.3f}" if "AUROC" in title else "{:9.1f}"
        print(f"{lab_:22s}{len(L):5d}" + "".join(f.format(x) for x in v) + f"   {S[int(np.argmax(v[:4]))]}")
print("\n=== per-episode best reference size (by AUROC) vs scale ratio: share of episodes where each size is best")
for lab_, lo, hi in BINS[:-1]:
    L = [r for r in recs if lo <= r["sr"] < hi]
    if len(L) < 8:
        continue
    best = [S[int(np.argmax([r[f"a{s}"] for s in S]))] for r in L]
    gain = np.mean([max(r[f"a{s}"] for s in S) - r["a1024"] for r in L])
    print(f"{lab_:22s}{len(L):5d}  " + "  ".join(f"{s}:{100 * np.mean([x == s for x in best]):3.0f}%" for s in S) + f"   mean AUROC gain of best size over 1024: {gain:+.3f}")
print("corr(scale ratio, log2(1024 / best size by AUROC)) =", round(float(np.corrcoef([r["sr"] for r in recs], [np.log2(1024 / S[int(np.argmax([r[f'a{s}'] for s in S]))]) for r in recs])[0, 1]), 3))
