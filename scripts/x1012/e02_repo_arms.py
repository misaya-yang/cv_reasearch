"""Score every cached repository arm against cached FoRIS on all main dev episodes (CLI-1024, class-pooled mIoU)."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rows = lab.bindings()[::STEP]
ARMS = ("B", "R", "G", "N", "C", "ownK", "source_budget")
M = {a: lab.Meter() for a in ARMS}
M["best-of-arms per episode (GT pick)"] = lab.Meter()
M["best of {B, R} per episode (GT pick)"] = lab.Meter()
cache = {}
t0 = time.time()
for k, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"])
    res = {}
    for a in ARMS:
        e = b["old_arms"][a]
        z = np.load(e["path"])
        p = torch.from_numpy(np.unpackbits(z[e["cli_key"]])[: 1024 * 1024].reshape(1024, 1024).astype(bool))
        res[a] = lab.iu(p, gt)
        M[a].add(b, *res[a])
    j = lambda t: t[0] / max(t[1], 1)
    M["best-of-arms per episode (GT pick)"].add(b, *max(res.values(), key=j))
    M["best of {B, R} per episode (GT pick)"].add(b, *max((res["B"], res["R"]), key=j))
    if k % 1000 == 0:
        print(k, len(rows), round(time.time() - t0, 1), flush=True)
cnt = {d: sum(r["dataset"] == d for r in rows) for d in lab.DOMAINS}
print("\nclass-pooled mIoU (per-episode mean IoU); episodes " + ", ".join(f"{d}={n}" for d, n in cnt.items()))
print(f"{'arm':40s}" + "".join(f"{d[:11]:>16s}" for d in lab.DOMAINS))
names = {"B": "B  FoRIS complete", "R": "R  anchor.global (whole ridge)", "G": "G  mean.archived", "N": "N  supported_mean",
         "C": "C  shared_edits_complete", "ownK": "ownK own_margin_sameK", "source_budget": "source_budget"}
for a, m in M.items():
    r = m.row(lab.DOMAINS)
    print(f"{names.get(a, a):40s}" + "".join(f"{r[d][0]:8.2f} ({r[d][1]:5.1f})" for d in lab.DOMAINS))
