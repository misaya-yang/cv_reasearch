"""W07: is the candidate behind FoRIS on COCO / ISIC because of ranking or because of the cut?

2x2 on dev1360, cached fields only: ranking in {complete FoRIS score64, candidate field} x
area in {FoRIS final mask area, candidate final mask area}; each cell = top-k pixels of the
ranking with k = that area.  Diagnostic only: the cross cells consume the other method's output.
Candidate field = VP_UN, or c when rho > .9 (the frozen fallback).

    python scripts/x1012/w07_rank_cut_cross.py OUT.jsonl
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import DEV

R1 = lab.ASSETS / "a/x1012_R1"
rows = json.loads((R1 / "manifest.json").read_text())
key = lambda b: f"{b['index']:06d}" if isinstance(b["index"], int) else str(b["index"])
bits = lambda a: torch.from_numpy(np.unpackbits(a)[: 1024 * 1024].reshape(1024, 1024).astype(bool))
out = open(sys.argv[1], "w")


def topk(field, k):
    if k <= 0:
        return torch.zeros_like(field, dtype=torch.bool)
    t = torch.topk(field.flatten(), min(k, field.numel())).values[-1]
    return field >= t


for kk, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    ig = lab.ignore1024(b)
    keep = torch.ones_like(gt) if ig is None else ~ig.to(DEV)
    g = gt & keep
    fl = np.load(R1 / "fields" / f"{key(b)}.npz")
    rho = float(fl["reference_rho"][0])
    if rho > 0.9:
        y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
        r, q = lab.debias(lab.unit(lab.feats(b, "reference"))), lab.debias(lab.unit(lab.feats(b, "query")))
        mu = 0.5 * (r.mean(0) + q.mean(0))
        fu = (torch.softmax(lab.unit(q - mu) @ lab.unit(r - mu).T / 0.05, dim=1) @ y).reshape(64, 64)
    else:
        fu = torch.from_numpy(fl["VP_UN"].astype(np.float32)).to(DEV)
    z = np.load(b["score64"]["path"])
    ff = torch.from_numpy(z[b["score64"]["key"]].astype(np.float32)).reshape(64, 64).to(DEV)
    F = {"F": lab.up(ff).masked_fill(~keep, -1e9), "U": lab.up(fu).masked_fill(~keep, -1e9)}
    M = {"F": lab.foris_cli(b).to(DEV) & keep, "U": bits(np.load(R1 / "predictions" / f"{key(b)}.npz")["VP_UN"]).to(DEV) & keep}
    rec = {"index": key(b), "dataset": b["dataset"], "fold": b["row"]["fold"], "cls": b["row"]["global_class_id"], "gt": int(g.sum())}
    for a in "FU":
        rec[f"final{a}"] = [int(v) for v in lab.iu(M[a], g)]
        for r_ in "FU":
            rec[f"rank{r_}_area{a}"] = [int(v) for v in lab.iu(topk(F[r_], int(M[a].sum())), g)]
    rec["rankF_gt"] = [int(v) for v in lab.iu(topk(F["F"], int(g.sum())), g)]
    rec["rankU_gt"] = [int(v) for v in lab.iu(topk(F["U"], int(g.sum())), g)]
    out.write(json.dumps(rec) + "\n")
    lab.free()
    if kk % 300 == 0:
        print(kk, len(rows), flush=True)
out.close()
E = [json.loads(l) for l in open(sys.argv[1])]
D = ["coco", "lvis", "pascal_part", "paco_part", "suim", "isaid", "isic", "lung", "fundus", "deepglobe_road"]


def miou(L, a):
    I, U = {}, {}
    for e in L:
        k = (e["fold"], e["cls"]); I[k] = I.get(k, 0) + e[a][0]; U[k] = U.get(k, 0) + e[a][1]
    return float(np.mean([100 * I[k] / max(U[k], 1) for k in I]))


cols = ["finalF", "finalU", "rankF_areaF", "rankU_areaF", "rankF_areaU", "rankU_areaU", "rankF_gt", "rankU_gt"]
print("class-pooled mIoU, dev1360; rankX_areaY = top-k of X's field with k = Y's final area; _gt = k = true area (diagnostic)")
print(f"{'':10s}" + "".join(f"{c:>13s}" for c in cols) + "   ranking effect | cut effect")
rng = np.random.default_rng(0)
for d in D + ["mean"]:
    if d == "mean":
        v = np.mean([[miou([e for e in E if e["dataset"] == x], c) for c in cols] for x in D], 0)
    else:
        L = [e for e in E if e["dataset"] == d]; v = [miou(L, c) for c in cols]
    rank = 0.5 * ((v[3] - v[2]) + (v[5] - v[4])); cut = 0.5 * ((v[4] - v[2]) + (v[5] - v[3]))
    print(f"{d[:10]:10s}" + "".join(f"{x:13.1f}" for x in v) + f"   {rank:+6.1f} (ours-FoRIS) | {cut:+6.1f} (our area - FoRIS area)")
