"""W04: are the failures FoRIS and the independent candidate share ranking failures or cut failures?

Dev1360, cached fields only.  Per episode, for three score fields (complete FoRIS score64 before
CRF, kernel posterior c, candidate field VP_UN): natural min-max>.5 IoU, GT-optimal single-level
IoU (diagnostic), token-grid ceiling, and pixel AUROC.  Query GT is used only for these diagnostics.

    python scripts/x1012/w04_failure_anatomy.py OUT.jsonl
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
out = open(sys.argv[1], "w")
bits = lambda a: torch.from_numpy(np.unpackbits(a)[: 1024 * 1024].reshape(1024, 1024).astype(bool))
for kk, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    ig = lab.ignore1024(b)
    keep = None if ig is None else ~ig.to(DEV)
    g = gt if keep is None else gt & keep
    iu = lambda p: lab.iu(p if keep is None else p & keep, g)
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.unit(lab.feats(b, "reference"))), lab.debias(lab.unit(lab.feats(b, "query")))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    c = torch.softmax(lab.unit(q - mu) @ lab.unit(r - mu).T / 0.05, dim=1) @ y
    z = np.load(b["score64"]["path"])
    fl = np.load(R1 / "fields" / f"{key(b)}.npz")
    F = {"foris": torch.from_numpy(z[b["score64"]["key"]].astype(np.float32)).reshape(64, 64).to(DEV),
         "c": c.reshape(64, 64), "un": torch.from_numpy(fl["VP_UN"].astype(np.float32)).to(DEV)}
    rec = {"index": key(b), "dataset": b["dataset"], "fold": b["row"]["fold"], "cls": b["row"]["global_class_id"],
           "gt": int(g.sum()), "rho": float(fl["reference_rho"][0]), "ref": float(y.mean())}
    rec["foris_final"] = [int(v) for v in iu(lab.foris_cli(b).to(DEV))]
    rec["un_final"] = [int(v) for v in iu(bits(np.load(R1 / "predictions" / f"{key(b)}.npz")["VP_UN"]).to(DEV))]
    # token-grid ceiling: GT coverage on the 64 grid, bilinear back, > .5
    rec["grid"] = [int(v) for v in iu(lab.up(lab.coverage64(gt)) > 0.5)]
    for k, f in F.items():
        u = lab.up(f)
        rec[k + "_nat"] = [int(v) for v in iu(lab.minmax(u) > 0.5)]
        io, uo, t = lab.best_threshold_iou(u if keep is None else u.masked_fill(~keep, float(u.min()) - 1), g)
        rec[k + "_orc"] = [int(io), int(uo)]
        mm = lab.minmax(u)
        rec[k + "_t"] = float((t - float(u.min())) / max(float(u.max() - u.min()), 1e-9))
        rec[k + "_auc"] = float(lab.auroc(u, g))
    out.write(json.dumps(rec) + "\n")
    lab.free()
    if kk % 200 == 0:
        print(kk, len(rows), flush=True)
print("done", flush=True)
