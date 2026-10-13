"""Headroom split on existing caches: resolution ceiling, ranking ceiling, extent."""
import sys, time
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import lab
from lab import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rows = lab.subset(N)
names = ["foris", "res_ceiling", "proto_minmax", "proto_oracle", "proto_raw_oracle", "foris_score_oracle", "nn1_vote"]
M = {k: Meter() for k in names}
auc = {k: lab.defaultdict(list) for k in ("proto", "proto_raw", "foris_score")}


def auroc(score, gt):
    s, g = score.flatten(), gt.flatten()
    idx = torch.randperm(s.numel(), device=s.device)[:60000]
    s, g = s[idx], g[idx]
    if g.all() or not g.any():
        return float("nan")
    r = torch.argsort(torch.argsort(s)).float()
    n1, n0 = g.sum().float(), (~g).sum().float()
    return float((r[g].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


t0 = time.time()
for k, b in enumerate(rows):
    d = b["dataset"]
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    mr = lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)
    cr = lab.coverage64(mr).flatten()
    M["foris"].add(b, *iu(lab.foris_cli(b).to(DEV), gt))
    M["res_ceiling"].add(b, *iu(up(lab.coverage64(gt)) > 0.5, gt))
    R, Q = lab.feats(b, "reference"), lab.feats(b, "query")
    for tag, f in (("proto", lab.debias), ("proto_raw", lab.unit)):
        r, q = f(R), f(Q)
        w = cr.clone()
        if w.sum() < 1e-6:
            w[cr.argmax()] = 1
        mu = lab.unit((r * w[:, None]).sum(0))
        s = up((q @ mu).reshape(64, 64))
        i, u, _ = lab.best_threshold_iou(s, gt)
        M[tag + "_oracle"].add(b, i, u)
        auc[tag][d].append(auroc(s, gt))
        if tag == "proto":
            sn = (s - s.min()) / (s.max() - s.min()).clamp_min(1e-6)
            M["proto_minmax"].add(b, *iu(sn > 0.5, gt))
            nn = (q @ r.T).argmax(1)
            M["nn1_vote"].add(b, *iu(up((cr[nn] > 0.5).float().reshape(64, 64)) > 0.5, gt))
    fs = torch.from_numpy(np.load(b["score64"]["path"])["score"]).to(DEV)
    s = up(fs)
    i, u, _ = lab.best_threshold_iou(s, gt)
    M["foris_score_oracle"].add(b, i, u)
    auc["foris_score"][d].append(auroc(s, gt))
    if k % 100 == 0:
        print(k, round(time.time() - t0, 1), flush=True)

print(f"\nclass-pooled mIoU / per-episode mean IoU, CLI-1024, n={N} per domain")
print(f"{'arm':22s}" + "".join(f"{d:>18s}" for d in DOMAINS))
for k in names:
    r = M[k].row()
    print(f"{k:22s}" + "".join(f"{r[d][0]:9.2f}{r[d][1]:9.2f}" for d in DOMAINS))
for k in auc:
    print(f"AUROC {k:16s}" + "".join(f"{np.nanmean(auc[k][d]):18.4f}" for d in DOMAINS))
