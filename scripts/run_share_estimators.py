#!/usr/bin/env python3
"""Label-free estimators of the object's share of the query image that do not read the score field (GPU).

The IoU-optimal cut of a field whose class-conditional distributions are fixed depends on one unknown, the object's
share of the query. Estimators here use only the reference mask and token correspondences:
  ref_area      the reference object's share of the reference image
  scale_*       the scale change from reference to query, from mutual nearest neighbours of reference-object tokens:
                median ratio of pairwise distances (scale_pair), ratio of coordinate spreads (scale_spread),
                and the spread of the nearest query token of every reference-object token (scale_fwd)
  nn_share      share of query tokens whose nearest reference token is on the object
Saved per episode for offline scoring. Reads no query truth.

  python scripts/run_share_estimators.py --root outputs/fresh600_root --run outputs/recheck_fresh600_v1 --out outputs/claude_share_fresh600
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda"); rows = json.loads((a.run / "manifest.json").read_text()); out = {k: [] for k in ("ref_area", "pairs", "scale_pair", "scale_spread", "scale_fwd", "nn_share", "mutual_share")}
    yy, xx = torch.meshgrid(torch.arange(64, device=dev), torch.arange(64, device=dev), indexing="ij"); pos = torch.stack([yy.flatten(), xx.flatten()], 1).float()
    spread = lambda x: float(x.var(0, unbiased=False).sum().sqrt()) if len(x) > 1 else float("nan")
    for r in rows:
        feat = torch.load(a.root / "cache/evidence_v1/feat" / (r["key"] + ".pt")); q, ref = (F.normalize(feat[k].to(dev).float(), dim=1) for k in ("q", "r"))
        with np.load(a.root / "results/extent_v1/run/packets" / (r["key"] + ".npz"), allow_pickle=False) as z:
            cov = torch.from_numpy(z["cov"]).to(dev).flatten()
        fg = cov >= .5
        if not fg.any():
            fg = cov == cov.max()
        sim = q @ ref.T; q2r = sim.argmax(1); r2q = sim.argmax(0)
        j = fg.nonzero().flatten(); j = j[q2r[r2q[j]] == j]; i = r2q[j]                # mutual pairs with the reference token on the object
        pq, pr = pos[i], pos[j]; s_pair = float("nan")
        if len(j) >= 3:
            dq, dr = torch.cdist(pq, pq), torch.cdist(pr, pr); ok = torch.triu(dr >= 2, 1)
            if ok.any():
                s_pair = float((dq[ok] / dr[ok]).median())
        out["ref_area"].append(float(cov.mean())); out["pairs"].append(len(j)); out["scale_pair"].append(s_pair)
        out["scale_spread"].append(spread(pq) / max(spread(pr), 1e-6) if len(j) >= 3 else float("nan"))
        out["scale_fwd"].append(spread(pos[r2q[fg]]) / max(spread(pos[fg]), 1e-6) if int(fg.sum()) >= 3 else float("nan"))
        out["nn_share"].append(float(fg[q2r].float().mean())); out["mutual_share"].append(len(j) / max(int(fg.sum()), 1))
    a.out.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out / "estimators.npz", **{k: np.array(v) for k, v in out.items()}); print("SHARE_DONE", flush=True)


if __name__ == "__main__":
    main()
