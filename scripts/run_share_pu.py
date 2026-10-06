#!/usr/bin/env python3
"""The object's share of the query as a mixture proportion: positives from the reference, the query as the unlabelled mixture (GPU).

Query tokens are a mixture pi * P1 + (1 - pi) * P0, with P1 the distribution of object tokens (sampled by the reference
object) and P0 unknown. For any region B of feature space, Q(B) / P1(B) >= pi, with equality where the background puts no
mass, so pi is the infimum of that ratio (mixture proportion estimation, Blanchard et al. 2010; Ramaswamy et al. 2016).
Here B_j is the cosine ball of radius tau around reference-object token j:
  ratio_j = (share of query tokens in B_j) / (share of reference-object tokens in B_j)
and the estimate is a low quantile of ratio_j over j. Saved: quantiles for every tau. Reads no query truth and no score field.

  python scripts/run_share_pu.py --root outputs/fresh600_root --run outputs/recheck_fresh600_v1 --out outputs/claude_share_pu_fresh600
"""
import argparse
import json
from pathlib import Path

import numpy as np

TAUS, QS = (0.4, 0.5, 0.6, 0.7, 0.8), (0., 0.05, 0.1, 0.25, 0.5, 0.75)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda"); rows = json.loads((a.run / "manifest.json").read_text()); out, nfg = [], []
    qs = torch.tensor(QS, device=dev)
    for r in rows:
        feat = torch.load(a.root / "cache/evidence_v1/feat" / (r["key"] + ".pt")); q, ref = (F.normalize(feat[k].to(dev).float(), dim=1) for k in ("q", "r"))
        with np.load(a.root / "results/extent_v1/run/packets" / (r["key"] + ".npz"), allow_pickle=False) as z:
            cov = torch.from_numpy(z["cov"]).to(dev).flatten()
        fg = cov >= .9
        if not fg.any():
            fg = cov == cov.max()
        pos = ref[fg]; sq, sp = pos @ q.T, pos @ pos.T; row = []
        for t in TAUS:
            ratio = (sq >= t).float().mean(1) / (sp >= t).float().mean(1)       # the token itself keeps the denominator positive
            row.append(torch.quantile(ratio, qs).cpu().numpy())
        out.append(np.stack(row)); nfg.append(int(fg.sum()))
    a.out.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out / "pu.npz", ratio=np.stack(out), taus=np.array(TAUS), quantiles=np.array(QS), reference_tokens=np.array(nfg)); print("PU_DONE", flush=True)


if __name__ == "__main__":
    main()
