#!/usr/bin/env python3
"""Does the labelled reference say where to cut the query field? Curves per cut level (GPU), scored offline.

The reference tokens are the only labelled tokens. For every level t of the sealed RCG field z (64 x 64):
  back_fg(t), back_bg(t)   share of reference foreground / background tokens whose nearest query token has z > t
  fwd_fg(t)                share of query tokens with z > t whose nearest reference token is foreground
together with intersection and union of the 1024 mask at that level. Truth is read only to count.

  python scripts/run_reference_cut.py --root outputs/fresh600_root --run outputs/recheck_fresh600_v1 --out outputs/claude_reference_cut_fresh600
"""
import argparse
import json
from pathlib import Path

import numpy as np

LEVELS = np.round(np.arange(0.10, 0.9001, 0.0125), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda"); lv = torch.from_numpy(LEVELS).float().to(dev)
    rows = json.loads((a.run / "manifest.json").read_text()); out = {k: [] for k in ("I", "U", "back_fg", "back_bg", "fwd_fg", "back_fg_soft", "back_bg_soft", "truth", "ref_area")}
    for r in rows:
        feat = torch.load(a.root / "cache/evidence_v1/feat" / (r["key"] + ".pt")); q, ref = (F.normalize(feat[k].to(dev).float(), dim=1) for k in ("q", "r"))
        with np.load(a.root / "results/extent_v1/run/packets" / (r["key"] + ".npz"), allow_pickle=False) as z:
            truth = torch.from_numpy(np.unpackbits(z["truth"]).reshape(1024, 1024).astype(bool)).to(dev); cov = torch.from_numpy(z["cov"]).to(dev).flatten()
        with np.load(a.run / "fields" / (r["key"] + ".npz"), allow_pickle=False) as z:
            f = torch.from_numpy(z["rcg"].astype(np.float32)).to(dev)
        m = F.interpolate(f[None, None], (1024, 1024), mode="bilinear", align_corners=False)[0] > lv[:, None, None]
        out["I"].append((m & truth).flatten(1).sum(1).cpu().numpy()); out["U"].append((m | truth).flatten(1).sum(1).cpu().numpy())
        sim = q @ ref.T; zf = f.flatten(); fg = cov >= .9
        if not fg.any():
            fg = cov == cov.max()
        bg = cov == 0
        v = zf[sim.argmax(0)]                                                   # z of the nearest query token of each reference token
        w = torch.softmax(sim / .05, 0); vs = (w * zf[:, None]).sum(0)          # the same, soft over query tokens
        share = lambda x, ix: ((x[ix][None] > lv[:, None]).float().mean(1) if ix.any() else torch.zeros_like(lv)).cpu().numpy()
        out["back_fg"].append(share(v, fg)); out["back_bg"].append(share(v, bg)); out["back_fg_soft"].append(share(vs, fg)); out["back_bg_soft"].append(share(vs, bg))
        nf = fg[sim.argmax(1)].float(); sel = (zf[None] > lv[:, None]).float()
        out["fwd_fg"].append(((sel * nf[None]).sum(1) / sel.sum(1).clamp_min(1)).cpu().numpy())
        out["truth"].append(int(truth.sum())); out["ref_area"].append(float(cov.mean()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "curves.npz", levels=LEVELS, **{k: np.array(v) for k, v in out.items()}); print("REFCUT_DONE", flush=True)


if __name__ == "__main__":
    main()
