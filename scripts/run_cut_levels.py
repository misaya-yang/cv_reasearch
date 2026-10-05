#!/usr/bin/env python3
"""Counts for every cut level of sealed 64 x 64 fields (GPU), so that any rule for choosing the level can be scored exactly.

For each episode of each sealed run: the RCG field (already on its final scale, cut at 0.5 by RCG) and the min-max FoRIS
score are upsampled to 1024 (bilinear, align_corners False) and cut at LEVELS; intersection and union with truth are
stored per level, together with the two fields. Truth is read only to count.

  python scripts/run_cut_levels.py --runs outputs/claude_official/run{0..6} --roots outputs/claude_official/root{0..6} --out outputs/claude_cut_levels4000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

LEVELS = np.round(np.arange(0.10, 0.9001, 0.0125), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True); p.add_argument("--roots", type=Path, nargs="+", required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda"); lv = torch.from_numpy(LEVELS).float().to(dev)[:, None, None]
    rows, fields, scores, I, U, T = [], [], [], {"rcg": [], "foris": []}, {"rcg": [], "foris": []}, []
    for b, (run, root) in enumerate(zip(a.runs, a.roots)):
        for r in json.loads((run / "manifest.json").read_text()):
            with np.load(root / "results/extent_v1/run/packets" / (r["key"] + ".npz"), allow_pickle=False) as z:
                truth = torch.from_numpy(np.unpackbits(z["truth"]).reshape(1024, 1024).astype(bool)).to(dev); s = z["score"].astype(np.float32)
            with np.load(run / "fields" / (r["key"] + ".npz"), allow_pickle=False) as z:
                f = z["rcg"].astype(np.float32)
            s = (s - s.min()) / max(float(s.max() - s.min()), 1e-6)
            for name, x in (("rcg", f), ("foris", s)):
                up = F.interpolate(torch.from_numpy(x).to(dev)[None, None], (1024, 1024), mode="bilinear", align_corners=False)[0]
                m = up > lv; I[name].append((m & truth).flatten(1).sum(1).cpu().numpy()); U[name].append((m | truth).flatten(1).sum(1).cpu().numpy())
            rows.append(dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"], batch=str(b)))
            fields.append(f.astype(np.float16)); scores.append(s.astype(np.float16)); T.append(int(truth.sum()))
            if len(rows) % 500 == 0:
                print(len(rows), flush=True)
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "rows.json").write_text(json.dumps(rows) + "\n")
    np.savez_compressed(a.out / "counts.npz", levels=LEVELS, truth=np.array(T), rcg_field=np.stack(fields), foris_score=np.stack(scores),
                        **{"%s_%s" % (k, n): np.stack(v[n]) for k, v in (("I", I), ("U", U)) for n in v})
    print("CUT_DONE", flush=True)


if __name__ == "__main__":
    main()
