#!/usr/bin/env python3
"""The prior term: is a query region an object at all, read from the query alone (GPU).

Argument fixed before this run. The target mask is a posterior: P(region is the target | reference, query) is
proportional to P(reference evidence | region) x P(region is a candidate object | query). FoRIS and RCG shape the
first factor only; the second is flat. Measured on fresh600: the first factor is exhausted (RCG is at the ceiling of
the query's own signature in 86 % of episodes) and in the remaining 14 % it prefers a wrong region in every layer and
at both scales (AUC of true object tokens against wrongly kept tokens 0.14 to 0.6). A rule monotone in reference
evidence cannot repair those episodes; the only untouched term is the prior. Prediction: a query-only objecthood
separates the true object from the wrongly kept region where reference evidence does not (AUC clearly above 0.5 in
the episodes with RCG best-cut IoU < 0.5), and component deletion that reads it beats the same rule without it.

Query-only fields (no reference, no truth), 64 x 64:
  obj_border  minus the mean of the 20 largest cosines to tokens on the image border (boundary-connectivity)
  obj_cls     cosine of the token to the class token of the query
  obj_cut     second eigenvector of the normalised Laplacian of the thresholded token affinity, signed so that the
              side holding fewer border tokens is positive (normalised cut bipartition)

  python scripts/run_prior.py --episodes .../train_episodes.json --start 0 --count 75 --out outputs/claude_prior_a
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, required=True); p.add_argument("--start", type=int, required=True); p.add_argument("--count", type=int, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--limit", type=int)
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent")); p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(a.host_root)); sys.path.insert(0, str(a.host_root / "scripts"))
    from extent_experiment import build_host, run_foris
    from run_rcg2 import solve
    man = json.loads(a.episodes.read_text()); rows = []
    for f in sorted({e["fold"] for e in man["episodes"]}):
        rows += [dict(e, key="%d_%d_%d" % (e["fold"], e["e"], e["c"])) for e in man["episodes"] if e["fold"] == f][a.start:a.start + a.count]
    rows = rows[:a.limit]; data, ann = Path(man["data_root"]), Path(man["annotation_root"]); dev = torch.device("cuda"); N = len(rows)
    a.out.mkdir(parents=True, exist_ok=False)
    tok = {k: np.zeros((N, 4096), np.float16) for k in ("truth", "score", "rcg", "cos", "obj_border", "obj_cls", "obj_cut", "ref_obj_border", "ref_obj_cls", "ref")}
    edge = torch.zeros(64, 64, dtype=torch.bool); edge[0] = edge[-1] = edge[:, 0] = edge[:, -1] = True; edge = edge.flatten().to(dev)
    begin = time.monotonic()
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda"); torch.set_num_threads(int(os.environ.get("RCG_THREADS", "4")))
        for n, row in enumerate(rows):
            sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
            mask, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            deb = got["deb"]; q, r = F.normalize(deb[0, -1].flatten(1).T.float(), dim=1), F.normalize(deb[0, 0].flatten(1).T.float(), dim=1)
            score = got["score"].float(); cov_t = F.interpolate(ref_mask[None, None].float(), tuple(score.shape), mode="area")[0, 0]; fg = cov_t.flatten() > .5
            z0 = solve(q.half().float().cpu(), r.half().float().cpu(), cov_t.cpu().numpy(), score.cpu().numpy().astype(np.float32), 16., "cpu")
            t_o = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == row["c"] + 1).copy()).to(dev)
            tt = F.avg_pool2d(F.interpolate(t_o[None, None].float(), (1024, 1024), mode="nearest"), 16).flatten()
            mu_r = F.normalize(r[fg].mean(0), dim=0) if fg.any() else F.normalize(r.mean(0), dim=0)
            out = host.encoder.m.forward_features(torch.stack([host._transform(sp).to(dev), tgt]))          # timm: class token, registers, patches
            x = F.normalize(out[:, host.encoder.m.num_prefix_tokens:].float(), dim=2); c = F.normalize(out[:, 0].float(), dim=1)
            border = lambda u: -(u @ u[edge].T).topk(20, dim=1).values.mean(1)
            A = ((q @ q.T) > .2).double() + 1e-5; d = A.sum(1); L = torch.eye(4096, device=dev, dtype=torch.double) - A / d.sqrt()[:, None] / d.sqrt()[None]
            v = (torch.linalg.eigh(L)[1][:, 1] / d.sqrt()).float(); v = v / v.abs().max(); v = -v if (v[edge] > 0).float().mean() > .5 else v
            fields = dict(truth=tt, score=score.flatten(), rcg=torch.from_numpy(z0).flatten(), cos=q @ mu_r, obj_border=border(q), obj_cls=x[1] @ c[1], obj_cut=v, ref_obj_border=border(r), ref_obj_cls=x[0] @ c[0], ref=cov_t.flatten())
            for k, val in fields.items():
                tok[k][n] = val.flatten().float().cpu().numpy()
            if (n + 1) % 25 == 0:
                print(json.dumps(dict(n=n + 1, total=N, seconds=round(time.monotonic() - begin, 1))), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "tokens.npz", **tok)
    print("PRIOR_DONE", flush=True)


if __name__ == "__main__":
    main()
