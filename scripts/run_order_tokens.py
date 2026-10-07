#!/usr/bin/env python3
"""Token-level record for the ordering role: what each query token was scored, and what could have been known (GPU).

Roles are defined before any method: ORDERING gives every query token a score (object above background); the CUT
turns the order into a set. The loss of the ordering role is what remains under the best cut of the episode. This run
stores, per 64 x 64 query token, the fields below so that ordering losses can be attributed afterwards with no GPU:

  truth        object share of the token (area)
  s2 s3 score  FoRIS evidence after its reference-contrast step, after its query-clustering step, and final
  rcg          the RCG field on the final score
  a_fg a_bg    largest cosine to a reference object token / reference background token (debiased features)
  knn_fg       share of object tokens among the 10 nearest reference tokens
  back_fg      whether the token is the nearest query token of some reference object token (back_bg likewise)
  nbr          the 20 nearest query tokens (feature cosine)

Truth is read only to be stored; no field uses it.

  python scripts/run_order_tokens.py --episodes .../train_episodes.json --start 0 --count 150 --out outputs/claude_order_fresh600
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
    f16 = lambda: np.zeros((N, 4096), np.float16)
    A = dict(truth=f16(), s2=f16(), s3=f16(), score=f16(), rcg=f16(), a_fg=f16(), a_bg=f16(), knn_fg=f16(), ref=f16())
    back_fg, back_bg, nbr, native = np.zeros((N, 4096), bool), np.zeros((N, 4096), bool), np.zeros((N, 4096, 20), np.int16), np.zeros((N, 4096), np.float16)
    iu = np.zeros((N, 2), np.int64); begin = time.monotonic()
    grid = lambda x: x.reshape(64, 64) if x.numel() == 4096 else F.interpolate(x.reshape(1, 1, *x.shape[-2:]).float(), (64, 64), mode="bilinear", align_corners=False)[0, 0]
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda"); torch.set_num_threads(int(os.environ.get("RCG_THREADS", "4")))
        for n, row in enumerate(rows):
            sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
            mask, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            deb = got["deb"][0]; q, r = F.normalize(deb[-1].flatten(1).T.float(), dim=1).half().float(), F.normalize(deb[0].flatten(1).T.float(), dim=1).half().float()
            score = got["score"].float(); cov_t = F.interpolate(ref_mask[None, None].float(), tuple(score.shape), mode="area")[0, 0]
            z = solve(q.cpu(), r.cpu(), cov_t.cpu().numpy(), score.cpu().numpy().astype(np.float32), 16., "cpu")
            cov = cov_t.flatten(); fg, bg = cov > .5, cov <= .5; sim = q @ r.T                                    # [query, reference]
            a_fg = sim[:, fg].max(1).values if fg.any() else torch.zeros(4096, device=dev); a_bg = sim[:, bg].max(1).values if bg.any() else torch.zeros(4096, device=dev)
            top = sim.topk(10, dim=1).indices; knn = fg[top].float().mean(1); near = sim.argmax(0)             # nearest query token of each reference token
            bf = torch.zeros(4096, dtype=torch.bool, device=dev); bb = bf.clone(); bf[near[fg]] = True; bb[near[bg]] = True
            nb = (q @ q.T).topk(21, dim=1).indices[:, 1:]
            t_o = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == row["c"] + 1).copy()).to(dev)
            truth = F.interpolate(t_o[None, None].float(), (1024, 1024), mode="nearest"); iu[n] = int((mask & truth[0, 0].bool()).sum()), int((mask | truth[0, 0].bool()).sum())
            for k, v in dict(truth=F.avg_pool2d(truth, 16)[0, 0], s2=grid(got["s2"]), s3=grid(got["s3"]), score=grid(score), rcg=torch.from_numpy(z), a_fg=a_fg, a_bg=a_bg, knn_fg=knn, ref=cov).items():
                A[k][n] = v.flatten().float().cpu().numpy()
            native[n] = F.avg_pool2d(mask[None, None].float(), 16).flatten().cpu().numpy(); back_fg[n], back_bg[n], nbr[n] = bf.cpu().numpy(), bb.cpu().numpy(), nb.cpu().numpy()
            if (n + 1) % 50 == 0:
                print(json.dumps(dict(n=n + 1, total=N, seconds=round(time.monotonic() - begin, 1))), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "tokens.npz", native=native, back_fg=back_fg, back_bg=back_bg, nbr=nbr, iu_native=iu, **A)
    print("ORDER_DONE", flush=True)


if __name__ == "__main__":
    main()
