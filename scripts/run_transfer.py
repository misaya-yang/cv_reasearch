#!/usr/bin/env python3
"""Where the ordering loss sits: representation, or transfer of the object's signature from reference to query (GPU).

Token fields on the debiased features FoRIS matches, no extra FoRIS pass. A ladder of ceilings, each read under the
best cut of the episode by score_transfer.py:

  cos            cosine to the reference object mean (what a prototype method can order)
  cos_oracle     cosine to the query's own object mean (truth): removes the shift of the signature between images
  mf_ref[l]      matched filter in the query's metric with the reference signature (run_evidence.py; l = shrinkage)
  mf_oracle[l]   the same filter with the query's own object mean (truth): the filter when the signature is right
  lin_oracle     ridge on truth, fitted on half of the 8 x 8 blocks, read on the other: what one direction can order
  lin_pseudo:*   the same ridge fitted on a field instead of truth (FoRIS final score, RCG): one self-training step
  s2 score rcg   FoRIS reference-contrast score, FoRIS final score, RCG

Fields with `oracle` read truth and are ceilings, not methods.

  python scripts/run_transfer.py --episodes .../train_episodes.json --start 0 --count 75 --out outputs/claude_transfer_a
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LAMS = (.1, 1., 10.)


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
    keys = ["truth", "s2", "score", "rcg", "cos", "cos_oracle", "lin_oracle", "lin_pseudo:score", "lin_pseudo:rcg"] + ["mf_ref[%g]" % l for l in LAMS] + ["mf_oracle[%g]" % l for l in LAMS]
    tok = {k: np.zeros((N, 4096), np.float16) for k in keys}; stat = np.zeros((N, 6), np.float32); iu = dict(native=np.zeros((N, 2), np.int64), rcg=np.zeros((N, 2), np.int64))
    grid = lambda x: x.reshape(64, 64) if x.numel() == 4096 else F.interpolate(x.reshape(1, 1, *x.shape[-2:]).float(), (64, 64), mode="bilinear", align_corners=False)[0, 0]
    up = lambda z: F.interpolate(z.reshape(1, 1, 64, 64).float(), (1024, 1024), mode="bilinear", align_corners=False)[0, 0] > .5
    blocks = ((torch.arange(64)[:, None] // 8 + torch.arange(64)[None] // 8) % 2).flatten().bool().to(dev)

    def inverse(x, lam):                                                # shrunk inverse covariance of the rows of x
        c = x - x.mean(0); ev, V = torch.linalg.eigh((c.T @ c).double() / len(c))
        return (V / (ev.clamp_min(0) + lam * ev.clamp_min(0).mean())) @ V.T, c

    begin = time.monotonic()
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda"); torch.set_num_threads(int(os.environ.get("RCG_THREADS", "4")))
        original = host._part2_stage2_contrastive_score
        for n, row in enumerate(rows):
            sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
            mask, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            deb = got["deb"]; q, r = F.normalize(deb[0, -1].flatten(1).T.float(), dim=1).half().float(), F.normalize(deb[0, 0].flatten(1).T.float(), dim=1).half().float()
            score = got["score"].float(); cov_t = F.interpolate(ref_mask[None, None].float(), tuple(score.shape), mode="area")[0, 0]; cov = cov_t.cpu().numpy(); fg = cov_t.flatten() > .5
            rcg = lambda s: torch.from_numpy(solve(q.cpu(), r.cpu(), cov, s.cpu().numpy().astype(np.float32), 16., "cpu")).to(dev)
            t_o = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == row["c"] + 1).copy()).to(dev)
            truth = F.interpolate(t_o[None, None].float(), (1024, 1024), mode="nearest")[0, 0].bool(); tt = F.avg_pool2d(truth[None, None].float(), 16).flatten()
            mu_r = F.normalize(r[fg].mean(0), dim=0) if fg.any() else F.normalize(r.mean(0), dim=0); m = q.mean(0)
            obj = tt > .5; mu_q = F.normalize(q[obj].mean(0), dim=0) if obj.any() else mu_r; mu_b = F.normalize(q[~obj].mean(0), dim=0) if (~obj).any() else mu_r
            stat[n] = float(tt.mean()), float(mu_r @ mu_q), float(mu_r @ mu_b), float(mu_q @ mu_b), float(fg.float().mean()), float((q[obj] @ mu_q).mean()) if obj.any() else 0.
            z0 = rcg(score); fields = dict(truth=tt, s2=grid(got["s2"]).flatten(), score=grid(score).flatten(), rcg=z0.flatten(), cos=q @ mu_r, cos_oracle=q @ mu_q)
            for lam in LAMS:
                P, c = inverse(q, lam)
                for tag, d in (("mf_ref", mu_r), ("mf_oracle", mu_q)):
                    d = (d - m).double(); fields["%s[%g]" % (tag, lam)] = (c.double() @ P @ d / (d @ P @ d).clamp_min(1e-12).sqrt()).float()
            for key, target in (("lin_oracle", tt), ("lin_pseudo:score", fields["score"]), ("lin_pseudo:rcg", fields["rcg"])):
                out = torch.zeros(4096, device=dev); target = target.float()
                for b in (blocks, ~blocks):
                    P, c = inverse(q[b], .1); w = P @ (c.double().T @ (target[b] - target[b].mean()).double()) / int(b.sum()); out[~b] = ((q[~b] - q[b].mean(0)).double() @ w).float() + target[b].mean()
                fields[key] = out
            masks = dict(native=mask, rcg=up(z0))
            for k, m in masks.items():
                iu[k][n] = int((m & truth).sum()), int((m | truth).sum())
            for k, v in fields.items():
                tok[k][n] = v.flatten().float().cpu().numpy()
            if (n + 1) % 25 == 0:
                print(json.dumps(dict(n=n + 1, total=N, seconds=round(time.monotonic() - begin, 1))), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "counts.npz", **{"iu:" + k: v for k, v in iu.items()}); np.savez_compressed(a.out / "tokens.npz", stat=stat, **tok)
    print("TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
