#!/usr/bin/env python3
"""The signature-transfer loss: how the object's mean feature moves between reference and query, and what removes it (GPU).

Measured before this run (run_transfer.py, 600 episodes, ordering score = token mIoU under the best cut per episode):
cosine to the reference object mean 66.09; FoRIS final 69.88; RCG 72.00; cosine to the query's own object mean 83.08;
one linear direction fitted on truth 87.31; token ceiling 91.70. Seventeen points lie between the two cosines, i.e. in
the move of the signature, and the present pipeline recovers six of them.

Two models of the move, each with its own estimator, stated before the run:
  image offset   every token of an image carries a common offset (and scale): then centring each image by its own token
                 mean (`cos_c`), or aligning second moments as well (`coral`), removes it, with no use of the query's
                 content. Prediction: `cos_c` closes most of the gap if this model holds.
  content        the signature moves because the instance differs: then only the query's own structure can find it.
                 Mean shift from the reference signature on the query tokens (`ms[t]`, kernel exp(cos/t)) moves the
                 signature to the nearest mode of the query. Prediction: gains where the reference signature already
                 leans to the object, none in the episodes where it does not.
Ceilings (truth): `cos_oracle`, `cos_c_oracle`. `stat` stores the geometry of the move per episode.

  python scripts/run_signature.py --episodes .../train_episodes.json --start 0 --count 75 --out outputs/claude_signature_a
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TEMPS = (.03, .1)


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
    keys = ["truth", "score", "rcg", "cos", "cos_oracle", "cos_c", "cos_c_oracle", "coral", "coral_c", "mf_c[1]", "mf_c[10]"] + ["ms[%g]" % t for t in TEMPS] + ["ms_c[%g]" % t for t in TEMPS]
    tok = {k: np.zeros((N, 4096), np.float16) for k in keys}; stat = np.zeros((N, 8), np.float32); iu = dict(native=np.zeros((N, 2), np.int64), rcg=np.zeros((N, 2), np.int64))
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
            mu_r = r[fg].mean(0) if fg.any() else r.mean(0); m, mr = q.mean(0), r.mean(0); obj = tt > .5; mu_q = q[obj].mean(0) if obj.any() else mu_r
            cs = lambda u, v: float(F.cosine_similarity(u, v, dim=0)); delta = mu_q - mu_r; qc = F.normalize(q - m, dim=1)
            ev, V = torch.linalg.eigh(((q - m).T @ (q - m)).double() / 4096); top = V[:, -10:].float()
            stat[n] = float(tt.mean()), cs(mu_r, mu_q), cs(mu_r - mr, mu_q - m), cs(delta, m - mr), float(delta.norm()), float((m - mr).norm()), float((top.T @ delta).norm() ** 2 / delta.norm().clamp_min(1e-9) ** 2), float(fg.float().mean())
            z0 = rcg(score); fields = dict(truth=tt, score=grid(score).flatten(), rcg=z0.flatten(), cos=q @ F.normalize(mu_r, dim=0), cos_oracle=q @ F.normalize(mu_q, dim=0),
                                           cos_c=qc @ F.normalize(mu_r - mr, dim=0), cos_c_oracle=qc @ F.normalize(mu_q - m, dim=0))

            def root(x, power):                                           # shrunk matrix power of the covariance of the rows of x
                c = (x - x.mean(0)).double(); e, U = torch.linalg.eigh(c.T @ c / len(c)); e = e.clamp_min(0) + e.clamp_min(0).mean()
                return (U * e ** power) @ U.T
            moved = (root(q, .5) @ root(r, -.5) @ (mu_r - mr).double()).float(); fields["coral"] = q @ F.normalize(moved + m, dim=0); fields["coral_c"] = qc @ F.normalize(moved, dim=0)
            for lam in (1., 10.):
                P, c = inverse(q, lam); d = (mu_r - mr).double(); fields["mf_c[%g]" % lam] = (c.double() @ P @ d / (d @ P @ d).clamp_min(1e-12).sqrt()).float()
            for t in TEMPS:
                for tag, x, start in (("ms", q, F.normalize(mu_r, dim=0)), ("ms_c", qc, F.normalize(mu_r - mr, dim=0))):
                    mu = start
                    for _ in range(10):
                        mu = F.normalize((torch.softmax(x @ mu / t, 0)[:, None] * x).sum(0), dim=0)
                    fields["%s[%g]" % (tag, t)] = x @ mu
            masks = dict(native=mask, rcg=up(z0))
            for k, m in masks.items():
                iu[k][n] = int((m & truth).sum()), int((m | truth).sum())
            for k, v in fields.items():
                tok[k][n] = v.flatten().float().cpu().numpy()
            if (n + 1) % 25 == 0:
                print(json.dumps(dict(n=n + 1, total=N, seconds=round(time.monotonic() - begin, 1))), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "counts.npz", **{"iu:" + k: v for k, v in iu.items()}); np.savez_compressed(a.out / "tokens.npz", stat=stat, **tok)
    print("SIGNATURE_DONE", flush=True)


if __name__ == "__main__":
    main()
