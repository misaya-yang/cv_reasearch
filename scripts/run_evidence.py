#!/usr/bin/env python3
"""Reference-to-query evidence derived as a detection problem, placed where FoRIS computes its own (GPU).

Measured before this run (score_order_tokens.py, 600 episodes): under the best cut of each episode FoRIS still loses
30 points; false area is four times the missed area, and 80 % of the false tokens lie in feature clusters of the query
that are background and are scored high as a whole. FoRIS scores a query token by its cosine to reference object
prototypes minus one reference-background direction; the query's own background never enters.

Derivation. One known signature d (a reference object prototype), a scene x_1..x_n (the query tokens) whose background
law is unknown and is not the reference background. The linear detector with unit response on d and least output
energy on the scene is the matched filter in the scene's own metric,

    y(x) = (x - m)' S^-1 (d - m) / sqrt((d - m)' S^-1 (d - m)),   m, S: mean and covariance of the query tokens,

with S shrunk towards its mean eigenvalue. Stated before the run: (1) the false area in coherent background clusters
falls, because directions in which the scene varies strongly are discounted; (2) the gain shrinks or reverses when the
object fills much of the query, because the object then contaminates S; (3) cosine to the prototype is the special
case S = I and should order worse wherever the scene is anisotropic.

Arms, each a complete mask at 1024: native (complete FoRIS, control); rcg; and per variant V the complete FoRIS with
its reference-contrast score replaced by y (matched to that score's mean and spread, the rest of FoRIS untouched),
alone and followed by RCG. Token fields are stored for the ordering score. `lin_oracle` reads truth (ceiling only).

  python scripts/run_evidence.py --episodes .../train_episodes.json --start 0 --count 75 --out outputs/claude_evidence_a
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VARIANTS = (("mf1[.1]", 1, .1), ("mfK[.03]", 0, .03), ("mfK[.1]", 0, .1), ("mfK[.3]", 0, .3))   # name, single prototype, shrinkage


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
    names = [v[0] for v in VARIANTS]; arms = ["native", "rcg"] + [n + x for n in names for x in ("", "+rcg")]
    iu = {k: np.zeros((N, 2), np.int64) for k in arms}
    tok = {k: np.zeros((N, 4096), np.float16) for k in ["truth", "s2", "score", "rcg", "cos", "ref_lda", "lin_oracle"] + ["y:" + n for n in names] + ["score:" + n for n in names] + ["rcg:" + n for n in names]}
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
            mds = F.interpolate(ref_mask[None, None].float(), tuple(score.shape), mode="nearest")[0] > .5
            mu_fg, _, protos = host._reference_contrastive_prototypes(deb[:, :1], mds, 1, mu_fg_per_reference=True)
            protos = mu_fg[None] if protos is None or len(protos) == 0 else protos
            fields = dict(cos=q @ mu_fg.float())
            for name, single, lam in VARIANTS:
                P, c = inverse(q, lam); d = ((mu_fg[None] if single else protos).float() - q.mean(0)).double()
                fields["y:" + name] = ((c.double() @ P @ d.T) / (d @ P @ d.T).diagonal().clamp_min(1e-12).sqrt()).max(1).values.float()
            P, c = inverse(r, .1); fields["ref_lda"] = ((q - q.mean(0)).double() @ P @ (r[fg].mean(0) - r[~fg].mean(0)).double()).float() if fg.any() and (~fg).any() else fields["cos"]
            t_o = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == row["c"] + 1).copy()).to(dev)
            truth = F.interpolate(t_o[None, None].float(), (1024, 1024), mode="nearest")[0, 0].bool(); tt = F.avg_pool2d(truth[None, None].float(), 16).flatten()
            lo = torch.zeros(4096, device=dev)                             # ceiling: ridge on truth, fitted on one half of the 8 x 8 blocks and read on the other
            for b in (blocks, ~blocks):
                P, c = inverse(q[b], .1); w = P @ (c.double().T @ (tt[b] - tt[b].mean()).double()) / int(b.sum()); lo[~b] = ((q[~b] - q[b].mean(0)).double() @ w).float()
            fields["lin_oracle"] = lo
            z0 = rcg(score); masks = dict(native=mask, rcg=up(z0)); fields.update(truth=tt, s2=grid(got["s2"]).flatten(), score=grid(score).flatten(), rcg=z0.flatten())
            for name, _, _ in VARIANTS:
                y = fields["y:" + name].reshape(tuple(score.shape))

                def replaced(*args, _y=y, **kw):
                    out = original(*args, **kw)
                    if out is None:
                        return None
                    s, sf, sbn, mu = out; yy = _y.to(s.dtype)
                    return (yy - yy.mean()) / yy.std().clamp_min(1e-6) * s.std() + s.mean(), (yy - yy.min()) / (yy.max() - yy.min()).clamp_min(1e-6), sbn, mu
                host._part2_stage2_contrastive_score = replaced
                try:
                    m_v, got_v = run_foris(host, sp, gold, qp)[:2]
                finally:
                    del host._part2_stage2_contrastive_score
                zv = rcg(got_v["score"].float()); masks[name] = m_v; masks[name + "+rcg"] = up(zv); fields["score:" + name] = grid(got_v["score"].float()).flatten(); fields["rcg:" + name] = zv.flatten()
            for k, m in masks.items():
                iu[k][n] = int((m & truth).sum()), int((m | truth).sum())
            for k, v in fields.items():
                tok[k][n] = v.flatten().float().cpu().numpy()
            if (n + 1) % 25 == 0:
                print(json.dumps(dict(n=n + 1, total=N, seconds=round(time.monotonic() - begin, 1))), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "counts.npz", **{"iu:" + k: v for k, v in iu.items()}); np.savez_compressed(a.out / "tokens.npz", **tok)
    print("EVIDENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
