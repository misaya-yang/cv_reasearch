#!/usr/bin/env python3
"""Is the evidence that separates the object from a wrongly chosen region absent at this scale, in this layer, or at all? (GPU)

Measured before this run (fresh600, ordering score under the best cut): RCG 72.00 against 83.08 for the query's own
object signature; replacing 83 episodes (RCG best-cut IoU < 0.5) alone by the latter gives 81.26. In those episodes no
stored reference evidence ranks object tokens above the wrongly chosen ones (AUC 0.3 to 0.5 for nearest-reference
vote, reference cosine, FoRIS scores). The loss is a wrong choice of region in one episode in seven, with no evidence
against it in the last-layer features at the native scale. Two places where evidence could still be, each a diagnostic
with truth, not a method:

  layer   cosine to the reference object mean in four encoder layers (`cos@L`)
  scale   the query window around the true object, enlarged until the object is as large as the reference object,
          re-encoded; inside the same window, ordering by the reference cosine with enlarged and with native tokens
          (`win_zoom`, `win_full`), and the cosine of the reference signature to the object's own signature both ways.

  python scripts/run_region.py --episodes .../train_episodes.json --start 0 --count 75 --out outputs/claude_region_a
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LAYERS = (11, 15, 19, 23)


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
    tok = {k: np.zeros((N, 4096), np.float16) for k in ["truth", "score", "rcg", "cos"] + ["cos@%d" % l for l in LAYERS]}
    win = {k: np.zeros((N, 4096), np.float16) for k in ("truth", "zoom", "full")}; stat = np.zeros((N, 8), np.float32)

    def best_iou(f, t):
        o = torch.argsort(f, descending=True); tt = t[o]; ci = tt.cumsum(0); cu = t.sum() + (1 - tt).cumsum(0); return float((ci / cu.clamp_min(1e-9)).max())
    begin = time.monotonic()
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda"); torch.set_num_threads(int(os.environ.get("RCG_THREADS", "4")))
        for n, row in enumerate(rows):
            sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
            mask, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            deb = got["deb"]; q, r = F.normalize(deb[0, -1].flatten(1).T.float(), dim=1), F.normalize(deb[0, 0].flatten(1).T.float(), dim=1)
            score = got["score"].float(); cov_t = F.interpolate(ref_mask[None, None].float(), tuple(score.shape), mode="area")[0, 0]; fg = cov_t.flatten() > .5
            z0 = torch.from_numpy(solve(q.half().float().cpu(), r.half().float().cpu(), cov_t.cpu().numpy(), score.cpu().numpy().astype(np.float32), 16., "cpu")).to(dev)
            debiased = bool((F.normalize(got["raw"][0], dim=1) - deb[0]).abs().max() > 1e-4)
            t_o = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == row["c"] + 1).copy()).to(dev)
            truth = F.interpolate(t_o[None, None].float(), (1024, 1024), mode="nearest"); tt = F.avg_pool2d(truth, 16).flatten(); obj = tt > .5
            mu_r = F.normalize(r[fg].mean(0), dim=0) if fg.any() else F.normalize(r.mean(0), dim=0)
            fields = dict(truth=tt, score=score.flatten(), rcg=z0.flatten(), cos=q @ mu_r)
            ref_img = host._transform(sp).to(dev)
            outs = host.encoder.get_intermediate_layers(torch.stack([ref_img, tgt]), n=list(LAYERS), reshape=True)
            for l, o in zip(LAYERS, outs):
                x = F.normalize(o.flatten(2).transpose(1, 2).float(), dim=2); fields["cos@%d" % l] = x[1] @ (F.normalize(x[0][fg].mean(0), dim=0) if fg.any() else F.normalize(x[0].mean(0), dim=0))
            for k, v in fields.items():
                tok[k][n] = v.flatten().float().cpu().numpy()
            ys, xs = torch.nonzero(truth[0, 0], as_tuple=True)
            if len(ys) and fg.any():
                qs, rs = float(truth.mean()), float(ref_mask.float().mean()); y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                side = min(1024, max(int(1024 * (qs / max(rs, 1e-6)) ** .5), int(1.2 * max(y1 - y0, x1 - x0)), 64)); side = max(16, side // 16 * 16)
                cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; top = min(max(cy - side // 2, 0), 1024 - side) // 16 * 16; left = min(max(cx - side // 2, 0), 1024 - side) // 16 * 16
                crop = F.interpolate(tgt[None, :, top:top + side, left:left + side], (1024, 1024), mode="bilinear", align_corners=False)
                f = F.normalize(host._extract_features(crop[None]), p=2, dim=2); f = host._debias_features(f) if debiased else f
                qz = F.normalize(f[0, 0].flatten(1).T.float(), dim=1); tz = F.avg_pool2d(F.interpolate(truth[:, :, top:top + side, left:left + side], (1024, 1024), mode="nearest"), 16).flatten()
                zoomed = (qz @ mu_r).reshape(1, 1, 64, 64); g = side // 16                                  # both read on the 64 x 64 grid of the enlarged window
                native = F.interpolate((q @ mu_r).reshape(64, 64)[top // 16:top // 16 + g, left // 16:left // 16 + g][None, None], (64, 64), mode="bilinear", align_corners=False)
                win["truth"][n], win["zoom"][n], win["full"][n] = tz.cpu().numpy(), zoomed.flatten().cpu().numpy(), native.flatten().cpu().numpy()
                oz = tz > .5; mu_q = F.normalize(q[obj].mean(0), dim=0) if obj.any() else mu_r; mu_z = F.normalize(qz[oz].mean(0), dim=0) if oz.any() else mu_r
                stat[n] = qs, rs, side / 1024, float(mu_r @ mu_q), float(mu_r @ mu_z), best_iou(native.flatten(), tz), best_iou(zoomed.flatten(), tz), 1
            if (n + 1) % 25 == 0:
                print(json.dumps(dict(n=n + 1, total=N, seconds=round(time.monotonic() - begin, 1))), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "tokens.npz", stat=stat, **tok, **{"win:" + k: v for k, v in win.items()})
    print("REGION_DONE", flush=True)


if __name__ == "__main__":
    main()
