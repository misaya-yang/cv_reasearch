#!/usr/bin/env python3
"""Finer tokens of the query for boundary work (GPU, minutes; same frozen DINOv3, nothing added). Measured on DEV241: true
pixels within 16 px of FoRIS's own boundary are worth +15.8 mIoU, colour cues recover 0.5, and a mixing-share estimate on the
64 x 64 tokens +0.9. Stored per episode, float16, reduced to --dims by an uncentred SVD of the query's own tokens:

  s8   128 x 128 tokens of the 1024 view on a stride-8 grid (four encodings shifted by 0 / 8 px, interleaved; token k is centred
       at pixel 8k), debiased exactly when FoRIS debiases this pair
  z2   128 x 128 tokens of the same view enlarged to 2048
  mu   the reference object's mean token (FoRIS's own, at 1024) and the reference's hard-background mean, in both bases
  base the usual 64 x 64 query tokens in the s8 basis (check: s8 at odd positions must equal them)

No label of a query is read.

  python scripts/hires_bank.py --manifest M/dev_episodes.json --out B/hires_dev [--limit N] [--tiny 1]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

EXTENT = "/root/autodl-tmp/demo9_extent"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--limit", type=int)
    p.add_argument("--tiny", type=int, default=0); p.add_argument("--device", default="cuda"); p.add_argument("--extent", default=EXTENT)
    p.add_argument("--dims", type=int, default=256); p.add_argument("--report", default="report.json"); p.add_argument("--zoom", type=int, default=1)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path[:0] = [a.extent, a.extent + "/scripts"]
    from extent_experiment import build_host
    man, dev = json.loads(Path(a.manifest).read_text()), a.device
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    host = build_host(SimpleNamespace(fixture=a.tiny or None, foris_root=None, demo4_root="/root/autodl-tmp/demo4"), man, dev)
    half = lambda t: t.half().cpu().numpy()
    enc = lambda x: F.normalize(host.encoder.get_intermediate_layers(x, n=1, reshape=True)[0].float(), dim=1)  # [B, C, h, w]
    tok = lambda x: x.flatten(-2).transpose(-1, -2)                                                            # [.., C, h, w] -> [.., hw, C]
    rows, done, began, worst = man["episodes"][:a.limit], 0, time.monotonic(), 0.
    with torch.inference_mode():
        for n, r in enumerate(rows):
            path = out / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"]))
            if path.exists():
                continue
            sp, qp = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1).copy())
            try:
                host.set_reference(sp, gold); host.set_target(qp)
                imgs, masks = torch.cat([host._ref_images, host._tgt_image[None]]), host._ref_masks.unsqueeze(1)
            finally:
                host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None
            last = enc(imgs)[None]                                         # [1, 2, C, h, w]
            flag = bool(host._should_apply_positional_debias(last, masks, 1))
            deb = (lambda x: host._debias_features(x)) if flag else (lambda x: x)
            last = deb(last); h, w = last.shape[-2:]; ps = imgs.shape[-1] // w; sh = ps // 2
            ref, base = tok(last[0, 0]), tok(last[0, 1])
            m = F.interpolate(masks.float(), size=(h, w), mode="nearest")[0, 0].flatten() > .5
            if not m.any():
                raise SystemExit("empty reference mask on the token grid: %s" % r)
            mu = F.normalize(ref[m].mean(0), dim=0)
            if (~m).any():
                sim = ref[~m] @ mu; hard = F.normalize(ref[~m][sim.topk(max(1, int(.2 * len(sim)))).indices].mean(0), dim=0)
            else:
                hard = torch.zeros_like(mu)
            # stride-8 grid: the view moved down/right by half a patch puts token i at centre ps*i; unmoved, at ps*i + sh
            q = imgs[1:2]; grid = torch.zeros(last.shape[2], 2 * h, 2 * w, device=q.device)
            for dy in (0, 1):
                for dx in (0, 1):
                    x = F.pad(q, (sh * dx, 0, sh * dy, 0), mode="replicate")[..., :q.shape[-2], :q.shape[-1]]
                    grid[:, (1 - dy)::2, (1 - dx)::2] = deb(enc(x)[None])[0, 0]
            s8 = tok(grid)                                                 # [4hw, C]
            worst = max(worst, float((tok(grid[:, 1::2, 1::2]) - base).abs().max()))
            pack = dict(debiased=flag, grid=np.array([2 * h, 2 * w]))
            groups = [("s8", s8)]
            if a.zoom:
                big = deb(enc(F.interpolate(q, scale_factor=2, mode="bicubic", align_corners=False))[None])[0, 0]
                groups.append(("z2", tok(big)))
            for name, x in groups:
                k = min(a.dims, x.shape[1])
                _, _, v = torch.pca_lowrank(x, q=k, center=False, niter=2)
                pack[name] = half(x @ v); pack["mu_" + name] = half(mu @ v); pack["hard_" + name] = half(hard @ v)
                if name == "s8":
                    pack["base"] = half(base @ v)
            np.savez(path, **pack)
            done += 1
            if n < 3 or n % 40 == 0:
                print("%d/%d debias=%s %.1f s" % (n + 1, len(rows), flag, time.monotonic() - began), flush=True)
    spent = time.monotonic() - began
    (out / a.report).write_text(json.dumps(dict(state="COMPLETED", episodes=len(rows), written=done, dims=a.dims, max_abs_unshifted_vs_base=worst,
                                                seconds=round(spent, 1), seconds_per_episode=round(spent / max(done, 1), 3), tiny=bool(a.tiny))))
    print((out / a.report).read_text())


if __name__ == "__main__":
    main()
