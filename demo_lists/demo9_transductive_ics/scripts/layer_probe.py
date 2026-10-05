#!/usr/bin/env python3
"""One closing measurement (GPU, minutes): FoRIS uses the reference as one mean vector of the last layer of two images encoded
apart. Does that one-vector reading rank the query better on another layer, without the positional projection, or when both
images are encoded in one forward pass? Only 64 x 64 score maps are stored (float16), no tokens; no query label is read.

  python scripts/layer_probe.py --manifest M/dev_episodes.json --out B/layers_dev [--limit N] [--tiny 1]

Maps per episode: <view>_one (cosine to the reference object's mean) and <view>_bg (minus 0.55 x the orthogonalised hard
background mean, FoRIS's own form), for view in l8, l12, l16, l20, l22, l24raw, l24 (as FoRIS has it), joint (one canvas).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

EXTENT = "/root/autodl-tmp/demo9_extent"
LAYERS = (8, 12, 16, 20, 22, 24)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--limit", type=int)
    p.add_argument("--tiny", type=int, default=0); p.add_argument("--device", default="cuda"); p.add_argument("--extent", default=EXTENT)
    p.add_argument("--report", default="report.json")
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
    tok = lambda x: x.flatten(-2).transpose(-1, -2)

    def read(ref, q, m):  # ref, q: [hw, C] unit tokens; m: reference mask on the grid
        mu = F.normalize(ref[m].mean(0), dim=0)
        one = q @ mu
        if (~m).any():
            sim = ref[~m] @ mu
            hard = F.normalize(ref[~m][sim.topk(max(1, int(.2 * len(sim)))).indices].mean(0), dim=0)
            hard = F.normalize(hard - (hard @ mu) * mu, dim=0)
            return one, one - .55 * (q @ hard)
        return one, one

    rows, done, began = man["episodes"][:a.limit], 0, time.monotonic()
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
            got = host.encoder.get_intermediate_layers(imgs, n=[k - 1 for k in LAYERS], reshape=True)
            got = list(got) * len(LAYERS) if len(got) == 1 else list(got)  # the CPU stand-in has one layer
            got = [F.normalize(x.float(), dim=1) for x in got]
            h, w = got[-1].shape[-2:]
            m = F.interpolate(masks.float(), size=(h, w), mode="nearest")[0, 0].flatten() > .5
            if not m.any():
                raise SystemExit("empty reference mask on the token grid: %s" % r)
            flag = bool(host._should_apply_positional_debias(got[-1][None], masks, 1))
            deb = (lambda x: host._debias_features(x)) if flag else (lambda x: x)
            pack = dict(debiased=flag)
            views = {("l%d" % k if k != LAYERS[-1] else "l24raw"): x for k, x in zip(LAYERS, got)}
            views["l24"] = deb(got[-1][None])[0]
            canvas = torch.cat([imgs[0], imgs[1]], dim=-1)[None]  # reference on the left, query on the right
            joint = F.normalize(host.encoder.get_intermediate_layers(canvas, n=1, reshape=True)[0].float(), dim=1)
            views["joint"] = deb(torch.stack([joint[0, :, :, :w], joint[0, :, :, w:]])[None])[0]
            for name, x in views.items():
                one, bg = read(tok(x[0]), tok(x[1]), m)
                pack[name + "_one"], pack[name + "_bg"] = one.view(h, w).half().cpu().numpy(), bg.view(h, w).half().cpu().numpy()
            np.savez(path, **pack)
            done += 1
            if n < 3 or n % 40 == 0:
                print("%d/%d debias=%s %.1f s" % (n + 1, len(rows), flag, time.monotonic() - began), flush=True)
    spent = time.monotonic() - began
    (out / a.report).write_text(json.dumps(dict(state="COMPLETED", episodes=len(rows), written=done, seconds=round(spent, 1),
                                                seconds_per_episode=round(spent / max(done, 1), 3), tiny=bool(a.tiny))))
    print((out / a.report).read_text())


if __name__ == "__main__":
    main()
