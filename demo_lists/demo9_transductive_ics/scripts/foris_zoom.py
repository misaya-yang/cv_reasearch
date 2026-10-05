#!/usr/bin/env python3
"""Look again at each region FoRIS found, at the region's own scale (GPU; complete public FoRIS, nothing fitted).

  python scripts/foris_zoom.py --manifest M/dev_episodes.json --out X/zoom_dev [--limit N]

FoRIS cuts its score at a constant level, which is right when the target fills a usual share of the view and too low
when the target is small (the label-chosen level is 0.71 for small targets and 0.51 otherwise, DEV241). So every
connected component of FoRIS's mask is cropped to a square of 1.5 x its longer side and FoRIS is run on that crop, up
to three rounds, each round cropping around the previous result; a round stops when the crop no longer shrinks.
Arms: `q` keeps the reference image as it is, `p` also crops the reference to 1.5 x its object.
Stored per episode (512 x 512 bitmaps): the native mask, each component, each component's result after round 1 and
at the end, and the raw (not normalised) score peaks of each crop pass; the truth is stored for the reading only.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

EXTENT = "/root/autodl-tmp/demo9_extent"
sys.path[:0] = [EXTENT, EXTENT + "/scripts"]
SIDE, ROUNDS, TOP = 512, 3, 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--limit", type=int)
    p.add_argument("--margin", type=float, default=1.5); p.add_argument("--min-side", type=int, default=128)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from scipy import ndimage
    from extent_experiment import build_host, run_foris
    from tics.extent_cut import paste, to_original
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root="/root/autodl-tmp/demo4"), man, "cuda")

    def box_of(mask):
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if len(ys) == 0:
            return None
        H, W = mask.shape
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        side = max(a.min_side, int(math.ceil(a.margin * max(y1 - y0, x1 - x0))))
        if side >= 0.9 * min(H, W):
            return None
        bx = int(min(max(round((x0 + x1) / 2 - side / 2), 0), W - side)); by = int(min(max(round((y0 + y1) / 2 - side / 2), 0), H - side))
        return bx, by, bx + side, by + side

    small = lambda m: np.packbits(F.interpolate(m[None, None].float(), (SIDE, SIDE), mode="area")[0, 0].gt(.5).cpu().numpy())
    begin = time.monotonic()
    with torch.inference_mode(), open(out / "episodes.jsonl", "w") as stream:
        for n, r in enumerate(man["episodes"][:a.limit]):
            sp, qp = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1).copy())
            native, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            hw = tuple(native.shape)
            raw0 = got["score"].float()
            # the reference cropped to its object's scale, for arm p
            rb = box_of(ref_mask)
            ref_p = (sp, gold)
            if rb is not None:
                rob, _ = to_original(rb, hw, sp.size)
                crop = gold[rob[1]:rob[3], rob[0]:rob[2]]
                if crop.any():
                    ref_p = (sp.crop(rob), crop)
            lab, count = ndimage.label(native.cpu().numpy(), structure=np.ones((3, 3)))
            areas = ndimage.sum(np.ones_like(lab), lab, range(1, count + 1)) if count else np.zeros(0)
            order = [i for i in np.argsort(-areas)[:TOP] if areas[i] >= 256]
            rest = torch.from_numpy(~np.isin(lab, [i + 1 for i in order]) & (lab > 0)).to(native.device)
            bits = dict(native=small(native), rest=small(rest))
            comps = []
            for j, i in enumerate(order):
                comp = torch.from_numpy(lab == i + 1).to(native.device)
                rec = dict(area=int(comp.sum()), peak0=float(F.interpolate(raw0[None, None], hw, mode="bilinear", align_corners=False)[0, 0][comp].max()))
                bits["c%d" % j] = small(comp)
                for arm, (r_img, r_mask) in (("q", (sp, gold)), ("p", ref_p)):
                    cur, prev, log = comp, None, []
                    for rnd in range(ROUNDS):
                        box = box_of(cur)
                        if box is None or (prev is not None and box[2] - box[0] > 0.8 * prev):
                            break
                        obox, mbox = to_original(box, hw, qp.size)
                        try:
                            pred, g, _, _ = run_foris(host, r_img, r_mask, qp.crop(obox))
                        except RuntimeError as err:
                            if "No foreground tokens" not in str(err):
                                raise
                            break
                        s, s2 = g["score"].float(), g["s2"].float()
                        log.append(dict(side=box[2] - box[0], peak=float(s.max()), low=float(s.min()), peak2=float(s2.max()), low2=float(s2.min()),
                                        share=float(pred.float().mean())))
                        prev, cur = box[2] - box[0], paste(pred, mbox, hw)
                        if rnd == 0:
                            bits["%s1_%d" % (arm, j)] = small(cur)
                        if not cur.any():
                            break
                    bits["%sf_%d" % (arm, j)] = small(cur)
                    rec[arm] = log
                comps.append(rec)
            truth = torch.from_numpy((np.asarray(Image.open(ann / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1).copy())  # for the reading only
            bits["truth"] = small(F.interpolate(truth[None, None].float(), hw, mode="nearest")[0, 0].to(native.device))
            np.savez_compressed(out / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])), **bits)
            stream.write(json.dumps(dict(fold=r["fold"], e=r["e"], c=r["c"], comps=comps, peak=float(raw0.max()), low=float(raw0.min()))) + "\n"); stream.flush()
            if n % 20 == 0:
                print("%d/%d, %.1f s" % (n + 1, len(man["episodes"][:a.limit]), time.monotonic() - begin), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=n + 1, elapsed_s=round(time.monotonic() - begin, 1))))


if __name__ == "__main__":
    main()
