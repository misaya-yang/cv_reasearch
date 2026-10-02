"""Region features for training / analysing recognisers.

For every image, pools each source over (a) the ground-truth class regions and (b) the class regions the segmenter predicts.
A predicted region is labelled with the ground-truth class histogram of its pixels (majority class + the 4 largest fractions).

  python scripts/extract_regions.py --split train --seg mask2former:models/m2f-swin-large-ade:640 --encoder models/dinov2-large:dinov2l
writes  $DEMO2_CACHE/rf/<dataset>_<split>_<segtag|noseg>_<source>.pt  for source in {own, <encoder tag>}.
`--fast` (meant for the training split): half-precision segmenter and regions at the segmenter's output resolution.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wwd import *

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="ade20k"); ap.add_argument("--split", default="train")
ap.add_argument("--seg", default=None, help="family:dir:short"); ap.add_argument("--encoder", default=None, help="dir:tag[:res[:ratio]]")
ap.add_argument("--no-gt", action="store_true", help="skip ground-truth regions"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--stride", type=int, default=1, help="use every k-th image"); ap.add_argument("--fast", action="store_true")
ap.add_argument("--max-pixels", type=float, default=1.2e6)
a = ap.parse_args()

ds = DATASETS[a.dataset](); C = ds.C; fs = ds.files(a.split)[::a.stride]
if a.limit: fs = fs[:a.limit]
seg = enc = None; preps = []
if a.seg: seg = make_segmenter(a.seg); preps.append(seg.prep)
if a.encoder: enc, etag = make_encoder(a.encoder); preps.append(enc.prep)
segtag = seg.tag if seg else "noseg"
outs = {name: f"{CACHE}/rf/{a.dataset}_{a.split}_{segtag}_{name}.pt" for name in (["own"] if seg else []) + ([etag] if enc else [])}
for p in outs.values():
    if os.path.exists(p): raise SystemExit(f"refusing to overwrite {p}")
os.makedirs(f"{CACHE}/rf", exist_ok=True)
KEYS = ("x", "y", "kind", "pc", "purity", "area", "img", "tk_cls", "tk_frac")
buf = {name: [] for name in outs}; done = {name: {k: [] for k in KEYS} for name in outs}


def flush():
    """Move buffered rows to the CPU in one go (one synchronisation per flush instead of several per image)."""
    for name, rows in buf.items():
        if not rows: continue
        cat = [torch.cat([r[j] for r in rows]) for j in range(len(KEYS) + 1)]; keep = cat[-1]
        for k, t in zip(KEYS, cat): done[name][k].append(t[keep].cpu())
        rows.clear()


t0 = time.time()
with torch.inference_mode():
    for n, (xs, gt, i) in enumerate(loader(ds, fs, preps)):
        gt = gt.to(DEV).long(); xs = [x.to(DEV) for x in xs]
        if seg:
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.fast): out = seg(xs[0])
        size = tuple(gt.shape)
        if seg and (a.fast or gt.numel() > a.max_pixels):     # work at the segmenter's resolution: output (fast) or input (very large images)
            size = tuple(v // 4 for v in xs[0].shape[-2:]) if a.fast else tuple(xs[0].shape[-2:])
            gt = F.interpolate(gt[None, None].float(), size=size, mode="nearest")[0, 0].long()
        sources = {}
        if seg: sources["own"] = own_source(out)
        if enc: sources[etag] = enc(xs[-1])
        sets = []                                             # (masks, y, kind, pred class, purity, top-4 classes, top-4 fractions, valid)
        if not a.no_gt:
            ids, m = class_masks(gt); K = len(ids)
            tc = torch.full((K, 4), -1, device=DEV, dtype=torch.long); tc[:, 0] = ids; tf = torch.zeros(K, 4, device=DEV); tf[:, 0] = 1
            sets.append((m, ids, 0, torch.full_like(ids, -1), torch.ones(K, device=DEV), tc, tf, torch.ones(K, dtype=torch.bool, device=DEV)))
        if seg:
            am = seg.posterior(out, size).argmax(0); ids, m = class_masks(am)
            hist = joint_hist(am, gt, gt != 255, C)[ids].float(); nv = hist.sum(1, keepdim=True)
            tf, tc = (hist / nv.clamp(min=1)).topk(4, 1); tc = torch.where(tf > 0, tc, torch.full_like(tc, -1))
            sets.append((m, tc[:, 0].clamp(min=0), 1, ids, tf[:, 0], tc, tf, nv[:, 0] > 0))
        for name, src in sources.items():
            for m, y, kind, pc, purity, tc, tf, valid in sets:
                f, ok = region_features(src, m); K = len(y)
                buf[name].append((f.half(), y, torch.full((K,), kind, dtype=torch.uint8, device=DEV), pc, purity, m.flatten(1).mean(1),
                                  torch.full((K,), int(i), dtype=torch.int32, device=DEV), tc, tf, ok & valid))
        if (n + 1) % 100 == 0: flush()
        if (n + 1) % 1000 == 0: print(a.split, segtag, n + 1, f"{time.time() - t0:.0f}s", flush=True)
flush()
for name, r in done.items():
    d = {k: torch.cat(v) for k, v in r.items()}; d["files"] = len(fs); d["args"] = vars(a); torch.save(d, outs[name])
    print(name, "regions", len(d["y"]), "dim", d["x"].shape[1], "gt", int((d["kind"] == 0).sum()), "pred", int((d["kind"] == 1).sum()), "->", outs[name], flush=True)
print("EXTRACT_DONE")
