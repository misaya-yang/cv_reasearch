#!/usr/bin/env python3
"""Check the datasets prepared by get_data.sh, without a GPU.

  check_data.py [--data /root/autodl-tmp/datasets/ics] [--sets coco,lvis,...] [--coco-compare N]

For every dataset: build INSID3's own loader (fold 0, one shot), draw three episodes with a fixed seed and print the
image size, the foreground share of the query and reference masks, and how many classes and images the fold has.
A dataset passes when the three episodes load and every mask has foreground.

--coco-compare N: compare the official COCO-20i masks with the masks rebuilt from the instance annotations
(/root/demo4_cache/data/COCO2014, used for every number so far) on the first N val2014 files (0 = all), and compare
the split files. Prints the share of identical files and of differing pixels.
"""
import argparse, os, pickle, sys, types
import numpy as np
from _paths import DEMO4
sys.path.insert(0, os.path.join(DEMO4, "INSID3"))

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/root/autodl-tmp/datasets/ics")
ap.add_argument("--sets", default="coco,lvis,paco_part,pascal_part,isic,lung,suim,isaid")
ap.add_argument("--rebuilt", default="/root/demo4_cache/data/COCO2014")
ap.add_argument("--coco-compare", type=int, default=-1)
a = ap.parse_args()

from datasets import build_dataset   # INSID3's registry
from PIL import Image

ok = {}
for name in a.sets.split(","):
    try:
        ds = build_dataset(name, types.SimpleNamespace(data_root=a.data, fold=0, shots=1))
        np.random.seed(0); rows = []
        for i in range(3):
            b = ds[i]; q = np.asarray(b["tgt_mask"]); r = np.asarray(b["ref_masks"][0])
            rows.append((tuple(b["tgt_img"].size), float((q > 0).mean()), float((r > 0).mean()), int(b["class_id"])))
        n_cls = len(getattr(ds, "class_ids", [])); meta = getattr(ds, "img_metadata_classwise", None)
        n_img = len({x if isinstance(x, str) else str(x) for v in meta.values() for x in v}) if isinstance(meta, dict) else -1
        good = all(q > 0 and r > 0 for _, q, r, _ in rows); ok[name] = good
        print(f"{'PASS' if good else 'EMPTY-MASK':10s} {name:12s} episodes {len(ds):5d}  classes in fold {n_cls:3d}  images {n_img:6d}  "
              + "  ".join(f"{s[0]}x{s[1]} q{q:.3f} r{r:.3f} c{c}" for s, q, r, c in rows), flush=True)
    except Exception as e:
        ok[name] = False; print(f"{'FAIL':10s} {name:12s} {type(e).__name__}: {str(e)[:200]}", flush=True)

if a.coco_compare >= 0:
    off, reb = os.path.join(a.data, "COCO2014"), a.rebuilt
    print("\nCOCO-20i: official masks against rebuilt masks")
    for f in range(4):
        A = pickle.load(open(f"{off}/splits/val/fold{f}.pkl", "rb")); B = pickle.load(open(f"{reb}/splits/val/fold{f}.pkl", "rb"))
        print(f"  split fold{f}: identical {A == B}")
    names = sorted(os.listdir(f"{off}/annotations/val2014")); have = set(os.listdir(f"{reb}/annotations/val2014"))
    print(f"  val2014 masks: official {len(names)}, rebuilt {len(have)}, in both {len(have & set(names))}")
    names = [n for n in names if n in have]; names = names[:a.coco_compare] if a.coco_compare else names
    same = diff_px = tot_px = shape_bad = 0; worst = []
    for n in names:
        x = np.array(Image.open(f"{off}/annotations/val2014/{n}")); y = np.array(Image.open(f"{reb}/annotations/val2014/{n}"))
        if x.shape != y.shape: shape_bad += 1; continue
        d = int((x != y).sum()); same += d == 0; diff_px += d; tot_px += x.size
        if d: worst.append((d / x.size, n))
    worst.sort(reverse=True)
    print(f"  compared {len(names)} files: identical {same} ({100 * same / max(len(names), 1):.2f}%), shape mismatch {shape_bad}, "
          f"differing pixels {100 * diff_px / max(tot_px, 1):.4f}%")
    print("  largest differences:", [(f"{100 * d:.1f}%", n) for d, n in worst[:5]])
    for d, n in worst[:3]:      # what differs: class histograms of the two versions
        x = np.array(Image.open(f"{off}/annotations/val2014/{n}")); y = np.array(Image.open(f"{reb}/annotations/val2014/{n}"))
        hx = {int(k): int(v) for k, v in zip(*np.unique(x, return_counts=True))}; hy = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        print(f"    {n}: official {hx}\n    {' ' * len(n)}  rebuilt  {hy}")
    # the unit that matters for the metric: the binary mask of one class in one image of the split
    ious, empty_off, empty_reb, n_pairs = [], 0, 0, 0; per_fold = []
    for f in range(4):
        split = pickle.load(open(f"{off}/splits/val/fold{f}.pkl", "rb")); cache = {}; fi = []
        for c, imgs in split.items():
            for name in imgs:
                png = os.path.basename(name).replace(".jpg", ".png")
                if png not in have: continue
                if png not in cache:
                    cache[png] = (np.array(Image.open(f"{off}/annotations/val2014/{png}")), np.array(Image.open(f"{reb}/annotations/val2014/{png}")))
                x, y = cache[png]; p, q = x == c + 1, y == c + 1; u = (p | q).sum(); n_pairs += 1
                empty_off += not p.any(); empty_reb += not q.any()
                if u: fi.append((p & q).sum() / u)
        per_fold.append(float(np.mean(fi))); ious += fi
    ious = np.array(ious)
    print(f"  (image, class) pairs in the val splits: {n_pairs}; class missing in official {empty_off}, in rebuilt {empty_reb}")
    print(f"  IoU(official, rebuilt) per pair: mean {ious.mean():.4f}, per fold {[round(v, 4) for v in per_fold]}, "
          f"below 0.99: {100 * (ious < 0.99).mean():.1f}%, below 0.9: {100 * (ious < 0.9).mean():.1f}%, below 0.5: {100 * (ious < 0.5).mean():.1f}%")

print("\nsummary:", {k: ("ok" if v else "NOT READY") for k, v in ok.items()})
