#!/usr/bin/env python3
"""Constructed pairs: episodes whose query mask is known by construction, for the annotation-free decision fit.

PLAN 2026-10-04, stage D1. Source images are unlabelled inputs (the TRAIN2400 manifests' images: image-isolated
from DEV241 / CONFIRM600 / registered160). No annotation is read or used anywhere in this script.

For each image: one frozen-encoder pass, k-means on the patch grid, connected components of the clusters. A
component with area 1-60% of the image (border-free preferred) is the region R. Reference window = R's bounding box
expanded 1.35x, its mask is R inside the window. Query window = a jittered window around R (scale log-uniform
0.4-1.8x of the reference window, rotation +-20 deg, flip, brightness / contrast +-15%, JPEG 60-90 when written);
its mask is R inside that window, transformed together with the image. Both masks are exact by construction, and
the region appears at two apparent scales, which is the scale mismatch the decision has to handle. JPEGs and 0/1
PNGs are written in the layout scripts/decision_cache.py reads, plus a manifest in the same format as the other
suites (fold = the source image's collection fold, c = a group of source images, e = 100000 + i).

  python scripts/decision_selfmade_pairs.py --train-manifest SUITE/selfmade_episodes.json \
      --out /root/autodl-tmp/demo9_extent/selfmade_v1 --images 300 --pairs-per-image 4 --seed 0   # under the guard
  python scripts/decision_selfmade_pairs.py --fixture --out /tmp/selfmade_fixture                # CPU, no model

Two constructions (added 2026-10-04, second session), chosen with --mode:
  same   the construction above: reference and query are two windows of one image, the region is one component of a
         6-cluster k-means. Two things make these pairs easier than real episodes and are the reason for `paste`:
         the query shows the same scene as the reference (its background evidence is nearly exact), and the mask is
         by definition one feature cluster (the query's own grouping maps are nearly the answer).
  paste  the region (a component of a k-means with 2, 3, 4 or 6 clusters, so that it can span several fine clusters)
         is cut out of its image, jittered (rotation, flip, brightness, contrast, size) and pasted on ANOTHER
         unlabelled image; the reference window is the region's box expanded 1.2 to 4 times. The query then has a
         background and distractors the reference never showed, and the target appears at a different scale.
  mixed  both, alternating per region; the manifests selfmade_same.json and selfmade_paste.json list each part.

Card. Assumption: the decision map is a property of the host's evidence structure, not of the classes, so pairs
whose mask is known by construction carry it. Prediction: the linear rung fitted on the constructed cache has
cosine >= 0.9 with the label-fitted constants and gives >= +2.0 patch level on DEV241 (label fit +4.12).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

K = 6                    # clusters per image
MIN_AREA, MAX_AREA = 0.01, 0.60     # region area as a fraction of the image
MIN_SHARE, MAX_SHARE = 0.01, 0.85   # share of the region inside the query window
MARGIN = 1               # patch border that counts as touching the border
LIMIT_SIDE = 1600        # crops longer than this are resized down (image and mask together)
GROUPS = 20              # pairs carry c = (index of the source image) % GROUPS and their masks the value c + 1. The fit
                         # (tics/decision_heads.py) chooses its stopping epoch on 20% of the "classes"; with one class
                         # for every pair it would hold out everything, fit nothing and stop after one epoch.


def components(labels, value):
    """4-neighbour connected components of (labels == value) as boolean grids (numpy, no scipy)."""
    import numpy as np
    m = labels == value
    h, w = m.shape
    seen = np.zeros_like(m)
    out = []
    for y in range(h):
        for x in range(w):
            if not m[y, x] or seen[y, x]:
                continue
            stack, cells = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            g = np.zeros_like(m)
            for cy, cx in cells:
                g[cy, cx] = True
            out.append(g)
    return out


def kmeans_torch(feats, k, iters=12, seed=0):
    """k-means on [N, C] features (numpy in, numpy labels out), seeded random init on the unit sphere."""
    import numpy as np
    import torch
    x = torch.from_numpy(feats.astype("float32"))
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    lab = torch.zeros(len(x), dtype=torch.long)
    for _ in range(iters):
        lab = (x @ c.T).argmax(1)
        for j in range(k):
            if int((lab == j).sum()):
                c[j] = x[lab == j].mean(0)
                c[j] = c[j] / c[j].norm().clamp_min(1e-6)
    return lab.numpy()


def pick_regions(labels, n, rng):
    """Up to n connected components of clusters that do not touch the border (fallback: all in range)."""
    import numpy as np
    h, w = labels.shape
    cand = []
    for value in np.unique(labels):
        for g in components(labels, int(value)):
            frac = float(g.sum()) / (h * w)
            if not (MIN_AREA <= frac <= MAX_AREA):
                continue
            border = bool(g[:MARGIN].any() or g[-MARGIN:].any() or g[:, :MARGIN].any() or g[:, -MARGIN:].any())
            cand.append(dict(patch=g, frac=frac, border=border))
    free = [c for c in cand if not c["border"]]
    pool = free or cand
    if not pool:
        return []
    order = rng.permutation(len(pool))
    return [pool[int(j)] for j in order[:n]]


def window_from_region(region_px, rng, image_size, expand=1.35):
    """Reference window = the region's box expanded; query window = a jittered window around the region."""
    import numpy as np
    W, H = image_size
    ys, xs = np.nonzero(region_px)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    cx, cy, rw, rh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    ref_w, ref_h = max(int(rw * expand), 64), max(int(rh * expand), 64)
    rx0, ry0 = int(max(0, min(cx - ref_w / 2, W - ref_w))), int(max(0, min(cy - ref_h / 2, H - ref_h)))
    ref_box = (rx0, ry0, min(W, rx0 + ref_w), min(H, ry0 + ref_h))
    s = float(np.exp(rng.uniform(np.log(0.4), np.log(1.8))))
    aspect = float(np.exp(rng.uniform(np.log(0.85), np.log(1.18))))
    qw, qh = max(int(ref_w * s * aspect), 48), max(int(ref_h * s / aspect), 48)
    jx, jy = rng.uniform(-0.2, 0.2) * qw, rng.uniform(-0.2, 0.2) * qh
    qx0 = int(max(0, min(cx + jx - qw / 2, W - min(qw, W))))
    qy0 = int(max(0, min(cy + jy - qh / 2, H - min(qh, H))))
    qbox = (qx0, qy0, min(W, qx0 + qw), min(H, qy0 + qh))
    return ref_box, qbox, s


def cap_both(img, mask, limit=LIMIT_SIDE):
    """Resize a crop and its mask together when the long side exceeds the limit."""
    from PIL import Image
    if max(img.size) <= limit:
        return img, mask
    s = limit / max(img.size)
    size = (max(int(img.width * s), 1), max(int(img.height * s), 1))
    return img.resize(size, Image.BILINEAR), mask.resize(size, Image.NEAREST)


def jitter_query(image, mask, rng):
    """Rotation, flip, brightness and contrast on the query window; image and mask are transformed together."""
    import numpy as np
    from PIL import Image, ImageEnhance
    theta = float(rng.uniform(-20, 20))
    image = image.rotate(theta, resample=Image.BILINEAR, expand=False, fillcolor=(124, 116, 104))
    mask = mask.rotate(theta, resample=Image.NEAREST, expand=False, fillcolor=0)
    if rng.random() < 0.5:
        image, mask = image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    image = ImageEnhance.Brightness(image).enhance(float(np.exp(rng.uniform(-0.16, 0.16))))
    image = ImageEnhance.Contrast(image).enhance(float(np.exp(rng.uniform(-0.16, 0.16))))
    return image, mask


def paste_query(obj, obj_mask, back, rng):
    """A region cut out of its image, jittered and pasted on another image. obj: RGB crop of the region's box;
    obj_mask: mode L crop with 0/1; back: RGB image. Returns (query image, query mask with 0/1, relative size)."""
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    theta = float(rng.uniform(-20, 20))
    m = obj_mask.point(lambda v: 255 if v else 0)
    obj = obj.rotate(theta, resample=Image.BILINEAR, expand=True)
    m = m.rotate(theta, resample=Image.NEAREST, expand=True, fillcolor=0)
    if rng.random() < 0.5:
        obj, m = obj.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT)
    obj = ImageEnhance.Brightness(obj).enhance(float(np.exp(rng.uniform(-0.16, 0.16))))
    obj = ImageEnhance.Contrast(obj).enhance(float(np.exp(rng.uniform(-0.16, 0.16))))
    W, H = back.size
    u = float(np.exp(rng.uniform(np.log(0.12), np.log(0.7))))  # the region's longer side as a share of the shorter image side
    s = u * min(W, H) / max(obj.size)
    size = (max(int(round(obj.width * s)), 8), max(int(round(obj.height * s)), 8))
    if size[0] > W or size[1] > H:
        return None
    obj, m = obj.resize(size, Image.BILINEAR), m.resize(size, Image.NEAREST)
    x0, y0 = int(rng.integers(0, W - size[0] + 1)), int(rng.integers(0, H - size[1] + 1))
    out = back.copy()
    out.paste(obj, (x0, y0), m.filter(ImageFilter.GaussianBlur(1.0)))  # a soft edge for the pixels, the exact mask for the label
    mask = Image.new("L", back.size, 0)
    mask.paste(m.point(lambda v: 1 if v > 127 else 0), (x0, y0))
    return out, mask, u


def grid_to_pixels(patch, height, width):
    """Upsample a patch grid to the image's pixels by index mapping (exact to the last row and column)."""
    import numpy as np
    h, w = patch.shape
    iy = (np.arange(height) * h // height).clip(0, h - 1)
    ix = (np.arange(width) * w // width).clip(0, w - 1)
    return patch[np.ix_(iy, ix)].copy()


def encode_grid(host, image):
    """The frozen encoder's patch grid for one image, through the same transform FoRIS uses."""
    x = host._transform(image).unsqueeze(0).to(host.device)
    feats = host.encoder.get_intermediate_layers(x, n=1, reshape=True)[0][0].float()  # [C, h, w]
    # `feats` already has its batch axis removed: flatten spatial axes and transpose
    # to [h*w, C]. A further [0] here silently kept one channel and broke reshape.
    return feats.flatten(1).transpose(0, 1).cpu().numpy(), tuple(feats.shape[-2:])


def make_fixture_image(rng):
    """A CPU stand-in for a val2014 image: colour blocks, so the cluster/component code has structure to find."""
    import numpy as np
    from PIL import Image
    img = np.zeros((256, 256, 3), "uint8")
    img[:] = (np.array([150, 140, 130]) + rng.integers(-15, 15, 3)).clip(0, 255)
    for _ in range(4):
        y, x = int(rng.integers(20, 200)), int(rng.integers(20, 200))
        img[y:y + 48, x:x + 48] = rng.integers(0, 255, 3)
    return Image.fromarray(img)


def build(args):
    import numpy as np
    from PIL import Image
    if not args.fixture and not args.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    for d in ("data/ref", "data/query", "ann/ref", "ann/query"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paste_k = [int(v) for v in args.paste_k.split(",")]

    if args.fixture:
        man = dict(data_root=str(out / "source"))
        images, host = ["fixture-%02d" % i for i in range(args.images)], None
    else:
        man = json.loads(Path(args.train_manifest).read_text())
        fold_of = {}
        for r in man["episodes"]:
            fold_of.setdefault(r["support"], r["fold"])
            fold_of.setdefault(r["query"], r["fold"])
        images = sorted(fold_of)
        rng.shuffle(images)
        images = images[:args.images]
        from extent_experiment import build_host
        host = build_host(args, man, "cuda")

    rows, receipt, rejected = [], [], dict(no_region=0, ref_share=0, query_share=0, short=0)
    for i, name in enumerate(images):
        if args.fixture:
            img = make_fixture_image(rng)
            blocks = np.asarray(img, "float32").reshape(64, 4, 64, 4, 3).mean((1, 3))
            feats, shape = blocks.reshape(-1, 3), (64, 64)
            fold = i % 4
        else:
            img = Image.open(Path(man["data_root"]) / name).convert("RGB")
            feats, shape = encode_grid(host, img)
            fold = fold_of[name]
        n_same = dict(same=args.pairs_per_image, paste=0, mixed=args.pairs_per_image // 2)[args.mode]
        todo = []
        if n_same:
            labels = kmeans_torch(feats, K, seed=args.seed + i).reshape(shape)
            todo += [("same", reg) for reg in pick_regions(labels, n_same, rng)]
        if args.pairs_per_image - n_same:
            coarse = int(rng.choice(paste_k))
            labels = kmeans_torch(feats, coarse, seed=args.seed + i).reshape(shape)
            todo += [("paste", reg) for reg in pick_regions(labels, args.pairs_per_image - n_same, rng)]
        for made, reg in todo:
            region_px = grid_to_pixels(reg["patch"], img.height, img.width)
            expand = 1.35 if made == "same" else float(np.exp(rng.uniform(np.log(1.2), np.log(4.0))))
            ref_box, qbox, scale = window_from_region(region_px, rng, img.size, expand=expand)
            rmask = Image.fromarray((region_px[ref_box[1]:ref_box[3], ref_box[0]:ref_box[2]] * 1).astype("uint8"))
            if np.asarray(rmask).sum() < 16:
                rejected["ref_share"] += 1
                continue
            if made == "same":
                q_mask = Image.fromarray((region_px[qbox[1]:qbox[3], qbox[0]:qbox[2]] * 1).astype("uint8"))
                share0 = float(np.asarray(q_mask).mean())
                if not (MIN_SHARE <= share0 <= MAX_SHARE) or np.asarray(q_mask).sum() < 32:
                    rejected["query_share"] += 1
                    continue
                q_img, q_mask = cap_both(img.crop(qbox), q_mask)
                q_img, q_mask = jitter_query(q_img, q_mask, rng)
            else:
                ys, xs = np.nonzero(region_px)
                box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                if args.fixture:
                    back = make_fixture_image(rng)
                else:
                    other = images[int(rng.integers(0, len(images)))]
                    if other == name:
                        other = images[(i + 1) % len(images)]
                    back = Image.open(Path(man["data_root"]) / other).convert("RGB")
                    if max(back.size) > LIMIT_SIDE:
                        back = cap_both(back, Image.new("L", back.size, 0))[0]
                got = paste_query(img.crop(box), Image.fromarray((region_px[box[1]:box[3], box[0]:box[2]] * 1).astype("uint8")), back, rng)
                if got is None:
                    rejected["query_share"] += 1
                    continue
                q_img, q_mask, scale = got
                share0 = float(np.asarray(q_mask).mean())
                if not (MIN_SHARE <= share0 <= MAX_SHARE) or np.asarray(q_mask).sum() < 32:
                    rejected["query_share"] += 1
                    continue
            ref_img, rmask = cap_both(img.crop(ref_box), rmask)
            if np.asarray(q_mask).sum() < 16:
                rejected["short"] += 1
                continue
            c = i % GROUPS  # see GROUPS: the stopping epoch of the fit is chosen on held-out groups of source images
            rmask, q_mask = (m.point(lambda v, c=c: c + 1 if v else 0) for m in (rmask, q_mask))
            rid, qid = "r%05d" % len(rows), "q%05d" % len(rows)
            ref_img.save(out / "data/ref" / (rid + ".jpg"), quality=88)
            q_img.save(out / "data/query" / (qid + ".jpg"), quality=int(rng.integers(60, 91)))
            rmask.save(out / "ann/ref" / (rid + ".png"))
            q_mask.save(out / "ann/query" / (qid + ".png"))
            rows.append(dict(fold=int(fold), e=100000 + len(rows), c=c,
                             support="ref/%s.jpg" % rid, query="query/%s.jpg" % qid, made=made))
            receipt.append(dict(source=str(name), fold=int(fold), made=made, region_frac=reg["frac"], ref_box=ref_box,
                                query_box=qbox, scale=scale, ref_share=float((np.asarray(rmask) > 0).mean()),
                                query_share=float((np.asarray(q_mask) > 0).mean())))
        if (i + 1) % 20 == 0:
            print(json.dumps(dict(image=i + 1, of=len(images), pairs=len(rows), rejected=rejected)), flush=True)
        if args.fixture and len(rows) >= args.pairs_per_image * args.images:
            break

    source_man = dict() if args.fixture else json.loads(Path(args.train_manifest).read_text())
    suite = dict(state="PREPARED", data_root=str(out / "data"), annotation_root=str(out / "ann"),
                 foris_root=source_man.get("foris_root", ""), projection_basis=source_man.get("projection_basis", ""),
                 seed=args.seed, episodes=rows)
    (out / "selfmade_episodes.json").write_text(json.dumps(suite))
    for part in ("same", "paste"):  # each construction alone, for a fit on one of them
        (out / ("selfmade_%s.json" % part)).write_text(json.dumps(dict(suite, episodes=[r for r in rows if r["made"] == part])))
    pct = lambda key: [float(np.percentile([r[key] for r in receipt], p)) for p in (5, 50, 95)] if receipt else []
    rep = dict(state="COMPLETED", pairs=len(rows), images=len(images), seed=args.seed, k_clusters=K, rejected=rejected, mode=args.mode,
               made={part: sum(r["made"] == part for r in rows) for part in ("same", "paste")},
               query_share_by_made={part: [float(np.percentile([r["query_share"] for r in receipt if r["made"] == part] or [0], p)) for p in (5, 50, 95)]
                                    for part in ("same", "paste")},
               query_share=pct("query_share"), scale=pct("scale"), region_frac=pct("region_frac"),
               pairs_receipt=receipt if args.fixture else len(receipt))
    (out / "build_receipt.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "pairs_receipt"}), flush=True)


def verify_fixture(out):
    """CPU identity check: masks are 0/1, sit inside their crops, and match the crops' sizes."""
    import numpy as np
    from PIL import Image
    out = Path(out)
    suite = json.loads((out / "selfmade_episodes.json").read_text())
    rep = json.loads((out / "build_receipt.json").read_text())
    assert len(suite["episodes"]) == rep["pairs"] > 0, "no pairs were written"
    for row in suite["episodes"]:
        q = np.asarray(Image.open(out / "ann" / Path(row["query"]).with_suffix(".png")))
        r = np.asarray(Image.open(out / "ann" / Path(row["support"]).with_suffix(".png")))
        assert set(np.unique(q)) <= {0, row["c"] + 1} and set(np.unique(r)) <= {0, row["c"] + 1}, "masks must be 0 or c + 1"
        assert q.sum() > 16 and r.sum() > 16, "empty mask"
        qimg = Image.open(out / "data/query" / Path(row["query"]).name)
        rimg = Image.open(out / "data/ref" / Path(row["support"]).name)
        assert qimg.size == (q.shape[1], q.shape[0]), "query image and mask sizes differ: %s" % row["query"]
        assert rimg.size == (r.shape[1], r.shape[0]), "reference image and mask sizes differ: %s" % row["support"]
    parts = {part: json.loads((out / ("selfmade_%s.json" % part)).read_text())["episodes"] for part in ("same", "paste")}
    assert len(parts["same"]) + len(parts["paste"]) == len(suite["episodes"]), "the two part manifests must partition the pairs"
    assert len({r["e"] for r in suite["episodes"]}) == len(suite["episodes"]), "episode numbers must be unique"
    assert len({r["c"] for r in suite["episodes"]}) >= min(5, rep["images"]), "the pairs must fall into several groups"
    print(json.dumps(dict(state="FIXTURE_VERIFIED", pairs=len(suite["episodes"]), same=len(parts["same"]), paste=len(parts["paste"]))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-manifest", help="a suite manifest; only its image list, folds and roots are used")
    p.add_argument("--out", required=True)
    p.add_argument("--images", type=int, default=300)
    p.add_argument("--pairs-per-image", type=int, default=4, help="at most this many regions per image")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("same", "paste", "mixed"), default="same")
    p.add_argument("--paste-k", default="2,3,4,6", help="cluster counts drawn per image for the pasted regions")
    p.add_argument("--fixture", action="store_true", help="CPU, synthetic images, then verify the written pairs")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    a = p.parse_args()
    if a.fixture:
        build(a)
        verify_fixture(a.out)
    elif not a.train_manifest:
        raise SystemExit("--train-manifest is required (or --fixture)")
    else:
        build(a)


if __name__ == "__main__":
    main()
