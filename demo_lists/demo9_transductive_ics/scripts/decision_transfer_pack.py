#!/usr/bin/env python3
"""Episode packs of the other benchmarks in the layout of our manifests, so that the read-out fitted on COCO base
classes can be read on them WITHOUT refitting (CPU only: images and masks are copied, no model is loaded).

  python scripts/decision_transfer_pack.py --dataset lvis --like SUITE/dev_episodes.json --out transfer_v1/lvis \
      --episodes 300
  python scripts/decision_transfer_pack.py --fixture --out /tmp/pack_fixture

Episodes come from INSID3's own loaders (`/root/autodl-tmp/demo4/INSID3/datasets`: lvis, pascal_part, paco_part,
suim, lung), 1 shot, numpy seed fixed per fold, the first --episodes / len(--folds) items of each fold. They are not
INSID3's exact evaluation lists (its worker processes draw differently); every comparison made on a pack is paired
on the pack's own episodes. Images are stored losslessly (PNG), masks as PNG with the value c + 1 on the target,
where c is a dense class index of the pack. `fold` in a row is i % 4: it only says which of the four COCO fold
read-outs scores the episode (none of them was fitted on this benchmark). Images whose longer side exceeds
--limit-side are resized together with their masks (counted in the receipt).
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

FOLDS = dict(lvis="0,1,2,3", pascal_part="0,1,2,3", paco_part="0,1,2,3", suim="0", lung="0")


def to_mask(m):
    """A loader's mask (tensor or array, any dtype, possibly with a leading axis) as a 2-D bool array."""
    import numpy as np
    m = np.asarray(m.cpu().numpy() if hasattr(m, "cpu") else m)
    while m.ndim > 2:
        m = m[0]
    return m > 0.5


def cap(img, mask, limit):
    """Image and mask resized together when the longer side exceeds the limit. Returns (image, mask, resized)."""
    from PIL import Image
    if max(img.size) <= limit:
        return img, mask, False
    s = limit / max(img.size)
    size = (max(int(round(img.width * s)), 1), max(int(round(img.height * s)), 1))
    return img.resize(size, Image.BILINEAR), mask.resize(size, Image.NEAREST), True


def write(out, items, like, name, seed, limit):
    """items: iterable of (reference image, reference mask, query image, query mask, class key, source fold)."""
    import numpy as np
    from PIL import Image
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    for d in ("data/ref", "data/query", "ann/ref", "ann/query"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows, dense, skipped, resized = [], {}, dict(size=0, empty=0), 0
    for ref, ref_mask, query, query_mask, key, fold in items:
        ref_mask, query_mask = to_mask(ref_mask), to_mask(query_mask)
        if ref_mask.shape != ref.size[::-1] or query_mask.shape != query.size[::-1]:
            skipped["size"] += 1
            continue
        if ref_mask.sum() < 16 or query_mask.sum() < 1:
            skipped["empty"] += 1
            continue
        c = dense.setdefault(key, len(dense))
        if c > 253:
            raise SystemExit("more than 254 classes in one pack")
        i = len(rows)
        for kind, img, mask in (("ref", ref, ref_mask), ("query", query, query_mask)):
            img, m, did = cap(img.convert("RGB"), Image.fromarray((mask * (c + 1)).astype("uint8")), limit)
            resized += int(did)
            img.save(out / "data" / kind / ("%05d.png" % i), compress_level=1)  # lossless either way; level 1 is several times faster
            m.save(out / "ann" / kind / ("%05d.png" % i))
        rows.append(dict(fold=i % 4, e=200000 + i, c=c, support="ref/%05d.png" % i, query="query/%05d.png" % i,
                         source_fold=int(fold), source_class=str(key), query_share=float(query_mask.mean())))
    suite = dict(state="PREPARED", dataset=name, data_root=str(out / "data"), annotation_root=str(out / "ann"),
                 foris_root=like.get("foris_root", ""), projection_basis=like.get("projection_basis", ""), seed=seed, episodes=rows)
    (out / "episodes.json").write_text(json.dumps(suite))
    rep = dict(state="COMPLETED", dataset=name, episodes=len(rows), classes=len(dense), skipped=skipped, resized_images=resized,
               query_share=[float(np.percentile([r["query_share"] for r in rows], p)) for p in (5, 50, 95)] if rows else [])
    (out / "pack_receipt.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    return rep


def from_loader(a):
    import numpy as np
    import torch
    sys.path.insert(0, str(a.insid3))
    from datasets import build_dataset
    folds = [int(v) for v in (a.folds or FOLDS[a.dataset]).split(",")]
    per = -(-a.episodes // len(folds))
    for fold in folds:
        np.random.seed(a.seed)
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        ds = build_dataset(a.dataset, SimpleNamespace(data_root=str(a.data_root), fold=fold, shots=1))
        for idx in range(min(per, len(ds))):
            b = ds[idx]
            yield b["ref_imgs"][0], b["ref_masks"][0], b["tgt_img"], b["tgt_mask"], "%d:%d" % (fold, int(b["class_id"])), fold
        del ds


def fixture(a):
    """No dataset: synthetic items through the same writer, then the checks the reader relies on."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)

    def items():
        for i in range(10):
            w, h = (int(v) for v in rng.integers(80, 300, 2))
            imgs = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(2)]
            masks = [np.zeros((h, w), np.float32) for _ in range(2)]
            for m in masks:
                m[h // 4:h // 2, w // 4:w // 2] = 1
            if i == 3:
                masks[1] = masks[1][:-1]  # a wrong size: must be skipped
            if i == 4:
                masks[0][:] = 0           # an empty reference: must be skipped
            yield imgs[0], masks[0], imgs[1], masks[1], "0:%d" % (i % 3), 0
    rep = write(a.out, items(), dict(foris_root="F", projection_basis="P"), "fixture", 0, 200)
    out = Path(a.out)
    man = json.loads((out / "episodes.json").read_text())
    checks = [("two bad items skipped", rep["episodes"] == 8 and rep["skipped"] == dict(size=1, empty=1)),
              ("three classes", rep["classes"] == 3), ("state and roots", man["state"] == "PREPARED" and man["foris_root"] == "F")]
    for r in man["episodes"]:
        for k in ("support", "query"):
            img = Image.open(Path(man["data_root"]) / r[k])
            ann = np.asarray(Image.open(Path(man["annotation_root"]) / Path(r[k]).with_suffix(".png")))
            checks.append(("%s %s: sizes agree, mask is c + 1, longer side capped" % (k, r[k]),
                           img.size == ann.shape[::-1] and set(np.unique(ann)) == {0, r["c"] + 1} and max(img.size) <= 200))
    checks.append(("episode numbers unique, folds cycle", len({r["e"] for r in man["episodes"]}) == 8 and [r["fold"] for r in man["episodes"]][:5] == [0, 1, 2, 3, 0]))
    bad = [n for n, ok in checks if not ok]
    print("%d/%d checks pass%s" % (len(checks) - len(bad), len(checks), "" if not bad else "; FAILED: " + "; ".join(bad)))
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(FOLDS))
    p.add_argument("--out", required=True)
    p.add_argument("--like", type=Path, help="one of our manifests; its FoRIS root and projection basis are copied")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--folds", help="comma separated folds of the benchmark (default: its first four, or its only one)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-side", type=int, default=1600)
    p.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/datasets/ics"))
    p.add_argument("--insid3", type=Path, default=Path("/root/autodl-tmp/demo4/INSID3"))
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args()
    if a.fixture:
        fixture(a)
    write(a.out, from_loader(a), json.loads(a.like.read_text()), a.dataset, a.seed, a.limit_side)


if __name__ == "__main__":
    main()
