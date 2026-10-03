#!/usr/bin/env python3
"""What the wrongly included and wrongly excluded regions are, from the official COCO-20i class maps. CPU only.

  python scripts/diagnose_extent_regions.py --run results/extent_v1/run --manifest results/extent_v1/episodes.json \
      --out results/extent_v1/region_diagnosis.json

The class map of a query gives every annotated thing class per pixel (0 = not annotated). A false positive of the
native FoRIS mask is then on another annotated class, which either also appears in the reference image (the reference
could have named it as background) or does not, or on pixels with no annotation. Uses query labels: a ledger of
which information is missing, not a method.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_extent import SMALL, load, miou


def bits(z, k):
    return np.unpackbits(z[k])[:1 << 20].reshape(1024, 1024).astype(bool)


def class_map(path):
    return np.asarray(Image.open(path).resize((1024, 1024), Image.NEAREST))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    ann = Path(json.loads(a.manifest.read_text())["annotation_root"])
    recs = load(a.run)
    rows, pairs, agree = [], Counter(), []
    for r in recs:
        z = np.load(a.run / "packets" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])))
        pred, truth = bits(z, "native"), bits(z, "truth")
        q = class_map(ann / Path(r["query"]).with_suffix(".png"))
        s = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png")))
        mine = q == r["c"] + 1
        agree.append(float((mine & truth).sum() / max((mine | truth).sum(), 1)))
        in_ref = np.isin(q, np.setdiff1d(np.unique(s), [0, r["c"] + 1]))
        other = (q > 0) & ~mine
        fp = pred & ~truth
        kinds = dict(other_class_in_reference=fp & other & in_ref, other_class_not_in_reference=fp & other & ~in_ref,
                     not_annotated=fp & (q == 0))
        i = int((pred & truth).sum())
        row = dict(native=[i, int((pred | truth).sum())], fp=int(fp.sum()), fn=int((truth & ~pred).sum()))
        for k, m in kinds.items():
            row[k] = int(m.sum())
            keep = pred & ~m
            row["drop_" + k] = [i, int((keep | truth).sum())]
        row["drop_all_other_classes"] = [i, int(((pred & ~(fp & other)) | truth).sum())]
        # Other annotated classes present in the query at all: could a competitor exist?
        row["query_has_other_class"] = bool(other.any())
        for k in np.unique(q[fp & other]):
            pairs[(int(r["c"]), int(k) - 1)] += int((fp & (q == k)).sum())
        rows.append(row)
    cls = np.array([r["c"] for r in recs])
    area = np.array([r["area"] for r in recs])
    col = lambda k: np.array([x[k] for x in rows], float)
    score = lambda k, m=slice(None): float(miou(col(k)[m, 0], col(k)[m, 1], cls[m])[0])
    tot = lambda k: float(col(k).sum())
    out = dict(state="DIAGNOSED", episodes=len(recs), class_map_agrees_with_packet_truth_mean_iou=float(np.mean(agree)),
               scope="model-size masks, native FoRIS after refinement; uses query labels",
               false_positive_pixels=dict(other_class_in_reference=tot("other_class_in_reference") / max(tot("fp"), 1),
                                          other_class_not_in_reference=tot("other_class_not_in_reference") / max(tot("fp"), 1),
                                          not_annotated=tot("not_annotated") / max(tot("fp"), 1)),
               class_miou={k: score(k) for k in ("native", "drop_other_class_in_reference", "drop_other_class_not_in_reference",
                                                 "drop_not_annotated", "drop_all_other_classes")},
               small={k: score(k, area < SMALL) for k in ("native", "drop_all_other_classes", "drop_not_annotated")},
               large={k: score(k, area >= SMALL) for k in ("native", "drop_all_other_classes", "drop_not_annotated")},
               episodes_whose_query_has_another_annotated_class=int(sum(x["query_has_other_class"] for x in rows)),
               top_pairs_target_then_included=[[t, o, n] for (t, o), n in pairs.most_common(15)])
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
