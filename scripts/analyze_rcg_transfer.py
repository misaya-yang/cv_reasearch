#!/usr/bin/env python3
"""Audit saved complete outputs; GT is used only for diagnostic stratification.

No inference, parameter selection, or new method. Original-resolution I/U are
read from the sealed records. Net TP/FP changes cannot identify four edit sets.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


GROUPS = {
    "COCO B+C+D": ["groupB", "groupC", "groupD"],
    "PASCAL-Part": ["pascal_part"],
    "SUIM": ["suim"],
    "PACO-Part": [f"paco_part_f{i}" for i in range(4)],
    "LVIS-92i": [f"lvis_f{i}" for i in range(4)],
}
ARMS = ("native", "rcg", "rcg_fine", "rcg2")


def aggregate(iu, classes, select=None):
    if select is None:
        select = np.ones(len(classes), bool)
    labels = np.unique(classes[select])
    counts = np.array([iu[select & (classes == c)].sum(0) for c in labels])
    return labels, counts.astype(float)


def mean_iou(iu, classes, select=None):
    _, counts = aggregate(iu, classes, select)
    return float(np.mean(counts[:, 0] / np.maximum(counts[:, 1], 1)) * 100)


def comparison(base, new, classes):
    labels, b = aggregate(base, classes)
    _, n = aggregate(new, classes)
    j = b[:, 0] / np.maximum(b[:, 1], 1)
    di = n[:, 0] - b[:, 0]
    du = n[:, 1] - b[:, 1]  # Same GT => delta union equals delta false positives.
    tp = 100 * di / np.maximum(n[:, 1], 1)
    fp = -100 * j * du / np.maximum(n[:, 1], 1)
    delta = 100 * (n[:, 0] / np.maximum(n[:, 1], 1) - j)
    assert np.max(np.abs(tp + fp - delta)) < 1e-10
    return {
        "delta_pp": float(delta.mean()),
        "net_true_coverage_contribution_pp": float(tp.mean()),
        "net_false_foreground_contribution_pp": float(fp.mean()),
        "net_tp_pixels": int(di.sum()), "net_fp_pixels": int(du.sum()),
        "classes_better": int((delta > 1e-9).sum()),
        "classes_worse": int((delta < -1e-9).sum()),
        "per_class": [dict(c=str(c), delta_pp=float(d), tp_pp=float(t), fp_pp=float(f),
                           base_iou=float(old), new_iou=float(old + d / 100))
                      for c, d, t, f, old in zip(labels, delta, tp, fp, j)],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    report = {"protocol": {
        "metric": "original-resolution per-class pooled intersection/union, macro mean",
        "exposure": "reanalysis of reused development packs; no independent confirmation",
        "decomposition": "mean_c [(I_new-I_base)/U_new - J_base*(U_new-U_base)/U_new]",
        "limits": "net changes, not four exclusive edit sets; strata use GT; no causal attribution",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, "datasets": {}}
    for dataset, packs in GROUPS.items():
        counts = {k: [] for k in ARMS}
        classes, areas, fields, sources = [], [], [], []
        for pack in packs:
            root = args.outputs / ("claude_rcg2_" + pack)
            rp, zp = root / "rows.json", root / "counts.npz"
            rows = json.loads(rp.read_text())
            with np.load(zp, allow_pickle=False) as z:
                for k in ARMS:
                    assert z["orig:" + k].shape == (len(rows), 2)
                    counts[k].append(z["orig:" + k])
                areas.append(z["truth"].astype(float) / 1024**2)
                fields.append(z["fields"].astype(np.float32).reshape(len(rows), -1))
            # Disjoint benchmark folds use local dense IDs; COCO batches share IDs.
            prefix = pack + ":" if dataset in ("PACO-Part", "LVIS-92i") else ""
            classes.extend(prefix + str(r["c"]) for r in rows)
            sources.append({"path": str(root), "n": len(rows),
                            "rows_sha256": hashlib.sha256(rp.read_bytes()).hexdigest(),
                            "counts_sha256": hashlib.sha256(zp.read_bytes()).hexdigest()})
        counts = {k: np.concatenate(v) for k, v in counts.items()}
        classes, areas, fields = np.array(classes), np.concatenate(areas), np.concatenate(fields)
        out = {"n": len(classes), "classes": len(np.unique(classes)), "sources": sources,
               "miou": {k: mean_iou(v, classes) for k, v in counts.items()},
               "rcg_vs_native": comparison(counts["native"], counts["rcg"], classes),
               "fine_vs_rcg": comparison(counts["rcg"], counts["rcg_fine"], classes),
               "full_vs_fine": comparison(counts["rcg_fine"], counts["rcg2"], classes),
               "gt_area_strata": [],
               "saved_fp16_field": {"empty_at_half": int((fields.max(1) <= .5).sum()),
                                      "all_at_half": int((fields.min(1) > .5).sum()),
                                      "min": float(fields.min()), "max": float(fields.max())}}
        for lo, hi in ((0, .02), (.02, .1), (.1, .3), (.3, 1.01)):
            sel = (areas >= lo) & (areas < hi)
            if sel.any():
                out["gt_area_strata"].append({"range": [lo, hi], "n": int(sel.sum()),
                    "classes": int(len(np.unique(classes[sel]))),
                    "miou": {k: mean_iou(v, classes, sel) for k, v in counts.items()}})
        report["datasets"][dataset] = out
        print(dataset, json.dumps({k: v for k, v in out.items() if k in ("n", "classes", "miou", "gt_area_strata", "saved_fp16_field")}))
        print("NET", {k: v for k, v in out["rcg_vs_native"].items() if k != "per_class"}, flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
