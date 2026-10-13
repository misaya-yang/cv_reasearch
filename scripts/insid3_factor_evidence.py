"""Post-seal audit of identity ranking versus area expansion in INSID3.

No new segmentation candidate or model inference. Both masks were sealed by
the experiment agent before query labels were used. The diagnostic uses exact
added/deleted pixels, and evaluates existing cluster scores at the same ROI.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO.parent / "cv_data"
ROOT = ASSETS / "a/insid3_task_structure60_20261010"
TRACE = ASSETS / "a/reference_mask_task_trace_20261010"
OUT = REPO / "evidence/local/insid3_foris_generalization_20261010"
sys.path.insert(0, str(REPO / "src"))
from ics.official_data import array_hash, decoded_rgb


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unpack(x):
    return np.unpackbits(x, count=1024**2).reshape(1024, 1024).astype(bool)


def dense_mask(path):
    x = np.asarray(Image.open(path).convert("L")) > 0
    return F.interpolate(torch.from_numpy(x.astype(np.float32))[None, None], (1024, 1024), mode="nearest")[0, 0].numpy().astype(bool)


def mass(x):
    return x.reshape(64, 16, 64, 16).sum((1, 3)).ravel().astype(float)


def auc(score, fg, bg):
    if fg.sum() == 0 or bg.sum() == 0:
        return None
    # Expand each cell as two opposite labels with actual pixel-mass weights.
    weights = np.r_[fg, bg]
    keep = weights > 0
    return float(roc_auc_score(np.r_[np.ones(len(fg)), np.zeros(len(bg))][keep],
                               np.tile(score, 2)[keep], sample_weight=weights[keep]))


def components(pairs):
    parents = {}
    def root(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = root(parents[x])
        return parents[x]
    def identity(pair, role):
        row = pair["row"]
        photo = row.get(role + "_photo_id")
        # Legacy SUIM has null photo IDs; null is not a shared photograph.
        if photo is not None:
            return pair["dataset"] + "/photo/" + str(photo)
        return pair["dataset"] + "/rgb/" + row[role + "_rgb_hash"]
    for pair in pairs:
        a, b = [identity(pair, k) for k in ("reference", "query")]
        parents[root(a)] = root(b)
    return {p["id"]: root(identity(p, "query")) for p in pairs}


def paired_summary(rows, fn, groups):
    pairs = {}
    for row in rows:
        pairs.setdefault(row["id"], []).append(fn(row))
    pairs = {key: float(np.mean(values)) for key, values in pairs.items() if len(values) == 2 and all(v is not None for v in values)}
    if not pairs:
        return {"n_pairs": 0, "mean": None, "ci95": None}
    ids = sorted({groups[key] for key in pairs})
    buckets = [np.asarray([v for key, v in pairs.items() if groups[key] == group]) for group in ids]
    if len(ids) < 2:
        return {"n_pairs": len(pairs), "photo_components": len(ids), "mean": float(np.mean(list(pairs.values()))),
                "ci95": None, "limitation": "Only one connected photo group; a bootstrap interval is not estimable."}
    rng = np.random.default_rng(0)
    draws = [float(np.concatenate([buckets[i] for i in rng.integers(0, len(ids), len(ids))]).mean()) for _ in range(2000)]
    return {"n_pairs": len(pairs), "photo_components": len(ids), "mean": float(np.mean(list(pairs.values()))),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}


def show_cases(pairs, truth, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    index = {p["id"]: p for p in pairs}
    metrics = {(r["id"], r["variant"]): r for r in rows}
    figures = []
    # These two existing opposite A/B effects were named before this audit.
    for pid in ("coco_000", "paco_part_001"):
        pair = index[pid]
        rgb = {}
        for role in ("reference", "query"):
            row = pair["row"]
            im = decoded_rgb(ASSETS / row[role + "_path"], row.get(role + "_crop"))
            assert array_hash(np.asarray(im)) == row[role + "_rgb_hash"]
            rgb[role] = np.asarray(im.resize((512, 512), Image.Resampling.BILINEAR))
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.7), facecolor="white", constrained_layout=True)
        with np.load(ROOT / "predictions" / (pid + ".npz")) as pred:
            for i, variant in enumerate(("A", "B")):
                ref = dense_mask(pair["variants"][variant]["path"])
                gt = dense_mask(truth[pid]["masks"][variant]["path"])
                rr = metrics[(pid, variant)]
                axes[i, 0].imshow(rgb["reference"])
                axes[i, 0].contour(ref[::2, ::2], levels=[.5], colors=["#00ff8c"], linewidths=.9)
                axes[i, 0].set_title(f"Reference task {variant} / {pair['classes'][i]}", fontsize=10)
                axes[i, 1].imshow(rgb["query"])
                axes[i, 1].contour(gt[::2, ::2], levels=[.5], colors=["#00ff8c"], linewidths=.9)
                axes[i, 1].set_title("Query and target outline", fontsize=10)
                for col, arm, metric in [(2, "official_bilinear", "old_iou"), (3, "intra_one_bilinear", "new_iou")]:
                    mask = unpack(pred[variant + "/" + arm])
                    overlay = np.zeros((1024, 1024, 4), np.float32)
                    overlay[mask & gt] = [.05, .95, .36, .53]
                    overlay[mask & ~gt] = [1., .18, .12, .64]
                    overlay[~mask & gt] = [.10, .50, 1., .68]
                    axes[i, col].imshow(rgb["query"])
                    axes[i, col].imshow(overlay[::2, ::2])
                    title = "INSID3" if col == 2 else "intra = 1 control"
                    axes[i, col].set_title(f"{title} | IoU {rr[metric]:.2f}", fontsize=10)
        for ax in axes.flat:
            ax.set_axis_off()
        fig.suptitle(f"{pid}: same query, two lawful reference tasks\nGreen = true positive; red = false positive; blue = missed target", fontsize=12)
        path = OUT / (pid + "_task_switch.png")
        fig.savefig(path, dpi=155)
        plt.close(fig)
        figures.append({"path": str(path), "sha256": sha(path), "selection": "Previously named opposite A/B effect; explanatory case, not prevalence estimate"})
    return figures


def main():
    torch.set_num_threads(2)
    seal, report = read(ROOT / "sealed.json"), read(ROOT / "report.json")
    assert report["seal_sha256"] == sha(ROOT / "sealed.json")
    assert seal["config_sha256"] == sha(ROOT / "config.json")
    pairs, truth = read(TRACE / "pairs.json"), read(TRACE / "query_truth.json")
    assert sha(TRACE / "pairs.json") == seal["pairs_sha256"]
    assert sha(TRACE / "query_truth.json") == read(TRACE / "prepared.json")["query_truth_sha256"]
    natives = {r["id"]: r for r in map(json.loads, (TRACE / "inference.jsonl").read_text().splitlines())}
    assert sha(TRACE / "inference.jsonl") == read(TRACE / "sealed.json")["inference_sha256"]
    groups = components(pairs)
    rows = []
    for pair, rec in zip(pairs, seal["records"]):
        pid = pair["id"]
        assert pid == rec["id"]
        state_path = ROOT / "states" / (pid + ".npz")
        pred_path = ROOT / "predictions" / (pid + ".npz")
        native = natives[pid]
        assert sha(state_path) == rec["states_sha256"] and sha(pred_path) == rec["predictions_sha256"]
        assert sha(native["file"]) == native["sha256"]
        with np.load(state_path) as state, np.load(pred_path) as pred, np.load(native["file"]) as foris:
            for variant in ("A", "B"):
                binding = truth[pid]["masks"][variant]
                assert sha(binding["path"]) == binding["sha256"]
                gt = dense_mask(binding["path"])
                old = unpack(pred[variant + "/official_bilinear"])
                new = unpack(pred[variant + "/intra_one_bilinear"])
                added, deleted = new & ~old, old & ~new
                I, U = int((old & gt).sum()), int((old | gt).sum())
                tp, fp = int((added & gt).sum()), int((added & ~gt).sum())
                old_iou = 100 * I / max(U, 1)
                new_iou = 100 * (new & gt).sum() / max(int((new | gt).sum()), 1)
                # Exact explanation for add-only edits, not an inference rule.
                assert not deleted.any(), (pid, variant, "Expected add-only control on these actual fields")
                assert abs(new_iou - 100 * (I + tp) / max(U + fp, 1)) < 1e-12
                row = {"id": pid, "dataset": pair["dataset"], "variant": variant,
                       "old_iou": old_iou, "new_iou": float(new_iou), "delta_iou": float(new_iou - old_iou),
                       "added_TP": tp, "added_FP": fp, "deleted_pixels": int(deleted.sum()),
                       "added_precision": tp / (tp + fp) if tp + fp else None,
                       "improvement_precision_boundary": I / max(I + U, 1), "ranking": {}}
                k = int(state[variant + "/K"])
                if k:
                    labels = state[variant + "/labels"].ravel()
                    cross, intra, area = [state[variant + "/" + key] for key in ("cross", "intra", "area")]
                    a, b = (cross * area)[labels], (cross * area * intra)[labels]
                    eligible = (area > 0)[labels]
                    native_mask = unpack(foris[variant + "/crf"])
                    for roi in ("candidate_touched_clusters", "inside_foris"):
                        if roi == "candidate_touched_clusters":
                            fg, bg = mass(gt) * eligible, mass(~gt) * eligible
                        else:
                            fg, bg = mass(gt & native_mask), mass(~gt & native_mask)
                        x, y = auc(a, fg, bg), auc(b, fg, bg)
                        row["ranking"][roi] = {"without_intra": x, "official_with_intra": y,
                                               "without_minus_with": None if x is None or y is None else x - y}
                    row["intra_range"] = [float(intra.min()), float(intra.max())]
                rows.append(row)
    summary = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        selected = [r for r in rows if r["dataset"] == dataset]
        summary[dataset] = {"n_tasks": len(selected), "changed_tasks": sum(r["added_TP"] + r["added_FP"] > 0 for r in selected),
                            "added_TP": sum(r["added_TP"] for r in selected), "added_FP": sum(r["added_FP"] for r in selected),
                            "deleted_pixels": sum(r["deleted_pixels"] for r in selected),
                            "iou_difference": paired_summary(selected, lambda r: r["delta_iou"], groups), "ranking_difference": {}}
        for roi in ("candidate_touched_clusters", "inside_foris"):
            summary[dataset]["ranking_difference"][roi] = paired_summary(selected, lambda r: r["ranking"].get(roi, {}).get("without_minus_with"), groups)
    OUT.mkdir(parents=True, exist_ok=True)
    figures = show_cases(pairs, truth, rows)
    result = {"state": "POST_SEAL_DIAGNOSIS_COMPLETE", "source_sha256": sha(__file__),
              "parent_seal_sha256": sha(ROOT / "sealed.json"), "parent_report_sha256": sha(ROOT / "report.json"),
              "trace_seal_sha256": sha(TRACE / "sealed.json"), "datasets": summary, "rows": rows, "figures": figures,
              "photo_identity": "Use physical photo ID when present, otherwise verified decoded-RGB SHA256; missing IDs never collapse to one photograph.",
              "new_encoder_calls": 0, "new_predictions": 0, "mps_calls": 0,
              "interpretation": "Existing intra=1 control was add-only on all120 tasks. Deleting an attenuation factor is not evidence that target identity improved. AUC compares same candidate-touched cluster ROI or same external FoRIS mask.",
              "limits": "All labels used only after parent prediction seal. Two task-switch montages were chosen because prior report named their opposite responses. Neither oracle precision boundary nor GT-conditioned ROI is an inference rule."}
    (OUT / "factor_evidence.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
