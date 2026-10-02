"""Offline paired SAM quality diagnosis from actual postprocessed cached masks.

Input NPZ (allow_pickle=False): logits[P,K,H,W] or masks[P,K,H,W],
iou_prediction[P,K], gt[P,H,W]. Boolean masks save full-resolution float storage.
Optional low_resolution_logits[P,K,h,w] enables the exact source
SAM2 dynamic selection *rule*, not evaluation of a SAM2 pretrained model.
All rows refer to independent prompts; no cross-prompt aggregation or fusion.
Use four raw output masks for diagnostics; the multimask deployment policy
considers only indices 1,2,3, matching the local official SAM/SAM2 snapshots.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def inner_boundary(mask, radius):
    # The zero pad correctly regards image-edge foreground as boundary.
    eroded = mask.copy()
    h, w = mask.shape
    for _ in range(radius):
        pad = np.pad(eroded, 1, constant_values=False)
        eroded = np.logical_and.reduce([pad[y:y+h, x:x+w] for y in range(3) for x in range(3)])
    return np.logical_and(mask, ~eroded)


def paired_bootstrap(values, seed=2027, repetitions=1000):
    rng = np.random.default_rng(seed)
    n = len(values)
    # Prompt-row uncertainty only; image-cluster bootstrap is preferred for a paper.
    means = [np.mean(values[rng.integers(n, size=n)]) for _ in range(repetitions)]
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def diagnose(arrays, boundary_ratio=0.02):
    prediction, gt = arrays["iou_prediction"], arrays["gt"]
    if "masks" in arrays:
        masks = arrays["masks"]
        if not np.all((masks == 0) | (masks == 1)):
            raise ValueError("masks must be binary")
        masks = masks.astype(bool)
    else:
        logits = arrays["logits"]
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite logits")
        masks = logits > 0
    if masks.ndim != 4 or gt.shape != (masks.shape[0], *masks.shape[-2:]):
        raise ValueError("need full-resolution logits[P,K,H,W] and gt[P,H,W]")
    p, k, h, w = masks.shape
    if p == 0 or k != 4 or prediction.shape != (p, k):
        raise ValueError("need nonempty rows, all four masks and four IoU predictions")
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite outputs")
    if not np.all((gt == 0) | (gt == 1)) or not 0 < boundary_ratio < 1:
        raise ValueError("need binary ground truth and valid boundary ratio")
    gt = gt.astype(bool)
    radius = max(1, int(round(boundary_ratio * np.sqrt(h*h + w*w))))
    scores, boundary_scores = np.empty((p, k)), np.empty((p, k))
    for row in range(p):
        gt_boundary = inner_boundary(gt[row], radius)
        for candidate in range(k):
            scores[row, candidate] = iou(masks[row, candidate], gt[row])
            boundary_scores[row, candidate] = iou(inner_boundary(masks[row, candidate], radius), gt_boundary)
    row_ids = np.arange(p)
    multimask = prediction[:, 1:].argmax(-1) + 1
    policies = {"single_token0": np.zeros(p, dtype=int), "multimask_iou_head_1to3": multimask}
    if "low_resolution_logits" in arrays:
        low = arrays["low_resolution_logits"]
        if low.ndim != 4 or low.shape[:2] != (p, k) or not np.isfinite(low).all():
            raise ValueError("invalid low-resolution logits")
        low = low[:, 0].reshape(p, -1)
        inter, union = (low > 0.05).sum(-1), (low > -0.05).sum(-1)
        stable = np.divide(inter, union, out=np.ones(p), where=union > 0) >= 0.98
        policies["sam2_snapshot_dynamic_rule_on_supplied_outputs"] = np.where(stable, 0, multimask)
    selected = policies["multimask_iou_head_1to3"]
    oracle3 = scores[:, 1:].argmax(-1) + 1
    oracle4 = scores.argmax(-1)
    gap3 = scores[row_ids, oracle3] - scores[row_ids, selected]
    rows = []
    for row in range(p):
        rows.append({"row": row, "iou_per_mask": scores[row].tolist(), "boundary_iou_per_mask": boundary_scores[row].tolist(),
                     "selection_per_policy": {key: int(indices[row]) for key, indices in policies.items()},
                     "multimask_iou_oracle_index": int(oracle3[row]), "all4_iou_oracle_index": int(oracle4[row]),
                     "multimask_oracle_gap": float(gap3[row])})
    return {"status": "OFFLINE_REAL_QUALITY_ONLY_IF_INPUT_IS_PRETRAINED_ANNOTATED_REAL_DATA", "prompt_rows": p,
            "resolution": [h, w], "boundary_ratio": boundary_ratio, "boundary_radius": radius,
            "policies": {name: {"mean_iou": float(scores[row_ids, ids].mean()), "mean_boundary_iou": float(boundary_scores[row_ids, ids].mean())}
                         for name, ids in policies.items()},
            "oracle_diagnostic_extra_ground_truth_not_deployable": {
                "multimask_1to3_mean_iou": float(scores[row_ids, oracle3].mean()),
                "all4_mean_iou": float(scores[row_ids, oracle4].mean()),
                "multimask_1to3_mean_boundary_iou_at_iou_oracle": float(boundary_scores[row_ids, oracle3].mean()),
                "all4_boundary_oracle_mean_boundary_iou": float(boundary_scores.max(-1).mean()),
                "mean_multimask_iou_gap": float(gap3.mean()), "prompt_bootstrap_gap95": paired_bootstrap(gap3),
                "fraction_gap_gt_0.02": float((gap3 > 0.02).mean())},
            "rows": rows,
            "limitations": ["Prompt-row bootstrap is not image-cluster uncertainty", "A dynamic selection rule applied to SAM outputs is not a SAM2 model result", "Protocol assumes full-resolution official resize/crop/resize logits", "No image encoder or latency data included", "Do not pool different prompt regimes without separate strata"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-ratio", type=float, default=0.02)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as arrays:
        results = diagnose(arrays, args.boundary_ratio)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: value for key, value in results.items() if key != "rows"}, indent=2))
