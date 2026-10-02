"""NumPy-only quality aggregation shared by online and offline diagnostics."""
import numpy as np

METHODS = ("official", "cached", "dense_assoc", "factor_projected")


def summarize_quality(records):
    summary = {}
    for regime in ("central", "near_boundary", "box"):
        for method in METHODS:
            group = [item for item in records if item["regime"] == regime and item["method"] == method]
            rows = [row for item in group for row in item.get("quality", {}).get("rows", [])]
            if not rows:
                continue
            selected_iou, selected_boundary, oracle3, oracle4, gaps = [], [], [], [], []
            for row in rows:
                selected = row["selection_per_policy"]["multimask_iou_head_1to3"]
                selected_iou.append(row["iou_per_mask"][selected])
                selected_boundary.append(row["boundary_iou_per_mask"][selected])
                oracle3.append(max(row["iou_per_mask"][1:]))
                oracle4.append(max(row["iou_per_mask"]))
                gaps.append(row["multimask_oracle_gap"])
            policy_means = {}
            for policy in rows[0]["selection_per_policy"]:
                policy_means[policy] = {"mean_iou": float(np.mean([row["iou_per_mask"][row["selection_per_policy"][policy]] for row in rows])),
                                        "mean_boundary_iou": float(np.mean([row["boundary_iou_per_mask"][row["selection_per_policy"][policy]] for row in rows]))}
            summary[f"{regime}/{method}"] = {"prompt_rows": len(rows), "policies": policy_means,
                                              "mean_multimask_selected_iou": float(np.mean(selected_iou)),
                                              "mean_multimask_selected_boundary_iou": float(np.mean(selected_boundary)),
                                              "mean_oracle_iou_1to3": float(np.mean(oracle3)), "mean_oracle_iou_all4": float(np.mean(oracle4)),
                                              "mean_oracle_gap_1to3": float(np.mean(gaps)), "fraction_oracle_gap_gt_0.02": float(np.mean(np.array(gaps) > .02))}
    return summary


def attach_prompt_metadata(quality, record, annotation_ids):
    """Carry object identity/prompt coordinates without requiring original assets."""
    rows = quality.get("rows", [])
    if rows and len(rows) != len(annotation_ids):
        raise ValueError("quality and annotation row counts differ")
    metadata = record.get("prompt_metadata")
    if metadata is not None and len(metadata) != len(annotation_ids):
        raise ValueError("prompt metadata and annotation row counts differ")
    old_rows = record.get("quality", {}).get("rows", [])
    for index, (row, annotation_id) in enumerate(zip(rows, annotation_ids)):
        row.update(image_id=int(record["image_id"]), annotation_id=int(annotation_id), regime=record["regime"])
        if metadata is not None:
            item = metadata[index]
            if int(item["annotation_id"]) != int(annotation_id):
                raise ValueError("prompt metadata and NPZ annotation order differ")
            regime = record["regime"]
            row["prompt_xy"] = item["central_xy" if regime == "central" else "near_boundary_xy"] if regime != "box" else None
            row["prompt_box_xyxy"] = item["tight_box_xyxy"] if regime == "box" else None
        elif index < len(old_rows):
            old = old_rows[index]
            if "annotation_id" in old and int(old["annotation_id"]) != int(annotation_id):
                raise ValueError("legacy quality and NPZ annotation order differ")
            for key in ("prompt_xy", "prompt_box_xyxy"):
                if key in old:
                    row[key] = old[key]
