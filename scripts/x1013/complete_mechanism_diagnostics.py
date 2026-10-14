#!/usr/bin/env python3
"""Complete the cached x1013 mechanism audit without model inference.

Adds the planned CLI area-error summaries, R2 reference-holdout/fit strata, and
an S-only label-qualification audit for PASCAL-Part and PACO. No x1013 T masks,
predictions, or metrics are read or scored.

    python -B scripts/x1013/complete_mechanism_diagnostics.py
"""
from __future__ import annotations

import ast
import collections
import gzip
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "cv_data"
OUT = ASSETS / "a/x1013_mechanism_diagnostics"
EVIDENCE = ROOT / "evidence/research/x1013_mechanism_completion.json"
PACO_CATEGORIES = OUT / "paco_categories.py"
EXPECTED_PACO_CATEGORIES_SHA256 = "22d1b71ed065050561f0724ec74e57fb67cda7a3bdafc0f2e7bbf63ae7703866"

sys.path.insert(0, str(ROOT / "scripts/x1012"))
import w02_scale_eval as w02  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def photo(binding: dict) -> str:
    row = binding["row"]
    return str(row.get("query_photo_id") or row["query_path"])


def split(binding: dict) -> str:
    value = int(hashlib.sha256(photo(binding).encode()).hexdigest()[:8], 16)
    return "S" if value % 2 == 0 else "T"


def load_jsonl_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def class_area_summary(rows: list[dict], arms: tuple[str, ...]) -> dict:
    result = {}
    for arm in arms:
        class_pixels: dict[tuple[str, str], list[int]] = {}
        total_gt = total_fn = total_fp = 0
        for row in rows:
            metric = row["arms"][arm]
            key = (str(row["fold"]), str(row["cls"]))
            slot = class_pixels.setdefault(key, [0, 0])
            slot[0] += int(metric["I"])
            slot[1] += int(metric["U"])
            gt = int(row["gt_area"])
            inter = int(metric["I"])
            union = int(metric["U"])
            pred = union - gt + inter
            total_gt += gt
            total_fn += gt - inter
            total_fp += pred - inter
        class_ious = [100.0 * i / max(u, 1) for i, u in class_pixels.values()]
        result[arm] = {
            "n_classes": len(class_ious),
            "class_mIoU_percent": float(np.mean(class_ious)) if class_ious else 0.0,
            "FN_over_GT_percent": 100.0 * total_fn / max(total_gt, 1),
            "FP_over_GT_percent": 100.0 * total_fp / max(total_gt, 1),
        }
    return result


def delta_summary(rows: list[dict], a: str, b: str) -> dict:
    fields = ("iou", "cldice_style", "skeleton_precision", "skeleton_recall",
              "missed_gt_components", "spurious_pred_components",
              "split_gt_components", "merged_pred_components")
    return {
        "n_episodes": len(rows),
        "n_query_photos": len({str(row["photo"]) for row in rows}),
        "deltas": {
            field: float(np.mean([row["arms"][a][field] - row["arms"][b][field]
                                  for row in rows])) if rows else 0.0
            for field in fields
        },
    }


def quartile_groups(rows: list[dict], field: str, value_getter) -> dict:
    ordered = sorted(rows, key=lambda row: (float(value_getter(row)), row["index"]))
    groups = {}
    for q in range(4):
        lo = q * len(ordered) // 4
        hi = (q + 1) * len(ordered) // 4
        sample = ordered[lo:hi]
        groups[f"Q{q + 1}"] = {
            "metadata_field": field,
            "range": ([float(value_getter(sample[0])), float(value_getter(sample[-1]))]
                      if sample else None),
            "pixel_vs_VP_identity": delta_summary(sample, "pixel", "VP_identity"),
            "pixel_vs_pixel_permuted": delta_summary(sample, "pixel", "pixel_permuted"),
        }
    return groups


def fit_distribution(records: list[dict], field: str) -> dict:
    values = [float(record[field]) for record in records]
    return {"median": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90))}


def r2_summary(diagnostic_rows: list[dict], bindings: list[dict], sealed: dict) -> dict:
    by_index = {(binding["dataset"], str(binding["index"])): binding for binding in bindings}
    sealed_by_episode = {record["episode_id"]: record for record in sealed["records"]}
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in diagnostic_rows:
        if row["cohort"] != "R2_1360_exposed":
            continue
        binding = by_index.get((row["dataset"], str(row["index"])))
        if binding is None:
            raise RuntimeError(f"No binding for R2 diagnostic row {row['dataset']}/{row['index']}")
        record = sealed_by_episode.get(binding["row"]["episode_id"])
        if record is None or record["dataset"] != row["dataset"]:
            raise RuntimeError(f"R2 sealed identity mismatch at {row['dataset']}/{row['index']}")
        alpha_idx = record["selection"]["alphas"].index(record["selection"]["selected"])
        row = dict(row)
        row["r2_selection"] = record["selection"]
        row["r2_fit"] = record["fit"]
        row["r2_fit_permuted"] = record["fit_permuted"]
        row["_heldout_mixed_blocks"] = int(record["selection"]["heldout_mixed_blocks"])
        row["r2_mixed_query_blocks"] = int(record["mixed_query_blocks"])
        row["r2_selected_holdout_TP"] = int(
            record["selection"]["reference_heldout_TP_at_true_block_area"][alpha_idx])
        grouped[row["dataset"]].append(row)

    result = {}
    for domain, rows in grouped.items():
        alpha_counts = collections.Counter(float(row["r2_selection"]["selected"]) for row in rows)
        result[domain] = {
            "n_episodes": len(rows),
            "n_query_photos": len({str(row["photo"]) for row in rows}),
            "selected_alpha_counts": {str(alpha): alpha_counts[alpha]
                                      for alpha in (0.0, 0.25, 1.0)},
            "frozen_reference_holdout_and_fit": {
                "selected_holdout_TP_at_true_block_area": fit_distribution(rows, "r2_selected_holdout_TP"),
                "heldout_mixed_blocks": fit_distribution(
                    rows, "_heldout_mixed_blocks"),
                "mixed_query_blocks": fit_distribution(rows, "r2_mixed_query_blocks"),
                "fit": {field: fit_distribution([row["r2_fit"] for row in rows], field)
                        for field in ("pairs", "steps", "gradient_max_abs", "objective")},
                "fit_permuted": {field: fit_distribution(
                    [row["r2_fit_permuted"] for row in rows], field)
                    for field in ("pairs", "steps", "gradient_max_abs", "objective")},
            },
            "strata": {
                "selected_alpha": {
                    str(alpha): {
                        "n_episodes": sum(float(row["r2_selection"]["selected"]) == alpha
                                           for row in rows),
                        "pixel_vs_VP_identity": delta_summary(
                            [row for row in rows if float(row["r2_selection"]["selected"]) == alpha],
                            "pixel", "VP_identity"),
                        "pixel_vs_pixel_permuted": delta_summary(
                            [row for row in rows if float(row["r2_selection"]["selected"]) == alpha],
                            "pixel", "pixel_permuted"),
                    }
                    for alpha in (0.0, 0.25, 1.0)
                },
                "selected_reference_holdout_TP_quartiles": quartile_groups(
                    rows, "selected_reference_holdout_TP", lambda row: row["r2_selected_holdout_TP"]),
                "mixed_query_blocks_quartiles": quartile_groups(
                    rows, "mixed_query_blocks", lambda row: row["r2_mixed_query_blocks"]),
                "heldout_mixed_blocks_quartiles": quartile_groups(
                    rows, "heldout_mixed_blocks", lambda row: row["_heldout_mixed_blocks"]),
            },
            "_rows": rows,
        }
    return result


def paco_category_list() -> list[dict]:
    observed = sha256(PACO_CATEGORIES)
    if observed != EXPECTED_PACO_CATEGORIES_SHA256:
        raise RuntimeError(f"Unexpected PACO category source hash: {observed}")
    tree = ast.parse(PACO_CATEGORIES.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PACO_CATEGORIES"
                for target in node.targets):
            categories = ast.literal_eval(node.value)
            if len(categories) != 531:
                raise RuntimeError(f"Expected 531 PACO categories, found {len(categories)}")
            return categories
    raise RuntimeError("PACO_CATEGORIES assignment not found in official source")


def decode_coco_rle(segmentation: dict, expected_area: int | None = None) -> np.ndarray:
    """Decode a compressed COCO RLE without importing the model runtime."""
    height, width = [int(value) for value in segmentation["size"]]
    encoded = segmentation["counts"]
    if not isinstance(encoded, str):
        raise TypeError("PACO audit expected compressed string RLE")
    counts = []
    index = 0
    while index < len(encoded):
        value = 0
        shift = 0
        more = True
        while more:
            char = ord(encoded[index]) - 48
            value |= (char & 0x1F) << (5 * shift)
            more = bool(char & 0x20)
            if not more and (char & 0x10):
                value |= (-1) << (5 * (shift + 1))
            index += 1
            shift += 1
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    foreground = False
    for count in counts:
        end = position + count
        if foreground:
            flat[position:end] = True
        position = end
        foreground = not foreground
    if position != height * width:
        raise RuntimeError(f"Invalid COCO RLE length {position} != {height * width}")
    mask = flat.reshape((height, width), order="F")
    if expected_area is not None and int(mask.sum()) != int(expected_area):
        raise RuntimeError("Decoded PACO RLE area does not match annotation area")
    return mask


def adjacent8(a: np.ndarray, b: np.ndarray) -> tuple[bool, int]:
    overlap = int(np.count_nonzero(a & b))
    dilated = np.zeros_like(a)
    height, width = a.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y0, y1 = max(0, dy), min(height, height + dy)
            x0, x1 = max(0, dx), min(width, width + dx)
            sy0, sy1 = max(0, -dy), min(height, height - dy)
            sx0, sx1 = max(0, -dx), min(width, width - dx)
            dilated[y0:y1, x0:x1] |= a[sy0:sy1, sx0:sx1]
    return bool(np.any(dilated & b)), overlap


def box_from_paco(annotation: dict) -> tuple[int, int, int, int]:
    x, y, width, height = annotation["obj_bbox"]
    return (int(x), int(y), int(x + width), int(y + height))


def paco_audit(bindings: list[dict], categories: list[dict]) -> dict:
    id_to_name = {int(category["id"]): category["name"] for category in categories}
    side_labels = sorted(name for name in id_to_name.values()
                         if re.search(r"(^|:)(left|right)(_|:|$)", name))
    left_id, right_id = 2252, 2313
    if (id_to_name.get(left_id), id_to_name.get(right_id)) != (
            "mouse_(computer_equipment):left_button",
            "mouse_(computer_equipment):right_button"):
        raise RuntimeError("PACO side-button category mapping changed")

    rows = [binding for binding in bindings
            if binding["dataset"] == "paco_part" and split(binding) == "S"]
    by_category = collections.Counter(int(binding["row"]["global_class_id"]) for binding in rows)
    pickle_path = ASSETS / "datasets/ics/PACO-Part/paco/paco_part_val.pkl"
    with pickle_path.open("rb") as stream:
        paco = pickle.load(stream)

    right_queries: dict[tuple[int, tuple[int, ...]], list[dict]] = collections.defaultdict(list)
    for binding in rows:
        if int(binding["row"]["global_class_id"]) != right_id:
            continue
        query_path = Path(binding["row"]["query_path"])
        crop = tuple(int(value) for value in binding["row"]["query_crop"])
        right_queries[(int(query_path.stem), crop)].append(binding)

    cases = []
    for (image_id, crop), repeated_rows in sorted(right_queries.items()):
        annotations = paco["img2anno"].get(image_id, [])
        target = [annotation for annotation in annotations
                  if int(annotation["category_id"]) == right_id
                  and box_from_paco(annotation) == crop]
        if not target:
            raise RuntimeError(f"No right_button target annotation for S query {image_id}/{crop}")
        parent_ids = sorted({int(annotation["obj_ann_id"]) for annotation in target})
        for parent_id in parent_ids:
            right_masks = [decode_coco_rle(annotation["segmentation"], annotation.get("area"))
                           for annotation in target if int(annotation["obj_ann_id"]) == parent_id]
            left_annotations = [annotation for annotation in annotations
                                if int(annotation["category_id"]) == left_id
                                and int(annotation["obj_ann_id"]) == parent_id]
            case = {
                "image_id": image_id,
                "crop_xyxy": list(crop),
                "parent_annotation_id": parent_id,
                "right_task_rows": len(repeated_rows),
                "left_button_same_parent": bool(left_annotations),
            }
            if left_annotations:
                left_masks = [decode_coco_rle(annotation["segmentation"], annotation.get("area"))
                              for annotation in left_annotations]
                right = np.logical_or.reduce(right_masks)
                left = np.logical_or.reduce(left_masks)
                touches, overlap = adjacent8(left, right)
                case.update({"left_right_touching8": touches,
                             "overlap_pixels": overlap})
            cases.append(case)

    return {
        "annotation_split": "S query photo IDs only",
        "S_task_rows": len(rows),
        "S_unique_query_photos": len({photo(binding) for binding in rows}),
        "side_category_names_in_official_taxonomy": side_labels,
        "left_button_id": left_id,
        "right_button_id": right_id,
        "S_target_rows": {id_to_name[left_id]: by_category[left_id],
                          id_to_name[right_id]: by_category[right_id]},
        "S_right_button_unique_query_parents": len(cases),
        "S_right_button_same_parent_left_button_count": sum(
            case["left_button_same_parent"] for case in cases),
        "S_right_button_same_parent_cases": cases,
        "balanced_side_identity_eligible": False,
        "stop_reason": "No S-half left_button target and only two S query photos with a same-parent left_button contrast.",
        "all_S_category_row_counts_for_side_ids": {
            str(category_id): by_category[category_id]
            for category_id in (left_id, right_id)},
    }


def pascal_audit(bindings: list[dict]) -> dict:
    rows = [binding for binding in bindings
            if binding["dataset"] == "pascal_part" and split(binding) == "S"]
    root = ASSETS / "datasets/ics/Pascal-Part/VOCdevkit/VOC2010"
    annotation_dir = root / "Annotations_Part_json_merged_part_classes"
    target_side_rows = [binding for binding in rows
                        if re.search(r"(^|[+_ ])(left|right)([+_ ])",
                                     str(binding["row"]["global_class_id"]), re.I)]
    part_counts = collections.Counter()
    matched_parents = 0
    no_match = 0
    annotation_hashes = {}
    for binding in rows:
        path = annotation_dir / (Path(binding["row"]["query_path"]).stem + ".json")
        if not path.exists():
            raise FileNotFoundError(path)
        annotation_hashes[str(path.relative_to(ASSETS))] = sha256(path)
        doc = read_json(path)
        crop = tuple(int(value) for value in binding["row"]["query_crop"])
        found = False
        for obj in doc.get("object", []):
            box = tuple(int(obj["bndbox"][key])
                        for key in ("xmin", "ymin", "xmax", "ymax"))
            if box != crop:
                continue
            found = True
            matched_parents += 1
            for part in obj.get("parts", []):
                part_counts[str(part["name"])] += 1
        if not found:
            no_match += 1
    map_path = root / "all_obj_part_to_image.json"
    categories = read_json(map_path)
    return {
        "annotation_split": "S query photo IDs only",
        "S_task_rows": len(rows),
        "S_unique_query_photos": len({photo(binding) for binding in rows}),
        "S_unique_target_classes": len({str(binding["row"]["global_class_id"])
                                         for binding in rows}),
        "S_target_classes_with_left_or_right_name": len(target_side_rows),
        "S_target_parent_annotation_matches": matched_parents,
        "S_target_parent_annotation_unmatched": no_match,
        "S_merged_part_annotation_names": dict(sorted(part_counts.items())),
        "S_merged_part_annotation_names_with_left_or_right": {
            name: count for name, count in part_counts.items()
            if re.search(r"(^|[ _-])(left|right)([ _-]|$)", name, re.I)},
        "merged_class_map_sha256": sha256(map_path),
        "opened_S_annotation_files": len(annotation_hashes),
        "opened_S_annotation_sha256": hashlib.sha256(
            json.dumps(sorted(annotation_hashes.items()), separators=(",", ":")).encode()
        ).hexdigest(),
        "source_part_families": sorted(categories),
        "balanced_side_identity_eligible": False,
        "stop_reason": "The current merged task labels and S-half parent annotations use side-agnostic part names.",
    }


def main() -> None:
    summary_path = OUT / "summary.json"
    episodes_path = OUT / "episodes.jsonl.gz"
    sealed_path = ASSETS / "a/x1012_R2/sealed.json"
    bindings = w02.rows_all(0)
    diagnostic_rows = load_jsonl_gzip(episodes_path)
    base_summary = read_json(summary_path)
    sealed = read_json(sealed_path)
    if base_summary.get("state") != "OFFLINE_STRUCTURE_PIXEL_DIAGNOSTIC_COMPLETE":
        raise RuntimeError("Base A/B diagnostic is not marked complete")
    if base_summary.get("x1013_T_scored") is not False:
        raise RuntimeError("Refusing to continue: x1013 T was scored by the base diagnostic")
    if sealed.get("state") != "ALL_NATURAL_OUTPUTS_SEALED" or sealed.get("n") != 1360:
        raise RuntimeError("R2 outputs are not in the expected sealed state")

    x1013_s = [row for row in diagnostic_rows if row["cohort"] == "x1013_S"]
    x1013_metrics = {}
    for domain in ("fundus", "deepglobe_road"):
        selected = [row for row in x1013_s if row["dataset"] == domain]
        x1013_metrics[domain] = {
            "n_episodes": len(selected),
            "n_query_photos": len({str(row["photo"]) for row in selected}),
            "class_and_area_metrics": class_area_summary(selected, ("FoRIS", "A4", "C1")),
        }

    r2 = r2_summary(diagnostic_rows, bindings, sealed)
    r2_class_metrics = {}
    for domain, payload in r2.items():
        rows = payload["_rows"]
        r2_class_metrics[domain] = {
            "n_episodes": len(rows),
            "n_query_photos": len({str(row["photo"]) for row in rows}),
            "class_and_area_metrics": class_area_summary(
                rows, ("foris", "VP_identity", "pixel", "pixel_permuted", "patch_constant")),
        }
    category_list = paco_category_list()
    audit = {
        "state": "MECHANISM_DIAGNOSTIC_A_B_COMPLETE_C_LABEL_QUALIFICATION_COMPLETE",
        "new_dinov3_inference_calls": 0,
        "new_model_construction": False,
        "x1013_T_scored": False,
        "x1013_S_class_area_metrics": x1013_metrics,
        "R2_class_area_metrics": r2_class_metrics,
        "R2_reference_holdout_fit_and_strata": {
            domain: {key: value for key, value in payload.items() if key != "_rows"}
            for domain, payload in r2.items()},
        "C_label_qualification": {
            "PACO": paco_audit(bindings, category_list),
            "PASCAL-Part": pascal_audit(bindings),
        },
        "inputs_sha256": {
            "base_summary": sha256(summary_path),
            "base_episodes": sha256(episodes_path),
            "R2_sealed": sha256(sealed_path),
            "PACO_category_source": sha256(PACO_CATEGORIES),
            "PACO_val_pickle": sha256(ASSETS / "datasets/ics/PACO-Part/paco/paco_part_val.pkl"),
            "PASCAL_merged_class_map": sha256(
                ASSETS / "datasets/ics/Pascal-Part/VOCdevkit/VOC2010/all_obj_part_to_image.json"),
        },
        "external_taxonomy_sources": {
            "PACO_categories": "https://github.com/facebookresearch/paco/blob/main/paco/data/datasets/paco_categories.py",
            "PASCAL-Part_original_taxonomy": "https://cs.stanford.edu/~roozbeh/pascal-parts/pascal-parts.html",
        },
    }
    write_json(OUT / "completion_summary.json", audit)
    write_json(EVIDENCE, audit)
    print(f"Wrote {OUT / 'completion_summary.json'}")
    print(f"Wrote {EVIDENCE}")
    for domain, result in x1013_metrics.items():
        print(domain, json.dumps(result["class_and_area_metrics"], sort_keys=True))
    for domain, payload in audit["R2_reference_holdout_fit_and_strata"].items():
        print("R2", domain, "alpha", payload["selected_alpha_counts"])
        print("R2 class/area", domain,
              json.dumps(r2_class_metrics[domain]["class_and_area_metrics"], sort_keys=True))
    print("PACO", json.dumps(audit["C_label_qualification"]["PACO"], sort_keys=True))
    print("PASCAL-Part", json.dumps({key: value for key, value in
          audit["C_label_qualification"]["PASCAL-Part"].items()
          if key != "S_merged_part_annotation_names"}, sort_keys=True))


if __name__ == "__main__":
    main()
