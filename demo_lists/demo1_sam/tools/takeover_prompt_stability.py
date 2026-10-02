"""GT-free selection using four counterfactual prompt views of one cached image.

Perturbations are fixed at eight resized-image pixels, without foreground checks.
Only original masks are selectable. This changes the quality inference protocol,
not the exact-decoder contract. CPU matching/export is excluded from GPU times.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from takeover_virtual_flip import load_models, encode_prompt_rows, decode_rows
from takeover_flip_score import bootstrap_image_groups, iou, inner_boundary

OFFSETS = ((-8, 0), (8, 0), (0, -8), (0, 8))
POLICIES = ("original_iou_head", "perturb_consistency", "perturb_mean_iou_head",
            "largest_original_mask", "original_logit_stability",
            "perturb_single_left", "perturb_horizontal_pair")


def perturb_objects(objects, regime, original_size, input_size, offset):
    import copy
    out = copy.deepcopy(objects)
    h, w = original_size
    ih, iw = input_size
    dx, dy = offset[0] * w / iw, offset[1] * h / ih
    for obj in out:
        if regime in ("central", "near_boundary"):
            key = "central_xy" if regime == "central" else "near_boundary_xy"
            x, y = obj[key]
            obj[key] = [float(np.clip(x + dx, 0, w - 1)), float(np.clip(y + dy, 0, h - 1))]
        else:
            x0, y0, x1, y1 = obj["tight_box_xyxy"]
            dx1, dy1 = float(np.clip(dx, -x0, w-x1)), float(np.clip(dy, -y0, h-y1))
            obj["tight_box_xyxy"] = [x0+dx1, y0+dy1, x1+dx1, y1+dy1]
    return out


def select(masks, scores, low, views, view_scores):
    baseline = int(np.argmax(scores[1:])) + 1
    consistencies, heads = [], []
    for vm, vs in zip(views, view_scores):
        matrix = np.array([[iou(a, b) for b in vm[1:]] for a in masks[1:]])
        p = max(itertools.permutations(range(3)), key=lambda q: sum(matrix[j, q[j]] for j in range(3)))
        consistencies.append([matrix[j, p[j]] for j in range(3)])
        heads.append([vs[1+p[j]] for j in range(3)])
    consistency = np.mean(consistencies, axis=0)
    # An uninformative tie preserves the official head, including empty masks.
    winner = baseline if np.ptp(consistency) <= 1e-8 else int(np.argmax(consistency)) + 1
    denominator = (low[1:] > -1).sum(axis=(-2, -1))
    stability = np.divide((low[1:] > 1).sum(axis=(-2, -1)), denominator,
                          out=np.ones(3), where=denominator > 0)
    choices = {"original_iou_head": baseline, "perturb_consistency": winner,
               "perturb_mean_iou_head": int(np.argmax((scores[1:] + np.sum(heads, axis=0))/5)) + 1,
               "largest_original_mask": int(np.argmax(masks[1:].sum(axis=(-2, -1)))) + 1,
               "original_logit_stability": int(np.argmax(stability)) + 1}
    for name, count in (("perturb_single_left", 1), ("perturb_horizontal_pair", 2)):
        c = np.mean(consistencies[:count], axis=0)
        choices[name] = baseline if np.ptp(c) <= 1e-8 else int(np.argmax(c)) + 1
    return choices, consistency.tolist(), stability.tolist()


def atomic(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2)+"\n")
    tmp.replace(path)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads((args.source_run / "report.json").read_text())
    if any("encoded_inputs" not in x for x in source["images"]):
        raise ValueError("saved encoded inputs required")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    report = {"status": "RUNNING", "source_run": str(args.source_run), "cohort": args.cohort,
              "protocol": "four fixed cardinal offsets of 8 resized pixels; no GT checks; independently decoded counterfactual positive prompts; original candidates 1..3 only; six-permutation matching per view; equal mean consistency; uninformative tie preserves official head",
              "image_encoder_executions": 0, "GT_used_for_selection": False,
              "new_human_clicks": 0, "decoder_prompt_multiplier": 5,
              "ablation_deployment_prompt_multipliers": {"perturb_single_left": 2, "perturb_horizontal_pair": 3, "perturb_consistency": 5},
              "rules_fixed_before_results": list(POLICIES), "offsets_resized_pixels": OFFSETS,
              "primary_endpoints": ["central IoU", "near_boundary IoU"],
              "rows": [], "image_diagnostic_times": [], "summary": {}}
    atomic(args.output, report)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    decoder, encoder, _ = load_models(args.source_run/"mask_decoder_state.pt", args.checkpoint, device)
    with torch.inference_mode():
        for info in source["images"]:
            with np.load(args.source_run/info["encoded_inputs"]["npz"], allow_pickle=False) as encoded:
                features, pe, dense = [torch.from_numpy(encoded[k]).to(device) for k in ("image_embeddings", "image_pe", "dense_nomask")]
                size = tuple(int(x) for x in encoded["original_size"])
                input_size = tuple(int(x) for x in encoded["input_size"])
                ids = encoded["annotation_ids"].copy()
            for regime in ("central", "near_boundary", "box"):
                rec = next(r for r in source["records"] if r["image_id"] == info["image_id"] and r["method"] == "official" and r["regime"] == regime)
                objects = rec["prompt_metadata"]
                prompts = [encode_prompt_rows(encoder, perturb_objects(objects, regime, size, input_size, o), regime, size, 1024, device)[0] for o in OFFSETS]
                sparse = torch.cat(prompts)
                if device.type == "cuda": torch.cuda.synchronize()
                start = time.perf_counter()
                arrays = decode_rows(decoder, features, pe, dense, sparse, input_size, size, 0, len(sparse))
                elapsed = (time.perf_counter()-start)*1000
                report["image_diagnostic_times"].append({"image_id": info["image_id"], "regime": regime,
                    "four_views_decode_postprocess_CPU_copy_ms": elapsed, "prompts": len(sparse),
                    "timing_scope": "one pass includes CPU copies; not a steady state GPU benchmark"})
                masks_views = arrays["masks"].reshape(4, len(ids), 4, *size)
                scores_views = arrays["iou_prediction"].reshape(4, len(ids), 4)
                with np.load(args.source_run/rec["npz"], allow_pickle=False) as base:
                    if not np.array_equal(ids, base["annotation_ids"]): raise ValueError("row identity mismatch")
                    for j, annotation in enumerate(ids):
                        choices, consistency, stability = select(base["masks"][j], base["iou_prediction"][j], base["low_resolution_logits"][j], masks_views[:,j], scores_views[:,j])
                        # GT is accessed only after every selection above.
                        gt = base["gt"][j]
                        radius = max(1, int(round(.02*np.hypot(*size))))
                        gb = inner_boundary(gt, radius)
                        values = [iou(m, gt) for m in base["masks"][j]]
                        boundaries = [iou(inner_boundary(m, radius), gb) for m in base["masks"][j]]
                        baseline = choices["original_iou_head"]
                        row = {"image_id": info["image_id"], "annotation_id": int(annotation), "regime": regime,
                               "selections": choices, "iou_per_mask": values, "consistency": consistency,
                               "logit_stability": stability}
                        for name, index in choices.items():
                            row[name+"_iou"] = values[index]
                            row[name+"_boundary_iou"] = boundaries[index]
                            row[name+"_delta_iou"] = values[index]-values[baseline]
                            row[name+"_delta_boundary_iou"] = boundaries[index]-boundaries[baseline]
                            row[name+"_delta_vs_largest"] = values[index]-values[choices["largest_original_mask"]]
                        report["rows"].append(row)
            atomic(args.output, report)
            print(json.dumps({"completed_images": len({r['image_id'] for r in report['rows']}), "image_id": info["image_id"]}), flush=True)
    for regime in ("central", "near_boundary", "box"):
        rows = [r for r in report["rows"] if r["regime"] == regime]
        report["summary"][regime] = {"prompts": len(rows), "policies": {}}
        for name in POLICIES:
            report["summary"][regime]["policies"][name] = {
                "mean_iou": float(np.mean([r[name+"_iou"] for r in rows])),
                "mean_boundary_iou": float(np.mean([r[name+"_boundary_iou"] for r in rows])),
                "mean_delta_iou": float(np.mean([r[name+"_delta_iou"] for r in rows])),
                "image_cluster_delta_iou_95": bootstrap_image_groups(rows, name+"_delta_iou"),
                "image_cluster_boundary_delta_95": bootstrap_image_groups(rows, name+"_delta_boundary_iou"),
                "mean_delta_vs_largest": float(np.mean([r[name+"_delta_vs_largest"] for r in rows])),
                "image_cluster_delta_vs_largest_95": bootstrap_image_groups(rows, name+"_delta_vs_largest"),
                "improved": sum(r[name+"_delta_iou"] > 0 for r in rows),
                "worsened": sum(r[name+"_delta_iou"] < 0 for r in rows)}
    report["status"] = "COMPLETED"
    atomic(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
    p.add_argument("--cohort", required=True)
    run(p.parse_args())
