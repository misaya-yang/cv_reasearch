"""Equal-logit fusion of already exported real RGB flip views; CPU only.

Candidates match using masks without GT. Original-IoU-head and consistency indices
remain fixed; fusion is a boundary intervention, separate from mask selection.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from takeover_virtual_flip import Sam
from takeover_flip_score import bootstrap_image_groups, inner_boundary, iou


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    source = json.loads((args.original_run/"report.json").read_text())
    flip = json.loads((args.flipped_run/"report.json").read_text())
    rank = json.loads(args.rank_score.read_text())
    report = {"status": "RUNNING", "cohort": "follow-up on previously inspected RGB-flip cohort; not new heldout evidence",
              "protocol": "equal average of original and inverse-horizontal-flipped full-resolution logits after independent official resize/crop/resize; match and selection from existing GT-free consistency rule; no trained weights or tuned fusion coefficient",
              "new_encoder_executions": 0, "new_decoder_executions": 0,
              "deployment_cost": "requires original and real RGB flip encoder/decoder views; saved-output reuse does not remove deployment cost",
              "policies": ["original_iou_head", "flip_consistency_original", "original_iou_head_fused", "flip_consistency_fused"],
              "rows": [], "summary": {}}
    proxy = SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024))
    with torch.inference_mode():
        for info in source["images"]:
            other_info = next(x for x in flip["images"] if x["image_id"] == info["image_id"])
            for regime in ("central", "near_boundary", "box"):
                a = next(r for r in source["records"] if r["image_id"]==info["image_id"] and r["regime"]==regime and r["method"]=="official")
                b = next(r for r in flip["records"] if r["image_id"]==info["image_id"] and r["regime"]==regime and r["method"]=="official")
                with np.load(args.original_run/a["npz"], allow_pickle=False) as original, np.load(args.flipped_run/b["npz"], allow_pickle=False) as flipped:
                    assert np.array_equal(original["annotation_ids"], flipped["annotation_ids"])
                    ol = Sam.postprocess_masks(proxy, torch.from_numpy(original["low_resolution_logits"]), tuple(info["input_size"]), tuple(info["original_size"]))
                    fl = Sam.postprocess_masks(proxy, torch.from_numpy(flipped["low_resolution_logits"]), tuple(other_info["input_size"]), tuple(other_info["original_size"])).flip(-1)
                    assert np.array_equal((ol>0).numpy(), original["masks"])
                    assert np.array_equal((fl>0).numpy(), flipped["masks"][...,::-1])
                    radius = max(1,int(round(.02*np.hypot(*info["original_size"]))))
                    for j, annotation in enumerate(original["annotation_ids"]):
                        saved = next(r for r in rank["rows"] if r["image_id"]==info["image_id"] and r["regime"]==regime and r["annotation_id"]==int(annotation))
                        baseline = saved["selections"]["original_iou_head"]
                        selected = saved["selections"]["matched_consistency_only"]
                        matched = saved["matching"]
                        fused = ((ol[j,1:] + fl[j,[1+x for x in matched]]) / 2 > 0).numpy()
                        candidates = {"original_iou_head": original["masks"][j,baseline],
                                      "flip_consistency_original": original["masks"][j,selected],
                                      "original_iou_head_fused": fused[baseline-1],
                                      "flip_consistency_fused": fused[selected-1]}
                        gt = original["gt"][j]
                        gb = inner_boundary(gt,radius)
                        values = {name:iou(mask,gt) for name,mask in candidates.items()}
                        boundaries = {name:iou(inner_boundary(mask,radius),gb) for name,mask in candidates.items()}
                        row = {"image_id":info["image_id"],"annotation_id":int(annotation),"regime":regime}
                        for name in report["policies"]:
                            row[name+"_iou"] = values[name]
                            row[name+"_boundary_iou"] = boundaries[name]
                            row[name+"_delta_iou"] = values[name]-values["original_iou_head"]
                            row[name+"_delta_boundary_iou"] = boundaries[name]-boundaries["original_iou_head"]
                            row[name+"_delta_vs_consistency"] = values[name]-values["flip_consistency_original"]
                        report["rows"].append(row)
            print(json.dumps({"completed_images":len({r['image_id'] for r in report['rows']}),"image_id":info["image_id"]}),flush=True)
    for regime in ("central", "near_boundary", "box"):
        rows = [r for r in report["rows"] if r["regime"]==regime]
        report["summary"][regime] = {"prompts":len(rows),"policies":{}}
        for name in report["policies"]:
            report["summary"][regime]["policies"][name] = {
                "mean_iou":float(np.mean([r[name+"_iou"] for r in rows])),
                "mean_boundary_iou":float(np.mean([r[name+"_boundary_iou"] for r in rows])),
                "mean_delta_iou":float(np.mean([r[name+"_delta_iou"] for r in rows])),
                "image_cluster_delta_iou_95":bootstrap_image_groups(rows,name+"_delta_iou"),
                "mean_delta_vs_consistency":float(np.mean([r[name+"_delta_vs_consistency"] for r in rows])),
                "image_cluster_delta_vs_consistency_95":bootstrap_image_groups(rows,name+"_delta_vs_consistency"),
                "image_cluster_boundary_delta_95":bootstrap_image_groups(rows,name+"_delta_boundary_iou")}
    report["status"] = "COMPLETED_CPU_SAVED_OUTPUT_FOLLOWUP"
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["summary"],indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-run",type=Path,required=True)
    p.add_argument("--flipped-run",type=Path,required=True)
    p.add_argument("--rank-score",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
