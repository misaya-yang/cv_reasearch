"""Replay real encoded inputs through all next-round arms, retaining numeric failures."""
import argparse
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import numpy as np
from takeover_common import *
from run_real_sam import differences, save_arrays


def run(args):
    configure()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.run_dir/"report.json").read_text())
    exported = torch.load(args.run_dir/"mask_decoder_state.pt", weights_only=True, map_location="cpu")
    model = make_fixture(grid=4, batch=1)[0]
    model.load_state_dict(exported["state_dict"], strict=True)
    model = model.to(device).eval()
    sys.path.insert(0, str(ROOT/"assets/source/segment-anything"))
    from segment_anything.modeling.sam import Sam
    post_self = SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024))
    report = {"schema_version": 2, "status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_run_status": source["status"], "subset_manifest_sha256": source["subset_manifest_sha256"],
              "numeric_status": "RUNNING", "quality_status": "DEFERRED", "precision": "float32", "tf32": False,
              "tolerance": 5e-5, "device": str(device), "model": source.get("model", "pretrained_sam_vit_b"),
              "scope": "Eager numerical/quality export only, no performance inference; original prompts, all four masks and IoU",
              "images": [], "records": []}
    failed = False
    with torch.inference_mode():
        for image_info in source["images"]:
            image_id = image_info["image_id"]
            folder = args.output_dir/f"image_{image_id:012d}"
            folder.mkdir()
            encoded_path = args.run_dir/image_info["encoded_inputs"]["npz"]
            with np.load(encoded_path, allow_pickle=False) as encoded:
                image, pe, dense = [torch.from_numpy(encoded[k]).to(device) for k in ("image_embeddings", "image_pe", "dense_nomask")]
                input_size, original_size = tuple(encoded["input_size"].tolist()), tuple(encoded["original_size"].tolist())
                threshold = float(encoded["mask_threshold"])
                for regime in ("central", "near_boundary", "box"):
                    sp_all = torch.from_numpy(encoded[f"sparse_{regime}"]).to(device)
                    original = next(r for r in source["records"] if r["image_id"] == image_id and r["regime"] == regime and r["method"] == "official")
                    with np.load(args.run_dir/original["npz"], allow_pickle=False) as saved:
                        gt, annotation_ids = saved["gt"].copy(), saved["annotation_ids"].copy()
                        saved_low, saved_iou = saved["low_resolution_logits"].copy(), saved["iou_prediction"].copy()
                    for arm in ARMS:
                        build, forward, plans = arm_functions(arm, model, image, pe, dense, sp_all.shape[1]+5)
                        cache = build()
                        lows, scores, masks, checks, replay_checks = [], [], [], [], []
                        for lo in range(0, len(sp_all), args.microbatch):
                            sp = sp_all[lo:lo+args.microbatch]
                            reference = official_predict(model, image, pe, dense, sp)
                            reference_full = Sam.postprocess_masks(post_self, reference[0], input_size, original_size)>threshold
                            outputs = forward(sp, cache)
                            assert outputs[0].shape[:2] == (len(sp), 4) and outputs[1].shape == (len(sp), 4)
                            full = Sam.postprocess_masks(post_self, outputs[0], input_size, original_size)>threshold
                            check = differences(reference, outputs, reference_full, full)
                            checks.append({"row_start": lo, "row_count": len(sp), **check})
                            saved_ref = (torch.from_numpy(saved_low[lo:lo+len(sp)]).to(device), torch.from_numpy(saved_iou[lo:lo+len(sp)]).to(device))
                            replay_checks.append(compare(saved_ref, reference, 5e-5))
                            lows.append(outputs[0].cpu().numpy()); scores.append(outputs[1].cpu().numpy()); masks.append(full.cpu().numpy())
                            del reference, reference_full, outputs, full, saved_ref
                        arm_failed = any(c["status"] == "FAILED" for c in checks)
                        failed |= arm_failed
                        output_path = folder/f"{regime}_{arm[0]}.npz"
                        save_arrays(output_path, {"masks": np.concatenate(masks), "low_resolution_logits": np.concatenate(lows),
                                                  "iou_prediction": np.concatenate(scores), "gt": gt, "annotation_ids": annotation_ids}, "none")
                        report["records"].append({"image_id": image_id, "regime": regime, "method": arm[0], "decoder_method": arm[1],
                                                  "attention": arm[2], "plans": plans, "numeric_status": "FAILED" if arm_failed else "PASSED",
                                                  "numeric_chunks": checks, "official_replay_matches_saved_5e5": all(c["passed"] for c in replay_checks),
                                                  "official_replay_checks": replay_checks, "prompt_metadata": original["prompt_metadata"],
                                                  "npz": str(output_path.relative_to(args.output_dir)), "quality": {"status": "DEFERRED_OFFLINE", "rows": []}})
                        cache = None
                    atomic_json(args.output_dir/"report.json", report)
            report["images"].append({"image_id": image_id})
            print(json.dumps({"completed_images": len(report["images"]), "numeric_failure_observed": failed}), flush=True)
    report["numeric_status"] = "FAILED" if failed else "PASSED"
    report["official_replay_matches_saved_5e5"] = all(r["official_replay_matches_saved_5e5"] for r in report["records"])
    report["status"] = "FAILED_DIAGNOSTIC_OUTPUTS_ONLY" if failed else "PASSED_NUMERIC_OUTPUTS_EXPORTED_QUALITY_DEFERRED"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.output_dir/"report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--microbatch", type=int, default=4)
    args = parser.parse_args()
    if args.microbatch < 1:
        parser.error("microbatch must be positive")
    run(args)
