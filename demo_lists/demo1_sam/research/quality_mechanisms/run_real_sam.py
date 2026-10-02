"""Official pretrained SAM + full-output numeric and real-quality diagnosis.

This is an eager FP32 diagnostic, not a performance benchmark. No downloads,
training, extra prompts, threshold relaxation, or cross-prompt fusion.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from diagnose_oracle_gap import diagnose
from quality_summary import METHODS, attach_prompt_metadata, summarize_quality

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 5e-5


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def segment(function, device):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000


def save_arrays(path, arrays, compression):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        function = np.savez_compressed if compression == "compressed" else np.savez
        function(handle, **arrays)
    temporary.replace(path)


def source_provenance(source_dir):
    records = json.loads((ROOT / "sam_shared_decoder/source_audit/source_provenance.json").read_text())["files"]
    verified = []
    for stem in ("common", "transformer", "mask_decoder"):
        relative = f"segment_anything/modeling/{stem}.py"
        record = next(item for item in records if item["project"] == "sam" and item["source_path"] == relative)
        digest = hashlib.sha256((source_dir / relative).read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError(f"official source differs from pinned reproduction: {relative}")
        verified.append({"file": relative, "sha256": digest, "revision": record["revision_commit"]})
    return verified


def encode_prompts(sam, transform, image_size, objects, regime, device):
    if regime in ("central", "near_boundary"):
        key = "central_xy" if regime == "central" else "near_boundary_xy"
        coords = np.array([item[key] for item in objects], dtype=np.float32)[:, None]
        transformed = transform.apply_coords(coords, image_size)
        coordinates = torch.as_tensor(transformed, dtype=torch.float32, device=device)
        labels = torch.ones(coordinates.shape[:2], dtype=torch.int64, device=device)
        return sam.prompt_encoder(points=(coordinates, labels), boxes=None, masks=None)
    boxes = np.array([item["tight_box_xyxy"] for item in objects], dtype=np.float32)
    transformed = transform.apply_boxes(boxes, image_size)
    return sam.prompt_encoder(points=None, boxes=torch.as_tensor(transformed, dtype=torch.float32, device=device), masks=None)


def differences(reference, candidate, reference_full, candidate_full):
    masks, scores = reference
    got_masks, got_scores = candidate
    finite = bool(torch.isfinite(masks).all() and torch.isfinite(scores).all()
                  and torch.isfinite(got_masks).all() and torch.isfinite(got_scores).all())
    mask_error = float((got_masks - masks).abs().max().item())
    iou_error = float((got_scores - scores).abs().max().item())
    flips = (reference_full != candidate_full).sum((-1, -2)).cpu().tolist()
    selected_ref = scores[:, 1:].argmax(-1)
    selected_got = got_scores[:, 1:].argmax(-1)
    selected_changed = (selected_ref != selected_got).cpu().tolist()
    return {"mask_logit_max_abs": mask_error, "iou_prediction_max_abs": iou_error,
            "finite": finite, "tolerance": TOLERANCE,
            "status": "PASSED" if finite and mask_error <= TOLERANCE and iou_error <= TOLERANCE else "FAILED",
            "full_resolution_binary_flips_per_prompt_per_mask": flips,
            "multimask_iou_argmax_changed_per_prompt": selected_changed}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "assets/source/segment-anything")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True, choices=("cuda:0", "cpu"))
    parser.add_argument("--limit-images", type=int, default=24)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--attention", choices=("explicit", "sdpa"), default="explicit")
    parser.add_argument("--assoc-order", choices=("auto", "projected", "associated"), default="auto")
    parser.add_argument("--assoc-write-output", choices=("auto", "dense", "sparse"), default="auto")
    parser.add_argument("--diagnostic-on-failure", action="store_true")
    parser.add_argument("--defer-quality", action="store_true",
                        help="Only export outputs/numeric checks; score_saved_outputs.py computes quality later on CPU")
    parser.add_argument("--npz-compression", choices=("compressed", "none"), default="compressed",
                        help="none avoids CPU deflate blocking during GPU generation; default preserves prior storage")
    parser.add_argument("--save-encoded-inputs", action="store_true",
                        help="Save one CPU decoder state per run and image/PE/dense/sparse FP32 inputs per image")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.limit_images, args.microbatch, args.threads) < 1:
        parser.error("counts must be positive")
    if (args.output_dir / "report.json").exists():
        parser.error("report already exists; choose a new output-dir to preserve raw evidence")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    verified_sources = source_provenance(args.source_dir)
    manifest_path = args.subset_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("model_outputs_used_for_selection") is not False:
        parser.error("subset manifest must certify model-independent sampling")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_dir.resolve()))
    sys.path.insert(0, str(ROOT / "sam_shared_decoder/execution_baselines"))
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    from baselines import ImageCache, PositionCache, cached_predict, official_predict, shape_plan
    from factor_candidate import FactorCache, factor_predict
    report = {"schema_version": 2, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "RUNNING", "precision": "float32", "tf32_matmul": False, "tf32_cudnn": False,
              "numeric_status": "RUNNING", "quality_status": "DEFERRED" if args.defer_quality else "RUNNING",
              "npz_compression": args.npz_compression,
              "matmul_precision": "highest", "numerical_tolerance": TOLERANCE,
              "model": "official_pretrained_sam_vit_b", "torch": torch.__version__, "python": platform.python_version(),
              "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "official_decoder_provenance": verified_sources,
              "checkpoint": {"path": str(args.checkpoint), "bytes": args.checkpoint.stat().st_size},
              "subset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "prompt_protocol": "one independent positive point or one tight GT box per row; no mask inputs; no extra clicks or fusion",
              "timing_contract": "DIAGNOSTIC_ONLY: one-pass synchronized eager segment times, no warmup/repeated performance inference; full per-image wall includes comparisons, CPU copies, exports/NPZ I/O and quality metrics unless deferred",
              "quality_limits": "small fixed development diagnosis, not SOTA; GT oracles use extra ground truth and are not deployable",
              "images": [], "records": []}
    def save_report():
        report["quality_summary"] = summarize_quality(report["records"])
        path = args.output_dir / "report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)
    try:
        model_start = time.perf_counter()
        sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(device=device, dtype=torch.float32).eval()
        synchronize(device)
        report["model_load_ms"] = (time.perf_counter() - model_start) * 1000
        if args.save_encoded_inputs:
            state_path = args.output_dir / "mask_decoder_state.pt"
            if state_path.exists():
                raise FileExistsError("decoder export already exists; use a fresh output directory")
            export_start = time.perf_counter()
            decoder = sam.mask_decoder
            state = {"schema_version": 1, "model": "sam_vit_b_mask_decoder", "dtype": "float32",
                     "source_revision": verified_sources[0]["revision"],
                     "state_dict": {name: value.detach().cpu() for name, value in decoder.state_dict().items()},
                     "architecture": {"transformer_dim": decoder.transformer_dim,
                                      "num_multimask_outputs": decoder.num_multimask_outputs,
                                      "depth": len(decoder.transformer.layers),
                                      "num_heads": decoder.transformer.layers[0].self_attn.num_heads,
                                      "mlp_dim": decoder.transformer.layers[0].mlp.lin1.out_features}}
            torch.save(state, state_path)
            report["decoder_export"] = {"path": state_path.name, "bytes": state_path.stat().st_size,
                                         "diagnostic_export_ms": (time.perf_counter()-export_start)*1000,
                                         "save_count": 1, "checkpoint_rehashed": False}
            del state
        transform = ResizeLongestSide(sam.image_encoder.img_size)
        any_failure = False
        with torch.inference_mode():
            pe = sam.prompt_encoder.get_dense_pe()
            for image_info in manifest["images"][:args.limit_images]:
                image_start = time.perf_counter()
                image_id = int(image_info["image_id"])
                folder = args.output_dir / f"image_{image_id:012d}"
                folder.mkdir(exist_ok=True)
                image = np.array(Image.open(args.subset_dir / image_info["image_file"]).convert("RGB"))
                original_size = (image_info["height"], image_info["width"])
                if image.shape[:2] != original_size:
                    raise ValueError(f"JPEG size differs from COCO annotation for {image_id}")
                with np.load(args.subset_dir / image_info["gt_file"], allow_pickle=False) as truth:
                    gt = truth["gt"]
                    annotation_ids = truth["annotation_ids"]
                if len(gt) != len(image_info["objects"]) or annotation_ids.tolist() != [item["annotation_id"] for item in image_info["objects"]]:
                    raise ValueError("GT row order and manifest object order mismatch")
                transformed, resize_ms = segment(lambda: transform.apply_image(image), device)
                input_size = transformed.shape[:2]
                tensor, transfer_ms = segment(lambda: torch.as_tensor(transformed, device=device).permute(2, 0, 1).contiguous()[None], device)
                preprocessed, preprocess_ms = segment(lambda: sam.preprocess(tensor), device)
                features, encoder_ms = segment(lambda: sam.image_encoder(preprocessed), device)
                encoded_arrays = None
                if args.save_encoded_inputs:
                    encoded_arrays = {"image_embeddings": features.cpu().numpy(), "image_pe": pe.cpu().numpy(),
                                      "annotation_ids": annotation_ids, "original_size": np.array(original_size, dtype=np.int64),
                                      "input_size": np.array(input_size, dtype=np.int64), "image_id": np.array(image_id, dtype=np.int64),
                                      "mask_threshold": np.array(sam.mask_threshold, dtype=np.float32)}
                image_record = {"image_id": image_id, "image_encoder_executions": 1, "original_size": list(original_size),
                                "input_size": list(input_size), "resize_ms": resize_ms, "input_transfer_ms": transfer_ms,
                                "preprocess_ms": preprocess_ms, "encoder_ms": encoder_ms, "regimes": []}
                image_cache_bundle = None
                def export_image_inputs():
                    if encoded_arrays is None:
                        return
                    encoded_path = folder / "encoded_inputs.npz"
                    save_arrays(encoded_path, encoded_arrays, args.npz_compression)
                    metadata = {"schema_version": 1, "image_id": image_id,
                                "npz": str(encoded_path.relative_to(args.output_dir)),
                                "decoder_state": report["decoder_export"]["path"],
                                "subset_manifest_sha256": report["subset_manifest_sha256"],
                                "regimes_exported": [key[7:] for key in encoded_arrays if key.startswith("sparse_")],
                                "objects": image_info["objects"],
                                "contract": "real encoded FP32 tensors; each sparse row is an independent original prompt; no GT encoded in inputs"}
                    (folder / "encoded_input_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
                    image_record["encoded_inputs"] = metadata
                for regime in ("central", "near_boundary", "box"):
                    (sparse, dense_batch), prompt_ms = segment(lambda: encode_prompts(sam, transform, original_size, image_info["objects"], regime, device), device)
                    dense = dense_batch[:1]
                    if not torch.equal(dense_batch, dense.expand_as(dense_batch)):
                        raise ValueError("shared cache cannot reuse different dense prompts")
                    if encoded_arrays is not None:
                        dense_array = dense.cpu().numpy()
                        if "dense_nomask" in encoded_arrays and not np.array_equal(encoded_arrays["dense_nomask"], dense_array):
                            raise ValueError("encoded export requires identical no-mask dense prompt across regimes")
                        encoded_arrays["dense_nomask"] = dense_array
                        encoded_arrays[f"sparse_{regime}"] = sparse.cpu().numpy()
                    caches, plans, cache_ms = {}, {}, {}
                    cache_reused = bool(image_cache_bundle is not None and image_cache_bundle[0] == sparse.shape[1]+5
                                        and torch.equal(image_cache_bundle[1], dense))
                    if cache_reused:
                        caches, plans = image_cache_bundle[2:]
                        cache_ms = {method: 0.0 for method in METHODS[1:]}
                    else:
                        for method in METHODS[1:]:
                            plan = None if method == "factor_projected" else shape_plan(sam.mask_decoder, features.shape[-2]*features.shape[-1],
                                      sparse.shape[1]+5, method, args.assoc_order, args.assoc_write_output)
                            plans[method] = plan
                            def build_cache():
                                position = PositionCache.build(sam.mask_decoder, pe, plan)
                                cache_type = FactorCache if method == "factor_projected" else ImageCache
                                return cache_type.build(sam.mask_decoder, features, dense, position)
                            caches[method], cache_ms[method] = segment(build_cache, device)
                        image_cache_bundle = (sparse.shape[1]+5, dense, caches, plans)
                    collected = {method: {"low": [], "iou": [], "masks": [], "chunks": [], "decoder_ms": [], "postprocess_ms": []} for method in METHODS}
                    for lower in range(0, len(sparse), args.microbatch):
                        sp = sparse[lower:lower+args.microbatch]
                        reference, reference_full = None, None
                        for method in METHODS:
                            def predict():
                                if method == "official":
                                    return official_predict(sam.mask_decoder, features, pe, dense, sp)
                                if method == "factor_projected":
                                    return factor_predict(sam.mask_decoder, caches[method], sp, args.attention)
                                return cached_predict(sam.mask_decoder, caches[method], sp, method, args.attention,
                                                      args.assoc_order, args.assoc_write_output, plans=plans[method])
                            outputs, decoder_ms = segment(predict, device)
                            full_logits, postprocess_ms = segment(lambda: sam.postprocess_masks(outputs[0], input_size, original_size), device)
                            full = full_logits > sam.mask_threshold
                            if outputs[0].shape[:2] != (len(sp), 4) or outputs[1].shape != (len(sp), 4):
                                raise ValueError("must retain all four raw masks and IoU predictions")
                            if method == "official":
                                reference, reference_full = outputs, full
                                comparison = {"status": "REFERENCE", "finite": bool(torch.isfinite(outputs[0]).all() and torch.isfinite(outputs[1]).all())}
                                if not comparison["finite"]:
                                    raise ValueError("official pretrained output is nonfinite")
                            else:
                                comparison = differences(reference, outputs, reference_full, full)
                                any_failure |= comparison["status"] == "FAILED"
                            store = collected[method]
                            store["low"].append(outputs[0].cpu().numpy())
                            store["iou"].append(outputs[1].cpu().numpy())
                            store["masks"].append(full.cpu().numpy())
                            store["decoder_ms"].append(decoder_ms)
                            store["postprocess_ms"].append(postprocess_ms)
                            store["chunks"].append({"row_start": lower, "row_count": len(sp), **comparison})
                            del full_logits
                    for method, store in collected.items():
                        arrays = {"masks": np.concatenate(store["masks"]), "low_resolution_logits": np.concatenate(store["low"]),
                                  "iou_prediction": np.concatenate(store["iou"]), "gt": gt, "annotation_ids": annotation_ids}
                        output_path = folder / f"{regime}_{method}.npz"
                        save_arrays(output_path, arrays, args.npz_compression)
                        quality = {"status": "DEFERRED_OFFLINE", "rows": []}
                        if not args.defer_quality:
                            try:
                                quality = diagnose(arrays)
                            except ValueError as error:
                                quality = {"status": "INVALID_QUALITY_INPUT", "error": str(error), "rows": []}
                            attach_prompt_metadata(quality, {"image_id": image_id, "regime": regime,
                                                            "prompt_metadata": image_info["objects"]}, annotation_ids)
                            (folder / f"{regime}_{method}_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
                        failed = any(chunk["status"] == "FAILED" for chunk in store["chunks"])
                        record = {"image_id": image_id, "regime": regime, "method": method,
                                  "numeric_status": "REFERENCE" if method == "official" else ("FAILED" if failed else "PASSED"),
                                  "attention": "official_explicit" if method == "official" else args.attention,
                                  "factor_write_attention": "explicit" if method == "factor_projected" else None,
                                  "shape_plan": plans.get(method), "npz": str(output_path.relative_to(args.output_dir)),
                                  "prompt_metadata": image_info["objects"],
                                  "numeric_chunks": store["chunks"], "diagnostic_decoder_ms": store["decoder_ms"],
                                  "diagnostic_postprocess_ms": store["postprocess_ms"], "quality": quality}
                        report["records"].append(record)
                    image_record["regimes"].append({"regime": regime, "prompt_encoder_ms": prompt_ms,
                                                      "cache_build_including_PE_ms": cache_ms, "prompt_count": len(sparse),
                                                      "total_tokens": sparse.shape[1]+5, "image_cache_reused_from_prior_regime": cache_reused})
                    report["status"] = "FAILED_DIAGNOSTIC_OUTPUTS_ONLY" if any_failure else "RUNNING"
                    report["numeric_status"] = "FAILED" if any_failure else "RUNNING"
                    save_report()
                    if any_failure and not args.diagnostic_on_failure:
                        export_image_inputs()
                        report["status"] = "FAILED_STOPPED_AT_NUMERIC_GATE"
                        report["images"].append(image_record)
                        save_report()
                        return 2
                export_image_inputs()
                image_record["complete_diagnostic_wall_ms"] = (time.perf_counter()-image_start)*1000
                report["images"].append(image_record)
                save_report()
                print(json.dumps({"image_id": image_id, "completed_images": len(report["images"]), "numeric_failure_observed": any_failure}), flush=True)
        report["status"] = "FAILED_DIAGNOSTIC_OUTPUTS_ONLY" if any_failure else (
            "PASSED_NUMERIC_OUTPUTS_EXPORTED_QUALITY_DEFERRED" if args.defer_quality else "PASSED_NUMERIC_AND_COMPLETED_QUALITY_DIAGNOSIS")
        report["numeric_status"] = "FAILED" if any_failure else "PASSED"
        report["quality_status"] = "DEFERRED" if args.defer_quality else "COMPLETED"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_report()
        return 0 if not any_failure or args.diagnostic_on_failure else 2
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        save_report()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
