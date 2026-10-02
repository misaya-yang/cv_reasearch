"""Heuristic horizontal feature view of saved SAM inputs; no image encoder run.

This is a development-set diagnosis, not RGB-flip encoder equivalence. Encoder
absolute positions, global context and partial patch/padding mixtures survive in
the original features. Four official decoder masks/scores are retained per prompt.
Outputs stay in mirrored full-image coordinates for takeover_flip_score.py.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "assets/source/segment-anything"))
sys.path.insert(0, str(ROOT / "sam_shared_decoder/execution_baselines"))
from segment_anything.modeling import PromptEncoder
from segment_anything.modeling.sam import Sam
from segment_anything.utils.transforms import ResizeLongestSide
from baselines import official_modules

SCOPE = "HEURISTIC_VIRTUAL_FEATURE_VIEW_NOT_RGB_ENCODER_EQUIVALENCE"
REGIMES = ("central", "near_boundary", "box")


def virtual_feature_flip(features, input_size, canvas_size=1024):
    """Reflect token centers around the valid resized pixel interval [0, W].

    Whole padded rows/columns remain byte-identical. For noninteger patch widths,
    bilinear interpolation samples reflected centers, with border replication.
    Intersecting partial patches are included; their mixed content cannot be
    separated. Even aligned-width permutations are not encoder equivariance.
    """
    if features.ndim != 4 or features.shape[0] != 1:
        raise ValueError("one BCHW image feature map required")
    height, width = (int(v) for v in input_size)
    gh, gw = features.shape[-2:]
    if not (0 < height <= canvas_size and 0 < width <= canvas_size):
        raise ValueError("input_size must fit padded square canvas")
    valid_h, valid_w = height * gh / canvas_size, width * gw / canvas_size
    rows, cols = math.ceil(valid_h), math.ceil(valid_w)
    result = features.clone()
    if valid_w.is_integer():
        result[..., :rows, :cols] = features[..., :rows, :cols].flip(-1)
    else:
        # align_corners=False: normalized centers map to source index x - .5.
        y = torch.arange(rows, device=features.device, dtype=features.dtype) + .5
        x = valid_w - (torch.arange(cols, device=features.device, dtype=features.dtype) + .5)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((2 * xx / gw - 1, 2 * yy / gh - 1), -1)[None]
        result[..., :rows, :cols] = F.grid_sample(
            features, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return result, {"valid_extent_in_tokens_hw": [valid_h, valid_w],
                    "intersecting_token_rows_columns": [rows, cols],
                    "partial_bottom_patch": not valid_h.is_integer(),
                    "partial_right_patch": not valid_w.is_integer(),
                    "complete_padding_tokens_preserved": True,
                    "mapping": "x_source_center = input_width/patch_width - x_target_center",
                    "partial_patch_rule": "bilinear token-center reflection; border replication; mixed padding is inseparable"}


def mirror_objects(objects, width):
    mirrored = copy.deepcopy(objects)
    for obj in mirrored:
        for key in ("central_xy", "near_boundary_xy"):
            x, y = obj[key]
            obj[key] = [width - 1 - x, y]
        x0, y0, x1, y1 = obj["tight_box_xyxy"]
        obj["tight_box_xyxy"] = [width - x1, y0, width - x0, y1]
    return mirrored


def encode_prompt_rows(encoder, objects, regime, original_size, canvas_size, device):
    transform = ResizeLongestSide(canvas_size)
    if regime in ("central", "near_boundary"):
        key = "central_xy" if regime == "central" else "near_boundary_xy"
        coordinates = np.array([obj[key] for obj in objects], dtype=np.float32)[:, None]
        points = torch.as_tensor(transform.apply_coords(coordinates, original_size),
                                 device=device, dtype=torch.float32)
        labels = torch.ones(points.shape[:2], device=device, dtype=torch.int64)
        return encoder(points=(points, labels), boxes=None, masks=None)
    if regime != "box":
        raise ValueError(regime)
    boxes = np.array([obj["tight_box_xyxy"] for obj in objects], dtype=np.float32)
    boxes = torch.as_tensor(transform.apply_boxes(boxes, original_size), device=device, dtype=torch.float32)
    return encoder(points=None, boxes=boxes, masks=None)


def load_models(state_path, checkpoint_path, device):
    exported = torch.load(state_path, weights_only=True, map_location="cpu")
    architecture = exported["architecture"]
    transformer, decoder_module, _, provenance = official_modules()
    decoder = decoder_module.MaskDecoder(
        transformer_dim=architecture["transformer_dim"],
        num_multimask_outputs=architecture["num_multimask_outputs"],
        transformer=transformer.TwoWayTransformer(
            depth=architecture["depth"], embedding_dim=architecture["transformer_dim"],
            num_heads=architecture["num_heads"], mlp_dim=architecture["mlp_dim"]))
    decoder.load_state_dict(exported["state_dict"], strict=True)
    encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                            input_image_size=(1024, 1024), mask_in_chans=16)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    encoder.load_state_dict({key.removeprefix("prompt_encoder."): value
                             for key, value in checkpoint.items()
                             if key.startswith("prompt_encoder.")}, strict=True)
    del checkpoint, exported
    return (decoder.to(device).eval().requires_grad_(False),
            encoder.to(device).eval().requires_grad_(False), provenance)


def decode_rows(decoder, image, pe, dense, sparse, input_size, original_size,
                threshold, microbatch, canvas_size=1024):
    lows, scores, masks = [], [], []
    post_self = SimpleNamespace(image_encoder=SimpleNamespace(img_size=canvas_size))
    for start in range(0, len(sparse), microbatch):
        chunk = sparse[start:start + microbatch]
        # predict_masks is the official full-output path, before slicing tokens.
        low, score = decoder.predict_masks(
            image_embeddings=image, image_pe=pe, sparse_prompt_embeddings=chunk,
            dense_prompt_embeddings=dense.expand(len(chunk), -1, -1, -1))
        if low.shape[:2] != (len(chunk), 4) or score.shape != (len(chunk), 4):
            raise ValueError("must retain four masks and four IoU predictions")
        if not (torch.isfinite(low).all() and torch.isfinite(score).all()):
            raise ValueError("nonfinite official decoder output")
        full = Sam.postprocess_masks(post_self, low, input_size, original_size) > threshold
        lows.append(low.cpu().numpy())
        scores.append(score.cpu().numpy())
        masks.append(full.cpu().numpy())
    return {"low_resolution_logits": np.concatenate(lows),
            "iou_prediction": np.concatenate(scores), "masks": np.concatenate(masks)}


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source = json.loads((args.source_run / "report.json").read_text())
    if not source["images"] or any("encoded_inputs" not in image for image in source["images"]):
        raise ValueError("source must contain saved encoded inputs for every image")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    decoder, encoder, provenance = load_models(args.source_run / "mask_decoder_state.pt", args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 2, "scope": SCOPE, "status": "RUNNING",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_run": str(args.source_run), "source_run_status": source["status"],
              "subset_manifest_sha256": source.get("subset_manifest_sha256"),
              "cohort": "same saved-input development cohort; no new heldout evidence",
              "device": str(device), "precision": "float32", "quality_status": "DEFERRED",
              "image_encoder_executions": 0, "official_decoder_provenance": provenance,
              "checkpoint": str(args.checkpoint),
              "approximation_limits": "Encoded absolute positions/global context persist; partial patch padding mixtures cannot be separated; no RGB-encoder equivalence or SOTA claim",
              "prompt_protocol": "independent original single-positive points/GT boxes mirrored geometrically; official PromptEncoder, no new clicks, no sparse-embedding flip, no mask prompts or cross-prompt fusion",
              "output_coordinates": "mirrored full-image masks and GT; scorer inverse-flips after official resize->crop->resize and threshold; matching tokens 1..3 only",
              "point_mapping": "x'=W-1-x", "box_mapping": "exclusive XYXY [W-xmax,ymin,W-xmin,ymax]",
              "gt_use": "original prompt metadata only; GT tensors read after decoder output solely for evaluation export; no GT ranking",
              "images": [], "records": []}

    def save_report():
        path = args.output_dir / "report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    save_report()
    try:
        with torch.inference_mode():
            for info in source["images"]:
                image_id = info["image_id"]
                folder = args.output_dir / f"image_{image_id:012d}"
                folder.mkdir()
                with np.load(args.source_run / info["encoded_inputs"]["npz"], allow_pickle=False) as encoded:
                    features, pe, dense = [torch.from_numpy(encoded[key]).to(device)
                                           for key in ("image_embeddings", "image_pe", "dense_nomask")]
                    if any(t.dtype != torch.float32 or t.shape != (1, 256, 64, 64) for t in (features, pe, dense)):
                        raise ValueError("expected one FP32 SAM ViT-B encoded image/PE/dense grid")
                    input_size = tuple(int(v) for v in encoded["input_size"])
                    original_size = tuple(int(v) for v in encoded["original_size"])
                    if ResizeLongestSide.get_preprocess_shape(*original_size, 1024) != input_size:
                        raise ValueError("input_size differs from official ResizeLongestSide")
                    ids = encoded["annotation_ids"].copy()
                    if int(encoded["image_id"]) != image_id:
                        raise ValueError("encoded image ID mismatch")
                    if not torch.allclose(encoder.get_dense_pe(), pe, atol=5e-5, rtol=0):
                        raise ValueError("checkpoint PE differs from saved image PE")
                    flipped, mapping = virtual_feature_flip(features, input_size)
                    for regime in REGIMES:
                        original = next(r for r in source["records"] if r["image_id"] == image_id
                                        and r["method"] == "official" and r["regime"] == regime)
                        objects = original["prompt_metadata"]
                        if ids.tolist() != [obj["annotation_id"] for obj in objects]:
                            raise ValueError("prompt metadata row order differs from encoded IDs")
                        original_sparse, original_dense = encode_prompt_rows(
                            encoder, objects, regime, original_size, 1024, device)
                        if not torch.equal(original_dense, dense.expand_as(original_dense)):
                            raise ValueError("checkpoint no-mask embedding differs from saved input")
                        if not torch.allclose(original_sparse, torch.from_numpy(encoded[f"sparse_{regime}"]).to(device), atol=5e-5, rtol=0):
                            raise ValueError("original prompt replay differs from saved sparse inputs")
                        mirrored = mirror_objects(objects, original_size[1])
                        sparse, mirrored_dense = encode_prompt_rows(encoder, mirrored, regime, original_size, 1024, device)
                        if not torch.equal(mirrored_dense, dense.expand_as(mirrored_dense)):
                            raise ValueError("mirrored prompt changed no-mask embedding")
                        arrays = decode_rows(decoder, flipped, pe, dense, sparse, input_size,
                                             original_size, float(encoded["mask_threshold"]), args.microbatch)
                        # No ground-truth tensors are supplied to any decoder or ranking call.
                        with np.load(args.source_run / original["npz"], allow_pickle=False) as truth:
                            if not np.array_equal(ids, truth["annotation_ids"]) or truth["gt"].shape != (len(ids), *original_size):
                                raise ValueError("GT shape/row order differs from source encoded inputs")
                            arrays["gt"] = truth["gt"][:, :, ::-1].copy()
                        arrays["annotation_ids"] = ids
                        path = folder / f"{regime}_official.npz"
                        np.savez(path, **arrays)
                        report["records"].append({"image_id": image_id, "regime": regime,
                                                  "method": "official", "view": SCOPE,
                                                  "npz": str(path.relative_to(args.output_dir)),
                                                  "prompt_metadata": mirrored,
                                                  "original_prompt_replay_5e5": True,
                                                  "quality": {"status": "DEFERRED_OFFLINE", "rows": []}})
                        save_report()
                report["images"].append({"image_id": image_id, "original_size": list(original_size),
                                         "input_size": list(input_size), "image_encoder_executions": 0,
                                         "feature_flip": mapping})
                save_report()
                print(json.dumps({"completed_images": len(report["images"]), "image_id": image_id,
                                  "scope": SCOPE}), flush=True)
        report["status"] = "COMPLETED_HEURISTIC_DEVELOPMENT_OUTPUTS_QUALITY_DEFERRED"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_report()
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = {"message": str(error), "traceback": traceback.format_exc()}
        save_report()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "assets/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.microbatch, args.threads) < 1:
        parser.error("microbatch and threads must be positive")
    run(args)
