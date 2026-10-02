"""CPU contracts for real-asset preparation and prompt/numeric interfaces.

Uses explicit synthetic archive + injectable synthetic decoder. It does not
test pycocotools or a pretrained model. No downloads, checkpoint, or GPU.
"""
import io
import json
from pathlib import Path
import tempfile
from unittest import mock
import zipfile

import numpy as np
from PIL import Image
import torch

from prepare_coco_subset import prepare, prompts_from_mask
from run_real_sam import differences, encode_prompts, source_provenance, ROOT
import run_real_sam
from score_saved_outputs import score_run


mask = np.zeros((40, 50), dtype=bool)
mask[2:38, 3:47] = True
central, near, box, depth = prompts_from_mask(mask, np.random.default_rng(2027))
assert mask[int(central[1]), int(central[0])] and mask[int(near[1]), int(near[0])]
assert box == [3, 2, 47, 38] and depth == 18
assert min(near[0]-3, 46-near[0], near[1]-2, 37-near[1]) == 0


class PromptEncoder:
    def __call__(self, points, boxes, masks):
        assert masks is None
        if boxes is None:
            coordinates, labels = points
            assert coordinates.shape == (2, 1, 2) and labels.shape == (2, 1)
            assert labels.eq(1).all()
            return coordinates, labels
        assert points is None and boxes.shape == (2, 4)
        return boxes, None


class Model:
    prompt_encoder = PromptEncoder()


class Transform:
    def apply_coords(self, x, size):
        assert size == (40, 50)
        return x * 2
    def apply_boxes(self, x, size):
        assert size == (40, 50)
        return x * 2


objects = [{"central_xy": central, "near_boundary_xy": near, "tight_box_xyxy": box}] * 2
for regime in ("central", "near_boundary", "box"):
    encode_prompts(Model(), Transform(), (40, 50), objects, regime, torch.device("cpu"))
reference = (torch.zeros(2, 4, 3, 3), torch.zeros(2, 4))
candidate = (reference[0].clone(), reference[1].clone())
candidate[0][0, 3, 0, 0] = 1e-3
full_ref = torch.zeros(2, 4, 5, 5, dtype=torch.bool)
full_got = full_ref.clone()
full_got[0, 3, 0, 0] = True
delta = differences(reference, candidate, full_ref, full_got)
assert delta["status"] == "FAILED" and delta["tolerance"] == 5e-5
assert delta["full_resolution_binary_flips_per_prompt_per_mask"] == [[0, 0, 0, 1], [0, 0, 0, 0]]
with tempfile.TemporaryDirectory() as temporary:
    folder = Path(temporary)
    images, annotations = [], []
    for image_id in range(5):
        images.append({"id": image_id, "file_name": f"{image_id:012d}.jpg", "height": 40, "width": 50})
        for object_id in range(6):
            annotations.append({"id": image_id*10+object_id, "image_id": image_id, "category_id": 1,
                                "bbox": [3, 2, 44, 36], "area": 1584, "iscrowd": 0 if object_id < 5 else 1,
                                "segmentation": [[3, 2, 47, 2, 47, 38, 3, 38]]})
    with zipfile.ZipFile(folder / "annotations.zip", "w") as archive:
        archive.writestr("annotations/instances_val2017.json", json.dumps({"images": images, "annotations": annotations, "categories": []}))
    payload = io.BytesIO()
    Image.fromarray(np.zeros((40, 50, 3), dtype=np.uint8)).save(payload, format="JPEG")
    with zipfile.ZipFile(folder / "images.zip", "w") as archive:
        for image in images:
            archive.writestr("val2017/"+image["file_name"], payload.getvalue())
    decode = lambda annotation, h, w: mask.copy()
    first = prepare(folder / "annotations.zip", folder / "images.zip", folder / "first", num_images=2, decoder=decode)
    second = prepare(folder / "annotations.zip", folder / "images.zip", folder / "second", num_images=2, decoder=decode)
    assert first == second and first["num_instances"] == 8
    assert len(list((folder / "first/images").glob("*.jpg"))) == 2
    assert len(json.loads((folder / "first/selected_annotations.json").read_text())["annotations"]) == 8
    assert all(item["annotation_id"] % 10 != 5 for image in first["images"] for item in image["objects"])
    try:
        prepare(folder / "annotations.zip", folder / "images.zip", folder / "first", num_images=2, decoder=decode)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing selection must not be overwritten")
    # Execute the exact real harness with tiny synthetic official SAM components.
    # No pretrained checkpoint is loaded. Temporary outputs are never real evidence.
    import sys
    sys.path.insert(0, str(ROOT / "assets/source/segment-anything"))
    sys.path.insert(0, str(ROOT / "sam_shared_decoder/execution_baselines"))
    from segment_anything import sam_model_registry
    from segment_anything.modeling import Sam, PromptEncoder
    from baselines import make_fixture
    class TinyImageEncoder(torch.nn.Module):
        img_size = 64
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Conv2d(3, 256, 16, stride=16)
        def forward(self, images):
            return self.projection(images)
    def synthetic_builder(checkpoint):
        decoder_model = make_fixture(seed=333, grid=4, batch=1)[0]
        return Sam(image_encoder=TinyImageEncoder(),
                   prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(4, 4), input_image_size=(64, 64), mask_in_chans=16),
                   mask_decoder=decoder_model)
    (folder / "synthetic_empty_checkpoint").write_bytes(b"")
    arguments = ["run_real_sam.py", "--subset-dir", str(folder / "first"), "--checkpoint", str(folder / "synthetic_empty_checkpoint"),
                 "--output-dir", str(folder / "synthetic_run"), "--device", "cpu", "--limit-images", "1", "--microbatch", "3"]
    with mock.patch.dict(sam_model_registry, {"vit_b": synthetic_builder}), mock.patch.object(sys, "argv", arguments):
        assert run_real_sam.main() == 0
    harness_report = json.loads((folder / "synthetic_run/report.json").read_text())
    assert len(harness_report["records"]) == 12
    assert harness_report["images"][0]["image_encoder_executions"] == 1
    assert [item["image_cache_reused_from_prior_regime"] for item in harness_report["images"][0]["regimes"]] == [False, True, True]
    assert len(harness_report["quality_summary"]) == 12
    for record in harness_report["records"]:
        with np.load(folder / "synthetic_run" / record["npz"], allow_pickle=False) as archive:
            assert archive["masks"].shape == (4, 4, 40, 50)
            assert archive["low_resolution_logits"].shape == (4, 4, 16, 16)
            assert archive["iou_prediction"].shape == (4, 4)
    deferred_arguments = arguments.copy()
    deferred_arguments[deferred_arguments.index("--output-dir")+1] = str(folder / "deferred_run")
    deferred_arguments.extend(["--defer-quality", "--npz-compression", "none", "--save-encoded-inputs", "--diagnostic-on-failure"])
    with mock.patch.dict(sam_model_registry, {"vit_b": synthetic_builder}), mock.patch.object(sys, "argv", deferred_arguments), \
            mock.patch.object(run_real_sam, "diagnose", side_effect=AssertionError("deferred generator must not score quality")):
        assert run_real_sam.main() == 0
    deferred_bytes = (folder / "deferred_run/report.json").read_bytes()
    deferred_report = json.loads(deferred_bytes)
    assert deferred_report["quality_status"] == "DEFERRED" and deferred_report["numeric_status"] == "PASSED"
    assert deferred_report["quality_summary"] == {} and deferred_report["decoder_export"]["save_count"] == 1
    assert not list((folder / "deferred_run").rglob("*_quality.json"))
    offline = score_run(folder / "deferred_run", folder / "offline_score")
    assert offline["quality_summary"] == harness_report["quality_summary"]
    assert (folder / "deferred_run/report.json").read_bytes() == deferred_bytes
    assert [item["numeric_status"] for item in offline["records"]] == [item["numeric_status"] for item in deferred_report["records"]]
    for old_record, new_record in zip(harness_report["records"], deferred_report["records"]):
        with np.load(folder / "synthetic_run" / old_record["npz"], allow_pickle=False) as first_arrays, \
                np.load(folder / "deferred_run" / new_record["npz"], allow_pickle=False) as second_arrays:
            assert all(np.array_equal(first_arrays[key], second_arrays[key]) for key in first_arrays.files)
        with zipfile.ZipFile(folder / "deferred_run" / new_record["npz"]) as archive:
            assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
    checkpoint = torch.load(folder / "deferred_run/mask_decoder_state.pt", map_location="cpu", weights_only=True)
    assert checkpoint["architecture"] == {"transformer_dim": 256, "num_multimask_outputs": 3, "depth": 2, "num_heads": 8, "mlp_dim": 2048}
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in checkpoint["state_dict"].values())
    replay_decoder = make_fixture(seed=999, grid=4, batch=1)[0]
    replay_decoder.load_state_dict(checkpoint["state_dict"])
    from baselines import official_predict
    encoded_path = folder / "deferred_run" / deferred_report["images"][0]["encoded_inputs"]["npz"]
    with np.load(encoded_path, allow_pickle=False) as encoded:
        assert encoded["image_embeddings"].shape == (1, 256, 4, 4)
        assert encoded["dense_nomask"].shape == (1, 256, 4, 4)
        for regime in ("central", "near_boundary", "box"):
            assert encoded[f"sparse_{regime}"].shape == (4, 2, 256)
            with torch.inference_mode():
                replay = official_predict(replay_decoder, *[torch.from_numpy(encoded[key]) for key in
                    ("image_embeddings", "image_pe", "dense_nomask", f"sparse_{regime}")])
            official_record = next(item for item in deferred_report["records"] if item["method"] == "official" and item["regime"] == regime)
            with np.load(folder / "deferred_run" / official_record["npz"], allow_pickle=False) as actual:
                # Original microbatch3+tail1 vs replay batch4 can reorder reductions.
                assert np.allclose(replay[0].numpy(), actual["low_resolution_logits"], rtol=0, atol=5e-5)
                assert np.allclose(replay[1].numpy(), actual["iou_prediction"], rtol=0, atol=5e-5)
    real_difference = run_real_sam.differences
    def injected_numeric_failure(*values):
        result = real_difference(*values)
        result.update(mask_logit_max_abs=1e-3, status="FAILED")
        return result
    failed_reports = []
    for mode in ("online", "deferred"):
        failed_arguments = arguments.copy()
        failed_arguments[failed_arguments.index("--output-dir")+1] = str(folder / f"failed_{mode}")
        failed_arguments.append("--diagnostic-on-failure")
        if mode == "deferred":
            failed_arguments.append("--defer-quality")
        with mock.patch.dict(sam_model_registry, {"vit_b": synthetic_builder}), mock.patch.object(sys, "argv", failed_arguments), \
                mock.patch.object(run_real_sam, "differences", side_effect=injected_numeric_failure):
            assert run_real_sam.main() == 0
        failed_reports.append(json.loads((folder / f"failed_{mode}" / "report.json").read_text()))
    assert failed_reports[0]["status"] == failed_reports[1]["status"] == "FAILED_DIAGNOSTIC_OUTPUTS_ONLY"
    scored_failure = score_run(folder / "failed_deferred", folder / "failed_offline_score")
    assert scored_failure["source_numeric_status"] == "FAILED" and scored_failure["source_run_status"] == "FAILED_DIAGNOSTIC_OUTPUTS_ONLY"
    assert scored_failure["quality_summary"] == failed_reports[0]["quality_summary"]
    assert [item["numeric_status"] for item in scored_failure["records"]] == [item["numeric_status"] for item in failed_reports[1]["records"]]
source_provenance(ROOT / "assets/source/segment-anything")
status = {"status": "SYNTHETIC_CPU_CONTRACTS_ONLY", "annotation_only_seeded_sampling": "PASS",
          "selected_JPEG_only_extraction": "PASS", "max_instances_and_non_crowd": "PASS",
          "deep_central_and_legal_boundary_points": "PASS", "XY_box_transform_interfaces": "PASS",
          "original_5e-5_failure_gate_and_fullmask_flips": "PASS", "pinned_official_decoder_sources": "PASS",
          "full_harness_synthetic_official_components_3_regimes_4_methods_tail_microbatch": "PASS",
          "single_image_encoder_and_shared_cache_across_regimes": "PASS",
          "deferred_generator_never_calls_quality_and_uncompressed_npz": "PASS",
          "offline_scoring_matches_online_and_leaves_source_report_unchanged": "PASS",
          "single_cpu_decoder_state_export_and_encoded_input_replay": "PASS",
          "injected_numeric_failure_status_preserved_online_deferred_and_offline": "PASS",
          "limitations": ["pycocotools unavailable locally, decode tested with explicit synthetic injection", "No actual COCO archive or pretrained weights loaded", "No GPU run"]}
Path(__file__).with_name("real_pipeline_contract_results.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
