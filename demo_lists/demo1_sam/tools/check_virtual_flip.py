"""CPU-only synthetic contracts, not real-data scientific validation."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from takeover_virtual_flip import (PromptEncoder, REGIMES, SCOPE, decode_rows,
                                   encode_prompt_rows, mirror_objects,
                                   official_modules, run, virtual_feature_flip)
from takeover_flip_score import run as score_run


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2027)
    # 4x4 tokens on a 64-pixel canvas; complete padding must not move.
    features = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    aligned, metadata = virtual_feature_flip(features, (32, 32), 64)
    assert torch.equal(aligned[..., :2, :2], features[..., :2, :2].flip(-1))
    assert torch.equal(aligned[..., 2:, :], features[..., 2:, :])
    assert torch.equal(aligned[..., :, 2:], features[..., :, 2:])
    assert not metadata["partial_right_patch"]
    square, _ = virtual_feature_flip(features, (64, 64), 64)
    assert torch.equal(square, features.flip(-1))
    fractional, metadata = virtual_feature_flip(features, (36, 36), 64)
    # Valid width is 2.25 tokens: source indices 1.25, .25, -.75.
    expected = torch.tensor([[1.25, .25, 0], [5.25, 4.25, 4], [9.25, 8.25, 8]])
    torch.testing.assert_close(fractional[0, 0, :3, :3], expected, atol=1e-6, rtol=0)
    assert torch.equal(fractional[..., 3:, :], features[..., 3:, :])
    assert torch.equal(fractional[..., :, 3:], features[..., :, 3:])
    assert metadata["partial_right_patch"] and metadata["partial_bottom_patch"]
    assert torch.equal(features, torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4))
    try:
        virtual_feature_flip(features, (64, 65), 64)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid padding geometry accepted")

    objects = [{"annotation_id": 11, "central_xy": [1., 3.], "near_boundary_xy": [0., 1.],
                "tight_box_xyxy": [0, 1, 4, 8]},
               {"annotation_id": 12, "central_xy": [5., 10.], "near_boundary_xy": [6., 12.],
                "tight_box_xyxy": [4, 9, 7, 13]}]
    untouched = copy.deepcopy(objects)
    mirrored = mirror_objects(objects, 7)
    assert mirrored[0]["central_xy"] == [5., 3.]
    assert mirrored[0]["near_boundary_xy"] == [6., 1.]
    assert mirrored[0]["tight_box_xyxy"] == [3, 1, 7, 8]
    assert mirrored[1]["tight_box_xyxy"] == [0, 9, 3, 13]
    assert mirror_objects(mirrored, 7) == objects == untouched

    transformer, decoder_module, _, _ = official_modules()
    decoder = decoder_module.MaskDecoder(transformer_dim=32,
        transformer=transformer.TwoWayTransformer(depth=1, embedding_dim=32,
                                                  num_heads=4, mlp_dim=64)).eval()
    encoder = PromptEncoder(embed_dim=32, image_embedding_size=(4, 4),
                            input_image_size=(64, 64), mask_in_chans=16).eval()
    image = torch.randn(1, 32, 4, 4)
    pe = encoder.get_dense_pe()
    original_size, input_size = (13, 7), (64, 34)
    view, _ = virtual_feature_flip(image, input_size, 64)
    original_records, view_records = [], []
    pe_before = pe.clone()
    with tempfile.TemporaryDirectory(prefix="synthetic_virtual_flip_contract_") as temp, torch.inference_mode():
        root = Path(temp)
        try:
            run(SimpleNamespace(output_dir=root))
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output directory accepted")
        original_dir, view_dir = root / "original", root / "view"
        original_dir.mkdir()
        view_dir.mkdir()
        gt = np.zeros((2, *original_size), dtype=bool)
        gt[0, 1:8, :4] = True
        gt[1, 9:13, 4:7] = True
        for regime in REGIMES:
            sparse, dense = encode_prompt_rows(encoder, mirrored, regime, original_size, 64, "cpu")
            dense_before = dense.clone()
            original_sparse, original_dense = encode_prompt_rows(encoder, objects, regime, original_size, 64, "cpu")
            assert sparse.shape == (2, 2, 32)
            assert torch.equal(dense, original_dense)
            assert not torch.equal(sparse[:, :1], original_sparse[:, :1])
            # Encoding and decoding rows separately must reproduce batched rows.
            for index in range(2):
                single_sparse, single_dense = encode_prompt_rows(encoder, [mirrored[index]], regime, original_size, 64, "cpu")
                # CPU batched GEMM/trigonometry differs slightly across Torch builds.
                torch.testing.assert_close(single_sparse, sparse[index:index + 1], atol=5e-6, rtol=0)
                assert torch.equal(single_dense, dense[index:index + 1])
            outputs = decode_rows(decoder, view, pe, dense[:1], sparse, input_size, original_size, 0, 2, 64)
            singles = decode_rows(decoder, view, pe, dense[:1], sparse, input_size, original_size, 0, 1, 64)
            np.testing.assert_allclose(outputs["low_resolution_logits"], singles["low_resolution_logits"], atol=1e-5, rtol=0)
            np.testing.assert_allclose(outputs["iou_prediction"], singles["iou_prediction"], atol=1e-5, rtol=0)
            assert outputs["masks"].shape == (2, 4, 13, 7)
            assert outputs["iou_prediction"].shape == (2, 4)
            low = torch.from_numpy(outputs["low_resolution_logits"])
            resized = F.interpolate(low, (64, 64), mode="bilinear", align_corners=False)[..., :64, :34]
            expected_full = F.interpolate(resized, (13, 7), mode="bilinear", align_corners=False) > 0
            assert np.array_equal(outputs["masks"], expected_full.numpy())
            assert torch.equal(pe, pe_before) and torch.equal(dense, dense_before)
            original_outputs = decode_rows(decoder, image, pe, original_dense[:1], original_sparse,
                                           input_size, original_size, 0, 2, 64)
            ids = np.array([11, 12], dtype=np.int64)
            original_outputs.update(gt=gt, annotation_ids=ids)
            outputs.update(gt=gt[:, :, ::-1].copy(), annotation_ids=ids)
            np.savez(original_dir / f"{regime}.npz", **original_outputs)
            np.savez(view_dir / f"{regime}.npz", **outputs)
            record = {"image_id": 1, "regime": regime, "method": "official", "npz": f"{regime}.npz"}
            original_records.append({**record, "prompt_metadata": objects})
            view_records.append({**record, "prompt_metadata": mirrored})
        for folder, records in ((original_dir, original_records), (view_dir, view_records)):
            (folder / "report.json").write_text(json.dumps({"scope": SCOPE,
                "synthetic_fixture": True, "images": [{"image_id": 1}], "records": records}))
        with redirect_stdout(io.StringIO()):
            score_run(original_dir, view_dir, root / "score.json")
        scored = json.loads((root / "score.json").read_text())
        assert len(scored["rows"]) == 6
        assert all(1 <= choice <= 3 for row in scored["rows"] for choice in row["selections"].values())
    assert objects == untouched
    print(json.dumps({"status": "PASSED_CPU_SYNTHETIC_CONTRACTS", "real_data_tested": False,
                      "checks": ["aligned_and_partial_padding", "point_and_exclusive_box_mapping",
                                 "official_prompt_rows_independent", "four_masks_four_iou",
                                 "decoder_microbatch_independence", "PE_dense_unchanged",
                                 "official_full_resolution_postprocess", "flip_scorer_schema_and_inverse_map",
                                 "reject_existing_output_directory"],
                      "scope": SCOPE}))


if __name__ == "__main__":
    main()
