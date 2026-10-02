"""Synthetic CPU check of encoded replay, all new arms, and offline scoring."""
import json
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
from takeover_common import *
from takeover_replay import run


configure()
sys.path.insert(0, str(ROOT/"assets/source/segment-anything"))
from segment_anything.modeling.sam import Sam
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    raw = root/"raw"
    raw.mkdir()
    folder = raw/"image_000000000001"
    folder.mkdir()
    model, image, pe, dense, sparse, _ = make_fixture(grid=4, batch=3)
    torch.save({"state_dict": model.state_dict()}, raw/"mask_decoder_state.pt")
    encoded = {"image_embeddings": image.numpy(), "image_pe": pe.numpy(), "dense_nomask": dense.numpy(),
               "input_size": np.array([1024,1024]), "original_size": np.array([32,40]), "mask_threshold": np.array(0.),
               **{f"sparse_{r}": sparse.numpy() for r in ("central", "near_boundary", "box")}}
    np.savez(folder/"encoded_inputs.npz", **encoded)
    metadata = [{"annotation_id": i, "central_xy": [1.,1.], "near_boundary_xy": [1.,1.], "tight_box_xyxy": [0,0,40,32]} for i in range(3)]
    source = {"status": "SYNTHETIC_CPU_ONLY", "model": "random_synthetic_CPU_contract",
              "subset_manifest_sha256": "synthetic", "records": [],
              "images": [{"image_id": 1, "encoded_inputs": {"npz": "image_000000000001/encoded_inputs.npz"}}]}
    with torch.inference_mode():
        outputs = official_predict(model, image, pe, dense, sparse)
        full = Sam.postprocess_masks(SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024)), outputs[0], (1024,1024), (32,40))>0
    for regime in ("central", "near_boundary", "box"):
        npz = folder/f"{regime}_official.npz"
        np.savez(npz, low_resolution_logits=outputs[0].numpy(), iou_prediction=outputs[1].numpy(),
                 masks=full.numpy(), gt=full[:,0].numpy(), annotation_ids=np.arange(3))
        source["records"].append({"image_id": 1, "regime": regime, "method": "official", "npz": str(npz.relative_to(raw)), "prompt_metadata": metadata})
    atomic_json(raw/"report.json", source)
    out = root/"replay"
    run(SimpleNamespace(run_dir=raw, output_dir=out, device="cpu", microbatch=2))
    report = json.loads((out/"report.json").read_text())
    assert len(report["records"]) == 24
    assert report["numeric_status"] == "PASSED" and report["official_replay_matches_saved_5e5"]
    before = (out/"report.json").read_bytes()
    subprocess.run([sys.executable, str(ROOT/"tools/takeover_score.py"), "--run-dir", str(out), "--output-dir", str(root/"score")], check=True)
    scored = json.loads((root/"score/score_report.json").read_text())
    assert len(scored["quality_summary"]) == 24 and scored["source_records_scored"] == 24
    assert (out/"report.json").read_bytes() == before
    # The scoring adapter must preserve an existing numeric failure exactly.
    report["status"] = "FAILED_DIAGNOSTIC_OUTPUTS_ONLY"
    report["numeric_status"] = "FAILED"
    report["records"][0]["numeric_status"] = "FAILED"
    atomic_json(out/"report.json", report)
    subprocess.run([sys.executable, str(ROOT/"tools/takeover_score.py"), "--run-dir", str(out), "--output-dir", str(root/"score_failed")], check=True)
    failed = json.loads((root/"score_failed/score_report.json").read_text())
    assert failed["source_numeric_status"] == "FAILED" and failed["records"][0]["numeric_status"] == "FAILED"
print("PASS: synthetic CPU replay 3 regimes x 8 arms, tail microbatch, full masks, dynamic scoring, numeric failure preservation")
