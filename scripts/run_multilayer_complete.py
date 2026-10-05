#!/usr/bin/env python3
"""Single-encoder complete D/v1 and frozen delta-only comparison.

One shared pair forward per ordinary episode. The first episode additionally
checks the native layer tap and independent complete public FoRIS. Inference
never reads query labels. Score only after all predictions have been sealed.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--foris-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--save-layers", action="store_true",
                        help="Save raw float32 q/r layers16/24; 64 MiB per 1024 episode, about 15.1 GiB for 241")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.out.exists():
        parser.error("Choose a fresh output directory; existing outputs are never replaced")

    import numpy as np
    import torch
    from PIL import Image
    from ics.data import TimmDINOv3
    from ics.experiment import load_rows, sha
    from ics.foris import build_host, run_foris
    from ics.methods import multilayer, multilayer_controls
    from ics.methods.layer_extract import extract_pair

    manifest = json.loads(args.manifest.read_text())
    allowed = ("fold", "e", "c", "key", "support", "query", "batch",
               "packet_export", "feature_export", "split", "seed")
    rows = [{key: row[key] for key in allowed if key in row}
            for row in load_rows(args.manifest)[:args.limit]]
    if not rows:
        parser.error("Empty selected cohort")
    if not manifest.get("projection_basis"):
        parser.error("Require the existing frozen native basis; do not silently rebuild native")
    if not torch.cuda.is_available():
        raise RuntimeError("This prepared experiment requires a visible GPU")
    root = args.out
    root.mkdir(parents=True)
    for folder in ("predictions", "packets"):
        (root / folder).mkdir()
    layer_bytes = len(rows) * 4 * 4096 * 1024 * 4
    if args.save_layers:
        if shutil.disk_usage(root).free < layer_bytes + 1024 ** 3:
            raise RuntimeError("Insufficient free disk for all float32 layers plus 1 GiB output reserve")
        (root / "layers").mkdir()
    write_json(root / "manifest.json", rows)
    source_paths = [Path(__file__).resolve(), Path(multilayer.__file__),
                    Path(multilayer_controls.__file__),
                    Path(__file__).resolve().parents[1] / "src/ics/methods/layer_extract.py"]
    protocol = dict(
        state="INFERENCE_ONLY", config=multilayer.CONFIG, delta_control=multilayer_controls.CONFIG,
        source_sha256={str(path): sha(path) for path in source_paths},
        source_manifest_sha256=sha(args.manifest), native_basis_sha256=sha(manifest["projection_basis"]),
        model_resolution=1024, encoder_instances=1, ordinary_pair_forwards_per_episode=1,
        first_episode_extra_pair_forwards=2, first_episode_total_pair_forwards=3,
        setup_black_forwards=1, setup_svd_count=3,
        query_gt_usage="none; separate score_forward_run.py only after cohort sealing",
        development_only=True, fixed_arms=["native", "multilayer", "multilayer.control", "multilayer.delta.control"],
        action_diagnostics=["multilayer.delete_only", "multilayer.add_only"],
        save_layers=args.save_layers, layer_dtype="float32", estimated_uncompressed_layer_bytes=layer_bytes,
        runtime_state="frozen v1 recipes; one new predetermined delta-only control",
    )
    write_json(root / "protocol.json", protocol)
    data, ann = Path(manifest["data_root"]), Path(manifest["annotation_root"])
    host = build_host(manifest, args.device, args.foris_root)
    if len([module for module in host.modules() if isinstance(module, TimmDINOv3)]) != 1:
        raise RuntimeError("Expected one shared frozen encoder")
    bases, setup = multilayer_controls.prepare_bases(host)
    torch.save(bases, root / "arm_bases.pt")
    audits = {"setup": setup, "episodes": {}, "arm_bases_sha256": sha(root / "arm_bases.pt")}
    prediction_hashes, layer_hashes = {}, {}
    start = time.monotonic()
    with torch.inference_mode(), (root / "episodes.jsonl").open("w") as stream:
        for index, row in enumerate(rows):
            with Image.open(data / row["support"]) as image:
                support = image.convert("RGB")
            with Image.open(data / row["query"]) as image:
                query = image.convert("RGB")
            with Image.open(ann / Path(row["support"]).with_suffix(".png")) as image:
                support_mask = torch.from_numpy((np.asarray(image) == row["c"] + 1).copy())
            timings, receipts, masks, fields = {}, {}, {}, {}
            public_native, public_score = None, None
            if index == 0:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                public, observed, _, _ = run_foris(host, support, support_mask, query)
                public_native = public.cpu().numpy().astype(bool)
                public_score = observed["score"].float().cpu().numpy()
                timings["public_native_first_episode"] = dict(seconds=time.monotonic() - before,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated())
                del public, observed
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            before = time.monotonic()
            extras = extract_pair(host, support, support_mask, query, verify_native=index == 0)
            extras["arm_bases"] = bases
            timings["shared_extraction"] = dict(seconds=time.monotonic() - before,
                                                peak_cuda_bytes=torch.cuda.max_memory_allocated())
            receipts["extraction"] = extras["layer_metadata"]
            q, r = extras["q_layers"]["24"], extras["r_layers"]["24"]
            dummy = np.zeros((64, 64), np.float32)
            for arm, fn in (("native", multilayer.native), ("multilayer", multilayer.predict),
                            ("multilayer.control", multilayer.control),
                            ("multilayer.delta.control", multilayer_controls.predict)):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                field, receipt = fn(q, r, extras["cov"], dummy, device=str(host.device), extras=extras)
                prediction = multilayer.finalize(field, extras)
                if prediction.shape != (1024, 1024):
                    raise RuntimeError("Expected a complete native 1024 mask")
                receipt["native_finalization_pending"] = False
                timings[arm] = dict(seconds=time.monotonic() - before,
                                   peak_cuda_bytes=torch.cuda.max_memory_allocated())
                receipts[arm], masks[arm], fields[arm] = receipt, prediction, field
            if index == 0:
                native_audit = dict(
                    full_mask_exact=bool(np.array_equal(public_native, masks["native"])),
                    full_score_max_abs=float(np.max(np.abs(public_score - fields["native"]))),
                    full_score_close=bool(np.allclose(public_score, fields["native"], atol=1e-5, rtol=1e-5)))
                audits["first_public_vs_tapped_native"] = native_audit
                if not native_audit["full_mask_exact"] or not native_audit["full_score_close"]:
                    write_json(root / "audit_failure.json", native_audit)
                    raise RuntimeError("Tapped native failed independent public-native replay")
            # Fixed post-hoc action diagnostics from the frozen v1 mask. They
            # never route or select the primary method, and use no GT.
            masks["multilayer.delete_only"] = masks["native"] & masks["multilayer"]
            masks["multilayer.add_only"] = masks["native"] | masks["multilayer"]
            key = row["key"]
            if args.save_layers:
                before_save = time.monotonic()
                layer_path = root / "layers" / f"{key}.npz"
                layer_arrays = {f"{role}{layer}": extras[f"{role}_layers"][layer].numpy()
                                for role in ("q", "r") for layer in ("16", "24")}
                if any(value.dtype != np.float32 for value in layer_arrays.values()):
                    raise ValueError("Layer archive must preserve raw float32 activations")
                with layer_path.with_suffix(".npz.tmp").open("wb") as f:
                    np.savez(f, **layer_arrays)
                layer_path.with_suffix(".npz.tmp").replace(layer_path)
                layer_hashes[key] = sha(layer_path)
                timings["save_layers"] = dict(seconds=time.monotonic() - before_save,
                                              bytes=layer_path.stat().st_size)
                del layer_arrays
            pred_path = root / "predictions" / f"{key}.npz"
            np.savez_compressed(pred_path, **{arm: np.packbits(mask) for arm, mask in masks.items()})
            np.savez_compressed(root / "packets" / f"{key}.npz", **fields)
            prediction_hashes[key] = sha(pred_path)
            audits["episodes"][key] = dict(timings=timings, receipts=receipts)
            write_json(root / "audits.json", audits)
            stream.write(json.dumps(dict(row, timings=timings, receipts=receipts)) + "\n")
            stream.flush()
            print(json.dumps(dict(completed=index + 1, total=len(rows), key=key, timings=timings)), flush=True)
            del extras, q, r
    write_json(root / "sealed.json", dict(
        state="ALL_PREDICTIONS_SEALED", manifest_sha256=sha(root / "manifest.json"),
        protocol_sha256=sha(root / "protocol.json"), predictions=prediction_hashes,
        query_labels_opened=False, layers=layer_hashes, arm_bases_sha256=audits["arm_bases_sha256"]))
    write_json(root / "completion.json", dict(state="COMPLETED", episodes=len(rows),
                                              elapsed_seconds=time.monotonic() - start))


if __name__ == "__main__":
    main()
