#!/usr/bin/env python3
"""One DINO producer and spawn-isolated complete FoRIS/CRF readout workers.

Fixed reference-value five-arm suite, identical to run_reference_value.py.
Default mini50; use --limit 3 --verify-serial 3 for actual same-case parity.
--compare-run checks all complete masks/pre-masks against an existing sealed
run_reference_value run. This runner never opens query annotations or scores.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

# Spawn imports this script before its worker target. Set limits before NumPy,
# torch or sklearn imports, including in every spawned process.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_intervention import file_sha, paired_input_audit


NAMES = dict(native="native", key_only="reference_value.key_only.control",
             control="reference_value.control", reference_value="reference_value",
             foreground_only="reference_value.foreground_only")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def arm_context(vit, host, arm, capture=None):
    from ics.methods.intervention import attention_intervention
    from ics.methods.reference_value import ReferenceValueConfig, reference_value
    if arm == "plain_audit":
        return nullcontext({})
    if arm == "key_only":
        return attention_intervention(vit, lambda: host._ref_masks[0], arm="control")
    config = ReferenceValueConfig()
    value_arm = "identity" if arm == "native" else arm
    if arm == "zero_audit":
        value_arm, config = "reference_value", replace(config, residual_ratio=0.)
    return reference_value(vit, lambda: host._ref_masks[0], arm=value_arm,
                           config=config, capture=capture)


def comparison(actual, expected):
    import numpy as np
    mask_errors = int(np.count_nonzero(np.unpackbits(actual["prediction"])
                                      != np.unpackbits(expected["prediction"])))
    pre_errors = int(np.count_nonzero(np.unpackbits(actual["pre"])
                                     != np.unpackbits(expected["pre"])))
    if actual["score"].shape != expected["score"].shape or actual["pre_shape"] != expected["pre_shape"]:
        raise ValueError("Comparison output shape changed")
    return dict(mask_mismatched_pixels=mask_errors, pre_mismatched_pixels=pre_errors,
                score_exact=bool(np.array_equal(actual["score"], expected["score"])),
                score_close=bool(np.allclose(actual["score"], expected["score"], atol=1e-5, rtol=1e-5)),
                score_max_abs=float(np.max(np.abs(actual["score"] - expected["score"]))),
                passed=mask_errors == 0 and pre_errors == 0 and
                    bool(np.allclose(actual["score"], expected["score"], atol=1e-5, rtol=1e-5)))


def self_check():
    """CPU-only IPC-input contracts; no efficacy/speed or real CRF claim."""
    import torch
    from ics.parallel_readout import ForbiddenEncoder, cached_features, validate_job

    class Host:
        def _extract_features(self, images):
            raise RuntimeError("Original method must be restored after the scope")

    host = Host()
    ref = torch.zeros(1, 3, 1024, 1024)
    target = torch.ones(3, 1024, 1024)
    # Deliberately non-contiguous map; contract must retain its physical layout.
    fmap = torch.arange(1024, dtype=torch.float32)[None, None, :, None, None].expand(1, 2, 1024, 64, 64)
    job = dict(fmaps=fmap, ref_images=ref, ref_masks=torch.zeros(1, 1024, 1024, dtype=torch.bool),
               target=target, fmap_stride=list(fmap.stride()))
    pair = torch.cat((ref, target[None]), dim=0)[None]
    with cached_features(host, job) as receipt:
        assert host._extract_features(pair) is fmap
        assert receipt["calls"] == 1 and receipt["encoder_forwards"] == 0
    assert "_extract_features" not in host.__dict__
    try:
        with cached_features(host, job):
            host._extract_features(pair.flip(1))
    except ValueError:
        pass
    else:
        raise AssertionError("Swapped paired input was not rejected")
    assert "_extract_features" not in host.__dict__
    try:
        validate_job(dict(job, fmap_stride=[1, 1, 1, 1, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("Changed stride was not rejected")
    try:
        ForbiddenEncoder().get_intermediate_layers(pair)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Readout encoder forward was allowed")
    return dict(state="CPU_INPUT_CONTRACTS_PASSED", actual_cuda_ipc_or_real_image_test=False,
                checks=["exact fmap object and noncontiguous stride", "actual reference/query order",
                        "exception restoration", "stride-change rejection", "encoder forward prohibited"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    parser.add_argument("--foris-root")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, choices=(3, 4), default=4)
    parser.add_argument("--max-inflight", type=int, help="Bound live CUDA IPC jobs; default 2 x workers")
    parser.add_argument("--verify-serial", type=int, default=1,
                        help="First N episodes: replay every arm through uncached public FoRIS")
    parser.add_argument("--compare-run", type=Path,
                        help="Existing sealed run_reference_value directory for per-arm bitwise mask parity")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if not args.manifest or not args.out:
        parser.error("--manifest and --out are required")
    if args.limit < 1 or args.verify_serial < 0:
        parser.error("--limit must be positive and --verify-serial nonnegative")
    max_inflight = args.workers * 2 if args.max_inflight is None else args.max_inflight
    if not args.workers <= max_inflight <= 16:
        parser.error("--max-inflight must be between workers and 16")

    import numpy as np
    import torch
    from PIL import Image
    from ics.data import TimmDINOv3
    from ics.experiment import load_rows
    from ics.foris import build_host, run_foris
    from ics.methods.intervention import CONFIG as KEY_CONFIG
    from ics.methods.reference_value import CONFIG
    from ics.parallel_readout import ReadoutPool, clear_episode, configure_threads

    configure_threads(2)
    if not torch.cuda.is_available():
        parser.error("Real inference requires CUDA; --self-check is CPU-only")
    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest.get("projection_basis"):
        parser.error("Manifest must name the existing native projection_basis")
    rows = load_rows(args.manifest)[:args.limit]
    root = Path(args.out)
    if root.exists():
        parser.error("Output exists; choose a fresh directory")
    existing_rows, existing_seal = {}, None
    if args.compare_run:
        existing_seal = json.loads((args.compare_run / "sealed.json").read_text())
        if existing_seal.get("state") != "ALL_PREDICTIONS_SEALED":
            parser.error("Comparison run is not sealed")
        if file_sha(args.compare_run / "manifest.json") != existing_seal["manifest_sha256"]:
            parser.error("Comparison manifest changed after sealing")
        if file_sha(args.compare_run / "protocol.json") != existing_seal["protocol_sha256"]:
            parser.error("Comparison protocol changed after sealing")
        existing_rows = {r["key"]: r for r in json.loads((args.compare_run / "manifest.json").read_text())}
        for row in rows:
            if row["key"] not in existing_rows or any(row[k] != existing_rows[row["key"]][k]
                    for k in ("fold", "e", "c", "support", "query")):
                parser.error(f"Comparison episode identity differs: {row['key']}")
    root.mkdir(parents=True)
    for name in ("predictions", "packets"):
        (root / name).mkdir()
    write_json(root / "manifest.json", rows)
    sources = [Path(__file__), Path(__file__).parents[1] / "src/ics/parallel_readout.py",
               Path(__file__).parents[1] / "src/ics/foris.py",
               Path(__file__).parents[1] / "src/ics/methods/reference_value.py",
               Path(__file__).parents[1] / "src/ics/methods/intervention.py"]
    source_root = Path(args.foris_root or manifest["foris_root"])
    sources += [source_root / name for name in
                ("models/foris.py", "utils/refinement.py", "utils/clustering.py", "utils/data.py")]
    protocol = dict(state="INFERENCE_ONLY", config=CONFIG, key_only_config=KEY_CONFIG,
        manifest=str(Path(args.manifest).resolve()), manifest_sha256=file_sha(args.manifest),
        source_sha256={str(p.resolve()): file_sha(p) for p in sources},
        dataset="COCO-20i", split="existing exposed DEV", seed=0, episodes=len(rows),
        model_resolution=1024, finalization="unchanged source FoRIS.segment/predict and private native CRF",
        encoder_instances=1, readout_workers=args.workers, readout_encoder_instances=0,
        producer_cpu_threads=2, worker_cpu_threads=2, total_configured_cpu_threads=2*(args.workers+1),
        multiprocessing_start_method="spawn", feature_transport="CUDA IPC, original FP32 shape/strides",
        max_inflight_jobs=max_inflight, native_basis=manifest["projection_basis"], frozen_weights=True,
        setup_black_forwards=0, worker_encoder_forwards=0, worker_dummy_forwards=0,
        fixed_arms=list(NAMES.values()), query_gt_usage="never opened; score only after sealing",
        development_only=True, ordinary_pair_forwards_per_episode=5,
        first_episode_additional_audit_pair_forwards=2,
        serial_verification_episodes=min(args.verify_serial, len(rows)),
        serial_verification_extra_pair_forwards=5*min(args.verify_serial, len(rows))+2*int(args.verify_serial>0),
        compare_run=str(args.compare_run.resolve()) if args.compare_run else None,
        data_root=manifest["data_root"], annotation_root=manifest["annotation_root"])
    write_json(root / "protocol.json", protocol)
    setup_started = time.monotonic()
    pool = ReadoutPool(manifest, device="cuda", foris_root=args.foris_root, workers=args.workers)
    audits, hashes, pending_episodes = {}, {}, {}
    completed, max_observed_inflight = 0, 0
    wall_started = None
    try:
        ready = pool.start()
        host = build_host(manifest, "cuda", args.foris_root)
        encoders = [m for m in host.modules() if isinstance(m, TimmDINOv3)]
        if len(encoders) != 1:
            raise RuntimeError("Producer requires exactly one real frozen encoder")
        vit = encoders[0].m
        torch.cuda.synchronize()
        write_json(root / "setup.json", dict(workers=ready, producer_encoder_instances=1,
                   seconds=time.monotonic()-setup_started))
        wall_started = time.monotonic()
        audit_seconds = 0.
        stream = (root / "episodes.jsonl").open("w")

        def accept(result):
            nonlocal completed
            record = pending_episodes[result["key"]]
            arm = result["arm"]
            record["outputs"][arm] = result
            record["timings"][arm].update(result["timings"], worker=result["worker"])
            record["receipts"][arm]["readout"] = result["receipt"]
            if arm in record["serial"]:
                check = comparison(result, record["serial"].pop(arm))
                record["parity"][arm] = check
                if not check["passed"]:
                    write_json(root / "audit_failure.json", dict(kind="public_serial_parity",
                               key=result["key"], arm=arm, check=check))
                    raise RuntimeError(f"Public serial parity failed: {result['key']} {arm}")
            if len(record["outputs"]) != len(record["arms"]):
                return
            outputs, key = record["outputs"], result["key"]
            if "plain_audit" in outputs:
                for name in ("plain_audit", "zero_audit"):
                    check = comparison(outputs[name], outputs["native"])
                    record["parity"][name + ".vs_native"] = check
                    if not check["passed"]:
                        raise RuntimeError(f"Native/identity/zero audit failed: {name}")
            if args.compare_run:
                prediction_path = args.compare_run / "predictions" / f"{key}.npz"
                if file_sha(prediction_path) != existing_seal["predictions"][key]:
                    raise ValueError("Comparison predictions changed after sealing")
                with np.load(prediction_path, allow_pickle=False) as old_masks, np.load(
                        args.compare_run / "packets" / f"{key}.npz", allow_pickle=False) as old_packet:
                    for name, public_name in NAMES.items():
                        expected = dict(prediction=old_masks[public_name], score=old_packet[name+"_score"],
                                        pre=old_packet[name+"_pre"],
                                        pre_shape=old_packet[name+"_pre_shape"].tolist())
                        check = comparison(outputs[name], expected)
                        record["parity"][name + ".vs_existing_run"] = check
                        if not check["passed"]:
                            write_json(root / "audit_failure.json", dict(kind="existing_run_parity", key=key,
                                       arm=name, check=check))
                            raise RuntimeError(f"Existing-run parity failed: {key} {name}")
            np.savez_compressed(root / "predictions" / f"{key}.npz",
                                **{name: outputs[arm]["prediction"] for arm, name in NAMES.items()})
            packet = {}
            for name, value in outputs.items():
                packet.update({name:value["prediction"], name+"_score":value["score"],
                               name+"_pre":value["pre"],
                               name+"_pre_shape":np.asarray(value["pre_shape"], dtype=np.int64)})
            packet.update(record["diagnostics"])
            np.savez_compressed(root / "packets" / f"{key}.npz", **packet)
            hashes[key] = file_sha(root / "predictions" / f"{key}.npz")
            record["timings"]["episode_pipeline_latency"] = dict(seconds=time.monotonic()-record["started"])
            audits[key] = dict(timings=record["timings"], receipts=record["receipts"],
                               parity=record["parity"], producer_reference_parity=record["reference_checks"])
            stream.write(json.dumps(dict(record["row"], **audits[key])) + "\n")
            stream.flush()
            completed += 1
            print(json.dumps(dict(completed=completed, total=len(rows), key=key,
                  pending_jobs=len(pool.pending), timings=record["timings"])), flush=True)
            del pending_episodes[key]

        with torch.inference_mode():
            for index, row in enumerate(rows):
                with Image.open(Path(manifest["data_root"]) / row["support"]) as image:
                    support = image.convert("RGB")
                with Image.open(Path(manifest["data_root"]) / row["query"]) as image:
                    query = image.convert("RGB")
                with Image.open(Path(manifest["annotation_root"]) / Path(row["support"]).with_suffix(".png")) as image:
                    support_mask = torch.from_numpy((np.asarray(image) == row["c"] + 1).copy())
                arms = ["native"] + (["plain_audit", "zero_audit"] if index == 0 else []) + list(NAMES)[1:]
                record = dict(row=row, arms=arms, outputs={}, serial={}, parity={}, diagnostics={},
                              timings={}, receipts={}, reference_checks={}, started=time.monotonic())
                pending_episodes[row["key"]] = record
                if index < args.verify_serial:
                    before = time.monotonic()
                    for arm in arms:
                        with paired_input_audit(vit, host), arm_context(vit, host, arm):
                            pred, observed, _, _ = run_foris(host, support, support_mask, query)
                        pre = observed["pre"].bool().cpu().numpy()
                        record["serial"][arm] = dict(prediction=np.packbits(pred.cpu().numpy()),
                            score=observed["score"].float().cpu().numpy().copy(),
                            pre=np.packbits(pre), pre_shape=list(pre.shape))
                        del pred, observed
                    audit_elapsed = time.monotonic()-before
                    record["timings"]["public_serial_verification"] = dict(seconds=audit_elapsed)
                    audit_seconds += audit_elapsed
                host.set_reference(support, support_mask)
                host.set_target(query)
                ref_images, ref_masks, target = host._ref_images, host._ref_masks, host._tgt_image
                orig_target_size = host._orig_tgt_size
                # The exact source predict stacking/order; no per-image forwards.
                images = torch.cat((ref_images, target[None]), dim=0)[None]
                baseline, native_ref = None, None
                try:
                    for arm in arms:
                        while len(pool.pending) >= max_inflight:
                            accept(pool.collect())
                        torch.cuda.synchronize()
                        before = time.monotonic()
                        capture = {}
                        with paired_input_audit(vit, host) as pair, arm_context(vit, host, arm, capture) as receipt:
                            fmap = host._extract_features(images)
                        torch.cuda.synchronize()
                        if pair["calls"] != 1 or (receipt and receipt["attention_calls"] != 1):
                            raise RuntimeError("Each actual arm must make exactly one paired encoder forward")
                        record["timings"][arm] = dict(seconds_encoder_forward=time.monotonic()-before)
                        record["receipts"][arm] = dict(attention=dict(receipt), paired_input=dict(pair),
                                                      raw_fmap_stride=list(fmap.stride()))
                        if index == 0:
                            raw_ref = fmap[0, 0].float().cpu().numpy().copy()
                            if arm == "native":
                                native_ref = raw_ref
                            else:
                                refcheck = dict(exact=bool(np.array_equal(raw_ref, native_ref)),
                                    close=bool(np.allclose(raw_ref, native_ref, atol=1e-5, rtol=1e-5)),
                                    max_abs=float(np.max(np.abs(raw_ref-native_ref))))
                                record["reference_checks"][arm] = refcheck
                                if not refcheck["close"]:
                                    raise RuntimeError(f"Reference feature parity failed: {arm}")
                        if arm == "native":
                            baseline = capture
                        elif capture:
                            if not torch.equal(capture["native_sdpa"], baseline["native_sdpa"]):
                                raise RuntimeError("Pre-injection native SDPA changed across arms")
                            diagnostics = record["diagnostics"]
                            diagnostics[arm+"_projected_attention_delta_norm"] = (
                                capture["attention_output"]-baseline["attention_output"]).norm(dim=-1).cpu().numpy()
                            diagnostics[arm+"_projected_attention_native_norm"] = baseline["attention_output"].norm(dim=-1).cpu().numpy()
                            for name in ("density_margin", "entropy_fg", "entropy_bg", "contrast_uniform_cosine"):
                                diagnostics[arm+"_"+name] = capture[name].cpu().numpy()
                            diagnostics[arm+"_injected_delta_norm"] = capture["injected_delta"].norm(dim=-1).cpu().numpy()
                            diagnostics[arm+"_native_sdpa_norm"] = capture["native_sdpa"].norm(dim=-1).cpu().numpy()
                        pool.submit(dict(id=row["key"]+":"+arm, key=row["key"], arm=arm,
                            fmaps=fmap, fmap_stride=list(fmap.stride()), ref_images=ref_images,
                            ref_masks=ref_masks, target=target, orig_target_size=orig_target_size))
                        max_observed_inflight = max(max_observed_inflight, len(pool.pending))
                        del fmap, capture
                        while True:
                            result = pool.collect(block=False)
                            if result is None:
                                break
                            accept(result)
                finally:
                    clear_episode(host)
                del baseline, native_ref, ref_images, ref_masks, target, images
            while pool.pending:
                accept(pool.collect())
        stream.close()
        if pending_episodes or completed != len(rows):
            raise RuntimeError("Incomplete parallel episode cohort")
        write_json(root / "audits.json", audits)
        write_json(root / "sealed.json", dict(state="ALL_PREDICTIONS_SEALED",
            manifest_sha256=file_sha(root / "manifest.json"), protocol_sha256=file_sha(root / "protocol.json"),
            predictions=hashes, query_labels_opened=False))
        elapsed = time.monotonic()-wall_started
        write_json(root / "completion.json", dict(state="COMPLETED", episodes=len(rows),
            elapsed_seconds=elapsed, serial_verification_seconds=audit_seconds,
            pipeline_elapsed_excluding_serial_verification=elapsed-audit_seconds,
            max_observed_inflight=max_observed_inflight,
            note="Subtracting verification wall time is diagnostic; use --verify-serial 0 for throughput timing"))
    except BaseException as error:
        write_json(root / "failure.json", dict(state="FAILED_UNSEALED", completed=completed, error=str(error)))
        raise
    finally:
        pool.close()


if __name__ == "__main__":
    main()
