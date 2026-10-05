#!/usr/bin/env python3
"""Complete public FoRIS arms with a single shared frozen DINOv3 model.

Requires existing RGB, masks, model weights, native basis and public FoRIS code.
No downloads; one sequential paired forward per arm. Native/identity and native/zero-additive-mask equality are
checked on the first episode, including the reference feature effect. This runner never opens query truth. Score only after it exits and seals all masks.
"""
import argparse
import hashlib
from contextlib import contextmanager
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@contextmanager
def paired_input_audit(vit, host):
    """Verify batch order against live host images before patch embedding."""
    import torch
    receipt = dict(calls=0)

    def audit(module, args):
        x = args[0]
        if x.ndim != 4 or x.shape[0] != 2:
            raise RuntimeError("FoRIS must encode [reference, query] together")
        ref, query = host._ref_images[0], host._tgt_image
        if ref.ndim == 4:
            ref = ref[0]
        if query.ndim == 4:
            query = query[0]
        pair = torch.stack((ref, query)).to(x)
        if pair.shape != x.shape:
            raise RuntimeError("Host image shapes do not match paired encoder inputs")
        if torch.allclose(pair, x, atol=1e-6, rtol=1e-6):
            transform = "identity"
        else:
            mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
            std = x.new_tensor([.229, .224, .225])[None, :, None, None]
            if not torch.allclose((pair - mean) / std, x, atol=1e-6, rtol=1e-6):
                raise RuntimeError("Could not verify [reference, query] input order against host pixels")
            transform = "imagenet_normalization"
        receipt.update(calls=receipt["calls"] + 1, order="reference,query", transform=transform,
                       shape=list(x.shape))
    hook = vit.patch_embed.register_forward_pre_hook(audit)
    try:
        yield receipt
    finally:
        hook.remove()



def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--foris-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--include-multilayer", action="store_true", help="Run D arms sequentially using the same model")
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from ics.data import TimmDINOv3
    from ics.foris import build_host, run_foris
    from ics.methods.intervention import CONFIG, attention_intervention

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    manifest = json.loads(Path(args.manifest).read_text())
    episodes = manifest["episodes"][:args.limit]
    if not episodes:
        parser.error("Manifest has no selected episodes")
    from ics.experiment import load_rows
    episodes = load_rows(args.manifest)[:args.limit]
    keys = [(row["fold"], row["e"], row["c"]) for row in episodes]
    if len(set(keys)) != len(keys):
        parser.error("Duplicate episode IDs in selected manifest")
    root = Path(args.out)
    if root.exists():
        parser.error("Output directory exists; choose a fresh destination to preserve predictions")
    root.mkdir(parents=True)
    (root / "packets").mkdir()
    (root / "predictions").mkdir()
    (root / "manifest.json").write_text(json.dumps(episodes, indent=2) + "\n")
    data, ann = Path(manifest["data_root"]), Path(manifest["annotation_root"])
    host = build_host(manifest, args.device, args.foris_root)
    encoders = [module for module in host.modules() if isinstance(module, TimmDINOv3)]
    if len(encoders) != 1:
        raise RuntimeError(f"Expected exactly one shared frozen encoder, found {len(encoders)}")
    vit = encoders[0].m
    setup_audit = {}
    if args.include_multilayer:
        from ics.methods.layer_extract import prepare_bases, extract_pair
        from ics.methods import multilayer
        bases, setup_audit = prepare_bases(host)
    (root / "protocol.json").write_text(json.dumps(dict(
        state="INFERENCE_ONLY", config=CONFIG, manifest=str(Path(args.manifest).resolve()),
        annotation_root=str(ann.resolve()),
        encoder_instances=1, model_resolution=1024, query_gt_usage="never read by inference; separate score_forward_run.py after process exit",
        development_only=True, native_basis=manifest.get("projection_basis"),
        added_inputs=False, per_arm_paired_forwards=1, identity_audit_episodes=1,
        masked_identity_audit_episodes=1,
        first_episode_additional_audit_pair_forwards=2 + int(args.include_multilayer),
        ordinary_pair_forwards_per_episode=3 + int(args.include_multilayer),
        first_episode_total_pair_forwards=5 + 2 * int(args.include_multilayer),
        multilayer_setup_black_forwards=int(args.include_multilayer),
        native_basis_loaded=bool(manifest.get("projection_basis")),
        reference_mask_path_audit="first episode raw reference features compared with native; reference logits receive zero bias",
        controls=["native", "key_only_equal_rms"], include_multilayer=args.include_multilayer,
        multilayer_shared_pair_forwards_per_episode=int(args.include_multilayer),
        multilayer_config=multilayer.CONFIG if args.include_multilayer else None,
    ), indent=2) + "\n")
    started = time.monotonic()
    prediction_hashes = {}
    all_audits = {"setup": setup_audit, "episodes": {}}
    with torch.inference_mode(), (root / "episodes.jsonl").open("w") as stream:
        for index, episode in enumerate(episodes):
            with Image.open(data / episode["support"]) as image:
                support = image.convert("RGB")
            with Image.open(data / episode["query"]) as image:
                query = image.convert("RGB")
            with Image.open(ann / Path(episode["support"]).with_suffix(".png")) as image:
                support_mask = torch.from_numpy((np.asarray(image) == episode["c"] + 1).copy())
            predictions, fields, timings, receipts = {}, {}, {}, {}
            arms = ["native"] + (["identity", "masked_identity"] if index == 0 else []) + ["control", "intervention"]
            reference_features = {}
            for arm in arms:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                with paired_input_audit(vit, host) as pair_receipt:
                    if arm == "native":
                        pred, observed, _, _ = run_foris(host, support, support_mask, query)
                        receipt = {}
                    else:
                        with attention_intervention(vit, lambda: host._ref_masks[0], arm=arm) as receipt:
                            pred, observed, _, _ = run_foris(host, support, support_mask, query)
                        if receipt["attention_calls"] != 1 or receipt["sdpa_calls"] != 1:
                            raise RuntimeError("Expected exactly one modified block execution per arm")
                if pair_receipt["calls"] != 1:
                    raise RuntimeError("Expected exactly one paired encoder execution per arm")
                if args.device == "cuda":
                    torch.cuda.synchronize()
                timings[arm] = dict(seconds=time.monotonic() - before,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
                predictions[arm] = pred.cpu().numpy().astype(bool)
                fields[arm] = observed["score"].float().cpu().numpy()
                receipts[arm] = dict(attention=dict(receipt), paired_input=dict(pair_receipt))
                if index == 0:
                    raw = observed["raw"]
                    if raw.ndim == 5 and raw.shape[:2] == (1, 2):
                        raw_ref = raw[0, 0]
                    elif raw.ndim == 4 and raw.shape[0] == 2:
                        raw_ref = raw[0]
                    else:
                        raise RuntimeError(f"Cannot audit raw reference feature layout: {tuple(raw.shape)}")
                    reference_features[arm] = raw_ref.float().cpu().numpy().copy()
                if pred.shape != (1024, 1024):
                    raise RuntimeError(f"Expected complete 1024 mask, got {tuple(pred.shape)}")
            if index == 0:
                first_audit = {}
                for audit_arm in ("identity", "masked_identity"):
                    ref_delta = reference_features[audit_arm] - reference_features["native"]
                    record = dict(
                        full_mask_exact=bool(np.array_equal(predictions[audit_arm], predictions["native"])),
                        full_score_close=bool(np.allclose(fields[audit_arm], fields["native"], atol=1e-5, rtol=1e-5)),
                        full_score_max_abs=float(np.max(np.abs(fields[audit_arm] - fields["native"]))),
                        reference_exact=bool(np.array_equal(reference_features[audit_arm], reference_features["native"])),
                        reference_close=bool(np.allclose(reference_features[audit_arm], reference_features["native"], atol=1e-5, rtol=1e-5)),
                        reference_max_abs=float(np.max(np.abs(ref_delta))),
                        reference_directly_biased=False,
                        sdpa_path="native mask" if audit_arm == "identity" else "full zero additive mask, same shape as candidate",
                        numerical_tolerance=dict(atol=1e-5, rtol=1e-5))
                    first_audit[audit_arm] = record
                for arm in ("control", "intervention"):
                    delta = reference_features[arm] - reference_features["masked_identity"]
                    first_audit[arm + ".reference_vs_masked_identity"] = dict(
                        exact=bool(np.array_equal(reference_features[arm], reference_features["masked_identity"])),
                        close=bool(np.allclose(reference_features[arm], reference_features["masked_identity"], atol=1e-5, rtol=1e-5)),
                        max_abs=float(np.max(np.abs(delta))), reference_directly_biased=False)
                all_audits["first_episode_mask_path"] = first_audit
                all_audits["first_episode_pre_multilayer_timings"] = dict(timings)
                # Persist the kernel audit even when validation prevents a completed cohort seal.
                (root / "audits.json").write_text(json.dumps(all_audits, indent=2) + "\n")
                failed = [arm for arm in ("identity", "masked_identity")
                          if not all(first_audit[arm][key] for key in
                                     ("full_mask_exact", "full_score_close", "reference_close"))]
                failed += [arm for arm in ("control", "intervention")
                           if not first_audit[arm + ".reference_vs_masked_identity"]["close"]]
                if failed:
                    (root / "audit_failure.json").write_text(json.dumps(dict(
                        state="SDPA_KERNEL_OR_REPLAY_CONFOUND", failed_arms=failed,
                        inference_sealed=False, details=first_audit), indent=2) + "\n")
                    raise RuntimeError("SDPA masked-path/reference replay failed; kernel confound recorded, no valid candidate comparison")
                del reference_features
            if args.include_multilayer:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                dextra = extract_pair(host, support, support_mask, query, verify_native=(index == 0))
                dextra["arm_bases"] = bases
                receipts["multilayer.extraction"] = dict(dextra["layer_metadata"])
                dq, dr = dextra["q_layers"]["24"], dextra["r_layers"]["24"]
                dummy = np.zeros((64, 64), np.float32)
                timings["multilayer.shared_forward"] = dict(seconds=time.monotonic() - before,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
                for arm, fn in (("multilayer.native_audit", multilayer.native),
                                ("multilayer.control", multilayer.control), ("multilayer", multilayer.predict)):
                    before = time.monotonic()
                    field, info = fn(dq, dr, dextra["cov"], dummy, device=str(host.device), extras=dextra)
                    pred = multilayer.finalize(field, dextra)
                    info["native_finalization_pending"] = False
                    if pred.shape != (1024, 1024):
                        raise RuntimeError("Multilayer finalizer did not produce a complete 1024 mask")
                    if arm == "multilayer.native_audit":
                        if not np.array_equal(pred, predictions["native"]):
                            raise RuntimeError("Multilayer native readout differs from public native mask")
                    else:
                        predictions[arm] = pred.astype(bool)
                        fields[arm] = field
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    timings[arm] = dict(seconds=time.monotonic() - before)
                    receipts[arm] = info
                del dextra, dq, dr
            name = "%d_%d_%d.npz" % (episode["fold"], episode["e"], episode["c"])
            packet = {}
            for arm, pred in predictions.items():
                packet[arm] = np.packbits(pred)
                packet[arm + "_score"] = fields[arm].astype(np.float32)
            # Pure inference: this process never opens any query annotation.
            np.savez_compressed(root / "packets" / name, **packet)
            canonical_names = {"native": "native", "control": "intervention.control",
                               "intervention": "intervention", "multilayer": "multilayer",
                               "multilayer.control": "multilayer.control"}
            canonical = {canonical_names[arm]: np.packbits(pred) for arm, pred in predictions.items()
                         if arm in canonical_names}
            np.savez_compressed(root / "predictions" / name, **canonical)
            prediction_hashes[episode["key"]] = file_sha(root / "predictions" / name)
            all_audits["episodes"][episode["key"]] = dict(timings=timings, receipts=receipts)
            row = dict(episode, packet=name, timings=timings, receipts=receipts)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(dict(completed=index + 1, total=len(episodes), timings=timings)), flush=True)
    (root / "audits.json").write_text(json.dumps(all_audits, indent=2) + "\n")
    (root / "sealed.json").write_text(json.dumps(dict(
        state="ALL_PREDICTIONS_SEALED", manifest_sha256=file_sha(root / "manifest.json"),
        protocol_sha256=file_sha(root / "protocol.json"), predictions=prediction_hashes,
        query_labels_opened=False), indent=2) + "\n")
    (root / "completion.json").write_text(json.dumps(dict(state="COMPLETED", episodes=len(episodes),
        elapsed_seconds=time.monotonic() - started), indent=2) + "\n")


if __name__ == "__main__":
    main()
