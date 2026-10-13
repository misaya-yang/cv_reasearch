#!/usr/bin/env python3
"""Run a fixed GPU final-block task-context experiment, then score sealed outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "cv_data"
FEATURE = ASSETS / "a/task_attention_relation60_20261010"
PARENT = ASSETS / "a/reference_mask_task_trace_20261010"
DEFAULT_OUT = ASSETS / "a/task_context_compatibility120_20261012"
METHOD = ROOT / "src/ics/methods/task_context_compatibility.py"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import opus55_reference_mask_trace as base
from ics.methods import task_context_compatibility as method

ARMS = ("context", "linear_context", "null_context", "qk_density", "o24_density")
RENDERS = ("zero", "minmax")
torch.set_num_threads(2)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(out):
    if (out / "config.json").exists():
        verify(out)
        return
    complete = read(FEATURE / "X23_COMPLETE.json")
    parent_config = read(FEATURE / "config.json")
    pairs = read(FEATURE / "pairs.json")
    assert len(pairs) == 60 and len(complete["records"]) == 119
    assert set(p["dataset"] for p in pairs) == {"coco", "paco_part", "suim"}
    assert base.sha(FEATURE / "config.json") == complete["config_sha256"]
    assert base.sha(FEATURE / "rope64.npy") == complete["rope64_sha256"]
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("frozen", "fields", "predictions", "records"):
        (out / folder).mkdir(exist_ok=True)
    for source in (Path(__file__), METHOD):
        shutil.copyfile(source, out / "frozen" / source.name)
    base.write(out / "pairs.json", pairs)
    controls = {}
    for pair in pairs:
        path = PARENT / "traces" / (pair["id"] + ".npz")
        controls[pair["id"]] = {"path": str(path), "sha256": base.sha(path)}
    base.write(out / "external_controls.json", controls)
    config = {
        "kind": "frozen final-block conditional context compatibility; exploratory mechanism",
        "n_pairs": 60, "n_tasks": 120, "n_unique_x23": 119,
        "domains": ["coco", "paco_part", "suim"],
        "producer_sha256": base.sha(__file__), "method_sha256": base.sha(METHOD),
        "pairs_sha256": base.sha(out / "pairs.json"),
        "feature_seal_sha256": base.sha(FEATURE / "X23_COMPLETE.json"),
        "external_controls_sha256": base.sha(out / "external_controls.json"),
        "weights": parent_config["weights"],
        "weights_sha256": parent_config["weights_sha256"],
        "mix": method.MIX, "query_chunk": 256, "prefix": 5,
        "cross_relation": "actual pre-RoPE last-block Q/K; head dim64, scale1/8",
        "native_query_relation": "actual unmodified RoPE64 within-image attention",
        "reference_roles": "exact1024 pixel area coverage on64 grid; all4096 locations",
        "decoder": "mean unit-O24 context compatibility difference; no fitting",
        "arms": list(ARMS), "render": list(RENDERS),
        "primary": "context/zero", "null_seed": 0,
        "evaluation": "complete1024 masks, per-photo A/B mean; connected R/Q photo groups",
        "label_policy": "deny all parent query labels until all outputs sealed",
        "resource": {"device": "mps", "precision": "FP32", "cpu_threads": 2,
                     "mps_allocation_limit_bytes": 20_000_000_000},
        "no_new_photos": True, "new_full_encoder_calls": 0,
        "parameter_updates": 0,
        "caveat": "Reference memory injection is a counterfactual inference intervention, not native inter-image attention or demonstrated originality."
    }
    base.write(out / "config.json", config)
    base.write(out / "activity.json", {"state": "PREPARED", "total_pairs": 60})


def verify(out):
    config = read(out / "config.json")
    assert base.sha(__file__) == config["producer_sha256"]
    assert base.sha(out / "frozen" / Path(__file__).name) == config["producer_sha256"]
    assert base.sha(METHOD) == config["method_sha256"]
    assert base.sha(out / "frozen" / METHOD.name) == config["method_sha256"]
    assert base.sha(out / "pairs.json") == config["pairs_sha256"]
    assert base.sha(FEATURE / "X23_COMPLETE.json") == config["feature_seal_sha256"]
    assert base.sha(out / "external_controls.json") == config["external_controls_sha256"]
    assert config["mix"] == method.MIX == .5
    return config


def coverage(binding):
    assert base.sha(binding["path"]) == binding["file_sha256"]
    with Image.open(binding["path"]) as image:
        array = np.asarray(image.convert("L")) > 0
    dense = F.interpolate(torch.from_numpy(array.astype(np.float32))[None, None],
                          (1024, 1024), mode="nearest")
    return F.avg_pool2d(dense, 16)[0, 0].reshape(-1)


def load_raw(pair, device):
    arrays, receipts = {}, {}
    for role in ("reference", "query"):
        binding = pair["raw_binding"][role]
        assert base.sha(binding["entry"]) == binding["entry_sha256"]
        entry = read(binding["entry"])
        path = Path(binding["entry"]).parent / entry["file"]
        assert base.sha(path) == entry["file_sha256"]
        assert entry["input_tensor_hash"] == binding["input_tensor_hash"]
        with np.load(path, allow_pickle=False) as values:
            array = values["O/24"].copy()
        assert array.shape == (4096, 1024) and array.dtype == np.float32
        arrays[role] = torch.from_numpy(array).to(device)
        receipts[role] = {"key": binding["key"], "sha256": entry["file_sha256"]}
    return arrays, receipts


def dense_field(field, render):
    tensor = torch.from_numpy(np.asarray(field, dtype=np.float64).reshape(64, 64))
    if render == "minmax":
        span = float(tensor.max() - tensor.min())
        tensor = (tensor - tensor.min()) / span if span > 1e-12 else torch.zeros_like(tensor)
    return F.interpolate(tensor[None, None], (1024, 1024), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def pack_field(field, render):
    dense = dense_field(field, render)
    return np.packbits((dense > (.5 if render == "minmax" else 0.)).reshape(-1))


def model_digest(block, norm):
    digest = hashlib.sha256()
    for prefix, module in (("block", block), ("norm", norm)):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update((prefix + "/" + name).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def run(out, smoke=False):
    prepare(out)
    config = verify(out)
    if (out / "sealed.json").exists():
        return
    assert torch.backends.mps.is_available(), "No usable MPS device."
    assert base.sha(config["weights"]) == config["weights_sha256"]
    torch.manual_seed(0)
    torch.mps.set_per_process_memory_fraction(20e9 / torch.mps.recommended_max_memory())
    begin = time.perf_counter()
    from ics.data import TimmDINOv3
    encoder = TimmDINOv3(Path(config["weights"]).parent).eval().requires_grad_(False)
    model = encoder.m
    block = model.blocks[-1].to("mps")
    final_norm = model.norm.to("mps")
    assert len(model.blocks) == 24 and model.num_prefix_tokens == 5
    assert block.attn.num_heads == 16 and block.attn.gate is None
    assert all(not parameter.requires_grad for parameter in model.parameters())
    digest_before = model_digest(block, final_norm)
    fixtures = method.check_fixtures("mps")
    rope = torch.from_numpy(np.load(FEATURE / "rope64.npy")).to("mps")
    pairs = read(out / "pairs.json")
    complete = read(FEATURE / "X23_COMPLETE.json")
    features = {record["key"]: record for record in complete["records"]}
    permutation = torch.from_numpy(np.random.default_rng(0).permutation(4096)).to("mps")
    initialized = time.perf_counter() - begin
    records, checked_domains = [], set()
    max_driver = 0
    start = time.perf_counter()
    todo = pairs[:1] if smoke else pairs
    for index, pair in enumerate(todo):
        record_path = out / "records" / (pair["id"] + ".json")
        if record_path.exists():
            record = read(record_path)
            assert record["producer_sha256"] == config["producer_sha256"]
            assert record["method_sha256"] == config["method_sha256"]
            assert base.sha(out / "fields" / record["file"]) == record["fields_sha256"]
            assert base.sha(out / "predictions" / record["file"]) == record["predictions_sha256"]
            records.append(record)
            if record["native_replay_check"] is not None:
                checked_domains.add(pair["dataset"])
            continue
        tick = time.perf_counter()
        attempts, x, input_receipts = [], {}, {}
        with base.deny_labels(PARENT, attempts):
            for role in ("reference", "query"):
                key = pair["raw_binding"][role]["key"]
                source = features[key]
                path = FEATURE / "x23" / source["file"]
                assert base.sha(path) == source["sha256"]
                array = np.load(path, allow_pickle=False)
                assert array.shape == (4101, 1024) and array.dtype == np.float32
                x[role] = torch.from_numpy(array.copy())[None].to("mps")
                input_receipts[role] = {"key": key, "sha256": source["sha256"]}
            raw, raw_receipts = load_raw(pair, "mps")
            qr, kr, vr = method.qkv(block, x["reference"])
            qq, kq, vq = method.qkv(block, x["query"])
            del qr, kq, vq
            query_q = qq[0, :, 5:]
            reference_k = kr[0, :, 5:]
            reference_v = vr[0, :, 5:]
            native_context = {}
            def capture(module, args):
                native_context["raw"] = args[0].detach().clone()
            handle = block.attn.norm.register_forward_pre_hook(capture)
            try:
                native = final_norm(block(x["query"], rope=rope))
            finally:
                handle.remove()
            raw_context = native_context["raw"]
            assert raw_context.shape == (1, 4101, 1024)
            native_raw_error = float((native[0, 5:] - raw["query"]).abs().max())
            assert native_raw_error < 2e-6, native_raw_error
            native_raw_bit_exact = bool(torch.equal(native[0, 5:], raw["query"]))
            replay_check = None
            partial_calls = 1
            if pair["dataset"] not in checked_domains:
                replay = method.replay_with_context(block, final_norm, x["query"], raw_context)
                replay_error = float((replay - native).abs().max())
                assert replay_error < 2e-6, replay_error
                replay_check = {"max_error": replay_error, "bit_exact": bool(torch.equal(replay, native))}
                checked_domains.add(pair["dataset"])
                partial_calls += 1
                del replay
            unit_r = F.normalize(raw["reference"], dim=-1)
            unit_q = F.normalize(raw["query"], dim=-1)
            masses = {v: coverage(pair["variants"][v]).to("mps") for v in ("A", "B")}
            all_mass = {v: masses[v] for v in ("A", "B")}
            all_mass.update({"null" + v: masses[v][permutation] for v in ("A", "B")})
            memories = {v: [torch.empty((4096, 1024), device="mps") for _ in range(2)]
                        for v in all_mass}
            fields = {v: {arm: np.empty(4096, dtype=np.float32) for arm in ARMS}
                      for v in ("A", "B")}
            for left in range(0, 4096, config["query_chunk"]):
                right = min(left + config["query_chunk"], 4096)
                logits = (query_q[:, left:right] @ reference_k.transpose(-1, -2)) / 8.
                cosine = unit_q[left:right] @ unit_r.T / .07
                for variant, mass in all_mass.items():
                    foreground, background, evidence = method.role_context(logits, reference_v, mass)
                    memories[variant][0][left:right] = foreground.transpose(0, 1).reshape(right-left, 1024)
                    memories[variant][1][left:right] = background.transpose(0, 1).reshape(right-left, 1024)
                    if variant in ("A", "B"):
                        fields[variant]["qk_density"][left:right] = evidence.cpu().numpy()
                        fields[variant]["o24_density"][left:right] = method.cosine_role_density(cosine, mass).cpu().numpy()
                driver = int(torch.mps.driver_allocated_memory())
                max_driver = max(max_driver, driver)
                assert driver < 20_000_000_000
            for variant in ("A", "B"):
                foreground, background = [
                    method.blended_context(raw_context, memory)
                    for memory in memories[variant]
                ]
                fields[variant]["linear_context"][:] = method.compatibility(
                    raw_context[0, 5:], foreground[0, 5:], background[0, 5:]).cpu().numpy()
                zf = method.replay_with_context(block, final_norm, x["query"], foreground)
                zb = method.replay_with_context(block, final_norm, x["query"], background)
                fields[variant]["context"][:] = method.compatibility(
                    native[0, 5:], zf[0, 5:], zb[0, 5:]).cpu().numpy()
                partial_calls += 2
                del zf, zb, foreground, background
                null_foreground, null_background = [
                    method.blended_context(raw_context, memory)
                    for memory in memories["null" + variant]
                ]
                zf = method.replay_with_context(block, final_norm, x["query"], null_foreground)
                zb = method.replay_with_context(block, final_norm, x["query"], null_background)
                fields[variant]["null_context"][:] = method.compatibility(
                    native[0, 5:], zf[0, 5:], zb[0, 5:]).cpu().numpy()
                partial_calls += 2
                del zf, zb, null_foreground, null_background
            torch.mps.synchronize()
            values, predictions = {}, {}
            for variant in ("A", "B"):
                values[variant + "/reference_coverage"] = masses[variant].cpu().numpy().reshape(64, 64)
                for arm, field in fields[variant].items():
                    assert np.isfinite(field).all()
                    values[variant + "/" + arm] = field.reshape(64, 64)
                    for render in RENDERS:
                        predictions[variant + "/" + arm + "/" + render] = pack_field(field, render)
            assert not attempts
        file = pair["id"] + ".npz"
        field_path, prediction_path = out / "fields" / file, out / "predictions" / file
        np.savez_compressed(field_path, **values)
        np.savez_compressed(prediction_path, **predictions)
        record = {
            "id": pair["id"], "dataset": pair["dataset"], "file": file,
            "producer_sha256": config["producer_sha256"], "method_sha256": config["method_sha256"],
            "fields_sha256": base.sha(field_path), "predictions_sha256": base.sha(prediction_path),
            "x23": input_receipts, "o24": raw_receipts,
            "reference_mask_sha256": {v: pair["variants"][v]["file_sha256"] for v in ("A", "B")},
            "native_o24_max_error": native_raw_error, "native_o24_bit_exact": native_raw_bit_exact,
            "native_replay_check": replay_check,
            "partial_block_forward_calls": partial_calls, "new_full_encoder_calls": 0,
            "query_GT_access_attempts": attempts, "mps_driver_sampled_bytes": max_driver,
            "seconds": time.perf_counter() - tick,
            "field_ranges": {v: {arm: {"min": float(f.min()), "max": float(f.max()),
                                      "std": float(f.std())} for arm, f in fields[v].items()}
                             for v in ("A", "B")}
        }
        base.write(record_path, record)
        records.append(record)
        base.write(out / "activity.json", {
            "state": "SMOKE_INFERENCE" if smoke else "INFERENCE",
            "pid": os.getpid(), "completed_pairs": len(records), "total_pairs": len(todo),
            "seconds": time.perf_counter() - start, "sampled_mps_driver_bytes": max_driver})
        print(json.dumps({"pairs": len(records), "total": len(todo),
                          "seconds": round(time.perf_counter()-start, 2),
                          "partial_forwards": sum(r["partial_block_forward_calls"] for r in records),
                          "mps_driver_gib": round(max_driver/2**30, 3)}), flush=True)
        del x, raw, kr, vr, qq, query_q, reference_k, reference_v, native, raw_context
        del masses, all_mass, memories, fields, values, predictions, unit_r, unit_q
        del logits, cosine, evidence, native_context
        torch.mps.empty_cache()
    assert model_digest(block, final_norm) == digest_before
    assert all(not parameter.requires_grad for parameter in model.parameters())
    seal = {
        "state": "SMOKE_COMPLETE" if smoke else "ALL_120_TASKS_SEALED",
        "n_pairs": len(records), "n_tasks": 2*len(records), "n_masks": 20*len(records),
        "records": records, "config_sha256": base.sha(out / "config.json"),
        "feature_seal_sha256": config["feature_seal_sha256"],
        "seconds_this_run": time.perf_counter() - start,
        "sum_case_seconds": sum(r["seconds"] for r in records),
        "initialization_seconds": initialized,
        "partial_block_forward_calls": sum(r["partial_block_forward_calls"] for r in records),
        "sampled_mps_driver_bytes": max_driver,
        "parameter_digest_before_after": digest_before, "parameter_updates": 0,
        "new_full_encoder_calls": 0, "query_GT_access_attempts": 0, "fixtures": fixtures,
        "runtime": {"torch": str(torch.__version__), "device": "mps", "cpu_threads": 2}}
    if smoke:
        base.write(out / "smoke.json", seal)
    else:
        assert len(records) == 60
        base.write(out / "sealed.json", seal)
        base.write(out / "activity.json", {"state": "SEALED_READY_FOR_SCORING", "n_tasks": 120})


def photo_components(pairs):
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    for pair in pairs:
        keys = [pair["dataset"] + "/" + str(pair["row"].get(role+"_photo_id")
                or pair["row"][role+"_rgb_hash"]) for role in ("reference", "query")]
        parents[find(keys[0])] = find(keys[1])
    return {p["id"]: find(p["dataset"] + "/" + str(p["row"].get("query_photo_id")
                              or p["row"]["query_rgb_hash"])) for p in pairs}


def grouped_interval(rows, values, groups):
    values = np.asarray(values, dtype=np.float64)
    ids = sorted({groups[r["id"]] for r in rows})
    buckets = [np.array([i for i, row in enumerate(rows) if groups[row["id"]] == key]) for key in ids]
    rng = np.random.default_rng(0)
    boot = [float(values[np.concatenate([buckets[k] for k in rng.integers(0, len(ids), len(ids))])].mean())
            for _ in range(2000)]
    return {"mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "n_pairs": len(rows), "photo_components": len(ids)}


def pixel_metrics(prediction, gt, baseline):
    intersection = int(np.count_nonzero(prediction & gt))
    union = int(np.count_nonzero(prediction | gt))
    return {"I": intersection, "U": union, "iou": 100*intersection/max(union, 1),
            "FP": int(np.count_nonzero(prediction & ~gt)), "FN": int(np.count_nonzero(~prediction & gt)),
            "edits_vs_foris": {"addTP": int(np.count_nonzero(prediction & ~baseline & gt)),
                              "addFP": int(np.count_nonzero(prediction & ~baseline & ~gt)),
                              "deleteTP": int(np.count_nonzero(~prediction & baseline & gt)),
                              "deleteFP": int(np.count_nonzero(~prediction & baseline & ~gt))}}


def oracle_levelset(field, gt):
    """GT-only complete bilinear pixel IoU limit; never creates inference masks."""
    score = dense_field(field, "zero").reshape(-1)
    labels = gt.reshape(-1)
    order = np.argsort(-score, kind="stable")
    ordered = score[order]
    boundaries = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(ordered)-1]
    true_positive = np.cumsum(labels[order], dtype=np.int64)[boundaries]
    union = int(labels.sum()) + boundaries + 1 - true_positive
    iou = 100*true_positive/np.maximum(union, 1)
    winner = int(np.argmax(iou))
    return {"iou": float(iou[winner]), "selected_pixels": int(boundaries[winner]+1),
            "threshold": float(np.nextafter(ordered[boundaries[winner]], -np.inf)),
            "only_diagnostic": True}


def score(out):
    config = verify(out)
    seal = read(out / "sealed.json")
    assert seal["state"] == "ALL_120_TASKS_SEALED"
    assert seal["config_sha256"] == base.sha(out / "config.json")
    pairs = read(out / "pairs.json")
    truth = read(PARENT / "query_truth.json")
    controls = read(out / "external_controls.json")
    groups = photo_components(pairs)
    records = []
    names = [arm + "/" + render for arm in ARMS for render in RENDERS] + ["foris"]
    start = time.perf_counter()
    for pair, producer in zip(pairs, seal["records"]):
        assert pair["id"] == producer["id"]
        file = producer["file"]
        for folder, key in (("fields", "fields_sha256"), ("predictions", "predictions_sha256")):
            assert base.sha(out / folder / file) == producer[key]
        external = controls[pair["id"]]
        assert base.sha(external["path"]) == external["sha256"]
        tasks, ground = [], {}
        with np.load(out / "fields" / file, allow_pickle=False) as fields, \
                np.load(out / "predictions" / file, allow_pickle=False) as masks, \
                np.load(external["path"], allow_pickle=False) as old:
            for variant in ("A", "B"):
                binding = truth[pair["id"]]["masks"][variant]
                assert base.sha(binding["path"]) == binding["sha256"]
                gt0 = base.read_mask(binding["path"]).astype(np.float32)
                gt = F.interpolate(torch.from_numpy(gt0)[None, None],
                                   (1024, 1024), mode="nearest")[0, 0].numpy().astype(bool)
                ground[variant] = gt
                native = np.unpackbits(old[variant + "/crf"], count=1024**2).reshape(1024, 1024).astype(bool)
                metrics = {"foris": pixel_metrics(native, gt, native)}
                ranks = {roi: {} for roi in ("whole", "inside_external_foris", "outside_external_foris")}
                oracles = {}
                for arm in ARMS:
                    field = fields[variant + "/" + arm]
                    for render in RENDERS:
                        packed = masks[variant + "/" + arm + "/" + render]
                        prediction = np.unpackbits(packed, count=1024**2).reshape(1024, 1024).astype(bool)
                        assert np.array_equal(packed, pack_field(field, render))
                        metrics[arm + "/" + render] = pixel_metrics(prediction, gt, native)
                    for roi, keep in (("whole", np.ones_like(gt)), ("inside_external_foris", native),
                                      ("outside_external_foris", ~native)):
                        positive = (gt & keep).reshape(64, 16, 64, 16).sum((1, 3)).reshape(-1)
                        negative = (~gt & keep).reshape(64, 16, 64, 16).sum((1, 3)).reshape(-1)
                        ranks[roi][arm] = base.weighted_auc(field.reshape(-1), positive, negative)
                    if arm in ("context", "o24_density"):
                        oracles[arm] = oracle_levelset(field, gt)
                tasks.append({"variant": variant, "metrics": metrics, "ranking": ranks, "oracle": oracles})
            a_only = ground["A"] & ~ground["B"]
            b_only = ground["B"] & ~ground["A"]
            wa = a_only.reshape(64, 16, 64, 16).sum((1, 3)).reshape(-1)
            wb = b_only.reshape(64, 16, 64, 16).sum((1, 3)).reshape(-1)
            switch = {}
            for arm in ARMS:
                fa, fb = fields["A/"+arm].reshape(-1), fields["B/"+arm].reshape(-1)
                aa, bb = base.weighted_auc(fa, wa, wb), base.weighted_auc(fb, wb, wa)
                switch[arm] = {"task_switch_auc": None if aa is None or bb is None else (aa+bb)/2,
                               "response_auc": base.weighted_auc(fa-fb, wa, wb)}
        records.append({"id": pair["id"], "dataset": pair["dataset"], "tasks": tasks, "switch": switch})
        if len(records) % 10 == 0:
            print(json.dumps({"scored_pairs": len(records), "total": 60,
                              "seconds": round(time.perf_counter()-start, 2)}), flush=True)
    domains = {}
    for dataset in sorted({row["dataset"] for row in records}):
        rows = [r for r in records if r["dataset"] == dataset]
        values = {name: np.array([np.mean([task["metrics"][name]["iou"] for task in row["tasks"]])
                                 for row in rows]) for name in names}
        primary = values["context/zero"]
        comparisons = {control: grouped_interval(rows, primary-values[control], groups)
                       for control in ("linear_context/zero", "null_context/zero",
                                       "qk_density/zero", "o24_density/zero", "foris")}
        ranking = {}
        for roi in ("whole", "inside_external_foris", "outside_external_foris"):
            valid = [row for row in rows if all(task["ranking"][roi][arm] is not None
                        for task in row["tasks"] for arm in ARMS)]
            if not valid:
                ranking[roi] = {"n_pairs": 0}
                continue
            vals = {arm: np.array([np.mean([t["ranking"][roi][arm] for t in row["tasks"]])
                                   for row in valid]) for arm in ARMS}
            ranking[roi] = {"n_pairs": len(valid), "auc": {arm: float(v.mean()) for arm, v in vals.items()},
                           "context_minus": {arm: grouped_interval(valid, vals["context"]-vals[arm], groups)
                                             for arm in ARMS if arm != "context"}}
        domains[dataset] = {
            "n_pairs": len(rows), "n_tasks": 2*len(rows),
            "photo_components": len({groups[row["id"]] for row in rows}),
            "mean_complete_iou": {name: float(vals.mean()) for name, vals in values.items()},
            "primary_minus": comparisons, "ranking": ranking,
            "switch": {arm: {metric: float(np.mean([row["switch"][arm][metric] for row in rows
                                      if row["switch"][arm][metric] is not None]))
                            for metric in ("task_switch_auc", "response_auc")} for arm in ARMS},
            "oracle_complete_iou": {arm: float(np.mean([task["oracle"][arm]["iou"]
                                    for row in rows for task in row["tasks"]]))
                                    for arm in ("context", "o24_density")},
            "edits": {name: {action: int(sum(task["metrics"][name]["edits_vs_foris"][action]
                              for row in rows for task in row["tasks"]))
                            for action in ("addTP", "addFP", "deleteTP", "deleteFP")} for name in names}
        }
    report = {"state": "COMPLETE", "domains": domains, "records": records,
              "grain": "CLI1024 complete outputs; per-photo A/B mean IoU, not official class mIoU",
              "seal_sha256": base.sha(out / "sealed.json"), "config_sha256": base.sha(out / "config.json"),
              "scorer_sha256": base.sha(__file__), "seconds": time.perf_counter()-start,
              "new_full_encoder_calls": 0, "parameter_updates": 0,
              "caveat": "All photos previously exposed; exploratory mechanism and counterfactual inference, no novelty or generalization claim."}
    base.write(out / "report.json", report)
    lines = ["# 冻结末块任务上下文120：完整结果", "",
             "全部120任务／1200张完整1024 Mask先封存后评分。每照片两任务均值IoU，不是官方class mIoU；三域均为已曝光机制样本。",
             "", "| 方法 | COCO | PACO-Part | SUIM |", "|---|---:|---:|---:|"]
    for name in names:
        lines.append("| "+name+" | "+" | ".join(f"{domains[d]['mean_complete_iou'][name]:.6f}"
                      for d in ("coco", "paco_part", "suim"))+" |")
    lines += ["", "主场为context/zero；minmax是预设范围控制，不按域选赢家。GT最佳阈值只作排序能力诊断。",
              "", "| 主场减完整控制 | COCO均值及95%区间 | PACO均值及95%区间 | SUIM均值及95%区间 |",
              "|---|---|---|---|"]
    for name in ("linear_context/zero", "null_context/zero", "qk_density/zero", "o24_density/zero", "foris"):
        pieces = []
        for d in ("coco", "paco_part", "suim"):
            item = domains[d]["primary_minus"][name]
            pieces.append(f"{item['mean']:+.6f} [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}]")
        lines.append("| "+name+" | "+" | ".join(pieces)+" |")
    lines += ["", "RQ照片关联组数："+"/".join(str(domains[d]["photo_components"]) for d in ("coco", "paco_part", "suim"))+"。",
              "原native末块与旧O24的逐例身份、fixture、权重digest和实际部分前向成本见sealed.json；完整四编辑／全图与争议ROI排序／Mask切换／oracle见report.json。",
              "缓存内GPU读出时间不代表完整冷推理；本轮无新全encoder前向，没有改变旧缓存、默认方法或模型权重。"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n")
    base.write(out / "activity.json", {"state": "COMPLETE", "n_pairs": 60, "n_tasks": 120})
    print(json.dumps({d: {"iou": v["mean_complete_iou"], "oracle": v["oracle_complete_iou"],
                         "switch": v["switch"]} for d, v in domains.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "run", "score", "all"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            prepare(args.out)
        elif args.mode == "smoke":
            run(args.out, smoke=True)
        elif args.mode == "run":
            run(args.out)
        elif args.mode == "score":
            score(args.out)
        else:
            run(args.out)
            score(args.out)
    except Exception as exc:
        if args.out.exists():
            base.write(args.out / "failure.json", {"state": "FAILED", "mode": args.mode,
                       "exception": type(exc).__name__, "message": str(exc),
                       "producer_sha256": base.sha(__file__), "method_sha256": base.sha(METHOD)})
        raise


if __name__ == "__main__":
    main()
