#!/usr/bin/env python3
"""Prepare/acquire fresh full-coordinate native INSID3 metric episodes.

Default --prepare-only and --self-check use stdlib/PIL metadata only: no torch,
encoder, CUDA, remote access or download. Execution requires an explicit CPU
plan, prior-produced fixed BLACK-image positional basis, and a resource guard.
Original full 1024 coordinates/FP32 serialization; no PCA/complement compression.
Training/development query GT is stored only in their supervised episode files.
Inference input contains NO query GT; its evaluator is a separate file opened
after the native feature/host field is frozen. Sampling never ranks mask quality.

Pilot: --train-count24 --dev-count8 --inference-count10. Maximum:240/80/10.
FP32 maximum collection requires explicit RAM/tmpfs storage; disk is capped5GiB.
Prepared plans name future runtime artifacts honestly; they are not READY yet.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import sys
import tempfile
import time

from PIL import Image


HERE = Path(__file__).resolve().parent
MAX_DISK = 5 * 1024 ** 3
MAX_RAM = 20 * 1024 ** 3
BLACK_SOURCE = "original_INSID3_normalized_torch_zeros_black_image_FP32_positional_basis"


def photo_id(name):
    match = re.search(r"COCO_(?:train|val)2014_(\d+)", name)
    return "COCO:" + str(int(match[1])) if match else Path(name).stem


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_index(path):
    path = Path(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text())
    else:
        # Explicit existing official dataset metadata; never discover a retired
        # experiment pickle or execute a model checkpoint in this CPU phase.
        with path.open("rb") as stream:
            value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Explicit class->image-name protocol index required")
    result = {}
    for classid, names in value.items():
        classid = int(classid)
        if not 0 <= classid < 80:
            raise ValueError("Expected zero-based COCO-80 protocol classes")
        result[classid] = sorted(set(map(str, names)))
    return result


def select_episodes(base, evaluation, fold, train_count, dev_count, inference_count, seed):
    if not 0 <= fold < 4 or not 1 <= train_count <= 240 or not 1 <= dev_count <= 80 or not 1 <= inference_count <= 10:
        raise ValueError("Finite cohort counts/fold outside the declared protocol")
    test_classes = {fold + 4 * i for i in range(20)}
    base_classes = sorted(set(base) - test_classes)
    if len(base_classes) < 4:
        raise ValueError("Base protocol has insufficient distinct training/development classes")
    rng = random.Random(seed)
    rng.shuffle(base_classes)
    development_classes = sorted(base_classes[:max(1, len(base_classes) // 5)])
    training_classes = sorted(set(base_classes) - set(development_classes))
    inference_classes = sorted(set(evaluation) & test_classes)
    if not inference_classes:
        raise ValueError("Evaluation protocol lacks held-out fold classes")
    selected, used_photos = {}, set()
    # Reserve inference then development images first. Every photo is used at
    # most once across ALL roles/splits, including shared multiclass COCO photos.
    requests = (("inference", inference_classes, evaluation, inference_count),
                ("development", development_classes, base, dev_count),
                ("train", training_classes, base, train_count))
    for split, classes, index, count in requests:
        pools = {c: list(index[c]) for c in classes}
        for pool in pools.values():
            rng.shuffle(pool)
        rows, class_use = [], Counter()
        while len(rows) < count:
            eligible = []
            for c in classes:
                available = [n for n in pools[c] if photo_id(n) not in used_photos]
                if len({photo_id(n) for n in available}) >= 2:
                    eligible.append((class_use[c], c, available))
            if not eligible:
                raise ValueError(f"Cannot build {count} photo-disjoint {split} episodes without quality selection")
            _, c, available = min(eligible, key=lambda item: (item[0], item[1]))
            support = available[0]
            query = next(n for n in available[1:] if photo_id(n) != photo_id(support))
            roles = {"support": [photo_id(support)], "query": [photo_id(query)]}
            rows.append(dict(e=len(rows), classid=c, support=support, query=query, roles_photo_ids=roles))
            used_photos.update((photo_id(support), photo_id(query)))
            class_use[c] += 1
        selected[split] = rows
    class_sets = {split: {r["classid"] for r in rows} for split, rows in selected.items()}
    photo_sets = {split: {p for r in rows for ids in r["roles_photo_ids"].values() for p in ids}
                  for split, rows in selected.items()}
    for a, b in (("train", "development"), ("train", "inference"), ("development", "inference")):
        if class_sets[a] & class_sets[b] or photo_sets[a] & photo_sets[b]:
            raise RuntimeError("Class/all-role photo isolation failed")
    return selected, dict(class_sets={k: sorted(v) for k, v in class_sets.items()},
                          photo_counts={k: len(v) for k, v in photo_sets.items()},
                          class_disjoint=True, all_role_photo_disjoint=True,
                          sampler="seeded class-balanced protocol draw, unique photos; no GT-quality ranking")


def prepare(args):
    base, evaluation = read_index(args.base_index), read_index(args.evaluation_index)
    selected, isolation = select_episodes(base, evaluation, args.fold, args.train_count,
                                         args.dev_count, args.inference_count, args.seed)
    data, masks = args.data_root.resolve(), args.annotation_root.resolve()
    assets = {args.base_index.resolve(), args.evaluation_index.resolve()}
    for rows in selected.values():
        for row in rows:
            for role in ("support", "query"):
                image, annotation = data / row[role], masks / Path(row[role]).with_suffix(".png")
                if data not in image.resolve().parents or masks not in annotation.resolve().parents:
                    raise ValueError("Protocol image path escapes the declared source root")
                with Image.open(image) as value:
                    size = value.size
                    value.verify()
                with Image.open(annotation) as value:
                    if value.size != size:
                        raise ValueError("Official annotation/image size mismatch")
                    value.verify()
                assets.update((image.resolve(), annotation.resolve()))
    model_root = args.shared_root.resolve() / "models/dinov3-vitl16-timm"
    config_path, weights_path = model_root / "config.json", model_root / "model.safetensors"
    config = json.loads(config_path.read_text())
    if config.get("architecture") != "vit_large_patch16_dinov3":
        raise ValueError("Existing native DINOv3 ViT-L/16 required")
    assets.update((config_path, weights_path))
    source = args.demo4_root.resolve() / "INSID3/models/insid3.py"
    text = source.read_text()
    basis_block = text.split("def _build_positional_basis", 1)[-1].split("def _debias_features", 1)[0]
    if "torch.zeros" not in basis_block or "normalize(" not in basis_block or "torch.linalg.svd" not in basis_block:
        raise ValueError("Positional-basis source must be the original normalized BLACK image; no Gaussian transplant")
    assets.add(source)
    statistics = []
    for path in sorted(assets):
        stat = path.stat()
        if stat.st_size < 1:
            raise ValueError("Empty existing source/data/weight asset")
        statistics.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    # FP32 full dimensions retained. Serialization estimate includes a little
    # container overhead; train files do not store redundant native geometry.
    training_bytes = (args.train_count + args.dev_count) * (2 * 64 * 64 * 1024 * 4 + 1024 * 1024 * 5 + 1024 ** 2)
    inference_bytes = args.inference_count * (3 * 64 * 64 * 1024 * 4 + 1024 * 1024 * 5 + 1024 ** 2)
    estimate = training_bytes + inference_bytes
    if estimate > (MAX_DISK if args.storage == "disk" else MAX_RAM):
        raise ValueError("Full FP32 cohort exceeds declared storage cap; maximum cohort needs explicit RAM/tmpfs storage")
    tensor_root, output = args.tensor_root.resolve(), args.output_dir.resolve()
    if tensor_root in (args.shared_root.resolve(), data, masks) or any(
            protected == tensor_root or protected in tensor_root.parents for protected in (data, masks, model_root)):
        raise ValueError("Fresh owned tensor root must not be a shared asset directory")
    plan = dict(schema="demo9_native_metric_acquisition_v1", state="PREPARED_CPU_METADATA",
                authorized_future_tensors=True, seed=args.seed, fold=args.fold, rows=selected,
                isolation=isolation, data_root=str(data), annotation_root=str(masks),
                shared_root=str(args.shared_root.resolve()), demo4_root=str(args.demo4_root.resolve()),
                output_dir=str(output), tensor_root=str(tensor_root), storage=args.storage,
                estimated_bytes=estimate, disk_cap_bytes=MAX_DISK, ram_cap_bytes=MAX_RAM,
                fixed_basis=dict(path=str(args.projection_basis.resolve()), producer=args.basis_producer,
                                 state="FUTURE_RUNTIME_ARTIFACT", source=BLACK_SOURCE,
                                 ready=args.projection_basis.exists()),
                expected_feature_contract=dict(stream="native", coordinates="original_full_debiased",
                                               dimension=1024, serialization="float32", basis_source=BLACK_SOURCE),
                assets=statistics, native_host_source_sha256=file_sha(source),
                no_model_or_torch_import=True, no_quality_GT_sampling=True,
                host_field="source-faithful native_signed_field INSID3 prethreshold bilinear capture",
                runtime_ready=False, runtime_dependency="Prior setup must produce the declared fixed BLACK-image basis")
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    if args.plan_out.exists():
        raise ValueError("Preserve the old prepared cohort; choose a new plan path")
    args.plan_out.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def _guard(device_args):
    import torch
    if not device_args.allow_gpu or device_args.resource_guard_state is None:
        raise ValueError("Acquisition requires an explicitly armed resource-guard stage")
    guard = json.loads(device_args.resource_guard_state.read_text())
    if (guard.get("state") != "GPU_RUNNING" or guard.get("last_event", {}).get("own_child_pid") != os.getpid()
            or not any(e.get("state") == "GPU_CUDA_READINESS_CONFIRMED" for e in guard.get("events", []))):
        raise ValueError("Resource guard must own this process and confirm CUDA readiness")
    if not sys.platform.startswith("linux") or not torch.cuda.is_available():
        raise RuntimeError("No confirmed Linux CUDA stage")
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DEMO4_GPU_FRAC", ".3")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda")


def _verify_assets(plan):
    for row in plan["assets"]:
        stat = Path(row["path"]).stat()
        if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("CPU-prepared asset changed before encoder construction: " + row["path"])
    if plan["fixed_basis"]["source"] != BLACK_SOURCE or not Path(plan["fixed_basis"]["path"]).is_file():
        raise ValueError("Prior setup has not produced the declared original BLACK-image basis")


def _atomic_json(value, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _tmpfs(path):
    resolved = path.resolve()
    if Path("/dev/shm") == resolved or Path("/dev/shm") in resolved.parents:
        return True
    mounts = Path("/proc/mounts").read_text().splitlines()
    matches = []
    for line in mounts:
        parts = line.split()
        if len(parts) > 2:
            mount = Path(parts[1].replace("\\040", " ")).resolve()
            if mount == resolved or mount in resolved.parents:
                matches.append((len(str(mount)), parts[2]))
    return bool(matches and max(matches)[1] == "tmpfs")


def acquire(args):
    plan = json.loads(args.prepared_plan.read_text())
    if plan.get("schema") != "demo9_native_metric_acquisition_v1" or plan.get("state") != "PREPARED_CPU_METADATA":
        raise ValueError("Explicit CPU-frozen acquisition plan required")
    _verify_assets(plan)  # No torch/model before metadata/dependency validation.
    device = _guard(args)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from types import MethodType
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(4)
    os.environ["DEMO4_ROOT"], os.environ["DEMO4_CACHE"] = plan["demo4_root"], plan["shared_root"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, plan["demo4_root"])
    from icx.common import TimmDINOv3
    from models.insid3 import INSID3
    from utils.data import load_image, load_mask
    from tics.host_signed_field import native_signed_field
    root, output = Path(plan["tensor_root"]), Path(plan["output_dir"])
    if root.exists() and any(root.iterdir()):
        raise ValueError("Acquisition requires a fresh owned tensor root; never reuse retired tensors")
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "acquisition_report.json").exists():
        raise ValueError("Preserve previous acquisition evidence")
    if plan["storage"] == "ram" and not _tmpfs(root):
        raise ValueError("RAM collection must be on /dev/shm or verified tmpfs")
    if shutil.disk_usage(root).free < plan["estimated_bytes"] + 64 * 1024 ** 2:
        raise RuntimeError("Declared fresh buffer has insufficient capacity")
    basis_path = Path(plan["fixed_basis"]["path"])
    saved_basis = torch.load(basis_path, map_location="cpu", weights_only=True)
    if isinstance(saved_basis, dict):
        if saved_basis.get("source", BLACK_SOURCE) != BLACK_SOURCE:
            raise ValueError("Fixed positional basis provenance does not match original BLACK-image source")
        basis = saved_basis.get("basis", saved_basis.get("positional_basis"))
    else:
        basis = saved_basis
    if not isinstance(basis, torch.Tensor) or basis.shape != (1024, 500) or not torch.isfinite(basis).all():
        raise ValueError("Expected finite original 1024x500 BLACK positional basis")
    basis = basis.float()
    if float((basis.T @ basis - torch.eye(500)).abs().max()) > 1e-3:
        raise ValueError("Fixed positional columns are not orthonormal")
    contract = {**plan["expected_feature_contract"], "projection_id": "sha256:" + file_sha(basis_path),
                "encoding": "original SQ-paired native BF16 feature extraction; native normalization/debias; FP32 serialization"}
    report = dict(state="ACQUIRING", feature_contract=contract, records=[], buffer_bytes=0,
                  plan_sha256=file_sha(args.prepared_plan), query_test_labels_in_input=False,
                  query_GT_use="train/development supervised files; inference evaluator file only",
                  scope="Fresh authorized full-feature collection; not method performance", source_basis=BLACK_SOURCE)
    def save():
        _atomic_json(report, output / "acquisition_report.json")
    save()
    started = time.monotonic()
    try:
        encoder = TimmDINOv3().to(device).eval().requires_grad_(False)
        # Reuse the setup artifact instead of recomputing its content-free SVD.
        # Constructor source remains unchanged after this one local instantiation.
        original_builder = INSID3._build_positional_basis
        INSID3._build_positional_basis = lambda self, dev: basis.to(dev).clone()
        try:
            host = INSID3(encoder=encoder, image_size=1024, svd_components=500, tau=.6,
                          merge_threshold=.2, mask_refiner="bilinear", resize_to_orig_size=False,
                          device=str(device)).to(device).eval().requires_grad_(False)
        finally:
            INSID3._build_positional_basis = original_builder
        host.positional_basis.copy_(basis.to(device))
        if not torch.equal(host.positional_basis.cpu(), basis):
            raise RuntimeError("Source-frozen positional basis changed")
        manifest_rows = {"train": [], "development": []}
        inference_rows = []
        with torch.inference_mode():
            for split in ("train", "development", "inference"):
                for row in plan["rows"][split]:
                    c = row["classid"]
                    images = [Image.open(Path(plan["data_root"]) / row[role]).convert("RGB") for role in ("support", "query")]
                    support_annotation = np.array(Image.open(Path(plan["annotation_root"]) / Path(row["support"]).with_suffix(".png")))
                    original_support = torch.from_numpy((support_annotation == c + 1).copy())
                    si, qi = [load_image(image, host._transform, device)[0] for image in images]
                    sm = load_mask(original_support, 1024, device)
                    if not sm.any():
                        raise RuntimeError("Protocol support lacks foreground; stop, do not quality-resample")
                    raw = host._extract_features(torch.cat((si, qi), 0).unsqueeze(0))
                    normalized = F.normalize(raw, p=2, dim=2)
                    debiased = host._debias_features(normalized)
                    _, _, dimensions, height, width = debiased.shape
                    if dimensions != 1024 or (height, width) != (64, 64):
                        raise RuntimeError("Expected full native ViT-L1024/grid64 features")
                    fs, fq = [debiased[0, i].flatten(1).T.float().cpu() for i in (0, 1)]
                    geometry = normalized[0, 1].flatten(1).T.float().cpu()
                    coverage = F.interpolate(original_support.to(device).float()[None, None],
                                             (height, width), mode="area")[0, 0].flatten().cpu()
                    # Exact same native extracted pair reused by the ORIGINAL host
                    # decoder. First item verifies against a fresh original call.
                    direct = host.predict_mask(si, sm, qi) if not report["records"] else None
                    owned, previous = "_extract_features" in host.__dict__, host.__dict__.get("_extract_features")
                    original_extract = host._extract_features
                    host._extract_features = MethodType(lambda self, image_batch: raw, host)
                    try:
                        with native_signed_field(host, "insid3") as capture:
                            prediction = host.predict_mask(si, sm, qi)
                            signed = capture.only_field().cpu()
                    finally:
                        if owned:
                            host._extract_features = previous
                        else:
                            delattr(host, "_extract_features")
                    if not torch.equal(signed > 0, prediction.cpu()) or (direct is not None and not torch.equal(prediction, direct)):
                        raise RuntimeError("Actual native host/SQ replay/signed-field parity failed")
                    fields = dict(Fs=fs, coverage=coverage, Fq=fq, host_field=signed,
                                  grid={"support": [height, width], "query": [height, width]},
                                  classid=c, roles_photo_ids=row["roles_photo_ids"], feature_contract=contract)
                    name = f"{split}_{row['e']:04d}.pt"
                    path = root / name
                    if split == "inference":
                        fields["query_geometry"] = geometry
                        if {"query_truth", "query_gt", "query_mask", "native_mask"} & set(fields):
                            raise RuntimeError("Forbidden query supervision in inference input")
                        torch.save(fields, path)  # Freeze/write lawful input BEFORE query annotation opens.
                    query_annotation = np.array(Image.open(Path(plan["annotation_root"]) / Path(row["query"]).with_suffix(".png")))
                    original_truth = torch.from_numpy((query_annotation == c + 1).copy())
                    truth = load_mask(original_truth, 1024, device)[0].bool().cpu()
                    if split == "inference":
                        evaluator = root / f"inference_{row['e']:04d}_evaluator.pt"
                        torch.save(dict(native_mask=truth, original_mask=original_truth), evaluator)
                        inference_rows.append(dict(input=str(path), evaluator_labels=str(evaluator), **row))
                    else:
                        fields["query_truth"] = truth
                        torch.save(fields, path)
                        manifest_rows[split].append(dict(path=name, classid=c, roles_photo_ids=row["roles_photo_ids"]))
                    report["buffer_bytes"] = sum(p.stat().st_size for p in root.glob("*.pt"))
                    cap = MAX_DISK if plan["storage"] == "disk" else MAX_RAM
                    if report["buffer_bytes"] > cap:
                        raise RuntimeError("Owned buffer exceeded declared cap; no feature compression or shared cleanup")
                    report["records"].append(dict(split=split, e=row["e"], classid=c,
                        roles_photo_ids=row["roles_photo_ids"], native_signed_field_exact=True,
                        first_original_replay_parity=(direct is not None), input=str(path)))
                    report["elapsed_seconds"] = time.monotonic() - started
                    save()
                    print(json.dumps(report["records"][-1]), flush=True)
                    del raw, normalized, debiased, geometry, coverage, fs, fq, prediction, signed, sm, si, qi, fields, truth
        manifest = dict(schema="demo9_metric_training_v1", authorization="AUTHORIZED_FUTURE",
                        tensor_root=str(root), feature_contract=contract, splits=manifest_rows)
        training_manifest = output / "training_manifest.json"
        _atomic_json(manifest, training_manifest)
        _atomic_json(dict(rows=inference_rows, feature_contract=contract, isolation=plan["isolation"],
                         fold=plan["fold"], seed=plan["seed"],
                         source_protocol="Class/photo-disjoint base-class collection from existing val metadata; not official train split"),
                     output / "inference_cohort.json")
        # A direct check with the production trainer verifies all payload metadata.
        trainer = importlib.import_module("train_reference_metric")
        verified, checked_contract, audit = trainer.read_manifest(training_manifest)
        for rows in verified.values():
            for row in rows:
                trainer.load_episode(row, checked_contract)
        _atomic_json(dict(rank=32), output / "metric_config.json")
        commands = []
        python = sys.executable
        for variant in ("protected", "fixed_global", "unprotected"):
            checkpoint_root, deployed = output / variant, output / f"{variant}_frozen.pt"
            commands.append(dict(kind="gpu", name=f"metric_train_{variant}", argv=[python, str(HERE / "train_reference_metric.py"),
                "--manifest", str(training_manifest), "--output-dir", str(checkpoint_root), "--variant", variant,
                "--metric-config", str(output / "metric_config.json"), "--epochs", "10", "--patience", "0", "--device", "cuda", "--allow-gpu"]))
            commands.append(dict(kind="cpu", name=f"metric_export_{variant}", handoff_budget_seconds=60,
                argv=[python, str(HERE / "train_reference_metric.py"), "--export", str(checkpoint_root / "best.pt"), "--deployment-out", str(deployed)]))
        evaluator_argv = [python, str(HERE / "native_metric_cohort_experiment.py"),
                          "--cohort", str(output / "inference_cohort.json")]
        for variant in ("protected", "fixed_global", "unprotected"):
            evaluator_argv.extend(["--checkpoint", variant + "=" + str(output / f"{variant}_frozen.pt")])
        evaluator_argv.extend(["--device", "cuda", "--allow-gpu", "--resource-guard-state",
                               str(args.resource_guard_state), "--out", str(output / "evaluation.json")])
        commands.append(dict(kind="gpu", name="metric_cohort_evaluation", argv=evaluator_argv,
                             produces=[str(output / "evaluation.json")]))
        _atomic_json(dict(commands=commands, scientific_arms_independent=True,
                         runtime_dependencies_only=True), output / "route_commands.json")
        report.update(state="COMPLETED", training_manifest=str(training_manifest), inference_cohort=str(output / "inference_cohort.json"),
                      route_commands=str(output / "route_commands.json"), supervised_feature_contract_audited=True,
                      no_query_GT_in_inference_inputs=True, native_full_dimensions=1024, rank=32)
        save()
    except Exception as error:
        report.update(state="ERROR", error=repr(error), partial_owned_buffer_preserved=True)
        save()
        raise
    return report


def self_check(out=None):
    # Metadata-only fixture; no model/feature tensor generation, torch or CUDA.
    with tempfile.TemporaryDirectory(prefix="demo9-native-metric-metadata-") as folder:
        root = Path(folder)
        data, annotations, shared, demo4 = root / "data", root / "annotations", root / "shared", root / "demo4"
        model = shared / "models/dinov3-vitl16-timm"
        source = demo4 / "INSID3/models/insid3.py"
        for directory in (data, annotations, model, source.parent):
            directory.mkdir(parents=True)
        source.write_text("def _build_positional_basis():\n return torch.linalg.svd(normalize(torch.zeros(1,3,1024,1024)))\ndef _debias_features():\n pass\n")
        (model / "config.json").write_text(json.dumps(dict(architecture="vit_large_patch16_dinov3")))
        (model / "model.safetensors").write_bytes(b"SYNTHETIC_METADATA_ONLY_NOT_A_WEIGHT")
        index = {}
        for c in range(12):
            names = []
            for j in range(12):
                name = f"COCO_train2014_{c*100+j:012d}.jpg"
                names.append(name)
                Image.new("RGB", (12, 10), (c, j, 0)).save(data / name)
                Image.new("L", (12, 10), c + 1).save(annotations / Path(name).with_suffix(".png"))
            index[c] = names
        index[1].append(index[0][0])  # A multiclass photo must not leak across roles.
        path = root / "index.json"
        path.write_text(json.dumps(index))
        args = argparse.Namespace(base_index=path, evaluation_index=path, fold=0,
            train_count=8, dev_count=4, inference_count=3, seed=0, data_root=data,
            annotation_root=annotations, shared_root=shared, demo4_root=demo4,
            storage="disk", tensor_root=root / "future_tensors", output_dir=root / "results",
            projection_basis=root / "future_black_basis.pt", basis_producer="native_setup_v1",
            plan_out=root / "prepared.json")
        plan = prepare(args)
        if not plan["isolation"]["class_disjoint"] or not plan["isolation"]["all_role_photo_disjoint"]:
            raise AssertionError("Metadata fixture leaks")
        if plan["fixed_basis"]["ready"] or plan["runtime_ready"]:
            raise AssertionError("A future runtime asset was falsely marked ready")
        if plan["expected_feature_contract"]["coordinates"] != "original_full_debiased" or "black" not in plan["fixed_basis"]["source"]:
            raise AssertionError("Wrong projector/coordinate source")
        repeated, _ = select_episodes(read_index(path), read_index(path), 0, 8, 4, 3, 0)
        if repeated != plan["rows"]:
            raise AssertionError("Cohort is not deterministic")
        original_text = source.read_text()
        source.write_text(original_text.replace("torch.zeros", "torch.randn"))
        try:
            prepare(args)
        except ValueError as error:
            if "BLACK" not in str(error):
                raise
        else:
            raise AssertionError("A Gaussian projector was silently transplanted")
        finally:
            source.write_text(original_text)
        result = dict(state="CPU_METADATA_SELF_CHECK_PASSED", class_and_all_role_photo_isolation=True,
                      original_full_1024_coordinate_contract=True, fixed_black_basis_source=True,
                      gaussian_projector_rejected=True, future_producer_dependency_honest=True, no_quality_GT_sampling=True,
                      torch_imported="torch" in sys.modules, synthetic_only=True, actual_acquisition_run=False)
        if result["torch_imported"]:
            raise AssertionError("Prepare-only path imported torch")
    if out:
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Default metadata-only operation")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--self-check", "--selfcheck", action="store_true")
    parser.add_argument("--out", type=Path, help="Small self-check JSON only")
    parser.add_argument("--base-index", type=Path)
    parser.add_argument("--evaluation-index", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--demo4-root", type=Path)
    parser.add_argument("--projection-basis", type=Path)
    parser.add_argument("--basis-producer", default="native_setup_v1")
    parser.add_argument("--tensor-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--prepared-plan", type=Path)
    parser.add_argument("--storage", choices=("disk", "ram"), default="disk")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--dev-count", type=int, default=8)
    parser.add_argument("--inference-count", type=int, default=10)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--resource-guard-state", type=Path)
    args = parser.parse_args()
    if args.self_check:
        if args.acquire or args.allow_gpu:
            parser.error("Metadata self-check does not acquire or arm GPU")
        print(json.dumps(self_check(args.out)))
    elif args.acquire:
        if args.prepare_only or args.prepared_plan is None:
            parser.error("Acquisition needs --prepared-plan and cannot be combined with --prepare-only")
        print(json.dumps(acquire(args)))
    else:
        required = (args.base_index, args.evaluation_index, args.data_root, args.annotation_root,
                    args.shared_root, args.demo4_root, args.projection_basis, args.tensor_root,
                    args.output_dir, args.plan_out)
        if any(x is None for x in required) or args.allow_gpu:
            parser.error("Prepare-only needs explicit existing metadata/assets and fresh output/future basis paths; no GPU arming")
        plan = prepare(args)
        print(json.dumps(dict(state=plan["state"], counts={k: len(v) for k, v in plan["rows"].items()},
                              estimated_bytes=plan["estimated_bytes"], runtime_ready=False, torch_imported=False)))


if __name__ == "__main__":
    main()
