#!/usr/bin/env python3
"""Finite outer segmentation training for three equal-budget metric arms.

CPU is the default; no model/image acquisition or automatic cache discovery.
Input JSON must explicitly authorize FUTURE episode tensors and contain only
train/development splits. The required per-episode dict has Fs[P,D], coverage[P],
Fq[Q,D], host_field[H,W], query_truth[H,W], grid (pair or support/query dict),
classid, roles_photo_ids, and the identical coordinates/projection_id contract.
The module sees ONLY Fs/coverage/Fq/grid; query_truth is used by loss/evaluator.
CUDA additionally requires --allow-gpu and a resource-guard parent process.

Manifest schema:
 {"schema":"demo9_metric_training_v1","authorization":"AUTHORIZED_FUTURE",
  "tensor_root":"/explicit/new/tensors",
  "feature_contract":{"coordinates":"...","projection_id":"..."},
  "splits":{"train":[{"path":"train.pt","classid":0,
   "roles_photo_ids":{"support":["s0"],"query":["q0"]}}],
   "development":[{"path":"development.pt","classid":20,
   "roles_photo_ids":{"support":["s20"],"query":["q20"]}}]}}

--synthetic-smoke creates ONLY temporary synthetic files and runs one CPU epoch
per arm. It is a code smoke test, never a training or segmentation result.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tics.reference_metric import (MetricConfig, ReferenceMetric, correct_host_field,
                                   difference_design, local_triplets, preference)


VARIANTS = ("protected", "fixed_global", "unprotected")


class MetricArm(ReferenceMetric):
    """Identical U/gain opportunities; fixed-global also learns its r scales."""
    def __init__(self, dimension, config, seed, variant):
        super().__init__(dimension, config, seed)
        if variant not in VARIANTS:
            raise ValueError("Unknown metric arm")
        self.variant = variant
        if variant == "fixed_global":
            initial = (1 - config.minimum) / (config.maximum - config.minimum)
            self.global_scales = torch.nn.Parameter(torch.full((config.rank,), math.log(initial / (1 - initial))))

    def adapt(self, reference, coverage, grid):
        if self.variant == "protected":
            return super().adapt(reference, coverage, grid)
        directions = self.directions()
        if not bool((coverage >= .5).any()) or not bool((coverage < .5).any()):
            return directions, reference.new_ones(self.config.rank), dict(state="FALLBACK_MISSING_LABEL", alpha=0., triples=0)
        if self.variant == "fixed_global":
            weights = (self.config.minimum + (self.config.maximum - self.config.minimum)
                       * self.global_scales.sigmoid())
            return directions, weights, dict(state="FIXED_GLOBAL", alpha=1., triples=0,
                min_weight=float(weights.detach().min()), max_weight=float(weights.detach().max()))
        unit = F.normalize(reference, dim=-1)
        triples, mass = local_triplets(unit, coverage, grid, self.config)
        if not len(triples):
            return directions, unit.new_ones(self.config.rank), dict(state="FALLBACK_NO_LEGAL_TRIPLETS", alpha=0., triples=0)
        margins, design = difference_design(unit, directions, triples)
        desired = preference(design, margins, mass, self.config)
        # SCML-style shared-basis/triplet adaptation control. Same preference
        # and spectral ray bounds, but NO protection of existing good margins.
        delta = desired - 1
        positive = delta > self.config.maximum - 1
        negative = delta < self.config.minimum - 1
        upper = torch.where(positive, (self.config.maximum - 1)
                            / torch.where(positive, delta, torch.ones_like(delta)), torch.ones_like(delta))
        lower = torch.where(negative, (1 - self.config.minimum)
                            / torch.where(negative, -delta, torch.ones_like(delta)), torch.ones_like(delta))
        alpha = torch.cat((delta.new_ones(1), upper, lower)).amin()
        weights = 1 + alpha * delta
        return directions, weights, dict(state="UNPROTECTED_SPECTRAL", alpha=float(alpha.detach()), triples=len(triples),
            min_weight=float(weights.detach().min()), max_weight=float(weights.detach().max()))


def _photos(roles):
    if not isinstance(roles, dict) or not {"support", "query"}.issubset(roles):
        raise ValueError("All-role photo IDs must include support and query")
    clean = {}
    for role, ids in roles.items():
        ids = [ids] if isinstance(ids, str) else ids
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Each role needs nonempty explicit photo IDs")
        clean[role] = set(ids)
    if clean["support"] & clean["query"]:
        raise ValueError("Reference and query must be different photos")
    return clean, set().union(*clean.values())


def read_manifest(path):
    path = Path(path).resolve()
    raw_bytes = path.read_bytes()
    doc = json.loads(raw_bytes)
    if doc.get("schema") != "demo9_metric_training_v1" or doc.get("authorization") != "AUTHORIZED_FUTURE":
        raise ValueError("Explicit AUTHORIZED_FUTURE tensor manifest required; no retired-cache discovery")
    if set(doc.get("splits", {})) != {"train", "development"}:
        raise ValueError("Only train and development splits allowed; test labels cannot select a checkpoint")
    contract = doc.get("feature_contract", {})
    if not all(isinstance(contract.get(k), str) and contract[k] for k in ("coordinates", "projection_id")):
        raise ValueError("Coordinate system and projection identity must be explicit")
    root = Path(doc["tensor_root"])
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    records, classes, photos = {}, {}, {}
    for split in ("train", "development"):
        if not isinstance(doc["splits"][split], list) or not doc["splits"][split]:
            raise ValueError("Both authorized splits must have episodes")
        records[split], classes[split], photos[split] = [], set(), set()
        for row in doc["splits"][split]:
            _, role_ids = _photos(row["roles_photo_ids"])
            classid = str(row["classid"])
            tensor = (root / row["path"]).resolve()
            if root not in tensor.parents or tensor.suffix != ".pt" or not tensor.is_file():
                raise ValueError("Episode must be an existing .pt within the explicitly authorized tensor root")
            records[split].append({**row, "resolved_path": tensor, "class_key": classid})
            classes[split].add(classid)
            photos[split].update(role_ids)
    if classes["train"] & classes["development"]:
        raise ValueError("Training/development classes overlap")
    if photos["train"] & photos["development"]:
        raise ValueError("Training/development photos overlap in at least one role")
    audit = dict(manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(), feature_contract=contract,
                 strict_class_disjoint=True, strict_all_role_photo_disjoint=True,
                 split_counts={k: len(v) for k, v in records.items()}, test_split_loaded=False,
                 class_ids={k: sorted(v) for k, v in classes.items()},
                 all_role_photo_ids={k: sorted(v) for k, v in photos.items()})
    return records, contract, audit


def load_episode(row, contract):
    # Tensor/primitive-only loading; do not execute an arbitrary pickle class.
    episode = torch.load(row["resolved_path"], map_location="cpu", weights_only=True)
    if str(episode.get("classid")) != row["class_key"]:
        raise ValueError("Manifest/payload class mismatch")
    payload_roles, _ = _photos(episode.get("roles_photo_ids"))
    manifest_roles, _ = _photos(row["roles_photo_ids"])
    if payload_roles != manifest_roles or episode.get("feature_contract") != contract:
        raise ValueError("Photo roles or coordinate/projection contract differ from the authorized manifest")
    required = ("Fs", "coverage", "Fq", "host_field", "query_truth")
    if any(not isinstance(episode.get(key), torch.Tensor) for key in required):
        raise ValueError("Episode needs explicit frozen tensors and supervised query labels")
    fs, coverage, fq, host, truth = [episode[k] for k in required]
    if (fs.ndim != 2 or fq.ndim != 2 or fs.shape[1] != fq.shape[1]
            or coverage.shape != fs.shape[:1] or host.ndim != 2 or truth.shape != host.shape):
        raise ValueError("Episode tensor dimensions disagree")
    if any(not bool(torch.isfinite(t).all()) for t in (fs, coverage, fq, host, truth)):
        raise ValueError("Nonfinite frozen episode tensor")
    if bool(((coverage < 0) | (coverage > 1)).any()) or not bool(((truth == 0) | (truth == 1)).all()):
        raise ValueError("Coverage must be [0,1], query GT binary")
    if bool((fs.norm(dim=1) <= 1e-12).any()) or bool((fq.norm(dim=1) <= 1e-12).any()):
        raise ValueError("Zero feature vector has no metric correspondence")
    grid = episode.get("grid")
    if isinstance(grid, dict):
        support_grid, query_grid = tuple(grid["support"]), tuple(grid["query"])
    else:
        support_grid = query_grid = tuple(grid)
    if (len(support_grid) != 2 or len(query_grid) != 2 or min(*support_grid, *query_grid) < 1
            or math.prod(support_grid) != len(fs) or math.prod(query_grid) != len(fq)):
        raise ValueError("Support/query grid must describe all patches")
    return {**episode, "support_grid": support_grid, "query_grid": query_grid}


def segmentation_loss(field, truth):
    logits, truth = field / .1, truth.to(field.dtype)
    positive, negative = truth.sum(), (1 - truth).sum()
    if positive > 0 and negative > 0:
        mass = torch.where(truth > .5, .5 / positive, .5 / negative)
    else:
        mass = torch.ones_like(truth) / truth.numel()
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * mass).sum()
    probability = logits.sigmoid()
    intersection = (probability * truth).sum()
    union = probability.sum() + truth.sum() - intersection
    soft_iou = 1 - (intersection + 1e-6) / (union + 1e-6)
    return bce + soft_iou


def predict_field(model, episode, device):
    reference = episode["Fs"].to(device=device, dtype=torch.float32)
    coverage = episode["coverage"].to(device=device, dtype=torch.float32)
    query = episode["Fq"].to(device=device, dtype=torch.float32)
    # query_truth is deliberately absent from the model call.
    delta, audit = model(reference, coverage, query, episode["support_grid"])
    host = episode["host_field"].to(device=device, dtype=torch.float32)
    return correct_host_field(host, delta, episode["query_grid"]), audit


@torch.no_grad()
def evaluate(model, rows, contract, device):
    model.eval()
    counts = {}
    loss_sum = 0.
    for row in rows:
        episode = load_episode(row, contract)
        field, _ = predict_field(model, episode, device)
        truth = episode["query_truth"].to(device).bool()
        prediction = field > 0
        intersection, union = int((prediction & truth).sum()), int((prediction | truth).sum())
        total = counts.setdefault(row["class_key"], [0, 0])
        total[0] += intersection
        total[1] += union
        loss_sum += float(segmentation_loss(field, truth))
    per_class = {key: value[0] / max(value[1], 1) for key, value in counts.items()}
    return dict(class_miou=sum(per_class.values()) / len(per_class), per_class_iou=per_class,
                class_intersection_union=counts, segmentation_loss=loss_sum / len(rows), episodes=len(rows))


def _atomic_save(payload, path):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def export_deployment(best_checkpoint, deployment_out):
    """Export ONLY a completed, development-selected trained model.

    No optimizer, RNG, query labels or frozen feature tensors enter deployment.
    Variant must travel with the state: unprotected and protected happen to
    have identical parameter names but intentionally different adaptation.
    """
    source, destination = Path(best_checkpoint), Path(deployment_out)
    if destination.exists():
        raise ValueError("Preserve the existing deployment artifact; choose a new destination")
    saved = torch.load(source, map_location="cpu", weights_only=True)
    required = ("model", "settings", "manifest_audit", "history", "epoch", "best_score")
    if any(key not in saved for key in required) or not saved.get("resume_allowed", False):
        raise ValueError("A completed-epoch trainer checkpoint is required; failure evidence cannot be deployed")
    variant = saved["settings"].get("variant")
    if variant not in VARIANTS:
        raise ValueError("Deployment must retain a known training arm")
    history = saved["history"]
    steps = sum(row["optimizer_updates"] for row in history)
    if (not history or steps < 1 or history[-1]["epoch"] != saved["epoch"]
            or history[-1]["development"]["class_miou"] != saved["best_score"]):
        raise ValueError("Checkpoint is not a trained development-selected epoch")
    audit = saved["manifest_audit"]
    if (audit.get("test_split_loaded") is not False or not audit.get("strict_class_disjoint")
            or not audit.get("strict_all_role_photo_disjoint")):
        raise ValueError("Checkpoint lacks strict label/split isolation provenance")
    state = {name: value.detach().cpu().clone() for name, value in saved["model"].items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("A nonfinite training state cannot be frozen for deployment")
    payload = dict(state="FROZEN", training_split="base", trained_steps=steps,
                   query_test_labels_used=False, feature_contract=audit["feature_contract"],
                   config=saved["settings"]["metric_config"], model_state=state, variant=variant,
                   selected_epoch=saved["epoch"], development_class_miou=saved["best_score"],
                   source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   training_manifest_sha256=audit["manifest_sha256"],
                   training_class_ids=audit.get("class_ids", {}).get("train", []),
                   training_all_role_photo_ids=audit.get("all_role_photo_ids", {}).get("train", []),
                   development_class_ids=audit.get("class_ids", {}).get("development", []),
                   development_all_role_photo_ids=audit.get("all_role_photo_ids", {}).get("development", []),
                   inference_loader="train_reference_metric.load_deployment; preserves MetricArm variant semantics")
    # Validate variant/state compatibility and dictionary rank before publishing.
    _load_deployment_payload(payload, payload["feature_contract"], torch.device("cpu"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(payload, destination)
    return {key: value for key, value in payload.items() if key != "model_state"}


def _load_deployment_payload(payload, expected_feature_contract, device):
    if (payload.get("state") != "FROZEN" or payload.get("training_split") != "base"
            or payload.get("trained_steps", 0) < 1 or payload.get("query_test_labels_used") is not False):
        raise ValueError("Only a frozen base-trained artifact without test-label access is deployable")
    if payload.get("feature_contract") != expected_feature_contract:
        raise ValueError("Deployment feature coordinates/projection differ from the supplied episode")
    variant = payload.get("variant")
    if variant not in VARIANTS:
        raise ValueError("Never default an unknown deployment arm to protected behavior")
    state = payload.get("model_state", {})
    dictionary = state.get("dictionary")
    if not isinstance(dictionary, torch.Tensor) or dictionary.ndim != 2:
        raise ValueError("Explicit dictionary state required")
    if any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Nonfinite or unexpected model state")
    config = MetricConfig(**payload["config"])
    model = MetricArm(dictionary.shape[0], config, seed=2048, variant=variant)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        model.directions()  # Reject a rank-deficient dictionary, not a silent rank change.
    return model, payload


def load_deployment(path, expected_feature_contract, device=torch.device("cpu")):
    """Runner adapter; caller owns any future GPU authorization and split check."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _load_deployment_payload(payload, expected_feature_contract, torch.device(device))


def _rng_state(shuffle, device):
    state = dict(python=random.getstate(), shuffle=shuffle.getstate(), torch_cpu=torch.get_rng_state())
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _device(name, allow_gpu):
    if name == "cpu":
        return torch.device("cpu")  # No CUDA probe or initialization.
    if name != "cuda" or not allow_gpu:
        raise ValueError("CUDA requires explicit --device cuda --allow-gpu; CPU is the default")
    parent = subprocess.check_output(["ps", "-p", str(os.getppid()), "-o", "command="], text=True)
    if not any(Path(token).name == "experiment_resource_guard.py" for token in parent.split()):
        raise RuntimeError("Future CUDA training must be launched as a resource-guard stage")
    if not torch.cuda.is_available():
        raise RuntimeError("Explicitly armed GPU stage has no CUDA device")
    torch.cuda.set_per_process_memory_fraction(.3)
    return torch.device("cuda")


def train(manifest, output_dir, variant="protected", config=MetricConfig(), *, epochs=10,
          patience=0, learning_rate=1e-3, seed=2048, device=torch.device("cpu"), resume=None):
    if epochs < 1 or patience < 0 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Positive finite budget/rate required; patience=0 disables early stopping")
    rows, contract, manifest_audit = read_manifest(manifest)
    # Validate every authorized payload before making an optimizer or loading a checkpoint.
    dimensions = set()
    for split in rows:
        for row in rows[split]:
            dimensions.add(load_episode(row, contract)["Fs"].shape[1])
    if len(dimensions) != 1:
        raise ValueError("Different feature dimensions cannot share one learned dictionary")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any((output / name).exists() for name in
                              ("last.pt", "best.pt", "failure.pt", "report.json")):
        raise ValueError("Existing training evidence must be preserved; choose a fresh output or explicit completed-epoch resume")
    random.seed(seed)
    torch.manual_seed(seed)
    shuffle = random.Random(seed)
    model = MetricArm(dimensions.pop(), config, seed, variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    history, start_epoch, best_score, stale = [], 0, -math.inf, 0
    settings = dict(variant=variant, metric_config=asdict(config), epochs=epochs, patience=patience,
                    learning_rate=learning_rate, seed=seed, loss="balanced_BCE_logits(H/.1)+soft_IoU(sigmoid(H/.1))",
                    selection="development class-macro IoU; no test set/teacher MSE", device=str(device),
                    equal_budget_note="Same configured epochs/episodes/optimizer/gain across arms; patience0 keeps exact epoch parity")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if not saved.get("resume_allowed", False):
            raise ValueError("Failure/partial-epoch evidence is preserved but cannot be resumed as a completed epoch")
        if saved["manifest_audit"] != manifest_audit or saved["settings"] != settings:
            raise ValueError("Explicit resume must preserve split manifest, configuration and total budget")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best_score, stale, history = saved["epoch"], saved["best_score"], saved["stale"], saved["history"]
        random.setstate(saved["rng"]["python"])
        shuffle.setstate(saved["rng"]["shuffle"])
        torch.set_rng_state(saved["rng"]["torch_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["rng"]["torch_cuda"])

    def checkpoint(epoch, resume_allowed=True):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), settings=settings,
                    manifest_audit=manifest_audit, epoch=epoch, best_score=best_score, stale=stale,
                    history=history, rng=_rng_state(shuffle, device), resume_allowed=resume_allowed)

    current_epoch, current_row, started = start_epoch, None, time.monotonic()
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            order = list(rows["train"])
            shuffle.shuffle(order)
            model.train()
            train_loss, updates, skipped = 0., 0, 0
            for row in order:
                current_row = row
                episode = load_episode(row, contract)
                optimizer.zero_grad(set_to_none=True)
                field, audit = predict_field(model, episode, device)
                loss = segmentation_loss(field, episode["query_truth"].to(device))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training loss; no retry or hidden fallback")
                train_loss += float(loss.detach())
                if not loss.requires_grad:
                    skipped += 1
                    continue
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in model.parameters()):
                    raise RuntimeError("Nonfinite outer gradient; preserving failure")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                    raise RuntimeError("Nonfinite optimizer parameter; preserving failure")
                updates += 1
            development = evaluate(model, rows["development"], contract, device)
            summary = dict(epoch=epoch + 1, train_segmentation_loss=train_loss / len(order),
                           optimizer_updates=updates, no_gradient_fallbacks=skipped, development=development)
            history.append(summary)
            improved = development["class_miou"] > best_score
            if improved:
                best_score, stale = development["class_miou"], 0
            else:
                stale += 1
            payload = checkpoint(epoch + 1)
            _atomic_save(payload, output / "last.pt")
            if improved:
                _atomic_save(payload, output / "best.pt")
            print(json.dumps(dict(variant=variant, **summary)), flush=True)
            if patience and stale >= patience:
                break
    except Exception as error:
        _atomic_save(checkpoint(current_epoch, resume_allowed=False), output / "failure.pt")
        failure = dict(state="FAILED", error_type=type(error).__name__, error=str(error),
                       epoch=current_epoch, episode_path=str(current_row["resolved_path"]) if current_row else None,
                       settings=settings, manifest_audit=manifest_audit)
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    changes = {name: float((parameter.detach().cpu() - initial[name]).abs().max())
               for name, parameter in model.named_parameters()}
    report = dict(state="COMPLETED", settings=settings, manifest_audit=manifest_audit,
                  epochs_completed=len(history), best_development_class_miou=best_score,
                  parameter_max_changes=changes, history=history, seconds=time.monotonic() - started,
                  learned_parameter_count=sum(p.numel() for p in model.parameters()),
                  deployment_query_truth_input=False, checkpoint=str(output / "best.pt"))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def synthetic_smoke(out=None):
    """One CPU epoch per arm, no assets or persistent synthetic feature caches."""
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(512)
    with tempfile.TemporaryDirectory(prefix="demo9-metric-trainer-") as temporary:
        root = Path(temporary)
        contract = dict(coordinates="synthetic_unit_features_6d", projection_id="synthetic_identity_v1")
        splits = {"train": [], "development": []}
        for split, classid in (("train", 0), ("development", 20)):
            for index in range(2):
                coverage = (torch.arange(64).reshape(8, 8) % 8 < 4).flatten().float()
                truth_patch = (torch.arange(64).reshape(8, 8) % 8 < 4)
                fs = torch.randn(64, 6, generator=generator)
                fq = torch.randn(64, 6, generator=generator)
                fs[:, 0] += (coverage * 2 - 1) * .8
                fq[:, 0] += (truth_patch.flatten().float() * 2 - 1) * .7
                truth = F.interpolate(truth_patch.float()[None, None], (16, 16), mode="nearest")[0, 0]
                host = .08 * (truth * 2 - 1) + torch.randn(16, 16, generator=generator) * .14
                roles = {"support": [f"{split}-support-{index}"], "query": [f"{split}-query-{index}"]}
                name = f"{split}-{index}.pt"
                torch.save(dict(Fs=fs, coverage=coverage, Fq=fq, host_field=host, query_truth=truth,
                                grid=[8, 8], classid=classid, roles_photo_ids=roles, feature_contract=contract), root / name)
                splits[split].append(dict(path=name, classid=classid, roles_photo_ids=roles))
        document = dict(schema="demo9_metric_training_v1", authorization="AUTHORIZED_FUTURE",
                        tensor_root=str(root), feature_contract=contract, splits=splits)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(document))
        config = replace(MetricConfig(), rank=2, margin=.1, temperature=.2, regularization=1.,
                         bandwidth=.4, max_anchors_per_class=8, positives=1, negatives=1,
                         exclusion_radius=1, tolerance=1e-9)
        reports = {}
        audited_rows, _, _ = read_manifest(manifest)
        for variant in VARIANTS:
            report = train(manifest, root / variant, variant, config, epochs=1,
                           learning_rate=.02, seed=2048, device=torch.device("cpu"))
            if report["history"][0]["optimizer_updates"] != 2:
                raise AssertionError("Synthetic arm must update on the same two episodes")
            if report["parameter_max_changes"]["dictionary"] <= 0 or report["parameter_max_changes"]["gain_parameter"] <= 0:
                raise AssertionError("Both dictionary and global gain must receive actual outer updates")
            saved = torch.load(root / variant / "best.pt", map_location="cpu", weights_only=True)
            if "rng" not in saved or saved["manifest_audit"]["test_split_loaded"]:
                raise AssertionError("Checkpoint RNG or split contract violated")
            trained = MetricArm(6, config, 2048, variant)
            trained.load_state_dict(saved["model"])
            sample = load_episode(audited_rows["train"][0], contract)
            with torch.no_grad():
                directions, scales, adaptation = trained.adapt(sample["Fs"], sample["coverage"], sample["support_grid"])
                if float(scales.min()) < config.minimum - 1e-6 or float(scales.max()) > config.maximum + 1e-6:
                    raise AssertionError("Trained arm violates spectral bounds")
                if variant == "protected":
                    unit = F.normalize(sample["Fs"], dim=-1)
                    triples, _ = local_triplets(unit, sample["coverage"], sample["support_grid"], config)
                    margins, design = difference_design(unit, directions, triples)
                    guard = margins >= config.margin
                    if bool(((margins + design @ (scales - 1))[guard]
                             < config.retain_fraction * margins[guard] - 1e-5).any()):
                        raise AssertionError("Trained protected arm violates reference-margin protection")
            reports[variant] = dict(epochs=1, optimizer_updates=2,
                dictionary_change=report["parameter_max_changes"]["dictionary"],
                gain_change=report["parameter_max_changes"]["gain_parameter"], checkpoint_rng=True,
                trained_weights_legal=True, adaptation_state=adaptation["state"])
            artifact = root / f"deployment-{variant}.pt"
            export_deployment(root / variant / "best.pt", artifact)
            deployed, frozen = load_deployment(artifact, contract)
            with torch.no_grad():
                before, _ = trained(sample["Fs"], sample["coverage"], sample["Fq"], sample["support_grid"])
                after, _ = deployed(sample["Fs"], sample["coverage"], sample["Fq"], sample["support_grid"])
            if not torch.equal(before, after) or any(parameter.requires_grad for parameter in deployed.parameters()):
                raise AssertionError("Deployment changes the selected arm's correction or permits training")
            if "optimizer" in frozen or "rng" in frozen or frozen["variant"] != variant:
                raise AssertionError("Deployment retains training-only state or drops the arm")
            reports[variant].update(deployment_exact_correction_parity=True,
                                    deployment_variant_preserved=True, deployment_frozen=True)
        rejected = {}
        for condition in ("class_overlap", "photo_overlap", "test_split"):
            bad = json.loads(json.dumps(document))
            if condition == "class_overlap":
                bad["splits"]["development"][0]["classid"] = 0
            elif condition == "photo_overlap":
                bad["splits"]["development"][0]["roles_photo_ids"]["query"] = ["train-support-0"]
            else:
                bad["splits"]["test"] = []
            path = root / f"bad-{condition}.json"
            path.write_text(json.dumps(bad))
            try:
                read_manifest(path)
            except ValueError:
                rejected[condition] = True
            else:
                raise AssertionError("Leakage manifest accepted")
        altered = torch.load(root / "train-0.pt", map_location="cpu", weights_only=True)
        altered["feature_contract"] = {**contract, "projection_id": "incompatible_projection"}
        torch.save(altered, root / "train-0.pt")
        try:
            load_episode(audited_rows["train"][0], contract)
        except ValueError:
            rejected["payload_projection_mismatch"] = True
        else:
            raise AssertionError("Incompatible feature coordinates/projection accepted")
        result = dict(state="PASSED", scope="Synthetic one-epoch CPU trainer smoke; no real-data efficacy",
                      arms=reports, split_leakage_rejections=rejected, cpu_only=True,
                      cuda_initialization=False, real_training_executed=False, temporary_tensors_removed=True)
    if out:
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="protected")
    parser.add_argument("--metric-config", type=Path, help="Explicit JSON MetricConfig overrides")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--export", type=Path, help="Export an explicit completed best.pt without any dataset load")
    parser.add_argument("--deployment-out", type=Path, help="New frozen inference .pt destination")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--out", type=Path, help="Small synthetic-smoke result JSON")
    args = parser.parse_args()
    if args.export:
        if args.deployment_out is None or args.device != "cpu" or args.allow_gpu or args.manifest or args.synthetic_smoke:
            parser.error("Deployment export needs --deployment-out and is CPU-only, without a dataset load")
        print(json.dumps(export_deployment(args.export, args.deployment_out)), flush=True)
        return
    if args.deployment_out is not None:
        parser.error("--deployment-out requires --export best.pt")
    if args.synthetic_smoke:
        if args.device != "cpu" or args.allow_gpu or args.manifest or args.resume:
            parser.error("Synthetic smoke is local CPU only and does not load a supplied real dataset")
        print(json.dumps(synthetic_smoke(args.out)), flush=True)
        return
    if args.manifest is None or args.output_dir is None:
        parser.error("Real future training requires explicit --manifest and --output-dir")
    config = MetricConfig(**json.loads(args.metric_config.read_text())) if args.metric_config else MetricConfig()
    device = _device(args.device, args.allow_gpu)
    report = train(args.manifest, args.output_dir, args.variant, config, epochs=args.epochs,
                   patience=args.patience, learning_rate=args.learning_rate, seed=args.seed,
                   device=device, resume=args.resume)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
