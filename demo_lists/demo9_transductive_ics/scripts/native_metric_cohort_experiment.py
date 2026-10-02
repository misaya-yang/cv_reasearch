#!/usr/bin/env python3
"""One-PID E3 held-out cohort evaluator for three frozen metric arms.

The shared source/geometry/host field is read from native_metric_acquire's
inference_cohort.json. All three arms, the native baseline and identity-metric
full-distribution KDE control are frozen BEFORE opening each separate evaluator.
The oracle is a diagnostic choice among these ALREADY frozen predictions, not
a deployable selector, optimized task upper bound or label-derived new mask.
Output includes analyzer-compatible pixel I/U, ledgers and packed predictions.
No child executors, new encoder/model/data acquisition or automatic CUDA use.
"""
import argparse
from dataclasses import replace
import hashlib
import io
import contextlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference_feature_probe as probe

VARIANTS = ("protected", "fixed_global", "unprotected")


def _photos(row):
    roles = row.get("roles_photo_ids")
    if not isinstance(roles, dict) or not {"support", "query"}.issubset(roles):
        raise ValueError("Held-out cohort must retain support/query photo identities")
    groups = {role: set([ids] if isinstance(ids, str) else ids) for role, ids in roles.items()}
    if groups["support"] & groups["query"]:
        raise ValueError("Reference/query photo identity collision")
    return set().union(*groups.values())


def _write(report, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(records, key="iu"):
    totals = {}
    for row in records:
        for name, (intersection, union) in row[key].items():
            if str(row["arm_states"].get(name, "SCORED")).startswith("ERROR"):
                continue
            pair = totals.setdefault(name, {}).setdefault(str(row["c"]), [0, 0])
            pair[0] += intersection
            pair[1] += union
    per_class = {name: {c: 100 * i / max(u, 1) for c, (i, u) in classes.items()}
                 for name, classes in totals.items()}
    return {name: sum(classes.values()) / len(classes) for name, classes in per_class.items()}, per_class


def evaluate_cohort(cohort_path, checkpoints, out, device=torch.device("cpu"),
                    max_ram_bytes=512 * 1024 ** 2, require_native_contract=True):
    if set(checkpoints) != set(VARIANTS):
        raise ValueError("All three independently trained arms must be explicitly supplied")
    cohort_path, out = Path(cohort_path), Path(out)
    if out.exists():
        raise ValueError("Preserve existing cohort evidence; use a fresh report path")
    cohort = json.loads(cohort_path.read_text())
    rows, contract = cohort.get("rows"), cohort.get("feature_contract")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10:
        raise ValueError("This bounded pilot evaluates one to ten frozen inference episodes")
    if require_native_contract and (not isinstance(contract, dict)
            or contract.get("stream") != "native" or contract.get("coordinates") != "original_full_debiased"
            or contract.get("dimension") != 1024 or not str(contract.get("projection_id", "")).startswith("sha256:")):
        raise ValueError("Real E3 requires the acquired native full1024/source-fixed projection contract")
    # Deployment provenance is inspected before opening any inference/evaluator
    # payload. Model variants/configuration cannot be chosen from cohort GT.
    metadata, config = {}, None
    for variant in VARIANTS:
        saved = torch.load(checkpoints[variant], map_location="cpu", weights_only=True)
        if (saved.get("state") != "FROZEN" or saved.get("variant") != variant
                or saved.get("feature_contract") != contract or saved.get("training_split") != "base"
                or saved.get("trained_steps", 0) < 1 or saved.get("query_test_labels_used") is not False):
            raise ValueError("Checkpoint variant/projection/training provenance does not match the cohort")
        if config is None:
            config = saved["config"]
        elif saved["config"] != config:
            raise ValueError("All metric controls must share the identical adaptation/decoder configuration")
        seen_classes = set(map(str, saved.get("training_class_ids", []))) | set(map(str, saved.get("development_class_ids", [])))
        seen_photos = set(saved.get("training_all_role_photo_ids", [])) | set(saved.get("development_all_role_photo_ids", []))
        if not seen_classes or not seen_photos:
            raise ValueError("Deployment must disclose training and checkpoint-selection provenance")
        for row in rows:
            if str(row["classid"]) in seen_classes or _photos(row) & seen_photos:
                raise ValueError("Inference class/photo overlaps training or development checkpoint selection")
        metadata[variant] = {key: saved[key] for key in ("variant", "trained_steps", "selected_epoch",
            "development_class_miou", "feature_contract", "training_manifest_sha256")}
    used_photos = set()
    for row in rows:
        if Path(row["input"]).resolve() == Path(row["evaluator_labels"]).resolve():
            raise ValueError("Inference and evaluator must be different files")
        current = _photos(row)
        if current & used_photos:
            raise ValueError("Pilot cohort requires unique all-role photos")
        used_photos.update(current)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(state="RUNNING", experiment="E3_reference_difference_metric", dataset="COCO-20i",
        fold=cohort.get("fold", 0), seed=cohort.get("seed", 0), records=[], feature_contract=contract,
        training_budget="Each arm capped at the same ten epochs; bounded pilot, not convergence evidence",
        checkpoint_metadata=metadata, query_GT_passed_to_methods=False,
        cohort_sha256=probe.file_sha(cohort_path), baseline="insid3_native",
        naive="native_kde", learned_same_budget_controls=["metric_fixed_global", "metric_unprotected"],
        oracle="oracle_frozen_bank_native_IoU",
        oracle_scope="Per-episode native-IoU selection among valid frozen masks only; diagnostic, not a deployable rule or optimized task upper bound",
        source_scope="Separate class/photo-disjoint cohort from base-class val metadata; not official train split or matched standard-first10 cohort",
        scope="Ten-episode pilot; no method gain/convergence/four-fold claim",
        checkpoint_sha256={variant: probe.file_sha(path) for variant, path in checkpoints.items()},
        source_sha256={str(path): probe.file_sha(path) for path in (Path(__file__), HERE / "reference_feature_probe.py",
            HERE / "train_reference_metric.py", HERE.parent / "tics/reference_metric.py")})
    _write(report, out)
    started = time.monotonic()
    try:
        for ordinal, row in enumerate(rows):
            case_start = time.monotonic()
            d = probe.canonical_episode(probe.load_trusted(Path(row["input"])))
            reference_grid, query_grid, estimate = probe.validate_episode(d, 5, max_ram_bytes)
            if d["feature_contract"] != contract or str(d.get("classid")) != str(row["classid"]):
                raise ValueError("Frozen episode contract or class identity differs from the cohort")
            if d.get("roles_photo_ids") != row["roles_photo_ids"]:
                raise ValueError("Input/cohort all-role photo metadata disagree")
            for name in ("reference_features", "reference_coverage", "query_features", "query_geometry", "host_signed_field"):
                d[name] = d[name].to(device)
            outputs = {"host": d["host_signed_field"].detach().clone()}
            states = {}
            # Reuse the exact prepared feature executor; it never opens GT.
            for variant in VARIANTS:
                arm_outputs, arm_records, _ = probe.freeze_outputs(d, ["metric"],
                    metric_checkpoint=checkpoints[variant], max_ram_bytes=max_ram_bytes)
                name = "metric_" + variant
                outputs[name] = arm_outputs["metric"].detach().clone()
                states[name] = arm_records["metric"]
            with torch.no_grad():
                foreground = d["reference_coverage"].flatten() >= .5
                if not bool(foreground.any()) or bool(foreground.all()):
                    outputs["native_kde"] = outputs["host"].clone()
                    states["native_kde"] = dict(state="FALLBACK_DEGENERATE_SUPPORT", preserved_host_exact=True)
                else:
                    metric = probe.module("reference_metric")
                    # Identity metric is independent of U, using an empty
                    # direction matrix: the same complete density function.
                    reference = F.normalize(d["reference_features"].float(), dim=1)
                    query = F.normalize(d["query_features"].float(), dim=1)
                    empty = query.new_empty((query.shape[1], 0))
                    score = metric.density_ratio(query, reference, foreground, empty,
                        query.new_empty(0), bandwidth=config["bandwidth"])
                    outputs["native_kde"] = probe.lift_signed(score.reshape(query_grid), outputs["host"].shape).detach().clone()
                    states["native_kde"] = dict(state="SOLVED", bandwidth=config["bandwidth"],
                        recipe="Shared complete FG/BG distributions, identity distance; no new trained representation")
            if any(not bool(torch.isfinite(value).all()) for value in outputs.values()):
                raise RuntimeError("Nonfinite frozen field; evaluator remains unopened")
            frozen_at = time.monotonic()
            # The existing evaluator is the ONLY opening of query labels. Every
            # bank field already detached/cloned, including explicit error fallback.
            predictions = probe.evaluate_frozen(outputs, states, Path(row["evaluator_labels"]))
            predictions["insid3_native"] = predictions.pop("host")
            arm_states = {"insid3_native": "SCORED", **{name: value["state"] for name, value in states.items()}}
            iu, original_iu, pixels, prediction_bits = {}, {}, {}, {}
            for name, prediction in predictions.items():
                prediction_bits[name] = prediction["prediction_bits"]
                if "native_ledger" in prediction:
                    ledger = prediction["native_ledger"]
                    iu[name] = [ledger["intersection"], ledger["union"]]
                    pixels[name] = ledger
                if "original_ledger" in prediction:
                    ledger = prediction["original_ledger"]
                    original_iu[name] = [ledger["intersection"], ledger["union"]]
            # No GT-derived mask. Copy the ledger and bits of a frozen candidate.
            choice = max(iu, key=lambda name: iu[name][0] / max(iu[name][1], 1))
            oracle = report["oracle"]
            iu[oracle] = list(iu[choice])
            if choice in original_iu:
                original_iu[oracle] = list(original_iu[choice])
            pixels[oracle] = dict(pixels[choice])
            prediction_bits[oracle] = dict(prediction_bits[choice])
            arm_states[oracle] = "DIAGNOSTIC_GT_CHOICE_AMONG_FROZEN_BANK"
            record = dict(e=row.get("e", ordinal), c=row["classid"], classid=row["classid"],
                dataset="COCO-20i", fold=report["fold"], roles_photo_ids=row["roles_photo_ids"],
                support=row.get("support"), query=row.get("query"), iu=iu, original_iu=original_iu,
                pixels=pixels, prediction_bits=prediction_bits, arm_states=arm_states,
                methods=states, frozen_bank_choice=choice, all_bank_predictions_frozen_before_evaluator=True,
                feature_contract=contract, estimated_tensor_bytes=estimate,
                predictions_frozen_s=frozen_at - case_start, episode_seconds=time.monotonic() - case_start,
                input=str(row["input"]), evaluator_sha256=probe.file_sha(Path(row["evaluator_labels"])))
            report["records"].append(record)
            report["miou"], report["per_class"] = summarize(report["records"])
            report["original_miou"], report["original_per_class"] = summarize(report["records"], "original_iu")
            report["elapsed_seconds"] = time.monotonic() - started
            _write(report, out)
            print(json.dumps(dict(e=record["e"], c=record["c"], count=len(report["records"]),
                iu=iu, method_states=arm_states, frozen_bank_choice=choice)), flush=True)
            del d, outputs, predictions, arm_outputs
        report["failure_count"] = sum(value["state"] == "ERROR" for row in report["records"] for value in row["methods"].values())
        report["fallback_count"] = sum(str(value["state"]).startswith("FALLBACK") for row in report["records"] for value in row["methods"].values())
        report["state"] = "COMPLETED_WITH_ERRORS" if report["failure_count"] else "COMPLETED"
        _write(report, out)
        return report
    except Exception as error:
        report.update(state="ERROR", error=repr(error), evaluator_not_used_for_prediction=True)
        _write(report, out)
        raise


def self_check(out=None):
    torch.set_num_threads(1)
    torch.manual_seed(633)
    trainer = probe.metric_loader()
    with tempfile.TemporaryDirectory(prefix="demo9-metric-cohort-cpu-") as folder:
        root = Path(folder)
        contract = dict(stream="native", coordinates="synthetic_unit6", dimension=6, projection_id="synthetic_identity")
        splits = {"train": [], "development": []}
        for split, classid in (("train", 0), ("development", 20)):
            for e in range(2):
                coverage = (torch.arange(64) % 8 < 4).float()
                fs, fq = torch.randn(64, 6), torch.randn(64, 6)
                fs[:, 0] += .8 * (coverage * 2 - 1)
                fq[:, 0] += .7 * (coverage * 2 - 1)
                truth = F.interpolate(coverage.reshape(1, 1, 8, 8), (16, 16), mode="nearest")[0, 0]
                host = .1 * (truth * 2 - 1) + .12 * torch.randn(16, 16)
                roles = {"support": [f"{split}-s-{e}"], "query": [f"{split}-q-{e}"]}
                path = root / f"{split}_{e}.pt"
                torch.save(dict(Fs=fs, coverage=coverage, Fq=fq, host_field=host, query_truth=truth,
                    grid=[8, 8], classid=classid, roles_photo_ids=roles, feature_contract=contract), path)
                splits[split].append(dict(path=path.name, classid=classid, roles_photo_ids=roles))
        manifest = root / "train.json"
        manifest.write_text(json.dumps(dict(schema="demo9_metric_training_v1", authorization="AUTHORIZED_FUTURE",
            tensor_root=str(root), feature_contract=contract, splits=splits)))
        config = replace(trainer.MetricConfig(), rank=2, temperature=.2, regularization=1., bandwidth=.4,
            max_anchors_per_class=8, positives=1, negatives=1, exclusion_radius=1)
        checkpoints = {}
        for variant in VARIANTS:
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train(manifest, root / variant, variant, config, epochs=1, learning_rate=.02)
                trainer.export_deployment(root / variant / "best.pt", root / f"{variant}.pt")
            checkpoints[variant] = root / f"{variant}.pt"
        coverage = (torch.arange(64) % 8 < 4).float()
        fs, fq = torch.randn(64, 6), torch.randn(64, 6)
        fs[:, 0] += .8 * (coverage * 2 - 1)
        fq[:, 0] += .7 * (coverage * 2 - 1)
        truth = F.interpolate(coverage.reshape(1, 1, 8, 8), (16, 16), mode="nearest")[0, 0]
        roles = {"support": ["heldout-s"], "query": ["heldout-q"]}
        input_path, labels_path = root / "input.pt", root / "eval.pt"
        torch.save(dict(Fs=fs, coverage=coverage, Fq=fq, query_geometry=F.normalize(fq, dim=1),
            host_field=.1 * (truth * 2 - 1) + .12 * torch.randn(16, 16), grid=[8, 8],
            classid=40, roles_photo_ids=roles, feature_contract=contract), input_path)
        torch.save(dict(native_mask=truth.bool(), original_mask=truth.bool()), labels_path)
        cohort = root / "cohort.json"
        cohort.write_text(json.dumps(dict(rows=[dict(input=str(input_path), evaluator_labels=str(labels_path),
            classid=40, e=0, roles_photo_ids=roles)], feature_contract=contract)))
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluate_cohort(cohort, checkpoints, root / "result.json", require_native_contract=False)
        row = report["records"][0]
        expected = {"insid3_native", "native_kde", "metric_protected", "metric_fixed_global", "metric_unprotected", report["oracle"]}
        if report["state"] != "COMPLETED" or set(row["iu"]) != expected or set(row["prediction_bits"]) != expected:
            raise AssertionError("Frozen baseline/naive/oracle/control bank is incomplete")
        if not row["all_bank_predictions_frozen_before_evaluator"]:
            raise AssertionError("Evaluator was opened before the bank froze")
        analyzer = __import__("analyze_native_experiments")
        normalized = analyzer.normalize_row(row, report, 0)
        for bits in normalized["prediction_bits"].values():
            if analyzer.decode_bits(bits).shape != (16, 16):
                raise AssertionError("Packed prediction/analyzer contract failed")
        bad = json.loads(cohort.read_text())
        bad["rows"][0]["roles_photo_ids"]["query"] = ["train-s-0"]
        bad_path = root / "leak.json"
        bad_path.write_text(json.dumps(bad))
        try:
            evaluate_cohort(bad_path, checkpoints, root / "bad.json", require_native_contract=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Checkpoint-supervised photo leakage accepted")
        result = dict(state="CPU_COHORT_SELF_CHECK_PASSED", one_pid_inline=True,
            three_frozen_variants=True, native_baseline_identity_density_naive_frozen_oracle=True,
            all_predictions_before_evaluator=True, analyzer_IU_and_packed_prediction_compatible=True,
            supervised_class_photo_leakage_rejected=True, synthetic_only=True, real_data_or_GPU_run=False)
    if out:
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], help="Explicit VARIANT=/path/frozen.pt, three times")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--resource-guard-state", type=Path)
    parser.add_argument("--max-ram-bytes", type=int, default=512 * 1024 ** 2)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        if args.device != "cpu" or args.allow_gpu:
            parser.error("Self-check is local CPU only")
        print(json.dumps(self_check(args.out)))
        return
    if args.cohort is None or args.out is None or args.max_ram_bytes < 1:
        parser.error("Explicit --cohort --out and a positive working-memory cap required")
    checkpoints = {}
    for argument in args.checkpoint:
        variant, path = argument.split("=", 1)
        if variant in checkpoints or variant not in VARIANTS:
            parser.error("Known unique deployment variants required")
        checkpoints[variant] = Path(path)
    device = probe.device_gate(args.device, args.allow_gpu, args.resource_guard_state)
    with torch.no_grad():
        report = evaluate_cohort(args.cohort, checkpoints, args.out, device, args.max_ram_bytes)
    print(json.dumps(dict(state=report["state"], episodes=len(report["records"]),
                         failure_count=report["failure_count"], class_miou=report["miou"])))
    raise SystemExit(2 if report["failure_count"] else 0)


if __name__ == "__main__":
    main()
