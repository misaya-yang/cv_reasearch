"""Read-only protocol/cache inventory and independent saved-I/U aggregation.

No encoder, prediction, GT, mask, or feature payload is read. Only this audit's
inventory.json is written. Cache counts are metadata discovery, not full
payload validation or an episode-coverage claim.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[4]
ASSETS = REPO.parent / "cv_data"
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source(path):
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def expected_from_report(report):
    expected = defaultdict(set)
    for key in report["per_class"]:
        fold, label = key.split(":", 1)
        expected[fold].add(label)
    return {fold: sorted(labels) for fold, labels in expected.items()}


def aggregate(records, expected=None, frame=None, arms=None):
    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    arm_set = None
    for row in records:
        iu = row["iu"] if frame is None else row["iu"][frame]
        if arms is not None:
            iu = {arm: iu[arm] for arm in arms}
        if arm_set is None:
            arm_set = set(iu)
        if set(iu) != arm_set:
            raise ValueError("Different arm sets within paired records")
        key = str(row.get("fold", 0)), str(row.get("class_id", row.get("c", 0)))
        for arm, (inter, union) in iu.items():
            if not (0 <= inter <= union):
                raise ValueError("Invalid saved I/U")
            pooled[key][arm][0] += inter
            pooled[key][arm][1] += union
    if expected is None:
        expected = defaultdict(set)
        for fold, label in pooled:
            expected[fold].add(label)
        expected = {f: sorted(cs) for f, cs in expected.items()}
    per_fold = {}
    for fold, labels in expected.items():
        per_fold[fold] = {
            arm: mean(100 * pooled[(fold, label)][arm][0]
                      / max(pooled[(fold, label)][arm][1], 1) for label in labels)
            for arm in sorted(arm_set)
        }
    return dict(n=len(records), class_slots=sum(map(len, expected.values())),
                observed_class_slots=len(pooled),
                missing_class_slots=sum((f, c) not in pooled
                                        for f, cs in expected.items() for c in cs),
                per_fold=per_fold,
                miou={arm: mean(v[arm] for v in per_fold.values())
                      for arm in sorted(arm_set)})


def metric_check(identifier, ledger, report_path, report, frame=None, expected=None):
    data = rows(ledger)
    if expected is None and "per_class" in report:
        expected = expected_from_report(report)
    target = report["miou"] if "miou" in report else report["scores"]
    found = aggregate(data, expected, frame, arms=sorted(target))
    comparisons = {arm: dict(recomputed=value, recorded=target[arm],
                             abs_error=abs(value - target[arm]))
                   for arm, value in found["miou"].items() if arm in target}
    return dict(id=identifier, ledger=source(ledger), report=source(report_path),
                metric_frame=frame or "original/ledger documented frame",
                **found, comparisons=comparisons,
                passed=all(c["abs_error"] < 1e-9 for c in comparisons.values()),
                verification="Saved I/U aggregation only; no mask reconstruction or bootstrap rerun")


def main():
    official = ASSETS / "a/official_s0_manifest_20261008/prepared.json"
    frozen = read(official)
    actual_official_manifest = source(official.parent / "manifest.json")
    if actual_official_manifest["sha256"] != frozen["manifest_sha256"]:
        raise ValueError("Prepared official manifest SHA does not match file")
    folds = {}
    for key, item in frozen["folds"].items():
        folds[key] = {k: item[k] for k in ("official_length", "selected_length")}
        folds[key]["class_slots"] = len(item["expected_class_ids"])
    manifest_inventory = []
    for path in sorted((ASSETS / "a/fresh_manifests_20261008").glob("*/prepared.json")):
        document = read(path)
        if document.get("state") != "FRESH_MANIFEST_FROZEN":
            continue
        manifest_inventory.append(dict(**source(path), seed=document.get("seed"),
            split_role=document.get("split_role"), n=document.get("n"),
            manifest_sha256=document.get("manifest_sha256"),
            datasets=sorted({key.split("/")[0] for key in document["folds"]})))

    profile = "0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158"
    pool = ASSETS / "a/paco_mean200_20261008/raw_cache" / profile
    branches, shape_dtype, dataset_entries, roles = Counter(), Counter(), Counter(), Counter()
    missing_payloads, malformed = [], []
    metadata_digest = hashlib.sha256()
    entries = sorted(pool.glob("*/entry.json"))
    for path in entries:
        raw = path.read_bytes()
        metadata_digest.update(str(path.relative_to(pool)).encode())
        metadata_digest.update(hashlib.sha256(raw).digest())
        d = json.loads(raw)
        if d["profile_id"] != profile:
            malformed.append(str(path))
        if not (path.parent / d["file"]).is_file():
            missing_payloads.append(str(path))
        for name, info in d["features"].items():
            branches[name] += 1
            shape_dtype[str((name, info["shape"], info["dtype"]))] += 1
        dataset_tags = set()
        for p in d["provenance"]:
            eid = str(p.get("episode_id", ""))
            role = str(p.get("view_role", p.get("role", "unspecified")))
            roles[role] += 1
            for dataset in ("coco", "lvis", "paco_part", "pascal_part", "suim"):
                if "/" + dataset + "/" in eid or eid.startswith(dataset + "/"):
                    dataset_tags.add(dataset)
            if eid.startswith("dg18-") or "deepglobe" in str(p).lower():
                dataset_tags.add("deepglobe_road")
        for tag in dataset_tags or {"unclassified_provenance"}:
            dataset_entries[tag] += 1
    secondary_caches = []
    for rel in ("a/raw_feature_cache_complete5_20261008_v2/cache",
                "a/raw_feature_cache_probe_20261008/cache"):
        root = ASSETS / rel
        available = Counter()
        paths = sorted(root.glob("*/*/entry.json"))
        for path in paths:
            available.update(read(path)["features"].keys())
        secondary_caches.append(dict(root=str(root), entries=len(paths),
                                     branch_metadata_counts=dict(available)))

    checks = []
    pub = REPO / "evidence/local/research_20261005/pipeline_verified/frozen_public4000_v1"
    checks.append(metric_check("COCO_public4000_legacy_complete", pub / "episodes.jsonl",
        pub / "report.json", read(pub / "report.json"),
        expected={str(f): [str(f + 4 * i) for i in range(20)] for f in range(4)}))
    lvis = ASSETS / "a/lvis_component_completion600_20261009/score"
    checks.append(metric_check("LVIS_component600_observed60classes_perfold", lvis / "cumulative600_episodes.jsonl",
        lvis / "paired_results.json", read(lvis / "paired_results.json")["cumulative600"]))
    paco = ASSETS / "a/paco_role_competition600_20261010/score"
    checks.append(metric_check("PACO_competitive600_official303slots", paco / "scored_episodes.jsonl",
        paco / "paired_results.json", read(paco / "paired_results.json")["primary"], frame="original"))
    deep = ASSETS / "a/deepglobe_role_competition100_20261010/score"
    checks.append(metric_check("DeepGlobe_custom100_pooled_road", deep / "scored_episodes.jsonl",
        deep / "paired_results.json", read(deep / "paired_results.json")))
    coco = ASSETS / "a/coco_role_competition200_20261010/matched_raw/score"
    checks.append(metric_check("COCO_competitive200_matched40slots", coco / "scored_episodes.jsonl",
        coco / "paired_results.json", read(coco / "paired_results.json")["primary"], frame="original"))
    coco_ridge = REPO / "evidence/local/parallel_signal_20261010/02_task_identity/ridge_coco_validation"
    checks.append(metric_check("COCO_frozen_ridge200_matched40slots", coco_ridge / "scored_episodes.jsonl",
        coco_ridge / "report.json", read(coco_ridge / "report.json")["primary"]))
    current = REPO / "evidence/local/difficult_region_signal_20261010/paco_remaining500"
    current_records = rows(current / "scored_episodes.jsonl")
    current_report = read(current / "report.json")
    for label, selected in (("all600", current_records),
                            ("remaining500", [r for r in current_records if not r["pilot"]])):
        item = aggregate(selected, expected_from_report(current_report["groups"][label]))
        recorded = current_report["groups"][label]["miou"]
        errors = {a: abs(v - recorded[a]) for a, v in item["miou"].items()}
        checks.append(dict(id="PACO_frozen_ridge_" + label, ledger=source(current / "scored_episodes.jsonl"),
            report=source(current / "report.json"), **item, abs_errors=errors,
            passed=all(v < 1e-9 for v in errors.values())))
    controls = REPO / "evidence/local/difficult_region_signal_20261010/uniform_controls"
    control_records = rows(controls / "scored_episodes.jsonl")
    control_report = read(controls / "report.json")
    for dataset, target in control_report["groups"].items():
        item = aggregate([r for r in control_records if r["dataset"] == dataset],
                         expected_from_report(target))
        errors = {a: abs(v - target["miou"][a]) for a, v in item["miou"].items()}
        checks.append(dict(id="uniform_controls_" + dataset, ledger=source(controls / "scored_episodes.jsonl"),
            report=source(controls / "report.json"), **item, abs_errors=errors,
            passed=all(v < 1e-9 for v in errors.values())))

    source_paths = [REPO / "scripts/frozen_dino_plan.py", REPO / "scripts/raw_feature_cache.py",
        REPO / "scripts/cached_dino.py", REPO / "src/ics/official_data.py", REPO / "src/ics/metrics.py",
        ASSETS / "setup/download_receipt.json", ASSETS / "setup/ics_loader_pool_readiness_receipt.json",
        ASSETS / "setup/asset_downloads/iSAID/isaid_extract_receipt.json",
        ASSETS / "setup/asset_downloads/isic_input/isic_prepare_receipt.json",
        ASSETS / "a/deepglobe200_preparation_20261009/preparation_receipt.json",
        ASSETS / "third_party/foris_official/models/__init__.py",
        ASSETS / "third_party/foris_official/inference.py",
        ASSETS / "third_party/foris_official/opts.py"]
    public_source_checks = []
    pinned_public_sha = {
        "fundus.py": "f5c3b51a682f96f8631407d56db45427a1796e53b897576e70fffc99a211e919",
        "isic.py": "94f4beafa1878edcdef52993d068eccf0d0ee5ac5e7bce0c22521f28c3bf7003",
        "lung.py": "1dec2739b824954c28d867952b251753fa0108aaba28d554b01b70815b63ef3a",
        "isaid.py": "3efe42eb51220a35f70e47bc265cb6a663b19a2cad4a429bcca56099c5cdc837",
        "__init__.py": "c29fca6df16a6a21464132d8b5c1c6ab152d3c83832f7e5ea0a9db9a93d41a63",
    }
    for name, public_sha in pinned_public_sha.items():
        path = ASSETS / "third_party/foris_official/datasets" / name
        public_source_checks.append(dict(**source(path), public_sha256=public_sha,
            passed=sha(path) == public_sha, verification="Public raw file fetched in this audit turn at pinned commit1aa02a11"))
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope="No model/inference launched; metadata and saved I/U only; other agents' files left untouched",
        official_five_protocol=dict(prepared=source(official), seed=frozen["seed"],
            n=frozen["n"], manifest_sha256=frozen["manifest_sha256"],
            manifest=actual_official_manifest, manifest_sha_passed=True, folds=folds),
        frozen_development_validation=manifest_inventory,
        cache=dict(root=str(pool.parent), profile=profile, profile_document=source(pool / "profile.json"),
            metadata_entries=len(entries), metadata_snapshot_sha256=metadata_digest.hexdigest(),
            branches=dict(branches), shape_dtype=dict(shape_dtype),
            provenance_dataset_entry_counts=dict(dataset_entries), provenance_role_counts=dict(roles),
            secondary_caches=secondary_caches,
            missing_payload_paths=missing_payloads, malformed_profile_paths=malformed,
            limits="Metadata-only discovery. Counts are input/crop entries, not episode or unique-photo coverage. No payload SHA/tensor revalidation in this audit."),
        checked_sources=[source(p) for p in source_paths], public_loader_identity=public_source_checks,
        saved_iu_recomputations=checks,
        all_numeric_checks_passed=all(c["passed"] for c in checks),
        live_absent_pools={name: not (ASSETS / "datasets/ics" / name).exists()
                           for name in ("LungSegmentation", "Fundus")})
    output = OUT / "inventory.json"
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), entries=len(entries),
        checks=len(checks), all_passed=record["all_numeric_checks_passed"],
        dataset_cache_tags=dict(dataset_entries), scores={c["id"]: c["miou"] for c in checks})))


if __name__ == "__main__":
    main()
