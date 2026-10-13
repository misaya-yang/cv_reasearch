#!/usr/bin/env python3
"""Finish the frozen native-task oracle audit; never produce algorithm masks.

Cross-protocol membership is sealed before inspecting saved performance. Exact
RGB/Mask equality is required, and native crops are never silently generalized.
All performance comes from the old six-arm integer ledger or its saved masks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

import audit_task_identifiability as audit
from ics.official_data import array_hash, decoded_rgb

OUT = audit.OUT
ATLAS_DIR = audit.ATLAS.parent
ARMS = ("FoRIS", "MEAN", "R0", "C", "N", "K")
FRAMES = ("cli", "original")
EXPECTED = audit.ASSETS / "a/consensus_supported_foreground2380_v2_20261011/expected_classes.json"
NATIVE = audit.ASSETS / "a/consensus_official_pascal_suim4000_v1_20261011"


def write(name, value):
    audit.base.write(OUT / name, value)


def freeze():
    dependencies = [Path(__file__), audit.ATLAS, OUT / "protocol.json", OUT / "manifest.json",
        OUT / "reference_membership.json", OUT / "reference_membership_seal.json",
        OUT / "query_certificates.json", OUT / "oracle_membership_seal.json", EXPECTED,
        ATLAS_DIR / "inventory_report.json"]
    for domain in ("pascal_part", "suim"):
        dependencies += [NATIVE / domain / "composition" / name for name in
            ("expected_classes.json", "scored_episodes.json", "report.json")]
    dependencies += audit.catalog_paths()
    config = {"state": "FROZEN_ORACLE_COMPLETION_BEFORE_CROSS_PROTOCOL_AND_PERFORMANCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(p): audit.base.sha(p) for p in dependencies},
        "cross_protocol_family": "COCO-20i and LVIS-92i native uncropped tasks on available identical decoded RGB; all allowed native categories",
        "cross_protocol_membership": "Exact full reference Mask equality; identical native R/Q pixel inputs; no approximate equality, forced crop, or category-name matching",
        "cross_protocol_scope": "Only local counterparts; missing counterpart is not a negative finding. Other domains have no shared native frame in these local sources.",
        "cross_protocol_semantic_caveat": "A different annotated answer is a protocol witness, not automatically a natural semantic ambiguity; annotation boundary differences require adjudication.",
        "subsets": "all, full task unique, full query identifiable, hard reference unique/alias, hard query identifiable with/without query-positive prior, soft reference unique, unresolved",
        "metric": "Class-pooled I/U using the frozen expected per-fold class slots; missing slots zero only for the fixed-slot descriptive metric. Empty subsets return null.",
        "matching": "For each hard alias, nearest available nonalias of same fold and global category in log GT fraction and log image aspect/area; prefer disjoint R/Q photos; no replacement",
        "matching_caveat": "Four aliases only; descriptive sensitivity, no causal or population inference",
        "four_edits": "N versus FoRIS and N versus same-count K; reuse native4000 saved gross edits, reconstruct old2200 from sealed masks and official nearest-aligned GT",
        "new_encoder_calls": 0, "raw_feature_payload_reads": 0, "new_algorithm_predictions": 0,
        "parameter_fits": 0}
    path = OUT / "analysis_protocol.json"
    if path.exists():
        old = audit.read(path)
        assert old["source_sha256"] == config["source_sha256"], "Frozen analysis inputs changed"
    else:
        shutil.copyfile(__file__, OUT / Path(__file__).name)
        write(path.name, config)
    return config


def union_audit(manifest, membership):
    """Only available COCO/LVIS native counterpart observations qualify."""
    start = time.perf_counter()
    catalog = audit.Catalog()
    lvis_by_id = defaultdict(list)
    for name in catalog.lvis_images:
        lvis_by_id[int(Path(name).stem)].append(name)
    coco_base = audit.ASSETS / "demo4_cache/data/COCO2014"
    coco_by_id = {int(p.stem.rsplit("_", 1)[-1]): p for p in
        sorted((coco_base / "annotations/val2014").glob("*.png"))}
    observations = {}
    file_hashes = {}

    def counterpart(row, role):
        source = row["dataset"]
        if source not in ("coco", "lvis") or row.get(role + "_crop") is not None:
            return None, "NO_SHARED_NATIVE_FRAME_IN_LOCAL_SCOPE"
        key = source, row[role + "_rgb_hash"]
        if key in observations:
            return observations[key]
        image_id = int(Path(row[role + "_path"]).stem.rsplit("_", 1)[-1])
        if source == "coco":
            candidates = [("lvis", audit.ASSETS / "datasets/ics/LVIS/coco" / name, name)
                for name in sorted(lvis_by_id.get(image_id, []))]
        else:
            label = coco_by_id.get(image_id)
            candidates = [] if label is None else [("coco", coco_base / "val2014" /
                (label.stem + ".jpg"), "val2014/" + label.stem + ".jpg")]
        status = "NO_LOCAL_ANNOTATED_COUNTERPART"
        for domain, path, name in candidates:
            if not path.exists():
                status = "COUNTERPART_RGB_UNAVAILABLE"
                continue
            rgb = np.asarray(decoded_rgb(path, None))
            digest = array_hash(rgb)
            file_hashes[str(path)] = audit.base.sha(path)
            if digest != row[role + "_rgb_hash"]:
                status = "COUNTERPART_DECODED_RGB_DIFFERS"
                continue
            alternative = dict(row, dataset=domain)
            alternative[role + "_path"] = str(path.relative_to(audit.ASSETS))
            alternative[role + "_photo_id"] = name
            alternative[role + "_crop"] = None
            masks, provenance, _ = catalog.masks(alternative, role)
            result = ({"dataset": domain, "path": str(path), "rgb_sha256": digest,
                "masks": masks, "provenance": provenance}, "EXACT_NATIVE_RGB_COUNTERPART")
            observations[key] = result
            return result
        observations[key] = (None, status)
        return None, status

    records, counts = [], defaultdict(Counter)
    for position, item in enumerate(manifest):
        row, ident = item["row"], item["id"]
        record = {"id": ident, "dataset": row["dataset"], "witnesses": []}
        alt_r, r_status = counterpart(row, "reference")
        record["reference_status"] = r_status
        if alt_r is None:
            record["status"] = r_status
        else:
            alt_q, q_status = counterpart(row, "query")
            record["query_status"] = q_status
            record["counterpart_dataset"] = alt_r["dataset"]
            record["reference_counterpart_path"] = alt_r["path"]
            record["reference_provenance"] = alt_r["provenance"]
            target_hash = membership[ident]["reference_signatures"]["full"]
            shape = tuple(row["reference_mask_size_hw"])
            matches = [task for task, mask in alt_r["masks"].items()
                if mask.shape == shape and mask.any() and array_hash(mask.astype(np.uint8)) == target_hash]
            record["reference_matching_tasks"] = matches
            record["reference_complete_mask_hash"] = target_hash
            if alt_q is None:
                record["status"] = "REFERENCE_COUNTERPART_ONLY_QUERY_UNAVAILABLE"
            else:
                record["status"] = "COMPLETE_SAME_INPUT_CROSS_PROTOCOL_CHECK"
                record["query_counterpart_path"] = alt_q["path"]
                record["query_provenance"] = alt_q["provenance"]
                if matches:
                    observed = audit.base.read_mask(row["query_mask_path"])
                    assert array_hash(observed.astype(np.uint8)) == row["query_mask_hash"]
                    for task in matches:
                        other = alt_q["masks"].get(task)
                        missing = other is None
                        if missing:
                            other = np.zeros_like(observed)
                        if other.shape != observed.shape:
                            record["witnesses"].append({"task": task, "status": "QUERY_MASK_FRAME_MISMATCH"})
                            continue
                        digest = array_hash(other.astype(np.uint8))
                        xor = int(np.count_nonzero(observed ^ other))
                        record["witnesses"].append({"native_task": row["dataset"] + ":" + str(row["global_class_id"]),
                            "alternative_task": alt_r["dataset"] + ":" + task,
                            "status": "ANNOTATED_QUERY_ANSWERS_DIFFER" if xor else "QUERY_ANSWERS_AGREE",
                            "same_reference_rgb": row["reference_rgb_hash"], "same_query_rgb": row["query_rgb_hash"],
                            "same_reference_mask": target_hash, "reference_crop": None, "query_crop": None,
                            "native_query_mask": row["query_mask_hash"], "alternative_query_mask": digest,
                            "native_query_area": int(observed.sum()), "alternative_query_area": int(other.sum()),
                            "xor_pixels": xor, "query_pixels": int(observed.size),
                            "alternative_query_catalog_absent": missing,
                            "query_positive_allows_alternative": bool(other.any()),
                            "semantic_status": "NOT_ADJUDICATED; native annotation definitions only"})
        counts[row["dataset"]][record["status"]] += 1
        counts[row["dataset"]]["reference_cross_protocol_match"] += bool(record.get("reference_matching_tasks"))
        counts[row["dataset"]]["different_query_annotation"] += any(w.get("xor_pixels", 0) > 0 for w in record["witnesses"])
        counts[row["dataset"]]["different_positive_query_annotation"] += any(w.get("xor_pixels", 0) > 0 and
            w["query_positive_allows_alternative"] for w in record["witnesses"])
        records.append(record)
        if (position + 1) % 400 == 0:
            print(json.dumps({"cross_protocol": position + 1, "seconds": round(time.perf_counter() - start, 2)}), flush=True)
    write("cross_protocol_membership.json", records)
    result = {"state": "EXACT_LOCAL_CROSS_PROTOCOL_MEMBERSHIP_SEALED_BEFORE_PERFORMANCE",
        "counts": {d: dict(v) for d, v in counts.items()}, "n": len(records),
        "members_sha256": audit.base.sha(OUT / "cross_protocol_membership.json"),
        "analysis_protocol_sha256": audit.base.sha(OUT / "analysis_protocol.json"),
        "counterpart_RGB_file_sha256": file_hashes, "seconds": time.perf_counter() - start,
        "new_encoder_calls": 0, "new_algorithm_predictions": 0,
        "coverage_caveat": "Counts apply to exact available local native counterpart pairs, not the whole task family or all five datasets"}
    write("cross_protocol_seal.json", result)
    return result


def load_edits(rows):
    """Native4000 already saved gross edits. Old2200 need saved-mask accounting."""
    start = time.perf_counter()
    native = {}
    for domain in ("pascal_part", "suim"):
        path = NATIVE / domain / "composition/scored_episodes.json"
        saved = audit.read(path)
        report = audit.read(path.parent / "report.json")
        assert audit.base.sha(path) == report["scored_records_sha256"]
        for frame in FRAMES:
            for row in saved[frame]:
                key = domain, frame, row["episode_id"]
                assert key not in native
                native[key] = row
    records, hashed, checks = {}, {}, Counter()
    for position, row in enumerate(rows):
        ident = row["binding_index"]
        values = {}
        if row["dataset"] in ("pascal_part", "suim"):
            for frame in FRAMES:
                saved = native[row["dataset"], frame, row["episode_id"]]
                assert saved["metrics"]["supported_mean"] == row["frames"][frame]["metrics"]["N"]
                values[frame] = {"N_minus_FoRIS": saved["edits"]["supported_mean"]["foris.crf"],
                    "N_minus_K": saved["edits"]["supported_mean"]["own_margin_sameK"]}
            checks["reused_native_saved_edit_tasks"] += 1
        else:
            gt_raw = audit.base.read_mask(row["query_mask_path"])
            assert array_hash(gt_raw.astype(np.uint8)) == row["query_mask_hash"]
            checks["query_GT_mask_reads_for_saved_output_accounting"] += 1
            archives = {}
            for arm in ("FoRIS", "N", "K"):
                binding = row["prediction_sources"][arm]
                path = binding["path"]
                if path not in hashed:
                    hashed[path] = audit.base.sha(path)
                assert hashed[path] == binding["sha256"]
                if path not in archives:
                    with np.load(path, allow_pickle=False) as archive:
                        archives[path] = {k: archive[k].copy() for k in
                            {b[f + "_key"] for b in row["prediction_sources"].values()
                                if b["path"] == path for f in FRAMES}}
            for frame in FRAMES:
                shape = (1024, 1024) if frame == "cli" else tuple(row["query_size_hw"])
                h, w = shape
                y, x = np.arange(h) * gt_raw.shape[0] // h, np.arange(w) * gt_raw.shape[1] // w
                gt = gt_raw[y[:, None], x[None, :]]
                masks = {}
                for arm in ("FoRIS", "N", "K"):
                    b = row["prediction_sources"][arm]
                    packed = archives[b["path"]][b[frame + "_key"]]
                    masks[arm] = np.unpackbits(packed)[:h * w].reshape(shape).astype(bool)
                    v = row["frames"][frame]["metrics"][arm]
                    assert int((masks[arm] & gt).sum()) == v["I"]
                    assert int((masks[arm] | gt).sum()) == v["U"]
                    checks["direct_saved_mask_IU"] += 2
                values[frame] = {}
                for baseline in ("FoRIS", "K"):
                    add = masks["N"] & ~masks[baseline]
                    remove = masks[baseline] & ~masks["N"]
                    values[frame]["N_minus_" + baseline] = [int((add & gt).sum()), int((add & ~gt).sum()),
                        int((remove & gt).sum()), int((remove & ~gt).sum())]
        for frame in FRAMES:
            for baseline in ("FoRIS", "K"):
                at, af, dt, df = values[frame]["N_minus_" + baseline]
                n = row["frames"][frame]["metrics"]["N"]
                b = row["frames"][frame]["metrics"][baseline]
                assert n["TP"] - b["TP"] == at - dt
                assert n["FP"] - b["FP"] == af - df
                checks["gross_edit_endpoint_reconciliation"] += 2
        records[ident] = values
        if (position + 1) % 400 == 0:
            print(json.dumps({"saved_edit_tasks": position + 1, "seconds": round(time.perf_counter() - start, 2)}), flush=True)
    write("saved_four_edits.json", [{"id": ident, "frames": values} for ident, values in records.items()])
    write("saved_four_edits_receipt.json", {"seconds": time.perf_counter() - start,
        "checks": dict(checks), "saved_prediction_files_checked": len(hashed),
        "source_prediction_hashes_sha256": hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest(),
        "four_edits_sha256": audit.base.sha(OUT / "saved_four_edits.json"),
        "semantics": "added true, added false, deleted true, deleted false; arithmetic output accounting, not causal attribution",
        "new_encoder_calls": 0, "raw_feature_payload_reads": 0, "new_algorithm_predictions": 0})
    return records


def summary(rows, expected, edits):
    if not rows:
        return {"n": 0, "metrics": None, "four_edits": None}
    by_class = defaultdict(list)
    for row in rows:
        by_class[str(row["fold"]), str(row["loader_class_id"])].append(row)
    photos = {role: {r[role + "_rgb_hash"] for r in rows} for role in ("reference", "query")}
    result = {"n": len(rows), "observed_class_slots": len(by_class),
        "official_class_slots": sum(map(len, expected.values())),
        "R_photos": len(photos["reference"]), "Q_photos": len(photos["query"]),
        "RQ_photos": len(photos["reference"] | photos["query"]), "frames": {}}
    for frame in FRAMES:
        official = dict.fromkeys(ARMS, 0.)
        observed = dict.fromkeys(ARMS, 0.)
        totals = {arm: Counter() for arm in ARMS}
        for fold, classes in expected.items():
            for cid in classes:
                members = by_class.get((str(fold), str(cid)), [])
                for arm in ARMS:
                    counts = Counter()
                    for r in members:
                        counts.update({k: r["frames"][frame]["metrics"][arm][k] for k in ("I", "U", "TP", "FP", "FN", "GT")})
                    iou = 100 * counts["I"] / counts["U"] if counts["U"] else 0.
                    official[arm] += iou / len(expected) / len(classes)
                    if members:
                        observed[arm] += iou / len(by_class)
                    totals[arm].update(counts)
        gross = {pair: [sum(edits[r["binding_index"]][frame][pair][k] for r in rows)
            for k in range(4)] for pair in ("N_minus_FoRIS", "N_minus_K")}
        result["frames"][frame] = {"fixed_official_slot_mIoU": official,
            "observed_class_slot_mIoU": observed, "pooled_integers": {a: dict(v) for a, v in totals.items()},
            "episode_mean_IoU": {a: float(np.mean([r["frames"][frame]["metrics"][a]["IoU"] for r in rows])) for a in ARMS},
            "four_edits": gross, "delta_N_minus_FoRIS_pp": official["N"] - official["FoRIS"],
            "delta_N_minus_K_pp": official["N"] - official["K"]}
    return result


def performance(manifest, members, queries):
    start = time.perf_counter()
    rows = [json.loads(line) for line in audit.ATLAS.open()]
    assert [r["binding_index"] for r in rows] == [m["id"] for m in manifest]
    assert len({r["binding_index"] for r in rows}) == 6200
    checks = Counter()
    for row, item in zip(rows, manifest):
        for key in ("episode_id", "dataset", "fold", "global_class_id", "reference_rgb_hash", "query_rgb_hash", "reference_mask_hash", "query_mask_hash"):
            assert row[key] == item["row"][key]
            checks["identity_fields"] += 1
        for frame in FRAMES:
            for arm in ARMS:
                v = row["frames"][frame]["metrics"][arm]
                assert all(isinstance(v[k], int) and v[k] >= 0 for k in ("I", "U", "TP", "FP", "FN", "GT"))
                assert v["I"] == v["TP"] and v["U"] == v["TP"] + v["FP"] + v["FN"]
                assert v["GT"] == v["TP"] + v["FN"]
                assert abs(v["IoU"] - 100 * v["I"] / max(1, v["U"])) < 1e-10
                checks["integer_metric_identities"] += 4
    edits = load_edits(rows)
    expected = audit.read(EXPECTED)
    for domain in ("pascal_part", "suim"):
        expected[domain] = audit.read(NATIVE / domain / "composition/expected_classes.json")[domain]
    inventory = audit.read(ATLAS_DIR / "inventory_report.json")
    result = {}
    for domain in audit.DOMAINS:
        all_rows = [r for r in rows if r["dataset"] == domain]
        def choose(predicate):
            return [r for r in all_rows if predicate(members[r["binding_index"]], queries[r["binding_index"]])]
        selections = {"all": all_rows,
            "full_reference_unique": choose(lambda m, q: m["status"] == "REFERENCE_TASK_UNIQUE"),
            "full_query_identifiable": choose(lambda m, q: q["certificates"]["full"]["status"] in ("QUERY_CONSENSUS", "REFERENCE_UNIQUE_IMPLIES_QUERY_TASK_UNIQUE")),
            "hard_reference_unique": choose(lambda m, q: len(m["matches"]["hard_joint"]) == 1),
            "hard_reference_alias": choose(lambda m, q: len(m["matches"]["hard_joint"]) > 1),
            "hard_query_identifiable_open": choose(lambda m, q: q["certificates"]["hard_joint"]["status"] in ("QUERY_CONSENSUS", "REFERENCE_UNIQUE_IMPLIES_QUERY_TASK_UNIQUE")),
            "hard_query_identifiable_positive": choose(lambda m, q: q["certificates"]["hard_joint"]["status"] == "REFERENCE_UNIQUE_IMPLIES_QUERY_TASK_UNIQUE" or q["certificates"]["hard_joint"].get("n_answers_query_positive") == 1),
            "soft_reference_unique": choose(lambda m, q: len(m["matches"]["soft_joint"]) == 1),
            "unresolved": choose(lambda m, q: m["status"].startswith("UNRESOLVED") or q["status"].startswith("UNRESOLVED"))}
        result[domain] = {name: summary(rs, expected[domain], edits) for name, rs in selections.items()}
        for name in ("full_reference_unique", "full_query_identifiable", "soft_reference_unique", "hard_query_identifiable_positive"):
            assert result[domain][name] == result[domain]["all"]
        for frame in FRAMES:
            for arm in ARMS:
                assert abs(result[domain]["all"]["frames"][frame]["fixed_official_slot_mIoU"][arm] - inventory["summary"][domain][frame]["miou"][arm]) < 1e-10
                checks["prior_full_ledger_mIoU_reconciliation"] += 1
    # Match only the four compressed-input aliases; full-input subset is all.
    aliases = [r for r in rows if len(members[r["binding_index"]]["matches"]["hard_joint"]) > 1]
    used, matches = set(), []
    def coordinates(r):
        h, w = r["query_size_hw"]
        frac = r["frames"]["original"]["metrics"]["FoRIS"]["GT"] / (h * w)
        return np.array([math.log(max(frac, 1 / (h * w))), math.log(h / w), math.log(h * w)])
    for alias in aliases:
        eligible = [r for r in rows if r["dataset"] == alias["dataset"] and r["fold"] == alias["fold"] and
            r["global_class_id"] == alias["global_class_id"] and r["binding_index"] not in used and
            len(members[r["binding_index"]]["matches"]["hard_joint"]) == 1]
        alias_photos = {alias["reference_rgb_hash"], alias["query_rgb_hash"]}
        disjoint = [r for r in eligible if not alias_photos & {r["reference_rgb_hash"], r["query_rgb_hash"]}]
        candidates = disjoint or eligible
        if not candidates:
            matches.append({"alias_id": alias["binding_index"], "control_id": None, "status": "NO_SAME_CLASS_NONALIAS"})
            continue
        control = min(candidates, key=lambda r: (float(np.abs(coordinates(r) - coordinates(alias)).sum()), r["binding_index"]))
        used.add(control["binding_index"])
        matches.append({"alias_id": alias["binding_index"], "control_id": control["binding_index"],
            "disjoint_RQ_photos": bool(disjoint), "global_class_id": alias["global_class_id"],
            "log_area_fraction_aspect_pixels_distance": float(np.abs(coordinates(control) - coordinates(alias)).sum()),
            "alias_coordinates": coordinates(alias).tolist(), "control_coordinates": coordinates(control).tolist(),
            "episode_deltas": {f: {"alias_N_minus_FoRIS": alias["frames"][f]["per_case_delta_N_minus_FoRIS"],
                "control_N_minus_FoRIS": control["frames"][f]["per_case_delta_N_minus_FoRIS"]} for f in FRAMES}})
    write("subset_performance.json", result)
    write("matched_compressed_alias_sensitivity.json", {"pairs": matches,
        "n": len(matches), "interpretation": "Descriptive same-class/size sensitivity only; not a causal ambiguity effect or a full-input oracle effect"})
    report = {"state": "COMPLETE_FROZEN_ORACLE_SUBSETS_AND_SAVED_PERFORMANCE", "n": len(rows),
        "counts": audit.COUNTS, "checks": dict(checks), "seconds": time.perf_counter() - start,
        "subset_performance_sha256": audit.base.sha(OUT / "subset_performance.json"),
        "new_encoder_calls": 0, "raw_feature_payload_reads": 0, "new_algorithm_predictions": 0,
        "full_query_alias_n": 0, "hard_reference_alias_n": len(aliases)}
    write("analysis_receipt.json", report)
    return report


def main():
    start = time.perf_counter()
    freeze()
    reference_seal = audit.read(OUT / "reference_membership_seal.json")
    oracle_seal = audit.read(OUT / "oracle_membership_seal.json")
    assert audit.base.sha(OUT / "reference_membership.json") == reference_seal["membership_sha256"]
    assert audit.base.sha(OUT / "query_certificates.json") == oracle_seal["query_certificates_sha256"]
    manifest = audit.read(OUT / "manifest.json")
    members = {r["id"]: r for r in audit.read(OUT / "reference_membership.json")}
    queries = {r["id"]: r for r in audit.read(OUT / "query_certificates.json")}
    assert len(manifest) == len(members) == len(queries) == 6200
    cross = union_audit(manifest, members)
    perf = performance(manifest, members, queries)
    write("analysis_completion.json", {"state": "COMPLETE_ORACLE_SUBSETS_AND_LOCAL_UNION_AUDIT_PENDING_FINAL_VERIFICATION",
        "source_thread": "01a12dda-fbdf-79b3-9f92-12975fb0438e", "seconds": time.perf_counter() - start,
        "cross_protocol_seconds": cross["seconds"], "saved_performance_seconds": perf["seconds"],
        "new_encoder_calls": 0, "raw_feature_payload_reads": 0, "new_algorithm_predictions": 0,
        "scientific_goal": "Five-domain unified Strong method is not established by this diagnostic audit"})
    print(json.dumps({"cross_counts": cross["counts"], "performance": perf}), flush=True)


if __name__ == "__main__":
    main()
