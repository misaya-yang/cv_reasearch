"""Sole post-seal scorer/direct verifier. Writing/importing it never opens GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT.parent / "cv_data"
sys.path.insert(0, str(ROOT / "src"))
from ics.official_data import array_hash

PROTOCOL_SHA = "9135633b9d719fcb7a41eb587aa4461af199966a52da04e167786677fb4e1605"
NEW = ("background_vetted_complete", "unvetted_anchor_complete", "reference_only_protected_complete", "same_edit_budget_foris_margin")
OLD_ATLAS = {"B": "FoRIS", "G": "MEAN", "R0": "R0", "C": "C", "N": "N", "K": "K"}
OLD_BINDING = {"B": "B", "G": "G", "R0": "R", "C": "C", "N": "N", "K": "ownK"}
OLD4 = {
    "old4fd6_anchor_relative_complete": "anchor_relative_complete",
    "old4fd6_reference_relative_complete": "reference_relative_complete",
    "old4fd6_same_edit_budget_foris_margin": "same_edit_budget_foris_margin",
}
OLD4_MAIN = "old4fd6_anchor_relative_complete"
MAIN_CONTROLS = ("C", "N", OLD4_MAIN, NEW[1], NEW[2], NEW[3])
ARMS = tuple(OLD_ATLAS) + tuple(OLD4) + NEW
EXPECTED_COUNTS = {"coco": 200, "paco_part": 600, "lvis": 1400, "pascal_part": 2000, "suim": 2000}
EXPECTED_SLOTS = {"coco": 40, "paco_part": 303, "lvis": 920, "pascal_part": 56, "suim": 7}
POLICIES = (
    (DATA / "a/consensus_supported_foreground2380_v2_20261011/expected_classes.json", "2fb86fc8ab80299427a8a140081c98ce458439d7b89b9a6a51c19a26e68d9e7f", ("coco", "paco_part", "lvis")),
    (DATA / "a/consensus_official_pascal_suim4000_v1_20261011/pascal_part/composition/expected_classes.json", "e437d57a1b344544b3e58d73bdbbde5bcefcc8e4aacc42e3fcfaf56d6c081116", ("pascal_part",)),
    (DATA / "a/consensus_official_pascal_suim4000_v1_20261011/suim/composition/expected_classes.json", "b95cb0493ea1758ed7c55047066438f105b5bf2bc3f9240cea3405659323365f", ("suim",)),
)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    p = Path(path)
    temp = p.with_suffix(p.suffix + ".pending")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(p)


def resize(a, shape, mode="bilinear"):
    x = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)[None, None]
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(x, size=tuple(shape), mode=mode, **kw)[0, 0].numpy()


def unpack(z, key, shape, checks):
    a = z[key]
    n = int(np.prod(shape))
    assert a.dtype == np.uint8 and a.ndim == 1 and a.size == (n + 7) // 8, (key, shape, a.shape, a.dtype)
    padding = a.size * 8 - n
    assert not padding or (int(a[-1]) & ((1 << padding) - 1)) == 0, key
    checks["packed_schema_and_padding"] += 1
    return np.unpackbits(a, count=n).reshape(shape).astype(bool)


def expected_classes():
    result = {}
    for path, digest, datasets in POLICIES:
        assert sha(path) == digest, path
        data = read(path)
        for dataset in datasets:
            result[dataset] = {str(f): list(map(str, cs)) for f, cs in data[dataset].items()}
    assert {d: sum(map(len, fs.values())) for d, fs in result.items()} == EXPECTED_SLOTS
    return result


def gate_all_masks(checks):
    protocol_path = HERE / "protocol.json"
    assert sha(protocol_path) == PROTOCOL_SHA
    protocol = read(protocol_path)
    assert tuple(protocol["new_arms"]) == NEW and protocol["scope"]["new_final_mask_count"] == 49600
    seal, config = read(HERE / "sealed.json"), read(HERE / "config.json")
    assert seal["state"] == "ALL6200_FOUR_COMPLETE_ARMS_TWO_FRAMES_SEALED_BEFORE_GT"
    assert seal["n"] == config["n"] == 6200 and seal["mask_count"] == 49600 and len(seal["records"]) == 6200
    assert tuple(config["arms"]) == NEW
    assert Path(config["protocol_path"]).resolve() == protocol_path.resolve()
    assert seal["protocol_sha256"] == config["protocol_sha256"] == PROTOCOL_SHA
    assert seal["config_sha256"] == sha(HERE / "config.json")
    assert sha(HERE / "protocol.frozen.json") == PROTOCOL_SHA
    bindings = protocol["source_bindings"]
    input_root = Path(config["input_root"]).resolve()
    assert input_root == (ROOT / bindings["execution_inputs_path"]).resolve().parent
    preflight_path = ROOT / bindings["execution_preflight_receipt_path"]
    assert sha(preflight_path) == bindings["execution_preflight_receipt_sha256"]
    preflight = read(preflight_path)
    for name, field in (("inputs.jsonl", "inputs_sha256"), ("score_bindings.jsonl", "score_bindings_sha256"), ("raw_metadata.json", "raw_metadata_sha256")):
        assert sha(input_root / name) == config["input_sha256"][name] == preflight[field] == bindings["execution_" + field]
    for path, digest in config["source_sha256"].items():
        assert sha(path) == digest, path
    historical = bindings["completed_4fd6"]
    assert sha(ROOT / historical["protocol"]["path"]) == historical["protocol"]["sha256"]
    for asset in historical["assets"].values():
        assert sha(ROOT / asset["path"]) == asset["sha256"], asset["path"]
    previous_verify = read(ROOT / historical["assets"]["verification.json"]["path"])
    previous_seal = read(ROOT / historical["assets"]["sealed.json"]["path"])
    assert previous_verify["episodes_sha256"] == historical["assets"]["episodes.jsonl"]["sha256"]
    assert previous_verify["report_sha256"] == historical["assets"]["report.json"]["sha256"]
    assert previous_verify["state"] == "ALL6200_NEW_THREE_ARMS_TWO_FRAMES_SCORED_DIRECTLY_VERIFIED"
    assert previous_seal["protocol_sha256"] == historical["protocol"]["sha256"]
    assert previous_seal["n"] == 6200 and previous_seal["mask_count"] == 37200
    scores, inputs = lines(input_root / "score_bindings.jsonl"), lines(input_root / "inputs.jsonl")
    assert len(scores) == len(inputs) == 6200
    assert Counter(row["row"]["dataset"] for row in scores) == EXPECTED_COUNTS
    expected_keys = {"original_hw"} | {f + "/" + arm for f in ("cli", "original") for arm in NEW}
    filenames, record_hashes = set(), {}
    for i, (bound, source, entry) in enumerate(zip(scores, inputs, seal["records"])):
        row = bound["row"]
        assert bound["index"] == source["index"] == entry["index"] == i
        assert bound["binding_index"] == source["binding_index"] == entry["binding_index"]
        assert row["episode_id"] == source["row"]["episode_id"] == entry["episode_id"]
        assert row["query_size_hw"] == source["row"]["query_size_hw"]
        assert entry["filename"] == f"{i:06}.npz"
        path, rec_path = HERE / "predictions" / entry["filename"], HERE / "records" / f"{i:06}.json"
        assert sha(path) == entry["prediction_sha256"], path
        record_hashes[i] = sha(rec_path)
        rec = read(rec_path)
        assert rec["prediction_sha256"] == entry["prediction_sha256"]
        assert rec["index"] == i and rec["episode_id"] == row["episode_id"] and rec["binding_index"] == bound["binding_index"]
        assert rec["source_N"] == bound["old_final_masks"]["N"]
        assert rec["query_GT_reads"] == rec["new_encoder_calls"] == rec["new_CRF_calls"] == 0
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == expected_keys
            assert z["original_hw"].tolist() == row["query_size_hw"]
            for frame, shape in (("cli", (1024, 1024)), ("original", tuple(row["query_size_hw"]))):
                for arm in NEW:
                    unpack(z, frame + "/" + arm, shape, checks)
        filenames.add(entry["filename"])
        checks["complete_preGT_packet_SHA_identity"] += 1
    assert {path.name for path in (HERE / "predictions").glob("*.npz")} == filenames
    assert {path.name for path in (HERE / "records").glob("*.json")} == {f"{i:06}.json" for i in range(6200)}
    checks["all49600_masks_gated_before_truth"] = 49600
    return protocol, seal, scores, inputs, record_hashes


def old_numbers(protocol, scores, checks):
    path = ROOT / protocol["source_bindings"]["atlas_membership_path"]
    assert sha(path) == protocol["source_bindings"]["atlas_membership_sha256"]
    by_index = {r["binding_index"]: r for r in lines(path)}
    assert len(by_index) == 6200
    for s in scores:
        r, old = s["row"], by_index[s["binding_index"]]
        for field in ("episode_id", "dataset", "fold", "loader_class_id", "global_class_id", "query_rgb_hash", "reference_mask_hash"):
            assert r[field] == old[field], (s["binding_index"], field)
        checks["old_atlas_integer_identity_binding"] += 1
    return by_index


def previous_numbers(protocol, scores, atlas, checks):
    assets = protocol["source_bindings"]["completed_4fd6"]["assets"]
    seal_path = ROOT / assets["sealed.json"]["path"]
    seal = read(seal_path)
    result = {}
    with (ROOT / assets["episodes.jsonl"]["path"]).open() as stream:
        for i, line in enumerate(stream):
            old = json.loads(line)
            source, entry = scores[i], seal["records"][i]
            row = source["row"]
            assert old["index"] == source["index"] == entry["index"] == i
            assert old["binding_index"] == source["binding_index"] == entry["binding_index"]
            assert old["episode_id"] == row["episode_id"] == entry["episode_id"]
            assert old["dataset"] == row["dataset"] and old["fold"] == int(row["fold"])
            assert old["class_id"] == old["loader_class_id"] == int(row["loader_class_id"])
            assert old["global_class_id"] == row["global_class_id"]
            metrics = {}
            for frame in ("cli", "original"):
                saved = old["frames"][frame]["metrics"]
                for arm, name in OLD_ATLAS.items():
                    assert saved[arm] == atlas[source["binding_index"]]["frames"][frame]["metrics"][name]
                metrics[frame] = {prefix: dict(saved[name]) for prefix, name in OLD4.items()}
            assert entry["filename"] == f"{i:06}.npz"
            result[i] = {"metrics": metrics, "prediction_path": seal_path.parent / "predictions" / entry["filename"],
                         "prediction_sha256": entry["prediction_sha256"]}
            checks["previous3_complete_IU_identity_and_old6_binding"] += 1
    assert len(result) == len(scores) == 6200
    return result


def truth(row, checks):
    with Image.open(row["query_mask_path"]) as im:
        a = (np.asarray(im.convert("L")) > 0).astype(np.uint8)
    assert list(a.shape) == row["query_mask_size_hw"]
    assert array_hash(a) == row["query_mask_hash"]
    checks["query_truth_file_identity"] += 1
    return a


def old_support(source, shape, frame, hashed, checks):
    masks = {}
    for name in ("B", "G", "R0", "C", "N"):
        binding = source["old_final_masks"][OLD_BINDING[name]]
        if binding["path"] not in hashed:
            assert sha(binding["path"]) == binding["sha256"]
            hashed.add(binding["path"])
        with np.load(binding["path"], allow_pickle=False) as z:
            masks[name] = unpack(z, binding[frame + "_key"], shape, checks)
    assert source["old_final_masks"]["R"][frame + "_key"] == frame + "/anchor.global"
    return masks


def source_margin(binding, shape, hashed):
    if binding["path"] not in hashed:
        assert sha(binding["path"]) == binding["sha256"]
        hashed.add(binding["path"])
    with np.load(binding["path"], allow_pickle=False) as z:
        score = z[binding["key"]].copy().reshape(64, 64).astype(np.float64)
    assert np.isfinite(score).all()
    span = float(np.ptp(score))
    d = 2 * resize((score - score.min()) / max(span, 1e-6), shape) - 1
    return d, np.abs(d) if span else np.zeros(shape)


def verify_budget(new, masks, d, conf, producer_record, frame, checks):
    B, G, R0, N = (masks[k] for k in ("B", "G", "R0", "N"))
    main, budget, reference_only = new[NEW[0]], new[NEW[3]], new[NEW[2]]
    assert not np.any(N & ~(B | G | R0))
    add, delete = main & ~B, B & ~main
    Kadd, Kdel = int(add.sum()), int(delete.sum())
    assert int((budget & ~B).sum()) == Kadd and int((B & ~budget).sum()) == Kdel
    Da, Dd = ~B & (G | R0 | N), B & ~N
    agreed, protected = (B == G) & (G == R0), B & N
    frame_record = producer_record["frames"][frame]
    assert frame_record["D_add"] == int(Da.sum()) and frame_record["D_delete"] == int(Dd.sum())
    assert frame_record["B_N_common"] == int(protected.sum()) and frame_record["agreed_pixels"] == int(agreed.sum())
    stats = producer_record["dense_stats"]
    mf, mb, mau = (float(stats[k]) for k in ("MF", "MB", "MA_unvetted"))
    assert all(math.isfinite(x) for x in (mf, mb, mau)) and mf + mb == 4096 and 0 <= mau <= 4096
    ref_reason = stats["reference_reason"]
    if ref_reason is None:
        assert mf > 0 and mb > 0 and math.isfinite(stats["source_scale_squared"]) and stats["source_scale_squared"] > 0
        mav = float(stats["MA_vetted"])
        assert math.isfinite(mav) and 0 <= mav <= mau
        assert stats["reference_rejected_anchor_mass"] == mau - mav
        assert 0 <= stats["reference_positive_native_tokens"] <= 4096
        for name in ("reference_FG", "reference_BG"):
            assert len(stats[name + "_index_sha256"]) == 64
        for name, mass in (("vetted", mav), ("unvetted", mau)):
            reason = stats[name + "_reason"]
            if reason is None:
                assert mass > 0 and len(stats[name + "_query_FG_index_sha256"]) == 64
            elif reason == "query_anchor_empty_or_rejected":
                assert mass == 0
            else:
                assert reason == "numeric_cue_invalid"
    else:
        assert ref_reason in ("reference_role_unavailable", "reference_scale_invalid", "numeric_cue_invalid")
        assert stats["vetted_reason"] == stats["unvetted_reason"] == ref_reason
    fallback = {NEW[0]: ref_reason is not None or stats["vetted_reason"] is not None,
                NEW[1]: ref_reason is not None or stats["unvetted_reason"] is not None,
                NEW[2]: ref_reason is not None}
    fallback[NEW[3]] = fallback[NEW[0]]
    for arm in NEW:
        prediction = new[arm]
        assert np.array_equal(prediction[agreed], B[agreed])
        assert np.array_equal(prediction[protected], B[protected])
        assert not np.any((prediction & ~B) & ~Da) and not np.any((B & ~prediction) & ~Dd)
        assert not np.any((prediction != B) & (conf >= .5))
        recorded = frame_record["edit_counts"][arm]
        assert recorded["add"] == int((prediction & ~B).sum()) and recorded["delete"] == int((B & ~prediction).sum())
        if fallback[arm]:
            assert np.array_equal(prediction, B), (arm, frame, "fallback must preserve B")
        checks["same_frame_conflict_N_protection_fallback_and_count_binding"] += 1
    for arm in NEW[:2]:
        assert not np.any((new[arm] & ~B) & ~(reference_only & ~B))
        assert not np.any((B & ~new[arm]) & ~(B & ~reference_only))
        checks["direct_reference_only_edit_superset"] += 1
    rebuilt = B.copy()
    for eligible, K, descending, value in ((Da & (conf < .5), Kadd, True, True), (Dd & (conf < .5), Kdel, False, False)):
        ids = np.flatnonzero(eligible.ravel())
        assert K <= len(ids)
        rank = np.argsort(-d.ravel()[ids] if descending else d.ravel()[ids], kind="stable")
        rebuilt.ravel()[ids[rank[:K]]] = value
    assert np.array_equal(rebuilt, budget)
    checks["direct_budget_positions_and_row_major_ties"] += 1
    checks["query_bank_small_record_contract_no_dense_recompute"] += 1
    return {"K_add": Kadd, "K_delete": Kdel, "same_frame": frame,
            "fallback": fallback, "N_protected_pixels": int(protected.sum())}


def new_counts(p, B, t, old_B, checks):
    I, U = int(np.count_nonzero(p & t)), int(np.count_nonzero(p | t))
    GT, P = int(t.sum()), int(p.sum())
    TP, FP, FN = I, P - I, GT - I
    assert U == TP + FP + FN
    add, delete = p & ~B, B & ~p
    edits = {"add_TP": int((add & t).sum()), "add_FP": int((add & ~t).sum()),
             "delete_TP": int((delete & t).sum()), "delete_FP": int((delete & ~t).sum())}
    assert TP == old_B["TP"] + edits["add_TP"] - edits["delete_TP"]
    assert FP == old_B["FP"] + edits["add_FP"] - edits["delete_FP"]
    assert old_B["GT"] == GT == TP + FN
    assert P == old_B["TP"] + old_B["FP"] + int(add.sum()) - int(delete.sum())
    checks["direct_new_AND_OR_IU_and_confusion"] += 1
    checks["gross_edits_oldB_confusion_closure"] += 1
    return {"I": I, "U": U, "TP": TP, "FP": FP, "FN": FN, "GT": GT, "IoU": 100 * I / max(U, 1)}, edits


def direct_gross(p, baseline, t, actual, prior, checks):
    """Finite complete-method contrast; no baseline I/U rescoring."""
    add, delete = p & ~baseline, baseline & ~p
    e = {"add_TP": int((add & t).sum()), "add_FP": int((add & ~t).sum()),
         "delete_TP": int((delete & t).sum()), "delete_FP": int((delete & ~t).sum())}
    assert actual["TP"] == prior["TP"] + e["add_TP"] - e["delete_TP"]
    assert actual["FP"] == prior["FP"] + e["add_FP"] - e["delete_FP"]
    assert actual["GT"] == prior["GT"] == int(t.sum())
    checks["main_finite_control_direct_gross_TP_FP_closure"] += 1
    return e


def aggregate(records, expected):
    groups = defaultdict(list)
    for r in records:
        groups[(str(r["fold"]), str(r["class_id"]))].append(r)
    assert not set(groups) - {(f, c) for f, cs in expected.items() for c in cs}
    per_class, per_fold, missing = {}, {}, {}
    for f, cs in expected.items():
        missing[f] = [c for c in cs if not groups.get((f, c))]
        per_fold[f] = dict.fromkeys(ARMS, 0.0)
        for c in cs:
            g = groups.get((f, c), [])
            values = {}
            for arm in ARMS:
                I = sum(r["iu"][arm][0] for r in g)
                U = sum(r["iu"][arm][1] for r in g)
                values[arm] = 100 * I / max(U, 1)
                per_fold[f][arm] += values[arm] / len(cs)
            per_class[f + ":" + c] = {"n": len(g), "IoU_pct": values,
                "official_contribution_pct": {a: v / len(expected) / len(cs) for a, v in values.items()}}
    point = {a: float(np.mean([v[a] for v in per_fold.values()])) for a in ARMS}
    return {"n": len(records), "miou_pct": point, "per_fold": per_fold, "per_class": per_class,
            "expected_slots": sum(map(len, expected.values())), "observed_slots": len(groups),
            "missing_classes": missing, "statistic": "per-class pooled I/U -> class mean -> fold mean; missing official slots zero",
            "total_confusion": {a: {k: sum(r["metrics"][a][k] for r in records) for k in ("TP", "FP", "FN")} for a in ARMS},
            "gross_vs_B": {a: {k: sum(r["edits_vs_B"][a][k] for r in records) for k in ("add_TP", "add_FP", "delete_TP", "delete_FP")} for a in NEW},
            "gross_main_vs_controls": {a: {k: sum(r["gross_main_vs_controls"][a][k] for r in records) for k in ("add_TP", "add_FP", "delete_TP", "delete_FP")} for a in MAIN_CONTROLS}}


def score():
    started, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(4)
    checks = Counter()
    protocol, seal, scores, inputs, record_hashes = gate_all_masks(checks)
    gate_seconds = time.monotonic() - started
    expected = expected_classes()
    atlas = old_numbers(protocol, scores, checks)
    previous = previous_numbers(protocol, scores, atlas, checks)
    groups, hashed = defaultdict(list), set()
    old_prediction_bindings = Counter()
    pending = HERE / "episodes.jsonl.pending"
    with pending.open("w") as stream:
        for i, (source, inp, entry) in enumerate(zip(scores, inputs, seal["records"])):
            row = source["row"]
            class_id, fold = int(row["loader_class_id"]), int(row["fold"])
            assert str(class_id) in expected[row["dataset"]][str(fold)]
            old = atlas[source["binding_index"]]
            rec_path = HERE / "records" / f"{i:06}.json"
            assert sha(rec_path) == record_hashes[i]
            producer_record = read(rec_path)
            for arm, atlas_name in OLD_ATLAS.items():
                assert old["prediction_sources"][atlas_name] == source["old_final_masks"][OLD_BINDING[arm]]
                old_prediction_bindings[arm] += 1
            prior = previous[i]
            assert sha(prior["prediction_path"]) == prior["prediction_sha256"]
            raw_truth = truth(row, checks)
            item = {"index": i, "binding_index": source["binding_index"], "episode_id": row["episode_id"],
                    "dataset": row["dataset"], "fold": fold, "class_id": class_id, "loader_class_id": class_id,
                    "global_class_id": row["global_class_id"], "query_photo_id": row["query_photo_id"],
                    "reference_photo_id": row["reference_photo_id"], "frames": {},
                    "producer_dense_stats": producer_record["dense_stats"], "producer_record_sha256": record_hashes[i]}
            path = HERE / "predictions" / entry["filename"]
            assert sha(path) == entry["prediction_sha256"]
            with np.load(path, allow_pickle=False) as z, np.load(prior["prediction_path"], allow_pickle=False) as previous_z:
                assert previous_z["original_hw"].tolist() == row["query_size_hw"]
                for frame, shape in (("cli", (1024, 1024)), ("original", tuple(row["query_size_hw"]))):
                    target = resize(raw_truth, shape, "nearest") > .5
                    saved = old["frames"][frame]["metrics"]
                    metrics = {arm: dict(saved[name]) for arm, name in OLD_ATLAS.items()}
                    metrics.update({arm: dict(value) for arm, value in prior["metrics"][frame].items()})
                    assert all(metric["GT"] == int(target.sum()) for metric in metrics.values())
                    support = old_support(source, shape, frame, hashed, checks)
                    d, conf = source_margin(inp["source_score64"], shape, hashed)
                    predictions = {arm: unpack(z, frame + "/" + arm, shape, checks) for arm in NEW}
                    previous_main = unpack(previous_z, frame + "/" + OLD4[OLD4_MAIN], shape, checks)
                    budget = verify_budget(predictions, support, d, conf, producer_record, frame, checks)
                    edits = {}
                    for arm in NEW:
                        metrics[arm], edits[arm] = new_counts(predictions[arm], support["B"], target, metrics["B"], checks)
                    baselines = {"C": support["C"], "N": support["N"], OLD4_MAIN: previous_main,
                                 NEW[1]: predictions[NEW[1]], NEW[2]: predictions[NEW[2]], NEW[3]: predictions[NEW[3]]}
                    assert tuple(baselines) == MAIN_CONTROLS
                    control_gross = {arm: direct_gross(predictions[NEW[0]], baseline, target, metrics[NEW[0]], metrics[arm], checks)
                                     for arm, baseline in baselines.items()}
                    frame_record = {"metrics": metrics, "iu": {arm: [metrics[arm]["I"], metrics[arm]["U"]] for arm in ARMS},
                                    "edits_vs_B": edits, "gross_main_vs_controls": control_gross, "budget": budget}
                    item["frames"][frame] = frame_record
                    groups[(row["dataset"], frame)].append({**{k: item[k] for k in ("episode_id", "fold", "class_id", "query_photo_id", "reference_photo_id")}, **frame_record})
                    del target, support, d, conf, predictions, previous_main
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                stream.flush()
                print(json.dumps({"stage": "post_seal_score_direct_verify", "completed": i + 1, "n": 6200}), flush=True)
    pending.replace(HERE / "episodes.jsonl")
    summaries = {dataset: {frame: aggregate(groups[(dataset, frame)], expected[dataset]) for frame in ("cli", "original")} for dataset in EXPECTED_COUNTS}
    assert len(groups) == 10 and all(len(groups[(dataset, frame)]) == n for dataset, n in EXPECTED_COUNTS.items() for frame in ("cli", "original"))
    historical = protocol["source_bindings"]["completed_4fd6"]["assets"]
    old_report = read(ROOT / historical["report.json"]["path"])
    for dataset, frames in summaries.items():
        for frame, summary in frames.items():
            for arm, old_name in {**{arm: arm for arm in OLD_ATLAS}, **OLD4}.items():
                assert abs(summary["miou_pct"][arm] - old_report["groups"][dataset][frame]["miou_pct"][old_name]) < 1e-10
                checks["old9_complete_official_points_bound_to_verified_report"] += 1
    assert checks["direct_new_AND_OR_IU_and_confusion"] == 49600
    assert checks["main_finite_control_direct_gross_TP_FP_closure"] == 6200 * 2 * len(MAIN_CONTROLS)
    report = {"state": "ALL6200_NEW_FOUR_ARMS_TWO_FRAMES_SCORED_DIRECTLY_VERIFIED", "n": 6200, "new_masks": 49600,
              "arms": list(ARMS), "new_arms": list(NEW), "old4fd6_arm_mapping": OLD4, "groups": summaries,
              "intervals_state": "PENDING_SEPARATE_CONTINUATION_NOT_A_SCORE_GATE",
              "protocol_sha256": PROTOCOL_SHA, "seal_sha256": sha(HERE / "sealed.json"), "episodes_sha256": sha(HERE / "episodes.jsonl"),
              "scorer_source_sha256": sha(__file__), "expected_class_sources": {str(path): digest for path, digest, _ in POLICIES},
              "old_IU_policy": "Old6 complete metrics copied from the SHA-bound atlas; old4fd6 three complete metrics copied from its verified ledger with an explicit prefix. No old GT IU rescoring",
              "old4fd6_sources": historical,
              "scope": "Exposed fixed6200, CLI/original independent frame composition, fixed official classes missing0, no domain or frame winner selection"}
    write(HERE / "report.json", report)
    verification = {"state": report["state"], "n": 6200, "checks": dict(checks),
                    "old_atlas_metric_bindings": dict(old_prediction_bindings), "old4fd6_complete_rows_reused": 6200,
                    "query_GT_reads": checks["query_truth_file_identity"], "new_mask_count": 49600,
                    "old_IU_rescoring": 0, "new_raw_reads": 0, "new_fits": 0, "new_predictions": 0,
                    "query_anchor_verification": "Frozen-code and postseal small-record mass/reason/hash consistency plus final B-fallback identity; no raw/dense cue replay",
                    "preGT_complete_gate_seconds": gate_seconds, "wall_seconds": time.monotonic() - started,
                    "CPU_seconds": time.process_time() - cpu, "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "report_sha256": sha(HERE / "report.json"), "episodes_sha256": sha(HERE / "episodes.jsonl"),
                    "seal_sha256": sha(HERE / "sealed.json"), "protocol_sha256": PROTOCOL_SHA, "scorer_source_sha256": sha(__file__)}
    write(HERE / "verification.json", verification)
    print(json.dumps({"state": report["state"], "groups": {dataset: {frame: summary["miou_pct"] for frame, summary in frames.items()} for dataset, frames in summaries.items()}, "verification": verification}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("score",), default="score")
    parser.parse_args()
    score()


if __name__ == "__main__":
    main()
