#!/usr/bin/env python3
"""Project sealed fixed1200 masks to original DEV241 and account from raw DINO.

Assembly reads masks/receipts only and seals all241 snapshots before GT scoring.
No model, encoder, threshold search, fitting, or changes to source artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from ics.experiment import metric, photo_groups, sha, summarize, unpack

FINE_ARMS = ("native", "rcg", "mean.control", "rcg64.control", "fine.rcg16.control", "fine.rcg64")
RAW = "model.raw_nn"
INSID3 = "insid3.release_crf.control"
FORWARD_NATIVE = "native.forward_replay.control"
EDIT_ORDER = ("add_TP", "add_FP", "delete_TP", "delete_FP")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(row):
    return int(row["c"]), Path(row["support"]).name, Path(row["query"]).name


def check(path, digest):
    if sha(path) != digest:
        raise ValueError("Changed sealed source: " + str(path))


def source(path, expected_n):
    seal = json.loads((path / "sealed.json").read_text())
    if seal.get("state") != "ALL_PREDICTIONS_SEALED":
        raise ValueError("Incomplete source: " + str(path))
    check(path / "manifest.json", seal["manifest_sha256"])
    rows = json.loads((path / "manifest.json").read_text())
    if len(rows) != expected_n or set(seal["predictions"]) != {r["key"] for r in rows}:
        raise ValueError("Incomplete source key set")
    by_identity = {identity(r): r for r in rows}
    if len(by_identity) != len(rows):
        raise ValueError("Ambiguous identity mapping")
    metadata = {}
    for field, filename in (("protocol_sha256", "protocol.json"), ("config_sha256", "config.json"), ("runtime_sha256", "runtime.json")):
        if field in seal:
            check(path / filename, seal[field])
            metadata[filename] = json.loads((path / filename).read_text())
    return dict(path=path, seal=seal, seal_hash=sha(path / "sealed.json"), rows=rows,
                identities=by_identity, metadata=metadata)


def packed(path, digest, names):
    check(path, digest)
    with np.load(path, allow_pickle=False) as z:
        result = {name: z[name].copy() for name in names}
    if any(v.dtype != np.uint8 or v.shape != (131072,) for v in result.values()):
        raise ValueError("Expected packed1024 comparison masks")
    return result


def assemble_one(job):
    row, fine_row, paths, hashes, include_insid3 = job
    original = packed(paths["original"], hashes["original"], ("native",))
    family = packed(paths["family"], hashes["family"], ("origin", "native", "rcg"))
    stage = packed(paths["stage"], hashes["stage"], (RAW,))
    fine = packed(paths["fine"], hashes["fine"], FINE_ARMS)
    if not np.array_equal(family["origin"], stage[RAW]):
        raise ValueError("Actual origin is not the sealed raw stage NN mask: " + row["key"])
    differences = {name: int(np.unpackbits(family[name] ^ fine[name]).sum()) for name in ("native", "rcg")}
    if any(differences.values()):
        raise ValueError("Original family/public baseline bit parity failed: " + row["key"])
    check(paths["old_packet"], hashes["old_packet"])
    check(paths["new_packet"], hashes["new_packet"])
    # Only the old native comparison is indexed; query truth is never indexed here.
    with np.load(paths["old_packet"], allow_pickle=False) as z:
        old_native = z["native"].copy()
    if not np.array_equal(old_native, fine["native"]):
        raise ValueError("Original native packet/public native bit parity failed")
    masks = dict(fine)
    masks[RAW] = family["origin"]
    masks[FORWARD_NATIVE] = original["native"]
    if include_insid3:
        released = packed(paths["insid3"], hashes["insid3"], (INSID3, "origin", "native", "rcg"))
        for name in ("origin", "native", "rcg"):
            if not np.array_equal(released[name], family[name]):
                raise ValueError("Released INSID3 comparator identity/baseline mismatch")
        masks[INSID3] = released[INSID3]
    differences["original_forward_native"] = int(np.unpackbits(original["native"] ^ fine["native"]).sum())
    return masks, dict(original_key=row["key"], fine_key=fine_row["key"], identity=list(identity(row)),
                       paths={k: str(v) for k, v in paths.items()}, hashes=hashes, replay_differences=differences)


def prepare(a):
    sources = dict(original=source(a.original, 241), family=source(a.family, 241), fine=source(a.fine, 1200))
    family_protocol = sources["family"]["metadata"]["protocol.json"]
    if family_protocol["origin"] != "stage:model.raw_nn":
        raise ValueError("Parent accounting origin is not raw NN")
    stage_ref = family_protocol["sources"]["stage"]
    sources["stage"] = source(Path(stage_ref["path"]), 241)
    if sources["stage"]["seal_hash"] != stage_ref["seal_sha256"]:
        raise ValueError("Raw stage parent seal changed")
    stage_config = sources["stage"]["metadata"]["config.json"]
    original_protocol = sources["original"]["metadata"]["protocol.json"]
    if stage_config["origin"] != RAW or "before any projection" not in stage_config["token_space"]:
        raise ValueError("Raw origin's unprojected provenance is missing")
    if stage_config["layers_run_seal_sha256"] != sources["original"]["seal_hash"]:
        raise ValueError("Raw stage is not bound to the original layer producer")
    if original_protocol["layer_dtype"] != "float32" or not original_protocol["save_layers"]:
        raise ValueError("Original serialized raw-layer provenance is missing")
    if sources["stage"]["seal"]["query_labels_opened"] is not False:
        raise ValueError("Raw origin was not label-isolated")
    fine_score_state = json.loads((a.fine / "score_state.json").read_text())
    if fine_score_state["state"] != "CPU_GT_SCORE_COMPLETE" or fine_score_state["completed"] != 1200:
        raise ValueError("Require the completed fixed1200 comparison")
    check(a.fine / "report.json", fine_score_state["report_sha256"])
    if sources["fine"]["seal"]["query_truth_opened"] is not False or tuple(sources["fine"]["seal"]["arms"]) != FINE_ARMS:
        raise ValueError("Unexpected fine recipe or label use")
    rows = sources["original"]["rows"]
    for row in rows:
        for name in ("family", "stage", "fine"):
            r = sources[name]["identities"].get(identity(row))
            if r is None or int(r["fold"]) != int(row["fold"]):
                raise ValueError("Missing/changed DEV identity in " + name)
            if name != "fine" and r["key"] != row["key"]:
                raise ValueError("Original source key mismatch")
    insid3_status = dict(included=False, reason="not requested")
    if a.insid3:
        try:
            released = source(a.insid3, 241)
            protocol = released["metadata"]["protocol.json"]
            if released["seal"]["query_gt_in_inference"] is not False or protocol["query_gt_in_inference"] is not False:
                raise ValueError("INSID3 query-label isolation absent")
            if protocol["family_seal_sha256"] != sources["family"]["seal_hash"]:
                raise ValueError("INSID3 comparison parent changed")
            for row in rows:
                r = released["identities"].get(identity(row))
                if r is None or int(r["fold"]) != int(row["fold"]):
                    raise ValueError("INSID3 identity mismatch")
                old = family_protocol["input_receipts"][row["key"]]
                if protocol["input_receipts"][r["key"]]["packet_sha256"] != old["packet_sha256"]:
                    raise ValueError("INSID3 GT packet binding mismatch")
            sources["insid3"] = released
            insid3_status = dict(included=True, actual_arm=INSID3, implementation=protocol["implementation"],
                                 config=protocol["config"], seed=protocol["seed"], resolution=protocol["resolution"],
                                 family_parent_and_GT_packet_binding=True)
        except (ValueError, KeyError, FileNotFoundError) as error:
            insid3_status = dict(included=False, reason=str(error))
    jobs = []
    for row in rows:
        key = row["key"]
        fine_row = sources["fine"]["identities"][identity(row)]
        paths, hashes = {}, {}
        for name in ("original", "family", "stage", "fine"):
            selected_key = fine_row["key"] if name == "fine" else key
            paths[name] = sources[name]["path"] / "predictions" / (selected_key + ".npz")
            hashes[name] = sources[name]["seal"]["predictions"][selected_key]
        paths["old_packet"] = a.old_packets / (key + ".npz")
        paths["new_packet"] = Path(fine_row["packet_export"])
        old = family_protocol["input_receipts"][key]
        if Path(old["packet"]) != paths["old_packet"]:
            raise ValueError("Old packet receipt points elsewhere")
        hashes["old_packet"] = old["packet_sha256"]
        hashes["new_packet"] = sources["fine"]["seal"]["inputs"][fine_row["key"]]["packet_sha256"]
        stage_key = sources["stage"]["identities"][identity(row)]["key"]
        if sources["stage"]["seal"]["inputs"][stage_key]["packet_sha256"] != hashes["old_packet"]:
            raise ValueError("Stage/family GT packet provenance mismatch")
        if insid3_status["included"]:
            r = sources["insid3"]["identities"][identity(row)]
            paths["insid3"] = a.insid3 / "predictions" / (r["key"] + ".npz")
            hashes["insid3"] = sources["insid3"]["seal"]["predictions"][r["key"]]
        jobs.append((row, fine_row, paths, hashes, insid3_status["included"]))
    protocol = dict(
        n=241, dataset="COCO-20i original exposed DEV241", seed=family_protocol["seed"], resolution=1024,
        shots=1, exposure="all original241 DEV identities in original order; frozen1200 subset projection, not confirmation",
        origin=RAW, origin_definition="sealed raw unprojected final-layer nearest-reference-token label transfer",
        raw_origin_provenance=dict(parent_origin=family_protocol["origin"], stage_token_space=stage_config["token_space"],
            original_layer_dtype=original_protocol["layer_dtype"], stage_query_labels_opened=False,
            parent_other_arms_used_DEV_fitting=sources["family"]["seal"].get("query_gt_used_for_DEV_fold_fitting"),
            origin_branch="exact clone of label-isolated stage:model.raw_nn; GT-selected parent choices are unused"),
        sources={name: dict(path=str(s["path"]), seal_sha256=s["seal_hash"]) for name, s in sources.items()},
        frozen_fine_config=sources["fine"]["metadata"]["config.json"], insid3=insid3_status,
        native_definition="original cached packet/family native, bit-identical to mapped1200 native",
        original_forward_native_definition="separately retained original GPU native replay; known25-pixel drift remains explicit",
        matching="class/supportbasename/querybasename; not episode key", query_GT_during_assembly=False,
        encoder_forwards=0, retuning=False, masks_projected_without_changes=True,
        source_code={str(Path(__file__).resolve()): sha(__file__), str(REPO / "src/ics/experiment.py"): sha(REPO / "src/ics/experiment.py")})
    return rows, jobs, protocol


def assemble(a):
    if a.out.exists():
        raise FileExistsError("Fresh owned output required")
    rows, jobs, protocol = prepare(a)
    if a.verify_only:
        for i in (0, 60, 120, 180):
            _, receipt = assemble_one(jobs[i])
            print(json.dumps(dict(verified=receipt["original_key"], differences=receipt["replay_differences"])), flush=True)
        print("MASK_ONLY_SOURCE_SMOKE_OK", flush=True)
        return
    begin = time.monotonic()
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(assemble_one, jobs, chunksize=4))
    a.out.mkdir(parents=True)
    (a.out / "predictions").mkdir()
    hashes = {}
    for row, (masks, _) in zip(rows, results):
        path = a.out / "predictions" / (row["key"] + ".npz")
        np.savez_compressed(path, **masks)
        hashes[row["key"]] = sha(path)
    protocol["input_receipts"] = {r["key"]: result[1] for r, result in zip(rows, results)}
    protocol["replay_differences"] = {r["key"]: result[1]["replay_differences"] for r, result in zip(rows, results)}
    protocol["seconds_assembly"] = time.monotonic() - begin
    write(a.out / "manifest.json", rows)
    write(a.out / "protocol.json", protocol)
    arms = list(FINE_ARMS) + [RAW, FORWARD_NATIVE] + ([INSID3] if protocol["insid3"]["included"] else [])
    write(a.out / "sealed.json", dict(state="ALL_PREDICTIONS_SEALED", n=241, arms=arms,
        manifest_sha256=sha(a.out / "manifest.json"), protocol_sha256=sha(a.out / "protocol.json"),
        predictions=hashes, query_GT_during_assembly=False, encoder_forwards=0))
    print(json.dumps(dict(state="ALL_PREDICTIONS_SEALED", n=241, seconds=protocol["seconds_assembly"],
                         insid3=protocol["insid3"]["included"])), flush=True)


def edits(mask, origin, truth):
    add, delete = mask & ~origin, origin & ~mask
    return dict(add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                delete_TP=int((delete & truth).sum()), delete_FP=int((delete & ~truth).sum()))


def score_one(job):
    row, receipt, prediction, prediction_hash, arms = job
    check(prediction, prediction_hash)
    for kind in ("old_packet", "new_packet"):
        check(receipt["paths"][kind], receipt["hashes"][kind])
    with np.load(receipt["paths"]["old_packet"], allow_pickle=False) as z:
        old_truth = z["truth"].copy()
    with np.load(receipt["paths"]["new_packet"], allow_pickle=False) as z:
        new_truth = z["truth"].copy()
    if not np.array_equal(old_truth, new_truth):
        raise ValueError("Pixel GT identity parity failure: " + row["key"])
    truth = unpack(old_truth)
    masks = packed(prediction, prediction_hash, arms)
    masks = {name: unpack(value) for name, value in masks.items()}
    result = dict(iu={}, raw_edits={}, native_edits={}, GT_pixel_parity=True)
    for name, mask in masks.items():
        result["iu"][name] = [int((mask & truth).sum()), int((mask | truth).sum())]
        for origin, target in ((RAW, "raw_edits"), ("native", "native_edits")):
            count = edits(mask, masks[origin], truth)
            base = [int((masks[origin] & truth).sum()), int((masks[origin] | truth).sum())]
            reconstructed = [base[0] + count["add_TP"] - count["delete_TP"], base[1] + count["add_FP"] - count["delete_FP"]]
            if reconstructed != result["iu"][name]:
                raise ValueError("Exact edit accounting failed")
            result[target][name] = count
    return result


def aggregate(corrections, rows, field=None):
    if field is None:
        return {name: {k: sum(r[k] for r in values) for k in EDIT_ORDER} for name, values in corrections.items()}
    labels = sorted({str(r.get(field, "DEV")) for r in rows})
    return {name: {label: {k: sum(r[k] for r in values if str(r.get(field, "DEV")) == label) for k in EDIT_ORDER}
                   for label in labels} for name, values in corrections.items()}


def score(a):
    begin = time.monotonic()
    seal = json.loads((a.out / "sealed.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or (a.out / "report.json").exists():
        raise ValueError("Incomplete or already-scored output")
    for field, file in (("manifest_sha256", "manifest.json"), ("protocol_sha256", "protocol.json")):
        check(a.out / file, seal[field])
    protocol = json.loads((a.out / "protocol.json").read_text())
    for path, digest in protocol["source_code"].items():
        check(path, digest)
    rows = json.loads((a.out / "manifest.json").read_text())
    jobs = [(r, protocol["input_receipts"][r["key"]], a.out / "predictions" / (r["key"] + ".npz"),
             seal["predictions"][r["key"]], seal["arms"]) for r in rows]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(score_one, jobs, chunksize=4))
    arrays = {arm: np.array([r["iu"][arm] for r in results], np.int64) for arm in seal["arms"]}
    raw_corrections, native_corrections = {}, {}
    for name in seal["arms"]:
        for key, destination in (("raw_edits", raw_corrections), ("native_edits", native_corrections)):
            destination[name] = [dict(key=row["key"], c=row["c"], fold=row["fold"], batch=row.get("batch", "DEV"),
                                      **r[key][name]) for row, r in zip(rows, results)]
    alias = "raw_DINO.control"
    report, draws = summarize(rows, dict(arrays, **{alias: arrays[RAW]}), raw_corrections)
    report["scores"].pop(alias)
    report["contrasts"].pop(alias)
    for values in report["contrasts"].values():
        if alias in values:
            values[RAW] = values.pop(alias)
    report["contrasts"][RAW].pop(RAW, None)
    for field in ("folds", "batchs"):
        for entry in report[field].values():
            entry["scores"].pop(alias)
            entry["gain_vs_native"].pop(alias)
            entry["gain_vs_raw_DINO"] = {name: value - entry["scores"][RAW] for name, value in entry["scores"].items()}
    report["corrections_vs_raw_DINO"] = report.pop("corrections_vs_native")
    report["corrections_vs_raw_DINO_by_class"] = report.pop("corrections_by_class")
    report["corrections_vs_raw_DINO_by_batch"] = report.pop("corrections_by_batch")
    report["corrections_vs_native"] = aggregate(native_corrections, rows)
    report["corrections_vs_native_by_class"] = aggregate(native_corrections, rows, "c")
    report["corrections_vs_native_by_batch"] = aggregate(native_corrections, rows, "batch")
    report.update(primary="fine.rcg64", edit_origin=RAW, origin_definition=protocol["origin_definition"],
        exposure=protocol["exposure"], seed=protocol["seed"], dataset=protocol["dataset"], resolution=1024,
        matched_original_rows=241, native_RCG_bit_parity=True, GT_pixel_parity=True,
        original_forward_native_pixels_different=sum(r["original_forward_native"] for r in protocol["replay_differences"].values()),
        raw_origin_provenance=protocol["raw_origin_provenance"], insid3=protocol["insid3"],
        retuning=False, encoder_forwards=0, source_seal_sha256=sha(a.out / "sealed.json"), seconds_score=time.monotonic() - begin)
    # Independent parent1200 I/U check after all own masks are sealed and scored.
    fine_source = Path(protocol["sources"]["fine"]["path"])
    parent_episodes = {r["key"]: r for r in [json.loads(line) for line in (fine_source / "episodes.jsonl").read_text().splitlines()]}
    for row, result in zip(rows, results):
        mapped_key = protocol["input_receipts"][row["key"]]["fine_key"]
        for arm in FINE_ARMS:
            if parent_episodes[mapped_key]["iu"][arm] != result["iu"][arm]:
                raise ValueError("Fixed1200 original I/U parity failed: " + arm)
    report["all_six_parent1200_IU_parity"] = True
    report["primary_contrast_decisions"] = {name: ("positive" if value["ci95"][0] > 0 else "negative" if value["ci95"][1] < 0 else "unresolved")
        for name, value in report["contrasts"]["fine.rcg64"].items()}
    write(a.out / "report.json", report)
    details = [dict(row, iu=result["iu"], edits_vs_raw_DINO=result["raw_edits"], edits_vs_native=result["native_edits"])
               for row, result in zip(rows, results)]
    (a.out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in details))
    np.savez_compressed(a.out / "counts.npz", **{"iu:" + k: v for k, v in arrays.items()},
        **{"raw_edits:" + k: np.array([[r[key] for key in EDIT_ORDER] for r in values], np.int64) for k, values in raw_corrections.items()},
        **{"native_edits:" + k: np.array([[r[key] for key in EDIT_ORDER] for r in values], np.int64) for k, values in native_corrections.items()})
    np.save(a.out / "bootstrap_photo_draws.npy", draws)
    write(a.out / "score_receipt.json", dict(source_seal_sha256=sha(a.out / "sealed.json"), GT_pixel_parity_cases=241,
        source_packet_receipts={r["key"]: {k: protocol["input_receipts"][r["key"]]["hashes"][k] for k in ("old_packet", "new_packet")} for r in rows},
        exact_four_edit_reconstruction_cases=241, edit_order=list(EDIT_ORDER), score_seconds=report["seconds_score"]))
    bases = (RAW, "native", "rcg", "mean.control", "rcg64.control", "fine.rcg16.control", INSID3)
    bases = [b for b in bases if b in arrays]
    lines = ["# Frozen complete fine readout on original DEV241", "", protocol["exposure"], "",
             "| arm | class mIoU | " + " | ".join("vs " + b for b in bases) + " |",
             "|---|---:|" + "---|" * len(bases)]
    for arm in seal["arms"]:
        values = []
        for base in bases:
            contrast = report["contrasts"][arm].get(base)
            values.append("" if contrast is None else "%+.3f [%+.3f,%+.3f]" % (contrast["gain"], *contrast["ci95"]))
        lines.append("| %s | %.6f | %s |" % (arm, report["scores"][arm], " | ".join(values)))
    lines += ["", "All241 canonical native/RCG masks and old/new truth are bit-identical. Original-forward native replay differs by25pixels in two cases and is shown separately.",
              "All own masks sealed before GT diagnostics; no fitting, new encoder, or efficacy-based subset.",
              "", "## Edits from actual raw DINO origin", "",
              "| arm | add TP | add FP | delete TP | delete FP |", "|---|---:|---:|---:|---:|"]
    for arm in seal["arms"]:
        e = report["corrections_vs_raw_DINO"][arm]
        lines.append("| %s | %d | %d | %d | %d |" % (arm, *(e[k] for k in EDIT_ORDER)))
    lines += ["", "Native-relative counts are separately retained in report.json and episodes.jsonl. An interval crossing zero is unresolved. This DEV view is bookkeeping of the unchanged frozen1200 recipe, not fresh confirmation."]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(state="RAW_DEV241_DONE", scores=report["scores"], primary=report["contrasts"]["fine.rcg64"],
                         edits_vs_raw=report["corrections_vs_raw_DINO"]["fine.rcg64"], score_seconds=report["seconds_score"])), flush=True)


def self_test():
    truth = np.zeros((1024, 1024), bool)
    truth[:10] = True
    raw = np.zeros_like(truth)
    raw[:20] = True
    count = edits(truth, raw, truth)
    assert count == dict(add_TP=0, add_FP=0, delete_TP=0, delete_FP=10240)
    assert identity(dict(c=1, support="a/x.jpg", query="b/y.jpg")) == (1, "x.jpg", "y.jpg")
    assert np.array_equal(unpack(np.packbits(truth)), truth)
    print("SELF_TEST_OK", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("all", "assemble", "score"), default="all")
    p.add_argument("--original", type=Path)
    p.add_argument("--family", type=Path)
    p.add_argument("--fine", type=Path)
    p.add_argument("--insid3", type=Path)
    p.add_argument("--old-packets", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.out or not 1 <= a.workers <= 4:
        p.error("Require out and1..4workers")
    if a.phase in ("all", "assemble"):
        if not all((a.original, a.family, a.fine, a.old_packets)):
            p.error("Require all primary sources")
        assemble(a)
        if a.verify_only:
            return
    if a.phase in ("all", "score"):
        score(a)


if __name__ == "__main__":
    main()
