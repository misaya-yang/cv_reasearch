#!/usr/bin/env python3
"""Independent CPU statistics verification of a COMPLETE scored4000 bundle.

--run is the inference directory (remote outputs/frozen_subtoken4000_v1):
sealed.json/config.json, optional original manifest.json.
--scored is the separate CPU scoring directory (remote
outputs/frozen_subtoken4000_scored_v1): report.json, episodes.jsonl,
score_state.json, receipt.json, optional scored manifest.json copy.
Scorer receipt source_seal_sha256/config_sha256/manifest_sha256 connect the
two original identities; local downloads do not become one native producer.
No mask/packet/field/model is opened. Complete seal + complete score gates are
checked before saved I/U.

Uses the same ics.experiment class-summed I/U, full-cohort connected-photo
RandomState(0) 2,000 draws, absent-class handling and percentile intervals as
the verified1200 analysis. Repeated episode identities are never deduplicated.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
from score_frozen_subtoken4000 import ARMS, EDIT_NAMES, ROOT, public_key, same_identity
from verify_frozen_subtoken1200 import net_terms
from ics.experiment import metric, photo_groups, sha, summarize

PRIMARY = "fine.rcg64"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def complete_metadata(run, scored):
    """Connect separate inference and scoring identities before episode I/U."""
    if run.resolve() == scored.resolve():
        raise ValueError("Use separate original inference and CPU scoring directories")
    seal = json.loads((run / "sealed.json").read_text())
    state = json.loads((scored / "score_state.json").read_text())
    if seal.get("state") != "ALL_PREDICTIONS_SEALED" or seal.get("n") != 4000 or state.get("state") != "CPU_GT_SCORE_COMPLETE" or state.get("n") != 4000 or state.get("completed") != 4000:
        raise ValueError("Require all4000 sealed AND CPU-scored before saved I/U verification")
    if set(seal.get("arms", [])) != set(ARMS):
        raise ValueError("Complete inference seal must advertise all six fixed arms")
    run_files = ["sealed.json", "config.json"]
    scored_files = ["score_state.json", "receipt.json", "report.json", "episodes.jsonl"]
    if (run / "manifest.json").is_file():
        run_files.append("manifest.json")
    if (scored / "manifest.json").is_file():
        scored_files.append("manifest.json")
    if (scored / "counts.npz").is_file():
        scored_files.append("counts.npz")
    hashes = {"inference": {name: sha(run / name) for name in run_files},
              "scoring": {name: sha(scored / name) for name in scored_files}}
    receipt = json.loads((scored / "receipt.json").read_text())
    source = json.loads((scored / "report.json").read_text())
    config = json.loads((run / "config.json").read_text())
    if (hashes["inference"]["sealed.json"] != receipt["source_seal_sha256"]
            or hashes["inference"]["config.json"] != seal["config_sha256"]
            or receipt["config_sha256"] != seal["config_sha256"]
            or receipt["manifest_sha256"] != seal["manifest_sha256"]
            or hashes["scoring"]["report.json"] != state["report_sha256"]
            or source["config"] != config):
        raise ValueError("Scorer receipt does not connect the correct inference/scoring metadata")
    if not Path(receipt["source_run"]).is_absolute():
        raise ValueError("Scorer receipt must retain its original absolute inference source_run")
    for role in ("inference", "scoring"):
        if "manifest.json" in hashes[role] and hashes[role]["manifest.json"] != seal["manifest_sha256"]:
            raise ValueError("Fetched " + role + " manifest differs from the inference seal")
    provenance = {"inference": {"local_directory": str(run.resolve()), "sha256": hashes["inference"]},
                  "scoring": {"local_directory": str(scored.resolve()), "sha256": hashes["scoring"], "inference_source_run_from_receipt": receipt["source_run"]},
                  "receipt_link": {"source_seal_sha256": receipt["source_seal_sha256"], "config_sha256": receipt["config_sha256"], "manifest_sha256": receipt["manifest_sha256"]}}
    return source, config, hashes, provenance


def arrays_and_edits(rows):
    arrays = {arm: np.array([row["iu"][arm] for row in rows], dtype=np.int64) for arm in ARMS}
    for values in arrays.values():
        if np.any((values[:, 0] < 0) | (values[:, 1] <= 0) | (values[:, 0] > values[:, 1])):
            raise ValueError("Invalid saved I/U")
    present = ["edits_vs_native" in row for row in rows]
    if any(present) and not all(present):
        raise ValueError("Partially saved native-relative edit records")
    corrections = {}
    if all(present):
        for arm in ARMS:
            records = []
            for row in rows:
                counts = row["edits_vs_native"][arm]
                if set(counts) != set(EDIT_NAMES) or any(not isinstance(value, int) or value < 0 for value in counts.values()):
                    raise ValueError("Malformed native-relative edit count")
                native, candidate = row["iu"]["native"], row["iu"][arm]
                if candidate[0] != native[0] + counts["add_TP"] - counts["delete_TP"] or candidate[1] != native[1] + counts["add_FP"] - counts["delete_FP"]:
                    raise ValueError("Per-draw edit identity failed: " + row["key"] + " " + arm)
                records.append(dict(key=row["key"], c=row["c"], fold=row["fold"], batch=str(row.get("batch", "unspecified")), **counts))
            corrections[arm] = records
    return arrays, corrections


def compare_report(recomputed, source):
    errors = []
    if recomputed["bootstrap"] != source["bootstrap"]:
        raise ValueError("Source bootstrap convention differs from verified1200")
    for field in ("n", "classes", "photo_groups", "largest_photo_group"):
        if recomputed[field] != source[field]:
            raise ValueError("Sampling/group count differs: " + field)
    for arm in ARMS:
        errors.append(abs(recomputed["scores"][arm] - source["scores"][arm]))
        for base, result in recomputed["contrasts"][arm].items():
            previous = source["contrasts"][arm][base]
            errors.append(abs(result["gain"] - previous["gain"]))
            errors.extend(abs(np.array(result["ci95"]) - previous["ci95"]))
            if any(result[field] != previous[field] for field in ("up", "down", "tie")):
                raise ValueError("Saved up/down/tie count differs")
    for table in ("folds", "batchs"):
        if set(recomputed[table]) != set(source[table]):
            raise ValueError("Fold/batch label set differs")
        for label, group in recomputed[table].items():
            previous = source[table][label]
            if group["n"] != previous["n"]:
                raise ValueError("Fold/batch sample count differs")
            for arm in ARMS:
                errors.append(abs(group["scores"][arm] - previous["scores"][arm]))
                errors.append(abs(group["gain_vs_native"][arm] - previous["gain_vs_native"][arm]))
            if "primary_gain_vs_controls" in previous:
                for arm, gain in previous["primary_gain_vs_controls"].items():
                    errors.append(abs(group["scores"][PRIMARY] - group["scores"][arm] - gain))
    if max(errors) > 1e-10:
        raise ValueError("Source statistical report differs from saved I/U")
    return float(max(errors))


def subgroup_intervals(rows, arrays, draws, recomputed, source):
    groups = photo_groups(rows)
    group_count = int(groups.max()) + 1
    weights = np.stack([np.bincount(draw, minlength=group_count) for draw in draws])[:, groups]
    classes = np.array([row["c"] for row in rows])
    errors, unavailable = [], []
    for field, table in (("fold", "folds"), ("batch", "batchs")):
        for label, group in recomputed[table].items():
            mask = np.array([str(row.get(field, "unspecified")) == label for row in rows])
            sampled = {arm: np.array([metric(value[mask], classes[mask], weight[mask]) for weight in weights]) for arm, value in arrays.items()}
            group["primary_paired_contrasts"] = {}
            for arm in ARMS[:-1]:
                delta = arrays[PRIMARY][mask, 0] / arrays[PRIMARY][mask, 1] - arrays[arm][mask, 0] / arrays[arm][mask, 1]
                result = dict(gain=group["scores"][PRIMARY] - group["scores"][arm], ci95=np.percentile(sampled[PRIMARY] - sampled[arm], [2.5, 97.5]).tolist(), up=int((delta > 1e-12).sum()), down=int((delta < -1e-12).sum()), tie=int((np.abs(delta) <= 1e-12).sum()))
                group["primary_paired_contrasts"][arm] = result
                previous = source[table][label].get("primary_paired_contrasts", {}).get(arm)
                if previous is None:
                    unavailable.append(f"{table}/{label}/{arm}")
                    continue
                errors.append(abs(result["gain"] - previous["gain"]))
                errors.extend(abs(np.array(result["ci95"]) - previous["ci95"]))
                if any(result[key] != previous[key] for key in ("up", "down", "tie")):
                    raise ValueError("Fold/batch episode outcome counts differ")
    if errors and max(errors) > 1e-10:
        raise ValueError("Source subgroup paired CIs differ from the existing common bootstrap")
    return {"max_error_pp": float(max(errors)) if errors else None, "source_subgroup_intervals_not_saved": unavailable}


def verify_prior_rows(rows, prior4000, prior1200):
    if len(prior4000) != 4000 or len({r["key"] for r in prior4000}) != 4000:
        raise ValueError("Require original all4000 baseline draw records")
    if len(prior1200) != 1200 or len({r["key"] for r in prior1200}) != 1200:
        raise ValueError("Require all1200 earlier six-arm draw records")
    previous = {row["key"]: row for row in prior4000}
    keys = [public_key(row, previous) for row in rows]
    if keys != [row["key"] for row in prior4000]:
        raise ValueError("All4000 original sampled draws and order must be preserved")
    target = dict(zip(keys, rows))
    for row, key in zip(rows, keys):
        if not same_identity(row, previous[key]) or any(row["iu"][arm] != previous[key]["iu"][arm] for arm in ARMS[:3]):
            raise ValueError("Existing4000 baseline I/U or sampled identity differs")
    used = set()
    for old in prior1200:
        key = public_key(old, previous)
        if key in used or not same_identity(old, target[key]) or any(old["iu"][arm] != target[key]["iu"][arm] for arm in ARMS):
            raise ValueError("Earlier1200 six-arm I/U or sampled identity differs")
        used.add(key)
    identities = [(int(row["c"]), Path(row["query"]).name, Path(row["support"]).name) for row in rows]
    return {"baseline4000": {"n": 4000, "arms": list(ARMS[:3]), "mismatches": 0}, "earlier1200": {"n": len(used), "arms": list(ARMS), "mismatches": 0}, "sampled_draws": len(rows), "unique_episode_identities": len(set(identities)), "natural_repeated_draws_retained": len(rows) - len(set(identities))}


def verify_count_artifact(path, arrays, corrections):
    if not path.is_file():
        return "not downloaded; saved episode counts remain the statistical source"
    with np.load(path, allow_pickle=False) as stored:
        for arm in ARMS:
            if not np.array_equal(stored["iu:" + arm], arrays[arm]):
                raise ValueError("Saved counts.npz I/U differs from episodes: " + arm)
            if corrections:
                expected = np.array([[record[name] for name in EDIT_NAMES] for record in corrections[arm]], dtype=np.int64)
                if not np.array_equal(stored["edits_vs_native:" + arm], expected):
                    raise ValueError("Saved counts.npz edit records differ from episodes: " + arm)
    return "all six-arm I/U and available four-way edits match every saved draw"


def sideeffects(rows, arrays, corrections, draws):
    classes = np.array([row["c"] for row in rows])
    ids, index = np.unique(classes, return_inverse=True)
    groups = photo_groups(rows)
    g = int(groups.max()) + 1
    weights = np.stack([np.bincount(draw, minlength=g) for draw in draws])[:, groups]
    result = {"net_stage_decompositions": {}}
    for arm, base in ((PRIMARY, "native"), (PRIMARY, "rcg64.control"), ("fine.rcg16.control", "rcg"), ("rcg64.control", "rcg")):
        point = net_terms(arrays, arm, base, classes)
        boot = np.array([net_terms(arrays, arm, base, classes, w) for w in weights])
        result["net_stage_decompositions"][arm + " minus " + base] = dict(gain_pp=float(point[0]), ci95_pp=np.percentile(boot[:, 0], [2.5, 97.5]).tolist(), net_TP_contribution_pp=float(point[1]), net_TP_ci95_pp=np.percentile(boot[:, 1], [2.5, 97.5]).tolist(), net_FP_contribution_pp=float(point[2]), net_FP_ci95_pp=np.percentile(boot[:, 2], [2.5, 97.5]).tolist(), pooled_net_TP_change=int((arrays[arm][:, 0] - arrays[base][:, 0]).sum()), pooled_net_FP_change=int((arrays[arm][:, 1] - arrays[base][:, 1]).sum()))
    if corrections:
        values = np.array([[record[name] for name in EDIT_NAMES] for record in corrections[PRIMARY]])
        def terms(w):
            present = np.bincount(index, weights=w, minlength=len(ids)) > 0
            total = lambda value: np.bincount(index, weights=w * value, minlength=len(ids))
            j0 = total(arrays["native"][:, 0]) / np.maximum(total(arrays["native"][:, 1]), 1)
            u1 = np.maximum(total(arrays[PRIMARY][:, 1]), 1)
            factors = (np.ones(len(ids)), -j0, -np.ones(len(ids)), j0)
            contribution = np.array([100 * np.mean((total(values[:, k]) * factors[k] / u1)[present]) for k in range(4)])
            expected = metric(arrays[PRIMARY], classes, w) - metric(arrays["native"], classes, w)
            if abs(contribution.sum() - expected) > 1e-10:
                raise ValueError("Four-way class-macro contribution identity failed")
            return contribution
        point = terms(np.ones(len(rows)))
        boot = np.array([terms(w) for w in weights])
        result["primary_four_way_native_relative"] = {name: {"contribution_pp": float(point[k]), "ci95_pp": np.percentile(boot[:, k], [2.5, 97.5]).tolist(), "pooled_pixel_count": int(values[:, k].sum())} for k, name in enumerate(EDIT_NAMES)}
    else:
        result["primary_four_way_native_relative"] = "not saved"
    result["interpretation"] = "Exact observed-mask accounting with class totals/final candidate union. Only within-pair terms add; pooled totals are diagnostics. No causal or GT-geometry inference."
    return result


def self_test():
    rows = [dict(key=f"public0:{j}", fold=0, e=j, c=0 if j < 2 else 1, batch="test", public_batch=0, query="a.jpg" if j < 2 else f"q{j}.jpg", support="b.jpg" if j < 2 else f"s{j}.jpg") for j in range(4)]
    native = [[1, 2], [1, 2], [10, 20], [30, 40]]
    fine = [[2, 2], [2, 2], [11, 20], [32, 40]]
    for j, row in enumerate(rows):
        row["iu"] = {arm: fine[j] if arm == PRIMARY else native[j] for arm in ARMS}
        row["edits_vs_native"] = {arm: dict(add_TP=row["iu"][arm][0] - native[j][0], add_FP=0, delete_TP=0, delete_FP=0) for arm in ARMS}
    arrays, corrections = arrays_and_edits(rows)
    source, draws = summarize(rows, arrays, corrections)
    effects = sideeffects(rows, arrays, corrections, draws)
    assert abs(sum(v["contribution_pp"] for v in effects["primary_four_way_native_relative"].values()) - source["contrasts"][PRIMARY]["native"]["gain"]) < 1e-10
    assert abs(source["scores"]["native"] - 100 * (.5 + 40 / 60) / 2) < 1e-10
    assert len(rows) == 4 and len(set((r["c"], r["query"], r["support"]) for r in rows)) == 3
    assert compare_report(source, copy.deepcopy(source)) == 0
    bad = copy.deepcopy(source)
    bad["contrasts"][PRIMARY]["native"]["ci95"][0] += .01
    try:
        compare_report(source, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Altered confidence interval was accepted")
    bad_rows = copy.deepcopy(rows)
    bad_rows[-1]["edits_vs_native"][PRIMARY]["add_TP"] += 1
    try:
        arrays_and_edits(bad_rows)
    except ValueError:
        pass
    else:
        raise AssertionError("Bad per-draw edit identity was accepted")
    # This tests metadata transfer/identity only; the synthetic directories
    # contain no real episode I/U or query truth.
    with tempfile.TemporaryDirectory() as directory:
        run, scored = (Path(directory) / role for role in ("infer-download", "score-download"))
        run.mkdir(); scored.mkdir()
        def save(path, value):
            path.write_text(json.dumps(value, indent=2) + "\n")
        config = {"parameters": {"0": {"sigma": 1.25, "tau": .07}}}
        save(run / "config.json", config)
        save(run / "manifest.json", [])
        seal = dict(state="ALL_PREDICTIONS_SEALED", n=4000, arms=list(ARMS), config_sha256=sha(run / "config.json"), manifest_sha256=sha(run / "manifest.json"))
        save(run / "sealed.json", seal)
        save(scored / "report.json", dict(config=config))
        (scored / "episodes.jsonl").write_text("")
        save(scored / "score_state.json", dict(state="CPU_GT_SCORE_COMPLETE", n=4000, completed=4000, report_sha256=sha(scored / "report.json")))
        receipt = dict(source_run="/root/study/outputs/frozen_subtoken4000_v1", source_seal_sha256=sha(run / "sealed.json"), config_sha256=seal["config_sha256"], manifest_sha256=seal["manifest_sha256"])
        save(scored / "receipt.json", receipt)
        _, _, _, provenance = complete_metadata(run, scored)
        assert provenance["inference"]["local_directory"] != provenance["scoring"]["local_directory"]
        assert provenance["scoring"]["inference_source_run_from_receipt"] == receipt["source_run"]
        receipt["config_sha256"] = "0" * 64
        save(scored / "receipt.json", receipt)
        try:
            complete_metadata(run, scored)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched scorer/inference receipt was accepted")
    print("SELF_TEST_PASS: class-summed estimand, sampled repeats, exact edits, CI/edit corruption rejection, separate run/scored identity and receipt mismatch rejection")


def main():
    parent = ROOT / "evidence/local/research_20261005/pipeline_verified"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=parent / "frozen_subtoken4000_v1")
    parser.add_argument("--scored", type=Path, default=parent / "frozen_subtoken4000_scored_v1")
    parser.add_argument("--prior4000", type=Path, default=parent / "frozen_public4000_v1/episodes.jsonl")
    parser.add_argument("--prior1200", type=Path, default=parent / "frozen_subtoken1200_v1")
    parser.add_argument("--out", type=Path, default=parent / "frozen_subtoken4000_verification_v1")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    started = time.monotonic()
    # Complete inference/score gates and cross-directory receipt linkage run
    # before reading any final episode I/U.
    source, config, hashes, provenance = complete_metadata(args.run, args.scored)
    if config["parameters"] != json.loads((args.prior1200 / "config.json").read_text())["parameters"]:
        raise ValueError("Earlier1200 frozen parameters changed")
    prior_paths = {"public4000": args.prior4000, "earlier1200_episodes": args.prior1200 / "episodes.jsonl", "earlier1200_config": args.prior1200 / "config.json"}
    prior_hashes = {name: sha(path) for name, path in prior_paths.items()}
    rows = read_rows(args.scored / "episodes.jsonl")
    if len(rows) != 4000 or len({row["key"] for row in rows}) != 4000 or set(int(r["c"]) for r in rows) != set(range(80)) or any(sum(int(r["fold"]) == fold for r in rows) != 1000 for fold in range(4)):
        raise ValueError("Require all4000 sampled draws,80 classes,1000/fold")
    parity = verify_prior_rows(rows, read_rows(args.prior4000), read_rows(args.prior1200 / "episodes.jsonl"))
    arrays, corrections = arrays_and_edits(rows)
    count_artifact = verify_count_artifact(args.scored / "counts.npz", arrays, corrections)
    recomputed, draws = summarize(rows, arrays, corrections)
    max_error = compare_report(recomputed, source)
    subgroup = subgroup_intervals(rows, arrays, draws, recomputed, source)
    editing = sideeffects(rows, arrays, corrections, draws)
    edit_tables_verified = []
    if corrections:
        for field in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
            if field in source:
                if recomputed[field] != source[field]:
                    raise ValueError("Saved edit totals/class/batch report differs: " + field)
                edit_tables_verified.append(field)
    baseline = recomputed["contrasts"][PRIMARY]["native"]
    controls = ("mean.control", "rcg64.control", "fine.rcg16.control")
    result = {"source": {**provenance, "prior_sha256": prior_hashes, "script_sha256": sha(Path(__file__))},
              "verification": {"all_source_scores_CIs_fold_batch_max_error_pp": max_error, "global_episode_up_down_tie_match": True, "subgroup_primary_CIs": subgroup, "prior_parity": parity, "frozen_parameters_unchanged": True, "inference_manifest_hash_verified_locally": "manifest.json" in hashes["inference"], "scored_manifest_copy_hash_verified_locally": "manifest.json" in hashes["scoring"], "inference_scoring_receipt_link_verified": True, "saved_counts_artifact": count_artifact, "native_relative_per_draw_edit_identities": "all six arms passed" if corrections else "not saved", "edit_report_tables_verified": edit_tables_verified},
              "statistics": recomputed, "edit_sideeffects": editing, "fixed_primary_decision": {"primary": PRIMARY, "observed_gain_at_least2_pp": baseline["gain"] >= 2, "ci95_entirely_at_least2_pp": baseline["ci95"][0] >= 2, "mandatory_strong_controls": list(controls), "strong_controls_all_resolved": all(recomputed["contrasts"][PRIMARY][arm]["ci95"][0] > 0 for arm in controls), "interpretation": "Report every fixed contrast. CI crossing zero is unresolved; point wins are not superiority. Existing4000 benchmark reuse is not independent confirmation."},
              "limits": ["No original packet/prediction/field masks were reopened or hashed locally; this verifies statistics from saved I/U and saved edit counts.", "Per-draw edit identities do not independently recount individual four-way categories from masks.", "Raw-DINO complete4000 origin remains unavailable; DEV241 common-origin comparison is separate.", "Native quality, GT-distance and object geometry are not inferred from missing fields.", "Subgroups reuse full-cohort photo multiplicities, exactly as the existing4000 scorer; no alternative RNG or estimand was introduced."]}
    for role, directory in (("inference", args.run), ("scoring", args.scored)):
        for name, expected_hash in hashes[role].items():
            if sha(directory / name) != expected_hash:
                raise ValueError("Original " + role + " download changed during verification")
    for name, path in prior_paths.items():
        if sha(path) != prior_hashes[name]:
            raise ValueError("Earlier verified records changed during verification")
    result["runtime_seconds"] = time.monotonic() - started
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Frozen subtoken4000 independent statistical verification", "", f"Inference metadata was read from `{args.run}`; CPU scoring outputs were read from `{args.scored}`. The files retain separate inference and CPU scoring origins, connected by the scorer receipt's seal/config/manifest hashes. The receipt declares inference source `{provenance['scoring']['inference_source_run_from_receipt']}`.", "", f"All six-arm class-summed macro scores, stored contrasts/CIs and fold/batch point results match saved I/U with maximum error {max_error:.3g} pp. Global up/down/tie counts match. All4000 prior native/RCG/MEAN and all1200 earlier six-arm I/U match exactly; all sampled draws and natural repeated identities are retained.", "", "| Fine64 minus control | Gain [95% CI], pp | Up / down / tie |", "|---|---:|---:|"]
    for arm in ARMS[:-1]:
        value = recomputed["contrasts"][PRIMARY][arm]
        lines.append(f"| {arm} | {value['gain']:+.6f} [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] | {value['up']} / {value['down']} / {value['tie']} |")
    lines += ["", "| Arm | Class-summed macro mIoU |", "|---|---:|"]
    lines += [f"| {arm} | {recomputed['scores'][arm]:.6f} |" for arm in ARMS]
    lines += ["", "| Subgroup | Draws | Fine64 | vs native | vs coarse64 | vs MEAN | vs fine16 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for table in ("folds", "batchs"):
        for label, group in recomputed[table].items():
            contrasts = group["primary_paired_contrasts"]
            entries = [f"{contrasts[arm]['gain']:+.3f} [{contrasts[arm]['ci95'][0]:+.3f}, {contrasts[arm]['ci95'][1]:+.3f}]" for arm in ("native", "rcg64.control", "mean.control", "fine.rcg16.control")]
            lines.append(f"| {table}/{label} | {group['n']} | {group['scores'][PRIMARY]:.3f} | " + " | ".join(entries) + " |")
    if corrections:
        lines += ["", "| Fine64 signed edit contribution vs native | Class-macro contribution [95% CI], pp | Pooled count |", "|---|---:|---:|"]
        for name, values in editing["primary_four_way_native_relative"].items():
            lines.append(f"| {name} | {values['contribution_pp']:+.3f} [{values['ci95_pp'][0]:+.3f}, {values['ci95_pp'][1]:+.3f}] | {values['pooled_pixel_count']} |")
    lines += ["", "Fold/batch primary contrasts are independently recomputed using the same full-cohort connected-photo RandomState(0)2,000 multiplicities; details and source availability are retained in JSON.", "", "Native-relative per-draw edit identities: " + result["verification"]["native_relative_per_draw_edit_identities"] + ". These checks do not replace a mask recount.", "", "Fine64 remains the fixed primary. Evaluate stable>=2 against native and MEAN/coarse64/originalfine16 controls; intervals crossing zero remain unresolved. Full4000 is benchmark/development reuse, not fresh confirmation.", ""]
    (args.out / "verification.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.out), "verification": result["verification"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
