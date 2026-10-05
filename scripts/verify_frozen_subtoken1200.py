#!/usr/bin/env python3
"""Independently recompute completed1200 statistics from saved I/U, CPU only.

No packet, prediction, field, model or image is opened. Saved pooled four-way
RCG edit totals can be checked against net I/U changes, but cannot be recounted
or decomposed per class without masks/per-draw edit counts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ics.experiment import metric, photo_groups, sha, summarize

ARMS = ("native", "rcg", "mean.control", "rcg64.control", "fine.rcg16.control", "fine.rcg64")


def net_terms(arrays, arm, base, classes, weights=None):
    ids, index = np.unique(classes, return_inverse=True)
    weights = np.ones(len(classes)) if weights is None else weights
    present = np.bincount(index, weights=weights, minlength=len(ids)) > 0
    totals = []
    for name in (base, arm):
        values = arrays[name]
        totals.append(np.stack([np.bincount(index, weights=weights * values[:, k], minlength=len(ids)) for k in (0, 1)], axis=1))
    b, a = totals
    j0 = b[:, 0] / np.maximum(b[:, 1], 1)
    u1 = np.maximum(a[:, 1], 1)
    tp = 100 * np.mean(((a[:, 0] - b[:, 0]) / u1)[present])
    fp = 100 * np.mean((j0 * (b[:, 1] - a[:, 1]) / u1)[present])
    gain = metric(arrays[arm], classes, weights) - metric(arrays[base], classes, weights)
    if abs(tp + fp - gain) > 1e-10:
        raise ValueError("Exact class net-TP/net-FP decomposition failed")
    return np.array([gain, tp, fp])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parent = ROOT / "evidence/local/research_20261005/pipeline_verified"
    parser.add_argument("--source", type=Path, default=parent / "frozen_subtoken1200_v1")
    parser.add_argument("--prior4000", type=Path, default=parent / "frozen_public4000_v1/episodes.jsonl")
    parser.add_argument("--out", type=Path, default=parent / "frozen_subtoken1200_verification_v1")
    args = parser.parse_args()
    start = time.monotonic()
    filenames = ("episodes.jsonl", "report.json", "sealed.json", "config.json")
    hashes = {name: sha(args.source / name) for name in filenames}
    rows = [json.loads(line) for line in (args.source / "episodes.jsonl").read_text().splitlines() if line.strip()]
    source = json.loads((args.source / "report.json").read_text())
    seal = json.loads((args.source / "sealed.json").read_text())
    config = json.loads((args.source / "config.json").read_text())
    if len(rows) != 1200 or len({row["key"] for row in rows}) != 1200 or seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 1200:
        raise ValueError("Require all completed1200 sampled draws")
    if hashes["config.json"] != seal["config_sha256"] or hashes["sealed.json"] != source["source_prediction_seal_sha256"] or config != source["config"]:
        raise ValueError("Fetched config/seal identity does not match scored report")
    arrays = {arm: np.array([row["iu"][arm] for row in rows], dtype=np.int64) for arm in ARMS}
    if any(np.any((values[:, 0] < 0) | (values[:, 1] <= 0) | (values[:, 0] > values[:, 1])) for values in arrays.values()):
        raise ValueError("Invalid saved I/U")
    recomputed, draws = summarize(rows, arrays, {})
    errors = []
    for key in ("n", "classes", "photo_groups", "largest_photo_group"):
        if recomputed[key] != source[key]:
            raise ValueError("Source sampling/group field differs: " + key)
    for arm in ARMS:
        errors.append(abs(recomputed["scores"][arm] - source["scores"][arm]))
        for base, contrast in recomputed["contrasts"][arm].items():
            previous = source["contrasts"][arm][base]
            errors.append(abs(contrast["gain"] - previous["gain"]))
            errors.extend(abs(np.array(contrast["ci95"]) - previous["ci95"]))
            if any(contrast[key] != previous[key] for key in ("up", "down", "tie")):
                raise ValueError("Source episode outcome counts differ")
    for table in ("folds", "batchs"):
        if set(recomputed[table]) != set(source[table]):
            raise ValueError("Subgroup labels differ")
        for label, values in recomputed[table].items():
            if values["n"] != source[table][label]["n"]:
                raise ValueError("Subgroup sampled count differs")
            errors.extend(abs(values["scores"][arm] - source[table][label]["scores"][arm]) for arm in ARMS)
    if max(errors) > 1e-10:
        raise ValueError("Stored scores/CIs do not match saved I/U")

    # Independently verify the report's native/RCG/MEAN continuity to public4000.
    prior_hash = sha(args.prior4000)
    prior_rows = [json.loads(line) for line in args.prior4000.read_text().splitlines() if line.strip()]
    prior = {row["key"]: row for row in prior_rows}
    if len(prior_rows) != 4000 or len(prior) != 4000:
        raise ValueError("Require preserved public4000 draw keys")
    for row in rows:
        key = f"public{row['public_batch']}:{row.get('source_key', row['key'])}"
        previous = prior[key]
        if any(int(row[k]) != int(previous[k]) for k in ("fold", "c")) or any(Path(row[k]).name != Path(previous[k]).name for k in ("support", "query")):
            raise ValueError("Public4000 sampled identity differs")
        for arm in ARMS[:3]:
            if row["iu"][arm] != previous["iu"][arm]:
                raise ValueError("Public4000 baseline I/U continuity failed")

    pooled = {}
    for arm in ARMS:
        counts = source["edits_vs_RCG_supplemental"][arm]
        net_tp = int((arrays[arm][:, 0] - arrays["rcg"][:, 0]).sum())
        net_fp = int((arrays[arm][:, 1] - arrays["rcg"][:, 1]).sum())
        if net_tp != counts["add_TP"] - counts["delete_TP"] or net_fp != counts["add_FP"] - counts["delete_FP"]:
            raise ValueError("Reported pooled edit counts fail net I/U identity: " + arm)
        pooled[arm] = {"reported_counts": counts, "net_TP_change": net_tp, "net_FP_change": net_fp,
                       "verification": "pooled add/delete differences match saved I/U exactly; individual categories not independently recounted"}

    classes = np.array([row["c"] for row in rows])
    groups = photo_groups(rows)
    group_count = int(groups.max()) + 1
    weights = np.stack([np.bincount(draw, minlength=group_count) for draw in draws])[:, groups]
    pairs = (("fine.rcg64", "native"), ("rcg", "native"), ("rcg64.control", "rcg"), ("fine.rcg64", "rcg64.control"), ("fine.rcg16.control", "rcg"), ("fine.rcg64", "rcg"), ("fine.rcg64", "fine.rcg16.control"))
    stages, samples = {}, {}
    for arm, base in pairs:
        name = arm + " minus " + base
        point = net_terms(arrays, arm, base, classes)
        boot = np.array([net_terms(arrays, arm, base, classes, weight) for weight in weights])
        samples[name] = boot
        stages[name] = {"gain_pp": float(point[0]), "ci95_pp": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                        "net_TP_contribution_pp": float(point[1]), "net_TP_contribution_ci95_pp": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
                        "net_FP_contribution_pp": float(point[2]), "net_FP_contribution_ci95_pp": np.percentile(boot[:, 2], [2.5, 97.5]).tolist(),
                        "pooled_net_TP_change": int((arrays[arm][:, 0] - arrays[base][:, 0]).sum()),
                        "pooled_net_FP_change": int((arrays[arm][:, 1] - arrays[base][:, 1]).sum())}
    arithmetic = {"full_gain_pp": stages["fine.rcg64 minus native"]["gain_pp"],
                  "coarse_RCG_vs_native_pp": stages["rcg minus native"]["gain_pp"],
                  "coarse_lambda_change_pp": stages["rcg64.control minus rcg"]["gain_pp"],
                  "fine_vs_coarse64_pp": stages["fine.rcg64 minus rcg64.control"]["gain_pp"]}
    if abs(arithmetic["full_gain_pp"] - sum(arithmetic[key] for key in ("coarse_RCG_vs_native_pp", "coarse_lambda_change_pp", "fine_vs_coarse64_pp"))) > 1e-10:
        raise ValueError("Three-stage arithmetic identity failed")
    interaction = samples["fine.rcg64 minus rcg64.control"][:, 0] - samples["fine.rcg16.control minus rcg"][:, 0]
    secondary = {"fine_increment_lambda64_minus_lambda16_pp": stages["fine.rcg64 minus rcg64.control"]["gain_pp"] - stages["fine.rcg16.control minus rcg"]["gain_pp"], "ci95_pp": np.percentile(interaction, [2.5, 97.5]).tolist(), "interpretation": "paired arithmetic difference of fixed fine/coarse contrasts; secondary, not evidence of a causal lambda threshold"}
    report = {"source": {"directory": str(args.source), "sha256": hashes, "public4000_sha256": prior_hash, "script_sha256": sha(Path(__file__))},
              "verification": {"n": len(rows), "classes": recomputed["classes"], "photo_groups": group_count, "largest_photo_group": recomputed["largest_photo_group"], "all_scores_contrasts_CIs_fold_batch_max_error_pp": float(max(errors)), "all_episode_up_down_tie_counts_match": True, "all1200_native_RCG_MEAN_match_public4000": True, "config_and_source_seal_identity_match": True, "pooled_edit_net_IU_identities_match": True},
              "statistics": recomputed, "stage_arithmetic": arithmetic, "net_TP_FP_stage_decompositions": stages, "pooled_edits_vs_RCG": pooled, "secondary_interaction": secondary,
              "full4000_decision": {"primary": "fine.rcg64", "mandatory_strong_controls": ["rcg64.control", "mean.control", "fine.rcg16.control"], "status1200": "fine64 resolves gain over coarse64, while MEAN/original RCG/fine16 contrasts cross zero; point>=2 is observed but at-least2 is not resolved by its CI", "criterion": "retain frozen primary; full4000 must assess stable>=2 vs native and superiority to all strong controls; CI crossing zero remains unresolved", "exposure": "1200 already exposed development, 4000 reused benchmark; neither is independent confirmation"},
              "limits": ["No masks, packets or fields opened; their seals/hash tables were not independently checked locally.", "Manifest file is not in fetched bundle; original prediction manifest hash cannot be independently checked here.", "Reported four-category pixel counts are only checked via pooled net-I/U identities; arbitrary compensating count errors could pass.", "Per-draw/native-relative fine64 edit categories and GT-distance/object geometry are not saved in this bundle.", "Net I change is net TP change and net U change is net FP change for the same GT, but those nets do not identify additions versus deletions.", "Exact two-term attribution uses final arm union, class totals and native/base IoU; pooled pixel totals are not the macro estimand.", "All labels are post-hoc diagnostics; no control threshold or interaction is causal evidence."]}
    for name in filenames:
        if sha(args.source / name) != hashes[name]:
            raise ValueError("Source changed during verification")
    if sha(args.prior4000) != prior_hash:
        raise ValueError("Prior4000 changed during verification")
    report["runtime_seconds"] = time.monotonic() - start
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Frozen subtoken1200: independent I/U verification", "", f"All six-arm scores, paired2,000 RandomState(0) connected-photo confidence intervals, four-fold/two-batch scores and episode up/down/tie counts reproduce the completed report; maximum difference {max(errors):.3g} pp. All1200 native/RCG/MEAN counts also match the previously verified public4000 draws.", "", "| Fine64 minus arm | Gain [95% CI], pp | Up / down / tie |", "|---|---:|---:|"]
    for arm in ARMS[:-1]:
        value = recomputed["contrasts"]["fine.rcg64"][arm]
        lines.append(f"| {arm} | {value['gain']:+.6f} [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] | {value['up']} / {value['down']} / {value['tie']} |")
    lines += ["", "Stage arithmetic: fine64−native = (RCG−native) + (coarse64−RCG) + (fine64−coarse64).", "", f"{arithmetic['full_gain_pp']:+.6f} = {arithmetic['coarse_RCG_vs_native_pp']:+.6f} + {arithmetic['coarse_lambda_change_pp']:+.6f} + {arithmetic['fine_vs_coarse64_pp']:+.6f} pp.", "", "| Pair | Net TP contribution | Net FP contribution | Pooled net TP / FP change |", "|---|---:|---:|---:|"]
    for name, values in stages.items():
        lines.append(f"| {name} | {values['net_TP_contribution_pp']:+.6f} | {values['net_FP_contribution_pp']:+.6f} | {values['pooled_net_TP_change']} / {values['pooled_net_FP_change']} |")
    lines += ["", "The exact two-term class-macro identity is `DeltaI/U_final + J_base*(U_base-U_final)/U_final`, averaged across classes. It separates net TP from net FP; it cannot recover all four edit categories.", "", "Saved fine64 edits against RCG: " + json.dumps(pooled["fine.rcg64"]["reported_counts"]) + ". Their net TP/FP changes match I/U sums exactly. Individual categories were not recounted from masks.", "", f"Fine64 versus coarse64 yields +{stages['fine.rcg64 minus rcg64.control']['pooled_net_TP_change']:,} net TP with +{stages['fine.rcg64 minus rcg64.control']['pooled_net_FP_change']:,} net FP. These are pooled side-effect diagnostics, not class-macro contributions or direct addition/deletion counts.", "", "The full4000 readout keeps fine64 fixed and compares native, coarse64, MEAN and original fine16. A point win whose CI crosses zero does not resolve superiority; 1200's point>=2 with a CI below2 does not resolve stable>=2. Dataset reuse is not fresh confirmation.", "", "Unavailable locally: prediction/packet/field replay, original manifest file, per-draw four-way fine edit counts, raw-DINO origin, GT-distance/object geometry. No such fields are inferred.", ""]
    (args.out / "verification.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.out), "verification": report["verification"], "stage_arithmetic": arithmetic, "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
