#!/usr/bin/env python3
"""CPU-only diagnostic audit of sealed replay masks against fixed cached native.

Query truth is read only to score already-sealed masks and label four edit actions.
This script never chooses masks, changes either baseline, or imports a model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ics.experiment import load_rows, packet, sha, summarize

PIXELS = 1024 * 1024
PACKED_BYTES = PIXELS // 8
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
ACTIONS = ("add_TP", "delete_FP", "delete_TP", "add_FP")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def checked_hash(path, expected, description):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"{description} changed: {path}: {actual} != {expected}")
    return actual


def packed(value, label):
    if value.dtype != np.uint8 or value.shape != (PACKED_BYTES,):
        raise ValueError(f"{label}: expected {PACKED_BYTES} uint8 packed bytes")
    return value


def count(value):
    return int(POPCOUNT[value].sum(dtype=np.int64))


def iu(mask, truth):
    return [count(mask & truth), count(mask | truth)]


def actions(mask, baseline, truth, row):
    add, delete = mask & ~baseline, baseline & ~mask
    return dict(key=row["key"], c=row["c"], fold=row["fold"],
                batch=str(row.get("batch", "unspecified")),
                add_TP=count(add & truth), delete_FP=count(delete & ~truth),
                delete_TP=count(delete & truth), add_FP=count(add & ~truth))


def read_run(path, expected_n):
    seal_path = path / "sealed.json"
    seal = json.loads(seal_path.read_text())
    checks = {"seal_sha256": sha(seal_path)}
    checks["manifest_sha256"] = checked_hash(
        path / "manifest.json", seal["manifest_sha256"], "Sealed manifest")
    raw_rows = json.loads((path / "manifest.json").read_text())
    rows = load_rows(path / "manifest.json")
    if len(rows) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} episodes, found {len(rows)}")
    if any(raw["key"] != row["key"] for raw, row in zip(raw_rows, rows)):
        raise ValueError(f"{path}: stored key disagrees with fold/episode/class")
    keys = {row["key"] for row in rows}
    if set(seal["predictions"]) != keys:
        raise ValueError(f"{path}: sealed prediction key set differs from manifest")
    for seal_key, filename in (("protocol_sha256", "protocol.json"),
                               ("config_sha256", "config.json"),
                               ("arm_bases_sha256", "arm_bases.pt")):
        if seal_key in seal:
            checks[seal_key] = checked_hash(path / filename, seal[seal_key], filename)
    if "inputs" in seal and set(seal["inputs"]) != keys:
        raise ValueError(f"{path}: sealed input key set differs from manifest")
    return rows, seal, checks


def markdown(report):
    drift = report["native_drift"]
    lines = ["# Native replay audit: complete DEV241", "",
             f"State: **{report['state']}**. **{report['public_replay_state']}**.", "",
             "Question: quantify the difference between the fixed cached-native packets and the sealed",
             "tapped-native replay, and keep every candidate comparison explicit under both baselines.",
             "Outcome: preserve and reuse the completed cohort; public replay on discrepant cases is",
             "still required to locate the cause. Pixel drift is neither automatic invalidation nor",
             "proof of full native identity. CPU mask analysis only; no encoder/model/GPU execution.", "",
             f"COCO-20i 1-shot, frozen DINOv3, 1024 masks; {report['n']} episodes, "
             f"{report['classes']} classes, {report['photo_groups']} connected photograph groups. "
             "All cases are DEV, not independent confirmation. Seed 0 follows the current cohort record;",
             "the exported source/RCG episode rows do not independently encode a per-row seed field.",
             "Query GT was used only for class-summed I/U scoring and labeled four-action diagnostics,",
             "after prediction/input seals were checked. No GT-based routing or mask choice occurred.", "",
             f"Drift: **{drift['episodes_with_drift']}/{report['n']} episodes**, "
             f"**{drift['total_mismatched_pixels']} pixels** overall. Maximum per-episode difference "
             f"is {drift['maximum_mismatched_pixels']} pixels "
             f"({100 * drift['maximum_mismatched_ratio']:.8f}% of 1024²).", "",
             "| Complete sealed arm | Class mIoU | Gain vs cached-native [95% CI] | Gain vs replayed-native [95% CI] |",
             "|---|---:|---:|---:|"]
    for arm, score in report["scores"].items():
        cells = []
        for base in ("cached_native", "replayed_native"):
            item = report["contrasts_vs_baseline"].get(arm, {}).get(base)
            cells.append("—" if item is None else
                         f"{item['gain']:+.6f} [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}]")
        lines.append(f"| {arm} | {score:.6f} | {cells[0]} | {cells[1]} |")
    for field in ("folds", "batches"):
        lines += ["", f"## {field.capitalize()}: fixed scores and explicit baseline gains", "",
                  "| Stratum | n | cached / replayed native | Replay minus cached | D / concat / delta gain vs cached | D / concat / delta gain vs replayed |",
                  "|---|---:|---:|---:|---:|---:|"]
        for label, result in report[field].items():
            s = result["scores"]
            arms = ("multilayer", "multilayer.control", "multilayer.delta.control")
            vals = [" / ".join(f"{result['gain_vs_baseline'][base][a]:+.6f}" for a in arms)
                    for base in ("cached_native", "replayed_native")]
            lines.append(f"| {label} | {result['n']} | {s['cached_native']:.6f} / "
                         f"{s['replayed_native']:.6f} | {s['replayed_native'] - s['cached_native']:+.6f} "
                         f"| {vals[0]} | {vals[1]} |")
    lines += ["", "## All discrepant episode keys", "",
              "Ratios below use the full 1024² image, not the foreground area.", "",
              "| Key | Fold | Batch | Different pixels | Ratio | Replay minus cached episode IoU (pp) |",
              "|---|---:|---|---:|---:|---:|"]
    for item in drift["discrepant_episodes"]:
        lines.append(f"| {item['key']} | {item['fold']} | {item['batch']} | "
                     f"{item['mismatched_pixels']} | {item['mismatched_ratio']:.9f} | "
                     f"{item['replayed_minus_cached_episode_iou']:+.9f} |")
    lines += ["", "## Four actions: scoring diagnostics only", "",
              "| Arm | Fixed baseline | add TP | delete FP | delete TP | add FP |",
              "|---|---|---:|---:|---:|---:|"]
    for base, arms in report["corrections_vs_baseline"].items():
        for arm, counts in arms.items():
            lines.append(f"| {arm} | {base} | " + " | ".join(str(counts[a]) for a in ACTIONS) + " |")
    lines += ["", "## Integrity and remaining limits", "",
              "- Source and RCG manifests, all 241 prediction SHA256 values for each run, and all 241",
              "  original packet SHA256 values were checked. Exact source/RCG key sets and episode",
              "  metadata agree. RCG config, D protocol and D arm-basis hashes were checked.",
              "- Stored first-episode public-versus-tapped audit is reproduced in report.json; this is",
              "  prior-run evidence. This CPU task does not execute any public replay.",
              "- Raw-layer and input-feature SHA256 values are not reread: they are not scoring inputs.",
              "  D raw score packets lack seal entries and are not scoring inputs either.",
              "- Paired uncertainty uses the unchanged 2,000 RandomState(0) connected-photo draws from",
              "  ics.experiment.summarize. The two baseline analyses use exactly equal draws.",
              "- Candidate/control pairwise contrasts, per-class/per-batch action counts, all episode",
              "  I/U counts and full hash receipts are retained in the owned output directory.", "",
              "Sources:", ""]
    for key, value in report["sources"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", f"CPU audit runtime: {report['runtime_seconds']:.3f} seconds.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Original cached packet root (e.g. demo9_extent)")
    parser.add_argument("--source", required=True, type=Path,
                        help="Sealed complete forward run containing native replay and candidates")
    parser.add_argument("--rcg", required=True, type=Path, help="Sealed same-cohort RCG run")
    parser.add_argument("--out", required=True, type=Path, help="Fresh owned output directory")
    parser.add_argument("--expected-n", type=int, default=241)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh owned output directory; existing results are never replaced")
    started = time.monotonic()
    rows, source_seal, source_checks = read_run(args.source, args.expected_n)
    rcg_rows, rcg_seal, rcg_checks = read_run(args.rcg, args.expected_n)
    if source_seal.get("state") != "ALL_PREDICTIONS_SEALED":
        raise ValueError("Source run is not completely sealed")
    source_keys = {r["key"] for r in rows}
    rcg_lookup = {r["key"]: r for r in rcg_rows}
    if source_keys != set(rcg_lookup):
        raise ValueError("Source and RCG cohort key sets differ")
    for row in rows:
        other = rcg_lookup[row["key"]]
        for field in ("fold", "e", "c", "support", "query", "batch", "split", "seed"):
            if row.get(field) != other.get(field):
                raise ValueError(f"Cohort metadata differs: {row['key']}: {field}")
    if "inputs" not in rcg_seal:
        raise ValueError("RCG lacks the sealed original-packet input hashes")
    # Check the entire sealed cohort before opening query GT for diagnostics.
    receipts = []
    for row in rows:
        key = row["key"]
        receipts.append(dict(key=key,
            forward_prediction_sha256=checked_hash(args.source / "predictions" / f"{key}.npz",
                source_seal["predictions"][key], "Forward prediction"),
            rcg_prediction_sha256=checked_hash(args.rcg / "predictions" / f"{key}.npz",
                rcg_seal["predictions"][key], "RCG prediction"),
            packet_sha256=checked_hash(packet(args.root, row),
                rcg_seal["inputs"][key]["packet_sha256"], "Original cached packet")))
    print(json.dumps({"stage": "all_sealed_scoring_inputs_verified", "n": len(rows)}), flush=True)
    arrays, corrections_cached, corrections_replayed = {}, {}, {}
    details, discrepant, drift_by_episode = [], [], []
    forward_arms, rcg_arms = None, None
    for row in rows:
        key = row["key"]
        with np.load(packet(args.root, row), allow_pickle=False) as f:
            truth = packed(f["truth"], f"{key}: truth")
            cached = packed(f["native"], f"{key}: cached native")
        with np.load(args.source / "predictions" / f"{key}.npz", allow_pickle=False) as f:
            current = set(f.files)
            if forward_arms is None:
                forward_arms = current
            if current != forward_arms or "native" not in current:
                raise ValueError(f"{key}: incomplete/inconsistent forward arm set")
            masks = {("replayed_native" if name == "native" else name):
                     packed(f[name], f"{key}: {name}") for name in f.files}
        with np.load(args.rcg / "predictions" / f"{key}.npz", allow_pickle=False) as f:
            current = set(f.files)
            if rcg_arms is None:
                rcg_arms = current
            if current != rcg_arms:
                raise ValueError(f"{key}: incomplete/inconsistent RCG arm set")
            if any(name in masks for name in current):
                raise ValueError("RCG and forward arm names collide")
            masks.update({name: packed(f[name], f"{key}: {name}") for name in f.files})
        masks = {"cached_native": cached, **masks}
        replayed = masks["replayed_native"]
        changed = count(cached ^ replayed)
        ciu, riu = iu(cached, truth), iu(replayed, truth)
        drift = dict(key=key, c=row["c"], fold=row["fold"],
                     batch=str(row.get("batch", "unspecified")), mismatched_pixels=changed,
                     mismatched_ratio=changed / PIXELS,
                     replayed_minus_cached_episode_iou=100 *
                     (riu[0] / max(riu[1], 1) - ciu[0] / max(ciu[1], 1)))
        drift_by_episode.append(drift)
        if changed:
            discrepant.append(drift)
        for arm, mask in masks.items():
            counts = iu(mask, truth)
            arrays.setdefault(arm, []).append(counts)
            cc, cr = actions(mask, cached, truth, row), actions(mask, replayed, truth, row)
            corrections_cached.setdefault(arm, []).append(cc)
            corrections_replayed.setdefault(arm, []).append(cr)
            details.append(dict(key=key, c=row["c"], fold=row["fold"], batch=cc["batch"],
                                arm=arm, intersection=counts[0], union=counts[1],
                                actions_vs_cached_native={a: cc[a] for a in ACTIONS},
                                actions_vs_replayed_native={a: cr[a] for a in ACTIONS}))
    arrays = {name: np.asarray(values, dtype=np.int64) for name, values in arrays.items()}
    if any(len(values) != args.expected_n for values in arrays.values()):
        raise ValueError("Incomplete arm cohort")
    print(json.dumps({"stage": "packed_masks_audited", "mismatched_episodes": len(discrepant),
                      "mismatched_pixels": sum(d["mismatched_pixels"] for d in discrepant)}), flush=True)
    summaries, draws_by_baseline = {}, {}
    for baseline, corrections in (("cached_native", corrections_cached),
                                  ("replayed_native", corrections_replayed)):
        named_arrays = {("native" if name == baseline else name): value for name, value in arrays.items()}
        named_corrections = {("native" if name == baseline else name): value
                             for name, value in corrections.items()}
        summaries[baseline], draws_by_baseline[baseline] = summarize(rows, named_arrays, named_corrections)
    if not np.array_equal(draws_by_baseline["cached_native"], draws_by_baseline["replayed_native"]):
        raise ValueError("The baseline summaries used different bootstrap draws")
    # An input manifest/seal edit during the read would make the audit ambiguous.
    for path, checks in ((args.source, source_checks), (args.rcg, rcg_checks)):
        checked_hash(path / "sealed.json", checks["seal_sha256"], "Seal during audit")
        checked_hash(path / "manifest.json", checks["manifest_sha256"], "Manifest during audit")
    cached_summary = summaries["cached_native"]
    contrasts = {arm: {} for arm in arrays}
    for base, summary in summaries.items():
        for arm in arrays:
            if arm != base:
                contrasts[arm][base] = summary["contrasts"][arm]["native"]
    report = {k: cached_summary[k] for k in ("n", "classes", "photo_groups", "largest_photo_group", "bootstrap", "exposure")}
    report.update(state="native drift observed" if discrepant else "no native mask drift observed",
                  public_replay_state="public replay on discrepant cases still pending",
                  query_gt_usage="diagnostic scoring and four-action labels only; no inference, routing or selection",
                  sources=dict(root=str(args.root), source=str(args.source), rcg=str(args.rcg), out=str(args.out)),
                  scores={name: cached_summary["scores"]["native" if name == "cached_native" else name]
                          for name in arrays},
                  contrasts_vs_baseline=contrasts,
                  candidate_control_contrasts={("cached_native" if arm == "native" else arm):
                                               {base: value for base, value in values.items() if base != "native"}
                                               for arm, values in cached_summary["contrasts"].items()},
                  integrity=dict(source=source_checks, rcg=rcg_checks, n_forward_predictions_checked=len(receipts),
                                 n_rcg_predictions_checked=len(receipts), n_original_packets_checked=len(receipts),
                                 cohort_key_and_metadata_identity=True, raw_layer_hashes_checked=False,
                                 feature_hashes_checked=False, raw_score_packets_sealed=False,
                                 source_query_labels_opened_at_sealing=source_seal.get("query_labels_opened")),
                  native_drift=dict(episodes_with_drift=len(discrepant), exact_episodes=len(rows)-len(discrepant),
                                    total_mismatched_pixels=sum(d["mismatched_pixels"] for d in discrepant),
                                    maximum_mismatched_pixels=max(d["mismatched_pixels"] for d in drift_by_episode),
                                    maximum_mismatched_ratio=max(d["mismatched_ratio"] for d in drift_by_episode),
                                    discrepant_episodes=discrepant),
                  runtime_seconds=time.monotonic() - started)
    for field in ("folds", "batches"):
        report[field] = {}
        summary_field = "batchs" if field == "batches" else field
        for label, sub in cached_summary[summary_field].items():
            scores = {("cached_native" if arm == "native" else arm): value for arm, value in sub["scores"].items()}
            report[field][label] = dict(n=sub["n"], scores=scores,
                gain_vs_baseline={base: {arm: value - scores[base] for arm, value in scores.items()}
                                  for base in summaries})
    for suffix in ("vs_native", "by_class", "by_batch"):
        report["corrections_" + ("vs_baseline" if suffix == "vs_native" else suffix)] = {
            base: {("native" if arm == "native" else arm): value
                   for arm, value in summary["corrections_" + suffix].items()}
            for base, summary in summaries.items()}
        for base, values in report["corrections_" + ("vs_baseline" if suffix == "vs_native" else suffix)].items():
            values[base] = values.pop("native")
    audit_path = args.source / "audits.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        report["prior_first_public_vs_tapped_native"] = audit.get("first_public_vs_tapped_native")
        report["integrity"]["audits_sha256_unsealed_metadata"] = sha(audit_path)
    args.out.mkdir(parents=True)
    write_json(args.out / "report.json", report)
    write_json(args.out / "episode_metrics.json", details)
    write_json(args.out / "hash_receipts.json", receipts)
    write_json(args.out / "native_drift_by_episode.json", drift_by_episode)
    np.save(args.out / "bootstrap_photo_draws.npy", draws_by_baseline["cached_native"])
    (args.out / "report.md").write_text(markdown(report))
    print(json.dumps(dict(state=report["state"], n=report["n"], scores=report["scores"],
                         native_drift=report["native_drift"],
                         contrasts_vs_baseline=report["contrasts_vs_baseline"],
                         runtime_seconds=report["runtime_seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
