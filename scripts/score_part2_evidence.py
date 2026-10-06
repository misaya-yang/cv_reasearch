#!/usr/bin/env python3
"""Score sealed complete Part2 interventions and matching existing controls on CPU.

No partial/prefinal run is eligible. Query truth is first opened after every
prediction, source, manifest and original comparison packet has been checked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ics.experiment import packet, sha, summarize, unpack
from run_part2_evidence import verify_prefinal


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def identity(row):
    return tuple(row[name] for name in ("fold", "e", "c", "support", "query"))


def verify_complete(run, expected):
    seal = json.loads((run / "sealed.json").read_text())
    if seal.get("state") != "ALL_PREDICTIONS_SEALED":
        raise ValueError("Complete native CRF predictions must be sealed before scoring")
    config = verify_prefinal(run, seal)
    if sha(run / "finalize_audit.json") != seal["finalize_audit_sha256"]:
        raise ValueError("Finalization audit changed")
    rows = json.loads((run / "manifest.json").read_text())
    keys = {row["key"] for row in rows}
    if len(keys) != len(rows) or (expected is not None and len(rows) != expected):
        raise ValueError("Duplicate keys or unexpected episode count")
    if keys != set(seal["predictions"]) or keys != set(seal["inputs"]):
        raise ValueError("Incomplete prediction/input seal")
    for row in rows:
        key = row["key"]
        if sha(run / "predictions" / f"{key}.npz") != seal["predictions"][key]:
            raise ValueError(f"Prediction changed: {key}")
        if sha(packet(config["root"], row)) != seal["inputs"][key]["packet_sha256"]:
            raise ValueError(f"Original comparison packet changed: {key}")
    return config, rows, seal


def verify_controls(run, rows):
    seal = json.loads((run / "sealed.json").read_text())
    if seal.get("state") != "ALL_PREDICTIONS_SEALED":
        raise ValueError("External complete controls must be sealed")
    if sha(run / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("External control manifest changed")
    source = {row["key"]: row for row in json.loads((run / "manifest.json").read_text())}
    for row in rows:
        key = row["key"]
        if key not in source or identity(source[key]) != identity(row):
            raise ValueError(f"External control episode mismatch: {key}")
        if sha(run / "predictions" / f"{key}.npz") != seal["predictions"][key]:
            raise ValueError(f"External control prediction changed: {key}")
    return seal


def correction(mask, native, truth, row):
    add, delete = mask & ~native, native & ~mask
    return dict(c=row["c"], batch=str(row.get("batch", "fresh600_dev")),
                add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                delete_FP=int((delete & ~truth).sum()), delete_TP=int((delete & truth).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-run", type=Path,
                        help="Matching sealed RCG and MEAN_CONTROL predictions")
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()
    config, rows, seal = verify_complete(args.out, args.expected)
    external_seal = verify_controls(args.control_run, rows) if args.control_run else None
    arms = ["native"] + list(seal["arms"])
    external = {"rcg": "RCG", "mean.control": "MEAN_CONTROL"} if external_seal else {}
    if set(arms) & set(external):
        raise ValueError("Control names collide with candidate arm names")
    arms += list(external)
    arrays = {arm: np.zeros((len(rows), 2), np.int64) for arm in arms}
    edits = {arm: [] for arm in arms if arm != "native"}
    details = []
    for n, row in enumerate(rows):
        key = row["key"]
        with np.load(packet(config["root"], row), allow_pickle=False) as cached:
            truth, native = unpack(cached["truth"]), unpack(cached["native"])
        masks = {"native": native}
        with np.load(args.out / "predictions" / f"{key}.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(seal["arms"]):
                raise ValueError(f"Candidate arm set changed: {key}")
            masks.update({arm: unpack(saved[arm]) for arm in seal["arms"]})
        if external:
            with np.load(args.control_run / "predictions" / f"{key}.npz", allow_pickle=False) as saved:
                masks.update({arm: unpack(saved[source]) for arm, source in external.items()})
        one = dict(key=key, fold=row["fold"], c=row["c"], metrics={})
        for arm, mask in masks.items():
            i, u = int((mask & truth).sum()), int((mask | truth).sum())
            arrays[arm][n] = i, u
            one["metrics"][arm] = dict(intersection=i, union=u)
            if arm != "native":
                edits[arm].append(correction(mask, native, truth, row))
        details.append(one)
    report, draws = summarize(rows, arrays, edits)
    report.update(state="COMPLETE_MASKS_SCORED", resolution=[1024, 1024],
                  metric="mean over classes of summed intersection / summed union",
                  primary="part2.bg_lse", baseline="native",
                  same_cache_baseline="native.cache.control",
                  controls=["native.cache.control", "part2.bg_max.control"] + list(external),
                  finalizer="Unchanged native FoRIS CUDA CRF after its binarizer",
                  query_truth_opened_only_after_complete_seal=True,
                  prediction_seal_sha256=sha(args.out / "sealed.json"),
                  inference_config_sha256=seal["config_sha256"],
                  native_drift_audit_sha256=seal["finalize_audit_sha256"],
                  scorer_sha256=sha(__file__), seed="Inherited source manifest; no new draws",
                  feature_precision="Retained post-Part1 FP16 cache, cast to FP32; no reencoding",
                  parameter_provenance="Native constants; fixed BG LSE candidate and max control",
                  external_control_provenance=(dict(run=str(args.control_run),
                    seal_sha256=sha(args.control_run / "sealed.json"), arms=external)
                    if external_seal else None))
    np.savez_compressed(args.out / "counts.npz", **arrays)
    np.save(args.out / "bootstrap_photo_draws.npy", draws)
    write_json(args.out / "episode_metrics.json", details)
    write_json(args.out / "report.json", report)
    text = ["# Complete Part2 evidence comparison", "",
            f"{len(rows)} reused DEV episodes; 1024 pixels; class-summed IoU; unchanged native CRF.",
            "The post-Part1 FP16 replay is compared with both original native and same-cache native.", "",
            "| Arm | mIoU | Gain vs original native | 95% paired CI |",
            "|---|---:|---:|---:|"]
    for arm, score in report["scores"].items():
        contrast = report["contrasts"].get(arm, {}).get("native")
        gain = f"{contrast['gain']:+.3f}" if contrast else "—"
        interval = f"[{contrast['ci95'][0]:+.3f}, {contrast['ci95'][1]:+.3f}]" if contrast else "—"
        text.append(f"| {arm} | {score:.3f} | {gain} | {interval} |")
    text += ["", "Primary comparisons against same-cache native, BG max and available RCG/MEAN",
             "controls, fold/batch gains, and harmful TP deletions are in report.json.",
             "This reused cohort is development evidence, not independent confirmation."]
    (args.out / "report.md").write_text("\n".join(text) + "\n")
    print(json.dumps(dict(state=report["state"], n=len(rows), scores=report["scores"],
                         primary=report["contrasts"]["part2.bg_lse"])), flush=True)


if __name__ == "__main__":
    main()
