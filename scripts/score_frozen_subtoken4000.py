#!/usr/bin/env python3
"""Score six fixed arms only after all 4,000 complete predictions are sealed.

Contract with the streaming producer:
  sealed.json: state=ALL_PREDICTIONS_SEALED, n=4000, arms; SHA256 of
  manifest.json/config.json; predictions/fields/inputs keyed by unique draw key.
  Each row carries an absolute original packet_export (packet_path accepted).
  Input receipts carry packet_sha256. Packed masks are uint8[131072]; fine
  scalar fields are float32[128,128]. Explicit aliases in sealed config are
  accepted as prediction_arm_aliases/field_arm_aliases, never as unhashed CLI.
  Manifest order must reproduce the existing public4000 sampled draws, including
  the natural repeated photo/class identity. Draw keys may be aliases when
  public_batch/source_key or public_key identifies the original sampled draw.

All output/field/input hashes for ALL draws are checked before any truth member
is accessed. Only this score phase opens packet['truth']; no model is imported.
The earlier1200 source must already be fully sealed AND CPU-scored. Accepted
counts files: episode_metrics.json (rows or flat arm records), episodes.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from ics.experiment import metric, photo_groups, sha, summarize, unpack

ARMS = ("native", "rcg", "mean.control", "rcg64.control", "fine.rcg16.control", "fine.rcg64")
FINE = ARMS[-2:]
EDIT_NAMES = ("add_TP", "add_FP", "delete_TP", "delete_FP")
ALIASES = {
    "native": ("native", "NATIVE"), "rcg": ("rcg", "RCG"),
    "mean.control": ("mean.control", "MEAN_CONTROL", "mean_control"),
    "rcg64.control": ("rcg64.control", "rcg64", "RCG64_CONTROL"),
    "fine.rcg16.control": ("fine.rcg16.control", "fine16", "FINE_RCG16_CONTROL"),
    "fine.rcg64": ("fine.rcg64", "fine64", "FINE_RCG64"),
}


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data["episodes"]


def validate_rows(rows, expected):
    if len(rows) != expected or len({r["key"] for r in rows}) != expected:
        raise ValueError(f"Require {expected} distinct sampled draw keys; do not deduplicate identities")
    for row in rows:
        if "/" in row["key"] or "\\" in row["key"] or row["key"] in (".", ".."):
            raise ValueError("Draw key cannot escape the sealed output directory")
        for field in ("fold", "c", "e"):
            row[field] = int(row[field])
    if expected == 4000 and (set(r["c"] for r in rows) != set(range(80)) or any(sum(r["fold"] == f for r in rows) != 1000 for f in range(4))):
        raise ValueError("Require original 80 classes and 1,000 sampled draws per fold")


def checked(path, digest):
    actual = sha(path)
    if actual != digest:
        raise ValueError("Changed sealed file: " + str(path))
    return actual


def choose_alias(keys, arm, explicit):
    if arm in explicit:
        if explicit[arm] not in keys:
            raise ValueError("Configured arm alias missing: " + arm)
        return explicit[arm]
    matches = [name for name in ALIASES[arm] if name in keys]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous arm {arm}: {matches}")
    return matches[0]


def packet_path(row, receipt, require_original=True):
    values = [str(source[name]) for source in (row, receipt) for name in ("packet_export", "packet_path", "packet") if name in source]
    if not values or len(set(values)) != 1:
        raise ValueError("Require one consistent absolute original packet path")
    path = Path(values[0])
    if not path.is_absolute() or (require_original and not re.search(r"/outputs/claude_official/root\d+/results/extent_v1/run/packets/[^/]+\.npz$", str(path))):
        raise ValueError("Require original absolute claude_official/root*/.../packets/*.npz path")
    return path


def public_key(row, prior):
    explicit = [row[k] for k in ("public_key", "public_draw_key", "prior_key") if k in row]
    if row["key"] in prior:
        explicit.append(row["key"])
    if explicit:
        if len(set(explicit)) != 1 or explicit[0] not in prior:
            raise ValueError("Conflicting/unknown original sampled draw key")
        return explicit[0]
    block = row.get("public_batch")
    if block is None:
        batch = str(row.get("batch", ""))
        match = re.fullmatch(r"(?:official)?(\d+)", batch)
        if match:
            block = int(match[1])
        else:
            match = re.search(r"/(?:run|root)(\d+)(?:/|$)", str(row.get("recheck_run", row.get("packet_export", row.get("packet_path", "")))))
            if match:
                block = int(match[1])
    source_key = row.get("source_key", row["key"].split(":")[-1])
    candidate = f"public{block}:{source_key}"
    if block is None or candidate not in prior:
        raise ValueError("Cannot resolve sampled draw without explicit public/source key: " + row["key"])
    return candidate


def same_identity(row, previous):
    return all(int(row[name]) == int(previous[name]) for name in ("c", "fold")) and all(Path(row[name]).name == Path(previous[name]).name for name in ("query", "support"))


def preflight(run, expected=4000, require_original_packets=True):
    """Read hashes and prediction/field members, never a packet truth member."""
    run = Path(run)
    seal_path = run / "sealed.json"
    seal_hash = sha(seal_path)
    seal = json.loads(seal_path.read_text())
    if seal.get("state") != "ALL_PREDICTIONS_SEALED" or seal.get("n") != expected:
        raise ValueError(f"Require all{expected} complete predictions sealed before opening GT")
    for filename, key in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256")):
        checked(run / filename, seal[key])
    config = json.loads((run / "config.json").read_text())
    if "n" in config and config["n"] != expected:
        raise ValueError("Sealed config cohort size differs")
    if config.get("query_gt_in_inference") is not False:
        raise ValueError("Sealed config must explicitly declare query_gt_in_inference=false")
    for path, digest in config.get("source_code_sha256", {}).items():
        checked(path, digest)
    rows = read_records(run / "manifest.json")
    validate_rows(rows, expected)
    keys = {row["key"] for row in rows}
    for table in ("predictions", "fields", "inputs"):
        if set(seal[table]) != keys:
            raise ValueError("Seal table does not exactly cover every draw: " + table)
    aliases = config.get("prediction_arm_aliases", {})
    advertised = set(seal["arms"])
    for arm in ARMS:
        if arm not in advertised:
            choose_alias(advertised, arm, aliases)
    receipts = {}
    for row in rows:
        key = row["key"]
        prediction, field = (run / folder / (key + ".npz") for folder in ("predictions", "fields"))
        checked(prediction, seal["predictions"][key])
        checked(field, seal["fields"][key])
        packet = packet_path(row, seal["inputs"][key], require_original_packets)
        checked(packet, seal["inputs"][key]["packet_sha256"])
        receipt = {"prediction": str(prediction), "prediction_sha256": seal["predictions"][key], "field": str(field), "field_sha256": seal["fields"][key], "packet": str(packet), "packet_sha256": seal["inputs"][key]["packet_sha256"]}
        with np.load(prediction, allow_pickle=False) as stored:
            mapping = {arm: choose_alias(stored.files, arm, aliases) for arm in ARMS}
            for name in mapping.values():
                value = stored[name]
                if value.dtype != np.uint8 or value.shape != (131072,):
                    raise ValueError("Invalid packed complete mask: " + key + " " + name)
        with np.load(field, allow_pickle=False) as stored:
            field_aliases = config.get("field_arm_aliases", {})
            for arm in FINE:
                value = stored[choose_alias(stored.files, arm, field_aliases)]
                if value.dtype != np.float32 or value.shape != (128, 128) or not np.isfinite(value).all():
                    raise ValueError("Invalid fixed fine scalar field: " + key + " " + arm)
        receipt["arm_aliases"] = mapping
        receipts[key] = receipt
    checked(seal_path, seal_hash)
    return rows, seal, config, receipts, seal_hash


def prior1200(run, full_prior):
    rows, seal, config, _, seal_hash = preflight(run, expected=1200, require_original_packets=False)
    state_path = Path(run) / "score_state.json"
    state = json.loads(state_path.read_text())
    if state.get("state") != "CPU_GT_SCORE_COMPLETE" or state.get("n") != 1200 or state.get("completed") != 1200:
        raise ValueError("Earlier1200 must be fully sealed and CPU-scored")
    checked(Path(run) / "report.json", state["report_sha256"])
    options = [Path(run) / name for name in ("episode_metrics.json", "episodes.jsonl")]
    files = [path for path in options if path.is_file()]
    if not files:
        raise FileNotFoundError("Need sealed/scored earlier1200 episode_metrics.json or episodes.jsonl")
    path = files[0]
    digest = sha(path)
    records = read_records(path)
    keyed = {}
    for record in records:
        key = record["key"]
        if "iu" in record:
            if key in keyed:
                raise ValueError("Duplicate earlier1200 count row")
            keyed[key] = dict(record["iu"])
        else:
            values = keyed.setdefault(key, {})
            if record["arm"] in values:
                raise ValueError("Duplicate earlier1200 arm count")
            values[record["arm"]] = [record["intersection"], record["union"]]
    if set(keyed) != {row["key"] for row in rows}:
        raise ValueError("Earlier1200 counts must cover every sealed sampled draw")
    result = {}
    for row in rows:
        key = public_key(row, full_prior)
        if key in result or not same_identity(row, full_prior[key]):
            raise ValueError("Earlier1200 original draw mapping is duplicate or changed")
        result[key] = {arm: keyed[row["key"]][choose_alias(keyed[row["key"]], arm, {})] for arm in ("rcg64.control", *FINE)}
    return result, {"run": str(Path(run).resolve()), "seal_sha256": seal_hash, "counts": str(path), "counts_sha256": digest, "score_state_sha256": sha(state_path), "report_sha256": state["report_sha256"], "n": len(result), "frozen_parameters": config["parameters"]}


def mask_counts(masks, truth):
    iu, edits = {}, {}
    native = masks["native"]
    for arm, mask in masks.items():
        iu[arm] = [int((mask & truth).sum()), int((mask | truth).sum())]
        add, delete = mask & ~native, native & ~mask
        edits[arm] = {"add_TP": int((add & truth).sum()), "add_FP": int((add & ~truth).sum()), "delete_TP": int((delete & truth).sum()), "delete_FP": int((delete & ~truth).sum())}
        if iu[arm][0] != iu["native"][0] + edits[arm]["add_TP"] - edits[arm]["delete_TP"] or iu[arm][1] != iu["native"][1] + edits[arm]["add_FP"] - edits[arm]["delete_FP"]:
            raise ValueError("Complete-native edit accounting does not reconstruct I/U")
    return iu, edits


def self_test():
    # Repeated episode identities are retained as two separately sampled keys.
    rows = [dict(key=f"public0:k{j}", c=0, fold=0, e=j, support="s.jpg", query="q.jpg") for j in range(2)]
    validate_rows(rows, 2)
    assert len(rows) == 2
    native = np.array([[1, 1], [0, 0]], dtype=bool)
    truth = np.array([[1, 0], [1, 0]], dtype=bool)
    changed = np.array([[0, 0], [1, 1]], dtype=bool)
    _, edits = mask_counts({"native": native, "fine.rcg64": changed}, truth)
    assert edits["fine.rcg64"] == dict(add_TP=1, add_FP=1, delete_TP=1, delete_FP=1)
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        run = base / "run"
        (run / "predictions").mkdir(parents=True)
        (run / "fields").mkdir()
        packet = base / "outputs/claude_official/root0/results/extent_v1/run/packets/k.npz"
        packet.parent.mkdir(parents=True)
        # Intentionally not an NPZ: preflight must hash, never decode this GT packet.
        packet.write_bytes(b"unopened-GT-sentinel")
        for row in rows:
            row["packet_export"] = str(packet)
            np.savez(run / "predictions" / (row["key"] + ".npz"), **{arm: np.zeros(131072, np.uint8) for arm in ARMS})
            np.savez(run / "fields" / (row["key"] + ".npz"), **{arm: np.zeros((128, 128), np.float32) for arm in FINE})
        write(run / "manifest.json", rows)
        write(run / "config.json", dict(query_gt_in_inference=False))
        seal = dict(state="ALL_PREDICTIONS_SEALED", n=2, arms=list(ARMS), manifest_sha256=sha(run / "manifest.json"), config_sha256=sha(run / "config.json"), predictions={r["key"]: sha(run / "predictions" / (r["key"] + ".npz")) for r in rows}, fields={r["key"]: sha(run / "fields" / (r["key"] + ".npz")) for r in rows}, inputs={r["key"]: dict(packet_sha256=sha(packet)) for r in rows})
        write(run / "sealed.json", seal)
        got = preflight(run, expected=2)
        assert len(got[0]) == 2
        (run / "predictions" / (rows[-1]["key"] + ".npz")).write_bytes(b"changed-last-output")
        try:
            preflight(run, expected=2)
        except ValueError as error:
            assert "Changed sealed file" in str(error)
        else:
            raise AssertionError("Last-output corruption was accepted")
        seal["state"] = "RUNNING_GPU_INFERENCE"
        write(run / "sealed.json", seal)
        try:
            preflight(run, expected=2)
        except ValueError as error:
            assert "before opening GT" in str(error)
        else:
            raise AssertionError("Partial inference seal was accepted")
    print("SELF_TEST_PASS: repeated draws, all four edits, exact I/U recomposition, complete seal barrier, last-output corruption, no GT packet decoding during preflight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prior4000", type=Path, default=Path("outputs/frozen_public4000_v1/episodes.jsonl"))
    parser.add_argument("--prior1200-run", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.run is None or args.out is None or args.prior1200_run is None:
        parser.error("--run, --out and --prior1200-run are required for full4000 scoring")
    if args.out.exists():
        raise FileExistsError("Use a fresh scoring output directory; retained outputs are never removed")
    started = time.monotonic()
    rows, seal, config, receipts, seal_hash = preflight(args.run)
    prior_hash = sha(args.prior4000)
    previous = read_records(args.prior4000)
    validate_rows(previous, 4000)
    prior = {r["key"]: r for r in previous}
    mappings = [public_key(row, prior) for row in rows]
    if mappings != [row["key"] for row in previous] or any(not same_identity(row, prior[key]) for row, key in zip(rows, mappings)):
        raise ValueError("Require all original4000 draws in original order and unchanged photo/class identity")
    earlier, prior_receipt = prior1200(args.prior1200_run, prior)
    if len(earlier) != 1200:
        raise ValueError("Earlier fine readout parity requires all1200 sealed/scored draws")
    if config.get("parameters") != prior_receipt["frozen_parameters"]:
        raise ValueError("Full4000 readout must retain the earlier1200 frozen parameters")
    # At this point ALL4000 prediction, field, packet and metadata hashes passed.
    arrays = {arm: [] for arm in ARMS}
    corrections = {arm: [] for arm in ARMS}
    episodes = []
    for number, (row, original_key) in enumerate(zip(rows, mappings), 1):
        receipt = receipts[row["key"]]
        checked(receipt["prediction"], receipt["prediction_sha256"])
        checked(receipt["packet"], receipt["packet_sha256"])
        with np.load(receipt["prediction"], allow_pickle=False) as stored:
            masks = {arm: unpack(stored[receipt["arm_aliases"][arm]]) for arm in ARMS}
        with np.load(receipt["packet"], allow_pickle=False) as stored:
            truth = unpack(stored["truth"])
        iu, edits = mask_counts(masks, truth)
        for arm in ARMS[:3]:
            if iu[arm] != prior[original_key]["iu"][arm]:
                raise ValueError("Prior4000 native/RCG/MEAN I/U mismatch: " + original_key + " " + arm)
        if original_key in earlier:
            for arm, expected in earlier[original_key].items():
                if iu[arm] != expected:
                    raise ValueError("Earlier1200 fine/RCG64 I/U mismatch: " + original_key + " " + arm)
        record = dict(row, public_key=original_key, iu=iu, edits_vs_native=edits)
        episodes.append(record)
        for arm in ARMS:
            arrays[arm].append(iu[arm])
            corrections[arm].append(dict(key=row["key"], c=row["c"], fold=row["fold"], batch=str(row.get("batch", "unspecified")), **edits[arm]))
        if number % 400 == 0:
            print(json.dumps(dict(scored=number, total=4000, seconds=time.monotonic() - started)), flush=True)
    checked(args.run / "sealed.json", seal_hash)
    checked(args.prior4000, prior_hash)
    checked(prior_receipt["counts"], prior_receipt["counts_sha256"])
    arrays = {arm: np.asarray(values, dtype=np.int64) for arm, values in arrays.items()}
    report, draws = summarize(rows, arrays, corrections)
    primary = "fine.rcg64"
    controls = ("mean.control", "rcg64.control", "fine.rcg16.control")
    baseline = report["contrasts"][primary]["native"]
    strong_resolved = all(report["contrasts"][primary][arm]["ci95"][0] > 0 for arm in controls)
    report.update(primary=primary, config=config, parameter_updates=False,
                  exposure="Public benchmark/development reuse; not independent confirmation",
                  baseline4000_IU_parity=dict(draws=4000, arms=list(ARMS[:3]), mismatches=0),
                  earlier1200_IU_parity=dict(draws=1200, arms=["rcg64.control", *FINE], mismatches=0),
                  raw_DINO_origin=dict(status="full4000 unavailable; DEV241 common origin remains separate", edits=None),
                  INSID3="full4000 unavailable; not substituted by a control",
                  primary_criterion=dict(observed_gain_at_least2_pp=baseline["gain"] >= 2, ci95_entirely_at_least2_pp=baseline["ci95"][0] >= 2, superiority_to_strong_controls_resolved=strong_resolved, strict_fullcohort_criterion_met=baseline["ci95"][0] >= 2 and strong_resolved, interpretation="CI crossing zero is unresolved. A lower CI below2 does not establish stable at-least2. Reused4000 cannot establish independent confirmation."),
                  runtime=dict(cpu_score_seconds=time.monotonic() - started, gpu_inference_seconds=seal.get("seconds")))
    for group in (*report["folds"].values(), *report["batchs"].values()):
        group["primary_gain_vs_controls"] = {arm: group["scores"][primary] - group["scores"][arm] for arm in ARMS if arm != primary}
    # Subgroup intervals share the full cohort's photograph multiplicities;
    # dependencies through draws outside a subgroup remain connected.
    groups = photo_groups(rows)
    group_count = int(groups.max()) + 1
    row_weights = np.stack([np.bincount(draw, minlength=group_count) for draw in draws])[:, groups]
    classes = np.array([row["c"] for row in rows])
    for field, table in (("fold", "folds"), ("batch", "batchs")):
        for label, group in report[table].items():
            selected = np.array([str(row.get(field, "unspecified")) == label for row in rows])
            samples = {arm: np.array([metric(value[selected], classes[selected], weight[selected]) for weight in row_weights]) for arm, value in arrays.items()}
            group["primary_paired_contrasts"] = {}
            for arm in ARMS[:-1]:
                delta = arrays[primary][selected, 0] / np.maximum(arrays[primary][selected, 1], 1) - arrays[arm][selected, 0] / np.maximum(arrays[arm][selected, 1], 1)
                group["primary_paired_contrasts"][arm] = dict(gain=group["scores"][primary] - group["scores"][arm], ci95=np.percentile(samples[primary] - samples[arm], [2.5, 97.5]).tolist(), up=int((delta > 1e-12).sum()), down=int((delta < -1e-12).sum()), tie=int((np.abs(delta) <= 1e-12).sum()))
    args.out.mkdir(parents=True)
    write(args.out / "manifest.json", rows)
    write(args.out / "receipt.json", dict(source_run=str(args.run.resolve()), source_seal_sha256=seal_hash, manifest_sha256=seal["manifest_sha256"], config_sha256=seal["config_sha256"], scorer_sha256=sha(Path(__file__)), prior4000=str(args.prior4000.resolve()), prior4000_sha256=prior_hash, prior1200=prior_receipt, inputs=receipts, all4000_validated_before_first_GT=True))
    np.savez_compressed(args.out / "counts.npz", **{"iu:" + arm: value for arm, value in arrays.items()}, **{"edits_vs_native:" + arm: np.array([[record[name] for name in EDIT_NAMES] for record in corrections[arm]], dtype=np.int64) for arm in ARMS})
    np.save(args.out / "bootstrap_photo_draws.npy", draws)
    (args.out / "episodes.jsonl").write_text("".join(json.dumps(record) + "\n" for record in episodes))
    write(args.out / "episode_metrics.json", episodes)
    write(args.out / "report.json", report)
    lines = ["# Frozen subtoken: all4,000 sampled draws", "", "| Arm | Class-summed macro mIoU |", "|---|---:|"]
    lines += [f"| {arm} | {report['scores'][arm]:.6f} |" for arm in ARMS]
    lines += ["", "Fine lambda64 minus controls (paired2,000 RandomState(0) connected-photo95% CI):", ""]
    for arm in ARMS[:-1]:
        result = report["contrasts"][primary][arm]
        lines.append(f"- {arm}: {result['gain']:+.6f} [{result['ci95'][0]:+.6f}, {result['ci95'][1]:+.6f}] pp; up/down/tie {result['up']}/{result['down']}/{result['tie']}.")
    lines += ["", "All4000 native/RCG/MEAN and all1200 earlier lambda64/fine16/fine64 I/U match exactly. All sampled draws, including repeated identities, are retained.", "Raw DINO origin on4000 is unavailable; DEV241 remains a separate common-origin comparison. Edit counts here are relative to complete native.", "CI crossing zero is unresolved; point wins do not establish strong-control superiority. The criterion report separately records the point>=2 and CI entirely>=2 conditions. This reused benchmark is not fresh confirmation."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    write(args.out / "score_state.json", dict(state="CPU_GT_SCORE_COMPLETE", completed=4000, n=4000, report_sha256=sha(args.out / "report.json")))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
