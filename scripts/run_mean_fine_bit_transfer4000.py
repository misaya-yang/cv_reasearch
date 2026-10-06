#!/usr/bin/env python3
"""One fixed CPU packed-mask control; seal all4000 before opening GT.

R=RCG, M=MEAN, V=fine16. A=V&~R; B=R&~V; C=(M|A)&~B.
This mask-only control is separate from scalar residual graft and does not
replace the frozen fine64 primary. No encoder, image or scalar field is read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from ics.experiment import metric, photo_groups, sha, summarize

SOURCE_ARMS = ("native", "rcg", "mean.control", "rcg64.control", "fine.rcg16.control", "fine.rcg64")
CONTROL = "mean_fine_bit_edit_transfer.control"
COMPARE = ("native", "mean.control", "fine.rcg16.control", "fine.rcg64")
EDIT_NAMES = ("add_TP", "add_FP", "delete_TP", "delete_FP")
POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError("Changed sealed source: " + str(path))


def packed(value):
    if value.dtype != np.uint8 or value.shape != (131072,):
        raise ValueError("Require packed complete1024 mask")
    return value


def transfer(r, m, v):
    return (m | (v & ~r)) & ~(r & ~v)


def source_metadata(source_run, source_scored):
    seal_path = source_run / "sealed.json"
    seal = json.loads(seal_path.read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 4000 or set(seal["arms"]) != set(SOURCE_ARMS):
        raise ValueError("Require complete sealed six-arm4000 source")
    for name in ("manifest", "config"):
        checked(source_run / (name + ".json"), seal[name + "_sha256"])
    rows = json.loads((source_run / "manifest.json").read_text())
    if len(rows) != 4000 or len({r["key"] for r in rows}) != 4000 or any(sum(r["fold"] == f for r in rows) != 1000 for f in range(4)):
        raise ValueError("Preserve all4000 original sampled draw keys and folds")
    state = json.loads((source_scored / "score_state.json").read_text())
    receipt = json.loads((source_scored / "receipt.json").read_text())
    if state["state"] != "CPU_GT_SCORE_COMPLETE" or state["n"] != 4000 or state["completed"] != 4000:
        raise ValueError("Require completed source scoring")
    checked(source_scored / "report.json", state["report_sha256"])
    checked(seal_path, receipt["source_seal_sha256"])
    if Path(receipt["source_run"]).resolve() != source_run.resolve() or receipt["config_sha256"] != seal["config_sha256"] or receipt["manifest_sha256"] != seal["manifest_sha256"]:
        raise ValueError("Inference/scoring source identity mismatch")
    checked(source_scored / "manifest.json", seal["manifest_sha256"])
    if json.loads((source_scored / "manifest.json").read_text()) != rows:
        raise ValueError("Source scorer manifest order/photo identity mismatch")
    keys = {r["key"] for r in rows}
    if set(seal["predictions"]) != keys or set(seal["inputs"]) != keys:
        raise ValueError("Require a source receipt for every sampled draw")
    for row in rows:
        if "/" in row["key"] or "\\" in row["key"] or not Path(row["packet_export"]).is_absolute():
            raise ValueError("Unsafe draw key or nonabsolute original packet")
    return rows, seal, dict(source_run=str(source_run.resolve()), source_scored=str(source_scored.resolve()),
                            source_seal_sha256=sha(seal_path), source_manifest_sha256=seal["manifest_sha256"],
                            source_config_sha256=seal["config_sha256"], source_report_sha256=state["report_sha256"],
                            source_episodes_sha256=sha(source_scored / "episodes.jsonl"))


def make_one(job):
    key, source, digest, destination = job
    checked(source, digest)
    with np.load(source, allow_pickle=False) as stored:
        r, m, v = (packed(stored[name]) for name in ("rcg", "mean.control", "fine.rcg16.control"))
        candidate = transfer(r, m, v)
    path = Path(destination) / "predictions" / (key + ".npz")
    np.savez_compressed(path, **{CONTROL: candidate})
    return key, sha(path)


def infer(a):
    rows, source_seal, provenance = source_metadata(a.source_run, a.source_scored)
    if a.out.exists():
        raise FileExistsError("Fresh own namespace required; retain existing outputs")
    began = time.monotonic()
    # Validate every source complete-mask SHA before generating the fixed C.
    jobs = []
    for row in rows:
        path = a.source_run / "predictions" / (row["key"] + ".npz")
        checked(path, source_seal["predictions"][row["key"]])
        jobs.append((row["key"], str(path), source_seal["predictions"][row["key"]], str(a.out)))
    a.out.mkdir(parents=True)
    (a.out / "predictions").mkdir()
    write(a.out / "manifest.json", rows)
    config = dict(n=4000, assessed_control=CONTROL, original_primary_unchanged="fine.rcg64",
                  formula="R=rcg; M=mean.control; V=fine.rcg16.control; A=V&~R; B=R&~V; C=(M|A)&~B",
                  equivalent="use V at per-bit M/R agreement; otherwise keep M", parameter_search=False,
                  query_GT_read_in_inference=False, encoder_forwards=0, images_opened=0, scalar_fields_opened=0,
                  source=provenance, script_sha256=sha(Path(__file__)), statistics_sha256=sha(ROOT / "src/ics/experiment.py"),
                  workers=a.workers, exposure="all4000 reused public benchmark draws; not fresh confirmation",
                  role="necessary fixed cheapest mask-level same-information combination control; separate from scalar graft")
    write(a.out / "config.json", config)
    predictions = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for number, (key, digest) in enumerate(pool.map(make_one, jobs, chunksize=8), 1):
            predictions[key] = digest
            if number % 400 == 0:
                progress = dict(state="CPU_MASK_CONTROL_INFERENCE", completed=number, n=4000, seconds=time.monotonic() - began, query_GT_opened=False)
                write(a.out / "state.json", progress)
                print(json.dumps(progress), flush=True)
    checked(a.source_run / "sealed.json", provenance["source_seal_sha256"])
    if len(predictions) != 4000:
        raise ValueError("Incomplete control construction")
    seal = dict(state="ALL_PREDICTIONS_SEALED", n=4000, arms=[CONTROL],
                manifest_sha256=sha(a.out / "manifest.json"), config_sha256=sha(a.out / "config.json"),
                predictions=predictions, source=provenance, inputs=source_seal["inputs"],
                seconds=time.monotonic() - began, query_GT_opened=False, encoder_forwards=0)
    write(a.out / "sealed.json", seal)
    write(a.out / "state.json", dict(state="ALL_PREDICTIONS_SEALED", completed=4000, n=4000))
    print("ALL4000_BIT_TRANSFER_MASKS_SEALED", flush=True)


def count(value):
    return int(POPCOUNT[value].sum(dtype=np.int64))


def score_one(job):
    row, source_path, source_hash, candidate_path, candidate_hash, packet_hash, expected = job
    checked(source_path, source_hash)
    checked(candidate_path, candidate_hash)
    checked(row["packet_export"], packet_hash)
    with np.load(source_path, allow_pickle=False) as stored:
        masks = {arm: packed(stored[arm]) for arm in SOURCE_ARMS}
    with np.load(candidate_path, allow_pickle=False) as stored:
        masks[CONTROL] = packed(stored[CONTROL])
    with np.load(row["packet_export"], allow_pickle=False) as stored:
        truth = packed(stored["truth"])
    iu = {arm: [count(mask & truth), count(mask | truth)] for arm, mask in masks.items()}
    for arm in SOURCE_ARMS:
        if iu[arm] != expected[arm]:
            raise ValueError("Original six-arm I/U parity failed: " + row["key"] + " " + arm)
    edits = {}
    for base in (*COMPARE, "rcg"):
        add, delete = masks[CONTROL] & ~masks[base], masks[base] & ~masks[CONTROL]
        values = dict(add_TP=count(add & truth), add_FP=count(add & ~truth), delete_TP=count(delete & truth), delete_FP=count(delete & ~truth))
        if iu[CONTROL][0] != iu[base][0] + values["add_TP"] - values["delete_TP"] or iu[CONTROL][1] != iu[base][1] + values["add_FP"] - values["delete_FP"]:
            raise ValueError("Control edit I/U identity failed")
        edits[base] = values
    return dict(row, iu=iu, control_edits=edits)


def score(a):
    began = time.monotonic()
    rows, source_seal, source = source_metadata(a.source_run, a.source_scored)
    seal = json.loads((a.out / "sealed.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 4000 or seal["source"] != source:
        raise ValueError("Require all4000 unchanged control masks sealed before GT")
    for name in ("manifest", "config"):
        checked(a.out / (name + ".json"), seal[name + "_sha256"])
    if json.loads((a.out / "manifest.json").read_text()) != rows:
        raise ValueError("Control manifest identity/order changed")
    previous = [json.loads(line) for line in (a.source_scored / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(previous) != 4000 or [r["key"] for r in previous] != [r["key"] for r in rows]:
        raise ValueError("Require every original source count in unchanged draw order")
    jobs = []
    # ALL4000 C/source/packet hashes and previous identities precede first GT.
    for row, old in zip(rows, previous):
        if any(row[k] != old[k] for k in ("key", "fold", "c", "support", "query")):
            raise ValueError("Source count photo/class identity mismatch")
        key = row["key"]
        sp, cp = a.source_run / "predictions" / (key + ".npz"), a.out / "predictions" / (key + ".npz")
        ph = source_seal["inputs"][key]["packet_sha256"]
        checked(sp, source_seal["predictions"][key]); checked(cp, seal["predictions"][key]); checked(row["packet_export"], ph)
        jobs.append((row, str(sp), source_seal["predictions"][key], str(cp), seal["predictions"][key], ph, old["iu"]))
    details = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for number, record in enumerate(pool.map(score_one, jobs, chunksize=8), 1):
            details.append(record)
            if number % 400 == 0:
                print(json.dumps(dict(state="CPU_PACKED_GT_SCORE", completed=number, n=4000, seconds=time.monotonic() - began)), flush=True)
    arrays = {arm: np.array([row["iu"][arm] for row in details], dtype=np.int64) for arm in (*SOURCE_ARMS, CONTROL)}
    corrections = {CONTROL: [dict(key=row["key"], c=row["c"], fold=row["fold"], batch=row["batch"], **row["control_edits"]["native"]) for row in details]}
    report, draws = summarize(rows, arrays, corrections)
    cls, groups = np.array([row["c"] for row in rows]), photo_groups(rows)
    weights = np.stack([np.bincount(draw, minlength=int(groups.max()) + 1) for draw in draws])[:, groups]
    # Original fine64 retains its original name/primary identity; add its paired
    # contrast explicitly because summarize's base list excludes that name.
    for field, table in ((None, None), ("fold", "folds"), ("batch", "batchs")):
        targets = [(None, report)] if field is None else list(report[table].items())
        for label, target in targets:
            selected = np.ones(4000, dtype=bool) if field is None else np.array([str(r[field]) == label for r in rows])
            samples = {arm: np.array([metric(arrays[arm][selected], cls[selected], w[selected]) for w in weights]) for arm in (CONTROL, *COMPARE)}
            contrasts = {}
            for base in COMPARE:
                delta = arrays[CONTROL][selected, 0] / np.maximum(arrays[CONTROL][selected, 1], 1) - arrays[base][selected, 0] / np.maximum(arrays[base][selected, 1], 1)
                contrasts[base] = dict(gain=target["scores"][CONTROL] - target["scores"][base], ci95=np.percentile(samples[CONTROL] - samples[base], [2.5, 97.5]).tolist(), up=int((delta > 1e-12).sum()), down=int((delta < -1e-12).sum()), tie=int((np.abs(delta) <= 1e-12).sum()))
            target["fixed_control_contrasts"] = contrasts
    report["contrasts"][CONTROL]["fine.rcg64"] = report["fixed_control_contrasts"]["fine.rcg64"]
    original = json.loads((a.source_scored / "report.json").read_text())
    if any(abs(report["scores"][arm] - original["scores"][arm]) > 1e-10 for arm in SOURCE_ARMS):
        raise ValueError("Source six-arm class scores did not reproduce")
    report.update(assessed_control=CONTROL, original_primary_unchanged="fine.rcg64", source=source,
                  control_prediction_seal_sha256=sha(a.out / "sealed.json"), original_six_arm_IU_parity=dict(n=4000, arms=list(SOURCE_ARMS), mismatches=0),
                  formula="C=(M|(V&~R))&~(R&~V)", parameter_search=False, encoder_forwards=0,
                  exposure="all4000 reused public benchmark draws; not fresh confirmation", scalar_graft="separate construction, not run or equated here",
                  runtime=dict(cpu_mask_infer_seconds=seal["seconds"], cpu_score_seconds=time.monotonic() - began, note="cached packed-mask operations only; upstream producer runtime is separate"))
    np.savez_compressed(a.out / "counts.npz", **{"iu:" + arm: value for arm, value in arrays.items()}, **{"edits:" + base: np.array([[r["control_edits"][base][name] for name in EDIT_NAMES] for r in details]) for base in (*COMPARE, "rcg")})
    (a.out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in details))
    write(a.out / "report.json", report)
    write(a.out / "receipt.json", dict(source=source, config_sha256=seal["config_sha256"], manifest_sha256=seal["manifest_sha256"], control_seal_sha256=sha(a.out / "sealed.json"), all4000_control_masks_sealed_before_GT=True, script_sha256=sha(Path(__file__))))
    lines = ["# Fixed MEAN/fine16 bit-edit transfer control: all4000 draws", "", f"`{CONTROL}` = `(M | (V & ~R)) & ~(R & ~V)`; use V at bit-level M/R agreement, otherwise keep M. No parameter sweep, encoder, new scalar field or primary replacement.", "", f"Complete class-summed macro mIoU: **{report['scores'][CONTROL]:.6f}**.", "", "| Fixed comparison | Gain [95% CI], pp | Up / down / tie |", "|---|---:|---:|"]
    for base, value in report["fixed_control_contrasts"].items():
        lines.append(f"| {base} | {value['gain']:+.6f} [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] | {value['up']} / {value['down']} / {value['tie']} |")
    lines += ["", "All4000 original six-arm I/U match exactly. All masks sealed before GT. All sampled draws and natural repeats retained. Paired2000 RandomState(0) connected-photo intervals; fold/batch contrasts and native-relative four-way edits are in JSON.", "This is the fixed cheapest mask-level combination control for the separate scalar graft. It does not claim equivalence to that scalar construction or replace fine64. Cached CPU runtime excludes upstream inference; benchmark reuse is not fresh confirmation."]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    write(a.out / "score_state.json", dict(state="CPU_GT_SCORE_COMPLETE", completed=4000, n=4000, report_sha256=sha(a.out / "report.json")))
    print("\n".join(lines), flush=True)


def self_test():
    cases = np.arange(8, dtype=np.uint8)
    r, m, v = ((cases & bit) != 0 for bit in (1, 2, 4))
    expected = np.where(m == r, v, m)
    assert np.array_equal((m | (v & ~r)) & ~(r & ~v), expected)
    assert np.array_equal(transfer(np.packbits(r), np.packbits(m), np.packbits(v)), np.packbits(expected))
    generator = np.random.RandomState(0)
    rr, mm, vv = [generator.randint(256, size=131072).astype(np.uint8) for _ in range(3)]
    cc = transfer(rr, mm, vv)
    expected = np.where(np.unpackbits(mm) == np.unpackbits(rr), np.unpackbits(vv), np.unpackbits(mm))
    assert np.array_equal(np.unpackbits(cc), expected)
    print("SELF_TEST_PASS: all8 binary cases and complete1024 packed mask match agreement-site V/otherwise M; no GT required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("infer", "score", "all"), default="all")
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-scored", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    a = parser.parse_args()
    if a.self_test:
        self_test(); return
    if a.source_run is None or a.source_scored is None or a.out is None:
        parser.error("source-run/source-scored/out are required")
    if a.stage in ("infer", "all"):
        infer(a)
    if a.stage in ("score", "all"):
        score(a)


if __name__ == "__main__":
    main()
