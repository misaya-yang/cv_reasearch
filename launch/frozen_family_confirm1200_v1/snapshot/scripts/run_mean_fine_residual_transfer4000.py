#!/usr/bin/env python3
"""One fixed full4000 residual transfer; source replay is not a new method arm.

prepare (CPU): validate all source hashes/headers/identities, inherited exact
lambda16 replay on1200, and original six-arm count provenance; never index GT.
infer (root-owned CUDA): consume shared original1200 MEAN fields; for remaining
2800 use the verified actual Part1-only paired FP32 producer and temporary FP16
q/r, then CPU replay exact original RCG16/MEAN16. Reuse every original fine16 Y.
C128 = bilinear(G64,128) + (Y128 - bilinear(F64,128)), FP32, coefficient1,
align_corners=False, no clipping/refit. CUDA bilinear1024 >.5 seals all masks.
score (CPU): only after the full seal, validate all six original I/U and report
class-summed macro scores, paired photo2000 RS0 intervals, folds and four edits.

This is a new fixed combination, not the original fine64 primary or directP(G).
No query GT in prepare/infer; no feature archive, download, deletion or sweep.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import metric, photo_groups, render, sha, summarize, unpack

ARM = "mean_fine_residual_transfer_v1"
ARMS = ("native", "rcg", "mean.control", "rcg64.control", "fine.rcg16.control", "fine.rcg64")
EDIT_NAMES = ("add_TP", "add_FP", "delete_TP", "delete_FP")
PREFIX_SHA = "e99b1c205054fe27555abfec74ecea86786f1ce2f0b6c7d262445c6b00236df3"


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, digest):
    actual = sha(path)
    if actual != digest:
        raise ValueError("Changed sealed file: " + str(path))
    return actual


def codes(config):
    for path, digest in config.get("source_code_sha256", {}).items():
        checked(path, digest)


def rows_at(path, n):
    rows = read(path)
    if not isinstance(rows, list) or len(rows) != n or len({r["key"] for r in rows}) != n:
        raise ValueError("Require every original distinct draw key; no identity deduplication")
    for row in rows:
        if "/" in row["key"] or "\\" in row["key"]:
            raise ValueError("Unsafe draw key")
        for key in ("c", "fold", "e"):
            if not isinstance(row[key], int):
                raise ValueError("Require integer original identity")
    if n == 4000 and (set(r["c"] for r in rows) != set(range(80)) or any(sum(r["fold"] == f for r in rows) != 1000 for f in range(4))):
        raise ValueError("Require original4000/80class/1000perfold cohort")
    return rows


def same_identity(a, b):
    return all(a[k] == b[k] for k in ("c", "fold", "e")) and all(Path(a[k]).name == Path(b[k]).name for k in ("support", "query"))


def original1200_key(row):
    return "public%d:%s" % (row["public_batch"], row.get("source_key", row["key"]))


def grid(value, size, label):
    if value.shape != (size, size) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Require finite FP32 grid: " + label)


def mask(value, label):
    if value.shape != (131072,) or value.dtype != np.uint8:
        raise ValueError("Require packed1024 mask: " + label)


def source_one(job):
    """All source members except truth; called on CPU only."""
    row, base, seal = job
    key = row["key"]
    fp, pp = (Path(base) / name / (key + ".npz") for name in ("fields", "predictions"))
    checked(fp, seal["fields"][key]); checked(pp, seal["predictions"][key])
    checked(row["source_field"], row["source_field_sha256"])
    checked(row["source_prediction"], row["source_prediction_sha256"])
    packet = Path(row["packet_export"])
    if not packet.is_absolute() or seal["inputs"][key]["packet_export"] != str(packet):
        raise ValueError("Require original consistent absolute packet")
    checked(packet, seal["inputs"][key]["packet_sha256"])
    with np.load(fp, allow_pickle=False) as z:
        f, y = z["rcg"].copy(), z["fine.rcg16.control"].copy()
        grid(f, 64, "F64"); grid(y, 128, "Y128")
        grid(z["rcg64.control"], 64, "coarse64"); grid(z["fine.rcg64"], 128, "fine64")
    with np.load(pp, allow_pickle=False) as z:
        if set(z.files) != set(ARMS):
            raise ValueError("Explicit six-arm keys required; no silent alias")
        masks = {a: z[a].copy() for a in ARMS}
        for a, v in masks.items(): mask(v, a)
    with np.load(row["source_field"], allow_pickle=False) as z:
        if not np.array_equal(f, z["rcg"]):
            raise ValueError("Source4000 F differs from original RCG16")
    with np.load(row["source_prediction"], allow_pickle=False) as z:
        if not np.array_equal(masks["rcg"], z["RCG"]) or not np.array_equal(masks["mean.control"], z["MEAN_CONTROL"]):
            raise ValueError("Original RCG/MEAN complete masks differ")
    with np.load(packet, allow_pickle=False) as z:
        grid(z["score"], 64, "score"); grid(z["cov"], 64, "reference coverage")
        if np.any(z["cov"] < 0) or np.any(z["cov"] > 1): raise ValueError("Invalid reference coverage")
        if not np.array_equal(masks["native"], z["native"]):
            raise ValueError("Independent stored native mask differs")
    return key, dict(field=str(fp.resolve()), field_sha256=seal["fields"][key],
        prediction=str(pp.resolve()), prediction_sha256=seal["predictions"][key],
        packet=str(packet), packet_sha256=seal["inputs"][key]["packet_sha256"],
        original_field=row["source_field"], original_field_sha256=row["source_field_sha256"],
        original_prediction=row["source_prediction"], original_prediction_sha256=row["source_prediction_sha256"])


def inherited1200(a, full):
    """Verify the already executed exact lambda16 replay, do not rerun it."""
    old = rows_at(a.fine1200 / "manifest.json", 1200)
    seal, config = read(a.fine1200 / "sealed.json"), read(a.fine1200 / "config.json")
    checked(a.fine1200 / "manifest.json", seal["manifest_sha256"])
    checked(a.fine1200 / "config.json", seal["config_sha256"])
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 1200: raise ValueError("Require original1200 complete seal")
    prepared = Path(config["prepared"]); ps, pc = read(prepared / "sealed.json"), read(prepared / "config.json")
    checked(prepared / "sealed.json", config["prepared_seal_sha256"])
    checked(prepared / "manifest.json", ps["manifest_sha256"]); checked(prepared / "config.json", ps["config_sha256"])
    checked(prepared / "audits.json", ps["audits_sha256"]); codes(pc)
    audits = read(prepared / "audits.json")
    if ps["state"] != "ALL1200_RCG64_FIELDS_AND_PROVIDER_MASKS_SEALED" or ps["n"] != 1200 or set(audits) != {r["key"] for r in old}:
        raise ValueError("Require sealed all1200 original lambda16 audits")
    mapping = {}
    for row in old:
        key, public = row["key"], original1200_key(row)
        if public in mapping or public not in full or not same_identity(row, full[public]): raise ValueError("Original1200 draw identity changed")
        audit = audits[key]["solver"]["16"]
        if any(audit[k] != 0 for k in ("cg_status", "fp32_field_mismatched_entries", "stored_rcg_mask_mismatched_pixels", "maximum_field_difference")):
            raise ValueError("Original1200 exact lambda16 parity failed")
        cp, fp, pp = prepared / "fields" / (key + ".npz"), a.fine1200 / "fields" / (key + ".npz"), a.fine1200 / "predictions" / (key + ".npz")
        checked(cp, ps["fields"][key]); checked(fp, seal["fields"][key]); checked(pp, seal["predictions"][key])
        fullfp, fullpp = a.base / "fields" / (public + ".npz"), a.base / "predictions" / (public + ".npz")
        with np.load(cp, allow_pickle=False) as p, np.load(fp, allow_pickle=False) as f, np.load(fullfp, allow_pickle=False) as ff:
            if not np.array_equal(p["rcg16"], ff["rcg"]) or not np.array_equal(f["fine.rcg16.control"], ff["fine.rcg16.control"]):
                raise ValueError("Earlier lambda16/Fine16 scalar reconstruction source changed")
        with np.load(pp, allow_pickle=False) as p, np.load(fullpp, allow_pickle=False) as ff:
            if any(not np.array_equal(p[arm], ff[arm]) for arm in ARMS): raise ValueError("Earlier1200 six-arm bit parity failed")
        mapping[public] = dict(row=row, exact_rcg16_audit=audit, prepared_field_sha256=ps["fields"][key])
    return mapping, dict(fine1200=str(a.fine1200.resolve()), fine1200_seal_sha256=sha(a.fine1200 / "sealed.json"),
        rcg1200_prepared=str(prepared), rcg1200_prepared_seal_sha256=sha(prepared / "sealed.json"),
        rcg1200_audits_sha256=ps["audits_sha256"], n=1200, inherited_exact_replay=True)


def prepare(a):
    if a.out.exists(): raise FileExistsError("Fresh owned output required")
    start = time.monotonic()
    rows = rows_at(a.base / "manifest.json", 4000); seal, source_config = read(a.base / "sealed.json"), read(a.base / "config.json")
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 4000 or tuple(seal["arms"]) != ARMS: raise ValueError("Require exact six-arm full4000 seal")
    checked(a.base / "manifest.json", seal["manifest_sha256"]); checked(a.base / "config.json", seal["config_sha256"]); codes(source_config)
    keys = {r["key"] for r in rows}
    if any(set(seal[table]) != keys for table in ("fields", "predictions", "inputs")): raise ValueError("Incomplete source seal")
    prior = rows_at(a.scored / "manifest.json", 4000); state = read(a.scored / "score_state.json"); receipt = read(a.scored / "receipt.json")
    if state["state"] != "CPU_GT_SCORE_COMPLETE" or state["completed"] != 4000 or state["n"] != 4000:
        raise ValueError("Require completed original4000 CPU score")
    checked(a.scored / "report.json", state["report_sha256"])
    if rows != prior or receipt["source_seal_sha256"] != sha(a.base / "sealed.json"): raise ValueError("Original scored4000 manifest/seal differs")
    inherited, inherited_receipt = inherited1200(a, {r["key"]: r for r in rows})
    benchmark = read(a.benchmark / "report.json"); prefix_config = read(a.benchmark / "config.json")
    evidence = read(a.benchmark / "evidence.json")
    checked(a.benchmark / "config.json", benchmark["config_sha256"])
    checked(a.benchmark / "evidence.json", benchmark["evidence_sha256"])
    if benchmark["state"] != "TWO_PAIR_PART1_PRODUCER_PARITY_AND_TIMING_COMPLETE" or benchmark["n"] != 2 or benchmark["evidence"] != evidence:
        raise ValueError("Actual Part1 benchmark is incomplete")
    if not all(pair["fixed_lambda16"]["passed"] and pair["instrumented_parity"]["passed"] and all(r["passed"] for r in pair["repeats"]) for pair in evidence):
        raise ValueError("Part1 benchmark feature/gate/native/score/lambda16 parity incomplete")
    codes(prefix_config)
    prefix_path = a.prefix_snapshot / "scripts/benchmark_part1_producer.py"
    checked(prefix_path, PREFIX_SHA)
    if prefix_config["source_code_sha256"].get(str(prefix_path.resolve())) != PREFIX_SHA:
        raise ValueError("Use the actually benchmarked immutable Part1 snapshot")
    original_sources = {}
    for run, digest in source_config["providers"].items():
        checked(Path(run) / "sealed.json", digest)
        ps = read(Path(run) / "sealed.json")
        for name in ("components.py", "rcg_readout.py", "hypothesis_source_contrast.py"):
            original_sources[str((a.source / name).resolve())] = checked(a.source / name, ps["astra_sources"][name])
    files = [Path(__file__).resolve(), REPO / "src/ics/experiment.py", REPO / "src/ics/methods/rcg.py"]
    config = dict(n=4000, candidate=ARM, arms=[*ARMS, ARM], original_primary_unchanged="fine.rcg64",
        constructor="C128=bilinear(G64,128)+(Y128-bilinear(F64,128)); FP32 align_corners=False coefficient1; no clip",
        renderer="CUDA bilinear128->1024 align_corners=False strict>0.5", parameter_updates=False,
        direct_mean_fine="separate strong control P(G); C is not P(G)", query_gt_in_inference=False,
        query_truth_opened=False, exposure="same exposed public4000 benchmark/development draws; not independent confirmation",
        source_code_sha256={**{str(p): sha(p) for p in files}, **prefix_config["source_code_sha256"], **original_sources},
        base=str(a.base.resolve()), base_seal_sha256=sha(a.base / "sealed.json"), base_config_sha256=seal["config_sha256"],
        scored=str(a.scored.resolve()), scored_counts_sha256=sha(a.scored / "counts.npz"), scored_episodes_sha256=sha(a.scored / "episodes.jsonl"),
        scored_report_sha256=state["report_sha256"], scored_state_sha256=sha(a.scored / "score_state.json"),
        mean1200=str(a.mean1200.resolve()), mean_field_key="mean.control", prefix_snapshot=str(a.prefix_snapshot.resolve()),
        benchmark=str(a.benchmark.resolve()), benchmark_report_sha256=sha(a.benchmark / "report.json"), benchmark_evidence_sha256=sha(a.benchmark / "evidence.json"),
        benchmark_config_sha256=sha(a.benchmark / "config.json"), basis=str(a.basis.resolve()), basis_sha256=prefix_config["basis_sha256"],
        host_manifest=str(a.host_manifest.resolve()), host_manifest_sha256=prefix_config["host_manifest_sha256"], host_builder_sha256=prefix_config["host_builder_sha256"],
        source=str(a.source.resolve()), inherited1200=inherited_receipt, saved_source_parameters=source_config["parameters"],
        encoder_replay="remaining2800 actual Part1 paired FP32 -> original channel normalization/gate/projection -> FP16 q/r in RAM only; existing sealed score/cov reused",
        native="independent original sealed complete native; Part1 returns no native; no new native recomputation claim",
        retained="G64 and fixedC128 scalar fields plus all4000 packed candidate masks; no q/r or fine-feature archive",
        workers=a.workers, preflight_workers=a.check_workers, inflight=a.inflight, render_batch=a.batch)
    checked(a.basis, config["basis_sha256"]); checked(a.host_manifest, config["host_manifest_sha256"])
    a.out.mkdir(parents=True); (a.out / "fields").mkdir(); (a.out / "predictions").mkdir()
    shutil.copyfile(a.base / "manifest.json", a.out / "manifest.json"); write(a.out / "config.json", config)
    receipts = {}
    with ProcessPoolExecutor(a.check_workers, mp_context=mp.get_context("spawn")) as pool:
        jobs = [(r, str(a.base), {table:{r["key"]:seal[table][r["key"]]} for table in ("fields","predictions","inputs")}) for r in rows]
        for n, (key, value) in enumerate(pool.map(source_one, jobs, chunksize=4), 1):
            receipts[key] = value
            if n % 400 == 0:
                print(json.dumps(dict(state="CPU_SOURCE_PREFLIGHT", checked=n, n=4000, seconds=time.monotonic()-start)), flush=True)
    write(a.out / "source_receipt.json", receipts); write(a.out / "inherited1200.json", inherited)
    terminal = dict(state="ALL4000_SOURCE_PREPARED", n=4000, manifest_sha256=sha(a.out / "manifest.json"), config_sha256=sha(a.out / "config.json"),
        source_receipt_sha256=sha(a.out / "source_receipt.json"), inherited1200_sha256=sha(a.out / "inherited1200.json"),
        seconds=time.monotonic()-start, query_truth_opened=False, mean1200_ready=(a.mean1200 / "sealed.json").is_file())
    write(a.out / "prepared.json", terminal); write(a.out / "state.json", terminal)
    print(json.dumps(terminal), flush=True)


def prepared(a):
    p, c = read(a.out / "prepared.json"), read(a.out / "config.json")
    if p["state"] != "ALL4000_SOURCE_PREPARED" or p["n"] != 4000: raise ValueError("Require full4000 source preflight")
    for filename, field in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256"), ("source_receipt.json", "source_receipt_sha256"), ("inherited1200.json", "inherited1200_sha256")):
        checked(a.out / filename, p[field])
    codes(c); checked(Path(c["base"]) / "sealed.json", c["base_seal_sha256"])
    return rows_at(a.out / "manifest.json", 4000), c, read(a.out / "source_receipt.json"), read(a.out / "inherited1200.json")


def mean_provider(config, inherited):
    p = Path(config["mean1200"]); seal, pc = read(p / "sealed.json"), read(p / "config.json")
    if seal["state"] != "ALL1200_ORIGINAL_MEAN_FIELDS_PARITY_SEALED" or seal["n"] != 1200 or seal["query_truth_opened"] is not False:
        raise ValueError("Require all1200 exact original continuous MEAN fields sealed")
    checked(p / "manifest.json", seal["manifest_sha256"]); checked(p / "config.json", seal["config_sha256"]); codes(pc)
    rows = rows_at(p / "manifest.json", 1200)
    if set(seal["fields"]) != {r["key"] for r in rows} or set(seal["inputs"]) != set(seal["fields"]): raise ValueError("Incomplete MEAN provider seal")
    mapping = {}
    for row in rows:
        public = original1200_key(row); key = row["key"]
        if public in mapping or public not in inherited or not same_identity(row, inherited[public]["row"]): raise ValueError("Shared1200 G identity changed")
        fp = p / "fields" / (key + ".npz"); checked(fp, seal["fields"][key])
        with np.load(fp, allow_pickle=False) as z:
            if set(z.files) != {"rcg", "mean.control"}: raise ValueError("Explicit sharedG field keys required")
            for name in z.files: grid(z[name], 64, name)
        if seal["inputs"][key]["mean_render_mismatched_pixels"] != 0: raise ValueError("Shared1200 originalMEAN parity failed")
        mapping[public] = dict(row=row, field=str(fp), field_sha256=seal["fields"][key], receipt=seal["inputs"][key])
    if set(mapping) != set(inherited): raise ValueError("Must reuse exactly all1200 sharedG fields")
    return mapping, dict(path=str(p), seal_sha256=sha(p / "sealed.json"), config_sha256=seal["config_sha256"], manifest_sha256=seal["manifest_sha256"])


def load_arrays(row, receipt):
    for path, digest in ((receipt["field"], receipt["field_sha256"]), (receipt["prediction"], receipt["prediction_sha256"]), (receipt["packet"], receipt["packet_sha256"])):
        checked(path, digest)
    with np.load(receipt["field"], allow_pickle=False) as z: f, y = z["rcg"].copy(), z["fine.rcg16.control"].copy()
    with np.load(receipt["prediction"], allow_pickle=False) as z: masks = {arm: z[arm].copy() for arm in ARMS}
    with np.load(receipt["packet"], allow_pickle=False) as z: cov, score = z["cov"].copy(), z["score"].copy()
    return f, y, masks, cov, score


def rebuild(job):
    """Original CPU solvers; fail on any field/mask discrepancy before grafting."""
    import torch
    from ics.methods import rcg
    torch.set_num_threads(1)
    q, r, cov, score, f, rcgmask, meanmask, source = job
    start = time.monotonic()
    if q.dtype != np.float16 or r.dtype != np.float16 or q.shape != (4096, 1024) or r.shape != q.shape: raise ValueError("Original exporter FP16 q/r required")
    field, ri = rcg.predict(q, r, cov, score, device="cpu")
    nf, nm = int(np.count_nonzero(field != f)), int(np.count_nonzero(render(field) != unpack(rcgmask)))
    if nf or nm: raise RuntimeError("Actual Part1 original RCG16 field/mask parity failed")
    sys.path.insert(0, source); components = importlib.import_module("components")
    g, mi = components.mean_control(q.astype(np.float32), r.astype(np.float32), cov, score, components.rcg)
    grid(g, 64, "originalMEAN16")
    mm = int(np.count_nonzero(render(g) != unpack(meanmask)))
    if mm: raise RuntimeError("Original MEAN16 complete mask parity failed")
    return g, dict(rcg16_field_mismatches=nf, rcg16_mask_mismatches=nm, mean_mask_mismatches=mm,
        rcg_solver=ri, mean_solver=mi, cpu_seconds=time.monotonic()-start)


def infer(a):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    rows, config, receipts, inherited = prepared(a)
    if (a.out / "sealed.json").exists() or any(any((a.out / name).iterdir()) for name in ("fields", "predictions")): raise FileExistsError("Infer requires untouched prepared field/mask directories")
    shared, shared_receipt = mean_provider(config, inherited)
    prefix = Path(config["prefix_snapshot"]); sys.path.insert(0, str(prefix / "scripts"))
    from benchmark_part1_producer import part1_only
    from run_frozen_subtoken4000 import make_host
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    checked(config["host_manifest"], config["host_manifest_sha256"])
    host, producer, man = make_host(a)
    checked(producer.__file__, config["host_builder_sha256"])
    runtime = dict(mean1200=shared_receipt, torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
        prepare_sha256=sha(a.out / "prepared.json"), host_builder_sha256=sha(producer.__file__),
        constructor=config["constructor"], native=config["native"], query_truth_opened=False)
    write(a.out / "inference_config.json", runtime)
    results, pending, batches, writes = [], [], [], []
    reused, encoded, encoder_seconds, render_seconds = 0, 0, 0., 0.
    start = time.monotonic()

    def save(item, c, packed):
        row, f, y, masks, g, receipt = item; key = row["key"]
        fp, pp = a.out / "fields" / (key + ".npz"), a.out / "predictions" / (key + ".npz")
        np.savez_compressed(fp, **{"mean.control": g, ARM: c}); np.savez_compressed(pp, **{ARM: packed})
        return key, sha(fp), sha(pp), receipt

    def flush(writers):
        nonlocal render_seconds
        if not batches: return
        torch.cuda.synchronize(); tick = time.monotonic()
        f = torch.from_numpy(np.stack([x[1] for x in batches])).to("cuda")[:, None]
        g = torch.from_numpy(np.stack([x[4] for x in batches])).to("cuda")[:, None]
        y = torch.from_numpy(np.stack([x[2] for x in batches])).to("cuda")[:, None]
        up = lambda v, size: F.interpolate(v, (size, size), mode="bilinear", align_corners=False)
        replay = (up(y, 1024)[:, 0] > .5).cpu().numpy()
        c = up(g, 128) + (y - up(f, 128))
        if c.dtype != torch.float32 or not bool(torch.isfinite(c).all()): raise RuntimeError("Nonfinite/nonFP32 C")
        packed = np.packbits((up(c, 1024)[:, 0] > .5).cpu().numpy().reshape(len(batches), -1), axis=1)
        c = c[:, 0].cpu().numpy()
        torch.cuda.synchronize(); render_seconds += time.monotonic()-tick
        for index, item in enumerate(batches):
            mismatch = int(np.count_nonzero(replay[index] != unpack(item[3]["fine.rcg16.control"])))
            if mismatch: raise RuntimeError("Earlier fine16 CUDA field-to-mask reconstruction failed")
            item[5]["fine16_render_mismatched_pixels"] = mismatch
            item[5]["C128_min"] = float(c[index].min()); item[5]["C128_max"] = float(c[index].max())
            writes.append(writers.submit(save, item, c[index].copy(), packed[index].copy()))
        batches.clear()
        while len(writes) >= 2*a.writers: results.append(writes.pop(0).result())

    def finish(item, writers):
        future, row, f, y, masks, receipt = item
        g, audit = future.result(); receipt["source_replay"] = audit
        batches.append((row, f, y, masks, g, receipt))
        if len(batches) >= a.batch: flush(writers)

    with torch.inference_mode(), ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool, ThreadPoolExecutor(a.writers) as writers:
        for n, row in enumerate(rows, 1):
            f, y, masks, cov, score = load_arrays(row, receipts[row["key"]])
            receipt = dict(receipts[row["key"]])
            if row["key"] in shared:
                source = shared[row["key"]]; checked(source["field"], source["field_sha256"])
                with np.load(source["field"], allow_pickle=False) as z:
                    g = z["mean.control"].copy()
                    if not np.array_equal(z["rcg"], f): raise ValueError("Shared1200 F differs from sealed4000 F")
                if source["receipt"]["packet_sha256"] != receipt["packet_sha256"]: raise ValueError("Shared1200 packet differs")
                receipt.update(mode="shared1200_G_and_inherited_exact_lambda16_replay", shared_mean_field_sha256=source["field_sha256"],
                    source_replay=dict(inherited_rcg16_audit=inherited[row["key"]]["exact_rcg16_audit"], mean_mask_mismatches=0,
                        mean_provider_receipt=source["receipt"]))
                # Independent CPU source finalizer, not a CUDA coarse-mask alias.
                if not np.array_equal(np.packbits(render(g)), masks["mean.control"]): raise ValueError("Shared1200 original MEAN mask parity failed")
                batches.append((row, f, y, masks, g, receipt)); reused += 1
                if len(batches) >= a.batch: flush(writers)
            else:
                support_path, query_path = Path(man["data_root"])/row["support"], Path(man["data_root"])/row["query"]
                reference_path = Path(man["annotation_root"])/Path(row["support"]).with_suffix(".png")
                with Image.open(support_path) as im: support = im.convert("RGB")
                with Image.open(query_path) as im: query = im.convert("RGB")
                with Image.open(reference_path) as im: reference_mask = torch.from_numpy((np.asarray(im) == row["c"]+1).copy())
                torch.cuda.synchronize(); tick = time.monotonic()
                values = part1_only(host, support, reference_mask, query)
                torch.cuda.synchronize(); seconds = time.monotonic()-tick; encoder_seconds += seconds
                if not np.array_equal(values["cov"], cov): raise RuntimeError("Source reference coverage drift")
                receipt.update(mode="remaining2800_actual_Part1_RAM_replay", encoder=dict(seconds=seconds, debiased=values["debiased"],
                    cov_maxdiff=0., support_image_sha256=sha(support_path), query_image_sha256=sha(query_path), reference_mask_sha256=sha(reference_path),
                    q_FP16_array_sha256=hashlib.sha256(values["q"].tobytes()).hexdigest(), r_FP16_array_sha256=hashlib.sha256(values["r"].tobytes()).hexdigest(),
                    q_r_archived=False, native_returned=False, score="independent unchanged sealed FoRIS continuous response"))
                job = (values["q"], values["r"], cov, score, f, masks["rcg"], masks["mean.control"], config["source"])
                pending.append((pool.submit(rebuild, job), row, f, y, masks, receipt)); encoded += 1
                del values, job, reference_mask
                if len(pending) >= a.inflight: finish(pending.pop(0), writers)
            while pending and pending[0][0].done(): finish(pending.pop(0), writers)
            if n == 1 or n % 20 == 0:
                state = dict(state="SOURCE_REPLAY_AND_FIXED_GRAFT", visited=n, n=4000, reused1200=reused, encoded2800=encoded,
                    completed=len(results), pending_CPU=len(pending), seconds=time.monotonic()-start, encoder_seconds=encoder_seconds,
                    render_seconds=render_seconds, query_truth_opened=False)
                write(a.out / "state.json", state); print(json.dumps(state), flush=True)
        for item in pending: finish(item, writers)
        flush(writers); results.extend(future.result() for future in writes)
    if reused != 1200 or encoded != 2800 or len(results) != 4000: raise RuntimeError("Full4000 accounting incomplete")
    checked(Path(config["mean1200"])/"sealed.json", shared_receipt["seal_sha256"])
    codes(config); checked(Path(config["base"])/"sealed.json", config["base_seal_sha256"])
    seal = dict(state="ALL_PREDICTIONS_SEALED", n=4000, arms=[ARM], candidate=ARM,
        manifest_sha256=sha(a.out/"manifest.json"), config_sha256=sha(a.out/"config.json"), inference_config_sha256=sha(a.out/"inference_config.json"),
        prepared_sha256=sha(a.out/"prepared.json"), predictions={k:p for k,_,p,_ in results}, fields={k:f for k,f,_,_ in results}, inputs={k:r for k,_,_,r in results},
        seconds=time.monotonic()-start, encoder_seconds=encoder_seconds, render_seconds=render_seconds,
        reused1200=reused, encoded2800=encoded, feature_archives_created=0, query_truth_opened=False, cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write(a.out/"sealed.json", seal); write(a.out/"state.json", dict(state=seal["state"], n=4000, completed=4000))
    print("FIXED_MEAN_FINE_RESIDUAL_TRANSFER4000_ALL_MASKS_SEALED", flush=True)


def score_one(job):
    row, out, seal, prior = job; key = row["key"]
    with np.load(seal["inputs"][key]["packet"], allow_pickle=False) as z: truth = unpack(z["truth"])
    with np.load(seal["inputs"][key]["prediction"], allow_pickle=False) as z: masks = {a:unpack(z[a]) for a in ARMS}
    with np.load(Path(out)/"predictions"/(key+".npz"), allow_pickle=False) as z: masks[ARM] = unpack(z[ARM])
    iu, edits, relative = {}, {}, {}
    for arm, value in masks.items():
        iu[arm] = [int((value & truth).sum()), int((value | truth).sum())]
        if arm in ARMS and iu[arm] != prior[arm]: raise ValueError("Original six-arm4000 I/U changed: "+key+" "+arm)
    for base in ("native", "rcg", "mean.control", "fine.rcg16.control", "fine.rcg64"):
        add, delete = masks[ARM] & ~masks[base], masks[base] & ~masks[ARM]
        counts = dict(add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()), delete_TP=int((delete & truth).sum()), delete_FP=int((delete & ~truth).sum()))
        if iu[ARM] != [iu[base][0]+counts["add_TP"]-counts["delete_TP"], iu[base][1]+counts["add_FP"]-counts["delete_FP"]]:
            raise ValueError("Four-edit I/U closure failed")
        relative[base] = counts
    edits = dict(key=key, c=row["c"], fold=row["fold"], batch=str(row["batch"]), **relative["native"])
    return dict(row, iu=iu, edits_vs_native=relative["native"], edits_vs_controls=relative), edits


def score(a):
    start = time.monotonic(); rows, config, _, _ = prepared(a)
    seal = read(a.out/"sealed.json")
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 4000 or seal["arms"] != [ARM]: raise ValueError("All4000 masks must seal before first GT access")
    for name, key in (("manifest.json","manifest_sha256"),("config.json","config_sha256"),("inference_config.json","inference_config_sha256"),("prepared.json","prepared_sha256")):
        checked(a.out/name, seal[key])
    if any(set(seal[k]) != {r["key"] for r in rows} for k in ("fields", "predictions", "inputs")): raise ValueError("Incomplete4000 candidate seal")
    for row in rows:
        key = row["key"]
        for folder in ("fields", "predictions"): checked(a.out/folder/(key+".npz"), seal[folder][key])
        for path, digest in ((seal["inputs"][key]["packet"],seal["inputs"][key]["packet_sha256"]),(seal["inputs"][key]["prediction"],seal["inputs"][key]["prediction_sha256"])):
            checked(path,digest)
        with np.load(a.out/"predictions"/(key+".npz"), allow_pickle=False) as z: mask(z[ARM],ARM)
        with np.load(a.out/"fields"/(key+".npz"), allow_pickle=False) as z: grid(z[ARM],128,ARM); grid(z["mean.control"],64,"G64")
    scored = Path(config["scored"]); checked(scored/"episodes.jsonl", config["scored_episodes_sha256"]); checked(scored/"counts.npz",config["scored_counts_sha256"])
    previous = [json.loads(line) for line in (scored/"episodes.jsonl").read_text().splitlines()]
    if [r["key"] for r in previous] != [r["key"] for r in rows] or any(not same_identity(r,p) for r,p in zip(rows,previous)): raise ValueError("Prior4000 exact identity/order changed")
    with np.load(scored/"counts.npz",allow_pickle=False) as z:
        old = {arm:z["iu:"+arm].copy() for arm in ARMS}
    if any(not np.array_equal(old[arm],np.asarray([r["iu"][arm] for r in previous],np.int64)) for arm in ARMS): raise ValueError("Old count source disagreement")
    episodes, corrections = [], []
    with ProcessPoolExecutor(a.check_workers,mp_context=mp.get_context("spawn")) as pool:
        jobs = [(r,str(a.out),{"inputs":{r["key"]:seal["inputs"][r["key"]]}},p["iu"]) for r,p in zip(rows,previous)]
        for n,(record,correction) in enumerate(pool.map(score_one,jobs,chunksize=4),1):
            episodes.append(record); corrections.append(correction)
            if n%400==0: print(json.dumps(dict(state="CPU_COMPLETE_MASK_SCORE",completed=n,n=4000,seconds=time.monotonic()-start)),flush=True)
    arrays = {**old, ARM:np.asarray([r["iu"][ARM] for r in episodes],np.int64)}
    # summarize's base selection excludes historical non-control fine64; explicit
    # statistics-only alias enables its paired CI and is removed from the report.
    alias = "historical_fine64.control"; arrays[alias] = arrays["fine.rcg64"]
    report, draws = summarize(rows,arrays,{ARM:corrections})
    for table in (report["scores"],report["contrasts"]): table.pop(alias,None)
    for contrasts in report["contrasts"].values():
        if alias in contrasts: contrasts["fine.rcg64"] = contrasts.pop(alias)
    for table in ("folds","batchs"):
        for group in report[table].values():
            group["scores"].pop(alias,None); group["gain_vs_native"].pop(alias,None)
            group["candidate_gain_vs_controls"] = {arm:group["scores"][ARM]-group["scores"][arm] for arm in ARMS}
    arrays.pop(alias)
    report.update(candidate=ARM, historical_primary_unchanged="fine.rcg64", config=config, parameter_updates=False,
        source_seal_sha256=sha(a.out/"sealed.json"), all4000_validated_before_first_GT=True,
        original_six_arm_IU_parity=dict(n=4000,mismatches=0), raw_origin_edits="full4000 raw origin unavailable; not replaced by native",
        direct_mean_fine="C128 is fixed residual transfer, not P(G); direct fullMeanFine1200 is a separate same-information control",
        corrections_vs_controls={base:{name:int(sum(r["edits_vs_controls"][base][name] for r in episodes)) for name in EDIT_NAMES} for base in episodes[0]["edits_vs_controls"]},
        runtime=dict(prepare_seconds=read(a.out/"prepared.json")["seconds"],source_replay_and_render_seconds=seal["seconds"],
            paired_encoder_seconds=seal["encoder_seconds"],fixed_render_seconds=seal["render_seconds"],cpu_score_seconds=time.monotonic()-start),
        interpretation="Intervals crossing zero remain unresolved. Same exposed4000 benchmark is not fresh confirmation. This newly combined candidate does not change the old primary fine64 report.")
    write(a.out/"report.json",report); np.save(a.out/"bootstrap_photo_draws.npy",draws)
    np.savez_compressed(a.out/"counts.npz",**{"iu:"+arm:v for arm,v in arrays.items()},**{"edits_vs_"+base:np.asarray([[r["edits_vs_controls"][base][name] for name in EDIT_NAMES] for r in episodes],np.int64) for base in episodes[0]["edits_vs_controls"]})
    (a.out/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in episodes))
    lines=["# Fixed MEAN/fine residual transfer on all4000 draws","","New candidate; original fine64 primary is unchanged. Coefficient1, no clip/refit; all source/mask parities required before GT.","","| Arm | Class-summed macro mIoU |","|---|---:|"]
    lines += [f"| {arm} | {report['scores'][arm]:.6f} |" for arm in (*ARMS,ARM)]
    lines += ["","Candidate minus fixed controls (paired2000 RandomState0 connected-photo95% CI):",""]
    for arm in ARMS:
        result=report["contrasts"][ARM][arm]; lines.append(f"- {arm}: {result['gain']:+.6f} [{result['ci95'][0]:+.6f}, {result['ci95'][1]:+.6f}] pp; up/down/tie {result['up']}/{result['down']}/{result['tie']}.")
    lines += ["","All4000 original six-arm I/U match exactly. Natural repeated draws remain retained. Four-edit counts are relative to actual complete native and named controls; full4000 raw-origin edits are unavailable.","C is not directP(G). A CI crossing zero is unresolved; this exposed benchmark is not independent confirmation."]
    (a.out/"report.md").write_text("\n".join(lines)+"\n")
    write(a.out/"score_state.json",dict(state="CPU_GT_SCORE_COMPLETE",n=4000,completed=4000,candidate=ARM,report_sha256=sha(a.out/"report.json")))
    print("\n".join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage",choices=("prepare","infer","score")); p.add_argument("--out",type=Path,default=Path("outputs/mean_fine_residual_transfer4000_v1"))
    p.add_argument("--base",type=Path,default=Path("outputs/frozen_subtoken4000_v1")); p.add_argument("--scored",type=Path,default=Path("outputs/frozen_subtoken4000_scored_v1"))
    p.add_argument("--fine1200",type=Path,default=Path("outputs/frozen_subtoken1200_v1"));p.add_argument("--mean1200",type=Path,default=Path("outputs/fine_mean1200_prepared_v1"))
    p.add_argument("--benchmark",type=Path,default=Path("outputs/part1_producer_two_pair_v1"));p.add_argument("--prefix-snapshot",type=Path,default=Path("launch/part1_producer_two_pair_v1/snapshot"))
    p.add_argument("--source",type=Path,default=Path("external/astra_emd/external_mean_delete600_modules"));p.add_argument("--host-manifest",type=Path,default=Path("outputs/claude_official/batch0.json"))
    p.add_argument("--host-root",type=Path,default=Path("/root/autodl-tmp/demo9_extent"));p.add_argument("--demo4-root",default="/root/autodl-tmp/demo4")
    p.add_argument("--basis",type=Path,default=Path("/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt"))
    p.add_argument("--workers",type=int,default=6);p.add_argument("--check-workers",type=int,default=4);p.add_argument("--writers",type=int,default=2);p.add_argument("--inflight",type=int,default=6);p.add_argument("--batch",type=int,default=16)
    a=p.parse_args()
    if any(v<1 for v in (a.workers,a.check_workers,a.writers,a.inflight,a.batch)):p.error("Positive execution sizes required")
    try: {"prepare":prepare,"infer":infer,"score":score}[a.stage](a)
    except Exception as error:
        if a.out.exists(): write(a.out/"state.json",dict(state="FAILED",stage=a.stage,error=repr(error),query_truth_opened=(a.stage=="score")))
        raise


if __name__ == "__main__":
    main()
