#!/usr/bin/env python3
"""Frozen DEV1200 subtoken readout with the existing lambda64 strong control.

Stages are separate: CPU prepare seals lambda16 replay/lambda64 fields and
provider masks; CUDA infer streams four query shifts and seals all1200 fine
masks; CPU score alone opens GT and checks prior native/RCG/MEAN I/U.
No refitting, parameter grid, CRF, new image/reference, or fine-feature archive.

python scripts/run_frozen_subtoken1200.py prepare \
    --manifest outputs/confirm1200_conditional_v1/inputs/manifest.json \
    --prepared outputs/subtoken1200_rcg64_prepared_v1
python scripts/run_frozen_subtoken1200.py infer \
    --prepared outputs/subtoken1200_rcg64_prepared_v1 --out outputs/frozen_subtoken1200_v1
python scripts/run_frozen_subtoken1200.py score \
    --out outputs/frozen_subtoken1200_v1 \
    --prior-episodes outputs/composed_full1200_dev_v1/episodes.jsonl
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import metric, photo_groups, render, sha, summarize, unpack

FINE16 = "fine.rcg16.control"
FINE64 = "fine.rcg64"
ARMS = ("native", "rcg", "mean.control", "rcg64.control", FINE16, FINE64)


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def manifest(path):
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else data["episodes"]
    if len(rows) != 1200 or len({r["key"] for r in rows}) != 1200:
        raise ValueError("Require all1200 original draw keys; never deduplicate identities or subset")
    for row in rows:
        for name in ("fold", "c", "e"):
            row[name] = int(row[name])
        if not all(Path(row[k]).is_absolute() for k in ("feature_export", "packet_export", "recheck_run")):
            raise ValueError("Provider manifest must contain absolute cache/packet/run paths")
    if any(sum(r["fold"] == f for r in rows) != 300 for f in range(4)):
        raise ValueError("Require original300/fold cohort")
    return rows


def parameters(path):
    report = json.loads(Path(path).read_text())
    selection = report["selection"]["feat.nested"]
    if report["n"] != 600 or set(selection) != {"0", "1", "2", "3"}:
        raise ValueError("Require recorded full600 fine-feature fold selection")
    expected = {0: (1.25, .07), 1: (1.25, .15), 2: (1.25, .15), 3: (1.25, .07)}
    result = {}
    for fold, arm in selection.items():
        match = re.fullmatch(r"feat\[s([0-9.]+),t([0-9.]+)\]", arm)
        if match is None or tuple(map(float, match.groups())) != expected[int(fold)]:
            raise ValueError("Recorded600 frozen parameters changed")
        result[fold] = dict(arm=arm, sigma=expected[int(fold)][0], tau=expected[int(fold)][1])
    return result


def verify_code(config):
    for path, digest in config["source_code_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Changed immutable stage implementation: " + path)


def prepare_one(job):
    """CPU-only; share similarities/graph, then use two genuinely different matrices."""
    import torch
    import torch.nn.functional as F
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.methods.rcg import minmax, rank

    torch.set_num_threads(1)
    row, destination, expected_field, expected_mask = job
    begin = time.monotonic()
    key, source_key = row["key"], row.get("source_key", row["key"])
    run = Path(row["recheck_run"])
    field_path, mask_path = run / "fields" / (source_key + ".npz"), run / "predictions" / (source_key + ".npz")
    inputs = dict(feature_sha256=sha(row["feature_export"]), packet_sha256=sha(row["packet_export"]),
                  source_field_sha256=sha(field_path), source_predictions_sha256=sha(mask_path))
    if inputs["source_field_sha256"] != expected_field or inputs["source_predictions_sha256"] != expected_mask:
        raise ValueError("Changed provider field/masks: " + key)
    cached = torch.load(row["feature_export"], map_location="cpu", weights_only=True)
    q, r = (cached[k].float() for k in ("q", "r"))
    if any(cached[k].dtype != torch.float16 for k in ("q", "r")) or q.shape != (4096, 1024) or r.shape != q.shape:
        raise ValueError("Require the retained4096x1024 FP16 q/r cache")
    with np.load(row["packet_export"], allow_pickle=False) as z:
        # Supplied-reference coverage, source score, and prediction only. No truth.
        cov, score, native = z["cov"].copy(), z["score"].copy(), z["native"].copy()
    with np.load(field_path, allow_pickle=False) as z:
        sealed_field = z["rcg"].copy()
    with np.load(mask_path, allow_pickle=False) as z:
        original_rcg, mean = z["RCG"].copy(), z["MEAN_CONTROL"].copy()
    if cov.shape != (64, 64) or score.shape != cov.shape or sealed_field.shape != cov.shape:
        raise ValueError("Invalid aligned source grids")
    if sealed_field.dtype != np.float32 or not all(np.isfinite(v).all() for v in (cov, score, sealed_field)):
        raise ValueError("Invalid sealed/source float fields")
    if cov.min() < 0 or cov.max() > 1 or not bool(torch.isfinite(q).all() and torch.isfinite(r).all()):
        raise ValueError("Invalid q/r or reference coverage")
    if bool((q.norm(dim=1) == 0).any() or (r.norm(dim=1) == 0).any()):
        raise ValueError("Zero-norm cached token")
    for packed in (native, original_rcg, mean):
        unpack(packed)
    q, r = F.normalize(q, dim=1), F.normalize(r, dim=1)
    pure = np.flatnonzero(cov.ravel() >= .9)
    if not len(pure):
        pure = np.flatnonzero(cov.ravel() == cov.max())
    sim = q @ r.T
    dq, dr = sim.topk(10, dim=1).values.mean(1), sim.topk(10, dim=0).values.mean(0)
    guide = ((2 * sim[:, pure] - dr[pure][None, :]).max(1).values - dq).numpy()
    del sim, dq, dr, r
    s = minmax(score).ravel()
    y = (s + .5 * (rank(guide) - rank(s))).astype(np.float64)
    sim = q @ q.T
    sim.fill_diagonal_(-2)
    values, ids = sim.topk(20, dim=1)
    del sim, q
    distance = (1 - values).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), ids.numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel()
    w = w / max(float(degree.mean()), 1e-8)
    degree = np.asarray(w.sum(1)).ravel()
    laplacian = sparse.diags(degree) - w
    confidence = .1 + np.abs(2 * s - 1)
    confidence = (confidence / confidence.mean()).astype(np.float64)
    anchor = sparse.diags(confidence)
    rhs = confidence * y
    fields, audits = {}, {}
    for lam in (16, 64):
        matrix = anchor + lam * laplacian
        iterations = [0]
        def callback(_):
            iterations[0] += 1
        solved, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
        residual = float(np.linalg.norm(matrix @ solved - rhs))
        tolerance = max(1e-9, 1e-7 * float(np.linalg.norm(rhs)))
        if status != 0 or not np.isfinite(solved).all() or residual > 1.01 * tolerance:
            raise RuntimeError("Fixed lambda%d CG convergence failed: %s" % (lam, key))
        field = solved.reshape(64, 64).astype(np.float32)
        fields["rcg%d" % lam] = field
        audits[str(lam)] = dict(graph_lambda=lam, cg_status=int(status), cg_iterations=iterations[0],
                               absolute_residual=residual, residual_tolerance=tolerance,
                               weighted_mean_error=float(abs(np.dot(confidence, solved - y))))
        if lam == 16:
            replay = render(field)
            mismatch_field = int(np.count_nonzero(field != sealed_field))
            mismatch_mask = int(np.count_nonzero(replay != unpack(original_rcg)))
            audits["16"].update(fp32_field_mismatched_entries=mismatch_field,
                                  stored_rcg_mask_mismatched_pixels=mismatch_mask,
                                  maximum_field_difference=float(np.abs(field - sealed_field).max()))
            if mismatch_field or mismatch_mask:
                raise RuntimeError("Exact lambda16 field/mask replay failed before lambda64: " + key)
    masks = {"native": native, "rcg": original_rcg, "mean.control": mean,
             "rcg64.control": np.packbits(render(fields["rcg64"]))}
    out = Path(destination)
    fp, pp = out / "fields" / (key + ".npz"), out / "predictions" / (key + ".npz")
    np.savez_compressed(fp, **fields)
    np.savez_compressed(pp, **masks)
    inputs["debiased"] = bool(cached["debiased"])
    return key, sha(fp), sha(pp), dict(inputs=inputs, solver=audits,
                                      source_pure_tokens=int(len(pure)), seconds=time.monotonic() - begin)


def prepare(a):
    rows = manifest(a.manifest)
    frozen = parameters(a.parameter_report)
    if a.prepared.exists():
        raise FileExistsError("Use a fresh prepared directory; never remove retained inputs")
    providers, jobs = {}, []
    for row in rows:
        run = Path(row["recheck_run"])
        if str(run) not in providers:
            seal = json.loads((run / "sealed.json").read_text())
            if seal["state"] != "ALL_PREDICTIONS_SEALED" or sha(run / "manifest.json") != seal["manifest_sha256"]:
                raise ValueError("Provider manifest is not sealed")
            data = json.loads((run / "manifest.json").read_text())
            data = data if isinstance(data, list) else data["episodes"]
            providers[str(run)] = dict(seal=seal, rows={r["key"]: r for r in data},
                                       seal_sha256=sha(run / "sealed.json"))
        source = providers[str(run)]
        key = row.get("source_key", row["key"])
        original = source["rows"][key]
        if int(original["c"]) != row["c"] or int(original["fold"]) != row["fold"] or any(
                Path(original[k]).name != Path(row[k]).name for k in ("support", "query")):
            raise ValueError("Provider photo/class/fold identity changed: " + row["key"])
        jobs.append((row, str(a.prepared), source["seal"]["fields"][key], source["seal"]["predictions"][key]))
    a.prepared.mkdir(parents=True)
    (a.prepared / "fields").mkdir()
    (a.prepared / "predictions").mkdir()
    write(a.prepared / "manifest.json", rows)
    code = [Path(__file__).resolve(), REPO / "src/ics/experiment.py", REPO / "src/ics/methods/rcg.py"]
    config = dict(stage="CPU_ONLY_FIXED_RCG64_PREPARATION", n=1200,
                  exposure="all1200 previously exposed DEV draws; no independent confirmation or SOTA claim",
                  source_manifest=str(a.manifest.resolve()), source_manifest_sha256=sha(a.manifest),
                  parameter_report=str(a.parameter_report.resolve()), parameter_report_sha256=sha(a.parameter_report),
                  parameters=frozen, source_code_sha256={str(p): sha(p) for p in code},
                  providers={name: dict(seal_sha256=v["seal_sha256"]) for name, v in providers.items()},
                  graph_lambdas=[16, 64], alpha=.5, cross_k=10, query_k=20, purity=.9,
                  confidence_floor=.1, cg_rtol=1e-7, cg_atol=1e-9, cg_maxiter=300,
                  lambda16_parity="exact FP32 source field and complete stored mask before lambda64 solve",
                  workers=a.workers, query_gt_in_inference=False, encoder_forwards=0, images_opened=0,
                  legacy_provider_inputs="not assumed; fresh packet/cache hashes captured and prior I/U checked at score")
    write(a.prepared / "config.json", config)
    begin, records = time.monotonic(), {}
    write(a.prepared / "state.json", dict(state="RUNNING_CPU_PREPARATION", pid=os.getpid(), completed=0, n=1200))
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(prepare_one, job) for job in jobs]
        for future in as_completed(futures):
            key, field_hash, prediction_hash, audit = future.result()
            records[key] = dict(fields_sha256=field_hash, predictions_sha256=prediction_hash, **audit)
            if len(records) % 25 == 0:
                state = dict(state="RUNNING_CPU_PREPARATION", pid=os.getpid(), completed=len(records), n=1200,
                             seconds=time.monotonic() - begin)
                write(a.prepared / "state.json", state)
                print(json.dumps(state), flush=True)
    for name, info in providers.items():
        if sha(Path(name) / "sealed.json") != info["seal_sha256"]:
            raise ValueError("Provider seal changed during CPU preparation")
    write(a.prepared / "audits.json", records)
    seal = dict(state="ALL1200_RCG64_FIELDS_AND_PROVIDER_MASKS_SEALED", n=1200,
                manifest_sha256=sha(a.prepared / "manifest.json"), config_sha256=sha(a.prepared / "config.json"),
                audits_sha256=sha(a.prepared / "audits.json"),
                fields={k: v["fields_sha256"] for k, v in records.items()},
                predictions={k: v["predictions_sha256"] for k, v in records.items()},
                inputs={k: v["inputs"] for k, v in records.items()},
                seconds=time.monotonic() - begin, query_truth_opened=False)
    write(a.prepared / "sealed.json", seal)
    write(a.prepared / "state.json", dict(state=seal["state"], pid=os.getpid(), completed=1200, n=1200,
                                         seconds=seal["seconds"]))
    print("SUBTOKEN1200_CPU_PREPARATION_SEALED", flush=True)


def prepared_metadata(directory):
    seal = json.loads((directory / "sealed.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if seal["state"] not in ("ALL1200_RCG64_FIELDS_AND_PROVIDER_MASKS_SEALED", "ALL_PREDICTIONS_SEALED") or seal["n"] != 1200:
        raise ValueError("Require all1200 CPU-prepared fields and provider masks sealed")
    for file, key in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256")):
        if sha(directory / file) != seal[key]:
            raise ValueError("Changed prepared " + file)
    if "audits_sha256" in seal and sha(directory / "audits.json") != seal["audits_sha256"]:
        raise ValueError("Changed prepared solver audits")
    if "source_code_sha256" in config:
        verify_code(config)
    rows = manifest(directory / "manifest.json")
    keys = {r["key"] for r in rows}
    if any(set(seal[k]) != keys for k in ("fields", "predictions", "inputs")):
        raise ValueError("Prepared seal must include every original draw key")
    return rows, seal, config


def infer(a):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, prepared_seal, prepared_config = prepared_metadata(a.prepared)
    frozen = parameters(a.parameter_report)
    if "parameters" in prepared_config and frozen != prepared_config["parameters"]:
        raise ValueError("Prepared/fine-readout frozen parameters differ")
    if a.out.exists():
        raise FileExistsError("Use a fresh inference directory")
    ancestor = a.out.parent.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < (1 << 30):
        raise OSError("Need1GiB headroom for packed masks and scalar fields; no fine feature cache is created")
    a.out.mkdir(parents=True)
    (a.out / "predictions").mkdir()
    (a.out / "fields").mkdir()
    write(a.out / "manifest.json", rows)
    sys.path.insert(0, str(a.host_root))
    sys.path.insert(0, str(a.host_root / "scripts"))
    import extent_experiment as host_module

    host_manifest = json.loads(a.host_manifest.read_text())
    code = [Path(__file__).resolve(), REPO / "src/ics/experiment.py"]
    config = dict(n=1200, arms=list(ARMS), primary=FINE64,
                  exposure="all1200 existing exposed DEV draws; no independent confirmation or SOTA claim",
                  parameters=frozen, fitting="recorded600 fold choices frozen unchanged; no1200 fitting or grid",
                  prepared=str(a.prepared.resolve()), prepared_seal_sha256=sha(a.prepared / "sealed.json"),
                  prepared_config_sha256=sha(a.prepared / "config.json"), prepared_metadata=prepared_config,
                  source_code_sha256={str(p): sha(p) for p in code},
                  host_manifest=str(a.host_manifest.resolve()), host_manifest_sha256=sha(a.host_manifest),
                  host_builder=str(Path(host_module.__file__).resolve()), host_builder_sha256=sha(host_module.__file__),
                  host_root=str(a.host_root.resolve()), demo4_root=a.demo4_root,
                  feature_space="source run_subtoken normalized/debiased four shifted grids, interleaved FP32; no FP16 re-quantization",
                  feature_lifecycle="one128x128x1024 fine tensor in bounded GPU RAM; both fixed readouts consume before release; no full fine cache",
                  retained="all1200 packed complete masks, scalar fields, source receipts; existing features/assets untouched",
                  shifts=[[-4, -4], [-4, 4], [4, -4], [4, 4]], padding="reflect4", window=2,
                  renderer="CUDA bilinear128->1024 align_corners=False then>0.5; noCRF",
                  baseline_renderer="CPU fixed RCG64 bilinear64->1024, same locked RCG renderer",
                  query_gt_in_inference=False, extra_encoder_forwards=4800, reference_images_per_episode=1,
                  dataset="COCO-20i", resolution=1024, seed=0, tf32=False,
                  readers=a.readers, writers=a.writers, prefetch=a.prefetch, chunk=a.chunk,
                  torch_version=str(torch.__version__), device_name=torch.cuda.get_device_name(0))
    write(a.out / "config.json", config)
    write(a.out / "state.json", dict(state="RUNNING_GPU_INFERENCE", pid=os.getpid(), completed=0, n=1200,
                                    query_truth_opened=False))
    begin = time.monotonic()
    device = torch.device("cuda")
    fi = torch.arange(128, device=device)
    base = ((fi * 8 + 4 - 8).float() / 16).round().long()
    off = torch.arange(-2, 3, device=device)
    ti = base[:, None] + off[None]
    distance = ((ti * 16 + 8) - (fi * 8 + 4)[:, None]).float() / 16
    ok = (ti >= 0) & (ti < 64)
    ti = ti.clamp(0, 63)
    ny = ti[:, None, :, None].expand(128, 128, 5, 5)
    nx = ti[None, :, None, :].expand(128, 128, 5, 5)
    nb = (ny * 64 + nx).reshape(16384, 25)
    d2 = (distance[:, None, :, None].square() + distance[None, :, None, :].square()).reshape(16384, 25)
    valid = (ok[:, None, :, None] & ok[None, :, None, :]).reshape(16384, 25)
    data = Path(host_manifest["data_root"])

    def load(row):
        key = row["key"]
        receipt = dict(prepared_seal["inputs"][key])
        if sha(row["feature_export"]) != receipt["feature_sha256"]:
            raise ValueError("Changed retained feature cache: " + key)
        fp, pp = a.prepared / "fields" / (key + ".npz"), a.prepared / "predictions" / (key + ".npz")
        if sha(fp) != prepared_seal["fields"][key] or sha(pp) != prepared_seal["predictions"][key]:
            raise ValueError("Changed prepared scalar fields or masks: " + key)
        cached = torch.load(row["feature_export"], map_location="cpu", weights_only=True)
        q = cached["q"]
        if q.shape != (4096, 1024) or q.dtype != torch.float16 or not bool(torch.isfinite(q).all()):
            raise ValueError("Require finite original FP16 cached query tokens")
        debiased = bool(cached["debiased"])
        if "debiased" in receipt and debiased != receipt["debiased"]:
            raise ValueError("Cached feature-space flag changed")
        with np.load(fp, allow_pickle=False) as z:
            if prepared_seal["state"] == "ALL1200_RCG64_FIELDS_AND_PROVIDER_MASKS_SEALED":
                coarse = {FINE16: z["rcg16"].copy(), FINE64: z["rcg64"].copy()}
            else:
                coarse = {FINE16: z["rcg"].copy(), FINE64: z["rcg64.control"].copy()}
        if any(v.shape != (64, 64) or v.dtype != np.float32 or not np.isfinite(v).all() for v in coarse.values()):
            raise ValueError("Invalid prepared scalar fields")
        with np.load(pp, allow_pickle=False) as z:
            masks = {name: z[name].copy() for name in ARMS[:4]}
        query_path = data / row["query"]
        with Image.open(query_path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        receipt.update(query_image_sha256=sha(query_path), query_image=str(query_path.resolve()),
                       prepared_fields_sha256=prepared_seal["fields"][key],
                       prepared_predictions_sha256=prepared_seal["predictions"][key])
        return q, debiased, rgb, coarse, masks, receipt

    def save(row, fields, masks, receipt):
        key = row["key"]
        fp, pp = a.out / "fields" / (key + ".npz"), a.out / "predictions" / (key + ".npz")
        np.savez_compressed(fp, **fields)
        np.savez_compressed(pp, **masks)
        return key, sha(fp), sha(pp), receipt

    results, writer_jobs = [], []
    gpu_seconds = 0.
    with torch.inference_mode(), ThreadPoolExecutor(a.readers) as readers, ThreadPoolExecutor(a.writers) as writers:
        host = host_module.build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), host_manifest, "cuda")
        pending = {i: readers.submit(load, rows[i]) for i in range(min(a.prefetch, len(rows)))}
        for n, row in enumerate(rows):
            cached_q, debiased, rgb, coarse, masks, receipt = pending.pop(n).result()
            later = n + a.prefetch
            if later < len(rows):
                pending[later] = readers.submit(load, rows[later])
            tgt = host._transform(Image.fromarray(rgb)).to(device)
            if tgt.shape != (3, 1024, 1024):
                raise ValueError("Host must transform RGB to3x1024x1024")
            q = F.normalize(cached_q.to(device).float(), dim=1)
            pad = F.pad(tgt[None], (4, 4, 4, 4), mode="reflect")[0]
            torch.cuda.synchronize()
            tick = time.monotonic()
            fine = torch.zeros(128, 128, 1024, device=device, dtype=torch.float32)
            for ay, ax in itertools.product((0, 1), repeat=2):
                sy, sx = (-4, 4)[ay], (-4, 4)[ax]
                shifted = pad[:, 4 + sy:4 + sy + 1024, 4 + sx:4 + sx + 1024]
                f = F.normalize(host._extract_features(shifted[None, None]), p=2, dim=2)
                if debiased:
                    f = host._debias_features(f)
                if f.shape != (1, 1, 1024, 64, 64):
                    raise ValueError("Host last-layer feature shape changed")
                fine[ay::2, ax::2] = F.normalize(f[0, 0], dim=0).permute(1, 2, 0)
            fine = fine.reshape(16384, 1024)
            params = frozen[str(row["fold"])]
            sigma, tau = params["sigma"], params["tau"]
            scalar = {name: torch.from_numpy(value).to(device).float().flatten() for name, value in coarse.items()}
            pieces = {name: [] for name in scalar}
            for cell in range(0, 16384, a.chunk):
                stop = min(cell + a.chunk, 16384)
                neighbors = nb[cell:stop]
                cos = torch.einsum("pc,pkc->pk", fine[cell:stop], q[neighbors])
                weights = torch.exp(-d2[cell:stop] / (2 * sigma ** 2)) * torch.exp((cos - 1) / tau) * valid[cell:stop]
                denominator = weights.sum(1).clamp_min(1e-12)
                for name, field in scalar.items():
                    pieces[name].append((weights * field[neighbors]).sum(1) / denominator)
            fields_gpu = {name: torch.cat(parts).reshape(128, 128) for name, parts in pieces.items()}
            fields = {}
            for name, field in fields_gpu.items():
                mask = F.interpolate(field[None, None], (1024, 1024), mode="bilinear", align_corners=False)[0, 0] > .5
                masks[name] = np.packbits(mask.cpu().numpy())
                fields[name] = field.cpu().numpy()
            torch.cuda.synchronize()
            gpu_seconds += time.monotonic() - tick
            if any(not np.isfinite(value).all() for value in fields.values()):
                raise RuntimeError("Nonfinite fine-readout scalar field")
            writer_jobs.append(writers.submit(save, row, fields, masks, receipt))
            if len(writer_jobs) >= 2 * a.writers:
                results.append(writer_jobs.pop(0).result())
            if (n + 1) % 10 == 0 or n == 0:
                state = dict(state="RUNNING_GPU_INFERENCE", pid=os.getpid(), completed=n + 1, n=1200,
                             seconds=time.monotonic() - begin, gpu_seconds=gpu_seconds, query_truth_opened=False)
                write(a.out / "state.json", state)
                print(json.dumps(state), flush=True)
            # No feature archive is created: both fixed arms consumed this tensor.
            del cached_q, rgb, tgt, q, pad, fine, f, scalar, pieces, fields_gpu, field, mask, cos, weights
        results.extend(future.result() for future in writer_jobs)
    if sha(a.prepared / "sealed.json") != config["prepared_seal_sha256"]:
        raise ValueError("Prepared seal changed during GPU inference")
    seal = dict(state="ALL_PREDICTIONS_SEALED", n=1200, arms=list(ARMS),
                manifest_sha256=sha(a.out / "manifest.json"), config_sha256=sha(a.out / "config.json"),
                fields={key: f for key, f, _, _ in results}, predictions={key: p for key, _, p, _ in results},
                inputs={key: receipt for key, _, _, receipt in results},
                seconds=time.monotonic() - begin, gpu_seconds=gpu_seconds,
                cuda_peak_bytes=torch.cuda.max_memory_allocated(), query_truth_opened=False,
                fine_feature_archives_created=0, complete_encoder_forwards=4800)
    write(a.out / "sealed.json", seal)
    write(a.out / "state.json", dict(state=seal["state"], completed=1200, n=1200, seconds=seal["seconds"]))
    print("FROZEN_SUBTOKEN1200_GPU_INFERENCE_SEALED", flush=True)


def score(a):
    seal = json.loads((a.out / "sealed.json").read_text())
    config = json.loads((a.out / "config.json").read_text())
    rows = manifest(a.out / "manifest.json")
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 1200:
        raise ValueError("Require all1200 complete masks sealed before opening GT")
    for file, key in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256")):
        if sha(a.out / file) != seal[key]:
            raise ValueError("Changed sealed " + file)
    verify_code(config)
    prepared = Path(config["prepared"])
    _, prepared_seal, _ = prepared_metadata(prepared)
    if sha(prepared / "sealed.json") != config["prepared_seal_sha256"]:
        raise ValueError("Changed CPU-prepared seal")
    if not a.prior_episodes.is_file():
        raise FileNotFoundError("Prior baseline I/U source must exist: " + str(a.prior_episodes))
    prior_records = [json.loads(line) for line in a.prior_episodes.read_text().splitlines() if line.strip()]
    prior = {r["key"]: r for r in prior_records}
    if len(prior_records) != 1200 or len(prior) != 1200 or set(prior) != {r["key"] for r in rows}:
        raise ValueError("Prior baseline I/U must contain exactly the unchanged1200 draw keys")
    # Preflight EVERY complete output and packet before accessing the first GT member.
    for row in rows:
        key = row["key"]
        if any(sha(a.out / folder / (key + ".npz")) != seal[folder][key] for folder in ("predictions", "fields")):
            raise ValueError("Changed sealed complete output: " + key)
        if sha(row["packet_export"]) != seal["inputs"][key]["packet_sha256"]:
            raise ValueError("Changed GT packet: " + key)
        old = prior[key]
        if int(old["c"]) != row["c"] or int(old["fold"]) != row["fold"] or any(
                Path(old[name]).name != Path(row[name]).name for name in ("support", "query")):
            raise ValueError("Prior baseline photo/class identity mismatch: " + key)
    begin = time.monotonic()
    write(a.out / "score_state.json", dict(state="RUNNING_CPU_GT_SCORE", completed=0, n=1200))
    arrays = {arm: [] for arm in ARMS}
    details = []
    supplemental = {arm: dict(add_TP=0, add_FP=0, delete_TP=0, delete_FP=0) for arm in ARMS}
    for n, row in enumerate(rows):
        key = row["key"]
        with np.load(row["packet_export"], allow_pickle=False) as z:
            truth = unpack(z["truth"])
        with np.load(a.out / "predictions" / (key + ".npz"), allow_pickle=False) as z:
            masks = {arm: unpack(z[arm]) for arm in ARMS}
        episode = dict(row, iu={})
        for arm, mask in masks.items():
            iu = [int((mask & truth).sum()), int((mask | truth).sum())]
            if arm in ("native", "rcg", "mean.control") and iu != prior[key]["iu"][arm]:
                raise ValueError("Prior complete baseline I/U parity failed: " + key + " " + arm)
            arrays[arm].append(iu)
            episode["iu"][arm] = iu
            add, delete = mask & ~masks["rcg"], masks["rcg"] & ~mask
            values = dict(add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                          delete_TP=int((delete & truth).sum()), delete_FP=int((delete & ~truth).sum()))
            for name, value in values.items():
                supplemental[arm][name] += value
        details.append(episode)
        if (n + 1) % 100 == 0:
            print(json.dumps(dict(scored=n + 1, total=1200, seconds=time.monotonic() - begin)), flush=True)
    arrays = {arm: np.asarray(values, dtype=np.int64) for arm, values in arrays.items()}
    report, draws = summarize(rows, arrays, {})
    for name in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
        report.pop(name, None)
    report.update(config=config, primary=FINE64, exposure=config["exposure"], parameter_updates=False,
                  source_prediction_seal_sha256=sha(a.out / "sealed.json"),
                  prior_episode_source=str(a.prior_episodes.resolve()), prior_episode_sha256=sha(a.prior_episodes),
                  baseline_IU_parity=dict(episodes=1200, arms=["native", "rcg", "mean.control"], mismatches=0),
                  raw_DINO_origin=dict(status="not exported in provider1200; raw-origin add/delete unavailable", edits=None),
                  edits_vs_RCG_supplemental=supplemental,
                  INSID3_1200="not available in this fixed comparison; not substituted",
                  runtime=dict(cpu_prepare_seconds=prepared_seal.get("seconds"), gpu_infer_seconds=seal["seconds"],
                               gpu_compute_seconds=seal["gpu_seconds"], cuda_peak_bytes=seal["cuda_peak_bytes"],
                               cpu_score_seconds=time.monotonic() - begin))
    for fold in report["folds"].values():
        scores = fold["scores"]
        fold["primary_gain_vs_controls"] = {arm: scores[FINE64] - scores[arm] for arm in ARMS if arm != FINE64}
    np.savez_compressed(a.out / "counts.npz", **{"iu:" + arm: values for arm, values in arrays.items()})
    np.save(a.out / "bootstrap_photo_draws.npy", draws)
    (a.out / "episodes.jsonl").write_text("".join(json.dumps(d) + "\n" for d in details))
    write(a.out / "report.json", report)
    lines = ["# Frozen subtoken readout: complete reused DEV1200", "", "| arm | class mIoU at1024 |", "|---|---:|"]
    lines += ["| %s | %.6f |" % (arm, report["scores"][arm]) for arm in ARMS]
    lines += ["", "Fine lambda64 versus mandatory controls (paired95% photo-connected intervals):", ""]
    for arm in ARMS[:-1]:
        c = report["contrasts"][FINE64][arm]
        lines.append("- %s: %+.6f [%+.6f, %+.6f] percentage points." % (arm, c["gain"], *c["ci95"]))
    lines += ["", "All1200 prior native/RCG/MEAN I/U match exactly. Frozen600 readout parameters are unchanged.",
              "Raw DINO origin masks are absent: raw-origin edit accounting is unavailable. RCG edits are supplemental.",
              "This is reused development evidence, not fresh confirmation or a SOTA result."]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    write(a.out / "score_state.json", dict(state="CPU_GT_SCORE_COMPLETE", completed=1200, n=1200,
                                          report_sha256=sha(a.out / "report.json")))
    print("\n".join(lines), flush=True)
    print("FROZEN_SUBTOKEN1200_SCORE_DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "infer", "score"))
    p.add_argument("--manifest", type=Path, default=Path("outputs/confirm1200_conditional_v1/inputs/manifest.json"))
    p.add_argument("--parameter-report", type=Path, default=Path("outputs/claude_subtoken_fresh600/report.json"))
    p.add_argument("--prepared", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--prior-episodes", type=Path)
    p.add_argument("--host-manifest", type=Path, default=Path("outputs/claude_official/batch0.json"))
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent"))
    p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--readers", type=int, default=6)
    p.add_argument("--writers", type=int, default=2)
    p.add_argument("--prefetch", type=int, default=12)
    p.add_argument("--chunk", type=int, default=512)
    a = p.parse_args()
    if any(v < 1 for v in (a.workers, a.readers, a.writers, a.prefetch, a.chunk)):
        p.error("execution sizes must be positive")
    if a.stage in ("prepare", "infer") and a.prepared is None:
        p.error("prepare/infer require --prepared")
    if a.stage in ("infer", "score") and a.out is None:
        p.error("infer/score require --out")
    if a.stage == "score" and a.prior_episodes is None:
        p.error("score requires the existing --prior-episodes baseline I/U source")
    if a.stage == "prepare" and a.prepared.exists():
        p.error("prepare requires a fresh directory")
    if a.stage == "infer" and a.out.exists():
        p.error("infer requires a fresh directory")
    try:
        if a.stage == "prepare":
            prepare(a)
        else:
            infer(a) if a.stage == "infer" else score(a)
    except Exception as error:
        destination = a.prepared if a.stage == "prepare" else a.out
        if destination is not None and (destination / "config.json").exists():
            write(destination / ("score_state.json" if a.stage == "score" else "state.json"),
                  dict(state="FAILED", stage=a.stage, pid=os.getpid(), error=repr(error)))
        raise


if __name__ == "__main__":
    main()
