#!/usr/bin/env python3
"""Held full4000 streaming adapter; root launches only after the1200 decision.

Smoke replays two cached1200 pairs without query GT. Infer preserves every
draw of public blocks0..6, reuses all sealed1200 outputs, and streams only
the remaining2800 paired q/r + four shifted query grids in bounded RAM.
The ACTUAL exporter run_foris path owns the source gate/projection order.
No feature archive, cleanup, download, new parameter, or GT access.

python scripts/run_frozen_subtoken4000.py smoke --out outputs/subtoken4000_smoke_v1
python scripts/run_frozen_subtoken4000.py infer --out outputs/frozen_subtoken4000_v1

Requires sibling run_frozen_subtoken1200.py, src/ics, and the existing host
and CRF PYTHONPATH. Scoring is delegated separately; standard complete-mask
seal/manifest/receipts are written for independent all4000 CPU scoring.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import render, sha, unpack
from run_frozen_subtoken1200 import ARMS, FINE16, FINE64, parameters, verify_code, write

BASIS_SHA = "9b9b20755a796cbda11bb7220d246ee540106e40cb024e249a5f63f884b6a116"


def source_rows(a):
    rows, providers = [], {}
    for block in range(7):
        run = a.public / ("run%d" % block)
        seal = json.loads((run / "sealed.json").read_text())
        if seal["state"] != "ALL_PREDICTIONS_SEALED" or sha(run / "manifest.json") != seal["manifest_sha256"]:
            raise ValueError("Require all seven sealed public providers")
        data = json.loads((run / "manifest.json").read_text())
        data = data if isinstance(data, list) else data["episodes"]
        providers[str(run.resolve())] = sha(run / "sealed.json")
        for original in data:
            key = original["key"]
            rows.append(dict(original, key="public%d:%s" % (block, key), source_key=key,
                public_batch=block, batch="official%d" % block,
                recheck_run=str(run.resolve()),
                packet_export=str((a.public / ("root%d" % block) / "results/extent_v1/run/packets" / (key + ".npz")).resolve()),
                source_field=str((run / "fields" / (key + ".npz")).resolve()),
                source_prediction=str((run / "predictions" / (key + ".npz")).resolve()),
                source_field_sha256=seal["fields"][key], source_prediction_sha256=seal["predictions"][key]))
    if len(rows) != 4000 or len({r["key"] for r in rows}) != 4000 or any(sum(r["fold"] == f for r in rows) != 1000 for f in range(4)):
        raise ValueError("Require all4000 original draws,1000/fold; no identity deduplication")
    return rows, providers


def read_source(row):
    if sha(row["source_field"]) != row["source_field_sha256"] or sha(row["source_prediction"]) != row["source_prediction_sha256"]:
        raise ValueError("Changed source field/masks: " + row["key"])
    with np.load(row["packet_export"], allow_pickle=False) as z:
        cov, score, native = z["cov"].copy(), z["score"].copy(), z["native"].copy()
    with np.load(row["source_field"], allow_pickle=False) as z:
        rcg = z["rcg"].copy()
    with np.load(row["source_prediction"], allow_pickle=False) as z:
        masks = {"native": native, "rcg": z["RCG"].copy(), "mean.control": z["MEAN_CONTROL"].copy()}
    receipt = dict(packet_export=row["packet_export"], packet_path=row["packet_export"],
                   packet_sha256=sha(row["packet_export"]), source_field_sha256=row["source_field_sha256"],
                   source_prediction_sha256=row["source_prediction_sha256"])
    return cov, score, rcg, masks, receipt


def solve64(job):
    """Exact CPU locked equations on FP16-quantized tokens; two distinct lambdas."""
    import torch
    import torch.nn.functional as F
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.methods.rcg import minmax, rank
    torch.set_num_threads(1)
    q16, r16, cov, score, sealed_field, sealed_mask = job
    q, r = F.normalize(torch.from_numpy(q16).float(), dim=1), F.normalize(torch.from_numpy(r16).float(), dim=1)
    pure = np.flatnonzero(cov.ravel() >= .9)
    if not len(pure):
        pure = np.flatnonzero(cov.ravel() == cov.max())
    sim = q @ r.T
    dq, dr = sim.topk(10, dim=1).values.mean(1), sim.topk(10, dim=0).values.mean(0)
    guide = ((2 * sim[:, pure] - dr[pure][None]).max(1).values - dq).numpy()
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
    lap = sparse.diags(np.asarray(w.sum(1)).ravel()) - w
    conf = .1 + np.abs(2 * s - 1)
    conf = (conf / conf.mean()).astype(np.float64)
    rhs, anchor = conf * y, sparse.diags(conf)
    fields, audit = {}, {}
    for lam in (16, 64):
        matrix = anchor + lam * lap
        solved, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300)
        residual = float(np.linalg.norm(matrix @ solved - rhs))
        tolerance = max(1e-9, 1e-7 * float(np.linalg.norm(rhs)))
        if status != 0 or not np.isfinite(solved).all() or residual > 1.01 * tolerance:
            raise RuntimeError("Fixed lambda%d CG failed" % lam)
        field = solved.reshape(64, 64).astype(np.float32)
        name = "rcg" if lam == 16 else "rcg64.control"
        fields[name] = field
        audit[str(lam)] = dict(graph_lambda=lam, cg_status=int(status), absolute_residual=residual)
        if lam == 16:
            nf, nm = int(np.count_nonzero(field != sealed_field)), int(np.count_nonzero(render(field) != unpack(sealed_mask)))
            audit["16"].update(field_mismatches=nf, mask_mismatches=nm,
                               field_max_difference=float(np.abs(field - sealed_field).max()))
            if nf or nm:
                raise RuntimeError("Exact lambda16 parity failed; lambda64 is not used")
    return fields, np.packbits(render(fields["rcg64.control"])), audit


def make_host(a):
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha(a.basis) != BASIS_SHA:
        raise ValueError("Source FP32 positional basis changed")
    basis = torch.load(a.basis, map_location="cpu", weights_only=True)["basis"]
    if basis.shape != (1024, 500) or basis.dtype != torch.float32:
        raise ValueError("Require unchanged native FP32 U500; noQR/rescaling")
    sys.path.insert(0, str(a.host_root)); sys.path.insert(0, str(a.host_root / "scripts"))
    import extent_experiment as producer
    man = json.loads(a.host_manifest.read_text())
    host = producer.build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in host.parameters()):
        raise ValueError("Source encoder is FP32; do not silently change dtype/autocast")
    return host, producer, man


def encode_pair(host, producer, man, row, source):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    sp, qp = data / row["support"], data / row["query"]
    mp = ann / Path(row["support"]).with_suffix(".png")
    with Image.open(sp) as im:
        support = im.convert("RGB")
    with Image.open(qp) as im:
        query = im.convert("RGB")
    with Image.open(mp) as im:
        ref_mask = torch.from_numpy((np.asarray(im) == row["c"] + 1).copy())
    # Actual exporter set_reference/set_target/segment with unchanged source taps.
    native, got, transformed_mask, target = producer.run_foris(host, support, ref_mask, query)
    raw, deb = F.normalize(got["raw"][0], dim=1), got["deb"][0]
    if raw.dtype != torch.float32 or deb.dtype != torch.float32:
        raise ValueError("Paired source features must remain FP32")
    q16, r16 = deb[-1].flatten(1).T.half().cpu().numpy(), deb[0].flatten(1).T.half().cpu().numpy()
    if q16.shape != (4096, 1024) or r16.shape != q16.shape:
        raise ValueError("Unexpected paired q/r shape")
    debiased = bool((raw - deb).abs().max() > 1e-4)
    cov, score, _, masks, _ = source
    native_mismatch = int(np.unpackbits(np.packbits(native.cpu().numpy()) ^ masks["native"]).sum())
    regenerated_cov = F.interpolate(transformed_mask[None, None].float(), (64, 64), mode="area")[0, 0].cpu().numpy()
    score_difference = float(np.abs(got["score"].float().cpu().numpy() - score).max())
    cov_difference = float(np.abs(regenerated_cov - cov).max())
    audit = dict(native_mismatched_pixels=native_mismatch, score_max_difference=score_difference,
                 cov_max_difference=cov_difference, debiased=debiased,
                 producer="actual exporter paired reference/query FP32 -> source debias -> FP16 q/r",
                 query_image_sha256=sha(qp), support_image_sha256=sha(sp), reference_mask_sha256=sha(mp))
    if native_mismatch or score_difference or cov_difference:
        raise RuntimeError("Source FoRIS producer drift; no4000 parity claim: " + json.dumps(audit))
    return q16, r16, debiased, target, audit


def fine_grid(host, target, debiased):
    import torch
    import torch.nn.functional as F
    pad = F.pad(target[None], (4, 4, 4, 4), mode="reflect")[0]
    fine = torch.zeros(128, 128, 1024, device="cuda", dtype=torch.float32)
    for ay, ax in itertools.product((0, 1), repeat=2):
        sy, sx = (-4, 4)[ay], (-4, 4)[ax]
        shifted = pad[:, 4 + sy:4 + sy + 1024, 4 + sx:4 + sx + 1024]
        f = F.normalize(host._extract_features(shifted[None, None]), p=2, dim=2)
        if debiased:
            f = host._debias_features(f)
        fine[ay::2, ax::2] = F.normalize(f[0, 0], dim=0).permute(1, 2, 0)
    return fine.reshape(16384, 1024).cpu().numpy()


def readout(fine_array, q16, coarse, params, chunk):
    import torch
    import torch.nn.functional as F
    fine = torch.from_numpy(fine_array).to("cuda")
    q = F.normalize(torch.from_numpy(q16).to("cuda").float(), dim=1)
    fi = torch.arange(128, device="cuda"); off = torch.arange(-2, 3, device="cuda")
    base = ((fi * 8 + 4 - 8).float() / 16).round().long()
    ti = base[:, None] + off[None]
    distance = ((ti * 16 + 8) - (fi * 8 + 4)[:, None]).float() / 16
    ok = (ti >= 0) & (ti < 64); ti = ti.clamp(0, 63)
    ny = ti[:, None, :, None].expand(128, 128, 5, 5); nx = ti[None, :, None, :].expand(128, 128, 5, 5)
    nb = (ny * 64 + nx).reshape(16384, 25)
    d2 = (distance[:, None, :, None].square() + distance[None, :, None, :].square()).reshape(16384, 25)
    valid = (ok[:, None, :, None] & ok[None, :, None, :]).reshape(16384, 25)
    scalar = {FINE16: torch.from_numpy(coarse["rcg"]).to("cuda").flatten(),
              FINE64: torch.from_numpy(coarse["rcg64.control"]).to("cuda").flatten()}
    pieces = {name: [] for name in scalar}
    for start in range(0, 16384, chunk):
        end = min(start + chunk, 16384); ids = nb[start:end]
        cos = torch.einsum("pc,pkc->pk", fine[start:end], q[ids])
        weights = torch.exp(-d2[start:end] / (2 * params["sigma"] ** 2)) * torch.exp((cos - 1) / params["tau"]) * valid[start:end]
        denominator = weights.sum(1).clamp_min(1e-12)
        for name, field in scalar.items():
            pieces[name].append((weights * field[ids]).sum(1) / denominator)
    fields, masks = dict(coarse), {}
    for name, parts in pieces.items():
        field = torch.cat(parts).reshape(128, 128)
        masks[name] = np.packbits((F.interpolate(field[None, None], (1024, 1024), mode="bilinear", align_corners=False)[0, 0] > .5).cpu().numpy())
        fields[name] = field.cpu().numpy()
    return fields, masks


def reuse_metadata(a, frozen):
    config = json.loads((a.reuse1200 / "config.json").read_text())
    seal = json.loads((a.reuse1200 / "sealed.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 1200 or config["parameters"] != frozen:
        raise ValueError("Require exact fixed six-arm sealed1200 reuse")
    if sha(a.reuse1200 / "manifest.json") != seal["manifest_sha256"] or sha(a.reuse1200 / "config.json") != seal["config_sha256"]:
        raise ValueError("Changed1200 metadata")
    verify_code(config)
    if config["host_manifest_sha256"] != sha(a.host_manifest) or config["host_builder_sha256"] != sha(config["host_builder"]):
        raise ValueError("1200 host producer/config changed")
    rows = json.loads((a.reuse1200 / "manifest.json").read_text())
    mapping = {"public%d:%s" % (r["public_batch"], r.get("source_key", r["key"])): r for r in rows}
    if len(mapping) != 1200:
        raise ValueError("1200 source draw mapping is not one-to-one")
    prepared = Path(config["prepared"])
    ps = json.loads((prepared / "sealed.json").read_text())
    if sha(prepared / "sealed.json") != config["prepared_seal_sha256"]:
        raise ValueError("Changed reused lambda64 preparation")
    return mapping, seal, prepared, ps, config


def save(out, row, fields, masks, receipt):
    fp, pp = out / "fields" / (row["key"] + ".npz"), out / "predictions" / (row["key"] + ".npz")
    np.savez_compressed(fp, **fields); np.savez_compressed(pp, **masks)
    return row["key"], sha(fp), sha(pp), receipt


def smoke(a):
    import torch
    frozen = parameters(a.parameter_report)
    mapping, _, _, _, _ = reuse_metadata(a, frozen)
    public, _ = source_rows(a); lookup = {r["key"]: r for r in public}
    selected = list(mapping.values())
    selected = [selected[0], next(r for r in selected if r["fold"] != selected[0]["fold"])]
    a.out.mkdir(parents=True, exist_ok=False)
    host, producer, man = make_host(a)
    audits = []
    with torch.inference_mode():
        for original in selected:
            key = "public%d:%s" % (original["public_batch"], original.get("source_key", original["key"]))
            row = lookup[key]; source = read_source(row)
            q16, r16, flag, _, audit = encode_pair(host, producer, man, row, source)
            cached = torch.load(original["feature_export"], map_location="cpu", weights_only=True)
            audit.update(key=key, cached_q_mismatches=int(np.count_nonzero(q16 != cached["q"].numpy())),
                         cached_r_mismatches=int(np.count_nonzero(r16 != cached["r"].numpy())),
                         cached_gate_match=flag == bool(cached["debiased"]), cache_sha256=sha(original["feature_export"]))
            audits.append(audit); write(a.out / "parity.json", dict(state="SMOKE_RUNNING", audits=audits, query_truth_opened=False))
            if audit["cached_q_mismatches"] or audit["cached_r_mismatches"] or not audit["cached_gate_match"]:
                raise RuntimeError("Paired exporter FP16/gate parity failed; full4000 held")
            _, _, cg_audit = solve64((q16, r16, source[0], source[1], source[2], source[3]["rcg"]))
            audit["fixed_rcg"] = cg_audit
    write(a.out / "parity.json", dict(state="TWO_PAIR_SOURCE_PARITY_PASSED", audits=audits,
                                      source_code_sha256=sha(Path(__file__)), basis_sha256=BASIS_SHA, query_truth_opened=False))
    print("SUBTOKEN4000_TWO_PAIR_PARITY_PASSED", flush=True)


def infer(a):
    import torch
    frozen = parameters(a.parameter_report)
    reuse, reuse_seal, prepared, prepared_seal, reuse_config = reuse_metadata(a, frozen)
    rows, providers = source_rows(a)
    if len(set(reuse) & {r["key"] for r in rows}) != 1200:
        raise ValueError("All1200 protected draws must be reused")
    if a.smoke_receipt is None:
        raise ValueError("Full4000 requires the root-launched --smoke-receipt")
    smoke_record = json.loads(a.smoke_receipt.read_text())
    if smoke_record["state"] != "TWO_PAIR_SOURCE_PARITY_PASSED" or smoke_record["source_code_sha256"] != sha(Path(__file__)):
        raise ValueError("Source two-pair smoke must pass on this unchanged adapter")
    a.out.mkdir(parents=True, exist_ok=False); (a.out / "fields").mkdir(); (a.out / "predictions").mkdir()
    write(a.out / "manifest.json", rows)
    host, producer, man = make_host(a)
    if sha(producer.__file__) != reuse_config["host_builder_sha256"]:
        raise ValueError("Fresh/exporter host differs from reused1200")
    code = [Path(__file__).resolve(), REPO / "scripts/run_frozen_subtoken1200.py", REPO / "src/ics/experiment.py",
            REPO / "src/ics/methods/rcg.py", REPO / "src/ics/data.py", REPO / "src/ics/native_basis.py"]
    config = dict(n=4000, arms=list(ARMS), primary=FINE64, parameters=frozen,
                  exposure="all4000 original public draws including repeats; benchmark reuse, not fresh confirmation or SOTA",
                  source_code_sha256={str(p): sha(p) for p in code}, providers=providers,
                  reuse1200=str(a.reuse1200.resolve()), reuse1200_seal_sha256=sha(a.reuse1200 / "sealed.json"),
                  smoke_receipt=str(a.smoke_receipt.resolve()), smoke_receipt_sha256=sha(a.smoke_receipt),
                  basis=str(a.basis.resolve()), basis_sha256=BASIS_SHA, producer_sha256=sha(producer.__file__),
                  host_manifest_sha256=sha(a.host_manifest), parameter_report_sha256=sha(a.parameter_report),
                  lifecycle="reuse1200 complete outputs; remaining2800 temporary q/r FP16 + fine FP32 in bounded RAM; no archives or deletion",
                  dtype="actual source paired FP32, no autocast; exporter q/r half->CPU float32->unit; fine FP32 unchanged",
                  projection="actual source gate and explicit (I-U@U.T)@X order preserved by run_foris; noQR/lowrank substitute",
                  query_gt_in_inference=False, workers=a.workers, writers=a.writers, inflight=a.inflight, chunk=a.chunk)
    write(a.out / "config.json", config)
    begin = time.monotonic(); results = []; pending = []; writes = []; reused = 0; encoded = 0
    def finished(item):
        future, row, q16, fine, masks, receipt = item
        coarse, mask64, audit = future.result()
        fields, new_masks = readout(fine, q16, coarse, frozen[str(row["fold"])], a.chunk)
        masks.update(new_masks); masks["rcg64.control"] = mask64
        receipt["solver"] = audit
        return fields, masks, receipt
    with torch.inference_mode(), ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool, ThreadPoolExecutor(a.writers) as writers:
        for n, row in enumerate(rows):
            source = read_source(row); masks, receipt = source[3], source[4]
            if row["key"] in reuse:
                old = reuse[row["key"]]; key = old["key"]
                if old["c"] != row["c"] or any(Path(old[k]).name != Path(row[k]).name for k in ("support", "query")):
                    raise ValueError("Reused draw identity mismatch")
                fp, pp = a.reuse1200 / "fields" / (key + ".npz"), a.reuse1200 / "predictions" / (key + ".npz")
                cp = prepared / "fields" / (key + ".npz")
                if sha(fp) != reuse_seal["fields"][key] or sha(pp) != reuse_seal["predictions"][key] or sha(cp) != prepared_seal["fields"][key]:
                    raise ValueError("Changed sealed1200 output")
                if receipt["packet_sha256"] != reuse_seal["inputs"][key]["packet_sha256"]:
                    raise ValueError("Reused1200 packet identity differs")
                with np.load(pp, allow_pickle=False) as z:
                    copied = {name: z[name].copy() for name in ARMS}
                if any(not np.array_equal(copied[name], masks[name]) for name in masks):
                    raise ValueError("Reused1200/provider complete baseline differs")
                with np.load(fp, allow_pickle=False) as z:
                    fields = {FINE16: z[FINE16].copy(), FINE64: z[FINE64].copy()}
                with np.load(cp, allow_pickle=False) as z:
                    if prepared_seal["state"] == "ALL1200_RCG64_FIELDS_AND_PROVIDER_MASKS_SEALED":
                        fields.update(rcg=z["rcg16"].copy(), **{"rcg64.control": z["rcg64"].copy()})
                    else:
                        fields.update(rcg=z["rcg"].copy(), **{"rcg64.control": z["rcg64.control"].copy()})
                receipt.update(mode="sealed1200_reuse_no_encoder", reuse_prediction_sha256=sha(pp), reuse_fine_fields_sha256=sha(fp), reuse_coarse_fields_sha256=sha(cp))
                writes.append(writers.submit(save, a.out, row, fields, copied, receipt)); reused += 1
            else:
                q16, r16, flag, target, audit = encode_pair(host, producer, man, row, source)
                future = pool.submit(solve64, (q16, r16, source[0], source[1], source[2], masks["rcg"]))
                fine = fine_grid(host, target, flag)
                receipt.update(mode="temporary_RAM_paired_exporter", encoder=audit)
                pending.append((future, row, q16, fine, masks, receipt)); encoded += 1
                del r16, target
                if len(pending) >= a.inflight:
                    item = pending.pop(0); fields, out_masks, inputs = finished(item)
                    writes.append(writers.submit(save, a.out, item[1], fields, out_masks, inputs))
            while pending and pending[0][0].done():
                item = pending.pop(0); fields, out_masks, inputs = finished(item)
                writes.append(writers.submit(save, a.out, item[1], fields, out_masks, inputs))
            while len(writes) >= 2 * a.writers:
                results.append(writes.pop(0).result())
            if (n + 1) % 20 == 0:
                state = dict(state="STREAMING_INFERENCE", n=4000, visited=n + 1, reused=reused, encoded=encoded,
                             completed=len(results), pending_fine_RAM=len(pending), seconds=time.monotonic() - begin, query_truth_opened=False)
                write(a.out / "state.json", state); print(json.dumps(state), flush=True)
        for item in pending:
            fields, masks, receipt = finished(item)
            writes.append(writers.submit(save, a.out, item[1], fields, masks, receipt))
        results.extend(future.result() for future in writes)
    if reused != 1200 or encoded != 2800 or len(results) != 4000:
        raise RuntimeError("Incomplete4000 streaming/reuse accounting")
    for path, digest in providers.items():
        if sha(Path(path) / "sealed.json") != digest:
            raise ValueError("Provider seal changed during stream")
    seal = dict(state="ALL_PREDICTIONS_SEALED", n=4000, arms=list(ARMS), manifest_sha256=sha(a.out / "manifest.json"),
                config_sha256=sha(a.out / "config.json"), predictions={k: p for k, _, p, _ in results},
                fields={k: f for k, f, _, _ in results}, inputs={k: r for k, _, _, r in results},
                seconds=time.monotonic() - begin, reused1200=reused, encoded2800=encoded,
                fine_feature_archives_created=0, query_truth_opened=False, cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write(a.out / "sealed.json", seal); write(a.out / "state.json", dict(state=seal["state"], n=4000, completed=4000))
    print("FROZEN_SUBTOKEN4000_ALL_MASKS_SEALED", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("smoke", "infer")); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--public", type=Path, default=Path("outputs/claude_official"))
    p.add_argument("--reuse1200", type=Path, default=Path("outputs/frozen_subtoken1200_v1"))
    p.add_argument("--smoke-receipt", type=Path)
    p.add_argument("--parameter-report", type=Path, default=Path("outputs/claude_subtoken_fresh600/report.json"))
    p.add_argument("--host-manifest", type=Path, default=Path("outputs/claude_official/batch0.json"))
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent"))
    p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    p.add_argument("--basis", type=Path, default=Path("/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt"))
    p.add_argument("--workers", type=int, default=6); p.add_argument("--writers", type=int, default=2)
    p.add_argument("--inflight", type=int, default=6); p.add_argument("--chunk", type=int, default=512)
    a = p.parse_args()
    if a.out.exists():
        p.error("Use a fresh output directory; no cleanup/overwrite")
    if any(v < 1 for v in (a.workers, a.writers, a.inflight, a.chunk)):
        p.error("Execution sizes must be positive")
    try:
        smoke(a) if a.stage == "smoke" else infer(a)
    except Exception as error:
        if a.out.exists():
            write(a.out / "state.json", dict(state="FAILED", stage=a.stage, error=repr(error), query_truth_opened=False))
        raise


if __name__ == "__main__":
    main()
