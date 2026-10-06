#!/usr/bin/env python3
"""Fixed reference-BG OAS maximum-SNR readout, complete exposed DEV241, CPU only."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), "/root/demo4_cache/env"]
import numpy as np
from ics.experiment import metric, photo_groups, sha, summarize, unpack

ARMS = ["raw.oas", "raw.identity.control", "projected.oas", "projected.identity.control"]
CONFIG = dict(
    method="reference-background maximum-SNR conditioner", spaces=["raw_FP32_q24_r24", "cached_projected_FP16"],
    token_normalization="FP32 per-token unit normalization, then FP64 reference statistics/solve",
    reference_FG="cov>=0.9; if empty all maximally covered tokens", reference_BG="cov==0; if empty all minimally covered tokens",
    equation="d=muF-muB; w=solve(OAS(BG),d); score=(unit(q)-(muF+muB)/2)@w",
    identity_control="Sigma=I, same unnormalized d, midpoint and zero threshold",
    covariance="installed sklearn OAS, centered BG samples, store_precision=False; no ridge/grid",
    undefined_covariance_fallback="Sigma=I only if fewer than2 BG tokens or covariance trace==0; record",
    renderer="FP32 signed64 field; bilinear1024 align_corners=False >0; no minmax/CRF",
    cross_space_caveat="raw FP32 versus conditional projected FP16 also changes quantization; not pure projection ablation",
    common_origin="unchanged sealed model.raw_nn nearest-reference-token label transfer",
    query_GT_in_inference=False, encoder_forwards=0, downloads=0, variants_searched=0,
    exposure="all original241 reused DEV identities; not independent confirmation",
    diagnostic="after seal only: high score=FG deep FN/FP token AUROC on fixed fine16 errors, same241 cohort",
)


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def render_signed(field):
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(field, np.float32))[None, None],
                         (1024, 1024), mode="bilinear", align_corners=False)[0, 0].numpy() > 0


def unit(x):
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(x).float()
    if t.shape != (4096, 1024) or not bool(torch.isfinite(t).all()) or bool((t.norm(dim=1) == 0).any()):
        raise ValueError("Require finite nonzero 4096x1024 tokens")
    return F.normalize(t, dim=1).numpy().astype(np.float64)


def one(job):
    import torch
    from scipy.linalg import cho_factor, cho_solve
    from sklearn.covariance import OAS
    j, out, baseline, expected_baseline = job
    torch.set_num_threads(1)
    started = time.monotonic()
    for role in ("raw", "projected", "packet"):
        if sha(j["paths"][role]) != j["sha256"][role]:
            raise ValueError("Changed inference input: " + role)
    key = j["old"]["key"]
    bp = Path(baseline) / "predictions" / (key + ".npz")
    if sha(bp) != expected_baseline:
        raise ValueError("Changed complete baseline masks")
    with np.load(j["paths"]["raw"], allow_pickle=False) as z:
        raw_q, raw_r = z["q24"].copy(), z["r24"].copy()
    if raw_q.dtype != np.float32 or raw_r.dtype != np.float32:
        raise ValueError("Raw source must stay FP32")
    cached = torch.load(j["paths"]["projected"], map_location="cpu", weights_only=True)
    if any(cached[k].dtype != torch.float16 for k in ("q", "r")):
        raise ValueError("Projected source must stay FP16")
    with np.load(j["paths"]["packet"], allow_pickle=False) as z:
        cov = z["cov"].copy()  # Only reference coverage; no query GT, score or native indexed.
    if cov.shape != (64, 64) or not np.isfinite(cov).all() or cov.min() < 0 or cov.max() > 1:
        raise ValueError("Invalid reference coverage")
    fi, bi = np.flatnonzero(cov.ravel() >= .9), np.flatnonzero(cov.ravel() == 0)
    fg_fallback, bg_fallback = not len(fi), not len(bi)
    if fg_fallback:
        fi = np.flatnonzero(cov.ravel() == cov.max())
    if bg_fallback:
        bi = np.flatnonzero(cov.ravel() == cov.min())
    fields, conditioners, info = {}, {}, {}
    for space, q0, r0 in (("raw", raw_q, raw_r), ("projected", cached["q"].numpy(), cached["r"].numpy())):
        q, r = unit(q0), unit(r0)
        muF, muB = r[fi].mean(0), r[bi].mean(0)
        d, midpoint = muF - muB, (muF + muB) / 2
        fallback = "fewer_than2_BG" if len(bi) < 2 else None
        shrinkage, covariance_trace, residual = None, None, None
        if fallback is None:
            estimator = OAS(store_precision=False, assume_centered=False).fit(r[bi])
            sigma, shrinkage = estimator.covariance_, float(estimator.shrinkage_)
            covariance_trace = float(np.trace(sigma))
            if covariance_trace == 0:
                fallback = "zero_covariance_trace"
            else:
                w = cho_solve(cho_factor(sigma, lower=True, check_finite=True), d)
                residual = float(np.linalg.norm(sigma @ w - d) / max(np.linalg.norm(d), 1e-12))
                if residual > 1e-8 or not np.isfinite(w).all():
                    raise RuntimeError("Invalid OAS linear solve")
        if fallback is not None:
            w = d.copy()
        for suffix, direction in (("oas", w), ("identity.control", d)):
            f = ((q - midpoint) @ direction).reshape(64, 64).astype(np.float32)
            if not np.isfinite(f).all():
                raise ValueError("Nonfinite signed field")
            fields[space + "." + suffix] = f
        conditioners.update({space + ".muF": muF, space + ".muB": muB, space + ".w": w})
        info[space] = dict(nFG=len(fi), nBG=len(bi), FG_fallback=fg_fallback, BG_fallback=bg_fallback,
                           covariance_fallback=fallback, OAS_shrinkage=shrinkage, covariance_trace=covariance_trace,
                           solve_relative_residual=residual, mean_difference_norm=float(np.linalg.norm(d)))
    masks = {k: np.packbits(render_signed(f)) for k, f in fields.items()}
    with np.load(bp, allow_pickle=False) as z:
        masks.update({k: z[k].copy() for k in z.files})
    out = Path(out)
    fp, pp, wp = out / "fields" / (key + ".npz"), out / "predictions" / (key + ".npz"), out / "conditioners" / (key + ".npz")
    np.savez_compressed(fp, **fields)
    np.savez_compressed(pp, **masks)
    np.savez_compressed(wp, **conditioners)
    return key, dict(fields_sha256=sha(fp), predictions_sha256=sha(pp), conditioners_sha256=sha(wp), spaces=info,
                     inputs={k: dict(path=j["paths"][k], sha256=j["sha256"][k]) for k in ("raw", "projected", "packet")},
                     baseline_prediction_sha256=expected_baseline, cached_debiased=bool(cached["debiased"]),
                     seconds=time.monotonic() - started)


def infer(a):
    import sklearn
    from sklearn.covariance import OAS
    from diagnose_region_prototypes_dev241 import source_jobs
    jobs, source_hashes = source_jobs(a.raw.resolve(), a.fine.resolve())
    baseline_seal = json.loads((a.baseline / "sealed.json").read_text())
    if baseline_seal["state"] != "ALL_PREDICTIONS_SEALED" or sha(a.baseline / "manifest.json") != baseline_seal["manifest_sha256"]:
        raise ValueError("Unsealed baseline")
    rows = json.loads((a.baseline / "manifest.json").read_text())
    if rows != [j["old"] for j in jobs] or len(rows) != 241:
        raise ValueError("Original DEV241 order/identity mismatch")
    source_hashes[str(a.baseline / "sealed.json")] = sha(a.baseline / "sealed.json")
    if a.smoke:
        jobs, rows = jobs[:2], rows[:2]
    a.out.mkdir(parents=True, exist_ok=False)
    for folder in ("fields", "predictions", "conditioners"):
        (a.out / folder).mkdir()
    write(a.out / "manifest.json", rows)
    write(a.out / "providers.json", jobs)
    config = {**CONFIG, "n": len(rows), "smoke": a.smoke, "workers": a.workers,
              "source_metadata_sha256": source_hashes, "OAS_version": sklearn.__version__,
              "OAS_implementation": inspect.getsourcefile(OAS), "OAS_implementation_sha256": sha(inspect.getsourcefile(OAS)),
              "source_code_sha256": sha(__file__), "baseline": str(a.baseline.resolve())}
    write(a.out / "config.json", config)
    begin, records = time.monotonic(), {}
    work = [(j, str(a.out), str(a.baseline), baseline_seal["predictions"][j["old"]["key"]]) for j in jobs]
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(one, j) for j in work]):
            key, record = future.result()
            records[key] = record
            if len(records) % 10 == 0 or len(records) == len(rows):
                state = dict(state="INFERENCE_WITHOUT_QUERY_GT", completed=len(records), n=len(rows), seconds=time.monotonic()-begin)
                write(a.out / "state.json", state)
                print(json.dumps(state), flush=True)
    for path, digest in source_hashes.items():
        if sha(path) != digest:
            raise ValueError("Source metadata changed")
    write(a.out / "audits.json", records)
    write(a.out / "sealed.json", dict(state="ALL_PREDICTIONS_SEALED", n=len(rows), query_GT_in_inference=False,
          manifest_sha256=sha(a.out / "manifest.json"), config_sha256=sha(a.out / "config.json"),
          providers_sha256=sha(a.out / "providers.json"), audits_sha256=sha(a.out / "audits.json"),
          predictions={k:v["predictions_sha256"] for k,v in records.items()}, fields={k:v["fields_sha256"] for k,v in records.items()},
          conditioners={k:v["conditioners_sha256"] for k,v in records.items()}, inference_seconds=time.monotonic()-begin))
    print("ALL_PREDICTIONS_SEALED", flush=True)


def score(a):
    from scipy.ndimage import distance_transform_edt
    from sklearn.metrics import roc_auc_score
    seal = json.loads((a.out / "sealed.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 241:
        raise ValueError("Only full sealed DEV241 can be scored")
    for name in ("manifest", "config", "providers", "audits"):
        if sha(a.out / (name + ".json")) != seal[name + "_sha256"]:
            raise ValueError("Changed sealed metadata")
    config = json.loads((a.out / "config.json").read_text())
    rows, jobs = json.loads((a.out / "manifest.json").read_text()), json.loads((a.out / "providers.json").read_text())
    baseline = Path(config["baseline"])
    prior = {r["key"]: r for r in [json.loads(l) for l in (baseline / "episodes.jsonl").read_text().splitlines() if l]}
    arrays, corrections, details, auc = {}, {}, [], {k:[] for k in ARMS}
    for row, job in zip(rows, jobs):
        key = row["key"]
        for folder in ("fields", "predictions", "conditioners"):
            if sha(a.out / folder / (key + ".npz")) != seal[folder][key]:
                raise ValueError("Changed prediction artifact")
        if sha(job["paths"]["packet"]) != job["sha256"]["packet"]:
            raise ValueError("Changed truth packet")
        with np.load(job["paths"]["packet"], allow_pickle=False) as z:
            truth = unpack(z["truth"])
        with np.load(a.out / "predictions" / (key + ".npz"), allow_pickle=False) as z:
            masks = {k:unpack(z[k]) for k in z.files}
        item = dict(row, iu={})
        for arm, mask in masks.items():
            iu = [int((mask & truth).sum()), int((mask | truth).sum())]
            if arm not in ARMS and iu != prior[key]["iu"][arm]:
                raise ValueError("Original baseline I/U drift: " + key + "/" + arm)
            arrays.setdefault(arm, []).append(iu)
            item["iu"][arm] = iu
            add, delete = mask & ~masks["native"], masks["native"] & ~mask
            corrections.setdefault(arm, []).append(dict(key=key,c=row["c"],fold=row["fold"],batch=str(row.get("batch","unspecified")),
                add_TP=int((add & truth).sum()),add_FP=int((add & ~truth).sum()),delete_FP=int((delete & ~truth).sum()),delete_TP=int((delete & truth).sum())))
        raw_origin = masks["model.raw_nn"]
        item["edits_vs_raw_origin"] = {}
        for arm in ARMS:
            add, delete = masks[arm] & ~raw_origin, raw_origin & ~masks[arm]
            item["edits_vs_raw_origin"][arm] = dict(add_TP=int((add&truth).sum()),add_FP=int((add&~truth).sum()),delete_FP=int((delete&~truth).sum()),delete_TP=int((delete&truth).sum()))
        goldcov = truth.reshape(64,16,64,16).mean((1,3))
        center = np.s_[8::16,8::16]
        deepFN = (goldcov >= .9) & ~masks["fine.rcg16.control"][center] & (distance_transform_edt(truth)[center] > 16)
        deepFP = (goldcov <= .1) & masks["fine.rcg16.control"][center] & (distance_transform_edt(~truth)[center] > 16)
        item["deep_FN_FP_tokens"] = [int(deepFN.sum()),int(deepFP.sum())]
        with np.load(a.out / "fields" / (key + ".npz"), allow_pickle=False) as z:
            for arm in ARMS:
                v = z[arm]
                auc[arm].append(float(roc_auc_score(np.r_[np.ones(deepFN.sum()),np.zeros(deepFP.sum())],np.r_[v[deepFN],v[deepFP]])) if deepFN.any() and deepFP.any() else None)
        details.append(item)
    arrays = {k:np.asarray(v,np.int64) for k,v in arrays.items()}
    result, draws = summarize(rows, arrays, corrections)
    groups, classes = photo_groups(rows), np.array([r["c"] for r in rows])
    weights = np.stack([np.bincount(d,minlength=int(groups.max())+1) for d in draws])[:,groups]
    origin_samples = np.array([metric(arrays["model.raw_nn"],classes,w) for w in weights])
    for arm in ARMS:
        samples = np.array([metric(arrays[arm],classes,w) for w in weights])
        result["contrasts"][arm]["model.raw_nn"] = dict(gain=result["scores"][arm]-result["scores"]["model.raw_nn"],ci95=np.percentile(samples-origin_samples,[2.5,97.5]).tolist())
    result["deep_FN_FP_token_AUROC_diagnostic"] = {}
    for arm, values in auc.items():
        ok = np.array([x is not None for x in values]);v = np.array([x or 0 for x in values]);den = weights[:,ok].sum(1)
        boots = (weights[:,ok]@v[ok])/np.maximum(den,1)
        result["deep_FN_FP_token_AUROC_diagnostic"][arm] = dict(n_eligible=int(ok.sum()),mean=float(v[ok].mean()) if ok.any() else None,
           ci95=np.percentile(boots[den>0],[2.5,97.5]).tolist() if ok.any() else None,per_episode=values,orientation="higher=FG, no sign flip",conditioned_on="same fixed fine16 errors after full seal")
    result.update(config=config, source_prediction_seal_sha256=sha(a.out/"sealed.json"), original_baseline_IU_exact=True,
                  common_origin="unchanged sealed model.raw_nn", complete_masks_primary=True)
    np.savez_compressed(a.out/"IU.npz",**arrays)
    np.save(a.out/"bootstrap_photo_draws.npy",draws)
    (a.out/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in details))
    write(a.out/"report.json",result)
    print(json.dumps(dict(n=241,scores=result["scores"],contrasts={k:result["contrasts"][k] for k in ARMS})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["infer","score","all"])
    p.add_argument("--raw",type=Path,default=Path("outputs/gpu_multilayer_dev241_v1"))
    p.add_argument("--fine",type=Path,default=Path("outputs/frozen_subtoken1200_v1"))
    p.add_argument("--baseline",type=Path,default=Path("outputs/frozen_fine_raw_dev241_v1"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if not 1<=a.workers<=6 or (a.smoke and a.stage!="infer"):
        p.error("CPU1..6; smoke is infer-only")
    if a.stage in ["infer","all"]:infer(a)
    if a.stage in ["score","all"]:score(a)


if __name__=="__main__":main()
