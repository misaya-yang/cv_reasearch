#!/usr/bin/env python3
"""CPU-only region-mean reference diagnostic on the full exposed DEV241.

Query truth supplies privileged semantic connected regions, never proposals or
an inference method. Read only existing sealed raw/projected features and masks.
No encoder, image, fitting, sign reversal, threshold search, or GPU computation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

CUES = ("raw_mean_prototype", "projected_mean_prototype", "stored_nn_mean", "fine64_field_mean")
THRESHOLDS = dict(zip(CUES, (0.0, 0.0, 0.0, 0.5)))
CENTERS = np.arange(64) * 16 + 8
CONFIG = dict(
    purpose="fixed, GT-privileged region-mean semantic-reference feasibility; not an inference method",
    dataset="COCO-20i", cohort="all241 existing exposed DEV identities; not independent confirmation",
    resolution=1024, seed=0, raw_layer=24, grid=64,
    region_geometry="8-connected semantic GT components (not instances) and sealed fine.rcg64 components",
    positive_regions="GT components with zero pixel overlap with fine.rcg64",
    negative_regions="fine.rcg64 components with zero pixel overlap with GT",
    eligible_positive_tokens="16x16 GT coverage>=0.9 and token center belongs to that GT component",
    eligible_negative_tokens="16x16 GT coverage<=0.1 and token center belongs to that stray prediction component",
    token_center="pixel index16*i+8, same geometry for every cue",
    region_descriptor="unit(mean(unit(query_tokens))) over region's eligible tokens",
    reference_FG="unit(mean(unit(r[cov>=0.9]))); if absent, single row-major argmax(cov) token",
    reference_BG="unit(mean(unit(r[cov==0]))); require at least one exact-zero-coverage reference token",
    raw="sealed q24/r24 FP32 encoder-intermediate norm_true activations, before FoRIS positional projection",
    projected="matching1200 cached FP16 FoRIS q/r, with its recorded conditional debias flag; cast FP32 then token-normalize",
    precision_caveat="raw FP32 versus cached FP16 includes quantization difference; only cache-debiased=True episodes applied projection, reported separately",
    prototype_margin="fixed cosine(region_descriptor, FGprototype)-cosine(region_descriptor, BGprototype)",
    controls="region mean of stored packet fg_max-bg_max; region mean of2x2-averaged128-grid fine64 field on same eligible64-grid tokens",
    orientation="higher means target FG, never reverse posthoc",
    fitted_weights=0, variants_searched=0, thresholds=THRESHOLDS,
    primary="mean per-episode region AUROC, only episodes with both eligible positive and negative regions",
    secondary="region-pooled AUROC and region-area-weighted pooled AUROC; fixed-threshold balanced accuracy",
    bootstrap=dict(draws=2000, rng="RandomState(0)", unit="connected reference/query photographs", paired=True),
    conditioning_caveat="fine64 field scores are conditioned on errors of its own rendered mask; poor separation can be tautological",
    interpretation="positive cue does not supply legal object proposals or complete-mask gains; negative result limits this fixed mean-prototype construction",
    encoder_forwards=0, images_opened=0, downloads=0, device="cpu", query_GT_for_diagnostics=True,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(row):
    return int(row["c"]), Path(row["support"]).name, Path(row["query"]).name


def unpack(value):
    if value.dtype != np.uint8 or value.shape != (131072,):
        raise ValueError("Require a packed1024 mask")
    return np.unpackbits(value).reshape(1024, 1024).astype(bool)


def unit(value):
    value = np.asarray(value, np.float32)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Nonfinite or zero-norm descriptor")
    return value / norms


def auroc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    ranks = rankdata(np.concatenate((positive, negative)), method="average")
    p, n = len(positive), len(negative)
    return float((ranks[:p].sum() - p * (p + 1) / 2) / (p * n))


def weighted_auc(scores, labels, weights):
    """Weighted pair probability; ties receive half-credit, no GT sign flip."""
    scores, labels, weights = np.asarray(scores), np.asarray(labels, bool), np.asarray(weights, float)
    if not labels.any() or labels.all():
        return None
    order = np.argsort(scores, kind="stable")
    scores, labels, weights = scores[order], labels[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    pos = np.add.reduceat(weights * labels, starts)
    neg = np.add.reduceat(weights * ~labels, starts)
    return float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)) / (pos.sum() * neg.sum()))


def regions(truth, prediction):
    if truth.shape != (1024, 1024) or prediction.shape != truth.shape:
        raise ValueError("Unexpected mask geometry")
    coverage = truth.reshape(64, 16, 64, 16).mean((1, 3))
    positive_labels, np_ = ndimage.label(truth, np.ones((3, 3), bool))
    negative_labels, nn_ = ndimage.label(prediction, np.ones((3, 3), bool))
    hits = np.bincount(positive_labels[prediction], minlength=np_ + 1) > 0
    touch = np.bincount(negative_labels[truth], minlength=nn_ + 1) > 0
    hits[0] = touch[0] = True
    records, token_ids = [], []
    for role, labels, selected, pure in (
        ("whole_missed_GT", positive_labels, np.flatnonzero(~hits), coverage >= 0.9),
        ("stray_fine64", negative_labels, np.flatnonzero(~touch), coverage <= 0.1),
    ):
        sizes = np.bincount(labels.ravel())
        membership = labels[np.ix_(CENTERS, CENTERS)].ravel()
        for region in selected:
            ids = np.flatnonzero(pure.ravel() & (membership == region))
            records.append(dict(role=role, region=int(region), area=int(sizes[region]),
                                pure_tokens=int(len(ids)), eligible=bool(len(ids))))
            token_ids.append(ids)
    return records, token_ids


def reference_roles(coverage):
    coverage = np.asarray(coverage, np.float32).reshape(-1)
    if coverage.shape != (4096,) or not np.isfinite(coverage).all() or coverage.min() < 0 or coverage.max() > 1:
        raise ValueError("Invalid supplied-reference coverage")
    fg, bg = coverage >= 0.9, coverage == 0
    pure_count = int(fg.sum())
    fallback = not bool(pure_count)
    if fallback:
        fg[int(coverage.argmax())] = True
    if not bg.any():
        raise ValueError("Reference has no exact-background tokens for the declared prototype")
    return fg, bg, dict(pure_FG_tokens=pure_count, selected_FG_tokens=int(fg.sum()),
                       exact_BG_tokens=int(bg.sum()), mixed_tokens=int(((coverage > 0) & (coverage < 0.9)).sum()),
                       fallback=fallback, fallback_argmax_index=int(coverage.argmax()) if fallback else None,
                       max_coverage=float(coverage.max()), selected_FG_min_coverage=float(coverage[fg].min()),
                       selected_FG_mean_coverage=float(coverage[fg].mean()))


def score_regions(records, token_ids, qraw, rraw, qprojected, rprojected, coverage, nnmargin, finefield):
    fg, bg, ref = reference_roles(coverage)
    eligible = [i for i, r in enumerate(records) if r["eligible"]]
    for cue, q, r in ((CUES[0], qraw, rraw), (CUES[1], qprojected, rprojected)):
        q, r = unit(q), unit(r)
        fgp, bgp = unit(r[fg].mean(0)), unit(r[bg].mean(0))
        for i in eligible:
            descriptor = unit(q[token_ids[i]].mean(0))
            records[i].setdefault("scores", {})[cue] = float(descriptor @ fgp - descriptor @ bgp)
    for i in eligible:
        records[i]["scores"][CUES[2]] = float(nnmargin.ravel()[token_ids[i]].mean())
        records[i]["scores"][CUES[3]] = float(finefield.ravel()[token_ids[i]].mean())
    cues = {}
    for cue in CUES:
        p = [r["scores"][cue] for r in records if r["eligible"] and r["role"] == "whole_missed_GT"]
        n = [r["scores"][cue] for r in records if r["eligible"] and r["role"] == "stray_fine64"]
        tp = sum(v > THRESHOLDS[cue] for v in p)
        fp = sum(v > THRESHOLDS[cue] for v in n)
        cues[cue] = dict(auc=auroc(p, n), rule=dict(TP=tp, FN=len(p) - tp, FP=fp, TN=len(n) - fp),
                         balanced_accuracy=float((tp / len(p) + 1 - fp / len(n)) / 2) if p and n else None)
    return dict(regions=records, reference=ref, cues=cues)


def one(job):
    import torch
    torch.set_num_threads(1)
    paths, expected = job["paths"], job["sha256"]
    for name, path in paths.items():
        if sha(path) != expected[name]:
            raise ValueError("Changed sealed source " + name + " " + job["old"]["key"])
    with np.load(paths["packet"], allow_pickle=False) as z:
        packed = z["truth"].copy()
        truth = unpack(packed)
        coverage = z["cov"].copy()
        nnmargin = (z["fg_max"] - z["bg_max"]).reshape(64, 64).copy()
    with np.load(paths["prediction"], allow_pickle=False) as z:
        prediction = unpack(z["fine.rcg64"])
        iu = {arm: [int((unpack(z[arm]) & truth).sum()), int((unpack(z[arm]) | truth).sum())]
              for arm in job["prior_iu"]}
    if iu != job["prior_iu"]:
        raise ValueError("Prior scored1200 six-arm I/U mismatch " + job["new"]["key"])
    with np.load(paths["field"], allow_pickle=False) as z:
        finefield = np.asarray(z["fine.rcg64"], np.float32).copy()
    if finefield.shape != (128, 128):
        raise ValueError("Unexpected128-grid fine64 field")
    # Mean over the four existing fine tokens in each coarse cell, no new interpolation or mask.
    finefield = finefield.reshape(64, 2, 64, 2).mean((1, 3))
    with np.load(paths["raw"], allow_pickle=False) as z:
        qraw, rraw = z["q24"].copy(), z["r24"].copy()
    cache = torch.load(paths["projected"], map_location="cpu", weights_only=True)
    if bool(cache["debiased"]) != job["cached_debiased"]:
        raise ValueError("Recorded and actual FoRIS conditional-debias flags disagree")
    if any(cache[k].dtype != torch.float16 for k in ("q", "r")):
        raise ValueError("Unexpected projected cache precision")
    qprojected, rprojected = (cache[k].float().numpy() for k in ("q", "r"))
    if any(x.shape != (4096, 1024) or x.dtype != np.float32 for x in (qraw, rraw, qprojected, rprojected)):
        raise ValueError("Unexpected feature geometry or precision")
    if not np.isfinite(nnmargin).all() or not np.isfinite(finefield).all():
        raise ValueError("Nonfinite stored cue")
    records, token_ids = regions(truth, prediction)
    result = score_regions(records, token_ids, qraw, rraw, qprojected, rprojected, coverage, nnmargin, finefield)
    result.update(cached_debiased=bool(cache["debiased"]),
                  truth_packed_sha256=array_sha(packed), reference_coverage_array_sha256=array_sha(coverage),
                  exact_prior_six_arm_IU=iu, truth_pixels=int(truth.sum()), prediction_pixels=int(prediction.sum()))
    return result


def photo_groups(rows):
    parent, seen = list(range(len(rows))), {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(rows):
        for role in ("support", "query"):
            photo = Path(row[role]).name
            if photo in seen:
                a, b = find(i), find(seen[photo])
                parent[max(a, b)] = min(a, b)
            seen[photo] = i
    groups, mapping = [], {}
    for i in range(len(rows)):
        root = find(i)
        mapping.setdefault(root, len(mapping))
        groups.append(mapping[root])
    return np.array(groups)


def summarize(rows, results):
    groups = photo_groups(rows)
    g = int(groups.max()) + 1
    draws = np.random.RandomState(0).randint(g, size=(2000, g))
    weights = np.stack([np.bincount(d, minlength=g) for d in draws])
    def mean(values):
        values = np.asarray([np.nan if v is None else v for v in values], float)
        valid = np.isfinite(values)
        count = np.bincount(groups[valid], minlength=g)
        total = np.bincount(groups[valid], weights=values[valid], minlength=g)
        denominator = weights @ count
        samples = np.divide(weights @ total, denominator, out=np.full(2000, np.nan), where=denominator > 0)
        finite = np.isfinite(samples)
        return dict(mean=float(values[valid].mean()) if valid.any() else None,
                    ci95=np.percentile(samples[finite], (2.5, 97.5)).tolist() if finite.any() else None,
                    eligible_episodes=int(valid.sum()), eligible_photo_groups=int((count > 0).sum()),
                    finite_bootstrap_draws=int(finite.sum()))
    def contrast(a, b):
        return mean([x - y if x is not None and y is not None else None for x, y in zip(a, b)])
    all_regions = [r for result in results for r in result["regions"] if r["eligible"]]
    labels = [r["role"] == "whole_missed_GT" for r in all_regions]
    area = [r["area"] for r in all_regions]
    report = dict(state="COMPLETED", n=len(rows), classes=len({r["c"] for r in rows}),
                  photo_groups=g, largest_photo_group=int(np.bincount(groups).max()), config=CONFIG,
                  cues={}, paired_contrasts={}, regions={}, folds={}, reference={}, source_space_subsets={})
    auc = {cue: [r["cues"][cue]["auc"] for r in results] for cue in CUES}
    for cue in CUES:
        report["cues"][cue] = dict(
            episode_auc=mean(auc[cue]), episode_auc_minus_chance=mean([v - 0.5 if v is not None else None for v in auc[cue]]),
            balanced_accuracy=mean([r["cues"][cue]["balanced_accuracy"] for r in results]),
            region_pooled_auc=weighted_auc([r["scores"][cue] for r in all_regions], labels, np.ones(len(all_regions))),
            region_area_weighted_pooled_auc=weighted_auc([r["scores"][cue] for r in all_regions], labels, area),
            fixed_threshold=THRESHOLDS[cue],
            pooled_rule={k: sum(r["cues"][cue]["rule"][k] for r in results) for k in ("TP", "FN", "FP", "TN")})
    for a, b in ((CUES[0], CUES[1]), (CUES[0], CUES[2]), (CUES[1], CUES[2]), (CUES[0], CUES[3])):
        report["paired_contrasts"][a + "-minus-" + b] = contrast(auc[a], auc[b])
    for applied in (True, False):
        subset_auc = {cue: [v if r["cached_debiased"] == applied else None for v, r in zip(auc[cue], results)] for cue in CUES}
        report["source_space_subsets"]["projection_applied" if applied else "projection_not_applied"] = dict(
            n=sum(r["cached_debiased"] == applied for r in results),
            episode_auc={cue: mean(subset_auc[cue]) for cue in CUES},
            raw_minus_projected=contrast(subset_auc[CUES[0]], subset_auc[CUES[1]]),
            raw_minus_stored_nn=contrast(subset_auc[CUES[0]], subset_auc[CUES[2]]))
    for role in ("whole_missed_GT", "stray_fine64"):
        rr = [r for result in results for r in result["regions"] if r["role"] == role]
        eligible = [r for r in rr if r["eligible"]]
        ineligible = [r for r in rr if not r["eligible"]]
        report["regions"][role] = dict(
            all_regions=len(rr), eligible_regions=len(eligible), regions_without_pure_tokens=len(ineligible),
            total_pixels=sum(r["area"] for r in rr), eligible_pixels=sum(r["area"] for r in eligible),
            pixels_without_pure_tokens=sum(r["area"] for r in ineligible), pure_tokens=sum(r["pure_tokens"] for r in rr),
            episodes_with_regions=sum(any(r["role"] == role for r in result["regions"]) for result in results),
            episodes_with_eligible_regions=sum(any(r["role"] == role and r["eligible"] for r in result["regions"]) for result in results),
            small_regions_area_lt256=sum(r["area"] < 256 for r in rr),
            small_regions_without_pure_tokens=sum(r["area"] < 256 for r in ineligible),
            no_pure_token_area_percentiles=np.percentile([r["area"] for r in ineligible], [0, 50, 90, 100]).tolist() if ineligible else None)
    for fold in sorted({r["fold"] for r in rows}):
        ix = [i for i, r in enumerate(rows) if r["fold"] == fold]
        values = lambda cue: [auc[cue][i] for i in ix if auc[cue][i] is not None]
        report["folds"][str(fold)] = dict(n=len(ix), eligible_episodes=len(values(CUES[0])),
            episode_auc={cue: float(np.mean(values(cue))) if values(cue) else None for cue in CUES},
            raw_minus_projected=float(np.mean(np.array(values(CUES[0])) - values(CUES[1]))) if values(CUES[0]) else None,
            raw_minus_stored_nn=float(np.mean(np.array(values(CUES[0])) - values(CUES[2]))) if values(CUES[0]) else None)
    refs = [r["reference"] for r in results]
    fallback = [r for r in refs if r["fallback"]]
    report["reference"] = dict(fallback_episodes=len(fallback),
        fallback_max_coverage=[r["max_coverage"] for r in fallback],
        pure_FG_tokens_percentiles=np.percentile([r["pure_FG_tokens"] for r in refs], [0, 25, 50, 75, 100]).tolist(),
        exact_BG_tokens_percentiles=np.percentile([r["exact_BG_tokens"] for r in refs], [0, 25, 50, 75, 100]).tolist(),
        selected_FG_min_coverage_percentiles=np.percentile([r["selected_FG_min_coverage"] for r in refs], [0, 25, 50, 75, 100]).tolist(),
        selected_FG_mean_coverage_percentiles=np.percentile([r["selected_FG_mean_coverage"] for r in refs], [0, 25, 50, 75, 100]).tolist())
    return report, groups, draws


def source_jobs(raw, fine):
    raw_seal = json.loads((raw / "sealed.json").read_text())
    fine_seal = json.loads((fine / "sealed.json").read_text())
    old = json.loads((raw / "manifest.json").read_text())
    new = json.loads((fine / "manifest.json").read_text())
    scored = [json.loads(line) for line in (fine / "episodes.jsonl").read_text().splitlines() if line.strip()]
    report = json.loads((fine / "report.json").read_text())
    if len(old) != 241 or len(new) != 1200 or len(scored) != 1200:
        raise ValueError("Require full241 raw cohort and fully scored1200 masks")
    if len({identity(r) for r in old}) != 241 or len({identity(r) for r in new}) != 1200:
        raise ValueError("Ambiguous semantic episode identity")
    for folder, seal in ((raw, raw_seal), (fine, fine_seal)):
        if seal["state"] != "ALL_PREDICTIONS_SEALED" or sha(folder / "manifest.json") != seal["manifest_sha256"]:
            raise ValueError("Unsealed or changed source manifest")
    if raw_seal["query_labels_opened"] or fine_seal["query_truth_opened"]:
        raise ValueError("Inference sealing reports query-label exposure")
    if sha(raw / "protocol.json") != raw_seal["protocol_sha256"]:
        raise ValueError("Changed raw feature protocol")
    if sha(fine / "config.json") != fine_seal["config_sha256"] or sha(fine / "sealed.json") != report["source_prediction_seal_sha256"]:
        raise ValueError("Scored fine1200 config/seal identity differs")
    metadata = {r["key"]: r for r in [json.loads(line) for line in (raw / "episodes.jsonl").read_text().splitlines()]}
    byidentity, bykey = {identity(r): r for r in new}, {r["key"]: r for r in scored}
    jobs = []
    for row in old:
        matched = byidentity[identity(row)]
        if row["fold"] != matched["fold"] or identity(matched) != identity(bykey[matched["key"]]):
            raise ValueError("Old/new/scored identity differs")
        extraction = metadata[row["key"]]["receipts"]["extraction"]
        if extraction["space"] != "encoder_intermediate_norm_true" or extraction["intervention"]:
            raise ValueError("Unexpected raw encoder feature provenance")
        key, oldkey = matched["key"], row["key"]
        source = fine_seal["inputs"][key]
        jobs.append(dict(old=row, new=matched, prior_iu=bykey[key]["iu"], raw_extraction_receipt=extraction,
            cached_debiased=bool(source["debiased"]),
            paths=dict(raw=str(raw / "layers" / (oldkey + ".npz")), projected=matched["feature_export"],
                       packet=matched["packet_export"], prediction=str(fine / "predictions" / (key + ".npz")),
                       field=str(fine / "fields" / (key + ".npz"))),
            sha256=dict(raw=raw_seal["layers"][oldkey], projected=source["feature_sha256"],
                        packet=source["packet_sha256"], prediction=fine_seal["predictions"][key], field=fine_seal["fields"][key])))
    hashes = {str(folder / name): sha(folder / name) for folder, names in (
        (raw, ("manifest.json", "sealed.json", "protocol.json", "episodes.jsonl")),
        (fine, ("manifest.json", "sealed.json", "config.json", "episodes.jsonl", "report.json"))) for name in names}
    return jobs, hashes


def self_test():
    assert auroc([1.0], [0.0]) == 1.0 and auroc([0.0], [1.0]) == 0.0
    assert auroc([1.0], [1.0]) == 0.5 and auroc([], [1.0]) is None
    assert weighted_auc([1, 1, 0], [True, False, False], [3, 2, 4]) == 5 / 6
    truth, pred = np.zeros((1024, 1024), bool), np.zeros((1024, 1024), bool)
    truth[64:128, 64:128] = True
    truth[20:24, 20:24] = True
    pred[256:320, 256:320] = True
    rr, ids = regions(truth, pred)
    assert [r["pure_tokens"] for r in rr] == [0, 16, 16]
    assert [r["area"] for r in rr] == [16, 4096, 4096]
    # Two semantic regions joined only diagonally are one8-connected component.
    diagonal = truth.copy()
    diagonal[128, 128] = True
    assert len(regions(diagonal, pred)[0]) == 3
    q = np.tile([0.0, 1.0], (4096, 1)).astype(np.float32)
    q[ids[1]] = [1.0, 0.0]
    r = np.tile([0.0, 1.0], (4096, 1)).astype(np.float32)
    r[0] = [1.0, 0.0]
    cov = np.zeros(4096, np.float32)
    cov[0] = 1
    cue = np.zeros((64, 64), np.float32)
    cue.ravel()[ids[1]], cue.ravel()[ids[2]] = 1, -1
    result = score_regions(rr, ids, q, r, q, r, cov, cue, cue)
    result["cached_debiased"] = True
    assert all(result["cues"][c]["auc"] == 1 for c in CUES)
    cov[0] = 0.2
    fg, bg, ref = reference_roles(cov)
    assert fg.sum() == 1 and fg[0] and bg.sum() == 4095 and ref["fallback"]
    rows = [dict(fold=0, c=0, support="a", query="b"), dict(fold=0, c=0, support="b", query="c")]
    assert photo_groups(rows).tolist() == [0, 0]
    report, _, _ = summarize(rows, [result, result])
    assert report["cues"][CUES[0]]["episode_auc"]["mean"] == 1
    assert report["paired_contrasts"][CUES[0] + "-minus-" + CUES[1]]["mean"] == 0
    print("SELF_TEST_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--fine", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    a = parser.parse_args()
    if a.self_test:
        self_test()
        return
    if not all((a.raw, a.fine, a.out)) or not 1 <= a.workers <= 2:
        parser.error("Require raw/fine/out and1..2 CPU workers")
    if a.out.exists():
        raise FileExistsError("Never overwrite existing evidence: " + str(a.out))
    begin = time.monotonic()
    jobs, source_hashes = source_jobs(a.raw.resolve(), a.fine.resolve())
    a.out.mkdir(parents=True)
    write(a.out / "config.json", CONFIG)
    write(a.out / "manifest.json", [dict(old=j["old"], new=j["new"]) for j in jobs])
    state = dict(state="CPU_GT_DIAGNOSTIC_RUNNING", pid=os.getpid(), workers=a.workers, completed=0, n=241,
                 started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source_code_sha256=sha(__file__))
    write(a.out / "state.json", state)
    results = []
    with ProcessPoolExecutor(a.workers) as pool:
        for n, result in enumerate(pool.map(one, jobs, chunksize=1), 1):
            results.append(result)
            if n % 40 == 0 or n == 241:
                state.update(completed=n, seconds=time.monotonic() - begin)
                write(a.out / "state.json", state)
                print(json.dumps(dict(completed=n, n=241, seconds=state["seconds"])), flush=True)
    rows = [j["old"] for j in jobs]
    report, groups, draws = summarize(rows, results)
    for path, digest in source_hashes.items():
        if sha(path) != digest:
            raise ValueError("Source metadata changed during diagnostic " + path)
    report.update(seconds=time.monotonic() - begin, source_code_sha256=sha(__file__),
                  all241_source_hashes_verified=True, all241_old_new_semantic_identities_matched=True,
                  exact_prior1200_six_arm_IU_parity=True)
    write(a.out / "report.json", report)
    write(a.out / "receipt.json", dict(source_metadata_sha256=source_hashes, source_code_sha256=sha(__file__),
        inputs=[dict(j, truth_packed_sha256=r["truth_packed_sha256"],
                     reference_coverage_array_sha256=r["reference_coverage_array_sha256"]) for j, r in zip(jobs, results)],
        query_truth_used_for_GT_geometry_only=True, all_source_complete_masks_sealed_before_diagnostic=True))
    with (a.out / "episodes.jsonl").open("w") as stream:
        for job, result in zip(jobs, results):
            stream.write(json.dumps(dict(old=job["old"], new=job["new"], diagnostic=result), allow_nan=False) + "\n")
    np.save(a.out / "bootstrap_photo_draws.npy", draws)
    np.savez_compressed(a.out / "counts.npz", photo_groups=groups,
        **{"auc:" + c: np.array([np.nan if r["cues"][c]["auc"] is None else r["cues"][c]["auc"] for r in results]) for c in CUES})
    lines = ["# Region-mean semantic reference diagnostic: full exposed DEV241", "",
             "GT geometry is privileged. No legal proposal generator, complete recovery method, encoder call, fitting or sign inversion.", "",
             "| Fixed cue | Eligible episodes | Episode region AUROC [95% CI] | Region-pooled AUROC | Pixel-mass-weighted AUROC |",
             "|---|---:|---:|---:|---:|"]
    for c in CUES:
        s = report["cues"][c]
        v = s["episode_auc"]
        auc_text = "unavailable" if v["mean"] is None else "%.4f [%.4f, %.4f]" % (v["mean"], *v["ci95"])
        pooled = lambda x: "unavailable" if x is None else "%.4f" % x
        lines.append("| %s | %d | %s | %s | %s |" % (c, v["eligible_episodes"], auc_text,
                     pooled(s["region_pooled_auc"]), pooled(s["region_area_weighted_pooled_auc"])))
    lines += ["", "## Paired contrasts", "", json.dumps(report["paired_contrasts"], indent=2),
              "", "## Region availability and pixel mass", "", json.dumps(report["regions"], indent=2),
              "", "## Reference purity and fallback", "", json.dumps(report["reference"], indent=2),
              "", "## Four folds", "", json.dumps(report["folds"], indent=2),
              "", "## Actual conditional projection subsets", "", json.dumps(report["source_space_subsets"], indent=2), "",
              CONFIG["conditioning_caveat"], "", CONFIG["precision_caveat"], "", CONFIG["interpretation"]]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    state.update(state="CPU_GT_DIAGNOSTIC_COMPLETE", completed=241, seconds=report["seconds"], report_sha256=sha(a.out / "report.json"))
    write(a.out / "state.json", state)
    print(json.dumps(dict(state=state["state"], seconds=report["seconds"], cues=report["cues"],
                          paired_contrasts=report["paired_contrasts"], regions=report["regions"])), flush=True)


if __name__ == "__main__":
    main()
