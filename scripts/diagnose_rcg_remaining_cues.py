#!/usr/bin/env python3
"""Fixed CPU diagnostics of existing reference cues on RCG's remaining errors.

No encoder, inference, recovery component, fitting, or threshold search. Query
GT defines post-hoc semantic connected-region/error cohorts only. Scores are
the existing packet NN margin, FoRIS minmax score, and sealed RCG field.
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

CUES = ("signed_nn", "foris_score", "rcg_field")
STRATA = ("whole_regions", "deep_errors")
THRESHOLDS = {"signed_nn": 0.0, "foris_score": 0.5, "rcg_field": 0.5}
CENTERS = np.arange(64) * 16 + 8
CONFIG = dict(
    purpose="post-hoc existing-cue feasibility; no inference or deployable recovery claim",
    resolution=1024, grid=64, GT_pure_FG="16x16 coverage>=0.9",
    GT_pure_BG="16x16 coverage<=0.1", token_membership="mask/region at pixel index16*i+8",
    whole_regions="positive: zero-RCG-overlap 8-connected semantic GT region; negative: zero-GT-overlap 8-connected RCG region",
    deep_errors="GT-pure token center is FN/FP and Euclidean distance to opposite GT label>16",
    strata_overlap="whole-region and deep-error cohorts may overlap; never sum them",
    score_orientation="higher score is target FG; no retrospective sign inversion",
    cues="packet fg_max-bg_max; packet score minmax with floor1e-6; sealed rcg field",
    thresholds=THRESHOLDS, thresholds_searched=False, encoder_forwards=0,
    primary="mean per-episode AUROC on episodes with both eligible labels",
    secondary="existing fixed-rule per-episode balanced accuracy; pooled recall/FPR are descriptive",
    bootstrap=dict(draws=2000, rng="RandomState(0)", unit="connected support/query photographs", paired=True),
    conditioning_caveat="RCG field is conditioned on errors of its rendered mask; poor field AUROC can be partly tautological and does not prove weak features",
    region_proxy="8-connected semantic/prediction regions, not instances; small regions may contain no GT-pure token centers",
    availability="NN>0 in GT-selected regions is a diagnostic budget, not an inference detector",
    exposure="4000 preserved public benchmark draws; reused development, not independent confirmation",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def unpack(value):
    if value.dtype != np.uint8 or value.shape != (131072,):
        raise ValueError("Expected packed1024 mask")
    return np.unpackbits(value).reshape(1024, 1024).astype(bool)


def auroc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    ranks = rankdata(np.concatenate((positive, negative)), method="average")
    p, n = len(positive), len(negative)
    return float((ranks[:p].sum() - p * (p + 1) / 2) / (p * n))


def region_availability(labels, selected_ids, eligible, margin):
    sizes = np.bincount(labels.ravel())
    centers = labels[np.ix_(CENTERS, CENTERS)]
    tokens = np.bincount(centers[eligible], minlength=len(sizes))
    positive = np.bincount(centers[eligible & (margin > 0)], minlength=len(sizes))
    records = [dict(region=int(i), area=int(sizes[i]), pure_tokens=int(tokens[i]),
                    positive_nn_tokens=int(positive[i])) for i in selected_ids]
    availability = dict(
        regions=len(records), pixels=sum(r["area"] for r in records),
        regions_with_pure_tokens=sum(r["pure_tokens"] > 0 for r in records),
        regions_without_pure_tokens=sum(r["pure_tokens"] == 0 for r in records),
        pixels_without_pure_tokens=sum(r["area"] for r in records if r["pure_tokens"] == 0),
        regions_with_positive_nn=sum(r["positive_nn_tokens"] > 0 for r in records),
        eligible_regions_no_positive_nn=sum(r["pure_tokens"] > 0 and r["positive_nn_tokens"] == 0 for r in records),
        pure_tokens=sum(r["pure_tokens"] for r in records),
        positive_nn_tokens=sum(r["positive_nn_tokens"] for r in records),
        small_regions_area_lt256=sum(r["area"] < 256 for r in records),
        small_regions_without_pure_tokens=sum(r["area"] < 256 and r["pure_tokens"] == 0 for r in records),
    )
    return availability, records


def analyze(truth, rcg, native, cues):
    if truth.shape != (1024, 1024) or any(v.shape != (64, 64) for v in cues.values()):
        raise ValueError("Wrong mask or cue geometry")
    if any(not np.isfinite(v).all() for v in cues.values()):
        raise ValueError("Nonfinite cue")
    coverage = truth.reshape(64, 16, 64, 16).mean((1, 3))
    pure_fg, pure_bg = coverage >= 0.9, coverage <= 0.1
    tc, rc = (m[np.ix_(CENTERS, CENTERS)] for m in (truth, rcg))
    eight = np.ones((3, 3), bool)
    gt_labels, ng = ndimage.label(truth, eight)
    hits = np.bincount(gt_labels[rcg], minlength=ng + 1) > 0
    hits[0] = True
    lost = ~hits[gt_labels]
    rcg_labels, nr = ndimage.label(rcg, eight)
    overlaps = np.bincount(rcg_labels[truth], minlength=nr + 1) > 0
    overlaps[0] = True
    stray = ~overlaps[rcg_labels]
    if truth.any() and not truth.all():
        inner = ndimage.distance_transform_edt(truth)[np.ix_(CENTERS, CENTERS)]
        outer = ndimage.distance_transform_edt(~truth)[np.ix_(CENTERS, CENTERS)]
    else:
        inner = outer = np.full((64, 64), np.inf)
    cohorts = dict(
        whole_regions=(pure_fg & lost[np.ix_(CENTERS, CENTERS)],
                       pure_bg & stray[np.ix_(CENTERS, CENTERS)]),
        deep_errors=(pure_fg & tc & ~rc & (inner > 16),
                     pure_bg & ~tc & rc & (outer > 16)),
    )
    result = dict(
        pure_fg_tokens=int(pure_fg.sum()), pure_bg_tokens=int(pure_bg.sum()),
        mixed_GT_tokens=int((~pure_fg & ~pure_bg).sum()),
        pure_FG_centers_not_FG=int((pure_fg & ~tc).sum()),
        pure_BG_centers_not_BG=int((pure_bg & tc).sum()),
        iu={name: [int((m & truth).sum()), int((m | truth).sum())]
            for name, m in (("native", native), ("rcg", rcg))},
        strata={}, regions={}, region_details={},
    )
    for name, (positive, negative) in cohorts.items():
        entry = dict(positive_tokens=int(positive.sum()), negative_tokens=int(negative.sum()), cues={})
        for cue, values in cues.items():
            p, n = values[positive], values[negative]
            tp, fp = int((p > THRESHOLDS[cue]).sum()), int((n > THRESHOLDS[cue]).sum())
            rule = dict(TP=tp, FN=len(p) - tp, FP=fp, TN=len(n) - fp)
            entry["cues"][cue] = dict(auc=auroc(p, n), rule=rule,
                balanced_accuracy=float((tp / len(p) + 1 - fp / len(n)) / 2) if len(p) and len(n) else None)
        result["strata"][name] = entry
    for name, labels, ids, eligible in (
        ("whole_missed_GT", gt_labels, np.flatnonzero(~hits), cohorts["whole_regions"][0]),
        ("stray_RCG", rcg_labels, np.flatnonzero(~overlaps), cohorts["whole_regions"][1]),
    ):
        summary, details = region_availability(labels, ids, eligible, cues["signed_nn"])
        result["regions"][name], result["region_details"][name] = summary, details
    result["strata_overlap_tokens"] = dict(
        positive=int((cohorts["whole_regions"][0] & cohorts["deep_errors"][0]).sum()),
        negative=int((cohorts["whole_regions"][1] & cohorts["deep_errors"][1]).sum()))
    return result


def one(job):
    row, packet, prediction, field, expected = job
    for name, path in (("packet", packet), ("prediction", prediction), ("field", field)):
        if sha(path) != expected[name]:
            raise ValueError("Changed " + name + " " + row["key"])
    with np.load(packet, allow_pickle=False) as z:
        truth, native = unpack(z["truth"]), unpack(z["native"])
        score = np.asarray(z["score"], np.float32).reshape(64, 64).copy()
        nn = (np.asarray(z["fg_max"], np.float32) - np.asarray(z["bg_max"], np.float32)).reshape(64, 64).copy()
    with np.load(prediction, allow_pickle=False) as z:
        rcg = unpack(z["RCG"])
    with np.load(field, allow_pickle=False) as z:
        rcg_field = np.asarray(z["rcg"], np.float32).reshape(64, 64).copy()
    normalized = (score - score.min()) / max(float(score.max() - score.min()), 1e-6)
    result = analyze(truth, rcg, native, dict(signed_nn=nn, foris_score=normalized, rcg_field=rcg_field))
    for name in ("native", "rcg"):
        if result["iu"][name] != row["iu"][name]:
            raise ValueError("Prior4000 I/U parity failure " + name + " " + row["key"])
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
    ids, groups = {}, []
    for i in range(len(rows)):
        root = find(i)
        ids.setdefault(root, len(ids))
        groups.append(ids[root])
    return np.array(groups)


def bootstrap(rows):
    groups = photo_groups(rows)
    g = int(groups.max()) + 1
    draws = np.random.RandomState(0).randint(g, size=(2000, g))
    weights = np.stack([np.bincount(d, minlength=g) for d in draws])
    def mean(values):
        values = np.asarray([np.nan if v is None else v for v in values], float)
        valid = np.isfinite(values)
        count = np.bincount(groups[valid], minlength=g)
        total = np.bincount(groups[valid], weights=values[valid], minlength=g)
        den = weights @ count
        samples = np.divide(weights @ total, den, out=np.full(2000, np.nan), where=den > 0)
        finite = np.isfinite(samples)
        return dict(mean=float(values[valid].mean()) if valid.any() else None,
                    ci95=np.percentile(samples[finite], (2.5, 97.5)).tolist() if finite.any() else None,
                    eligible_episodes=int(valid.sum()), eligible_photo_groups=int((count > 0).sum()),
                    finite_bootstrap_draws=int(finite.sum()))
    def ratio(numerator, denominator):
        num, den = np.asarray(numerator), np.asarray(denominator)
        gn = np.bincount(groups, weights=num, minlength=g)
        gd = np.bincount(groups, weights=den, minlength=g)
        sampled_d = weights @ gd
        sampled = np.divide(weights @ gn, sampled_d, out=np.full(2000, np.nan), where=sampled_d > 0)
        finite = np.isfinite(sampled)
        return dict(ratio=float(num.sum() / den.sum()) if den.sum() else None,
                    ci95=np.percentile(sampled[finite], (2.5, 97.5)).tolist() if finite.any() else None)
    return groups, draws, mean, ratio


def summarize(rows, results):
    groups, draws, mean, ratio = bootstrap(rows)
    report = dict(n=len(rows), classes=len({r["c"] for r in rows}), photo_groups=int(groups.max()) + 1,
                  largest_photo_group=int(np.bincount(groups).max()), config=CONFIG, strata={}, regions={}, folds={})
    for name in STRATA:
        entries = [r["strata"][name] for r in results]
        summary = dict(positive_tokens=sum(e["positive_tokens"] for e in entries),
                       negative_tokens=sum(e["negative_tokens"] for e in entries),
                       positive_eligible_episodes=sum(e["positive_tokens"] > 0 for e in entries),
                       negative_eligible_episodes=sum(e["negative_tokens"] > 0 for e in entries), cues={}, paired_contrasts={})
        for cue in CUES:
            cue_rows = [e["cues"][cue] for e in entries]
            rules = {k: sum(e["rule"][k] for e in cue_rows) for k in ("TP", "FN", "FP", "TN")}
            summary["cues"][cue] = dict(
                auc=mean([e["auc"] for e in cue_rows]),
                balanced_accuracy=mean([e["balanced_accuracy"] for e in cue_rows]),
                fixed_threshold=THRESHOLDS[cue], pooled_rule_counts=rules,
                pooled_recall=rules["TP"] / (rules["TP"] + rules["FN"]) if rules["TP"] + rules["FN"] else None,
                pooled_FPR=rules["FP"] / (rules["FP"] + rules["TN"]) if rules["FP"] + rules["TN"] else None)
        for other in CUES[1:]:
            a = [e["cues"]["signed_nn"]["auc"] for e in entries]
            b = [e["cues"][other]["auc"] for e in entries]
            summary["paired_contrasts"]["signed_nn-minus-" + other] = mean(
                [x - y if x is not None and y is not None else None for x, y in zip(a, b)])
        report["strata"][name] = summary
    for region in ("whole_missed_GT", "stray_RCG"):
        entries = [r["regions"][region] for r in results]
        report["regions"][region] = {k: sum(e[k] for e in entries) for k in entries[0]}
        report["regions"][region]["positive_nn_fraction_all_regions"] = ratio(
            [e["regions_with_positive_nn"] for e in entries], [e["regions"] for e in entries])
        report["regions"][region]["positive_nn_fraction_eligible_regions"] = ratio(
            [e["regions_with_positive_nn"] for e in entries], [e["regions_with_pure_tokens"] for e in entries])
    for fold in sorted({r["fold"] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r["fold"] == fold]
        report["folds"][str(fold)] = dict(n=len(ids), strata={})
        for name in STRATA:
            report["folds"][str(fold)]["strata"][name] = {}
            for cue in CUES:
                aucs = [results[i]["strata"][name]["cues"][cue]["auc"] for i in ids]
                valid = [v for v in aucs if v is not None]
                report["folds"][str(fold)]["strata"][name][cue] = dict(
                    eligible_episodes=len(valid), auc=float(np.mean(valid)) if valid else None)
    for name in STRATA:
        ci = report["strata"][name]["cues"]["signed_nn"]["auc"]["ci95"]
        report["strata"][name]["NN_ordering_decision"] = (
            "unavailable" if ci is None else "positive ordering evidence" if ci[0] > 0.5
            else "no positive ordering evidence in this tested cue" if ci[1] <= 0.5 else "unresolved")
    return report, groups, draws


def self_test():
    assert auroc(np.array([1.0]), np.array([0.0])) == 1
    assert auroc(np.array([0.0]), np.array([1.0])) == 0
    assert auroc(np.array([1.0]), np.array([1.0])) == 0.5
    assert auroc(np.array([]), np.array([1.0])) is None
    truth = np.zeros((1024, 1024), bool)
    truth[64:128, 64:128] = True
    truth[20:24, 20:24] = True
    rcg = np.zeros_like(truth)
    rcg[256:320, 256:320] = True
    center_truth = truth[np.ix_(CENTERS, CENTERS)]
    nn = np.where(center_truth, 1.0, -1.0)
    field = np.where(center_truth, 0.2, 0.8)
    result = analyze(truth, rcg, np.zeros_like(truth), dict(signed_nn=nn, foris_score=field, rcg_field=field))
    assert result["strata"]["whole_regions"]["positive_tokens"] == 16
    assert result["strata"]["whole_regions"]["negative_tokens"] == 16
    assert result["strata"]["deep_errors"]["positive_tokens"] == 4
    for name in STRATA:
        assert result["strata"][name]["cues"]["signed_nn"]["auc"] == 1
        assert result["strata"][name]["cues"]["rcg_field"]["auc"] == 0
    assert result["regions"]["whole_missed_GT"]["regions"] == 2
    assert result["regions"]["whole_missed_GT"]["regions_without_pure_tokens"] == 1
    empty = analyze(np.zeros_like(truth), rcg, np.zeros_like(truth), dict(signed_nn=nn, foris_score=field, rcg_field=field))
    assert empty["strata"]["whole_regions"]["cues"]["signed_nn"]["auc"] is None
    rows = [dict(c=0, fold=0, support="a", query="b"), dict(c=0, fold=0, support="b", query="c")]
    assert photo_groups(rows).tolist() == [0, 0]
    report, _, _ = summarize(rows, [result, result])
    assert report["strata"]["whole_regions"]["paired_contrasts"]["signed_nn-minus-rcg_field"]["mean"] == 1
    print("SELF_TEST_OK", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--verified-receipt", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--verify-only", action="store_true", help="Check four real inputs without writing output")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if not all((a.root, a.manifest, a.verified_receipt, a.out)) or not 1 <= a.workers <= 4:
        p.error("Require root/manifest/verified-receipt/out and1..4workers")
    if a.out.exists():
        raise FileExistsError("Fresh owned output required: " + str(a.out))
    rows = json.loads(a.manifest.read_text())
    receipt = json.loads(a.verified_receipt.read_text())
    if len(rows) != 4000:
        raise ValueError("Require all4000 preserved draws")
    jobs, source_seals = [], {}
    for row in rows:
        block, key = row["public_batch"], row["key"].split(":", 1)[-1]
        run = a.root / f"outputs/claude_official/run{block}"
        if block not in source_seals:
            seal = json.loads((run / "sealed.json").read_text())
            digest = sha(run / "sealed.json")
            if digest != receipt["source_seals"][str(block)] or seal["state"] != "ALL_PREDICTIONS_SEALED":
                raise ValueError("Changed/unsealed source")
            if sha(run / "manifest.json") != seal["manifest_sha256"]:
                raise ValueError("Changed source manifest")
            source_seals[block] = dict(seal=seal, hash=digest)
        inputs = receipt["inputs"][row["key"]]
        prediction = run / "predictions" / (key + ".npz")
        field = run / "fields" / (key + ".npz")
        packet = a.root / f"outputs/claude_official/root{block}/results/extent_v1/run/packets/{key}.npz"
        expected = dict(packet=inputs["packet_sha256"], prediction=inputs["prediction_sha256"],
                        field=source_seals[block]["seal"]["fields"][key])
        if expected["prediction"] != source_seals[block]["seal"]["predictions"][key]:
            raise ValueError("Source prediction receipts disagree")
        jobs.append((row, str(packet), str(prediction), str(field), expected))
    if a.verify_only:
        begin = time.monotonic()
        for i in (0, 1000, 2000, 3000):
            result = one(jobs[i])
            print(json.dumps(dict(verified=rows[i]["key"], strata={k: {x: v for x, v in s.items() if x != "cues"}
                                      for k, s in result["strata"].items()})), flush=True)
        print(json.dumps(dict(state="REAL_INPUT_SMOKE_OK", seconds=time.monotonic() - begin)), flush=True)
        return
    a.out.mkdir(parents=True)
    write(a.out / "manifest.json", rows)
    write(a.out / "config.json", CONFIG)
    begin, results = time.monotonic(), []
    with ProcessPoolExecutor(a.workers) as pool:
        for n, result in enumerate(pool.map(one, jobs, chunksize=4), 1):
            results.append(result)
            if n % 400 == 0:
                print(json.dumps(dict(completed=n, total=4000, seconds=time.monotonic() - begin)), flush=True)
    report, groups, draws = summarize(rows, results)
    report.update(source_code_sha256=sha(__file__), exact_native_RCG_IU_parity=True,
                  seconds=time.monotonic() - begin, state="COMPLETED")
    write(a.out / "report.json", report)
    write(a.out / "receipt.json", dict(
        manifest_sha256=sha(a.manifest), verified_receipt_sha256=sha(a.verified_receipt),
        source_code_sha256=sha(__file__), source_seals={str(k): s["hash"] for k, s in source_seals.items()},
        inputs={job[0]["key"]: job[-1] for job in jobs}, query_GT_used_for_diagnostics=True))
    with (a.out / "episodes.jsonl").open("w") as stream:
        for row, result in zip(rows, results):
            stream.write(json.dumps(dict(row, diagnostic=result), allow_nan=False) + "\n")
    arrays = dict(photo_groups=groups)
    for name in STRATA:
        arrays["tokens:" + name] = np.array([[r["strata"][name][k] for k in ("positive_tokens", "negative_tokens")] for r in results])
        for cue in CUES:
            arrays["auc:" + name + ":" + cue] = np.array([np.nan if r["strata"][name]["cues"][cue]["auc"] is None
                                                        else r["strata"][name]["cues"][cue]["auc"] for r in results])
            arrays["rule:" + name + ":" + cue] = np.array([[r["strata"][name]["cues"][cue]["rule"][k]
                                                          for k in ("TP", "FN", "FP", "TN")] for r in results])
    np.savez_compressed(a.out / "counts.npz", **arrays)
    np.save(a.out / "bootstrap_photo_draws.npy", draws)
    lines = ["# Existing cues on RCG remaining errors:4000 reused draws", "",
             "GT diagnostics only; semantic connected regions are not instances. No encoder, fitting or threshold search.", "",
             "| cohort | cue | eligible episodes | mean episode AUROC [95% CI] | fixed-rule balanced accuracy | pooled recall / FPR |",
             "|---|---|---:|---|---:|---|"]
    for name in STRATA:
        for cue in CUES:
            s = report["strata"][name]["cues"][cue]
            auc, ci = s["auc"]["mean"], s["auc"]["ci95"]
            auc_text = "unavailable" if auc is None else "%.4f [%.4f, %.4f]" % (auc, *ci)
            ba = s["balanced_accuracy"]["mean"]
            rule_text = "unavailable" if s["pooled_recall"] is None or s["pooled_FPR"] is None else "%.4f / %.4f" % (s["pooled_recall"], s["pooled_FPR"])
            lines.append("| %s | %s | %d | %s | %s | %s |" % (name, cue, s["auc"]["eligible_episodes"], auc_text,
                          "unavailable" if ba is None else "%.4f" % ba, rule_text))
    lines += ["", CONFIG["conditioning_caveat"], "", "## Region availability", "",
              json.dumps(report["regions"], indent=2), "", "## Paired comparisons", "",
              json.dumps({k: v["paired_contrasts"] for k, v in report["strata"].items()}, indent=2), "",
              "Positive NN in a GT-selected region does not supply a deployable detector or complete-mask recovery result."]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(state="REMAINING_CUES_DONE", seconds=report["seconds"], strata=report["strata"], regions=report["regions"])), flush=True)


if __name__ == "__main__":
    main()
