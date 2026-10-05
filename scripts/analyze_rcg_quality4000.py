#!/usr/bin/env python3
"""CPU-only descriptive baseline-quality analysis of all 4,000 sampled draws.

No masks are selected, inferred or changed. Native-IoU bins use query truth and
are diagnostic only. Shared photographs connect draws before subgroup analysis;
every subgroup uses the full cohort's paired RandomState(0) bootstrap weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ics.experiment import metric, photo_groups, sha

BINS = np.array([0, .25, .5, .75, .9, 1.000001])
BIN_NAMES = ("[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,.9)", "[.9,1]")
DRAW_COUNT = 2000


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def write_tsv(path, records, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in records)


class Analysis:
    def __init__(self, rows):
        self.rows = rows
        self.classes = np.array([row["c"] for row in rows])
        self.batches = np.array([row["batch"] for row in rows])
        self.native = np.array([row["iu"]["native"] for row in rows], dtype=np.int64)
        self.rcg = np.array([row["iu"]["rcg"] for row in rows], dtype=np.int64)
        for values in (self.native, self.rcg):
            if np.any(values[:, 1] <= 0) or np.any(values[:, 0] < 0) or np.any(values[:, 0] > values[:, 1]):
                raise ValueError("Invalid paired intersection/union counts")
        self.baseline = self.native[:, 0] / self.native[:, 1]
        self.delta = self.rcg[:, 0] / self.rcg[:, 1] - self.baseline
        self.bin = np.searchsorted(BINS, self.baseline, side="right") - 1
        if np.any((self.bin < 0) | (self.bin >= len(BIN_NAMES))):
            raise ValueError("Native IoU outside the declared bins")
        self.groups = photo_groups(rows)
        group_count = int(self.groups.max()) + 1
        self.draws = np.random.RandomState(0).randint(group_count, size=(DRAW_COUNT, group_count))
        self.group_weights = np.stack([np.bincount(draw, minlength=group_count) for draw in self.draws]).astype(np.int16)
        self.row_weights = self.group_weights[:, self.groups]

    def summary(self, mask, bootstrap=True, fixed_weights=None):
        indices = np.flatnonzero(mask)
        if not len(indices):
            return {"n": 0}
        classes = self.classes[indices]
        ids, class_index = np.unique(classes, return_inverse=True)
        n, r = self.native[indices], self.rcg[indices]
        baseline, delta = self.baseline[indices], self.delta[indices]
        perfect = baseline == 1
        error = 1 - baseline
        weights = np.ones(len(indices)) if fixed_weights is None else np.asarray(fixed_weights)
        if weights.shape != (len(indices),) or np.any(weights <= 0):
            raise ValueError("Expected positive weights for each selected draw")

        def values(w):
            present = np.bincount(class_index, weights=w, minlength=len(ids)) > 0
            scores = []
            for iu in (n, r):
                intersection = np.bincount(class_index, weights=w * iu[:, 0], minlength=len(ids))
                union = np.bincount(class_index, weights=w * iu[:, 1], minlength=len(ids))
                scores.append(100 * np.mean((intersection / np.maximum(union, 1))[present]))
            mean_delta = 100 * np.sum(w * delta) / w.sum()
            denominator = np.sum(w[~perfect] * error[~perfect])
            reduction = 100 * np.sum(w[~perfect] * delta[~perfect]) / denominator if denominator else np.nan
            return np.array([scores[0], scores[1], scores[1] - scores[0], mean_delta, reduction])

        point = values(weights)
        result = {
            "n": len(indices), "classes": len(ids),
            "photo_groups_present": len(np.unique(self.groups[indices])),
            "native_class_macro_miou": float(point[0]), "rcg_class_macro_miou": float(point[1]),
            "class_macro_gain_pp": float(point[2]),
            "mean_episode_native_iou_percent": float(100 * np.average(baseline, weights=weights)),
            "mean_episode_gain_pp": float(point[3]),
            "mean_episode_ceiling_headroom_pp": float(100 * np.average(error, weights=weights)),
            "episode_error_reduction_percent": float(point[4]) if np.isfinite(point[4]) else None,
            "perfect_native_draws_excluded_from_error_reduction": int(perfect.sum()),
            "up": int((delta > 1e-12).sum()), "down": int((delta < -1e-12).sum()),
            "tie": int((np.abs(delta) <= 1e-12).sum()),
            "quality_bin_counts": np.bincount(self.bin[indices], minlength=5).tolist(),
            "class_draw_counts": {str(c): int((classes == c).sum()) for c in ids},
        }
        if bootstrap:
            samples = np.array([values(w[indices] * weights) for w in self.row_weights])
            result["class_macro_gain_ci95_pp"] = interval(samples[:, 2])
            result["mean_episode_gain_ci95_pp"] = interval(samples[:, 3])
            result["episode_error_reduction_ci95_percent"] = interval(samples[:, 4])
            return result, samples
        return result

    def composition_decomposition(self, first, last, strata):
        """Symmetric exact mean-episode decomposition on shared stratum support."""
        shared = [s for s in np.unique(strata) if np.any(first & (strata == s)) and np.any(last & (strata == s))]
        support = np.isin(strata, shared)
        first, last = first & support, last & support
        p = np.array([np.mean(strata[first] == s) for s in shared])
        q = np.array([np.mean(strata[last] == s) for s in shared])
        a = np.array([self.delta[first & (strata == s)].mean() for s in shared])
        b = np.array([self.delta[last & (strata == s)].mean() for s in shared])
        composition = 100 * np.sum((q - p) * (a + b) / 2)
        within = 100 * np.sum((p + q) * (b - a) / 2)
        difference = 100 * (self.delta[last].mean() - self.delta[first].mean())
        if not np.isclose(composition + within, difference, atol=1e-12):
            raise ValueError("Composition decomposition failed arithmetic identity")
        target = (p + q) / 2
        return {
            "estimand": "mean episode DeltaIoU; not class-summed macro mIoU",
            "earlier_n_on_shared_support": int(first.sum()), "later_n_on_shared_support": int(last.sum()),
            "shared_strata": len(shared), "later_minus_earlier_pp": float(difference),
            "composition_component_pp": float(composition), "within_stratum_component_pp": float(within),
            "earlier_common_composition_gain_pp": float(100 * target.dot(a)),
            "later_common_composition_gain_pp": float(100 * target.dot(b)),
            "formula": "composition=sum((p_later-p_earlier)*(d_earlier+d_later)/2); within=sum((p_earlier+p_later)/2*(d_later-d_earlier))",
        }

    def class_quality_standardization(self, earlier, later):
        """Reweight shared quality bins within each class to a common distribution.

        Fixed observed weights are descriptive; no adjusted inferential CI is
        asserted. Unions still determine within-class pixel weighting.
        """
        cell = self.classes * len(BIN_NAMES) + self.bin
        shared = [s for s in np.unique(cell) if np.any(earlier & (cell == s)) and np.any(later & (cell == s))]
        support = np.isin(cell, shared)
        masks = (earlier & support, later & support)
        class_ids = np.intersect1d(np.unique(self.classes[masks[0]]), np.unique(self.classes[masks[1]]))
        weights = [np.ones(len(self.rows)), np.ones(len(self.rows))]
        counts = []
        for c in class_ids:
            cells = [s for s in shared if s // len(BIN_NAMES) == c]
            totals = [int((m & (self.classes == c)).sum()) for m in masks]
            for s in cells:
                ns = [int((m & (cell == s)).sum()) for m in masks]
                counts.extend(ns)
                # Equal early/later weighting, normalized separately within class.
                target = .5 * (ns[0] / totals[0] + ns[1] / totals[1])
                for j in (0, 1):
                    weights[j][masks[j] & (cell == s)] = target / (ns[j] / totals[j])
        raw = [self.summary(m, bootstrap=False) for m in masks]
        adjusted = [self.summary(m, bootstrap=False, fixed_weights=w[m]) for m, w in zip(masks, weights)]
        return {
            "estimand": "class-summed macro mIoU on shared class-by-native-IoU-bin support",
            "target": "equal early/later average of observed bin probabilities within each class",
            "shared_cells": len(shared), "classes_retained": len(class_ids),
            "earlier_retained_n": int(masks[0].sum()), "later_retained_n": int(masks[1].sum()),
            "minimum_draw_count_in_shared_cell": min(counts),
            "median_draw_count_in_shared_cell": float(np.median(counts)),
            "raw_earlier_gain_pp": raw[0]["class_macro_gain_pp"],
            "raw_later_gain_pp": raw[1]["class_macro_gain_pp"],
            "raw_later_minus_earlier_pp": raw[1]["class_macro_gain_pp"] - raw[0]["class_macro_gain_pp"],
            "standardized_earlier_gain_pp": adjusted[0]["class_macro_gain_pp"],
            "standardized_later_gain_pp": adjusted[1]["class_macro_gain_pp"],
            "standardized_later_minus_earlier_pp": adjusted[1]["class_macro_gain_pp"] - adjusted[0]["class_macro_gain_pp"],
            "standardized_earlier_native_miou": adjusted[0]["native_class_macro_miou"],
            "standardized_later_native_miou": adjusted[1]["native_class_macro_miou"],
            "caveat": "Post-hoc descriptive reweighting, sparse cells, observed weights fixed, omitted nonoverlap draws; no causal interpretation or adjusted CI.",
        }


def markdown(report):
    overall, compare = report["overall"], report["earlier_later_comparison"]
    def ci(row):
        return "[" + ", ".join(f"{x:.3f}" for x in row["class_macro_gain_ci95_pp"]) + "]"
    lines = [
        "# RCG baseline-quality analysis: all 4,000 sampled draws", "",
        f"Native {overall['native_class_macro_miou']:.6f}, RCG {overall['rcg_class_macro_miou']:.6f}: **+{overall['class_macro_gain_pp']:.6f} pp, 95% CI {ci(overall)}**.",
        f"Mean episode gain is +{overall['mean_episode_gain_pp']:.6f} pp; this is a different estimand.",
        "", "All sampled draws, including the natural repeated episode identity, are retained. Query-GT native-quality bins are diagnostic only. This cohort reuses benchmark/development data, not independent confirmation.",
        "", "Observed gains shrink in the high-quality bins and become negative for native IoU >= .9. The quality-bin and class-frequency composition accounting does not explain the smaller gains in the last two batches; the loss of gain persists within strata. These observations do not identify a causal mechanism.",
        "", "## Fixed native-quality bins", "",
        "| Native episode IoU | Draws | Classes | Native macro | RCG macro | Macro gain [95% CI] | Mean episode gain | Error reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in report["quality_bins"].items():
        lines.append(f"| {label} | {row['n']} | {row['classes']} | {row['native_class_macro_miou']:.3f} | {row['rcg_class_macro_miou']:.3f} | {row['class_macro_gain_pp']:+.3f} {ci(row)} | {row['mean_episode_gain_pp']:+.3f} | {row['episode_error_reduction_percent']:+.2f}% |")
    lines += ["", "## Batch composition and paired gains", "",
              "| Batch | Draws | Native macro | Macro gain [95% CI] | Mean native episode IoU | Mean episode gain | Bins: <.25 / .25-.5 / .5-.75 / .75-.9 / >=.9 |",
              "|---|---:|---:|---:|---:|---:|---|",
              ]
    for label, row in report["batches"].items():
        lines.append(f"| {label} | {row['n']} | {row['native_class_macro_miou']:.3f} | {row['class_macro_gain_pp']:+.3f} {ci(row)} | {row['mean_episode_native_iou_percent']:.3f} | {row['mean_episode_gain_pp']:+.3f} | {' / '.join(map(str, row['quality_bin_counts']))} |")
    lines += ["", "## Earlier versus last two batches", "",
              "Earlier = official0–4 (3,000 draws); later = official5–6 (1,000 draws). This comparison is descriptive and was selected after the batch results were visible.", "",
              "| Group | Native macro | Macro gain [95% CI] | Mean native episode IoU | Mean episode gain | Episode error reduction |",
              "|---|---:|---:|---:|---:|---:|",
              ]
    for label in ("earlier", "later"):
        row = compare[label]
        lines.append(f"| {label} | {row['native_class_macro_miou']:.3f} | {row['class_macro_gain_pp']:+.3f} {ci(row)} | {row['mean_episode_native_iou_percent']:.3f} | {row['mean_episode_gain_pp']:+.3f} | {row['episode_error_reduction_percent']:+.2f}% |")
    diff = compare["later_minus_earlier"]
    lines += ["", f"Later minus earlier macro gain: {diff['class_macro_gain_pp']:+.3f} pp, 95% CI [{diff['class_macro_gain_ci95_pp'][0]:.3f}, {diff['class_macro_gain_ci95_pp'][1]:.3f}].", "",
              "| Native IoU bin | Early / late classes | Earlier mean episode gain | Later mean episode gain | Earlier macro gain | Later macro gain |",
              "|---|---:|---:|---:|---:|---:|",
              ]
    for label, row in compare["same_bin_effects"].items():
        a, b = row["earlier"], row["later"]
        lines.append(f"| {label} | {a['classes']} / {b['classes']} | {a['mean_episode_gain_pp']:+.3f} | {b['mean_episode_gain_pp']:+.3f} | {a['class_macro_gain_pp']:+.3f} | {b['class_macro_gain_pp']:+.3f} |")
    ceiling = compare["constant_error_reduction_reference"]
    lines += ["", f"If the earlier group's aggregate episode error-reduction fraction were unchanged, the later group's headroom would correspond to {ceiling['later_episode_gain_at_earlier_error_reduction_pp']:+.3f} pp, versus the observed {compare['later']['mean_episode_gain_pp']:+.3f}. The headroom change accounts for {ceiling['headroom_only_later_minus_earlier_pp']:+.3f} pp under that descriptive reference, versus the actual {diff['mean_episode_gain_pp']:+.3f} pp episode-mean change. This is arithmetic accounting, not a causal prediction."]
    lines += ["", "Symmetric composition accounting concerns **mean episode gain**, not macro mIoU:", "",
              "| Strata | Shared early / late draws | Later minus earlier | Composition component | Within-stratum component |",
              "|---|---:|---:|---:|---:|",
              ]
    for label, row in compare["composition_decompositions"].items():
        lines.append(f"| {label} | {row['earlier_n_on_shared_support']} / {row['later_n_on_shared_support']} | {row['later_minus_earlier_pp']:+.3f} | {row['composition_component_pp']:+.3f} | {row['within_stratum_component_pp']:+.3f} |")
    standardized = compare["class_quality_standardization"]
    lines += ["", f"Within-class quality standardization retains {standardized['earlier_retained_n']}/3000 earlier and {standardized['later_retained_n']}/1000 later draws, {standardized['classes_retained']} classes and {standardized['shared_cells']} shared cells. Raw overlap macro gains: {standardized['raw_earlier_gain_pp']:+.3f} / {standardized['raw_later_gain_pp']:+.3f}; common-quality gains: {standardized['standardized_earlier_gain_pp']:+.3f} / {standardized['standardized_later_gain_pp']:+.3f}. Minimum shared-cell draw count: {standardized['minimum_draw_count_in_shared_cell']}. These are descriptive weighted I/U ratios; no adjusted inferential CI is asserted.",
              "", "## Interpretation limits", "",
              "- Each batch includes the same 80 classes, equally weighted in the primary macro metric. Class-frequency composition can affect the episode mean, but cannot directly change the primary class weights. The sampled quality, object-size and union composition within classes can still vary.",
              "- Same-bin subgroup macro means average the classes observed in each subgroup; early and late class sets can differ. The joint class-quality standardization uses shared support in all 80 classes.",
              "- Error reduction = 100 × sum(episode DeltaIoU) / sum(1 − native episode IoU), excluding exact native IoU = 1. This ratio of sums avoids exploding per-example ratios near perfection. It is not class-macro error reduction and is not a pixel FP/FN correction rate.",
              "- Negative DeltaIoU-versus-native-IoU correlation is mathematically coupled (native appears in both axes), and positive gain is bounded by 1 − native IoU. It cannot establish causation. Binning on query GT also cannot define an inference-time gate.",
              "- The bootstrap uses the full cohort's connected support/query-photo components, retaining cross-batch and cross-bin photo dependence. Classes absent in a bootstrap replicate are omitted from its macro mean.",
              "- Existing I/U counts reproduce the source report. No native-mask replay, encoder or annotation parity is newly established; I/U alone cannot separate recovered TP, deleted FP, erroneous deletions and new FP.", "",
              ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT / "evidence/local/research_20261005/pipeline_verified"
    parser.add_argument("--input", type=Path, default=default / "frozen_public4000_v1")
    parser.add_argument("--output", type=Path, default=default / "rcg_quality4000_v1")
    args = parser.parse_args()
    start = time.monotonic()
    episode_path, report_path = args.input / "episodes.jsonl", args.input / "report.json"
    source_hashes = {"episodes_jsonl": sha(episode_path), "report_json": sha(report_path)}
    rows = [json.loads(line) for line in episode_path.read_text().splitlines() if line.strip()]
    source = json.loads(report_path.read_text())
    if len(rows) != 4000 or len(rows) != source["n"]:
        raise ValueError("This analysis requires all 4,000 original sampled draws")
    analysis = Analysis(rows)
    all_rows = np.ones(len(rows), dtype=bool)
    overall, _ = analysis.summary(all_rows)
    source_contrast = source["contrasts"]["rcg"]["native"]
    checks = {
        "native_score_absolute_error": abs(overall["native_class_macro_miou"] - source["scores"]["native"]),
        "rcg_score_absolute_error": abs(overall["rcg_class_macro_miou"] - source["scores"]["rcg"]),
        "gain_absolute_error": abs(overall["class_macro_gain_pp"] - source_contrast["gain"]),
        "ci95_max_absolute_error": float(np.max(np.abs(np.array(overall["class_macro_gain_ci95_pp"]) - source_contrast["ci95"]))),
    }
    if max(checks.values()) > 1e-10:
        raise ValueError(f"Source report parity failed: {checks}")
    group_sizes = np.bincount(analysis.groups)
    if len(group_sizes) != source["photo_groups"] or int(group_sizes.max()) != source["largest_photo_group"]:
        raise ValueError("Source connected-photo group structure differs")
    quality = {name: analysis.summary(analysis.bin == j)[0] for j, name in enumerate(BIN_NAMES)}
    batches = {}
    global_bin_delta = np.array([analysis.delta[analysis.bin == j].mean() for j in range(5)])
    global_class_delta = {c: analysis.delta[analysis.classes == c].mean() for c in np.unique(analysis.classes)}
    for batch in sorted(set(analysis.batches)):
        mask = analysis.batches == batch
        row, _ = analysis.summary(mask)
        row["quality_bin_proportions"] = (np.array(row["quality_bin_counts"]) / row["n"]).tolist()
        row["episode_gain_expected_from_pooled_quality_bin_effects_pp"] = float(100 * np.dot(row["quality_bin_proportions"], global_bin_delta))
        row["episode_gain_expected_from_pooled_class_effects_pp"] = float(100 * np.mean([global_class_delta[c] for c in analysis.classes[mask]]))
        row["fold_draw_counts"] = dict(sorted(Counter(str(r["fold"]) for r, selected in zip(rows, mask) if selected).items()))
        for arm, key in (("native", "native_class_macro_miou"), ("rcg", "rcg_class_macro_miou")):
            if abs(row[key] - source["batchs"][batch]["scores"][arm]) > 1e-10:
                raise ValueError(f"Source batch parity failed: {batch}/{arm}")
        batches[batch] = row
    earlier = np.isin(analysis.batches, [f"official{j}" for j in range(5)])
    later = np.isin(analysis.batches, ["official5", "official6"])
    if not np.all(earlier ^ later) or earlier.sum() != 3000 or later.sum() != 1000:
        raise ValueError("Unexpected batch membership")
    a, a_samples = analysis.summary(earlier)
    b, b_samples = analysis.summary(later)
    comparison = {
        "definition": "Post-hoc comparison: earlier official0-4 versus later official5-6",
        "earlier": a, "later": b,
        "later_minus_earlier": {
            "class_macro_gain_pp": b["class_macro_gain_pp"] - a["class_macro_gain_pp"],
            "class_macro_gain_ci95_pp": interval(b_samples[:, 2] - a_samples[:, 2]),
            "mean_episode_gain_pp": b["mean_episode_gain_pp"] - a["mean_episode_gain_pp"],
            "mean_episode_gain_ci95_pp": interval(b_samples[:, 3] - a_samples[:, 3]),
        },
        "constant_error_reduction_reference": {
            "description": "Apply earlier aggregate episode error-reduction fraction to later mean residual error; descriptive arithmetic, not causal prediction",
            "later_episode_gain_at_earlier_error_reduction_pp": a["mean_episode_gain_pp"] * b["mean_episode_ceiling_headroom_pp"] / a["mean_episode_ceiling_headroom_pp"],
            "headroom_only_later_minus_earlier_pp": a["mean_episode_gain_pp"] * (b["mean_episode_ceiling_headroom_pp"] / a["mean_episode_ceiling_headroom_pp"] - 1),
        },
        "same_bin_effects": {name: {"earlier": analysis.summary(earlier & (analysis.bin == j))[0], "later": analysis.summary(later & (analysis.bin == j))[0]} for j, name in enumerate(BIN_NAMES)},
        "composition_decompositions": {
            "quality_bin": analysis.composition_decomposition(earlier, later, analysis.bin),
            "class": analysis.composition_decomposition(earlier, later, analysis.classes),
            "class_by_quality_bin": analysis.composition_decomposition(earlier, later, analysis.classes * 5 + analysis.bin),
        },
        "class_quality_standardization": analysis.class_quality_standardization(earlier, later),
    }
    folds = {str(fold): analysis.summary(np.array([r["fold"] == fold for r in rows]))[0] for fold in sorted(set(r["fold"] for r in rows))}
    identities = [(r["c"], Path(r["support"]).name, Path(r["query"]).name) for r in rows]
    report = {
        "question": "Does RCG gain shrink at high per-draw native IoU, and do native quality/ceiling/class mix account for the last batches' smaller gains?",
        "source": {"input": str(args.input), "hashes": source_hashes, "script_sha256": sha(Path(__file__)), "experiment_statistics_sha256": sha(ROOT / "src/ics/experiment.py")},
        "protocol": {"dataset": "COCO-20i public benchmark reuse", "split": "4 folds, existing sampled test draws", "seed": 0, "resolution": "1024 working pixels per current project record; not newly replayed", "arms": ["native", "rcg"], "sampling": "All 4,000 original sampled draws; no deduplication", "exposure": source["exposure"], "native_quality_bin_edges": BINS.tolist(), "bootstrap_draws": DRAW_COUNT, "rng": "RandomState(0)", "bootstrap_unit": "Full-cohort connected support/query photographs; shared paired weights for all subgroups", "classes_absent_in_bootstrap": "omit absent classes from that replicate's macro mean"},
        "integrity": {"n": len(rows), "unique_episode_identities": len(set(identities)), "natural_repeated_draws": len(rows) - len(set(identities)), "classes": len(set(analysis.classes)), "photo_groups": len(group_sizes), "largest_photo_group": int(group_sizes.max()), "source_report_parity": checks},
        "overall": overall, "quality_bins": quality, "batches": batches, "folds": folds,
        "earlier_later_comparison": comparison,
        "diagnostic_correlation": {"pearson_delta_vs_native_iou": float(np.corrcoef(analysis.baseline, analysis.delta)[0, 1]), "interpretation": "Descriptive only: mathematically coupled by Delta=RCG-native and constrained by ceiling. Not causal evidence."},
        "error_reduction_definition": "100*sum(DeltaIoU)/sum(1-nativeIoU), excluding exact perfect native draws; ratio of sums, not mean of per-draw ratios or class-macro normalization",
        "limits": ["Ground-truth baseline-quality strata are post-hoc diagnostics, not deployable selection.", "All batches use equal weight for the same 80 classes in primary class macro; class frequency directly affects only episode-mean composition.", "Shared class-quality support and fixed reweighting are descriptive, with sparse cells and omitted nonoverlap draws.", "Full I/U counts alone cannot identify add/delete TP/FP mechanisms.", "No new native mask, backbone/annotation parity or independent confirmation is established."],
    }
    if sha(episode_path) != source_hashes["episodes_jsonl"] or sha(report_path) != source_hashes["report_json"]:
        raise ValueError("Source input changed during analysis")
    args.output.mkdir(parents=True, exist_ok=True)
    draw_records = []
    for j, row in enumerate(rows):
        draw_records.append({"draw_index": j, "key": row["key"], "batch": row["batch"], "fold": row["fold"], "class": row["c"], "query": row["query"], "support": row["support"], "photo_group": int(analysis.groups[j]), "native_I": int(analysis.native[j, 0]), "native_U": int(analysis.native[j, 1]), "rcg_I": int(analysis.rcg[j, 0]), "rcg_U": int(analysis.rcg[j, 1]), "native_iou": float(analysis.baseline[j]), "delta_iou": float(analysis.delta[j]), "ceiling_headroom": float(1 - analysis.baseline[j]), "quality_bin": BIN_NAMES[analysis.bin[j]]})
    write_tsv(args.output / "draw_metrics.tsv", draw_records, list(draw_records[0]))
    for filename, mapping in (("quality_bins.tsv", quality), ("batches.tsv", batches)):
        records = [{"label": label, **row, "ci95_low": row["class_macro_gain_ci95_pp"][0], "ci95_high": row["class_macro_gain_ci95_pp"][1]} for label, row in mapping.items()]
        write_tsv(args.output / filename, records, ["label", "n", "classes", "native_class_macro_miou", "rcg_class_macro_miou", "class_macro_gain_pp", "ci95_low", "ci95_high", "mean_episode_native_iou_percent", "mean_episode_gain_pp", "episode_error_reduction_percent"])
    report["runtime_seconds"] = time.monotonic() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "report.md").write_text(markdown(report))
    print(json.dumps({"output": str(args.output), "runtime_seconds": report["runtime_seconds"], "overall_gain_pp": overall["class_macro_gain_pp"], "overall_ci95_pp": overall["class_macro_gain_ci95_pp"], "source_parity": checks, "later_minus_earlier": comparison["later_minus_earlier"]}, indent=2))


if __name__ == "__main__":
    main()
