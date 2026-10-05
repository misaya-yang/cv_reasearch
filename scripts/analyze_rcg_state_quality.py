#!/usr/bin/env python3
"""CPU-only exact stage and edit accounting of sealed P/N/R/T pixel states.

P=pre-CRF FoRIS, N=complete native, R=complete RCG, T=query truth.
The analysis retains every sampled draw and uses the existing full-cohort
RandomState(0) connected-photo bootstrap. All GT strata are diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from analyze_rcg_quality4000 import Analysis, BIN_NAMES, ROOT, interval
from ics.experiment import sha

BANDS = ("GT_distance_le8", "GT_distance_8to16", "GT_distance_over16")
ACTIONS = ("add_TP", "delete_TP", "delete_FP", "add_FP")
STAGES = ("RCG_minus_pre", "native_minus_pre", "RCG_minus_native")


def decode(counts):
    state = np.arange(16)
    p, n, r, truth = [(state & bit) != 0 for bit in (1, 2, 4, 8)]
    pooled = counts.sum(axis=1)
    arrays = {name: np.stack([pooled[:, mask & truth].sum(axis=1), pooled[:, mask | truth].sum(axis=1)], axis=1) for name, mask in (("pre", p), ("native", n), ("rcg", r))}
    selectors = (r & ~n & truth, n & ~r & truth, n & ~r & ~truth, r & ~n & ~truth)
    events = np.stack([counts[:, :, selector].sum(axis=2) for selector in selectors], axis=2)
    opportunities = np.stack([counts[:, :, selector].sum(axis=2) for selector in (~n & truth, n & truth, n & ~truth, ~n & ~truth)], axis=2)
    if not np.array_equal(arrays["native"][:, 0] + events[:, :, 0].sum(axis=1) - events[:, :, 1].sum(axis=1), arrays["rcg"][:, 0]):
        raise ValueError("Intersection edit accounting failed")
    if not np.array_equal(arrays["native"][:, 1] - events[:, :, 2].sum(axis=1) + events[:, :, 3].sum(axis=1), arrays["rcg"][:, 1]):
        raise ValueError("Union edit accounting failed")
    return arrays, events, opportunities


def class_values(arrays, events, class_index, weights, class_count):
    totals = {}
    scores = {}
    present = np.bincount(class_index, weights=weights, minlength=class_count) > 0
    for name, iu in arrays.items():
        totals[name] = np.stack([np.bincount(class_index, weights=weights * iu[:, j], minlength=class_count) for j in (0, 1)], axis=1)
        scores[name] = 100 * np.mean((totals[name][:, 0] / np.maximum(totals[name][:, 1], 1))[present])
    edit_totals = np.empty((class_count, len(BANDS), len(ACTIONS)))
    for band in range(len(BANDS)):
        for action in range(len(ACTIONS)):
            edit_totals[:, band, action] = np.bincount(class_index, weights=weights * events[:, band, action], minlength=class_count)
    j0 = totals["native"][:, 0] / np.maximum(totals["native"][:, 1], 1)
    u1 = np.maximum(totals["rcg"][:, 1], 1)
    factors = np.stack([np.ones(class_count), -np.ones(class_count), j0, -j0], axis=1)
    # The final RCG union is shared by all terms, so attribution is exact even
    # when native and RCG unions differ. This is accounting, not isolated masks.
    terms = 100 * np.mean((edit_totals * factors[:, None, :] / u1[:, None, None])[present], axis=0)
    gains = np.array([scores["rcg"] - scores["pre"], scores["native"] - scores["pre"], scores["rcg"] - scores["native"]])
    if abs(gains[0] - gains[1] - gains[2]) > 1e-10 or abs(terms.sum() - gains[2]) > 1e-10:
        raise ValueError("Class-macro stage or four-term identity failed")
    return scores, gains, terms


class StateAnalysis:
    def __init__(self, quality, counts, arrays):
        self.quality = quality
        decoded, self.events, self.opportunities = decode(counts)
        for name in ("pre", "native", "rcg"):
            if not np.array_equal(decoded[name], arrays[name]):
                raise ValueError(f"16-state I/U reconstruction differs: {name}")
        self.arrays = arrays

    def summary(self, mask, bootstrap=True):
        indices = np.flatnonzero(mask)
        if not len(indices):
            return {"n": 0}, None
        classes = self.quality.classes[indices]
        ids, class_index = np.unique(classes, return_inverse=True)
        arrays = {name: value[indices] for name, value in self.arrays.items()}
        events, opportunities = self.events[indices], self.opportunities[indices]
        scores, gains, terms = class_values(arrays, events, class_index, np.ones(len(indices)), len(ids))
        totals, denominators = np.zeros((len(ids), 3, 4)), np.zeros((len(ids), 3, 4))
        for j in range(3):
            for k in range(4):
                totals[:, j, k] = np.bincount(class_index, weights=events[:, j, k], minlength=len(ids))
                denominators[:, j, k] = np.bincount(class_index, weights=opportunities[:, j, k], minlength=len(ids))
        rates = {}
        for k, action in enumerate(ACTIONS):
            rates[action] = {}
            for j, band in enumerate((*BANDS, "all_distances")):
                numerator = totals[:, j, k] if j < 3 else totals[:, :, k].sum(axis=1)
                denominator = denominators[:, j, k] if j < 3 else denominators[:, :, k].sum(axis=1)
                eligible = denominator > 0
                rates[action][band] = {"class_balanced_percent": float(100 * np.mean(numerator[eligible] / denominator[eligible])) if eligible.any() else None, "eligible_classes": int(eligible.sum())}
        result = {"n": len(indices), "classes": len(ids), "scores": scores,
                  "stage_gains_pp": {name: float(gains[j]) for j, name in enumerate(STAGES)},
                  "edit_contributions_pp": {action: {"all_distances": float(terms[:, k].sum()), **{band: float(terms[j, k]) for j, band in enumerate(BANDS)}} for k, action in enumerate(ACTIONS)},
                  "class_balanced_opportunity_rates": rates,
                  "stage_identity_absolute_error_pp": float(abs(gains[0] - gains[1] - gains[2])),
                  "edit_identity_absolute_error_pp": float(abs(terms.sum() - gains[2])),
                  "quality_bin_counts": np.bincount(self.quality.bin[indices], minlength=5).tolist()}
        if not bootstrap:
            return result, None
        boot_gains, boot_terms = [], []
        for row_weights in self.quality.row_weights:
            _, g, t = class_values(arrays, events, class_index, row_weights[indices], len(ids))
            boot_gains.append(g)
            boot_terms.append(t)
        samples = {"stage": np.array(boot_gains), "terms": np.array(boot_terms)}
        result["stage_gains_ci95_pp"] = {name: interval(samples["stage"][:, j]) for j, name in enumerate(STAGES)}
        result["edit_contributions_ci95_pp"] = {action: {"all_distances": interval(samples["terms"][:, :, k].sum(axis=1)), **{band: interval(samples["terms"][:, j, k]) for j, band in enumerate(BANDS)}} for k, action in enumerate(ACTIONS)}
        return result, samples


def difference(earlier, later, first_samples, last_samples):
    result = {"stage_gain_changes_pp": {stage: later["stage_gains_pp"][stage] - earlier["stage_gains_pp"][stage] for stage in STAGES},
              "edit_contribution_changes_pp": {action: {band: later["edit_contributions_pp"][action][band] - earlier["edit_contributions_pp"][action][band] for band in (*BANDS, "all_distances")} for action in ACTIONS}}
    result["distance_total_changes_pp"] = {band: sum(result["edit_contribution_changes_pp"][action][band] for action in ACTIONS) for band in BANDS}
    if first_samples is not None and last_samples is not None:
        ds, dt = last_samples["stage"] - first_samples["stage"], last_samples["terms"] - first_samples["terms"]
        result["stage_gain_changes_ci95_pp"] = {stage: interval(ds[:, j]) for j, stage in enumerate(STAGES)}
        result["edit_contribution_changes_ci95_pp"] = {action: {"all_distances": interval(dt[:, :, k].sum(axis=1)), **{band: interval(dt[:, j, k]) for j, band in enumerate(BANDS)}} for k, action in enumerate(ACTIONS)}
        result["distance_total_changes_ci95_pp"] = {band: interval(dt[:, j, :].sum(axis=1)) for j, band in enumerate(BANDS)}
    total = sum(result["edit_contribution_changes_pp"][action]["all_distances"] for action in ACTIONS)
    if abs(total - result["stage_gain_changes_pp"]["RCG_minus_native"]) > 1e-10:
        raise ValueError("Late-minus-early edit attribution identity failed")
    return result


def markdown(report):
    def ci(values):
        return "[" + ", ".join(f"{x:.3f}" for x in values) + "]"
    comparison = report["earlier_later"]
    change = comparison["later_minus_earlier"]
    lines = ["# Exact RCG state and quality accounting: 4,000 sampled draws", "",
             "P = pre-CRF FoRIS mask, N = complete native, R = complete RCG. All results are post-hoc GT diagnostics; all sampled draws and photo dependencies are retained.", "",
             f"The last two batches lose {abs(change['stage_gain_changes_pp']['RCG_minus_native']):.3f} pp of RCG gain relative to the earlier five. The largest negative signed changes are less FP-removal benefit ({change['edit_contribution_changes_pp']['delete_FP']['all_distances']:+.3f} pp) and more TP-deletion harm ({change['edit_contribution_changes_pp']['delete_TP']['all_distances']:+.3f} pp). New-FP harm improves ({change['edit_contribution_changes_pp']['add_FP']['all_distances']:+.3f} pp), partially offsetting them. TP-addition change is small with a CI crossing zero. Native postprocessing gain changes by only {change['stage_gain_changes_pp']['native_minus_pre']:+.3f} pp, also with a CI crossing zero.", "",
             "## Exact stage identity", "",
             "`(R − N) = (R − P) − (N − P)` uses differences of paired class-summed macro mIoU, not pooled pixel IoU. The native-minus-pre term describes native CRF/postprocessing; the RCG-minus-pre term includes RCG's complete post-pre processing.", "",
             "| Group | Draws | Pre | Native | RCG | RCG − pre [95% CI] | Native − pre [95% CI] | RCG − native [95% CI] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in {"all": report["overall"], **report["batches"], "earlier0-4": comparison["earlier"], "later5-6": comparison["later"]}.items():
        entries = [f"{row['stage_gains_pp'][name]:+.3f} {ci(row['stage_gains_ci95_pp'][name])}" for name in STAGES]
        lines.append(f"| {label} | {row['n']} | {row['scores']['pre']:.3f} | {row['scores']['native']:.3f} | {row['scores']['rcg']:.3f} | " + " | ".join(entries) + " |")
    lines += ["", "| Stage gain change: later − earlier | Change [95% CI] |", "|---|---:|"]
    for name in STAGES:
        lines.append(f"| {name} | {change['stage_gain_changes_pp'][name]:+.3f} {ci(change['stage_gain_changes_ci95_pp'][name])} |")
    lines += ["", "## Exact class-balanced four-way attribution", "",
              "For each class, `R − N = (addTP − deleteTP + J_native × (deleteFP − addFP)) / U_RCG`. Sum counts within the group/class before taking each ratio, then average classes ×100. All four terms share the final RCG union; this is an additive identity, not four independently applied masks or causal effects.", "",
              "| Signed contribution | Earlier | Later | Later − earlier [95% CI] |", "|---|---:|---:|---:|"]
    for action in ACTIONS:
        first = comparison["earlier"]["edit_contributions_pp"][action]["all_distances"]
        last = comparison["later"]["edit_contributions_pp"][action]["all_distances"]
        lines.append(f"| {action} | {first:+.3f} | {last:+.3f} | {change['edit_contribution_changes_pp'][action]['all_distances']:+.3f} {ci(change['edit_contribution_changes_ci95_pp'][action]['all_distances'])} |")
    lines += ["", "Each band below uses the same class native IoU and final RCG union as the full identity, so the distance terms sum exactly to the overall attribution. Distances are to the nearest opposite GT label at 1024.", "",
              "| Band | Δ addTP | Δ deleteTP | Δ deleteFP | Δ addFP | Total late − early [95% CI] |", "|---|---:|---:|---:|---:|---:|"]
    for band in BANDS:
        values = [change["edit_contribution_changes_pp"][action][band] for action in ACTIONS]
        lines.append(f"| {band} | " + " | ".join(f"{v:+.3f}" for v in values) + f" | {sum(values):+.3f} {ci(change['distance_total_changes_ci95_pp'][band])} |")
    lines += ["", "## Opportunity-normalized rates", "",
              "Rates average class-specific event/opportunity ratios. addTP/nativeFN = recovery; deleteFP/nativeFP = removal; deleteTP/nativeTP = damage; addFP/nativeTN = new false positives. Each distance band uses matching-band opportunities. Zero-opportunity classes are omitted and eligible-class counts are retained in JSON. These rates have different denominators and are not additive mIoU components.", "",
              "| Event | Earlier all-distance rate | Later all-distance rate |", "|---|---:|---:|"]
    for action in ACTIONS:
        a = comparison["earlier"]["class_balanced_opportunity_rates"][action]["all_distances"]["class_balanced_percent"]
        b = comparison["later"]["class_balanced_opportunity_rates"][action]["all_distances"]["class_balanced_percent"]
        lines.append(f"| {action} | {a:.3f}% | {b:.3f}% |")
    lines += ["", "| GT-distance band | addTP recovery: earlier / later | deleteTP damage: earlier / later | deleteFP removal: earlier / later | addFP damage: earlier / later |", "|---|---:|---:|---:|---:|"]
    for band in BANDS:
        vals = []
        for action in ACTIONS:
            pair = [comparison[group]["class_balanced_opportunity_rates"][action][band]["class_balanced_percent"] for group in ("earlier", "later")]
            vals.append(" / ".join(f"{v:.3f}%" if v is not None else "N/A" for v in pair))
        lines.append(f"| {band} | " + " | ".join(vals) + " |")
    lines += ["", "## Fixed native-quality bins", "",
              "| Native IoU bin | Earlier / later draws | Δ RCG − pre | Δ native − pre | Δ RCG − native | Δ addTP | Δ deleteTP | Δ deleteFP | Δ addFP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["quality_bins"].items():
        d = row["later_minus_earlier"]
        vals = [d["stage_gain_changes_pp"][stage] for stage in STAGES] + [d["edit_contribution_changes_pp"][action]["all_distances"] for action in ACTIONS]
        lines.append(f"| {name} | {row['earlier']['n']} / {row['later']['n']} | " + " | ".join(f"{v:+.3f}" for v in vals) + " |")
    lines += ["", "Different quality subgroups can contain different classes; their macro contrasts are conditional diagnostics. Distance >16 does not prove an object-level effect or whole-object correction. Stage comparisons cannot isolate token evidence from rendering without the corresponding sealed alternatives. Bootstrap CIs are paired full-cohort connected-photo draws; late/early grouping was chosen after observing the batch readout. This is reused benchmark data, not independent confirmation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parent = ROOT / "evidence/local/research_20261005/pipeline_verified"
    parser.add_argument("--states", type=Path, default=parent / "rcg_native_states4000_v1")
    parser.add_argument("--verified", type=Path, default=parent / "frozen_public4000_v1")
    parser.add_argument("--output", type=Path, default=parent / "rcg_quality4000_v1")
    args = parser.parse_args()
    start = time.monotonic()
    paths = {"states": args.states / "states.npz", "manifest": args.states / "manifest.json", "state_source_report": args.states / "report.json", "verified_episodes": args.verified / "episodes.jsonl"}
    hashes = {name: sha(path) for name, path in paths.items()}
    rows = [json.loads(line) for line in paths["verified_episodes"].read_text().splitlines() if line.strip()]
    manifest = json.loads(paths["manifest"].read_text())
    state_source = json.loads(paths["state_source_report"].read_text())
    if state_source["state_bits"] != {"pre": 1, "native": 2, "RCG": 4, "truth": 8}:
        raise ValueError("State bit schema differs")
    if len(rows) != 4000 or len(manifest) != 4000:
        raise ValueError("Require all 4,000 sampled draws in original order")
    for a, b in zip(rows, manifest):
        if any(a[key] != b[key] for key in ("key", "c", "fold", "query", "support")):
            raise ValueError("State manifest does not preserve verified sampled draw order")
    with np.load(paths["states"], allow_pickle=False) as stored:
        counts = stored["counts"].copy()
        arrays = {name: stored[key].copy() for name, key in (("pre", "foris_pre.control"), ("native", "native"), ("rcg", "rcg"))}
    if counts.shape != (4000, 3, 16) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0) or not np.all(counts.sum(axis=(1, 2)) == 1024 ** 2):
        raise ValueError("State counts must exhaust every 1024x1024 image")
    quality = Analysis(rows)
    for name, expected in (("native", quality.native), ("rcg", quality.rcg)):
        if not np.array_equal(arrays[name], expected):
            raise ValueError("State masks' I/U differs from verified 4,000-draw counts")
    analysis = StateAnalysis(quality, counts, arrays)
    all_rows = np.ones(4000, dtype=bool)
    overall, _ = analysis.summary(all_rows)
    source_parity_errors = []
    for name, source_name in (("pre", "foris_pre.control"), ("native", "native"), ("rcg", "rcg")):
        source_parity_errors.append(abs(overall["scores"][name] - state_source["scores"][source_name]))
    for stage, arm, base in (("RCG_minus_pre", "rcg", "foris_pre.control"), ("native_minus_pre", "native", "foris_pre.control"), ("RCG_minus_native", "rcg", "native")):
        source_parity_errors.extend(abs(np.array(overall["stage_gains_ci95_pp"][stage]) - state_source["contrasts"][arm][base]["ci95"]))
    if max(source_parity_errors) > 1e-10:
        raise ValueError("Source stage report point/CI parity failed")
    earlier = np.isin(quality.batches, [f"official{j}" for j in range(5)])
    later = np.isin(quality.batches, ["official5", "official6"])
    first, first_samples = analysis.summary(earlier)
    last, last_samples = analysis.summary(later)
    batches = {batch: analysis.summary(quality.batches == batch)[0] for batch in sorted(set(quality.batches))}
    by_quality = {}
    for j, name in enumerate(BIN_NAMES):
        a, ast = analysis.summary(earlier & (quality.bin == j))
        b, bst = analysis.summary(later & (quality.bin == j))
        by_quality[name] = {"overall": analysis.summary(quality.bin == j)[0], "earlier": a, "later": b, "later_minus_earlier": difference(a, b, ast, bst)}
    comparison = {"earlier": first, "later": last, "later_minus_earlier": difference(first, last, first_samples, last_samples)}
    report = {"source": {"paths": {name: str(path) for name, path in paths.items()}, "sha256": hashes, "script_sha256": sha(Path(__file__)), "quality_analysis_script_sha256": sha(ROOT / "scripts/analyze_rcg_quality4000.py")},
              "protocol": {"n": 4000, "class_macro_classes": 80, "state_bits": {"pre": 1, "native": 2, "rcg": 4, "truth": 8}, "distance_bands": list(BANDS), "draws": 2000, "rng": "RandomState(0)", "bootstrap_unit": "full-cohort connected support/query photographs", "exposure": "benchmark/development reuse; not independent confirmation", "late_grouping": "post-hoc official5-6 versus official0-4"},
              "integrity": {"state_counts_exhaustive": True, "all_three_masks_reconstructed_exactly_from_states": True, "native_rcg_IU_matches_verified4000": True, "all_draws_preserved_in_order": True, "all_stage_and_edit_identities_tolerance_pp": 1e-10, "source_stage_report_score_and_ci_max_error_pp": float(max(source_parity_errors)), "photo_groups": int(quality.groups.max()) + 1},
              "attribution": {"formula": "For class c: (aTP-dTP+J_native*(dFP-aFP))/U_rcg; mean classes times 100", "distance": "Split event numerator by distance using the same class J_native and final U_rcg", "scope": "Exact descriptive decomposition of observed masks, not isolated counterfactual masks or causal components"},
              "overall": overall, "batches": batches, "earlier_later": comparison, "quality_bins": by_quality,
              "opportunity_rate_denominators": {"add_TP": "native FN", "delete_TP": "native TP", "delete_FP": "native FP", "add_FP": "native TN"},
              "limits": ["GT strata are diagnostics only, not deployable gates.", "Rates use distinct opportunity denominators; only signed final-union terms add to macro mIoU.", "Native-minus-pre is the observed native CRF/postprocess effect; RCG-minus-pre does not isolate graph, guide or renderer individually.", "Far GT distance alone does not establish whole-object correction.", "Conditional quality groups can contain different classes."]}
    for name, path in paths.items():
        if sha(path) != hashes[name]:
            raise ValueError("Source changed during analysis")
    report["runtime_seconds"] = time.monotonic() - start
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "state_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "state_report.md").write_text(markdown(report))
    print(json.dumps({"output": str(args.output), "runtime_seconds": report["runtime_seconds"], "earlier_later_stage_changes": comparison["later_minus_earlier"]["stage_gain_changes_pp"], "earlier_later_edit_changes": {action: comparison["later_minus_earlier"]["edit_contribution_changes_pp"][action]["all_distances"] for action in ACTIONS}}, indent=2))


if __name__ == "__main__":
    main()
