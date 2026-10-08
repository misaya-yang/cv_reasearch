#!/usr/bin/env python3
"""Measure deployable native-component boundary signal on saved DEV241 packets.

Candidate edges are selected only from the native mask area-downsampled to the score grid.
Ground truth is consulted afterwards to label TP-FP seams versus TP-TP / FP-FP interior edges.
No model, threshold, multilevel graph, cluster labels, or mIoU correction is fitted or invented.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

SEED_SHUFFLE = 31052
SEED_BOOTSTRAP = 0
BOOTSTRAP_DRAWS = 2000
MEAN_BIN_WIDTH = 0.1
JUMP_BIN_WIDTH = 0.05


def unpack_square(bits: np.ndarray, name: str) -> np.ndarray:
    n = int(bits.size) * 8
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"{name}: packed bit count {n} is not a square")
    return np.unpackbits(bits)[:n].reshape(side, side).astype(bool, copy=False)


def area_to_score_grid(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    H, W = mask.shape
    if H % h or W % w:
        raise ValueError(f"native mask shape {mask.shape} does not tile score grid {shape}")
    sh, sw = H // h, W // w
    return mask.reshape(h, sh, w, sw).mean(axis=(1, 3), dtype=np.float64)


def label_components4(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-connected native foreground components; this uses no truth or feature values."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes = [0]
    component = 0
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        component += 1
        labels[y, x] = component
        q = deque([(int(y), int(x))])
        size = 0
        while q:
            cy, cx = q.popleft()
            size += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = component
                    q.append((ny, nx))
        sizes.append(size)
    return labels, np.asarray(sizes, dtype=np.int32)


def candidate_edges(packet: dict[str, np.ndarray], native_patch: np.ndarray,
                    truth_coverage: np.ndarray, component_labels: np.ndarray,
                    component_sizes: np.ndarray) -> dict[str, np.ndarray]:
    """Build all native-only adjacent edge rows, then attach descriptive truth labels.

    Each row retains endpoint coordinates/scores, orientation-matched aff_r/d and sig_*,
    and native component area. Raw rows stay in memory for this packet; the compact result
    JSON records aggregates and can be reproduced from the source packets plus this script.
    """
    h, w = native_patch.shape
    parts: list[dict[str, np.ndarray]] = []
    specs = (
        ("right", native_patch[:, :-1] & native_patch[:, 1:],
         (np.zeros((h, w - 1), dtype=np.int16), np.arange(w - 1, dtype=np.int16)[None, :]),
         ("aff_r", "sig_right", "sig_left"), 0, 1),
        ("down", native_patch[:-1, :] & native_patch[1:, :],
         (np.arange(h - 1, dtype=np.int16)[:, None], np.zeros((h - 1, w), dtype=np.int16)),
         ("aff_d", "sig_down", "sig_up"), 1, 0),
    )
    for direction, eligible, _, (aff_key, sig_a_key, sig_b_key), dy, dx in specs:
        y, x = np.nonzero(eligible)
        if not len(y):
            continue
        y1, x1 = y + dy, x + dx
        lab = component_labels[y, x]
        if np.any(lab != component_labels[y1, x1]) or np.any(lab == 0):
            raise AssertionError("adjacent native foreground edge crossed component labels")
        parts.append(dict(
            direction=np.full(len(y), direction),
            coords=np.column_stack((y, x, y1, x1)).astype(np.int16),
            score_a=packet["score_norm"][y, x], score_b=packet["score_norm"][y1, x1],
            affinity=packet[aff_key][y, x].astype(np.float64),
            sig_a=packet[sig_a_key][y, x].astype(np.float64),
            sig_b=packet[sig_b_key][y, x].astype(np.float64),
            component_area=component_sizes[lab].astype(np.int32),
            truth_a=(truth_coverage[y, x] >= 0.5),
            truth_b=(truth_coverage[y1, x1] >= 0.5),
            mixed_a=((truth_coverage[y, x] > 0.0) & (truth_coverage[y, x] < 1.0)),
            mixed_b=((truth_coverage[y1, x1] > 0.0) & (truth_coverage[y1, x1] < 1.0)),
        ))
    if not parts:
        return {k: np.empty((0, 4), np.int16) if k == "coords" else np.empty(0)
                for k in ("direction", "coords", "score_a", "score_b", "affinity", "sig_a", "sig_b",
                          "component_area", "truth_a", "truth_b", "mixed_a", "mixed_b")}
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def conditional_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Tie-aware ROC AUC (positive=TP-FP seam), using the Mann-Whitney definition."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    wins = 0.0
    negatives_below = 0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        group_pos = int(sorted_labels[i:j].sum())
        group_neg = (j - i) - group_pos
        wins += group_pos * (negatives_below + 0.5 * group_neg)
        negatives_below += group_neg
        i = j
    return float(wins / (len(positive) * len(negative)))


def conditional_auc_for_episode(edges: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    """Compute one fixed signal against three nested error-class contrasts.

    Affinity is shuffled once per nonempty episode/bin, then reused for the combined,
    TP-TP-only, and FP-FP-only negative sets. This is a label-stratified diagnostic,
    not three candidate methods.
    """
    n = len(edges["affinity"])
    negative_sets = {
        "all_sameclass_internal": edges["truth_a"] == edges["truth_b"],
        "tp_tp_internal": edges["truth_a"] & edges["truth_b"],
        "fp_fp_internal": ~edges["truth_a"] & ~edges["truth_b"],
    }
    seam = edges["truth_a"] != edges["truth_b"]
    if n == 0:
        empty = dict(arms={"negative_affinity": None, "score_pair_abs_difference": None,
                          "episode_bin_affinity_shuffle": None},
                     coverage=dict(total_nonempty=0, eligible=0, positive_only=0, negative_only=0,
                                   eligible_edges=0, uncovered_edges=0, eligible_pairs=0,
                                   positive_only_edges=0, negative_only_edges=0,
                                   candidate_edges_in_comparison=0, nonfinite_affinity_edges=0))
        return {"comparisons": {name: dict(empty) for name in negative_sets}}

    mean_score = (edges["score_a"] + edges["score_b"]) / 2.0
    jump = np.abs(edges["score_a"] - edges["score_b"])
    mean_bin = np.minimum(np.floor(mean_score / MEAN_BIN_WIDTH).astype(int), 9)
    jump_bin = np.minimum(np.floor(jump / JUMP_BIN_WIDTH).astype(int), 19)
    bin_id = mean_bin * 20 + jump_bin
    affinity_signal = -edges["affinity"].astype(np.float64)
    score_signal = jump.astype(np.float64)
    finite = np.isfinite(affinity_signal) & np.isfinite(score_signal)
    acc = {}
    for name, negative in negative_sets.items():
        compared = seam | negative
        acc[name] = dict(sums={"affinity": 0.0, "score_pair": 0.0, "shuffle": 0.0},
                         pairs=0,
                         coverage=dict(total_nonempty=0, eligible=0, positive_only=0, negative_only=0,
                                       eligible_edges=0, uncovered_edges=0, eligible_pairs=0,
                                       positive_only_edges=0, negative_only_edges=0,
                                       candidate_edges_in_comparison=int(compared.sum()),
                                       nonfinite_affinity_edges=int((compared & ~finite).sum())))

    for b in np.unique(bin_id):
        idx = (bin_id == b) & finite
        if not idx.any():
            continue
        shuffled = affinity_signal[idx].copy()
        rng.shuffle(shuffled)  # exactly one within-bin permutation, reused by every label contrast
        for name, negative in negative_sets.items():
            compared = seam[idx] | negative[idx]
            if not compared.any():
                continue
            y = seam[idx][compared]
            p, q = int(y.sum()), int((~y).sum())
            cov = acc[name]["coverage"]
            cov["total_nonempty"] += 1
            if p and q:
                cov["eligible"] += 1
                pair_weight = p * q
                cov["eligible_edges"] += p + q
                cov["eligible_pairs"] += pair_weight
                acc[name]["pairs"] += pair_weight
                acc[name]["sums"]["affinity"] += conditional_auc(affinity_signal[idx][compared], y) * pair_weight
                acc[name]["sums"]["score_pair"] += conditional_auc(score_signal[idx][compared], y) * pair_weight
                acc[name]["sums"]["shuffle"] += conditional_auc(shuffled[compared], y) * pair_weight
            elif p:
                cov["positive_only"] += 1
                cov["positive_only_edges"] += p
                cov["uncovered_edges"] += p
            else:
                cov["negative_only"] += 1
                cov["negative_only_edges"] += q
                cov["uncovered_edges"] += q

    comparisons = {}
    for name, state in acc.items():
        cov = state["coverage"]
        # Nonfinite affinities stay in the candidate denominator but cannot enter an AUC.
        cov["uncovered_edges"] += cov["nonfinite_affinity_edges"]
        denom = state["pairs"]
        comparisons[name] = dict(
            arms={"negative_affinity": state["sums"]["affinity"] / denom if denom else None,
                  "score_pair_abs_difference": state["sums"]["score_pair"] / denom if denom else None,
                  "episode_bin_affinity_shuffle": state["sums"]["shuffle"] / denom if denom else None},
            coverage=cov,
        )
    return {"comparisons": comparisons}


def episode_area_cost(native_patch: np.ndarray, truth_coverage: np.ndarray,
                      component_labels: np.ndarray, component_sizes: np.ndarray,
                      pixel_equiv_per_patch: float) -> dict:
    gt_patch = truth_coverage >= 0.5
    ids = np.unique(component_labels[native_patch])
    ids = ids[ids > 0]
    attached_ids = [int(cid) for cid in ids if np.any(gt_patch[component_labels == cid])]
    attached = np.isin(component_labels, attached_ids) if attached_ids else np.zeros_like(native_patch)
    attached_fp = float((1.0 - truth_coverage[attached]).sum())
    attached_tp = float(truth_coverage[attached].sum())
    delete = attached & ~gt_patch
    delete_fp = float((1.0 - truth_coverage[delete]).sum())
    delete_tp_lost = float(truth_coverage[delete].sum())
    mixed_candidate = native_patch & (truth_coverage > 0.0) & (truth_coverage < 1.0)
    candidate_count = int(native_patch.sum())
    mixed_count = int(mixed_candidate.sum())
    return dict(
        native_components=int(len(ids)), attached_components=int(len(attached_ids)),
        candidate_native_patches=candidate_count,
        mixed_candidate_patches=mixed_count,
        mixed_candidate_patch_fraction=(mixed_count / candidate_count if candidate_count else None),
        attached_candidate_patches=int(attached.sum()),
        attached_fp_patch_equiv=attached_fp,
        attached_tp_patch_equiv=attached_tp,
        attached_fp_native_pixel_equiv=attached_fp * pixel_equiv_per_patch,
        attached_tp_native_pixel_equiv=attached_tp * pixel_equiv_per_patch,
        label_oracle_delete_majority_fp_patches=int(delete.sum()),
        label_oracle_fp_mass_removed_patch_equiv=delete_fp,
        label_oracle_tp_mass_lost_patch_equiv=delete_tp_lost,
        label_oracle_fp_mass_removed_native_pixel_equiv=delete_fp * pixel_equiv_per_patch,
        label_oracle_tp_mass_lost_native_pixel_equiv=delete_tp_lost * pixel_equiv_per_patch,
        candidate_component_area_mean=(float(component_sizes[ids].mean()) if len(ids) else None),
    )


def union_find_image_groups(episodes: list[dict]) -> tuple[list[str], int]:
    parent = list(range(len(episodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owner: dict[str, int] = {}
    for i, row in enumerate(episodes):
        for image in (row["support"], row["query"]):
            if image in owner:
                union(i, owner[image])
            else:
                owner[image] = i
    roots = [find(i) for i in range(len(episodes))]
    sorted_roots = {r: n for n, r in enumerate(sorted(set(roots), key=lambda x: min(i for i, q in enumerate(roots) if q == x)))}
    return [f"image_group_{sorted_roots[r]:03d}" for r in roots], len(sorted_roots)


def mean_or_none(values) -> float | None:
    x = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return float(x.mean()) if len(x) else None


def ci_bootstrap_grouped(deltas: np.ndarray, valid: np.ndarray, group_index: np.ndarray,
                         group_count: int, draws: int = BOOTSTRAP_DRAWS,
                         seed: int = SEED_BOOTSTRAP) -> list[float] | None:
    if not valid.any() or group_count == 0:
        return None
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(group_count, np.full(group_count, 1.0 / group_count), size=draws)
    weights = multiplicities[:, group_index[valid]].astype(np.float64)
    denom = weights.sum(axis=1)
    # Explicit elementwise summation avoids platform BLAS warnings for sparse zero weights.
    numer = np.sum(weights * deltas[valid][None, :], axis=1)
    means = numer[denom > 0] / denom[denom > 0]
    if not len(means):
        return None
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def _episode_arms(row: dict, comparison: str) -> dict:
    if comparison == "all_sameclass_internal":
        return row["auc"]
    return row["auc_by_negative_group"][comparison]["arms"]


def paired_metric_summary(rows: list[dict], name_a: str, name_b: str,
                          group_index: np.ndarray, group_count: int,
                          draws: int = BOOTSTRAP_DRAWS, seed: int = SEED_BOOTSTRAP,
                          comparison: str = "all_sameclass_internal") -> dict:
    a = np.asarray([np.nan if _episode_arms(r, comparison)[name_a] is None
                    else _episode_arms(r, comparison)[name_a] for r in rows], dtype=np.float64)
    b = np.asarray([np.nan if _episode_arms(r, comparison)[name_b] is None
                    else _episode_arms(r, comparison)[name_b] for r in rows], dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    d = a[valid] - b[valid]
    return dict(
        n_episodes=int(valid.sum()), n_image_groups=int(len(set(group_index[valid]))),
        mean_a=(float(a[valid].mean()) if valid.any() else None),
        mean_b=(float(b[valid].mean()) if valid.any() else None),
        mean_paired_difference=(float(d.mean()) if len(d) else None),
        paired_image_group_bootstrap95=ci_bootstrap_grouped(a - b, valid, group_index,
                                                             group_count, draws, seed),
        episodes_up=int((d > 1e-12).sum()), episodes_down=int((d < -1e-12).sum()),
        episodes_tied=int((np.abs(d) <= 1e-12).sum()),
    )


def coverage_summary(rows: list[dict], comparison: str = "all_sameclass_internal") -> dict:
    totals = defaultdict(int)
    for row in rows:
        b = (row["auc_coverage"] if comparison == "all_sameclass_internal"
             else row["auc_by_negative_group"][comparison]["coverage"])
        for k in ("total_nonempty", "eligible", "positive_only", "negative_only",
                  "eligible_edges", "uncovered_edges", "eligible_pairs",
                  "positive_only_edges", "negative_only_edges", "candidate_edges_in_comparison",
                  "nonfinite_affinity_edges"):
            totals[k] += b[k]
    n = len(rows)
    has_edge = sum(r["candidate_edge_count"] > 0 for r in rows)
    has_auc = sum(_episode_arms(r, comparison)["negative_affinity"] is not None for r in rows)
    totals.update(episodes=n, episodes_with_candidate_edges=has_edge,
                  episodes_with_conditional_auc=has_auc,
                  episodes_without_conditional_auc=n - has_auc,
                  total_candidate_edges=totals["candidate_edges_in_comparison"])
    totals["edge_coverage_fraction"] = (totals["eligible_edges"] / totals["total_candidate_edges"]
                                        if totals["total_candidate_edges"] else None)
    return dict(totals)


def grouped_area_summary(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out = {}
    for key in keys:
        vals = np.asarray([r["area_cost"][key] for r in rows if r["area_cost"][key] is not None],
                          dtype=np.float64)
        out[key] = dict(mean=(float(vals.mean()) if len(vals) else None),
                        median=(float(np.median(vals)) if len(vals) else None),
                        sum=(float(vals.sum()) if len(vals) else None))
    return out


def validate_inputs(input_root: Path, episodes: list[dict], packets_root: Path) -> dict:
    report_path = input_root / "run" / "report.json"
    run_report = json.loads(report_path.read_text()) if report_path.exists() else None
    if len(episodes) != 241:
        raise ValueError(f"expected DEV241 metadata, got {len(episodes)} episodes")
    keys = [(r["fold"], r["e"], r["c"]) for r in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError("episode metadata keys are not unique")
    missing = [f"{f}_{e}_{c}.npz" for f, e, c in keys if not (packets_root / f"{f}_{e}_{c}.npz").is_file()]
    packet_files = list(packets_root.glob("*.npz"))
    if missing or len(packet_files) != len(episodes):
        raise ValueError(f"packet/manifest mismatch: missing={len(missing)}, packets={len(packet_files)}")
    expected_iu_count = sum("expect_iu" in r for r in episodes)
    expected_original_iu_count = sum("expect_original_iu" in r for r in episodes)
    if expected_iu_count == 0:
        raise ValueError("manifest has no per-episode native IU metadata to verify")
    return {"run_report_path": str(report_path), "run_report": run_report,
            "packet_count": len(packet_files), "episode_keys_unique": True,
            "expected_iu_count": expected_iu_count,
            "expected_original_iu_count": expected_original_iu_count}


def process_episode(row: dict, packet_path: Path, rng_shuffle: np.random.Generator) -> dict:
    with np.load(packet_path, allow_pickle=False) as z:
        required = ("score", "aff_r", "aff_d", "truth", "native", "sig_right", "sig_left", "sig_down", "sig_up")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"{packet_path.name} missing fields {missing}")
        score = np.asarray(z["score"], dtype=np.float32)
        if score.ndim != 2:
            raise ValueError(f"{packet_path.name}: score must be 2-D")
        h, w = score.shape
        aff_r, aff_d = np.asarray(z["aff_r"]), np.asarray(z["aff_d"])
        if aff_r.shape != (h, w - 1) or aff_d.shape != (h - 1, w):
            raise ValueError(f"{packet_path.name}: unexpected aff_r/d shapes {aff_r.shape}/{aff_d.shape}")
        for key, expected in (("sig_right", (h, w - 1)), ("sig_left", (h, w - 1)),
                              ("sig_down", (h - 1, w)), ("sig_up", (h - 1, w))):
            if z[key].shape != expected:
                raise ValueError(f"{packet_path.name}: unexpected {key} shape {z[key].shape}")
        truth_full = unpack_square(np.asarray(z["truth"]), "truth")
        native_full = unpack_square(np.asarray(z["native"]), "native")
        if truth_full.shape != native_full.shape:
            raise ValueError(f"{packet_path.name}: truth/native unpacked shape mismatch")
        native_i = int(np.count_nonzero(truth_full & native_full))
        native_u = int(np.count_nonzero(truth_full | native_full))
        expected_iu = row.get("expect_iu")
        native_iu_match = None
        if expected_iu is not None:
            native_iu_match = [native_i, native_u] == [int(x) for x in expected_iu]
            if not native_iu_match:
                raise ValueError(f"{packet_path.name}: decoded native IU {[native_i, native_u]} != manifest {expected_iu}")
        native_coverage = area_to_score_grid(native_full, (h, w))
        truth_coverage = area_to_score_grid(truth_full, (h, w))
        native_patch = native_coverage >= 0.5
        score_min, score_max = float(score.min()), float(score.max())
        score_range = score_max - score_min
        score_norm = ((score.astype(np.float64) - score_min) / score_range
                      if score_range > 0 else np.zeros_like(score, dtype=np.float64))
        score_norm = np.clip(score_norm, 0.0, 1.0)
        packet = {k: np.asarray(z[k]) for k in required if k not in ("truth", "native")}
        packet["score_norm"] = score_norm
        component_labels, component_sizes = label_components4(native_patch)
        edges = candidate_edges(packet, native_patch, truth_coverage, component_labels, component_sizes)

    seam = edges["truth_a"] != edges["truth_b"]
    tp_tp = edges["truth_a"] & edges["truth_b"]
    fp_fp = ~edges["truth_a"] & ~edges["truth_b"]
    edge_mixed = edges["mixed_a"] | edges["mixed_b"]
    auc_info = conditional_auc_for_episode(edges, rng_shuffle)
    auc_by_group = auc_info["comparisons"]
    primary_auc = auc_by_group["all_sameclass_internal"]
    auc_record = primary_auc["arms"]
    patch_pixels = (truth_full.shape[0] * truth_full.shape[1]) / (h * w)
    area_cost = episode_area_cost(native_patch, truth_coverage, component_labels, component_sizes, patch_pixels)
    return dict(
        fold=int(row["fold"]), e=int(row["e"]), c=int(row["c"]), support=row["support"], query=row["query"],
        packet=packet_path.name, native_patch_frame_shape=list(native_full.shape), score_grid_shape=[h, w],
        decoded_native_iu=[native_i, native_u], manifest_native_iu=row.get("expect_iu"),
        manifest_native_iu_match=native_iu_match,
        candidate_edge_count=int(len(edges["affinity"])),
        candidate_edges_by_orientation={"right": int(np.count_nonzero(edges["direction"] == "right")),
                                        "down": int(np.count_nonzero(edges["direction"] == "down"))},
        edge_types={"tp_fp_seam": int(seam.sum()), "tp_tp_internal": int(tp_tp.sum()),
                    "fp_fp_internal": int(fp_fp.sum())},
        mixed_candidate_endpoint_edges=int(edge_mixed.sum()),
        mixed_candidate_endpoint_edge_fraction=(float(edge_mixed.mean()) if len(edge_mixed) else None),
        score_range=[score_min, score_max],
        auc=auc_record, auc_coverage=primary_auc["coverage"], auc_by_negative_group=auc_by_group,
        nonfinite_affinity_edges=primary_auc["coverage"]["nonfinite_affinity_edges"], area_cost=area_cost)


def mean_auc(rows: list[dict], arm: str, comparison: str = "all_sameclass_internal") -> float | None:
    return mean_or_none([_episode_arms(r, comparison)[arm] for r in rows])


def paired_metric_summary_no_ci(rows: list[dict], name_a: str, name_b: str,
                                comparison: str = "all_sameclass_internal") -> dict:
    a = np.asarray([np.nan if _episode_arms(r, comparison)[name_a] is None
                    else _episode_arms(r, comparison)[name_a] for r in rows])
    b = np.asarray([np.nan if _episode_arms(r, comparison)[name_b] is None
                    else _episode_arms(r, comparison)[name_b] for r in rows])
    valid = np.isfinite(a) & np.isfinite(b)
    d = a[valid] - b[valid]
    return dict(n_episodes=int(valid.sum()), mean_difference=(float(d.mean()) if len(d) else None),
                episodes_up=int((d > 1e-12).sum()), episodes_down=int((d < -1e-12).sum()),
                episodes_tied=int((np.abs(d) <= 1e-12).sum()))


def aggregate_by_fold(rows: list[dict]) -> dict:
    result = {}
    for fold in sorted({r["fold"] for r in rows}):
        subset = [r for r in rows if r["fold"] == fold]
        comparisons = {}
        for comparison in ("tp_tp_internal", "fp_fp_internal"):
            comparisons[comparison] = dict(
                conditional_auc={name: mean_auc(subset, name, comparison) for name in
                                 ("negative_affinity", "score_pair_abs_difference", "episode_bin_affinity_shuffle")},
                affinity_minus_score_pair=paired_metric_summary_no_ci(
                    subset, "negative_affinity", "score_pair_abs_difference", comparison),
                affinity_minus_shuffle=paired_metric_summary_no_ci(
                    subset, "negative_affinity", "episode_bin_affinity_shuffle", comparison),
                coverage=coverage_summary(subset, comparison))
        result[str(fold)] = dict(
            episodes=len(subset),
            conditional_auc={name: mean_auc(subset, name) for name in
                             ("negative_affinity", "score_pair_abs_difference", "episode_bin_affinity_shuffle")},
            affinity_minus_score_pair=paired_metric_summary_no_ci(
                subset, "negative_affinity", "score_pair_abs_difference"),
            affinity_minus_shuffle=paired_metric_summary_no_ci(
                subset, "negative_affinity", "episode_bin_affinity_shuffle"),
            negative_group_contrasts=comparisons,
            coverage=coverage_summary(subset),
            edge_types={kind: sum(r["edge_types"][kind] for r in subset)
                        for kind in ("tp_fp_seam", "tp_tp_internal", "fp_fp_internal")},
            mixed_candidate_patches=grouped_area_summary(subset, ("mixed_candidate_patches", "candidate_native_patches")),
            area_cost=grouped_area_summary(subset, ("attached_fp_native_pixel_equiv", "attached_tp_native_pixel_equiv",
                                                     "label_oracle_fp_mass_removed_native_pixel_equiv",
                                                     "label_oracle_tp_mass_lost_native_pixel_equiv")))
    return result


def run(input_root: Path, output_path: Path, draws: int) -> dict:
    input_root = input_root.resolve()
    manifest_path = input_root / "episodes.json"
    packets_root = input_root / "run" / "packets"
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]
    input_check = validate_inputs(input_root, episodes, packets_root)
    image_group_ids, image_group_count = union_find_image_groups(episodes)
    group_index = np.asarray([int(g.rsplit("_", 1)[1]) for g in image_group_ids], dtype=np.int32)
    rng_shuffle = np.random.default_rng(SEED_SHUFFLE)
    rows = []
    for row in episodes:
        packet_path = packets_root / f"{row['fold']}_{row['e']}_{row['c']}.npz"
        rec = process_episode(row, packet_path, rng_shuffle)
        rec["image_group_id"] = image_group_ids[len(rows)]
        rows.append(rec)
    # Per-episode IU metadata survives for the original DEV40 rows only; all 241 packets are still decoded.
    report = input_check["run_report"]
    verified_iu_count = sum(r["manifest_native_iu_match"] is True for r in rows)
    unverified_iu_count = sum(r["manifest_native_iu_match"] is None for r in rows)
    auc_names = ("negative_affinity", "score_pair_abs_difference", "episode_bin_affinity_shuffle")
    paired = dict(
        affinity_minus_score_pair=paired_metric_summary(rows, auc_names[0], auc_names[1], group_index,
                                                        image_group_count, draws, SEED_BOOTSTRAP),
        affinity_minus_episode_bin_shuffle=paired_metric_summary(rows, auc_names[0], auc_names[2], group_index,
                                                                 image_group_count, draws, SEED_BOOTSTRAP),
    )
    negative_group_contrasts = {}
    for comparison in ("tp_tp_internal", "fp_fp_internal"):
        negative_group_contrasts[comparison] = dict(
            conditional_auc={name: mean_auc(rows, name, comparison) for name in
                             ("negative_affinity", "score_pair_abs_difference", "episode_bin_affinity_shuffle")},
            paired_auc_differences={
                "affinity_minus_score_pair": paired_metric_summary(
                    rows, "negative_affinity", "score_pair_abs_difference", group_index,
                    image_group_count, draws, SEED_BOOTSTRAP, comparison),
                "affinity_minus_episode_bin_shuffle": paired_metric_summary(
                    rows, "negative_affinity", "episode_bin_affinity_shuffle", group_index,
                    image_group_count, draws, SEED_BOOTSTRAP, comparison)},
            coverage=coverage_summary(rows, comparison))
    report_out = dict(
        state="COMPLETED_CPU_DIAGNOSTIC", dataset="COCO-20i 1-shot DEV241, seed 0",
        question="Does negative native adjacent-patch affinity rank TP-FP seams within fixed score strata, beyond the score-pair control?",
        candidate_domain="All 4-connected native foreground components area-downsampled to the 64x64 score grid at >=0.5; all right/down adjacent native-foreground patch pairs; no GT filtering.",
        truth_labels="Post-hoc only: score-grid GT coverage >=0.5 is positive; <0.5 negative. Mixed patches remain in edge counts and are also reported.",
        units={"primary": "episode-macro conditional ROC AUC", "edge": "unique 4-neighbor patch pair within episode",
               "area": "64x64 score-grid patch equivalent and 1024x1024 native-pixel equivalent"},
        arms={"primary": "negative_affinity = -aff_r for right edges and -aff_d for down edges",
              "same_information_control": "absolute normalized score endpoint difference",
              "conditional_shuffle": "one within-bin shuffle of affinity per episode/bin, RNG seed 31052"},
        score_strata={"normalization": "min-max per episode to [0,1] (constant score map becomes all zero)",
                      "score_mean_bin_width": MEAN_BIN_WIDTH, "endpoint_jump_bin_width": JUMP_BIN_WIDTH,
                      "auc": "tie-aware ROC AUC within bins with both edge classes; bin pair weights n_positive*n_negative; episode AUC macro-averaged in final summaries",
                      "negative_edge_contrasts": ["all same-class internal edges", "TP-TP internal only", "FP-FP internal only"]},
        confidence_interval={"method": "paired bootstrap resampling connected image groups (any shared support/query image links episodes)",
                             "image_groups": image_group_count, "draws": draws, "seed": SEED_BOOTSTRAP},
        fold_episode_counts={str(f): sum(int(r["fold"] == f) for r in episodes) for f in range(4)},
        input_validation={"manifest_path": str(manifest_path), "packet_root": str(packets_root),
                          "manifest_episodes": len(episodes), "packets": input_check["packet_count"],
                          "per_episode_decoded_native_iu_matches_manifest": verified_iu_count,
                          "per_episode_native_iu_without_manifest_expectation": unverified_iu_count,
                          "manifest_native_iu_expectation_count": input_check["expected_iu_count"],
                          "manifest_original_iu_expectation_count": input_check["expected_original_iu_count"],
                          "sample_native_iu": {"packet": rows[0]["packet"], "decoded": rows[0]["decoded_native_iu"],
                                               "manifest": rows[0]["manifest_native_iu"],
                                               "match": rows[0]["manifest_native_iu_match"]},
                          "saved_run_report_native_class_miou": (report["class_miou"]["native"] if report else None),
                          "note": "The 201 non-DEV40 rows lack expected IU in the retained manifest; their masks were decoded and measured, but per-example IU cannot be cross-checked against an absent saved expectation."},
        feature_fields_retained_per_candidate_in_memory=["endpoint coordinates", "score endpoints", "aff_r/d", "orientation-matched sig_*", "native component area", "post-hoc GT coverage labels"],
        not_available_or_not_used=["cross-layer adjacency", "cluster labels beyond native 4-connectivity", "learned head", "deployment threshold", "full-pipeline corrected masks", "method mIoU"],
        overall=dict(
            conditional_auc={name: mean_or_none([r["auc"][field] for r in rows]) for name, field in
                             (("negative_affinity", "negative_affinity"),
                              ("score_pair_abs_difference", "score_pair_abs_difference"),
                              ("episode_bin_affinity_shuffle", "episode_bin_affinity_shuffle"))},
            paired_auc_differences=paired,
            negative_group_contrasts=negative_group_contrasts,
            coverage=coverage_summary(rows),
            edge_type_counts={kind: sum(r["edge_types"][kind] for r in rows)
                              for kind in ("tp_fp_seam", "tp_tp_internal", "fp_fp_internal")},
            mixed_candidate_patch_fraction=(sum(r["area_cost"]["mixed_candidate_patches"] for r in rows) /
                                            max(sum(r["area_cost"]["candidate_native_patches"] for r in rows), 1)),
            mixed_endpoint_edge_fraction=(sum(r["mixed_candidate_endpoint_edges"] for r in rows) /
                                          max(sum(r["candidate_edge_count"] for r in rows), 1)),
            area_cost=grouped_area_summary(rows, ("attached_fp_native_pixel_equiv", "attached_tp_native_pixel_equiv",
                                                  "label_oracle_fp_mass_removed_native_pixel_equiv",
                                                  "label_oracle_tp_mass_lost_native_pixel_equiv",
                                                  "candidate_native_patches", "attached_candidate_patches")),
        ),
        by_fold=aggregate_by_fold(rows),
        episode_records=rows,
        interpretation_limit="This is a DEV241 patch-edge signal/error-cost diagnostic. It is not a deployment predictor, not an independent FoRIS candidate, and not original-resolution full-pipeline mIoU.",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_out, indent=1, allow_nan=False) + "\n")
    return report_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path,
                        help="extent_v1 directory containing episodes.json and run/packets/ (read-only)")
    parser.add_argument("--out", required=True, type=Path, help="small JSON result path")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    result = run(args.input_root, args.out, args.bootstrap_draws)
    print(json.dumps({"state": result["state"], "episodes": result["input_validation"]["manifest_episodes"],
                      "image_groups": result["confidence_interval"]["image_groups"],
                      "overall": result["overall"], "by_fold": result["by_fold"], "out": str(args.out)}, indent=1))


if __name__ == "__main__":
    main()
