"""Independently recount all sealed scene600 binary outputs and class metrics.

No feature extraction, field fitting, segmentation inference or CRF replay.
The seal/config/source identities are checked before any query annotation opens.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from audit_saved_outputs import point_and_intervals

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = REPO.parent / "cv_data/a/autonomous_scene_reconstruction600_20261010"
ARMS = ("scene", "source_kernel", "support_ridge")
CHECKS = Counter()
HASHES = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path).resolve()
    if str(path) not in HASHES:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        HASHES[str(path)] = h.hexdigest()
    return HASHES[str(path)]


def check_sha(path, digest):
    assert sha(path) == digest, (str(path), "SHA mismatch")
    CHECKS["file_SHA"] += 1


def nearest_truth(truth, hw):
    """Exact nearest legacy Torch geometry for integer input sizes ->1024."""
    hw = tuple(hw)
    if truth.shape == hw:
        return truth
    # CLI1024's power-of-two divisor makes these coordinates exact in FP32 too.
    assert hw == (1024, 1024)
    ys = (np.arange(hw[0], dtype=np.int64) * truth.shape[0]) // hw[0]
    xs = (np.arange(hw[1], dtype=np.int64) * truth.shape[1]) // hw[1]
    return truth[ys[:, None], xs[None, :]]


def truth_for(row):
    with Image.open(row["query_mask_path"]) as image:
        gt = (np.asarray(image.convert("L")) > 0).astype(np.uint8)
    h = hashlib.sha256(json.dumps([list(gt.shape), gt.dtype.str]).encode())
    h.update(np.ascontiguousarray(gt).tobytes())
    assert h.hexdigest() == row["query_mask_hash"]
    assert gt.shape == tuple(row["query_size_hw"])
    CHECKS["GT_array_identity"] += 1
    return gt.astype(bool)


def unpack(z, key, hw):
    assert z[key].dtype == np.uint8 and z[key].ndim == 1
    assert len(z[key]) == (int(np.prod(hw)) + 7) // 8, (key, "packed geometry mismatch")
    CHECKS["packed_mask_geometry"] += 1
    return np.unpackbits(z[key], count=int(np.prod(hw))).reshape(hw).astype(bool)


def iu(pred, truth):
    return [int(np.count_nonzero(pred & truth)), int(np.count_nonzero(pred | truth))]


def edits(pred, baseline, truth):
    add, delete = pred & ~baseline, ~pred & baseline
    return [int(np.count_nonzero(add & truth)), int(np.count_nonzero(add & ~truth)),
            int(np.count_nonzero(delete & truth)), int(np.count_nonzero(delete & ~truth))]


def mass64(pixels):
    assert pixels.shape == (1024, 1024)
    return pixels.reshape(64, 16, 64, 16).sum((1, 3), dtype=np.float64).reshape(-1)


def weighted_rank(score, positive, negative):
    """Exact-score ties, with each64 node weighted by its actual pixel masses."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    total_p, total_n = float(positive.sum()), float(negative.sum())
    if total_p <= 0 or total_n <= 0:
        return dict(auc=None, ap=None, positive_pixels=total_p, negative_pixels=total_n,
                    reason="empty foreground or background in this episode ROI")
    order = np.argsort(np.asarray(score).reshape(-1), kind="stable")
    value = np.asarray(score).reshape(-1)[order]
    starts = np.r_[0, np.flatnonzero(value[1:] != value[:-1]) + 1]
    p, n = np.add.reduceat(positive[order], starts), np.add.reduceat(negative[order], starts)
    occupied = p + n > 0
    p, n = p[occupied], n[occupied]
    below = np.cumsum(n) - n
    auc_numerator = float(np.sum(p * (below + .5*n)))
    tp, fp = np.cumsum(p[::-1]), np.cumsum(n[::-1])
    ap = float(np.sum((p[::-1]/total_p) * tp/(tp+fp)))
    sign = np.asarray(score).reshape(-1) > 0
    return dict(auc=auc_numerator/(total_p*total_n), ap=ap,
                positive_pixels=total_p, negative_pixels=total_n,
                positive_mass_at_score_gt0=float(positive[sign].sum()),
                negative_mass_at_score_gt0=float(negative[sign].sum()),
                auc_numerator=auc_numerator, auc_denominator=total_p*total_n)


def rank_episode(score_fields, gt, rois):
    result = {}
    for name, roi in rois.items():
        positive, negative = mass64(roi & gt), mass64(roi & ~gt)
        result[name] = {arm: weighted_rank(score, positive, negative)
                        for arm, score in score_fields.items()}
    return result


def summarize_mechanisms(episodes):
    rank_groups, fits = defaultdict(list), defaultdict(list)
    for episode in episodes:
        ds = episode["dataset"]
        for roi, arms in episode["ranks"].items():
            for arm, record in arms.items():
                rank_groups[(ds, roi, arm)].append(record)
        for arm, fit in episode["source_fit"].items():
            fits[(ds, arm)].append(fit)
    def numeric(values):
        v = np.asarray(values, dtype=np.float64)
        return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                    p10=float(np.quantile(v, .1)), p90=float(np.quantile(v, .9)))
    ranks = {}
    for (ds, roi, arm), rr in rank_groups.items():
        valid = [r for r in rr if r["auc"] is not None]
        value = dict(n=len(rr), valid_episodes=len(valid), invalid_episodes=len(rr)-len(valid),
                     positive_pixels=sum(r["positive_pixels"] for r in rr),
                     negative_pixels=sum(r["negative_pixels"] for r in rr))
        if valid:
            value.update(auc=numeric([r["auc"] for r in valid]), ap=numeric([r["ap"] for r in valid]),
                within_episode_pair_weighted_auc=(sum(r["auc_numerator"] for r in valid)/
                                                  sum(r["auc_denominator"] for r in valid)),
                pooled_positive_mass_at_score_gt0=sum(r["positive_mass_at_score_gt0"] for r in valid),
                pooled_negative_mass_at_score_gt0=sum(r["negative_mass_at_score_gt0"] for r in valid))
        ranks.setdefault(ds, {}).setdefault(roi, {})[arm] = value
    paired = {}
    for ds in {r["dataset"] for r in episodes}:
        subset = [r for r in episodes if r["dataset"] == ds]
        paired[ds] = {}
        for roi in subset[0]["ranks"]:
            rr = [r["ranks"][roi] for r in subset]
            valid = [r for r in rr if all(r[a]["auc"] is not None for a in ARMS)]
            paired[ds][roi] = {
                comparison: dict(auc_delta=numeric([r[a]["auc"]-r[b]["auc"] for r in valid]),
                                 ap_delta=numeric([r[a]["ap"]-r[b]["ap"] for r in valid]))
                for comparison, a, b in (("scene_minus_source_kernel", "scene", "source_kernel"),
                                         ("scene_minus_support_ridge", "scene", "support_ridge"))} if valid else {}
    source_fit = {}
    for (ds, arm), values in fits.items():
        source_fit.setdefault(ds, {})[arm] = {
            key: numeric([r[key] for r in values]) for key in values[0]}
    return dict(ranks=ranks, paired_rank_deltas=paired, source_fit=source_fit,
        scope="Post-seal GT diagnosis. Native64 field scores with exact canonical1024 FG/BG pixel masses inside each fixed ROI; ties indivisible. Not final mIoU, a threshold search or independent confirmation.",
        limits=["Macro statistics average valid episodes; ROI-conditioned valid populations differ.",
                "Same ROI is used for all three field scores in each paired comparison.",
                "Added/deleted ROIs are defined by the named actual saved direct/CRF output relative to FoRIS; they are output-conditioned.",
                "64-node score treats each encoder cell as constant for ranking; deployed direct masks use bilinear1024.",
                "Reference binary-role loss is the supervised fit-subset training loss, not held-out cross-image accuracy.",
                "Scene and source kernels have an unlabelled query/source basis respectively; none of these diagnostics tunes a method."])


def validate_identity():
    # This function must finish before truth_for is reachable.
    assert (RUN / "sealed.json").exists(), "Await full prediction seal; do not read query GT"
    cfg, seal = read(RUN / "config.json"), read(RUN / "sealed.json")
    assert cfg["n"] == seal["n"] == len(seal["receipts"]) == 600
    assert seal["query_GT_read"] is False
    assert all(r["query_GT_reads"] == 0 and r["baseline_mask_reads"] == 0
               and r["encoder_calls"] == 0 and r["raw_writes"] == 0 for r in seal["receipts"])
    check_sha(RUN / "config.json", seal["config_sha256"])
    check_sha(RUN / "manifest.json", cfg["manifest_sha256"])
    check_sha(RUN / "tasks.json", cfg["tasks_sha256"])
    check_sha(cfg["profile_path"], cfg["profile_sha256"])
    check_sha(cfg["basis_path"], cfg["basis_sha256"])
    for relative, digest in cfg["source_sha256"].items():
        check_sha(RUN / "frozen" / relative, digest)
    for path, digest in cfg["external_source_sha256"].items():
        check_sha(path, digest)
    manifest, sources = read(RUN / "manifest.json"), read(RUN / "tasks.json")
    assert len(manifest) == len(sources) == 600
    assert len({r["episode_id"] for r in manifest}) == 600
    for row, source, rec in zip(manifest, sources, seal["receipts"]):
        assert row["episode_id"] == source["episode_id"] == rec["episode_id"]
        check_sha(RUN / "predictions" / rec["filename"], rec["prediction_sha256"])
        check_sha(RUN / "fields" / rec["filename"], rec["fields_sha256"])
    return cfg, seal, manifest, sources


def main():
    start = time.monotonic()
    cfg, seal, manifest, sources = validate_identity()
    print("All600 sealed receipts/config/source/prediction/field SHA verified; query GT scoring permitted", flush=True)
    records, grouped, mechanisms = [], defaultdict(list), []
    field_stats = defaultdict(list)
    for row, source, rec in zip(manifest, sources, seal["receipts"]):
        truth = truth_for(row)
        field_path = RUN / "fields" / rec["filename"]
        with np.load(field_path, allow_pickle=False) as field:
            assert set(field.files) == set(ARMS)
            score_fields = {a: field[a].copy() for a in ARMS}
            for arm in ARMS:
                value = field[arm]
                assert value.shape == (64, 64) and np.isfinite(value).all()
                field_stats[row["dataset"] + "/" + arm].append(
                    [float(value.min()), float(value.max()), float((value > 0).mean())])
                CHECKS["field_shape_and_finiteness"] += 1
        with np.load(RUN / "predictions" / rec["filename"], allow_pickle=False) as z:
            assert tuple(z["original_hw"]) == truth.shape
            expected_keys = {"original_hw"} | {
                f + "/" + a + "." + s for f in ("cli1024", "original")
                for a in ARMS for s in ("direct", "crf")}
            assert set(z.files) == expected_keys
            for frame, hw in (("cli1024", (1024, 1024)), ("original", truth.shape)):
                gt = nearest_truth(truth, hw)
                previous = {}
                for arm, b in source["baselines"].items():
                    check_sha(b["path"], b["sha256"])
                    with np.load(b["path"], allow_pickle=False) as prior:
                        previous[arm] = unpack(prior, b["keys"][frame], hw)
                result = dict(episode_id=row["episode_id"], dataset=row["dataset"], fold=row["fold"],
                    class_id=row["loader_class_id"], query_photo_id=row["query_photo_id"], frame=frame,
                    iu={a: iu(p, gt) for a, p in previous.items()}, edits={})
                CHECKS["baseline_original_or_cli_IU"] += len(previous)
                if frame == "cli1024":
                    rois = {"whole": np.ones(hw, dtype=bool), "FoRIS_FG": previous["foris.crf"]}
                for arm in ARMS:
                    for suffix in ("direct", "crf"):
                        a = arm + "." + suffix
                        pred = unpack(z, frame + "/" + a, hw)
                        result["iu"][a] = iu(pred, gt)
                        e = edits(pred, previous["foris.crf"], gt)
                        result["edits"][a] = e
                        i, u = result["iu"]["foris.crf"]
                        assert result["iu"][a] == [i + e[0] - e[2], u + e[1] - e[3]]
                        CHECKS["candidate_IU_and_edit_identity"] += 1
                        if frame == "cli1024":
                            rois[a + "/added"] = pred & ~previous["foris.crf"]
                            rois[a + "/deleted"] = ~pred & previous["foris.crf"]
                if frame == "cli1024":
                    mechanisms.append(dict(episode_id=row["episode_id"], dataset=row["dataset"],
                        ranks=rank_episode(score_fields, gt, rois),
                        source_fit={a: {k: float(rec["diagnostics"][a]["fit"][k]) for k in
                            ("reference_binary_role_loss", "reference_weighted_rmse", "regularized_objective",
                             "coefficient_l2", "bias", "dual_relative_residual",
                             "weighted_stationarity_l2")} for a in ARMS}))
                records.append(result)
                grouped[(row["dataset"], frame)].append(result)
    report = {}
    for (dataset, frame), rr in grouped.items():
        expected = cfg["coco_expected_classes"] if dataset == "coco" else None
        report[dataset + "/" + frame] = point_and_intervals(rr, expected)
        if dataset == "paco_part":
            report[dataset + "/" + frame + "/fixed303"] = point_and_intervals(rr, cfg["paco_expected_classes"])
    # Compare to the parent's published ledger/report if present; the audit can
    # also be run immediately after the complete seal before that scorer finishes.
    compared = {}
    parent_ledger = RUN / "scored_episodes.jsonl"
    if parent_ledger.exists():
        old = {(r["episode_id"], r["frame"]): r for r in
               (json.loads(s) for s in parent_ledger.read_text().splitlines() if s)}
        assert len(old) == len(records) == 1200
        for r in records:
            prior = old[r["episode_id"], r["frame"]]
            assert all(r[k] == prior[k] for k in ("iu", "edits", "dataset", "fold", "class_id"))
            CHECKS["parent_episode_frame_ledger_identity"] += 1
    parent_report = RUN / "report.json"
    if parent_report.exists():
        old = read(parent_report)["results"]
        assert set(old) == set(report)
        for k, r in report.items():
            difference = max(abs(r["miou"][a] - old[k]["miou"][a]) for a in r["miou"])
            assert difference < 1e-10, (k, difference)
            compared[k] = difference
    summary = dict(state="VERIFIED", n=600, frames=2, candidates=6,
        seal_sha256=sha(RUN / "sealed.json"), config_sha256=sha(RUN / "config.json"),
        checks=dict(CHECKS), report=report, parent_report_maximum_differences=compared,
        field_stats={k: dict(score_min=min(v[0] for v in values), score_max=max(v[1] for v in values),
                            mean_positive_token_fraction=sum(v[2] for v in values)/len(values))
                     for k, values in field_stats.items()},
        saved_file_SHA_identities=HASHES, seconds=time.monotonic()-start,
        raw_feature_reads=0, new_inference=0, new_prediction_masks_saved=0,
        exposure=cfg["exposure"], primary_frame="cli1024")
    (OUT / "scene600_verified.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (OUT / "scene600_recomputed_episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    (OUT / "scene600_mechanism_episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in mechanisms))
    mechanism_summary = summarize_mechanisms(mechanisms)
    (OUT / "scene600_mechanism_summary.json").write_text(
        json.dumps(mechanism_summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(state="VERIFIED", checks=dict(CHECKS), seconds=summary["seconds"],
        points={k: v["miou"] for k, v in report.items()})), flush=True)


if __name__ == "__main__":
    main()
