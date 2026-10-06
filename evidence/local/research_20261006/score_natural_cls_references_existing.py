"""Natural object CLS as single references, using only existing vectors.

Every eligible natural positive crop is evaluated separately as one reference;
scores are never ensembled and no best reference is selected. Average those
single-reference AUROCs within each original paired episode, then across episodes.
This changes reference images and crop construction, so it is NOT a causal test
of RGB erasure. All query object proposals are still privileged GT diagnostics.
"""
from pathlib import Path
import json
import time

import numpy as np

from score_cls_components_existing import ROOT, SOURCE, digest, photo_groups, unit


OUT = Path(__file__).with_name("natural_cls_references_existing_01a1100b")


def main():
    if OUT.exists():
        raise FileExistsError("Preserve completed analysis: " + str(OUT))
    start = time.process_time()
    data = json.loads((SOURCE / "labels.json").read_text())
    rows, records = data["source_rows"], data["regions"]
    index = {r["key"]: i for i, r in enumerate(rows)}
    groups = photo_groups(rows)
    bykey = {r["key"]: [] for r in rows}
    vectors, hashes = {}, {}
    seal = json.loads((SOURCE / "sealed.json").read_text())
    for key in sorted({r["episode"] for r in records}):
        path = SOURCE / "descriptors" / (key + ".npz")
        hashes[key] = digest(path)
        assert hashes[key] == seal["descriptors"][key]
        with np.load(path, allow_pickle=False) as z:
            for cid, vector in zip(z["crop_ids"].tolist(), unit(z["q"])):
                vectors[cid] = vector.copy()
    for region in records:
        bykey[region["episode"]].append(region)
    # Keep all recorded natural positive views. Repeated photographs remain
    # dependent and share bootstrap weights; do not choose by any score.
    anchors = [r for r in records if r["role"] == "whole_missed_GT"]
    ancestor = ROOT / "evidence/local/research_20261006/cls_components_existing_01a1100b"
    old = {r["key"]: r for r in json.loads((ancestor / "episode_auc.json").read_text())}
    paired, comparisons, unavailable = [], [], []
    for row in rows:
        key, i = row["key"], index[row["key"]]
        if key not in old:
            continue
        compatible = [a for a in anchors if rows[index[a["episode"]]]["c"] == row["c"]
                      and groups[index[a["episode"]]] != groups[i]]
        if not compatible:
            unavailable.append(key)
            continue
        pos = np.stack([vectors[r["crop_id"]] for r in bykey[key] if r["role"] == "whole_missed_GT"])
        neg = np.stack([vectors[r["crop_id"]] for r in bykey[key] if r["role"] == "stray_fine64"])
        outcomes = []
        for a in compatible:
            av = vectors[a["crop_id"]]
            p = np.sum(pos * av[None], axis=1)
            n = np.sum(neg * av[None], axis=1)
            difference = p[:, None] - n[None]
            value = float(np.mean((difference > 0) + .5 * (difference == 0)))
            ag = int(groups[index[a["episode"]]])
            outcomes.append((ag, value))
            comparisons.append(dict(query_episode=key, reference_crop=a["crop_id"],
                query_group=int(groups[i]), reference_group=ag, auc=value))
        paired.append(dict(key=key, c=int(row["c"]), fold=int(row["fold"]), group=int(groups[i]),
            references=len(outcomes), reference_photos=len({Path(rows[index[a['episode']]]['query']).name for a in compatible}),
            auc=float(np.mean([v for _, v in outcomes])),
            original_margin_auc=old[key]["auc"]["original_margin"],
            original_foreground_auc=old[key]["auc"]["foreground_cosine"], outcomes=outcomes))
    assert len(old) == 77 and len(paired) + len(unavailable) == 77
    assert all(r["query_group"] != r["reference_group"] for r in comparisons)
    draws = np.load(SOURCE / "bootstrap_photo_draws.npy", allow_pickle=False)
    g = int(groups.max()) + 1
    assert draws.shape == (2000, g) and g == 1046
    weights = np.stack([np.bincount(draw, minlength=g) for draw in draws])
    # Resample both endpoints from the same existing photo-group draw. The
    # reference weights enter each query's one-reference expectation; target
    # weights enter its outer mean. This preserves shared-anchor dependencies.
    numerator = {k: np.zeros(len(draws)) for k in ("natural", "original_fg", "original_margin")}
    denominator = np.zeros(len(draws))
    missing_anchor_counts = np.zeros(len(draws), int)
    for r in paired:
        reference_groups = np.array([a for a, _ in r["outcomes"]], int)
        values = np.array([v for _, v in r["outcomes"]])
        aw = weights[:, reference_groups]
        an = aw.sum(1)
        means = np.divide(np.sum(aw * values[None], axis=1), an,
                          out=np.zeros(len(draws)), where=an > 0)
        tw = weights[:, r["group"]] * (an > 0)
        missing_anchor_counts += (an == 0) & (weights[:, r["group"]] > 0)
        denominator += tw
        numerator["natural"] += tw * means
        numerator["original_fg"] += tw * r["original_foreground_auc"]
        numerator["original_margin"] += tw * r["original_margin_auc"]
    good = denominator > 0
    samples = {k: v[good] / denominator[good] for k, v in numerator.items()}
    def summary(name, point):
        return dict(mean=float(point), ci95=np.percentile(samples[name], [2.5, 97.5]).tolist(),
                    paired_episodes=len(paired))
    means = {"natural": np.mean([r["auc"] for r in paired]),
        "original_fg": np.mean([r["original_foreground_auc"] for r in paired]),
        "original_margin": np.mean([r["original_margin_auc"] for r in paired])}
    primary = {k: summary(k, v) for k, v in means.items()}
    contrasts = {}
    for name, baseline in (("natural_minus_chance", None), ("natural_minus_original_fg", "original_fg"),
                           ("natural_minus_original_margin", "original_margin")):
        v = samples["natural"] - (samples[baseline] if baseline else .5)
        contrasts[name] = dict(mean=float(means["natural"] - (means[baseline] if baseline else .5)),
                              ci95=np.percentile(v, [2.5, 97.5]).tolist())
    folds = {str(f): dict(n=sum(r["fold"] == f for r in paired),
                         natural_auc=float(np.mean([r["auc"] for r in paired if r["fold"] == f])))
             for f in range(4)}
    report = dict(scope="post-hoc natural single-reference representation diagnostic on existing exposed vectors",
        primary=primary, contrasts=contrasts, folds=folds,
        original_paired_episodes=77, no_eligible_reference=unavailable,
        eligible_classes=len({r["c"] for r in paired}), reference_pool_regions=len(anchors),
        single_reference_comparisons=len(comparisons),
        reference_count_range=[min(r["references"] for r in paired), max(r["references"] for r in paired)],
        bootstrap="2000 original connected-photo group draws; recompute inner reference expectation and outer episode mean with jointly sampled endpoint weights",
        bootstrap_valid_draws=int(good.sum()), bootstrap_max_queries_missing_resampled_anchors=int(missing_anchor_counts.max()),
        limits=["GT-defined query regions and natural reference object views are privileged diagnostics, not legal query proposals.",
                "Changing reference image and crop is confounded with removing RGB erasure; this is not an erasure causal experiment.",
                "Natural references are existing entirely missed GT objects, not a random deployment reference distribution.",
                "All data were exposed; descriptive intervals are not independent confirmation.",
                "No reference is selected by its result; each AUROC uses exactly one reference, not an ensemble."],
        complete_segmentation_method=False, encoder_forwards=0, images_opened=0, masks_generated=0, server_access=False,
        descriptor_sha256=hashes, source_sha256={p: digest(SOURCE / p) for p in
            ("labels.json", "sealed.json", "bootstrap_photo_draws.npy")},
        original_component_analysis_sha256=digest(ancestor / "episode_auc.json"),
        script_sha256=digest(__file__), cpu_seconds=time.process_time() - start)
    OUT.mkdir()
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (OUT / "single_reference_auc.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    (OUT / "episode_auc.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "outcomes"} for r in paired], indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("primary", "contrasts", "folds", "eligible_classes", "no_eligible_reference", "single_reference_comparisons", "cpu_seconds")}, indent=2))


if __name__ == "__main__":
    main()
