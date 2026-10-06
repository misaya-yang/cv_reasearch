"""Split one existing CLS margin into its fixed components; no inference.

The existing 77 paired episodes, privileged regions, vectors, score orientation,
and saved photo-group bootstrap draws stay unchanged. This is a post-hoc
diagnostic of existing results, not a complete segmentation method.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "evidence/local/research_20261005/pipeline_verified/object_cls_dev1200_v1"
OUT = Path(__file__).with_name("cls_components_existing_01a1100b")
CUES = ("original_margin", "foreground_cosine", "negative_background_cosine", "stored_nn_mean")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(value):
    value = np.asarray(value, np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm <= 1e-12):
        raise ValueError("Invalid existing descriptor")
    return value / norm


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
    ids, result = {}, []
    for i in range(len(rows)):
        key = find(i)
        ids.setdefault(key, len(ids))
        result.append(ids[key])
    return np.array(result)


def main():
    if OUT.exists():
        raise FileExistsError("Preserve completed analysis: " + str(OUT))
    start = time.process_time()
    labels = json.loads((SOURCE / "labels.json").read_text())
    rows, regions = labels["source_rows"], labels["regions"]
    old = {r["crop_id"]: r for r in json.loads((SOURCE / "scored_regions.json").read_text())}
    seal = json.loads((SOURCE / "sealed.json").read_text())
    report_old = json.loads((SOURCE / "report.json").read_text())
    values, bykey, hashes = {}, {r["key"]: [] for r in rows}, {}
    for key in sorted({r["episode"] for r in regions}):
        path = SOURCE / "descriptors" / (key + ".npz")
        hashes[key] = digest(path)
        assert hashes[key] == seal["descriptors"][key], key
        with np.load(path, allow_pickle=False) as data:
            query = unit(data["q"])
            fg = np.sum(query * unit(data["r_fg"])[None], axis=1)
            bg = np.sum(query * unit(data["r_bg"])[None], axis=1)
            for cid, f, b in zip(data["crop_ids"].tolist(), fg, bg):
                margin = float(f - b)
                assert margin == old[cid]["scores"]["object_cls"], cid
                values[cid] = dict(original_margin=margin, foreground_cosine=float(f),
                    negative_background_cosine=-float(b),
                    stored_nn_mean=old[cid]["scores"]["stored_nn_mean"])
    assert set(values) == {r["crop_id"] for r in regions}
    for region in regions:
        bykey[region["episode"]].append(region)
    aucs = {name: np.full(len(rows), np.nan) for name in CUES}
    for i, row in enumerate(rows):
        pos = [r for r in bykey[row["key"]] if r["role"] == "whole_missed_GT"]
        neg = [r for r in bykey[row["key"]] if r["role"] == "stray_fine64"]
        if not pos or not neg:
            continue
        for name in CUES:
            p = np.array([values[r["crop_id"]][name] for r in pos])
            n = np.array([values[r["crop_id"]][name] for r in neg])
            difference = p[:, None] - n[None]
            aucs[name][i] = float(np.mean((difference > 0) + .5 * (difference == 0)))
    valid = np.isfinite(aucs["original_margin"])
    assert int(valid.sum()) == 77 and len(rows) == 1200 and len(regions) == 736
    assert all(np.array_equal(np.isfinite(v), valid) for v in aucs.values())
    groups = photo_groups(rows)
    group_count = int(groups.max()) + 1
    draws = np.load(SOURCE / "bootstrap_photo_draws.npy", allow_pickle=False)
    assert group_count == 1046 and draws.shape == (2000, group_count)
    assert np.array_equal(draws, np.random.RandomState(0).randint(group_count, size=draws.shape))
    weights = np.stack([np.bincount(draw, minlength=group_count) for draw in draws])
    count = np.bincount(groups[valid], minlength=group_count)
    denominator = np.sum(weights * count[None], axis=1)
    assert np.all(denominator > 0)
    def summarize(v):
        total = np.bincount(groups[valid], weights=v[valid], minlength=group_count)
        samples = np.sum(weights * total[None], axis=1) / denominator
        return dict(mean=float(np.mean(v[valid])), ci95=np.percentile(samples, [2.5, 97.5]).tolist(),
                    eligible_episodes=int(valid.sum()))
    primary = {k: summarize(v) for k, v in aucs.items()}
    for new, original in (("original_margin", "object_cls"), ("stored_nn_mean", "stored_nn_mean")):
        assert primary[new] == report_old["primary"][original], (new, primary[new])
    contrasts = {"foreground_minus_margin": summarize(aucs["foreground_cosine"] - aucs["original_margin"]),
        "foreground_minus_chance": summarize(aucs["foreground_cosine"] - .5),
        "negative_background_minus_chance": summarize(aucs["negative_background_cosine"] - .5)}
    per_episode = [dict(key=r["key"], c=r["c"], fold=r["fold"], photo_group=int(groups[i]),
        auc={k: float(v[i]) for k, v in aucs.items()}) for i, r in enumerate(rows) if valid[i]]
    folds = {str(f): {k: float(np.mean([r["auc"][k] for r in per_episode if r["fold"] == f]))
                     for k in CUES} for f in range(4)}
    result = dict(scope="post-hoc fixed component diagnostic on existing exposed CLS1200 vectors",
        primary=primary, paired_contrasts=contrasts, folds=folds,
        formula="original_margin = foreground_cosine + negative_background_cosine",
        selection="all original77 paired episodes/296 regions; no subset, fit, sign reversal or threshold search",
        GT_geometry_privileged=True, complete_method_result=False,
        encoder_forwards=0, images_opened=0, server_access=False, masks_generated=0,
        original_margin_exact_parity=True, old_primary_and_intervals_exact_parity=True,
        source_rows=1200, paired_episodes=77, region_vectors=736, descriptor_files=len(hashes),
        bootstrap="unchanged2000 RandomState(0) draws over1046 connected photo groups",
        source_sha256={name: digest(SOURCE / name) for name in
            ("labels.json", "scored_regions.json", "sealed.json", "report.json", "bootstrap_photo_draws.npy")},
        descriptor_sha256=hashes, script_sha256=digest(__file__), cpu_seconds=time.process_time() - start)
    OUT.mkdir()
    (OUT / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (OUT / "episode_auc.json").write_text(json.dumps(per_episode, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("primary", "paired_contrasts", "folds", "cpu_seconds")}, indent=2))


if __name__ == "__main__":
    main()
