"""Independent CPU reconstruction from sealed compact CLS vectors."""
import hashlib
import json
from pathlib import Path
import numpy as np

p = Path(__file__).resolve().parent
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
seal = json.loads((p / "sealed.json").read_text())
prep = json.loads((p / "prepared.json").read_text())
report = json.loads((p / "report.json").read_text())
cases = json.loads((p / "crops.json").read_text())
labels = json.loads((p / "labels.json").read_text())
scored = json.loads((p / "scored_regions.json").read_text())
assert seal["state"] == "ALL_CLS_DESCRIPTORS_SEALED" and seal["descriptor_forwards"] == 367
assert sha(p / "prepared.json") == seal["prepared_sha256"]
assert sha(p / "report.json") == json.loads((p / "score_state.json").read_text())["report_sha256"]
for key, name in (("config_sha256", "config.json"), ("crops_sha256", "crops.json"), ("labels_sha256", "labels.json")):
    assert sha(p / name) == prep[key]
score_lookup = {r["crop_id"]: r for r in scored}
margin, margin_errors, fp32errors = {}, [], []
for case in cases:
    path = p / "descriptors" / (case["key"] + ".npz")
    assert sha(path) == seal["descriptors"][case["key"]]
    with np.load(path, allow_pickle=False) as z:
        assert z["crop_ids"].tolist() == [r["crop_id"] for r in case["regions"]]
        assert z["q"].shape == (len(case["regions"]), 1024)
        assert all(z[k].dtype == np.float32 for k in ("q", "r_fg", "r_bg"))
        q, fg, bg = (z[k].astype(np.float64) for k in ("q", "r_fg", "r_bg"))
        value = np.sum(q * fg[None], axis=1) / (np.sqrt(np.sum(q * q, axis=1)) * np.sqrt(np.sum(fg * fg)))
        value -= np.sum(q * bg[None], axis=1) / (np.sqrt(np.sum(q * q, axis=1)) * np.sqrt(np.sum(bg * bg)))
        unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
        exact = np.sum(unit(z["q"]) * unit(z["r_fg"])[None], axis=1) - np.sum(unit(z["q"]) * unit(z["r_bg"])[None], axis=1)
        for crop_id, v, e in zip(z["crop_ids"].tolist(), value, exact):
            original = score_lookup[crop_id]["scores"]["object_cls"]
            margin_errors.append(abs(float(v) - original))
            fp32errors.append(abs(float(e) - original))
            margin[crop_id] = float(v)
assert len(margin) == 197 and max(margin_errors) < 1e-6 and max(fp32errors) == 0
rows, photos = labels["source_rows"], {}
for i, row in enumerate(rows):
    for role in ("support", "query"):
        photos.setdefault(Path(row[role]).name, []).append(i)
adj = [set() for row in rows]
for ids in photos.values():
    for i in ids:
        adj[i].update(ids)
groups, g = np.full(len(rows), -1, int), 0
for i in range(len(rows)):
    if groups[i] >= 0:
        continue
    stack, groups[i] = [i], g
    while stack:
        current = stack.pop()
        for j in adj[current]:
            if groups[j] < 0:
                groups[j] = g
                stack.append(j)
    g += 1
assert g == 239
draws = np.random.RandomState(0).randint(g, size=(2000, g))
assert np.array_equal(draws, np.load(p / "bootstrap_photo_draws.npy"))
weights = np.stack([np.bincount(d, minlength=g) for d in draws])
bycase = {r["key"]: [] for r in rows}
for r in scored:
    bycase[r["episode"]].append(r)
curves, errors = {}, []
for cue in ("object_cls", "raw_mean_prototype", "projected_mean_prototype"):
    values = []
    for row in rows:
        rr = bycase[row["key"]]
        get = lambda r: margin[r["crop_id"]] if cue == "object_cls" else r["scores"][cue]
        pos = np.array([get(r) for r in rr if r["role"] == "whole_missed_GT"])
        neg = np.array([get(r) for r in rr if r["role"] == "stray_fine64"])
        values.append(float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean()) if len(pos) and len(neg) else np.nan)
    curves[cue] = np.array(values)
    assert np.allclose(np.load(p / "episode_auc.npz")[cue], values, equal_nan=True, rtol=0, atol=1e-15)
contrasts = [("object_cls-minus-" + c, curves["object_cls"] - curves[c]) for c in ("raw_mean_prototype", "projected_mean_prototype")]
for name, v in list(curves.items()) + contrasts + [("object_cls_minus_chance", curves["object_cls"] - 0.5)]:
    valid = np.isfinite(v)
    den = np.sum(weights * np.bincount(groups[valid], minlength=g)[None], axis=1)
    totals = np.bincount(groups[valid], weights=v[valid], minlength=g)
    boot = np.sum(weights * totals[None], axis=1) / den
    mean, ci = float(v[valid].mean()), np.percentile(boot, [2.5, 97.5])
    source = report["paired_contrasts"][name] if "-minus-" in name else report["primary"][name]
    errors.extend([abs(mean - source["mean"]), *abs(ci - source["ci95"])])
    assert valid.sum() == 18
for fold, entry in report["folds"].items():
    ix = [i for i, r in enumerate(rows) if r["fold"] == int(fold) and np.isfinite(curves["object_cls"][i])]
    for cue in curves:
        errors.append(abs(float(np.mean(curves[cue][ix])) - entry["episode_auc"][cue]))
for cue, source in report["supplement"].items():
    pos = [r for r in scored if r["role"] == "whole_missed_GT"]
    neg = [r for r in scored if r["role"] == "stray_fine64"]
    get = lambda r: margin[r["crop_id"]] if cue == "object_cls" else r["scores"][cue]
    d = np.array([get(r) for r in pos])[:, None] - np.array([get(r) for r in neg])[None]
    win = (d > 0) + 0.5 * (d == 0)
    mass = np.array([r["area"] for r in pos], float)[:, None] * np.array([r["area"] for r in neg], float)[None]
    errors.extend([abs(float(win.mean()) - source["region_pooled_auc"]), abs(float((win * mass).sum() / mass.sum()) - source["pixel_mass_weighted_auc"])])
assert np.isfinite(errors).all() and max(errors) < 1e-14
result = dict(state="FINAL_CLS_INDEPENDENTLY_VERIFIED", descriptor_files=85, compact_vectors=367, region_ids=197,
    all_descriptor_seal_and_preparation_report_hashes_verified=True,
    FP64_direct_cosine_vs_report_max_error=float(max(margin_errors)), FP32_recomputed_margin_max_error=float(max(fp32errors)),
    per_episode_AUC_method="direct positive-negative score pair comparisons; half-credit ties",
    connected_photo_group_method="independent photo-graph traversal", photo_groups=g, exact2000_RS0_draws=True,
    all_AUC_CI_paired_CI_fold_and_pooled_statistics_max_error=float(max(errors)), primary_eligible_episodes=18,
    primary_regions=71, primary=report["primary"], paired_contrasts=report["paired_contrasts"], report_sha256=sha(p / "report.json"),
    classification="unresolved; above-chance and raw-control superiority intervals include zero")
(p / "final_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("primary", "paired_contrasts")}, indent=2))
