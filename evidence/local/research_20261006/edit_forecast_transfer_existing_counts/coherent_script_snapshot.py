"""Retrospective forecasts from existing counts; never render or change a mask.

One fixed transfer check: COCO correct-edit fractions in the already frozen six
size bins predict the same frozen size-cut rule on other recorded datasets.
The predictor only receives mask areas and edit volumes. Historical labels fit
its constants; target labels score the saved predictions. All data are exposed.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "evidence/local/research_20261005"
OUT = Path(__file__).parent / "edit_forecast_transfer_existing_counts"
PIXELS = 1024**2
FROZEN = json.loads((OLD / "rcg2_frozen.json").read_text())
EDGES = np.array(FROZEN["size_cut"]["mask_area_at_half_edges"])
CUTS = np.array(FROZEN["size_cut"]["levels"])
SOURCES = {}


def record(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pack(name):
    path = OLD / name
    record(path / "counts.npz")
    record(path / "rows.json")
    rows = json.loads((path / "rows.json").read_text())
    with np.load(path / "counts.npz", allow_pickle=False) as z:
        b, s = z["iu:rcg_fine"], z["iu:rcg2"]
        area = b.sum(1) - z["truth"]
        new_area = s.sum(1) - z["truth"]
        selected = z["picked"]
        levels = z["levels"]
        half = int(np.argmin(abs(levels - 0.5)))
        k = abs(levels[None] - selected[:, None]).argmin(1)
        ix = np.arange(len(rows))
        assert np.allclose(levels[k], selected, rtol=0, atol=3e-8)
        assert np.array_equal(b, np.stack((z["I_fine"][:, half], z["U_fine"][:, half]), 1))
        assert np.array_equal(s, np.stack((z["I_fine"][ix, k], z["U_fine"][ix, k]), 1))
        assert np.all(np.diff(z["I_fine"], axis=1) <= 0)
        assert np.all(np.diff(z["U_fine"], axis=1) <= 0)
    bins = np.searchsorted(EDGES, area / PIXELS, side="right") - 1
    assert np.allclose(CUTS[bins], selected, rtol=0, atol=3e-8)
    dm = new_area - area
    a, d = np.maximum(dm, 0), np.maximum(-dm, 0)
    # Nested masks make the area difference equal the entire edit volume.
    assert np.all(((dm >= 0) == (selected <= 0.5)) | (dm == 0))
    w = np.where(dm >= 0, s[:, 0] - b[:, 0], b[:, 1] - s[:, 1])
    assert np.all((0 <= w) & (w <= a + d))
    assert np.array_equal(b[:, 0] + w - d, s[:, 0])
    assert np.array_equal(b[:, 1] + a - w, s[:, 1])
    v = np.zeros((len(rows), len(CUTS)), dtype=np.int64)
    good = np.zeros_like(v)
    v[np.arange(len(rows)), bins] = a + d
    good[np.arange(len(rows)), bins] = w
    return dict(rows=rows, base=b, changed=s, area=area, volume=v, good=good,
                add=a, delete=d, pixels=np.full(len(rows), PIXELS, dtype=np.int64))


def combine(parts, merged_name=None):
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "rows"}
    merged["rows"] = [r for p in parts for r in p["rows"]]
    if merged_name is not None:
        path = OLD / merged_name
        record(path / "rows.json")
        record(path / "counts.npz")
        rows = json.loads((path / "rows.json").read_text())
        assert len(rows) == len(merged["rows"])
        for official, local in zip(rows, merged["rows"]):
            assert official["key"].split("/", 1)[-1] == local["key"]
            assert all(official[k] == local[k] for k in ("query", "support"))
        with np.load(path / "counts.npz", allow_pickle=False) as z:
            assert np.array_equal(z["iu:rcg_fine"], merged["base"])
            assert np.array_equal(z["iu:rcg2"], merged["changed"])
        merged["rows"] = rows  # Keep the report's actual fold-class identities.
    return merged


def aggregate(data, keep=None):
    cls = np.array([r["c"] for r in data["rows"]])
    keep = np.ones(len(cls), dtype=bool) if keep is None else keep
    ids = np.unique(cls[keep])
    return dict(ids=ids, **{k: np.stack([data[k][keep & (cls == c)].sum(0) for c in ids])
                           for k in data if k != "rows"})


def fit(data, keep=None):
    keep = np.ones(len(data["rows"]), dtype=bool) if keep is None else keep
    v, g = data["volume"][keep].sum(0), data["good"][keep].sum(0)
    p = np.divide(g, v, out=np.zeros(len(v)), where=v != 0)
    assert np.all(v[CUTS != 0.5] > 0)
    m = data["area"][keep].sum()
    counts = data["base"][keep].sum(0)
    c = aggregate(data, keep)
    constant = 100 * np.mean(c["changed"][:, 0] / c["changed"][:, 1] -
                             c["base"][:, 0] / c["base"][:, 1])
    return dict(purity=p.tolist(), intersection_per_area=float(counts[0] / m),
                union_per_area=float(counts[1] / m), constant_gain=float(constant),
                training_episodes=int(keep.sum()))


def forecast(area, volume, model):
    """No target intersection, union or truth-area argument exists here."""
    w = np.sum(volume * np.array(model["purity"]), axis=1)
    a, d = volume[:, CUTS < 0.5].sum(1), volume[:, CUTS > 0.5].sum(1)
    i = area * model["intersection_per_area"]
    u = area * model["union_per_area"]
    active = a + d > 0
    assert np.all((u[active] > 0) & (u[active] + a[active] - w[active] > 0))
    change = np.zeros(len(area))
    change[active] = gain(i[active], u[active], a[active], d[active], w[active])
    return dict(correct=w.tolist(), baseline_I=i.tolist(), baseline_U=u.tolist(),
                gain=change.tolist(), constant=model["constant_gain"])


def photo_groups(rows):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        parent[find(r["support"])] = find(r["query"])
    return np.array([find(r["query"]) for r in rows])


def gain(i, u, a, d, w):
    return 100 * ((i + w - d) / (u + a - w) - i / u)


def error_radius(i, u, area, a, d, w, pixels):
    """Exact adverse class-purity radius at these fixed baseline counts.

    This is sensitivity, not an estimated error guarantee or confidence bound.
    The feasible lower endpoint follows from the capacities inside/outside O.
    """
    e = a + d
    low = np.maximum(0, a - (pixels - u)) + np.maximum(0, d - i)
    high = np.minimum(a, u - area) + np.minimum(d, area - i)
    if np.any((w < low) | (w > high)):
        return dict(radius=None, reason="point estimate violates true baseline count capacities")
    if np.mean(gain(i, u, a, d, w)) <= 0:
        return dict(radius=0.0, reason="point forecast is already nonpositive")
    if np.mean(gain(i, u, a, d, low)) > 0:
        return dict(radius=None, reason="positive throughout the feasible count interval")
    left, right = 0.0, 1.0
    for _ in range(55):
        mid = (left + right) / 2
        wm = np.maximum(low, w - mid * e)
        if np.mean(gain(i, u, a, d, wm)) > 0:
            left = mid
        else:
            right = mid
    return dict(radius=right, reason="exact zero crossing with true baseline fixed")


def evaluate(c, prediction, model):
    i, u = c["base"].T
    a, d, w = c["add"], c["delete"], np.array(prediction["correct"])
    observed = 100 * (c["changed"][:, 0] / c["changed"][:, 1] - i / u)
    known = gain(i, u, a, d, w)
    primary = np.array(prediction["gain"])
    ih, uh = np.array(prediction["baseline_I"]), np.array(prediction["baseline_U"])
    ni, nu, nm = ih + w - d, uh + a - w, c["area"] + a - d
    valid = (ih >= 0) & (ih <= c["area"]) & (uh >= c["area"]) & (uh <= c["pixels"])
    valid &= (ni >= 0) & (ni <= nm) & (nu >= nm) & (nu <= c["pixels"])
    sv, sg = c["volume"].sum(0), c["good"].sum(0)
    pool = np.divide(sg, sv, out=np.zeros(len(sv)), where=sv != 0)
    rows = [dict(c=int(cid), actual=float(observed[j]), predicted=float(primary[j]),
                 known_baseline_prediction=float(known[j]), edits=int(a[j] + d[j]),
                 correct=int(c["good"][j].sum()), predicted_correct=float(w[j]),
                 counts_valid=bool(valid[j])) for j, cid in enumerate(c["ids"])]
    return dict(classes=len(i), episodes=int(c["pixels"].sum() // PIXELS),
                actual_gain=float(observed.mean()), predicted_gain=float(primary.mean()),
                known_baseline_prediction=float(known.mean()), constant_gain=prediction["constant"],
                class_MAE=float(abs(primary - observed).mean()),
                sign_correct=int(((primary > 0) == (observed > 0)).sum()),
                actual_positive=int((observed > 0).sum()),
                predicted_positive=int((primary > 0).sum()),
                invalid_prediction_classes=c["ids"][~valid].tolist(),
                edit_volume_by_bin=sv.tolist(), actual_purity_by_bin=pool.tolist(),
                model=model, per_class=rows,
                fixed_true_baseline_adverse_purity_radius=error_radius(i, u, c["area"], a, d, w, c["pixels"]))


def main():
    start = time.process_time()
    assert not (OUT / "report.json").exists(), "Never overwrite a completed analysis."
    assert not (OUT / "predictions.json").exists(), "Never overwrite saved forecasts."
    OUT.mkdir(exist_ok=True)
    write("config.json", dict(
        scope="retrospective existing-count arithmetic; no new masks or inference",
        comparison="unchanged frozen RCG2 vs unchanged RCG fine at level 0.5",
        resolution=1024, training="COCO groups B+C+D, 1800 old draws",
        model="pixel-pooled correct-edit share in each of the six already frozen size bins",
        baseline="historical pooled I/mask-area and U/mask-area; same estimator as prior check",
        target_observables=["baseline mask area", "net edit volume in each frozen size bin"],
        target_label_use="only algebraic reconstruction of GT-invariant areas, then saved-forecast scoring",
        controls=["source mean gain", "target true baseline I/U diagnostic"],
        coco_readout="train other class folds, excluding photo components touching held fold",
        caveats=["all records previously exposed", "LVIS shares COCO imagery",
                 "PACO/LVIS preserve existing merged fold-class ids",
                 "sensitivity radii are not confidence intervals", "only 1024 counts are compared"],
        frozen_size_cuts=FROZEN["size_cut"], no_parameter_search=True,
        no_server=True, no_images_or_feature_arrays=True))
    coco = combine([pack(s) for s in ("rcg2_groupB", "rcg2_groupC", "rcg2_group_groupD")])
    target = dict(SUIM=pack("rcg2_group_suim"), PASCAL_Part=pack("rcg2_grouppascal_part"))
    for name, slug in (("PACO_Part", "paco_part"), ("LVIS", "lvis")):
        target[name] = combine([pack(f"rcg2_group_{slug}_f{f}") for f in range(4)],
                               f"rcg2_group_{slug}")
    model = fit(coco)
    forecasts, aggregates, models = {}, {}, {}
    for name, data in target.items():
        c = aggregate(data)
        aggregates[name], models[name] = c, model
        forecasts[name] = forecast(c["area"], c["volume"], model)
    f = np.array([r["fold"] for r in coco["rows"]])
    groups = photo_groups(coco["rows"])
    for fold in range(4):
        test = f == fold
        train = ~test & ~np.isin(groups, groups[test])
        assert not set(groups[train]) & set(groups[test])
        name, c, m = f"COCO_fold{fold}", aggregate(coco, test), fit(coco, train)
        aggregates[name], models[name] = c, m
        forecasts[name] = forecast(c["area"], c["volume"], m)
    write("predictions.json", forecasts)
    seal = hashlib.sha256((OUT / "predictions.json").read_bytes()).hexdigest()
    reports = {name: evaluate(aggregates[name], p, models[name]) for name, p in forecasts.items()}
    record(OLD / "rcg2_frozen.json")
    record(ROOT / "scripts/run_rcg2_stream.py")
    report = dict(prediction_sha256=seal, sources=SOURCES, results=reports,
                  local_process_cpu_seconds=time.process_time() - start,
                  checks="all level-count parity, nestedness, frozen cuts, merged rows and edit identities pass")
    write("report.json", report)
    print(json.dumps({name: {k: r[k] for k in ("classes", "actual_gain", "predicted_gain",
          "known_baseline_prediction", "invalid_prediction_classes",
          "fixed_true_baseline_adverse_purity_radius")} for name, r in reports.items()}, indent=2))


def atom_volumes(data):
    """Unlabelled volumes of 00, 10, 01, 11 (baseline, changed) in each bin."""
    area, a, d = data["area"], data["add"], data["delete"]
    bins = np.searchsorted(EDGES, area / PIXELS, side="right") - 1
    v = np.stack((data["pixels"] - area - a, d, a, area - d), 1)
    assert np.all(v >= 0)
    assert np.array_equal(v.sum(1), data["pixels"])
    result = np.zeros((len(area), len(CUTS), 4), dtype=np.int64)
    result[np.arange(len(area)), bins] = v
    return result, bins


def fit_coherent(data, keep=None):
    keep = np.ones(len(data["rows"]), dtype=bool) if keep is None else keep
    v, bins = atom_volumes(data)
    i, u = data["base"].T
    a, d = data["add"], data["delete"]
    # Thresholds are nested within each episode, so these four truth masses are exact.
    added_true = np.where(a > 0, data["changed"][:, 0] - i, 0)
    deleted_true = np.where(d > 0, i - data["changed"][:, 0], 0)
    t = np.stack((u - data["area"] - added_true, deleted_true, added_true, i - deleted_true), 1)
    truth = np.zeros_like(v)
    truth[np.arange(len(t)), bins] = t
    assert np.all((truth >= 0) & (truth <= v))
    total, good = v[keep].sum(0), truth[keep].sum(0)
    p = np.divide(good, total, out=np.zeros_like(total, dtype=float), where=total != 0)
    return dict(atom_truth_probability=p.tolist(), source_atom_volume=total.tolist(),
                constant_gain=fit(data, keep)["constant_gain"], training_episodes=int(keep.sum()))


def coherent_forecast(data, model, keep=None):
    # This function reads only unlabelled mask volumes and class ids from data.
    v, _ = atom_volumes(data)
    cls = np.array([r["c"] for r in data["rows"]])
    keep = np.ones(len(cls), dtype=bool) if keep is None else keep
    ids = np.unique(cls[keep])
    volume = np.stack([v[keep & (cls == c)].sum(0) for c in ids])
    assert not np.any((volume.sum(0) > 0) & (np.array(model["source_atom_volume"]) == 0))
    truth = (volume * np.array(model["atom_truth_probability"])).sum(1)
    atoms = volume.sum(1)
    i, u = truth[:, 1] + truth[:, 3], atoms[:, 1] + atoms[:, 3] + truth[:, 0] + truth[:, 2]
    ni, nu = truth[:, 2] + truth[:, 3], atoms[:, 2] + atoms[:, 3] + truth[:, 0] + truth[:, 1]
    w = truth[:, 2] + atoms[:, 1] - truth[:, 1]
    assert np.allclose(i + w - atoms[:, 1], ni, rtol=0, atol=2e-8)
    assert np.allclose(u + atoms[:, 2] - w, nu, rtol=0, atol=2e-8)
    assert np.all((u > 0) & (nu > 0))
    change = 100 * (ni / nu - i / u)
    return dict(correct=w.tolist(), baseline_I=i.tolist(), baseline_U=u.tolist(),
                gain=change.tolist(), constant=model["constant_gain"])


def coherent_main():
    """One required consistency repair, not a tuned set of alternative models."""
    start = time.process_time()
    assert (OUT / "report.json").exists()
    assert not (OUT / "coherent_predictions.json").exists()
    write("coherent_config.json", dict(
        reason="first forecast violates count capacities in some target classes",
        change="derive baseline and edited counts jointly from four mask-membership atoms per frozen bin",
        scope="same sources, folds, frozen cuts, masks and correct-edit estimates as first forecast",
        target_labels_in_predictor=False, parameter_search=False,
        qualification="retrospective research forecast fitted with historical query labels; not a segmentation method"))
    coco = combine([pack(s) for s in ("rcg2_groupB", "rcg2_groupC", "rcg2_group_groupD")])
    target = dict(SUIM=pack("rcg2_group_suim"), PASCAL_Part=pack("rcg2_grouppascal_part"))
    for name, slug in (("PACO_Part", "paco_part"), ("LVIS", "lvis")):
        target[name] = combine([pack(f"rcg2_group_{slug}_f{f}") for f in range(4)], f"rcg2_group_{slug}")
    forecasts, aggregates, models = {}, {}, {}
    m = fit_coherent(coco)
    for name, data in target.items():
        aggregates[name], models[name] = aggregate(data), m
        forecasts[name] = coherent_forecast(data, m)
    folds = np.array([r["fold"] for r in coco["rows"]])
    groups = photo_groups(coco["rows"])
    for f in range(4):
        held = folds == f
        train = ~held & ~np.isin(groups, groups[held])
        name, m = f"COCO_fold{f}", fit_coherent(coco, train)
        aggregates[name], models[name] = aggregate(coco, held), m
        forecasts[name] = coherent_forecast(coco, m, held)
    prior = json.loads((OUT / "predictions.json").read_text())
    for name, p in forecasts.items():
        assert np.allclose(p["correct"], prior[name]["correct"], rtol=0, atol=2e-8)
    write("coherent_predictions.json", forecasts)
    seal = hashlib.sha256((OUT / "coherent_predictions.json").read_bytes()).hexdigest()
    results = {name: evaluate(aggregates[name], p, models[name]) for name, p in forecasts.items()}
    assert all(not r["invalid_prediction_classes"] for r in results.values())
    record(Path(__file__).resolve())
    write("coherent_report.json", dict(results=results, sources=SOURCES, prediction_sha256=seal,
          local_process_cpu_seconds=time.process_time() - start,
          checks="all predicted count constraints pass; correct-edit forecasts equal original exactly to roundoff"))
    print(json.dumps({name: {k: r[k] for k in ("actual_gain", "predicted_gain", "known_baseline_prediction",
          "invalid_prediction_classes")} for name, r in results.items()}, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--coherent"]:
        coherent_main()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: forecast_transfer_from_counts.py [--coherent]")
