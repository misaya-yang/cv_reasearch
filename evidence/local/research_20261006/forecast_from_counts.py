"""Local arithmetic on old sufficient counts; no images, inference or server access.

One declared retrospective check: remove held-class baseline I/U from the supplied
edit-purity forecaster. Keep the segmentation method and all masks unchanged.
Historical labels fit the forecast, never a new segmentation component.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "evidence/local/research_20261005"
OUT = Path(__file__).parent / "edit_forecast_existing_counts"
KINDS = ("add:hole", "add:near", "add:far", "del:component", "del:near", "del:interior")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    start = time.process_time()
    count_path = OLD / "rcg_anatomy4000/counts.npz"
    state_path = OLD / "pipeline_verified/rcg_native_states4000_v1/states.npz"
    manifest_path = state_path.parent / "manifest.json"
    rows = json.loads(manifest_path.read_text())
    with np.load(count_path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in KINDS + ("iu:pre", "iu:native", "iu:rcg")}
    with np.load(state_path, allow_pickle=False) as z:
        assert all(np.array_equal(data[k], z[v]) for k, v in (
            ("iu:pre", "foris_pre.control"), ("iu:native", "native"), ("iu:rcg", "rcg")))
        states = z["counts"].sum(axis=1)
    assert len(rows) == len(states) == 4000
    assert np.all(states.sum(1) == 1024 ** 2)
    # Marginalizing the truth bit leaves a mask-only observable. Any swap of the
    # truth=0/1 cells with fixed prediction bits leaves mask_area unchanged.
    mask_area = states[:, (np.arange(16) & 1) != 0].sum(1)
    volume = np.stack([data[k].sum(1) for k in KINDS], 1)
    correct = np.stack([data[k][:, 0 if i < 3 else 1] for i, k in enumerate(KINDS)], 1)
    a, d, w = volume[:, :3].sum(1), volume[:, 3:].sum(1), correct.sum(1)
    assert np.array_equal(data["iu:pre"][:, 0] + w - d, data["iu:rcg"][:, 0])
    assert np.array_equal(data["iu:pre"][:, 1] + a - w, data["iu:rcg"][:, 1])
    cls = np.array([r["c"] for r in rows])
    assert all(r["fold"] == r["c"] % 4 for r in rows)
    ids = np.unique(cls)
    assert np.array_equal(ids, np.arange(80))
    sums = lambda array: np.stack([array[cls == c].sum(0) for c in ids])
    v, good, m = sums(volume), sums(correct), sums(mask_area)
    base, changed = sums(data["iu:pre"]), sums(data["iu:rcg"])
    add, delete = v[:, :3].sum(1), v[:, 3:].sum(1)
    actual = 100 * (changed[:, 0] / changed[:, 1] - base[:, 0] / base[:, 1])
    label_positive = np.array([int(x[0]) * int(y[1]) > int(y[0]) * int(x[1])
                               for x, y in zip(changed, base)])

    OUT.mkdir(parents=True, exist_ok=False)
    config = dict(
        question="Does the old edit-purity forecast retain value without held-class baseline I/U?",
        sampling="same old PUBLIC4000, 80 classes; train three class folds, hold one",
        method="RCG versus FoRIS pre-CRF at 1024; masks unchanged",
        purity="pooled correct pixel fraction per six observable edit types in other three folds",
        baseline_estimator="I_hat=M*sum_train(I)/sum_train(M); U_hat=M*sum_train(U)/sum_train(M)",
        controls=["always predict a gain", "other-three-fold mean class gain", "actual held baseline I/U diagnostic"],
        held_observables=["six edit volumes", "baseline predicted mask area", "class/fold metadata"],
        held_labels_in_primary_prediction=False,
        history_labels_used_for_forecaster=True,
        exposure="retrospective, reused development data; not prospective validation",
        no_segmentation_inference=True,
        no_parameter_search=True,
        no_server_or_gpu_compute=True,
        local_work="small arithmetic on existing compact count arrays only",
        sources={str(p.relative_to(ROOT)): digest(p) for p in (count_path, state_path, manifest_path)},
    )
    write("config.json", config)
    predictions = []
    for fold in range(4):
        train, test = ids % 4 != fold, ids % 4 == fold
        assert np.all(v[train].sum(0) > 0)
        purity = good[train].sum(0) / v[train].sum(0)
        precision = base[train, 0].sum() / m[train].sum()
        union_ratio = base[train, 1].sum() / m[train].sum()
        constant = float(actual[train].mean())
        for c in ids[test]:
            estimated_w = float(v[c] @ purity)
            ih, uh = float(m[c] * precision), float(m[c] * union_ratio)
            new_i, new_u = ih + estimated_w - float(delete[c]), uh + float(add[c]) - estimated_w
            assert uh > 0 and new_u > 0
            predicted = 100 * (new_i / new_u - ih / uh)
            margin0 = estimated_w - float(delete[c])
            margin1 = 2 * estimated_w - float(add[c] + delete[c])
            j_free = ("win" if min(margin0, margin1) > 0 else
                      "not_win" if max(margin0, margin1) <= 0 else "undetermined")
            final_area = float(m[c] + add[c] - delete[c])
            physically_valid = (0 <= new_i <= min(new_u, final_area) and new_u >= final_area)
            predictions.append(dict(
                c=int(c), fold=fold, add=int(add[c]), delete=int(delete[c]),
                baseline_mask_area=int(m[c]), estimated_correct_edits=estimated_w,
                estimated_baseline_I=ih, estimated_baseline_U=uh,
                predicted_gain=predicted, constant_gain_control=constant,
                estimated_baseline_J=ih / uh,
                predicted_count_constraints_valid=bool(physically_valid),
                j_free_prediction=j_free,
                purity_estimates=purity.tolist(),
            ))
    predictions.sort(key=lambda r: r["c"])
    write("predictions.json", predictions)
    prediction_seal = digest(OUT / "predictions.json")

    # All primary forecasts above are fixed before this held-label scoring stage.
    diagnostic = np.zeros(80)
    for fold in range(4):
        train, test = ids % 4 != fold, ids % 4 == fold
        purity = good[train].sum(0) / v[train].sum(0)
        wh = v[test] @ purity
        i, u = base[test, 0], base[test, 1]
        diagnostic[test] = 100 * ((i + wh - delete[test]) / (u + add[test] - wh) - i / u)

    def summary(values):
        values = np.asarray(values)
        pred_positive = values > 0
        return dict(predicted_mean=float(values.mean()), actual_mean=float(actual.mean()),
                    class_MAE=float(np.abs(values - actual).mean()),
                    sign_correct=int((pred_positive == label_positive).sum()),
                    predicted_positive=int(pred_positive.sum()),
                    actual_positive=int(label_positive.sum()),
                    true_positive=int((pred_positive & label_positive).sum()),
                    false_positive=int((pred_positive & ~label_positive).sum()),
                    true_negative=int((~pred_positive & ~label_positive).sum()),
                    false_negative=int((~pred_positive & label_positive).sum()),
                    folds=[dict(fold=f, predicted=float(values[ids % 4 == f].mean()),
                                actual=float(actual[ids % 4 == f].mean())) for f in range(4)])

    point = summary([r["predicted_gain"] for r in predictions])
    constant = summary([r["constant_gain_control"] for r in predictions])
    kept = [r for r in predictions if r["j_free_prediction"] != "undetermined"]
    report = dict(
        primary=point, constant_gain_control=constant,
        known_held_baseline_diagnostic=summary(diagnostic),
        always_win_control=dict(sign_correct=int(label_positive.sum()), classes=80),
        j_free_point_model=dict(
            scope="robust over J in [0,1] only; historical purity uncertainty is not bounded",
            decided=len(kept), correct=sum((r["j_free_prediction"] == "win") == bool(label_positive[r["c"]]) for r in kept),
            predicted_wins=sum(r["j_free_prediction"] == "win" for r in kept),
            false_win_classes=[r["c"] for r in kept if r["j_free_prediction"] == "win" and not label_positive[r["c"]]],
        ),
        invalid_predicted_count_classes=[r["c"] for r in predictions if not r["predicted_count_constraints_valid"]],
        source_alignment="all 4000 native/pre/RCG I/U arrays equal the manifest-bound state archive",
        exact_edit_identity_mismatches=0,
        prediction_sha256=prediction_seal,
        local_process_cpu_seconds=time.process_time() - start,
        primary_does_not_use_held_baseline_IU=True,
        scope="old-count retrospective arithmetic; no segmentation method change or new inference",
    )
    write("report.json", report)
    write("per_class_evaluation.json", [dict(predictions[c], actual_gain=float(actual[c]),
        actual_baseline_J=float(base[c, 0] / base[c, 1]),
        labelled_baseline_prediction=float(diagnostic[c])) for c in ids])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
