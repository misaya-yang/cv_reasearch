"""Score the sealed complete composition with official slots and strong controls."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import sol_bg_admission_composition as experiment
from insid3_factor_evidence import components
from ics.metrics import summarize
from ics.official_data import array_hash
from edit_balance import summarize_edits

core, OUT = experiment.core, experiment.OUT
PACO = core.MAIN / "paco_remaining500"
COCO = core.COCO


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def stats(pred, gt):
    tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
    return {"I": tp, "U": tp + fp + fn, "TP": tp, "FP": fp, "FN": fn,
            "GT": int(gt.sum()), "IoU": 100 * tp / max(tp + fp + fn, 1)}


def edits(pred, base, gt):
    return [int((pred & ~base & gt).sum()), int((pred & ~base & ~gt).sum()),
            int((~pred & base & gt).sum()), int((~pred & base & ~gt).sum())]


def connected_intervals(rows, source_rows, expected, baselines, repetitions=2000):
    identities = components([{ "id": r["row"]["episode_id"], "dataset": r["row"]["dataset"], "row": r["row"]} for r in source_rows])
    group_names = sorted({identities[r["episode_id"]] for r in rows})
    arms = sorted(rows[0]["iu"])
    if len(group_names) < 2:
        return {"photo_components": len(group_names), "ci95": None, "reason": "One connected R/Q group; not estimable."}
    lookup = {g: i for i, g in enumerate(group_names)}
    groups = np.array([lookup[identities[r["episode_id"]]] for r in rows])
    rng = np.random.RandomState(0)
    weights = rng.multinomial(len(group_names), np.ones(len(group_names)) / len(group_names), size=repetitions)[:, groups]
    declared = {str(f): list(map(str, ids)) for f, ids in expected.items()}
    samples = np.zeros((repetitions, len(arms)))
    for fold, ids in declared.items():
        for cls in ids:
            selected = [i for i, r in enumerate(rows) if str(r["fold"]) == fold and str(r["class_id"]) == cls]
            if not selected:
                continue
            iu = np.asarray([[rows[i]["iu"][a] for a in arms] for i in selected], dtype=np.float64)
            pooled = (weights[:, selected] @ iu.reshape(len(selected), -1)).reshape(repetitions, len(arms), 2)
            samples += (100 / (len(declared) * len(ids))) * pooled[:, :, 0] / np.maximum(pooled[:, :, 1], 1)
    result = {"photo_components": len(group_names), "largest_component_episodes": int(np.bincount(groups).max()),
              "repetitions": repetitions, "unit": "connected reference/query photo groups across the fixed dataset cohort",
              "scope": "Conditional fixed-cohort cluster bootstrap; absent resampled official slots remain zero, not unseen-photo validation.",
              "contrasts": {}}
    for base in baselines:
        b = arms.index(base)
        result["contrasts"][base] = {a: np.percentile(samples[:, j] - samples[:, b], [2.5, 97.5]).tolist()
                                          for j, a in enumerate(arms)}
    return result


def main():
    core.torch.set_num_threads(2)
    start = time.monotonic()
    seal, config = core.read(OUT / "sealed.json"), core.read(OUT / "config.json")
    assert seal["state"] == "ALL900_COMPOSITION_PREDICTIONS_SEALED" and seal["n"] == 900
    assert seal["config_sha256"] == core.sha(OUT / "config.json")
    assert seal["manifest_sha256"] == core.sha(OUT / "manifest.json")
    for path, digest in config["source_sha256"].items():
        assert core.sha(path) == digest
    sources = core.read(OUT / "manifest.json")
    paco_cfg, paco_index = core.read(PACO / "config.json"), core.read(PACO / "source_index.json")
    assert core.sha(PACO / "source_index.json") == paco_cfg["source_index_sha256"]
    coco_cfg = core.read(COCO / "config.json")
    old_metrics = {r["episode_id"]: r for r in lines(PACO / "scored_episodes.jsonl")}
    old_metrics.update({r["episode_id"]: r for r in lines(core.MAIN / "candidate_reference_v1/scored_episodes.jsonl")})
    if (COCO / "scored_episodes.jsonl").exists():
        old_metrics.update({r["episode_id"]: r for r in lines(COCO / "scored_episodes.jsonl")})
    rows, checks = [], 0
    for source, rec in zip(sources, seal["records"]):
        row = source["row"]
        eid, shape = row["episode_id"], tuple(row["query_size_hw"])
        assert eid == rec["episode_id"]
        packet = OUT / "predictions" / rec["file"]
        assert core.sha(packet) == rec["sha256"]
        assert core.sha(source["baseline"]) == source["baseline_sha256"]
        assert core.sha(source["old"]) == source["old_sha256"]
        with Image.open(row["query_mask_path"]) as image:
            binary = (np.asarray(image.convert("L")) > 0).astype(np.uint8)
        assert array_hash(binary) == row["query_mask_hash"]
        gt = core.resize(binary, shape, "nearest") > .5
        predictions = {}
        with np.load(packet, allow_pickle=False) as z:
            assert tuple(z["original_hw"]) == shape
            for arm in experiment.ARMS:
                predictions[arm] = core.unpack(z[arm], shape)
        with np.load(source["old"], allow_pickle=False) as z:
            predictions["anchor.global"] = core.unpack(z["anchor.global"], shape)
        with np.load(source["baseline"], allow_pickle=False) as z:
            predictions["foris.crf"] = core.unpack(z["original/" + source["baseline_key"]], shape)
            if row["dataset"] == "coco":
                predictions["mean"] = core.unpack(z["original/mean.archived"], shape)
                predictions["task.competitive"] = core.unpack(z["original/task.matched"], shape)
            else:
                predictions["region.fast"] = core.unpack(z["original/region.fast"], shape)
                predictions["task.competitive"] = core.unpack(z["original/task.competitive"], shape)
                if row["dataset"] == "deepglobe_road":
                    predictions["mean"] = core.unpack(z["original/mean"], shape)
        if row["dataset"] == "paco_part":
            mean = paco_index[eid]["mean"]
            assert core.sha(mean["path"]) == mean["sha256"]
            with np.load(mean["path"], allow_pickle=False) as z:
                predictions["mean"] = core.unpack(z["original/mean"], shape)
            checks += 1
        checks += 5
        metrics = {a: stats(p, gt) for a, p in predictions.items()}
        baseline, old, full, control = [predictions[a] for a in ("foris.crf", "anchor.global", *experiment.ARMS)]
        assert np.array_equal(full & baseline, old & baseline)
        assert not np.any(full & ~old)
        assert int((full & ~baseline).sum()) == int((control & ~baseline).sum()) == rec["accepted_added_pixels"]
        assert np.array_equal(control & baseline, old & baseline)
        rejected = old & ~full
        rtp, rfp = int((rejected & gt).sum()), int((rejected & ~gt).sum())
        assert rtp + rfp == rec["rejected_old_additions"] and not np.any(rejected & baseline)
        assert metrics["full_composition"]["I"] == metrics["anchor.global"]["I"] - rtp
        assert metrics["full_composition"]["U"] == metrics["anchor.global"]["U"] - rfp
        checks += 7
        accounting = {a: {b: edits(predictions[a], predictions[b], gt) for b in ("foris.crf", "anchor.global", "same_added_area_source_order")}
                      for a in ("full_composition", "same_added_area_source_order", "anchor.global")}
        for a, baselines in accounting.items():
            for b, e in baselines.items():
                at, af, dt, df = e
                assert metrics[a]["I"] == metrics[b]["I"] + at - dt
                assert metrics[a]["U"] == metrics[b]["U"] + af - df
                checks += 1
        previous = old_metrics.get(eid)
        if previous:
            for arm in ("foris.crf", "anchor.global", "mean"):
                if arm in previous["iu"]:
                    assert previous["iu"][arm] == [metrics[arm]["I"], metrics[arm]["U"]], (eid, arm)
                    checks += 1
        rows.append({"episode_id": eid, "dataset": row["dataset"], "fold": row["fold"],
                     "class_id": row["loader_class_id"], "query_photo_id": row["query_photo_id"],
                     "reference_photo_id": row["reference_photo_id"], "metrics": metrics,
                     "iu": {a: [m["I"], m["U"]] for a, m in metrics.items()}, "edits": accounting,
                     "constraint_triggered": rec["threshold_constraint_triggered"],
                     "rejected_old_additions": {"TP": rtp, "FP": rfp}, "inference_record": rec})
    core.write(OUT / "scored_episodes.json", rows)
    datasets = {}
    for dataset in ("deepglobe_road", "coco", "paco_part"):
        sub = [r for r in rows if r["dataset"] == dataset]
        expected = paco_cfg["expected_classes"] if dataset == "paco_part" else coco_cfg["expected_classes"] if dataset == "coco" else {"0": [0]}
        bases = [a for a in ("foris.crf", "anchor.global", "same_added_area_source_order", "mean", "region.fast", "task.competitive") if a in sub[0]["iu"]]
        summary = summarize(sub, baselines=tuple(bases), expected_classes=expected, repetitions=10000, unit="query_photo")
        connected = connected_intervals(sub, [s for s in sources if s["row"]["dataset"] == dataset], expected, bases)
        balances = {base: summarize_edits(sub, base, "full_composition", expected_classes=expected)
                    for base in ("foris.crf", "anchor.global", "same_added_area_source_order")}
        for base, balance in balances.items():
            assert abs(balance["delta_pp"]["full"] - summary["paired"][base]["full_composition"]["delta_pp"]) < 1e-10
        datasets[dataset] = {"summary": summary, "connected_RQ_photo_intervals": connected, "edit_balance": balances,
                             "constraint_triggered": sum(r["constraint_triggered"] for r in sub),
                             "outputs_changed": sum(r["inference_record"]["rejected_old_additions"] > 0 for r in sub),
                             "rejected_TP": sum(r["rejected_old_additions"]["TP"] for r in sub),
                             "rejected_FP": sum(r["rejected_old_additions"]["FP"] for r in sub),
                             "old_baseline_fg_deletions_unchanged": True, "same_added_area_control_identity": True}
    known = {"deepglobe_road": 29.378, "coco": 63.358900092540765, "paco_part": 38.4608684255599}
    for d, value in known.items():
        assert abs(datasets[d]["summary"]["miou"]["foris.crf"] - value) < .001
    result = {"state": "COMPLETE_SCORED900_COMPOSITION", "datasets": datasets, "checks": checks,
              "seal_sha256": core.sha(OUT / "sealed.json"), "scored_records_sha256": core.sha(OUT / "scored_episodes.json"),
              "scorer_sha256": core.sha(__file__), "metrics_helper_sha256": core.sha(REPO / "src/ics/metrics.py"),
              "edit_helper_sha256": core.sha(REPO / "scripts/edit_balance.py"), "seconds": time.monotonic() - start,
              "new_encoder_calls": 0, "new_reference_fits": 0, "frame": "original",
              "metric": "PACO303/COCO40 official fold-class slots including absent zeros; Deep totalI/U",
              "exposure": config["exposure"], "mean_coco_note": "Same-input archived complete MEAN output, not a new current-raw replay."}
    core.write(OUT / "report.json", result)
    print(json.dumps({"state": result["state"], "checks": checks, "seconds": result["seconds"],
                      "datasets": {d: {"scores": r["summary"]["miou"], "primary_deltas": {b: r["summary"]["paired"][b]["full_composition"] for b in r["summary"]["paired"]},
                                         "triggers": r["constraint_triggered"], "changed": r["outputs_changed"], "rejected_TP": r["rejected_TP"], "rejected_FP": r["rejected_FP"]}
                                   for d, r in datasets.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
