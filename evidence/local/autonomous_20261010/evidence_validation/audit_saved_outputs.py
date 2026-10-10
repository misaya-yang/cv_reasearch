"""Independent saved-output audit. No feature extraction or new inference.

Only binary predictions, frozen manifests, ground-truth scoring masks and
per-episode I/U ledgers are read. No candidate output is created or modified.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SIGNAL = REPO / "evidence/local/difficult_region_signal_20261010"
COCO = REPO / "evidence/local/parallel_signal_20261010/02_task_identity/ridge_coco_validation"
PILOT = REPO.parent / "cv_data/a/joint_role_pilot200_20261010"
HASHED = {}
MASK_CHECKS = Counter()
FILE_IDENTITIES = {}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    path = Path(path)
    if path not in HASHED:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        HASHED[path] = h.hexdigest()
    return HASHED[path]


def checked_file(path, expected):
    actual = sha(path)
    assert actual == expected, (str(path), "file SHA mismatch", expected, actual)
    FILE_IDENTITIES[str(Path(path).resolve())] = actual


def gt_for(row):
    with Image.open(row["query_mask_path"]) as image:
        truth = (np.asarray(image.convert("L")) > 0).astype(np.uint8)
    h = hashlib.sha256(json.dumps([list(truth.shape), truth.dtype.str]).encode())
    h.update(np.ascontiguousarray(truth).tobytes())
    assert h.hexdigest() == row["query_mask_hash"], row["episode_id"]
    assert tuple(truth.shape) == tuple(row["query_size_hw"]), row["episode_id"]
    MASK_CHECKS["ground_truth_identity"] += 1
    return truth.astype(bool)


def check_mask(packed, gt, expected_iu, label, baseline=None, expected_edits=None):
    pred = np.unpackbits(packed, count=gt.size).reshape(gt.shape).astype(bool)
    iu = [int(np.count_nonzero(pred & gt)), int(np.count_nonzero(pred | gt))]
    assert iu == expected_iu, (label, iu, expected_iu)
    MASK_CHECKS["original_mask_IU"] += 1
    if baseline is not None and expected_edits is not None:
        add, delete = pred & ~baseline, ~pred & baseline
        edits = [int(np.count_nonzero(add & gt)), int(np.count_nonzero(add & ~gt)),
                 int(np.count_nonzero(delete & gt)), int(np.count_nonzero(delete & ~gt))]
        assert edits == expected_edits, (label, edits, expected_edits)
        MASK_CHECKS["original_mask_edit_ledger"] += 1
    return pred


def point_and_intervals(rows, expected_classes=None, pairs=()):
    """Independent per-class pooling, equal fold mean, paired photo bootstrap."""
    arms = sorted(set.intersection(*(set(r["iu"]) for r in rows)))
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["fold"]), str(row["class_id"]))].append(row)
    if expected_classes is None:
        classes = {f: sorted(c for ff, c in groups if ff == f)
                   for f in sorted({f for f, c in groups})}
    else:
        classes = {str(f): sorted(map(str, cs)) for f, cs in expected_classes.items()}
    folds, pooled_classes, per_class, bootstrap_groups = {}, {}, {}, {}
    for f, cs in classes.items():
        folds[f] = {a: 0. for a in arms}
        for c in cs:
            group = groups.get((f, c), [])
            iu = {a: [sum(r["iu"][a][0] for r in group),
                      sum(r["iu"][a][1] for r in group)] for a in arms}
            value = {a: 100 * v[0] / max(v[1], 1) for a, v in iu.items()}
            per_class[f + ":" + c] = value
            pooled_classes[f + ":" + c] = iu
            for a in arms:
                folds[f][a] += value[a] / len(cs)
            photos = defaultdict(list)
            for r in group:
                photos[str(r["query_photo_id"])].append(r)
            if photos:
                bootstrap_groups[(f, c)] = np.asarray([
                    [[sum(r["iu"][a][j] for r in photos[p]) for j in (0, 1)]
                     for a in arms] for p in sorted(photos)], dtype=np.float64)
    point = {a: sum(v[a] for v in folds.values()) / len(folds) for a in arms}
    paired = {}
    if pairs:
        rng = np.random.RandomState(0)
        draws = np.zeros((10000, len(arms)), dtype=np.float64)
        for (f, c), x in sorted(bootstrap_groups.items()):
            n = len(x)
            weight = 100 / (len(folds) * len(classes[f]))
            for start in range(0, 10000, 1000):
                multiplicity = rng.multinomial(n, np.full(n, 1 / n), size=1000)
                counts = (multiplicity @ x.reshape(n, -1)).reshape(1000, len(arms), 2)
                draws[start:start + 1000] += weight * counts[:, :, 0] / np.maximum(counts[:, :, 1], 1)
        for a, b in pairs:
            diff = draws[:, arms.index(a)] - draws[:, arms.index(b)]
            paired[a + " - " + b] = dict(delta_pp=point[a] - point[b],
                ci95_pp=np.quantile(diff, [.025, .975]).tolist())
    return dict(n=len(rows), class_count=sum(map(len, classes.values())),
                observed_class_count=len(groups),
                query_photos=len({str(r["query_photo_id"]) for r in rows}),
                miou=point, per_fold=folds, per_class=per_class,
                pooled_class_IU=pooled_classes, paired=paired,
                interval_scope="10,000 paired query-photo bootstrap samples within fold/class; conditional on exposed image pool")


def compare_points(result, reported):
    difference = {a: result["miou"][a] - reported[a] for a in result["miou"]}
    assert max(map(abs, difference.values())) < 1e-10, difference
    result["max_reported_point_difference"] = max(map(abs, difference.values()))


def verify_uniform(rows):
    root = SIGNAL / "uniform_controls"
    seal, cfg, inputs = (read(root / f) for f in ("sealed.json", "config.json", "inputs.json"))
    checked_file(root / "config.json", seal["config_sha256"])
    checked_file(root / "inputs.json", cfg["inputs_sha256"])
    index = {r["episode_id"]: r for r in rows}
    paco_sources = read(SIGNAL / "paco_remaining500/source_index.json")
    for item, rec in zip(inputs, seal["receipts"]):
        row = item["row"]
        assert row["episode_id"] == rec["episode_id"]
        ledger = index[row["episode_id"]]
        assert ledger["fold"] == row["fold"] and ledger["class_id"] == row["loader_class_id"]
        gt = gt_for(row)
        checked_file(item["baseline_path"], item["baseline_sha256"])
        with np.load(item["baseline_path"], allow_pickle=False) as z:
            baseline = check_mask(z["original/foris.crf"], gt, ledger["iu"]["foris.crf"], row["episode_id"])
            for key in z.files:
                if key.startswith("original/") and key.split("/", 1)[1] in ledger["iu"]:
                    a = key.split("/", 1)[1]
                    if a == "foris.crf":
                        continue
                    check_mask(z[key], gt, ledger["iu"][a], row["episode_id"] + "/" + a,
                               baseline, ledger["edits"].get(a))
        checked_file(item["ridge_prediction_path"], item["ridge_prediction_sha256"])
        with np.load(item["ridge_prediction_path"], allow_pickle=False) as z:
            for a in ("anchor.global", "anchor.local4"):
                check_mask(z[a], gt, ledger["iu"][a], row["episode_id"] + "/" + a,
                           baseline, ledger["edits"][a])
        if row["dataset"] == "paco_part":
            mean = paco_sources[row["episode_id"]]["mean"]
            checked_file(mean["path"], mean["sha256"])
            with np.load(mean["path"], allow_pickle=False) as z:
                check_mask(z["original/mean"], gt, ledger["iu"]["mean"], row["episode_id"] + "/mean",
                           baseline, ledger["edits"].get("mean"))
        p = root / "predictions" / rec["filename"]
        checked_file(p, rec["sha256"])
        with np.load(p, allow_pickle=False) as z:
            for a in rec["amplitudes"]:
                check_mask(z[a], gt, ledger["iu"][a], row["episode_id"] + "/" + a,
                           baseline, ledger["edits"][a])
                assert ledger["edits"][a][:2] == [0, 0]
    assert len(inputs) == len(seal["receipts"]) == len(rows) == 700


def verify_coco(rows):
    seal, cfg, manifest, sources = (read(COCO / f) for f in
        ("sealed.json", "config.json", "manifest.json", "source_index.json"))
    checked_file(COCO / "config.json", seal["config_sha256"])
    checked_file(COCO / "manifest.json", cfg["manifest_sha256"])
    checked_file(COCO / "source_index.json", cfg["source_index_sha256"])
    index = {r["episode_id"]: r for r in rows}
    archived_root = REPO.parent / "cv_data/a/coco_role_competition200_20261010"
    archived = {r["episode_id"]: r for r in lines(archived_root / "inference.jsonl")}
    for row, rec in zip(manifest, seal["receipts"]):
        eid = row["episode_id"]
        assert eid == rec["episode_id"]
        gt, ledger, source = gt_for(row), index[eid], sources[eid]
        checked_file(source["baseline_path"], source["baseline_sha256"])
        with np.load(source["baseline_path"], allow_pickle=False) as z:
            baseline = check_mask(z[source["baseline_key"]], gt, ledger["iu"]["foris.crf"], eid + "/foris")
        old = archived[eid]
        old_path = archived_root / "predictions" / old["filename"]
        checked_file(old_path, old["prediction_sha256"])
        with np.load(old_path, allow_pickle=False) as z:
            for a in ("foris.crf", "mean", "task.competitive"):
                check_mask(z["original/" + a], gt, ledger["iu"][a], eid + "/" + a,
                           baseline, ledger["edits"][a])
        p = COCO / "predictions" / rec["filename"]
        checked_file(p, rec["prediction_sha256"])
        checked_file(COCO / "fields" / rec["filename"], rec["fields_sha256"])
        with np.load(p, allow_pickle=False) as z:
            check_mask(z["anchor.global"], gt, ledger["iu"]["anchor.global"], eid + "/anchor.global",
                       baseline, ledger["edits"]["anchor.global"])
    assert len(manifest) == len(seal["receipts"]) == len(rows) == 200


def verify_pilot(rows):
    cfg, seal, manifest = (read(PILOT / f) for f in ("config.json", "sealed.json", "manifest.json"))
    for f, k in (("config.json", "config_sha256"), ("manifest.json", "manifest_sha256"),
                 ("inference.jsonl", "inference_sha256")):
        checked_file(PILOT / f, seal[k])
    for p, digest in cfg["frozen_sources"].items():
        checked_file(p, digest)
    records = {r["episode_id"]: r for r in lines(PILOT / "inference.jsonl")}
    index = {r["episode_id"]: r for r in rows}
    for row in manifest:
        eid = row["episode_id"]
        gt, rec, ledger = gt_for(row), records[eid], index[eid]
        p = PILOT / "predictions" / rec["filename"]
        checked_file(p, rec["prediction_sha256"])
        with np.load(p, allow_pickle=False) as z:
            baseline = np.unpackbits(z["original/foris.crf"], count=gt.size).reshape(gt.shape).astype(bool)
            for a, iu in ledger["iu"]["original"].items():
                check_mask(z["original/" + a], gt, iu, eid + "/" + a,
                           baseline, ledger["edits"]["original"][a])
    assert len(manifest) == len(records) == len(rows) == 200
    return dict(executed_wrapper_sha256=sha(PILOT / "frozen/scripts/joint_role_pilot.py"),
                current_wrapper_sha256=sha(REPO / "scripts/joint_role_pilot.py"),
                current_matches_executed_wrapper=(sha(REPO / "scripts/joint_role_pilot.py") ==
                                                sha(PILOT / "frozen/scripts/joint_role_pilot.py")))


def main():
    start = time.monotonic()
    uniform = lines(SIGNAL / "uniform_controls/scored_episodes.jsonl")
    coco = lines(COCO / "scored_episodes.jsonl")
    pilot = lines(PILOT / "score/scored_episodes.jsonl")
    paco_cfg = read(SIGNAL / "paco_remaining500/config.json")
    report = read(SIGNAL / "uniform_controls/report.json")["groups"]
    groups = {}
    for ds in ("deepglobe_road", "paco_part"):
        rr = [r for r in uniform if r["dataset"] == ds]
        result = point_and_intervals(rr, paco_cfg["expected_classes"] if ds == "paco_part" else None,
            pairs=(("anchor.global", "foris.crf"), ("anchor.global", "uniform.matched_global"),
                   ("anchor.local4", "uniform.matched_local4")))
        compare_points(result, report[ds]["miou"])
        groups["uniform/" + ds] = result
    result = point_and_intervals(coco, read(COCO / "config.json")["expected_classes"],
        pairs=(("anchor.global", "foris.crf"), ("anchor.global", "mean"),
               ("anchor.global", "task.competitive")))
    compare_points(result, read(COCO / "report.json")["primary"]["miou"])
    groups["ridge/COCO200"] = result
    pilot_report = read(PILOT / "score/paired_results.json")["frames"]
    for ds in ("deepglobe_road", "paco_part"):
        rr = [dict(r, iu=r["iu"]["original"]) for r in pilot if r["dataset"] == ds]
        result = point_and_intervals(rr, pairs=(("role.joint", "role.unary"), ("role.joint", "foris.crf")))
        compare_points(result, pilot_report[ds]["original"]["miou"])
        groups["pilot/" + ds] = result
    manifest = read(SIGNAL / "paco_remaining500/manifest.json")
    first = [r for r in manifest if r["episode_id"] in set(paco_cfg["pilot_episode_ids"])]
    remaining = [r for r in manifest if r["episode_id"] not in set(paco_cfg["pilot_episode_ids"])]
    overlap = {}
    for key in ("query_photo_id", "reference_photo_id", "query_rgb_hash", "reference_rgb_hash"):
        a, b = {r[key] for r in first}, {r[key] for r in remaining}
        overlap[key] = dict(pilot_unique=len(a), remaining_unique=len(b), common=sorted(a & b))
    for name, rr in (("pilot100", [r for r in uniform if r["dataset"] == "paco_part" and r["episode_id"] in set(paco_cfg["pilot_episode_ids"])]),
                     ("remaining500", [r for r in uniform if r["dataset"] == "paco_part" and r["episode_id"] not in set(paco_cfg["pilot_episode_ids"])])):
        groups["PACO/" + name] = point_and_intervals(rr, paco_cfg["expected_classes"])
    print(json.dumps({k: v["miou"] for k, v in groups.items()}), flush=True)
    verify_uniform(uniform)
    print("uniform saved masks verified", flush=True)
    verify_coco(coco)
    print("COCO saved masks verified", flush=True)
    wrapper = verify_pilot(pilot)
    output = dict(state="VERIFIED", created_at=datetime.now(timezone.utc).isoformat(),
        groups=groups, PACO_overlap=overlap, original_mask_checks=dict(MASK_CHECKS),
        saved_file_identities=FILE_IDENTITIES, joint_wrapper=wrapper,
        seconds=time.monotonic() - start,
        encoder_forward=0, raw_feature_reads=0, new_segmentation_outputs=0,
        limits="Development images and labels were already exposed. This independently audits arithmetic, saved masks and provenance; it does not establish untouched cross-dataset generalization.")
    (OUT / "verified_saved_outputs.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(state=output["state"], checks=output["original_mask_checks"],
                         verified_files=len(FILE_IDENTITIES), seconds=output["seconds"])), flush=True)


if __name__ == "__main__":
    main()
