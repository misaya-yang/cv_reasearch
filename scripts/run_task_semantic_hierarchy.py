#!/usr/bin/env python3
"""Fixed cache-only five-domain semantic-granularity method and full scoring."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import builtins
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "cv_data"
ATLAS = ROOT / "evidence/local/five_dataset_case_mechanisms_20261011/integer_episodes.jsonl"
BINDINGS = ASSETS / "a/consensus_reference_calibration6380_v1_20261011/source_bindings.json"
OUT = ASSETS / "a/task_semantic_hierarchy_five1000_20261012"
BASIS = ASSETS / "native_assets/positional_basis.pt"
SPEC = ROOT / "docs/research/TASK_SEMANTIC_HIERARCHY_FIVE_20261012.md"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import opus55_reference_mask_trace as base
import run_task_context_compatibility as helpers
from ics.methods import task_semantic_hierarchy as method
from raw_feature_cache import tensor_hash
from ics.official_data import decoded_rgb, array_hash
from utils.data import build_transform

DOMAINS = ("coco", "lvis", "pascal_part", "paco_part", "suim")
ARMS = ("adaptive", "flat", "density", "null")
CONTROLS = ("FoRIS", "MEAN", "R0", "N")
FRAMES = ("cli", "original")
torch.set_num_threads(2)


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    import ics.native_basis
    return [Path(__file__), Path(method.__file__), Path(helpers.__file__), Path(base.__file__),
            ROOT / "scripts/raw_feature_cache.py", ROOT / "src/ics/official_data.py",
            Path(ics.native_basis.__file__), ASSETS / "third_party/foris_official/utils/data.py"]


def prepare(out):
    if (out / "config.json").exists():
        return verify(out)
    bindings = {b["index"]: b for b in read(BINDINGS)}
    atlas = [json.loads(line) for line in ATLAS.open()]
    manifests = []
    for domain in DOMAINS:
        chosen = [r for r in atlas if r["dataset"] == domain][:200]
        assert len(chosen) == 200
        for i, item in enumerate(chosen):
            bound = bindings[item["binding_index"]]
            assert item["episode_id"] == bound["episode_id"]
            assert domain not in ("pascal_part", "suim") or bound["cohort"] == "native2000"
            row = bound["row"]
            ref, query = bound["fit"]["reference_raw"], bound["fit"]["query_raw_identity"]
            assert all(Path(x["payload_path"]).exists() for x in (ref, query))
            controls = {name: item["prediction_sources"][name] for name in CONTROLS}
            manifests.append({"id": f"{domain}_{i:03d}", "dataset": domain,
                "atlas_index": item["binding_index"], "row": row,
                "reference_mask_sha256": base.sha(row["reference_mask_path"]),
                "raw": {"reference": ref, "query": query}, "controls": controls,
                "foris_field": item["small_field_sources"]["score64"]})
    expected_paths = [ASSETS / "a/consensus_supported_foreground2380_v2_20261011/expected_classes.json"]
    expected = read(expected_paths[0])
    for domain in ("pascal_part", "suim"):
        path = ASSETS / "a/consensus_official_pascal_suim4000_v1_20261011" / domain / "composition/expected_classes.json"
        expected_paths.append(path)
        expected[domain] = read(path)[domain]
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("fields", "predictions", "records", "frozen"):
        (out / folder).mkdir(exist_ok=True)
    for path in sources():
        shutil.copyfile(path, out / "frozen" / path.name)
    shutil.copyfile(SPEC, out / "frozen" / SPEC.name)
    base.write(out / "manifest.json", manifests)
    base.write(out / "expected_classes.json", {d: expected[d] for d in DOMAINS})
    config = {"state": "FROZEN_FIVE_DOMAIN_CACHE_ONLY_METHOD", "n": 1000,
        "domains": list(DOMAINS), "per_domain": 200, "arms": list(ARMS),
        "primary": "adaptive/cli", "frames": list(FRAMES),
        "sources": {str(p): base.sha(p) for p in sources()},
        "atlas_sha256": base.sha(ATLAS), "bindings_sha256": base.sha(BINDINGS),
        "manifest_sha256": base.sha(out / "manifest.json"),
        "expected_classes_sha256": base.sha(out / "expected_classes.json"),
        "expected_source_sha256": {str(p): base.sha(p) for p in expected_paths},
        "basis_sha256": base.sha(BASIS), "spec_snapshot_sha256": base.sha(SPEC),
        "depth": method.DEPTH, "lloyd_steps": method.LLOYD_STEPS,
        "prior_mass": method.PRIOR_MASS, "density_tau": method.TAU,
        "preprocessing": "unit native O24; same source-ridge conditional native APD then unit",
        "role_mass": "all4096 true1024 soft reference coverage, no sampling",
        "readout": "FP64 direct bilinear to each actual frame; strictly>0; no CRF/minmax",
        "exposure": "previously exposed atlas development; no independent confirmation",
        "new_encoder_calls": 0, "new_raw_writes": 0, "parameter_updates": 0,
        "query_labels": "deny all manifest query Mask paths until all8000 full masks sealed",
        "device": "mps", "precision": "FP32 geometry and similarities; CPU FP64 coding/render",
        "mps_allocation_limit_bytes": 20_000_000_000, "cpu_threads": 2}
    base.write(out / "config.json", config)
    base.write(out / "activity.json", {"state": "PREPARED", "total": 1000})
    return config


def verify(out):
    config = read(out / "config.json")
    for path, expected in config["sources"].items():
        assert base.sha(path) == expected, "Frozen source changed: " + path
        assert base.sha(out / "frozen" / Path(path).name) == expected
    for name in ("manifest", "expected_classes"):
        assert base.sha(out / (name+".json")) == config[name+"_sha256"]
    assert base.sha(BASIS) == config["basis_sha256"]
    assert base.sha(out / "frozen" / SPEC.name) == config["spec_snapshot_sha256"]
    return config


@contextmanager
def deny_query_masks(manifest, attempts):
    forbidden = {Path(r["row"]["query_mask_path"]).resolve() for r in manifest}
    saved = builtins.open, io.open, Image.open
    def wrap(original):
        def opened(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() in forbidden:
                attempts.append(str(path))
                raise RuntimeError("Query Mask access before full prediction seal")
            return original(path, *args, **kwargs)
        return opened
    builtins.open, io.open, Image.open = map(wrap, saved)
    try:
        yield
    finally:
        builtins.open, io.open, Image.open = saved


def mask1024(path):
    array = base.read_mask(path)
    return F.interpolate(torch.from_numpy(array.astype(np.float32))[None, None],
                         (1024, 1024), mode="nearest")[0, 0]


def load_inputs(item, transform):
    values, receipts, images = {}, {}, {}
    for role in ("reference", "query"):
        req, row = item["raw"][role], item["row"]
        image = decoded_rgb(ASSETS / row[role+"_path"], row.get(role+"_crop"))
        assert array_hash(np.asarray(image)) == row[role+"_rgb_hash"]
        tensor = transform(image)
        assert tensor_hash(tensor.numpy()) == req["input_sha256"]
        assert base.sha(req["entry_path"]) == req["entry_sha256"]
        entry = read(req["entry_path"])
        assert entry["input_tensor_hash"] == req["input_sha256"]
        assert entry["file_sha256"] == req["payload_sha256"]
        assert base.sha(req["payload_path"]) == req["payload_sha256"]
        with np.load(req["payload_path"], allow_pickle=False) as payload:
            raw = payload["O/24"].copy()
        assert raw.shape == (4096, 1024) and raw.dtype == np.float32
        assert tensor_hash(raw) == req["tensor_sha256"] == entry["features"]["O/24"]["tensor_sha256"]
        values[role] = torch.from_numpy(raw).to("mps")
        receipts[role] = {k: req[k] for k in ("key", "entry_sha256", "payload_sha256", "tensor_sha256", "input_sha256")}
        images[role] = image
    assert base.sha(item["row"]["reference_mask_path"]) == item["reference_mask_sha256"]
    source = base.read_mask(item["row"]["reference_mask_path"])
    assert array_hash(source.astype(np.uint8)) == item["row"]["reference_mask_hash"]
    dense = mask1024(item["row"]["reference_mask_path"])
    return values, receipts, dense, images


def render(field, shape):
    field = torch.from_numpy(np.asarray(field, dtype=np.float64).reshape(64, 64))
    dense = F.interpolate(field[None, None], tuple(shape), mode="bilinear", align_corners=False)[0, 0].numpy()
    return np.packbits((dense > 0).reshape(-1))


@torch.inference_mode()
def run(out, smoke=False):
    config = prepare(out)
    if (out / "sealed.json").exists():
        return
    assert torch.backends.mps.is_available()
    torch.mps.set_per_process_memory_fraction(20e9/torch.mps.recommended_max_memory())
    torch.manual_seed(0)
    from ics.native_basis import load_native_basis
    begin = time.perf_counter()
    basis, basis_receipt = load_native_basis(BASIS)
    projection = (torch.eye(1024)-basis@basis.T).to("mps")
    fixtures = method.check_fixtures()
    transform = build_transform(1024)
    manifest = read(out / "manifest.json")
    todo = [next(r for r in manifest if r["dataset"] == d) for d in DOMAINS] if smoke else manifest
    initialization = time.perf_counter()-begin
    start, records, driver_max = time.perf_counter(), [], 0
    attempts = []
    with deny_query_masks(manifest, attempts):
        for index, item in enumerate(todo):
            record_path = out / "records" / (item["id"]+".json")
            if record_path.exists():
                record = read(record_path)
                assert record["config_sha256"] == base.sha(out / "config.json")
                for folder in ("fields", "predictions"):
                    assert base.sha(out / folder / record["file"]) == record[folder+"_sha256"]
                records.append(record)
                driver_max = max(driver_max, record["mps_driver_sampled_bytes"])
                continue
            tick = time.perf_counter()
            raw, receipts, dense, images = load_inputs(item, transform)
            input_seconds = time.perf_counter()-tick
            r, q = (F.normalize(raw[role], dim=-1) for role in ("reference", "query"))
            nearest = F.interpolate(dense[None, None], (64, 64), mode="nearest")[0, 0].reshape(-1) > .5
            mean_q = q.mean(0)
            semantic = float(F.normalize(r[nearest].mean(0), dim=0)@mean_q/(mean_q.norm()+1e-6)) if nearest.any() else None
            apd = semantic is None or semantic < .8
            if apd:
                r, q = (F.normalize(x@projection.T, dim=-1) for x in (r, q))
            torch.mps.synchronize()
            gpu_start = time.perf_counter()
            paths, tree_records = method.shared_tree(r, q)
            coverage = F.avg_pool2d(dense[None, None], 16)[0, 0].reshape(-1).numpy()
            shifted = F.avg_pool2d(torch.roll(dense, (512, 512), (0, 1))[None, None], 16)[0, 0].reshape(-1).numpy()
            assert float(coverage.sum()) == float(shifted.sum())
            fields, task_info, tree_values = method.label_tree(paths, coverage, 4096)
            null_fields, null_info, null_values = method.label_tree(paths, shifted, 4096)
            fields["null"] = null_fields["adaptive"]
            fields["density"] = method.role_density(r, q, coverage)
            torch.mps.synchronize()
            algorithm_seconds = time.perf_counter()-gpu_start
            driver = int(torch.mps.driver_allocated_memory())
            driver_max = max(driver_max, driver)
            assert driver < config["mps_allocation_limit_bytes"]
            values = {arm: fields[arm].reshape(64, 64) for arm in ARMS}
            assert all(np.isfinite(field).all() for field in values.values())
            values.update({"paths": paths, "coverage": coverage.reshape(64, 64),
                           "shifted_coverage": shifted.reshape(64, 64), **tree_values,
                           **{"null/"+key: value for key, value in null_values.items()}})
            predictions = {}
            shapes = {"cli": (1024, 1024), "original": tuple(item["row"]["query_size_hw"])}
            for arm in ARMS:
                for frame, shape in shapes.items():
                    predictions[frame+"/"+arm] = render(fields[arm], shape)
            file = item["id"]+".npz"
            np.savez_compressed(out / "fields" / file, **values)
            np.savez_compressed(out / "predictions" / file, **predictions)
            record = {"id": item["id"], "dataset": item["dataset"], "file": file,
                "config_sha256": base.sha(out / "config.json"),
                "fields_sha256": base.sha(out / "fields" / file),
                "predictions_sha256": base.sha(out / "predictions" / file),
                "input_receipts": receipts, "mask_sha256": item["reference_mask_sha256"],
                "apd": apd, "source_semantic_score": semantic, "tree": tree_records,
                "task_info": task_info, "null_info": null_info,
                "field_ranges": {name: {"min": float(x.min()), "max": float(x.max()), "std": float(x.std())} for name, x in fields.items()},
                "query_GT_access_attempts": 0, "new_encoder_calls": 0,
                "mps_driver_sampled_bytes": driver, "input_seconds": input_seconds,
                "algorithm_seconds": algorithm_seconds, "seconds": time.perf_counter()-tick}
            base.write(record_path, record)
            records.append(record)
            if smoke or len(records) % 10 == 0:
                state = {"state": "SMOKE_INFERENCE" if smoke else "INFERENCE", "pid": os.getpid(),
                    "completed": len(records), "total": len(todo), "last_dataset": item["dataset"],
                    "seconds": round(time.perf_counter()-start, 3), "sampled_mps_driver_gib": driver_max/2**30}
                base.write(out / "activity.json", state)
                print(json.dumps(state), flush=True)
            del raw, r, q, fields, values, predictions, dense, images, paths, tree_values, null_values
            torch.mps.empty_cache()
    assert not attempts
    seal = {"state": "SMOKE_COMPLETE" if smoke else "ALL_8000_COMPLETE_MASKS_SEALED",
        "n": len(records), "n_masks": len(records)*8, "records": records,
        "config_sha256": base.sha(out / "config.json"), "parameter_updates": 0,
        "new_encoder_calls": 0, "new_raw_writes": 0, "query_GT_access_attempts": 0,
        "initialization_seconds": initialization, "seconds_this_run": time.perf_counter()-start,
        "sum_case_seconds": sum(r["seconds"] for r in records),
        "sum_input_seconds": sum(r["input_seconds"] for r in records),
        "sum_algorithm_seconds": sum(r["algorithm_seconds"] for r in records),
        "sampled_mps_driver_bytes": driver_max, "basis": basis_receipt, "fixtures": fixtures,
        "runtime": {"torch": str(torch.__version__), "device": "mps", "threads": 2}}
    base.write(out / ("smoke.json" if smoke else "sealed.json"), seal)
    if not smoke:
        assert len(records) == 1000
        base.write(out / "activity.json", {"state": "SEALED_READY_FOR_SCORING", "n": 1000})


def class_aggregate(rows, expected, name, frame, observed_only):
    totals = defaultdict(lambda: [0, 0])
    for row in rows:
        key = str(row["fold"]), str(row["class_id"])
        value = row["metrics"][frame][name]
        totals[key][0] += value["I"]; totals[key][1] += value["U"]
    folds = []
    for fold, class_ids in expected.items():
        keys = [(str(fold), str(c)) for c in class_ids if not observed_only or (str(fold), str(c)) in totals]
        if keys:
            folds.append(float(np.mean([100*totals[k][0]/max(totals[k][1], 1) for k in keys])))
    return float(np.mean(folds)) if folds else 0.


def score(out):
    verify(out)
    seal = read(out / "sealed.json")
    assert seal["state"] == "ALL_8000_COMPLETE_MASKS_SEALED" and seal["n"] == 1000
    assert seal["config_sha256"] == base.sha(out / "config.json")
    manifest = read(out / "manifest.json")
    # Validate every new packet before any query labels are opened.
    for item, record in zip(manifest, seal["records"]):
        assert item["id"] == record["id"]
        for folder in ("fields", "predictions"):
            assert base.sha(out / folder / record["file"]) == record[folder+"_sha256"]
    groups = helpers.photo_components([{"id": r["id"], "dataset": r["dataset"], "row": r["row"]} for r in manifest])
    expected = read(out / "expected_classes.json")
    names = (*ARMS, *CONTROLS)
    records, start = [], time.perf_counter()
    for item, producer in zip(manifest, seal["records"]):
        row = item["row"]
        gt0 = base.read_mask(row["query_mask_path"])
        assert array_hash(gt0.astype(np.uint8)) == row["query_mask_hash"]
        metrics, ranking, oracles = {}, {}, {}
        with np.load(out / "fields" / producer["file"]) as fields, np.load(out / "predictions" / producer["file"]) as prediction:
            controls = {}
            for name, source in item["controls"].items():
                assert base.sha(source["path"]) == source["sha256"]
                with np.load(source["path"]) as packet:
                    controls[name] = {frame: packet[source[frame+"_key"]].copy() for frame in FRAMES}
            for frame in FRAMES:
                shape = (1024, 1024) if frame == "cli" else tuple(row["query_size_hw"])
                gt = F.interpolate(torch.from_numpy(gt0.astype(np.float32))[None, None], shape,
                                   mode="nearest")[0, 0].numpy().astype(bool)
                baseline = np.unpackbits(controls["FoRIS"][frame], count=int(np.prod(shape))).reshape(shape).astype(bool)
                metrics[frame] = {}
                for name in names:
                    packed = prediction[frame+"/"+name] if name in ARMS else controls[name][frame]
                    if name in ARMS:
                        assert np.array_equal(packed, render(fields[name], shape))
                    mask = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
                    metrics[frame][name] = helpers.pixel_metrics(mask, gt, baseline)
                if frame == "cli":
                    for roi, keep in (("whole", np.ones_like(gt)), ("inside_foris", baseline)):
                        positive = (gt&keep).reshape(64, 16, 64, 16).sum((1, 3)).ravel()
                        negative = (~gt&keep).reshape(64, 16, 64, 16).sum((1, 3)).ravel()
                        ranking[roi] = {name: base.weighted_auc(fields[name].ravel(), positive, negative) for name in ARMS}
                        source = item["foris_field"]
                        assert base.sha(source["path"]) == source["sha256"]
                        with np.load(source["path"]) as packet:
                            field = packet[source["key"]].reshape(64, 64)
                        ranking[roi]["FoRIS_score"] = base.weighted_auc(field.ravel(), positive, negative)
                    for arm in ("adaptive", "flat", "density"):
                        oracles[arm] = helpers.oracle_levelset(fields[arm], gt)
        records.append({"id": item["id"], "dataset": item["dataset"], "episode_id": row["episode_id"],
            "fold": row["fold"], "class_id": row["loader_class_id"], "metrics": metrics,
            "ranking": ranking, "oracle_cli": oracles, "task_info": producer["task_info"]})
        if len(records) % 25 == 0:
            state = {"state": "SCORING", "completed": len(records), "total": 1000,
                     "seconds": round(time.perf_counter()-start, 3)}
            base.write(out / "activity.json", state)
            print(json.dumps(state), flush=True)
    summary = {}
    for domain in DOMAINS:
        rows = [r for r in records if r["dataset"] == domain]
        summary[domain] = {"n": len(rows), "observed_classes": len({(r["fold"], r["class_id"]) for r in rows}),
            "photo_components": len({groups[r["id"]] for r in rows}),
            "class_miou": {frame: {scope: {name: class_aggregate(rows, expected[domain], name, frame, scope=="observed")
                                         for name in names} for scope in ("observed", "declared_slots")} for frame in FRAMES},
            "primary_cli_minus_per_episode": {name: helpers.grouped_interval(rows,
                [r["metrics"]["cli"]["adaptive"]["iou"]-r["metrics"]["cli"][name]["iou"] for r in rows], groups)
                for name in names if name != "adaptive"},
            "ranking": {roi: {name: {"mean": float(np.mean([r["ranking"][roi][name] for r in rows if r["ranking"][roi][name] is not None])),
                                     "valid_n": sum(r["ranking"][roi][name] is not None for r in rows)}
                              for name in (*ARMS, "FoRIS_score")} for roi in ("whole", "inside_foris")},
            "oracle_cli_mean_episode_iou": {arm: float(np.mean([r["oracle_cli"][arm]["iou"] for r in rows])) for arm in ("adaptive", "flat", "density")},
            "effective_query_depth": float(np.mean([r["task_info"]["effective_query_depth"] for r in rows])),
            "edits_original": {name: {action: int(sum(r["metrics"]["original"][name]["edits_vs_foris"][action] for r in rows))
                              for action in ("addTP", "addFP", "deleteTP", "deleteFP")} for name in ARMS}}
    report = {"state": "COMPLETE", "summary": summary, "records": records,
        "seal_sha256": base.sha(out / "sealed.json"), "config_sha256": base.sha(out / "config.json"),
        "query_GT_reads": 1000, "new_encoder_calls": 0, "new_raw_writes": 0,
        "seconds": time.perf_counter()-start, "independent_confirmation": False,
        "novelty_established": False, "main_research_goal_complete": False}
    base.write(out / "report.json", report)
    lines = ["# 同一任务层级算法：五域完整开发结果", "", "各域200、共1000；先8000双帧Mask封存后GT。0新encoder。CLI为主端点，观察class/fold mIoU如下；不是完整官方测试池或独立确认。", "",
             "| 方法 | COCO | LVIS | PASCAL-Part | PACO-Part | SUIM |", "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        lines.append("| "+name+" | "+" | ".join(f"{summary[d]['class_miou']['cli']['observed'][name]:.6f}" for d in DOMAINS)+" |")
    lines += ["", "全部声明槽位、原图、逐例整数、完整level-set诊断及RQ关联组的逐例均值差区间见report.json。区间不是class mIoU区间；不按域选赢家。现有CRF后端未直接可用，本轮使用固定双帧bilinear完整读出，FoRIS/MEAN控制保持其原封存完整输出。"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n")
    base.write(out / "activity.json", {"state": "COMPLETE", "n": 1000, "new_encoder_calls": 0})
    print(json.dumps({d: summary[d]["class_miou"]["cli"]["observed"] for d in DOMAINS}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "run", "score", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.out)
    elif args.mode == "smoke": run(args.out, True)
    elif args.mode == "run": run(args.out)
    elif args.mode == "score": score(args.out)
    else: run(args.out); score(args.out)


if __name__ == "__main__":
    main()
