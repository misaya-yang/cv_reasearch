"""Read retained historical ledgers; verify aggregates without model inference."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
import json
import math
import subprocess


OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
DATA = REPO.parent / "cv_data"
sources = {}
checks = []


def read(path):
    path = path.resolve()
    content = path.read_bytes()
    sources[str(path)] = {
        "bytes": len(content), "sha256": sha256(content).hexdigest()
    }
    return content.decode()


def obj(path):
    return json.loads(read(path))


def rows(path, expected):
    result = [json.loads(line) for line in read(path).splitlines() if line.strip()]
    assert len(result) == expected, (path, len(result), expected)
    assert len({r["episode_id"] for r in result}) == expected, path
    checks.append({"name": f"{path.parent.name}/{path.name}: unique episodes", "n": expected})
    return result


def same(name, actual, reported):
    assert math.isclose(actual, reported, rel_tol=0, abs_tol=1e-10), (name, actual, reported)
    checks.append({"name": name, "recomputed": actual, "reported": reported})


def auc_mean(records, getter):
    values = [getter(r) for r in records]
    valid = [v for v in values if v is not None and math.isfinite(v)]
    return {"n_valid": len(valid), "auc": mean(valid)}


def class_fold_miou(records, getter):
    pooled = defaultdict(lambda: [0, 0])
    for r in records:
        for method, (intersection, union) in getter(r).items():
            pair = pooled[method, r["fold"], r["class_id"]]
            pair[0] += intersection
            pair[1] += union
    by_fold = defaultdict(list)
    for (method, fold, _), (intersection, union) in pooled.items():
        by_fold[method, fold].append(100 * intersection / union if union else 0)
    methods = sorted({key[0] for key in by_fold})
    return {
        method: mean(mean(v) for (m, _), v in by_fold.items() if m == method)
        for method in methods
    }


result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
          "scope": "Historical ledger aggregation only; no inference, new masks, or experiments.",
          "limitations": "AUC is recomputed from saved per-episode AUC, not raw tensors; mIoU from saved I/U, not rerendered masks. Existing exposed development sets are not independent confirmation."}

root = DATA / "a/lvis_tail_signal1400_20261009"
records = rows(root / "episodes.jsonl", 1400)
report = obj(root / "report.json")["groups"]["all1400"]["regions"]
tail = {}
for region in report:
    tail[region] = {}
    for signal, reported in report[region]["macro_weighted_AUC"].items():
        value = auc_mean(records, lambda r: r["regions"][region]["auc"][signal])
        same(f"tail/{region}/{signal}", value["auc"], reported)
        same(f"tail/{region}/{signal}/n", value["n_valid"], report[region]["n_auc_valid"])
        tail[region][signal] = value
result["tail_auc_1400"] = tail

root = DATA / "a/lvis_matched_background100_20261009"
records = rows(root / "diagnostic_episode_metrics.jsonl", 100)
report = obj(root / "report.json")["rois"]
matched = {}
for region in report:
    matched[region] = {}
    for signal in ["p_global", "p_matched"]:
        value = auc_mean(records, lambda r: r["rois"][region][signal]["auc"])
        reference = report[region]["macro"][signal]
        same(f"matched/{region}/{signal}", value["auc"], reference["auc"])
        same(f"matched/{region}/{signal}/n", value["n_valid"], reference["valid_episodes"])
        matched[region][signal] = value
result["matched_background_auc_100"] = matched
calibration = obj(DATA / "a/lvis_matched_background_calibration100_20261009/report.json")
result["matched_background_complete_output_report"] = {
    frame: value["miou"] for frame, value in calibration["frames"].items()
}

root = DATA / "a/lvis_reference_discriminant100_20261009"
records = rows(root / "source_transfer_episodes.jsonl", 100)
report = obj(root / "source_transfer_diagnostics.json")
transfer = {}
for region, reference in report["groups"]["all100"]["regions"].items():
    transfer[region] = {}
    for signal, reported in reference["AUC"].items():
        value = auc_mean(records, lambda r: r["regions"][region]["AUC"][signal])
        same(f"transfer/{region}/{signal}", value["auc"], reported)
        same(f"transfer/{region}/{signal}/n", value["n_valid"], reference["n_valid"])
        transfer[region][signal] = value
ratio = median(r["regularized_reference_objective_fisher"] /
               r["regularized_reference_objective_raw"] for r in records)
same("transfer/median_reference_objective_ratio", ratio, report["median_reference_objective_ratio"])
result["reference_transfer_100"] = {"reference_objective_ratio_median": ratio, "query_auc": transfer}

root = DATA / "a/lvis_resolution_loss1400_20261009"
records = rows(root / "episodes.jsonl", 1400)
report = obj(root / "report.json")["groups"]["all1400"]
zero = Counter()
for r in records:
    intersections = r["intersections"]
    if intersections["mean"] != 0:
        continue
    if intersections["mean.rank"] > 0:
        zero["lost_at_graph_after_rank_hit"] += 1
    elif intersections["foris.pre"] > 0:
        zero["lost_at_rank_after_pre_hit"] += 1
    elif intersections["foris.fg"] > 0:
        zero["lost_in_frontend_after_FG_hit"] += 1
    else:
        zero["no_FG_prefix_hit"] += 1
for key, reported in report["zero_mean_breakdown"].items():
    same(f"resolution/zero/{key}", zero[key], reported)
fn_pixels = sum(r["regions"]["mean_final_FN"]["pixels"] for r in records)
pure_fn = sum(r["regions"]["mean_final_FN"]["token_mass_query_cov_ge_09"] for r in records)
without_pure = sum(r["query_token_geometry"]["n_pure_ge_09"] == 0 for r in records)
same("resolution/without_pure_token", without_pure, report["query_without_pure_ge_09"])
same("resolution/fn_pure_mass_fraction", pure_fn / fn_pixels,
     report["regions"]["mean_final_FN"]["token_mass_query_cov_ge_09_fraction"])
result["resolution_loss_1400"] = {"zero_mean_breakdown": dict(zero), "zero_mean_total": sum(zero.values()),
                                  "without_pure_token": without_pure, "fn_pixels": fn_pixels,
                                  "fn_in_pure_tokens": pure_fn, "fn_pure_mass_fraction": pure_fn / fn_pixels}

root = DATA / "a/lvis_atomic1400_20261009"
records = rows(root / "episode_metrics.jsonl", 1400)
report = obj(root / "report.json")
atomic = {}
for frame in ["cli", "original"]:
    atomic[frame] = class_fold_miou(records, lambda r: r["frames"][frame]["iu"])
    for method, value in atomic[frame].items():
        same(f"atomic/{frame}/{method}", value, report["frames"][frame]["all1400"]["miou"][method])
result["atomic_miou_1400"] = atomic

presence = {}
for name in ["lvis_context_presence200_20261009", "lvis_component_presence200_20261009"]:
    root = DATA / "a" / name / "score"
    records = rows(root / "scored_episodes.jsonl", 200)
    report = obj(root / "paired_results.json")
    computed = class_fold_miou(records, lambda r: r["iu"])
    for method, value in computed.items():
        same(f"{name}/{method}", value, report["miou"][method])
    presence[name] = {"frame": report["frame"], "n": len(records), "miou": computed}
result["context_presence_miou_200"] = presence

old = {}
for name in ["probe29_native4_v2", "screen12_native200_v3"]:
    report = obj(REPO / "evidence/local/cpu100_20261006/server" / name / "methods_summary.json")
    selected = []
    for r in report["rows"]:
        if r["id"] not in ["local_001", "local_002", "QP01", "QP02", "inv_huber_reference_readout"]:
            continue
        score = mean(100 * v["final_I"] / v["final_U"] if v["final_U"] else 0
                     for v in r["class_actions_vs_prototype"].values())
        same(f"{name}/{r['id']}", score, r["score"])
        selected.append({"id": r["id"], "score": score, "matched_controls": r["matched_controls"],
                         "actions_vs_prototype": r["actions_vs_prototype"]})
    old[name] = {"case_count": report["case_count"], "rows": selected}
result["historical_native_scores"] = old

raw_root = DATA / "a/paco_mean200_20261008/raw_cache/0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158"
result["assets"] = {"main_o24_root": str(raw_root),
                    "live_entry_directories_with_metadata": sum(p.is_dir() and (p / "entry.json").is_file() for p in raw_root.iterdir()),
                    "note": "Directory count at audit time, not episodes; raw tensor payloads were not rehashed or loaded."}
result["current_pilot_failure_ledger"] = obj(DATA / "a/joint_role_pilot200_20261010/failure_evidence_audit.json")

documentation = [
    REPO / "evidence/local/cpu100_20261006/METHODS_AND_FAILURES.md",
    REPO / "evidence/local/results/reference_qk_v2/REPORT.md",
    REPO / "evidence/local/results/reference_qk_v2/summary.json",
    REPO / "evidence/local/results/reference_qk_v2/diagnosis.json",
    DATA / "setup/disk_audit_20261010/REPORT.md",
    DATA / "setup/disk_audit_20261010/cache_index.jsonl",
    DATA / "setup/disk_audit_20261010/cache_index_receipt.json",
    DATA / "a/joint_role_pilot200_20261010/REPORT.md",
    DATA / "a/joint_role_pilot200_20261010/verification.json",
    DATA / "a/lvis_component_conflicts400_20261009/REPORT.md",
    DATA / "a/lvis_component_completion600_20261009/REPORT.md",
    DATA / "a/paco_fast9_600_20261009/analysis/ROOT_CAUSE.md",
    DATA / "a/experiment_status_review_20261010/verified_results.json",
]
for path in documentation:
    read(path)
result["unavailable_primary_reports"] = []
for method, filename in [("M04", "paired_score_report.json"), ("M06", "score_report.json"), ("M18", "score_report.json")]:
    path = REPO / "evidence/local/pro30_20261007/runtime" / f"{method}_fixed600" / filename
    result["unavailable_primary_reports"].append({"path": str(path), "exists": path.exists(),
                                                "status": "Historical markdown claim only unless primary report is located."})

git_ref = "6e986b9^:src/ics/cpu100/local_structure.py"
archived = subprocess.check_output(["git", "show", git_ref], cwd=REPO)
result["archived_geometry_source"] = {"git_ref": git_ref, "sha256": sha256(archived).hexdigest(),
                                     "note": "Mechanism inspection only; not asserted to be the exact source of the v2 native4 scores."}
read(Path(__file__).resolve())
result["sources"] = sources
result["checks"] = checks
result["checks_passed"] = len(checks)
result["source_file_count"] = len(sources)
(OUT / "verified_evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"checks_passed": len(checks), "source_file_count": len(sources),
                  "output": str(OUT / "verified_evidence.json")}, ensure_ascii=False))
