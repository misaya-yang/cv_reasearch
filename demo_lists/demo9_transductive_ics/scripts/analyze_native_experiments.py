#!/usr/bin/env python3
"""CPU analysis of frozen native experiment JSON; no encoder/evaluator files.

Supports multi-episode iu/original_iu reports and reference_feature_probe's
single-episode ledger schema. Failed safe-host outputs are never method scores.
Sources remain separate: incompatible baselines/contexts are not silently joined.
Class sums precede ratios; bootstrap clusters all known photo roles transitively.
E9/E10 only report diagnostics actually present in these frozen artifacts.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np


def photo_id(value):
    text = str(value).replace("\\", "/")
    match = re.fullmatch(r"COCO_(?:train|val)(?:2014|2017)_(\d+)", Path(text).stem)
    return "COCO:" + str(int(match.group(1))) if match else text


def flatten_photos(value):
    if isinstance(value, dict):
        return [x for item in value.values() for x in flatten_photos(item)]
    if isinstance(value, (list, tuple, set)):
        return [x for item in value for x in flatten_photos(item)]
    return [] if value is None else [photo_id(value)]


def photos(row):
    for key in ("roles_photo_ids", "image_ids", "photo_ids", "all_role_photo_ids", "allrole_photo_ids"):
        if key in row:
            return frozenset(flatten_photos(row[key]))
    result = []
    for key in ("support", "query", "pool", "donors", "negative_donors"):
        if key in row:
            result.extend(flatten_photos(row[key]))
    # A numerical pool size is not a photo ID.
    result = [p for p in result if not re.fullmatch(r"\d+(?:\.\d+)?", p)]
    return frozenset(result)


def ordered_roles(row):
    if row.get("support") is not None and row.get("query") is not None:
        return [flatten_photos(row["support"]), flatten_photos(row["query"])]
    roles = row.get("roles_photo_ids")
    if isinstance(roles, dict) and "support" in roles and "query" in roles:
        return [flatten_photos(roles["support"]), flatten_photos(roles["query"])]
    return None


def checked_iu(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("I/U must contain two pixel counts, not a patch-IoU scalar")
    i, u = map(float, value)
    if not all(math.isfinite(x) and x >= 0 and abs(x-round(x)) < 1e-6 for x in (i, u)) or i > u:
        raise ValueError("Invalid binary pixel intersection/union")
    return [i, u]


def normalize_row(raw, document, ordinal):
    identity = document.get("episode_identity", {})
    row = dict(identity, **raw)
    class_value = row.get("c", row.get("classid", row.get("class_id")))
    if class_value is None:
        raise ValueError("No class ID; cannot manufacture a class-mIoU cohort")
    e = row.get("e", row.get("episode", ordinal))
    fold = row.get("fold", document.get("fold", document.get("args", {}).get("fold", 0)))
    dataset = str(row.get("dataset", document.get("dataset", "unspecified")) or "unspecified")
    result = dict(e=e, c=int(class_value), fold=int(fold), dataset=dataset,
        photos=photos(row), iu={}, original_iu={}, pixels={}, arm_states={},
        semantic_fp=row.get("semantic_fp", {}),
        ordered_roles=ordered_roles(row),
        prediction_bits=row.get("prediction_bits", {}),
        metadata={k: row[k] for k in ("pair_id", "task_tag", "task_name", "support",
            "query", "encoder_calls", "wall_s", "dataset", "rgb_sha256",
            "support_rgb_sha256", "query_rgb_sha256", "support_mask_sha256",
            "condition", "perturbation", "cost", "costs", "feature_seconds",
            "host_seconds", "episode_seconds", "fit_cost", "predictions_frozen_s",
            "peak_allocated_bytes", "peak_reserved_bytes") if k in row})
    states = row.get("arm_states", row.get("method_states", row.get("armstate", row.get("readers", {}))))
    for key in ("iu", "original_iu"):
        for name, counts in row.get(key, {}).items():
            state = states.get(name, "SCORED")
            state = state.get("state", "UNKNOWN") if isinstance(state, dict) else state
            result["arm_states"][name] = state
            if str(state).startswith("ERROR"):
                continue
            result[key][name] = checked_iu(counts)
    result["pixels"] = row.get("pixels", {})
    if "predictions" in row:
        method_records = document.get("methods", {})
        method_records = method_records if isinstance(method_records, dict) else {}
        for name, prediction in row["predictions"].items():
            state = method_records.get(name, {}).get("state", "SCORED")
            result["arm_states"][name] = state
            if "prediction_bits" in prediction:
                result["prediction_bits"][name] = prediction["prediction_bits"]
            # Only explicitly successful/fallback native ledgers count.
            if str(state).startswith("ERROR"):
                continue
            for source_key, target in (("native_ledger", "iu"), ("original_ledger", "original_iu")):
                ledger = prediction.get(source_key)
                if ledger is not None:
                    result[target][name] = checked_iu([ledger["intersection"], ledger["union"]])
                    if target == "iu":
                        result["pixels"][name] = ledger
    if not result["iu"]:
        raise ValueError("No valid native I/U rows")
    return result


def components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for i, row in enumerate(rows):
        for identity in row["photos"]:
            if identity in owner:
                parent[find(i)] = find(owner[identity])
            else:
                owner[identity] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    groups = sorted(groups.values(), key=lambda g: g[0])
    membership = np.zeros(len(rows), dtype=int)
    for j, group in enumerate(groups):
        membership[group] = j
    return membership, [dict(episodes=[rows[i]["e"] for i in group],
        classes=sorted({rows[i]["c"] for i in group}), size=len(group),
        photos=len(set().union(*(rows[i]["photos"] for i in group)))) for group in groups]


def class_score(rows, name, metric="iu", weights=None):
    totals = {}
    for j, row in enumerate(rows):
        weight = 1.0 if weights is None else float(weights[j])
        if weight == 0:
            continue
        key = (row["dataset"], row["fold"], row["c"])
        acc = totals.setdefault(key, [0.0, 0.0])
        i, u = row[metric][name]
        acc[0] += weight*i
        acc[1] += weight*u
    folds = {}
    per_class = []
    for (dataset, fold, cid), (i, u) in sorted(totals.items()):
        score = 100*i/max(u, 1)
        folds.setdefault((dataset, fold), []).append(score)
        per_class.append(dict(dataset=dataset, fold=fold, c=cid, I=i, U=u, iou=score))
    # Separate dataset estimands; do not manufacture a cross-dataset score.
    per_dataset = {}
    for (dataset, fold), values in folds.items():
        per_dataset.setdefault(dataset, []).append(float(np.mean(values)))
    scores = {dataset: float(np.mean(values)) for dataset, values in per_dataset.items()}
    return scores, per_class


def comparison(rows, method, baseline, metric, repetitions, seed):
    selected = [r for r in rows if method in r[metric] and baseline in r[metric]]
    if not selected:
        return dict(state="NOT_EVALUABLE",reason="No paired valid pixel I/U")
    chosen, chosen_classes = class_score(selected, method, metric)
    base, base_classes = class_score(selected, baseline, metric)
    result = dict(state="DESCRIPTIVE_PAIRED_SUBSET", paired_episodes=len(selected),
        excluded_episodes=len(rows)-len(selected), model_refit_in_bootstrap=False,
        per_dataset={d: dict(method_class_miou=chosen[d], baseline_class_miou=base[d],
            delta_pp=chosen[d]-base[d]) for d in chosen},
        per_class=[dict(x, baseline_iou=b["iou"], delta_pp=x["iou"]-b["iou"])
            for x, b in zip(chosen_classes, base_classes)])
    result["class_signs"] = dict(positive=sum(v["delta_pp"] > 1e-10 for v in result["per_class"]),
        negative=sum(v["delta_pp"] < -1e-10 for v in result["per_class"]),
        tie=sum(abs(v["delta_pp"]) <= 1e-10 for v in result["per_class"]))
    if any(not row["photos"] for row in selected):
        result["photo_component_interval"] = dict(state="NOT_EVALUABLE",
            reason="Missing all-role photo identities; episode independence is not assumed")
        return result
    membership, groups = components(selected)
    count = len(groups)
    result["photo_components"] = groups
    result["effective_photo_components"] = count
    result["small_sample_warning"] = (
        "Development diagnostic only; few components/classes and fixed models; no method-gain claim")
    if count < 2:
        result["photo_component_interval"] = dict(state="NOT_IDENTIFIABLE",
            reason="Only one connected photo component; a degenerate bootstrap is not an uncertainty estimate")
        return result
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(count, np.full(count, 1/count), size=repetitions)
    deltas = {dataset: [] for dataset in chosen}
    for draw in draws:
        weights = draw[membership]
        a, _ = class_score(selected, method, metric, weights)
        b, _ = class_score(selected, baseline, metric, weights)
        for dataset in a:
            deltas[dataset].append(a[dataset]-b[dataset])
    result["photo_component_interval"] = dict(state="EXPLORATORY_CONDITIONAL",
        bootstrap_repetitions=repetitions, unit="connected all-role photo components",
        note="Absent resampled classes/folds omitted; datasets never averaged together; no multiplicity correction",
        delta95_pp={d: np.quantile(v, [.025, .975]).tolist() if len(v) > 1 else None
            for d, v in deltas.items()})
    return result


def ledger_sum(rows, name):
    aliases = dict(recovered_fn=("recovered_fn", "recovered_host_FN"),
        lost_tp=("lost_tp", "removed_host_TP"), added_fp=("added_fp", "added_FP"),
        removed_fp=("removed_fp", "removed_host_FP"))
    total = {}
    for target, keys in aliases.items():
        values = [next((r["pixels"][name][k] for k in keys if k in r["pixels"][name]), None)
                  for r in rows if name in r["pixels"]]
        available = [v for v in values if v is not None]
        if available:
            total[target] = sum(map(float, available))
            total[target+"_episodes"] = len(available)
    return total


def decode_bits(bits):
    """Only frozen prediction bits, not any evaluator mask or label file."""
    shape = tuple(bits["shape"])
    if bits.get("encoding") == "numpy.packbits-big/base64":
        encoded = bits["data"]
    elif "base64" in bits and bits.get("bitorder", "big") == "big":
        encoded = bits["base64"]
    else:
        raise ValueError("Unsupported frozen mask-bit encoding")
    array = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
    size = math.prod(shape)
    if size > len(array)*8 or len(array)*8-size >= 8:
        raise ValueError("Frozen mask bit count/shape mismatch")
    return np.unpackbits(array)[:size].astype(bool).reshape(shape)


def frozen_bank_diagnostic(rows):
    bank = []
    choices = Counter()
    for row in rows:
        eligible = [name for name in row["iu"] if not (
            name.lower() == "true" or "oracle" in name.lower() or "true-" in name.lower()
            or "-true" in name.lower() or name.lower().startswith("gt"))]
        if not eligible:
            continue
        name = max(eligible, key=lambda key: row["iu"][key][0]/max(row["iu"][key][1], 1))
        choices[name] += 1
        bank.append(dict(row, iu={"frozen_bank_oracle": row["iu"][name]}))
    if not bank:
        return dict(state="NOT_EVALUABLE")
    scores, per_class = class_score(bank, "frozen_bank_oracle")
    return dict(state="POSTHOC_FINITE_OUTPUT_BANK_ONLY", per_dataset=scores,
        per_class=per_class, choices=dict(choices), bank_sizes=[len(r["iu"]) for r in rows],
        qualification="GT per-episode IoU choice among already scored non-oracle outputs; no GT-input oracle row is added. Not a method, not a strict class-mIoU upper bound, and potentially unequal budgets.")


def concept_pairs(rows):
    groups = {}
    for row in rows:
        if "pair_id" in row["metadata"]:
            groups.setdefault(str(row["metadata"]["pair_id"]), []).append(row)
    results = []
    for pair, group in groups.items():
        if len(group) != 2:
            results.append(dict(pair_id=pair, state="NOT_EVALUABLE", tasks=len(group)))
            continue
        a, b = group
        # A path alone does not prove immutable RGB; require hashes.
        ma, mb = a["metadata"], b["metadata"]
        fields = ("support_rgb_sha256", "query_rgb_sha256")
        verified = all(k in ma and k in mb and ma[k] == mb[k] for k in fields)
        methods = {}
        for name in sorted(set(a["iu"]) & set(b["iu"])):
            values = [r["iu"][name][0]/max(r["iu"][name][1], 1) for r in group]
            item = dict(task_ious=values, both_tasks_iou_at_least_half=all(v >= .5 for v in values),
                threshold_scope="Fixed descriptive .5 diagnostic, not a gate or a gain claim")
            if name in a["prediction_bits"] and name in b["prediction_bits"]:
                first = decode_bits(a["prediction_bits"][name])
                second = decode_bits(b["prediction_bits"][name])
                if first.shape != second.shape:
                    raise ValueError("Same-RGB pair masks have different native resolutions")
                item["changed_prediction_pixels"] = int(np.count_nonzero(first != second))
            methods[name] = item
        results.append(dict(pair_id=pair, state="RGB_HASH_VERIFIED" if verified else "PAIR_CONTRACT_UNVERIFIED",
            class_ids=[a["c"], b["c"]], methods=methods))
    return dict(actual_pair_groups=len(groups), pair_records=results,
        state="PAIRED_OUTPUT_DIAGNOSTICS" if results else "NOT_EVALUABLE",
        qualification="Prediction changes do not imply correct concept switches. Query semantic attribution needs saved post-evaluator semantic counts; none are invented.")


def analyze_source(path, baselines, repetitions, seed):
    document = json.loads(path.read_text())
    raws = document.get("records")
    if raws is None and "predictions" in document:
        raws = [dict(document.get("episode_identity", {}), predictions=document["predictions"])]
    if raws is None:
        raws = []
    rows, rejected = [], []
    for i, raw in enumerate(raws):
        try:
            rows.append(normalize_row(raw, document, i))
        except (ValueError, KeyError, TypeError) as error:
            rejected.append(dict(ordinal=i, error=str(error)))
    names = sorted({name for row in rows for name in row["iu"]})
    references = [name for name in baselines if name in names]
    if not references and not baselines:
        references = [next((name for name in ("native_raw", "native_dense", "host",
            "insid3_native", "1shot") if name in names), names[0])] if names else []
    method_records=document.get("methods", {})
    method_records=method_records if isinstance(method_records, dict) else {}
    result = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_state=document.get("state", "UNKNOWN"), source_scope=document.get("scope", document.get("method_claim")),
        source_error=document.get("error"), source_args=document.get("args"),
        planned_episodes=len(document.get("frozen_episodes", [])) or None,
        available_episodes=len(rows), rejected_records=rejected, baselines=references,
        missing_requested_baselines=[name for name in baselines if name not in names],
        partial=document.get("state") not in ("COMPLETED", "COMPLETED_REUSED"),
        failure_count=document.get("failure_count", 0),
        explicit_method_failures={name:value for name,value in method_records.items()
            if isinstance(value,dict) and str(value.get("state", "")).startswith("ERROR")},
        arm_state_counts=dict(Counter(str(state) for r in rows for state in r["arm_states"].values())),
        frozen_case_counts=[dict(e=r["e"],c=r["c"],fold=r["fold"],dataset=r["dataset"],
            ordered_roles=r["ordered_roles"],iu=r["iu"],original_iu=r["original_iu"],
            arm_states=r["arm_states"]) for r in rows],
        method_scores={}, comparisons={}, pixel_ledgers={},
        pixel_ledger_reference=document.get("pixel_ledger_reference", "Source's frozen host/default baseline; not automatically --base"),
        observed_cost={k:document[k] for k in ("elapsed_s","elapsed_concurrent_s",
            "elapsed_concurrent_seconds","peak_allocated_bytes","peak_reserved_bytes",
            "encoder_cost_excluded") if k in document})
    for name in names:
        valid = [r for r in rows if name in r["iu"]]
        score, by_class = class_score(valid, name)
        result["method_scores"][name] = dict(valid_episodes=len(valid), per_dataset=score, per_class=by_class)
        result["pixel_ledgers"][name] = ledger_sum(valid, name)
        for baseline in references:
            if name != baseline:
                key = name+" vs "+baseline
                result["comparisons"][key] = {
                    metric: comparison(rows, name, baseline, metric, repetitions, seed)
                    for metric in ("iu", "original_iu")}
    # No unseen concept/stress fixtures are invented from ordinary episode scores.
    result["E9_posthoc_diagnostics"] = concept_pairs(rows)
    result["E9_frozen_output_bank"] = frozen_bank_diagnostic(rows)
    semantic={}
    for row in rows:
        for name,value in row["semantic_fp"].items():
            histogram=value.get("label_histogram", {})
            total=semantic.setdefault(name,{})
            for label,count in histogram.items():
                total[label]=total.get(label,0)+int(count)
    result["E9_saved_semantic_FP"]=dict(method_label_histograms=semantic,
        qualification="Saved post-prediction evaluator counts only. Label0 is unannotated background, never an identified object.")
    datasets = sorted({r["dataset"] for r in rows})
    result["E10_posthoc_diagnostics"] = dict(actual_datasets=datasets,
        measured_cost_fields=result["observed_cost"],
        actual_per_episode_cost=[dict(e=r["e"],c=r["c"],
            **{k:r["metadata"][k] for k in ("cost","costs","encoder_calls","wall_s",
                "feature_seconds","host_seconds","episode_seconds","fit_cost",
                "predictions_frozen_s","peak_allocated_bytes","peak_reserved_bytes") if k in r["metadata"]})
            for r in rows if any(k in r["metadata"] for k in ("cost","costs","encoder_calls",
                "wall_s","feature_seconds","host_seconds","episode_seconds","fit_cost",
                "predictions_frozen_s","peak_allocated_bytes","peak_reserved_bytes"))],
        state="MULTI_DATASET_FROZEN_QUALITY" if len(set(datasets)-{"unspecified"})>1
            else ("QUALITY_COST_ONLY" if result["observed_cost"] and rows else "NOT_EVALUABLE"),
        per_dataset_quality={name: value["per_dataset"] for name,value in result["method_scores"].items()},
        reason="No stress/other-dataset outcome is inferred unless the frozen source contains that fixture")
    return result


def cross_source_anchor_audit(reports):
    """Counts equality is weaker than mask identity; no cross-source merging."""
    first, checked, mismatches = {}, 0, []
    for report in reports:
        for row in report.get("frozen_case_counts", []):
            if row["ordered_roles"] is None:
                continue
            identity = (row["dataset"],row["fold"],row["e"],row["c"],
                tuple(tuple(ids) for ids in row["ordered_roles"]))
            for name in ("insid3_native", "host", "1shot"):
                if name not in row["iu"]:
                    continue
                key=identity+(name,)
                value=dict(iu=row["iu"][name],original_iu=row["original_iu"].get(name))
                if key in first:
                    checked += 1
                    old=first[key]
                    if value != old["counts"]:
                        mismatches.append(dict(identity=str(identity),anchor=name,
                            first_source=old["source"],second_source=report["path"],
                            first_counts=old["counts"],second_counts=value))
                else:
                    first[key]=dict(source=report["path"],counts=value)
    return dict(overlapping_anchor_checks=checked,mismatches=mismatches,
        qualification="Only matching dataset/fold/episode/class/ordered-photo roles and same named host. I/U equality is not byte-exact mask identity; sources remain separate.")


def cross_source_frozen_bank(reports):
    """E9 posthoc bank, gated by identical same-named host counts on same roles."""
    grouped={}
    for report in reports:
        for row in report.get("frozen_case_counts",[]):
            if row["ordered_roles"] is None or "insid3_native" not in row["iu"]:
                continue
            identity=(row["dataset"],row["fold"],row["e"],row["c"],
                tuple(tuple(v) for v in row["ordered_roles"]))
            grouped.setdefault(identity,[]).append((report["path"],row))
    banks=[];excluded=[]
    for identity,entries in grouped.items():
        if len(entries)<2:
            continue
        anchors=[(r["iu"]["insid3_native"],r["original_iu"].get("insid3_native")) for _,r in entries]
        if any(value!=anchors[0] for value in anchors[1:]):
            excluded.append(dict(identity=str(identity),reason="Shared host I/U differs"))
            continue
        counts={"insid3_shared":anchors[0][0]}
        for source,row in entries:
            for name,value in row["iu"].items():
                if name!="insid3_native":
                    counts[source+"::"+name]=value
        dataset,fold,e,c,roles=identity
        banks.append(dict(dataset=dataset,fold=fold,e=e,c=c,iu=counts,original_iu={},
            photos=frozenset(x for group in roles for x in group),metadata={},pixels={},arm_states={}))
    result=frozen_bank_diagnostic(banks)
    result.update(matched_multi_source_episodes=len(banks),excluded=excluded,
        provenance_gate="Same dataset/fold/e/c/ordered support-query photos and exact shared host I/U; not mask identity",
        inference_use_forbidden=True)
    if banks:
        result["shared_host_class_miou"]=class_score(banks,"insid3_shared")[0]
    return result


def self_check():
    # Ratios after class sums, not mean episode IoU; no original/model mixing.
    d = dict(dataset="toy", fold=0, c=0, pixels={}, metadata={}, arm_states={})
    rows = [dict(d,e=0,photos=frozenset({"a","b"}),
        iu={"base":[1,1],"new":[1,2]}, original_iu={"base":[1,4],"new":[1,8]}),
        dict(d,e=1,photos=frozenset({"b","c"}),
        iu={"base":[1,100],"new":[2,100]}, original_iu={"base":[1,4],"new":[1,8]}),
        dict(d,e=2,photos=frozenset({"d","e"}),c=1,
        iu={"base":[1,2],"new":[2,2]},original_iu={"base":[1,2],"new":[1,2]})]
    score,_=class_score(rows,"base")
    assert abs(score["toy"]-(100*2/101+50)/2)<1e-12
    membership, groups=components(rows)
    assert membership.tolist()==[0,0,1] and len(groups)==2
    assert photo_id("val2014/COCO_val2014_000000000123.jpg")==photo_id("train2014/COCO_train2014_000000000123.png")
    paired=comparison(rows,"new","base","iu",200,0)
    original=comparison(rows,"new","base","original_iu",200,0)
    assert paired["per_dataset"]["toy"]["delta_pp"]!=original["per_dataset"]["toy"]["delta_pp"]
    try:
        checked_iu(.5)
        raise AssertionError("Scalar patch score accepted")
    except ValueError:
        pass
    fake=dict(episode_identity=dict(e=1,c=0,roles_photo_ids={"support":["s"],"query":["q"]}),
        methods={"failed":dict(state="ERROR")})
    raw=dict(fake["episode_identity"],predictions={
        "host":{"native_ledger":{"intersection":1,"union":2}},
        "failed":{"ERROR_safe_host_ledger":{"intersection":1,"union":2}}})
    assert "failed" not in normalize_row(raw,fake,0)["iu"]
    bank = frozen_bank_diagnostic([dict(rows[0],iu={"base":[1,2],"new":[3,4],"true":[1,1]})])
    assert bank["choices"] == {"new":1}
    def bits(mask):
        a=np.asarray(mask,dtype=bool)
        return dict(shape=list(a.shape),encoding="numpy.packbits-big/base64",
            data=base64.b64encode(np.packbits(a.reshape(-1)).tobytes()).decode())
    pair=[]
    for c, mask in enumerate(([1,0,1,0],[0,1,0,1])):
        meta=dict(pair_id="shared",support_rgb_sha256="s",query_rgb_sha256="q")
        pair.append(dict(rows[0],c=c,metadata=meta,prediction_bits={"base":bits(mask)}))
    assert concept_pairs(pair)["pair_records"][0]["methods"]["base"]["changed_prediction_pixels"]==4
    with tempfile.TemporaryDirectory(prefix="native_analysis_cpu_") as folder:
        path=Path(folder)/"empty.json"
        path.write_text(json.dumps(dict(state="ERROR",error="fixture stage fault",records=[])))
        report=analyze_source(path,[],10,0)
        assert report["source_error"]=="fixture stage fault" and not report["method_scores"]
    print(json.dumps(dict(state="CPU_SELF_CHECK_PASSED",class_sum_before_ratio=True,
        transitive_all_role_components=True,model_original_IU_separate=True,
        failed_host_fallback_excluded=True,no_GT_files_opened=True,no_CUDA_calls=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files",nargs="*",type=Path)
    parser.add_argument("--base",action="append",default=[])
    parser.add_argument("--B",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=2044)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.files or args.out is None or args.B<2:
        parser.error("files and --out required; --B>=2")
    reports=[]
    for path in args.files:
        try:
            reports.append(analyze_source(path,args.base,args.B,args.seed))
        except (OSError,ValueError,KeyError,TypeError) as error:
            reports.append(dict(path=str(path),state="ANALYSIS_ERROR",error=str(error)))
    analysis_errors=sum(r.get("state")=="ANALYSIS_ERROR" for r in reports)
    output=dict(state="CPU_FROZEN_OUTPUT_ANALYSIS" if not analysis_errors else "CPU_ANALYSIS_WITH_ERRORS",
        analysis_error_count=analysis_errors,scope="Descriptive development diagnostics; no method-gain claim",
        aggregate="Class sumI/sumU then fold mean; datasets stay separate",
        sources_remain_separate=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reports=reports,cross_source_anchor_audit=cross_source_anchor_audit(reports),
        E9_cross_source_frozen_output_bank=cross_source_frozen_bank(reports),
        no_new_encoder_GPU_or_stress_execution=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temp=args.out.with_suffix(args.out.suffix+".tmp")
    temp.write_text(json.dumps(output,allow_nan=False,indent=2))
    temp.replace(args.out)
    print(json.dumps(dict(state=output["state"],reports=len(reports),out=str(args.out))))
    if analysis_errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
