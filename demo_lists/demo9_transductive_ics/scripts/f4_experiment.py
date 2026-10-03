#!/usr/bin/env python3
"""Finite, zero-training seven-arm F4 experiment on complete public FoRIS.

Prepare is metadata-only. Root owns real CUDA launch and the outer resource
guard. All seven masks are durably frozen before opening that query's labels.
Completed cases are validated and reused on resume; software errors are never
reported as negative scientific effects or silently replaced by native masks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARM_IDS = ("native", "anchor_only", "full001", "perFG002",
           "matchedGlobal", "Part4cal", "querycore")
RECIPE = {
    "arms": list(ARM_IDS), "beta": .55, "tau": .07,
    "new_encoder_calls": 0, "query_GT_for_any_decision": False,
    "host": "complete public set_reference/set_target/segment; original CRF",
    "G_exception": "delete from native FINAL mask; no second CRF, exact existing query-core rule",
    "original_mapping": "bilinear binary working mask, align_corners=False, strictly >.5",
    "training": False, "downloads": False, "parameters_selected_on_query_GT": False,
    "budget_control": "D/E share independent BG token exposure and per-branch prototype-slot budget; effective direction counts separately reported",
}
CARD = [
    "Assumption: one global negative direction mis-penalizes some legitimate foreground modes; mode-conditioned reference counterevidence can change complete decisions beyond gain redistribution and a matched global bank.",
    "Required task evidence, not a predicted gain: B/C/D must beat native and G with paired lower bound>0 on all-fold DEV; conditional coverage attribution additionally requires D>C and D>E with paired lower bounds>0. The only prior measured gain is G's +0.456pp on already-inspected DEV241.",
    "Match: freeze the simplest supported arm, source hashes and rule before the root-verified unseen cohort; inspect per-class U(c-a)+I(b-d), FP removed and TP lost rather than borrowing G's gain.",
    "Mismatch: shared native/cache or numeric failure is an engineering error and stops this stage; otherwise no bank/threshold/normalization sweep. B explaining C/D removes the bank claim; F-only success is calibration; no win over G rejects complexity.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=1, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def row_key(row):
    return (int(row["fold"]), int(row["e"]), int(row["c"]),
            row["support"], row["query"])


def stem(row):
    return "%d_%d_%d" % row_key(row)[:3]


def load_records(path):
    path = Path(path)
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_record(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def source_files(foris_root, demo4_root):
    return [Path(__file__), ROOT / "tics/ten_direction_context.py",
            ROOT / "tics/native_assets.py", ROOT / "tics/f4_conditional.py",
            ROOT / "tics/f4_calibration.py", Path(demo4_root) / "icx/common.py",
            *[Path(foris_root) / rel for rel in ("models/foris.py", "utils/data.py",
                                                "utils/clustering.py", "utils/refinement.py")]]


def prepare(a):
    """No torch import, encoder, query-mask pixels or numerical tests."""
    parent = json.loads(a.from_manifest.read_text())
    rows = parent.get("episodes", parent.get("frozen_episodes", []))
    if not rows: raise ValueError("Nonempty existing source episode manifest required")
    if a.scope == "smoke8":
        selected = []
        for j in range(2):
            for f in range(4):
                fold = [r for r in rows if r["fold"] == f]
                if len(fold) < 2: raise ValueError("Two existing DEV examples per fold required")
                selected.append(fold[j])
    elif a.scope == "dev241":
        if len(rows) != 241: raise ValueError("Exact old DEV241 manifest required")
        selected = rows
    else:
        if a.unseen_receipt is None: raise ValueError("Root-verified unseen receipt required; old CONFIRM600 is not fresh")
        receipt = json.loads(a.unseen_receipt.read_text())
        if (receipt.get("state") != "ROOT_FROZEN_UNSEEN_COHORT_VERIFIED" or
            receipt.get("query_GT_previously_unopened") is not True or
            receipt.get("all_role_UID_exposure_check_passed") is not True or
            receipt.get("source_manifest_sha256") != sha(a.from_manifest)):
            raise ValueError("Unseen cohort receipt does not establish the required exposure/embargo contract")
        selected = rows
    selected = [{k: r[k] for k in ("fold", "e", "c", "support", "query")} for r in selected]
    if len({row_key(r) for r in selected}) != len(selected): raise ValueError("Duplicate episode identity")
    sources = source_files(parent["foris_root"], a.demo4_root)
    for path in sources:
        if not path.is_file(): raise FileNotFoundError(path)
        compile(path.read_text(), str(path), "exec")
    baseline_rows = {}
    if a.baseline_run:
        baseline_rows = {row_key(r.get("row", r)): r for r in load_records(a.baseline_run / "episodes.jsonl")}
    if a.scope != "frozen_unseen" and not baseline_rows:
        raise ValueError("Existing complete public DEV baseline records and packed masks required")
    assets, baselines = [], {}
    for row in selected:
        for role in ("support", "query"):
            image = Path(parent["data_root"]) / row[role]
            label = Path(parent["annotation_root"]) / Path(row[role]).with_suffix(".png")
            for path in (image, label):
                st = path.stat(); assets.append(dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns))
        if row_key(row) in baseline_rows:
            old = baseline_rows[row_key(row)]
            packet = a.baseline_run / "packets" / (stem(row) + ".npz")
            if not packet.is_file(): raise FileNotFoundError(packet)
            baseline_iu = old["original_iu"]["native"]
            baselines[stem(row)] = dict(packet=str(packet), packet_sha256=sha(packet), original_iu=baseline_iu)
        elif a.scope != "frozen_unseen":
            raise ValueError("Prepared old DEV row missing its complete native baseline")
    basis = a.projection_basis or parent.get("projection_basis")
    if not basis: raise ValueError("Existing source-native projection basis path required")
    basis = Path(basis); st = basis.stat()
    assets.append(dict(path=str(basis.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns))
    if shutil.disk_usage(ROOT).free < a.reserve_bytes: raise RuntimeError("Keep 5GiB disk reserve")
    manifest = dict(schema="f4_seven_arms_v1", state="METADATA_PREPARED_CPU_SMOKE_REQUIRED",
        episodes=selected, scope=a.scope, design_DEV=a.scope != "frozen_unseen",
        parent_manifest=str(a.from_manifest), parent_manifest_sha256=sha(a.from_manifest),
        data_root=parent["data_root"], annotation_root=parent["annotation_root"],
        foris_root=parent["foris_root"], demo4_root=a.demo4_root, seed=parent.get("seed", 0),
        projection_basis=str(basis.resolve()), source_hashes={str(p.resolve()): sha(p) for p in sources},
        assets=assets, baseline_rows=baselines, first8_native_pixel_and_original_IU_gate=True,
        recipe=RECIPE, card=CARD, query_GT_opened_after_all_seven_predictions=True,
        maximum_output_bytes=a.max_output_bytes, reserve_bytes=a.reserve_bytes,
        unseen_receipt=str(a.unseen_receipt) if a.unseen_receipt else None,
        unseen_receipt_sha256=sha(a.unseen_receipt) if a.unseen_receipt else None)
    if a.manifest.exists(): raise ValueError("Preserve immutable manifests; choose a new path")
    write_json(a.manifest, manifest)
    print(json.dumps(dict(state=manifest["state"], episodes=len(selected), scope=a.scope,
                          numerical_checks_run=False, GPU_started=False)), flush=True)


def arm_results(ctx):
    from tics.f4_conditional import source_evidence, run_conditional
    from tics.f4_calibration import run004, query_core_remove
    evidence = source_evidence(ctx)
    base_audit = dict(source_evidence_state=evidence.get("state"),
                      source_evidence_metadata=evidence.get("metadata", {}),
                      new_encoder_calls=0, mode="original_unmodified_public_FoRIS")
    if evidence.get("state") == "SUPPORTED":
        base_audit.update(foreground_mode_count=len(evidence["fg_prototypes"]),
                          original_hard_bank_slots=len(evidence["hard_bank"]))
    yield "native", {**ctx.native, "f4_audit": base_audit}
    for name, policy in (("anchor_only", "anchor"), ("full001", "001"),
                         ("perFG002", "002"), ("matchedGlobal", "global_matched")):
        yield name, run_conditional(ctx, policy=policy, evidence=evidence)
    yield "Part4cal", run004(ctx, evidence=evidence)
    yield "querycore", query_core_remove(ctx)


def small_json(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        if value.numel() > 4096: raise ValueError("Do not serialize full feature/bank/NxN tensors")
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size > 4096: raise ValueError("Only small audit fields allowed")
        value = value.tolist()
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, dict): return {str(k): small_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [small_json(v) for v in value]
    json.dumps(value, allow_nan=False)
    return value


def freeze_case(ctx, row, out, man_sha):
    """Seven arms, no annotation path available to the decision functions."""
    import numpy as np
    import torch
    arrays, audits, clocks = {}, {}, {}
    for name, result in arm_results(ctx):
        tick = time.monotonic()
        mask = result.get("mask")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(ctx.work_hw):
            raise ValueError("Full source working bool mask required: " + name)
        score = result.get("score")
        if not isinstance(score, torch.Tensor) or tuple(score.shape) != tuple(ctx.grid_hw) or not torch.isfinite(score).all():
            raise ValueError("Finite native-grid score required: " + name)
        arrays[name] = np.packbits(mask.detach().cpu().numpy().reshape(-1))
        arrays["score__" + name] = score.detach().float().cpu().numpy()
        for field in ("part2_score", "part2_sf", "part2_sbn", "part4_penalty", "part4_cluster_delta"):
            tensor = result.get("stages", {}).get(field)
            if isinstance(tensor, torch.Tensor) and tensor.numel() <= 4096:
                arrays[field + "__" + name] = tensor.detach().float().cpu().numpy()
        audits[name] = small_json(dict(audit=result.get("audit", {}), f4_audit=result.get("f4_audit", {})))
        clocks[name] = time.monotonic() - tick
        print(json.dumps(dict(pair=stem(row), arm=name, state="PREDICTION_IN_MEMORY_FROZEN")), flush=True)
    if tuple(audits) != ARM_IDS: raise RuntimeError("All seven arms must succeed before query GT is opened")
    path = out / "predictions" / (stem(row) + ".npz")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    temporary.replace(path)
    receipt = dict(state="ALL_SEVEN_PREDICTIONS_FROZEN", row=row,
        manifest_sha256=man_sha, prediction_path=str(path), predictions_sha256=sha(path),
        work_hw=list(ctx.work_hw), grid_hw=list(ctx.grid_hw),
        original_hw=[ctx.query_rgb.height, ctx.query_rgb.width], arm_ids=list(ARM_IDS),
        no_query_GT_before_prediction_freeze=True, frozen_at_ns=time.time_ns(),
        full_native_uncached_cached_exact=bool(ctx.native.get("uncached_public_exact")),
        identity_hooks_exact=bool(ctx.native.get("identity_exact")),
        current_DEV_baseline_bit_exact=ctx.native.get("current_DEV_baseline_bit_exact"),
        actual_encoder_B2_calls=ctx.budget.actual_b2_calls,
        extra_encoder_B2_calls=ctx.budget.extra_b2_calls,
        full_cached_replay_calls=ctx.replay_calls, method_audits=audits, arm_freeze_copy_seconds=clocks,
        cost_scope="Actual quality execution; shared-server elapsed time is not formal speed evidence")
    if receipt["actual_encoder_B2_calls"] != 1 or receipt["extra_encoder_B2_calls"] != 0:
        raise RuntimeError("F4 must share exactly one original paired encoder forward, without new views")
    write_json(out / "frozen" / (stem(row) + ".json"), receipt)
    return receipt


def original_mask(mask, hw):
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.as_tensor(mask)[None, None].float(), hw,
                         mode="bilinear", align_corners=False)[0, 0] > .5


def count_ledger(mask, native, truth):
    import numpy as np
    mask, native, truth = [np.asarray(x, dtype=bool) for x in (mask, native, truth)]
    i, u = int((mask & truth).sum()), int((mask | truth).sum())
    return [i, u], dict(TP=i, FP=int((mask & ~truth).sum()), FN=int((~mask & truth).sum()),
        deleted_FP=int((native & ~mask & ~truth).sum()), lost_TP=int((native & ~mask & truth).sum()),
        added_TP=int((~native & mask & truth).sum()), added_FP=int((~native & mask & ~truth).sum()),
        changed_pixels=int((native ^ mask).sum()))


def score_frozen(receipt, manifest):
    """CPU evaluation; no method/encoder called after first query-label open."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    if receipt.get("state") != "ALL_SEVEN_PREDICTIONS_FROZEN" or receipt.get("no_query_GT_before_prediction_freeze") is not True:
        raise ValueError("Unconditionally reject premature scoring")
    if not receipt.get("full_native_uncached_cached_exact") or not receipt.get("identity_hooks_exact"):
        raise RuntimeError("Complete public native identity must pass before scoring")
    path = Path(receipt["prediction_path"])
    if sha(path) != receipt["predictions_sha256"]: raise RuntimeError("Frozen prediction drift")
    with np.load(path, allow_pickle=False) as pack:
        masks = {arm: np.unpackbits(pack[arm])[:int(receipt["work_hw"][0])*int(receipt["work_hw"][1])].reshape(receipt["work_hw"]).astype(bool) for arm in ARM_IDS}
    row = receipt["row"]
    query_label = Path(manifest["annotation_root"]) / Path(row["query"]).with_suffix(".png")
    truth_original = np.asarray(Image.open(query_label)) == row["c"] + 1
    if list(truth_original.shape) != receipt["original_hw"]: raise RuntimeError("Image/label original shape mismatch")
    truth_model = F.interpolate(torch.from_numpy(truth_original.copy())[None, None].float(),
                                receipt["work_hw"], mode="nearest")[0, 0].bool().numpy()
    originals = {arm: original_mask(mask, truth_original.shape).numpy() for arm, mask in masks.items()}
    iu, model_iu, ledger, model_ledger, oracle = {}, {}, {}, {}, {}
    for arm in ARM_IDS:
        iu[arm], ledger[arm] = count_ledger(originals[arm], originals["native"], truth_original)
        model_iu[arm], model_ledger[arm] = count_ledger(masks[arm], masks["native"], truth_model)
        # This is a diagnostic intervention restricted to changed original pixels.
        ideal = (originals["native"] & originals[arm]) | (truth_original & (originals["native"] ^ originals[arm]))
        oracle[arm] = [int((ideal & truth_original).sum()), int((ideal | truth_original).sum())]
        base_i, base_u = iu.get("native", [0, 0])
        l = ledger[arm]
        l["exact_IoU_improvement_numerator"] = base_u*(l["added_TP"]-l["lost_TP"]) + base_i*(l["deleted_FP"]-l["added_FP"])
    baseline = manifest.get("baseline_rows", {}).get(stem(row))
    if baseline and iu["native"] != baseline["original_iu"]:
        raise RuntimeError("Complete native original-I/U does not reproduce current DEV baseline")
    if sha(path) != receipt["predictions_sha256"]: raise RuntimeError("Predictions changed during scoring")
    return dict(row=row, original_iu=iu, model_iu=model_iu,
        original_error_ledger=ledger, model_error_ledger=model_ledger,
        changed_original_pixel_oracle_iu=oracle, baseline_original_IU_exact=bool(baseline),
        baseline_working_mask_bit_exact=receipt.get("current_DEV_baseline_bit_exact"),
        predictions_frozen_before_query_GT=True, full_native_exact=True,
        prediction_path=str(path), predictions_sha256=receipt["predictions_sha256"],
        frozen_receipt_path=str(path.parent.parent / "frozen" / (stem(row)+".json")),
        actual_encoder_B2_calls=receipt["actual_encoder_B2_calls"], extra_encoder_B2_calls=receipt["extra_encoder_B2_calls"],
        full_cached_replay_calls=receipt["full_cached_replay_calls"], method_audits=receipt["method_audits"])


def validate_manifest(manifest, a):
    if manifest.get("schema") != "f4_seven_arms_v1": raise ValueError("Prepared F4 manifest required")
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest: raise RuntimeError("Frozen source changed: " + path)
    for item in manifest["assets"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise RuntimeError("Prepared asset changed: " + item["path"])
    if manifest.get("unseen_receipt") and sha(manifest["unseen_receipt"]) != manifest["unseen_receipt_sha256"]:
        raise RuntimeError("Frozen unseen-cohort receipt changed")
    if a.cpu_receipt:
        proof = json.loads(a.cpu_receipt.read_text())
        if (proof.get("state") != "CPU_F4_SEVEN_SOURCE_PASSED" or proof.get("cases") != 10 or
            proof.get("CUDA_initialized") is not False or proof.get("no_query_GT_in_decisions") is not True):
            raise RuntimeError("Ten public-source CPU seven-arm cases required before root CUDA launch")
        for path, digest in proof["source_hashes"].items():
            if sha(path) != digest: raise RuntimeError("CPU proof predates runtime source: " + path)
    elif a.run:
        raise ValueError("CPU receipt required")


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def run(a):
    import numpy as np
    import torch
    from PIL import Image
    if not a.allow_gpu or os.environ.get("DEMO9_F4_ROOT_LAUNCH") != "1":
        raise RuntimeError("Root explicitly owns GPU launch; prepare does not start a job")
    manifest = json.loads(a.manifest.read_text()); validate_manifest(manifest, a)
    rows = manifest["episodes"][:a.max_pairs] if a.max_pairs else manifest["episodes"]
    man_sha = sha(a.manifest); started = time.monotonic()
    if a.out.exists() and any(a.out.iterdir()) and not a.resume: raise ValueError("Fresh output or explicit --resume required")
    a.out.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "frozen", "errors"): (a.out/name).mkdir(exist_ok=True)
    records = load_records(a.out / "episodes.jsonl")
    done = {row_key(r["row"]): r for r in records}
    if len(done) != len(records): raise RuntimeError("Duplicated committed case")
    if not set(done) <= {row_key(r) for r in rows}: raise RuntimeError("Resume prefix is outside this requested cohort")
    report_path = a.out / "report.json"
    if a.resume and report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("manifest_sha256") != man_sha: raise RuntimeError("Resume manifest changed")
        old = a.out / ("report.before_resume_%d.json" % time.time_ns()); shutil.copyfile(report_path, old)
    for row in rows:
        if row_key(row) not in done: continue
        r = done[row_key(row)]
        frozen = json.loads(Path(r["frozen_receipt_path"]).read_text())
        if (frozen["manifest_sha256"] != man_sha or row_key(frozen["row"]) != row_key(row) or
            sha(r["prediction_path"]) != r["predictions_sha256"] or frozen["predictions_sha256"] != r["predictions_sha256"]):
            raise RuntimeError("Completed resume prefix failed identity/SHA validation")
    report = dict(state="RUNNING", manifest_sha256=man_sha, source_hashes=manifest["source_hashes"],
        scope=manifest["scope"], design_DEV=manifest["design_DEV"], expected_episodes=len(rows),
        arms=list(ARM_IDS), card=CARD, recipe=RECIPE, query_GT_in_decisions=False,
        completed_prefix_reused=len(done), records=records, errors_are_scientific_negatives=False)
    def save():
        report.update(episodes=len(records), elapsed_seconds=time.monotonic()-started,
                      bytes_written=directory_bytes(a.out))
        write_json(report_path, report)
    save()
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, manifest["foris_root"])
    sys.path.insert(0, manifest["demo4_root"])
    torch.set_num_threads(2); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    host = encoder = None
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows):
                if row_key(row) in done: continue
                if time.monotonic()-started >= a.budget_seconds:
                    report["state"]="COMPLETED_FINITE_BUDGET"; break
                if shutil.disk_usage(a.out).free < manifest["reserve_bytes"] or directory_bytes(a.out) > manifest["maximum_output_bytes"]:
                    raise RuntimeError("Bounded output/5GiB reserve gate")
                frozen_path = a.out / "frozen" / (stem(row)+".json")
                if a.resume and frozen_path.exists():
                    frozen=json.loads(frozen_path.read_text())
                    if frozen["manifest_sha256"] != man_sha or row_key(frozen["row"]) != row_key(row):
                        raise RuntimeError("Unscored frozen prefix identity changed")
                    rec=score_frozen(frozen,manifest); rec["resumed_without_encoding"] = True
                else:
                    if host is None:
                        import models.foris as source
                        from icx.common import TimmDINOv3
                        from tics.native_assets import reuse_native_basis
                        encoder=TimmDINOv3().cuda().eval().requires_grad_(False)
                        with reuse_native_basis(source.FoRIS, manifest["projection_basis"]):
                            host=source.FoRIS(encoder=encoder,image_size=1024,svd_components=500,tau=.6,
                                mask_refiner="crf",resize_to_orig_size=False,device="cuda").eval().requires_grad_(False)
                    from tics.ten_direction_context import TenDirectionContext, ViewBudget, capture_exact_native_and_middle
                    support=Image.open(Path(manifest["data_root"])/row["support"]).convert("RGB")
                    query=Image.open(Path(manifest["data_root"])/row["query"]).convert("RGB")
                    reference=torch.from_numpy((np.asarray(Image.open(Path(manifest["annotation_root"])/Path(row["support"]).with_suffix(".png"))) == row["c"]+1).copy())
                    captured={}
                    try:
                        host.set_reference(support,reference);host.set_target(query)
                        with capture_exact_native_and_middle(host,encoder) as captured:
                            public=host.segment().reshape(1024,1024).bool().clone()
                    finally:
                        host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None
                    budget=ViewBudget(max_extra_b2=0,native_b2_calls=1,
                                      actual_b2_calls=captured.get("encoder_forward_calls",0))
                    ctx=TenDirectionContext(host,encoder,support,query,reference,
                        raw_native=captured["raw_native"],layer_maps={6:captured[6],12:captured[12]},budget=budget)
                    native=ctx.initialize_native()
                    if not torch.equal(public,native["mask"]): raise RuntimeError("Uncached/cached complete public native differs")
                    native["uncached_public_exact"]=True
                    old=manifest.get("baseline_rows",{}).get(stem(row))
                    if old:
                        if sha(old["packet"]) != old["packet_sha256"]: raise RuntimeError("Old complete baseline packet changed")
                        with np.load(old["packet"],allow_pickle=False) as packet:
                            old_native=np.unpackbits(packet["native"])[:1024*1024].reshape(1024,1024).astype(bool)
                        if not np.array_equal(old_native,native["mask"].cpu().numpy()):
                            raise RuntimeError("Complete native DEV mask gate failed before any query GT")
                        native["current_DEV_baseline_bit_exact"]=True
                    frozen=freeze_case(ctx,row,a.out,man_sha)
                    del ctx, public, native, captured
                    rec=score_frozen(frozen,manifest)
                append_record(a.out/"episodes.jsonl",rec);records.append(rec);done[row_key(row)]=rec;save()
                print(json.dumps(dict(state="CASE_SCORED",case=stem(row),episodes=len(records),
                    original_iu=rec["original_iu"],native_original_IU_exact=rec["baseline_original_IU_exact"])),flush=True)
            else: report["state"]="COMPLETED"
        save()
    except BaseException as exc:
        error=dict(state="ERROR_SOFTWARE_OR_INTERFACE",error_type=type(exc).__name__,message=str(exc),
                   traceback=traceback.format_exc(),not_a_method_negative=True,completed_cases=len(records))
        write_json(a.out/"errors"/("error_%d.json"%time.time_ns()),error)
        report.update(error);save();raise


def score_only(a):
    manifest=json.loads(a.manifest.read_text()); validate_manifest(manifest,a)
    committed={row_key(r["row"]) for r in load_records(a.out/"episodes.jsonl")}
    for row in manifest["episodes"]:
        if row_key(row) in committed: continue
        path=a.out/"frozen"/(stem(row)+".json")
        if not path.exists(): continue
        receipt=json.loads(path.read_text())
        if receipt["manifest_sha256"] != sha(a.manifest): raise RuntimeError("Scoring manifest drift")
        rec=score_frozen(receipt,manifest);append_record(a.out/"episodes.jsonl",rec)
        print(json.dumps(dict(state="CPU_SCORED_FROZEN",case=stem(row))),flush=True)


def analyze(a):
    """CPU-only class-sum paired analysis, photograph-connected resampling."""
    import numpy as np
    records=load_records(a.out/"episodes.jsonl")
    if not records: raise ValueError("No committed frozen/scored cases")
    for r in records:
        if (not r["predictions_frozen_before_query_GT"] or not r["full_native_exact"] or
            sha(r["prediction_path"]) != r["predictions_sha256"]): raise RuntimeError("Frozen result identity gate")
    def score(rs,arm,key="original_iu"):
        sums={}
        for r in rs:
            c=r["row"]["c"];i,u=r[key][arm];v=sums.setdefault(c,[0,0]);v[0]+=i;v[1]+=u
        return 100*float(np.mean([i/max(u,1) for i,u in sums.values()]))
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(records):
        for role in ("support","query"):
            photo=r["row"][role]
            if photo in seen:parent[find(i)]=find(seen[photo])
            else:seen[photo]=i
    group_map={}
    for i,r in enumerate(records):group_map.setdefault(find(i),[]).append(r)
    groups=list(group_map.values());rng=np.random.default_rng(4021)
    draws=[[r for j in rng.integers(0,len(groups),len(groups)) for r in groups[j]] for _ in range(a.bootstrap_draws)]
    pairs=[(arm,"native") for arm in ARM_IDS if arm!="native"]
    pairs += [("full001","anchor_only"),("perFG002","full001"),("perFG002","matchedGlobal")]
    pairs += [(arm,"querycore") for arm in ARM_IDS if arm not in ("native","querycore")]
    comparisons={}
    for arm,base in pairs:
        delta=[score(rs,arm)-score(rs,base) for rs in draws]
        comparisons[arm+"__minus__"+base]=dict(delta_pp=score(records,arm)-score(records,base),
            photo_connected_paired_CI95=np.quantile(delta,[.025,.975]).tolist(),
            folds={str(f):score([r for r in records if r["row"]["fold"]==f],arm)-score([r for r in records if r["row"]["fold"]==f],base)
                   for f in range(4) if any(r["row"]["fold"]==f for r in records)})
    metrics={}
    for arm in ARM_IDS:
        values=[r["original_iu"][arm][0]/max(r["original_iu"][arm][1],1) for r in records]
        changes=[r["original_iu"][arm][0]/max(r["original_iu"][arm][1],1)-r["original_iu"]["native"][0]/max(r["original_iu"]["native"][1],1) for r in records]
        class_ledger={}
        for r in records:
            x=class_ledger.setdefault(str(r["row"]["c"]),dict(native_I=0,native_U=0,deleted_FP=0,lost_TP=0,added_TP=0,added_FP=0))
            i,u=r["original_iu"]["native"];x["native_I"]+=i;x["native_U"]+=u
            for k in ("deleted_FP","lost_TP","added_TP","added_FP"):x[k]+=r["original_error_ledger"][arm][k]
        for x in class_ledger.values():
            x["exact_IoU_improvement_numerator"]=x["native_U"]*(x["added_TP"]-x["lost_TP"])+x["native_I"]*(x["deleted_FP"]-x["added_FP"])
        metrics[arm]=dict(original_class_sum_mIoU_pp=score(records,arm),original_episode_mean_IoU_pp=100*float(np.mean(values)),
            model_class_sum_mIoU_pp=score(records,arm,"model_iu"),
            up=int(sum(x>0 for x in changes)),down=int(sum(x<0 for x in changes)),tie=int(sum(x==0 for x in changes)),
            original_error_ledger={k:sum(r["original_error_ledger"][arm][k] for r in records) for k in ("TP","FP","FN","deleted_FP","lost_TP","added_TP","added_FP","changed_pixels")},
            model_error_ledger={k:sum(r["model_error_ledger"][arm][k] for r in records) for k in ("TP","FP","FN","deleted_FP","lost_TP","added_TP","added_FP","changed_pixels")},
            per_class_IoU_improvement_ledger=class_ledger,
            changed_pixel_GT_oracle_class_mIoU_pp=score(records,arm,"changed_original_pixel_oracle_iu"))
    manifest=json.loads(a.manifest.read_text())
    result=dict(state="CPU_F4_SEVEN_ANALYZED",episodes=len(records),expected_episodes=len(manifest["episodes"]),
        scope=manifest["scope"],design_DEV=manifest["design_DEV"],all_four_folds_present={r["row"]["fold"] for r in records}=={0,1,2,3},
        photo_connected_groups=len(groups),bootstrap_seed=4021,bootstrap_draws=a.bootstrap_draws,
        metrics=metrics,comparisons=comparisons,
        actual_encoder_B2_calls=sum(r["actual_encoder_B2_calls"] for r in records),
        extra_encoder_B2_calls=sum(r["extra_encoder_B2_calls"] for r in records),
        prototype_budget_book={stem(r["row"]):r["method_audits"] for r in records},
        claim_limits=["Old smoke8/DEV241 are design evidence, not an independent method score",
            "Changed-pixel GT oracle is not deployable or a reachable-information certificate",
            "Class IoU condition is evaluated per class; aggregate deleted_FP/lost_TP is not an mIoU proof",
            "Shared-server quality elapsed time is not a formal speed comparison"],card=CARD)
    write_json(a.out/"analysis.json",result)
    print(json.dumps(dict(state=result["state"],episodes=len(records),original_mIoU={k:v["original_class_sum_mIoU_pp"] for k,v in metrics.items()})),flush=True)


def cpu_smoke(a):
    """Root/server runs this; no local numerical execution by preparer."""
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
    from ten_direction_context_cpu import make_fixture
    torch.set_num_threads(1)
    if torch.cuda.is_initialized(): raise RuntimeError("CPU smoke initialized CUDA")
    if a.out.exists() and any(a.out.iterdir()): raise ValueError("Fresh CPU output required")
    (a.out/"predictions").mkdir(parents=True);(a.out/"frozen").mkdir()
    cases=[]
    with torch.inference_mode():
        for j in range(10):
            ctx,encoder,audit=make_fixture(a.foris_root,seed=7100+j,image_size=128)
            row=dict(fold=j%4,e=j,c=j,support="fixture_s",query="fixture_q")
            before=encoder.calls
            receipt=freeze_case(ctx,row,a.out,"CPU_SYNTHETIC_SOURCE_FIXTURE")
            if encoder.calls!=before or receipt["actual_encoder_B2_calls"]!=1:
                raise RuntimeError("F4 replay invoked encoder")
            with np.load(receipt["prediction_path"],allow_pickle=False) as pack:
                if any(pack[name].shape!=(128*128//8,) for name in ARM_IDS):raise RuntimeError("Packed working masks differ")
            if not ctx.native.get("identity_exact"):raise RuntimeError("Native callbacks not exact")
            # Labels for a scoring arithmetic fixture are made only after all masks freeze.
            arbitrary_truth=np.zeros((128,128),dtype=bool);arbitrary_truth[32:96,32:96]=True
            with np.load(receipt["prediction_path"],allow_pickle=False) as pack:
                masks={k:np.unpackbits(pack[k]).reshape(128,128).astype(bool) for k in ARM_IDS}
            for name in ARM_IDS:
                count_ledger(masks[name],masks["native"],arbitrary_truth)
            cases.append(dict(case=j,seven_arms_frozen=True,encoder_calls=encoder.calls,
                full_native_exact=True,original_source_executed=True,no_query_GT_in_decisions=True,
                source_fixture=audit,degenerate_states=receipt["method_audits"]))
    files=source_files(a.foris_root,a.demo4_root)
    proof=dict(state="CPU_F4_SEVEN_SOURCE_PASSED",cases=10,records=cases,
        CUDA_initialized=False,no_query_GT_in_decisions=True,source_hashes={str(p.resolve()):sha(p) for p in files},
        scope="Actual public source with D8/image128/rank2 synthetic encoder and bilinear refiner; not production DINO/CRF or segmentation gain")
    write_json(a.out/"cpu_receipt.json",proof)
    print(json.dumps(dict(state=proof["state"],cases=10,CUDA_initialized=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    for mode in ("prepare","run","analyze","score-only","cpu-smoke"):modes.add_argument("--"+mode,action="store_true")
    p.add_argument("--manifest",type=Path);p.add_argument("--from-manifest",type=Path)
    p.add_argument("--scope",choices=("smoke8","dev241","frozen_unseen"),default="smoke8")
    p.add_argument("--baseline-run",type=Path);p.add_argument("--unseen-receipt",type=Path)
    p.add_argument("--projection-basis",type=Path);p.add_argument("--cpu-receipt",type=Path)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--allow-gpu",action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--max-pairs",type=int)
    p.add_argument("--budget-seconds",type=int,default=1800);p.add_argument("--memory-fraction",type=float,default=.35)
    p.add_argument("--max-output-bytes",type=int,default=512*1024**2);p.add_argument("--reserve-bytes",type=int,default=5*1024**3)
    p.add_argument("--bootstrap-draws",type=int,default=2000)
    p.add_argument("--foris-root",type=Path);p.add_argument("--demo4-root",default="/root/autodl-tmp/demo4")
    a=p.parse_args()
    if a.cpu_smoke:cpu_smoke(a)
    elif a.prepare:prepare(a)
    elif a.run:run(a)
    elif a.score_only:score_only(a)
    else:analyze(a)


if __name__=="__main__":main()
