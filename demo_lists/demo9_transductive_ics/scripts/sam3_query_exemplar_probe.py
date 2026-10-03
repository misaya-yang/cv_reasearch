#!/usr/bin/env python3
"""Bounded C2 probe, prepared statically; no boot, fitting or asset download.

One legal image encoding supplies independent prompt states. The original DEV
index and saved source box are retained. Query labels are opened only after the
whole legal40 cohort is frozen; an optional true-query-box phase is privileged
and pays for a separate encoding. Real masks/source are mandatory dependencies.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

GPU_SHA = "276a53fd8c5fd1fa45ce64317b695c22feef247ee25f3bade14b7587cedbc07d"
PROCESSOR_SHA = "d8738a0efb6138b01c0dc5deceffd29de9e675860a9a1ed3822b08766373333b"
ROOT = Path(__file__).resolve().parents[1]
ARMS = ("native", "foris", "query_foris", "query_self", "query_random", "query_foris_only", "cheap_OR", "cheap_AND")
CARD = [
    "Assumption: a legal second positive box on the query supplies localization evidence unavailable to the source-only stitched interface, rather than merely adding a prompt.",
    "Hypothesis predictions, not measured gains: query_foris-native >=2pp on old DEV40, >=.5pp versus query_self/random, paired lower bounds>0 and >=3 positive folds; every real native/NoOp parity gate has zero differing pixels. Query_foris must also beat both cheap unions/intersections before attributing value to re-prompting.",
    "Match: retain only this fixed query-box hypothesis for later independent testing; true-query-box is a separately labelled GT geometry-input replacement, with no perfect-concept claim.",
    "Mismatch: native/state/schema failure is an interface error and stops; no win over random/self indicates extra-prompt rather than locator evidence; no win over OR/AND withdraws prompt-mechanism value. No box/threshold/position sweep on these exposed tasks.",
]


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def key(row): return tuple(int(row[name]) for name in ("fold", "e", "c"))
def stem(row): return "%d_%d_%d" % key(row)


def unique_index(records,label):
    result={}
    for row in records:
        identifier=key(row)
        if identifier in result:raise ValueError("Duplicate (fold,e,c) in "+label+": "+str(identifier))
        result[identifier]=row
    return result


def require_same_photos(row,other,label):
    if key(row)!=key(other) or any(row[role]!=other.get(role) for role in ("support","query")):
        raise ValueError("Key or S/Q mismatch in "+label+": "+str(key(row)))


def frozen_base(path):
    if sha(path) != GPU_SHA: raise RuntimeError("Original frozen v2 GPU/score source is required")
    spec = importlib.util.spec_from_file_location("c2_original_sam3", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def require_files(files):
    for path, digest in files.items():
        if not Path(path).is_file(): raise FileNotFoundError("Real dependency unavailable; no synthetic replacement: " + path)
        if sha(path) != digest: raise RuntimeError("Frozen dependency changed: " + path)


def prepare(a):
    """Metadata only; this does not award a CPU/model acceptance state."""
    original = read(a.dev_manifest)
    if len(original.get("episodes", [])) != 241: raise ValueError("Original DEV241 manifest required")
    unique_index(original["episodes"],"original DEV manifest")
    source = rows(a.sam_native / "predictions_shard0.jsonl")
    indexed = unique_index(source,"SAM prediction ledger")
    scored_sam=unique_index(rows(a.sam_native/"episodes_shard0.jsonl"),"SAM scored ledger")
    scored_foris=unique_index(rows(a.foris_records),"FoRIS scored ledger")
    if len(indexed) != 241: raise ValueError("Complete real SAM3 prediction ledger required")
    freeze = read(a.sam_native / "prediction_freeze_shard0.json")
    if (freeze.get("state") != "PREDICTIONS_FROZEN" or freeze.get("source_sha256") != GPU_SHA or
        freeze.get("manifest_sha256") != sha(a.dev_manifest) or
        freeze.get("predictions_jsonl_sha256") != sha(a.sam_native / "predictions_shard0.jsonl")):
        raise ValueError("Original SAM native prediction freeze identity required")
    asset_index = {item["path"]: item for item in freeze["prediction_files"]}
    per_fold = {fold: [] for fold in range(4)}
    for global_index, row in enumerate(original["episodes"]):
        if len(per_fold[row["fold"]]) < 10:
            saved = indexed[key(row)]
            require_same_photos(row,scored_sam[key(row)],"SAM source/scored join")
            require_same_photos(row,scored_foris[key(row)],"FoRIS source/scored join")
            if saved["global_manifest_index"] != global_index or any(saved[role] != row[role] for role in ("support", "query")):
                raise ValueError("Subset re-indexing would change the source exemplar")
            if len(saved.get("exemplar_box", [])) != 4 or len(saved.get("query_shape", [])) != 2:
                raise ValueError("Saved original source bbox and original query geometry required")
            mask = a.sam_native / saved["prediction_file"]
            packet = a.foris_packets / (stem(row) + ".npz")
            if not mask.is_file() or not packet.is_file():
                raise FileNotFoundError("Real native masks unavailable: " + str(mask) + " / " + str(packet))
            asset = asset_index[saved["prediction_file"]]
            if mask.stat().st_size != asset["bytes"] or sha(mask) != asset["sha256"]:
                raise ValueError("Native SAM packed prediction differs from original immutable freeze")
            per_fold[row["fold"]].append({"row": row, "original_global_index": global_index,
                "source_bbox": saved["exemplar_box"], "query_shape": saved["query_shape"],
                "saved_sam_mask": str(mask.resolve()), "foris_packet": str(packet.resolve())})
    if any(len(group) != 10 for group in per_fold.values()): raise ValueError("Ten original first cases per fold required")
    cases = [per_fold[fold][i] for i in range(10) for fold in range(4)]
    base_path = ROOT / "scripts/sam3_stitch_gpu_v2.py"
    processor = a.sam3 / "sam3/model/sam3_image_processor.py"
    builder = a.sam3 / "sam3/model_builder.py"
    for path in (base_path, processor, builder):
        if not path.is_file(): raise FileNotFoundError("Actual original implementation required: " + str(path))
    if sha(base_path) != GPU_SHA: raise ValueError("Original A predictor source drift")
    if sha(processor) != PROCESSOR_SHA: raise ValueError("Pinned actual processor differs from reviewed source contract")
    if a.previous_plan:
        previous = read(a.previous_plan)
        for path in (processor, builder):
            if previous["source_hashes"].get(str(path.resolve())) != sha(path):
                raise ValueError("Official source differs from completed A")
    status = read(a.checkpoint_status)
    if (status.get("state") != "WEIGHT_VERIFIED" or status.get("actual_bytes") != 3450062241 or
        status.get("actual_sha256") != "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e" or
        a.checkpoint.stat().st_size != 3450062241): raise ValueError("Existing official verified checkpoint required")
    if not 120 <= a.budget_seconds <= 300: raise ValueError("Minute-scale finite total cap120..300s required")
    inputs = {str(path.resolve()): sha(path) for path in
        (a.dev_manifest, a.sam_native / "predictions_shard0.jsonl", a.sam_native / "prediction_freeze_shard0.json",
         a.foris_records, a.sam_native / "episodes_shard0.jsonl", a.checkpoint_status)}
    for case in cases:
        for name in ("saved_sam_mask", "foris_packet"): inputs[case[name]] = sha(case[name])
    sources = {str(path.resolve()): sha(path) for path in (Path(__file__), ROOT/"scripts/sam3_query_exemplar_cpu.py", base_path, processor, builder,
                                                        ROOT / "scripts/experiment_resource_guard.py")}
    manifest = dict(state="STATIC_METADATA_PREPARED_PENDING_SERVER_CPU_PREFLIGHT", schema="sam3_query_exemplar_v1",
        cases=cases, arms=list(ARMS), original_DEV_manifest=str(a.dev_manifest.resolve()),
        data_root=original["data_root"], annotation_root=original["annotation_root"],
        original_sam_scored_records=str((a.sam_native / "episodes_shard0.jsonl").resolve()),
        original_foris_records=str(a.foris_records.resolve()), base_source=str(base_path),
        sam3=str(a.sam3.resolve()), checkpoint=str(a.checkpoint.resolve()), checkpoint_status=str(a.checkpoint_status.resolve()),
        sources=sources, inputs=inputs, out=str(a.out.resolve()), card=CARD,
        query_box_rule="tight bbox of all predicted foreground pixels; empty means no added prompt",
        random_rule="same FoRIS bbox width/height; only location uniform from SHA256(S/Q UID, original index) seed",
        source_box_rule="saved original box AND component_box(Random(original DEV241 global index)) must agree",
        cached_state_rule="one set_image; recursive tensor cloning from prompt-free encoded state per arm; unknown state types fail",
        baseline_mapping="old A query crop nearest->original; FoRIS native1024 binary CUDA bilinear->original strictly>.5",
        oracle_enabled=a.with_oracle, oracle_scope="true query bbox after entire legal40 freeze; separate re-encoding charged",
        hard_cap_seconds=a.budget_seconds, reserve_bytes=5*1024**3, maximum_output_bytes=96*1024**2,
        exposed_DEV=True, independent_confirmation=False, static_only=True, GPU_started=False,
        source_API_state_clone_real_model_checked=False)
    manifest["cost_scope"]="Cached real FoRIS masks are reused in this hypothesis probe; their original DINO/decoder production cost is excluded here and must be added for deployment costs. No speed claim."
    manifest["query_only_role"]="Optional source-label reconstruction diagnostic: no source positive box when active; never selects a legal primary. Empty FoRIS mask explicitly abstains to original native."
    if (a.out/"manifest.json").exists(): raise ValueError("Preserve old preparation; use a fresh output directory")
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "manifest.json", manifest)
    print(json.dumps({"state": manifest["state"], "cases": 40, "GPU_started": False}))


def cpu_preflight(a):
    """SERVER only: real source and real cached-mask schema, no fake processor."""
    if not sys.platform.startswith("linux") or not str(ROOT).startswith("/root/"):
        raise RuntimeError("CPU preflight is server-only; no local numerical/model check")
    import numpy as np
    manifest = read(a.manifest); require_files(manifest["sources"]); require_files(manifest["inputs"])
    source_path = Path(manifest["sam3"]) / "sam3/model/sam3_image_processor.py"
    if sha(source_path)!=PROCESSOR_SHA: raise RuntimeError("Pinned reviewed official processor changed")
    tree = ast.parse(source_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Sam3Processor")
    methods = {node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    needed = {"set_image", "reset_all_prompts", "add_geometric_prompt", "_forward_grounding"}
    if not needed <= methods: raise RuntimeError("Pinned public processor API changed")
    methods_source={node.name:ast.get_source_segment(source_path.read_text(),node)
                    for node in cls.body if isinstance(node,ast.FunctionDef)}
    init=methods_source["__init__"];image=methods_source["set_image"]
    geo=methods_source["add_geometric_prompt"];reset=methods_source["reset_all_prompts"];ground=methods_source["_forward_grounding"]
    source_contracts={
        "public_processor_default_resolution1008": "resolution=1008" in init,
        "public_confidence_threshold_point5": "confidence_threshold=0.5" in init,
        "set_image_real_backbone_cache": 'state["backbone_out"] = self.model.backbone.forward_image(image)' in image,
        "geometry_default_visual_dummy": '["visual"]' in geo and "forward_text" in geo,
        "geometry_mutates_nested_language_dict": 'state["backbone_out"].update(dummy_text_outputs)' in geo,
        "geometry_appends_actual_prompt_boxes": '.append_boxes(boxes, labels)' in geo,
        "positive_box_source_dtype_roles": "dtype=torch.float32" in geo and "dtype=torch.bool" in geo,
        "reset_removes_nested_language_features": 'del state["backbone_out"][key]' in reset and '"language_features"' in reset,
        "reset_removes_geometric_prompt": '"geometric_prompt"' in reset and "del state[key]" in reset,
        "grounding_hard_masks_from_probability": 'state["masks"] = out_masks > 0.5' in ground and 'state["masks_logits"] = out_masks' in ground,
    }
    if not all(source_contracts.values()): raise RuntimeError("Official source contract changed")
    for case in manifest["cases"]:
        with np.load(case["saved_sam_mask"], allow_pickle=False) as packed:
            if list(packed["shape"]) != case["query_shape"] or packed["visual"].dtype != np.uint8 or packed["visual"].ndim != 1:
                raise RuntimeError("Real SAM original packed-mask schema changed")
            if packed["visual"].size != (int(np.prod(case["query_shape"]))+7)//8:
                raise RuntimeError("Truncated original SAM mask")
        with np.load(case["foris_packet"], allow_pickle=False) as packet:
            # Query GT exists as another key in this legacy packet. Never read it.
            value = packet["native"]
            if value.dtype != np.uint8 or value.shape != (1024*1024//8,):
                raise RuntimeError("Real complete FoRIS native1024 packed schema changed")
    from sam3_query_exemplar_cpu import interface_cases, score_fixture
    interface=interface_cases(manifest["sam3"],manifest["base_source"])
    score_bench=score_fixture(manifest["base_source"])
    if score_bench["elapsed_seconds"]>15:
        write(a.out/"cpu_score_not_ready.json",score_bench)
        raise RuntimeError("Complete synthetic40 score exceeded the preset15s CPU phase; no guard generated")
    receipt = dict(state="SERVER_ACTUAL_PROCESSOR_TINY10_SCHEMA_SCORE_PASSED", manifest_sha256=sha(a.manifest),
        source_hashes=manifest["sources"], cases=40, fake_processor=False, synthetic_tiny_model=True, query_GT_pixels_read=False,
        CUDA_initialized=False, real_model_state_clone_or_segmentation_parity_checked=False,
        source_contract_checks=source_contracts,source_contract_only_not_model_interface_tests=True,
        interface_fixture=interface,synthetic_score_timing=score_bench,
        remaining_gate="First8 real CUDA image-cache/NoOp/original-native parity; source/schema checks are not inference evidence")
    write(a.out / "cpu_preflight.json", receipt)
    final = {"path": str(a.out.resolve() / "report.json"), "json_equals": {"state": "COMPLETED"}}
    env = {"DEMO9_CUDA_GUARD": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "PYTHONPATH": str(Path(manifest["sam3"]).parent / "python") + ":" + manifest["sam3"] + ":/root/demo4_cache/env"}
    requires=[{"path": str(a.out.resolve()/"cpu_preflight.json"),
                  "json_equals": {"state": receipt["state"]}, "sha256": sha(a.out/"cpu_preflight.json")},
                  {"path": str(a.manifest.resolve()), "sha256": sha(a.manifest)}]
    cap=manifest["hard_cap_seconds"];legal_seconds=int(cap*.55);oracle_seconds=int(cap*.35)
    legal={"path":str(a.out.resolve()/"legal_freeze.json"),"json_equals":{"state":"ALL_LEGAL40_PREDICTIONS_FROZEN","episodes":40}}
    oracle={"path":str(a.out.resolve()/"oracle_freeze.json"),"json_equals":{"state":"ORACLE40_PREDICTIONS_FROZEN","episodes":40}}
    def stage(name,mode,kind,seconds,dependencies,output):
        return dict(name=name,kind=kind,**({"role":"finalize"} if kind=="cpu" else {}),cwd=str(ROOT),timeout_seconds=seconds,
            argv=[a.python,str(Path(__file__).resolve()),"--"+mode,"--manifest",str(a.manifest.resolve()),"--out",str(a.out.resolve())],
            env=env,requires=[*requires,*dependencies],cpu_artifacts=[{"path":path,"sha256":digest} for path,digest in manifest["inputs"].items()],
            code_files=list(manifest["sources"]),produces=[output],success_checks=[output])
    stages=[stage("C2_legal40","legal","gpu",legal_seconds,[],legal)]
    if manifest["oracle_enabled"]:stages.append(stage("C2_GT_bbox_input40","oracle","gpu",oracle_seconds,[legal],oracle))
    stages.append(stage("C2_CPU_frozen_score40","score","cpu",15,[legal,*([oracle] if manifest["oracle_enabled"] else [])],final))
    plan={"platform":"autodl","cuda_python":a.python,"stages":stages,
          "total_timeout_budget_seconds":cap,"sum_stage_timeouts_seconds":sum(s["timeout_seconds"] for s in stages),
          "startup_and_handoff_reserve_seconds":cap-sum(s["timeout_seconds"] for s in stages)}
    write(a.out / "guard_plan.json", plan)
    print(json.dumps({"state": receipt["state"], "real_model_checked": False, "GPU_started": False}))


def clone_image_state(value):
    """Independent tensors, no shallow alias to the original image cache."""
    import torch
    if torch.is_tensor(value): return value.clone()
    if isinstance(value, dict): return {key: clone_image_state(item) for key, item in value.items()}
    if isinstance(value, list): return [clone_image_state(item) for item in value]
    if isinstance(value, tuple): return tuple(clone_image_state(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise RuntimeError("Unverified image-state type; no deepcopy/fake handle fallback: " + type(value).__module__ + "." + type(value).__name__)


def equal_image_state(left, right):
    import torch
    if torch.is_tensor(left): return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict): return isinstance(right, dict) and left.keys()==right.keys() and all(equal_image_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple,list)): return type(left) is type(right) and len(left)==len(right) and all(equal_image_state(x,y) for x,y in zip(left,right))
    return left == right


def bbox(mask):
    import numpy as np
    y, x = np.nonzero(mask)
    if not len(x): return None
    return [int(x.min()), int(y.min()), int(x.max()-x.min()+1), int(y.max()-y.min()+1)]


def query_box_on_canvas(box, shape, base):
    h,w = shape; x,y,bw,bh = box; qx,qy,qw,qh = base.rectangles()[1]
    if not (bw>0 and bh>0 and 0<=x<=w-bw and 0<=y<=h-bh): raise ValueError("Query bbox outside original image")
    return [(qx+(x+bw/2)*qw/w)/base.CANVAS, (qy+(y+bh/2)*qh/h)/base.CANVAS,
            bw*qw/w/base.CANVAS, bh*qh/h/base.CANVAS]


def random_box(box, case):
    if box is None: return None
    h,w = case["query_shape"]; _,_,bw,bh = box
    seed = hashlib.sha256((case["row"]["support"]+"|"+case["row"]["query"]+"|"+str(case["original_global_index"])).encode()).digest()
    rng = random.Random(int.from_bytes(seed[:16],"big"))
    return [rng.randint(0,w-bw),rng.randint(0,h-bh),bw,bh]


def make_model(manifest):
    import torch
    base = frozen_base(manifest["base_source"])
    sys.path.insert(0, manifest["sam3"])
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.cuda.set_per_process_memory_fraction(.3); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model = build_sam3_image_model(bpe_path=str(Path(manifest["sam3"])/"sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        device="cuda", checkpoint_path=manifest["checkpoint"], load_from_HF=False,
        eval_mode=True, enable_inst_interactivity=False, compile=False).float().eval()
    if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()): raise RuntimeError("Original FP32 contract failed")
    precision = base.configure_fp32_mlp_backend(model)
    return base, Sam3Processor(model, device="cuda", confidence_threshold=.5), precision


def prompted(processor, image_state, source_box, extra_box, base):
    import torch
    state = clone_image_state(image_state); processor.reset_all_prompts(state)
    original_find = processor.find_stage
    # FindStage is another shared processor object passed to the model. Use the
    # actual source class and exactly the source fields, without sharing tensors.
    fields=("img_ids","text_ids","input_boxes","input_boxes_mask","input_boxes_label","input_points","input_points_mask")
    processor.find_stage=type(original_find)(**{name:clone_image_state(getattr(original_find,name)) for name in fields})
    seen = []; inner = processor.model.forward_grounding
    def capture(**kwargs):
        output = inner(**kwargs)
        if any(output[name].dtype!=torch.float32 for name in ("pred_logits","pred_masks","presence_logit_dec")):
            raise RuntimeError("Official grounding output violates original FP32 contract")
        seen.append(True); return output
    processor.model.forward_grounding = capture
    try:
        source_union=None
        if source_box is not None:
            state = processor.add_geometric_prompt(state=state, box=source_box, label=True)
            masks = state["masks"]
            source_union = masks[:,0].any(0).clone() if masks.shape[0] else torch.zeros((base.CANVAS,base.CANVAS),dtype=torch.bool,device=masks.device)
        if extra_box is not None: state = processor.add_geometric_prompt(state=state, box=extra_box,label=True)
        if source_box is None and extra_box is None: raise ValueError("No real positive exemplar exists")
        masks = state["masks"]
        union = masks[:,0].any(0) if masks.shape[0] else torch.zeros((base.CANVAS,base.CANVAS),dtype=torch.bool,device=masks.device)
        return union.clone(), source_union, len(seen)
    finally:
        processor.model.forward_grounding = inner
        processor.find_stage = original_find


def joint_query_prompt(processor,image_state,source_box,box,shape,base,native_canvas):
    """Shared production/fixture path: a genuine empty predictor adds no box."""
    import torch
    if box is None:return native_canvas.clone(),0
    result,noop,calls=prompted(processor,image_state,source_box,query_box_on_canvas(box,shape,base),base)
    if not torch.equal(noop,native_canvas):raise RuntimeError("Fresh cloned source-only NoOp/state-isolation failed")
    return result,calls


def paired_all_arms(records,arms,draws=2000):
    """Same seed/group draws/class-sum estimator as frozen paired(), once."""
    import numpy as np
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,row in enumerate(records):
        for role in ("support","query"):
            photo=row.get(role)
            if photo is None:continue
            if photo in seen:parent[find(i)]=find(seen[photo])
            else:seen[photo]=i
    groups={}
    for i in range(len(records)):groups.setdefault(find(i),[]).append(i)
    groups=list(groups.values());classes=sorted({r["c"] for r in records});class_idx={c:i for i,c in enumerate(classes)}
    totals=np.zeros((len(groups),len(classes),len(arms),2),np.float64)
    present=np.zeros((len(groups),len(classes)),np.int32)
    for g,indices in enumerate(groups):
        for i in indices:
            c=class_idx[records[i]["c"]];present[g,c]+=1
            for j,arm in enumerate(arms):totals[g,c,j]+=records[i]["original_iu"][arm]
    pick=np.random.default_rng(0).integers(0,len(groups),size=(draws,len(groups)))
    weights=np.zeros((draws,len(groups)),np.int32)
    np.add.at(weights,(np.arange(draws)[:,None],pick),1)
    sampled=np.einsum("dg,gcai->dcai",weights,totals,optimize=True)
    valid=weights@present>0
    per_class=sampled[...,0]/np.maximum(sampled[...,1],1)
    values=100*(per_class*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
    original=totals.sum(0);raw=100*np.mean(original[...,0]/np.maximum(original[...,1],1),axis=0)
    meta=dict(bootstrap_unit="connected support/query photograph groups",groups=len(groups),
              largest_group=max(map(len,groups)),draws=draws,seed=0)
    return {arm:float(raw[j]) for j,arm in enumerate(arms)}, {arm:values[:,j] for j,arm in enumerate(arms)},meta


def validate_freeze(path, manifest_path, phase, count):
    freeze = read(path)
    if (freeze.get("state") != phase or freeze.get("episodes") != count or
        freeze.get("manifest_sha256") != sha(manifest_path)): raise RuntimeError("Complete cohort freeze required")
    if phase=="ALL_LEGAL40_PREDICTIONS_FROZEN" and freeze.get("query_GT_used_as_input") is not False:
        raise RuntimeError("Legal cohort query-GT embargo violated")
    require_files(freeze["files"])
    return freeze


def infer(a, oracle=False):
    import numpy as np
    import torch
    from PIL import Image
    if os.environ.get("DEMO9_CUDA_GUARD")!="1": raise RuntimeError("Only a future single resource guard may run CUDA")
    manifest = read(a.manifest); require_files(manifest["sources"]); require_files(manifest["inputs"])
    receipt = read(a.out/"cpu_preflight.json")
    if (receipt.get("state")!="SERVER_ACTUAL_PROCESSOR_TINY10_SCHEMA_SCORE_PASSED" or
        receipt.get("manifest_sha256")!=sha(a.manifest) or receipt.get("source_hashes")!=manifest["sources"]):
        raise RuntimeError("Actual server CPU source/schema receipt required; static preparation is insufficient")
    if oracle: validate_freeze(a.out/"legal_freeze.json",a.manifest,"ALL_LEGAL40_PREDICTIONS_FROZEN",40)
    own_freeze=a.out/("oracle_freeze.json" if oracle else "legal_freeze.json")
    if own_freeze.exists():
        validate_freeze(own_freeze,a.manifest,"ORACLE40_PREDICTIONS_FROZEN" if oracle else "ALL_LEGAL40_PREDICTIONS_FROZEN",40)
        print(json.dumps(dict(state="REUSED_COMPLETED_COHORT",new_image_encodings=0,phase="oracle" if oracle else "legal")),flush=True);return
    phase="oracle" if oracle else "legal"; dest=a.out/phase; dest.mkdir(parents=True,exist_ok=True)
    ledger = dest/"predictions.jsonl"; existing=rows(ledger) if ledger.exists() else []
    if [key(r["row"]) for r in existing]!=[key(c["row"]) for c in manifest["cases"][:len(existing)]]:
        raise RuntimeError("Frozen resume prefix mismatch")
    for rec in existing: require_files({rec["path"]:rec["sha256"]})
    prefix_count=len(existing);start=time.monotonic();base=processor=None;precision=None
    try:
        with torch.inference_mode(), ledger.open("a") as stream:
            for case in manifest["cases"][len(existing):]:
                if shutil.disk_usage(a.out).free<manifest["reserve_bytes"]: raise RuntimeError("5GiB reserve gate")
                if base is None: base,processor,precision=make_model(manifest)
                row=case["row"]; data=Path(manifest["data_root"]); ann=Path(manifest["annotation_root"])
                support=Image.open(data/row["support"]).convert("RGB"); query=Image.open(data/row["query"]).convert("RGB")
                if [query.height,query.width]!=case["query_shape"] or row["support"]==row["query"]:
                    raise RuntimeError("Original query dimensions/independent reference roles changed")
                ref=np.asarray(Image.open(ann/Path(row["support"]).with_suffix(".png")))==row["c"]+1
                old_box=base.component_box(ref,random.Random(case["original_global_index"]))
                if old_box!=case["source_bbox"]: raise RuntimeError("Original indexed source component/box changed")
                canvas,source_box=base.stitch(support,query,old_box)
                with np.load(case["saved_sam_mask"],allow_pickle=False) as saved:
                    old_native=np.unpackbits(saved["visual"])[:query.height*query.width].reshape(case["query_shape"]).astype(bool)
                with np.load(case["foris_packet"],allow_pickle=False) as packet:
                    working=np.unpackbits(packet["native"])[:1024*1024].reshape(1024,1024).copy()
                foris=torch.nn.functional.interpolate(torch.from_numpy(working).to("cuda").float()[None,None],
                    size=tuple(case["query_shape"]),mode="bilinear",align_corners=False)[0,0].cpu().numpy()>.5
                counter=[0]; forward=processor.model.backbone.forward_image
                def image_forward(*args,**kwargs): counter[0]+=1; return forward(*args,**kwargs)
                processor.model.backbone.forward_image=image_forward
                try:
                    image_state=processor.set_image(canvas); immutable=clone_image_state(image_state)
                    native_canvas,_,native_calls=prompted(processor,image_state,source_box,None,base)
                    native=base.to_original(native_canvas.cpu().numpy().astype(np.uint8),query.size)
                    if not np.array_equal(native,old_native): raise RuntimeError("Native original A bit-parity failed")
                    replay,_,noop_calls=prompted(processor,image_state,source_box,None,base)
                    if not torch.equal(replay,native_canvas): raise RuntimeError("Unconditional source-only clone/reset NoOp failed")
                    masks={"native":native}; prompts={"native":0}; noops=1; calls=native_calls+noop_calls; reference_diagnostic=None
                    if oracle:
                        # Query truth is never read by the legal phase. The whole
                        # legal40 cohort was already validated/frozen above.
                        truth=np.asarray(Image.open(ann/Path(row["query"]).with_suffix(".png")))==row["c"]+1
                        extras={"oracle_true_query_box":bbox(truth)}
                    else:
                        fb=bbox(foris)
                        extras={"query_foris":fb,"query_self":bbox(native),"query_random":random_box(fb,case)}
                    boxes={}
                    for arm,box in extras.items():
                        boxes[arm]=box; prompts[arm]=int(box is not None)
                        result,used=joint_query_prompt(processor,image_state,source_box,box,case["query_shape"],base,native_canvas)
                        if used:noops+=1
                        calls+=used
                        masks[arm]=base.to_original(result.cpu().numpy().astype(np.uint8),query.size)
                    if not oracle:
                        boxes["query_foris_only"]=fb;prompts["query_foris_only"]=int(fb is not None)
                        if fb is None:
                            masks["query_foris_only"]=native.copy()
                            reference_diagnostic=dict(active=False,fallback="original source-only native; query-only intervention unavailable")
                        else:
                            result,_,used=prompted(processor,image_state,None,query_box_on_canvas(fb,case["query_shape"],base),base)
                            calls+=used;masks["query_foris_only"]=base.to_original(result.cpu().numpy().astype(np.uint8),query.size)
                            sx,sy,sw,sh=base.rectangles()[0]
                            reference_pred=np.asarray(Image.fromarray(result[sy:sy+sh,sx:sx+sw].cpu().numpy().astype(np.uint8)).resize(support.size,Image.NEAREST))>0
                            reference_diagnostic=dict(active=True,I=int((reference_pred&ref).sum()),U=int((reference_pred|ref).sum()),
                                TP=int((reference_pred&ref).sum()),FP=int((reference_pred&~ref).sum()),FN=int((~reference_pred&ref).sum()),
                                selection_uses_this_label_score=False,source_positive_box_used=False)
                    if counter[0]!=1 or not equal_image_state(image_state,immutable):
                        raise RuntimeError("Image cache mutated or more than one image encoding occurred")
                finally: processor.model.backbone.forward_image=forward
                if not oracle: masks.update(foris=foris,cheap_OR=native|foris,cheap_AND=native&foris)
                path=dest/(stem(row)+".npz")
                if path.exists(): raise RuntimeError("Preserve uncommitted prediction; no overwrite or synthetic replacement")
                np.savez_compressed(path,**{name:np.packbits(value) for name,value in masks.items()},shape=np.array(case["query_shape"]))
                rec=dict(row=row,original_global_index=case["original_global_index"],source_bbox=old_box,
                    path=str(path.resolve()),sha256=sha(path),query_shape=case["query_shape"],extra_query_boxes=boxes,
                    extra_prompt_counts=prompts,actual_image_encoder_calls=counter[0],grounding_calls=calls,
                    NoOp_source_replays_exact=noops,native_original_A_exact=True,image_state_immutable=True,
                    NoOp_validation_grounding_overhead=noop_calls,
                    phase=phase,query_GT_used_as_input=oracle,precision_backend=precision,
                    query_only_source_reconstruction=reference_diagnostic,
                    total_positive_boxes={name: (1 if name=="query_foris_only" and prompts[name] else 1+count)
                                          for name,count in prompts.items()})
                stream.write(json.dumps(rec)+"\n");stream.flush();existing.append(rec)
                if not oracle and len(existing)==8:
                    write(a.out/"first8_real_parity.json",dict(state="FIRST8_REAL_NATIVE_NOOP_PARITY_PASSED",cases=8,
                        manifest_sha256=sha(a.manifest),source_GPU_sha256=GPU_SHA,balanced_folds=[r["row"]["fold"] for r in existing],
                        query_GT_pixels_opened=False,new_case_reencoding=False))
                print(json.dumps(dict(state="CASE_PREDICTIONS_FROZEN",phase=phase,episodes=len(existing),encoder_calls=counter[0])),flush=True)
                if sum(p.stat().st_size for p in a.out.rglob("*") if p.is_file())>manifest["maximum_output_bytes"]:
                    raise RuntimeError("96MiB bounded output gate")
        if len(existing)!=40: raise RuntimeError("Whole40 cohort required before scoring")
        files={r["path"]:r["sha256"] for r in existing};files[str(ledger.resolve())]=sha(ledger)
        write(a.out/(phase+"_freeze.json"),dict(state="ORACLE40_PREDICTIONS_FROZEN" if oracle else "ALL_LEGAL40_PREDICTIONS_FROZEN",
            episodes=40,manifest_sha256=sha(a.manifest),files=files,phase=phase,
            query_GT_used_as_input=oracle,actual_new_image_encodings=len(existing)-prefix_count,
            reused_prefix_episodes=prefix_count,cohort_image_encodings=sum(r["actual_image_encoder_calls"] for r in existing),
            cost_note="Oracle requires a separate paid image encoding per newly inferred case; legal prefix reused without encoding",
            elapsed_s=time.monotonic()-start,original_native_bit_exact=True))
    except BaseException as exc:
        write(a.out/(phase+"_error_%d.json"%time.time_ns()),dict(state="ERROR_INTERFACE_OR_RUNTIME",type=type(exc).__name__,
            detail=str(exc),traceback=traceback.format_exc(),completed_cases=len(existing),not_a_method_negative=True))
        raise


def score(a):
    import numpy as np
    from PIL import Image
    manifest=read(a.manifest);validate_freeze(a.out/"legal_freeze.json",a.manifest,"ALL_LEGAL40_PREDICTIONS_FROZEN",40)
    if (a.out/"report.json").exists():
        old=read(a.out/"report.json")
        if (old.get("state")=="COMPLETED" and old.get("manifest_sha256")==sha(a.manifest) and
            old.get("legal_freeze_sha256")==sha(a.out/"legal_freeze.json")):
            print(json.dumps(dict(state="REUSED_COMPLETED_REPORT",query_GT_reopened=False)));return
        raise RuntimeError("Preserve changed/partial report; no silent rescore")
    legal=rows(a.out/"legal/predictions.jsonl"); oracle={}
    legal_index=unique_index([r["row"] for r in legal],"new legal prediction ledger")
    expected_index=unique_index([r["row"] for r in manifest["cases"]],"prepared cases")
    if len(legal)!=40 or set(legal_index)!=set(expected_index):raise RuntimeError("Exact complete40 scope required")
    for identifier,row in legal_index.items():require_same_photos(row,expected_index[identifier],"prepared/legal join")
    if manifest["oracle_enabled"]:
        validate_freeze(a.out/"oracle_freeze.json",a.manifest,"ORACLE40_PREDICTIONS_FROZEN",40)
        oracle_rows=rows(a.out/"oracle/predictions.jsonl")
        unique_index([r["row"] for r in oracle_rows],"oracle prediction ledger")
        oracle={key(r["row"]):r for r in oracle_rows}
        if set(oracle)!=set(expected_index):raise RuntimeError("Oracle scope differs from legal40")
        for identifier,r in oracle.items():require_same_photos(r["row"],expected_index[identifier],"oracle/legal join")
    originals=unique_index(rows(manifest["original_sam_scored_records"]),"old SAM scored ledger")
    foris=unique_index(rows(manifest["original_foris_records"]),"old FoRIS scored ledger")
    measured=[]
    for rec in legal:
        row=rec["row"];require_same_photos(row,originals[key(row)],"SAM scoring join");require_same_photos(row,foris[key(row)],"FoRIS scoring join")
        truth=np.asarray(Image.open(Path(manifest["annotation_root"])/Path(row["query"]).with_suffix(".png")))==row["c"]+1
        with np.load(rec["path"],allow_pickle=False) as pack:
            masks={arm:np.unpackbits(pack[arm])[:truth.size].reshape(truth.shape).astype(bool) for arm in ARMS}
        if oracle:
            with np.load(oracle[key(row)]["path"],allow_pickle=False) as pack:
                masks["oracle_true_query_box"]=np.unpackbits(pack["oracle_true_query_box"])[:truth.size].reshape(truth.shape).astype(bool)
        native=masks["native"]; iu={};edits={}
        for arm,pred in masks.items():
            iu[arm]=[int((pred&truth).sum()),int((pred|truth).sum())]
            edits[arm]=dict(removed_FP=int((native&~pred&~truth).sum()),lost_TP=int((native&~pred&truth).sum()),
                            recovered_FN=int((~native&pred&truth).sum()),added_FP=int((~native&pred&~truth).sum()))
        if iu["native"]!=originals[key(row)]["original_iu"]["visual"]: raise RuntimeError("Scored native I/U does not reproduce old A")
        if iu["foris"]!=foris[key(row)]["original_iu"]["native"]: raise RuntimeError("FoRIS original binary mapping does not reproduce paired I/U")
        measured.append({**row,"original_iu":iu,"edits":edits,"extra_prompt_counts":rec["extra_prompt_counts"]})
    base=frozen_base(manifest["base_source"])
    comparisons={}
    all_arms=[*ARMS,*( ["oracle_true_query_box"] if oracle else [])]
    original_scores,bootstrap_scores,bootstrap_meta=paired_all_arms(measured,all_arms)
    for arm in all_arms:
        for control in (["native","query_self","query_random","cheap_OR","cheap_AND"] if arm=="query_foris" else ["native"]):
            if arm==control:continue
            delta=bootstrap_scores[arm]-bootstrap_scores[control]
            pair={"miou":original_scores[arm],"gain":original_scores[arm]-original_scores[control],
                  "ci95":np.quantile(delta,[.025,.975]).tolist(),**bootstrap_meta}
            pair["per_fold"]=[base.class_miou([r for r in measured if r["fold"]==f],arm)-
                              base.class_miou([r for r in measured if r["fold"]==f],control) for f in range(4)]
            comparisons[arm+"__minus__"+control]=pair
    report=dict(state="COMPLETED",episodes=40,manifest_sha256=sha(a.manifest),legal_freeze_sha256=sha(a.out/"legal_freeze.json"),
        scope="synthetic server-only score fixture, not task evidence" if manifest.get("fixture") else "old exposed DEV40, fourfold10; source-index-preserving C2 hypothesis probe",
        comparisons=comparisons,records=measured,card=CARD,query_GT_opened_after_entire_legal_cohort_freeze=True,
        random_extra_prompt_count_matches_foris_all_cases=all(r["extra_prompt_counts"]["query_random"]==r["extra_prompt_counts"]["query_foris"] for r in measured),
        self_extra_prompt_budget_mismatch_cases=sum(r["extra_prompt_counts"]["query_self"]!=r["extra_prompt_counts"]["query_foris"] for r in measured),
        query_only_source_reconstruction=[r.get("query_only_source_reconstruction") for r in legal],
        encoder_cost=dict(legal_cohort_image_encodings=sum(r["actual_image_encoder_calls"] for r in legal),oracle_separate_encoding_cases=len(oracle)),
        privileged_oracle_note="Identical tight-BBox operation with GT query mask substituted after legal freeze; no perfect-concept claim",
        publication_goal_complete=False,automatic_new_stages=False)
    write(a.out/"report.json",report)
    print(json.dumps({"state":"COMPLETED","episodes":40,"comparisons":comparisons}),flush=True)


def run_body(a):
    if os.environ.get("DEMO9_CUDA_GUARD")!="1": raise RuntimeError("Future outer resource guard required")
    manifest=read(a.manifest);started=time.monotonic()
    with (a.out/"execution.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for phase in ["--legal",*( ["--oracle"] if manifest["oracle_enabled"] else []),"--score"]:
            remaining=manifest["hard_cap_seconds"]-(time.monotonic()-started)
            if remaining<=0: raise RuntimeError("Finite batch budget exhausted")
            with (a.out/(phase[2:]+".log")).open("a") as stream:
                completed=subprocess.run([a.python,str(Path(__file__).resolve()),phase,"--manifest",str(a.manifest),"--out",str(a.out)],
                    env=os.environ.copy(),stdout=stream,stderr=stream,timeout=min(60,remaining) if phase=="--score" else remaining)
            if completed.returncode: raise RuntimeError("Prepared phase failed: "+phase)


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    for name in ("prepare","cpu-preflight","legal","oracle","score","run-body"):mode.add_argument("--"+name,action="store_true")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--manifest",type=Path)
    p.add_argument("--dev-manifest",type=Path);p.add_argument("--sam-native",type=Path)
    p.add_argument("--foris-packets",type=Path);p.add_argument("--foris-records",type=Path)
    p.add_argument("--sam3",type=Path);p.add_argument("--checkpoint",type=Path);p.add_argument("--checkpoint-status",type=Path)
    p.add_argument("--previous-plan",type=Path);p.add_argument("--python",default="/root/miniconda3/bin/python")
    p.add_argument("--budget-seconds",type=int,default=300);p.add_argument("--with-oracle",dest="with_oracle",action="store_true",default=True)
    a=p.parse_args()
    if a.prepare and any(value is None for value in (a.dev_manifest,a.sam_native,a.foris_packets,a.foris_records,a.sam3,a.checkpoint,a.checkpoint_status)):
        p.error("--prepare requires original DEV manifest, real SAM native directory, real FoRIS packets/records and existing SAM3 source/checkpoint/status")
    if not a.prepare and a.manifest is None:p.error("--manifest required")
    if a.prepare:prepare(a)
    elif a.cpu_preflight:cpu_preflight(a)
    elif a.legal:infer(a)
    elif a.oracle:infer(a,oracle=True)
    elif a.score:score(a)
    else:run_body(a)


if __name__=="__main__":main()
