#!/usr/bin/env python3
"""SERVER-only matched author SAM3 frontends, four folds x first50 (seed0).

This200-case subset is not the author's4000-case66.6 reproduction. One bundled
author model/processor serves literal standard, legacy visual frontend using
the SAME author instance box, and optional literal max_neg0 positive mining.
No old Meta precision patch, semantic-component box or category text is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import sam3_author_reproduction as base

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=("author_standard","legacy_same_instance_box")
MINING="author_mining_maxneg0"
ARMS=(*REQUIRED,MINING)


def load_legacy(path):
    spec=importlib.util.spec_from_file_location("frozen_legacy_frontend_only",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prep_proof(preparation):
    manifest=base.read(preparation/"manifest.json");proof=base.read(preparation/"cpu_preflight.json")
    metadata=base.read(preparation/"episodes4000.json")
    base.exact_files(manifest["sources"]);base.exact_files(manifest["inputs"])
    if (proof.get("state")!="AUTHOR_SERVER_CPU_PREFLIGHT_PASSED_GPU_UNTESTED" or
        proof.get("manifest_sha256")!=base.sha(preparation/"manifest.json") or
        proof.get("episode_metadata_sha256")!=base.sha(preparation/"episodes4000.json") or
        metadata.get("state")!="AUTHOR_4000_EPISODE_METADATA_FROZEN" or len(metadata["episodes"])!=4000):
        raise RuntimeError("Verified v3 full4000 source/asset/metadata preparation required")
    return manifest,proof,metadata


def prepare(args):
    if args.out.exists() and any(args.out.iterdir()):raise RuntimeError("Fresh200 output required; preserve old evidence")
    old,proof,metadata=prep_proof(args.preparation)
    rows=[row for row in metadata["episodes"] if row["episode"]<50]
    if len(rows)!=200 or any(sum(row["fold"]==fold for row in rows)!=50 for fold in range(4)):
        raise RuntimeError("Exactly first50 per original fold required")
    for index,row in enumerate(metadata["episodes"]):
        if (row["fold"],row["episode"])!=(index//1000,index%1000):raise RuntimeError("Original4000 order differs")
    inputs=dict(old["inputs"])
    for name in ("manifest.json","cpu_preflight.json","episodes4000.json","metadata_process.json"):
        path=args.preparation/name;inputs[str(path.resolve())]=base.sha(path)
    sources=dict(old["sources"])
    for path in (Path(__file__),Path(base.__file__),args.legacy,ROOT/"scripts/experiment_resource_guard.py",ROOT/"tics/paired_frozen_stats.py"):
        sources[str(path.resolve())]=base.sha(path)
    manifest=dict(state="MATCHED_AUTHOR200_PREPARED_PENDING_SERVER_CPU",schema="matched_author200_v1",
        original_preparation=str(args.preparation.resolve()),author_source=old["source"],author_asset_manifest=old,
        sources=sources,inputs=inputs,legacy_frontend=str(args.legacy.resolve()),episodes=rows,seed=0,
        class_id_to_name=proof["class_id_to_name"],required_arms=list(REQUIRED),optional_arm=MINING,
        sampling="Original official local seed0 items; first50 in each of4folds, not fold0first200",
        source_box="Original instance annotation bbox; all intervening global random.choice draws consumed and checked",
        author_numerical_path="Bundled timm.Mlp, original builder defaults and processor; no precision patches",
        legacy_calls_only=["stitch","Runner","to_original"],legacy_component_box_not_called=True,
        same_original_instance_box_for_all_arms=True,legacy_prior_semantic_box_scores_not_reused=True,
        original_author_evaluator_run_episode_unmodified=True,QueryGT_protocol="Original loader opens GT before predictions; scoring only",
        total_cap_seconds=960,GPU_cap_seconds=900,CPU_report_cap_seconds=30,startup_reserve_seconds=30,
        allocator_GPU_memory_fraction=.8,reserve_bytes=5*1024**3,max_packed_bytes=128*1024**2,
        first8_gate="Input API/parity and measured runtime only; output differences recorded, no quality/CI stop; same run continues",
        optional_mining_cost_skip_allowed=True,full4000_paper_reproduction=False,GPU_started=False,automatic_boot=False,
        card=dict(assumption="On identical images/instance boxes and author backend the pure old visual frontend is equivalent to literal standard; mining is a separately specified algorithm.",
            prediction="Standard/legacy0 differing pixels expected; mining task gain is unmeasured. Required200 and optional mining200 must fit measured900s; no invented runtime prediction.",
            match="Attribute frontend parity only on this200 subset; report mining separately with paired four-fold class-sum intervals.",
            mismatch="Source/API/RNG/parity failures stop as implementation faults; runtime insufficiency holds finite scope without resampling or extending budget. No quality halt."))
    base.write(args.out/"manifest.json",manifest)
    print(json.dumps(dict(state=manifest["state"],cases=200,fold_counts=[50]*4,GPU_started=False)),flush=True)


def capture_author(evaluator,references,targets,cat_id):
    """Observer around literal public run_episode; no changed model arithmetic."""
    import numpy as np
    saved=[];original=evaluator.metric.update
    def capture(pred,truth,category):
        original(pred,truth,category)
        saved.append(np.asarray(pred).astype(bool).copy())
    evaluator.metric.update=capture
    try:evaluator.run_episode(references,targets,cat_id,vis_prefix=None,cls_name="visual")
    finally:evaluator.metric.update=original
    if len(saved)!=1:raise RuntimeError("Author1-shot must update exactly one original query mask")
    return saved[0]


def paired_fourfold(records,arm,baseline,draws=2000):
    """One shared exact connected-photo resampling; author mean-of-fold macros."""
    import numpy as np
    sys.path.insert(0,str(ROOT))
    from tics.paired_frozen_stats import _sampling_plan,_class_totals
    if not records:raise ValueError("No matched observations")
    plan=_sampling_plan(records,draws);class_fold={row["c"]:row["fold"] for row in records}
    totals=[_class_totals(np.array([row["IU"][name] for row in records],np.float64),plan) for name in (arm,baseline)]
    def macro(values,present):
        ratios=values[...,0]/np.maximum(values[...,1],1);fold_scores=[];fold_present=[]
        for fold in range(4):
            eligible=np.array([class_fold[int(cat)]==fold for cat in plan["class_ids"]])
            count=present[...,eligible].sum(axis=-1)
            fold_scores.append((ratios[...,eligible]*present[...,eligible]).sum(axis=-1)/np.maximum(count,1))
            fold_present.append(count>0)
        scores=np.stack(fold_scores,axis=-1);valid=np.stack(fold_present,axis=-1)
        return 100*(scores*valid).sum(axis=-1)/np.maximum(valid.sum(axis=-1),1)
    point=[float(macro(values.sum(axis=0),np.ones(len(plan["class_ids"]),bool))) for values in totals]
    sampled=[np.einsum("dg,gci->dci",plan["weights"],values,optimize=True) for values in totals]
    boot=macro(sampled[0],plan["sampled_present"])-macro(sampled[1],plan["sampled_present"])
    return dict(miou=point[0],baseline_miou=point[1],gain=point[0]-point[1],
        ci95=[float(x) for x in np.percentile(boot,[2.5,97.5])] if len(plan["groups"])>1 else None,
        bootstrap_unit="connected Source/Query photograph groups",draws=draws,seed=0,
        groups=len(plan["groups"]),largest_group=max(map(len,plan["groups"])),
        metric="mean of4fold class-sum I/U macros; resampled absent classes/folds excluded")


def cpu_preflight(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    manifest=base.read(args.manifest);base.exact_files(manifest["sources"]);base.exact_files(manifest["inputs"])
    old=manifest["author_asset_manifest"]
    sys.path[0:0]=[old["own_dependency_prefix"],old["approved_dependency_prefix"],"/root/demo4_cache/env"]
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    standard=base.author_module(Path(old["source"]));mining=base.author_module(Path(old["source"]),"mining_positive")
    legacy=load_legacy(Path(manifest["legacy_frontend"]))
    class Prompt:
        def append_boxes(self,*values):pass
    class Backbone:
        def forward_text(self,*values,**kwargs):return {"language_features":torch.zeros((1,1,256))}
    class Model:
        def __init__(self):self.backbone=Backbone()
        def _get_dummy_prompt(self):return Prompt()
        def forward_grounding(self,**kwargs):
            state=kwargs["fixture_state"];binary=state["fixture_mask"]
            return dict(pred_logits=torch.full((1,1,1),10.),pred_masks=torch.where(binary,10.,-10.)[None,None],presence_logit_dec=torch.full((1,1),10.))
    class Processor:
        device="cpu"
        def __init__(self):self.model=Model();self.threshold=.5
        def set_confidence_threshold(self,value):self.threshold=value
        def set_image(self,image):
            w,h=image.size;mask=torch.zeros((h,w),dtype=torch.bool)
            if (w,h)==(1008,1008):mask[70:300,200:800]=True
            else:mask[h//4:3*h//4,w//4:3*w//4]=True
            return dict(backbone_out={},fixture_mask=mask)
        def reset_all_prompts(self,state):
            state.pop("geometric_prompt",None);state["backbone_out"].pop("language_features",None)
        def add_geometric_prompt(self,*,box,label,state):return self._forward_grounding(state)
        def _forward_grounding(self,state):
            self.model.forward_grounding(fixture_state=state)
            h,w=state["fixture_mask"].shape
            state.update(masks=state["fixture_mask"][None,None],boxes=torch.tensor([[w/4,h/4,3*w/4,3*h/4]],dtype=torch.float32),scores=torch.tensor([.9]))
            return state
    checks=[]
    for index in range(10):
        processor=Processor();a=standard.PairwiseSam3Evaluator(processor,swap_order=True,use_text=False)
        b=mining.PairwiseSam3Evaluator(processor,max_neg=0)
        ref=dict(image=Image.new("RGB",(640+index,480)),box=[10.,20.,30.,40.])
        query=Image.new("RGB",(320+index,240));canvas,norm=legacy.stitch(ref["image"],query,ref["box"])
        truth=legacy.to_original(processor.set_image(canvas)["fixture_mask"].numpy().astype(np.uint8),query.size)
        target=dict(image=query,mask=truth)
        author_canvas,placements=a.create_input(ref,target)
        if not np.array_equal(np.asarray(canvas),np.asarray(author_canvas)) or norm!=a.get_norm_box(placements):raise RuntimeError("Synthetic frontend geometry mismatch")
        pa=capture_author(a,[ref],[target],1)
        union,*_=legacy.Runner(processor)(canvas,norm,text=None)
        pb=legacy.to_original(union.numpy().astype(np.uint8),query.size)
        pm=capture_author(b,[ref],[target],1)
        if not np.array_equal(pa,pb) or not np.array_equal(pa,truth) or not np.array_equal(pm,truth):raise RuntimeError("Literal evaluator10-case synthetic mask/metric contract failed")
        checks.append(dict(case=index,standard_legacy_differing_pixels=0,mining_synthetic_IU=[int(truth.sum())]*2))
    synthetic=[dict(c=row["cat_id"],fold=row["fold"],support=row["support"],query=row["query"],
        IU={REQUIRED[0]:[100,200],REQUIRED[1]:[100,200],MINING:[110,200]}) for row in manifest["episodes"]]
    began=time.monotonic();stats=paired_fourfold(synthetic,MINING,REQUIRED[0]);zero=paired_fourfold(synthetic,REQUIRED[1],REQUIRED[0]);elapsed=time.monotonic()-began
    if abs(stats["gain"]-5)>1e-10 or zero["gain"]!=0 or zero["ci95"]!=[0.,0.]:raise RuntimeError("Exact fourfold paired stats synthetic contract failed")
    if elapsed>manifest["CPU_report_cap_seconds"]:raise RuntimeError("Actual200-row synthetic statistics exceeds registered report cap")
    if torch.cuda.is_initialized():raise RuntimeError("No-card CPU fixture initialized CUDA")
    proof=dict(state="MATCHED_AUTHOR200_SERVER_CPU_PASSED_GPU_UNTESTED",manifest_sha256=base.sha(args.manifest),
        checks=checks,count=10,synthetic_processor_not_pretrained_SAM=True,actual_author_evaluator_functions_executed=True,
        real_annotations_or_GT_pixels_loaded=False,full200_stats_seconds=elapsed,full200_stats_draws=2000,
        CUDA_initialized=False,source_hashes=manifest["sources"],legacy_precision_patch_called=False)
    base.write(args.out/"cpu_preflight.json",proof);make_guard(args,manifest)
    print(json.dumps(dict(state=proof["state"],checks=10,CPU_statistics_seconds=elapsed)),flush=True)


def make_guard(args,manifest):
    old=manifest["author_asset_manifest"]
    env=dict(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DEMO9_AUTHOR200_ROOT_LAUNCH="1",
        PYTHONPATH=old["source"]+":"+old["own_dependency_prefix"]+":"+old["approved_dependency_prefix"]+":/root/demo4_cache/env",
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    prereq=[dict(path=str(args.manifest.resolve()),sha256=base.sha(args.manifest)),
        dict(path=str((args.out/"cpu_preflight.json").resolve()),sha256=base.sha(args.out/"cpu_preflight.json"),json_equals=dict(state="MATCHED_AUTHOR200_SERVER_CPU_PASSED_GPU_UNTESTED"))]
    prereq.extend(dict(path=path,sha256=digest) for path,digest in manifest["sources"].items() if Path(path).suffix!=".py")
    execution=dict(path=str((args.out/"execution.json").resolve()),json_equals=dict(finite_stage_finished=True))
    report=dict(path=str((args.out/"report.json").resolve()),json_equals=dict(finite_report_finished=True))
    stages=[]
    for action,kind,cap,requires,product in (("run","gpu",900,[],execution),("report","cpu",30,[execution],report)):
        stages.append(dict(name="author200_"+action,kind=kind,**(dict(role="finalize") if kind=="cpu" else {}),
            cwd=str(args.out.resolve()),timeout_seconds=cap,env=env,code_files=[path for path in manifest["sources"] if Path(path).suffix==".py"],requires=[*prereq,*requires],
            argv=[args.python,str(Path(__file__).resolve()),"--"+action,"--manifest",str(args.manifest.resolve()),"--out",str(args.out.resolve())],produces=[product],success_checks=[product]))
    base.write(args.out/"guard_plan.json",dict(platform="autodl",cuda_python=args.python,stages=stages,
        total_timeout_budget_seconds=960,sum_stage_timeouts_seconds=930,startup_reserve_seconds=30,
        automatic_boot=False,full4000_queue_resumed=False))


def iu(pred,truth):
    import numpy as np
    return [int(np.logical_and(pred,truth).sum()),int(np.logical_or(pred,truth).sum())]


def run(args):
    if os.environ.get("DEMO9_AUTHOR200_ROOT_LAUNCH")!="1":raise RuntimeError("Root-owned finite guard required")
    manifest=base.read(args.manifest);base.exact_files(manifest["sources"]);base.exact_files(manifest["inputs"])
    proof=base.read(args.out/"cpu_preflight.json")
    if proof.get("state")!="MATCHED_AUTHOR200_SERVER_CPU_PASSED_GPU_UNTESTED" or proof.get("manifest_sha256")!=base.sha(args.manifest):raise RuntimeError("Actual SERVER200 CPU preflight required")
    if (args.out/"episodes.jsonl").exists():raise RuntimeError("No restart/reseed: finite200 continues in one invocation only")
    import numpy as np
    import torch
    old=manifest["author_asset_manifest"];metadata=base.read(Path(manifest["original_preparation"])/"episodes4000.json")
    standard=base.author_module(Path(old["source"]));mining=base.author_module(Path(old["source"]),"mining_positive")
    legacy=load_legacy(Path(manifest["legacy_frontend"]))
    torch.cuda.set_per_process_memory_fraction(.8);standard.setup_seed(0)
    began=time.monotonic();load_start=time.monotonic()
    model=standard.build_sam3_image_model(bpe_path=old["bpe"],checkpoint_path=old["checkpoint"],load_from_HF=False)
    processor=standard.Sam3Processor(model);load_seconds=time.monotonic()-load_start
    legacy_run=legacy.Runner(processor);records=[];constructors=[];mining_enabled=True;skipped_boxes=0;packed_bytes=0
    (args.out/"masks").mkdir(exist_ok=True)
    state="MATCHED_AUTHOR200_COMPLETED";cost=None
    try:
        for fold in range(4):
            ds_start=time.monotonic()
            dataset=standard.COCOFSSDataset(lists_root=str(Path(old["source"])/"data/lists/coco/fss_list"),
                data_root=str(Path(old["source"])/"data/MSCOCO2014"),fold=fold,k_shot=1,mode="val",seed=0)
            constructors.append(time.monotonic()-ds_start)
            expected=metadata["episodes"][fold*1000:(fold+1)*1000]
            if dataset.test_items!=[(row["cat_id"],row["query_id"],[row["support_id"]]) for row in expected]:raise RuntimeError("Original author local sampling differs")
            a=standard.PairwiseSam3Evaluator(processor,swap_order=True,use_text=False)
            m=mining.PairwiseSam3Evaluator(processor,max_neg=0)
            for index,row in enumerate(expected):
                if base.rng_digest(random.getstate())!=row["global_box_rng_before"]:raise RuntimeError("Original global support bbox RNG mismatch")
                if index>=50:
                    anns=dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=row["support_id"],catIds=row["cat_id"]))
                    chosen=random.choice([ann for ann in anns if ann["area"]>1000 and not ann.get("iscrowd",0)])
                    if chosen["id"]!=row["support_annotation_id"] or chosen["bbox"]!=row["support_box"]:raise RuntimeError("Skipped original bbox draw differs")
                    skipped_boxes+=1
                else:
                    started=time.monotonic();refs,targets,cat,_=dataset[index]
                    if (cat!=row["cat_id"] or refs[0]["name"]!=row["support"] or targets[0]["name"]!=row["query"] or refs[0]["box"]!=row["support_box"]):raise RuntimeError("Actual selected author images/instance bbox differ")
                    arm_seconds={};predictions={};tick=time.monotonic()
                    predictions[REQUIRED[0]]=capture_author(a,refs,targets,cat);arm_seconds[REQUIRED[0]]=time.monotonic()-tick
                    canvas,box=legacy.stitch(refs[0]["image"],targets[0]["image"],row["support_box"])
                    author_canvas,placements=a.create_input(refs[0],targets[0])
                    if box!=a.get_norm_box(placements) or not np.array_equal(np.asarray(canvas),np.asarray(author_canvas)):raise RuntimeError("Same-instance frontend canvas/box parity failed")
                    tick=time.monotonic();union,*_=legacy_run(canvas,box,text=None)
                    predictions[REQUIRED[1]]=legacy.to_original(union.cpu().numpy().astype(np.uint8),targets[0]["image"].size);arm_seconds[REQUIRED[1]]=time.monotonic()-tick
                    differences=int(np.logical_xor(predictions[REQUIRED[0]],predictions[REQUIRED[1]]).sum())
                    if mining_enabled:
                        tick=time.monotonic();predictions[MINING]=capture_author(m,refs,targets,cat);arm_seconds[MINING]=time.monotonic()-tick
                    if base.rng_digest(random.getstate())!=row["global_box_rng_after"]:raise RuntimeError("Selected bbox draw or frontend unexpectedly consumed Python RNG")
                    truth=targets[0]["mask"]>0
                    if any(pred.shape!=truth.shape for pred in predictions.values()):raise RuntimeError("Original query dimensions differ")
                    packed={arm:np.packbits(pred.reshape(-1)) for arm,pred in predictions.items()}
                    path=args.out/"masks"/("f%d_e%04d_c%d.npz"%(fold,index,cat))
                    np.savez_compressed(path,shape=np.array(truth.shape,np.int64),**packed)
                    packed_bytes+=path.stat().st_size
                    if packed_bytes>manifest["max_packed_bytes"]:raise RuntimeError("Registered compact-mask storage bound exceeded")
                    record=dict(row,c=cat,IU={arm:iu(pred,truth) for arm,pred in predictions.items()},arm_seconds=arm_seconds,
                        standard_legacy_differing_pixels=differences,prediction_path=str(path.resolve()),prediction_sha256=base.sha(path),
                        original_query_shape=list(truth.shape),wall_seconds=time.monotonic()-started,
                        original_author_GT_loaded_before_prediction=True,legacy_semantic_component_box_used=False)
                    records.append(record)
                    with (args.out/"episodes.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n");stream.flush();os.fsync(stream.fileno())
                    print(json.dumps(dict(state="AUTHOR200_CASE_COMPLETED",cases=len(records),fold=fold,episode=index,IU=record["IU"],wall_seconds=record["wall_seconds"])),flush=True)
                    if len(records)==8:
                        remaining=900-(time.monotonic()-began);means={arm:sum(r["arm_seconds"][arm] for r in records)/8 for arm in ARMS}
                        overhead=max(0,sum(r["wall_seconds"] for r in records)/8-sum(means.values()))
                        required=(sum(means[arm] for arm in REQUIRED)+overhead)*192+3*constructors[0]
                        all_arms=required+means[MINING]*192
                        cost=dict(state="AUTHOR200_FIRST8_COST_GATE",cases=8,model_load_seconds=load_seconds,arm_mean_seconds=means,
                            estimated_required_remaining_seconds=required,estimated_all_arms_remaining_seconds=all_arms,
                            remaining_registered_seconds=remaining,required_can_fit=required<=remaining,optional_mining_can_fit=all_arms<=remaining,
                            no_quality_or_CI_halt=True,no_first8_rerun=True,not_a_tail_runtime_guarantee=True)
                        base.write(args.out/"first8_cost.json",cost)
                        if required>remaining:state="RESOURCE_HOLD_AFTER_MATCHED8";break
                        mining_enabled=all_arms<=remaining
                    if time.monotonic()-began>=900:state="RESOURCE_HOLD_WITH_MATCHED_PREFIX";break
                    if shutil.disk_usage(args.out).free<manifest["reserve_bytes"]:raise RuntimeError("5GiB disk reserve failed")
                if base.rng_digest(random.getstate())!=row["global_box_rng_after"]:raise RuntimeError("Original support bbox draw or algorithm consumed unexpected Python RNG")
            del dataset
            if state!="MATCHED_AUTHOR200_COMPLETED":break
        if state=="MATCHED_AUTHOR200_COMPLETED" and (len(records)!=200 or any(sum(r["fold"]==fold for r in records)!=50 for fold in range(4))):raise RuntimeError("All-fold200 coverage failed")
    except BaseException as error:
        base.write(args.out/("error_"+str(time.time_ns())+".json"),dict(state="ERROR_AUTHOR200_INTERFACE",error=repr(error),traceback=traceback.format_exc(),completed_cases=len(records),not_a_scientific_negative=True));raise
    base.write(args.out/"execution.json",dict(state=state,finite_stage_finished=True,cases=len(records),manifest_sha256=base.sha(args.manifest),
        records_sha256=base.sha(args.out/"episodes.jsonl"),elapsed_seconds=time.monotonic()-began,model_build_calls=1,
        author_processor_shared=True,legacy_precision_patch_calls=0,skipped_original_bbox_draws=skipped_boxes,
        optional_mining_full200=mining_enabled and len(records)==200,cost_gate=cost,full4000_paper_reproduction=False))


def report(args):
    manifest=base.read(args.manifest);base.exact_files(manifest["sources"])
    execution=base.read(args.out/"execution.json")
    if execution["manifest_sha256"]!=base.sha(args.manifest) or execution["records_sha256"]!=base.sha(args.out/"episodes.jsonl"):raise RuntimeError("Frozen report provenance differs")
    records=[json.loads(line) for line in (args.out/"episodes.jsonl").read_text().splitlines()]
    comparisons={};folds={}
    for arm in ARMS:
        eligible=[row for row in records if arm in row["IU"]]
        if not eligible:continue
        comparisons[arm]=paired_fourfold(eligible,arm,REQUIRED[0])
        comparisons[arm].update(cases=len(eligible),fourfold200=len(eligible)==200)
        folds[arm]={}
        for fold in range(4):
            chosen=[row for row in eligible if row["fold"]==fold]
            if chosen:folds[arm][str(fold)]=paired_fourfold(chosen,arm,REQUIRED[0])
    base.write(args.out/"report.json",dict(state="MATCHED_AUTHOR200_REPORT",finite_report_finished=True,
        execution=execution,comparisons=comparisons,folds=folds,class_id_to_name=manifest["class_id_to_name"],
        frontend_mask_comparison=dict(exact_cases=sum(row["standard_legacy_differing_pixels"]==0 for row in records),
            total_cases=len(records),differing_pixels=sum(row["standard_legacy_differing_pixels"] for row in records),
            differing_case_ids=[[row["fold"],row["episode"]] for row in records if row["standard_legacy_differing_pixels"]]),
        oldfrontend_prediction_union_with_sameinstancebox=True,oldsemanticmaskprotocol_and_precision_patch_not_reproduced=True,
        mining_positive_is_algorithm_difference=True,official200_subset_not_paper66_6_reproduction=True,
        only_same_RGB_instancebox_cohort_is_paired=True,no_new_method_claim=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare","cpu-preflight","run","report"):modes.add_argument("--"+name,action="store_true")
    parser.add_argument("--preparation",type=Path,default=ROOT/"results/sam3_author_v1/ready_preparation_v3")
    parser.add_argument("--legacy",type=Path,default=ROOT/"scripts/sam3_stitch_gpu_v2.py")
    parser.add_argument("--manifest",type=Path);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--python",default="/root/miniconda3/bin/python")
    args=parser.parse_args();base.server_only()
    if args.prepare:prepare(args)
    elif args.manifest is None:parser.error("--manifest required")
    elif args.cpu_preflight:cpu_preflight(args)
    elif args.run:run(args)
    else:report(args)


if __name__=="__main__":main()
