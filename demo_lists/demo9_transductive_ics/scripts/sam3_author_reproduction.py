#!/usr/bin/env python3
"""Literal pinned FSS-SAM3 COCO1-shot baseline, not our stitch adaptation.

SERVER-only. No boot/download/training/precision repair. The author reads Query
GT through its official loader before predicting, then scores online; this is
the authorised baseline protocol, not a whole-cohort QueryGT embargo claim.
One author main call runs all4x1000; a first8 cost gate continues IN PLACE.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
COMMIT="b82ba838678ae8c2f27f618d9500a50c171e223e"
EVALUATE_BLOB="d118c477682107fa4ddc5ac9d7d3d146aa3ec849"
DATASET_BLOB="68010ece3d9312a881a46533db15e70fc8647ff2"
MINING_BLOB="1afb97bfd9fadf8fb47880787e6e0b49dd1e1431"
CHECKPOINT_SHA="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CHECKPOINT_BYTES=3450062241
AUTHOR_MODES={
    "mining_positive":dict(entry="model/evaluate_neg.py",argv=["--seed","0","--max_neg","0"],visualizations=40,
        table_anchor="README negative-prompt table Nneg0:66.6; two-stage positive mining"),
    "standard_visual":dict(entry="model/evaluate.py",argv=["--dataset","coco","--shot","1","--seed","0","--disable_text"],visualizations=20,
        table_anchor="Standard public visual-only entry; not silently equated with mining table"),
}


class AuthorHardFailure(BaseException):pass
class AuthorCostHold(BaseException):pass


def server_only():
    if not sys.platform.startswith("linux") or not str(ROOT).startswith("/root/"):
        raise RuntimeError("Only SERVER preparation/execution; local source/AST access only")


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def git_blob(path):
    value=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(value)).encode()+b"\0"+value).hexdigest()


def rng_digest(state):
    return hashlib.sha256(repr(state).encode()).hexdigest()


def source_files(source):
    return sorted([*source.rglob("*.py"),source/"pyproject.toml",source/"LICENSE",source/"README.md"])


def verify_revision(source):
    critical={"model/evaluate.py":EVALUATE_BLOB,"model/evaluate_neg.py":MINING_BLOB,"data/dataset_tool.py":DATASET_BLOB}
    if any(git_blob(source/name)!=digest for name,digest in critical.items()):
        raise RuntimeError("Literal pinned author evaluator/mining/dataset blobs required")
    receipt=source/".author_source_revision.json"
    if receipt.is_file():
        proof=read(receipt);declared=proof.get("source_files",{});blobs=proof.get("git_blobs",{})
        actual={str(path.relative_to(source)) for path in source_files(source)}
        if proof.get("commit")!=COMMIT or set(declared)!=actual or set(blobs)!=actual:
            raise RuntimeError("Pinned transported author revision/source coverage differs")
        for name,digest in declared.items():
            if sha(source/name)!=digest or git_blob(source/name)!=blobs[name]:
                raise RuntimeError("Transported author source differs from pinned full SHA/blob proof: "+name)
        return dict(method="root_verified_full_transport_SHA_and_Git_blob_receipt",commit=COMMIT,
            path=str(receipt.resolve()),sha256=sha(receipt),git_checkout_present=(source/".git").exists())
    commit=subprocess.run(["git","-C",str(source),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    if commit!=COMMIT:raise RuntimeError("Author checkout commit changed")
    return dict(method="actual_pinned_git_checkout_and_critical_full_blobs",commit=commit,git_checkout_present=True)


def exact_files(values):
    for path,digest in values.items():
        if not Path(path).is_file() or sha(path)!=digest:raise RuntimeError("Frozen author source/input changed: "+path)


def asset_link(target,source):
    target=Path(target);source=Path(source).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve()!=source:raise RuntimeError("Existing asset view points elsewhere: "+str(target))
    else:target.symlink_to(source,target_is_directory=source.is_dir())


def author_module(source,mode="standard_visual"):
    # evaluate.py appends root itself; explicitly put the AUTHOR bundle first.
    sys.path.insert(0,str(source.resolve()))
    for name in list(sys.modules):
        if name=="sam3" or name.startswith("sam3."):
            filename=getattr(sys.modules[name],"__file__",None)
            if filename and not Path(filename).resolve().is_relative_to(source.resolve()):
                raise RuntimeError("Another SAM3 package was already imported; do not shadow/patch it")
    path=source/AUTHOR_MODES[mode]["entry"]
    spec=importlib.util.spec_from_file_location("literal_FSS_SAM3_author_"+mode,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    import sam3
    if not Path(sam3.__file__).resolve().is_relative_to(source.resolve()):raise RuntimeError("Wrong SAM3 implementation imported")
    dataset_file=Path(sys.modules[module.COCOFSSDataset.__module__].__file__).resolve()
    if dataset_file!=(source/"data/dataset_tool.py").resolve():raise RuntimeError("Wrong author COCO dataset implementation imported")
    return module


def prepare(args):
    source=args.source.resolve()
    revision=verify_revision(source)
    if args.out.exists() and any(args.out.iterdir()):raise ValueError("Preserve previous evidence; use a fresh output directory")
    paths=source_files(source);sources={str(path.resolve()):sha(path) for path in paths}
    if revision.get("path"):sources[revision["path"]]=revision["sha256"]
    sources[str(Path(__file__).resolve())]=sha(__file__)
    sources[str((ROOT/"scripts/experiment_resource_guard.py").resolve())]=sha(ROOT/"scripts/experiment_resource_guard.py")
    status=read(args.checkpoint_status)
    if (status.get("state")!="WEIGHT_VERIFIED" or status.get("actual_sha256")!=CHECKPOINT_SHA or
            status.get("actual_bytes")!=CHECKPOINT_BYTES or args.checkpoint.stat().st_size!=CHECKPOINT_BYTES):
        raise RuntimeError("Existing approved official checkpoint/status required; no download fallback")
    if not args.bpe.is_file() or not args.image_root.is_dir():raise FileNotFoundError("Existing BPE and native val2014 JPEG directory required")
    if args.total_seconds!=4500:raise ValueError("This finite baseline freezes the75-minute cap; no silent budget increase")
    inputs={str(args.checkpoint_status.resolve()):sha(args.checkpoint_status),str(args.bpe.resolve()):sha(args.bpe)}
    missing=[]
    if args.instances is None or not args.instances.is_file():missing.append("official instances_val2014.json")
    else:inputs[str(args.instances.resolve())]=sha(args.instances)
    asset_link(source/"data/MSCOCO2014/val2014",args.image_root)
    if not missing:asset_link(source/"data/MSCOCO2014/annotations/instances_val2014.json",args.instances)
    # evaluate_neg.py hardcodes this author's asset path. A read-only asset
    # view preserves the literal entry; never edit DATA_ROOT in its source.
    asset_link(Path("/home/tsai091/fsssam3/data/MSCOCO2014"),source/"data/MSCOCO2014")
    if shutil.disk_usage(args.out.parent).free<5*1024**3:raise RuntimeError("5GiB reserve gate")
    manifest=dict(state="HELD_MISSING_AUTHOR_ANNOTATIONS" if missing else "AUTHOR_METADATA_PREPARED_PENDING_SERVER_CPU_PREFLIGHT",
        schema="literal_FSS_SAM3_author_COCO1shot_v1",author_commit=COMMIT,source_revision=revision,source=str(source),sources=sources,inputs=inputs,
        image_root=str(args.image_root.resolve()),instances=str(args.instances.resolve()) if args.instances else None,
        checkpoint=str(args.checkpoint.resolve()),checkpoint_status=str(args.checkpoint_status.resolve()),bpe=str(args.bpe.resolve()),
        own_dependency_prefix=str(args.own_prefix.resolve()),approved_dependency_prefix=str(args.shared_prefix.resolve()),
        author_modes=AUTHOR_MODES,mode_priority=["mining_positive","standard_visual"],author_cwd=str(source/"model"),folds=4,episodes_per_fold=1000,total_episodes_per_mode=4000,
        split_ratio=.6,canvas=1008,orientation="vertical",COCO_default_swap=True,disable_text=True,
        instance_support_rule="noncrowd annotation area>1000; literal global random.choice",
        target_rule="literal noncrowd annToMask union; target pool without support area restriction",
        episode_rng="local random.Random(seed+fold); unsorted public COCO getImgIds order preserved",
        box_rng="global random seeded once by literal main; no per-fold reset",
        metadata_preparation="Separate no-Torch child compiles original dataset class/method AST with nominal Dataset=object; actual GPU imports unmodified torch Dataset and checks all4000 draws",
        lists_root_unused=True,no_list_or_image_or_weight_download=True,
        numerical_path="author bundled SAM3 + its timm.Mlp; builder defaults; no old precision repair/autocast/dtype/TF32 override",
        asset_only_builder_kwargs=["bpe_path","checkpoint_path","load_from_HF=False"],
        total_cap_seconds=4500,infer_cap_seconds=4455,CPU_report_cap_seconds=15,startup_reserve_seconds=30,
        first_cost_gate_episode=8,cost_gate_not_a_quality_or_CI_stop=True,
        allocator_GPU_memory_fraction=.8,allocator_limit_not_a_numerical_model_change=True,
        QueryGT_loading="Author loader opens target masks before inference; author metrics score online",
        matched_adaptation_comparison="Only identical RGB images, actual instance boxes, RNG draws and ground truth may be paired; other cohorts descriptive only",
        reserve_bytes=5*1024**3,max_prediction_bytes=256*1024**2,missing_assets=missing,
        GPU_started=False,automatic_boot=False,no_new_method_claim=True)
    write(args.out/"manifest.json",manifest)
    print(json.dumps(dict(state=manifest["state"],missing_assets=missing,GPU_started=False)),flush=True)


def freeze_episode_metadata(module,manifest):
    """Literal constructors + independent RNG simulation, no Query mask decode.

    Box expectations are verified against actual global RNG at run time. The
    constructors generate their original test_items; no sorting is introduced.
    """
    draw=random.Random(0);episodes=[];fold_stats=[];began=time.monotonic()
    for fold in range(4):
        dataset=module.COCOFSSDataset(lists_root=str(Path(manifest["source"])/"data/lists/coco/fss_list"),
            data_root=str(Path(manifest["source"])/"data/MSCOCO2014"),fold=fold,k_shot=1,mode="val",seed=0)
        if len(dataset)!=1000:raise RuntimeError("Literal author dataset did not produce1000 episodes in fold"+str(fold))
        fold_stats.append(dict(fold=fold,active_category_ids=dataset.active_cat_ids,
            target_pool_counts={str(cat):len(dataset.target_pool[cat]) for cat in dataset.active_cat_ids},
            support_pool_counts={str(cat):len(dataset.support_pool[cat]) for cat in dataset.active_cat_ids}))
        for index,(cat,target_id,ref_ids) in enumerate(dataset.test_items):
            if len(ref_ids)!=1:raise RuntimeError("Author1-shot did not select one reference")
            ref_id=ref_ids[0];query=dataset.coco.loadImgs(target_id)[0];support=dataset.coco.loadImgs(ref_id)[0]
            valid=[ann for ann in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=ref_id,catIds=cat))
                   if ann["area"]>1000 and not ann.get("iscrowd",0)]
            if not valid:raise RuntimeError("Prepared support has no literal valid instance annotation")
            before=rng_digest(draw.getstate());chosen=draw.choice(valid);after=rng_digest(draw.getstate())
            for image in (support,query):
                if not (Path(manifest["image_root"])/image["file_name"]).is_file():
                    raise FileNotFoundError("Actual native JPEG missing: "+image["file_name"])
            episodes.append(dict(fold=fold,episode=index,cat_id=cat,query_id=target_id,support_id=ref_id,
                query=query["file_name"],support=support["file_name"],query_hw=[query["height"],query["width"]],
                support_hw=[support["height"],support["width"]],support_box=chosen["bbox"],support_annotation_id=chosen["id"],
                global_box_rng_before=before,global_box_rng_after=after))
        del dataset
        import gc
        gc.collect()
        print(json.dumps(dict(state="AUTHOR_METADATA_FOLD_FROZEN",fold=fold,episodes=len(episodes),
            elapsed_seconds=time.monotonic()-began,Query_mask_pixels_decoded=False)),flush=True)
    return dict(state="AUTHOR_4000_EPISODE_METADATA_FROZEN",episodes=episodes,folds=fold_stats,
        class_id_to_name=module.COCO_ID_TO_NAME,
        source_mask_or_Query_mask_pixels_decoded=False,box_rng_simulation_must_match_actual_main=True)


def lightweight_author_dataset(source):
    """Compile literal dataset body for metadata only; nominal base is object.

    No import of dataset_tool.py or evaluator is performed here because their
    transitive Torch/SAM imports must not coexist with COCO JSON/index memory.
    The original class AST, all methods and constants remain unchanged. This
    compilation is never used by a GPU run, which imports the actual bundle.
    """
    if any(name=="torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("Lightweight metadata phase must precede every Torch import")
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    path=source/"data/dataset_tool.py";text=path.read_text();tree=ast.parse(text,str(path))
    names={"COCO_ID_TO_NAME","SORTED_COCO_IDS"};constants=[];classes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in names for target in node.targets):
            constants.append(node)
        elif isinstance(node,ast.ClassDef) and node.name=="COCOFSSDataset":classes.append(node)
    if len(constants)!=2 or len(classes)!=1 or ast.dump(classes[0].bases[0])!="Name(id='Dataset', ctx=Load())" or len(classes[0].bases)!=1:
        raise RuntimeError("Pinned original dataset constants/class/base contract changed")
    selected=ast.Module(body=[*constants,classes[0]],type_ignores=[])
    namespace=dict(os=os,random=random,np=np,Image=Image,COCO=COCO,Dataset=object,
        __name__="author_metadata_only_original_dataset_body")
    exec(compile(selected,str(path),"exec"),namespace)
    if any(name=="torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("Lightweight author metadata dependencies imported Torch")
    method_hashes={node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        for node in classes[0].body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    contract=dict(dataset_file=str(path),dataset_file_sha256=sha(path),dataset_git_blob=git_blob(path),
        original_class_AST_sha256=hashlib.sha256(ast.dump(classes[0],include_attributes=False).encode()).hexdigest(),
        original_method_AST_sha256=method_hashes,
        original_constant_AST_sha256={node.targets[0].id:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest() for node in constants},
        nominal_Dataset_base="object",original_class_and_method_AST_changed=False,
        original_constructor_and_test_items_body_used=True,metadata_only_compilation_used_for_GPU=False,
        Torch_imported=False,pretrained_model_loaded=False,Query_mask_pixels_decoded=False)
    return SimpleNamespace(COCOFSSDataset=namespace["COCOFSSDataset"],COCO_ID_TO_NAME=namespace["COCO_ID_TO_NAME"]),contract


def metadata_only(args):
    """Bounded fresh child: full4x1000 original metadata, then process exit."""
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import resource
    manifest=read(args.manifest);exact_files(manifest["sources"]);exact_files(manifest["inputs"])
    if manifest["missing_assets"]:raise RuntimeError("Literal author metadata requires the original instances_val2014.json")
    if (args.out/"episodes4000.json").exists():raise RuntimeError("Preserve existing frozen metadata; no silent overwrite")
    sys.path[0:0]=[manifest["own_dependency_prefix"],manifest["approved_dependency_prefix"],"/root/demo4_cache/env"]
    began=time.monotonic()
    write(args.out/"metadata_process.json",dict(state="AUTHOR_METADATA_ONLY_STARTED",PID=os.getpid(),Torch_imported=False,
        manifest_sha256=sha(args.manifest),total_expected_episodes=4000))
    print(json.dumps(dict(state="AUTHOR_METADATA_ONLY_STARTED",PID=os.getpid(),Torch_imported=False)),flush=True)
    try:
        module,contract=lightweight_author_dataset(Path(manifest["source"]))
        frozen=freeze_episode_metadata(module,manifest);frozen["manifest_sha256"]=sha(args.manifest)
        frozen["metadata_source_contract"]=contract
        if len(frozen["episodes"])!=4000:raise RuntimeError("Original metadata phase did not freeze all4000 episodes")
        write(args.out/"episodes4000.json",frozen)
        proof=dict(state="AUTHOR_METADATA_ONLY4000_PASSED_TORCH_NOT_IMPORTED",PID=os.getpid(),
            manifest_sha256=sha(args.manifest),episode_metadata_sha256=sha(args.out/"episodes4000.json"),
            episode_metadata_count=4000,source_contract=contract,elapsed_seconds=time.monotonic()-began)
        proof["observed_peak_RSS_KiB"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write(args.out/"metadata_process.json",proof)
        print(json.dumps(dict(state=proof["state"],episodes=4000,elapsed_seconds=proof["elapsed_seconds"])),flush=True)
    except BaseException as error:
        write(args.out/("metadata_error_"+str(time.time_ns())+".json"),dict(state="ERROR_AUTHOR_METADATA_ONLY",
            PID=os.getpid(),error=repr(error),traceback=traceback.format_exc(),Torch_imported="torch" in sys.modules))
        raise


def ensure_metadata_before_torch(args,manifest):
    if manifest["missing_assets"]:return None
    if any(name=="torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("CPU preflight must complete lightweight metadata before importing Torch")
    proof_path=args.out/"metadata_process.json";frozen_path=args.out/"episodes4000.json"
    if not frozen_path.is_file():
        with (args.out/"metadata_only.log").open("a") as log:
            command=[args.python,str(Path(__file__).resolve()),"--metadata-only","--manifest",str(args.manifest.resolve()),"--out",str(args.out.resolve())]
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=False)
        if result.returncode:raise RuntimeError("Lightweight author metadata child failed; preserve metadata_only.log")
    proof=read(proof_path)
    if (proof.get("state")!="AUTHOR_METADATA_ONLY4000_PASSED_TORCH_NOT_IMPORTED" or
        proof.get("manifest_sha256")!=sha(args.manifest) or proof.get("episode_metadata_count")!=4000 or
        proof.get("episode_metadata_sha256")!=sha(frozen_path) or proof.get("source_contract",{}).get("Torch_imported") is not False or
        proof.get("source_contract",{}).get("dataset_file_sha256")!=sha(Path(manifest["source"])/"data/dataset_tool.py")):
        raise RuntimeError("Sequential original4000 metadata proof/hash/source differs")
    return proof


def cpu_preflight(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    manifest=read(args.manifest);exact_files(manifest["sources"]);exact_files(manifest["inputs"])
    metadata_proof=ensure_metadata_before_torch(args,manifest)
    sys.path[0:0]=[manifest["own_dependency_prefix"],manifest["approved_dependency_prefix"],"/root/demo4_cache/env"]
    import numpy as np
    import torch
    torch.set_num_threads(1)
    module=author_module(Path(manifest["source"]))
    # Literal numerical functions on tiny known arrays, never a pretrained model.
    metric=module.ClassWiseIOUMetric();truth=np.array([[1,1],[0,0]],np.uint8)
    metric.update(np.array([[1,0],[0,0]],np.uint8),truth,1)
    metric.update(truth,truth,1);metric.update(np.zeros_like(truth),truth,2)
    if metric.intersections!={1:3.,2:0.} or metric.unions!={1:4.,2:2.} or metric.compute()!=.375:
        raise RuntimeError("Literal class-sum author metric fixture differs")
    from PIL import Image
    dummy=type("ProcessorMetadataOnly",(),{"device":"cpu"})()
    evaluator=module.PairwiseSam3Evaluator(dummy,swap_order=True,use_text=False)
    ref=dict(image=Image.new("RGB",(640,480)),box=[10,20,30,40])
    target=dict(image=Image.new("RGB",(320,240)),mask=np.zeros((240,320),np.uint8))
    canvas,placements=evaluator.create_input(ref,target);box=evaluator.get_norm_box(placements)
    if canvas.size!=(1008,1008) or placements["ref"]["offset"]!=(0,404) or placements["ref"]["curr_size"]!=(1008,604) or placements["tgt"]["curr_size"]!=(1008,404):
        raise RuntimeError("Literal author1008/.6/COCOswap geometry differs")
    if len(box)!=4 or any(not math.isfinite(value) for value in box):raise RuntimeError("Author native box normalisation invalid")
    mining=author_module(Path(manifest["source"]),"mining_positive")
    mining_metric=mining.ClassWiseIOUMetric();mining_metric.update(truth,truth,1)
    mining_evaluator=mining.PairwiseSam3Evaluator(dummy,max_neg=0)
    mining_canvas,mining_placements=mining_evaluator.create_input(ref,target)
    if mining_metric.compute()!=1 or mining_canvas.size!=(1008,1008) or mining_placements["ref"]["offset"]!=(0,404):
        raise RuntimeError("Literal mining max_neg0 metric/canvas fixture differs")
    from sam3.model.vitdet import Mlp
    if Mlp.__module__!="timm.layers.mlp":raise RuntimeError("Author bundled native timm.Mlp import identity differs")
    if torch.cuda.is_initialized():raise RuntimeError("No-card CPU preflight initialized CUDA")
    receipt=dict(state="AUTHOR_IMPORT_AND_LITERAL_CPU_FIXTURES_PASSED_ASSET_HELD" if manifest["missing_assets"] else "AUTHOR_SERVER_CPU_PREFLIGHT_PASSED_GPU_UNTESTED",
        manifest_sha256=sha(args.manifest),source_hashes=manifest["sources"],literal_metric_fixture_passed=True,
        literal_canvas_and_box_fixture_passed=True,pretrained_model_loaded=False,CUDA_initialized=False,
        literal_mining_positive_metric_and_canvas_fixture_passed=True,
        numerical_dtype_or_TF32_override_applied=False,missing_assets=manifest["missing_assets"],
        imported_author_SAM3_package=str(sys.modules["sam3"].__file__),bundled_timm_MLP=True,
        author_MLP_import_module=Mlp.__module__,
        library_versions=dict(torch=torch.__version__,numpy=np.__version__),class_id_to_name=module.COCO_ID_TO_NAME,GPU_started=False)
    if not manifest["missing_assets"]:
        receipt["episode_metadata_sha256"]=metadata_proof["episode_metadata_sha256"]
        receipt["episode_metadata_count"]=4000
        receipt["metadata_phase"]=metadata_proof
        receipt["COCO_JSON_and_Torch_live_in_separate_processes"]=True
    write(args.out/"cpu_preflight.json",receipt)
    if not manifest["missing_assets"]:make_guard(args,manifest,receipt)
    print(json.dumps(dict(state=receipt["state"],missing_assets=manifest["missing_assets"],GPU_started=False)),flush=True)


def make_guard(args,manifest,receipt):
    env=dict(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DEMO9_AUTHOR_ROOT_LAUNCH="1",
        PYTHONPATH=manifest["source"]+":"+manifest["own_dependency_prefix"]+":"+manifest["approved_dependency_prefix"]+":/root/demo4_cache/env",
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    prerequisites=[dict(path=str(args.manifest.resolve()),sha256=sha(args.manifest)),
        dict(path=str((args.out/"cpu_preflight.json").resolve()),sha256=sha(args.out/"cpu_preflight.json"),json_equals=dict(state=receipt["state"])),
        dict(path=str((args.out/"episodes4000.json").resolve()),sha256=sha(args.out/"episodes4000.json"))]
    prerequisites.extend(dict(path=path,sha256=digest) for path,digest in manifest["sources"].items() if Path(path).suffix!=".py")
    execution=dict(path=str((args.out/"execution.json").resolve()),json_equals=dict(finite_stage_finished=True))
    final=dict(path=str((args.out/"report.json").resolve()),json_equals=dict(finite_report_finished=True))
    stages=[]
    for mode,kind,cap,dependencies,output in (("run","gpu",manifest["infer_cap_seconds"],[],execution),
                                            ("report","cpu",manifest["CPU_report_cap_seconds"],[execution],final)):
        stages.append(dict(name="author_"+mode,kind=kind,**(dict(role="finalize") if kind=="cpu" else {}),cwd=str(args.out.resolve()),
            timeout_seconds=cap,argv=[args.python,str(Path(__file__).resolve()),"--"+mode,"--manifest",str(args.manifest.resolve()),"--out",str(args.out.resolve())],
            env=env,requires=[*prerequisites,*dependencies],code_files=[path for path in manifest["sources"] if Path(path).suffix==".py"],
            produces=[output],success_checks=[output]))
    write(args.out/"guard_plan.json",dict(platform="autodl",cuda_python=args.python,stages=stages,
        total_timeout_budget_seconds=4500,sum_stage_timeouts_seconds=4470,startup_and_handoff_reserve_seconds=30,
        one_literal_author_main_call=True,automatic_boot=False,no_quality_halt_after8=True))


def run_author_mode(args,manifest,metadata,mode,batch_start):
    """One unchanged author main, observed once; first8 remain in its metric."""
    if os.environ.get("DEMO9_AUTHOR_ROOT_LAUNCH")!="1":raise RuntimeError("Explicit root-owned finite guard required")
    import numpy as np
    import torch
    mode_out=args.out/mode;mode_out.mkdir(exist_ok=True)
    if (mode_out/"episodes.jsonl").exists():raise RuntimeError("No restart/reseed resume; first8 reused only inside one author main")
    module=author_module(Path(manifest["source"]),mode);start=time.monotonic()
    state=dict(successful=0,fold_counts={str(fold):0 for fold in range(4)},fold_metrics={},current=None,
        constructor_seconds=[],visual_seconds=0.,visual_count=0,model_seconds=None,current_evaluator=None,
        model_dtype_counts=None,source_code_unmodified=True,online_QueryGT_as_author=True)
    dataset_class=module.COCOFSSDataset
    original_init=dataset_class.__init__;original_get=dataset_class.__getitem__
    original_run=module.PairwiseSam3Evaluator.run_episode;original_build=module.build_sam3_image_model
    visual_name="save_visualization" if mode=="mining_positive" else "save_visualization_1shot"
    original_vis=getattr(module,visual_name)
    ledger=mode_out/"episodes.jsonl";prediction_dir=mode_out/"masks";prediction_dir.mkdir(exist_ok=True)
    original_metric_update=module.ClassWiseIOUMetric.update
    def observed_init(instance,*values,**kwargs):
        began=time.monotonic()
        try:original_init(instance,*values,**kwargs)
        except Exception as error:raise AuthorHardFailure("Author dataset initialization failed: "+repr(error))
        state["constructor_seconds"].append(time.monotonic()-began)
        expected=[row for row in metadata["episodes"] if row["fold"]==instance.fold]
        if len(instance)!=1000 or [(row["cat_id"],row["query_id"],[row["support_id"]]) for row in expected]!=instance.test_items:
            raise AuthorHardFailure("Literal local-RNG episode generation differs from frozen author metadata")
    def observed_get(instance,index):
        row=metadata["episodes"][instance.fold*1000+index]
        state["current"]=dict(row,episode_started=time.monotonic(),visual_before=state["visual_seconds"])
        if rng_digest(random.getstate())!=row["global_box_rng_before"]:
            raise AuthorHardFailure("Global author support-box RNG changed; no re-seed/approximate substitution")
        try:output=original_get(instance,index)
        except Exception as error:raise AuthorHardFailure("Author dataset getitem failed: "+repr(error))
        references,targets,cat_id,_=output
        if (len(references)!=1 or len(targets)!=1 or cat_id!=row["cat_id"] or references[0]["name"]!=row["support"] or
                targets[0]["name"]!=row["query"] or references[0]["box"]!=row["support_box"] or
                rng_digest(random.getstate())!=row["global_box_rng_after"]):
            raise AuthorHardFailure("Original author physical RGB/instance-box/RNG draw differs; no hidden loader skip")
        return output
    def asset_build(*values,**kwargs):
        before=time.monotonic();kwargs.update(bpe_path=manifest["bpe"],checkpoint_path=manifest["checkpoint"],load_from_HF=False)
        try:model=original_build(*values,**kwargs)
        except Exception as error:raise AuthorHardFailure("Unmodified bundled author model build failed: "+repr(error))
        state["model_seconds"]=time.monotonic()-before
        state["model_dtype_counts"]=dict(Counter(str(parameter.dtype) for parameter in model.parameters()))
        state["runtime_precision"]=dict(default_dtype=str(torch.get_default_dtype()),
            autocast_cuda_enabled=torch.is_autocast_enabled("cuda"),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
            TF32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,torch=torch.__version__)
        return model
    def observed_vis(*values,**kwargs):
        before=time.monotonic()
        try:return original_vis(*values,**kwargs)
        finally:state["visual_seconds"]+=time.monotonic()-before;state["visual_count"]+=1
    def observed_update(metric,pred,gt,cat_id):
        original_metric_update(metric,pred,gt,cat_id)
        row=state["current"]
        if row is None:raise AuthorHardFailure("Original metric update outside a recorded author episode")
        binary=np.asarray(pred)>0;truth=np.asarray(gt)>0
        path=prediction_dir/("f%d_e%04d_c%d.npz"%(row["fold"],row["episode"],cat_id))
        np.savez_compressed(path,prediction=np.packbits(binary.reshape(-1)),shape=np.array(binary.shape,np.int64))
        row["original_IU"]=[int(np.logical_and(binary,truth).sum()),int(np.logical_or(binary,truth).sum())]
        row["prediction_path"]=str(path);row["prediction_sha256"]=sha(path)
        row["author_class_intersections"]=dict(metric.intersections);row["author_class_unions"]=dict(metric.unions)
    def observed_run(evaluator,*values,**kwargs):
        state["current_evaluator"]=evaluator
        changed=evaluator.split_ratio!=.6 or evaluator.canvas_size!=1008
        if mode=="mining_positive":changed=changed or evaluator.max_neg!=0
        else:changed=changed or evaluator.use_text or evaluator.orientation!="vertical" or not evaluator.swap_order
        if changed:
            raise AuthorHardFailure("Literal author1-shot defaults changed")
        try:result=original_run(evaluator,*values,**kwargs)
        except Exception as error:raise AuthorHardFailure("Unmodified author inference/metric failed: "+repr(error))
        row=state["current"]
        if "original_IU" not in row:raise AuthorHardFailure("Author episode produced no metric update")
        row.update(wall_seconds=time.monotonic()-row.pop("episode_started"),
            visualization_seconds=state["visual_seconds"]-row.pop("visual_before"),
            RNG_observer_draws=0,author_GT_loaded_before_prediction=True,author_numerical_functions_unmodified=True)
        with ledger.open("a") as stream:stream.write(json.dumps(row)+"\n");stream.flush();os.fsync(stream.fileno())
        state["successful"]+=1;state["fold_counts"][str(row["fold"])]+=1
        state["fold_metrics"][str(row["fold"])]=dict(miou=float(evaluator.metric.compute()),
            intersections=dict(evaluator.metric.intersections),unions=dict(evaluator.metric.unions))
        print(json.dumps(dict(state="AUTHOR_EPISODE_COMPLETED",fold=row["fold"],episode=row["episode"],cat_id=row["cat_id"],
            successful=state["successful"],original_IU=row["original_IU"],actual_bbox=row["support_box"],
            wall_seconds=row["wall_seconds"],elapsed_seconds=time.monotonic()-start)),flush=True)
        if state["successful"]==8:
            first=[json.loads(line) for line in ledger.read_text().splitlines()]
            nonvis=sum(item["wall_seconds"]-item["visualization_seconds"] for item in first)/8
            vismean=state["visual_seconds"]/max(state["visual_count"],1)
            remaining=manifest["infer_cap_seconds"]-(time.monotonic()-batch_start)
            total_vis=AUTHOR_MODES[mode]["visualizations"]
            predicted=3992*nonvis+max(total_vis-state["visual_count"],0)*vismean+3*state["constructor_seconds"][0]
            cost=dict(state="AUTHOR8_COST_GATE",actual_first8_seconds=sum(item["wall_seconds"] for item in first),
                model_load_seconds=state["model_seconds"],nonvisual_case_mean_seconds=nonvis,
                author_visualization_mean_seconds=vismean,author_visualizations_total=total_vis,
                conservative_remaining_seconds=predicted,remaining_registered_seconds=remaining,
                can_fit=predicted<=remaining,first8_not_rerun=True,no_quality_or_CI_gate=True,
                not_a_tail_runtime_guarantee=True)
            write(mode_out/"first8_cost.json",cost)
            if predicted>remaining:raise AuthorCostHold("Measured author runtime cannot fit75min; stop before full, no rerental")
        if time.monotonic()-batch_start>=manifest["infer_cap_seconds"]:raise AuthorCostHold("Shared finite author cap reached")
        if shutil.disk_usage(args.out).free<manifest["reserve_bytes"] or sum(path.stat().st_size for path in prediction_dir.glob("*.npz"))>manifest["max_prediction_bytes"]:
            raise AuthorHardFailure("Author evidence size/disk reserve gate")
        return result
    dataset_class.__init__=observed_init;dataset_class.__getitem__=observed_get
    module.PairwiseSam3Evaluator.run_episode=observed_run;module.ClassWiseIOUMetric.update=observed_update
    module.build_sam3_image_model=asset_build;setattr(module,visual_name,observed_vis)
    old_argv=sys.argv;old_cwd=Path.cwd();mode_args=AUTHOR_MODES[mode]["argv"]
    sys.argv=[str(Path(manifest["source"])/AUTHOR_MODES[mode]["entry"]),*mode_args]
    status="AUTHOR_COMPLETE4000"
    try:
        os.chdir(manifest["author_cwd"]);module.main()
        if state["successful"]!=4000 or set(state["fold_counts"].values())!={1000}:
            raise AuthorHardFailure("Literal main returned without4x1000 successful episodes; silent skips prohibited")
    except AuthorCostHold as error:status="RESOURCE_HOLD_AFTER_MEASURED_AUTHOR_PREFIX";state["resource_reason"]=str(error)
    except BaseException as error:
        write(mode_out/("error_"+str(time.time_ns())+".json"),dict(state="ERROR_AUTHOR_INTERFACE",error=repr(error),traceback=traceback.format_exc(),
            successful=state["successful"],not_a_scientific_negative=True));raise
    finally:
        sys.argv=old_argv;os.chdir(old_cwd)
        dataset_class.__init__=original_init;dataset_class.__getitem__=original_get
        module.PairwiseSam3Evaluator.run_episode=original_run;module.ClassWiseIOUMetric.update=original_metric_update
        module.build_sam3_image_model=original_build;setattr(module,visual_name,original_vis)
    state.pop("current_evaluator",None);state.pop("current",None)
    result=dict(state=status,finite_stage_finished=True,mode=mode,manifest_sha256=sha(args.manifest),
        source_hashes=manifest["sources"],author_argv=mode_args,elapsed_seconds=time.monotonic()-start,
        first8_continued_in_same_main=True,author_main_calls=1,numerical_source_or_precision_patches=0,
        author_online_GT_protocol=True,statistics=state,skipped_author_episodes=0)
    write(mode_out/"execution.json",result);return result


def run_author(args):
    if os.environ.get("DEMO9_AUTHOR_ROOT_LAUNCH")!="1":raise RuntimeError("Explicit root-owned finite guard required")
    manifest=read(args.manifest);exact_files(manifest["sources"]);exact_files(manifest["inputs"])
    proof=read(args.out/"cpu_preflight.json")
    if proof.get("state")!="AUTHOR_SERVER_CPU_PREFLIGHT_PASSED_GPU_UNTESTED" or proof.get("manifest_sha256")!=sha(args.manifest):
        raise RuntimeError("Actual SERVER import/source/assets4000 preflight required")
    metadata=read(args.out/"episodes4000.json")
    if sha(args.out/"episodes4000.json")!=proof["episode_metadata_sha256"]:raise RuntimeError("Prepared4000 metadata changed")
    if (args.out/"execution.json").exists():raise RuntimeError("No duplicate/resumed author batch")
    import torch
    torch.cuda.set_per_process_memory_fraction(manifest["allocator_GPU_memory_fraction"])
    begin=time.monotonic();results={}
    for mode in manifest["mode_priority"]:
        remaining=manifest["infer_cap_seconds"]-(time.monotonic()-begin)
        if remaining<=0:
            results[mode]=dict(state="RESOURCE_SKIPPED_NO_REMAINING_CAP",finite_stage_finished=True,successful=0);continue
        results[mode]=run_author_mode(args,manifest,metadata,mode,begin)
    write(args.out/"execution.json",dict(state="FINITE_AUTHOR_MODES_FINISHED",finite_stage_finished=True,
        manifest_sha256=sha(args.manifest),modes=results,elapsed_seconds=time.monotonic()-begin,
        one_paid_boot_scope=True,shared_cap_seconds=4500,primary="mining_positive",secondary="standard_visual"))


def report(args):
    execution=read(args.out/"execution.json");manifest=read(args.manifest)
    if execution.get("manifest_sha256")!=sha(args.manifest):raise RuntimeError("Execution/report manifest changed")
    results={}
    for mode,result in execution["modes"].items():
        complete=result["state"]=="AUTHOR_COMPLETE4000";folds=result.get("statistics",{}).get("fold_metrics",{})
        score=100*sum(folds[str(f)]["miou"] for f in range(4))/4 if complete else None
        results[mode]=dict(state=result["state"],complete_author4000=complete,
            episodes=result.get("statistics",{}).get("successful",0),fold_class_macro_IU=folds,
            mean_four_fold_macro_mIoU_percent=score,table_anchor=AUTHOR_MODES[mode]["table_anchor"],
            measured_prefix_not_full_paper_reproduction=not complete)
    write(args.out/"report.json",dict(state="FINITE_LITERAL_AUTHOR_REPORT",finite_report_finished=True,
        modes=results,author_commit=COMMIT,class_id_to_name=read(args.out/"cpu_preflight.json")["class_id_to_name"],
        protocol=manifest,execution=execution,model_or_dataset_adaptation_used=False,
        author_instance_boxes_native_JPEG_and_noncrowd_Query_GT=True,
        published66_6_anchors_mining_maxneg0_not_standard=True,no_paired_claim_against_other_cohorts=True,
        no_method_or_acceptance_claim=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare","metadata-only","cpu-preflight","run","report"):modes.add_argument("--"+mode,action="store_true")
    parser.add_argument("--source",type=Path);parser.add_argument("--image-root",type=Path)
    parser.add_argument("--instances",type=Path);parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--checkpoint-status",type=Path);parser.add_argument("--bpe",type=Path)
    parser.add_argument("--own-prefix",type=Path,default=Path("/root/autodl-tmp/sam3_author_reproduction/python"))
    parser.add_argument("--shared-prefix",type=Path,default=Path("/root/autodl-tmp/sam3_preparation/python"))
    parser.add_argument("--manifest",type=Path);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--python",default="/root/miniconda3/bin/python");parser.add_argument("--total-seconds",type=int,default=4500)
    args=parser.parse_args();server_only()
    if args.prepare:
        if any(getattr(args,name) is None for name in ("source","image_root","checkpoint","checkpoint_status","bpe")):
            parser.error("Existing pinned source, native JPEGs and approved checkpoint/status/BPE required")
        prepare(args)
    elif args.manifest is None:parser.error("--manifest required")
    elif args.metadata_only:metadata_only(args)
    elif args.cpu_preflight:cpu_preflight(args)
    elif args.run:run_author(args)
    else:report(args)


if __name__=="__main__":main()
