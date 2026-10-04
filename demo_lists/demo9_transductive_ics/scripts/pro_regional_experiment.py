#!/usr/bin/env python3
"""Pro masked regional construction, isolated SERVER-only oldDEV40 preparation.

Source AND Query regional RGB are masked as specified in the supplied draft.
162 region crops shared by all controls; fullFoRIS stays an external comparison.
No natural-Query substitution, fresh40, oldqueue, downloads or automatic launch.
Whole-cohort predictions freeze before QueryGT. CPU preparation is not a score.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"scripts"
RUNTIME_PATHS=("/root/autodl-tmp/demo8_local_verification/crf_source/src",
               "/root/autodl-tmp/demo8_local_verification/runtime/extensions")
sys.path[:0]=list(RUNTIME_PATHS)+[str(ROOT),str(HERE),"/root/demo4_cache/env"]
BASE_ARMS=("raw_local","regional_local","old_global_local","main","source_cls_shuffled",
           "global_only","route_top1","route_top4","global_plus_local")
ARMS=("native",*BASE_ARMS)
GROUPS={"Pro_joint":dict(primary="main",controls=["regional_local","old_global_local",
         "route_top1","route_top4","global_plus_local","source_cls_shuffled","global_only"])}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w") as stream:json.dump(value,stream,indent=1,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def server_only():
    if not sys.platform.startswith("linux") or not str(ROOT).startswith("/root/"):raise RuntimeError("SERVER-only; local source/AST access, never model/numerical work")


def pure(row):return {key:row[key] for key in ("fold","e","c","support","query")}
def identity(row):return tuple(row[key] for key in ("fold","e","c","support","query"))


def validate(manifest):
    for path,digest in manifest["source_hashes"].items():
        if not Path(path).is_file() or sha(path)!=digest:raise RuntimeError("Frozen source/metadata changed: "+path)
    for item in manifest["assets"]:
        stat=Path(item["path"]).stat()
        if stat.st_size!=item["size"] or stat.st_mtime_ns!=item["mtime_ns"]:raise RuntimeError("Existing asset changed: "+item["path"])


def prepare(args):
    import numpy as np
    from PIL import Image
    if args.out.exists() and any(args.out.iterdir()):raise RuntimeError("Fresh Pro output required")
    old=read(args.old_manifest);fixed=read(args.fixed8_manifest)
    lookup={identity(pure(case["row"])):case for case in old["cases"]}
    eight=[pure(row) for row in fixed["frozen_episodes"]]
    if len(lookup)!=40 or len(eight)!=8 or any(sum(r["fold"]==f for r in eight)!=2 for f in range(4)):
        raise RuntimeError("Existing four-fold DEV40 and balanced8 required")
    selected=[lookup[identity(row)] for row in eight]+[case for case in old["cases"] if identity(pure(case["row"])) not in set(map(identity,eight))]
    rows=[dict(**pure(case["row"]),cohort="oldDEV40",case_id="dev_%d_%d_%d"%(case["row"]["fold"],case["row"]["e"],case["row"]["c"])) for case in selected]
    sources=[Path(__file__),HERE/"reference_qk_experiment.py",HERE/"experiment_resource_guard.py",
             ROOT/"tics/regional_observation.py",ROOT/"tics/regional_label_transfer.py",ROOT/"tics/native_assets.py",ROOT/"tics/paired_frozen_stats.py",
             Path(args.demo4_root)/"icx/common.py",args.old_manifest,args.fixed8_manifest,
             *[args.foris_root/name for name in ("models/foris.py","utils/data.py","utils/clustering.py","utils/refinement.py")]]
    for folder in RUNTIME_PATHS:
        if not Path(folder).is_dir():raise FileNotFoundError("Existing CRF runtime required: "+folder)
    hashes={str(path.resolve()):sha(path) for path in sources}
    assets=[];headers={};baselines={}
    for row,case in zip(rows,selected):
        for role in ("support","query"):
            image=Path(old["data_root"])/row[role];label=Path(old["annotation_root"])/Path(row[role]).with_suffix(".png")
            with Image.open(image) as im:hw=[im.height,im.width]
            with Image.open(label) as im:lhw=[im.height,im.width]
            if hw!=lhw:raise RuntimeError("RGB/annotation headers differ")
            headers.setdefault(row["case_id"],{})[role]=hw
            for path in (image,label):
                st=path.stat();assets.append(dict(path=str(path.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns))
        path=Path(case["foris_packet"])
        with np.load(path,allow_pickle=False) as packet:bits=packet["native"].copy()
        if bits.dtype!=np.uint8 or bits.size!=1024*1024//8:raise RuntimeError("Original native bitmap missing")
        baselines[row["case_id"]]=dict(path=str(path),sha256=sha(path));hashes[str(path.resolve())]=sha(path)
    model_root=Path("/root/demo4_cache/models/dinov3-vitl16-timm")
    for path in (args.projection_basis,model_root/"config.json",model_root/"model.safetensors"):
        st=path.stat();assets.append(dict(path=str(path.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.out.parent).free<5*1024**3:raise RuntimeError("5GiB reserve required")
    manifest=dict(state="PRO40_PREPARED_GPU_HELD",schema="pro_masked_regional_v1",episodes=rows,expected_cases=40,pilot_cases=8,
        data_root=old["data_root"],annotation_root=old["annotation_root"],headers=headers,baselines=baselines,
        foris_root=str(args.foris_root),demo4_root=str(args.demo4_root),projection_basis=str(args.projection_basis.resolve()),DINO_model_root=str(model_root),
        source_hashes=hashes,assets=assets,seed=0,groups=GROUPS,arms=list(ARMS),primary="main",
        primary_observation="Pro construction: BOTH Source and Query regions use masked RGB; Source adds true FG/BG views",
        recipe=dict(work_size=1024,crop_size=256,regions=[16,64],source_crop_images=82,query_crop_images=80,
            native1024_images=2,regional_images_per_case=162,whole_image_view="reuse native full-image encoding",
            Ward="spatial4adjacent SSE on original raw features; identical to prepared library",tau_local=.07,tau_global=.07,
            observation_mixture="uniform3 Query x4 Source views; per-role area normalized",natural_query=False,
            postprocess="sigmoid -> bilinear1024 -> threshold.5 -> fixed CRF -> original; exact pre/post masks saved"),
        card=["Assumption: correct regionalCLS improves the role of local Source-to-Query correspondences beyond same-observation local/routing controls.",
              "Predeclared hypothesis target, not measured forecast: main improves complete native and strongest same-observation control by>=2pp; unchanged native replay differs0pixels. No win probability.",
              "Match: retain the masked regional mechanism for later independent confirmation; oldDEV40 alone is not a paper result.",
              "Mismatch: main below strong controls rejects this construction; no natural-crop substitution, threshold sweep or fresh-label selection. Interface failures stop separately."],
        GPU_total_cap_seconds=2400,GPU_run_cap_seconds=2400,CPU_report_cap_seconds=60,startup_reserve_seconds=30,total_cap_seconds=2490,
        GPU_memory_fraction=.55,reserve_bytes=5*1024**3,max_output_bytes=128*1024**2,
        all_QueryGT_after_whole_actual_cohort_freeze=True,SourceBG_empty="native fallback, counted",
        first8_gate="software/native parity and projected cost only; no label-based gate",no_automatic_boot=True,
        fresh_labels_used=False,preparation_only_authorized=True)
    write(args.out/"manifest.json",manifest)
    print(json.dumps(dict(state=manifest["state"],cases=40,masked_query=True)),flush=True)


@contextmanager
def true_native_cls(backbone):
    """Read actual finalnorm before the native wrapper splits fiveprefix."""
    data={};handles=[]
    def last(module,values,output):data["last_block_seen"]=True
    def norm(module,values,output):
        if data.get("last_block_seen") and output.ndim==3 and output.shape[:2]==(2,4101):
            if "sequence" in data:raise RuntimeError("More than one native finalnorm sequence")
            data["sequence"]=output.detach().clone()
    handles.append(backbone.blocks[-1].register_forward_hook(last));handles.append(backbone.norm.register_forward_hook(norm))
    try:
        yield data
        if "sequence" not in data:raise RuntimeError("Actual native fullseq CLS missing; no pooled/derived substitute")
    finally:
        for handle in handles:handle.remove()


def encode_views(model,collections,device):
    """One shared batch8 stream over all Source/Query observation variants."""
    import torch
    sizes=[tensor.shape[0] for tensor in collections];all_rgb=torch.cat(collections,dim=0)
    tokens=[];cls=[];calls=0
    for start in range(0,len(all_rgb),8):
        batch=all_rgb[start:start+8].to(device)
        with torch.autocast("cuda",dtype=torch.bfloat16):sequence=model.forward_features(batch)
        if not isinstance(sequence,torch.Tensor) or sequence.shape!=(len(batch),261,1024) or not torch.isfinite(sequence).all():raise RuntimeError("Actual256 forward_features fullCLS/patch sequence contract failed")
        cls.append(sequence[:,0].float().detach());tokens.append(sequence[:,5:].float().reshape(len(batch),16,16,1024).detach());calls+=1
    whole_tokens=torch.cat(tokens);whole_cls=torch.cat(cls)
    chunks=[];position=0
    for size in sizes:chunks.append((whole_tokens[position:position+size],whole_cls[position:position+size]));position+=size
    return chunks,dict(regional_encoder_calls=calls,regional_images=len(all_rgb),regional_batch_cap=8,last_batch_images=len(all_rgb)%8 or 8,
        encoder_context="same nativeBF16",actual_features_input_dtype=str(sequence.dtype),correspondence_explicit_FP32=True)


def suite_inputs(source_obs,query_obs,source_tokens,query_tokens,source_cls,query_cls,source_whole,query_whole,source_whole_cls,query_whole_cls):
    import torch
    from tics.regional_observation import sample_descriptors,build_global_bank,old_pool_bank
    S=sample_descriptors(source_tokens,source_whole,source_obs);Q=sample_descriptors(query_tokens,query_whole,query_obs)
    sp=old_pool_bank(source_whole,source_obs);qp=old_pool_bank(query_whole,query_obs)
    if not sp["valid"].all() or not qp["valid"].all() or not S["valid"].all() or not Q["valid"].all():raise RuntimeError("Missing real observation/role prototype; no synthetic replacement")
    return dict(source_local=S["local"],query_local=Q["local"],source_raw=S["raw"],query_raw=Q["raw"],
        source_global=build_global_bank(source_cls,source_whole_cls,source_obs),query_global=build_global_bank(query_cls,query_whole_cls,query_obs),
        source_old_global=sp["bank"],query_old_global=qp["bank"],
        source_template_ids=torch.as_tensor(S["template_ids"],dtype=torch.long,device=S["local"].device),
        query_template_ids=torch.as_tensor(Q["template_ids"],dtype=torch.long,device=Q["local"].device),
        source_roles=torch.as_tensor(S["roles"],dtype=torch.bool,device=S["local"].device),
        source_area=torch.as_tensor(S["area"],dtype=S["local"].dtype,device=S["local"].device))


def postprocess(host,query_work,score,original_hw):
    import torch
    import torch.nn.functional as F
    if score.shape!=(4096,) or not torch.isfinite(score).all():raise RuntimeError("Finite full4096 classifier score required")
    pre=F.interpolate(score.sigmoid().reshape(1,1,64,64),(1024,1024),mode="bilinear",align_corners=False)[0,0]>.5
    work=host._finalize_mask(pre,query_work[None]).reshape(1024,1024).bool()
    return F.interpolate(work[None,None].float(),original_hw,mode="bilinear",align_corners=False)[0,0]>.5


def infer_pair(host,row,manifest,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from reference_qk_experiment import public_pass
    from tics.regional_observation import build_spatial_ward_tree,build_views
    from tics.regional_label_transfer import run_transfer_suite
    support=Image.open(Path(manifest["data_root"])/row["support"]).convert("RGB");query=Image.open(Path(manifest["data_root"])/row["query"]).convert("RGB")
    source_mask=torch.from_numpy((np.asarray(Image.open(Path(manifest["annotation_root"])/Path(row["support"]).with_suffix(".png")))==row["c"]+1).copy())
    if not source_mask.any():raise RuntimeError("Invalid empty Source foreground, not a case filter")
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda",dtype=torch.bfloat16),true_native_cls(host.encoder.m) as captured:native,trace=public_pass(host,support,source_mask,query)
    sequence=captured["sequence"]
    if trace["raw"].shape!=(1,2,1024,64,64) or not torch.equal(sequence[:,5:].transpose(1,2).reshape(2,1024,64,64).float(),trace["raw"][0].float()):raise RuntimeError("Actual nativeCLS/patch input identity failed")
    whole_cls=sequence[:,0].float().clone();whole=trace["raw"][0].float().permute(0,2,3,1).contiguous();del captured,sequence
    with torch.autocast("cuda",dtype=torch.bfloat16):noop,no_trace=public_pass(host,support,source_mask,query,cached=trace)
    if not torch.equal(native,noop) or not torch.equal(trace["score"],no_trace["score"]):raise RuntimeError("Untouched full-native/NoOp mask or score mismatch")
    if row["case_id"] in manifest["baselines"]:
        ref=manifest["baselines"][row["case_id"]]
        with np.load(ref["path"],allow_pickle=False) as package:old=np.unpackbits(package["native"]).reshape(1024,1024).astype(bool)
        if not np.array_equal(old,native.cpu().numpy()):raise RuntimeError("Old complete-public native mask mismatch")
    query_work=host._transform(query).to(native.device);support_work=host._transform(support).to(native.device)
    smask=trace["source_work_mask"].cpu().numpy().astype(bool);arrays={};clocks={};audits={};encoder_audit={};fallback=not (~smask).any()
    native_original=F.interpolate(native[None,None].float(),(query.height,query.width),mode="bilinear",align_corners=False)[0,0]>.5
    predictions={"native":native_original};diag={}
    if fallback:
        predictions.update({arm:native_original for arm in ARMS[1:]});arrays.update({"pre__"+arm:np.packbits(native_original.cpu().numpy().reshape(-1)) for arm in BASE_ARMS});audit=dict(state="SOURCE_BACKGROUND_EMPTY_ALL_ARMS_NATIVE_FALLBACK",not_case_filtered=True)
    else:
        tick=time.monotonic();S_tree=build_spatial_ward_tree(whole[0].cpu().numpy());Q_tree=build_spatial_ward_tree(whole[1].cpu().numpy())
        S=build_views(support_work,S_tree.cut(16),S_tree.cut(64),smask);Q=build_views(query_work,Q_tree.cut(16),Q_tree.cut(64))
        if S.num_views!=82 or Q.num_views!=80 or not S.point_valid.all() or not Q.point_valid.all():raise RuntimeError("Real spatial82/80 view/physical point contract failed")
        clocks["spatial_tree_and_views"]=time.monotonic()-tick
        tick=time.monotonic();encoded,encoder_audit=encode_views(host.encoder.m,[S.masked_rgb,Q.masked_rgb],native.device)
        clocks["all_shared_region_encodings"]=time.monotonic()-tick
        sm,sc=encoded[0];qm,qmc=encoded[1]
        primary=suite_inputs(S,Q,sm,qm,sc,qmc,whole[0],whole[1],whole_cls[0],whole_cls[1])
        fixed_seed=int(hashlib.sha256(json.dumps({key:row[key] for key in ("support","query")},sort_keys=True).encode()).hexdigest()[:8],16)
        tick=time.monotonic();result=run_transfer_suite(primary,include_ssp=False,query_chunk=128,shuffle_seed=fixed_seed)
        clocks["masked_shared_kernel_stream"]=time.monotonic()-tick
        scores={name:result["scores"][name] for name in BASE_ARMS}
        for arm,score in scores.items():
            tick=time.monotonic();predictions[arm]=postprocess(host,query_work,score,(query.height,query.width));clocks["postprocess__"+arm]=time.monotonic()-tick
            arrays["score__"+arm]=score.detach().float().cpu().numpy()
            pre=F.interpolate(score.sigmoid().reshape(1,1,64,64),(1024,1024),mode="bilinear",align_corners=False)[0,0]>.5
            pre_original=F.interpolate(pre[None,None].float(),(query.height,query.width),mode="bilinear",align_corners=False)[0,0]>.5
            arrays["pre__"+arm]=np.packbits(pre_original.cpu().numpy().reshape(-1))
        for name,value in result["diagnostics"].items():
            if value.shape!=(4096,) or not torch.isfinite(value).all():raise RuntimeError("Invalid g/l/r/h small-map contract")
            arrays["diagnostic__"+name]=value.detach().float().cpu().numpy().astype(np.float32)
        diag=result["audit"];audit=dict(primary=result["audit"],masked_query=True,shuffle_seed=fixed_seed,
            Source_physical_anchors=len(S.points_xy),Source_area_pixels=float(S.area.sum()),Source_FG_anchors=int(S.roles.sum()),Source_BG_anchors=int((~S.roles).sum()),
            S_crop_maskshape=list(S.maskshape),Q_crop_maskshape=list(Q.maskshape),CLS_not_pooled=True)
        arrays.update(query_labels16=Q.labels16.astype(np.int16),query_labels64=Q.labels64.astype(np.int16))
    if set(predictions)!=set(ARMS):raise RuntimeError("All fixed predictions required before frozen case")
    for arm,pred in predictions.items():arrays[arm]=np.packbits(pred.cpu().numpy().reshape(-1))
    arrays["original_hw"]=np.array([query.height,query.width],np.int64)
    dest=out/"masks"/(row["case_id"]+".npz");np.savez_compressed(dest,**arrays)
    torch.cuda.synchronize()
    rec=dict(row=row,case_id=row["case_id"],prediction_path=str(dest.resolve()),prediction_sha256=sha(dest),original_hw=[query.height,query.width],
        SourceBG_empty_fallback=fallback,all_predictions_FROZEN_QueryGT_closed=True,native_NoOp_exact=True,native_B2_encoder_calls=1,native1024_images=2,
        shared_region_encoder=encoder_audit,substage_seconds=clocks,audit=audit,score_storage="FP32; pre/post original masks saved exactly",scores_used_for_IU=False,
        seconds=time.monotonic()-begin,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(out/"frozen"/(row["case_id"]+".json"),rec)
    print(json.dumps(dict(state="PRO_CASE_FROZEN",case=row["case_id"],seconds=rec["seconds"],SourceBG_fallback=fallback)),flush=True)
    return rec


def cpu_preflight(args):
    os.environ["CUDA_VISIBLE_DEVICES"]="";manifest=read(args.manifest);validate(manifest)
    sys.path[:0]=[str(ROOT),str(HERE),manifest["foris_root"]]
    import torch
    from tics.regional_observation import cpu_selfcheck as roi_check
    from tics.regional_label_transfer import cpu_selfcheck as core_check
    from tics.paired_frozen_stats import paired
    torch.set_num_threads(1)
    from reference_qk_experiment import runtime_import
    runtime_import(manifest)
    from icx.common import TimmDINOv3
    import CRF
    ROI=roi_check();core=core_check()
    if torch.cuda.is_initialized():raise RuntimeError("CPU check initialized CUDA")
    # Original40 physical grouping/classes, synthetic integer I/U only. No
    # QueryGT pixels, model or regional descriptor simulation is passed off as
    # an actual encoder test. All final statistical calls are timed here.
    synthetic=[dict(**row,original_iu={arm:[100,200] for arm in (*ARMS,"fixed_family_pixel_oracle",*("pre__"+arm for arm in BASE_ARMS))}) for row in manifest["episodes"]]
    began=time.monotonic();summarize(synthetic)
    elapsed=time.monotonic()-began
    if elapsed>60:raise RuntimeError("Actual40 synthetic statistical summary exceeds fixed60s report cap")
    proof=dict(state="SERVER_PRO40_CPU_PASSED_GPU_UNTESTED",manifest_sha256=sha(args.manifest),source_hashes=manifest["source_hashes"],ROI=ROI,core=core,
        all40_statistics_seconds=elapsed,statistics_synthetic=True,CUDA_initialized=False,actual_pretrained_native_CLS_and_crop_forward_unverified=True,
        real_QueryGT_pixels_opened=False,mandatory_frames=40)
    write(args.out/"cpu_preflight.json",proof);make_guard(args,manifest)
    print(json.dumps(dict(state=proof["state"],statistics_seconds=elapsed)),flush=True)


def make_guard(args,manifest):
    env=dict(DEMO9_PRO_ROOT_LAUNCH="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
        DEMO4_ROOT=manifest["demo4_root"],DEMO4_CACHE="/root/demo4_cache",DEMO4_GPU_FRAC=str(manifest["GPU_memory_fraction"]),
        PYTHONPATH=str(ROOT)+":"+str(HERE)+":"+manifest["foris_root"]+":"+manifest["demo4_root"]+":/root/demo4_cache/env:"+":".join(RUNTIME_PATHS))
    prerequisites=[dict(path=str(args.manifest.resolve()),sha256=sha(args.manifest)),dict(path=str((args.out/"cpu_preflight.json").resolve()),sha256=sha(args.out/"cpu_preflight.json"),json_equals=dict(state="SERVER_PRO40_CPU_PASSED_GPU_UNTESTED"))]
    prerequisites += [dict(path=path,sha256=digest) for path,digest in manifest["source_hashes"].items()]
    freeze=dict(path=str((args.out/"execution.json").resolve()),json_equals=dict(state="PRO_PREDICTIONS_FROZEN_WHOLE_ACTUAL_COHORT"))
    final=dict(path=str((args.out/"summary.json").resolve()),json_equals=dict(state="PRO_COHORT_SCORED"))
    stages=[]
    for action,kind,cap,requires,product in (("run","gpu",2400,[],freeze),("report","cpu",60,[freeze],final)):
        stages.append(dict(name="pro40_"+action,kind=kind,**(dict(role="finalize") if kind=="cpu" else {}),cwd=str(args.out.resolve()),timeout_seconds=cap,
            env=env,argv=[args.python,str(Path(__file__).resolve()),"--"+action,"--manifest",str(args.manifest.resolve()),"--out",str(args.out.resolve())],
            code_files=[path for path in manifest["source_hashes"] if Path(path).suffix==".py"],requires=[*prerequisites,*requires],produces=[product],success_checks=[product]))
    write(args.out/"guard_plan.json",dict(platform="autodl",cuda_python=args.python,stages=stages,total_timeout_budget_seconds=2490,sum_stage_timeouts_seconds=2460,startup_reserve_seconds=30,no_auto_boot=True))


def run(args):
    if os.environ.get("DEMO9_PRO_ROOT_LAUNCH")!="1":raise RuntimeError("Root-owned finite guard required")
    manifest=read(args.manifest);validate(manifest);proof=read(args.out/"cpu_preflight.json")
    if proof["state"]!="SERVER_PRO40_CPU_PASSED_GPU_UNTESTED" or proof["manifest_sha256"]!=sha(args.manifest):raise RuntimeError("Actual SERVER source/ROI/core preflight required")
    if (args.out/"frozen").exists():raise RuntimeError("Fresh finite invocation only; no resumption or oldqueue")
    for name in ("masks","frozen","errors"):(args.out/name).mkdir(exist_ok=True)
    import torch
    from reference_qk_experiment import runtime_import
    from tics.native_assets import reuse_native_basis
    FoRIS=runtime_import(manifest)
    from icx.common import TimmDINOv3
    torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(manifest["GPU_memory_fraction"])
    started=time.monotonic();records=[];outcome="ALL40_COMPLETED";cost=None
    try:
        encoder=TimmDINOv3(mdir=manifest["DINO_model_root"]).cuda().eval().requires_grad_(False)
        with reuse_native_basis(FoRIS,manifest["projection_basis"]):host=FoRIS(encoder,image_size=1024,svd_components=500,tau=.6,mask_refiner="crf",resize_to_orig_size=False,device="cuda").eval().requires_grad_(False)
        with torch.inference_mode():
            for row in manifest["episodes"]:
                if shutil.disk_usage(args.out).free<manifest["reserve_bytes"]:raise RuntimeError("5GiB reserve violated")
                rec=infer_pair(host,row,manifest,args.out);records.append(rec)
                size=sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file())
                if size>manifest["max_output_bytes"]:raise RuntimeError("128MiB output bound exceeded; keep existing frozen cases, never delete evidence")
                if len(records)==8:
                    mean=sum(item["seconds"] for item in records)/8;remaining=2400-(time.monotonic()-started)
                    cost=dict(state="TECHNICAL8_ACTUAL_COST_GATE",actual_pair_mean_seconds=mean,remaining32_estimate_seconds=32*mean,remaining_registered_seconds=remaining,
                        can_fit=32*mean<=remaining,no_quality_or_CI_gate=True,first8_reused_in_same_invocation=True,not_tail_runtime_guarantee=True)
                    write(args.out/"first8_cost.json",cost)
                    if not cost["can_fit"]:outcome="RESOURCE_HOLD_AFTER_COMPLETE_TECHNICAL8";break
                if time.monotonic()-started>2400:raise TimeoutError("Finite full-shape budget exhausted")
        if len(records) not in (8,40):raise RuntimeError("Only complete fixed8 or mandatory40 cohorts may freeze")
        write(args.out/"execution.json",dict(state="PRO_PREDICTIONS_FROZEN_WHOLE_ACTUAL_COHORT",manifest_sha256=sha(args.manifest),source_hashes=manifest["source_hashes"],
            records=records,actual_cases=len(records),expected_cases=40,outcome=outcome,first8_cost=cost,seconds=time.monotonic()-started,QueryGT_opened=False,
            all_case_frozen_artifacts_preserved=True))
    except BaseException as error:
        write(args.out/"errors"/("error_"+str(time.time_ns())+".json"),dict(state="ERROR_PRO_INTERFACE_OR_RESOURCE",error=repr(error),traceback=traceback.format_exc(),completed_cases=len(records),scientific_negative=False));raise


def summarize(records):
    from tics.paired_frozen_stats import paired
    def compare(chosen,arm,base):return paired(chosen,lambda row:row["original_iu"][arm],lambda row:row["original_iu"][base],draws=2000)
    scopes={"all_actual":records}
    scopes.update({cohort:[row for row in records if row["cohort"]==cohort] for cohort in ("oldDEV40",)})
    results={}
    for scope,chosen in scopes.items():
        if not chosen:continue
        metrics={arm:compare(chosen,arm,"native") for arm in (*ARMS,"fixed_family_pixel_oracle")}
        groups={}
        for name,spec in GROUPS.items():
            comparisons={control:compare(chosen,spec["primary"],control) for control in spec["controls"]}
            strongest=max(spec["controls"],key=lambda arm:metrics[arm]["miou"])
            groups[name]=dict(primary=spec["primary"],all_controls=comparisons,strongest_same_information_control=strongest,
                vs_strongest_same_information=comparisons[strongest],vs_untouched_native=metrics[spec["primary"]],postlabel_strongest_is_comparator_not_method_selection=True)
        pre_refinement={arm:compare(chosen,"pre__"+arm,"pre__regional_local") for arm in BASE_ARMS}
        folds={}
        for fold in range(4):
            subset=[row for row in chosen if row["fold"]==fold]
            if subset:folds[str(fold)]={arm:compare(subset,arm,"native") for arm in ARMS}
        results[scope]=dict(cases=len(chosen),metrics=metrics,groups=groups,folds=folds,pre_refinement=pre_refinement)
    return results


def report(args):
    import numpy as np
    from PIL import Image
    sys.path.insert(0,str(ROOT))
    manifest=read(args.manifest);validate(manifest);frozen=read(args.out/"execution.json")
    if frozen["state"]!="PRO_PREDICTIONS_FROZEN_WHOLE_ACTUAL_COHORT" or frozen["manifest_sha256"]!=sha(args.manifest):raise RuntimeError("Whole actual prediction cohort must freeze before QueryGT")
    count=frozen["actual_cases"]
    if count not in (8,40) or [rec["case_id"] for rec in frozen["records"]]!=[row["case_id"] for row in manifest["episodes"][:count]]:raise RuntimeError("Unique physical case/QueryGT join mismatch")
    for rec in frozen["records"]:
        if sha(rec["prediction_path"])!=rec["prediction_sha256"]:raise RuntimeError("Frozen mask payload changed")
    scored=[]
    for rec in frozen["records"]:
        row=rec["row"];truth=np.asarray(Image.open(Path(manifest["annotation_root"])/Path(row["query"]).with_suffix(".png")))==row["c"]+1
        with np.load(rec["prediction_path"],allow_pickle=False) as package:
            if tuple(package["original_hw"])!=truth.shape:raise RuntimeError("Original mask/QueryGT shape differs")
            predictions={arm:np.unpackbits(package[arm])[:truth.size].reshape(truth.shape).astype(bool) for arm in ARMS};native=predictions["native"]
            IUs={};pixels={}
            for arm,pred in predictions.items():
                tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());IUs[arm]=[tp,tp+fp+fn]
                pixels[arm]=dict(TP=tp,FP=fp,FN=fn,recovered_native_FN=int((~native&pred&truth).sum()),lost_native_TP=int((native&~pred&truth).sum()),
                    deleted_native_FP=int((native&~pred&~truth).sum()),added_native_FP=int((~native&pred&~truth).sum()))
            for arm in BASE_ARMS:
                pre=np.unpackbits(package["pre__"+arm])[:truth.size].reshape(truth.shape).astype(bool)
                IUs["pre__"+arm]=[int((pre&truth).sum()),int((pre|truth).sum())]
            masks=np.stack(list(predictions.values()));inter=masks.all(0);union=masks.any(0);oracle=inter|(truth&(union&~inter))
            IUs["fixed_family_pixel_oracle"]=[int((oracle&truth).sum()),int((oracle|truth).sum())]
            diagnostics={}
            if not rec["SourceBG_empty_fallback"]:
                yy=np.arange(64)*truth.shape[0]//64;xx=np.arange(64)*truth.shape[1]//64;gt=truth[yy[:,None],xx[None,:]].reshape(-1)
                n64=native[(np.arange(64)*truth.shape[0]//64)[:,None],(np.arange(64)*truth.shape[1]//64)[None,:]].reshape(-1)
                for name in ("g","l","r","h","h_minus_l","r_top1"):
                    values=package["diagnostic__"+name].astype(np.float32)
                    diagnostics[name]={label:dict(patches=int(select.sum()),mean=float(values[select].mean()) if select.any() else None,
                        positive=int((select&(values>0)).sum())) for label,select in (("native_FN",~n64 & gt),("native_FP",n64 & ~gt),("true_FG",gt),("true_BG",~gt))}
                # GT-only diagnostic on the frozen64 partition, unrefined;
                # majority minimises pixel disagreement, not an IoU upperbound.
                labels=package["query_labels64"]
                mapped=labels[(np.arange(truth.shape[0])*64//truth.shape[0])[:,None],(np.arange(truth.shape[1])*64//truth.shape[1])[None,:]]
                region=np.zeros_like(truth)
                for label in np.unique(mapped):
                    select=mapped==label
                    if truth[select].sum()>select.sum()/2:region[select]=True
                IUs["GT_region64_majority_diagnostic"]=[int((region&truth).sum()),int((region|truth).sum())]
        scored.append(dict(**row,original_iu=IUs,pixels=pixels,diagnostic_margin_groups=diagnostics,SourceBG_empty_fallback=rec["SourceBG_empty_fallback"],seconds=rec["seconds"],encoding=rec["shared_region_encoder"]))
    results=summarize(scored)
    write(args.out/"scored_records.json",dict(state="REGIONAL_ORIGINAL_IU_AND_DELTAS",records=scored))
    write(args.out/"summary.json",dict(state="PRO_COHORT_SCORED",actual_cases=count,full40=count==40,results=results,primary_before_labels=manifest["primary"],
        exposed_DEV40=True,fresh_labels_used=False,first8_cost=frozen["first8_cost"],SourceBG_fallbacks=sum(rec["SourceBG_empty_fallback"] for rec in scored),
        oracle="GT restricted per-pixel chooser of produced masks; region64 majority is diagnostic not optimalIoU or information access proof",
        no_CI_or_negativefold_automatic_failure=True,no_newmethod_acceptance_claim=True,all_QueryGT_after_whole_actual_prediction_freeze=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare","cpu-preflight","run","report"):modes.add_argument("--"+name,action="store_true")
    oldroot=Path("/root/autodl-tmp/demo9_transductive_ics")
    parser.add_argument("--old-manifest",type=Path,default=oldroot/"results/sam3_query_exemplar_v1/manifest.json")
    parser.add_argument("--fixed8-manifest",type=Path,default=oldroot/"results/ten_direction_v1/manifest8.json")
    parser.add_argument("--foris-root",type=Path,default=Path("/root/autodl-tmp/demo8_local_verification/foris_source"))
    parser.add_argument("--demo4-root",type=Path,default=Path("/root/autodl-tmp/demo4"))
    parser.add_argument("--projection-basis",type=Path,default=oldroot/"results/native_runtime_v1/positional_basis.pt")
    parser.add_argument("--manifest",type=Path);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--python",default="/root/miniconda3/bin/python")
    args=parser.parse_args();server_only()
    if args.prepare:prepare(args)
    elif args.manifest is None:parser.error("--manifest required")
    elif args.cpu_preflight:cpu_preflight(args)
    elif args.run:run(args)
    else:report(args)


if __name__=="__main__":main()
