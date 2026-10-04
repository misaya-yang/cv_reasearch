#!/usr/bin/env python3
"""Finite full-public-FoRIS QK descriptor experiment, SERVER only.

Actual native last-block post-norm/post-RoPE Q/K, one B2 encoder pass per pair;
original Part1 then each fixed2048/3072 branch recomputes complete Part2--4.
First8 are a software/cost pilot, reused in the same40 invocation. No query
label pixels are opened until the entire actually completed cohort freezes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"scripts"
ARMS=("native","duplicate_z","constant_c","centroid_z","qonly","pre_part1_query_only","main","local_bank","shared_bank")
CHEAP=("constant_c","centroid_z","qonly","pre_part1_query_only")
DOC_SHA="6d480556bd2ff9080421d9ccbdb66b17caff1c58567d6cedb6c59669678da4d9"
CARD=[
    "Assumption: source-role conditioned native QK relative responses distinguish missed target appearances/confusers beyond final appearance and internal-coordinate readouts.",
    "Hypothesis, not measured prediction: main-native>=3pp and main-strongestcheap>=1pp in aggregate. Report paired intervals and everyfold as uncertainty/heterogeneity, never automatically fail for CI crossing0 or some negativefolds. Native cached/NoOp/capture differ in0pixels; matched local/shared3072 tests bank attribution.",
    "Match: retain the complete fixed mechanism only with full-mask improvements and intended nativeFN/FP context-direction changes; these40 are already-exposed DEV, not independent confirmation or acceptance evidence.",
    "Mismatch: source/API/nonfinite faults stop for repair; cost holds preserve the completed8 without lowering resolution or changing parameters. A cheap control explaining gain withdraws the corresponding QK-necessity claim; no layer/head/temperature/threshold sweep.",
]
HOST_DEFAULTS=dict(cluster_logsumexp_temp=.07,dino_bg_weight=.55,use_raw_target_clustering=True,use_raw_target_scoring=True,
    candidate_boost=.20,seed_cluster_boost=.25,semantic_disagreement_weight=.08,semantic_bg_coupling_weight=.05,
    semantic_cluster_fg_boost=.20,semantic_cluster_conflict_suppress=.18,semantic_penalty_uncertainty_power=1.5,
    semantic_penalty_max=.22,semantic_cluster_neg_cap=.12,enable_clustering=True)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w") as stream:json.dump(value,stream,indent=1,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def server_only():
    if not sys.platform.startswith("linux") or not str(ROOT).startswith("/root/"):raise RuntimeError("SERVER-only numerical/metadata/model work; local AST/source access only")


def pure(row):return {name:row[name] for name in ("fold","e","c","support","query")}
def key(row):return tuple(row[name] for name in ("fold","e","c","support","query"))
def stem(row):return "%d_%d_%d"%(row["fold"],row["e"],row["c"])


def runtime_import(manifest):
    sys.path[:0]=[manifest["foris_root"],str(ROOT),str(HERE),manifest["demo4_root"]]
    from models.foris import FoRIS
    if Path(inspect.getfile(FoRIS)).resolve()!=(Path(manifest["foris_root"])/"models/foris.py").resolve():raise RuntimeError("Wrong public FoRIS module imported")
    return FoRIS


def validate(manifest):
    for path,digest in manifest["source_hashes"].items():
        if not Path(path).is_file() or sha(path)!=digest:raise RuntimeError("Frozen source/input changed: "+path)
    for item in manifest["assets"]:
        stat=Path(item["path"]).stat()
        if stat.st_size!=item["size"] or stat.st_mtime_ns!=item["mtime_ns"]:raise RuntimeError("Existing asset identity changed: "+item["path"])


def prepare(args):
    from PIL import Image
    import numpy as np
    if args.out.exists() and any(args.out.iterdir()):raise RuntimeError("Fresh QK output required; preserve previous evidence")
    c2=read(args.c2_manifest);eight=read(args.fixed8_manifest)
    cases=c2["cases"];rows8=[pure(row) for row in eight["frozen_episodes"]]
    lookup={key(pure(case["row"])):case for case in cases}
    if len(cases)!=40 or len(lookup)!=40 or len(rows8)!=8 or any(sum(row["fold"]==fold for row in rows8)!=2 for fold in range(4)):raise RuntimeError("Exact oldDEV40 and balanced fixed8 required")
    ordered=[lookup[key(row)] for row in rows8]+[case for case in cases if key(pure(case["row"])) not in set(map(key,rows8))]
    rows=[pure(case["row"]) for case in ordered]
    sources=[Path(__file__),HERE/"complete_pair_evidence_extract.py",
        ROOT/"tics/reference_qk_descriptor.py",ROOT/"tics/native_assets.py",ROOT/"tics/paired_frozen_stats.py",
        HERE/"experiment_resource_guard.py",Path(args.demo4_root)/"icx/common.py",
        *[args.foris_root/name for name in ("models/foris.py","utils/data.py","utils/clustering.py","utils/refinement.py")]]
    hashes={str(path.resolve()):sha(path) for path in sources}
    for path in (args.c2_manifest,args.fixed8_manifest):hashes[str(path.resolve())]=sha(path)
    if args.method_doc:
        if sha(args.method_doc)!=DOC_SHA:raise RuntimeError("User's fixed QK design document changed")
        hashes[str(args.method_doc.resolve())]=DOC_SHA
    assets=[];baseline={};headers=[]
    for row,case in zip(rows,ordered):
        for role in ("support","query"):
            image=Path(c2["data_root"])/row[role];label=Path(c2["annotation_root"])/Path(row[role]).with_suffix(".png")
            with Image.open(image) as im:hw=[im.height,im.width]
            with Image.open(label) as im:lhw=[im.height,im.width]
            if hw!=lhw:raise RuntimeError("RGB/label header geometry differs")
            headers.append(dict(case=stem(row),role=role,original_hw=hw,query_label_pixels_opened=False))
            for path in (image,label):
                stat=path.stat();assets.append(dict(path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
        path=Path(case["foris_packet"])
        with np.load(path,allow_pickle=False) as packet:bits=packet["native"].copy()
        if bits.dtype!=np.uint8 or bits.size!=1024*1024//8:raise RuntimeError("Actual old full-public native1024 mask missing; no expected-IU substitute")
        baseline[stem(row)]=dict(path=str(path),sha256=sha(path),key="native",shape=[1024,1024],codec="np_packbits_big")
        hashes[str(path.resolve())]=baseline[stem(row)]["sha256"]
    for path in (args.projection_basis,Path("/root/demo4_cache/models/dinov3-vitl16-timm/config.json"),Path("/root/demo4_cache/models/dinov3-vitl16-timm/model.safetensors")):
        stat=path.stat();assets.append(dict(path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.out.parent).free<5*1024**3:raise RuntimeError("5GiB reserve required")
    manifest=dict(state="QK_METADATA_PREPARED_GPU_HELD",schema="reference_QK_full_public_v1",episodes=rows,expected_cases=40,pilot_cases=8,
        data_root=c2["data_root"],annotation_root=c2["annotation_root"],foris_root=str(args.foris_root),demo4_root=str(args.demo4_root),
        DINO_model_root="/root/demo4_cache/models/dinov3-vitl16-timm",
        projection_basis=str(args.projection_basis.resolve()),source_hashes=hashes,assets=assets,baselines=baseline,headers=headers,
        arms=list(ARMS),card=CARD,user_document_sha256=DOC_SHA,scope="old_exposed_DEV40",seed=0,
        recipe=dict(image_size=1024,svd_components=500,tau=.6,heads=16,head_width=64,prefix=5,last_block_index=23,
            dictionary_max64=True,assignment_threshold=.6,assignment_temperature=.07,attention_scale=.125,
            default_width=2048,bank_pair_width=3072,full_public_RGB_position=True,refiner="crf",original_mapping="bilinear>.5",
            insertion="after original Part1; complete Part2--4 recomputed; original basis1024 only",no_extra_encoder=True),
        GPU_total_cap_seconds=900,GPU_run_cap_seconds=870,CPU_report_cap_seconds=30,startup_reserve_seconds=30,total_cap_seconds=930,
        GPU_memory_fraction=.45,reserve_bytes=5*1024**3,max_output_bytes=64*1024**2,
        query_GT_after_whole_actual_cohort_freeze=True,first8_gate="Exact software/capture + actual cost only; continue40 without repeating8 if remaining budget fits",
        no_quality_or_CI_halt_on8=True,no_training_downloads_or_autoboot=True,method_quality_unmeasured=True)
    write(args.out/"manifest.json",manifest)
    print(json.dumps(dict(state=manifest["state"],cases=40,pilot8_balanced=True,GPU_started=False)),flush=True)


@contextmanager
def replace_instance(host,values):
    saved={name:(name in host.__dict__,host.__dict__.get(name)) for name in values}
    try:
        for name,fn in values.items():setattr(host,name,fn)
        yield
    finally:
        for name,(owned,old) in saved.items():setattr(host,name,old) if owned else delattr(host,name)


def public_pass(host,support,source_mask,query,*,cached=None,injection=None,pre_query=None):
    """Scoped instance tap/injection while literal public segment runs allparts."""
    import torch
    import torch.nn.functional as F
    if any(getattr(host,name)!=expected for name,expected in HOST_DEFAULTS.items()):raise RuntimeError("Public host hyperparameter defaults differ from fixed design")
    if any(getattr(host,name) is not None for name in ("_ref_images","_ref_masks","_tgt_image","_orig_tgt_size")):raise RuntimeError("Host must be idle; do not take foreign state")
    trace=dict(stage_audit={},encoder_calls=0)
    methods={name:getattr(host,name) for name in ("_extract_features","_part1_positional_debias","_part2_background_suppression", "_part3_clustering","_build_seed_cluster_prior","_part4_semantic_consistency_correction","_binarize_response")}
    def extract(imgs):
        if cached is None:
            out=methods["_extract_features"](imgs);trace.update(raw=out.detach().clone(),encoder_input=imgs.detach().clone());trace["encoder_calls"]+=1
            return out
        expected=cached["encoder_input"]
        if imgs.dtype!=expected.dtype or imgs.shape!=expected.shape or imgs.stride()!=expected.stride() or not torch.equal(imgs,expected):raise RuntimeError("Branch tried to encode a changed Source/Query input")
        trace["cache_hits"]=trace.get("cache_hits",0)+1;return cached["raw"]
    def part1(*values,**kwargs):
        incoming=values[0] if values else kwargs["fmaps_norm"]
        result=methods["_part1_positional_debias"](*values,**kwargs)
        trace.update(pre_part1=incoming.detach().clone(),part1=result.detach().clone())
        if cached is not None and not torch.equal(result,cached["part1"]):raise RuntimeError("Original Part1/condition changed acrossbranches")
        if injection is None:return result
        h,w=result.shape[-2:]
        combined=torch.stack((injection["source"].T.reshape(-1,h,w),injection["query"].T.reshape(-1,h,w)))[None]
        if combined.shape[:2]!=(1,2) or combined.shape[-2:]!=(h,w) or not torch.isfinite(combined).all():raise RuntimeError("Invalid post-Part1 full-pair feature injection")
        trace["stage_audit"]["injected_width"]=combined.shape[2]
        return combined
    def seed(*values,**kwargs):
        bound=inspect.signature(methods["_build_seed_cluster_prior"]).bind(*values,**kwargs).arguments
        scope=getattr(methods["_build_seed_cluster_prior"],"__func__",methods["_build_seed_cluster_prior"]).__globals__
        cluster=scope["agglomerative_clustering"];prototype=scope["compute_cluster_prototypes"]
        if host._tgt_image is None:raise RuntimeError("Public target state absent: RGB/position branch would be skipped")
        C=bound["tgt_feat"].shape[1];h,w=bound["h"],bound["w"]
        raw_tokens=None
        if pre_query is not None:raw_tokens=F.normalize(pre_query.reshape(C,-1).T,p=2,dim=1)
        def grouping(X,*args,**kw):
            trace["stage_audit"]["seed_clustering_width"]=X.shape[1]
            if X.shape!=(h*w,C+5):raise RuntimeError("Actual seed RGB/position clustering width invalid")
            if raw_tokens is not None:
                # Keep the literal colour/position suffix. Undo its joint L2
                # scale via the unit semantic-prefix norm, then replace ONLY
                # semantic grouping coordinates. Cross still uses native z.
                semantic_norm=X[:,:C].norm(dim=1,keepdim=True)
                if not (semantic_norm>0).all():raise RuntimeError("Zero native semantic grouping prefix")
                X=F.normalize(torch.cat((raw_tokens,X[:,C:]/semantic_norm),dim=1),p=2,dim=1)
            return cluster(X,*args,**kw)
        def prototypes(X,labels,K):
            if raw_tokens is not None:
                if X.shape!=raw_tokens.shape or not (torch.bincount(labels,minlength=K)>0).all():raise RuntimeError("Pre-Part1 intra prototype labels invalid")
                X=raw_tokens
            return prototype(X,labels,K)
        scope["agglomerative_clustering"]=grouping;scope["compute_cluster_prototypes"]=prototypes
        try:return methods["_build_seed_cluster_prior"](*values,**kwargs)
        finally:scope["agglomerative_clustering"]=cluster;scope["compute_cluster_prototypes"]=prototype
    def observed(name):
        fn=methods[name]
        def call(*values,**kwargs):
            bound=inspect.signature(fn).bind(*values,**kwargs).arguments;result=fn(*values,**kwargs)
            if name=="_part2_background_suppression":
                if result is None:raise RuntimeError("Original Source foreground domain invalid")
                trace["stage2_score"]=result[0].detach().clone()
                trace["stage_audit"].update(part2_width=bound["fmaps_norm"].shape[2],mu_fg_width=result[3].numel(),gated_query_width=result[4].shape[1])
            elif name=="_part3_clustering":trace["stage_audit"].update(part3_source_width=bound["ref_feats_raw"].shape[2],part3_query_width=bound["tgt_feat_raw"].shape[1])
            elif name=="_part4_semantic_consistency_correction":trace["stage_audit"].update(part4_gated_width=bound["tgt_feat"].shape[1],part4_recomputed=True)
            elif name=="_binarize_response":trace["score"]=bound["score_hw"].detach().clone()
            return result
        return call
    hooks={"_extract_features":extract,"_part1_positional_debias":part1,"_build_seed_cluster_prior":seed}
    hooks.update({name:observed(name) for name in methods if name not in hooks})
    try:
        host.set_reference(support,source_mask);host.set_target(query)
        trace["source_work_mask"]=host._ref_masks[0].detach().clone()
        trace["query_work_hw"]=tuple(host._tgt_image.shape[-2:])
        with replace_instance(host,hooks):prediction=host.segment().reshape(trace["query_work_hw"]).bool().clone()
        if cached is not None and (trace.get("cache_hits")!=1 or trace["encoder_calls"]!=0):raise RuntimeError("Actual cached branch unexpectedly re-encoded")
        return prediction,trace
    finally:host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None


def bundle_from_capture(host,trace,captured,prefix):
    import torch
    from tics.reference_qk_descriptor import build_qk_bundle
    z=trace["part1"][0].flatten(2).transpose(1,2).contiguous()
    q=captured["qkv"][23]["q"][:,:,prefix:];k=captured["qkv"][23]["k"][:,:,prefix:]
    scale=captured["sdpa_metadata"][23]["scale"]
    scale=1/math.sqrt(q.shape[-1]) if scale is None else float(scale)
    scope=getattr(host._locate_candidates,"__func__",host._locate_candidates).__globals__
    role=scope["downsample_mask"](trace["source_work_mask"][None,None],*trace["part1"].shape[-2:]).reshape(-1).bool()
    bundle=build_qk_bundle(z[0],z[1],q.float(),k.float(),role,attention_scale=scale,cluster_fn=scope["agglomerative_clustering"])
    bundle["audit"].update(captured_QK_native_dtype=str(q.dtype),readout_QK_dtype="torch.float32",captured_values_cast_to_FP32_explicit=True,
        cast_changed_dtype=q.dtype!=torch.float32,captured_QK_were_BF16=q.dtype==torch.bfloat16,
        readout_cast_not_encoder_precision_change=True,removed_actual_prefix_tokens=prefix,
        captured_physical_patches=q.shape[2],capture_stage="actual SDPA post-QKnorm/post-RoPE",extra_encoder_calls=0)
    return bundle,role,z


def infer_pair(host,row,manifest,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from complete_pair_evidence_extract import original_evidence
    support=Image.open(Path(manifest["data_root"])/row["support"]).convert("RGB")
    query=Image.open(Path(manifest["data_root"])/row["query"]).convert("RGB")
    source_mask=torch.from_numpy((np.asarray(Image.open(Path(manifest["annotation_root"])/Path(row["support"]).with_suffix(".png")))==row["c"]+1).copy())
    backbone=host.encoder.m;prefix=int(backbone.num_prefix_tokens)
    if len(backbone.blocks)!=24 or prefix!=5 or not isinstance(backbone.blocks[23].attn.q_norm,torch.nn.Identity) or not isinstance(backbone.blocks[23].attn.k_norm,torch.nn.Identity):raise RuntimeError("Actual fixed ViT-L24/prefix5/IdentityQKnorm required")
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with original_evidence(backbone,indices=(23,)) as captured:native,trace=public_pass(host,support,source_mask,query)
    if trace["encoder_calls"]!=1 or trace["raw"].shape!=(1,2,1024,64,64) or captured["qkv"][23]["q"].shape!=(2,16,4101,64):raise RuntimeError("Full4096-patch actualB2/capture contract failed")
    old=manifest["baselines"][stem(row)]
    if sha(old["path"])!=old["sha256"]:raise RuntimeError("Saved native baseline payload changed")
    with np.load(old["path"],allow_pickle=False) as packet:stored=np.unpackbits(packet["native"]).reshape(1024,1024).astype(bool)
    if not np.array_equal(stored,native.cpu().numpy()):raise RuntimeError("Actual complete native working mask differs from old full-public baseline")
    noop,no_trace=public_pass(host,support,source_mask,query,cached=trace)
    if not torch.equal(native,noop) or not torch.equal(trace["score"],no_trace["score"]):raise RuntimeError("Public cached/NoOp native mask or score differs")
    bundle,role,z=bundle_from_capture(host,trace,captured,prefix)
    del captured
    predictions={"native":native};audits={"native":trace["stage_audit"]};clocks={};arrays={};grid=(64,64)
    if bundle["state"]=="NATIVE_FALLBACK":
        for arm in ARMS[1:]:predictions[arm]=native;audits[arm]=dict(fallback_to_complete_native=True)
    else:
        for arm in ARMS[1:]:
            tick=time.monotonic();kwargs=dict(cached=trace)
            if arm=="pre_part1_query_only":kwargs["pre_query"]=trace["pre_part1"][0,1]
            else:kwargs["injection"]=bundle["features"][arm]
            pred,got=public_pass(host,support,source_mask,query,**kwargs)
            width=1024 if arm=="pre_part1_query_only" else (3072 if arm in ("local_bank","shared_bank") else 2048)
            if any(got["stage_audit"][field]!=width for field in ("part2_width","mu_fg_width","gated_query_width","part3_source_width","part3_query_width","part4_gated_width")) or got["stage_audit"]["seed_clustering_width"]!=width+5:raise RuntimeError("Branch reused stale width/Part2--4 or skippedRGB/position")
            predictions[arm]=pred;audits[arm]=got["stage_audit"]
            if arm=="main":arrays["main_stage2_minus_native"]=((got["stage2_score"]-trace["stage2_score"]).float().cpu().numpy())
            torch.cuda.synchronize();clocks[arm]=time.monotonic()-tick
            del got
        def margin(fields):
            sf=F.normalize(fields["source"][role].mean(0),dim=0);sb=F.normalize(fields["source"][~role].mean(0),dim=0)
            return fields["query"]@(sf-sb)
        z_margin=margin(dict(source=z[0],query=z[1]));c_margin=margin(bundle["mechanism"]["qk_context"])
        arrays.update(z_margin=z_margin.reshape(grid).float().cpu().numpy(),context_margin=c_margin.reshape(grid).float().cpu().numpy(),
            equal_weight_margin=((z_margin+c_margin)/2).reshape(grid).float().cpu().numpy(),
            centroid_context_margin=margin(bundle["mechanism"]["centroid_context"]).reshape(grid).float().cpu().numpy(),
            assignment_zero=np.packbits(bundle["mechanism"]["query_assignment_zero"].cpu().numpy()),
            FG_assignment_zero=np.packbits(bundle["mechanism"]["query_common_FG_assignment_zero"].cpu().numpy()))
    for arm,pred in predictions.items():
        original=F.interpolate(pred[None,None].float(),(query.height,query.width),mode="bilinear",align_corners=False)[0,0]>.5
        arrays[arm]=np.packbits(original.cpu().numpy().reshape(-1))
        arrays["grid__"+arm]=np.packbits(pred[::16,::16].cpu().numpy().reshape(-1))
    arrays["original_hw"]=np.array([query.height,query.width],np.int64)
    path=out/"masks"/(stem(row)+".npz");np.savez_compressed(path,**arrays)
    torch.cuda.synchronize()
    rec=dict(row=pure(row),prediction_path=str(path.resolve()),prediction_sha256=sha(path),original_hw=[query.height,query.width],
        method_state=bundle["state"],method_audit=bundle["audit"],branch_audit=audits,branch_seconds=clocks,
        native_saved_full_public_exact=True,native_cached_NoOp_exact=True,encoder_B2_calls=1,extra_encoder_calls=0,
        duplicate_final_differing_work_pixels=int((predictions["duplicate_z"]!=native).sum()),
        duplicate_isometry_not_assumed_mask_parity=True,native_candidate_height_axis_preserved=True,
        mechanism_margin_definition="Sourcehard-role unit-mean FG/BG template direction; diagnostic, not host hardBG or calibrated reliability",
        source_role_FG=int(role.sum()),query_GT_pixels_opened=False,elapsed_seconds=time.monotonic()-begin,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(out/"frozen"/(stem(row)+".json"),rec)
    print(json.dumps(dict(state="QK_PAIR_FROZEN_NO_QUERY_GT",case=stem(row),method_state=rec["method_state"],seconds=rec["elapsed_seconds"])),flush=True)
    return rec


def cpu_preflight(args):
    os.environ["CUDA_VISIBLE_DEVICES"]="";manifest=read(args.manifest);validate(manifest)
    sys.path[:0]=[str(ROOT),str(HERE),manifest["foris_root"]]
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from functools import partial
    from unittest.mock import patch
    from timm.models.eva import EvaBlock
    from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3
    from complete_pair_evidence_extract import original_evidence,cpu_check
    from tics.reference_qk_descriptor import _server_cpu_check
    torch.set_num_threads(1);FoRIS=runtime_import(manifest)
    core=_server_cpu_check();write(args.out/"core_cpu.json",core)
    cpu_check(args.out/"capture_cpu.json")
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.num_prefix_tokens=5
            self.blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(23)]+[EvaBlock(dim=1024,num_heads=16,mlp_ratio=1.,num_prefix_tokens=5,
                attn_type="eva",rotate_half=True,qkv_bias=True,norm_layer=partial(torch.nn.LayerNorm,eps=1e-5),attn_drop=0.,proj_drop=0.,drop_path=0.)])
            self.blocks[-1].attn.fused_attn=True
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.m=Backbone();self.calls=0
            self.register_buffer("projection",torch.randn((3,1024),generator=torch.Generator().manual_seed(8010)))
        def get_intermediate_layers(self,x,n=1,reshape=True):
            if n!=1 or not reshape:raise RuntimeError("Native n1 fixture only")
            self.calls+=1;v=F.avg_pool2d(x,16);tokens=v.flatten(2).transpose(1,2)@self.projection
            tokens=torch.cat([tokens.mean(1,keepdim=True).expand(-1,5,-1),tokens],dim=1)
            for block in self.m.blocks[:-1]:tokens=block(tokens)
            rope=RotaryEmbeddingDinoV3(dim=64,rotate_half=True).get_embed(shape=(4,4))
            tokens=self.m.blocks[-1](tokens,rope=rope)
            return [tokens[:,5:].transpose(1,2).reshape(2,1024,4,4)]
    cases=[]
    for seed in range(3):
        torch.manual_seed(9010+seed);enc=Encoder().eval()
        colours=torch.empty((64,64,3),dtype=torch.uint8)
        colours[:32]=torch.tensor([220-seed,40+seed,40],dtype=torch.uint8)
        colours[32:]=torch.tensor([40,220-seed,40+seed],dtype=torch.uint8)
        source=Image.fromarray(colours.numpy());query=source.copy()
        mask=torch.zeros((64,64),dtype=torch.bool);mask[16:48,16:48]=True
        with torch.inference_mode(),patch.object(FoRIS,"_build_positional_basis",lambda host,device:torch.eye(1024)[:,:500]):
            host=FoRIS(enc,image_size=64,svd_components=500,tau=.6,mask_refiner="bilinear",resize_to_orig_size=False,device="cpu").eval()
            with original_evidence(enc.m,indices=(23,)) as captured:native,trace=public_pass(host,source,mask,query)
            noop,no_trace=public_pass(host,source,mask,query,cached=trace)
            if not torch.equal(native,noop) or not torch.equal(trace["score"],no_trace["score"]):raise RuntimeError("Actual-public synthetic NoOp mismatch")
            bundle,_,_=bundle_from_capture(host,trace,captured,5)
            if bundle["state"]!="READY":raise RuntimeError("Synthetic same-image branch must exercise all actual2048/3072 source paths")
            if not all(torch.equal(bundle["features"]["local_bank"][role],bundle["features"]["shared_bank"][role]) for role in ("source","query")):raise RuntimeError("Actual same-image matched bank-pair identity failed")
            arms={}
            for arm in ARMS[1:]:
                kw=dict(cached=trace)
                if arm=="pre_part1_query_only":kw["pre_query"]=trace["pre_part1"][0,1]
                else:kw["injection"]=bundle["features"][arm]
                pred,got=public_pass(host,source,mask,query,**kw)
                width=1024 if arm=="pre_part1_query_only" else (3072 if arm in ("local_bank","shared_bank") else 2048)
                if got["stage_audit"]["part2_width"]!=width or got["stage_audit"]["part4_gated_width"]!=width or got["stage_audit"]["seed_clustering_width"]!=width+5:raise RuntimeError("Actual-public dynamic width/RGB branch fixture failed")
                arms[arm]=dict(width=width,work_shape=list(pred.shape),Part2_to_4_actual_source=True)
            if enc.calls!=1:raise RuntimeError("CPU complete-host branches re-encoded")
            cases.append(dict(seed=seed,native_NoOp_mask_and_score_bitexact=True,actual_encoder_calls=1,branches=arms))
    if torch.cuda.is_initialized():raise RuntimeError("SERVER CPU fixture initialized CUDA")
    # Benchmark the actual final report on a synthetic40 shaped frozen cohort,
    # not a model result or any real QueryGT read.
    import numpy as np
    from tics.paired_frozen_stats import paired
    synthetic=[dict(row=row,c=row["c"],support=row["support"],query=row["query"],IU={arm:[100,200] for arm in (*ARMS,"fixed_family_pixel_oracle")}) for row in manifest["episodes"]]
    began=time.monotonic()
    for arm in (*ARMS,"fixed_family_pixel_oracle"):paired(synthetic,lambda r,a=arm:r["IU"][a],lambda r:r["IU"]["native"])
    for arm in (*CHEAP,"shared_bank"):paired(synthetic,lambda r:r["IU"]["main" if arm!="shared_bank" else "local_bank"],lambda r,a=arm:r["IU"][a])
    for fold in range(4):
        subset=[row for row in synthetic if row["row"]["fold"]==fold]
        for arm in ARMS:paired(subset,lambda r,a=arm:r["IU"][a],lambda r:r["IU"]["native"])
    seconds=time.monotonic()-began
    if seconds>30:raise RuntimeError("Actual40-case/all51comparison2000-draw synthetic statistics exceeds frozen30s CPU cap")
    proof=dict(state="SERVER_REFERENCE_QK_CPU_PREFLIGHT_PASSED_GPU_UNTESTED",manifest_sha256=sha(args.manifest),source_hashes=manifest["source_hashes"],
        core_contract_count=len(core["checks"]),capture_contract_count=10,actual_full_public_synthetic_cases=cases,
        actual_host_source=True,pretrained_DINO_or_CRF_not_verified=True,CUDA_initialized=False,real_QueryGT_opened=False,
        full40_2000draw_stats_seconds=seconds,statistics_comparisons=51,statistics_input_synthetic=True)
    write(args.out/"cpu_preflight.json",proof);make_guard(args,manifest)
    print(json.dumps(dict(state=proof["state"],core_checks=len(core["checks"]),capture_cases=10,full_public_cases=3)),flush=True)


def make_guard(args,manifest):
    prerequisite=[dict(path=str(args.manifest.resolve()),sha256=sha(args.manifest)),dict(path=str((args.out/"cpu_preflight.json").resolve()),sha256=sha(args.out/"cpu_preflight.json"),json_equals=dict(state="SERVER_REFERENCE_QK_CPU_PREFLIGHT_PASSED_GPU_UNTESTED"))]
    prerequisite += [dict(path=path,sha256=digest) for path,digest in manifest["source_hashes"].items()]
    frozen=dict(path=str((args.out/"execution.json").resolve()),json_equals=dict(state="PREDICTIONS_FROZEN_WHOLE_ACTUAL_COHORT"))
    report=dict(path=str((args.out/"summary.json").resolve()),json_equals=dict(state="QK_FROZEN_COHORT_SCORED"))
    env=dict(DEMO9_REFERENCE_QK_ROOT_LAUNCH="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
        DEMO4_ROOT=manifest["demo4_root"],DEMO4_CACHE="/root/demo4_cache",DEMO4_GPU_FRAC=str(manifest["GPU_memory_fraction"]),
        PYTHONPATH=str(ROOT)+":"+str(HERE)+":"+manifest["foris_root"]+":"+manifest["demo4_root"]+":/root/demo4_cache/env")
    stages=[]
    for action,kind,cap,dependencies,product in (("run","gpu",870,[],frozen),("report","cpu",30,[frozen],report)):
        stages.append(dict(name="reference_QK_"+action,kind=kind,**(dict(role="finalize") if kind=="cpu" else {}),cwd=str(args.out.resolve()),timeout_seconds=cap,
            env=env,argv=[args.python,str(Path(__file__).resolve()),"--"+action,"--manifest",str(args.manifest.resolve()),"--out",str(args.out.resolve())],
            requires=[*prerequisite,*dependencies],code_files=[path for path in manifest["source_hashes"] if Path(path).suffix==".py"],produces=[product],success_checks=[product]))
    write(args.out/"guard_plan.json",dict(platform="autodl",cuda_python=args.python,stages=stages,total_timeout_budget_seconds=930,
        sum_stage_timeouts_seconds=900,startup_reserve_seconds=30,no_auto_boot=True,GPU_total_including_startup_cap900=True))


def run(args):
    if os.environ.get("DEMO9_REFERENCE_QK_ROOT_LAUNCH")!="1":raise RuntimeError("Explicit root finite guard required; no autoGPU")
    manifest=read(args.manifest);validate(manifest);proof=read(args.out/"cpu_preflight.json")
    if proof["state"]!="SERVER_REFERENCE_QK_CPU_PREFLIGHT_PASSED_GPU_UNTESTED" or proof["manifest_sha256"]!=sha(args.manifest):raise RuntimeError("Actual SERVER source/core/fullhost CPU receipt required")
    if (args.out/"execution.json").exists() or (args.out/"frozen").exists():raise RuntimeError("Fresh bounded QK invocation only; no oldqueue/prefix resumption")
    for name in ("masks","frozen","errors"):(args.out/name).mkdir(exist_ok=True)
    import torch
    from tics.native_assets import reuse_native_basis
    FoRIS=runtime_import(manifest)
    from icx.common import TimmDINOv3
    torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(manifest["GPU_memory_fraction"])
    begin=time.monotonic();records=[];outcome="ALL40_COMPLETED";cost=None
    try:
        encoder=TimmDINOv3(mdir=manifest["DINO_model_root"]).cuda().eval().requires_grad_(False)
        with reuse_native_basis(FoRIS,manifest["projection_basis"]):host=FoRIS(encoder,image_size=1024,svd_components=500,tau=.6,mask_refiner="crf",resize_to_orig_size=False,device="cuda").eval().requires_grad_(False)
        with torch.inference_mode():
            for row in manifest["episodes"]:
                if shutil.disk_usage(args.out).free<manifest["reserve_bytes"]:raise RuntimeError("5GiB reserve failed")
                if sum(path.stat().st_size for path in (args.out/"masks").glob("*.npz"))>manifest["max_output_bytes"]:raise RuntimeError("Bounded small-evidence storage exceeded")
                rec=infer_pair(host,row,manifest,args.out);records.append(rec)
                if len(records)==8:
                    mean=sum(item["elapsed_seconds"] for item in records)/8
                    remaining=870-(time.monotonic()-begin)
                    cost=dict(state="FIXED8_ACTUAL_COST_GATE",completed_cases=8,actual_pair_mean_seconds=mean,estimated_remaining32_seconds=32*mean,
                        remaining_registered_seconds=remaining,can_fit=32*mean<=remaining,no_quality_or_CI_gate=True,first8_not_reencoded=True,not_tail_runtime_guarantee=True)
                    write(args.out/"first8_cost.json",cost)
                    if not cost["can_fit"]:outcome="RESOURCE_HOLD_AFTER_COMPLETE_FIXED8";break
                if time.monotonic()-begin>870:raise TimeoutError("Finite full-shape run budget exhausted; no fake load or hidden truncation")
        if len(records) not in (8,40):raise RuntimeError("Only full fixed8 or40 may freeze; incomplete scientific cohort")
        write(args.out/"execution.json",dict(state="PREDICTIONS_FROZEN_WHOLE_ACTUAL_COHORT",manifest_sha256=sha(args.manifest),source_hashes=manifest["source_hashes"],
            records=records,actual_cases=len(records),expected_cases=40,outcome=outcome,first8_cost=cost,elapsed_seconds=time.monotonic()-begin,
            Query_GT_opened=False,partial8_is_technical_pilot_not_full_method_score=True))
    except BaseException as error:
        write(args.out/"errors"/("error_"+str(time.time_ns())+".json"),dict(state="ERROR_QK_INTERFACE_OR_FINITE_BUDGET",error=repr(error),traceback=traceback.format_exc(),completed_cases=len(records),scientific_negative=False));raise


def report(args):
    import numpy as np
    from PIL import Image
    sys.path.insert(0,str(ROOT));from tics.paired_frozen_stats import paired
    manifest=read(args.manifest);validate(manifest);frozen=read(args.out/"execution.json")
    if frozen["state"]!="PREDICTIONS_FROZEN_WHOLE_ACTUAL_COHORT" or frozen["manifest_sha256"]!=sha(args.manifest):raise RuntimeError("All actual cohort predictions must freeze before QueryGT")
    count=frozen["actual_cases"]
    if count not in (8,40) or len(frozen["records"])!=count or [key(rec["row"]) for rec in frozen["records"]]!=[key(row) for row in manifest["episodes"][:count]]:raise RuntimeError("Exact unique fixed8/40 prediction/QueryGT identity join failed")
    for rec in frozen["records"]:
        if sha(rec["prediction_path"])!=rec["prediction_sha256"]:raise RuntimeError("Frozen mask payload changed before labels")
    records=[]
    for rec in frozen["records"]:
        row=rec["row"];truth=np.asarray(Image.open(Path(manifest["annotation_root"])/Path(row["query"]).with_suffix(".png")))==row["c"]+1
        IU={};pixels={};mechanism=None
        with np.load(rec["prediction_path"],allow_pickle=False) as package:
            if tuple(package["original_hw"])!=truth.shape:raise RuntimeError("Original QueryGT/prediction shape differs")
            predictions={arm:np.unpackbits(package[arm])[:truth.size].reshape(truth.shape).astype(bool) for arm in ARMS}
            native=predictions["native"]
            for arm,pred in predictions.items():
                tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());IU[arm]=[tp,tp+fp+fn]
                pixels[arm]=dict(TP=tp,FP=fp,FN=fn,recovered_native_FN=int((~native&pred&truth).sum()),lost_native_TP=int((native&~pred&truth).sum()),
                    deleted_native_FP=int((native&~pred&~truth).sum()),added_native_FP=int((~native&pred&~truth).sum()))
            every=np.stack(list(predictions.values()));common=every.all(0);union=every.any(0)
            oracle=common|(truth&(union&~common));tp=int((oracle&truth).sum());IU["fixed_family_pixel_oracle"]=[tp,int((oracle|truth).sum())]
            if rec["method_state"]=="READY":
                y=(np.arange(64)*truth.shape[0]//64).astype(int);x=(np.arange(64)*truth.shape[1]//64).astype(int);gt=truth[y[:,None],x[None,:]]
                ng=np.unpackbits(package["grid__native"]).reshape(64,64).astype(bool);mg=np.unpackbits(package["grid__main"]).reshape(64,64).astype(bool)
                zero=np.unpackbits(package["assignment_zero"]).reshape(64,64).astype(bool);cm=package["context_margin"];zm=package["z_margin"];aug=package["equal_weight_margin"]
                def group(mask):
                    count=int(mask.sum())
                    return dict(patches=count,context_positive=int((mask&(cm>0)).sum()),context_negative=int((mask&(cm<0)).sum()),
                        appearance_negative_to_augmented_positive=int((mask&(zm<0)&(aug>0)).sum()),mean_context_margin=float(cm[mask].mean()) if count else None,
                        recovered_by_complete_main=int((mask & mg & gt).sum()),deleted_by_complete_main=int((mask & ~mg & ~gt).sum()))
                mechanism=dict(native_FN_assignment_zero=group(~ng & gt & zero),native_FN_assignment_nonzero=group(~ng & gt & ~zero),native_FP=group(ng & ~gt),
                    core_common_FG_exists=bool(rec["method_audit"]["common_FG_bins"]),patch_GT_nearest_from_original=True,source_template_margin_not_host_reliability=True)
        record=dict(row,c=row["c"],IU=IU,pixels=pixels,mechanism=mechanism,method_state=rec["method_state"],elapsed_seconds=rec["elapsed_seconds"])
        records.append(record)
    metrics={arm:paired(records,lambda row,a=arm:row["IU"][a],lambda row:row["IU"]["native"]) for arm in (*ARMS,"fixed_family_pixel_oracle")}
    controls={arm:paired(records,lambda row:row["IU"]["main"],lambda row,a=arm:row["IU"][a]) for arm in CHEAP}
    bank_pair=paired(records,lambda row:row["IU"]["local_bank"],lambda row:row["IU"]["shared_bank"])
    fold_metrics={str(fold):{arm:paired([row for row in records if row["fold"]==fold],lambda row,a=arm:row["IU"][a],lambda row:row["IU"]["native"]) for arm in ARMS} for fold in range(4)}
    write(args.out/"scored_records.json",dict(state="ALL_FROZEN_COHORT_ORIGINAL_IU",records=records))
    write(args.out/"summary.json",dict(state="QK_FROZEN_COHORT_SCORED",actual_cases=len(records),scope=manifest["scope"],seed=0,metrics=metrics,main_minus_cheap=controls,
        local_minus_shared_matched3072=bank_pair,folds=fold_metrics,fallback_cases=sum(row["method_state"]!="READY" for row in records),
        cost=frozen["first8_cost"],source_hashes=manifest["source_hashes"],source_document_sha256=DOC_SHA,
        oracle_definition="QueryGT-assisted restricted per-pixel choice among produced fixed9 masks; not information access proof or feature upperbound",
        no_method_acceptance_claim=True,full40_completed=len(records)==40,query_GT_opened_after_whole_actual_prediction_cohort=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare","cpu-preflight","run","report"):modes.add_argument("--"+name,action="store_true")
    parser.add_argument("--c2-manifest",type=Path,default=ROOT/"results/sam3_query_exemplar_v1/manifest.json")
    parser.add_argument("--fixed8-manifest",type=Path,default=ROOT/"results/ten_direction_v1/manifest8.json")
    parser.add_argument("--foris-root",type=Path,default=Path("/root/autodl-tmp/demo8_local_verification/foris_source"))
    parser.add_argument("--demo4-root",type=Path,default=Path("/root/autodl-tmp/demo4"))
    parser.add_argument("--projection-basis",type=Path,default=ROOT/"results/native_runtime_v1/positional_basis.pt")
    parser.add_argument("--method-doc",type=Path);parser.add_argument("--manifest",type=Path);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--python",default="/root/miniconda3/bin/python")
    args=parser.parse_args();server_only()
    if args.prepare:prepare(args)
    elif args.manifest is None:parser.error("--manifest required")
    elif args.cpu_preflight:cpu_preflight(args)
    elif args.run:run(args)
    else:report(args)


if __name__=="__main__":main()
