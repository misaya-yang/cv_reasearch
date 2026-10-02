#!/usr/bin/env python3
"""Actual native-INSID3 E9 concept or E10 fixed-brightness diagnostic.

Native baseline ONLY: does not claim the seven candidate algorithms passed a
concept/stress test. One guarded PID, existing weights/data, typed basis reuse.
All unit predictions freeze before query annotation opens. No training, raw
feature cache, downloads, retries, or task-selection feedback.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from tics.native_assets import reuse_native_basis
from prepare_runtime_diagnostics import rgb_hash
from reference_feature_probe import device_gate,file_sha
from native_transport_experiment import source_receipt,require_sources
from global_representation_probe import summary,pixel_ledger


def brighten(array,factor=.75):
    if factor!=.75:raise ValueError("Only the preregistered brightness .75 is prepared")
    return np.rint(np.asarray(array,dtype=np.float32)*factor).clip(0,255).astype(np.uint8)


def bits(mask):
    a=mask.detach().cpu().numpy().astype(bool)
    return dict(shape=list(a.shape),bitorder="big",
        base64=base64.b64encode(np.packbits(a.reshape(-1),bitorder="big").tobytes()).decode())


def evaluate_case(predictions,semantic,c):
    """No values return to the already frozen native predictor."""
    first=next(iter(predictions.values()));shape=tuple(first.shape)
    if first.ndim!=2 or any(p.dtype!=torch.bool or tuple(p.shape)!=shape for p in predictions.values()):
        raise ValueError("Frozen predictions must be same-size bool masks")
    semantic=semantic.to(first.device)
    truth=F.interpolate((semantic==c+1).float()[None,None],shape,mode="nearest")[0,0]>.5
    labels=F.interpolate(semantic.float()[None,None],shape,mode="nearest")[0,0].long()
    base=predictions["insid3_native"];row=dict(iu={},original_iu={},pixels={},prediction_bits={},semantic_fp={})
    original_gt=semantic==c+1
    for name,pred in predictions.items():
        row["iu"][name]=[int((pred&truth).sum()),int((pred|truth).sum())]
        original=F.interpolate(pred.float()[None,None],tuple(semantic.shape),
            mode="bilinear",align_corners=False)[0,0]>.5
        row["original_iu"][name]=[int((original&original_gt).sum()),int((original|original_gt).sum())]
        row["pixels"][name]=pixel_ledger(base,pred,truth)
        row["prediction_bits"][name]=bits(pred)
        values,counts=torch.unique(labels[pred&~truth],return_counts=True)
        hist={str(int(v)):int(n) for v,n in zip(values,counts)}
        row["semantic_fp"][name]=dict(label_histogram=hist,
            unannotated_background_pixels=hist.get("0",0),person_pixels=hist.get("1",0),
            qualification="Exclusive semantic PNG labels; label0 does not identify a physical object")
    return row


def pair_diagnostic(rows,predictions,semantic,targets):
    scores={}
    for i,c in enumerate(targets):
        pred=predictions[i]["insid3_native"];own=rows[i]["iu"]["insid3_native"]
        other=targets[1-i]
        truth=F.interpolate((semantic.to(pred.device)==other+1).float()[None,None],
            tuple(pred.shape),mode="nearest")[0,0]>.5
        cross=float((pred&truth).sum())/max(float((pred|truth).sum()),1)
        scores[str(c)]=dict(own_iou=own[0]/max(own[1],1),cross_concept_iou=cross)
    return dict(changed_prediction_pixels=int((predictions[0]["insid3_native"]!=predictions[1]["insid3_native"]).sum()),
        native_task_scores=scores,both_prefer_own_target=all(v["own_iou"]>v["cross_concept_iou"] for v in scores.values()),
        qualification="Native baseline diagnostic only; changed predictions are not candidate-method success")


def verify_manifest(path):
    doc=json.loads(path.read_text())
    if doc.get("state")!="PREPARED_DIAGNOSTICS" or doc.get("kind") not in ("concepts","brightness"):
        raise ValueError("A nonempty CPU-prepared diagnostic manifest is required")
    if not doc.get("records") or len(doc["records"])>5 or doc.get("predictions_cap")>10 or doc.get("brightness_factor")!=.75:
        raise ValueError("Bounded frozen diagnostic dimensions differ")
    for asset in doc.get("assets",[]):
        stat=Path(asset["path"]).stat()
        if stat.st_size!=asset["size"] or stat.st_mtime_ns!=asset["mtime_ns"]:
            raise ValueError("Prepared immutable asset changed: "+asset["path"])
    for row in doc["records"]:
        expected=2 if doc["kind"]=="concepts" else 1
        if len(row["targets"])!=expected or len(set(row["targets"]))!=expected:
            raise ValueError("Target pair contract differs")
    return doc


def self_check():
    torch.set_num_threads(1)
    sem=torch.ones(8,12,dtype=torch.long);sem[4:]=37
    pa=torch.zeros(8,8,dtype=torch.bool);pa[4:]=True
    pb=~pa
    ra=evaluate_case({"insid3_native":pa},sem,36)
    rb=evaluate_case({"insid3_native":pb},sem,0)
    assert ra["iu"]["insid3_native"]==[32,32]
    assert ra["original_iu"]["insid3_native"]==[48,48]
    diag=pair_diagnostic([ra,rb],[{"insid3_native":pa},{"insid3_native":pb}],sem,[36,0])
    assert diag["changed_prediction_pixels"]==64 and diag["both_prefer_own_target"]
    rgb=np.array([0,128,255],dtype=np.uint8)
    assert brighten(rgb).tolist()==[0,96,191] and np.array_equal(rgb,[0,128,255])
    original={k:v.clone() for k,v in {"insid3_native":pa,"insid3_brightness075":pb}.items()}
    output=evaluate_case(original,sem,36)
    assert all(torch.equal(original[k],v) for k,v in {"insid3_native":pa,"insid3_brightness075":pb}.items())
    assert output["pixels"]["insid3_brightness075"]["lost_tp"]==32
    print(json.dumps(dict(state="CPU_DIAGNOSTIC_CHECK_PASSED",native_original_IU_separate=True,
        concept_score_not_just_response=True,brightness_fixed=True,semantic_FP_counts=True,
        masks_frozen=True,GPU_calls=0,real_native_method_evaluated=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-manifest",type=Path)
    p.add_argument("--source-guard",type=Path)
    p.add_argument("--prepared-root",type=Path,default=Path("/root/autodl-tmp/demo9"))
    p.add_argument("--projection-basis",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--allow-gpu",action="store_true")
    p.add_argument("--resource-guard-state",type=Path)
    p.add_argument("--self-check",action="store_true")
    args=p.parse_args()
    if args.self_check:self_check();return
    if not all((args.prepared_manifest,args.source_guard,args.out)):
        p.error("--prepared-manifest --source-guard --out required")
    if args.out.exists() and any(args.out.iterdir()):raise ValueError("Fresh output directory required")
    args.out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    report=dict(state="PREPARING",records=[],pair_diagnostics=[],dataset="COCO-20i",
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope="Actual native baseline-only diagnostic; no candidate-method, all-fold or cross-dataset success claim",
        pixel_ledger_reference="insid3_native for the corresponding concept / identity brightness task",
        source_sha256={str(f):file_sha(f) for f in (Path(__file__),HERE/"prepare_runtime_diagnostics.py",
            HERE.parent/"tics/native_assets.py")})
    def save():
        report["class_miou"]=summary(report["records"])
        report["elapsed_concurrent_s"]=time.monotonic()-started
        tmp=args.out/"report.tmp";tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(args.out/"report.json")
    save()
    try:
        prepared=verify_manifest(args.prepared_manifest);verified=source_receipt(args.source_guard)
        if str(Path(__file__).resolve()) not in verified:raise ValueError("Diagnostic executable missing from prepared source guard")
        report.update(kind=prepared["kind"],fold=prepared["fold"],seed=0,
            prepared_manifest_sha256=file_sha(args.prepared_manifest),
            source_guard_sha256=file_sha(args.source_guard),
            fixture_query_presence_exposure=prepared["query_presence_used_only_to_construct_fixture"],
            frozen_units=prepared["records"],predictions_cap=prepared["predictions_cap"])
        save()
        device=device_gate("cuda",args.allow_gpu,args.resource_guard_state)
        torch.set_num_threads(2);os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        sys.path.insert(0,str(args.prepared_root/"scripts"));import _paths
        sys.path.insert(0,_paths.DEMO4)
        from icx.common import build_model,TimmDINOv3
        from models.insid3 import INSID3
        from utils.data import load_image,load_mask
        require_sources(verified,[build_model,TimmDINOv3,INSID3,load_image,load_mask])
        if Path(_paths.COCO_ANN).resolve()!=Path(prepared["official_mask_root"]).resolve():
            raise ValueError("Official semantic mask root differs from prepared fixture")
        with torch.no_grad(),reuse_native_basis(INSID3,args.projection_basis) as basis_receipt:
            model=build_model().eval().requires_grad_(False)
        report["basis_receipt"]=basis_receipt;report["state"]="RUNNING";save()
        data=Path(prepared["data_root"]);annotations=Path(prepared["official_mask_root"])
        original_extract=model._extract_features;calls=[]
        def counted(this,images):
            calls.append(dict(images=int(images.shape[1]),ordered_native_pair=True))
            return original_extract(images)
        model._extract_features=MethodType(counted,model)
        try:
            with torch.inference_mode():
                for unit in prepared["records"]:
                    if time.monotonic()-started>300:raise TimeoutError("Fixed5min diagnostic wall cap")
                    torch.cuda.reset_peak_memory_stats();tick=time.monotonic();before=len(calls)
                    support_rgb=np.asarray(Image.open(data/unit["support"]).convert("RGB")).copy()
                    query_rgb=np.asarray(Image.open(data/unit["query"]).convert("RGB")).copy()
                    if rgb_hash(support_rgb)!=unit["support_rgb_sha256"] or rgb_hash(query_rgb)!=unit["query_rgb_sha256"]:
                        raise ValueError("Prepared same-RGB fixture hash changed")
                    s=load_image(Image.fromarray(support_rgb),model._transform,device)[0]
                    t=load_image(Image.fromarray(query_rgb),model._transform,device)[0]
                    support_sem=torch.from_numpy(np.asarray(Image.open(annotations/Path(unit["support"]).with_suffix(".png"))).copy())
                    frozen=[]
                    for c in unit["targets"]:
                        sm=load_mask(support_sem==c+1,1024,device)
                        if not sm.any():raise ValueError("Prepared support target disappeared")
                        native=model.predict_mask(s,sm,t).reshape(1024,1024).bool().detach().clone()
                        predictions={"insid3_native":native}
                        if prepared["kind"]=="brightness":
                            changed=brighten(query_rgb)
                            dark=load_image(Image.fromarray(changed),model._transform,device)[0]
                            predictions["insid3_brightness075"]=model.predict_mask(s,sm,dark).reshape(1024,1024).bool().detach().clone()
                            del dark
                        frozen.append(predictions)
                    torch.cuda.synchronize();inference_seconds=time.monotonic()-tick
                    # Every concept/brightness output for this unit is frozen NOW.
                    semantic=torch.from_numpy(np.asarray(Image.open(annotations/Path(unit["query"]).with_suffix(".png"))).copy())
                    unit_rows=[]
                    for i,c in enumerate(unit["targets"]):
                        row=evaluate_case(frozen[i],semantic,c)
                        row.update(e=unit["e"],c=c,fold=prepared["fold"],dataset="COCO-20i",
                            support=unit["support"],query=unit["query"],roles_photo_ids=unit["roles_photo_ids"],
                            support_rgb_sha256=unit["support_rgb_sha256"],query_rgb_sha256=unit["query_rgb_sha256"],
                            condition="concept_mask_switch" if prepared["kind"]=="concepts" else "query_brightness075",
                            outputs_frozen_before_query_gt=True,
                            cost=dict(unit_id=unit["pair_id"],shared_unit_cost=True,inference_seconds=inference_seconds,
                                encoder_calls=len(calls)-before,native_pair_image_forwards=2*(len(calls)-before),
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                        if prepared["kind"]=="concepts":row.update(pair_id=unit["pair_id"],task_tag="concept_"+str(c))
                        report["records"].append(row);unit_rows.append(row)
                    if prepared["kind"]=="concepts":
                        report["pair_diagnostics"].append(dict(pair_id=unit["pair_id"],
                            **pair_diagnostic(unit_rows,frozen,semantic,unit["targets"])))
                    report["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
                    report["peak_reserved_bytes"]=torch.cuda.max_memory_reserved();save()
                    print(json.dumps(dict(kind=prepared["kind"],unit=unit["pair_id"],
                        completed_records=len(report["records"]),class_miou=report["class_miou"])),flush=True)
                    del s,t,support_sem,frozen,semantic,unit_rows,native,predictions,sm
            report.update(state="COMPLETED",scientific_outcome="Native-only diagnostic, not novel method efficacy")
            save()
        finally:
            del model._extract_features
    except BaseException as error:
        report.update(state="ERROR",error=repr(error),traceback=traceback.format_exc(),stage_fatal=True)
        save();raise


if __name__=="__main__":main()
