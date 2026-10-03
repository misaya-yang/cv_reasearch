#!/usr/bin/env python3
"""SERVER-only fixed native-host OR/AND control, no model or GPU execution.

Reads original-resolution FoRIS native masks from the completed B freeze and
visual SAM3 masks from the completed A freeze. Readout/text/working masks are
never used. Existing per-case native I/U must match exactly before aggregation.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zlib

CARD=[
    "Assumption: complementary errors of two frozen native hosts can be recovered by fixed mask union/intersection without a learned decision rule.",
    "Prediction derived from exposed DEV40 OR-native+4.926pp with crossing CI, not measured full-cohort evidence: oldCONF600 OR-SAM gain2..5pp, >=3 positive folds, paired lower bound>0. Every841 original native I/U must be exact for BOTH stored baselines.",
    "Match: retain OR/AND as strong same-information naive controls only; independent data and complete equal-cost production remain necessary, with no method/novelty claim.",
    "Mismatch: metadata/shape/SHA/native-IU mismatch stops and preserves errors; DEV-only gain or crossingCONF CI does not trigger expansion. No threshold/fitting/selector sweep. Pixel disagreement oracle is labelled GT arithmetic, not observable decision ability.",
]
ARMS=("sam","foris","OR","AND","disagreement_oracle")


def read(path):return json.loads(Path(path).read_text())
def records(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def key(row):return tuple(int(row[k]) for k in ("fold","e","c"))
def uid(row):return canonical({k:row[k] for k in ("fold","e","c","support","query")})
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def unique(rows,label):
    result={}
    for row in rows:
        k=key(row)
        if k in result:raise ValueError("Duplicate key in "+label+": "+str(k))
        result[k]=row
    return result
def same(row,other):
    if key(row)!=key(other) or any(row[n]!=other.get(n) for n in ("support","query")):raise ValueError("Exact key/SQ mismatch: "+str(key(row)))


def cohort(root,split,count,out,deadline):
    import numpy as np
    from PIL import Image
    from sam3_query_exemplar_probe import paired_all_arms
    a=root/"results/sam3_handover_v3"/("A_"+split);b=root/"results/sam3_handover_v3"/("B_"+split)
    prepared=read(root/"results/sam3_handover_v3"/("B_"+split+"_prepared.json"));manifest=read(prepared["manifest"])
    if sha(prepared["manifest"])!=prepared["manifest_sha256"] or len(manifest["episodes"])!=count:raise ValueError("Prepared full manifest drift")
    expected=unique(manifest["episodes"],"manifest");apred=unique(records(a/"predictions_shard0.jsonl"),"A predictions")
    ascored=unique(records(a/"episodes_shard0.jsonl"),"A scored");bscored=unique(records(b/"episodes.jsonl"),"B scored")
    owner=unique(records(prepared["baseline_records"]),"read-only FoRIS baseline")
    if sha(prepared["baseline_records"])!=prepared["baseline_records_sha256"]:raise ValueError("FoRIS source baseline drift")
    for label,index in (("A predictions",apred),("A scored",ascored),("B scored",bscored)):
        if len(index)!=count or set(index)!=set(expected):raise ValueError("Incomplete "+label)
    afreeze=read(a/"prediction_freeze_shard0.json");bfreeze=read(b/"freeze_report.json")
    if (afreeze["state"]!="PREDICTIONS_FROZEN" or afreeze["episodes"]!=count or afreeze["query_annotation_opened"] is not False or
        afreeze["manifest_sha256"]!=prepared["manifest_sha256"] or afreeze["predictions_jsonl_sha256"]!=sha(a/"predictions_shard0.jsonl")):
        raise ValueError("A complete prediction freeze mismatch")
    areport=read(a/"report_shard0.json");breport=read(b/"report.json")
    if (areport["state"]!="COMPLETED" or areport["scored_jsonl_sha256"]!=sha(a/"episodes_shard0.jsonl") or
        areport["prediction_freeze_sha256"]!=sha(a/"prediction_freeze_shard0.json") or breport["state"]!="COMPLETED"):
        raise ValueError("Completed baseline score provenance mismatch")
    if (bfreeze["state"]!="PREDICTIONS_FROZEN" or bfreeze["frozen_episodes"]!=count or bfreeze["cohort_episodes"]!=count or
        bfreeze["query_GT_pixels_opened"] is not False or bfreeze["contract_sha256"]!=prepared["contract_sha256"] or
        bfreeze["all_prediction_sha256"]!=breport["all_prediction_sha256"]):raise ValueError("B complete original-mask freeze mismatch")
    acache={asset["path"]:asset for asset in afreeze["prediction_files"] if asset["path"].startswith("masks/")}
    if len(acache)!=count:raise ValueError("A mask freeze scope differs")
    frozen=bfreeze["frozen_cases"]
    if len(frozen)!=count or [r["index"] for r in frozen]!=list(range(count)) or bfreeze["all_case_uids"]!=[uid(r) for r in manifest["episodes"]]:
        raise ValueError("B exact original frozen case order/UID mismatch")
    if canonical([(r["case_uid"],r["sha256"]) for r in frozen])!=bfreeze["all_prediction_sha256"]:raise ValueError("B global freeze digest mismatch")
    measured=[];stream_path=out/(split+"_episodes.jsonl")
    if stream_path.exists():raise ValueError("Preserve prior partial records; use fresh output, never silently rescore")
    with stream_path.open("x") as stream:
        for i,row in enumerate(manifest["episodes"]):
            if time.monotonic()>deadline:raise TimeoutError("Finite180s CPU task cap reached")
            k=key(row)
            for other in (apred[k],ascored[k],bscored[k],owner[k]):same(row,other)
            afile=a/apred[k]["prediction_file"];asset=acache[apred[k]["prediction_file"]]
            if afile.stat().st_size!=asset["bytes"] or sha(afile)!=asset["sha256"]:raise ValueError("SAM original visual bitmap changed")
            case_meta=frozen[i];bfile=Path(case_meta["path"])
            if case_meta["case_uid"]!=uid(row) or sha(bfile)!=case_meta["sha256"] or bscored[k]["predictions_sha256"]!=case_meta["sha256"]:
                raise ValueError("FoRIS original bitmap/UID provenance changed")
            with gzip.open(bfile,"rt") as src:case=json.load(src)
            same(row,case["row"])
            if (case["index"]!=i or case["case_uid"]!=uid(row) or case["contract_sha256"]!=prepared["contract_sha256"] or
                case["query_GT_pixels_opened"] is not False or case["actual_public_FoRIS"] is not True or case["actual_CRF"] is not True):
                raise ValueError("B native full-pipeline contract mismatch")
            packed=case["original_masks"]["native"]
            if packed["codec"]!="zlib_np_packbits_big" or list(packed["shape"])!=case["original_hw"]:raise ValueError("FoRIS original schema unavailable; no working-mask approximation")
            bits=zlib.decompress(base64.b64decode(packed["data"]));shape=list(packed["shape"]);pixels=shape[0]*shape[1]
            if sha_bytes(bits)!=packed["bits_sha256"] or len(bits)!=(pixels+7)//8:raise ValueError("FoRIS packed bits/shape corruption")
            fm=np.unpackbits(np.frombuffer(bits,dtype=np.uint8))[:pixels].reshape(shape).astype(bool)
            with np.load(afile,allow_pickle=False) as z:
                if list(z["shape"])!=shape or z["visual"].dtype!=np.uint8 or z["visual"].size!=(pixels+7)//8:raise ValueError("Two original host shapes/schema differ")
                sm=np.unpackbits(z["visual"])[:pixels].reshape(shape).astype(bool)
            if shape!=apred[k]["query_shape"]:raise ValueError("SAM declared geometry changed")
            with Image.open(Path(manifest["data_root"])/row["query"]) as rgb:
                if [rgb.height,rgb.width]!=shape:raise ValueError("Original RGB dimensions mismatch")
            truth=np.asarray(Image.open(Path(manifest["annotation_root"])/Path(row["query"]).with_suffix(".png")))==row["c"]+1
            if list(truth.shape)!=shape:raise ValueError("Original query GT shape mismatch")
            masks={"sam":sm,"foris":fm,"OR":sm|fm,"AND":sm&fm,"disagreement_oracle":(sm&fm)|(truth&(sm^fm))}
            iu={arm:[int((pred&truth).sum()),int((pred|truth).sum())] for arm,pred in masks.items()}
            if iu["sam"]!=ascored[k]["original_iu"]["visual"]:raise ValueError("SAM native original I/U mismatch; no tolerance")
            if iu["foris"]!=bscored[k]["original_iu"]["native"] or iu["foris"]!=owner[k]["original_iu"]["native"]:
                raise ValueError("FoRIS native original I/U mismatch; no tolerance")
            edits={}
            for control,baseline in (("sam",sm),("foris",fm)):
                edits[control]={arm:dict(removed_FP=int((baseline&~pred&~truth).sum()),lost_TP=int((baseline&~pred&truth).sum()),
                    recovered_FN=int((~baseline&pred&truth).sum()),added_FP=int((~baseline&pred&~truth).sum())) for arm,pred in masks.items()}
            rec={**row,"original_iu":iu,"edits":edits,"shape":shape,"SAM_native_IU_exact":True,"FoRIS_native_IU_exact":True,
                 "sam_bitmap_sha256":asset["sha256"],"foris_case_sha256":case_meta["sha256"]}
            stream.write(json.dumps(rec)+"\n");stream.flush();measured.append(rec)
            write(out/"status.json",dict(state="CPU_SCORING",split=split,episodes=i+1,expected=count,native_IU_both_exact=True))
            print(json.dumps(dict(state="NATIVE_PARITY_AND_FIXED_CONTROLS_SCORED",split=split,episode=i+1,expected=count)),flush=True)
    if len(measured)!=count or {r["fold"] for r in measured}!={0,1,2,3}:raise ValueError("Complete all-fold scoring required")
    original,draws,meta=paired_all_arms(measured,list(ARMS),2000)
    comparisons={}
    for arm in ("OR","AND","disagreement_oracle"):
        for control in ("sam","foris"):
            difference=draws[arm]-draws[control]
            comparisons[arm+"__minus__"+control]=dict(delta_pp=original[arm]-original[control],CI95=np.quantile(difference,[.025,.975]).tolist(),
                folds={str(f):class_miou([r for r in measured if r["fold"]==f],arm)-class_miou([r for r in measured if r["fold"]==f],control) for f in range(4)})
    difference=draws["disagreement_oracle"]-draws["OR"]
    comparisons["disagreement_oracle__minus__OR"]=dict(delta_pp=original["disagreement_oracle"]-original["OR"],CI95=np.quantile(difference,[.025,.975]).tolist())
    aggregate={control:{arm:{name:sum(r["edits"][control][arm][name] for r in measured) for name in ("removed_FP","lost_TP","recovered_FN","added_FP")}
        for arm in ARMS} for control in ("sam","foris")}
    result=dict(state="COMPLETED",split=split,episodes=count,class_sum_original_mIoU=original,comparisons=comparisons,bootstrap=meta,
        original_host_baseline_exact_cases=dict(SAM=count,FoRIS=count),original_masks_used=True,working_interpolation_used=False,
        edits=aggregate,scored_records=str(stream_path),scored_records_sha256=sha(stream_path),card=CARD,
        scope="Previously examined DEV241 / oldCONF600; fixed same-information naive control, no new method or independent confirmation",
        oracle_definition="(SAM AND FoRIS) OR [queryGT AND (SAM XOR FoRIS)]; restricted per-pixel choice between two fixed masks, no information-reachability proof",
        GPU_used=False,model_inference=False,readout_or_category_text_used=False,downloads=False)
    write(out/(split+"_report.json"),result);return result


def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def class_miou(recs,arm):
    acc={}
    for row in recs:
        value=acc.setdefault(row["c"],[0,0]);value[0]+=row["original_iu"][arm][0];value[1]+=row["original_iu"][arm][1]
    return 100*sum(i/max(u,1) for i,u in acc.values())/len(acc)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--budget-seconds",type=int,default=180);a=p.parse_args()
    if not sys.platform.startswith("linux") or not str(a.root.resolve()).startswith("/root/"):raise RuntimeError("SERVER-only numerical audit; no local execution")
    if not 30<=a.budget_seconds<=180:raise ValueError("One bounded CPU process30..180s required")
    if a.out.exists() and any(a.out.iterdir()):raise ValueError("Preserve existing outputs; use a fresh directory")
    a.out.mkdir(parents=True);write(a.out/"card.json",dict(card=CARD,arms=list(ARMS),source_sha256=sha(__file__),GPU_used=False))
    start=time.monotonic()
    try:
        reports={split:cohort(a.root.resolve(),split,count,a.out,start+a.budget_seconds) for split,count in (("dev",241),("confirm",600))}
        write(a.out/"status.json",dict(state="COMPLETED",episodes=841,elapsed_seconds=time.monotonic()-start,GPU_used=False,model_inference=False,server_left_running=True))
        print(json.dumps(dict(state="COMPLETED",elapsed_seconds=time.monotonic()-start,scores={k:v["class_sum_original_mIoU"] for k,v in reports.items()})),flush=True)
    except BaseException as exc:
        write(a.out/("error_%d.json"%time.time_ns()),dict(state="ERROR_PRESERVED_NO_APPROXIMATION",error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),GPU_used=False))
        raise


if __name__=="__main__":main()
