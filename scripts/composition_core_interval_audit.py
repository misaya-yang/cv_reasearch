"""Describe existing ridge edits relative to observable, protected query cores.

This is a post-seal development audit, not a new mask producer.  The cores use
the already fixed bounded-anchor confidence=.5 boundary; their truth purity is
only reported after all legal descriptors have been frozen.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(name,"2")
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent/"cv_data"
MAIN = REPO/"evidence/local/difficult_region_signal_20261010"
PILOT = DATA/"a/joint_role_pilot200_20261010"
COCO = REPO/"evidence/local/parallel_signal_20261010/02_task_identity/ridge_coco_validation"
OUT = REPO/"evidence/local/insid3_foris_generalization_20261010/core_interval"
sys.path[:0] = [str(REPO/"scripts"),str(REPO/"src")]
from insid3_factor_evidence import read,sha
from ics.official_data import array_hash


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def resize(value, shape, mode="bilinear"):
    tensor=torch.from_numpy(np.asarray(value,dtype=np.float64))[None,None]
    kw={} if mode=="nearest" else {"align_corners":False}
    return F.interpolate(tensor,tuple(shape),mode=mode,**kw)[0,0].numpy()


def unpack(value,shape):
    return np.unpackbits(value,count=int(np.prod(shape))).reshape(shape).astype(bool)


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def median(score,weight):
    score,weight=np.asarray(score).ravel(),np.asarray(weight).ravel().astype(float)
    positive=weight>0
    if not positive.any():return None
    s,w=score[positive],weight[positive]
    order=np.argsort(s,kind="stable")
    s,w=s[order],w[order]
    return float(s[np.searchsorted(np.cumsum(w),.5*w.sum(),side="left")])


def sources():
    rows=[]
    original={r["episode_id"]:r for r in lines(PILOT/"inference.jsonl")}
    score_index={r["episode_id"]:r for r in lines(DATA/"a/cache_mechanism_study_20261010/frontend_fields_episodes.jsonl")}
    raw={r["episode_id"]:r for r in read(MAIN/"raw/sealed.json")["receipts"]}
    candidate={r["episode_id"]:r for r in read(MAIN/"candidate_reference_v1/sealed.json")["receipts"]}
    for row in read(PILOT/"manifest.json"):
        eid=row["episode_id"];o=original[eid];r=raw[eid];c=candidate[eid];s=score_index[eid]
        rows.append(dict(row=row,field=str(MAIN/"raw/fields"/r["filename"]),field_sha256=r["sha256"],field_key="source_apd.ridge.global",
                         old=str(MAIN/"candidate_reference_v1/predictions"/c["filename"]),old_sha256=c["prediction_sha256"],
                         baseline=str(PILOT/"predictions"/o["filename"]),baseline_sha256=o["prediction_sha256"],
                         baseline_key="foris.crf",source_score=s["source_score_path"],source_score_sha256=s["source_score_sha256"],score_key="score",
                         calibration=c["calibration"]))
    source_index=read(COCO/"source_index.json")
    candidate={r["episode_id"]:r for r in read(COCO/"sealed.json")["receipts"]}
    for row in read(COCO/"manifest.json"):
        eid=row["episode_id"];s=source_index[eid];c=candidate[eid]
        rows.append(dict(row=row,field=str(COCO/"fields"/c["filename"]),field_sha256=c["fields_sha256"],field_key="ridge_global",
                         old=str(COCO/"predictions"/c["filename"]),old_sha256=c["prediction_sha256"],
                         baseline=s["baseline_path"],baseline_sha256=s["baseline_sha256"],baseline_key=s["baseline_key"].split("/",1)[1],
                         source_score=s["source_score"]["path"],source_score_sha256=s["source_score"]["sha256"],score_key=s["source_score"]["score_key"],
                         calibration=c["calibration"]))
    assert len(rows)==400
    return rows


def descriptor(source):
    for key in ("field","old","baseline","source_score"):
        assert sha(source[key])==source[key+"_sha256"]
    row=source["row"]
    with np.load(source["field"]) as z:r=z[source["field_key"]].copy()
    with np.load(source["source_score"]) as z:s=z[source["score_key"]].reshape(64,64).astype(float)
    with np.load(source["baseline"]) as z:
        b=unpack(z["cli1024/"+source["baseline_key"]],(1024,1024))
    assert r.shape==(128,128) and np.isfinite(r).all()
    span=float(s.max()-s.min())
    confidence=np.abs(2*resize((s-s.min())/max(span,1e-6),(1024,1024))-1) if span else np.zeros((1024,1024))
    cores={"foreground":b&(confidence>.5),"background":(~b)&(confidence>.5)}
    weights={k:v.reshape(128,8,128,8).sum((1,3)) for k,v in cores.items()}
    med={k:median(r,v) for k,v in weights.items()}
    threshold=float(source["calibration"]["threshold"])
    lo,hi=med["background"],med["foreground"]
    if lo is None or hi is None:condition="missing_core"
    elif lo>=hi:condition="inverted_or_equal"
    elif threshold<lo:condition="threshold_below_background_core"
    elif threshold>hi:condition="threshold_above_foreground_core"
    else:condition="threshold_within_core_interval"
    record=dict(episode_id=row["episode_id"],dataset=row["dataset"],
                core_median=med,core_pixels={k:int(v.sum()) for k,v in cores.items()},
                reference_threshold=threshold,reference_separation=float(source["calibration"]["separation"]),
                condition=condition,field_sha256=source["field_sha256"],source_score_sha256=source["source_score_sha256"])
    return record,r,cores


def prepare():
    assert not(OUT/"descriptors.json").exists(),"Preserve prior descriptor audit"
    torch.set_num_threads(2)
    ss=sources()
    write(OUT/"config.json",dict(state="PRE_GT_QUERY_CORE_DESCRIPTOR_PROTOCOL",source_sha256=sha(__file__),
         n=400,cohort="original Deep100/PACO100 signal pilot plus original COCO200 ridge transfer",
         core="same fixed bounded-anchor source confidence >.5 and full FoRIS CLI label",
         median="lower weighted median of saved128 ridge field, weights exact core-pixel mass of each8x8 footprint",
         purpose="Identify visible source-threshold contradictions with unchanged query cores before proposing the complete bounded composition",
         new_predictor=False,new_mask=False,new_encoder=False,query_GT=False))
    descriptors=[]
    for source in ss:
        d,_,_=descriptor(source);descriptors.append(d)
    write(OUT/"sources.json",ss)
    write(OUT/"descriptors.json",dict(state="SEALED_LEGAL_DESCRIPTORS",rows=descriptors,
        config_sha256=sha(OUT/"config.json"),sources_sha256=sha(OUT/"sources.json"),query_GT_reads=0))
    print(json.dumps({"state":"SEALED_LEGAL_DESCRIPTORS","n":len(descriptors)}))


def score():
    torch.set_num_threads(2)
    start=time.monotonic()
    cfg,desc=read(OUT/"config.json"),read(OUT/"descriptors.json")
    assert sha(__file__)==cfg["source_sha256"] and sha(OUT/"config.json")==desc["config_sha256"]
    assert sha(OUT/"sources.json")==desc["sources_sha256"]
    result=[]
    for source,record in zip(read(OUT/"sources.json"),desc["rows"]):
        actual,r,cores=descriptor(source)
        assert actual==record
        row=source["row"];shape=tuple(row["query_size_hw"])
        truth=(np.asarray(Image.open(row["query_mask_path"]).convert("L"))>0).astype(np.uint8)
        assert array_hash(truth)==row["query_mask_hash"]
        gt=resize(truth,shape,"nearest")>.5;gt_cli=resize(truth,(1024,1024),"nearest")>.5
        with np.load(source["baseline"]) as z:b=unpack(z["original/"+source["baseline_key"]],shape)
        with np.load(source["old"]) as z:old=unpack(z["anchor.global"],shape)
        r_native=resize(r,shape)
        added=old&~b;deleted=(~old)&b
        masks={"added_tp":added & gt,"added_fp":added & ~gt,"deleted_tp":deleted & gt,"deleted_fp":deleted & ~gt}
        lo,hi=record["core_median"]["background"],record["core_median"]["foreground"]
        buckets={}
        if lo is not None and hi is not None and lo<hi:
            for name,keep in (("below_background_median",r_native<lo),("between_core_medians",(r_native>=lo)&(r_native<=hi)),("above_foreground_median",r_native>hi)):
                buckets[name]={k:int((v&keep).sum()) for k,v in masks.items()}
            for key,mask in masks.items():assert sum(x[key] for x in buckets.values())==int(mask.sum())
        record=record.copy()
        record.update(edits={k:int(v.sum()) for k,v in masks.items()},buckets=buckets,
            core_fg_precision=float((cores["foreground"]&gt_cli).sum()/cores["foreground"].sum()) if cores["foreground"].any() else None,
            core_bg_true_background=float((cores["background"]&~gt_cli).sum()/cores["background"].sum()) if cores["background"].any() else None,
            base_iu=[int((b & gt).sum()),int((b | gt).sum())],old_iu=[int((old & gt).sum()),int((old | gt).sum())])
        result.append(record)
    summary={}
    for dataset in sorted({r["dataset"] for r in result}):
        sub=[r for r in result if r["dataset"]==dataset]
        groups={}
        for condition in sorted({r["condition"] for r in sub}):
            selected=[r for r in sub if r["condition"]==condition]
            groups[condition]=dict(n=len(selected),edits={k:sum(r["edits"][k] for r in selected) for k in ("added_tp","added_fp","deleted_tp","deleted_fp")},
                core_fg_precision_mean=float(np.mean([r["core_fg_precision"] for r in selected if r["core_fg_precision"] is not None])) if any(r["core_fg_precision"] is not None for r in selected) else None,
                core_bg_precision_mean=float(np.mean([r["core_bg_true_background"] for r in selected if r["core_bg_true_background"] is not None])) if any(r["core_bg_true_background"] is not None for r in selected) else None)
        buckets={}
        for name in ("below_background_median","between_core_medians","above_foreground_median"):
            buckets[name]={k:sum(r["buckets"].get(name,{}).get(k,0) for r in sub) for k in ("added_tp","added_fp","deleted_tp","deleted_fp")}
        summary[dataset]=dict(n=len(sub),conditions=groups,existing_edit_buckets=buckets)
    report=dict(state="COMPLETE_EXISTING_EDIT_CONDITION_AUDIT",summary=summary,rows=result,
                descriptor_sha256=sha(OUT/"descriptors.json"),seconds=time.monotonic()-start,
                caution="Counts describe old saved candidate edits only. No new candidate mask, IoU or guaranteed benefit is produced; original-pixel pooled counts are not macro mIoU contributions.")
    write(OUT/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("prepare","score"));args=parser.parse_args()
    {"prepare":prepare,"score":score}[args.phase]()
