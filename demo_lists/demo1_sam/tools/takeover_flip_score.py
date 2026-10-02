"""Fixed, GT-free flip ranking rules; GT is used only after selections are made."""
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/quality_mechanisms"))
from diagnose_oracle_gap import iou,inner_boundary


def match_and_select(masks,scores,flipped_masks,flipped_scores):
    # Only the official multimask tokens 1..3 participate; never add token0.
    matrix=np.array([[iou(a,b) for b in flipped_masks[1:]] for a in masks[1:]])
    permutations=list(itertools.permutations(range(3)))
    matched=max(permutations,key=lambda p:sum(matrix[j,p[j]] for j in range(3)))
    consistency=np.array([matrix[j,matched[j]] for j in range(3)])
    view_scores=np.array([flipped_scores[1+matched[j]] for j in range(3)])
    selections={"original_iou_head":int(scores[1:].argmax())+1,
                "matched_mean_iou_head":int(((scores[1:]+view_scores)/2).argmax())+1,
                "matched_consistency_only":int(consistency.argmax())+1}
    return selections,matrix,matched,consistency


def bootstrap_image_groups(rows,key,repetitions=2000):
    ids=sorted({r["image_id"] for r in rows})
    sums=np.array([sum(r[key] for r in rows if r["image_id"]==i) for i in ids])
    counts=np.array([sum(r["image_id"]==i for r in rows) for i in ids])
    rng=np.random.default_rng(2027)
    draws=rng.integers(len(ids),size=(repetitions,len(ids)))
    return np.quantile(sums[draws].sum(-1)/counts[draws].sum(-1),[.025,.975]).tolist()


def run(original,flipped,output):
    if output.exists():raise FileExistsError(output)
    source=json.loads((original/"report.json").read_text())
    view=json.loads((flipped/"report.json").read_text())
    assert [r["image_id"] for r in source["images"]]==[r["image_id"] for r in view["images"]]
    num_images=len(source["images"])
    num_objects=sum(len(r["prompt_metadata"]) for r in source["records"] if r["method"]=="official" and r["regime"]=="central")
    report={"status":"RUNNING","started_at_utc":datetime.now(timezone.utc).isoformat(),
            "protocol":f"{num_images} fixed images/{num_objects} original objects; only horizontal flip, independent original prompts, no extra clicks; 1..3 candidates, matching over all 6 permutations; fixed rules with no tuning/training",
            "rules_fixed_before_view_results":["original_iou_head","matched_mean_iou_head","matched_consistency_only"],
            "scope":"Fixed-rule development or prospective cohort diagnosis, not SOTA; view encoder+decoder cost is extra",
            "rows":[],"summary":{}}
    for record in source["records"]:
        if record["method"]!="official":continue
        other=next(r for r in view["records"] if r["image_id"]==record["image_id"] and r["regime"]==record["regime"])
        with np.load(original/record["npz"],allow_pickle=False) as a,np.load(flipped/other["npz"],allow_pickle=False) as b:
            assert np.array_equal(a["annotation_ids"],b["annotation_ids"])
            assert np.array_equal(a["gt"],b["gt"][:,:,::-1])
            masks=a["masks"].copy(); fm=b["masks"][:,:,:,::-1].copy()
            scores=a["iou_prediction"].copy(); fs=b["iou_prediction"].copy();gt=a["gt"].copy()
            radius=max(1,int(round(.02*np.hypot(*gt.shape[-2:]))))
            for j in range(len(gt)):
                selected,matrix,matched,consistency=match_and_select(masks[j],scores[j],fm[j],fs[j])
                values=[iou(m,gt[j]) for m in masks[j]]
                boundary=inner_boundary(gt[j],radius)
                boundaries=[iou(inner_boundary(m,radius),boundary) for m in masks[j]]
                baseline=selected["original_iou_head"]
                row={"image_id":record["image_id"],"annotation_id":int(a["annotation_ids"][j]),"regime":record["regime"],
                     "selections":selected,"iou_per_mask":values,"boundary_iou_per_mask":boundaries,
                     "matching":list(matched),"matching_iou_matrix":matrix.tolist(),"consistency":consistency.tolist(),
                     "oracle_1to3_iou":max(values[1:])}
                for name,index in selected.items():
                    row[name+"_iou"]=values[index]
                    row[name+"_boundary_iou"]=boundaries[index]
                    row[name+"_delta_iou"]=values[index]-values[baseline]
                    row[name+"_delta_boundary_iou"]=boundaries[index]-boundaries[baseline]
                report["rows"].append(row)
    for regime in ("central","near_boundary","box"):
        rows=[r for r in report["rows"] if r["regime"]==regime]
        report["summary"][regime]={"prompts":len(rows),"oracle_1to3_mean_iou":float(np.mean([r["oracle_1to3_iou"] for r in rows])),"policies":{}}
        for name in report["rules_fixed_before_view_results"]:
            report["summary"][regime]["policies"][name]={
                "mean_iou":float(np.mean([r[name+"_iou"] for r in rows])),
                "mean_boundary_iou":float(np.mean([r[name+"_boundary_iou"] for r in rows])),
                "mean_delta_iou":float(np.mean([r[name+"_delta_iou"] for r in rows])),
                "image_cluster_delta_iou_95":bootstrap_image_groups(rows,name+"_delta_iou"),
                "image_cluster_delta_boundary_iou_95":bootstrap_image_groups(rows,name+"_delta_boundary_iou"),
                "improved":sum(r[name+"_delta_iou"]>0 for r in rows),"worsened":sum(r[name+"_delta_iou"]<0 for r in rows)}
    report["status"]="COMPLETED_DEVELOPMENT_DIAGNOSIS"
    report["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["summary"],indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-run",type=Path,required=True)
    p.add_argument("--flipped-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.original_run,a.flipped_run,a.output)
