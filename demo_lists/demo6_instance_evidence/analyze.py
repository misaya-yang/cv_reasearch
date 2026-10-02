"""CPU task diagnostics: category-matched identity errors and visibility errors."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image

DATA=Path("/root/autodl-tmp/demo1_sam/assets/datasets/DAVIS")


def auc(scores,labels):
    s=np.array(scores);y=np.array(labels)
    p,n=s[y==1],s[y==0]
    if not len(p) or not len(n):return None
    return float(((p[:,None]>n[None]).sum()+.5*(p[:,None]==n[None]).sum())/(len(p)*len(n)))


def metrics(rows,method):
    visible=[r for r in rows if r["presence"]==1]
    hits=[r["methods"][method]["point_hit"] for r in visible]
    groups=defaultdict(list)
    for r in rows:
        if r["presence"]!=-1:groups[(r["sequence"],r["object"])].append(r)
    aa=[];fpr=[]
    for rr in groups.values():
        scores=[r["methods"][method]["score"] for r in rr];labs=[r["presence"] for r in rr]
        a=auc(scores,labs)
        if a is not None:
            aa.append(a)
            scores=np.array(scores);labs=np.array(labs)
            # Descriptive ROC operating point. GT-selected threshold is NOT a deployment rule.
            threshold=np.quantile(scores[labs==1],.1)
            fpr.append(float((scores[labs==0]>=threshold).mean()))
    return dict(visible=len(visible),point_hit_fraction=float(np.mean(hits)) if hits else None,
                macro_object_visibility_auc=float(np.mean(aa)) if aa else None,
                macro_fpr_at_development_tpr90=float(np.mean(fpr)) if fpr else None,
                objects_with_absence=len(aa),roc_threshold_is_evaluation_only=True)


def main(args):
    cat=json.loads(Path(args.groups).read_text())["groups"]
    output={}
    for path in args.records:
        rows=json.loads(Path(path).read_text())
        same=[r for r in rows if r["sequence"] in cat and r["object"] in cat[r["sequence"]]["objects"]]
        names=list(rows[0]["methods"])
        image_cache={}
        per={}
        for method in names:
            result=metrics(same,method)
            wrong_same=[];wrong_other=[];floor=[];unobservable=[]
            for r in same:
                if r["presence"]!=1:continue
                key=(r["sequence"],r["frame"])
                if key not in image_cache:
                    image_cache[key]=np.asarray(Image.open(DATA/"Annotations/480p"/key[0]/(key[1]+".png")))
                gt=image_cache[key];px,py=r["methods"][method]["point"]
                label=int(gt[py,px]);ids=cat[r["sequence"]]["objects"]
                wrong_same.append(label in ids and label!=r["object"])
                wrong_other.append(label not in (0,255,r["object"]) and label not in ids)
                if "grid_center_target_available" in r:
                    floor.append(not r["grid_center_target_available"])
                    if r["grid_center_target_available"]:unobservable.append(r["methods"][method]["point_hit"])
            result.update(same_kind_switch_fraction=float(np.mean(wrong_same)),
                          other_kind_error_fraction=float(np.mean(wrong_other)),
                          grid_impossible_fraction=float(np.mean(floor)) if floor else None,
                          point_hit_when_grid_possible=float(np.mean(unobservable)) if unobservable else None)
            per[method]=result
        output[Path(path).parent.name]=dict(all={m:metrics(rows,m) for m in names},same_kind=per,
            group_definition="manually first-frame-verified development groups, not method-selected",
            same_kind_rows=len(same),scope="visible-instance identity; invisibility != physical nonexistence")
    Path(args.out).write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--records",nargs="+",required=True)
    p.add_argument("--groups",required=True);p.add_argument("--out",required=True);main(p.parse_args())
