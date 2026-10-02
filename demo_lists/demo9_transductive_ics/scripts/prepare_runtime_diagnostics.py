#!/usr/bin/env python3
"""CPU-only E9/E10 fixture preparation from existing official first10 assets.

No torch/model/GPU import. Concept availability uses semantic class presence in
the already frozen RGB pairs, not model outcomes, IoU or difficulty. That query
presence exposure is disclosed: this is a constructed diagnostic, not a blind
generalization cohort. No missing fixture is replaced by a new/favorable image.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image


def rgb_hash(array):
    a=np.ascontiguousarray(array,dtype=np.uint8)
    return hashlib.sha256(str(tuple(a.shape)).encode()+a.tobytes()).hexdigest()


def read_pair(row,data,annotation):
    values={}
    for role in ("support","query"):
        image=data/row[role];mask=annotation/Path(row[role]).with_suffix(".png")
        with Image.open(image) as value:rgb=np.asarray(value.convert("RGB")).copy()
        with Image.open(mask) as value:labels=np.asarray(value).copy()
        if labels.ndim!=2 or rgb.shape[:2]!=labels.shape:
            raise ValueError("Semantic PNG must be indexed HxW and match RGB")
        classes=sorted(int(v)-1 for v in np.unique(labels) if 1<=int(v)<=80)
        values[role]=dict(path=str(image),mask_path=str(mask),shape=list(rgb.shape[:2]),
            rgb_sha256=rgb_hash(rgb),classes=classes)
    return values


def prepare(source_path,kind,limit,data):
    source=json.loads(source_path.read_text())
    if source.get("state")!="PREPARED_ASSETS" or source.get("seed")!=0 or not source.get("frozen_episodes"):
        raise ValueError("Existing fixed standard PREPARED_ASSETS manifest required")
    if not 1<=limit<=5:raise ValueError("Bounded diagnostic limit1..5 units (<=10 native predictions)")
    for asset in source.get("assets",[]):
        stat=Path(asset["path"]).stat()
        if stat.st_size!=asset["size"] or stat.st_mtime_ns!=asset["mtime_ns"]:
            raise ValueError("Source prepared asset changed: "+asset["path"])
    annotation=Path(source["official_mask_root"])
    records=[];excluded=[]
    # Preserve the existing first10 order; never expand its cohort.
    for row in source["frozen_episodes"][:10]:
        info=read_pair(row,data,annotation)
        original=int(row["c"]);shared=sorted(set(info["support"]["classes"])&set(info["query"]["classes"]))
        if original not in shared:
            excluded.append(dict(e=row["e"],reason="Original class absent in at least one official semantic PNG"))
            continue
        targets=[original]
        if kind=="concepts":
            alternatives=[c for c in shared if c!=original]
            if not alternatives:
                excluded.append(dict(e=row["e"],reason="No second co-occurring annotated concept"))
                continue
            targets.append(alternatives[0])
        records.append(dict(e=int(row["e"]),original_c=original,support=row["support"],
            query=row["query"],targets=targets,pair_id=("concept_" if kind=="concepts" else "brightness_")+str(row["e"]),
            support_rgb_sha256=info["support"]["rgb_sha256"],
            query_rgb_sha256=info["query"]["rgb_sha256"],
            roles_photo_ids={"support":[row["support"]],"query":[row["query"]]},
            dimensions={role:info[role]["shape"] for role in ("support","query")}))
        if len(records)>=limit:break
    return dict(state="PREPARED_DIAGNOSTICS" if records else "NOT_EVALUABLE",kind=kind,
        dataset="COCO-20i",fold=int(source["fold"]),seed=0,limit=limit,actual_units=len(records),
        predictions_cap=2*len(records),brightness_factor=.75,records=records,
        exclusions=excluded,data_root=str(data),official_mask_root=str(annotation),
        source_manifest=str(source_path),source_manifest_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        assets=source.get("assets",[]),query_presence_used_only_to_construct_fixture=True,
        selection="Existing frozen first10 order; original target plus smallest different shared class. No inference/difficulty/IoU selection.",
        scope="Native-only diagnostic fixtures, not candidate-method success or all-fold/cross-dataset evidence",
        exposure="CPU preparation may inspect query class presence. Runtime algorithms receive RGB and sole support mask; query GT opens after all unit outputs freeze.")


def self_check():
    with tempfile.TemporaryDirectory(prefix="runtime_diag_prepare_cpu_") as folder:
        root=Path(folder);data=root/"data";annotation=root/"ann";data.mkdir();annotation.mkdir()
        rgb=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        label=np.ones((8,8),dtype=np.uint8);label[4:]=37
        for name in ("s","q"):
            Image.fromarray(rgb).save(data/(name+".png"))
            Image.fromarray(label).save(annotation/(name+".png"))
        path=root/"source.json"
        path.write_text(json.dumps(dict(state="PREPARED_ASSETS",seed=0,fold=0,
            official_mask_root=str(annotation),assets=[],
            frozen_episodes=[dict(e=3,c=36,support="s.png",query="q.png")])))
        result=prepare(path,"concepts",5,data)
        assert result["records"][0]["targets"]==[36,0] and result["actual_units"]==1
        assert result["predictions_cap"]==2 and result["records"][0]["support_rgb_sha256"]==rgb_hash(rgb)
        result=prepare(path,"brightness",1,data)
        assert result["records"][0]["targets"]==[36] and result["brightness_factor"]==.75
    assert "torch" not in sys.modules
    print(json.dumps(dict(state="CPU_PREPARATION_CHECK_PASSED",torch_imported=False,
        no_model_or_GPU=True,fixed_presence_selection=True,source_RGB_hashes=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest",type=Path)
    parser.add_argument("--kind",choices=("concepts","brightness"))
    parser.add_argument("--limit",type=int,default=5)
    parser.add_argument("--data-root",type=Path,default=Path("/root/demo4_cache/data/COCO2014"))
    parser.add_argument("--out",type=Path)
    parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    if args.self_check:self_check();return
    if not all((args.source_manifest,args.kind,args.out)):parser.error("--source-manifest --kind --out required")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Explicit CUDA_VISIBLE_DEVICES empty required for preparation")
    if args.out.exists():raise ValueError("Fresh diagnostic manifest required")
    doc=prepare(args.source_manifest,args.kind,args.limit,args.data_root)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(doc,indent=2))
    print(json.dumps({k:doc[k] for k in ("state","kind","actual_units","predictions_cap","scope")}))


if __name__=="__main__":main()
