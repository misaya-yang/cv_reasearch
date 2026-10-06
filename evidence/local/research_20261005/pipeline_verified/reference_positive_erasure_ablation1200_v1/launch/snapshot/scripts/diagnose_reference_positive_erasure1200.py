#!/usr/bin/env python3
"""Single causal ablation: remove only reference-positive RGB erasure.

All736 query CLS,332 BG CLS, boxes,77 paired episodes and score orientation
remain fixed. Root launches332 natural reference-positive views. This is a
GT-privileged diagnostic, not a legal proposal generator or method result.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import diagnose_object_cls_dev1200 as frozen
base = frozen.base
import numpy as np
from PIL import Image


def prepare(a):
    if a.out.exists():raise FileExistsError("Fresh own output required")
    source=a.source.resolve()
    prep,config,cases=base.prepared(argparse.Namespace(out=source))
    seal=json.loads((source/"sealed.json").read_text())
    state=json.loads((source/"score_state.json").read_text())
    report=json.loads((source/"report.json").read_text())
    if seal["state"]!="ALL_CLS_DESCRIPTORS_SEALED" or seal["n"]!=332 or seal["query_regions"]!=736 or seal["total_descriptor_images"]!=1400:
        raise ValueError("Require complete frozen1200 CLS source")
    if base.sha(source/"prepared.json")!=seal["prepared_sha256"] or base.sha(source/"report.json")!=state["report_sha256"]:
        raise ValueError("Changed source preparation/report")
    if report["primary"]["object_cls"]["eligible_episodes"]!=77:
        raise ValueError("Fixed77-episode cohort differs")
    for case in cases:
        path=source/"descriptors"/(case["key"]+".npz")
        if base.sha(path)!=seal["descriptors"][case["key"]]:raise ValueError("Changed fixed source descriptor")
        with np.load(path,allow_pickle=False) as z:
            if z["crop_ids"].tolist()!=[r["crop_id"] for r in case["regions"]] or z["q"].shape!=(len(case["regions"]),1024):
                raise ValueError("Fixed query IDs/shape differ")
            if any(z[k].dtype!=np.float32 or not np.isfinite(z[k]).all() for k in ("q","r_fg","r_bg")):
                raise ValueError("Invalid sealed source vectors")
        if base.sha(case["reference"])!=case["image_sha256"]["reference"]:
            raise ValueError("Changed original reference RGB")
    deps={name:dict(path=dep["path"],sha256=base.sha(dep["path"])) for name,dep in config["dependency_sha256"].items()
          if name in ("model_config","model_weights","timm_eva","host_transform")}
    for name in deps:
        if deps[name]["sha256"]!=config["dependency_sha256"][name]["sha256"]:raise ValueError("Changed frozen dependency "+name)
    for name,path in dict(entry=Path(__file__),frozen_stats=Path(frozen.__file__),frozen_CLS=Path(base.__file__),
                          wrapper=base.REPO/"src/ics/data.py",package_init=base.REPO/"src/ics/__init__.py").items():
        deps[name]=dict(path=str(path.resolve()),sha256=base.sha(path))
    cfg=dict(source=str(source),model_dir=config["model_dir"],host_root=config["host_root"],dependency_sha256=deps,
        intervention="reference positive: same square+10percent-context box, original natural RGB instead of erasing pixels outside FG",
        fixed="all736 q and332 BG CLS,332 source boxes,77 paired episodes,NN scores,weights/transform/FP32/batch4/sign",
        changed_reference_views=332,encoder_calls=83,GT_geometry_privileged=True,no_fit_flip_grid_or_BG_change=True,
        bootstrap=dict(draws=2000,rng="RandomState(0)",unit="all1200 connected photographs,1046 groups"),
        source_code_sha256=base.sha(__file__),source_descriptor_seal_sha256=base.sha(source/"sealed.json"),
        source_prepared_sha256=base.sha(source/"prepared.json"),source_crops_sha256=base.sha(source/"crops.json"),
        source_labels_sha256=base.sha(source/"labels.json"),source_report_sha256=base.sha(source/"report.json"))
    a.out.mkdir(parents=True)
    base.write(a.out/"config.json",cfg)
    base.write(a.out/"prepared.json",dict(state="CAUSAL_REFERENCE_ERASURE_ABLATION_PREPARED_NO_GPU",config_sha256=base.sha(a.out/"config.json"),
        new_reference_views=332,encoder_calls=83,fixed_query_vectors=736,fixed_BG_vectors=332,paired_episodes=77,
        source_descriptor_seal_sha256=cfg["source_descriptor_seal_sha256"],source_code_sha256=cfg["source_code_sha256"]))
    base.write(a.out/"state.json",dict(state="CAUSAL_REFERENCE_ERASURE_ABLATION_PREPARED_NO_GPU",GPU_run=False))
    print(json.dumps(json.loads((a.out/"prepared.json").read_text())),flush=True)


def prepared(a):
    prep=json.loads((a.out/"prepared.json").read_text());cfg=json.loads((a.out/"config.json").read_text())
    if base.sha(a.out/"config.json")!=prep["config_sha256"]:raise ValueError("Changed ablation preparation")
    source=Path(cfg["source"])
    for filename,field in (("sealed.json","source_descriptor_seal_sha256"),("crops.json","source_crops_sha256"),
                           ("labels.json","source_labels_sha256"),("report.json","source_report_sha256"),("prepared.json","source_prepared_sha256")):
        if base.sha(source/filename)!=cfg[field]:raise ValueError("Changed frozen source "+filename)
    return cfg,source,json.loads((source/"crops.json").read_text())


def infer(a):
    cfg,source,cases=prepared(a)
    if (a.out/"natural_reference_cls.npz").exists() or (a.out/"sealed.json").exists():raise FileExistsError("Do not replace partial/completed descriptors")
    for name,dep in cfg["dependency_sha256"].items():
        if base.sha(dep["path"])!=dep["sha256"]:raise ValueError("Changed dependency "+name)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(base.REPO/"src"));from ics.data import TimmDINOv3
    sys.path.insert(0,cfg["host_root"]);from utils.data import build_transform
    model=TimmDINOv3(cfg["model_dir"]).to("cuda").float().eval().requires_grad_(False).m
    if type(model).__module__!="timm.models.eva" or model.cls_token is None or model.num_prefix_tokens!=5:raise ValueError("Wrong actual CLS interface")
    transform=build_transform(1024);vectors=[];begin=time.monotonic()
    for start in range(0,332,4):
        batch=cases[start:start+4];images=[]
        for case in batch:
            if base.sha(case["reference"])!=case["image_sha256"]["reference"]:raise ValueError("Changed reference RGB")
            with Image.open(case["reference"]) as im:
                # The only intervention: do not call erase(). Keep original RGB.
                images.append(base.square_crop(im.convert("RGB"),case["reference_fg_crop"]))
        x=torch.stack([transform(im) for im in images]).to(device="cuda",dtype=torch.float32)
        with torch.inference_mode(),torch.autocast(device_type="cuda",enabled=False):
            tokens=model.forward_features(x)
            if tokens.shape!=(len(batch),4101,1024) or tokens.dtype!=torch.float32:raise ValueError("Wrong normalized CLS output")
            cls=tokens[:,0];error=float((F.normalize(cls,dim=-1).norm(dim=-1)-1).abs().max())
            values=cls.detach().cpu().numpy().copy()
        if not np.isfinite(values).all() or error>1e-6:raise ValueError("Invalid natural FG CLS")
        vectors.extend(values)
        state=dict(state="NATURAL_FG_CLS_RUNNING",completed=len(vectors),n=332,encoder_calls=start//4+1,seconds=time.monotonic()-begin)
        base.write(a.out/"state.json",state)
        if start==0:base.write(a.out/"model_setup_smoke.json",dict(state="ACTUAL_FP32_CLS_BATCH4_OK",shape=[4,4101,1024],CLS_index=0,unit_norm_error=error,
            cuda_peak_bytes=int(torch.cuda.max_memory_allocated()),no_autocast=True,descriptors_retained=True))
        if len(vectors)%40==0 or len(vectors)==332:print(json.dumps(state),flush=True)
    np.savez_compressed(a.out/"natural_reference_cls.npz",keys=np.array([c["key"] for c in cases]),r_fg_natural=np.stack(vectors))
    base.write(a.out/"sealed.json",dict(state="ALL_NATURAL_FG_CLS_SEALED",new_reference_views=332,encoder_calls=83,
        fixed_query_vectors=736,fixed_BG_vectors=332,paired_episodes=77,natural_reference_sha256=base.sha(a.out/"natural_reference_cls.npz"),
        source_descriptor_seal_sha256=cfg["source_descriptor_seal_sha256"],config_sha256=base.sha(a.out/"config.json"),
        labels_opened_in_infer=False,GT_boxes_privileged=True,seconds=time.monotonic()-begin,cuda_peak_bytes=int(torch.cuda.max_memory_allocated())))
    base.write(a.out/"state.json",dict(state="ALL_NATURAL_FG_CLS_SEALED",completed=332,n=332))


def score(a):
    cfg,source,cases=prepared(a);seal=json.loads((a.out/"sealed.json").read_text())
    if seal["state"]!="ALL_NATURAL_FG_CLS_SEALED" or seal["new_reference_views"]!=332 or base.sha(a.out/"natural_reference_cls.npz")!=seal["natural_reference_sha256"]:
        raise ValueError("Require all332 sealed intervention descriptors")
    if (a.out/"report.json").exists():raise FileExistsError("Do not overwrite scored result")
    fixed_seal=json.loads((source/"sealed.json").read_text());labels=json.loads((source/"labels.json").read_text())
    old_report=json.loads((source/"report.json").read_text());old_scores=json.loads((source/"scored_regions.json").read_text())
    expected={r["crop_id"]:r["scores"]["object_cls"] for r in old_scores};margins={}
    with np.load(a.out/"natural_reference_cls.npz",allow_pickle=False) as z:
        if z["keys"].tolist()!=[c["key"] for c in cases] or z["r_fg_natural"].shape!=(332,1024) or z["r_fg_natural"].dtype!=np.float32:
            raise ValueError("Intervention case IDs/precision differ")
        natural=z["r_fg_natural"].copy()
    for case,rnat in zip(cases,natural):
        path=source/"descriptors"/(case["key"]+".npz")
        if base.sha(path)!=fixed_seal["descriptors"][case["key"]]:raise ValueError("Changed fixed q/BG vector source")
        with np.load(path,allow_pickle=False) as z:
            q,bg,masked=base.unit(z["q"]),base.unit(z["r_bg"]),base.unit(z["r_fg"])
            bgsim=np.sum(q*bg[None],axis=1)
            old=np.sum(q*masked[None],axis=1)-bgsim
            new=np.sum(q*base.unit(rnat)[None],axis=1)-bgsim
            for crop_id,v,w in zip(z["crop_ids"].tolist(),old,new):
                if float(v)!=expected[crop_id]:raise ValueError("Fixed original masked CLS margin does not reproduce")
                margins[crop_id]=dict(maskedFG=float(v),naturalFG=float(w))
    records=labels["regions"]
    for r in records:r["scores"]=dict(object_cls=margins[r["crop_id"]]["naturalFG"],stored_nn_mean=r["controls"]["stored_nn_mean"])
    natural_report,draws,natural_values=frozen.statistics(labels["source_rows"],records)
    for r in records:r["scores"]["object_cls"]=margins[r["crop_id"]]["maskedFG"]
    masked_report,masked_draws,masked_values=frozen.statistics(labels["source_rows"],records)
    if not np.array_equal(draws,masked_draws) or masked_report["primary"]!=old_report["primary"] or masked_report["paired_contrasts"]!=old_report["paired_contrasts"]:
        raise ValueError("Same-cohort masked/NN statistics do not reproduce")
    groups=base.photo_groups(labels["source_rows"]);g=int(groups.max())+1
    delta=np.array([np.nan if x is None else x for x in natural_values["object_cls"]])-np.array([np.nan if x is None else x for x in masked_values["object_cls"]])
    valid=np.isfinite(delta);weights=np.stack([np.bincount(d,minlength=g) for d in draws])
    den=np.sum(weights*np.bincount(groups[valid],minlength=g)[None],axis=1)
    boot=np.sum(weights*np.bincount(groups[valid],weights=delta[valid],minlength=g)[None],axis=1)/den
    if valid.sum()!=77 or natural_report["primary"]["object_cls"]["eligible_episodes"]!=77:raise ValueError("Fixed77 cohort changed")
    report=dict(state="REFERENCE_POSITIVE_ERASURE_ABLATION_SCORED",config=cfg,source_n=1200,photo_groups=g,paired_episodes=77,
        primary=dict(naturalFG=natural_report["primary"]["object_cls"],maskedFG=masked_report["primary"]["object_cls"],NN=natural_report["primary"]["stored_nn_mean"]),
        natural_minus_masked=dict(mean=float(delta[valid].mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist(),eligible_episodes=77),
        natural_vs_NN_and_chance=natural_report["paired_contrasts"],natural_folds=natural_report["folds"],masked_folds=masked_report["folds"],
        natural_supplement=natural_report["supplement"],all736_q_and332_BG_fixed_source_hashes_verified=True,
        all736_maskedFG_margin_and77_AUC_CI_exact=True,new_reference_views=332,GT_geometry_privileged=True,complete_method_result=False,
        result_boundary="one reference-positive-erasure intervention only; no sign flip,cohort selection,BG change,fit or grid")
    base.write(a.out/"report.json",report);base.write(a.out/"region_margins.json",margins)
    np.save(a.out/"bootstrap_photo_draws.npy",draws)
    base.write(a.out/"score_state.json",dict(state=report["state"],report_sha256=base.sha(a.out/"report.json")))
    (a.out/"report.md").write_text("# Fixed reference-positive RGB-erasure ablation\n\nGT boxes remain privileged; one causal intervention changes only reference-FG erasure.\n\n"+json.dumps({k:report[k] for k in ("primary","natural_minus_masked","natural_vs_NN_and_chance","natural_folds")},indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("state","primary","natural_minus_masked")}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","infer","score"));p.add_argument("--out",type=Path,required=True);p.add_argument("--source",type=Path)
    a=p.parse_args()
    if a.stage=="prepare" and not a.source:p.error("Prepare requires sealed --source")
    {"prepare":prepare,"infer":infer,"score":score}[a.stage](a)


if __name__=="__main__":main()
