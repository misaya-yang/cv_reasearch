#!/usr/bin/env python3
"""Expand the unchanged object-crop/actual-CLS diagnostic on sealed DEV1200.

CPU prepare only is authorized here; root owns infer scheduling. Reuse all367
legacy CLS vectors by (class, reference, query) identity and exact region boxes.
The full-cohort simple control is fixed region-average packet FGmax-BGmax.
Raw region-mean controls exist only for the original241 diagnostic cohort.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import diagnose_object_cls_dev241 as base
import numpy as np
from PIL import Image
from scipy import ndimage

CUES = ("object_cls", "stored_nn_mean")
CENTERS = np.arange(64) * 16 + 8


def identity(row):
    return int(row["c"]), Path(row["support"]).name, Path(row["query"]).name


def fixed_regions(truth, prediction, nnmargin):
    if truth.shape!=(1024,1024) or prediction.shape!=truth.shape or nnmargin.shape!=(64,64) or not np.isfinite(nnmargin).all():
        raise ValueError("Unexpected source geometry or nonfinite stored NN cue")
    coverage = truth.reshape(64,16,64,16).mean((1,3))
    result, availability = [], {}
    for role, mask, other, pure in (
        ("whole_missed_GT", truth, prediction, coverage >= .9),
        ("stray_fine64", prediction, truth, coverage <= .1),
    ):
        labels, count = ndimage.label(mask, np.ones((3,3),bool))
        touches = np.bincount(labels[other], minlength=count+1)>0
        touches[0] = True
        sizes = np.bincount(labels.ravel(), minlength=count+1)
        membership = labels[np.ix_(CENTERS,CENTERS)].ravel()
        objects = ndimage.find_objects(labels)
        selected = np.flatnonzero(~touches)
        region_records = []
        for region in selected:
            ids = np.flatnonzero(pure.ravel() & (membership==region))
            entry = dict(role=role, region=int(region), area=int(sizes[region]), pure_tokens=int(len(ids)))
            region_records.append(entry)
            if len(ids):
                y,x = objects[region-1]
                result.append(dict(entry, working_bbox=[x.start,y.start,x.stop,y.stop],
                                   stored_nn_mean=float(nnmargin.ravel()[ids].mean())))
        availability[role] = dict(all_regions=len(selected),eligible_regions=sum(r["pure_tokens"]>0 for r in region_records),
            regions_without_pure_tokens=sum(r["pure_tokens"]==0 for r in region_records),
            total_pixels=sum(r["area"] for r in region_records),
            eligible_pixels=sum(r["area"] for r in region_records if r["pure_tokens"]),
            pixels_without_pure_tokens=sum(r["area"] for r in region_records if not r["pure_tokens"]),
            small_regions_without_pure_tokens=sum(r["area"]<256 and not r["pure_tokens"] for r in region_records))
    return result,availability


def prepare(a):
    if a.out.exists():
        raise FileExistsError("Fresh owned namespace required; no overwrite or deletion")
    fine = a.fine.resolve()
    manifest = json.loads((fine/"manifest.json").read_text())
    seal = json.loads((fine/"sealed.json").read_text())
    score_state = json.loads((fine/"score_state.json").read_text())
    fine_report = json.loads((fine/"report.json").read_text())
    scored = {r["key"]:r for r in [json.loads(l) for l in (fine/"episodes.jsonl").read_text().splitlines() if l.strip()]}
    if len(manifest)!=1200 or len(scored)!=1200 or len({identity(r) for r in manifest})!=1200:
        raise ValueError("Require all1200 sealed sampled identities")
    if seal["state"]!="ALL_PREDICTIONS_SEALED" or base.sha(fine/"manifest.json")!=seal["manifest_sha256"]:
        raise ValueError("Unsealed or changed1200 manifest")
    if score_state["state"]!="CPU_GT_SCORE_COMPLETE" or score_state["completed"]!=1200 or base.sha(fine/"report.json")!=score_state["report_sha256"]:
        raise ValueError("Require fully scored/sealed1200 source")
    if base.sha(fine/"sealed.json")!=fine_report["source_prediction_seal_sha256"]:
        raise ValueError("Scored1200 source seal differs")
    legacy = a.legacy.resolve()
    legacy_prep, legacy_config, legacy_cases = base.prepared(argparse.Namespace(out=legacy))
    legacy_seal = json.loads((legacy/"sealed.json").read_text())
    legacy_labels = json.loads((legacy/"labels.json").read_text())
    legacy_report = json.loads((legacy/"report.json").read_text())
    legacy_scores = json.loads((legacy/"scored_regions.json").read_text())
    if legacy_seal["state"]!="ALL_CLS_DESCRIPTORS_SEALED" or legacy_seal["descriptor_forwards"]!=367:
        raise ValueError("Require all367 sealed legacy descriptors")
    if base.sha(legacy/"prepared.json")!=legacy_seal["prepared_sha256"] or base.sha(legacy/"labels.json")!=legacy_prep["labels_sha256"]:
        raise ValueError("Changed legacy crop/label preparation")
    case_by_key = {r["key"]:r for r in legacy_cases}
    by_legacy_id = {identity(r):r for r in legacy_labels["source_rows"]}
    regions_by_key = {}
    for r in legacy_scores:
        regions_by_key.setdefault(r["episode"],[]).append(r)
    host = json.loads(a.host_manifest.read_text())
    data,annotations = Path(host["data_root"]),Path(host["annotation_root"])
    cases,labels,inputs,availability,legacy_matched = [],[],[],[],0
    reused_images,reused_regions = 0,0
    begin=time.monotonic()
    for index,row in enumerate(manifest):
        key=row["key"]
        if identity(scored[key])!=identity(row) or scored[key]["fold"]!=row["fold"]:
            raise ValueError("Manifest/scored episode identity mismatch")
        packet,prediction=Path(row["packet_export"]),fine/"predictions"/(key+".npz")
        expected=seal["inputs"][key]
        if base.sha(packet)!=expected["packet_sha256"] or base.sha(prediction)!=seal["predictions"][key]:
            raise ValueError("Changed1200 packet/prediction "+key)
        with np.load(packet,allow_pickle=False) as z:
            truth_packed,cov=z["truth"].copy(),z["cov"].copy()
            truth=base.unpack(truth_packed)
            nnmargin=(z["fg_max"]-z["bg_max"]).reshape(64,64).copy()
        with np.load(prediction,allow_pickle=False) as z:
            pred=base.unpack(z["fine.rcg64"])
        iu=[int((truth&pred).sum()),int((truth|pred).sum())]
        if iu!=scored[key]["iu"]["fine.rcg64"]:
            raise ValueError("Source truth/fine64 I/U differs "+key)
        rr,available=fixed_regions(truth,pred,nnmargin)
        availability.append(available)
        query,reference=data/row["query"],data/row["support"]
        reference_mask=annotations/Path(row["support"]).with_suffix(".png")
        with Image.open(query) as im:query_size=im.size
        with Image.open(reference) as im:reference_size=im.size
        with Image.open(reference_mask) as im:fg=np.array(im)==row["c"]+1
        if fg.shape!=reference_size[::-1]:
            raise ValueError("Reference mask/header geometry differs "+key)
        yy=(np.arange(1024)*fg.shape[0]/1024).astype(int)
        xx=(np.arange(1024)*fg.shape[1]/1024).astype(int)
        work=fg[np.ix_(yy,xx)].reshape(64,16,64,16).mean((1,3)).astype(np.float32)
        if not np.array_equal(work,cov):
            raise ValueError("Reference mask does not reproduce sealed coverage "+key)
        hashes=dict(query=base.sha(query),reference=base.sha(reference),reference_mask=base.sha(reference_mask))
        if hashes["query"]!=expected["query_image_sha256"]:
            raise ValueError("Query RGB differs from1200 source "+key)
        inputs.append(dict(key=key,packet=str(packet),packet_sha256=expected["packet_sha256"],
            prediction=str(prediction),prediction_sha256=seal["predictions"][key],truth_packed_sha256=base.array_sha(truth_packed),
            reference_coverage_sha256=base.array_sha(cov),fine64_IU=iu,image_sha256=hashes,
            query_size=query_size,reference_size=reference_size))
        previous=by_legacy_id.get(identity(row))
        legacy_matched+=int(previous is not None)
        if not rr:
            if previous and previous["key"] in case_by_key:
                raise ValueError("Legacy eligible regions disappeared")
            continue
        case=dict(key=key,c=row["c"],fold=row["fold"],query=str(query),reference=str(reference),reference_mask=str(reference_mask),
            query_size=list(query_size),reference_size=list(reference_size),image_sha256=hashes,reference_FG_array_sha256=base.array_sha(fg),
            reference_fg_crop=base.square_box(base.bbox(fg),reference_size),
            reference_bg_crop=base.square_box([0,0,*reference_size],reference_size,context=0),regions=[])
        paired=len({r["role"] for r in rr})==2
        for slot,r in enumerate(rr):
            crop_id=f"{key}--{slot}"
            case["regions"].append(dict(crop_id=crop_id,working_bbox=r["working_bbox"],crop=base.square_box(r["working_bbox"],query_size,working=True)))
            labels.append(dict(crop_id=crop_id,episode=key,role=r["role"],original_region_id=r["region"],
                area=r["area"],pure_tokens=r["pure_tokens"],primary_paired_episode=paired,
                controls=dict(stored_nn_mean=r["stored_nn_mean"])))
        if previous:
            old_case=case_by_key[previous["key"]]
            old_regions=regions_by_key[previous["key"]]
            old_lookup={(r["role"],r["original_region_id"]):r for r in old_regions}
            if set(old_lookup)!={(r["role"],r["region"]) for r in rr}:
                raise ValueError("Legacy/full1200 region IDs differ")
            for name in ("reference_fg_crop","reference_bg_crop","query_size","reference_size","image_sha256","reference_FG_array_sha256"):
                if old_case[name]!=case[name]:
                    raise ValueError("Changed frozen legacy crop source "+name)
            old_crops={r["crop_id"]:r for r in old_case["regions"]}
            source_indices={r["crop_id"]:i for i,r in enumerate(old_case["regions"])}
            reuse_indices=[]
            for r,crop,new_label in zip(rr,case["regions"],labels[-len(rr):]):
                old=old_lookup[(r["role"],r["region"])]
                if old["area"]!=r["area"] or old["pure_tokens"]!=r["pure_tokens"] or old_crops[old["crop_id"]]["crop"]!=crop["crop"]:
                    raise ValueError("Legacy frozen region geometry differs")
                reuse_indices.append(source_indices[old["crop_id"]])
                new_label.update(legacy_region=dict(old_episode=previous["key"],old_crop_id=old["crop_id"],
                    object_cls=old["scores"]["object_cls"],raw_mean_prototype=old["scores"]["raw_mean_prototype"],
                    projected_mean_prototype=old["scores"]["projected_mean_prototype"]))
            descriptor=legacy/"descriptors"/(previous["key"]+".npz")
            if base.sha(descriptor)!=legacy_seal["descriptors"][previous["key"]]:
                raise ValueError("Changed reusable descriptor seal")
            case.update(reuse_descriptor=str(descriptor),reuse_descriptor_sha256=base.sha(descriptor),
                reuse_source_crop_ids=[r["crop_id"] for r in old_case["regions"]],reuse_query_indices=reuse_indices)
            reused_images+=2+len(rr);reused_regions+=len(rr)
        cases.append(case)
        if (index+1)%200==0:
            print(json.dumps(dict(verified_source_rows=index+1,n=1200,seconds=time.monotonic()-begin)),flush=True)
    if legacy_matched!=241 or reused_images!=367 or reused_regions!=197:
        raise ValueError("All241 identities/367 vectors/197 fixed regions must be preserved")
    total_images=sum(2+len(c["regions"]) for c in cases)
    new_images=total_images-reused_images
    counts={role:sum(r["role"]==role for r in labels) for role in ("whole_missed_GT","stray_fine64")}
    config=dict(base.CONFIG,cohort="all1200 fully sealed exposed DEV draws; same frozen crop/CLS construction, no new GT selection rule",
        primary="mean episode AUROC on all1200 episodes with both eligible missed/stray regions",
        supplement="pooled eligible regions are secondary; raw control remains restricted to original241",
        full1200_control="stored packet FGmax-BGmax, averaged over identical eligible pure64-grid tokens per region",
        raw_full1200_available=False,raw_control_scope="original241 source cohort only, original18 paired episodes",
        model_dir=legacy_config["model_dir"],host_root=legacy_config["host_root"],source_episodes=1200,
        encoded_episodes=len(cases),query_regions=len(labels),expected_descriptor_forwards=new_images,
        total_descriptor_images=total_images,reused_descriptor_images=367,
        source_code_sha256=base.sha(base.__file__),entry_source_code_sha256=base.sha(__file__),
        descriptor_storage_estimate_bytes=total_images*1024*4,no_encoder_or_GPU_execution_in_prepare=True)
    deps=dict(legacy_config["dependency_sha256"])
    deps["wrapper"]=dict(path=str(base.REPO/"src/ics/data.py"),sha256=base.sha(base.REPO/"src/ics/data.py"))
    deps["package_init"]=dict(path=str(base.REPO/"src/ics/__init__.py"),sha256=base.sha(base.REPO/"src/ics/__init__.py"))
    deps["entry_script"]=dict(path=str(Path(__file__).resolve()),sha256=base.sha(__file__))
    deps["frozen_cls_implementation"]=dict(path=str(Path(base.__file__).resolve()),sha256=base.sha(base.__file__))
    for name,dep in deps.items():
        if base.sha(dep["path"])!=dep["sha256"]:
            raise ValueError("Changed frozen dependency "+name)
    config["dependency_sha256"]=deps
    aggregate={role:{name:sum(r[role][name] for r in availability) for name in availability[0][role]}
               for role in ("whole_missed_GT","stray_fine64")}
    source_hashes={str(folder/name):base.sha(folder/name) for folder,names in (
        (fine,("manifest.json","sealed.json","episodes.jsonl","report.json","score_state.json")),
        (legacy,("prepared.json","config.json","crops.json","labels.json","sealed.json","report.json","scored_regions.json"))) for name in names}
    config["source_metadata_sha256"]=source_hashes
    a.out.mkdir(parents=True)
    base.write(a.out/"config.json",config);base.write(a.out/"crops.json",cases)
    base.write(a.out/"labels.json",dict(regions=labels,source_rows=manifest,legacy241_report=legacy_report))
    base.write(a.out/"source_receipt.json",dict(inputs=inputs,source_metadata_sha256=source_hashes,
        all1200_source_hashes_IU_RGB_headers_reference_coverage_verified=True,query_GT_used_for_geometry=True))
    summary=dict(state="CPU_CROPS_PREPARED_NO_ENCODER_RUN",source_episodes=1200,encoded_episodes=len(cases),
        query_regions=len(labels),counts=counts,paired_episodes=sum(len({r["role"] for r in labels if r["episode"]==c["key"]})==2 for c in cases),
        paired_regions=sum(r["primary_paired_episode"] for r in labels),expected_descriptor_forwards=new_images,
        total_descriptor_images=total_images,reused_descriptor_images=367,reused_query_regions=197,
        new_encoder_calls=sum(math.ceil((2+len(c["regions"]))/4) for c in cases if not c.get("reuse_descriptor")),
        fixed_max_batch=4,availability=aggregate,raw_full1200_available=False,
        config_sha256=base.sha(a.out/"config.json"),crops_sha256=base.sha(a.out/"crops.json"),labels_sha256=base.sha(a.out/"labels.json"),
        source_receipt_sha256=base.sha(a.out/"source_receipt.json"),source_code_sha256=base.sha(__file__),seconds=time.monotonic()-begin)
    base.write(a.out/"prepared.json",summary)
    base.write(a.out/"state.json",dict(state="CPU_CROPS_PREPARED_NO_ENCODER_RUN",encoder_forwards=0,GPU_run=False))
    print(json.dumps(summary),flush=True)


def statistics(rows,records):
    groups=base.photo_groups(rows);g=int(groups.max())+1
    draws=np.random.RandomState(0).randint(g,size=(2000,g))
    weights=np.stack([np.bincount(d,minlength=g) for d in draws])
    def mean(values):
        v=np.array([np.nan if x is None else x for x in values]);valid=np.isfinite(v)
        counts=np.bincount(groups[valid],minlength=g);totals=np.bincount(groups[valid],weights=v[valid],minlength=g)
        den=np.sum(weights*counts[None],axis=1);samples=np.divide(np.sum(weights*totals[None],axis=1),den,out=np.full(2000,np.nan),where=den>0)
        samples=samples[np.isfinite(samples)]
        return dict(mean=float(v[valid].mean()) if valid.any() else None,
            ci95=np.percentile(samples,[2.5,97.5]).tolist() if len(samples) else None,eligible_episodes=int(valid.sum()))
    bykey={r["key"]:[] for r in rows}
    for r in records:bykey[r["episode"]].append(r)
    values={cue:[base.auc([r["scores"][cue] for r in bykey[row["key"]] if r["role"]=="whole_missed_GT"],
                         [r["scores"][cue] for r in bykey[row["key"]] if r["role"]=="stray_fine64"]) for row in rows] for cue in CUES}
    result=dict(primary={cue:mean(values[cue]) for cue in CUES},
        paired_contrasts=dict(CLS_minus_NN=mean([a-b if a is not None and b is not None else None for a,b in zip(values[CUES[0]],values[CUES[1]])]),
                             CLS_minus_chance=mean([v-.5 if v is not None else None for v in values[CUES[0]]])),folds={},supplement={},source_n=len(rows),photo_groups=g)
    for fold in range(4):
        ix=[i for i,r in enumerate(rows) if r["fold"]==fold and values[CUES[0]][i] is not None]
        result["folds"][str(fold)]=dict(eligible_episodes=len(ix),episode_auc={c:float(np.mean([values[c][i] for i in ix])) if ix else None for c in CUES},
            CLS_minus_NN=float(np.mean([values[CUES[0]][i]-values[CUES[1]][i] for i in ix])) if ix else None)
    positive=np.array([r["role"]=="whole_missed_GT" for r in records])
    def pooled(cue,mass):
        scores=np.array([r["scores"][cue] for r in records]);order=np.argsort(scores,kind="stable")
        scores,fg,w=scores[order],positive[order],np.asarray(mass,float)[order]
        starts=np.r_[0,np.flatnonzero(np.diff(scores))+1]
        p=np.add.reduceat(w*fg,starts);n=np.add.reduceat(w*~fg,starts)
        return float(np.sum(p*(np.cumsum(n)-n+.5*n))/(p.sum()*n.sum())) if p.sum() and n.sum() else None
    for cue in CUES:
        scores=np.array([r["scores"][cue] for r in records])
        result["supplement"][cue]=dict(region_pooled_auc=pooled(cue,np.ones(len(records))),
            pixel_mass_weighted_auc=pooled(cue,[r["area"] for r in records]),positive_regions=int(positive.sum()),negative_regions=int((~positive).sum()),
            fixed_zero_margin_TP=int(((scores>0)&positive).sum()),fixed_zero_margin_FP=int(((scores>0)&~positive).sum()))
    return result,draws,values


def score(a):
    prep,config,cases=base.prepared(a)
    seal=json.loads((a.out/"sealed.json").read_text())
    if seal["state"]!="ALL_CLS_DESCRIPTORS_SEALED" or seal["descriptor_forwards"]!=prep["expected_descriptor_forwards"] or seal["reused_descriptor_images"]!=367:
        raise ValueError("Require complete frozen1200 descriptor seal")
    if base.sha(a.out/"prepared.json")!=seal["prepared_sha256"] or base.sha(a.out/"labels.json")!=prep["labels_sha256"]:
        raise ValueError("Changed prepared cohort")
    if (a.out/"report.json").exists():raise FileExistsError("Never overwrite scored evidence")
    labels=json.loads((a.out/"labels.json").read_text());margin={}
    for case in cases:
        path=a.out/"descriptors"/(case["key"]+".npz")
        if base.sha(path)!=seal["descriptors"][case["key"]]:raise ValueError("Changed CLS descriptor")
        with np.load(path,allow_pickle=False) as z:
            if z["crop_ids"].tolist()!=[r["crop_id"] for r in case["regions"]]:raise ValueError("Region IDs differ")
            scores=np.sum(base.unit(z["q"])*base.unit(z["r_fg"])[None],axis=1)-np.sum(base.unit(z["q"])*base.unit(z["r_bg"])[None],axis=1)
            margin.update(zip(z["crop_ids"].tolist(),map(float,scores)))
    records=labels["regions"]
    if set(margin)!={r["crop_id"] for r in records}:raise ValueError("Incomplete region descriptors")
    for r in records:
        r["scores"]=dict(r["controls"],object_cls=margin[r["crop_id"]])
        if r.get("legacy_region") and r["scores"]["object_cls"]!=r["legacy_region"]["object_cls"]:
            raise ValueError("Reused197-region CLS margin differs")
    report,draws,values=statistics(labels["source_rows"],records)
    if report["primary"]["object_cls"]["eligible_episodes"]!=prep["paired_episodes"]:
        raise ValueError("Fixed paired cohort changed")
    report.update(state="CPU_GT_CLS1200_DIAGNOSTIC_COMPLETE",config=config,region_counts=prep["counts"],
        availability=prep["availability"],descriptor_seal_sha256=base.sha(a.out/"sealed.json"),raw_full1200_available=False,
        original241_control=dict(scope="original241 source,18 paired episodes/71 regions,not a full1200 control",
            report=labels["legacy241_report"]["primary"],paired_contrasts=labels["legacy241_report"]["paired_contrasts"]),
        exact197_legacy_CLS_margin_parity=True,complete_method_result=False,GT_geometry_privileged=True)
    base.write(a.out/"report.json",report);base.write(a.out/"scored_regions.json",records)
    np.save(a.out/"bootstrap_photo_draws.npy",draws)
    np.savez_compressed(a.out/"episode_auc.npz",**{c:np.array([np.nan if x is None else x for x in v]) for c,v in values.items()})
    lines=["# Frozen object-crop CLS diagnostic: exposed1200","","GT crop geometry is privileged; no legal proposal generator or complete segmentation gain is established.","",
        "| cue | paired episodes | episode AUROC [95% CI] |","|---|---:|---:|"]
    for cue in CUES:
        v=report["primary"][cue];lines.append("| %s | %d | %.4f [%.4f,%.4f] |"%(cue,v["eligible_episodes"],v["mean"],*v["ci95"]))
    lines += ["","Paired contrasts:","",json.dumps(report["paired_contrasts"],indent=2),"","Folds:","",json.dumps(report["folds"],indent=2),"",
        "Secondary pooled-region summaries:","",json.dumps(report["supplement"],indent=2),"",
        "Full1200 raw activations are missing. The raw/projection controls remain explicitly limited to original241; region-averaged NN is a different full1200 control. All197 original CLS margins reproduce exactly."]
    (a.out/"report.md").write_text("\n".join(lines)+"\n")
    base.write(a.out/"score_state.json",dict(state=report["state"],report_sha256=base.sha(a.out/"report.json")))
    print(json.dumps(dict(state=report["state"],primary=report["primary"],paired_contrasts=report["paired_contrasts"])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","infer","score","self-test"))
    for name in ("out","fine","legacy","host-manifest"):p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.stage=="self-test":base.self_test();return
    if not a.out:p.error("Require out")
    if a.stage=="prepare" and not all((a.fine,a.legacy,a.host_manifest)):p.error("Prepare requires fine/legacy/host-manifest")
    {"prepare":prepare,"infer":base.infer,"score":score}[a.stage](a)


if __name__=="__main__":main()
