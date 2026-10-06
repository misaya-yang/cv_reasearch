#!/usr/bin/env python3
"""Exact fixed-size-cut composition from existing sealed128-grid fields.

No encoder, image, fitting, sign change, or peer queue modification. CUDA is
used only for the source-identical bilinear finalizer. All4000 fold-temperature
controls plus the2000 already matching uniform-tau15 primary masks seal before
GT. A successor supplies the missing uniform-tau15 fold0/3 fields/masks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from ics.experiment import sha, summarize

CONTROL="fixed_size_cut_on_foldtemp_fine16.control"
PRIMARY="provided_uniform_t15_size_cut.primary"
BASE=("native","rcg","mean.control","rcg64.control","fine.rcg16.control","fine.rcg64")
POPCOUNT=np.array([i.bit_count() for i in range(256)],np.uint8)


def write(path,value):
    p=Path(path);temporary=p.with_suffix(p.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(p)


def expected_recipe(path):
    f=json.loads(path.read_text())
    if f["readout"]!={"kind":"shifted-grid feature-guided upsampling to 128 x 128","sigma":1.25,"tau":.15,"window":2}:
        raise ValueError("Do not change supplied frozen readout")
    if f["size_cut"]!={"mask_area_at_half_edges":[0,.01,.03,.08,.2,.4,1.01],"levels":[.5,.5625,.5875,.5625,.45,.375]}:
        raise ValueError("Do not change supplied six thresholds/bins")
    return f


def source(a):
    seal=json.loads((a.source/"sealed.json").read_text());cfg=json.loads((a.source/"config.json").read_text())
    rows=json.loads((a.source/"manifest.json").read_text())
    if seal["state"]!="ALL_PREDICTIONS_SEALED" or seal["n"]!=4000 or len(rows)!=4000:
        raise ValueError("Require complete sealed4000 source")
    for file,key in (("manifest.json","manifest_sha256"),("config.json","config_sha256")):
        if sha(a.source/file)!=seal[key]:raise ValueError("Changed source metadata")
    params={str(f):dict(sigma=1.25,tau=.07 if f in (0,3) else .15) for f in range(4)}
    for f,p in params.items():
        if any(cfg["parameters"][f][k]!=v for k,v in p.items()):raise ValueError("Unexpected source fold temperature")
    return rows,seal,cfg


def packed(value):
    if value.shape!=(131072,) or value.dtype!=np.uint8:raise ValueError("Require packed1024 mask")
    return value


def load(job):
    row,root,seal=job
    key=row["key"];fp=Path(root)/"fields"/(key+".npz");pp=Path(root)/"predictions"/(key+".npz")
    if sha(fp)!=seal["fields"][key] or sha(pp)!=seal["predictions"][key]:raise ValueError("Changed source field/prediction "+key)
    with np.load(fp,allow_pickle=False) as z:field=z["fine.rcg16.control"].copy()
    with np.load(pp,allow_pickle=False) as z:half=packed(z["fine.rcg16.control"]).copy()
    if field.shape!=(128,128) or field.dtype!=np.float32 or not np.isfinite(field).all():raise ValueError("Wrong source field")
    return row,field,half


def infer(a):
    if a.out.exists():raise FileExistsError("Fresh owned output; never overwrite")
    rows,source_seal,source_cfg=source(a);recipe=expected_recipe(a.frozen)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.out.mkdir(parents=True);(a.out/"predictions").mkdir()
    config=dict(n=4000,source=str(a.source.resolve()),source_seal_sha256=sha(a.source/"sealed.json"),frozen=str(a.frozen.resolve()),
        frozen_sha256=sha(a.frozen),recipe=recipe,primary=PRIMARY,control=CONTROL,
        primary_available_n=2000,primary_missing_n=2000,primary_missing_folds=[0,3],
        primary_rule="uniformtau.15; only fold1/2 existing fields match",control_rule="same fixed sizecut on fold0/3tau.07,fold1/2tau.15",
        version="fresh600-global-fit65.86004769,not crossfold65.44",crossfold_recipe="not serialized in source run_rcg2 counts/rows; unavailable",
        source_feature_semantics="unit(FP16 coarseq/r),FP32 shiftedfine; lambda16 sealed RCG,alpha.5,sigma1.25,window2",
        renderer="CUDA bilinear128->1024 align_corners=False; strict>threshold; source halfmask exact parity before cut",
        area="pixel count of source-exact fine>.5 /1048576; searchsorted(edges,area,right)-1",
        encoder_forwards=0,images_opened=0,query_truth_opened=False,source_code_sha256=sha(__file__),
        batch=a.batch,readers=a.readers,writers=a.writers,exposure="all4000 existing DEV/public draws including fitted600;no independent confirmation")
    write(a.out/"config.json",config);write(a.out/"manifest.json",rows)
    begin=time.monotonic();result=[];pending_writes=[];edges=np.array(recipe["size_cut"]["mask_area_at_half_edges"])
    levels=np.array(recipe["size_cut"]["levels"]);matched=[];missing=[]
    def save(row,masks,receipt):
        path=a.out/"predictions"/(row["key"]+".npz");np.savez_compressed(path,**masks)
        return row["key"],sha(path),receipt
    with ThreadPoolExecutor(a.readers) as readers,ThreadPoolExecutor(a.writers) as writers,torch.inference_mode():
        lookahead=min(2*a.batch,4000)
        pending={i:readers.submit(load,(rows[i],str(a.source),source_seal)) for i in range(lookahead)}
        for start in range(0,4000,a.batch):
            chunk=[pending.pop(i).result() for i in range(start,min(start+a.batch,4000))]
            for i in range(start+lookahead,min(start+lookahead+len(chunk),4000)):
                pending[i]=readers.submit(load,(rows[i],str(a.source),source_seal))
            fields=torch.from_numpy(np.stack([x[1] for x in chunk]))[:,None].to("cuda")
            up=F.interpolate(fields,(1024,1024),mode="bilinear",align_corners=False)[:,0]
            half=(up>.5).cpu().numpy()
            areas=[];receipts=[]
            for i,(row,field,expected) in enumerate(chunk):
                reproduced=np.packbits(half[i])
                mismatch=int(POPCOUNT[reproduced^expected].sum())
                if mismatch:raise RuntimeError("Source CUDA halfmask parity failed "+row["key"]+" pixels="+str(mismatch))
                area=int(half[i].sum())/1048576
                bin=int(np.searchsorted(edges,area,side="right")-1);level=float(levels[bin]);areas.append(level)
                receipts.append(dict(source_field=str((a.source/"fields"/(row["key"]+".npz")).resolve()),source_field_key="fine.rcg16.control",
                    source_field_sha256=source_seal["fields"][row["key"]],source_prediction_sha256=source_seal["predictions"][row["key"]],
                    packet_export=row["packet_export"],packet_sha256=source_seal["inputs"][row["key"]]["packet_sha256"],
                    source_halfmask_mismatched_pixels=0,area_half=area,size_bin=bin,cut_level=level,
                    source_tau=source_cfg["parameters"][str(row["fold"])]["tau"],uniform_primary_available=row["fold"] in (1,2)))
            cuts=torch.tensor(areas,device="cuda",dtype=torch.float32)[:,None,None]
            masks=(up>cuts).cpu().numpy()
            for i,(row,_,_) in enumerate(chunk):
                value=np.packbits(masks[i]);arms={CONTROL:value}
                if row["fold"] in (1,2):arms[PRIMARY]=value;matched.append(row)
                else:missing.append(dict(row,missing_reason="required uniformtau.15 field absent; existingtau.07 is control only"))
                pending_writes.append(writers.submit(save,row,arms,receipts[i]))
            while len(pending_writes)>2*a.writers:result.append(pending_writes.pop(0).result())
            if start==0 or (start+len(chunk))%400==0:
                state=dict(state="PURE_CUDA_SIZECUT_RENDER_RUNNING",completed=start+len(chunk),n=4000,pid=os.getpid(),seconds=time.monotonic()-begin,encoder_forwards=0)
                write(a.out/"state.json",state);print(json.dumps(state),flush=True)
        result.extend(f.result() for f in pending_writes)
    if len(matched)!=2000 or len(missing)!=2000:raise ValueError("Source coverage mismatch")
    receipts={k:r for k,_,r in result};hashes={k:h for k,h,_ in result}
    seal=dict(state="ALL_AVAILABLE_COMPOSITION_MASKS_SEALED",n=4000,control_n=4000,uniform_primary_available_n=2000,
        predictions=hashes,inputs=receipts,source_seal_sha256=config["source_seal_sha256"],config_sha256=sha(a.out/"config.json"),
        manifest_sha256=sha(a.out/"manifest.json"),query_truth_opened=False,all4000_source_halfmask_exact=True,
        seconds=time.monotonic()-begin,cuda_peak_bytes=int(torch.cuda.max_memory_allocated()))
    write(a.out/"sealed.json",seal)
    def maskrow(row):
        key=row["key"];return dict(row,field_source=receipts[key]["source_field"],field_key="fine.rcg16.control",field_sha256=receipts[key]["source_field_sha256"],
            prediction=str((a.out/"predictions"/(key+".npz")).resolve()),prediction_sha256=hashes[key],
            prediction_arms=[CONTROL,PRIMARY] if row["fold"] in (1,2) else [CONTROL],recipe_version=config["version"],source_tau=receipts[key]["source_tau"])
    write(a.out/"complete_mask_manifest.json",[maskrow(r) for r in rows])
    write(a.out/"primary_matching2000_manifest.json",[maskrow(r) for r in matched])
    write(a.out/"missing_uniform2000_manifest.json",missing)
    write(a.out/"source_receipt.json",dict(inputs=receipts,source_seal_sha256=config["source_seal_sha256"],frozen_sha256=config["frozen_sha256"]))
    write(a.out/"state.json",dict(state=seal["state"],completed=4000,n=4000,control_n=4000,primary_available_n=2000))
    print(json.dumps({k:v for k,v in seal.items() if k not in ("predictions","inputs")}),flush=True)


def score(a):
    rows,source_seal,_=source(a);seal=json.loads((a.out/"sealed.json").read_text())
    if seal["state"]!="ALL_AVAILABLE_COMPOSITION_MASKS_SEALED" or seal["n"]!=4000:raise ValueError("Seal all masks before GT")
    for name,key in (("config.json","config_sha256"),("manifest.json","manifest_sha256")):
        if sha(a.out/name)!=seal[key]:raise ValueError("Changed own sealed metadata")
    if sha(a.source/"sealed.json")!=seal["source_seal_sha256"]:raise ValueError("Changed source seal")
    if (a.out/"report.json").exists():raise FileExistsError("Never overwrite scored results")
    prior_path=a.prior/"episodes.jsonl";prior={r["key"]:r for r in [json.loads(l) for l in prior_path.read_text().splitlines() if l.strip()]}
    if len(prior)!=4000:raise ValueError("Require scored original4000 I/U")
    begin=time.monotonic();arrays={arm:[] for arm in (*BASE,CONTROL)};primary_arrays={arm:[] for arm in (*BASE,CONTROL,PRIMARY)}
    matched=[];records=[]
    def iu(mask,truth):return [int(POPCOUNT[mask&truth].sum()),int(POPCOUNT[mask|truth].sum())]
    for n,row in enumerate(rows,1):
        key=row["key"];own=a.out/"predictions"/(key+".npz");bp=a.source/"predictions"/(key+".npz")
        if sha(own)!=seal["predictions"][key] or sha(bp)!=source_seal["predictions"][key] or sha(row["packet_export"])!=seal["inputs"][key]["packet_sha256"]:
            raise ValueError("Changed mask/truth packet "+key)
        previous=prior[key]
        if any(row[k]!=previous[k] for k in ("fold","c")) or any(Path(row[k]).name!=Path(previous[k]).name for k in ("support","query")):
            raise ValueError("Prior scored episode identity changed")
        with np.load(row["packet_export"],allow_pickle=False) as z:truth=packed(z["truth"]).copy()
        with np.load(bp,allow_pickle=False) as z:masks={arm:packed(z[arm]).copy() for arm in BASE}
        with np.load(own,allow_pickle=False) as z:masks.update({arm:packed(z[arm]).copy() for arm in z.files})
        record=dict(row,iu={})
        for arm,mask in masks.items():
            value=iu(mask,truth);record["iu"][arm]=value
            if arm in BASE and value!=previous["iu"][arm]:raise ValueError("Exact original4000 baseline I/U parity failed")
            if arm!=PRIMARY:arrays[arm].append(value)
            if row["fold"] in (1,2):primary_arrays[arm].append(value)
        if row["fold"] in (1,2):matched.append(row)
        records.append(record)
        if n%800==0:print(json.dumps(dict(state="SEALED_GT_SCORE",completed=n,seconds=time.monotonic()-begin)),flush=True)
    arrays={k:np.array(v,np.int64) for k,v in arrays.items()};primary_arrays={k:np.array(v,np.int64) for k,v in primary_arrays.items()}
    report,draws=summarize(rows,arrays,{})
    matching_report,matching_draws=summarize(matched,primary_arrays,{})
    report.update(state="CACHED_COMPOSITION_SCORE_COMPLETE",config=json.loads((a.out/"config.json").read_text()),
        complete_control=CONTROL,provided_primary_complete4000=False,provided_primary_matching2000=matching_report,
        exact_all4000_six_baseline_IU_parity=True,all4000_source_halfmask_pixel_parity=True,
        remaining_required_fields="2000 fold0/3 uniformtau.15 fields/masks; boundary successor owns completion",seconds=time.monotonic()-begin,
        seal_sha256=sha(a.out/"sealed.json"),prior_episode_sha256=sha(prior_path))
    write(a.out/"report.json",report);np.save(a.out/"bootstrap_photo_draws.npy",draws);np.save(a.out/"matching2000_bootstrap_photo_draws.npy",matching_draws)
    np.savez_compressed(a.out/"counts.npz",**{"iu:"+arm:value for arm,value in arrays.items()},**{"matched2000_iu:"+arm:value for arm,value in primary_arrays.items()})
    (a.out/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    lines=["# Frozen size-cut composition: cached4000 control,matching2000 primary","",
        "Fresh600 globally fitted65.860048 recipe; not the quoted crossfold65.44. No4000 refit,encoder,image or sign change.","",
        "Complete4000 fold-temperature size-cut CONTROL scores:","",json.dumps(report["scores"],indent=2),"",
        "Matching2000 uniformtau.15 PRIMARY only (fold1/2),same-cohort comparisons:","",json.dumps(matching_report["scores"],indent=2),"",
        "The provided uniformtau.15 primary is not complete4000 until the successor supplies both missingfolds. Field/mask manifests preserve all draw identities and coverage."]
    (a.out/"report.md").write_text("\n".join(lines)+"\n")
    write(a.out/"score_state.json",dict(state=report["state"],n=4000,report_sha256=sha(a.out/"report.json")))
    print(json.dumps(dict(state=report["state"],scores=report["scores"],matching2000_scores=matching_report["scores"])),flush=True)


def validate(a):
    rows,seal,_=source(a);expected_recipe(a.frozen)
    if a.out.exists():raise FileExistsError("Fresh output required")
    for fold in range(4):
        row=next(r for r in rows if r["fold"]==fold)
        load((row,str(a.source),seal))
    if not (a.prior/"episodes.jsonl").exists():raise FileNotFoundError("Scored baseline receipt required")
    print("VALIDATION_OK_NO_GPU_ENCODER_OR_GT",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("validate","infer","score"))
    for name in ("source","frozen","out","prior"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--batch",type=int,default=16);p.add_argument("--readers",type=int,default=2);p.add_argument("--writers",type=int,default=2)
    a=p.parse_args()
    if not 1<=a.batch<=32 or not 1<=a.readers<=2 or not 1<=a.writers<=2:p.error("Bounded finalizer:batch1..32,readers/writers1..2")
    {"validate":validate,"infer":infer,"score":score}[a.stage](a)


if __name__=="__main__":main()
