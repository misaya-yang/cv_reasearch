#!/usr/bin/env python3
"""Fixed already-sealed complete methods versus DirectMEAN on its exact1200 draws.

No recipe selection, subset choice, encoder or field reconstruction. Full draw
identity includes public batch, e, fold, class and both photos. All source mask
and packet hashes pass before independent GT count/edit evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from ics.experiment import metric,photo_groups,sha

BASES=("native","rcg","mean.control","rcg64.control","fine.rcg16.control","fine.rcg64")
DIRECT="fine.mean16.control";GLOBAL="selected.global12";HELD="selected.heldfold12";GRAFT="scalar_graft.control"
ARMS=(*BASES,DIRECT,GRAFT,GLOBAL,HELD)
EDIT=("add_TP","add_FP","delete_TP","delete_FP")
POP=np.array([bin(x).count("1") for x in range(256)],np.uint8)
GLOBAL_GRAFT_ID="mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1"


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def checked(p,h):
    if sha(p)!=h:raise ValueError("Changed sealed source: "+str(p))
def identity(r):return(int(r["public_batch"]),int(r["fold"]),int(r["e"]),int(r["c"]),r["support"],r["query"])
def public_key(r):return "public%d:%s"%(r["public_batch"],r.get("source_key",r["key"]))
def packed(v):
    if v.dtype!=np.uint8 or v.shape!=(131072,):raise ValueError("Expected packed complete1024 mask")
    return v


def evaluate_one(job):
    row,paths,global_arm,expected=job
    with np.load(paths["mean"],allow_pickle=False) as z:masks={a:packed(z[a].copy()) for a in (*BASES,DIRECT)}
    with np.load(paths["old"],allow_pickle=False) as z:
        if any(not np.array_equal(masks[a],z[a]) for a in BASES):raise ValueError("Original six-arm mask bit parity failed")
    with np.load(paths["global"],allow_pickle=False) as z:masks[GLOBAL]=packed(z[global_arm].copy())
    with np.load(paths["held"],allow_pickle=False) as z:masks[HELD]=packed(z["recipe0"].copy())
    with np.load(paths["graft"],allow_pickle=False) as z:masks[GRAFT]=packed(z["mean_fine_residual_transfer_v1"].copy())
    with np.load(row["packet_export"],allow_pickle=False) as z:truth=packed(z["truth"].copy())
    count=lambda v:int(POP[v].sum(dtype=np.int64))
    iu={a:[count(m&truth),count(m|truth)] for a,m in masks.items()}
    if any(iu[a]!=expected[a] for a in ARMS):raise ValueError("Existing sealed method I/U differs on exact draw: "+row["key"])
    edits={}
    for a,m in masks.items():
        add,delete=m&~masks["native"],masks["native"]&~m
        edits[a]=[count(add&truth),count(add&~truth),count(delete&truth),count(delete&~truth)]
        if iu[a]!=[iu["native"][0]+edits[a][0]-edits[a][2],iu["native"][1]+edits[a][1]-edits[a][3]]:raise ValueError("Native four-edit I/U closure failed")
    paired={}
    for a in (GLOBAL,HELD):
        add,delete=masks[a]&~masks[DIRECT],masks[DIRECT]&~masks[a]
        paired[a]=[count(add&truth),count(add&~truth),count(delete&truth),count(delete&~truth)]
    return dict(key=row["key"],iu=iu,edits_vs_native=edits,edits_vs_direct_mean=paired)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mean",type=Path,default=Path("outputs/fine_mean1200_v1"));p.add_argument("--global-run",type=Path,default=Path("outputs/exact_family_selection_v1/public4000_v3"));p.add_argument("--held",type=Path,default=Path("outputs/exact_family_selection_v1/public4000_v3_crossfold"));p.add_argument("--out",type=Path,default=Path("outputs/selected_composition_vs_direct_mean1200_v1"));p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    if a.out.exists():raise FileExistsError("Fresh comparison output required")
    start=time.monotonic();mean=a.mean.resolve();glob=a.global_run.resolve();held=a.held.resolve()
    ms,mc,mr,mstate=[read(mean/name) for name in ("sealed.json","config.json","report.json","score_state.json")]
    if ms["state"]!="ALL_PREDICTIONS_AND_KERNELS_SEALED" or ms["n"]!=1200 or mstate["state"]!="CPU_SCORE_COMPLETE" or mstate["n"]!=1200:raise ValueError("Completed original DirectMEAN1200 required")
    checked(mean/"manifest.json",ms["manifest_sha256"]);checked(mean/"config.json",ms["config_sha256"])
    rows=read(mean/"manifest.json");n=len(rows)
    if n!=1200 or len({r["key"] for r in rows})!=n or len({identity(r) for r in rows})!=n or any(sum(r["fold"]==f for r in rows)!=300 for f in range(4)):raise ValueError("Require all original1200 unique draws/300perfold")
    gs,gr,gm=[read(glob/name) for name in ("finalists_sealed.json","report.json","manifest.json")]
    hs,hr,hm=[read(held/name) for name in ("sealed.json","report.json","manifest.json")]
    if gs["state"]!="ALL_FINALIST_MASKS_SEALED" or gs["n"]!=4000 or hs["state"]!="ALL_HELDFOLD_MASKS_SEALED" or hs["n"]!=4000:raise ValueError("Existing full4000 global/held masks must seal")
    checked(glob/"manifest.json",gs["manifest_sha256"]);checked(glob/"final_recipes.json",gs["recipes_sha256"]);checked(held/"manifest.json",hs["manifest_sha256"]);checked(held/"selection.json",hs["selection_sha256"])
    if gm!=hm:raise ValueError("Global/held original4000 library identity differs")
    if gr["selected_global_DEV_recipes"]!=["recipe3"]:raise ValueError("Actual fixed main global arm must be recipe3; never reselect")
    global_arm="recipe3";full=gm["rows"]
    if len(full)!=4000 or len({identity(r) for r in full})!=4000 or len({r["key"] for r in full})!=4000:raise ValueError("Ambiguous original4000 full draw identity")
    lookup={identity(r):(i,r) for i,r in enumerate(full)};mapping=[];used=set()
    for row in rows:
        key=identity(row)
        if key not in lookup:raise ValueError("OriginalMean1200 draw absent from full4000")
        i,match=lookup[key];public=public_key(row)
        if match["key"]!=public or i in used or public in {m["public_key"] for m in mapping}:raise ValueError("Duplicate/ambiguous public draw mapping")
        if row["packet_export"]!=match["packet"]["path"]:raise ValueError("Exact draw packet path differs")
        used.add(i);mapping.append(dict(mean_key=row["key"],public_key=public,original4000_index=i,identity=list(key)))
    with np.load(mean/"counts.npz",allow_pickle=False) as z:mean_counts={x:z["iu:"+x].copy() for x in (*BASES,DIRECT)}
    with np.load(glob/"final_counts.npz",allow_pickle=False) as z:global_counts={x:z[x].copy() for x in (global_arm,GLOBAL_GRAFT_ID,*["frozen_subtoken4000_v1::"+v for v in BASES])}
    with np.load(held/"counts.npz",allow_pickle=False) as z:held_counts=z["heldfold_recipe"].copy()
    receipts={};jobs=[]
    for j,(row,mapping_row) in enumerate(zip(rows,mapping)):
        i,public=mapping_row["original4000_index"],mapping_row["public_key"];match=full[i]
        original=match["masks"]["frozen_subtoken4000_v1::native"];graft=match["masks"][GLOBAL_GRAFT_ID]
        paths={"mean":str(mean/"predictions"/(row["key"]+".npz")),"old":original["path"],"global":str(glob/"finalists"/(public+".npz")),"held":str(held/"finalists"/(public+".npz")),"graft":graft["path"]}
        hashes={"mean":ms["predictions"][row["key"]],"old":original["sha256"],"global":gs["predictions"][public],"held":hs["predictions"][public],"graft":graft["sha256"],"packet":match["packet"]["sha256"]}
        if ms["inputs"][row["key"]]["packet_sha256"]!=hashes["packet"]:raise ValueError("OriginalMean/complete-method query truth identity differs")
        for name,path in paths.items():checked(path,hashes[name])
        checked(row["packet_export"],hashes["packet"])
        expected={x:mean_counts[x][j].tolist() for x in (*BASES,DIRECT)}
        for x in BASES:
            if not np.array_equal(mean_counts[x][j],global_counts["frozen_subtoken4000_v1::"+x][i]):raise ValueError("Original six-arm matched I/U parity failed")
        expected.update({GLOBAL:global_counts[global_arm][i].tolist(),HELD:held_counts[i].tolist(),GRAFT:global_counts[GLOBAL_GRAFT_ID][i].tolist()})
        jobs.append((row,paths,global_arm,expected));receipts[row["key"]]=dict(mapping_row,paths=paths,sha256=hashes)
    source_files={str(path/name):sha(path/name) for path,names in ((mean,["sealed.json","manifest.json","config.json","report.json","counts.npz","score_state.json"]),(glob,["finalists_sealed.json","final_recipes.json","report.json","final_counts.npz","manifest.json"]),(held,["sealed.json","selection.json","report.json","counts.npz","manifest.json"])) for name in names}
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn")) as pool:records=list(pool.map(evaluate_one,jobs,chunksize=4))
    arrays={arm:np.asarray([r["iu"][arm] for r in records],np.int64) for arm in ARMS};cls=np.array([r["c"] for r in rows]);groups=photo_groups(rows);g=int(groups.max())+1
    draws=np.random.RandomState(0).randint(g,size=(2000,g));weights=np.stack([np.bincount(d,minlength=g) for d in draws])[:,groups]
    if not np.array_equal(draws,np.load(mean/"bootstrap_photo_draws.npy",allow_pickle=False)):raise ValueError("OriginalMean1200 photograph bootstrap draws differ")
    scores={arm:metric(v,cls) for arm,v in arrays.items()};samples={arm:np.array([metric(v,cls,w) for w in weights]) for arm,v in arrays.items()}
    if any(abs(scores[x]-mr["scores"][x])>1e-10 for x in (*BASES,DIRECT)):raise ValueError("DirectMean/original6 class scores do not reproduce")
    comparisons={};compare_bases=(DIRECT,"native","fine.rcg16.control","fine.rcg64",GRAFT)
    for method in (GLOBAL,HELD):
        comparisons[method]={}
        for base in compare_bases:
            delta=arrays[method][:,0]/np.maximum(arrays[method][:,1],1)-arrays[base][:,0]/np.maximum(arrays[base][:,1],1)
            comparisons[method][base]=dict(gain=scores[method]-scores[base],ci95=np.percentile(samples[method]-samples[base],[2.5,97.5]).tolist(),up=int((delta>1e-12).sum()),down=int((delta< -1e-12).sum()),tie=int((np.abs(delta)<=1e-12).sum()))
    folds={}
    for fold in range(4):
        ix=np.array([r["fold"]==fold for r in rows]);point={arm:metric(v[ix],cls[ix]) for arm,v in arrays.items()};sub={arm:np.array([metric(v[ix],cls[ix],w[ix]) for w in weights]) for arm,v in arrays.items()}
        folds[str(fold)]=dict(n=int(ix.sum()),scores=point,comparisons={method:{base:dict(gain=point[method]-point[base],ci95=np.percentile(sub[method]-sub[base],[2.5,97.5]).tolist()) for base in compare_bases} for method in (GLOBAL,HELD)})
    edits={arm:np.asarray([r["edits_vs_native"][arm] for r in records],np.int64) for arm in ARMS};paired_edits={arm:np.asarray([r["edits_vs_direct_mean"][arm] for r in records],np.int64) for arm in (GLOBAL,HELD)}
    for path,digest in source_files.items():checked(path,digest)
    a.out.mkdir(parents=True);write(a.out/"manifest.json",rows);write(a.out/"source_receipt.json",dict(files=source_files,draws=receipts,all1200_sources_hashed_before_GT=True,original1200_order_retained=True,original6_IU_and_mask_bit_parity_exact=True))
    report=dict(state="MATCHED_DIRECT_MEAN1200_COMPLETE_COMPARISON",n=n,classes=len(set(cls)),photo_groups=g,scores=scores,comparisons=comparisons,folds=folds,
        global_recipe=gr["recipes"][3],heldfold_recipes=hr["selection"],four_edits_vs_native={arm:{name:int(edits[arm][:,i].sum()) for i,name in enumerate(EDIT)} for arm in ARMS},
        four_edits_vs_direct_mean={arm:{name:int(paired_edits[arm][:,i].sum()) for i,name in enumerate(EDIT)} for arm in (GLOBAL,HELD)},
        original6_baseline_IU_all1200_exact=True,all_fixed_global_held_graft_IU_source_counts_exact=True,all1200_draw_identity_matches_exact=True,
        bootstrap=dict(draws=2000,rng="RandomState(0)",unit="original1200 connected support/query photographs",folds="same full1200 photo multiplicities"),
        exposure="Same exact original1200 exposed development draws; global recipe selected using whole4000 GT; held recipes select other folds but source parameter exposure remains. No per-example GT routing. This does not replace pending full4000 DirectMEAN control.",
        raw_origin="Full1200 raw origin not available; edits retain actual complete-native or DirectMEAN origin",new_recipe_selection=False,new_encoder=False,
        source_receipt_sha256=sha(a.out/"source_receipt.json"),script_sha256=sha(Path(__file__)),seconds=time.monotonic()-start)
    write(a.out/"report.json",report);np.save(a.out/"bootstrap_photo_draws.npy",draws)
    np.savez_compressed(a.out/"counts.npz",**{"iu:"+x:v for x,v in arrays.items()},**{"edits_native:"+x:v for x,v in edits.items()},**{"edits_direct_mean:"+x:v for x,v in paired_edits.items()})
    (a.out/"episodes.jsonl").write_text("".join(json.dumps(dict(row,public_draw=mapping[j]["public_key"],**records[j]))+"\n" for j,row in enumerate(rows)))
    lines=["# Complete selected methods versus DirectMEAN: exact original1200","","Original1200 order, public batch/e/class/photos and packets are identical; no draw deduplication or recipe substitution. All original six-arm mask/I/U parity passes.","","| Arm | Class-summed macro mIoU |","|---|---:|"]
    lines += [f"| {arm} | {scores[arm]:.6f} |" for arm in ARMS]
    lines += ["","| Fixed method minus comparator | Gain [paired95% CI], pp |","|---|---:|"]
    for method,table in comparisons.items():
        for base,r in table.items():lines.append(f"| {method} minus {base} | {r['gain']:+.6f} [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] |")
    lines += ["","Original1200 connected-photo2000 RandomState0 intervals; fold intervals/four-edits/source hashes remain in JSON. Global and heldfold recipes retain their actual4000 selection provenance. Intervals crossing0 remain unresolved; reused1200 is not fresh confirmation or a replacement for the pending4000 DirectMEAN comparison."]
    (a.out/"report.md").write_text("\n".join(lines)+"\n");write(a.out/"state.json",dict(state=report["state"],n=n,report_sha256=sha(a.out/"report.json")))
    print(json.dumps(dict(scores=scores,comparisons=comparisons,seconds=report["seconds"])),flush=True)


if __name__=="__main__":main()
