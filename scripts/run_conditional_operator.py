#!/usr/bin/env python3
"""Choose effective additions/deletions of RCG using reference-only marginal value.

Fixed candidate families come from complete FoRIS, MEAN and Astra masks. Add and
remove sets are disjoint, their expected I/U changes are exact, and total edited
pixels are capped before label-free joint/greedy selection. CPU scoring opens GT
only after all final masks have been sealed. Raw DINO remains the accounting origin.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import run_calibrated_operator as calibrated
CONFIG = dict(name="conditional_effective_edits", max_edit_fraction=.05,
              budget_fractions=[.25, .5, 1.], current_combination="RCG",
              accounting_origin="stage:model.raw_nn",
              objective="maximize minimum predicted IoU gain from two reference-only estimators",
              query_gt_in_inference=False, additional_encoder_forwards=0,
              fallback="unchanged RCG when reference/query anchor calibration is unavailable",
              candidate_families="FoRIS/MEAN additions and Astra/MEAN/FoRIS deletions, plus agreement sets")
calibrated.CONFIG["conditional_selector"] = CONFIG


def value_fields(q, r, cov, masks, field):
    import numpy as np
    import torch
    from ics.experiment import unpack
    cover = lambda mask: unpack(mask).reshape(64, 16, 64, 16).mean((1, 3)).reshape(-1)
    source = "astra.control"
    candidates = np.flatnonzero(cover(masks[source]) > .5)
    if not len(candidates):
        source = "rcg"; candidates = np.flatnonzero(cover(masks[source]) > .5)
    fg_ids = candidates[np.argsort(-field.reshape(-1)[candidates], kind="stable")[:max(1, len(candidates)//2)]]
    outside = (cover(masks["native"]) == 0) & (cover(masks["rcg"]) == 0) & (cover(masks["mean.control"]) == 0)
    candidates = np.flatnonzero(outside)
    bg_ids = candidates[np.argsort(field.reshape(-1)[candidates], kind="stable")[:max(1, len(candidates)//2)]]
    fg = torch.zeros(4096, dtype=torch.bool, device="cuda"); fg[fg_ids] = True
    bg = torch.zeros_like(fg); bg[bg_ids] = True
    cross, info = calibrated.posterior(torch.as_tensor(q, device="cuda"), torch.as_tensor(r, device="cuda"), cov, None, fg, bg)
    info.update(anchor_source=source, query_foreground_anchors=len(fg_ids), query_background_anchors=len(bg_ids))
    if cross is None: return None, None, info
    consensus = np.stack([unpack(masks[name]) for name in ("native", "rcg", "astra.control", "mean.control")])
    lower, upper = float(consensus.all(0).mean()), float(consensus.any(0).mean())
    target = float(np.clip(float(cross.mean()), lower, upper))
    rank = torch.as_tensor(field.reshape(-1), device="cuda").clamp(1e-4, 1-1e-4)
    logits = torch.logit(rank); low, high = -30., 30.
    for _ in range(50):
        middle = (low+high)/2
        if float(torch.sigmoid(logits+middle).mean()) < target: low=middle
        else: high=middle
    rank = torch.sigmoid(logits+(low+high)/2)
    info.update(rank_mass_target=target, consensus_lower=lower, consensus_upper=upper)
    expand = lambda p: p.reshape(64, 64).repeat_interleave(16, 0).repeat_interleave(16, 1).reshape(-1)
    return expand(rank), expand(cross), info


def frontiers(domains, value, budget):
    import torch
    items=[dict(name="none", mask=torch.zeros_like(value, dtype=torch.bool), domain=None, pixels=0)]
    for name, domain in domains.items():
        ids=torch.nonzero(domain & (value > 0))[:, 0]
        if not len(ids): continue
        ids=ids[torch.argsort(value[ids], descending=True, stable=True)]
        sizes=sorted({min(len(ids), int(budget*f)) for f in CONFIG["budget_fractions"]} - {0})
        for size in sizes:
            mask=torch.zeros_like(domain); mask[ids[:size]]=True
            items.append(dict(name=name+":"+str(size), mask=mask, domain=name, pixels=size))
    return items


def solve(origin, probabilities, additions, deletions, budget, dual=True):
    import numpy as np
    import torch
    p=torch.stack(probabilities); total=p.sum(1)
    inter=(p*origin).sum(1); union=total+origin.sum()-inter; baseline=inter/union.clamp_min(1e-8)
    add_margin=p-baseline[:, None]*(1-p)
    delete_margin=-add_margin
    normalized_add=add_margin/union.clamp_min(1e-8)[:, None]
    normalized_delete=delete_margin/union.clamp_min(1e-8)[:, None]
    av=normalized_add.min(0).values if dual else normalized_add[0]
    dv=normalized_delete.min(0).values if dual else normalized_delete[0]
    A=frontiers(additions, av, budget);B=frontiers(deletions, dv, budget)
    am=torch.stack([a["mask"] for a in A]);bm=torch.stack([b["mask"] for b in B])
    if bool((am & origin).any()) or bool((bm & ~origin).any()):raise ValueError("Effective add/delete domains overlap")
    at=am.float() @ p.T;bt=bm.float() @ p.T
    an=am.sum(1);bn=bm.sum(1)
    expected_inter=inter[None,None]+at[:,None]-bt[None]
    expected_union=union[None,None]+(an[:,None]-at)[:,None]-(bn[:,None]-bt)[None]
    scores=expected_inter/expected_union.clamp_min(1e-8)-baseline[None,None]
    utility=scores.min(2).values if dual else scores[:,:,0]
    area=an[:,None]+bn[None];feasible=area <= budget
    utility=torch.where(feasible,utility,-torch.inf).cpu().numpy();area=area.cpu().numpy()
    def best(candidates):
        return max(candidates,key=lambda ij:(float(utility[ij]),-int(area[ij]),-int(ij[0]!=0)-int(ij[1]!=0)))
    joint=best([(i,j) for i in range(len(A)) for j in range(len(B)) if np.isfinite(utility[i,j])])
    add=best([(i,0) for i in range(len(A))]);delete=best([(0,j) for j in range(len(B))])
    forward=best([(add[0],j) for j in range(len(B)) if np.isfinite(utility[add[0],j])])
    reverse=best([(i,delete[1]) for i in range(len(A)) if np.isfinite(utility[i,delete[1]])])
    greedy=best([forward,reverse])
    if utility[joint]+1e-8 < utility[greedy]:raise ValueError("Joint/greedy containment failed")
    def mask(pair):return (origin | A[pair[0]]["mask"]) & ~B[pair[1]]["mask"]
    i,j=joint
    audit=dict(add=A[i]["name"],delete=B[j]["name"],add_pixels=A[i]["pixels"],delete_pixels=B[j]["pixels"],
        budget=budget,candidates=int(feasible.sum()),expected_gain=scores[i,j].cpu().tolist(),
        minimum_gain=float(utility[joint]),greedy_gain=float(utility[greedy]),raw_origin_unchanged=True)
    return dict(joint=mask(joint),greedy=mask(greedy),add_only=mask(add),delete_only=mask(delete)), audit, A[i],B[j]


def predict(q, r, cov, comparisons, field):
    import numpy as np
    import torch
    from ics.experiment import unpack
    masks={name:torch.as_tensor(unpack(mask).reshape(-1),device="cuda") for name,mask in comparisons.items()}
    origin=masks["rcg"];budget=int(int(origin.sum())*CONFIG["max_edit_fraction"])
    p1,p2,info=value_fields(q,r,cov,comparisons,field)
    names=("conditional.joint","conditional.greedy.control","conditional.add_only","conditional.delete_only", "conditional.same_count.control","conditional.rcg_value.control")
    if p1 is None or not budget:
        info.update(budget=budget,add_pixels=0,delete_pixels=0)
        return {name:comparisons["rcg"] for name in names},np.full(4096,.5,np.float32),info
    additions={"mean":masks["mean.control"] & ~origin,"foris":masks["native"] & ~origin}
    additions["agreement"]=additions["mean"] & additions["foris"]
    additions["union"]=additions["mean"] | additions["foris"]
    deletions={"astra":origin & ~masks["astra.control"],"mean":origin & ~masks["mean.control"],"foris":origin & ~masks["native"]}
    deletions["agreement"]=(deletions["astra"] & deletions["mean"]) | (deletions["astra"] & deletions["foris"]) | (deletions["mean"] & deletions["foris"])
    result,audit,a,b=solve(origin,[p1,p2],additions,deletions,budget,dual=True)
    simple,_,_,_=solve(origin,[p1,p2],additions,deletions,budget,dual=False)
    z=torch.as_tensor(field,device="cuda").repeat_interleave(16,0).repeat_interleave(16,1).reshape(-1)
    def matched(domain,count,descending):
        chosen=torch.zeros_like(origin)
        if count:
            ids=torch.nonzero(domain)[:,0];ids=ids[torch.argsort(z[ids],descending=descending,stable=True)[:count]];chosen[ids]=True
        return chosen
    ma=matched(additions[a["domain"]],a["pixels"],True) if a["domain"] else torch.zeros_like(origin)
    mb=matched(deletions[b["domain"]],b["pixels"],False) if b["domain"] else torch.zeros_like(origin)
    matched_mask=(origin|ma)&~mb
    output={"conditional.joint":result["joint"],"conditional.greedy.control":result["greedy"],
        "conditional.add_only":result["add_only"],"conditional.delete_only":result["delete_only"],
        "conditional.same_count.control":matched_mask,"conditional.rcg_value.control":simple["joint"]}
    info.update(audit)
    output={name:np.packbits(mask.cpu().numpy()) for name,mask in output.items()}
    return output,p1.reshape(64,16,64,16).mean((1,3)).reshape(-1).cpu().numpy(),info


def infer(a):
    import numpy as np
    import torch
    from ics.experiment import load_rows,load_inputs,sha
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.out.exists():raise FileExistsError("Fresh output required")
    seal=json.loads((a.family/"sealed.json").read_text());calibrated.check(a.family/"manifest.json",seal["manifest_sha256"])
    calibrated.check(a.family/"protocol.json",seal["protocol_sha256"]);source=json.loads((a.family/"protocol.json").read_text())
    all_rows=load_rows(a.family/"manifest.json")
    if len(all_rows)!=241:raise ValueError("Full DEV241 required")
    rows=([all_rows[0],next(r for r in all_rows if r["fold"]!=all_rows[0]["fold"])] if a.smoke else all_rows)
    recheck=source["sources"]["recheck"];calibrated.check(Path(recheck["path"])/recheck["seal_name"],recheck["seal_sha256"])
    a.out.mkdir();(a.out/"predictions").mkdir();(a.out/"posterior").mkdir()
    calibrated.write(a.out/"manifest.json",rows)
    protocol=dict(n=len(rows),config=calibrated.CONFIG,input_receipts={},origin=CONFIG["accounting_origin"],
        selection="reference-only conditional edit selection; no query labels or DEV fold fitting",
        source_code={str(Path(__file__).resolve()):sha(__file__),str(Path(calibrated.__file__).resolve()):sha(calibrated.__file__),str(REPO/"src/ics/experiment.py"):sha(REPO/"src/ics/experiment.py")},
        source_family_seal_sha256=sha(a.family/"sealed.json"),resolution=1024,seed=0,extra_encoder_forwards=0,query_gt_in_inference=False)
    calibrated.write(a.out/"protocol.json",protocol);hashes,posterior_hashes,audit={},{},{};begin=time.monotonic()
    for num,row in enumerate(rows,1):
        key=row["key"];(q,r,cov,score),receipt=load_inputs(a.root,row);protocol["input_receipts"][key]=receipt
        pp=a.family/"predictions"/(key+".npz");calibrated.check(pp,seal["predictions"][key])
        with np.load(pp,allow_pickle=False) as z:comparisons={name:z[name].copy() for name in ("native","origin","rcg","astra.control","mean.control","insid3.control")}
        fp=Path(recheck["path"])/"fields"/(key+".npz");calibrated.check(fp,recheck["fields"][key])
        with np.load(fp,allow_pickle=False) as z:field=z["rcg"].copy()
        output,prob,info=predict(q,r,cov,comparisons,field)
        dest=a.out/"predictions"/(key+".npz");np.savez_compressed(dest,**comparisons,**output);hashes[key]=sha(dest)
        dest=a.out/"posterior"/(key+".npz");np.savez_compressed(dest,probability=prob);posterior_hashes[key]=sha(dest)
        audit[key]=info;print(json.dumps(dict(n=num,total=len(rows),seconds=round(time.monotonic()-begin,1),add=info.get("add_pixels"),delete=info.get("delete_pixels"),fallback=info["fallback"])),flush=True)
    calibrated.write(a.out/"protocol.json",protocol);calibrated.write(a.out/"choices.json",audit)
    calibrated.write(a.out/"sealed.json",dict(state="ALL_PREDICTIONS_SEALED",n=len(rows),manifest_sha256=sha(a.out/"manifest.json"),
        protocol_sha256=sha(a.out/"protocol.json"),choices_sha256=sha(a.out/"choices.json"),predictions=hashes,posterior=posterior_hashes,
        query_gt_used_for_DEV_fold_fitting=False,per_query_gt_routing=False,seconds=time.monotonic()-begin,cuda_peak_bytes=torch.cuda.max_memory_allocated()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("phase",choices=("infer","score"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--family",type=Path);p.add_argument("--out",type=Path,required=True);p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.phase=="infer" and a.family is None:p.error("--family required")
    {"infer":infer,"score":calibrated.score}[a.phase](a)

if __name__=="__main__":main()
