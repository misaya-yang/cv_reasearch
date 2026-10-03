#!/usr/bin/env python3
"""Frozen single-reference readout audit; no fitting across episodes, mask method or encoder.

CUDA is optional arithmetic acceleration on the SAME saved tensors/readouts.
Query annotations are loaded only after every query score has been materialised.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPRESENTATIONS = ("raw_final", "pre_debias", "actual_debiased", "block12")
TAU = .1
SCORE_BIN = .05
CARD = [
    "Assumption: a specific legal reference FG/BG readout may rank missed targets above true background AND true foreground above false-positive background, beyond the original source-score ordering.",
    "Prediction: report old score and exact old NN margin alongside all three fixed readouts; both score-conditional FN/TN and TP/FP directions must improve with paired evidence before claiming a useful signal. No point forecast is derived from the unrelated241-task .664 AUC.",
    "Match: preserve the smallest successful readout/representation for a subsequent complete inference test; AUC is neither mIoU nor evidence that the existing extent decoder can use it.",
    "Mismatch: reject only this readout on this12-pair development cohort. Empty score overlap is INCONCLUSIVE. Do not infer absence of information in complete DINO vectors, layers or native relations, or restore a generic trained head."
]


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def auc(y, x):
    p = np.asarray(y, dtype=bool)
    n, m = int(p.sum()), int((~p).sum())
    if not n or not m:
        return None
    # Average tied ranks; pure NumPy because the preserved server runtime has no scipy.
    x=np.asarray(x);order=np.argsort(x,kind="stable");sx=x[order]
    starts=np.r_[0,np.flatnonzero(sx[1:]!=sx[:-1])+1]
    stops=np.r_[starts[1:],len(sx)]
    ranks=np.empty(len(x),float)
    ranks[order]=np.repeat((starts+stops+1)/2,stops-starts)
    return float((ranks[p].sum()-n*(n+1)/2)/(n*m))


def source_ref_grid(mask, shape):
    """Exact fixed FoRIS downsample_mask from source utils/data.py; reference-only."""
    a=mask.float().reshape(1,1,*mask.shape[-2:])
    down=F.interpolate(a,size=shape,mode="bilinear",align_corners=False)[0,0]>.5
    if not down.any():
        down=F.interpolate(a,size=shape,mode="nearest")[0,0]>.5
        if not down.any():
            coords=torch.argwhere(a[0,0]>0)
            if not len(coords):
                raise ValueError("No legal foreground reference")
            cy,cx=(coords.float().mean(0)/(a.shape[-1]//shape[1])).int()
            down[cy,cx]=True
    return down.flatten()


def tokens(t, prefix, grid):
    if t.ndim == 5:
        if t.shape[:2] != (1,2) or tuple(t.shape[-2:])!=tuple(grid):
            raise ValueError("Source map shape mismatch")
        return t[0].flatten(2).transpose(1,2)
    if t.ndim == 3 and t.shape[0] == 2:
        if t.shape[1] != prefix+grid[0]*grid[1]:
            raise ValueError("Native block patch count mismatch")
        return t[:,prefix:]
    raise ValueError("Unsupported native tensor shape")


def readouts(t, fg, chunk):
    # FP32 source values are preserved; L2 is the DECLARED cosine readout, not a new representation.
    z=F.normalize(t.float(),p=2,dim=-1)
    s,q=z[0],z[1]
    if not fg.any() or fg.all():
        raise ValueError("No legal two-class reference anchors")
    if not torch.isfinite(z).all():
        raise ValueError("Nonfinite cached vectors")
    mu=F.normalize(s[fg].mean(0),dim=0)-F.normalize(s[~fg].mean(0),dim=0)
    mean=(q@mu).detach().cpu().numpy()
    nn,lme=[],[]
    for i in range(0,len(q),chunk):
        sim=q[i:i+chunk]@s.T
        f,b=sim[:,fg],sim[:,~fg]
        nn.append((f.amax(1)-b.amax(1)).detach().cpu().numpy())
        lme.append((torch.logsumexp(f/TAU,dim=1)-np.log(f.shape[1])
                    -torch.logsumexp(b/TAU,dim=1)+np.log(b.shape[1])).detach().cpu().numpy())
    return dict(mean_margin=mean,nn_margin=np.concatenate(nn),equal_mass_lme=np.concatenate(lme))


def contrast(positive, negative, field, source_score):
    mask=positive|negative
    unconditional=auc(positive[mask],field[mask])
    strata=np.minimum((source_score/SCORE_BIN).astype(int),int(1/SCORE_BIN)-1)
    rows=[]
    for k in range(int(1/SCORE_BIN)):
        select=mask & (strata==k)
        np_,nn_=int((positive&select).sum()),int((negative&select).sum())
        if np_ and nn_:
            rows.append(dict(bin=k,n_positive=np_,n_negative=nn_,
                             auc=auc(positive[select],field[select])))
    return dict(n_positive=int(positive.sum()),n_negative=int(negative.sum()),
        unconditioned_auc=unconditional,score_conditioned_auc=None if not rows else float(np.mean([r["auc"] for r in rows])),
        matched_bins=len(rows),matched_positive=sum(r["n_positive"] for r in rows),matched_negative=sum(r["n_negative"] for r in rows),
        state="SCORE_OVERLAP_PRESENT" if rows else "INCONCLUSIVE_NO_SHARED_SCORE_STRATUM",bins=rows)


def partition(query, native, shape, grid):
    truth=F.interpolate(torch.from_numpy(query.copy()).float()[None,None],size=shape,mode="nearest")
    cov=F.interpolate(truth,size=grid,mode="area")[0,0].numpy().ravel()
    native_cov=F.interpolate(torch.from_numpy(native.copy()).float()[None,None],size=grid,mode="area")[0,0].numpy().ravel()
    fg,bg=cov>=.9,cov<=.1;pred=native_cov>=.5
    return fg&~pred,bg&pred,fg&pred,bg&~pred,int((~(fg|bg)).sum())


def selfcheck(out):
    torch.manual_seed(31063)
    checks=[]
    mask=torch.zeros(8,8,dtype=torch.bool);mask[:,:4]=True
    assert torch.equal(source_ref_grid(mask,(2,2)),torch.tensor([True,False,True,False]))
    checks.append("actual_source_bilinear_reference_grid")
    mask.zero_();mask[0,0]=True
    assert source_ref_grid(mask,(2,2)).tolist()==[True,False,False,False]
    checks.append("actual_source_nearest_fallback")
    mask.zero_();mask[3,3]=True
    assert source_ref_grid(mask,(2,2)).tolist()==[True,False,False,False]
    checks.append("actual_source_tiny_target_center_fallback")
    full=torch.arange(54).reshape(2,9,3)
    assert torch.equal(tokens(full,5,(2,2)),full[:,5:])
    checks.append("prefix_removed_without_redefining_native_block")
    t=torch.randn(2,5,7);fg=torch.tensor([True,True,False,False,False])
    one,many=readouts(t,fg,1),readouts(t,fg,4)
    assert all(np.allclose(one[k],many[k],atol=2e-6,rtol=2e-6) for k in one)
    checks.append("three_readouts_chunk_consistency")
    swapped=readouts(t,~fg,2)
    assert all(np.allclose(one[k],-swapped[k],atol=2e-6,rtol=2e-6) for k in one)
    checks.append("reference_label_swap_reverses_every_readout")
    perm=torch.tensor([3,0,4,1,2]);z=t.clone();z[0]=t[0,perm]
    shuffled=readouts(z,fg[perm],3)
    assert all(np.allclose(one[k],shuffled[k],atol=2e-6,rtol=2e-6) for k in one)
    checks.append("reference_order_invariance_with_labels_aligned")
    original_load=np.load
    def deny_labels(*args,**kwargs):raise AssertionError("readout attempted annotation access")
    np.load=deny_labels
    try:blind=readouts(t,fg,2)
    finally:np.load=original_load
    assert all(np.allclose(one[k],blind[k],atol=2e-6,rtol=2e-6) for k in one)
    query=np.array([[True,False,False],[False,False,False]])
    native=np.zeros((4,4),bool)
    fn,fp,tp,tn,mixed=partition(query,native,(4,4),(2,2))
    assert fn.tolist()==[True,False,False,False] and not fp.any() and not tp.any() and tn.sum()==3 and mixed==0
    checks.append("no_annotation_access_in_readouts_and_nearest_then_area_geometry")
    assert auc(np.array([True,False]),np.array([1.,1.]))==.5
    assert auc(np.array([True,False,True,False]),np.array([.1,.1,.2,.3]))==.375
    checks.append("auc_ties_are_half")
    pos=np.array([True,True,False,False]);neg=~pos;score=np.array([.8,.9,.1,.2])
    no_overlap=contrast(pos,neg,score,score)
    assert no_overlap["score_conditioned_auc"] is None and auc(pos,score)==1 and auc(pos,-score)==0
    checks.append("empty_score_overlap_inconclusive_and_inversion_harms_true_positive_direction")
    report=dict(state="PASSED",checks=checks,cases=len(checks),no_encoder=True,no_training=True,
                CUDA_initialized=torch.cuda.is_initialized(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if report["CUDA_initialized"]:raise AssertionError("CPU selfcheck initialized CUDA")
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


def finish_summary(rows):
    out={}
    names=sorted({k for r in rows for k in r["readouts"]})
    rng=np.random.default_rng(31061)
    # Resample connected photo groups, rather than treating repeated photographs as independent.
    parent=list(range(len(rows)));seen={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        for name in (r["row"]["support"],r["row"]["query"]):
            if name in seen:parent[find(i)]=find(seen[name])
            else:seen[name]=i
    groups={}
    for i in range(len(rows)):groups.setdefault(find(i),[]).append(i)
    gs=list(groups.values())
    for task in ("FN_vs_FP","FN_vs_TN","TP_vs_FP"):
        table={}
        for name in names:
            valid=[i for i,r in enumerate(rows) if name in r["readouts"] and r["readouts"][name][task]["score_conditioned_auc"] is not None]
            if not valid:
                table[name]=dict(state="INCONCLUSIVE_NO_SHARED_SCORE_STRATUM",pairs=0);continue
            values={i:rows[i]["readouts"][name][task]["score_conditioned_auc"] for i in valid}
            delta={i:values[i]-rows[i]["readouts"]["old_source_score"][task]["score_conditioned_auc"] for i in valid}
            boot=[]
            for _ in range(1000):
                ids=[i for g in rng.integers(0,len(gs),len(gs)) for i in gs[g] if i in delta]
                if ids:boot.append(float(np.mean([delta[i] for i in ids])))
            table[name]=dict(pairs=len(valid),macro_pair_conditioned_auc=float(np.mean(list(values.values()))),
                delta_vs_old_score=float(np.mean(list(delta.values()))),exploratory_photo_group_CI95=np.quantile(boot,[.025,.975]).tolist() if boot else None,
                per_fold={str(f):dict(pairs=sum(rows[i]["row"]["fold"]==f for i in valid),
                    delta=None if not any(rows[i]["row"]["fold"]==f for i in valid) else float(np.mean([delta[i] for i in valid if rows[i]["row"]["fold"]==f]))) for f in range(4)})
        out[task]=table
    return out,len(gs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path);p.add_argument("--out",type=Path)
    p.add_argument("--selfcheck-out",type=Path);p.add_argument("--attention-variation",action="store_true")
    p.add_argument("--representations",default=",".join(REPRESENTATIONS),
                   help="Comma-separated native representation keys, or 'none' for attention-only audit")
    p.add_argument("--old-packets",type=Path);p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--limit",type=int,default=12);p.add_argument("--chunk",type=int,default=256)
    p.add_argument("--max-seconds",type=float,default=300)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if a.selfcheck_out:selfcheck(a.selfcheck_out);return
    if not a.cache or not a.out:p.error("--cache and --out required for a real cache audit")
    native_names=[] if a.representations=="none" else a.representations.split(",")
    if any(x not in REPRESENTATIONS for x in native_names):p.error("Unsupported native representation")
    if not native_names and not a.attention_variation:p.error("No representations selected")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.device=="cuda":torch.cuda.set_per_process_memory_fraction(.3)
    start=time.monotonic();cache_report=json.loads((a.cache/"report.json").read_text())
    if cache_report["state"]!="COMPLETED" or not cache_report.get("query_labels_opened_after_all_feature_commits"):
        raise ValueError("Evidence cache is not frozen before annotations")
    rows=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    def save(state):
        summary,groups=finish_summary(rows) if rows and state!="RUNNING" else ({},0)
        report=dict(state=state,scope="12 reused development pairs,3perfold; pair means, not a benchmark score or independent confirmation",
            algorithm="Three fixed reference-only readouts, no cross-episode supervision, no parameter search, no encoder or new mask method.",
            device=a.device,threads=2,TF32=False,representations=native_names+attention_names,tau=TAU,score_bin=SCORE_BIN,
            card=CARD,pairs=len(rows),seconds=time.monotonic()-start,photo_groups=groups,
            conditional_scope="Within each pair, equal weight over occupied source-score bins; pairs equally weighted. .05 bins are approximate score conditioning. FN/FP may have no overlap and is not an increment claim.",
            labels="Query GT only AFTER all scores freeze; pure query patch coverage>=.9 or<=.1; native prediction patch coverage>=.5. Mixed GT patches excluded, counted separately.",
            false_shortcut="old_source_score_inverted is a CHEATING diagnostic for FN/FP only; reversing source score is NOT a correction mechanism. It also damages the other directions.",
            limitations="A negative fixed cosine/mean/NN/LME readout is NOT a DINO-information upper bound. Positive AUC is NOT mIoU or safe mask correction. Native block12 output is not silently final-normalized. Optional attention representations are explicitly DERIVED native self-attention U=AV versus V in the same V coordinates; no native cross-image attention is claimed.",
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=="cuda" else None,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary=summary,rows=rows)
        tmp=a.out.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.out)
    attention_names=[]
    with torch.inference_mode():
        for index in range(min(a.limit,12)):
            if time.monotonic()-start>a.max_seconds:save("STOPPED_FINITE_ARITHMETIC_BUDGET");return
            path=a.cache/f"pair_{index:02}.pt"
            pack=torch.load(path,map_location="cpu",weights_only=True)
            meta,features=pack["metadata"],pack["features"]
            grid=tuple(meta["grid"]);prefix=meta["prefix_tokens"]
            reference=source_ref_grid(features["reference_native_mask"],grid).to(a.device)
            score=features["original_score"].float().cpu().numpy().ravel()
            score=(score-score.min())/max(float(np.ptp(score)),1e-6)
            fields={"old_source_score":score.copy(),"old_source_score_inverted":-score.copy()}
            attention_audit=None
            for rep in native_names:
                t=tokens(features[rep],prefix,grid).to(a.device)
                for name,value in readouts(t,reference,a.chunk).items():fields[rep+"/"+name]=value
                del t
            if a.attention_variation:
                from native_attention_variation import derive_attention_representations, signed_value_average, get_last_audit
                extra=derive_attention_representations(pack,prefix,a.device)
                for rep,t in extra.items():
                    if tuple(t.shape)!=(2,grid[0]*grid[1],features["block12"].shape[-1]):
                        raise ValueError("Derived attention V-coordinate shape mismatch")
                    for name,value in readouts(t,reference,a.chunk).items():fields[rep+"/"+name]=value
                attention_names=list(extra)
                del extra,t
                linear=signed_value_average(pack,prefix,reference,a.device)
                for name,value in linear.items():
                    if tuple(value.shape)!=(grid[0]*grid[1],):
                        raise ValueError("Fixed-reference-V linear scalar shape mismatch")
                    fields[name]=value.detach().cpu().numpy()
                attention_audit=get_last_audit()
                attention_audit["helper_sha256"]=hashlib.sha256((Path(__file__).parent/"native_attention_variation.py").read_bytes()).hexdigest()
                del linear
            old_state="NOT_SUPPLIED_NO_EXACT_OLD_NN_CONTROL"
            if a.old_packets:
                r=meta["row"];old=a.old_packets/f'{r["fold"]}_{r["e"]}_{r["c"]}.npz'
                if not old.is_file():raise FileNotFoundError(str(old))
                with np.load(old) as z:
                    fields["old_packet_nn_margin"]=z["fg_max"].astype(float).ravel()-z["bg_max"].astype(float).ravel()
                old_state="EXACT_SAVED_OLD_PACKET_FG_MINUS_BG"
            if not all(np.isfinite(x).all() for x in fields.values()):raise ValueError("Nonfinite score, not information negative")
            frozen={name:array_sha(x) for name,x in fields.items()}
            # No query GT was touched above. The scores are now frozen.
            with np.load(a.cache/f"labels_{index:02}.npz") as lab:
                truth=torch.from_numpy(lab["query"].copy()).float()[None,None]
                shape=tuple(int(x) for x in lab["model_shape"])
                native=np.unpackbits(lab["native"])[:int(np.prod(shape))].reshape(shape)
            FN,FP,TP,TN,mixed=partition(truth[0,0].numpy(),native,shape,grid)
            contrasts={"FN_vs_FP":(FN,FP),"FN_vs_TN":(FN,TN),"TP_vs_FP":(TP,FP)}
            results={name:{task:contrast(pos,neg,x,score) for task,(pos,neg) in contrasts.items()} for name,x in fields.items()}
            fixed_sign={}
            for name,x in fields.items():
                threshold=.5 if name=="old_source_score" else (-.5 if name=="old_source_score_inverted" else 0.)
                positive=x>threshold
                fixed_sign[name]=dict(threshold=threshold,FN_recovered=int((FN&positive).sum()),
                    TN_falsely_added=int((TN&positive).sum()),FP_rejected=int((FP&~positive).sum()),
                    TP_core_rejected=int((TP&~positive).sum()),
                    scope="Readout-sign error-direction diagnostic on pure patches, NOT a complete mask correction or calibrated probability.")
            row=dict(index=index,row=meta["row"],grid=grid,n_reference_fg=int(reference.sum()),n_reference_bg=int((~reference).sum()),
                source_SAFR_applied=meta.get("actual_source_SAFR_applied"),native_rounding=meta.get("rounding"),
                source_native_identity=meta.get("matched_old_mask_exact"),old_nn_control=old_state,
                counts=dict(FN=int(FN.sum()),FP=int(FP.sum()),TP=int(TP.sum()),TN=int(TN.sum()),mixed_GT=mixed),
                no_queryGT_before_scores=True,frozen_score_sha256=frozen,readouts=results,fixed_sign_error_directions=fixed_sign,
                attention_derivation=attention_audit)
            rows.append(row);save("RUNNING")
            print(json.dumps(dict(pair=index,seconds=time.monotonic()-start,counts=row["counts"],
                source_score={k:v["score_conditioned_auc"] for k,v in results["old_source_score"].items()})),flush=True)
            del pack,features,reference,truth,fields
    if a.device=="cpu" and torch.cuda.is_initialized():raise ValueError("CPU audit initialized CUDA")
    save("COMPLETED_FIXED_READOUT_AUDIT")


if __name__=="__main__":main()
