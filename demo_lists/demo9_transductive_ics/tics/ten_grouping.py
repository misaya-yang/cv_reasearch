"""Methods03/06 ONLY: reference-conditioned tree grouping and independent multi-seed prior.
Pure pair-local inference and actual public-FoRIS replay hooks. No backbone calls,
query labels, disk feature caches, training, downloads or source modifications.
"""
from types import SimpleNamespace
import time
import numpy as np
import torch
import torch.nn.functional as F

LAMBDA_GRID=(0.,.25,1.,4.,16.)
TEMPERATURE=.1
MAX_SEEDS=3
EPS=1e-5
MODULE_BUDGET={"m03":dict(extra_encoder_B2=0,max_pair_seconds=90,max_points=4096,query_trees=2,reference_trees=2,persistent_tensor_bytes=0),
               "m06":dict(extra_encoder_B2=0,max_pair_seconds=40,max_points=4096,max_seeds=3,persistent_tensor_bytes=0)}
for _budget in MODULE_BUDGET.values():
    _budget.update(batch_hard_cap_seconds=1800,batch_cap_overrides_arm_sum=True,
        timeout_outcome="INCOMPLETE_RESOURCE_BUDGET_NOT_METHOD_FAILURE",
        arm_limits_are_not_measured_runtime=True)

class GroupingBudgetIncomplete(RuntimeError):
    """Administrative incompletion; never a scientific negative result."""
    outcome="INCOMPLETE_RESOURCE_BUDGET"
CARDS03=[
 "Assumption: full source hierarchy contains useful subregions that fixedtau grouping incorrectly mixes; legal reference evidence can select split cost.",
 "Prediction: method must improve complete native mask and beat SAME evidence/tree fixedlambda1 naive; old truck mixed-group witness is a premise, not a numeric gain prediction. Both seed/semantic source recomputation and nativeidentity control must pass.",
 "Match: freeze the smallest successful tree rule for independent confirmation; treeGT oracle only diagnoses structure and is never a deployment score.",
 "Mismatch: if complete method fails, inspect whether reference heldoutlambda, query evidence or hierarchy lacks separation; no GTlambda sweep, no direct tree-label final mask."
]
CARDS06=[
 "Assumption: multiple spatially separate reference-supported groups explain true instances that cannot inherit the single native seed's appearance.",
 "Prediction: K1 must return original prior/wholemask BITEXACT; independent max(K<=3) must beat native and same-budget naive multi-seed WITHOUT identity support while preserving original BG/consistency. No point forecast from oracle gaps.",
 "Match: freeze K3, identity>BG and quarter-grid spatialNMS; check all added/lost TP/FP and multiinstance versus singleinstance after outputs freeze.",
 "Mismatch: additional seeds that increase false regions invalidate this fixed independent-localisation construct; do not remove original seed penalty or add GT-tuned weights."
]

def _scope(ctx):
    return type(ctx.host)._build_seed_cluster_prior.__globals__

def _sf(ctx):
    return ctx.native["stages"]["part2_sf"].reshape(ctx.grid_hw)

def _margin(tokens,fg,bg):
    if not bool(fg.any()) or not bool(bg.any()):return None
    plus=F.normalize(tokens[fg].mean(0),dim=0)
    minus=F.normalize(tokens[bg].mean(0),dim=0)
    return tokens@(plus-minus)

def full_average_tree(values,tau):
    """Same metric/linkage/input as actual public agglomeration, full children retained."""
    from sklearn.cluster import AgglomerativeClustering
    n=len(values)
    if n>4096:raise ValueError("frozen 64square maximum hierarchy budget")
    if n==1:return np.empty((0,2),np.int64),np.zeros(1,np.int64)
    distance=(1-(values@values.T).clamp(-1,1)).detach().cpu().numpy()
    model=AgglomerativeClustering(n_clusters=None,metric="precomputed",linkage="average",
        distance_threshold=float(1-tau),compute_full_tree=True)
    labels=model.fit_predict(distance)
    children=np.asarray(model.children_,dtype=np.int64).copy()
    if children.shape!=(n-1,2):raise ValueError("complete binary source hierarchy required")
    return children,np.asarray(labels,np.int64)

def tree_dp(children,probability,split_cost):
    """C(R,y) sum NLL; compare stop0/stop1/split. Returns GROUPS, not a final mask."""
    p=np.asarray(probability,dtype=np.float64).reshape(-1)
    n=len(p);ch=np.asarray(children,dtype=np.int64)
    if not n or ch.shape!=(n-1,2) or not np.isfinite(p).all():raise ValueError("full tree/finite p")
    if not np.isfinite(split_cost) or split_cost<0:raise ValueError("nonnegative frozenlambda")
    p=np.clip(p,EPS,1-EPS)
    c0=np.zeros(2*n-1);c1=c0.copy();best=c0.copy()
    split=np.zeros(2*n-1,bool);kind=np.zeros(2*n-1,np.int8)
    c0[:n]=-np.log1p(-p);c1[:n]=-np.log(p)
    best[:n]=np.minimum(c0[:n],c1[:n]);kind[:n]=(c1[:n]<c0[:n])
    for offset,(left,right) in enumerate(ch):
        node=n+offset
        if left<0 or right<0 or left>=node or right>=node or left==right:raise ValueError("invalid source hierarchy")
        c0[node]=c0[left]+c0[right];c1[node]=c1[left]+c1[right]
        stop=min(c0[node],c1[node]);divide=best[left]+best[right]+split_cost
        split[node]=divide<stop-1e-12
        best[node]=divide if split[node] else stop;kind[node]=c1[node]<c0[node]
    groups=np.full(n,-1,np.int64);calibration_labels=np.zeros(n,bool)
    frontier=[];queue=[2*n-2]
    while queue:
        node=queue.pop()
        if node>=n and split[node]:
            left,right=ch[node-n];queue.extend([int(right),int(left)])
        else:frontier.append(node)
    for label,node in enumerate(frontier):
        stack=[int(node)]
        while stack:
            current=stack.pop()
            if current<n:
                if groups[current]!=-1:raise ValueError("leaf reused")
                groups[current]=label;calibration_labels[current]=bool(kind[node])
            else:stack.extend(map(int,ch[current-n]))
    if np.any(groups<0):raise ValueError("unassigned tree leaf")
    return groups,calibration_labels,dict(groups=len(frontier),nodes=2*n-1,split_cost=float(split_cost),
        energy=float(best[-1]),output_role="partition only; calibration labels never become querymask")

def _reference_values(ctx,stage):
    x=ctx.ref_tokens
    if stage!="seed":return x
    scope=_scope(ctx);h,w=ctx.grid_hw
    low=F.interpolate(ctx.support_work,size=(h,w),mode="bilinear",align_corners=False)[0]
    rgb=scope["denormalize"](low).clamp(0,1).flatten(1).T
    color=F.normalize(rgb,p=2,dim=1)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device,dtype=x.dtype),
        torch.linspace(-1,1,w,device=x.device,dtype=x.dtype),indexing="ij")
    pos=F.normalize(torch.stack([yy,xx],dim=-1).reshape(-1,2),p=2,dim=1)
    return F.normalize(torch.cat([x,color*.35,pos*.20],dim=1),p=2,dim=1)

def calibrate_reference_lambda(ctx,stage):
    """Only LEGAL support labels; fixed four spatial folds, balanced holdout classification error.
    Base feature/sourcePart1 is fixed by full legal support. This is per-task calibration,
    not independent conformal data. Semantic reference tree uses ungated nativePart1 reference.
    """
    cache=getattr(ctx,"_ten_grouping_reference_calibration",{})
    if stage in cache:return cache[stage]
    h,w=ctx.grid_hw
    hard=(ctx.reference_coverage.reshape(-1)>.5)
    pureF=ctx.reference_coverage.reshape(-1)>=.9;pureB=ctx.reference_coverage.reshape(-1)<=.1
    yy,xx=torch.meshgrid(torch.arange(h,device=hard.device),torch.arange(w,device=hard.device),indexing="ij")
    folds=((yy//max(1,h//4)+xx//max(1,w//4))%4).flatten()
    refvalues=_reference_values(ctx,stage)
    children,_=full_average_tree(refvalues,float(ctx.host.tau))
    x=ctx.ref_tokens;records=[];acc={v:[] for v in LAMBDA_GRID}
    for fold in range(4):
        held=folds==fold;train=~held
        if not bool((held&hard).any()) or not bool((held&~hard).any()):continue
        margin=_margin(x,train&pureF,train&pureB)
        if margin is None:continue
        # Both reference and query use same fixed readout mix. Reference sf proxy is
        # a foreground-unit-mean response, minmax over UNLABELED support features.
        mu=F.normalize(x[train&pureF].mean(0),dim=0)
        foreground=x@mu
        sf=(foreground-foreground.min())/(foreground.max()-foreground.min()).clamp_min(1e-6)
        p=.5*sf+.5*torch.sigmoid(margin/TEMPERATURE)
        p=p.detach().cpu().numpy();y=hard.cpu().numpy();val=held.cpu().numpy()
        row=dict(spatial_fold=fold)
        for lam in LAMBDA_GRID:
            _,guess,_=tree_dp(children,p,lam)
            error=.5*np.mean(guess[val&y]!=y[val&y])+.5*np.mean(guess[val&~y]!=y[val&~y])
            acc[lam].append(float(error));row[str(lam)]=float(error)
        records.append(row)
    if records:
        lam=min(LAMBDA_GRID,key=lambda t:(float(np.mean(acc[t])),-t))
        state="REFERENCE_ONLY_SPATIAL_HOLDOUT"
    else:
        lam=1.;state="FROZEN_DEFAULT1_NO_TWO_CLASS_SPATIAL_HOLDOUT"
    audit=dict(state=state,lambda_value=float(lam),lambda_grid=list(LAMBDA_GRID),records=records,
        criterion="support-only balanced error, largerlambda breaks equal errors",
        pure_anchor_rules=[.9,.1],temperature=TEMPERATURE,feature_note="nativePart1 support features already use alllegal reference; not independent calibration")
    result=(float(lam),audit);cache[stage]=result;ctx._ten_grouping_reference_calibration=cache
    return result

def _query_probability(ctx):
    coverage=ctx.reference_coverage.flatten()
    fg=coverage>=.9;bg=coverage<=.1
    if not bool(fg.any()) or not bool(bg.any()):return None
    x=ctx.ref_tokens;q=ctx.query_tokens
    muF=F.normalize(x[fg].mean(0),dim=0);muB=F.normalize(x[bg].mean(0),dim=0)
    margin=q@(muF-muB)
    return (.5*_sf(ctx).flatten()+.5*torch.sigmoid(margin/TEMPERATURE)).clamp(EPS,1-EPS)

def _tree_run(ctx,calibrate=True):
    started=time.monotonic()
    audit={"id":"m03" if calibrate else "m03_fixed_lambda_naive","stages":{},
        "no_query_GT":True,"no_encoder_calls":True,"final_mask_from_actual_public_source":True}
    p=_query_probability(ctx)
    if p is None:
        out=ctx.replay();out["audit"]={**out.get("audit",{}),**audit,"state":"NATIVE_NO_TWO_PURE_ANCHOR_CLASSES"}
        return out
    # Small exact tree cache is context-local, RAM only. No raw tensor written.
    cache=getattr(ctx,"_ten_grouping_query_tree_cache",{})
    def override(values,labels,stage,ctx):
        if time.monotonic()-started>90:raise GroupingBudgetIncomplete("m03 finite90s pair budget")
        key=stage;entry=cache.get(key)
        if entry is None or not torch.equal(entry["values"],values):
            children,replica=full_average_tree(values,float(ctx.host.tau))
            if not np.array_equal(replica,labels.detach().cpu().numpy()):
                raise RuntimeError("full source tree replay changes native partition")
            entry=dict(values=values.detach().clone(),children=children);cache[key]=entry
        lam,cal=calibrate_reference_lambda(ctx,stage) if calibrate else (1.,dict(state="FIXED_LAMBDA1_NAIVE"))
        if time.monotonic()-started>90:raise GroupingBudgetIncomplete("m03 referencecal exceeded finite90s")
        replacement,unused_labels,details=tree_dp(entry["children"],p.detach().cpu().numpy(),lam)
        audit["stages"][stage]={**details,"calibration":cal,"native_groups":int(labels.max())+1,
            "native_tree_replay_exact":True,"query_tree_labels_not_used_as_final_mask":True}
        return torch.from_numpy(replacement).to(device=labels.device,dtype=labels.dtype)
    out=ctx.replay(group_override=override);ctx._ten_grouping_query_tree_cache=cache
    out["audit"]={**out.get("audit",{}),**audit}
    return out

def method03(ctx):return _tree_run(ctx,True)
def naive03(ctx):return _tree_run(ctx,False)

def _seed_components(ctx,kw):
    scope=_scope(ctx);h,w=kw["h"],kw["w"];feat=kw["tgt_feat"][0]
    x=F.normalize(feat.reshape(feat.shape[0],-1).T,p=2,dim=1)
    labels=ctx.native["stages"]["seed_cluster_labels"].flatten().to(device=x.device,dtype=torch.long)
    if len(labels)!=h*w:raise ValueError("actual native seed cluster layout")
    k=int(labels.max())+1
    prototypes=scope["compute_cluster_prototypes"](x,labels,K=k)
    refFG=[];refBG=[]
    for r in range(kw["n_refs"]):
        mask=scope["downsample_mask"](kw["ref_masks"][r:r+1],h,w)
        source=kw["ref_feats"][0,r]
        if bool(mask.any()):refFG.append(source[:,mask].mean(1))
        if bool((~mask).any()):refBG.append(source[:,~mask].mean(1))
    if not refFG:return None
    muFG=F.normalize(torch.stack(refFG).mean(0),p=2,dim=0)
    muBG=None if not refBG else F.normalize(torch.stack(refBG).mean(0),p=2,dim=0)
    matched=labels[kw["candidate_mask"].flatten()]
    if not len(matched):return None
    ids,counts=matched.unique(return_counts=True)
    area=torch.bincount(labels,minlength=k).to(prototypes.dtype).clamp_min(1)
    weights=torch.zeros(k,device=x.device,dtype=prototypes.dtype);weights[ids]=counts.to(weights.dtype);weights/=area
    fg_map=torch.einsum("chw,c->hw",feat,muFG).flatten()
    cross=torch.zeros(k,device=x.device,dtype=prototypes.dtype)
    centers=torch.zeros(k,2,device=x.device,dtype=prototypes.dtype)
    yy,xx=torch.meshgrid(torch.arange(h,device=x.device,dtype=x.dtype),torch.arange(w,device=x.device,dtype=x.dtype),indexing="ij")
    coordinates=torch.stack([yy,xx],dim=-1).reshape(-1,2)
    for group in range(k):
        take=labels==group
        cross[group]=fg_map[take].mean();centers[group]=coordinates[take].mean(0)
    seed_scores=cross*weights
    native_seed=int(ids[seed_scores[ids].argmax()])
    identity=None if muBG is None else prototypes@(muFG-muBG)
    return dict(labels=labels,prototypes=prototypes,ids=ids,cross=cross,weights=weights,
        centers=centers,native_seed=native_seed,identity=identity,seed_scores=seed_scores,grid=(h,w))

def seed_prior_from_components(data,seed):
    intra=torch.einsum("c,kc->k",data["prototypes"][seed],data["prototypes"]).clamp_min(0)
    combined=data["cross"]*intra*data["weights"]
    combined[seed]=combined.max().clamp_min(1e-6)
    return combined

def _multiseed_run(ctx,identity=True,max_seeds=MAX_SEEDS):
    events=[];strict_k1=int(max_seeds)==1;started=time.monotonic()
    def override(original_result,ctx,**kwargs):
        if time.monotonic()-started>40:raise GroupingBudgetIncomplete("m06 finite40s pair budget")
        if strict_k1:
            events.append(dict(K=1,native_identity_returned=True))
            return original_result
        d=_seed_components(ctx,kwargs)
        if d is None or len(d["prototypes"])<=1:
            events.append(dict(K=1,state="NATIVE_DEGENERATE"));return original_result
        source=seed_prior_from_components(d,d["native_seed"])
        normal=(source-source.min())/(source.max()-source.min()).clamp_min(1e-6)
        normal=normal[d["labels"]].reshape(kwargs["h"],kwargs["w"])
        if not torch.equal(normal,original_result):
            raise RuntimeError("native single-seed component replay not BITEXACT")
        seeds=[d["native_seed"]];scores=d["seed_scores"][d["ids"]]
        order=d["ids"][torch.argsort(scores,descending=True,stable=True)]
        distance=max(kwargs["h"],kwargs["w"])/4.
        for candidate in order.tolist():
            if candidate in seeds:continue
            if identity and (d["identity"] is None or float(d["identity"][candidate])<=0):continue
            if all(float((d["centers"][candidate]-d["centers"][j]).norm())>=distance for j in seeds):
                seeds.append(candidate)
            if len(seeds)>=int(max_seeds):break
        if len(seeds)==1:
            events.append(dict(K=1,seeds=seeds,native_identity_returned=True));return original_result
        # Accepted a_k=1 iff extra seed beats referenceBG; original seed always retained.
        # All raw source prior terms have the SAME units; max, not repeated sum/minmax.
        combined=torch.stack([seed_prior_from_components(d,seed) for seed in seeds]).amax(0)
        prior=(combined-combined.min())/(combined.max()-combined.min()).clamp_min(1e-6)
        events.append(dict(K=len(seeds),seeds=seeds,native_source_component_exact=True,
            identity_required=bool(identity),extra_identity_margin=[None if d["identity"] is None else float(d["identity"][s]) for s in seeds[1:]],
            spatial_nms_grid_distance=distance,aggregation="MAX source-compatible terms; one sharedminmax",
            a_k="binary referenceFG greater than BG eligibility; original seed unchanged"))
        return prior[d["labels"]].reshape(kwargs["h"],kwargs["w"])
    out=ctx.replay(seed_prior_override=override)
    if strict_k1 and (not torch.equal(out["mask"],ctx.native["mask"]) or not torch.equal(out["score"],ctx.native["score"])):
        raise RuntimeError("K1 complete public source identity failed")
    out["audit"]={**out.get("audit",{}),"id":"m06" if identity else "m06_multi_without_identity_naive",
        "seed_events":events,"max_seeds":int(max_seeds),"BG_part4_source_retained":True,"no_encoder_calls":True,
        "no_query_GT":True,"no_seed_term_deletion":True}
    return out

def method06(ctx):return _multiseed_run(ctx,True,MAX_SEEDS)
def naive06(ctx):return _multiseed_run(ctx,False,MAX_SEEDS)
def k1_control(ctx):return _multiseed_run(ctx,True,1)

METHODS=[
 dict(id="m03",function=method03,naive=naive03,card=CARDS03,budget=MODULE_BUDGET["m03"]),
 dict(id="m06",function=method06,naive=naive06,card=CARDS06,budget=MODULE_BUDGET["m06"],controls={"K1_native_identity":k1_control})]

def self_check():
    """Tiny SERVER-only arithmetic; no CUDA/model/GT files. Root may capture stdout."""
    if torch.cuda.is_initialized():raise RuntimeError("CPU-only check")
    children=np.array([[0,1],[2,3],[4,5]],np.int64)
    groups,guess,audit=tree_dp(children,np.array([.9,.9,.1,.1]),.25)
    assert groups.tolist()==[0,0,1,1] and guess.tolist()==[True,True,False,False]
    one,_,_=tree_dp(children,np.full(4,.6),16);assert len(np.unique(one))==1
    fine,_,_=tree_dp(children,np.array([.99,.01,.99,.01]),0)
    assert len(np.unique(fine))==4
    # Actual sklearn source-compatible linkage/input tiny check, full children count.
    x=F.normalize(torch.tensor([[1.,0.],[.99,.1],[0.,1.],[.1,.99]]),dim=1)
    ch,labels=full_average_tree(x,.6);assert ch.shape==(3,2) and len(labels)==4
    data=dict(prototypes=x,cross=torch.tensor([.9,.8,.7,.6]),weights=torch.ones(4))
    a=seed_prior_from_components(data,0);b=seed_prior_from_components(data,2)
    joint=torch.stack([a,b]).amax(0);assert torch.all(joint>=a) and torch.all(joint>=b)
    # K1 bypass returns exact original tensors through a source-replay-compatible stub.
    original=torch.tensor([[.2,.8]]);score=torch.tensor([[1.,2.]])
    class Replay:
        native=dict(mask=original>.5,score=score)
        def replay(self,seed_prior_override=None):
            assert seed_prior_override(original_result=original,ctx=self) is original
            return dict(mask=self.native["mask"],score=score,audit={})
    result=k1_control(Replay());assert torch.equal(result["mask"],Replay.native["mask"])
    return dict(state="CPU_TEN_GROUPING_PURE_CHECKED",tests=6,CUDA_initialized=False,
        learned_or_pretrained_model=False,source_full_pipeline_real_context="UNTESTED_PREPARED_HARNESS_GATE")
