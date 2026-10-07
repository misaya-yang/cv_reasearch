"""Bounded D181--D200 numerical operators and label-clean source selection.

Only genuine RGB callbacks provide encoder observations. No query labels enter
this module. Unspecified recipe choices are listed in the batch assumptions.
"""
from __future__ import annotations
from dataclasses import replace
import heapq
import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.special import expit, logsumexp
from .common import artifact, ArtifactUnavailable
from . import d_helpers_151_200 as dh

EPS=1e-6


def unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),EPS)


def sqdist(x,c):
    return np.maximum(0.,np.sum(x*x,axis=1)[:,None]+np.sum(c*c,axis=1)-2*dh.mm(x,c.T))


def pseudo_query(ep,train):
    reduced=dh.restricted(ep,train)
    def provider(_ep,name):
        if name in ('frozen_encode_rgb','frozen_input_gradient'):
            fn=artifact(ep,name)
            return lambda role,*args,**kwargs:fn('r',*args,**kwargs)
        return artifact(ep,'r_'+name[2:] if name.startswith('q_') else name)
    # Remove direct q resources so every pseudo-Q observation resolves to R.
    kept={k:v for k,v in reduced.artifacts.items() if not k.startswith('q_') and not callable(v)}
    return replace(reduced,q=ep.r,q_hw=ep.r_hw,q_valid=ep.wvalid.copy(),q_rgb=ep.r_rgb,
                   query_geometry=ep.reference_geometry,original_shape=ep.r_hw,
                   artifacts=kept,provider=provider)


def calibrate(ep,method,builder,configs=dh.DEFAULT_CONFIGS,*,probability=False):
    """Rebuild every supervised dependency; reference is a physical pseudo-Q."""
    dh.validate(ep);base=dh.b0_field(ep);folds=dh.folds(ep)
    if ep.wf.sum()<=EPS or ep.wb.sum()<=EPS or not folds:
        return dh.finish(ep,base,method,branch='single_role_or_missing_fold_B0')
    selected=np.zeros(len(ep.r),bool);baseline=np.zeros(len(ep.r));pairs=[]
    for train,held in folds:
        pseudo=pseudo_query(ep,train);pairs.append((pseudo,held));selected|=held
        baseline[held]=dh.b0_field(pseudo)[held]
    best=(dh.risk(ep,baseline,selected),0,-1);chosen=None;audit=[]
    for j,cfg in enumerate(configs):
        scores=np.zeros(len(ep.r));legal=True;details=[]
        for pseudo,held in pairs:
            field,detail=builder(pseudo,cfg);details.append(detail)
            if field is None:legal=False;break
            field=np.asarray(field,float).ravel()
            if field.shape!=(len(ep.r),) or not np.isfinite(field[held]).all():
                raise FloatingPointError(method+' source field invalid')
            scores[held]=field[held]
        if not legal:audit.append(dict(configuration=cfg,legal=False,folds=details));continue
        if probability:
            threshold=.5;error=dh.risk(ep,scores,selected,threshold)
            edits=int(np.count_nonzero((scores[selected]>.5)!=(baseline[selected]>0)))
        else:threshold,error,edits=dh.best_threshold(ep,scores,selected,baseline)
        audit.append(dict(configuration=cfg,legal=True,source_error=error,threshold=threshold,
                          edits=edits,folds=details))
        if (error,edits,j)<best:best=(error,edits,j);chosen=(cfg,threshold)
    if chosen is None:
        return dh.finish(ep,base,method,branch='source_selected_zero_adaptation',source_candidates=audit,
                         source_folds=len(folds),calibration_query='reference_RGB_and_grid_pseudo_query')
    field,detail=builder(ep,chosen[0])
    if field is None:return dh.finish(ep,base,method,branch='full_fit_unavailable_B0',source_candidates=audit)
    return dh.finish(ep,np.asarray(field).ravel()-chosen[1],method,branch='source_selected_method',
                     configuration=chosen[0],source_threshold=chosen[1],source_error=best[0],
                     source_candidates=audit,source_folds=len(folds),held_labels_removed=True,
                     calibration_query='reference_RGB_and_grid_pseudo_query',**detail)


def modes(ep,k=8,*,original=True):
    x=ep.r if original else dh.project(ep)[0];f,b,dirty=dh.pure(ep);out=[]
    for selected,weight in ((f,ep.wf),(b,ep.wb)):
        centers,ids=dh.source_modes(x[selected],k,weights=weight[selected],spherical=original)
        if not len(centers):return None
        assignment=np.full(len(ep.r),-1,int);assignment[selected]=ids
        out.append((centers,assignment,selected))
    return out,dirty


def bandwidth(ep):
    x=ep.r[ep.wvalid>0]
    if len(x)<2:return .1
    # Median of ALL original-D source pair distances, no Q bandwidth selection.
    distance=np.maximum(0.,2.-2.*dh.mm(x,x.T));ids=np.triu_indices(len(x),1)
    return max(float(np.median(distance[ids])),1e-5)


def kernel_profile(x,centers,scale):
    return np.exp(-sqdist(x,centers)/max(float(scale),1e-5))


def kernel_head(ep,features_r,features_q,extra=None):
    model=dh.head(ep,features_r,extra=extra)
    return None if model is None else dh.predict(model,features_q)


def rbf_control(ep,cfg):
    k,_,strength=cfg;m=modes(ep,k)
    if m is None:return None,dict(reason='insufficient_modes')
    bank=np.vstack([a[0] for a in m[0]]);scale=bandwidth(ep)*strength
    return kernel_head(ep,kernel_profile(ep.r,bank,scale),kernel_profile(ep.q,bank,scale)),dict(codewords=len(bank),control='same_mode_RBF_head')


def groups(ep,count=4):
    """Source correlation clustering of <=32 fixed-projection coordinates."""
    xr,xq,p=dh.project(ep);active=ep.wvalid>0;z=xr[active]
    if len(z)<8:return None
    correlation=np.corrcoef(z,rowvar=False);correlation=np.nan_to_num(correlation)
    d=xr.shape[1];ng=min(count,d);seeds=[0]
    while len(seeds)<ng:
        distance=1.-np.max(np.abs(correlation[:,seeds]),axis=1);distance[seeds]=-np.inf
        seeds.append(int(np.argmax(distance)))
    assignment=np.argmax(np.abs(correlation[:,seeds]),axis=1)
    for j,s in enumerate(seeds):assignment[s]=j
    return xr,xq,p,[np.flatnonzero(assignment==j) for j in range(ng)]


def attributes(ep,count=4):
    result=groups(ep,count)
    if result is None:return None
    xr,xq,p,partition=result;target=dh.sample_weights(ep)
    if target is None:return None
    y,w=target;directions=[]
    for ids in partition:
        model=dh.logistic(xr[:,ids],y,w)
        direction=np.zeros(xr.shape[1]);direction[ids]=model[0]
        direction/=max(float(np.linalg.norm(direction)),EPS);directions.append(direction)
    a=np.asarray(directions).T
    return dh.mm(xr,a),dh.mm(xq,a),p,a,partition


def oof_head(ep,features):
    values=np.full(len(ep.r),np.nan)
    for train,held in dh.folds(ep):
        model=dh.head(dh.restricted(ep,train),features)
        if model is not None:values[held]=dh.predict(model,features[held])
    return values


def high_precision_gates(ep,features):
    score=oof_head(ep,features);f,b=dh.role(ep);block=dh.blocks(ep.r_hw)
    output=[]
    for positive,selected,other in ((True,f,b),(False,b,f)):
        adverse=score[other&np.isfinite(score)]
        gate=max(0.,float(np.max(adverse))) if positive and len(adverse) else (min(0.,float(np.min(adverse))) if len(adverse) else np.nan)
        reliable=selected&np.isfinite(score)&((score>gate) if positive else (score<gate))
        qualified=reliable.sum()>=8 and len(np.unique(block[reliable]))>=2
        output.append((gate,bool(qualified)))
    return output


def candidate_regions(ep):
    """All leaves, <=32 depth-sampled tree nodes, every B0 component; O(N)."""
    height,width=ep.q_hw;n=len(ep.q);valid=ep.q_valid>0;parent=np.arange(n)
    node=np.arange(n);depth=np.zeros(n,int);children={};nodes=[]
    ids=np.arange(n).reshape(height,width)
    aa=np.r_[ids[:-1].ravel(),ids[:,:-1].ravel()];bb=np.r_[ids[1:].ravel(),ids[:,1:].ravel()]
    legal=valid[aa]&valid[bb];aa=aa[legal];bb=bb[legal]
    costs=np.sum((ep.q[aa]-ep.q[bb])**2,axis=1)
    def root(j):
        while parent[j]!=j:parent[j]=parent[parent[j]];j=parent[j]
        return j
    for edge in np.argsort(costs,kind='stable'):
        u,v=root(int(aa[edge])),root(int(bb[edge]))
        if u==v:continue
        if u>v:u,v=v,u
        identifier=n+len(nodes);children[identifier]=(int(node[u]),int(node[v]))
        level=max(int(depth[u]),int(depth[v]))+1;nodes.append((level,identifier))
        parent[v]=u;node[u]=identifier;depth[u]=level
    nodes.sort()
    picked=[]
    for index in np.unique(np.linspace(0,max(len(nodes)-1,0),min(32,len(nodes)),dtype=int)) if nodes else []:
        pending=[nodes[index][1]];leaves=[]
        while pending:
            current=pending.pop()
            if current<n:leaves.append(current)
            else:pending.extend(children[current])
        picked.append(np.asarray(sorted(leaves),int))
    components,count=label((dh.b0_field(ep)>0).reshape(ep.q_hw))
    picked += [np.flatnonzero(components.ravel()==j) for j in range(1,count+1)]
    unique={tuple(c.tolist()):c for c in picked if len(c)>1}
    return list(unique.values()),dict(singleton_leaves=int(valid.sum()),nonleaf_candidates=len(unique),
        merge_rule='four_neighbor_original_D_single_linkage_depth_sample32',tree_storage='linear_parent_children')


def ring(ep,ids):
    inside=np.zeros(ep.q_hw,bool);inside.ravel()[ids]=True
    return np.flatnonzero((binary_dilation(inside)&~inside).ravel()&(ep.q_valid>0))


def merge_field(base,updates):
    out=np.asarray(base,float).copy()
    for ids,value in updates:
        value=np.asarray(value,float);old=out[ids];replace_mask=np.abs(value)>np.abs(old)+1e-12
        out[ids[replace_mask]]=value[replace_mask]
    return out


def mixture_weights(logs,penalty=0.,iterations=20):
    if not len(logs):return np.full(logs.shape[1],1./logs.shape[1])
    weights=np.full(logs.shape[1],1./logs.shape[1]);best=(np.inf,weights.copy())
    for _ in range(iterations):
        logjoint=logs+np.log(np.maximum(weights,1e-12));normal=logsumexp(logjoint,axis=1)
        objective=float(-normal.sum()+penalty*np.count_nonzero(weights>1e-8))
        if objective<best[0]:best=(objective,weights.copy())
        mass=np.exp(logjoint-normal[:,None]).sum(axis=0)
        # Proximal active-component cost permits exactly zero optional parts.
        mass=np.maximum(0.,mass-penalty);weights=mass/max(float(mass.sum()),EPS)
        if mass.sum()<=EPS:weights=np.eye(1,logs.shape[1],int(np.argmax(logs.mean(0)))).ravel()
    return best[1]


def balanced_entropy(ep,assignment,count):
    f=np.bincount(assignment,weights=ep.wf/min(max(ep.wf.sum(),EPS),np.inf),minlength=count)
    b=np.bincount(assignment,weights=ep.wb/min(max(ep.wb.sum(),EPS),np.inf),minlength=count)
    mass=f+b;p=f/np.maximum(mass,EPS)
    return float(np.sum(mass*(-p*np.log(np.maximum(p,EPS))-(1-p)*np.log(np.maximum(1-p,EPS))))/2)


def nnls_codes(x,atoms,penalty,*,forbid=None,steps=20):
    atoms=np.asarray(atoms,float);gram=dh.mm(atoms,atoms.T);corr=dh.mm(x,atoms.T);codes=np.zeros_like(corr)
    for _ in range(steps):
        for j in range(len(atoms)):
            update=(corr[:,j]-dh.mm(codes,gram[:,j])+codes[:,j]*gram[j,j]-penalty/2.)/max(gram[j,j],EPS)
            codes[:,j]=np.maximum(0.,update)
            if forbid is not None:codes[forbid[:,j],j]=0.
    reconstruction=dh.mm(codes,atoms)
    return codes,np.sum((x-reconstruction)**2,axis=1)+penalty*codes.sum(axis=1)
