"""Faithful episode-local implementations of the supplied D151--E225 cards.

Registries contain completed implementations only. Missing genuine observations
raise ArtifactUnavailable rather than being replaced by an unrelated baseline.
The companion recipes record choices left unspecified by the supplied cards.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.special import expit, logsumexp
from scipy.ndimage import label
from .common import Result
from . import d_helpers_151_200 as dh


def _reference_builder(ep, config=None):
    xr,xq,p=dh.project(ep); model=dh.head(ep,xr)
    return (None,{}) if model is None else (lambda x: dh.predict(model, dh.mm(x,p)), dict(steps=100))


def d_reference_logistic(ep):
    scorer,_=_reference_builder(ep)
    if scorer is None:
        return dh.baseline(ep)
    return dh.finish(ep,scorer(ep.q),'D_control_reference_logistic',temporary_reference_fitting=True,steps=100)


def d_strong_logistic(ep):
    from ics.cpu100.decision_risk import average_logistic
    out=average_logistic(ep)
    return Result(out.margin,0.,info=dict(out.info,astra_control='unchanged_CPU100_average_logistic'))


def d_strong_ridge(ep):
    from ics.cpu100.invariance_support import huber_ridge_control
    out=huber_ridge_control(ep)
    return Result(out.margin,0.,info=dict(out.info,astra_control='unchanged_CPU100_fullD_ridge'))


def _mixture_fit(q, anchors, variance, anchor_strength, count_penalty, proposed, proposal_groups=None):
    """Free mixing masses, source-identified F centers, query-only BG births."""
    nf=len(anchors); means=np.asarray(anchors,float).copy(); weights=np.full(nf,1./nf)
    d=q.shape[1]; variance=max(float(variance),1e-5)
    em_steps=0
    def fit(means, weights,iterations):
        nonlocal em_steps
        best=None
        for _ in range(iterations):
            em_steps+=1
            distance=np.maximum(0.,np.sum(q*q,axis=1)[:,None]+np.sum(means*means,axis=1)-2*dh.mm(q,means.T))
            logs=np.log(np.maximum(weights,1e-12))-distance/(2*variance)-.5*d*np.log(2*np.pi*variance)
            ll=logsumexp(logs,axis=1); responsibilities=np.exp(logs-ll[:,None])
            objective=float(-ll.sum()+anchor_strength*np.sum((means[:nf]-anchors)**2)/(2*variance)
                            +count_penalty*(len(means)-nf))
            if best is None or objective<best[0]:
                best=(objective,means.copy(),weights.copy(),logs.copy())
            mass=responsibilities.sum(axis=0)
            updated=dh.mm(responsibilities.T,q)/np.maximum(mass[:,None],1e-12)
            updated[:nf]=(dh.mm(responsibilities[:,:nf].T,q)+anchor_strength*anchors)/(mass[:nf,None]+anchor_strength)
            means=updated; weights=np.maximum(mass,1e-12)/max(float(mass.sum()),1e-12)
        return best
    # One candidate has at most TWENTY EM updates in total. Four initialize the
    # inherited model; each of up to eight birth proposals gets two updates.
    incumbent=fit(means,weights,4); attempted=accepted=0
    remaining=list(range(len(proposed)))
    for _ in range(min(8,len(proposed))):
        # Largest currently unexplained Q cluster, with residual sum breaking
        # equal counts. It is never a synthetic BG inherited from the R mask.
        distances=np.maximum(0.,np.sum(q*q,axis=1)[:,None]+np.sum(incumbent[1]*incumbent[1],axis=1)-2*dh.mm(q,incumbent[1].T))
        unexplained=np.min(distances,axis=1)>2.*variance*d
        groups=np.argmin(np.sum((q[:,None]-proposed[None])**2,axis=2),axis=1) if proposal_groups is None else proposal_groups
        scores={k:(int(np.sum(unexplained&(groups==k))),float(np.min(distances[groups==k],axis=1).sum()),-k) for k in remaining}
        j=max(remaining,key=lambda k:scores[k])
        if scores[j][0]<8:
            break
        remaining.remove(j); attempted+=1
        newmeans=np.vstack((incumbent[1],proposed[j])); newweights=np.r_[incumbent[2]*(1.-1./len(newmeans)),1./len(newmeans)]
        candidate=fit(newmeans,newweights,2)
        if candidate[0]<incumbent[0]-1e-12:
            incumbent=candidate; accepted+=1
        if not remaining:
            break
    return incumbent,nf,dict(bg_proposals=attempted,bg_accepted=accepted,mixture_objective=incumbent[0],
                              target_mixing_mass=float(incumbent[2][:nf].sum()),em_steps_total=em_steps,
                              em_budget=20)


def _d151_builder(ep,config):
    k,rank,penalty=config; xr,xq,p=dh.project(ep); f,b,contamination=dh.pure(ep); valid=ep.q_valid>0
    if not f.any() or not b.any() or not valid.any():
        return None,{}
    centers,_=dh.source_modes(xr[f],k,spherical=False,weights=ep.wf[f])
    if not len(centers):return None,dict(reason='fewer_than_eight_F_support_samples')
    source=xr[f]; distances=np.maximum(0.,np.sum(source*source,axis=1)[:,None]+np.sum(source*source,axis=1)-2*dh.mm(source,source.T))
    np.fill_diagonal(distances,np.inf)
    nearest=np.min(distances,axis=1); finite=nearest[np.isfinite(nearest)]
    variance=float(np.median(finite))/max(xr.shape[1],1) if len(finite) else .01
    proposed,proposal_groups=dh.source_modes(xq[valid],8,spherical=False)
    if not len(proposed):return None,dict(reason='fewer_than_eight_query_BG_proposal_samples')
    fit,nf,detail=_mixture_fit(xq[valid],centers,variance,penalty*len(xq[valid])/max(len(centers),1),
                             penalty*np.log1p(valid.sum())*xr.shape[1],proposed,proposal_groups)
    if len(fit[1])==nf:
        return None,dict(**detail,reason='all_BG_births_rejected')
    def score(x):
        z=dh.mm(x,p); distance=np.maximum(0.,np.sum(z*z,axis=1)[:,None]+np.sum(fit[1]*fit[1],axis=1)-2*dh.mm(z,fit[1].T))
        # Class posterior log odds: the shared normalizer cancels exactly.
        logs=np.log(np.maximum(fit[2],1e-12))-distance/(2*max(variance,1e-5))
        return logsumexp(logs[:,:nf],axis=1)-logsumexp(logs[:,nf:],axis=1)
    return score,dict(**detail,soft_coverage_fallback=bool(contamination),source_F_component_counts_minimum=8)


def d151(ep):
    return dh.calibrate(ep,'D151',_d151_builder)


def _d151_control_builder(ep,config):
    k,_,_=config; f,b,_=dh.pure(ep); score0=dh.b0_field(ep); valid=ep.q_valid>0
    if not f.any() or not b.any():
        return None,{}
    fg,_=dh.source_modes(ep.r[f],k,weights=ep.wf[f])
    if not len(fg):return None,{}
    qbg=ep.q[valid & (score0<0)]
    pool=np.vstack((ep.r[b],qbg)) if len(qbg) else ep.r[b]
    bg,_=dh.source_modes(pool,8)
    if not len(bg):return None,{}
    return lambda x: np.max(dh.mm(x,fg.T),axis=1)-np.max(dh.mm(x,bg.T),axis=1),dict(bg_capacity=8,bg_modes=len(bg),target_capacity=k)


def d151_control(ep):
    return dh.calibrate(ep,'D151_control_same_budget_negative_prototypes',_d151_control_builder)


def _pseudo_head(ep,xr,xq,ids,labels,strength):
    if not len(ids) or strength==0:
        return dh.head(ep,xr)
    extra=(xq[ids],np.asarray(labels,float),np.full(len(ids),strength/len(ids)))
    return dh.head(ep,xr,extra=extra)


def _d152_builder(ep,config,control=False):
    strength=float(config); xr,xq,p=dh.project(ep); valid=np.flatnonzero(ep.q_valid>0)
    spatial=dh.folds(ep)
    if not len(valid) or not spatial:
        return None,{}
    # K=16 is the requested cap. Underpopulated independent cluster hypotheses
    # are deterministically pooled by reducing K until each has eight samples.
    _,qgroup=dh.source_modes(ep.q[valid],16)
    if not len(qgroup):return None,dict(reason='query_clusters_have_less_than_eight_samples')
    authorized=[]; authorized_labels=[]; comparisons=[]
    for group in np.unique(qgroup):
        ids=valid[qgroup==group]; signs=[]
        if strength>0:
            for train,held in spatial:
                reduced=dh.restricted(ep,train); errors=[]
                for hypothetical in (1.,0.):
                    model=_pseudo_head(reduced,xr,xq,ids,np.full(len(ids),hypothetical),strength)
                    errors.append(dh.risk(ep,dh.predict(model,xr),held))
                difference=errors[1]-errors[0]
                signs.append(1 if difference>1e-12 else -1 if difference< -1e-12 else 0)
        winner=signs[0] if signs and signs[0]!=0 and all(v==signs[0] for v in signs) else 0
        comparisons.append(dict(cluster=int(group),heldout_signs=signs,authorized=winner))
        if winner:
            authorized.extend(ids.tolist()); authorized_labels.extend([1. if winner>0 else 0.]*len(ids))
    ids=np.asarray(authorized,int); labels=np.asarray(authorized_labels)
    if control and len(ids):
        base=dh.b0_field(ep)
        for group in np.unique(qgroup):
            members=valid[qgroup==group]; chosen=np.isin(ids,members)
            if chosen.any():
                labels[chosen]=float(np.mean(base[members])>0)
    audit=[]; safe=True
    for train,held in spatial:
        reduced=dh.restricted(ep,train); original=dh.head(reduced,xr)
        changed=_pseudo_head(reduced,xr,xq,ids,labels,strength)
        centers,_=dh.foreground_modes(reduced,8)
        modes=np.argmax(dh.mm(ep.r,centers.T),axis=1) if len(centers) else np.zeros(len(ep.r),int)
        good=dh.fg_recall_non_decrease(ep,dh.predict(original,xr),dh.predict(changed,xr),modes,held)
        audit.append(bool(good)); safe &= good
    if not safe:
        ids=np.empty(0,int); labels=np.empty(0); strength=0.
    model=_pseudo_head(ep,xr,xq,ids,labels,strength)
    return lambda x: dh.predict(model,dh.mm(x,p)),dict(authorized_points=len(ids),authorized_clusters=sum(v['authorized']!=0 for v in comparisons),
                  competition=comparisons,mode_recall_audit=audit,pseudo_strength=strength,pseudo_refits=1)


def d152(ep):
    return dh.calibrate(ep,'D152',_d152_builder,(0.,.25,1.))


def _d152_simple_builder(ep,config):
    strength=float(config); xr,xq,p=dh.project(ep); valid=np.flatnonzero(ep.q_valid>0)
    _,groups=dh.kmeans(ep.q[valid],16); baseline=dh.b0_field(ep); labels=np.zeros(len(valid))
    for group in np.unique(groups):
        labels[groups==group]=np.mean(baseline[valid[groups==group]])>0
    model=_pseudo_head(ep,xr,xq,valid,labels,strength)
    return lambda x: dh.predict(model,dh.mm(x,p)),dict(pseudo_points=len(valid),pseudo_strength=strength,one_shot=True)


def d152_control(ep):
    return dh.calibrate(ep,'D152_control_same_count_nearest_cluster',lambda ep,c:_d152_builder(ep,c,True),(0.,.25,1.))


def _stable_query_negatives(ep):
    f,b=dh.role(ep); base=dh.b0_field(ep); xr,xq,p=dh.project(ep); classifier=dh.head(ep,xr)
    if classifier is None or not b.any():
        return np.empty(0,int)
    radius=dh.source_radius(ep,b); distance=dh.nearest_distance(dh.pair(ep),b)
    if not np.isfinite(radius):
        return np.empty(0,int)
    spatial=dh.folds(ep); source_scores=np.full(len(ep.r),np.nan)
    for train,held in spatial:
        model=dh.head(dh.restricted(ep,train),xr); source_scores[held]=dh.predict(model,xr[held])
    supported=b & np.isfinite(source_scores)
    if supported.sum()<8 or len(np.unique(dh.blocks(ep.r_hw)[supported]))<2:
        return np.empty(0,int)
    # A zero-FG-error source interval; an empty qualified interval exits honestly.
    fscore=source_scores[f & np.isfinite(source_scores)]
    if not len(fscore):
        return np.empty(0,int)
    threshold=min(0.,float(np.min(fscore)))
    qualifying=supported & (source_scores<threshold)
    if qualifying.sum()<8 or len(np.unique(dh.blocks(ep.r_hw)[qualifying]))<2:
        return np.empty(0,int)
    return np.flatnonzero((ep.q_valid>0)&(base<0)&(dh.predict(classifier,xq)<threshold)&(distance<=radius))


def _d153_builder(ep,config):
    k,fee_multiplier=config; f,b,_=dh.pure(ep)
    if not f.any() or not b.any():
        return None,{}
    fa=dh.fps_rows(ep.r[f],k); ba=dh.fps_rows(ep.r[b],k); negatives=_stable_query_negatives(ep)
    if len(negatives):
        ba=np.vstack((ba,dh.fps_rows(ep.q[negatives],k)))
    # The residual scale is measured on leave-block BG using only the other
    # source blocks' background dictionary, not in-sample reconstruction.
    source=[]
    for training,held in dh.folds(ep):
        train=b&training; test=b&held
        if train.any() and test.any():
            atom=dh.fps_rows(ep.r[train],k)
            source.extend(dh.sparse_cost(ep.r[test],atom,0.).tolist())
    scale=max(float(np.median(source)) if source else 1.,1e-5); penalty=.1*scale; fee=fee_multiplier*scale
    union=np.vstack((ba,fa)); radius=dh.source_radius(ep,f)
    if not np.isfinite(radius):
        return None,{}
    def score(x):
        bg=dh.sparse_cost(x,ba,penalty); both=dh.sparse_cost(x,union,penalty)
        # The feasible nested solution using zero F coefficients is retained;
        # finite coordinate updates cannot manufacture a negative reduction.
        both=np.minimum(bg,both); reduction=bg-both
        gate=radius-dh.nearest_distance(dh.mm(x,ep.r.T),f)
        return np.minimum((reduction-fee)/scale,gate/max(radius,1e-5))
    return score,dict(opening_fee=fee,l1_penalty=penalty,bg_atoms=len(ba),fg_atoms=len(fa),
                      query_negative_points=len(negatives),coordinate_sweeps=20,coefficient_constraint='signed_L1')


def d153(ep):
    # A nonnegative extra threshold preserves the strictly positive opening fee.
    return dh.calibrate(ep,'D153',_d153_builder,((4,.1),(4,1.),(8,.1),(8,1.),(16,1.)),minimum_threshold=0.)


def _d153_control_builder(ep,config):
    k,_=config; f,b,_=dh.pure(ep); neg=_stable_query_negatives(ep)
    if not f.any() or not b.any():
        return None,{}
    fg=dh.fps_rows(ep.r[f],k); bg=dh.fps_rows(ep.r[b],k)
    if len(neg):
        bg=np.vstack((bg,dh.fps_rows(ep.q[neg],k)))
    return lambda x: np.max(dh.mm(x,fg.T),axis=1)-np.max(dh.mm(x,bg.T),axis=1),dict(fg_atoms=len(fg),bg_atoms=len(bg))


def d153_control(ep):
    return dh.calibrate(ep,'D153_control_same_dictionary_nearest',_d153_control_builder,((4,.1),(8,.1),(16,1.)))


def _d155_builder(ep,config):
    strength=float(config); xr,xq,p=dh.project(ep); f,b=dh.role(ep); spatial=dh.folds(ep)
    if not spatial:
        return None,{}
    dist=dh.nearest_distance(dh.pair(ep,'rr'),f); far=dist>=np.quantile(dist[b],.5)
    stable=b.copy()
    for train,held in spatial:
        reduced=dh.restricted(ep,train); nearest=dh.b0_field(reduced,ep.r[held])
        stable[np.flatnonzero(held)]=nearest<0
    core=b&far&stable
    if core.sum()<8 or len(np.unique(dh.blocks(ep.r_hw)[core]))<2:
        return None,dict(core_points=int(core.sum()))
    allhead=dh.head(ep,xr); corehead=dh.role_head(ep,xr,background=core)
    def score(x):
        z=dh.mm(x,p)
        return (1-strength)*dh.predict(allhead,z)+strength*dh.predict(corehead,z)
    return score,dict(core_points=int(core.sum()),blend=strength,known_BG_relabelled=False)


def d155(ep):
    return dh.calibrate(ep,'D155',_d155_builder,(0.,.25,1.))


def d155_control(ep):
    return dh.calibrate(ep,'D155_control_discard_hard_BG',_d155_builder,(1.,))


def _nonadjacent_pairs(ep,rids,qids,difference):
    rb=dh.blocks(ep.r_hw,4)[rids]; qb=dh.blocks(ep.q_hw,4)[qids]
    for i in range(len(rids)):
        ry,rx=divmod(int(rb[i]),4); qy,qx=divmod(int(qb[i]),4)
        for j in range(i):
            sy,sx=divmod(int(rb[j]),4); ty,tx=divmod(int(qb[j]),4)
            if max(abs(ry-sy),abs(rx-sx))<=1 or max(abs(qy-ty),abs(qx-tx))<=1:
                continue
            if float(dh.mm(difference[i],difference[j]))>0:
                return True
    return False


def _d156_builder(ep,config,control=False):
    k,_,_=config; xr,xq,p=dh.project(ep); f,b,_=dh.pure(ep)
    if not f.any() or not b.any():
        return None,{}
    fg_radius=dh.source_radius(ep,f); bg_radius=dh.source_radius(ep,b)
    if not np.isfinite(fg_radius) or not np.isfinite(bg_radius):
        return None,{}
    reference=dh.head(ep,xr); bgcenters,bg_group=dh.source_modes(ep.r[b],k,weights=ep.wb[b]); bids=np.flatnonzero(b)
    if not len(bgcenters):return None,dict(reason='fewer_than_eight_BG_support_samples')
    qr=dh.pair(ep); qvalid=ep.q_valid>0; qnearest=np.argmax(np.where(ep.wvalid[None]>0,qr,-np.inf),axis=1)
    rnearest=np.argmax(np.where(qvalid[:,None],qr,-np.inf),axis=0)
    experts=[]; recurrence=[]
    for group in np.unique(bg_group):
        ids=bids[bg_group==group]; qm=rnearest[ids]; mutual=qnearest[qm]==ids
        distances=np.sqrt(np.maximum(0.,2.-2.*qr[qm,ids])); match=mutual&(distances<=bg_radius)
        pairs_r=ids[match]; pairs_q=qm[match]
        active=len(pairs_r)>=2 and _nonadjacent_pairs(ep,pairs_r,pairs_q,ep.q[pairs_q]-ep.r[pairs_r])
        chosen=np.zeros(len(ep.r),bool); chosen[ids]=True
        model=dh.role_head(ep,xr,background=chosen)
        experts.append((model,chosen,active,len(pairs_r)))
        recurrence.append(dict(type=int(group),mutual_pairs=len(pairs_r),active=bool(active)))
    def score(x):
        z=dh.mm(x,p); sim=dh.mm(x,ep.r.T)
        distance=dh.nearest_distance(sim,f)
        u=.5*dh.predict(reference,z)+.5*(1.-distance/max(fg_radius,1e-5))
        output=u.copy()
        for model,chosen,active,nmatch in experts:
            d=dh.nearest_distance(sim,chosen); here=d<=bg_radius
            if control:
                # Same u, same BG types and source radius, ordinary distance veto.
                veto=d/max(bg_radius,1e-5)-1.
                output[here]=np.minimum(output[here],veto[here])
            elif active:
                veto=dh.predict(model,z); output[here]=np.minimum(output[here],veto[here])
        return output
    return score,dict(bg_types=len(experts),recurrence=recurrence,fg_radius=fg_radius,bg_radius=bg_radius,
                      active_experts=sum(x[2] for x in experts),base_field='half_reference_logit_half_F_support')


def d156(ep):
    return dh.calibrate(ep,'D156',_d156_builder)


def d156_control(ep):
    return dh.calibrate(ep,'D156_control_same_u_local_BG_distance',lambda ep,c:_d156_builder(ep,c,True))


METHODS={'D151':d151,'D152':d152,'D153':d153,'D155':d155,'D156':d156}
CONTROLS={'D_B0':dh.baseline,'D_control_reference_logistic':d_reference_logistic,
          'D_control_strong_CPU100_average_logistic':d_strong_logistic,'D_control_strong_CPU100_ridge':d_strong_ridge,
          'D151_control_same_budget_negative_prototypes':d151_control,
          'D152_control_same_count_nearest_cluster':d152_control,
          'D153_control_same_dictionary_nearest':d153_control,
          'D155_control_discard_hard_BG':d155_control,
          'D156_control_same_u_local_BG_distance':d156_control}

from .d160_164 import METHODS as BATCH02_METHODS, CONTROLS as BATCH02_CONTROLS
METHODS.update(BATCH02_METHODS)
CONTROLS.update(BATCH02_CONTROLS)
