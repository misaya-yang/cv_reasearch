"""True RGB views / initial residual-state replay; never token-only stand-ins."""
from __future__ import annotations
import numpy as np
from .common import artifact,ArtifactUnavailable
from . import d_helpers_151_200 as dh
from .d160_164 import _matched,_nonadjacent_both,_geometric_median,_fit_residual_map
from ics.cpu100.common import rgb_view

_VIEW_CACHE={}


def _views(ep,kind):
    key=(ep.source_id,tuple(ep.producer.get('source_image_hashes',())),id(ep.r),id(ep.q),kind)
    if _VIEW_CACHE.get('key')!=key:
        _VIEW_CACHE.clear();_VIEW_CACHE['key']=key
    if 'value' in _VIEW_CACHE:
        return _VIEW_CACHE['value']
    encoder=artifact(ep,'frozen_encode_rgb');out={}
    for role in ('r','q'):
        image=np.asarray(rgb_view(ep,role),float)/255.
        if kind=='photometric':
            mean=image.mean(axis=(0,1),keepdims=True)
            images=(np.clip(image*.9,0,1),np.clip(image*1.1,0,1),
                    np.clip(mean+.9*(image-mean),0,1),np.clip(mean+1.1*(image-mean),0,1))
        elif kind=='flip_brightness':
            images=(image[:,::-1].copy(),np.clip(image*.9,0,1))
        else:
            raise ValueError('Unknown predefined genuine RGB view family')
        results=[]
        for j,changed in enumerate(images):
            z=np.asarray(encoder(role,changed,working_canvas=True,side=image.shape[0]),float)
            expected=ep.r_hw if role=='r' else ep.q_hw
            if z.shape!=tuple(expected)+(ep.r.shape[1],) or not np.isfinite(z).all():
                raise ValueError('True native extra-view output shape/finite mismatch')
            if kind=='flip_brightness' and j==0:
                z=z[:,::-1].copy()
            results.append(z.reshape(-1,z.shape[-1]))
        out[role]=np.asarray(results)
    _VIEW_CACHE['value']=out
    return out


def _lift_map(ep,x,matrix,p):
    z=dh.mm(x,p); right=np.linalg.solve(dh.mm(p.T,p)+1e-6*np.eye(p.shape[1]),p.T)
    corrected=x+dh.mm(dh.mm(z,matrix),right)
    corrected/=np.maximum(np.linalg.norm(corrected,axis=1,keepdims=True),1e-6)
    return corrected


def _d165_builder(ep,config,control=False):
    k,rank,strength=config; views=_views(ep,'photometric');f,b,_=dh.pure(ep)
    if not f.any() or not b.any():
        return None,{}
    xr,xq,p=dh.project(ep)
    response={role:np.stack(((v[1]-v[0])/.2,(v[3]-v[2])/.2),axis=1).reshape(v.shape[1],-1)
              for role,v in views.items()}
    for role in response:
        response[role]/=np.maximum(np.linalg.norm(response[role],axis=1,keepdims=True),1e-6)
    rid,qid=_matched(ep,f|b); support=dh.source_radius(ep,f|b)
    distance=np.sqrt(np.maximum(0.,2.-2.*dh.pair(ep)[qid,rid]))
    compatible=np.sum(response['r'][rid]*response['q'][qid],axis=1)>.5
    compatible &= distance<=support
    rid=rid[compatible];qid=qid[compatible]
    if len(rid)<8 or not _nonadjacent_both(ep,rid,qid):
        return None,dict(reason='insufficient_response_compatible_pairs',pairs=len(rid))
    matrix=_fit_residual_map(xq[qid],xr[rid],rank,.01)*float(strength)
    def score(x):
        if control and x is ep.q:
            fields=[dh.b0_field(ep,ep.q)]+[dh.b0_field(ep,v) for v in views['q']]
            return np.mean(fields,axis=0)
        return dh.b0_field(ep,_lift_map(ep,x,matrix,p))
    score.source_score=lambda ids:np.mean([dh.b0_field(ep,_lift_map(ep,v[ids],matrix,p)) for v in views['r']],axis=0)
    if control:
        score.source_score=lambda ids:np.mean([dh.b0_field(ep,ep.r[ids])]+[dh.b0_field(ep,v[ids]) for v in views['r']],axis=0)
    return score,dict(actual_extra_views=8,response_compatible_pairs=len(rid),map_norm=float(np.linalg.norm(matrix)),
                      transformations='exposure_.9_1.1_contrast_about_own_mean_.9_1.1',response_agreement_cosine=.5)


def d165(ep):
    f,b=dh.role(ep)
    if not f.any() or not b.any():return dh.finish(ep,dh.b0_field(ep),'D165',branch='single_role_documented_B0')
    _views(ep,'photometric')
    return dh.calibrate(ep,'D165',_d165_builder)


def d165_control(ep):
    _views(ep,'photometric')
    def builder(ep,c):
        views=_views(ep,'photometric');f,b=dh.role(ep)
        if not f.any() or not b.any():return None,{}
        bank=np.asarray([ep.r,*views['r']]).reshape(-1,ep.r.shape[1]);ft=np.tile(f,5);bt=np.tile(b,5)
        def average(rows):
            return np.mean([np.max(dh.mm(v,bank[ft].T),axis=1)-np.max(dh.mm(v,bank[bt].T),axis=1) for v in rows],axis=0)
        def score(x):return average(np.asarray([ep.q,*views['q']]) if x is ep.q else np.asarray([x]))
        score.source_score=lambda ids:average(np.asarray([ep.r,*views['r']])[:,ids])
        return score,dict(actual_extra_views=8,response_pair_gate_used=False)
    return dh.calibrate(ep,'D165_control_same_views_average_B0',builder,((4,2,1.),))


def _tail_resources(ep):
    r=np.asarray(artifact(ep,'r_pre_last2_patch_state'),float)
    q=np.asarray(artifact(ep,'q_pre_last2_patch_state'),float)
    if r.shape!=ep.r.shape or q.shape!=ep.q.shape or not np.isfinite(r).all() or not np.isfinite(q).all():
        raise ValueError('D169 genuine pre-last2 residual patch states required')
    return r,q,artifact(ep,'tail_replay')


def _d169_builder(ep,alpha,control=False):
    rs,qs,replay=_tail_resources(ep);f,b,_=dh.pure(ep);rid,qid=_matched(ep,b)
    base=dh.b0_field(ep);negative=base[qid]<0;rid=rid[negative];qid=qid[negative]
    if len(rid)<8 or not _nonadjacent_both(ep,rid,qid):
        return None,dict(reason='insufficient_BG_pairs',pairs=len(rid))
    differences=qs[qid]-rs[rid];direction=_geometric_median(differences)
    groups=dh.blocks(ep.r_hw,4)[rid]
    if any(float(dh.mm(differences[groups==g].mean(axis=0),direction))<=0 for g in np.unique(groups)):
        return None,dict(reason='BG_state_directions_conflict')
    # Source pseudo-domain differences use observed training-R BG pairs from
    # different blocks, not the held block's label or an invented latent state.
    selected=np.flatnonzero(b); source_blocks=dh.blocks(ep.r_hw)
    source_differences=[]; rr=dh.pair(ep,'rr')
    for source in selected:
        eligible=b&(source_blocks!=source_blocks[source])
        if eligible.any():
            target=np.flatnonzero(eligible)[np.argmax(rr[source,eligible])]
            source_differences.append(rs[source]-rs[target])
    if not source_differences:
        return None,dict(reason='no_source_pseudo_domain_pairs')
    source_shift=_geometric_median(np.asarray(source_differences))
    if control:
        corrected=np.asarray(artifact(ep,'q_raw'),float)-float(alpha)*direction
        corrected/=np.maximum(np.linalg.norm(corrected,axis=1,keepdims=True),1e-6)
        query_field=dh.b0_field(ep,corrected)
        source_corrected=np.asarray(artifact(ep,'r_raw'),float)-float(alpha)*source_shift
        source_corrected/=np.maximum(np.linalg.norm(source_corrected,axis=1,keepdims=True),1e-6)
        source_field=dh.b0_field(ep,source_corrected)
    else:
        rout=replay(ep,'r',patch_state_delta=np.broadcast_to(-float(alpha)*source_shift,rs.shape).copy(),return_attention=False)
        query_field=None
        source_field=dh.b0_field(ep,np.asarray(rout['unit']))
    detail=dict(alpha=float(alpha),matched_BG_pairs=len(rid),state_shift_norm=float(np.linalg.norm(direction)),
                source_pseudo_domain_pairs=len(source_differences),actual_tail_replays=0 if control else 1,
                special_tokens_untouched=True,replay_injection='initial_pre_block23_patch_state')
    def scorer(x):
        nonlocal query_field
        if x is ep.q:
            if query_field is None:
                qout=replay(ep,'q',patch_state_delta=np.broadcast_to(-float(alpha)*direction,qs.shape).copy(),return_attention=False)
                query_field=dh.b0_field(ep,np.asarray(qout['unit']));detail['actual_tail_replays']+=1
            return query_field
        ids=np.argmax(dh.mm(x,ep.r.T),axis=1);return source_field[ids]
    scorer.source_score=lambda ids:source_field[ids]
    return scorer,detail


def d169(ep):
    f,b=dh.role(ep)
    if not f.any() or not b.any():return dh.finish(ep,dh.b0_field(ep),'D169',branch='single_role_documented_B0')
    _tail_resources(ep)
    return dh.calibrate(ep,'D169',_d169_builder,(0.,.25,.5,1.))


def d169_control(ep):
    _tail_resources(ep)
    return dh.calibrate(ep,'D169_control_same_pairs_final_shift',lambda ep,a:_d169_builder(ep,a,True),(0.,.25,.5,1.))


def _d170_builder(ep,strength,control=False):
    views=_views(ep,'flip_brightness');xr,xq,p=dh.project(ep);spatial=dh.folds(ep);f,b=dh.role(ep)
    if not spatial:return None,{}
    rstack=np.asarray([ep.r,*views['r']]);qstack=np.asarray([ep.q,*views['q']])
    projected_r=np.asarray([dh.mm(v,p) for v in rstack]);projected_q=np.asarray([dh.mm(v,p) for v in qstack])
    def fit(reduced,ids=None,labels=None):
        targets=dh.sample_weights(reduced)
        if targets is None:return None
        yy,ww=targets
        extra=None
        if ids is not None and len(ids) and strength>0:
            extra=(projected_q[:,ids].reshape(-1,xr.shape[1]),np.tile(labels,3),np.full(3*len(ids),float(strength)/(3*len(ids))))
        return dh.logistic(projected_r.reshape(-1,xr.shape[1]),np.tile(yy,3),np.tile(ww/3.,3),extra=extra)
    original=fit(ep);qscore=np.asarray([dh.predict(original,z) for z in projected_q]);rscores=np.full((3,len(ep.r)),np.nan)
    for train,held in spatial:
        model=fit(dh.restricted(ep,train))
        for j in range(3):rscores[j,held]=dh.predict(model,projected_r[j,held])
    allpos=np.all(rscores>0,axis=0);allneg=np.all(rscores<0,axis=0)
    fvalues=np.min(rscores[:,f&allpos],axis=0);bvalues=np.max(rscores[:,b&allneg],axis=0)
    # Any observed same-view agreement error determines the finite zero-error
    # margin interval, rather than declaring every agreement a reliable label.
    wrongpos=np.min(rscores[:,b&allpos],axis=0);wrongneg=np.max(rscores[:,f&allneg],axis=0)
    fg_gate=max(0.,float(np.max(wrongpos)) if len(wrongpos) else 0.)
    bg_gate=min(0.,float(np.min(wrongneg)) if len(wrongneg) else 0.)
    block_ids=dh.blocks(ep.r_hw,4)
    if min(len(fvalues),len(bvalues))<8 or fg_gate<=bg_gate:
        # The source intervals can meet only at zero without overlap. Equality
        # zero is legal provided both roles have actual independent block mass.
        if fg_gate<bg_gate or min(len(fvalues),len(bvalues))<8:
            ids=np.empty(0,int);labels=np.empty(0)
        else:ids=None
    else:ids=None
    if ids is None:
        base=dh.b0_field(ep)
        positive=np.all(qscore>fg_gate,axis=0)&(base>0);negative=np.all(qscore<bg_gate,axis=0)&(base<0)
        support=np.zeros(len(ep.q),bool);sim=dh.pair(ep)
        for cls,selected in ((True,f),(False,b)):
            radius=dh.source_radius(ep,selected)
            if not np.isfinite(radius):continue
            eligible=positive if cls else negative
            for qi in np.flatnonzero(eligible&(ep.q_valid>0)):
                hit=selected&(np.sqrt(np.maximum(0.,2.-2.*sim[qi]))<=radius)
                distinct=np.unique(block_ids[hit])
                separated=any(max(abs(int(a)//4-int(c)//4),abs(int(a)%4-int(c)%4))>1 for a in distinct for c in distinct)
                if separated:support[qi]=True
        ids=np.flatnonzero(support);labels=positive[ids].astype(float)
    if control:
        ids=np.empty(0,int);labels=np.empty(0)
    mode_audit=[];safe=True
    for train,held in spatial:
        reduced=dh.restricted(ep,train);before=fit(reduced);after=fit(reduced,ids,labels)
        centers,_=dh.foreground_modes(reduced,8);modes=np.argmax(dh.mm(ep.r,centers.T),axis=1)
        old=np.mean([dh.predict(before,z) for z in projected_r],axis=0)
        changed=np.mean([dh.predict(after,z) for z in projected_r],axis=0)
        okay=dh.fg_recall_non_decrease(ep,old,changed,modes,held);mode_audit.append(bool(okay));safe &=okay
    if not safe:ids=np.empty(0,int);labels=np.empty(0)
    final=fit(ep,ids,labels)
    def augmented_b0(rows):
        bank=rstack.reshape(-1,ep.r.shape[1]);ft=np.tile(f,3);bt=np.tile(b,3)
        return np.mean([np.max(dh.mm(v,bank[ft].T),axis=1)-np.max(dh.mm(v,bank[bt].T),axis=1) for v in rows],axis=0)
    no_update=control or not len(ids) or float(strength)==0.
    def scorer(x):
        if no_update:
            return augmented_b0(qstack if x is ep.q else np.asarray([x]))
        if x is ep.q:return np.mean([dh.predict(final,z) for z in projected_q],axis=0)
        return dh.predict(final,dh.mm(x,p))
    scorer.source_score=(lambda ids:augmented_b0(rstack[:,ids])) if no_update else (lambda ids:np.mean([dh.predict(final,z[ids]) for z in projected_r],axis=0))
    return scorer,dict(actual_extra_views=4,pseudo_points=len(ids),pseudo_strength=float(strength),
                       source_F_gate=fg_gate,source_B_gate=bg_gate,mode_recall_audit=mode_audit,query_updates=0 if no_update else 1,
                       documented_augmented_B0_exit=no_update,
                       transforms='horizontal_flip_inverse_aligned_and_brightness_.9')


def d170(ep):
    f,b=dh.role(ep)
    if not f.any() or not b.any():return dh.finish(ep,dh.b0_field(ep),'D170',branch='single_role_documented_B0')
    _views(ep,'flip_brightness')
    return dh.calibrate(ep,'D170',_d170_builder,(0.,.25,1.))


def d170_control(ep):
    _views(ep,'flip_brightness')
    return dh.calibrate(ep,'D170_control_same_views_without_pseudo_update',lambda ep,c:_d170_builder(ep,c,True),(0.,))


METHODS={'D165':d165,'D169':d169,'D170':d170}
CONTROLS={'D165_control_same_views_average_B0':d165_control,'D169_control_same_pairs_final_shift':d169_control,
          'D170_control_same_views_without_pseudo_update':d170_control}
