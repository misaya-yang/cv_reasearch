"""Actual amplitude observation and supported domain adaptation (D160--164)."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .common import ArtifactUnavailable, artifact
from . import d_helpers_151_200 as dh


def _amplitude(ep,role):
    try:
        value=np.asarray(artifact(ep,role+'_pre_final_ln'),float)
    except ArtifactUnavailable:
        return None
    expected=ep.r.shape if role=='r' else ep.q.shape
    if value.shape!=expected or not np.isfinite(value).all():
        raise ValueError('D160 actual pre-final-LN state shape/finite contract')
    a=np.log(np.linalg.norm(value,axis=1)+1e-6)
    validity=ep.wvalid>0 if role=='r' else ep.q_valid>0
    if not validity.any():
        return None
    med=float(np.median(a[validity])); mad=float(np.median(np.abs(a[validity]-med)))
    return None if mad<1e-6 else (a-med)/mad


def _d160_builder(ep,strength):
    f,b,_=dh.pure(ep)
    if not f.any() or not b.any():
        return None,{}
    ra=_amplitude(ep,'r'); qa=_amplitude(ep,'q'); active=ra is not None and qa is not None and float(strength)>0
    classes=[]; rr=dh.pair(ep,'rr'); assignment=dh.blocks(ep.r_hw)
    for selected in (f,b):
        observed=ep.r[selected]; reps=dh.fps_rows(observed,16)
        member=np.argmax(dh.mm(observed,reps.T),axis=1)
        spatial_dist=[]
        for group in range(4):
            held=selected&(assignment==group); train=selected&(assignment!=group)
            if held.any() and train.any():
                spatial_dist.extend(dh.nearest_distance(rr[held],train).tolist())
        width=max(float(np.median(spatial_dist)) if spatial_dist else 1.,1e-5)
        amplitude_width=1.
        if active:
            # Source spatial pairs only; no query amplitude labels enter widths.
            a=ra[selected]; bid=assignment[selected]
            diff=np.abs(a[:,None]-a[None]); legal=bid[:,None]!=bid[None]
            amplitude_width=max(float(np.median(diff[legal])) if legal.any() else 1.,1e-5)
        classes.append((selected,reps,member,width,amplitude_width))
    def score(x,amplitude=None):
        values=[]
        for selected,reps,member,width,awidth in classes:
            cosine=np.clip(dh.mm(x,reps.T),-1.,1.)
            logs=-(2.-2.*cosine)/(2.*width*width)
            terms=[]; count=int(selected.sum())
            for j in range(len(reps)):
                ids=np.flatnonzero(selected)[member==j]
                if not len(ids):
                    continue
                if active and amplitude is not None:
                    alog=-float(strength)*(amplitude[:,None]-ra[ids][None])**2/(2.*awidth*awidth)
                    terms.append(logs[:,j]+logsumexp(alog,axis=1)-np.log(count))
                else:
                    terms.append(logs[:,j]+np.log(len(ids)/count))
            values.append(logsumexp(np.asarray(terms),axis=0))
        return values[0]-values[1]
    def scorer(x):
        if x is ep.q:
            a=qa
        else:
            # Source C asks for rows of R. Exact observed row lookup carries its
            # actual source amplitude, never the norm of final-unit descriptors.
            ids=np.argmax(dh.mm(x,ep.r.T),axis=1)
            a=None if ra is None else ra[ids]
        return score(x,a)
    scorer.source_score=lambda ids:score(ep.r[ids],None if ra is None else ra[ids])
    return scorer,dict(amplitude_active=active,amplitude_strength=float(strength),raw_observation='pre_final_LN',representatives_per_role=16)


def d160(ep):
    return dh.calibrate(ep,'D160',_d160_builder,(0.,1.,2.))


def d160_control(ep):
    return dh.calibrate(ep,'D160_control_same_angle_density',_d160_builder,(0.,))


def _matched(ep,selected):
    sim=dh.pair(ep); valid=ep.q_valid>0
    qnearest=np.argmax(np.where(ep.wvalid[None]>0,sim,-np.inf),axis=1)
    rnearest=np.argmax(np.where(valid[:,None],sim,-np.inf),axis=0)
    rid=np.flatnonzero(selected); qid=rnearest[rid]; mutual=qnearest[qid]==rid
    rid=rid[mutual]; qid=qid[mutual]
    return rid,qid


def _nonadjacent_both(ep,rid,qid):
    rb=dh.blocks(ep.r_hw,4)[rid]; qb=dh.blocks(ep.q_hw,4)[qid]
    for i in range(len(rid)):
        for j in range(i):
            ry,rx=divmod(int(rb[i]),4); sy,sx=divmod(int(rb[j]),4)
            qy,qx=divmod(int(qb[i]),4); ty,tx=divmod(int(qb[j]),4)
            if max(abs(ry-sy),abs(rx-sx))>1 and max(abs(qy-ty),abs(qx-tx))>1:
                return True
    return False


def _geometric_median(points):
    value=np.median(points,axis=0)
    for _ in range(20):
        distance=np.linalg.norm(points-value,axis=1)
        exact=np.flatnonzero(distance<1e-12)
        if len(exact):
            # A smoothed finite Weiszfeld update avoids falsely declaring any
            # observed coincident point the exact geometric median.
            distance=np.maximum(distance,1e-8)
        weight=1./distance; value=(points*weight[:,None]).sum(axis=0)/weight.sum()
    return value


def _d161_builder(ep,config,mean=False):
    cap=float(config); f,b,_=dh.pure(ep); rid,qid=_matched(ep,b); s0=dh.b0_field(ep)
    kept=s0[qid]<0; rid=rid[kept]; qid=qid[kept]
    if len(rid)<8 or not _nonadjacent_both(ep,rid,qid):
        return None,dict(pairs=len(rid),reason='insufficient_BG_pairs')
    difference=ep.q[qid]-ep.r[rid]
    _,group=dh.kmeans(difference,min(4,len(difference)))
    qualified=[]
    for k in np.unique(group):
        ids=np.flatnonzero(group==k)
        if len(ids)>=8 and _nonadjacent_both(ep,rid[ids],qid[ids]):
            rb=dh.blocks(ep.r_hw,4)[rid[ids]]
            means=[difference[ids[rb==v]].mean(axis=0) for v in np.unique(rb)]
            direction=difference[ids].mean(axis=0)
            if all(float(dh.mm(m,direction))>0 for m in means):
                qualified.extend(ids.tolist())
    if len(qualified)<8:
        return None,dict(pairs=len(rid),reason='conflicting_BG_directions')
    points=difference[qualified]
    shift=points.mean(axis=0) if mean else _geometric_median(points)
    norm=float(np.linalg.norm(shift)); limit=cap*max(dh.source_radius(ep,b),1e-5)
    if not np.isfinite(limit):
        return None,{}
    shift*=min(1.,limit/max(norm,1e-12))
    xr,xq,p=dh.project(ep); model=dh.head(ep,xr)
    def scorer(x):
        corrected=x-shift; corrected/=np.maximum(np.linalg.norm(corrected,axis=1,keepdims=True),1e-6)
        return dh.predict(model,dh.mm(corrected,p))
    return scorer,dict(matched_BG_pairs=len(rid),qualified_BG_pairs=len(qualified),shift_norm=float(np.linalg.norm(shift)),
                      median_iterations=20,shift_estimator='mean' if mean else 'geometric_median')


def d161(ep):
    return dh.calibrate(ep,'D161',_d161_builder,(.25,.5,1.))


def d161_control(ep):
    return dh.calibrate(ep,'D161_control_same_pairs_mean_shift',lambda ep,c:_d161_builder(ep,c,True),(.25,.5,1.))


def _fit_residual_map(x,y,rank,penalty):
    residual=y-x
    matrix=np.linalg.solve(dh.mm(x.T,x)+penalty*np.eye(x.shape[1]),dh.mm(x.T,residual))
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    matrix=(u[:,:rank]*s[:rank])@v[:rank]
    norm=np.linalg.norm(matrix)
    return matrix*min(1.,.1/max(float(norm),1e-12))


def _d162_builder(ep,config,control=False):
    k,rank,tolerance=config; xr,xq,p=dh.project(ep); f,b,_=dh.pure(ep); base=dh.b0_field(ep); head=dh.head(ep,xr)
    hs=dh.predict(head,xq); maps=[]; details=[]; matched=[]
    for selected,positive in ((f,True),(b,False)):
        rid,qid=_matched(ep,selected); agree=(base[qid]>0)&(hs[qid]>0) if positive else (base[qid]<0)&(hs[qid]<0)
        rid=rid[agree]; qid=qid[agree]
        if len(rid)<8 or len(np.unique(dh.blocks(ep.r_hw)[rid]))<2:
            return None,dict(reason='missing_role_pairs',pairs=len(rid))
        source=xq[qid]; dest=xr[rid]; penalty=.01*max(float(np.mean(np.sum(source*source,axis=1))),1e-6)
        full=_fit_residual_map(source,dest,rank,penalty); blocks=dh.blocks(ep.r_hw)[rid]; ratios=[]
        for block in np.unique(blocks):
            tr=blocks!=block; te=~tr
            if tr.sum()<rank+1:
                continue
            fit=_fit_residual_map(source[tr],dest[tr],rank,penalty)
            old=float(np.sum((source[te]-dest[te])**2)); new=float(np.sum((source[te]+dh.mm(source[te],fit)-dest[te])**2))
            ratios.append(new/old if old>1e-12 else (1. if new<=1e-12 else np.inf))
        if not ratios or max(ratios)>=1.-1e-8:
            return None,dict(reason='role_map_no_held_improvement',role='F' if positive else 'B',ratios=ratios)
        maps.append(full); matched.append((source,dest)); details.append(dict(role='F' if positive else 'B',pairs=len(rid),held_error_ratios=ratios))
    landmarks=dh.fps_rows(xr[ep.wvalid>0],min(64,len(xr)))
    if control:
        src=np.vstack([x[0] for x in matched]); dst=np.vstack([x[1] for x in matched])
        consensus=_fit_residual_map(src,dst,rank,.01)
        kept=rank
    else:
        average=.5*(maps[0]+maps[1]); u,s,v=np.linalg.svd(average,full_matrices=False); consensus=np.zeros_like(average); kept=0
        for j in range(min(rank,len(s))):
            component=np.outer(u[:,j]*s[j],v[j])
            a=dh.mm(landmarks,maps[0])@v[j]; bscore=dh.mm(landmarks,maps[1])@v[j]
            scale=max(float(np.mean(a*a+bscore*bscore)),1e-12)
            if float(np.mean((a-bscore)**2))<=float(tolerance)*scale and float(dh.mm(a,bscore))>0:
                consensus+=component; kept+=1
    if kept==0:
        return None,dict(reason='no_shared_map_direction',role_maps=details)
    def scorer(x):
        z=dh.mm(x,p); correction=dh.mm(z,consensus)
        # Lift only the identified auxiliary directions; all other original-D
        # coordinates remain untouched before the documented unit operation.
        right=np.linalg.solve(dh.mm(p.T,p)+1e-6*np.eye(p.shape[1]),p.T)
        corrected=x+dh.mm(correction,right)
        corrected/=np.maximum(np.linalg.norm(corrected,axis=1,keepdims=True),1e-6)
        return dh.b0_field(ep,corrected)
    return scorer,dict(common_directions=kept,role_maps=details,map_frobenius=float(np.linalg.norm(consensus)),landmark_disagreement_compared=True)


def d162(ep):
    return dh.calibrate(ep,'D162',_d162_builder)


def d162_control(ep):
    return dh.calibrate(ep,'D162_control_same_pairs_single_shrunk_map',lambda ep,c:_d162_builder(ep,c,True))


def _d163_builder(ep,config,free=False):
    k,rank,penalty=config; xr,xq,p=dh.project(ep); f,b,_=dh.pure(ep)
    if not f.any() or not b.any():
        return None,{}
    fg,group=dh.kmeans(xr[f],k,spherical=False); bg,_=dh.kmeans(xq[ep.q_valid>0],min(8,k),spherical=False)
    original_fg,_=dh.kmeans(ep.r[f],k); radius=dh.source_radius(ep,f)
    if len(fg)<2 or not np.isfinite(radius):
        return None,dict(reason='fewer_than_two_F_modes')
    sim=dh.mm(ep.q,original_fg.T); winners=np.argmax(sim,axis=1); distances=np.sqrt(np.maximum(0.,2.-2.*np.max(sim,axis=1)))
    # Direct support belongs to observed source tokens. A synthesized spherical
    # centroid may lie farther away than every observed member; it must not
    # silently impose an additional stricter support test on genuine anchors.
    support=np.zeros(len(ep.q),bool); support[dh.qualified_anchors(ep,True)]=True
    if len(np.unique(winners[support]))<2:
        return None,dict(reason='fewer_than_two_directly_supported_modes')
    _,_,axes=np.linalg.svd(fg-fg.mean(axis=0),full_matrices=False); basis=axes[:min(rank,len(axes))].T
    initial=fg.copy(); weights=np.full(len(fg)+len(bg),1./(len(fg)+len(bg))); shift=np.zeros_like(fg) if free else np.zeros(xr.shape[1])
    q=xq[ep.q_valid>0]; variance=max(float(np.median(np.sum((xr[f]-fg[group])**2,axis=1)))/xr.shape[1],1e-4)
    best=None
    for _ in range(20):
        centers=np.vstack((initial+shift,bg)); dist=np.maximum(0.,np.sum(q*q,axis=1)[:,None]+np.sum(centers*centers,axis=1)-2*dh.mm(q,centers.T))
        logs=np.log(np.maximum(weights,1e-12))-dist/(2*variance); normal=logsumexp(logs,axis=1); responsibility=np.exp(logs-normal[:,None])
        objective=float(-normal.sum()+penalty*len(q)*np.sum(shift*shift)/(2*variance))
        if best is None or objective<best[0]:
            best=(objective,centers.copy(),weights.copy(),np.array(shift,copy=True))
        mass=responsibility.sum(axis=0); weights=np.maximum(mass,1e-12)/max(float(mass.sum()),1e-12)
        if free:
            displacement=dh.mm(responsibility[:,:len(fg)].T,q)-mass[:len(fg),None]*initial
            shift=displacement/(mass[:len(fg),None]+penalty*len(q))
        else:
            displacement=dh.mm(responsibility[:,:len(fg)].T,q).sum(axis=0)-(mass[:len(fg),None]*initial).sum(axis=0)
            candidate=displacement/(mass[:len(fg)].sum()+penalty*len(q))
            shift=dh.mm(dh.mm(candidate,basis),basis.T)
        bg=dh.mm(responsibility[:,len(fg):].T,q)/np.maximum(mass[len(fg):,None],1e-12)
    def scorer(x):
        z=dh.mm(x,p); dist=np.maximum(0.,np.sum(z*z,axis=1)[:,None]+np.sum(best[1]*best[1],axis=1)-2*dh.mm(z,best[1].T))
        logs=np.log(np.maximum(best[2],1e-12))-dist/(2*variance)
        return logsumexp(logs[:,:len(fg)],axis=1)-logsumexp(logs[:,len(fg):],axis=1)
    actual=best[1][:len(fg)]
    discrepancy=float(np.max(np.abs((actual[:,None]-actual[None])-(initial[:,None]-initial[None]))))
    return scorer,dict(supported_modes=len(np.unique(winners[support])),fg_modes=len(fg),bg_modes=len(bg),
                      retained_best_objective=best[0],mode_difference_discrepancy=discrepancy,free_mode_shifts=free,iterations=20)


def d163(ep):
    return dh.calibrate(ep,'D163',_d163_builder)


def d163_control(ep):
    return dh.calibrate(ep,'D163_control_same_initial_modes_free_shifts',lambda ep,c:_d163_builder(ep,c,True))


def _layer_features(ep,x,groups,reference,selected,k,mode):
    """Cross-spatial-block routing and CDF: a scored point never enters its null."""
    f,b,_=dh.pure(ep); sim=dh.mm(x,ep.r.T); target=np.max(sim[:,f],axis=1); original=dh.b0_field(ep,x)
    output=np.c_[original,target]; counts=[]
    for block in np.unique(groups):
        tested=groups==block; pool=selected&(groups!=block)
        if pool.sum()<8:
            counts.append(0);continue
        centers,which=dh.kmeans(reference[pool],k)
        affinity=dh.mm(x[tested],centers.T); affinity-=np.max(affinity,axis=1,keepdims=True)
        routing=np.exp(affinity/.1); routing/=routing.sum(axis=1,keepdims=True)
        source_sim=dh.mm(reference[pool],ep.r[f].T); null=np.max(source_sim,axis=1)
        transformed=np.zeros(tested.sum()); good=np.zeros(tested.sum())
        for layer in range(len(centers)):
            values=null[which==layer]
            if len(values)<8:
                continue
            if mode=='zscore':
                statistic=(target[tested]-np.mean(values))/max(float(np.std(values)),1e-5)
            else:
                statistic=np.searchsorted(np.sort(values),target[tested],side='right')/len(values)
            transformed+=routing[:,layer]*statistic; good+=routing[:,layer]
        usable=good>1e-6; ids=np.flatnonzero(tested)[usable]
        output[ids,0]=transformed[usable]/good[usable]; counts.append(int(pool.sum()))
    return output,counts


def _d164_builder(ep,config,mode='cdf'):
    k,_,_=config; f,b,_=dh.pure(ep)
    if not f.any() or not b.any():
        return None,{}
    source_groups=dh.blocks(ep.r_hw); query_groups=dh.blocks(ep.q_hw)
    source_features,source_counts=_layer_features(ep,ep.r,source_groups,ep.r,b,k,mode)
    query_neg=(dh.b0_field(ep)<0)&(ep.q_valid>0)
    query_features,query_counts=_layer_features(ep,ep.q,query_groups,ep.q,query_neg,k,mode)
    model=dh.head(ep,source_features)
    def scorer(x):
        if x is ep.q:
            return dh.predict(model,query_features)
        ids=np.argmax(dh.mm(x,ep.r.T),axis=1)
        return dh.predict(model,source_features[ids])
    scorer.source_score=lambda ids:dh.predict(model,source_features[ids])
    return scorer,dict(source_layer_pool_sizes=source_counts,query_layer_pool_sizes=query_counts,
                      background_layers=k,statistic=mode,scored_block_excluded_from_routing_and_CDF=True)


def d164(ep):
    return dh.calibrate(ep,'D164',_d164_builder)


def d164_control(ep):
    return dh.calibrate(ep,'D164_control_same_layers_zscore',lambda ep,c:_d164_builder(ep,c,'zscore'))


METHODS={'D160':d160,'D161':d161,'D162':d162,'D163':d163,'D164':d164}
CONTROLS={'D160_control_same_angle_density':d160_control,'D161_control_same_pairs_mean_shift':d161_control,
          'D162_control_same_pairs_single_shrunk_map':d162_control,'D163_control_same_initial_modes_free_shifts':d163_control,
          'D164_control_same_layers_zscore':d164_control}
