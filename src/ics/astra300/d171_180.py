"""Faithful episode-local CPU implementations of supplied Astra D171--D180.

Actual DINO observations are mandatory where a card requests them. The source
selection always includes literal D B0 at zero; controls are separately named.
Unspecified numerical choices are recorded in the accompanying assumptions.
"""
from __future__ import annotations

from itertools import product
from functools import partial
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.ndimage import label
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from PIL import Image

from .common import EPS, ArtifactUnavailable, artifact, array_hash
from . import d_helpers_151_200 as dh
from . import d_helpers_171_180 as ph
from ics.cpu100.common import rgb_view


def _piecewise(ep,xr,k):
    centers,modes=ph.mode_ids(ep,min(k,8))
    if not len(centers):
        return None
    models=[]
    for j in range(len(centers)):
        fit=dh.role_head(ep,xr,foreground=(modes==j))
        if fit is None:
            return None
        models.append(ph.parameter(fit))
    return np.asarray(models),modes


def _piece_score(theta,z):
    return np.max(dh.mm(z,theta.T),axis=1)


def _piece_loss_grad(ep,theta,z):
    y,weights=ph.balance(ep);logits=dh.mm(z,theta.T);which=np.argmax(logits,axis=1)
    score=logits[np.arange(len(z)),which];residual=weights*(expit(score)-y)
    penalty=ph.source_penalty(z[:,:-1],weights)
    grad=np.zeros_like(theta)
    for j in range(len(theta)):
        grad[j]=dh.mm(z[which==j].T,residual[which==j])
    grad[:,:-1]+=penalty*theta[:,:-1]
    value=float(np.sum(weights*(np.logaddexp(0.,score)-y*score)))+.5*penalty*float(np.sum(theta[:,:-1]**2))
    return value,grad


def _dedup_density(ep):
    valid=ep.q_valid>0
    if not valid.any():
        return np.zeros(len(ep.q)),0
    centers,ids=dh.kmeans(ep.q[valid],min(64,int(valid.sum())))
    width=ph.median_width(centers)
    density=np.mean(np.exp(-ph.sqdist(centers,centers)/(2*width)),axis=1)
    counts=np.bincount(ids,minlength=len(centers))
    # Each observed cluster has ONE effective occurrence in the kernel density
    # and ONE total query mass. Copies of a token do not increase its force.
    rho=np.zeros(len(ep.q));rho[valid]=density[ids]/counts[ids]
    rho/=max(float(rho.sum()),EPS)
    return rho,len(centers)


def _d171_builder(ep,cfg,kind='density'):
    k,radius_factor,strength=cfg;xr,xq,_=dh.project(ep);zr,zq=ph.augmented(xr),ph.augmented(xq)
    initial=_piecewise(ep,xr,k)
    if initial is None:
        return None,{}
    theta,modes=initial;origin=theta.copy();rho,cluster_count=_dedup_density(ep)
    if kind=='supervised':
        strength=0.
    states=[]
    for train,held in dh.folds(ep):
        reduced=dh.restricted(ep,train);fit=_piecewise(reduced,xr,k)
        if fit is None:
            return None,dict(reason='missing_nested_source_mode')
        value,assignment=fit
        states.append([reduced,held,value,value.copy(),assignment,_piece_score(value,zr)])
    radius=.05*radius_factor*max(float(np.linalg.norm(origin)),1.)
    def objective_grad(reduced,t,anchor):
        loss,grad=_piece_loss_grad(reduced,t,zr)
        logits=dh.mm(zq,t.T);winner=np.argmax(logits,axis=1);score=logits[np.arange(len(zq)),winner]
        if kind=='entropy':
            p=expit(score);value=np.logaddexp(0.,score)-p*score
            derivative=-score*p*(1-p)
        else:
            value=np.exp(-np.abs(score));derivative=-np.sign(score)*value
        loss+=float(strength*np.sum(rho*value))+.5*float(np.sum((t-anchor)**2))
        for j in range(len(t)):
            here=winner==j;grad[j]+=strength*dh.mm(zq[here].T,rho[here]*derivative[here])
        grad+=t-anchor
        return loss,grad
    best=theta.copy();best_obj=objective_grad(ep,theta,origin)[0];accepted=0;history=[best_obj]
    # Twenty total adaptation rounds, including rejected proposals. Nested
    # source models evolve in the same rounds and never train on their holdout.
    for _ in range(20 if strength>0 else 0):
        _,gradient=objective_grad(ep,theta,origin)
        rate=.1/max(1.,float(np.linalg.norm(gradient)))
        proposed=ph.trust(theta-rate*gradient,origin,radius);changed=[];safe=True
        for reduced,held,t,anchor,assignment,before in states:
            _,g=objective_grad(reduced,t,anchor)
            candidate=ph.trust(t-rate*g,anchor,.05*radius_factor*max(float(np.linalg.norm(anchor)),1.))
            safe &= ph.recall_safe(ep,before,_piece_score(candidate,zr),assignment,held)
            changed.append(candidate)
        if not safe:
            history.append(best_obj);continue
        theta=proposed
        for state,t in zip(states,changed):
            state[2]=t
        accepted+=1;value=objective_grad(ep,theta,origin)[0]
        if value<best_obj:
            best_obj=value;best=theta.copy()
        history.append(best_obj)
    if kind=='density' and strength>0 and best_obj>=history[0]-1e-12:
        return None,dict(reason='no_feasible_low_density_improvement',adaptation_rounds=20,
                         accepted_rounds=accepted,objective_trace=history)
    projection=dh.project(ep)[2]
    scorer=ph.semantic_score(ep,lambda x:_piece_score(best,ph.augmented(dh.mm(x,projection))))
    return scorer,dict(local_FG_units=len(theta),density_kind=kind,query_unique_clusters=cluster_count,
                       density_uses_unique_cluster_mass=True,adaptation_rounds=20 if strength>0 else 0,
                       accepted_rounds=accepted,best_visited_objective=best_obj,objective_trace=history,
                       source_mode_OOF_recall_guard=True,trust_radius=radius)


def d171(ep):
    return dh.calibrate(ep,'D171',_d171_builder)


def _d172_builder(ep,cfg,kind='block_cone'):
    k,radius,strength=cfg;xr,xq,p=dh.project(ep);model=dh.head(ep,xr)
    if model is None:
        return None,{}
    theta=ph.parameter(model);zr,zq=ph.augmented(xr),ph.augmented(xq)
    positive=dh.qualified_anchors(ep,True);negative=dh.qualified_anchors(ep,False)
    ids=np.r_[positive,negative];targets=np.r_[np.ones(len(positive)),np.zeros(len(negative))]
    if not len(ids):
        return ph.semantic_score(ep,lambda x:dh.predict(model,dh.mm(x,p))),dict(anchors=0,adaptation_step=0.)
    qw=np.full(len(ids),strength/len(ids));_,gq=ph.loss_gradient(theta,zq[ids],targets,qw)
    _,modes=ph.mode_ids(ep,k);selectors=[]
    source_y,source_w=ph.balance(ep)
    for block in range(4):
        selected=(dh.blocks(ep.r_hw)==block)&(ep.wvalid>0)
        if selected.any():
            selectors.append((source_y,source_w*selected/max(float((source_w*selected).sum()),EPS)))
    for mode in np.unique(modes[ep.wf>0]):
        if mode>=0:
            fg=ep.wf*(modes==mode)
            selectors.append((np.ones(len(ep.r)),fg/max(float(fg.sum()),EPS)))
    if kind in ('global','shrink'):
        selectors=[(source_y,source_w)]
    gradients=[];reference=[]
    for y,w in selectors:
        value,gradient=ph.loss_gradient(theta,zr,y,w)
        gradients.append(gradient);reference.append((y,w,value))
    gradients=np.asarray(gradients)
    if kind=='plain':
        direction=-gq
    elif kind=='shrink':
        direction=-gq/(1+radius)
    else:
        bound=.05*radius*max(float(np.linalg.norm(theta)),1.)
        result=minimize(lambda v:.5*float((v+gq)@(v+gq)),np.zeros_like(theta),
                        jac=lambda v:v+gq,method='SLSQP',
                        constraints=[{'type':'ineq','fun':lambda v:-dh.mm(gradients,v),
                                      'jac':lambda v:-gradients},
                                     {'type':'ineq','fun':lambda v:bound*bound-float(v@v),
                                      'jac':lambda v:-2*v}],
                        options={'maxiter':50,'ftol':1e-10})
        direction=result.x
        if not result.success or np.max(dh.mm(gradients,direction),initial=0.)>1e-7 or np.linalg.norm(direction)>bound+1e-7:
            direction=np.zeros_like(theta)
    best=theta;best_step=0.;best_loss=ph.loss_gradient(theta,zq[ids],targets,qw)[0]
    checks=[]
    for step in (0.,.25,.5,1.):
        proposed=theta+step*direction
        safe=(kind=='plain' or all(ph.loss_gradient(proposed,zr,y,w)[0]<=old+1e-10 for y,w,old in reference))
        checks.append(bool(safe))
        value=ph.loss_gradient(proposed,zq[ids],targets,qw)[0]
        if safe and value<best_loss-1e-12:
            best=proposed;best_loss=value;best_step=step
    return ph.semantic_score(ep,lambda x:dh.mm(ph.augmented(dh.mm(x,p)),best)),dict(
        anchors=len(ids),constraint_kind=kind,constraints=len(gradients),direction_norm=float(np.linalg.norm(direction)),
        adaptation_step=best_step,actual_source_loss_checks=checks,QP_maxiter=50,pseudolabels_frozen_once=True)


def d172(ep):
    return dh.calibrate(ep,'D172',_d172_builder)


_OCCLUSION_CACHE={}


def _d173_views(ep,original_mask,random=False):
    image=np.asarray(rgb_view(ep,'r'),float)/255.;g=ep.reference_geometry;side=image.shape[0]
    sh,sw=g['resized_hw'];oy,ox=g['padding_top_left'];mask=np.zeros((side,side),bool)
    mask[oy:oy+sh,ox:ox+sw]=np.asarray(Image.fromarray(original_mask.astype(np.uint8)).resize((sw,sh),Image.Resampling.NEAREST))>0
    yy=np.minimum(np.arange(side)*ep.r_hw[0]//side,ep.r_hw[0]-1)
    xx=np.minimum(np.arange(side)*ep.r_hw[1]//side,ep.r_hw[1]-1)
    known=(ep.wvalid>0).reshape(ep.r_hw)[yy[:,None],xx[None,:]]
    physical=np.zeros((side,side),bool);physical[oy:oy+sh,ox:ox+sw]=True
    known &=physical;mask &=known
    block_canvas=dh.blocks(ep.r_hw).reshape(ep.r_hw)[yy[:,None],xx[None,:]]
    # Only fully observed, actual BG tiles may fill a removed foreground.
    tiles=[]
    for idx in np.flatnonzero((ep.wvalid>0)&(ep.wf==0)):
        y,x=divmod(int(idx),ep.r_hw[1]);ys=slice(y*side//ep.r_hw[0],(y+1)*side//ep.r_hw[0]);xs=slice(x*side//ep.r_hw[1],(x+1)*side//ep.r_hw[1])
        if np.all(known[ys,xs]&~mask[ys,xs]):
            tiles.append((int(idx),image[ys,xs].copy()))
    if not tiles:
        return [],dict(reason='no_legal_observed_BG_fill_tile')
    rng=np.random.default_rng(0);groups=[(0,),(1,),(2,),(3,),(0,1),(0,2),(1,3),(2,3)]
    views=[(ep.r.copy(),ep.wf.copy(),ep.wb.copy())];receipt=[]
    encoder=artifact(ep,'frozen_encode_rgb')
    for j,group in enumerate(groups):
        removed=mask&np.isin(block_canvas,group)
        if random:
            # Same removed pixel count, fixed random FG ranks. Visibility and
            # BG exclusions are recomputed from the actual replacement mask.
            candidates=np.flatnonzero(mask);take=int(removed.sum());removed=np.zeros_like(mask)
            removed.ravel()[rng.permutation(candidates)[:take]]=True
        if not removed.any():
            continue
        token_removed=np.asarray(Image.fromarray(removed.astype(np.float32),mode='F').resize((ep.r_hw[1],ep.r_hw[0]),Image.Resampling.BOX)).ravel()
        visible=ep.wf-np.minimum(ep.wf,token_removed)
        if np.count_nonzero(visible>0)<2:
            continue
        tid,tile=tiles[j%len(tiles)];tiled=np.tile(tile,(int(np.ceil(side/tile.shape[0])),int(np.ceil(side/tile.shape[1])),1))[:side,:side]
        changed=image.copy();changed[removed]=tiled[removed]
        key=(ep.source_id,array_hash(changed),id(encoder),tuple(ep.r_hw))
        if key not in _OCCLUSION_CACHE:
            z=np.asarray(encoder('r',changed,working_canvas=True,side=side,raw=False),float)
            if z.shape!=tuple(ep.r_hw)+(ep.r.shape[1],) or not np.isfinite(z).all():
                raise ValueError('D173 requires an actual aligned frozen-DINO view grid')
            if len(_OCCLUSION_CACHE)>=48:
                _OCCLUSION_CACHE.clear()
            _OCCLUSION_CACHE[key]=ph.normalize(z.reshape(ep.r.shape))
        z=_OCCLUSION_CACHE[key]
        # Removed pixels are neither FG nor negative training examples.
        views.append((z,visible,ep.wb.copy()))
        receipt.append(dict(group=group,BG_fill_token=tid,removed_pixels=int(removed.sum())))
    return views,dict(actual_reference_view_requests=len(receipt),original_plus_visible_views=len(views),views=receipt,
                      removed_pixels_never_BG=True,random_occlusion_control=random)


def _d173_builder(ep,aggregation,original_mask,kind='directed'):
    views,info=_d173_views(ep,original_mask,random=(kind=='random'))
    if not views or len(views)<2:
        return None,info
    capacity=32;banks=[]
    for observed,fw,bw in views:
        c=np.divide(fw,fw+bw,out=np.zeros_like(fw),where=fw+bw>0)
        f=(fw+bw>0)&(c>=.9);b=(fw+bw>0)&(c<=.1)
        if f.sum()<8:f=fw>0
        if b.sum()<8:b=bw>0
        if not f.any() or not b.any():
            continue
        banks.append((dh.fps_rows(observed[f],capacity),dh.fps_rows(observed[b],capacity)))
    if not banks:
        return None,info
    if kind=='original':
        banks=banks[:1]*len(banks)
    def score(x):
        fields=np.asarray([np.max(dh.mm(x,f.T),axis=1)-np.max(dh.mm(x,b.T),axis=1) for f,b in banks])
        if aggregation=='median':
            return np.median(fields,axis=0)
        ordered=np.sort(fields,axis=0);trim=int(len(fields)*.2)
        return np.mean(ordered[trim:len(fields)-trim],axis=0)
    return ph.semantic_score(ep,score),dict(info,aggregation=aggregation,prototypes_per_view_per_role=capacity)


def _d173(ep,kind):
    f,b=dh.role(ep)
    if not f.any() or not b.any():
        return dh.finish(ep,dh.b0_field(ep),'D173' if kind=='directed' else 'D173_control_'+kind,branch='single_role_documented_B0')
    if ep.reference_mask is None or ep.r_rgb is None or ep.reference_mask.shape!=ep.r_rgb.shape[:2] or not ep.reference_geometry:
        raise ArtifactUnavailable('D173 requires source-bound original R RGB, full mask and geometry')
    artifact(ep,'frozen_encode_rgb')
    # The full pixel mask is only consulted inside the fitting object's known
    # pixels. _d173_views clears ALL held/buffer pixels BEFORE grouping/filling.
    return dh.calibrate(ep,'D173' if kind=='directed' else 'D173_control_'+kind,
                        partial(_d173_builder,original_mask=ep.reference_mask,kind=kind),('median','trimmed'))


def d173(ep):
    return _d173(ep,'directed')


def _weighted_head(ep,xr,block_weights):
    weight=np.asarray(block_weights)[dh.blocks(ep.r_hw)]
    return dh.role_head(ep,xr,foreground=weight,background=weight)


def _d174_builder(ep,cfg,kind='influence'):
    k,_,fraction=cfg;xr,xq,p=dh.project(ep);base_head=dh.head(ep,xr)
    if base_head is None:
        return None,{}
    oof,records=ph.oof(ep,xr);available=np.isfinite(oof)
    if not available.any():
        return None,{}
    original_q=dh.predict(base_head,xq);influences=[];improvements=[]
    for block in range(4):
        weights=np.ones(4);weights[block]=0.;model=_weighted_head(ep,xr,weights)
        if model is None:
            influences.append(np.zeros(len(ep.q),bool));improvements.append(-np.inf);continue
        influences.append((dh.predict(model,xq)>0)!=(original_q>0))
        altered=np.full(len(ep.r),np.nan)
        for reduced,held,_ in records:
            fit=_weighted_head(reduced,xr,weights)
            if fit is not None:
                altered[held]=dh.predict(fit,xr[held])
        chosen=available&np.isfinite(altered)
        improvements.append(dh.risk(ep,oof,chosen)-dh.risk(ep,altered,chosen) if chosen.any() else -np.inf)
    influences=np.asarray(influences);eligible=[];ratios=[]
    for block in range(4):
        exclusive=influences[block]&(influences.sum(axis=0)==1)&(ep.q_valid>0)
        connected,count=label(exclusive.reshape(ep.q_hw));largest=max(np.bincount(connected.ravel())[1:],default=0)
        ratio=largest/max(int(np.count_nonzero(ep.q_valid>0)),1);ratios.append(ratio)
        eligible.append(improvements[block]>1e-12 and (kind=='errors_only' or ratio>=float(fraction)))
    if not any(eligible):
        return None,dict(reason='no_OOF_bad_block_with_exclusive_query_change',source_error_improvements=improvements,
                         exclusive_change_ratios=ratios)
    _,modes=ph.mode_ids(ep,k);best=(dh.risk(ep,oof,available),0);winning=None;count=0
    for weights in product((0.,.5,1.),repeat=4):
        if any(weights[j]<1. and not eligible[j] for j in range(4)):
            continue
        token_weights=np.asarray(weights)[dh.blocks(ep.r_hw)]
        if any(not np.any((modes==m)&(ep.wf>0)&(token_weights>0)) for m in np.unique(modes[ep.wf>0])):
            continue
        scores=np.zeros(len(ep.r));legal=True
        for reduced,held,_ in records:
            fit=_weighted_head(reduced,xr,weights)
            if fit is None:
                legal=False;break
            scores[held]=dh.predict(fit,xr[held])
        if not legal:
            continue
        count+=1;key=(dh.risk(ep,scores,available),sum(1.-np.asarray(weights)))
        if key<best:
            best=key;winning=weights
    if winning is None:
        return None,dict(reason='no_source_improving_legal_weighting',enumerated_legal_weights=count)
    final=_weighted_head(ep,xr,winning)
    return ph.semantic_score(ep,lambda x:dh.predict(final,dh.mm(x,p))),dict(block_weights=winning,
        source_error_improvements=improvements,exclusive_change_ratios=ratios,enumerated_legal_weights=count,
        maximum_weight_combinations=81,all_FG_modes_represented=True,FG_and_BG_weights_changed_together=True)


def d174(ep):
    return dh.calibrate(ep,'D174',_d174_builder)


def _attribute_groups(ep,xr):
    y,w=ph.balance(ep);active=w>0
    centered=xr[active]-np.average(xr[active],axis=0,weights=w[active])
    strength=np.abs(dh.mm(xr.T,w*(y-.5)))
    keep=np.flatnonzero(strength>EPS)
    keep=keep[np.argsort(-strength[keep],kind='stable')][:16]
    if len(keep)<2:
        return []
    covariance=dh.mm(centered[:,keep].T,w[active,None]*centered[:,keep])
    sd=np.sqrt(np.maximum(np.diag(covariance),EPS));corr=covariance/(sd[:,None]*sd[None])
    distance=np.clip(1.-np.abs(corr),0.,1.);distance=(distance+distance.T)/2;np.fill_diagonal(distance,0.)
    tree=linkage(squareform(distance,checks=False),method='average')
    labels=fcluster(tree,min(4,len(keep)),criterion='maxclust')
    # scipy's maxclust gives a single group for two leaves; two distinct
    # supervised directions still form two legal attributes.
    if len(np.unique(labels))<2:
        labels=np.arange(len(keep))%min(4,len(keep))+1
    return [keep[labels==j] for j in np.unique(labels)]


def _d175_fit(ep,cfg,kind='contrast'):
    xr,xq,p=dh.project(ep);groups=_attribute_groups(ep,xr)
    if len(groups)<2:
        return None,dict(reason='fewer_than_two_supervised_attributes')
    columns=np.concatenate(groups);xa=xr[:,columns];qa=xq[:,columns]
    valid=np.flatnonzero(ep.q_valid>0);blocks=dh.blocks(ep.q_hw)
    spatial=[j for j in range(4) if np.any((blocks==j)&(ep.q_valid>0))]
    if len(spatial)<len(groups):
        return None,dict(reason='insufficient_distinct_query_spatial_blocks')
    rng=np.random.default_rng(0);noise=np.zeros((2*len(ep.r),len(columns)))
    offset=0
    for j,group in enumerate(groups):
        indices=[]
        for row in range(len(noise)):
            block=spatial[(row+j)%len(spatial)];pool=valid[blocks[valid]==block]
            indices.append(int(pool[rng.integers(len(pool))]))
        noise[:,offset:offset+len(group)]=xq[np.asarray(indices)][:,group];offset+=len(group)
    width=ph.median_width(xa[ep.wvalid>0]);features=ph.rff(xa,width,64)
    fmass=float(ep.wf.sum())
    contrast_x=np.r_[features(xa),features(noise)];target=np.r_[np.ones(len(xa)),np.zeros(len(noise))]
    weights=np.r_[.5*ep.wf/max(fmass,EPS),np.full(len(noise),.5/len(noise))]
    contrast=dh.logistic(contrast_x,target,weights)
    def h(rows):
        return dh.predict(contrast,features(dh.mm(rows,p)[:,columns]))
    if kind=='frequency':
        variance=np.var(qa[valid],axis=0)+EPS;direction=np.average(xa,axis=0,weights=ep.wf)-np.average(xa,axis=0,weights=ep.wb)
        def h(rows):
            return dh.mm(dh.mm(rows,p)[:,columns],direction/np.sqrt(variance))
    if kind=='profile_ridge':
        phi,count=ph.profile(ep);pr=phi(ep.r);y,w=ph.balance(ep)
        centers=dh.fps_rows(pr[ep.wvalid>0],64);bw=ph.median_width(pr[ep.wvalid>0])
        def kernel(rows):
            return np.exp(-ph.sqdist(phi(rows),centers)/(2*bw))
        theta=ph.ridge(kernel(ep.r),2*y-1,w,.01)
        return ph.semantic_score(ep,lambda rows:dh.mm(ph.augmented(kernel(rows)),theta)),dict(profile_columns=count,control='full_profile_RBF_ridge')
    reference=np.c_[h(ep.r),dh.b0_field(ep,ep.r)];semantic=dh.head(ep,reference)
    def raw(rows):
        return dh.predict(semantic,np.c_[h(rows),dh.b0_field(ep,rows)])
    return ph.semantic_score(ep,raw),dict(attribute_groups=[g.tolist() for g in groups],RFF_features=64,
        synthetic_noise_rows=len(noise),synthetic_noise_is_semantic_BG=False,query_samples_not_relabelled=True,
        source_bandwidth_squared=width,contrast_kind=kind)


def _d175_builder(ep,cfg,kind='contrast'):
    scorer,info=_d175_fit(ep,cfg,kind)
    if scorer is None or kind!='contrast':
        return scorer,info
    changed=np.zeros(len(ep.r));base=np.zeros(len(ep.r));available=np.zeros(len(ep.r),bool)
    for train,held in dh.folds(ep):
        reduced=dh.restricted(ep,train);fit,_=_d175_fit(reduced,cfg,kind)
        if fit is None:
            return None,dict(info,reason='no_rebuildable_source_increment')
        changed[held]=fit.source_score(np.flatnonzero(held));base[held]=dh.b0_field(reduced,ep.r[held]);available|=held
    if not available.any() or dh.risk(ep,changed,available)>=dh.risk(ep,base,available)-1e-12:
        return None,dict(info,reason='no_OOF_attribute_increment')
    return scorer,dict(info,OOF_increment_guard=True)


def d175(ep):
    # No card parameter controls a contrast mechanism; one fixed fit plus B0
    # avoids listing identical nominal (K,r,lambda) copies as different fits.
    return dh.calibrate(ep,'D175',_d175_builder,((4,2,1.),))


METHODS={f'D{i}':globals()[f'd{i}'] for i in range(171,176)}
CONTROLS={
    'D171_control_supervised':lambda ep:dh.calibrate(ep,'D171_control_supervised',partial(_d171_builder,kind='supervised')),
    'D171_control_entropy':lambda ep:dh.calibrate(ep,'D171_control_entropy',partial(_d171_builder,kind='entropy')),
    'D172_control_plain_step':lambda ep:dh.calibrate(ep,'D172_control_plain_step',partial(_d172_builder,kind='plain')),
    'D172_control_global_loss':lambda ep:dh.calibrate(ep,'D172_control_global_loss',partial(_d172_builder,kind='global')),
    'D172_control_L2_step':lambda ep:dh.calibrate(ep,'D172_control_L2_step',partial(_d172_builder,kind='shrink')),
    'D173_control_random_occlusion':lambda ep:_d173(ep,'random'),
    'D173_control_same_dictionary_original':lambda ep:_d173(ep,'original'),
    'D174_control_errors_only':lambda ep:dh.calibrate(ep,'D174_control_errors_only',partial(_d174_builder,kind='errors_only')),
    'D175_control_profile_RBF_ridge':lambda ep:dh.calibrate(ep,'D175_control_profile_RBF_ridge',partial(_d175_builder,kind='profile_ridge'),((4,2,1.),)),
    'D175_control_query_channel_frequency':lambda ep:dh.calibrate(ep,'D175_control_query_channel_frequency',partial(_d175_builder,kind='frequency'),((4,2,1.),)),
}
