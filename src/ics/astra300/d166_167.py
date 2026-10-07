"""A background domain field and supervised importance weights, with full masks."""
from __future__ import annotations
import numpy as np
from .common import continuous_original,Result
from . import d_helpers_151_200 as dh
from .d160_164 import _matched,_nonadjacent_both,_geometric_median
from ics.methods.direct_dino_features import resize


def _d166_field(ep,smooth,zero=False):
    f,b,_=dh.pure(ep);rid,qid=_matched(ep,b);negative=dh.b0_field(ep)[qid]<0
    rid=rid[negative];qid=qid[negative]
    if len(rid)<8:return None,dict(reason='insufficient_BG_pairs')
    difference=ep.q[qid]-ep.r[rid];qh,qw=ep.q_hw
    cy=(qid//qw+.5)/qh;cx=(qid%qw+.5)/qw
    values=np.zeros((16,ep.r.shape[1]));mass=np.zeros(16);observed=[]
    for y in range(4):
        for x in range(4):
            cell=4*y+x;inside=(np.abs(cy-(y+.5)/4)<=.375)&(np.abs(cx-(x+.5)/4)<=.375)
            ids=np.flatnonzero(inside)
            if len(ids)<8 or not _nonadjacent_both(ep,rid[ids],qid[ids]):continue
            vector=_geometric_median(difference[ids]);rb=dh.blocks(ep.r_hw,4)[rid[ids]]
            if any(float(dh.mm(difference[ids[rb==v]].mean(axis=0),vector))<=0 for v in np.unique(rb)):continue
            values[cell]=vector;mass[cell]=len(ids);observed.append(cell)
    if not observed:return None,dict(reason='all_local_blocks_unanchored')
    edges=[]
    for y in range(4):
        for x in range(4):
            if y<3:edges.append((4*y+x,4*(y+1)+x))
            if x<3:edges.append((4*y+x,4*y+x+1))
    lap=np.zeros((16,16))
    for i,j in edges:lap[i,i]+=1;lap[j,j]+=1;lap[i,j]-=1;lap[j,i]-=1
    fidelity=mass/max(float(mass.mean()),1e-6)
    field=np.linalg.solve(np.diag(fidelity)+float(smooth)*lap,fidelity[:,None]*values)
    # A bound on DOMAIN differences comes only from source BG block means.
    source_blocks=dh.blocks(ep.r_hw,4);source=[]
    for g in range(16):
        ids=b&(source_blocks==g)
        source.append(ep.r[ids].mean(axis=0) if ids.any() else None)
    limits=[np.linalg.norm(source[i]-source[j]) for i,j in edges if source[i] is not None and source[j] is not None]
    if not limits:return None,dict(reason='source_domain_gradient_bound_unidentified')
    bound=max(float(np.quantile(limits,.95)),1e-6)
    largest=max(float(np.linalg.norm(field[i]-field[j])) for i,j in edges)
    mean=field.mean(axis=0);field=mean+(field-mean)*min(1.,bound/max(largest,1e-12))
    if zero:field.fill(0.)
    dense=resize(field.reshape(4,4,-1),ep.q_hw).reshape(ep.q.shape)
    corrected=ep.q-dense;corrected/=np.maximum(np.linalg.norm(corrected,axis=1,keepdims=True),1e-6)
    return dh.b0_field(ep,corrected),dict(observed_domain_blocks=observed,smoothing=float(smooth),domain_gradient_bound=bound,
                                        maximum_domain_gradient=max(float(np.linalg.norm(field[i]-field[j])) for i,j in edges),
                                        original_feature_pairing=True,zero_domain_control=zero,semantic_labels_propagated=False)


def d166(ep):return dh.calibrate_field(ep,'D166',_d166_field,(.25,1.))


def d166_control(ep):return dh.calibrate_field(ep,'D166_control_same_renderer_zero_domain',lambda ep,c:_d166_field(ep,c,True),(.25,1.))


def _d167_builder(ep,clip,control=False):
    xr,xq,p=dh.project(ep);f,b=dh.role(ep);valid=ep.q_valid>0;source=ep.wvalid>0
    if not f.any() or not b.any() or not valid.any():return None,{}
    x=np.vstack((xr[source],xq[valid]));labels=np.r_[np.zeros(source.sum()),np.ones(valid.sum())]
    weight=np.r_[np.full(source.sum(),.5/source.sum()),np.full(valid.sum(),.5/valid.sum())]
    domain=dh.logistic(x,labels,weight)
    ratio=np.clip(np.exp(np.clip(dh.predict(domain,xr),-20,20)),1./float(clip),float(clip))
    rr=dh.pair(ep);r_to_q=np.sqrt(np.maximum(0.,2.-2.*np.max(rr[valid],axis=0)))
    f_radius=dh.source_radius(ep,f);b_radius=dh.source_radius(ep,b)
    if not np.isfinite(f_radius) or not np.isfinite(b_radius):return None,{}
    radius=np.where(f,f_radius,b_radius)
    classified=np.zeros(len(ep.r),bool)
    for train,held in dh.folds(ep):
        model=dh.head(dh.restricted(ep,train),xr)
        hs=dh.predict(model,xr[held]);classes=dh.cov(ep)[held]>=.5
        classified[np.flatnonzero(held)]=(hs>0)==classes
    overlap=source&classified&(r_to_q<=radius)
    multiplier=np.ones(len(ep.r))
    if control:
        proximity=np.exp(-r_to_q/np.maximum(radius,1e-6))
        multiplier[overlap]=np.clip(proximity[overlap]/max(float(np.mean(proximity[overlap])),1e-6),1./float(clip),float(clip))
    else:multiplier[overlap]=ratio[overlap]
    fm=ep.wf*multiplier;bm=ep.wb*multiplier
    fw=.5*fm/max(float(fm.sum()),1e-6);bw=.5*bm/max(float(bm.sum()),1e-6);w=fw+bw;target=fw/np.maximum(w,1e-6)
    model=dh.logistic(xr,target,w)
    def support(x):
        sim=dh.mm(x,ep.r.T);ids=np.argmax(np.where(source[None],sim,-np.inf),axis=1)
        distance=np.sqrt(np.maximum(0.,2.-2.*sim[np.arange(len(x)),ids]))
        answer=overlap[ids]&(distance<=radius[ids])
        if x is ep.q:answer &=ep.q_valid>0
        return answer
    def scorer(x):return dh.predict(model,dh.mm(x,p))
    scorer.support=support
    scorer.source_ratio=lambda ids:np.clip(np.exp(np.clip(dh.predict(domain,xr[ids]),-20,20)),.25,4.)
    return scorer,dict(reference_overlap_samples=int(overlap.sum()),clipping=[1./float(clip),float(clip)],
                       query_class_labels_used=False,domain_classifier_input='feature_only',reference_mode_mass_floor='positive_original_mass_outside_support',
                       importance_normalized_per_role=True,domain_steps=100,supervised_head_steps=100)


def _locked_threshold(ep,score,allowed,selected,base,ratio):
    # Evaluated weights are predicted from an unlabeled domain classifier; they
    # do not supply query class truth or an estimate of target area.
    fw=ep.wf*ratio;bw=ep.wb*ratio;fm=fw[selected].sum();bm=bw[selected].sum()
    if min(fm,bm)<=1e-12:return 0.,np.inf,0
    inside=selected&allowed;outside=selected&~allowed
    constant=fw[outside&(base<=0)].sum()/fm+bw[outside&(base>0)].sum()/bm
    if not inside.any():return 0.,float(constant),0
    ids=np.flatnonzero(inside);order=ids[np.argsort(score[ids],kind='stable')];s=score[order]
    a=np.r_[0.,np.cumsum(fw[order])];b=np.r_[0.,np.cumsum(bw[order])]
    cuts=np.r_[0,np.flatnonzero(np.diff(s)>0)+1,len(s)]
    error=constant+a[cuts]/fm+(b[-1]-b[cuts])/bm
    t=np.empty(len(cuts));t[0]=np.nextafter(s[0],-np.inf);t[-1]=s[-1]
    if len(cuts)>2:
        c=cuts[1:-1];t[1:-1]=s[c-1]+(s[c]-s[c-1])/2
    best=np.flatnonzero(np.abs(error-error.min())<1e-12)
    edits=[int(np.count_nonzero((score[inside]>t[j])!=(base[inside]>0))) for j in best]
    j=int(best[np.argmin(edits)]);return float(t[j]),float(error[j]),min(edits)


def _d167_calibrate(ep,method,control=False):
    dh.validate(ep);spatial=dh.folds(ep);base=dh.b0_field(ep);base[ep.q_valid<=0]=-1.
    if not spatial:return dh.finish(ep,base,method,branch='missing_role_after_merge_B0')
    selected=np.zeros(len(ep.r),bool);base_oof=np.zeros(len(ep.r));episodes=[]
    for train,held in spatial:
        reduced=dh.restricted(ep,train);selected|=held;episodes.append((reduced,held));base_oof[held]=dh.b0_field(reduced,ep.r[held])
    chosen=None;summaries=[];best=(np.inf,10**20,np.inf)
    for clip in (1.5,2.,4.):
        score=np.zeros(len(ep.r));allowed=np.zeros(len(ep.r),bool);ratio=np.ones(len(ep.r));legal=True
        for reduced,held in episodes:
            scorer,info=_d167_builder(reduced,clip,control)
            if scorer is None:legal=False;break
            ids=np.flatnonzero(held);score[held]=scorer(ep.r[held]);allowed[held]=scorer.support(ep.r[held]);ratio[held]=scorer.source_ratio(ids)
        if not legal:summaries.append(dict(clip=clip,legal=False));continue
        threshold,error,edits=_locked_threshold(ep,score,allowed,selected,base_oof,ratio)
        # Recompute the zero adaptation with exactly these same weighted labels.
        fw=ep.wf*ratio;bw=ep.wb*ratio
        zero=float(fw[selected&(base_oof<=0)].sum()/fw[selected].sum()+bw[selected&(base_oof>0)].sum()/bw[selected].sum())
        summaries.append(dict(clip=clip,legal=True,source_error=error,source_zero_error=zero,edits=edits,threshold=threshold))
        if error<zero-1e-12 and (error,edits,clip)<best:best=(error,edits,clip);chosen=(clip,threshold)
    if chosen is None:return dh.finish(ep,base,method,branch='source_selected_zero_adaptation',source_candidates=summaries)
    scorer,info=_d167_builder(ep,chosen[0],control)
    if scorer is None:return dh.finish(ep,base,method,branch='full_fit_unavailable_B0')
    field=scorer(ep.q)-chosen[1];field[ep.q_valid<=0]=-1.;supported=scorer.support(ep.q).reshape(ep.q_hw)
    physical=continuous_original(ep,field.reshape(ep.q_hw))
    base_original=continuous_original(ep,base.reshape(ep.q_hw))>0
    domain_original=continuous_original(ep,supported.astype(float))>.5
    mask=np.where(domain_original,physical>0,base_original)
    return Result(field.reshape(ep.q_hw),0.,mask.astype(bool),dict(method=method,branch='source_selected_method',
                       source_candidates=summaries,source_threshold=chosen[1],configuration=chosen[0],
                       support_outside_preserves_original_M0=True,support_original_pixels=int(domain_original.sum()),**info))


def d167(ep):return _d167_calibrate(ep,'D167')
def d167_control(ep):return _d167_calibrate(ep,'D167_control_same_support_nearest_query_weights',True)

METHODS={'D166':d166,'D167':d167}
CONTROLS={'D166_control_same_renderer_zero_domain':d166_control,'D167_control_same_support_nearest_query_weights':d167_control}
