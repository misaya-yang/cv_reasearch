"""Fixed-bank audits, regional explanation, constrained cuts, DRO and mixtures."""
from __future__ import annotations
import itertools
import numpy as np
from scipy.ndimage import label,binary_dilation
from scipy.special import expit,logsumexp
from scipy.optimize import minimize,linprog
from . import d_helpers_151_200 as dh


def _negative_bank(ep,k=8):
    spatial=dh.folds(ep);background=dh.role(ep)[1];scores=np.full(len(ep.r),np.nan)
    for train,held in spatial:
        scores[held]=dh.b0_field(dh.restricted(ep,train),ep.r[held])
    usable=background&np.isfinite(scores)
    if not usable.any():return [],np.empty((0,ep.r.shape[1]))
    threshold=float(np.quantile(scores[usable],.25));s0=dh.b0_field(ep)
    eligible=np.flatnonzero((ep.q_valid>0)&(s0<threshold)&(s0<0))
    if len(eligible)<8:return [],np.empty((0,ep.r.shape[1]))
    centers,groups=dh.source_modes(ep.q[eligible],min(k,8))
    return [eligible[groups==j] for j in range(len(centers))],centers


def _extra_model(ep,xr,xq,groups,subset,strength=1.):
    ids=np.concatenate([groups[j] for j in subset]) if len(subset) else np.empty(0,int)
    extra=None if not len(ids) else (xq[ids],np.zeros(len(ids)),np.full(len(ids),float(strength)/len(ids)))
    return dh.head(ep,xr,extra=extra)


def _fg_error(ep,score,held):
    mass=ep.wf[held].sum()
    return float(ep.wf[held&(score<=0)].sum()/mass) if mass>0 else 0.


def _fold_banks(ep,k):
    """Each source fold rebuilds eligibility AND clusters before fitting.

    A shared unlabeled-Q codebook provides stable bucket identities only; no R
    label enters that correspondence. Actual bank atoms are fold-local negative
    clusters and remain fixed for L versus L-minus-S comparisons in that fold.
    """
    codebook,_=dh.source_modes(ep.q[ep.q_valid>0],min(k,8))
    if not len(codebook):return [],codebook
    xr,xq,p=dh.project(ep);data=[]
    for train,held in dh.folds(ep):
        reduced=dh.restricted(ep,train);groups,centers=_negative_bank(reduced,k)
        buckets=[[] for _ in range(len(codebook))]
        for points,center in zip(groups,centers):
            bucket=int(np.argmax(dh.mm(center,codebook.T)));buckets[bucket].extend(points.tolist())
        buckets=[np.unique(v).astype(int) for v in buckets]
        data.append((reduced,held,buckets))
    return data,codebook


def _d154_builder(ep,config,control=False):
    k,_,strength=config;xr,xq,p=dh.project(ep);fold_data,codebook=_fold_banks(ep,min(k,8))
    if not fold_data:return None,{}
    complete=tuple(range(len(codebook)));disputed=set();audit=[]
    for reduced,held,groups in fold_data:
        full=_extra_model(reduced,xr,xq,groups,complete,strength)
        full_error=_fg_error(ep,dh.predict(full,xr),held)
        differences=[]
        for j in complete:
            if not len(groups[j]):differences.append(None);continue
            without=_extra_model(reduced,xr,xq,groups,tuple(t for t in complete if t!=j),strength)
            without_error=_fg_error(ep,dh.predict(without,xr),held)
            # Positive means ADDING the group increased held FG errors.
            contamination=full_error-without_error;differences.append(float(contamination))
            if contamination>1e-12:disputed.add(j)
        audit.append(differences)
    retained=[j for j in complete if j not in disputed];tested=[];winner=();best=(np.inf,10**20,())
    choices=[complete] if control else [tuple(retained[j] for j in range(len(retained)) if bits&(1<<j))
                                      for bits in range(1<<len(retained))]
    for subset in choices:
        score=np.zeros(len(ep.r));selected=np.zeros(len(ep.r),bool);points=0
        for reduced,held,groups in fold_data:
            model=_extra_model(reduced,xr,xq,groups,subset,strength)
            score[held]=dh.predict(model,xr[held]);selected|=held
            points+=sum(len(groups[j]) for j in subset)
        error=dh.risk(ep,score,selected);key=(error,points,subset);tested.append(dict(subset=subset,source_error=error,source_bank_points=points))
        if key<best:best=key;winner=subset
    full_groups,full_centers=_negative_bank(ep,min(k,8));buckets=[[] for _ in complete]
    for points,center in zip(full_groups,full_centers):
        j=int(np.argmax(dh.mm(center,codebook.T)));buckets[j].extend(points.tolist())
    buckets=[np.unique(v).astype(int) for v in buckets]
    model=_extra_model(ep,xr,xq,buckets,winner,strength)
    return lambda x:dh.predict(model,dh.mm(x,p)),dict(disputed_buckets=sorted(disputed),retained_subset=winner,
               contamination_audit=audit,subsets_tested=len(tested),subset_source_results=tested,
               chosen_negative_points=sum(len(buckets[j]) for j in winner),fold_banks_rebuilt=True)


def d154(ep):return dh.calibrate(ep,'D154',_d154_builder)
def d154_control(ep):return dh.calibrate(ep,'D154_control_same_bank_without_group_audit',lambda ep,c:_d154_builder(ep,c,True))


def _subspace(rows,rank):
    mean=rows.mean(axis=0);_,s,v=np.linalg.svd(rows-mean,full_matrices=False)
    use=min(rank,int(np.sum(s>1e-8)))
    return mean,v[:use]


def _code(rows,model):
    mean,v=model;delta=rows-mean
    residual=np.sum(delta*delta,axis=1)-np.sum(dh.mm(delta,v.T)**2,axis=1)
    return np.maximum(residual,0.)


def _d157_field(ep,config,control=False):
    k,rank,strength=config;f,b,_=dh.pure(ep);base=dh.b0_field(ep);xr,xq,p=dh.project(ep)
    if not f.any() or not b.any():return None,{}
    rf=dh.source_radius(ep,f);rb=dh.source_radius(ep,b)
    if not np.isfinite(rf) or not np.isfinite(rb):return None,{}
    distf=dh.nearest_distance(dh.pair(ep),f);distb=dh.nearest_distance(dh.pair(ep),b)
    unknown=(distf>rf)&(distb>rb)&(ep.q_valid>0)
    regions,count=label(unknown.reshape(ep.q_hw));anchors=dh.qualified_anchors(ep,True)
    anchor_mask=np.zeros(len(ep.q),bool);anchor_mask[anchors]=True
    if not len(anchors):return None,dict(reason='no_qualified_F_anchors')
    source_values=[]
    for train,held in dh.folds(ep):
        use=f&held
        if use.any():source_values.extend(dh.b0_field(dh.restricted(ep,train),ep.r[use]).tolist())
    if not source_values:return None,{}
    minimum=float(np.min(source_values));field=base.copy();audits=[]
    yy,xx=np.indices(ep.q_hw);checker=((yy+xx)%2).ravel()
    for region in range(1,count+1):
        ids=np.flatnonzero(regions.ravel()==region)
        if len(ids)<8:continue
        # Unknown points cannot themselves satisfy the original radius anchor
        # test. The attached one-patch ring defines this region's anchor support.
        ring=binary_dilation((regions==region),structure=np.ones((3,3))).ravel()
        local=np.flatnonzero(ring&anchor_mask)
        if not len(local):continue
        halves=[ids[checker[ids]==j] for j in (0,1)]
        if min(len(v) for v in halves)<2:continue
        scores=np.zeros(len(ids));gaps=[]
        for fitting,testing in ((halves[0],halves[1]),(halves[1],halves[0])):
            # All query anchors used here are outside the scored half.
            fg_rows=np.vstack((xr[f],xq[local[~np.isin(local,testing)]]))
            foreground=_subspace(fg_rows,rank);background=_subspace(xq[fitting],rank)
            if control:
                positive_distance=np.sum((xq[testing]-fg_rows.mean(axis=0))**2,axis=1)
                negative_distance=np.sum((xq[testing]-xq[fitting].mean(axis=0))**2,axis=1)
                gap=negative_distance-positive_distance
            else:
                fcode=_code(xq[testing],foreground);bcode=_code(xq[testing],background)
                ffee=.5*(foreground[1].shape[0]+1)*xr.shape[1]*np.log1p(len(fg_rows))/max(len(fg_rows),1)
                bfee=.5*(background[1].shape[0]+1)*xr.shape[1]*np.log1p(len(fitting))/len(fitting)
                gap=bcode+bfee-fcode-ffee
            # Geometry / code gap alone cannot assert target identity.
            gate=base[testing]-minimum
            output=np.minimum(gap,gate)
            for qid,value in zip(testing,output):scores[np.flatnonzero(ids==qid)[0]]=value
            gaps.append(float(np.mean(gap)))
        field[ids]=scores
        audits.append(dict(region=region,points=len(ids),ring_F_anchors=len(local),half_code_gaps=gaps))
    if not audits:return None,dict(reason='no_anchored_eight_point_unknown_regions')
    return field,dict(unknown_regions=count,region_audits=audits,source_minimum_F_margin=minimum,
                       held_half_never_used_to_fit_its_BG_or_F_space=True)


def d157(ep):return dh.calibrate_field(ep,'D157',_d157_field)
def d157_control(ep):return dh.calibrate_field(ep,'D157_control_same_regions_anchor_centroid',lambda ep,c:_d157_field(ep,c,True))


def _audit_mode_recall(ep,xr,xq,ids,strength):
    labels=np.zeros(len(ids));audits=[];scores=np.zeros(len(ep.r));selected=np.zeros(len(ep.r),bool)
    for train,held in dh.folds(ep):
        reduced=dh.restricted(ep,train);original=dh.head(reduced,xr)
        extra=None if not len(ids) else (xq[ids],labels,np.full(len(ids),strength/len(ids)))
        changed=dh.head(reduced,xr,extra=extra);centers,_=dh.foreground_modes(reduced,8)
        modes=np.argmax(dh.mm(ep.r,centers.T),axis=1) if len(centers) else np.zeros(len(ep.r),int)
        good=dh.fg_recall_non_decrease(ep,dh.predict(original,xr),dh.predict(changed,xr),modes,held)
        audits.append(bool(good));scores[held]=dh.predict(changed,xr[held]);selected|=held
    return bool(audits) and all(audits),scores,selected,audits


def _d158_builder(ep,config,control=False):
    k,_,strength=config;xr,xq,p=dh.project(ep);f,b,_=dh.pure(ep)
    centers,_=dh.foreground_modes(ep,k);base=dh.b0_field(ep)
    if not len(centers):return None,{}
    radius=dh.source_radius(ep,f)
    if not np.isfinite(radius):return None,{}
    distance=np.sqrt(np.maximum(0.,2.-2.*np.max(dh.mm(ep.q,centers.T),axis=1)))
    positive=base>max(0.,float(np.median(base[base>0])) if np.any(base>0) else 0.)
    candidates=np.flatnonzero(positive&(distance>radius)&(ep.q_valid>0))
    if len(candidates)<8:return None,dict(reason='no_unsupported_positive_cluster')
    _,assignment=dh.source_modes(ep.q[candidates],min(k,8));groups=[candidates[assignment==j] for j in np.unique(assignment)]
    groups.sort(key=lambda ids:(-float(base[ids].mean()),int(ids.min())))
    accepted=[];iterations=[];planes=[];visited=[]
    for j,group in enumerate(groups[:5]):
        proposed=np.r_[np.asarray(accepted,int),group]
        legal,score,selected,audit=_audit_mode_recall(ep,xr,xq,proposed,strength)
        if control:legal=True
        iterations.append(dict(candidate=j,points=len(group),accepted=bool(legal),held_mode_audit=audit))
        if not legal:continue
        accepted=proposed.tolist();model=_extra_model(ep,xr,xq,[np.asarray(accepted,int)],(0,),strength)
        planes.append(dict(weights=model[0].tolist(),bias=model[1],accepted_negative_points=len(accepted)))
        error=dh.risk(ep,score,selected);visited.append((error,len(visited),np.asarray(accepted,int).copy()))
    if not visited:return None,dict(iterations=iterations,reason='no_accepted_negative_constraint')
    minimum=min(v[0] for v in visited);chosen=next(v for v in visited if v[0]<=minimum+1e-12)
    ids=chosen[2];model=_extra_model(ep,xr,xq,[ids],(0,),strength)
    return lambda x:dh.predict(model,dh.mm(x,p)),dict(cut_iterations=iterations,local_planes=planes,
                final_weights=model[0].tolist(),final_bias=model[1],chosen_cut_stage=chosen[1],
                accepted_negative_points=len(ids),max_candidate_rounds=5,held_mode_safety_not_query_guarantee=True)


def d158(ep):return dh.calibrate(ep,'D158',_d158_builder)
def d158_control(ep):return dh.calibrate(ep,'D158_control_same_candidates_static_negative_constraints',lambda ep,c:_d158_builder(ep,c,True))


def worst_weights(loss,rho,removed=()):
    """Exact finite LP over the literal ORIGINAL-K polytope, including removals."""
    loss=np.asarray(loss,float);k=len(loss);u=np.full(k,1./k)
    # Absolute-deviation auxiliaries: w-u<=t, u-w<=t, sumt<=rho.
    identity=np.eye(k);A=np.vstack((np.c_[identity,-identity],np.c_[-identity,-identity],np.c_[np.zeros((1,k)),np.ones((1,k))]))
    upper=np.r_[u,-u,float(rho)];bounds=[(0.,0.) if j in removed else (0.,2./k) for j in range(k)]+[(0.,None)]*k
    solution=linprog(np.r_[-loss,np.zeros(k)],A_ub=A,b_ub=upper,A_eq=np.c_[np.ones((1,k)),np.zeros((1,k))],b_eq=[1.],bounds=bounds,method='highs')
    return None if not solution.success else solution.x[:k]


def _dro_fit(ep,xr,xq,groups,rho,removed,strength,trim=False):
    y,weights=dh.sample_weights(ep);d=xr.shape[1];w=np.zeros(d);bias=0.;penalty=.01*max(float(np.sum(weights*np.sum(xr*xr,axis=1))/d),1e-6)
    gram=dh.mm(xr.T,weights[:,None]*xr);bound=.25*(float(np.linalg.eigvalsh(gram)[-1])+weights.sum())+penalty
    group_moments=[dh.mm(xq[ids].T,xq[ids])/len(ids) for ids in groups]
    bound+=strength*.25*(max(float(np.linalg.eigvalsh(g)[-1]) for g in group_moments)+1.)
    rate=1./max(bound,1e-6);worst=None;losses=[]
    for _ in range(100):
        current=[dh.mm(xq[ids],w)+bias for ids in groups]
        loss=np.asarray([np.logaddexp(0,z).mean() for z in current])
        if trim:
            worst=np.ones(len(groups));worst[list(removed)]=0;worst/=worst.sum()
        else:worst=worst_weights(loss,rho,removed)
        if worst is None:return None,dict(infeasible=True)
        residual=weights*(expit(dh.mm(xr,w)+bias)-y);gradient=dh.mm(xr.T,residual)+penalty*w;gb=float(residual.sum())
        for mass,ids,z in zip(worst,groups,current):
            if mass<=0:continue
            rr=expit(z);gradient+=strength*mass*dh.mm(xq[ids].T,rr)/len(ids);gb+=strength*mass*float(rr.mean())
        w-=rate*gradient;bias-=rate*gb
        losses.append(float(np.sum(weights*(np.logaddexp(0,dh.mm(xr,w)+bias)-y*(dh.mm(xr,w)+bias)))+strength*float(worst@loss)+.5*penalty*float(w@w)))
    return (w,float(bias)),dict(worst_weights=worst.tolist(),source_objective=losses[-1],steps=100)


def _d159_builder(ep,config,trim=False):
    h,rho_index,strength=config;xr,xq,p=dh.project(ep);anchors=dh.qualified_anchors(ep,False)
    if len(anchors)<8:return None,{}
    centers,which=dh.source_modes(ep.q[anchors],8);groups=[anchors[which==j] for j in range(len(centers))]
    if not len(groups):return None,{}
    # Same codebook RBG block frequencies give observed, source-only L1 radii.
    b=dh.role(ep)[1];source_groups=dh.blocks(ep.r_hw);vectors=[]
    for block in range(4):
        ids=np.flatnonzero(b&(source_groups==block))
        if len(ids)<8:continue
        assigned=np.argmax(dh.mm(ep.r[ids],centers.T),axis=1)
        freq=np.bincount(assigned,minlength=len(centers)).astype(float);freq/=freq.sum();vectors.append(freq)
    if len(vectors)<2:return None,dict(reason='insufficient_source_BG_frequency_blocks')
    radii=sorted({float(np.abs(a-b).sum()) for a in vectors for b in vectors})
    rho=radii[min(int(rho_index),len(radii)-1)];candidates=[]
    for count in range(min(int(h),len(groups)-1)+1):
        for removed in itertools.combinations(range(len(groups)),count):
            if worst_weights(np.zeros(len(groups)),rho,removed) is None:continue
            safe=True;score=np.zeros(len(ep.r));selected=np.zeros(len(ep.r),bool);audit=[]
            for train,held in dh.folds(ep):
                reduced=dh.restricted(ep,train);model,detail=_dro_fit(reduced,xr,xq,groups,rho,removed,strength,trim)
                if model is None:safe=False;break
                original=dh.head(reduced,xr);center,_=dh.foreground_modes(reduced,8)
                mode=np.argmax(dh.mm(ep.r,center.T),axis=1) if len(center) else np.zeros(len(ep.r),int)
                prediction=dh.predict(model,xr);good=dh.fg_recall_non_decrease(ep,dh.predict(original,xr),prediction,mode,held)
                safe &=good;audit.append(bool(good));score[held]=prediction[held];selected|=held
            if safe:candidates.append((dh.risk(ep,score,selected),len(removed),removed,audit))
    if not candidates:return None,dict(reason='no_source_F_recall_feasible_DRO_head',rho=rho)
    selected=min(candidates,key=lambda a:(a[0],a[1],a[2]));model,detail=_dro_fit(ep,xr,xq,groups,rho,selected[2],strength,trim)
    return lambda x:dh.predict(model,dh.mm(x,p)),dict(rho=rho,removed_clusters=selected[2],original_K=len(groups),
              source_frequency_radius_candidates=radii,held_mode_recall_audit=selected[3],feasible_retractions=len(candidates),
              polytope='originalK: sumw1,L1to_original_uniform<=rho,cap2/K,removed_weight0',**detail)


def d159(ep):return dh.calibrate(ep,'D159',_d159_builder,((0,0,.25),(1,1,.25),(2,1,.25),(1,2,1.),(2,2,1.)))
def d159_control(ep):return dh.calibrate(ep,'D159_control_same_retraction_uniform_trim',lambda ep,c:_d159_builder(ep,c,True),((0,0,.25),(1,1,.25),(2,1,.25),(1,2,1.),(2,2,1.)))


def _simplex_nnls(matrix,target):
    k=matrix.shape[1]
    result=minimize(lambda w:.5*np.sum((matrix@w-target)**2),np.full(k,1./k),
                    jac=lambda w:matrix.T@(matrix@w-target),bounds=[(0.,1.)]*k,
                    constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(k)},
                    method='SLSQP',options={'maxiter':20,'ftol':1e-10})
    w=np.maximum(result.x,0.);w/=max(float(w.sum()),1e-12)
    return w,float(np.linalg.norm(matrix@w-target)),bool(result.success)


def _d168_builder(ep,config,unchecked=False):
    k,rank,tolerance=config;f,b,_=dh.pure(ep);negative=dh.qualified_anchors(ep,False)
    if not f.any() or len(negative)<8:return None,{}
    fg,_=dh.source_modes(ep.r[f],k,weights=ep.wf[f]);bg,_=dh.source_modes(ep.q[negative],min(k,8))
    if not len(fg) or not len(bg):return None,{}
    dictionary=np.vstack((fg,bg));nf=len(fg);d=min(32,ep.r.shape[1])
    # Three DISJOINT original coordinates, selected by source-only variance.
    variance=np.var(ep.r[ep.wvalid>0],axis=0);selected=np.argsort(-variance,kind='stable')[:d]
    groups=np.array_split(selected,3)
    if min(len(g) for g in groups)<1:return None,{}
    target=ep.q[ep.q_valid>0].mean(axis=0);weights=[];checks=[]
    for omitted in range(3):
        fit_ids=np.concatenate([g for j,g in enumerate(groups) if j!=omitted]);matrix=dictionary[:,fit_ids].T
        singular=np.linalg.svd(matrix,compute_uv=False);fullrank=int(np.sum(singular>max(float(singular[0]) if len(singular) else 0.,1.)*1e-8))
        if fullrank<len(dictionary) and not unchecked:return None,dict(reason='mixture_dictionary_not_identifiable',rank=fullrank,components=len(dictionary))
        w,residual,converged=_simplex_nnls(matrix,target[fit_ids])
        held=groups[omitted];error=float(np.linalg.norm(dictionary[:,held].T@w-target[held]))
        # Same fixed mixture predicts genuine held-source block moments. The
        # reference pseudo-query limit uses train-source class dictionaries.
        pseudo=[]
        for train,test in dh.folds(ep):
            if not test.any():continue
            point=ep.r[test].mean(axis=0);ww,rr,ok=_simplex_nnls(matrix,point[fit_ids])
            pseudo.append(float(np.linalg.norm(dictionary[:,held].T@ww-point[held])))
        bound=max(float(np.quantile(pseudo,.95)) if pseudo else 0.,1e-5)*float(tolerance)
        if (error>bound or not converged) and not unchecked:return None,dict(reason='held_attribute_prediction_failed',held_attribute_group=omitted,error=error,bound=bound)
        weights.append(w);checks.append(dict(group=omitted,numerical_rank=fullrank,held_attribute_error=error,bound=bound))
    if max(float(np.linalg.norm(a-b)) for a in weights for b in weights)>.25 and not unchecked:
        return None,dict(reason='attribute_rotations_have_inconsistent_responsibilities')
    weight=np.mean(weights,axis=0);centers=dictionary[:,selected];source=np.asarray(ep.r)[:,selected]
    distances=np.sum((source[:,None]-centers[None])**2,axis=2)
    sigma2=max(float(np.median(np.min(distances,axis=1))),1e-5)
    def score(x):
        z=x[:,selected];distance=np.maximum(0.,np.sum(z*z,axis=1)[:,None]+np.sum(centers*centers,axis=1)-2*dh.mm(z,centers.T))
        logs=np.log(np.maximum(weight,1e-12))-distance/(2*sigma2)
        return logsumexp(logs[:,:nf],axis=1)-logsumexp(logs[:,nf:],axis=1)
    return score,dict(attribute_groups=[g.tolist() for g in groups],mixture_weights=weight.tolist(),attribute_checks=checks,
                      free_BG_residual_allowed=False,unchecked_control=unchecked)


def d168(ep):return dh.calibrate(ep,'D168',_d168_builder)
def d168_control(ep):return dh.calibrate(ep,'D168_control_same_mixture_without_attribute_test',lambda ep,c:_d168_builder(ep,c,True))

METHODS={'D154':d154,'D157':d157,'D158':d158,'D159':d159,'D168':d168}
CONTROLS={'D154_control_same_bank_without_group_audit':d154_control,'D157_control_same_regions_anchor_centroid':d157_control,
          'D158_control_same_candidates_static_negative_constraints':d158_control,'D159_control_same_retraction_uniform_trim':d159_control,
          'D168_control_same_mixture_without_attribute_test':d168_control}
