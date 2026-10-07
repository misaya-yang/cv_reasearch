"""Pro30 M02 low-rank orthogonal anchors and M07 anchored simplex semi-NMF."""
from __future__ import annotations

from functools import partial
import time
import numpy as np
from scipy.linalg import eigh

from .common import (EPS,validate,unit,blocks,fit_br,br_margin,finish,degenerate_margin,joint_basis,source_contract)


def squared(a,b):return np.maximum(np.sum(a*a,1)[:,None]+np.sum(b*b,1)[None]-2*a@b.T,0)


def weighted_modes(x,weight,k,*,spherical=False,query_init=False):
    ids=np.flatnonzero(weight>0);points=x[ids];weights=weight[ids]
    if not len(ids):return np.empty((0,x.shape[1])),[]
    k=min(k,len(ids));rng=np.random.RandomState(0)
    if query_init:
        selected=[int(rng.choice(len(ids),p=weights/weights.sum()))]
        distance=squared(points,points[selected])[:,0]
        while len(selected)<k:
            probability=weights*distance;probability[selected]=0
            if probability.sum()<=EPS:
                index=next(i for i in range(len(ids)) if i not in selected)
            else:index=int(rng.choice(len(ids),p=probability/probability.sum()))
            selected.append(index);distance=np.minimum(distance,squared(points,points[index:index+1])[:,0])
    else:
        center=np.sum(points*weights[:,None],0)/weights.sum();selected=[int(np.argmin(squared(points,center[None])[:,0]))]
        distance=squared(points,points[selected])[:,0]
        while len(selected)<k:
            distance[selected]=-1;index=int(np.argmax(distance));selected.append(index)
            distance=np.minimum(distance,squared(points,points[index:index+1])[:,0])
    centers=points[selected].copy()
    for _ in range(10):
        label=np.argmin(squared(points,centers),axis=1)
        centers=np.array([np.sum(points[label==j]*weights[label==j,None],0)/weights[label==j].sum()
                          for j in range(len(centers)) if np.any(label==j)])
        if spherical:centers=unit(centers)
    label=np.argmin(squared(points,centers),axis=1)
    return centers,[ids[label==j] for j in range(len(centers))]


def orthogonal_operator(reference,query):
    stacked=np.r_[reference,query];_,s,vt=np.linalg.svd(stacked,full_matrices=False)
    basis=vt[s>max(float(s[0]),EPS)*1e-10][:32].T
    if not basis.shape[1]:return basis,np.empty((0,0))
    r=reference@basis;q=query@basis
    # Equal per-anchor omega=1; no hidden normalization changes ridge-to-data ratio.
    a,_,bt=np.linalg.svd(q.T@r+np.eye(basis.shape[1]),full_matrices=False)
    return basis,a@bt


def transform(vector,basis,rotation):return vector+basis@(rotation-np.eye(len(rotation)))@(basis.T@vector)


def _pair_anchors(ep):
    reference,groups=weighted_modes(ep.r,ep.wb,32,spherical=True)
    query,qgroups=weighted_modes(ep.q,ep.q_valid,64,query_init=True);query=unit(query)
    similarity=reference@query.T
    if not len(reference) or not len(query):return None
    left=np.argsort(-similarity,axis=1,kind='stable')[:,:min(3,len(query))]
    right=np.argsort(-similarity,axis=0,kind='stable')[:min(3,len(reference))]
    possible=[(float(similarity[i,j]),i,int(j)) for i in range(len(reference)) for j in left[i] if i in right[:,j]]
    possible.sort(key=lambda item:(-item[0],item[1],item[2]));usedr=set();usedq=set();pairs=[]
    for value,i,j in possible:
        if i in usedr or j in usedq:continue
        usedr.add(i);usedq.add(j);pairs.append((i,j,value))
        if len(pairs)>=16:break
    representatives=[]
    for i,_,_ in pairs:
        group=groups[i];order=np.lexsort((group,-(ep.r[group]@reference[i])))
        representatives.append(int(group[order[0]]))
    regions=np.unique(blocks(ep.r_hw)[representatives]) if representatives else np.empty(0,int)
    return reference,query,pairs,representatives,regions


def background_orthogonal(ep,mode='orthogonal'):
    validate(ep);start=time.perf_counter();mid='PRO30_M02' if mode=='orthogonal' else 'PRO30_M02__'+mode
    degeneration=degenerate_margin(ep)
    if degeneration is not None:return finish(ep,degeneration[0],mid,degeneration[1])
    fitted=fit_br(ep);original=fitted.predict(ep.q);bank=_pair_anchors(ep)
    info=dict(source_contract(2),BR=fitted.info,mode=mode,new_encoder_forwards=0)
    if bank is None:return finish(ep,original,mid,dict(info,inactive=True,inactive_reason='no source/query modes'))
    r,q,pairs,representatives,regions=bank
    info.update(reference_modes=len(r),query_modes=len(q),mutual_pairs=len(pairs),reference_anchor_quadrants=regions.tolist(),
        reference_anchor_representatives=representatives,omega='one per matched anchor')
    if len(pairs)<8 or len(regions)<3:
        return finish(ep,original,mid,dict(info,inactive=True,inactive_reason='fewer8pairs_or3sourcequadrants'))
    ri=np.array([p[0] for p in pairs]);qi=np.array([p[1] for p in pairs]);ra,qa=r[ri],q[qi]
    fullbasis,fullrotation=orthogonal_operator(ra,qa);halves=[]
    foreground=unit(np.sum(ep.r*ep.wf[:,None],0))
    for parity in (0,1):
        train=np.arange(len(pairs))%2==parity;held=~train
        basis,rotation=orthogonal_operator(ra[train],qa[train])
        predicted=np.array([transform(value,basis,rotation) for value in ra[held]])
        before=float(np.median(np.sum((ra[held]-qa[held])**2,axis=1)))
        after=float(np.median(np.sum((predicted-qa[held])**2,axis=1)))
        halves.append(dict(before=before,after=after,passed=before>EPS and after<=.9*before,
            foreground_direction=transform(foreground,basis,rotation)))
    foreground_cos=float(unit(halves[0]['foreground_direction'])@unit(halves[1]['foreground_direction']))
    info.update(rank=fullbasis.shape[1],odd_even_foreground_cosine=foreground_cos,
        odd_even_residuals=[{k:v for k,v in h.items() if k!='foreground_direction'} for h in halves],
        nearest_vs_hungarian_unused='anchors greedy one-to-one as source specifies')
    active=all(h['passed'] for h in halves) and foreground_cos>=.95
    if not active:return finish(ep,original,mid,dict(info,inactive=True,inactive_reason='odd_even_stability_gate_failed'))
    if mode=='same_anchors_shift':
        shift=np.mean(qa-ra,axis=0);coefficient=fitted.coefficient;bias=fitted.bias-shift@coefficient
    elif mode=='permuted_orthogonal':
        perm=np.random.default_rng(0).permutation(len(qa));basis,rotation=orthogonal_operator(ra,qa[perm])
        coefficient=transform(fitted.coefficient,basis,rotation);bias=fitted.bias
    elif mode=='same_rank_global_CORAL':
        basis=fullbasis;rr=ep.r[ep.wvalid>0]@basis;qq=ep.q[ep.q_valid>0]@basis
        mr,mq=rr.mean(0),qq.mean(0);cr=(rr-mr).T@(rr-mr)/len(rr);cq=(qq-mq).T@(qq-mq)/len(qq)
        def power(c,p):
            value,vector=eigh(c);floor=max(float(np.trace(c)/len(c))*.01,1e-6)
            return (vector*np.maximum(value,floor)**p)@vector.T
        t=power(cq,.5)@power(cr,-.5);low=basis.T@fitted.coefficient
        newlow=np.linalg.solve(t.T,low);coefficient=fitted.coefficient+basis@(newlow-low)
        bias=fitted.bias+mr@low-mq@newlow
        info['CORAL_scope']='all valid R/Q covariance inside same anchored-span rank, with global means'
    elif mode=='orthogonal':coefficient=transform(fitted.coefficient,fullbasis,fullrotation);bias=fitted.bias
    else:raise ValueError(mode)
    score=ep.q@coefficient+bias;score[ep.q_valid<=0]=-1.
    return finish(ep,score,mid,dict(info,inactive=False,postprocess_seconds=time.perf_counter()-start))


def project_simplex(x,mass):
    x=np.asarray(x,float);mass=np.broadcast_to(np.asarray(mass,float),(len(x),))
    if x.shape[1]==0:return x.copy()
    ordered=np.sort(x,axis=1)[:,::-1];cumulative=np.cumsum(ordered,axis=1)-mass[:,None]
    valid=ordered-cumulative/(np.arange(x.shape[1])+1)>0;rho=np.maximum(valid.sum(1)-1,0)
    theta=cumulative[np.arange(len(x)),rho]/(rho+1)
    out=np.maximum(x-theta[:,None],0);out[mass<=0]=0
    return out


def _factor_initial(ep):
    basis,pca=joint_basis(ep,32);ri=np.flatnonzero(ep.wvalid>0);qi=np.flatnonzero(ep.q_valid>0)
    rx=ep.r@basis;qx=ep.q@basis
    f,_=weighted_modes(rx,ep.wf,4);b,_=weighted_modes(rx,ep.wb,4);dictionary=np.r_[f,b];nf=len(f)
    x=np.r_[rx[ri],qx[qi]];coverage=ep.wf[ri]/ep.wvalid[ri];h=np.zeros((len(x),len(dictionary)))
    fidx=np.argmin(squared(rx[ri],f),1);bidx=np.argmin(squared(rx[ri],b),1)
    h[np.arange(len(ri)),fidx]=coverage;h[np.arange(len(ri)),nf+bidx]=1-coverage
    h[len(ri)+np.arange(len(qi)),np.argmin(squared(qx[qi],dictionary),1)]=1.
    return basis,pca,ri,qi,x,dictionary,nf,h,coverage


def anchored_seminmf(ep,mode='joint'):
    validate(ep);start=time.perf_counter();mid='PRO30_M07' if mode=='joint' else 'PRO30_M07__'+mode
    degeneration=degenerate_margin(ep)
    if degeneration is not None:return finish(ep,degeneration[0],mid,degeneration[1])
    basis,pca,ri,qi,x,d0,nf,h,c=_factor_initial(ep);dictionary=d0.copy()
    omega=np.r_[.5*ep.wvalid[ri]/ep.wvalid[ri].sum(),(.0 if mode=='query_weight_zero' else .5)*ep.q_valid[qi]/max(ep.q_valid[qi].sum(),EPS)]
    def objective():return float(np.sum(omega[:,None]*(x-h@dictionary)**2)+.01*np.sum((dictionary-d0)**2))
    trace=[objective()]
    for _ in range(15):
        if mode!='fixed_D0':
            dictionary=np.linalg.solve(h.T@(omega[:,None]*h)+.01*np.eye(len(d0)),h.T@(omega[:,None]*x)+.01*d0)
        trace.append(objective())
        if mode=='constrained_hard_kmeans':
            options=np.array([c[:,None]*dictionary[f]+(1-c[:,None])*dictionary[b]
                for f in range(nf) for b in range(nf,len(dictionary))])
            costs=np.sum((options-x[:len(ri)][None])**2,axis=2);best=np.argmin(costs,axis=0)
            h[:len(ri)]=0;fc=best//(len(dictionary)-nf);bc=best%(len(dictionary)-nf)+nf
            h[np.arange(len(ri)),fc]=c;h[np.arange(len(ri)),bc]=1-c
            h[len(ri):]=0;idx=np.argmin(squared(x[len(ri):],dictionary),axis=1)
            h[len(ri)+np.arange(len(qi)),idx]=1.
        else:
            step=1/(2*float(np.linalg.eigvalsh(dictionary@dictionary.T).max())+1e-8)
            for _ in range(10):
                proposal=h-step*(2*(h@dictionary-x)@dictionary.T)
                h[:len(ri),:nf]=project_simplex(proposal[:len(ri),:nf],c)
                h[:len(ri),nf:]=project_simplex(proposal[:len(ri),nf:],1-c)
                h[len(ri):]=project_simplex(proposal[len(ri):],1.)
        trace.append(objective())
    margin=np.full(len(ep.q),-1.);margin[qi]=2*np.sum(h[len(ri):,:nf],axis=1)-1
    info=dict(source_contract(7),mode=mode,PCA=pca,factor_role_counts=[nf,len(d0)-nf],rounds=15,H_PG_steps_per_round=0 if mode=='constrained_hard_kmeans' else 10,
        dictionary_ridge=.01,reference_weight_mass=float(omega[:len(ri)].sum()),query_weight_mass=float(omega[len(ri):].sum()),
        objective_history=trace,maximum_recorded_objective_increase=float(max(np.max(np.diff(trace)),0)),
        reference_role_constraint_error=float(np.max(np.abs(h[:len(ri),:nf].sum(1)-c))),
        all_simplex_mass_error=float(np.max(np.abs(h.sum(1)-1))),minimum_coefficient=float(h.min()),
        factor_displacement_norm=float(np.linalg.norm(dictionary-d0)),empty_components=int(np.sum(h.sum(0)<=1e-10)),
        query_FG_coefficient_mean=float(np.average(h[len(ri):,:nf].sum(1),weights=ep.q_valid[qi])) if len(qi) else None,
        query_weight_zero_keeps_reference_mass_half=True,PG_row_gradient_not_multiplied_by_omega=True,
        postprocess_seconds=time.perf_counter()-start,new_encoder_forwards=0,quality='unmeasured candidate; coefficients are not calibrated occupancy')
    return finish(ep,margin,mid,info)


def install(methods,controls,requirements,contracts):
    methods['PRO30_M02']=background_orthogonal
    for mode in ('same_anchors_shift','same_rank_global_CORAL','permuted_orthogonal'):
        controls['PRO30_M02__'+mode]=partial(background_orthogonal,mode=mode)
    methods['PRO30_M07']=anchored_seminmf
    for mode in ('fixed_D0','constrained_hard_kmeans','query_weight_zero'):
        controls['PRO30_M07__'+mode]=partial(anchored_seminmf,mode=mode)
    for number in (2,7):
        mid='PRO30_M'+str(number).zfill(2);requirements[mid]=['native unit R/Q','complete MR area weights','physical valid grids']
        contracts[mid]=dict(source_contract(number),input_contract='N',host='B_R' if number==2 else None,
            renderer='CPU100 two-threshold',controls=[c for c in controls if c.startswith(mid+'__')]+['PRO30_B_R'],
            constants={'R_BG_modes':32,'Q_modes':64,'Lloyd_rounds':10,'mutual_top':3,'pair_max':16,'pair_min':8,'R_quadrants_min':3,'half_residual_ratio_max':.9,'FG_transformed_cos_min':.95,'Procrustes_identity_weight':1.,'anchor_weights':'equal1'} if number==2 else {'dimension_max':32,'role_factors_max':4,'ridge_D0':.01,'ALS_rounds':15,'H_PG_steps':10,'reference_query_weight':[.5,.5]},
            implementation_assumption='query modes weighted single-trial kmeans++seed0,10Euclidean Lloyd; source modes deterministicFPS10Lloyd; mode representatives nearest member, equal Procrustes anchors' if number==2 else 'weighted Euclidean reference kmeans uses deterministicFPS10Lloyd; hard reference assignment exact over FG×BGpairs; queryOmega0 leaves referenceOmega.5')
