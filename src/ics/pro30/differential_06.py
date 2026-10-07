"""M06 source-supervised skew-bilinear edge divergence and fixed integration."""
from __future__ import annotations

from functools import partial
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from ics.cpu100.common import neighbors
from .common import (EPS,validate,unit,br_margin,fit_br,degenerate_margin,finish,joint_basis,source_contract)

_CACHE=None


def physical_edges(hw,valid):
    i,j=neighbors(hw,diagonal=True);keep=(valid[i]>0)&(valid[j]>0)
    return i[keep],j[keep]


def _weighted_head(ep,x,strength=.01,target_override=None):
    ids=np.flatnonzero(ep.wvalid>0);f,b=ep.wf[ids],ep.wb[ids]
    weight=.5*f/max(f.sum(),EPS)+.5*b/max(b.sum(),EPS)
    y=2*f/ep.wvalid[ids]-1 if target_override is None else target_override[ids];z=x[ids]
    coefficient=np.linalg.solve(z.T@(weight[:,None]*z)+strength*np.eye(z.shape[1]),z.T@(weight*y))
    return coefficient,0.


def _prepare(ep):
    global _CACHE
    if _CACHE is not None and _CACHE['ep'] is ep:return _CACHE,True
    validate(ep);start=time.perf_counter();u,pca_info=joint_basis(ep,16)
    r=ep.r@u;q=ep.q@u;dimension=u.shape[1];aa,bb=np.triu_indices(dimension,1)
    ri,rj=physical_edges(ep.r_hw,ep.wvalid);qi,qj=physical_edges(ep.q_hw,ep.q_valid)
    coverage=np.divide(ep.wf,ep.wvalid,out=np.zeros(len(ep.r)),where=ep.wvalid>0);target=2*coverage-1
    difference=target[rj]-target[ri];distinct=np.abs(difference)>1e-8
    design=r[ri][:,aa]*r[rj][:,bb]-r[ri][:,bb]*r[rj][:,aa]
    base=np.minimum(ep.wvalid[ri],ep.wvalid[rj]);weights=np.zeros(len(ri))
    # The source leaves the denominator for its "multiplier<=10" unspecified.
    # The declared recipe caps *within-role edge-quality* multipliers before
    # the separate exact half-mass role normalization. No global inverse-class
    # cap is silently claimed (it is incompatible with halfmass on rare edges).
    for group in (distinct,~distinct):
        if not group.any():continue
        multiplier=np.minimum(base[group]/max(float(base[group].mean()),EPS),10.)
        weights[group]=multiplier/multiplier.sum()
    groups=int(distinct.any())+int((~distinct).any())
    if groups:weights/=groups
    coefficient=np.linalg.solve(design.T@(weights[:,None]*design)+.01*np.eye(len(aa)),design.T@(weights*difference)) if len(aa) else np.empty(0)
    k=np.zeros((dimension,dimension));k[aa,bb]=coefficient;k[bb,aa]=-coefficient
    queryweights=np.exp(-(1-np.sum(ep.q[qi]*ep.q[qj],axis=1))/.1)
    c=sparse.csr_matrix((np.r_[-np.ones(len(qi)),np.ones(len(qj))],
        (np.r_[np.arange(len(qi)),np.arange(len(qj))],np.r_[qi,qj])),shape=(len(qi),len(ep.q)))
    laplacian=c.T@sparse.diags(queryweights)@c;degree=np.asarray(laplacian.diagonal())
    gamma=max(float(degree[ep.q_valid>0].mean()/4) if np.any(ep.q_valid>0) else 0.,.1)
    unary,br_info=br_margin(ep)
    _CACHE=dict(ep=ep,u=u,r=r,q=q,aa=aa,bb=bb,ri=ri,rj=rj,qi=qi,qj=qj,
        C=c,W=queryweights,L=laplacian,gamma=gamma,K=k,s_B=unary,BR_info=br_info,
        fit_info=dict(PCA=pca_info,source_edges=len(ri),query_edges=len(qi),K_parameters=len(aa),ridge=.01,
            distinguishing_edges=int(distinct.sum()),source_group_masses=[float(weights[distinct].sum()),float(weights[~distinct].sum())],
            edge_multiplier_scope='within distinguishing/non-distinguishing quality; cap10 before half-mass normalization; globalinverseclasscap unspecified in source',
            gamma=gamma,prepare_seconds=time.perf_counter()-start))
    return _CACHE,False


def _ring_mean(x,hw,valid):
    i,j=physical_edges(hw,valid);total=np.zeros_like(x);count=np.zeros(len(x))
    np.add.at(total,i,x[j]);np.add.at(total,j,x[i]);np.add.at(count,i,1);np.add.at(count,j,1)
    return total/np.maximum(count[:,None],1)


def differential(ep,mode='skew',*,renderer=finish):
    start=time.perf_counter();degeneration=degenerate_margin(ep)
    if degeneration is not None:
        margin,info=degeneration;return renderer(ep,margin,'PRO30_M06' if mode=='skew' else 'PRO30_M06__'+mode,info)
    f,hit=_prepare(ep);q,r=f['q'],f['r'];k=f['K'];qi,qj=f['qi'],f['qj'];unary=f['s_B']
    if mode=='reverse':k=-k
    elif mode=='permuted_K':
        k=k.copy();coefficient=k[f['aa'],f['bb']];coefficient=coefficient[np.random.default_rng(0).permutation(len(coefficient))]
        k[f['aa'],f['bb']]=coefficient;k[f['bb'],f['aa']]=-coefficient
    if mode=='zero_e':edge=np.zeros(len(qi))
    elif mode=='gradient_e':edge=np.asarray(f['C']@unary)
    elif mode in ('pointwise_quadratic','center_neighbor_bilinear'):
        if mode=='pointwise_quadratic':
            rr=r[:,f['aa']]*r[:,f['bb']];qq=q[:,f['aa']]*q[:,f['bb']]
        else:
            rm=_ring_mean(r,ep.r_hw,ep.wvalid);qm=_ring_mean(q,ep.q_hw,ep.q_valid)
            rr=np.einsum('ni,nj->nij',r,rm).reshape(len(r),-1);qq=np.einsum('ni,nj->nij',q,qm).reshape(len(q),-1)
        target=2*np.divide(ep.wf,ep.wvalid,out=np.zeros(len(ep.r)),where=ep.wvalid>0)-1
        residual_target=target-fit_br(ep).predict(ep.r)
        w,b=_weighted_head(ep,rr,target_override=residual_target);unary=unary+qq@w;edge=np.zeros(len(qi))
    else:edge=np.clip(np.sum((q[qi]@k)*q[qj],axis=1),-2,2)
    divergence=np.asarray(f['C'].T@(f['W']*edge));matrix=f['gamma']*sparse.eye(len(q),format='csr')+f['L']
    rhs=f['gamma']*unary+divergence;iterations=[0]
    solve_start=time.perf_counter()
    if mode=='divergence_unary':margin=unary+divergence/f['gamma'];status=0
    else:
        def record(_):iterations[0]+=1
        margin,status=cg(matrix,rhs,x0=unary,rtol=1e-6,atol=0.,maxiter=100,callback=record)
        if status!=0:raise RuntimeError('PRO30_M06 CG failed within original100 steps: status='+str(status))
    margin[ep.q_valid<=0]=-1.
    info=dict(source_contract(6),**f['fit_info'],BR=f['BR_info'],component_cache_hit=hit,
        mode=mode,solver_iterations=iterations[0],solver_status=int(status),solver_seconds=time.perf_counter()-solve_start,
        relative_solver_residual=float(np.linalg.norm(matrix@margin-rhs)/max(np.linalg.norm(rhs),EPS)) if mode!='divergence_unary' else None,
        divergence_norm=float(np.linalg.norm(divergence)),edge_norm=float(np.linalg.norm(edge)),
        K_skew_defect=float(np.max(np.abs(k+k.T))) if k.size else 0.,
        postprocess_seconds=time.perf_counter()-start,new_encoder_forwards=0,quality='unmeasured candidate')
    return renderer(ep,margin,'PRO30_M06' if mode=='skew' else 'PRO30_M06__'+mode,info)


def install(methods,controls,requirements,contracts):
    methods['PRO30_M06']=differential
    for mode in ('zero_e','pointwise_quadratic','center_neighbor_bilinear','divergence_unary','gradient_e','reverse','permuted_K'):
        controls['PRO30_M06__'+mode]=partial(differential,mode=mode)
    requirements['PRO30_M06']=['native unit R/Q','softreferencecoverage','physicalvalid8-neighborgrids']
    contracts['PRO30_M06']=dict(source_contract(6),host='B_R',input_contract='N',
        constants={'unwhitened_dimension_max':16,'skew_ridge':.01,'edge_prediction_clip':[-2,2],'query_distance_width':.1,'gamma':'max(mean_degree/4,.1)'},
        solver_stop={'CG_maxiter':100,'relative_residual':1e-6,'failure':'raise, no baseline completion'},
        renderer='CPU100 two-threshold',controls=[c for c in controls if c.startswith('PRO30_M06__')]+['PRO30_B_R'],
        implementation_assumption='Source does not define denominator of cap10: capped within-edge-role quality, then exacthalfrole mass; needs source reviewer judgment, not a globalinverseclass cap claim')
