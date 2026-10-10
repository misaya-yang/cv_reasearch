"""Reference adjacency conditioned coverage conflict, with two mass controls.

This is an ambiguity guard, not a probability of the true query boundary.
Identical uncertain correspondences can have nonzero conflict by design.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import brentq
from .context_rank_graph import _update


def factorized_degree_weights(degree):
    """b_i b_j (i!=j) has the same reference degrees, with no self loops."""
    d=np.asarray(degree,np.float64);active=d>0;dd=d[active];b=np.zeros_like(d)
    if len(dd)<2:return b
    total=float(dd.sum());maximum=float(dd.max());pivot=int(dd.argmax())
    if maximum>=total-maximum and len(dd)>2:
        raise ValueError('Boundary degree sequence has no finite positive factorized null')
    lower=2*np.sqrt(maximum)
    def small(s):return 2*dd/(s+np.sqrt(np.maximum(s*s-4*dd,0)))
    def fsmall(s):return float(small(s).sum()-s)
    if fsmall(lower)>=0:
        upper=max(2*lower,2*np.sqrt(total))
        while fsmall(upper)>0:upper*=2
        scale=brentq(fsmall,lower,upper,xtol=1e-12,rtol=1e-14)
        values=small(scale)
    else:
        def flarge(s):
            values=small(s);return float(values.sum()-2*values[pivot])
        upper=2*lower
        while flarge(upper)<0:
            upper*=2
            if not np.isfinite(upper):raise ValueError('Factorized degree balancing failed')
        scale=brentq(flarge,lower,upper,xtol=1e-12,rtol=1e-14)
        values=small(scale);values[pivot]=scale-values[pivot]
    b[active]=values;error=float(np.max(np.abs(b*(b.sum()-b)-d)))
    if error>1e-9*max(1,float(d.max())):raise RuntimeError('Reference degree null does not match')
    return b


def reference_graph(r,device):
    rr=r.to(device);sim=rr@rr.T;sim.fill_diagonal_(-2)
    values,idx=sim.topk(min(20,len(r)-1),dim=1);del sim,rr
    values,idx=values.cpu(),idx.cpu();distance=(1-values).clamp_min(0)
    weights=torch.exp(-distance/distance[:,-1:].clamp_min(1e-6)).numpy()
    n,k=idx.shape
    w=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(n),k),idx.numpy().ravel())),shape=(n,n))
    w=w.multiply(w.T);w.data=np.sqrt(w.data)
    mean=float(np.asarray(w.sum(1)).mean());w=w/max(mean,1e-8)
    return w.astype(np.float64).tocsr()


def edge_dot(left,right,rows,cols):
    result=np.empty(len(rows),np.float64)
    for start in range(0,len(rows),128):
        sl=slice(start,start+128);result[sl]=np.sum(left[rows[sl]]*right[cols[sl]],axis=1,dtype=np.float64)
    return result


def symmetric_ratio(num,den,rows,cols,n):
    keys=rows.astype(np.int64)*n+cols;order=np.argsort(keys)
    reverse=order[np.searchsorted(keys[order],cols.astype(np.int64)*n+rows)]
    assert np.array_equal(keys[reverse],cols.astype(np.int64)*n+rows)
    num=.5*(num+num[reverse]);den=.5*(den+den[reverse])
    ratio=np.divide(num,den,out=np.zeros_like(num),where=den>0)
    if (ratio< -1e-7).any() or (ratio>1+1e-7).any() or not np.isfinite(ratio).all():
        raise RuntimeError('Invalid reference conflict ratio')
    return np.clip(ratio,0,1),num,den


def match_loss(weights,cost,budget):
    if budget==0:return np.zeros_like(cost),0.
    possible=float(weights[cost>0].sum())
    if budget>possible+1e-10*max(1,possible):raise ValueError('Null cannot match requested edge loss')
    upper=float(1/cost[cost>0].min())
    def loss(scale):return float(np.dot(weights,np.minimum(scale*cost,1)))
    scale=upper if abs(budget-possible)<1e-12*max(1,possible) else brentq(
        lambda x:loss(x)-budget,0,upper,xtol=1e-12,rtol=1e-14)
    return np.minimum(scale*cost,1),scale


def build(parent,s,q,r,cov,device='mps'):
    if parent.dtype!=np.float64 or not sparse.isspmatrix_csr(parent):raise ValueError('Actual FP64 parent H required')
    q,r=torch.as_tensor(q),torch.as_tensor(r);c=np.asarray(cov,np.float32).reshape(-1).astype(np.float64)
    s=np.asarray(s,np.float32).reshape(-1);n=len(s);m=len(c)
    if q.dtype!=torch.float32 or r.dtype!=torch.float32 or q.shape[0]!=n or r.shape[0]!=m or q.shape[1]!=r.shape[1] or parent.shape!=(n,n):raise ValueError('Invalid parent features')
    if not torch.isfinite(q).all() or not torch.isfinite(r).all() or (c<0).any() or (c>1).any():raise ValueError('Nonfinite features/coverage')
    if (q.norm(dim=1)-1).abs().max()>1e-3 or (r.norm(dim=1)-1).abs().max()>1e-3:raise ValueError('Use actual parent unit features')
    coo=parent.tocoo();off=coo.row!=coo.col;rows,cols=coo.row[off].copy(),coo.col[off].copy()
    w=-coo.data[off]/16;group=s>.5;affected=group[rows]|group[cols]
    if not np.isfinite(parent.data).all() or (w<0).any() or not np.isfinite(s).all():raise ValueError('Invalid parent graph/source')
    factors={key:np.ones(len(w),np.float64) for key in ('candidate','factorized','uniform')}
    arrays=dict(group=group,edge_rows=rows,edge_cols=cols,parent_weights=w,affected_edges=affected,coverage=c)
    fmass=float(c.sum());bmass=float((1-c).sum());fallback=None;beta=0.;eta=1.
    if fmass==0 or bmass==0:fallback='missing_reference_role_identity'
    elif not affected.any():fallback='empty_E_identity'
    elif np.ptp(c)==0:fallback='constant_reference_coverage_identity'
    if fallback is None:
        wr=reference_graph(r,device);cut=wr.copy();cr=cut.tocoo();cut.data*=np.square(c[cr.row]-c[cr.col]);cut.eliminate_zeros()
        if cut.nnz==0:fallback='no_observed_reference_cut_identity'
        else:
            ids=np.flatnonzero(affected);er,ec=rows[ids],cols[ids]
            active=np.unique(np.r_[er,ec]);positions=np.full(n,-1,np.int64);positions[active]=np.arange(len(active))
            pr,pc=positions[er],positions[ec]
            logits=(q[active].to(device)@r.to(device).T).cpu().numpy().astype(np.float64)
            kernel=np.exp((logits-logits.max(axis=1,keepdims=True))/.07)
            alpha=.5*(c/fmass+(1-c)/bmass);p=kernel*alpha;p/=p.sum(axis=1,keepdims=True)
            num=edge_dot(p,(cut@p.T).T,pr,pc);den=edge_dot(p,(wr@p.T).T,pr,pc)
            cost,num,den=symmetric_ratio(num,den,er,ec,n)
            d=np.asarray(wr.sum(1)).ravel();b=factorized_degree_weights(d);v=p*b
            # Nonnegative sums avoid cancellation of the zero-diagonal null.
            before=np.c_[np.zeros(len(v)),np.cumsum(v,axis=1)[:,:-1]]
            after=np.c_[np.cumsum(v[:,::-1],axis=1)[:,:-1][:,::-1],np.zeros(len(v))]
            null_den=edge_dot(v,before+after,pr,pc)
            levels,labels=np.unique(c,return_inverse=True)
            grouping=sparse.csr_matrix((np.ones(m),(labels,np.arange(m))),shape=(len(levels),m))
            mass=(grouping@v.T).T;dist=np.square(levels[:,None]-levels[None,:])
            null_num=edge_dot(mass,mass@dist,pr,pc)
            null_cost,null_num,null_den=symmetric_ratio(null_num,null_den,er,ec,n)
            budget=float(np.dot(w[ids],cost));matched_cost,beta=match_loss(w[ids],null_cost,budget)
            eta=1-budget/max(float(w[ids].sum()),1e-12)
            factors['candidate'][ids]=1-cost;factors['factorized'][ids]=1-matched_cost;factors['uniform'][ids]=eta
            arrays.update(active_query_nodes=active,reference_indptr=wr.indptr,reference_indices=wr.indices,
                reference_weights=wr.data,reference_degree=d,factorized_b=b,affected_edge_indices=ids,
                real_num=num,real_den=den,real_cost=cost,factorized_num=null_num,factorized_den=null_den,
                factorized_cost_raw=null_cost,factorized_cost_mass_matched=matched_cost,
                correspondence_tensor_sha256=np.asarray(hashlib.sha256(p.tobytes()).hexdigest()))
    result=dict(arrays=arrays,diagnostics=dict(fallback=fallback,n_G=int(group.sum()),affected_edges=int(affected.sum()),
        uniform_eta=eta,factorized_mass_scale=beta,reference_degree_preserved=True,same_P_can_cut=True,
        operator='adjacency-conditioned coverage conflict; ambiguity guard, not query boundary probability',
        dense_device=device,encoder_calls=0,query_GT_in_inference=False))
    masses={}
    for kind,factor in factors.items():
        neww=w*factor;new,degree,diff=_update(parent,rows,cols,neww-w)
        result[kind+'_H']=new;result[kind+'_Hdiff']=diff
        arrays[kind+'_weights']=neww;arrays[kind+'_delta_degree']=degree
        masses[kind]=float(neww[affected].sum())
        assert np.array_equal(neww[~affected],w[~affected])
    if max(masses.values())-min(masses.values())>1e-9*max(1,float(w[affected].sum())):raise RuntimeError('Query edge masses do not match')
    result['diagnostics']['query_edge_masses']=masses
    return result
