"""Full-anchor cosine unbalanced entropic transport; query labels never enter.

For supplied UNIT native descriptors, C_ij = 1 - <q_i,r_j>, a_i = 1/Nq.
By intentional default rule, b puts total .5 on annotated support FG and .5 on
BG, uniformly WITHIN each class. This is a support prior, not a claimed query
foreground fraction; finite KL relaxation can still transmit a soft quota bias.

Convex primal for finite rho_q,rho_r >= 0 and epsilon > 0:
 <C,P> + epsilon KL(P || a outer b)
       + rho_q KL(P1 || a) + rho_r KL(P.T1 || b), P >= 0.
KL(x||y) = sum(x log(x/y)-x+y), including non-unit transported mass.
Chunked log-Sinkhorn has bounded iterations; failure is explicit. Numerical
fixed-point convergence is not task gain or a guarantee about mask correctness.
"""
import math

import torch


class TransportError(RuntimeError):
    pass


class TransportConvergenceError(TransportError):
    pass


def _validate(reference, foreground, query, chunk_size, max_ram_bytes):
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError('reference/query need [N,C] in the SAME supplied native space')
    if not len(reference) or not len(query) or reference.device != query.device:
        raise ValueError('nonempty reference/query on the same device required')
    if reference.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64) or query.dtype!=reference.dtype:
        raise ValueError('matching floating descriptor dtypes required')
    if foreground.dtype!=torch.bool or foreground.numel()!=len(reference):
        raise ValueError('support foreground must be an annotated bool vector of length Nr')
    foreground=foreground.flatten().to(reference.device)
    if not foreground.any() or foreground.all():
        raise ValueError('both annotated support FG and BG anchors required; no invented anchors')
    if not isinstance(chunk_size,int) or not 1<=chunk_size<=4096 or max_ram_bytes<=0:
        raise ValueError('chunk_size must be 1..4096; max_ram_bytes positive')
    dtype=torch.float64 if reference.dtype==torch.float64 else torch.float32
    element=8 if dtype==torch.float64 else 4
    # Conservative tensor workspace bound: input+cast copies, dual/mass vectors,
    # and eight block-sized temporaries. BLAS allocator/runtime overhead excluded.
    nq,nr=len(query),len(reference);cq=min(chunk_size,nq);cr=min(chunk_size,nr)
    estimate=(reference.numel()+query.numel())*(reference.element_size()+element)
    estimate+=16*(nq+nr)*element+8*cq*cr*element+nr
    if estimate>max_ram_bytes:
        raise MemoryError(f'estimated tensor bytes {estimate} exceed explicit budget {max_ram_bytes}')
    reference=reference.to(dtype);query=query.to(dtype)
    if not torch.isfinite(reference).all() or not torch.isfinite(query).all():
        raise ValueError('nonfinite descriptors')
    if (reference.norm(dim=1)-1).abs().max()>2e-3 or (query.norm(dim=1)-1).abs().max()>2e-3:
        raise ValueError('UNIT descriptors required; method never renormalizes/projects native geometry')
    return reference,foreground,query,estimate


class _ChunkKernel:
    def __init__(self,query,reference,loga,logb,epsilon,chunk):
        self.query,self.reference=query,reference
        self.loga,self.logb,self.epsilon,self.chunk=loga,logb,epsilon,chunk
    def block(self,q0,q1,r0,r1):
        cost=1-self.query[q0:q1]@self.reference[r0:r1].T
        logkernel=self.loga[q0:q1,None]+self.logb[None,r0:r1]-cost/self.epsilon
        return cost,logkernel
    def apply(self,scaling,transpose=False):
        size=len(self.reference) if transpose else len(self.query)
        other=len(self.query) if transpose else len(self.reference)
        out=torch.empty(size,device=self.query.device,dtype=self.query.dtype)
        for start in range(0,size,self.chunk):
            stop=min(start+self.chunk,size)
            accum=torch.full((stop-start,),-torch.inf,device=out.device,dtype=out.dtype)
            for at in range(0,other,self.chunk):
                end=min(at+self.chunk,other)
                if transpose:
                    _,logk=self.block(at,end,start,stop)
                    term=torch.logsumexp(logk+scaling[at:end,None],dim=0)
                else:
                    _,logk=self.block(start,stop,at,end)
                    term=torch.logsumexp(logk+scaling[None,at:end],dim=1)
                accum=torch.logaddexp(accum,term)
            out[start:stop]=accum
        return out


def _kl(mass,logprior):
    safe=mass.clamp_min(torch.finfo(mass.dtype).tiny)
    return (mass*(safe.log()-logprior)-mass+logprior.exp()).sum()


@torch.no_grad()
def transport_scores(reference,foreground,query,*,mode='unbalanced',epsilon=.1,
                     rho_query=.1,rho_reference=.1,class_balance=True,
                     chunk_size=256,max_iterations=200,tolerance=1e-5,
                     max_ram_bytes=256*1024**2,require_convergence=True):
    """Return (row-conditional FG scores, diagnostics); all anchors, no query GT.

    modes share descriptors, full support anchors and class prior:
      kde: P=(a outer b)*exp(-C/epsilon), no marginal coupling (rho=0).
      balanced: exact row/column marginals; this INDUCES support FG mass in query.
      unbalanced: finite KL penalties, no imposed query FG-mass equality.
    score>0.5 is a fixed label rule, score==0.5 is BG; not calibrated probability.
    """
    if mode not in ('kde','balanced','unbalanced') or not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError('invalid mode/positive epsilon')
    if not all(math.isfinite(r) and r>=0 for r in (rho_query,rho_reference)):
        raise ValueError('KL penalties must be finite and nonnegative')
    if not isinstance(max_iterations,int) or not 1<=max_iterations<=2000 or not 0<tolerance<1:
        raise ValueError('max_iterations 1..2000; tolerance in (0,1)')
    reference,foreground,query,ram=_validate(reference,foreground,query,chunk_size,max_ram_bytes)
    nq,nr=len(query),len(reference);device=query.device;dtype=query.dtype
    a=torch.full((nq,),1/nq,device=device,dtype=dtype)
    b=torch.full((nr,),1/nr,device=device,dtype=dtype)
    if class_balance:
        b[foreground]=.5/int(foreground.sum());b[~foreground]=.5/int((~foreground).sum())
    loga,logb=a.log(),b.log();kernel=_ChunkKernel(query,reference,loga,logb,epsilon,chunk_size)
    logu=torch.zeros_like(a);logv=torch.zeros_like(b);converged=False;delta=math.inf
    if mode=='balanced':tau_q=tau_r=1.
    elif mode=='kde':tau_q=tau_r=0.
    else:tau_q=rho_query/(rho_query+epsilon);tau_r=rho_reference/(rho_reference+epsilon)
    if tau_q==0 and tau_r==0:
        # Exact closed-form KDE control: no unnecessary Sinkhorn/matmul passes.
        iteration=0;delta=0.;converged=True
    else:
        for iteration in range(1,max_iterations+1):
            newu=tau_q*(loga-kernel.apply(logv))
            newv=tau_r*(logb-kernel.apply(newu,transpose=True))
            delta=float(torch.maximum((newu-logu).abs().max(),(newv-logv).abs().max()))
            logu,logv=newu,newv
            if not torch.isfinite(logu).all() or not torch.isfinite(logv).all():
                raise TransportError('nonfinite Sinkhorn duals; no scores emitted')
            if delta<=tolerance:converged=True;break
    if not converged and require_convergence:
        raise TransportConvergenceError(f'{mode} did not reach fixed-point tolerance {tolerance} after {max_iterations} iterations; delta={delta}')
    rowlog=torch.full_like(a,-torch.inf);collog=torch.full_like(b,-torch.inf);fglog=torch.full_like(a,-torch.inf)
    cost_sum=torch.zeros((),device=device,dtype=dtype);entropy_sum=cost_sum.clone()
    for q0 in range(0,nq,chunk_size):
        q1=min(nq,q0+chunk_size)
        for r0 in range(0,nr,chunk_size):
            r1=min(nr,r0+chunk_size);cost,logk=kernel.block(q0,q1,r0,r1)
            lp=logu[q0:q1,None]+logk+logv[None,r0:r1]
            rowlog[q0:q1]=torch.logaddexp(rowlog[q0:q1],torch.logsumexp(lp,dim=1))
            collog[r0:r1]=torch.logaddexp(collog[r0:r1],torch.logsumexp(lp,dim=0))
            selected=foreground[r0:r1]
            if selected.any():fglog[q0:q1]=torch.logaddexp(fglog[q0:q1],torch.logsumexp(lp[:,selected],dim=1))
            plan=lp.exp();cost_sum+=(plan*cost).sum()
            entropy_sum+=(plan*(lp-loga[q0:q1,None]-logb[None,r0:r1])-plan).sum()
    rows,cols=rowlog.exp(),collog.exp();scores=(fglog-rowlog).exp()
    if not torch.isfinite(scores).all() or not torch.isfinite(rows).all() or float(rows.sum())<=0:
        raise TransportError('invalid/underflowed transport mass; no scores emitted')
    rq=0. if mode=='kde' else rho_query;rr=0. if mode=='kde' else rho_reference
    objective=cost_sum+epsilon*(entropy_sum+1)
    if mode!='balanced':objective+=rq*_kl(rows,loga)+rr*_kl(cols,logb)
    stationarity=None
    if mode!='balanced':
        residual_q=epsilon*logu+rq*(rowlog-loga)
        residual_r=epsilon*logv+rr*(collog-logb)
        stationarity=float(torch.maximum((residual_q.max()+residual_r.max()).abs(),(residual_q.min()+residual_r.min()).abs()))
    diagnostics=dict(mode=mode,converged=converged,iterations=iteration,fixed_point_delta=delta,
        epsilon=epsilon,rho_query=rq if mode!='balanced' else None,rho_reference=rr if mode!='balanced' else None,
        query_row_prior='uniform',support_prior='FG/BG each .5, uniform within class' if class_balance else 'uniform anchors (support area prior)',
        imposed_query_FG_quota=mode=='balanced',support_FG_prior_mass=float(b[foreground].sum()),
        transported_mass=float(rows.sum()),transported_FG_fraction=float(cols[foreground].sum()/cols.sum()),
        mean_row_conditional_FG=float(scores.mean()),hard_FG_fraction=float((scores>.5).float().mean()),
        row_marginal_L1=float((rows-a).abs().sum()),column_marginal_L1=float((cols-b).abs().sum()),
        convex_primal_value=float(objective),unbalanced_stationarity_max_abs=stationarity,
        estimated_tensor_bytes=ram,max_ram_bytes=max_ram_bytes,
        geometry='Supplied unit native descriptors; untouched cosine 1-dot; no projection, spatial cost, anchor compression or query labels',
        scope='Numerical convergence only; finite KL/support balancing may still impose soft quota bias; no task or global empirical guarantee')
    return scores,diagnostics


def unbalanced_transport_scores(reference,foreground,query,**kwargs):
    return transport_scores(reference,foreground,query,mode='unbalanced',**kwargs)


def balanced_transport_scores(reference,foreground,query,**kwargs):
    return transport_scores(reference,foreground,query,mode='balanced',**kwargs)


def full_anchor_kde_scores(reference,foreground,query,**kwargs):
    return transport_scores(reference,foreground,query,mode='kde',**kwargs)
