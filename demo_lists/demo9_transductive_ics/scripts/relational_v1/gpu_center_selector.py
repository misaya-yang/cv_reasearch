"""Optional F64 center-dot acceleration. The provided CPU core is unchanged.

Original build_stars defines every unit vector/companion. GPU dot products only
propose a role-balanced candidate superset. The original NumPy dot primitive
rescoring and stable(-cos,rootID) decide. Boundary/tie ambiguity falls back to
the ENTIRE original selector. No GPU endpoint cost, tolerance flip or NxN disk.
"""
from contextlib import contextmanager
import numpy as np
import torch
import relational_exclusion_cpu as core

EPS=np.finfo(np.float64).eps

class GPUCenterSelector:
    def __init__(self,original_selector,*,enabled=True,verify=False):
        self.original=original_selector;self.enabled=bool(enabled and torch.cuda.is_available());self.verify=verify
        self.audit=dict(enabled=self.enabled,dtype='float64',GPU_dot_calls=0,GPU_candidate_queries=0,CPU_original_fallbacks=0,
            parity_checks=0,parity_mismatches=0,ambiguous_queries=0,invalid_unit_queries=0,
            safety_bound='8*D*float64eps for verified unit centers; excluded upper bound and CPU near-ties checked',
            original_stars_and_endpoint_costs=True,full_similarity_disk_cache=False,fallback_reasons=[])
        self.source=None;self.ids=None;self.centers=None;self.is_F=None;self.qids={};self.qcenters=None;self.scores=None
        self.pending_q=None;self.validated_center_cache=None
        if self.enabled:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.cuda.set_per_process_memory_fraction(.4,device=0)
    def prepare(self,query_stars,source_stars,coverage):
        ids,centers,is_F=core._center_cache(source_stars,coverage)
        qid=np.asarray(sorted(query_stars),dtype=np.int64)
        self.source=source_stars;self.ids=ids;self.centers=centers;self.is_F=is_F;self.scores=None;self.qids={};self.validated_center_cache=None
        if not self.enabled or centers is None or not len(qid):return
        qc=np.stack([query_stars[int(i)].features[0] for i in qid])
        if centers.shape[1]!=qc.shape[1]:return
        D=centers.shape[1];tol=64*D*EPS
        if not (np.isfinite(centers).all() and np.isfinite(qc).all() and
                np.max(np.abs(np.sum(centers*centers,axis=1)-1))<=tol and
                np.max(np.abs(np.sum(qc*qc,axis=1)-1))<=tol):
            self.audit['invalid_unit_queries']+=1;return
        self.qcenters=qc;self.qids={int(i):j for j,i in enumerate(qid)}
        a=torch.from_numpy(np.ascontiguousarray(qc)).to('cuda',dtype=torch.float64)
        b=torch.from_numpy(np.ascontiguousarray(centers)).to('cuda',dtype=torch.float64)
        self.scores=(a@b.T).clamp(-1.,1.)
        self.audit['GPU_dot_calls']+=1;self.audit['last_GPU_score_shape']=list(self.scores.shape)
        del a,b
    def fallback(self,query,stars,coverage,center_cache,reason):
        self.audit['CPU_original_fallbacks']+=1
        if reason not in self.audit['fallback_reasons']:self.audit['fallback_reasons'].append(reason)
        return self.original(query,stars,coverage,center_cache=center_cache)
    def __call__(self,query,source_stars,source_coverage,*,center_cache=None):
        if not self.enabled:return self.fallback(query,source_stars,source_coverage,center_cache,'GPU unavailable/disabled')
        known=self.source is source_stars and query.root in self.qids
        if known:known=np.array_equal(self.qcenters[self.qids[query.root]],query.features[0])
        if not known:self.prepare({query.root:query},source_stars,source_coverage)
        if self.scores is None:return self.fallback(query,source_stars,source_coverage,center_cache,'unit contract/empty bank unavailable')
        if center_cache is not None and center_cache is not self.validated_center_cache:
            ids,centers,is_F=center_cache
            if not (np.array_equal(ids,self.ids) and np.array_equal(is_F,self.is_F) and np.array_equal(centers,self.centers)):
                return self.fallback(query,source_stars,source_coverage,center_cache,'center cache identity differs')
            self.validated_center_cache=center_cache
        k=min(8,int(self.is_F.sum()),int((~self.is_F).sum()))
        if not k:return self.fallback(query,source_stars,source_coverage,center_cache,'missing role')
        sim=self.scores[self.qids[query.root]].detach().cpu().numpy();bound=8*self.centers.shape[1]*EPS;roles=[]
        for mask in (self.is_F,~self.is_F):
            available=np.flatnonzero(mask);order=np.lexsort((self.ids[available],-sim[available]));kth=sim[available[order[k-1]]]
            candidate=available[sim[available]>=kth-bound]
            exact=np.clip(self.centers[candidate]@query.features[0],-1.,1.)
            ranked=np.lexsort((self.ids[candidate],-exact));take=ranked[:k]
            # GPU proposals never settle a near-boundary/strict-tie decision.
            near=exact[ranked[:min(k+1,len(ranked))]]
            excluded=available[sim[available]<kth-bound]
            ambiguous=(len(near)>1 and np.any(np.abs(np.diff(near))<=2*bound))
            if len(excluded) and exact[take].min()<=sim[excluded].max()+bound:ambiguous=True
            if ambiguous:
                self.audit['ambiguous_queries']+=1
                return self.fallback(query,source_stars,source_coverage,center_cache,'CPU near-tie/boundary; exact original whole bank')
            roles.append([source_stars[int(self.ids[candidate[i]])] for i in take])
        self.audit['GPU_candidate_queries']+=1
        if self.verify:
            old=self.original(query,source_stars,source_coverage,center_cache=center_cache);self.audit['parity_checks']+=1
            if [[s.root for s in role] for role in roles]!=[[s.root for s in role] for role in old]:
                self.audit['parity_mismatches']+=1;self.enabled=False
                self.audit['enabled']=False
                return self.fallback(query,source_stars,source_coverage,center_cache,'PARITY_FAILED: original CPU only')
        return roles[0],roles[1]

@contextmanager
def gpu_center_selection(*,enabled=True,verify=False):
    """Wrap ONLY the original build/selection functions; restore on all exits."""
    original_build=core.build_stars;original_select=core.select_role_banks
    selector=GPUCenterSelector(original_select,enabled=enabled,verify=verify)
    def build(*args,**kwargs):
        result=original_build(*args,**kwargs)
        coverage=kwargs.get('source_coverage')
        if coverage is None:selector.pending_q=result[0]
        elif selector.pending_q is not None:selector.prepare(selector.pending_q,result[0],coverage)
        return result
    try:
        core.build_stars=build;core.select_role_banks=selector
        yield selector.audit
    finally:
        core.build_stars=original_build;core.select_role_banks=original_select
        selector.scores=None;selector.qcenters=None;selector.centers=None;selector.pending_q=None
        selector.audit['restored']=True
