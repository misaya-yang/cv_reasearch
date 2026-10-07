"""Supplied Astra B076--B100 and C101--C150, original recipes.

The registry only includes implemented complete algorithms. Missing original
RGB/attention/layer/encoder assets raise explicit blockers, never substitutes.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation
from scipy.special import logsumexp, expit

from .common import Result,artifact,ArtifactUnavailable
from .group_076_150_common import (Context, Kernel, Prepared, G, blocks, dot, edges4,
                                  prepare, region_run, point_run, dynamic_region_run, region_point_run, rp, spherical, unit)


METHODS = {}
class _Controls(dict):
    def __setitem__(self,name,fn):
        def tagged(ep):
            out=fn(ep)
            original=out.info.get('method_id')
            out.info=dict(out.info,method_id=name,control_id=name,underlying_recipe_id=original)
            return out
        super().__setitem__(name,tagged)
CONTROLS = _Controls()
REQUIREMENTS = {}
ASSUMPTIONS = {}


def result(ep, z, method, info):
    info = dict(info, method_id=method, source_group='B/C supplied Astra300',
                renderer='continuous_field_direct_original_bilinear_align_corners_false',
                qgt_access=False, implementation_assumptions=ASSUMPTIONS.get(method, []))
    return Result(field=np.asarray(z).reshape(ep.q_hw), threshold=0., info=info)


def register(method, fn, assumptions=(), requirements=('final_unit_dino',)):
    METHODS[method] = fn
    ASSUMPTIONS[method] = list(assumptions)
    REQUIREMENTS[method] = list(requirements)


def region_method(method, descriptor):
    def run(ep):
        z, info = region_run(ep, descriptor)
        return result(ep, z, method, info)
    return run


def point_method(method, descriptor, additive=False):
    def run(ep):
        z, info = point_run(ep, descriptor, additive)
        return result(ep, z, method, info)
    return run


def region_point_method(method,sample_fn,additive=False):
    def run(ep):
        z,info=region_point_run(ep,sample_fn,additive)
        return result(ep,z,method,info)
    return run


def _ring(ctx, c, distance, excluded=None):
    mask = np.zeros(ctx.hw, bool); mask.ravel()[c] = True
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], bool)
    outer = binary_dilation(mask, structure=structure, iterations=distance)
    ring = outer & ~mask
    ring.ravel()[ctx.valid <= 0] = False
    if excluded is not None: ring.ravel()[excluded] = False
    return np.flatnonzero(ring.ravel())


def _c102(ctx, prepared):
    excluded = np.zeros(len(ctx.x), bool)
    for a in ctx.atoms:
        if np.any(ctx.positive[a]): excluded[a] = True
    descriptors = []
    for c in ctx.regions:
        ring = _ring(ctx,c,3,excluded)
        if len(ring)<4: descriptors.append([np.nan]*5); continue
        inside, outside = ctx.u[c].mean(), ctx.u[ring].mean()
        pi = unit(ctx.profile[c].mean(axis=0)); po = unit(ctx.profile[ring].mean(axis=0))
        descriptors.append([inside,outside,inside-outside,np.mean(ctx.negative[ring]),float(pi@po)])
    return np.asarray(descriptors)


register('C102',region_method('C102',_c102),
         ['Complete-profile cosine compares mean response vectors; ring is exact four-neighbor distance 1--3.'])


def _halves(ctx,c):
    y,x = np.unravel_index(c,ctx.hw); coords = np.column_stack((y,x)).astype(float)
    if len(c)<2: return None
    centered=coords-coords.mean(axis=0)
    vals,vec=np.linalg.eigh(centered.T@centered)
    axis=vec[:,-1]
    if axis[np.argmax(np.abs(axis))]<0: axis=-axis
    order=np.lexsort((c,centered@axis)); split=(len(c)+1)//2
    return c[order[:split]],c[order[split:]]


def _pair_profile(ctx,c,protos):
    halves=_halves(ctx,c)
    if halves is None:return None
    bank=np.concatenate(protos)
    return np.asarray([dot(ctx.x[h],bank).mean(axis=0) for h in halves])


def _half_pair_library(ctx,p,protos,shuffle=False):
    # All reference candidates reconstructed with the same fold-excluded bank.
    rc=p.context(True,ctx.fold)
    library=[];labels=[]
    for c in rc.regions:
        if ctx.fold is not None and np.any(p.rb[c]==ctx.fold):continue
        if np.all(p.c[c]>=.9):label=1.
        elif np.all(p.c[c]<=.1):label=0.
        else:continue
        row=_pair_profile(rc,c,protos)
        if row is not None:library.append(row);labels.append(label)
    if shuffle and library:
        lib=np.array(library);labels_array=np.asarray(labels)
        rng=np.random.default_rng(104)
        for role in(0,1):
            selected=np.flatnonzero(labels_array==role)
            if len(selected):lib[selected,1]=lib[rng.permutation(selected),1]
        library=lib.tolist()
    return library,labels


def _c104(ctx,p,joint_only=False,shuffle=False):
    proto=ctx.role_prototypes(16);library,labels=_half_pair_library(ctx,p,proto,shuffle)
    rows=[]
    if not library or len(set(labels))<2:
        return np.full((len(ctx.regions),3),np.nan)
    lib=np.asarray(library);label=np.asarray(labels)
    # One bandwidth/standardizer for joint and marginal kernels; permute slots.
    paired=np.r_[lib.reshape(len(lib),-1),lib[:,::-1].reshape(len(lib),-1)]
    model=Kernel(paired,np.r_[label,label],np.full(2*len(label),.5))
    standard=model.standardize(paired)[:,:model.n]
    halfdim=lib.shape[-1]
    def conditional(pair,role):
        z=model.standardize(pair.reshape(1,-1))[0,:model.n]
        side=standard[np.r_[label==role,label==role]]
        a=-np.sum((side[:,:halfdim]-z[:halfdim])**2,axis=1)/(2*model.sigma**2)
        b=-np.sum((side[:,halfdim:]-z[halfdim:])**2,axis=1)/(2*model.sigma**2)
        joint=logsumexp(a+b)-np.log(len(side))
        return joint if joint_only else joint-(logsumexp(a)-np.log(len(side)))-(logsumexp(b)-np.log(len(side)))
    for c in ctx.regions:
        pair=_pair_profile(ctx,c,proto)
        if pair is None:rows.append([np.nan]*3);continue
        # Slots are already symmetrized in the source library.
        halves=_halves(ctx,c)
        rows.append([ctx.u[halves[0]].mean(),ctx.u[halves[1]].mean(),
                     conditional(pair,1)-conditional(pair,0)])
    return np.asarray(rows)


def _c104_predict(ep,joint_only=False,shuffle=False):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C104',{'degenerate':deg[1]})
    descriptors=[];labels=[];weights=[]
    for fold in range(16):
        if not np.any((p.rb==fold)&(p.rv>0)):continue
        ctx=p.context(True,fold)
        if not len(ctx.fg) or not len(ctx.bg):continue
        for c,d in zip(ctx.regions,_c104(ctx,p,joint_only,shuffle)):
            if p.rb[c[len(c)//2]]!=fold or not np.isfinite(d).any():continue
            coverage=ep.wf[c].sum()/max(ep.wvalid[c].sum(),1e-12)
            descriptors.extend((d,d[[1,0,2]]));labels.extend((coverage,coverage));weights.extend((.5,.5))
    model=Kernel(np.asarray(descriptors).reshape(-1,3),np.asarray(labels),np.asarray(weights))
    ctx=p.context();d=_c104(ctx,p,joint_only,shuffle);a,b=model.log_densities(d);c,e=model.log_densities(d[:,[1,0,2]])
    h=np.clip(np.logaddexp(a,c)-np.logaddexp(b,e),-4,4);h[~np.isfinite(d).any(axis=1)]=0
    z=ctx.u+rp(ctx,h);z[ctx.valid<=0]=-4
    return result(ep,z,'C104',{'source_samples':len(descriptors),'swapped_final_density_average':True,'kernel_active':bool(model.active)})


register('C104',_c104_predict,
         ['Spatial principal-axis sign uses the largest absolute coordinate; odd extra point goes to H1.',
          'Joint/marginal kernels share the common standardized joint bandwidth; source slots and final K1 source/query densities are both swap-averaged.'])


def _three_blocks(ctx,c):
    if len(c)<3:return None
    yy,xx=np.unravel_index(c,ctx.hw);coords=np.column_stack((yy,xx)).astype(float)
    first=0;chosen=[first];dist=np.full(len(c),np.inf)
    for _ in range(2):
        dist=np.minimum(dist,np.sum((coords-coords[chosen[-1]])**2,axis=1));dist[chosen]=-np.inf
        chosen.append(int(np.argmax(dist)))
    distance=np.sum((coords[:,None]-coords[chosen][None])**2,axis=2)
    labels=np.argmin(distance,axis=1)
    groups=[c[labels==i] for i in range(3)]
    return groups if all(len(g) for g in groups) else None


def _triple(ctx,c,permutation=(0,1,2)):
    groups=_three_blocks(ctx,c)
    if groups is None:return np.full(7,np.nan)
    groups=[groups[i] for i in permutation]
    means=np.array([ctx.u[g].mean() for g in groups])
    centers=unit(np.array([ctx.x[g].mean(axis=0) for g in groups]))
    coords=np.array([np.column_stack(np.unravel_index(g,ctx.hw)).mean(axis=0) for g in groups])
    a,b=coords[1]-coords[0],coords[2]-coords[0]
    area=abs(a[0]*b[1]-a[1]*b[0])/2
    return np.r_[means, centers[0]@centers[1],centers[1]@centers[2],centers[0]@centers[2],area]


def _c105_predict(ep, permutation=True,pairwise=False):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C105',{'degenerate':deg[1]})
    orders=tuple(itertools.permutations(range(3))) if permutation else ((0,1,2),)
    d=[];labels=[];weights=[]
    for fold in range(16):
        if not np.any((p.rb==fold)&(p.rv>0)):continue
        ctx=p.context(True,fold)
        if not len(ctx.fg) or not len(ctx.bg):continue
        for c in ctx.regions:
            if p.rb[c[len(c)//2]]!=fold or len(c)<3:continue
            groups=_three_blocks(ctx,c)
            if groups is None:continue
            cov=np.array([np.sum(ep.wf[g])/max(np.sum(ep.wvalid[g]),1e-12) for g in groups])
            coverage=np.sum(ep.wf[c])/max(np.sum(ep.wvalid[c]),1e-12)
            mixed=not(np.all(cov>=.9)or np.all(cov<=.1))
            sample_weight=1-coverage if mixed else 1.
            for order in orders:
                d.append(_triple(ctx,c,order));labels.append(coverage);weights.append(sample_weight/len(orders))
    model=Kernel(np.asarray(d).reshape(-1,7),np.asarray(labels),np.asarray(weights))
    pair_models=[Kernel(np.asarray(d).reshape(-1,7)[:,cols],np.asarray(labels),np.asarray(weights))for cols in((0,1,3),(1,2,4),(0,2,5),(6,))]if pairwise else None
    ctx=p.context();increments=[]
    for c in ctx.regions:
        if len(c)<3:increments.append(0.);continue
        data=np.array([_triple(ctx,c,o) for o in orders])
        if pairwise:
            scores=[]
            for km,cols in zip(pair_models,((0,1,3),(1,2,4),(0,2,5),(6,))):
                lf,lb=km.log_densities(data[:,cols]);scores.append(logsumexp(lf)-logsumexp(lb))
            value=np.mean(scores)
        else:
            lf,lb=model.log_densities(data);value=logsumexp(lf)-logsumexp(lb)
        increments.append(float(np.clip(value,-4,4)))
    z=ctx.u+rp(ctx,increments);z[ctx.valid<=0]=-4
    return result(ep,z,'C105',{'kernel_active':bool(model.active),'source_samples':len(d),
                             'permutations':len(orders),'candidate_count':len(ctx.regions)})


register('C105',_c105_predict,
         ['Spatial FPS starts at the lowest row ID; each point joins its nearest seed.',
          'All six orders are marginalized as kernel densities, not averaged log odds; source label is actual three-block coverage, and mixed triples have sample weight 1-coverage.'])


def _c106(ctx,p,crossfit=True,centers=16):
    pointblocks=blocks(ctx.hw);v=ctx.u.copy()
    for fold in range(16):
        held=(pointblocks==fold)&(ctx.valid>0)
        allowed=(pointblocks!=fold)if crossfit else np.ones(len(ctx.x),bool)
        fg=np.flatnonzero(ctx.positive&allowed)
        bg=np.flatnonzero(ctx.negative&allowed)
        if not held.any() or not len(fg) or not len(bg):continue
        pf=spherical(ctx.x[fg],min(centers,len(fg)))[0]
        pb=spherical(ctx.x[bg],min(centers,len(bg)))[0]
        v[held]=np.clip((dot(ctx.x[held],pf).max(axis=1)-dot(ctx.x[held],pb).max(axis=1))/ctx.scale,-4,4)
    anchors=np.flatnonzero(ctx.positive|ctx.negative)
    distance=np.full(len(ctx.x),np.nan)
    if len(anchors):
        xy=np.column_stack(np.unravel_index(np.arange(len(ctx.x)),ctx.hw))
        for i in range(0,len(ctx.x),256):
            distance[i:i+256]=np.sqrt(np.sum((xy[i:i+256,None]-xy[anchors][None])**2,axis=2).min(axis=1))
    return np.column_stack((ctx.u,v,(np.sign(ctx.u)!=np.sign(v)).astype(float),distance))


register('C106',point_method('C106',_c106),
         ['Cross-fitted query prototype margin uses the fixed reference OOF scale; nearest-anchor distance is in native grid units.'])


def _triple_rows(ctx):
    grid=np.arange(len(ctx.x)).reshape(ctx.hw);rows=[]
    for axis in (0,1):
        view=grid if axis==0 else grid.T
        for line in view:
            for k in range(2,len(line)):
                ids=line[k-2:k+1]
                if np.all(ctx.valid[ids]>0):rows.extend((ids.copy(),ids[::-1].copy()))
    return np.asarray(rows,dtype=int).reshape(-1,3)


def _triple_descriptor(ctx,ids):
    if not len(ids):return np.empty((0,6))
    a,b,c=ctx.x[ids[:,0]],ctx.x[ids[:,1]],ctx.x[ids[:,2]]
    return np.column_stack((ctx.u[ids],np.einsum('id,id->i',a,b),np.einsum('id,id->i',b,c),np.einsum('id,id->i',a,c)))


class _States:
    def __init__(self,d,soft):
        self.states=np.array(list(itertools.product((0,1),repeat=3)))
        self.model=Kernel(d,np.full(len(d),.5))
        self.weights=np.prod(np.where(self.states[None]>0,soft[:,None],1-soft[:,None]),axis=2)
    def costs(self,d):
        d=np.atleast_2d(d);z=self.model.standardize(d);source=self.model.d
        logk=-np.sum((z[:,None]-source[None])**2,axis=2)/(2*self.model.sigma**2)
        costs=np.empty((len(d),8))
        for s in range(8):
            w=self.weights[:,s]
            if w.sum()>0:
                lw=np.log(w,where=w>0,out=np.full(len(w),-np.inf))
                costs[:,s]=-logsumexp(logk+lw,axis=1)+np.log(w.sum())
            else:
                u=d[:,:3];state=self.states[s]
                costs[:,s]=np.sum(np.logaddexp(0,u)-u*state,axis=1)
        return costs


def _state_library(ctx,p):
    rc=p.context(True,ctx.fold);ids=_triple_rows(rc)
    if ctx.fold is not None:ids=ids[~np.any(p.rb[ids]==ctx.fold,axis=1)]
    return _States(_triple_descriptor(rc,ids),p.c[ids])


def _viterbi_second_order(ctx,line,library):
    u=ctx.u[line];n=len(line)
    if n<3:return u>0
    triples=np.array([line[k-2:k+1] for k in range(2,n)])
    costs=library.costs(_triple_descriptor(ctx,triples))
    dp=np.array([np.logaddexp(0,u[:2]).sum()-np.dot(u[:2],(a,b)) for a,b in itertools.product((0,1),repeat=2)])
    paths=list(itertools.product((0,1),repeat=2));edits=np.array([np.sum(np.asarray(path)!=(u[:2]>0))for path in paths])
    for i in range(n-2):
        nxt=np.full(4,np.inf);new_edits=np.full(4,np.iinfo(np.int32).max);new_paths=[None]*4
        for a,b,c in itertools.product((0,1),repeat=3):
            old=2*a+b;new=2*b+c;value=dp[old]+costs[i,4*a+2*b+c]
            changed=int(edits[old])+int(bool(c)!=(u[i+2]>0));path=paths[old]+(c,)
            if (value,changed,path)<(nxt[new],new_edits[new],new_paths[new] or (2,)*n):
                nxt[new]=value;new_edits[new]=changed;new_paths[new]=path
        dp=nxt;edits=new_edits;paths=new_paths
    state=min(range(4),key=lambda s:(dp[s],edits[s],paths[s]))
    return np.array(paths[state],bool)


def _c108_field(ctx,p,independent=False,pairwise=False):
    library=_state_library(ctx,p);grid=np.arange(len(ctx.x)).reshape(ctx.hw)
    if pairwise:
        original=library
        class Factorized:
            def costs(self,d):
                costs=original.costs(d);out=np.empty_like(costs)
                for a,b,c in itertools.product((0,1),repeat=3):
                    ab=-logsumexp(-costs[:,[4*a+2*b,4*a+2*b+1]],axis=1)+np.log(2)
                    bc=-logsumexp(-costs[:,[2*b+c,4+2*b+c]],axis=1)+np.log(2)
                    out[:,4*a+2*b+c]=.5*(ab+bc)
                return out
        library=Factorized()
    votes=np.zeros(len(ctx.x));count=np.zeros(len(ctx.x))
    for view in (grid,grid.T):
        for line in view:
            # Padding is outside the physical query; never becomes a border BG prior.
            ids=line[ctx.valid[line]>0]
            for ordered in (ids,ids[::-1]):
                if len(ordered):
                    labels=(ctx.u[ordered]>0) if independent else _viterbi_second_order(ctx,ordered,library)
                    votes[ordered]+=labels;count[ordered]+=1
    z=np.divide(votes,count,out=np.zeros_like(votes),where=count>0)-.5+ctx.u/4
    z[ctx.valid<=0]=-4
    return z


def _c108(ep,pairwise=False):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C108',{'degenerate':deg[1]})
    ctx=p.context();z=_c108_field(ctx,p,pairwise=pairwise)
    return result(ep,z,'C108',{'directions':4,'state_library_source':'full lawful reference triples',
                             'missing_state':'independent logistic point costs','no_border_role_prior':True})


register('C108',_c108,
         ['The first two line states use independent logistic costs; every later transition uses normalized source triple density.',
          'Both orders of every horizontal/vertical line vote equally; no hard BG border prior.'])


def _source_only(ep):
    p=prepare(ep);deg=p.degenerate()
    z=deg[0] if deg is not None else p.context().u
    return result(ep,z,'control_C_source_NN',{})


CONTROLS['control_C_source_NN']=_source_only
CONTROLS['control_C102_region_mean']=region_method('control_C102_region_mean',lambda c,p:c.base_descriptors()[:,[0]])
CONTROLS['control_C104_independent_halves']=region_method('control_C104_independent_halves',lambda c,p:_c104(c,p)[:,:2])
CONTROLS['control_C104_full_joint_kernel']=lambda ep:_c104_predict(ep,True)
CONTROLS['control_C104_shuffle_half_pairs']=lambda ep:_c104_predict(ep,False,True)
CONTROLS['control_C105_no_permutation']=lambda ep:_c105_predict(ep,False)
CONTROLS['control_C105_pairwise_kernels']=lambda ep:_c105_predict(ep,True,True)
CONTROLS['control_C106_query_prototype_only']=point_method('control_C106_query_prototype_only',lambda c,p:_c106(c,p)[:,[0,1]])
CONTROLS['control_C106_all_anchors']=point_method('control_C106_all_anchors',lambda c,p:_c106(c,p,False))
CONTROLS['control_C106_mean_anchors']=point_method('control_C106_mean_anchors',lambda c,p:_c106(c,p,True,1))
CONTROLS['control_C108_independent_points']=_source_only
CONTROLS['control_C108_pairwise_factorized']=lambda ep:_c108(ep,True)


def _null_descriptor(ctx,p):
    # Full bank SVD, exact numerical rank; no convenient low-rank truncation.
    _,s,vt=np.linalg.svd(ctx.bank,full_matrices=False)
    tolerance=max(ctx.bank.shape)*np.finfo(ctx.bank.dtype).eps*(s[0] if len(s) else 0)
    rank=int(np.sum(s>tolerance));basis=vt[:rank]
    residual=ctx.x-np.einsum('ik,kd->id',dot(ctx.x,basis),basis,optimize=False)
    norm=np.linalg.norm(residual,axis=1)
    if rank>=ctx.x.shape[1]:return np.full((len(ctx.x),4),np.nan)
    W=ctx.W.maximum(__import__('ics.astra300.group_076_150_common',fromlist=['mutual_graph']).mutual_graph(ctx.x))
    i,j=W.nonzero();delta=ctx.profile[i]-ctx.profile[j]
    distance=np.sqrt(np.sum(delta*delta,axis=1))
    from scipy.spatial.distance import pdist
    reference_profiles=dot(ctx.bank[ctx.fg],ctx.bank)
    threshold=float(np.median(pdist(reference_profiles))) if len(reference_profiles)>1 else 0
    allow=distance<=threshold
    rn=unit(residual)
    weights=np.maximum(np.einsum('id,id->i',rn[i],rn[j]),0)
    Wp=sparse.csr_matrix((weights[allow],(i[allow],j[allow])),shape=W.shape)
    zp,_=G(ctx,Wp);zq,_=G(ctx,ctx.W)
    descriptor=np.column_stack((ctx.u,zp,zq,norm));descriptor[norm<=1e-12]=np.nan
    return descriptor


def _c109(ep):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C109',{'degenerate':deg[1]})
    ctx=p.context();_,s,_=np.linalg.svd(ctx.bank,full_matrices=False)
    threshold=max(ctx.bank.shape)*np.finfo(ctx.bank.dtype).eps*s[0]
    rank=int(np.sum(s>threshold))
    if rank>=ctx.x.shape[1]:return result(ep,ctx.u,'C109',{'nullspace_dimension':0,'increment_inactive':'full numerical reference rank'})
    z,info=point_run(ep,_null_descriptor)
    info.update(nullspace_dimension=ctx.x.shape[1]-rank,rank_threshold=threshold)
    return result(ep,z,'C109',info)


register('C109',_c109,
         ['Numerical rank uses the actual double-precision arithmetic dtype after common unit normalization; no low-rank truncation.',
          'Complete-profile Euclidean distances are compared with the exact all-pair pure-reference-FG median; residual edges use their cosine positive part exactly.'])


def _walk(ctx,nonback):
    W=ctx.W.tocsr();degree=np.asarray(W.sum(axis=1)).ravel();ticket=expit(ctx.u)
    if not nonback:
        T=sparse.diags(np.divide(1,degree,out=np.zeros_like(degree),where=degree>0))@W
        probability=ticket.copy()
        for iteration in range(2000):
            nxt=.05*ticket+.95*(T@probability)
            nxt[degree<=0]=ticket[degree<=0];nxt[ctx.positive]=1;nxt[ctx.negative]=0
            if np.max(np.abs(nxt-probability))<=1e-6:return nxt,iteration+1
            probability=nxt
        return ticket,-2000
    source,target=W.nonzero();weight=W.data
    state={(int(i),int(j)):k for k,(i,j) in enumerate(zip(source,target))}
    transition_i=[];transition_j=[];transition_w=[];fallback=0
    for k,(i,j) in enumerate(zip(source,target)):
        neighbors=W.indices[W.indptr[j]:W.indptr[j+1]]
        weights=W.data[W.indptr[j]:W.indptr[j+1]]
        keep=neighbors!=i
        if not np.any(keep):keep=np.ones(len(neighbors),bool);fallback+=1
        norm=weights[keep].sum()
        if norm<=0:continue
        for endpoint,w in zip(neighbors[keep],weights[keep]):
            transition_i.append(k);transition_j.append(state[(int(j),int(endpoint))]);transition_w.append(w/norm)
    transition=sparse.csr_matrix((transition_w,(transition_i,transition_j)),shape=(len(source),len(source)))
    probability=ticket[target].copy()
    for iteration in range(2000):
        nxt=.05*ticket[target]+.95*(transition@probability)
        nxt[ctx.positive[target]]=1;nxt[ctx.negative[target]]=0
        if np.max(np.abs(nxt-probability),initial=0)<=1e-6:
            sums=np.bincount(target,weights=weight*nxt,minlength=len(ctx.x))
            incoming=np.bincount(target,weights=weight,minlength=len(ctx.x))
            return np.divide(sums,incoming,out=ticket.copy(),where=incoming>0),iteration+1
        probability=nxt
    return ticket,-2000


def _c114_descriptor(ctx,p):
    pn,inb=_walk(ctx,True);pw,iw=_walk(ctx,False)
    if inb<0:return np.full((len(ctx.x),3),np.nan)
    return np.column_stack((ctx.u,pn,pw))


register('C114',point_method('C114',_c114_descriptor),
         ['State (i→j) is absorbed on arrival at j; isolated points keep their initial soft vote.',
          'Both ordinary and nonbacktracking walk use exactly .05 point-vote restart.'])
CONTROLS['control_C114_ordinary_walk']=point_method('control_C114_ordinary_walk',lambda c,p:np.column_stack((c.u,_walk(c,False)[0])))


def _edge_descriptors(ctx,i,j):
    cosine=np.einsum('id,id->i',ctx.x[i],ctx.x[j]);jump=1-cosine
    rank=np.empty(len(i))
    for node in np.unique(i):
        at=np.flatnonzero(i==node)
        order=np.argsort(jump[at],kind='stable');r=np.empty(len(at));r[order]=np.arange(len(at))/max(len(at)-1,1)
        rank[at]=r
    return np.column_stack((ctx.u[i],ctx.u[j],cosine,rank))


class _PairStates:
    def __init__(self,descriptors,soft,emission_columns=(0,1)):
        self.model=Kernel(descriptors,np.full(len(descriptors),.5))
        self.states=np.array(((1,1),(1,0),(0,1),(0,0)))
        self.emission_columns=emission_columns
        self.weights=np.prod(np.where(self.states[None]>0,soft[:,None],1-soft[:,None]),axis=2)
    def logs(self,d):
        z=self.model.standardize(d);out=np.zeros((len(d),4))
        for start in range(0,len(d),128):
            delta=z[start:start+128,None]-self.model.d[None]
            logk=-np.sum(delta*delta,axis=2)/(2*self.model.sigma**2)
            for state in range(4):
                w=self.weights[:,state]
                if w.sum()>0:
                    lw=np.log(w,where=w>0,out=np.full(len(w),-np.inf))
                    out[start:start+128,state]=logsumexp(logk+lw,axis=1)-np.log(w.sum())
                else:
                    u=np.asarray(d)[start:start+128][:,self.emission_columns]
                    out[start:start+128,state]=-np.sum(np.logaddexp(0,u)-u*self.states[state],axis=1)
        return out


def _pair_library(ctx,p):
    rc=p.context(True,ctx.fold);i,j=rc.W.nonzero()
    if ctx.fold is not None:
        keep=(p.rb[i]!=ctx.fold)&(p.rb[j]!=ctx.fold);i,j=i[keep],j[keep]
    return _PairStates(_edge_descriptors(rc,i,j),p.c[np.column_stack((i,j))])


def _c116_descriptor(ctx,p,symmetric=False):
    i,j=ctx.W.nonzero();library=_pair_library(ctx,p)
    logs=library.logs(_edge_descriptors(ctx,i,j))
    weights=(np.maximum(logs[:,0]-logs[:,1],0),np.maximum(logs[:,3]-logs[:,2],0))
    signals=[];asym=np.zeros(len(ctx.x))
    for side,weight in enumerate(weights):
        T=sparse.csr_matrix((weight,(i,j)),shape=ctx.W.shape)
        asym+=np.asarray(abs(T-T.T).sum(axis=1)).ravel()
        if symmetric:T=(T+T.T)/2
        degree=np.asarray(T.sum(axis=1)).ravel()
        T=sparse.diags(np.divide(1,degree,out=np.zeros_like(degree),where=degree>0))@T
        initial=expit(ctx.u) if side==0 else 1-expit(ctx.u);field=initial.copy()
        for _ in range(20):field=.5*initial+.5*(T.T@field)
        signals.append(field)
    descriptor=np.column_stack((ctx.u,signals[0]-signals[1],asym))
    descriptor[np.asarray(ctx.W.sum(axis=1)).ravel()<=0]=np.nan
    return descriptor


register('C116',point_method('C116',_c116_descriptor),
         ['Edge jump rank is the ordered 1-cosine rank among outgoing spatial edges.',
          'Directed positive log-density ratios are row-normalized before twenty mass-propagation steps; each retains .5 of its original ticket.'])
CONTROLS['control_C116_symmetric']=point_method('control_C116_symmetric',lambda c,p:_c116_descriptor(c,p,True))


def _path_cost(ctx):
    W=ctx.W.tocsr();i,j=W.nonzero();w=np.asarray(W[i,j]).ravel()
    return sparse.csr_matrix((-np.log(np.maximum(w,1e-12))+np.maximum(ctx.u[j],0),(i,j)),shape=W.shape)


def _c117_descriptor(ctx,p,direct=False):
    from scipy.sparse.csgraph import dijkstra
    cost=_path_cost(ctx);evidence=[];distances=[]
    for atom in ctx.atoms:
        anchors=atom[ctx.negative[atom]&~ctx.broad[atom]]
        if not len(anchors):continue
        dist,_,source=dijkstra(cost,directed=True,indices=anchors,min_only=True,return_predecessors=True)
        valid=source>=0;score=np.zeros(len(ctx.x))
        if np.any(valid):score[valid]=np.einsum('id,id->i',ctx.x[valid],ctx.x[source[valid]])*np.exp(-dist[valid])
        distances.append(dist);evidence.append(score)
    negative=np.zeros(len(ctx.x))
    if len(evidence):
        distance=np.array(distances);scores=np.array(evidence)
        chosen=np.argsort(distance,axis=0,kind='stable')[:4]
        negative=np.take_along_axis(scores,chosen,axis=0).max(axis=0)
    if direct:
        bank=np.flatnonzero(ctx.negative&~ctx.broad)
        negative=dot(ctx.x,ctx.x[bank]).max(axis=1)if len(bank)else np.zeros(len(ctx.x))
    positive=np.concatenate([a for a in ctx.atoms if np.any(ctx.positive[a])])if any(np.any(ctx.positive[a])for a in ctx.atoms)else np.empty(0,int)
    nearest=(dijkstra(cost,directed=True,indices=positive,min_only=True) if len(positive) else np.full(len(ctx.x),np.nan))
    nearest[~np.isfinite(nearest)]=np.nan
    return np.column_stack((ctx.u,negative,nearest))


register('C117',point_method('C117',_c117_descriptor),
         ['The four retained negative sources are distinct query atoms; within each atom a multi-source shortest path selects its actual minimum-cost anchor.',
          'Nearest positive distance uses the same directed -log(w)+positive-u transport cost.'])
CONTROLS['control_C117_direct_negative']=point_method('control_C117_direct_negative',lambda c,p:_c117_descriptor(c,p,True))


def _c119_descriptor(ctx,p,sum_paths=False):
    prototypes=ctx.role_prototypes(16)[0]
    if not len(prototypes):return np.full((len(ctx.x),4),np.nan)
    nearest=np.argmax(dot(ctx.x,prototypes),axis=1);messages=[]
    W=ctx.W.tocsr();i,j=W.nonzero();weight=np.asarray(W[i,j]).ravel()
    weight=weight* np.where(ctx.negative[j],expit(-4.),1.)
    for root,atom in enumerate(ctx.atoms):
        anchors=atom[ctx.positive[atom]]
        for mode in np.unique(nearest[anchors]):
            group=anchors[nearest[anchors]==mode]
            m=np.zeros(len(ctx.x));m[group]=np.maximum(ctx.u[group],0)
            for _ in range(20):
                new=m.copy()
                if sum_paths:
                    new+=np.bincount(j,weights=weight*m[i],minlength=len(m))
                else:np.maximum.at(new,j,weight*m[i])
                m=new
            messages.append(m)
    if not messages:return np.full((len(ctx.x),4),np.nan)
    values=np.array(messages);count=np.count_nonzero(values>0,axis=0)
    maximum=values.max(axis=0)
    # Explicit bounded/truncated sum: clamp every per-source ticket at 4.
    average=np.divide(np.clip(values,0,4).sum(axis=0),count,out=np.zeros(len(ctx.x)),where=count>0)
    descriptor=np.column_stack((ctx.u,average,maximum,count));descriptor[count==0]=np.nan
    return descriptor


register('C119',point_method('C119',_c119_descriptor),
         ['Messages retain (reference FG mode, query atom root) as their exact source key; a negative anchor multiplies incoming passage by sigmoid(-4).',
          'Truncated sum clips each source ticket to [0,4], then divides by its number of nonzero distinct sources.'])
CONTROLS['control_C119_sum_paths']=point_method('control_C119_sum_paths',lambda c,p:_c119_descriptor(c,p,True))
CONTROLS['control_C119_max_source']=point_method('control_C119_max_source',lambda c,p:_c119_descriptor(c,p)[:,[0,2,3]])


def _rank(ctx,ids):
    if not len(ids) or not len(ctx.fg) or not len(ctx.bg):return np.nan
    profile=ctx.profile[ids].mean(axis=0)
    difference=profile[ctx.fg,None]-profile[None,ctx.bg]
    return float(np.mean(difference>0)+.5*np.mean(difference==0))


def _reference_bg_buckets(ctx,p):
    rc=p.context(True,ctx.fold);bucket={}
    rb=blocks(rc.hw)
    for c in rc.regions:
        if ctx.fold is not None and np.any(p.rb[c]==ctx.fold):continue
        if np.all(p.c[c]<=.1):
            key=(len(c),len(np.unique(rb[c])))
            bucket.setdefault(key,[]).append(float(rc.u[c].sum()))
    return bucket


def _c127_descriptor(ctx,p):
    buckets=_reference_bg_buckets(ctx,p);qb=blocks(ctx.hw);out=[]
    for c in ctx.regions:
        sample=buckets.get((len(c),len(np.unique(qb[c]))),[])
        if np.any(ctx.positive[c]) or len(sample)<4:out.append([np.nan]*4);continue
        mu=np.mean(sample);variance=np.var(sample,ddof=1)
        z=(ctx.u[c].sum()-mu)/np.sqrt(max(variance,1e-12))
        positive_blocks=sum(ctx.u[c[qb[c]==block]].mean()>0 for block in np.unique(qb[c]))
        out.append([z,ctx.u[c].mean(),np.mean(ctx.negative[c]),positive_blocks])
    return np.asarray(out)


register('C127',region_method('C127',_c127_descriptor),
         ['Reference BG buckets match exact native candidate point count and exact number of occupied public 4×4 blocks; variance uses ddof=1.'])
CONTROLS['control_C127_unstandardized_sum']=region_method('control_C127_unstandardized_sum',lambda c,p:np.array([[c.u[a].sum(),len(a)]for a in c.regions]))


def _negative_objects(ctx):
    qb=blocks(ctx.hw);objects=[]
    for c in ctx.regions:
        if np.any(ctx.positive[c]):continue
        bs=np.unique(qb[c[ctx.negative[c]]])
        if any(abs(int(a)//4-int(b)//4)+abs(int(a)%4-int(b)%4)>1 for a,b in itertools.combinations(bs,2)):
            objects.append(c)
    return objects


def _c128_fields(ctx,p,overlap_exclusion=True):
    objects=_negative_objects(ctx)
    negative_profile=unit(np.array([ctx.profile[c].mean(axis=0) for c in objects])) if objects else np.empty((0,len(ctx.bank)))
    positives=unit(dot(ctx.bank[ctx.fg],ctx.bank))
    profile=unit(ctx.profile);positive=dot(profile,positives).max(axis=1)
    fields=[]
    for c in ctx.regions:
        keep=np.array([not np.intersect1d(c,b,assume_unique=True).size for b in objects],bool) if overlap_exclusion else np.ones(len(objects),bool)
        if keep.any():
            bg=dot(profile[c],negative_profile[keep]).max(axis=1)
            fields.append(np.column_stack((ctx.u[c],positive[c]-bg,ctx.max_f[c]-ctx.max_b[c],np.full(len(c),1-keep.mean()))))
        else:
            fields.append(np.full((len(c),4),np.nan))
    return ctx.regions,fields


register('C128',region_point_method('C128',_c128_fields),
         ['Negative objects and reference FG are compared in normalized complete-reference-response space.',
          'All candidate-conditioned point fields apply K before the common inverse-area RP; an empty candidate-conditioned negative bank uses its source-only u.'])
CONTROLS['control_C128_no_overlap_exclusion']=region_point_method('control_C128_no_overlap_exclusion',lambda c,p:_c128_fields(c,p,False))


def _perimeter(ctx,c):
    mask=np.zeros(ctx.hw,bool);mask.ravel()[c]=True
    area=len(c)
    internal=np.sum(mask[1:]&mask[:-1])+np.sum(mask[:,1:]&mask[:,:-1])
    return 4*area-2*int(internal)


def _c129_descriptor(ctx,p):
    from scipy.ndimage import label
    negative=ctx.negative.reshape(ctx.hw)
    labels,count=label(negative,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    table={};h,w=ctx.hw
    # All legal sliding axis-aligned power-of-two windows, no sampling/cap.
    sizes=set()
    for exponent in range(int(np.log2(max(h,w)))+1):
        s=2**exponent
        for height,width in ((s,s),(s,2*s),(2*s,s)):
            if height<=h and width<=w:sizes.add((height,width))
    for height,width in sorted(sizes):
        for y in range(h-height+1):
            for x in range(w-width+1):
                patch=labels[y:y+height,x:x+width]
                if patch.flat[0]<=0 or not np.all(patch==patch.flat[0]):continue
                area=height*width;ratio=2*(height+width)/area
                key=(int(np.floor(np.log2(area))),int(np.floor(np.log2(ratio))))
                table.setdefault(key,[]).append(ctx.u.reshape(ctx.hw)[y:y+height,x:x+width].mean())
    proto=ctx.role_prototypes(16)[0]
    mode=np.argmax(dot(ctx.x,proto),axis=1) if len(proto) else np.full(len(ctx.x),-1)
    out=[]
    for c in ctx.regions:
        key=(int(np.floor(np.log2(len(c)))),int(np.floor(np.log2(_perimeter(ctx,c)/len(c)))))
        sample=table.get(key,[])
        if len(sample)<8:out.append([np.nan]*3);continue
        mean=ctx.u[c].mean();tail=np.mean(np.asarray(sample)<=mean)
        modes=len(np.unique(mode[c[ctx.u[c]>0]]))
        out.append([tail,mean,modes])
    return np.asarray(out)


register('C129',region_method('C129',_c129_descriptor),
         ['Area and perimeter/area use floor(log2) buckets; source FG-mode count uses the positive-u members and fixed public reference modes.'])
CONTROLS['control_C129_area_mean']=region_method('control_C129_area_mean',lambda c,p:np.array([[c.u[a].mean(),len(a),_perimeter(c,a)/len(a)]for a in c.regions]))


def _c138_descriptor(ctx,p,penalty=.1):
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh
    W=ctx.W;degree=np.asarray(W.sum(axis=1)).ravel();valid=degree>0
    z=ctx.u.copy();high=np.abs(ctx.u)
    ids=np.flatnonzero(valid)
    if len(ids)>1:
        g=W[ids][:,ids];d=degree[ids]
        inv=sparse.diags(1/np.sqrt(d));L=sparse.eye(len(ids))-inv@g@inv
        components=connected_components(g,directed=False,return_labels=False)
        count=min(32+components,len(ids)-1)
        if len(ids)<=128:values,V=np.linalg.eigh(L.toarray())
        else:values,V=eigsh(L,k=count,which='SM',tol=1e-8,v0=np.ones(len(ids)))
        order=np.argsort(values);V=V[:,order][:,values[order]>1e-10][:,:min(32,len(ids)-1)]
        if V.shape[1]:
            u=ctx.u[ids];a=np.zeros(V.shape[1]);residual=u.copy()
            for _ in range(100):
                for k in range(V.shape[1]):
                    residual+=V[:,k]*a[k];raw=V[:,k]@residual
                    a[k]=np.sign(raw)*max(abs(raw)-penalty*len(ids)/2,0)
                    residual-=V[:,k]*a[k]
            coefficients=np.einsum('ik,i->k',V,u);h=u-np.einsum('ik,k->i',V,coefficients)
            z[ids]=np.einsum('ik,k->i',V,a)+h;high[ids]=np.abs(h)
    return np.column_stack((ctx.u,z,high))


register('C138',point_method('C138',_c138_descriptor),
         ['All zero Laplacian modes are nontriviality-excluded; isolated points preserve u; coordinate descent starts at zero.'])
CONTROLS['control_C138_spectral_LS']=point_method('control_C138_spectral_LS',lambda c,p:_c138_descriptor(c,p,0.))
CONTROLS['control_C138_graph_G']=point_method('control_C138_graph_G',lambda c,p:np.column_stack((c.u,G(c)[0])))


def _c147_descriptor(ctx,p):
    qb=blocks(ctx.hw);out=[]
    for c in ctx.regions:
        bs=np.unique(qb[c])
        if len(bs)<3:out.append([np.nan]*4);continue
        means=[];ranks=[];size=[]
        for b in bs:
            kept=c[qb[c]!=b];means.append(ctx.u[kept].mean());ranks.append(_rank(ctx,kept));size.append(len(kept))
        original=ctx.u[c].mean()
        out.append([min(ranks),min(means),max(original-np.asarray(means)),min(size)])
    return np.asarray(out)


register('C147',region_method('C147',_c147_descriptor),
         ['Complete-profile FG/BG rank is the class-balanced pairwise response ordering probability (ties .5); all 16 absolute spatial blocks are considered.'])
CONTROLS['control_C147_trimmed_mean']=region_method('control_C147_trimmed_mean',lambda c,p:np.array([[c.u[a].mean(),np.sort(c.u[a])[:max(1,int(.9*len(a)))].mean(),len(a)]for a in c.regions]))


def _c148_fields(ctx,p,split_outliers=True):
    from scipy.ndimage import label
    fields=[];regions=[]
    for c in ctx.regions:
        center=unit(ctx.x[c].mean(axis=0));distance=1-ctx.x[c]@center
        selected=c[distance>np.median(distance)] if split_outliers else c
        mask=np.zeros(ctx.hw,bool);mask.ravel()[selected]=True
        groups,count=label(mask,np.array([[0,1,0],[1,1,1],[0,1,0]]))
        for group in range(1,count+1):
            ids=np.flatnonzero(groups.ravel()==group)
            if _rank(ctx,ids)<=.5 or np.any(ctx.negative[ids]):continue
            shell=_ring(ctx,ids,1)
            if not len(shell):continue
            v=unit(ctx.x[ids].mean(axis=0))
            added=ctx.x[shell]@v
            margin=(np.maximum(ctx.max_f[shell],added)-ctx.max_b[shell])/ctx.scale
            fields.append(np.column_stack((ctx.u[shell],np.clip(margin,-4,4),np.full(len(shell),len(ids)/len(c)))))
            regions.append(shell)
    return regions,fields


register('C148',region_point_method('C148',_c148_fields),
         ['Positive complete-reference rank means class-balanced ordering >.5; any strong BG member vetoes that source face.',
          'Only each qualified face one-grid shell receives the extra frozen prototype; each shell applies K before common inverse-area RP.'])
CONTROLS['control_C148_all_local_faces']=region_point_method('control_C148_all_local_faces',lambda c,p:_c148_fields(c,p,False))


def _c111_descriptor(ctx,p,pair_interaction=True):
    roots=[a[ctx.positive[a]]for a in ctx.atoms if np.any(ctx.positive[a])]
    if not roots:return np.full((len(ctx.x),5),np.nan)
    baseline,info=G(ctx,hard=True)
    if info:return np.full((len(ctx.x),5),np.nan)
    single=[]
    for root in roots:
        positive=ctx.positive.copy();positive[root]=False
        changed,info=G(ctx,positive=positive,hard=True)
        if info:return np.full((len(ctx.x),5),np.nan)
        single.append(baseline-changed)
    maximum_single=np.max(single,axis=0);maximum_pair=np.zeros(len(ctx.x));interaction=np.zeros(len(ctx.x))
    if pair_interaction:
        for a,b in itertools.combinations(range(len(roots)),2):
            positive=ctx.positive.copy();positive[roots[a]]=False;positive[roots[b]]=False
            changed,info=G(ctx,positive=positive,hard=True)
            if info:return np.full((len(ctx.x),5),np.nan)
            delta=baseline-changed
            maximum_pair=np.maximum(maximum_pair,delta)
            interaction=np.maximum(interaction,np.abs(delta-single[a]-single[b]))
    return np.column_stack((ctx.u,baseline,maximum_single,maximum_pair,interaction))


register('C111',point_method('C111',_c111_descriptor),
         ['Root deletion influence is baseline hard-Dirichlet field minus the newly solved free-boundary field; all root pairs are enumerated.'])
CONTROLS['control_C111_single_roots']=point_method('control_C111_single_roots',lambda c,p:_c111_descriptor(c,p,False)[:,:3])


def _disjoint_cost(ctx,target,foreground,required=2):
    from .group_076_150_flow import MinCostFlow
    anchors=ctx.positive if foreground else ctx.negative
    blocked=ctx.negative if foreground else ctx.positive
    roots=[a[anchors[a]]for a in ctx.atoms if np.any(anchors[a])]
    if len(roots)<required or blocked[target] or ctx.valid[target]<=0:return np.inf
    n=len(ctx.x);source=2*n+len(roots);sink=2*target+1
    flow=MinCostFlow(source+1)
    for i in range(n):
        if ctx.valid[i]>0 and not blocked[i]:flow.add(2*i,2*i+1,required if i==target else 1,0)
    ii,jj=ctx.W.nonzero();ww=np.asarray(ctx.W[ii,jj]).ravel()
    for i,j,w in zip(ii,jj,ww):
        if not blocked[i] and not blocked[j]:flow.add(2*int(i)+1,2*int(j),required,-np.log(max(w,1e-12)))
    for k,root in enumerate(roots):
        node=2*n+k;flow.add(source,node,1,0)
        for i in root:
            if not blocked[i]:flow.add(node,2*int(i),1,0)
    received,cost=flow.solve(source,sink,required)
    return cost if received==required else np.inf


def _c112_descriptor(ctx,p,required=2):
    n=len(ctx.x);fg=np.array([_disjoint_cost(ctx,i,True,required)for i in range(n)])
    bg=np.array([_disjoint_cost(ctx,i,False,required)for i in range(n)])
    sf=np.array([_disjoint_cost(ctx,i,True,1)for i in range(n)])
    sb=np.array([_disjoint_cost(ctx,i,False,1)for i in range(n)])
    dual=np.full(n,np.nan);single=np.full(n,np.nan)
    both=np.isfinite(fg)&np.isfinite(bg);dual[both]=bg[both]-fg[both]
    one=np.isfinite(sf)&np.isfinite(sb);single[one]=sb[one]-sf[one]
    return np.column_stack((ctx.u,dual,np.isfinite(fg),np.isfinite(bg),single))


register('C112',point_method('C112',_c112_descriptor),
         ['Every vertex except the tested sink has capacity one; atom source groups also have capacity one, enforcing two different roots.',
          'Unavailable class routes are missing costs plus explicit existence flags, never zero cost.'])
CONTROLS['control_C112_single_paths']=point_method('control_C112_single_paths',lambda c,p:_c112_descriptor(c,p,1))


def _c141_descriptor(ctx,p,single_label=False):
    from scipy.sparse.csgraph import connected_components,dijkstra
    from .group_076_150_common import mutual_graph
    collision=mutual_graph(ctx.profile,20,profile=True)
    _,label=connected_components(collision,directed=False)
    out=np.column_stack((ctx.u,np.full((len(ctx.x),3),np.nan)))
    spatial=ctx.W.copy()
    for component in np.unique(label):
        ids=np.flatnonzero(label==component)
        positive=ids[ctx.positive[ids]];negative=ids[ctx.negative[ids]]
        if not len(positive)or not len(negative):continue
        local=np.full(len(ctx.x),-1,int);local[ids]=np.arange(len(ids))
        graph=spatial[ids][:,ids];ii,jj=graph.nonzero();weights=np.asarray(graph[ii,jj]).ravel()
        cost=sparse.csr_matrix((-np.log(np.maximum(weights,1e-12)),(ii,jj)),shape=graph.shape)
        df=dijkstra(cost,directed=False,indices=local[positive],min_only=True)
        db=dijkstra(cost,directed=False,indices=local[negative],min_only=True)
        difference=np.full(len(ids),np.nan);finite=np.isfinite(df)&np.isfinite(db);difference[finite]=db[finite]-df[finite]
        # Four-neighbor steps until leaving the collision component.
        i,j=edges4(ctx.hw,ctx.valid);boundary=np.unique(np.r_[i[(label[i]==component)&(label[j]!=component)],j[(label[j]==component)&(label[i]!=component)]])
        hop=sparse.csr_matrix((np.ones(len(ii)),(ii,jj)),shape=graph.shape)
        exit_distance=dijkstra(hop,directed=False,indices=local[boundary],min_only=True) if len(boundary) else np.full(len(ids),np.nan)
        exit_distance[~np.isfinite(exit_distance)]=np.nan
        if single_label:difference[:]=np.nanmean(difference) if np.isfinite(difference).any()else 0
        out[ids,1:]=np.column_stack((difference,exit_distance,np.full(len(ids),(len(positive)+len(negative))/len(ids))))
    return out


register('C141',point_method('C141',_c141_descriptor),
         ['Collision graph uses exact complete-profile Euclidean mutual-20; within-component paths use spatial -log(w) and exit distance uses unweighted four-neighbor hops.'])
CONTROLS['control_C141_one_label_per_collision']=point_method('control_C141_one_label_per_collision',lambda c,p:_c141_descriptor(c,p,True))


def _c142_rho(ctx,p,fixed=None):
    if fixed is not None:return None
    rc=p.context(True,ctx.fold);grid=np.arange(len(rc.x)).reshape(rc.hw);d=[];coverage=[]
    for y in range(0,rc.hw[0],2):
        for x in range(0,rc.hw[1],2):
            ids=grid[y:y+2,x:x+2].ravel();ids=ids[rc.valid[ids]>0]
            if not len(ids) or (ctx.fold is not None and np.any(p.rb[ids]==ctx.fold)):continue
            u=rc.u[ids];d.append([u.mean(),u.min(),u.max(),np.var(rc.profile[ids],axis=0).mean()])
            c=p.c[ids];coverage.append(float(np.prod(c)+np.prod(1-c)))
    return Kernel(np.asarray(d).reshape(-1,4),np.asarray(coverage))


def _c142(ep,fixed_rho=None):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C142',{'degenerate':deg[1]})
    ctx=p.context();grid=np.arange(len(ctx.x)).reshape(ctx.hw);groups=[];rho=[]
    model=_c142_rho(ctx,p,fixed_rho)
    for y in range(0,ctx.hw[0],2):
        for x in range(0,ctx.hw[1],2):
            ids=grid[y:y+2,x:x+2].ravel();ids=ids[ctx.valid[ids]>0]
            if not len(ids):continue
            u=ctx.u[ids];d=[u.mean(),u.min(),u.max(),np.var(ctx.profile[ids],axis=0).mean()]
            value=fixed_rho if fixed_rho is not None else max(float(model(d)[0]),0)
            groups.append(ids);rho.append(value)
    ii,jj=ctx.W.nonzero();weights=np.asarray(ctx.W[ii,jj]).ravel()
    incident=[]
    for ids in groups:
        at=np.isin(ii,ids)|np.isin(jj,ids)
        incident.append((ii[at],jj[at],weights[at]))
    def energy(labels,coarse):
        value=-ctx.u@labels+.5*np.sum(weights*np.abs(labels[ii]-labels[jj]))
        for ids,r,c in zip(groups,rho,coarse):value+=r*min(np.sum(np.abs(labels[ids]-c)),2)
        return float(value)
    outcomes=[]
    for start in (np.zeros(len(ctx.x),int),np.ones(len(ctx.x),int),(ctx.u>0).astype(int)):
        labels=start.copy();labels[ctx.valid<=0]=0
        coarse=np.array([int(labels[g].mean()>.5)for g in groups])
        for _ in range(20):
            for k,ids in enumerate(groups):
                best=None
                ei,ej,ew=incident[k]
                base=labels.copy()
                for bits in itertools.product((0,1),repeat=len(ids)+1):
                    labels[ids]=bits[:-1]
                    local=-ctx.u[ids]@labels[ids]+.5*np.sum(ew*np.abs(labels[ei]-labels[ej]))+rho[k]*min(np.sum(np.abs(labels[ids]-bits[-1])),2)
                    item=(float(local),int(np.sum(labels[ids]!=(ctx.u[ids]>0))),bits)
                    if best is None or item<best[0]:best=(item,bits)
                labels[ids]=best[1][:-1];coarse[k]=best[1][-1]
        outcomes.append((energy(labels,coarse),int(np.sum(labels!=(ctx.u>0))),tuple(labels.tolist()),labels))
    best=min(outcomes,key=lambda a:a[:3]);z=2*best[3]-1
    return result(ep,z,'C142',{'energy':best[0],'optimizer':'20 row-order block-ICM rounds / 3 starts','optimization_gap':'unknown','coarse_block_count':len(groups)})


register('C142',_c142,
         ['Coarse descriptors are [mean u,min u,max u,mean complete-profile variance]; same-class soft source weight is product(c)+product(1-c).',
          'Block ICM enumerates all coarse+fine assignments; tie-break is complete energy, total S0 edits, then global row-order labels.'])
CONTROLS['control_C142_fixed_robust_Potts']=lambda ep:_c142(ep,1.)


def _base_descriptor(ctx,c):
    u=ctx.u[c]
    return [u.mean(),np.median(u),u.min(),u.max(),ctx.max_f[c].mean(),ctx.max_b[c].mean()]


def _subcontext(ctx,ids):
    from dataclasses import replace
    return replace(ctx,x=ctx.x[ids],hw=(len(ids),1),valid=ctx.valid[ids],profile=ctx.profile[ids],
                   max_f=ctx.max_f[ids],max_b=ctx.max_b[ids],u=ctx.u[ids],
                   positive=ctx.positive[ids],negative=ctx.negative[ids],W=ctx.W[ids][:,ids])


def _c146_samples(ctx,p,recursive=True):
    from .group_076_150_flow import cut
    qb=blocks(ctx.hw);pending=[np.flatnonzero(ctx.valid>0)];terminal=[]
    while pending:
        c=pending.pop(0)
        fg=np.unique(qb[c[ctx.positive[c]]]);bg=np.unique(qb[c[ctx.negative[c]]])
        if len(fg)<2 or len(bg)<2:terminal.append(c);continue
        y,_,_=cut(ctx.u[c],ctx.W[c][:,c]);a,b=c[y],c[~y]
        if not len(a)or not len(b):terminal.append(c);continue
        if recursive:pending.extend((a,b))
        else:terminal.extend((a,b))
    d=np.array([[ctx.u[c].mean(),ctx.u[c].max(),ctx.u[c].min(),_rank(ctx,c)]for c in terminal]).reshape(-1,4)
    return terminal,d


def _c146(ep):
    z,info=dynamic_region_run(ep,_c146_samples)
    return result(ep,z,'C146',info)


register('C146',_c146,
         ['Fixed positive/negative public anchors trigger recursive spatial cuts only after each covers at least two distinct absolute 4×4 blocks; endpoint capacities use positive/negative u.'])
CONTROLS['control_C146_one_cut']=lambda ep:result(ep,*dynamic_region_run(ep,lambda c,p:_c146_samples(c,p,False))[:1],'control_C146_one_cut',{})


def _c130_samples(ctx,p,adaptive=True):
    qb=blocks(ctx.hw)
    K0=p.k0_for(ctx)
    def error(c):
        total=0.;count=0
        for block in np.unique(qb[c]):
            train=c[qb[c]!=block];test=c[qb[c]==block]
            if not len(train)or not len(test):continue
            label=bool(K0(_base_descriptor(ctx,train))[0]>0)
            total+=np.sum((ctx.u[test]>0)!=label);count+=len(test)
        return total/count if count else np.inf
    terminal=[a.copy()for a in ctx.atoms]
    if adaptive:
        while True:
            proposals=[]
            for k,c in enumerate(terminal):
                if len(c)<2 or not(ctx.u[c].min()<0<ctx.u[c].max()):continue
                profile=ctx.profile[c];center=profile-profile.mean(axis=0)
                _,s,v=np.linalg.svd(center,full_matrices=False)
                if not len(s)or s[0]<=1e-12:continue
                direction=v[0]
                if direction[np.argmax(np.abs(direction))]<0:direction=-direction
                projection=center@direction;order=np.lexsort((c,projection));split=(len(c)+1)//2
                a,b=c[order[:split]],c[order[split:]]
                if not len(b):continue
                labels=[K0(_base_descriptor(ctx,g))[0]>0 for g in(a,b)]
                if labels[0]==labels[1]:continue
                old=error(c);new=(len(a)*error(a)+len(b)*error(b))/len(c)
                decline=old-new
                if np.isfinite(decline)and decline>0:proposals.append((decline,float(s[0]**2/max(len(c),1)),-int(c.min()),k,a,b))
            if not proposals:break
            best=max(proposals,key=lambda a:a[:3]);terminal[best[3]:best[3]+1]=[best[4],best[5]]
    d=np.array([[ctx.u[c].mean(),ctx.u[c].max(),ctx.u[c].min(),np.mean(np.linalg.norm(ctx.profile[c]-ctx.profile[c].mean(axis=0),axis=1))]for c in terminal]).reshape(-1,4)
    return terminal,d


def _c130(ep):
    z,info=dynamic_region_run(ep,_c130_samples)
    return result(ep,z,'C130',info)


register('C130',_c130,
         ['Conflict proxy uses leave-one-absolute-4×4-block K0 predictions against frozen original u signs; accepted split must reduce the weighted held-block error and predict both classes.',
          'Proposals are selected by largest error decline, then largest complete-profile principal variance, then smallest row ID.'])
CONTROLS['control_C130_static_atoms']=lambda ep:result(ep,*dynamic_region_run(ep,lambda c,p:_c130_samples(c,p,False))[:1],'control_C130_static_atoms',{})


def _c122_samples(ctx,p,remove_all=False):
    regions=[];rows=[]
    shared=np.zeros(len(ctx.x),bool)
    if remove_all:
        count=np.zeros(len(ctx.x),int)
        for c in ctx.regions:count[c]+=1
        shared=count>1
    for a in range(len(ctx.regions)):
        C=ctx.regions[a]
        for b in range(a+1,len(ctx.regions)):
            D=ctx.regions[b];H=np.intersect1d(C,D,assume_unique=True)
            if not len(H):continue
            for container in(C,D):
                exclusive=np.setdiff1d(container,H,assume_unique=True)
                if not len(exclusive):continue
                local=_subcontext(ctx,container);baseline,info=G(local)
                positive=local.positive.copy();negative=local.negative.copy()
                unset=np.isin(container,H)if not remove_all else shared[container]
                positive[unset]=False;negative[unset]=False
                changed,other=G(local,positive=positive,negative=negative)
                if info or other:continue
                at=np.isin(container,exclusive)
                regions.append(exclusive)
                rows.append([ctx.u[exclusive].mean(),np.abs(ctx.u[exclusive]).mean(),(baseline[at]-changed[at]).mean(),np.max(np.abs(baseline[at]-changed[at]))])
    return regions,np.asarray(rows).reshape(-1,4)


def _c122(ep):
    z,info=dynamic_region_run(ep,_c122_samples)
    return result(ep,z,'C122',info)


register('C122',_c122,
         ['Each side is corrected only on C\\H / D\\H; signed mean influence and absolute maximum influence summarize the keep/remove-shared-anchor local solves.',
          'All overlapping candidate pairs are enumerated without capping; shared H retains its original u.'])
CONTROLS['control_C122_remove_all_shared_once']=lambda ep:result(ep,*dynamic_region_run(ep,lambda c,p:_c122_samples(c,p,True))[:1],'control_C122_remove_all_shared_once',{})


def _c123(ep,recursive=True):
    from scipy.sparse.linalg import eigsh
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C123',{'degenerate':deg[1]})
    ctx=p.context();accepted=np.zeros(len(ctx.x),bool);iterations=0;solves=0
    for _ in range(len(ctx.x)):
        ids=np.flatnonzero((ctx.valid>0)&~accepted)
        if len(ids)<2:break
        W=ctx.W[ids][:,ids];degree=np.asarray(W.sum(axis=1)).ravel();active=degree>0
        ids=ids[active];W=W[active][:,active];degree=degree[active]
        if len(ids)<2:break
        inv=sparse.diags(1/np.sqrt(degree));L=sparse.eye(len(ids))-inv@W@inv
        if len(ids)<=128:values,V=np.linalg.eigh(L.toarray())
        else:values,V=eigsh(L,k=min(5,len(ids)-1),which='SM',tol=1e-8,v0=np.ones(len(ids)))
        order=np.argsort(values);V=V[:,order][:,values[order]>1e-10][:,:4];solves+=1
        newly=np.zeros(len(ctx.x),bool)
        for vector in V.T:
            for sign in(-1,1):
                c=ids[sign*vector>0]
                if len(c)and p.k0(_base_descriptor(ctx,c))[0]>0:newly[c[ctx.u[c]>=ctx.weak_threshold]]=True
        newly&=~accepted
        if not newly.any():break
        accepted|=newly;iterations+=1
        if not recursive:break
    mask=accepted|(ctx.u>0);mask[ctx.valid<=0]=False
    return result(ep,2*mask.astype(float)-1,'C123',{'accepted_count':int(accepted.sum()),'iterations':iterations,'eigensolves':solves,'maximum_iterations':len(ctx.x)})


register('C123',_c123,
         ['Normalized positive-weight Laplacian uses the first four strictly nonzero modes; accepted visible points are frozen and deleted only from subsequent discovery graphs.'])
CONTROLS['control_C123_once_spectrum']=lambda ep:_c123(ep,False)


def _em_class(ctx,train,k):
    positives=train[ctx.u[train]>0]
    if not len(positives):return None
    fg=unit(ctx.bank[ctx.fg].mean(axis=0));bg=unit(ctx.bank[ctx.bg].mean(axis=0))
    if k==1:centers=unit(ctx.x[positives].mean(axis=0))[None]
    else:centers=spherical(ctx.x[positives],min(k,len(positives)),iterations=0)[0]
    if len(centers)!=k:return None
    for _ in range(5):
        log_fg=dot(ctx.x[train],centers)/.1-np.log(k);log_bg=ctx.x[train]@bg/.1
        total=logsumexp(np.column_stack((log_fg,log_bg)),axis=1)
        responsibilities=np.exp(log_fg-total[:,None])
        centers=unit(np.einsum('ik,id->kd',responsibilities,ctx.x[train])+fg[None])
    return centers,bg


def _em_score(ctx,ids,model):
    centers,bg=model
    return logsumexp(dot(ctx.x[ids],centers)/.1,axis=1)-np.log(len(centers))-ctx.x[ids]@bg/.1


def _c125(ep,fixed_k=None):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C125',{'degenerate':deg[1]})
    ctx=p.context();yy,xx=np.indices(ctx.hw);qb=((yy%2)*2+xx%2).ravel();fields=[];regions=[];twocount=0
    for c in ctx.regions:
        if not np.any(ctx.u[c]>0):continue
        scores=[]
        for k in(1,2):
            total=0.;valid=True;train_sizes=[]
            for fold in range(4):
                train=c[qb[c]!=fold];test=c[qb[c]==fold]
                if not len(test):continue
                model=_em_class(ctx,train,k)
                if model is None:valid=False;break
                total+=_em_score(ctx,test,model).sum();train_sizes.append(len(train))
            cost=(k-1)*(ctx.x.shape[1]-1)*np.log(max(2,len(c)))/2
            scores.append(total-cost if valid else-np.inf)
        winner=int(np.argmax(scores))+1 if fixed_k is None else fixed_k
        model=_em_class(ctx,c,winner)
        if model is None:continue
        twocount+=winner==2;regions.append(c);fields.append(2*expit(_em_score(ctx,c,model))-1)
    z=rp(ctx,fields,regions);covered=np.zeros(len(ctx.x),bool)
    for c in regions:covered[c]=True
    z[~covered]=ctx.u[~covered];z[ctx.valid<=0]=-4
    return result(ep,z,'C125',{'two_center_regions':twocount,'supported_regions':len(regions),'em_iterations':5,'crossfit_folds':4})


register('C125',_c125,
         ['Four interleaved spatial folds use row/column parity; each FG component receives one unit-weight reference FG-centroid pseudo-observation per EM update.',
          'FG density is a uniform k-component spherical mixture, BG is the fixed reference BG centroid; the extra-center BIC fee uses the final full-region training size.'])
CONTROLS['control_C125_one_center']=lambda ep:_c125(ep,1)
CONTROLS['control_C125_two_centers']=lambda ep:_c125(ep,2)


def _band_path(ctx,band,start,end):
    import heapq
    allowed=band.copy();allowed[[start,end]]=True;allowed[ctx.negative]=False
    if not allowed[start]or not allowed[end]:return None
    initial=(start,False);distance={initial:(0.,(start,))};queue=[(0.,(start,),(start,),start,False)]
    while queue:
        cost,path,states,node,seen=heapq.heappop(queue)
        if distance.get((node,seen))!=(cost,path):continue
        if node==end and seen:return path
        begin,stop=ctx.W.indptr[node:node+2]
        for neighbor,w in zip(ctx.W.indices[begin:stop],ctx.W.data[begin:stop]):
            if not allowed[neighbor]:continue
            flag=seen or bool(band[neighbor]and ctx.positive[neighbor])
            encoded=int(neighbor)+len(ctx.x)*int(flag)
            if encoded in states:continue
            value=cost-np.log(max(w,1e-12));newpath=path+(int(neighbor),)
            old=distance.get((int(neighbor),flag),(np.inf,(len(ctx.x)+1,)))
            if (value,newpath)<old:
                distance[(int(neighbor),flag)]=(value,newpath);heapq.heappush(queue,(value,newpath,states+(encoded,),int(neighbor),flag))
    return None


def _c107_descriptor(ctx,p,open_all=False,closed_all=False):
    from scipy.ndimage import label,distance_transform_cdt
    from scipy.sparse.csgraph import connected_components
    negative=ctx.negative.reshape(ctx.hw);groups,count=label(negative,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    structure=np.array([[0,1,0],[1,1,1],[0,1,0]],bool);bands=[];union=np.zeros(len(ctx.x),bool);paths=[]
    for g in range(1,count+1):
        band=binary_dilation(groups==g,structure=structure).ravel()&(ctx.valid>0);bands.append(band);union|=band
    ii,jj=ctx.W.nonzero();weights=np.asarray(ctx.W[ii,jj]).ravel()
    restored=set()
    if not closed_all:
        for band in bands:
            ids=np.flatnonzero((ctx.valid>0)&~band);graph=ctx.W[ids][:,ids]
            _,cc=connected_components(graph,directed=False);components=[]
            for group in np.unique(cc):
                c=ids[cc==group]
                neighbors=np.asarray(ctx.W[c][:,np.flatnonzero(band)].sum(axis=1)).ravel()>0
                endpoints=c[neighbors]
                if len(endpoints):components.append(int(endpoints.min()))
            for start,end in itertools.combinations(components,2):
                path=_band_path(ctx,band,start,end)
                if path is not None:
                    for a,b in zip(path,path[1:]):restored.add((int(a),int(b)));restored.add((int(b),int(a)))
    keep=~(union[ii]|union[jj])
    if open_all:keep[:]=True
    else:
        for k,(i,j)in enumerate(zip(ii,jj)):
            if (int(i),int(j))in restored:keep[k]=True
    altered=sparse.csr_matrix((weights[keep],(ii[keep],jj[keep])),shape=ctx.W.shape)
    z0,a=G(ctx);z1,b=G(ctx,altered)
    if a or b:return np.full((len(ctx.x),5),np.nan)
    distance=np.minimum(distance_transform_cdt(~negative,metric='taxicab').ravel(),3).astype(float)
    if not negative.any():distance[:]=3
    return np.column_stack((ctx.u,z0,z1,z1-z0,distance))


register('C107',point_method('C107',_c107_descriptor),
         ['Dijkstra state tracks whether a positive anchor strictly inside the dilated band has been visited; ties compare complete row-ID paths.',
          'All band cuts are formed before all legal paths are restored, making pair order irrelevant.'])
CONTROLS['control_C107_all_closed']=point_method('control_C107_all_closed',lambda c,p:_c107_descriptor(c,p,closed_all=True))
CONTROLS['control_C107_all_open']=point_method('control_C107_all_open',lambda c,p:_c107_descriptor(c,p,open_all=True))


def _c115_descriptor(ctx,p,remove=True,bypass=True):
    from .group_076_150_common import mutual_graph
    W=ctx.W.maximum(mutual_graph(ctx.x,20));degree=np.diff(W.indptr);threshold=np.quantile(degree,.75)
    fg_mode=np.argmax(dot(ctx.x,ctx.role_prototypes(16)[0]),axis=1)
    additions={};removed=set();pathcount=np.zeros(len(ctx.x));allcount=np.zeros(len(ctx.x))
    for hub in np.flatnonzero(degree>=threshold):
        neighbors=W.indices[W.indptr[hub]:W.indptr[hub+1]];weight=W.data[W.indptr[hub]:W.indptr[hub+1]]
        if not np.any(ctx.positive[neighbors])or not np.any(ctx.negative[neighbors]):continue
        if ctx.negative[hub]and bypass:
            for a,b in itertools.combinations(range(len(neighbors)),2):
                i,j=int(neighbors[a]),int(neighbors[b]);allcount[[i,j]]+=1
                if fg_mode[i]==fg_mode[j]:
                    value=weight[a]*weight[b];additions[(i,j)]=additions.get((i,j),0)+value;additions[(j,i)]=additions.get((j,i),0)+value;pathcount[[i,j]]+=1
        if remove:
            hy,hx=np.unravel_index(hub,ctx.hw)
            for neighbor in neighbors:
                y,x=np.unravel_index(neighbor,ctx.hw)
                if abs(y-hy)+abs(x-hx)>1:removed.add((int(hub),int(neighbor)));removed.add((int(neighbor),int(hub)))
    i,j=W.nonzero();weight=np.asarray(W[i,j]).ravel();keep=np.array([(int(a),int(b))not in removed for a,b in zip(i,j)])
    altered=sparse.csr_matrix((weight[keep],(i[keep],j[keep])),shape=W.shape)
    if additions:
        pairs=list(additions);altered+=sparse.csr_matrix(([additions[a]for a in pairs],([a[0]for a in pairs],[a[1]for a in pairs])),shape=W.shape)
    z0,a=G(ctx,W);z1,b=G(ctx,altered)
    if a or b:return np.full((len(ctx.x),4),np.nan)
    proportion=np.divide(pathcount,allcount,out=np.zeros(len(ctx.x)),where=allcount>0)
    return np.column_stack((ctx.u,z0,z1,proportion))


register('C115',point_method('C115',_c115_descriptor),
         ['The hub degree quantile is over the exact mutual-20 plus spatial graph; bypass edges add products of the original two edges.',
          'Only nonspatial edges incident to mixed hubs are deleted; original point u is retained.'])
CONTROLS['control_C115_delete_only']=point_method('control_C115_delete_only',lambda c,p:_c115_descriptor(c,p,True,False))
CONTROLS['control_C115_bypass_only']=point_method('control_C115_bypass_only',lambda c,p:_c115_descriptor(c,p,False,True))


def _first_order(ctx,ids,edge_costs):
    u=ctx.u[ids];paths=[(0,),(1,)];dp=[np.logaddexp(0,u[0]),np.logaddexp(0,-u[0])];edits=[int(u[0]>0),int(u[0]<=0)]
    for k in range(1,len(ids)):
        nxt=[];newpaths=[];newedits=[]
        for label in(0,1):
            possible=[]
            for old in(0,1):
                cost=dp[old]+edge_costs[k-1,2*old+label]
                changed=edits[old]+int(bool(label)!=(u[k]>0));path=paths[old]+(label,)
                possible.append((float(cost),changed,path))
            selected=min(possible);nxt.append(selected[0]);newedits.append(selected[1]);newpaths.append(selected[2])
        dp=nxt;edits=newedits;paths=newpaths
    chosen=min(range(2),key=lambda label:(dp[label],edits[label],paths[label]))
    return np.asarray(paths[chosen],bool)


def _mst(ctx):
    from scipy.sparse.csgraph import minimum_spanning_tree
    # Full spatial topology, including zero-affinity physical edges; exact max tree.
    i,j=edges4(ctx.hw,ctx.valid);w=np.maximum(np.einsum('id,id->i',ctx.x[i],ctx.x[j]),0)**4
    # Strict positive cost avoids scipy treating physical zero-cost edges as absent.
    cost=2-w
    graph=sparse.csr_matrix((np.r_[cost,cost],(np.r_[i,j],np.r_[j,i])),shape=ctx.W.shape)
    tree=minimum_spanning_tree(graph);return tree+tree.T


def _role_pair_library(ctx,p):
    rc=p.context(True,ctx.fold);i,j=rc.W.nonzero()
    if ctx.fold is not None:
        keep=(p.rb[i]!=ctx.fold)&(p.rb[j]!=ctx.fold);i,j=i[keep],j[keep]
    return _PairStates(_edge_descriptors(rc,i,j)[:,:3],p.c[np.column_stack((i,j))])


def _c118(ep,independent=False):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C118',{'degenerate':deg[1]})
    ctx=p.context();tree=_mst(ctx);roots=[int(a[np.flatnonzero(ctx.positive[a])[0]])for a in ctx.atoms if np.any(ctx.positive[a])]
    leaves=np.flatnonzero((np.diff(tree.indptr)<=1)&(ctx.valid>0));votes=np.zeros(len(ctx.x));norm=np.zeros(len(ctx.x));length=0
    library=_role_pair_library(ctx,p)
    # Pair library state order is FF,FB,BF,BB; line decoder uses BB,BF,FB,FF.
    for root in roots:
        parent=np.full(len(ctx.x),-1,int);parent[root]=root;queue=[root]
        for node in queue:
            for neighbor in tree.indices[tree.indptr[node]:tree.indptr[node+1]]:
                if parent[neighbor]<0:parent[neighbor]=node;queue.append(int(neighbor))
        for leaf in leaves:
            if parent[leaf]<0:continue
            path=[int(leaf)]
            while path[-1]!=root:path.append(int(parent[path[-1]]))
            ids=np.array(path[::-1]);length+=len(ids)
            if len(ids)>1:
                logs=library.logs(_edge_descriptors(ctx,ids[:-1],ids[1:])[:,:3])
                labels=_first_order(ctx,ids,-logs[:,[3,2,1,0]])if not independent else ctx.u[ids]>0
            else:labels=ctx.u[ids]>0
            votes[ids]+=labels/len(ids);norm[ids]+=1/len(ids)
    z=ctx.u.copy();covered=norm>0;z[covered]=ctx.u[covered]/4+np.divide(votes,norm,out=np.zeros_like(votes),where=covered)[covered]-.5
    z[ctx.valid<=0]=-4
    return result(ep,z,'C118',{'roots':len(roots),'leaves':len(leaves),'total_path_length':length})


register('C118',_c118,
         ['Each public positive atom uses its lowest-row positive anchor as path origin; every physical MST leaf is included.',
          'State kernel observations use only [u_i,u_j,cos_ij]; transition states are reorder-equivalent FF/FB/BF/BB.'])
CONTROLS['control_C118_independent_paths']=lambda ep:_c118(ep,True)


def _attention(ctx,p,last2=False):
    role='r'if ctx.source else'q';name=role+('_attention_last2'if last2 else'_attention_final')
    value=np.asarray(artifact(p.ep,name))
    expected=4 if last2 else 3
    if value.ndim!=expected or value.shape[-2:]!=(len(ctx.x),len(ctx.x))or not np.isfinite(value).all()or np.any(value<0):
        raise ValueError('actual head/layer attention must be finite nonnegative patch-to-patch softmax submatrices')
    if np.any(value.sum(axis=-1)>1+1e-5):raise ValueError('patch attention row mass must not exceed actual full softmax mass')
    return value


def _head_message_norm(ctx,p):
    role='r'if ctx.source else'q';value=np.asarray(artifact(p.ep,role+'_head_output_final'))
    if value.ndim!=3 or value.shape[1]!=len(ctx.x)or not np.isfinite(value).all():raise ValueError('actual complete AV head-output patch rows required')
    return np.linalg.norm(value,axis=-1).mean(axis=0)


def _c101_descriptor(ctx,p,flow_only=False):
    from scipy.sparse.linalg import lsmr
    attention=_attention(ctx,p).mean(axis=0);i,j=edges4(ctx.hw,ctx.valid)
    flow=(attention[i,j]-attention[j,i])*(ctx.u[i]+ctx.u[j])/2
    incidence=sparse.csr_matrix((np.r_[np.ones(len(i)),-np.ones(len(i))],
                                (np.r_[np.arange(len(i)),np.arange(len(i))],np.r_[i,j])),shape=(len(i),len(ctx.x)))
    solved=lsmr(incidence,flow,atol=1e-10,btol=1e-10,maxiter=2000)
    if solved[1]not in(0,1,2):raise ArithmeticError('C101 actual Hodge least-squares did not converge within 2000 iterations')
    potential=solved[0]
    gradient=incidence@potential;curl=flow-gradient
    rows=[]
    for c in ctx.regions:
        mask=np.zeros(len(ctx.x),bool);mask[c]=True
        internal=mask[i]&mask[j];cross=mask[i]^mask[j]
        ratio=np.sum(curl[internal]**2)/max(np.sum(flow[internal]**2),1e-12)
        net=np.sum(np.where(mask[i[cross]],flow[cross],-flow[cross]))
        near_positive=np.any(np.column_stack((ctx.positive[i],ctx.positive[j])),axis=1)
        near_negative=np.any(np.column_stack((ctx.negative[i],ctx.negative[j])),axis=1)
        fg=np.mean(np.abs(curl[internal&near_positive]))if np.any(internal&near_positive)else np.nan
        bg=np.mean(np.abs(curl[internal&near_negative]))if np.any(internal&near_negative)else np.nan
        rows.append([ratio,net,ctx.u[c].mean(),fg,bg]if not flow_only else[net,ctx.u[c].mean()])
    return np.asarray(rows)


register('C101',region_method('C101',_c101_descriptor),
         ['Hodge potential is the least-norm incidence least-squares solution; cross-boundary net flow uses original f (the residual curl has identically zero net flux).',
          'Foreground/background coupling is mean absolute residual flow on internal edges touching the corresponding public anchor.'],
         ('final_unit_dino','q_attention_final','r_attention_final'))
CONTROLS['control_C101_original_flow_only']=region_method('control_C101_original_flow_only',lambda c,p:_c101_descriptor(c,p,True))


def _c145_fields(ctx,p,symmetric=False,degree_only=False):
    A=_attention(ctx,p).mean(axis=0);message=_head_message_norm(ctx,p);fields=[];regions=[]
    for c in ctx.regions:
        raw=A[np.ix_(c,c)];degree=raw.sum(axis=1);P=np.divide(raw,degree[:,None],out=np.zeros_like(raw),where=degree[:,None]>0)
        empty=degree<=0;P[np.flatnonzero(empty),np.flatnonzero(empty)]=1
        if symmetric:
            P=(P+P.T)/2;P/=np.maximum(P.sum(axis=1,keepdims=True),1e-12)
        bf=np.maximum(ctx.u[c],0);bb=np.maximum(-ctx.u[c],0)
        if bf.sum()<=0 and bb.sum()<=0:continue
        bf=bf/max(bf.sum(),1e-12);bb=bb/max(bb.sum(),1e-12)
        def multiply(matrix,vector):return np.einsum('ij,j->i',matrix,vector,optimize=False)
        df=multiply(P,multiply(P.T,bf))-multiply(P.T,multiply(P,bf))
        db=multiply(P,multiply(P.T,bb))-multiply(P.T,multiply(P,bb))
        descriptor=np.column_stack((ctx.u[c],df,db,raw.sum(axis=1),raw.sum(axis=0),message[c]))
        fields.append(descriptor[:,[0,3,4,5]]if degree_only else descriptor);regions.append(c)
    return regions,fields


register('C145',region_point_method('C145',_c145_fields),
         ['The final-block head attention matrices are averaged before each candidate row-normalization; all-empty row becomes an exact self-loop.',
          'Every candidate applies point K before the documented inverse-area RP; row/column degrees are measured on actual attention before local row normalization, and complete AV message norms are required.'],
         ('final_unit_dino','q_attention_final','r_attention_final','q_head_output_final','r_head_output_final'))
CONTROLS['control_C145_symmetric_attention']=region_point_method('control_C145_symmetric_attention',lambda c,p:_c145_fields(c,p,True))
CONTROLS['control_C145_degree_only']=region_point_method('control_C145_degree_only',lambda c,p:_c145_fields(c,p,degree_only=True))


def _select_attention_destination(ctx,p):
    attention=np.asarray(artifact(p.ep,'r_attention_last2'))
    if attention.ndim!=4 or attention.shape[-2:]!=(len(p.r),len(p.r)):raise ValueError('actual reference last-two-block heads required')
    excluded=np.zeros(len(p.r),bool)if ctx.fold is None else p.rb==ctx.fold
    if not hasattr(p,'destination_reference_mass'):
        mass=np.zeros(attention.shape[:2]+(2,16,len(p.r)),dtype=float)
        for layer in range(attention.shape[0]):
            for head in range(attention.shape[1]):
                A=attention[layer,head]
                for role in range(2):
                    role_mask=(p.c>=.9)if role==0 else(p.c<=.1)
                    for block in range(16):
                        ids=np.flatnonzero((p.rv>0)&role_mask&(p.rb==block))
                        for start in range(0,len(p.r),128):
                            mass[layer,head,role,block,start:start+128]=A[start:start+128,ids].sum(axis=1,dtype=float)
        p.destination_reference_mass=mass
    mass=p.destination_reference_mass
    effects=[]
    for fold in range(16):
        keep=(p.rv>0)&~excluded&(p.rb!=fold);fg=np.flatnonzero(keep&(p.c>=.9));bg=np.flatnonzero(keep&(p.c<=.1))
        if not len(fg)or not len(bg):continue
        retained=[b for b in range(16)if b!=fold and(ctx.fold is None or b!=ctx.fold)]
        fg_mass=mass[:,:,0,retained].sum(axis=2)[:,:,fg].mean(axis=-1)
        bg_mass=mass[:,:,1,retained].sum(axis=2)[:,:,fg].mean(axis=-1)
        effects.append(fg_mass-bg_mass)
    if not effects:return None
    effect=np.mean(effects,axis=0);best=np.unravel_index(np.argmax(effect),effect.shape)
    return best if effect[best]>0 else None


def _destination_mass(A,ctx):
    mass=np.zeros(len(ctx.x));source_high=np.zeros(len(ctx.x))
    for atom in ctx.atoms:
        senders=atom[ctx.positive[atom]]
        if not len(senders):continue
        candidate=A[senders].sum(axis=0)/len(senders)
        win=candidate>mass;mass[win]=candidate[win];source_high[win]=ctx.u[senders].max()
    return mass,source_high


def _c150_descriptor(ctx,p,all_heads=False):
    selected=_select_attention_destination(ctx,p)
    if selected is None:return np.full((len(ctx.x),4),np.nan)
    raw=_attention(ctx,p,last2=True);layer,head=selected;A=raw.mean(axis=(0,1))if all_heads else raw[layer,head]
    mass,highest=_destination_mass(A,ctx)
    rc=p.context(True,ctx.fold);rraw=np.asarray(artifact(p.ep,'r_attention_last2'))
    rA=rraw.mean(axis=(0,1))if all_heads else rraw[layer,head]
    r_mass,_=_destination_mass(rA,rc)
    receivers=rc.bank_ids[rc.fg]
    if not len(receivers):return np.full((len(ctx.x),4),np.nan)
    threshold=float(np.quantile(r_mass[receivers],.25))
    descriptor=np.column_stack((ctx.u,mass,ctx.max_b-ctx.max_f,highest));descriptor[(mass<threshold)|(mass<=0)]=np.nan
    return descriptor


register('C150',point_method('C150',_c150_descriptor,additive=True),
         ['Each reference fold head-selection effect removes that fold from both labeled sender and receiver banks; source pseudoqueries additionally remove their outer centre block.',
          'Each positive source atom contributes its mean actual sender-row arrival mass, and different atoms combine by max; threshold is the reference-FG receiver mass 25% quantile.'],
         ('final_unit_dino','q_attention_last2','r_attention_last2'))
CONTROLS['control_C150_all_heads_mean']=point_method('control_C150_all_heads_mean',lambda c,p:_c150_descriptor(c,p,True),additive=True)


def _attention_head_regions(ctx,p,head):
    A=_attention(ctx,p)[head];descriptors=[];valid=ctx.valid>0
    for c in ctx.regions:
        mask=np.zeros(len(ctx.x),bool);mask[c]=True;outside=np.flatnonzero(valid&~mask)
        out=A[c][:,outside].sum(axis=1).mean()if len(outside)else 0.
        inward=A[outside][:,c].sum(axis=1).mean()if len(outside)else 0.
        internal=A[c][:,c].sum(axis=1).mean()
        descriptors.append([out-inward,internal,ctx.u[c].mean(),len(c)])
    return np.asarray(descriptors)


def _select_region_head(ctx,p):
    key=ctx.fold
    if not hasattr(p,'region_attention_head_selection'):p.region_attention_head_selection={}
    if key in p.region_attention_head_selection:return p.region_attention_head_selection[key]
    raw=np.asarray(artifact(p.ep,'r_attention_final'))
    if raw.ndim!=3 or raw.shape[1:]!=(len(p.r),len(p.r)):raise ValueError('actual reference per-head final attention required')
    byhead=[[]for _ in range(len(raw))];labels=[];weights=[];source_folds=[]
    for fold in range(16):
        if fold==ctx.fold or not np.any((p.rb==fold)&(p.rv>0)):continue
        excluded=(fold,)if ctx.fold is None else(fold,ctx.fold)
        rc=p.context(True,excluded)
        if not len(rc.fg)or not len(rc.bg):continue
        chosen=[k for k,c in enumerate(rc.regions)if p.rb[c[len(c)//2]]==fold]
        for head in range(len(raw)):
            byhead[head].extend(_attention_head_regions(rc,p,head)[chosen])
        for k in chosen:
            c=rc.regions[k];labels.append(p.ep.wf[c].sum()/max(p.rv[c].sum(),1e-12));weights.append(p.rv[c].mean());source_folds.append(fold)
    labels=np.asarray(labels);weights=np.asarray(weights);source_folds=np.asarray(source_folds)
    if not len(labels)or not np.sum(labels*weights)or not np.sum((1-labels)*weights):selected=None
    else:
        effects=[]
        for head,data in enumerate(byhead):
            d=np.asarray(data).reshape(-1,4);pred=np.zeros(len(d))
            for fold in np.unique(source_folds):
                held=source_folds==fold;model=Kernel(d[~held],labels[~held],weights[~held]);pred[held]=model(d[held])
            effect=abs(np.sum(pred*labels*weights)/np.sum(labels*weights)-np.sum(pred*(1-labels)*weights)/np.sum((1-labels)*weights))
            effects.append(effect)
        selected=int(np.argmax(effects))
    p.region_attention_head_selection[key]=selected
    return selected


def _c110_descriptor(ctx,p,mean_heads=False):
    head=_select_region_head(ctx,p)
    if head is None:return np.full((len(ctx.regions),4),np.nan)
    if mean_heads:
        return np.mean([_attention_head_regions(ctx,p,h)for h in range(len(_attention(ctx,p)))],axis=0)
    return _attention_head_regions(ctx,p,head)


register('C110',region_method('C110',_c110_descriptor),
         ['Head selection maximizes absolute class-balanced source-kernel prediction effect under true leave-one-centre-block-out evaluation; ties use the smallest head ID.',
          'The selector and all its source kernels are rebuilt without an outer source-pseudoquery block; incoming/outgoing masses are normalized by actual sender counts.'],
         ('final_unit_dino','q_attention_final','r_attention_final'))
CONTROLS['control_C110_mean_heads']=region_method('control_C110_mean_heads',lambda c,p:_c110_descriptor(c,p,True))


def _overlap_domains(ctx):
    # Exact all-candidate overlap groups. A full root can legitimately make this
    # one whole-query domain; do not prune it to manufacture local behavior.
    parent=np.arange(len(ctx.x));used=np.zeros(len(ctx.x),bool)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    for c in ctx.regions:
        if not len(c):continue
        root=find(int(c[0]));used[c]=True
        for point in c[1:]:
            other=find(int(point))
            if root!=other:parent[other]=root
    groups={}
    for i in np.flatnonzero(used):groups.setdefault(find(int(i)),[]).append(int(i))
    return [np.array(v,int)for _,v in sorted(groups.items())]


def _c124_descriptor(ctx,p,graph=True):
    from .group_076_150_flow import minimum_edit_cut,persistent
    domains=_overlap_domains(ctx);out=np.full((len(ctx.x),5),np.nan)
    for ids in domains:
        W=ctx.W[ids][:,ids]if graph else sparse.csr_matrix((len(ids),len(ids)))
        _,_,solver=minimum_edit_cut(ctx.u[ids],W);fg,bg,unknown=persistent(solver,len(ids),len(ids)+1,len(ids))
        votes=[]
        for role in(0,1):
            reference=ctx.fg if role==0 else ctx.bg
            _,assignment=spherical(ctx.bank[reference],min(16,len(reference)))
            for mode in np.unique(assignment):
                retained=reference[assignment!=mode]
                if not len(retained):continue
                f=retained if role==0 else ctx.fg;b=retained if role==1 else ctx.bg
                u=np.clip((ctx.profile[ids][:,f].max(axis=1)-ctx.profile[ids][:,b].max(axis=1))/ctx.scale,-4,4)
                _,_,changed=minimum_edit_cut(u,W);pf,pb,pu=persistent(changed,len(ids),len(ids)+1,len(ids))
                votes.append(np.column_stack((pf,pb,pu)))
        if votes:
            rate=np.mean(votes,axis=0)
            out[ids]=np.column_stack((ctx.u[ids],fg.astype(int)-bg.astype(int),rate))
    return out


register('C124',point_method('C124',_c124_descriptor),
         ['All candidate-overlap domains are retained, including a whole-query domain if the unpruned tree root connects them.',
          'Persistent classes come from the true residual minimum-cut lattice; original/perturbed cuts use exact SCC closure to choose minimum S0 edits without changing primary energy.'])
CONTROLS['control_C124_mode_unary_votes']=point_method('control_C124_mode_unary_votes',lambda c,p:_c124_descriptor(c,p,False))


def _source_cut(ep):
    from .group_076_150_flow import minimum_edit_cut
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'control_C_E0_cut',{'degenerate':deg[1]})
    ctx=p.context();y,energy,_=minimum_edit_cut(ctx.u,ctx.W);y[ctx.valid<=0]=False
    return result(ep,2*y.astype(float)-1,'control_C_E0_cut',{'energy':energy,'tie':'exact minimum-S0-edit residual-SCC closure'})
CONTROLS['control_C_E0_cut']=_source_cut


def _c144_fields(ctx,p,replacement=True):
    from .group_076_150_common import graph4
    donors=[a for a in ctx.atoms if np.any(ctx.negative[a])]
    if len(donors)<3:return [],[]
    donors=sorted(donors,key=lambda a:int(a.min()))[:3]
    fields=[];regions=[]
    for c in ctx.regions:
        ring=_ring(ctx,c,2)
        domain=np.sort(np.r_[c,ring]);local=_subcontext(ctx,domain)
        baseline,info=G(local)
        if info:continue
        outcomes=[]
        for donor in donors:
            edited=ctx.x.copy();edited[ring]=ctx.x[donor[np.arange(len(ring))%len(donor)]]
            W=graph4(edited,ctx.hw,ctx.valid)[domain][:,domain]if replacement else local.W
            field,status=G(local,W)
            if status:break
            outcomes.append(field[np.searchsorted(domain,c)])
        if len(outcomes)!=3:continue
        values=np.array(outcomes);old=baseline[np.searchsorted(domain,c)]
        fields.append(np.column_stack((ctx.u[c],old,values.min(axis=0),values.max(axis=0),np.median(values,axis=0),np.full(len(c),len(c)))))
        regions.append(c)
    return regions,fields


register('C144',region_point_method('C144',_c144_fields),
         ['Three donor atoms are the lowest-row distinct atoms containing a fixed negative anchor; each outer ring is replaced by cyclic alignment in row order.',
          'Original u and public anchors are unchanged; only the local graph over C plus its two-grid outer ring is rebuilt, and each candidate point K precedes RP.'])
CONTROLS['control_C144_original_local_G']=region_point_method('control_C144_original_local_G',lambda c,p:_c144_fields(c,p,False))


def _radial_regions(ctx,c):
    center=int(c[np.argmax(ctx.u[c])]);cy,cx=np.unravel_index(center,ctx.hw);yy,xx=np.unravel_index(c,ctx.hw)
    radius=np.abs(yy-cy)+np.abs(xx-cx);rings=[c[radius==r]for r in np.unique(radius)]
    rows=[];previous=None
    for ring in rings:
        profile=ctx.profile[ring].mean(axis=0)
        difference=np.linalg.norm(profile-previous)if previous is not None else 0.
        rows.append([ctx.u[ring].mean(),ctx.u[ring].min(),ctx.u[ring].max(),difference]);previous=profile
    return rings,np.asarray(rows)


def _radial_library(ctx,p):
    rc=p.context(True,ctx.fold);d=[];soft=[]
    for c in rc.regions:
        if ctx.fold is not None and np.any(p.rb[c]==ctx.fold):continue
        rings,rows=_radial_regions(rc,c)
        for k in range(1,len(rings)):
            d.append(np.r_[rows[k-1],rows[k]])
            soft.append([np.sum(p.ep.wf[g])/max(p.rv[g].sum(),1e-12)for g in(rings[k-1],rings[k])])
    return _PairStates(np.asarray(d).reshape(-1,8),np.asarray(soft).reshape(-1,2),emission_columns=(0,4))


def _c139_fields(ctx,p,independent=False):
    library=_radial_library(ctx,p);fields=[]
    for c in ctx.regions:
        rings,d=_radial_regions(ctx,c)
        ids=np.array([g[0]for g in rings]);observed=ctx.u[ids].copy()
        # The line emission is the actual ring mean, not the representative token.
        from dataclasses import replace
        proxy=replace(ctx,u=ctx.u.copy());proxy.u[ids]=d[:,0]
        if len(rings)>1:
            costs=-library.logs(np.array([np.r_[d[k-1],d[k]]for k in range(1,len(rings))]))[:,[3,2,1,0]]
            label=_first_order(proxy,ids,costs)if not independent else d[:,0]>0
        else:label=d[:,0]>0
        rows=np.empty((len(c),5));lookup={int(point):k for k,point in enumerate(c)}
        for k,ring in enumerate(rings):
            at=np.array([lookup[int(i)]for i in ring]);rows[at]=np.column_stack((ctx.u[ring],np.full(len(ring),label[k]),np.repeat(d[k,:3][None],len(ring),axis=0)))
        fields.append(rows)
    return ctx.regions,fields


register('C139',region_point_method('C139',_c139_fields),
         ['Four-neighbor radial distance is Manhattan distance on the physical grid; rings are restricted to the candidate support.',
          'Ring-transition kernels use the concatenated four-coordinate descriptors of two consecutive rings; final point descriptor is [u,decoded ring label,ring mean/min/max], with K before RP.'])
CONTROLS['control_C139_independent_rings']=region_point_method('control_C139_independent_rings',lambda c,p:_c139_fields(c,p,True))


def _tree_parent_children(ctx):
    n=len(ctx.x);parent=np.arange(n);members={int(i):np.array([i],int)for i in np.flatnonzero(ctx.valid>0)}
    i,j=edges4(ctx.hw,ctx.valid);weights=np.maximum(np.einsum('id,id->i',ctx.x[i],ctx.x[j]),0)**4
    relations=[];seen=set()
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    def append(C,H):
        key=(C.tobytes(),H.tobytes())
        if key not in seen and len(H)<len(C):relations.append((C,H));seen.add(key)
    for k in np.lexsort((j,i,-weights)):
        a,b=find(int(i[k])),find(int(j[k]))
        if a==b:continue
        if a>b:a,b=b,a
        A,B=members.pop(a),members.pop(b);C=np.sort(np.r_[A,B]);parent[b]=a;members[a]=C
        append(C,A);append(C,B)
    # Upper-level-set tree, including joint equal-value plateaus.
    parent=np.arange(n);members={};active=np.zeros(n,bool);neighbors=[[]for _ in range(n)]
    for a,b in zip(i,j):neighbors[int(a)].append(int(b));neighbors[int(b)].append(int(a))
    order=np.flatnonzero(ctx.valid>0);order=order[np.argsort(-ctx.u[order],kind='stable')];start=0
    while start<len(order):
        stop=start+1
        while stop<len(order)and ctx.u[order[stop]]==ctx.u[order[start]]:stop+=1
        new=order[start:stop];previous=[]
        for node in new:
            for neighbor in neighbors[int(node)]:
                if active[neighbor]:previous.append(members[find(neighbor)].copy())
        previous.extend(np.array([i],int)for i in new)
        for node in new:active[node]=True;members[int(node)]=np.array([node],int)
        for node in new:
            for neighbor in neighbors[int(node)]:
                if not active[neighbor]:continue
                a,b=find(int(node)),find(neighbor)
                if a==b:continue
                if a>b:a,b=b,a
                C=np.sort(np.r_[members.pop(a),members.pop(b)]);parent[b]=a;members[a]=C
        for old in previous:
            C=members[find(int(old[0]))];append(C,old)
        start=stop
    return relations


def _capacity_certificate(ctx,C,H):
    from .group_076_150_flow import cut
    shell=np.setdiff1d(C,H,assume_unique=True);outside=_ring(ctx,C,1)
    source=H[ctx.positive[H]];sink=outside[ctx.negative[outside]]
    if not len(source)or not len(sink)or not len(shell):return [np.nan]*4
    domain=np.sort(np.r_[C,outside]);position={int(x):k for k,x in enumerate(domain)}
    s=np.zeros(len(domain));t=np.zeros(len(domain));s[[position[int(x)]for x in source]]=1/len(source);t[[position[int(x)]for x in sink]]=1/len(sink)
    W=ctx.W[domain][:,domain];zero=np.zeros(len(domain));_,kappa,_=cut(zero,W,s,t)
    ts=np.zeros(len(domain));ts[[position[int(x)]for x in shell]]=1/len(shell)
    _,ks,_=cut(zero,W,s,ts)
    return [ctx.u[H].mean(),ctx.u[shell].mean(),ks/(kappa+1e-12),np.mean(ctx.negative[shell])]


def _c113_samples(ctx,p):
    pairs=_tree_parent_children(ctx);regions=[];d=[]
    for C,H in pairs:
        row=_capacity_certificate(ctx,C,H)
        if np.isfinite(row).any():regions.append(C);d.append(row)
    return regions,np.asarray(d).reshape(-1,4)


def _c113(ep):
    z,info=dynamic_region_run(ep,_c113_samples)
    return result(ep,z,'C113',info)


register('C113',_c113,
         ['Every true parent-child relation in both public trees supplies one (C,H) certificate, exact duplicate pairs are removed; no best-core label search is introduced.',
          'The shell certificate replaces the original outside-BG sink set with every shell point at normalized total sink capacity one.'])
CONTROLS['control_C113_region_means']=lambda ep:result(ep,*dynamic_region_run(ep,lambda c,p:(lambda regions,d:(regions,d[:,:2]))(*_c113_samples(c,p)))[:1],'control_C113_region_means',{})


def _c121_fields(ctx,p,shuffle=False):
    K0=p.k0_for(ctx);pairs=_tree_parent_children(ctx);donors=[]
    for C,H in pairs:
        shell=np.setdiff1d(C,H,assume_unique=True)
        if not len(shell):continue
        if K0(_base_descriptor(ctx,H))[0]>0 and K0(_base_descriptor(ctx,shell))[0]>0:
            delta=ctx.x[shell].mean(axis=0)-ctx.x[H].mean(axis=0)
            donors.append((C,H,shell,delta,ctx.profile[H].mean(axis=0)))
    if shuffle and donors:
        perm=np.random.default_rng(121).permutation(len(donors));deltas=[donors[k][3]for k in perm]
        donors=[(d[0],d[1],d[2],delta,d[4])for d,delta in zip(donors,deltas)]
    fields=[];regions=[]
    for C,H in pairs:
        shell=np.setdiff1d(C,H,assume_unique=True)
        if not len(shell):continue
        available=[d for d in donors if not np.intersect1d(H,d[0],assume_unique=True).size]
        if not available:continue
        profile=ctx.profile[H].mean(axis=0)
        chosen=min(enumerate(available),key=lambda item:(np.linalg.norm(profile-item[1][4]),int(item[1][0].min()),item[0]))[1]
        core=ctx.x[H].mean(axis=0);v=unit(core+chosen[3]);core=unit(core)
        translated=ctx.x[shell]@v;original=ctx.x[shell]@core
        fields.append(np.column_stack((ctx.u[shell],translated,original,ctx.max_b[shell])));regions.append(shell)
    return regions,fields


register('C121',region_point_method('C121',_c121_fields),
         ['The same public tree parent-child relations define core H and its disjoint own shell C\\H; donor core and shell both require the independently prebuilt source K0 positive gate.',
          'All overlapping donor containers are excluded; a one-time nonoverlapping complete-profile nearest donor supplies Delta, and only the receiver own shell is scored.'])
CONTROLS['control_C121_shuffle_paired_delta']=region_point_method('control_C121_shuffle_paired_delta',lambda c,p:_c121_fields(c,p,True))


def _grounded_resistance(W,anchors):
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import splu
    n=W.shape[0];_,cc=connected_components(W,directed=False);R=np.full(n,np.inf);factors={};component_of={}
    degree=np.asarray(W.sum(axis=1)).ravel();L=sparse.diags(degree)-W
    for component in np.unique(cc):
        ids=np.flatnonzero(cc==component);ground=ids[anchors[ids]]
        if not len(ground):continue
        R[ground]=0;free=ids[~anchors[ids]]
        if not len(free):continue
        matrix=L[free][:,free].tocsc();factor=splu(matrix);position={int(i):k for k,i in enumerate(free)}
        for start in range(0,len(free),64):
            stop=min(start+64,len(free));rhs=np.zeros((len(free),stop-start));rhs[np.arange(start,stop),np.arange(stop-start)]=1
            solution=factor.solve(rhs);R[free[start:stop]]=solution[np.arange(start,stop),np.arange(stop-start)]
        factors[int(component)]=(free,position,factor)
        for point in free:component_of[int(point)]=int(component)
    return R,factors,component_of


def _edge_resistance(W,anchors,baseline,factors,component_of,i,j,weight):
    component=component_of.get(int(i),component_of.get(int(j)))
    if component is None:return baseline.copy()
    free,position,factor=factors[component];rhs=np.zeros(len(free))
    if int(i)in position:rhs[position[int(i)]]+=1
    if int(j)in position:rhs[position[int(j)]]-=1
    response=factor.solve(rhs);denominator=1-weight*(rhs@response)
    if denominator>1e-10:
        changed=baseline.copy();changed[free]=baseline[free]+weight*response**2/denominator
        return changed
    edited=W.tolil();edited[i,j]=0;edited[j,i]=0;edited=edited.tocsr();edited.eliminate_zeros()
    return _grounded_resistance(edited,anchors)[0]


def _c120_descriptor(ctx,p,intervene=True):
    atom_id=np.full(len(ctx.x),-1)
    for k,a in enumerate(ctx.atoms):atom_id[a]=k
    i,j=edges4(ctx.hw,ctx.valid);uncertain=~(ctx.positive|ctx.negative)
    positive_weight=np.asarray(ctx.W[i,j]).ravel()>0
    keep=(atom_id[i]!=atom_id[j])&(uncertain[i]|uncertain[j])&positive_weight;i,j=i[keep],j[keep]
    out=np.full((len(ctx.x),5),np.nan)
    if not len(i):return out
    stats=np.zeros((len(ctx.x),4));covered=np.zeros(len(ctx.x),bool)
    for role,anchors in enumerate((ctx.positive,ctx.negative)):
        baseline,factors,component=_grounded_resistance(ctx.W,anchors)
        for a,b in zip(i,j):
            weight=float(ctx.W[a,b]);covered[[a,b]]=True
            changed=_edge_resistance(ctx.W,anchors,baseline,factors,component,int(a),int(b),weight)if intervene and weight>0 else baseline
            for point in(a,b):
                disconnected=not np.isfinite(changed[point])
                if np.isfinite(changed[point])and np.isfinite(baseline[point]):
                    delta=np.log1p(changed[point])-np.log1p(baseline[point])
                    stats[point,2*role]=max(stats[point,2*role],delta if intervene else np.log1p(baseline[point]))
                stats[point,2*role+1]=max(stats[point,2*role+1],disconnected)
    out[covered]=np.column_stack((ctx.u[covered],stats[covered]));return out


register('C120',point_method('C120',_c120_descriptor),
         ['Test edges are spatial atom-boundary edges with at least one point lacking a strong public identity anchor.',
          'Effective resistance uses exact grounded Laplacian diagonal inverses; deletions use the exact rank-one inverse identity when nonsingular and explicitly recompute disconnected components otherwise.'])
CONTROLS['control_C120_original_resistance']=point_method('control_C120_original_resistance',lambda c,p:_c120_descriptor(c,p,False))


def _layers(ctx,p,name):
    role='r'if ctx.source else'q';a=np.asarray(artifact(p.ep,role+'_'+name))
    if a.ndim!=3 or a.shape[1:]!=ctx.x.shape or not np.isfinite(a).all():raise ValueError('actual aligned per-layer native patch states required')
    a=unit(a)
    if not np.allclose(a[-1],ctx.x,atol=2e-5,rtol=2e-5):raise ArtifactUnavailable('Actual final-layer artifact does not match the bound native unit episode')
    return a


def _layer_fields(ctx,p,name):
    q=_layers(ctx,p,name);r=np.asarray(artifact(p.ep,'r_'+name));r=unit(r)
    if r.ndim!=3 or r.shape[1:]!=p.r.shape or q.shape[0]!=r.shape[0]:raise ValueError('reference/query actual layers differ')
    if not np.allclose(r[-1],p.r,atol=2e-5,rtol=2e-5):raise ArtifactUnavailable('Reference layer artifact final state differs from the bound native unit episode')
    fields=[]
    excluded=()if ctx.fold is None else((ctx.fold,)if isinstance(ctx.fold,(int,np.integer))else tuple(ctx.fold))
    eligible=(p.rv>0)&~np.isin(p.rb,excluded)
    for layer in range(len(r)):
        affinity=dot(r[layer],r[layer]);oof=[]
        for block in range(16):
            held=np.flatnonzero(eligible&(p.rb==block));bank=np.flatnonzero(eligible&(p.rb!=block))
            fg=bank[p.c[bank]>=.9];bg=bank[p.c[bank]<=.1]
            if len(held)and len(fg)and len(bg):oof.extend((affinity[np.ix_(held,fg)].max(axis=1)-affinity[np.ix_(held,bg)].max(axis=1)).tolist())
        scale=max(float(np.subtract(*np.percentile(oof,[75,25])))if oof else 0,.01)
        profile=dot(q[layer],r[layer,ctx.bank_ids]);m=profile[:,ctx.fg].max(axis=1)-profile[:,ctx.bg].max(axis=1)
        fields.append(np.clip(m/scale,-4,4))
    return np.array(fields),q


def _closed_holes(ctx,foreground):
    from scipy.ndimage import label
    structure=np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask=foreground.reshape(ctx.hw)&(ctx.valid.reshape(ctx.hw)>0)
    fg,fc=label(mask,structure);bg,bc=label(~mask&(ctx.valid.reshape(ctx.hw)>0),structure)
    holes=[]
    physical=ctx.valid.reshape(ctx.hw)>0
    boundary=np.zeros(ctx.hw,bool);boundary[[0,-1],:]=True;boundary[:,[0,-1]]=True
    boundary|=physical&binary_dilation(~physical,structure=structure)
    for component in range(1,bc+1):
        hole=bg==component
        if np.any(hole&boundary):continue
        ring=binary_dilation(hole,structure=structure)&~hole
        neighbors=np.unique(fg[ring]);neighbors=neighbors[neighbors>0]
        if len(neighbors)==1:holes.append(np.flatnonzero(hole.ravel()))
    return holes


def _c131_samples(ctx,p):
    fields,layers=_layer_fields(ctx,p,'layer_tokens_half_threequarter_final')
    if len(fields)!=3:raise ValueError('C131 requires actual half, three-quarter and final layers')
    holes=_closed_holes(ctx,fields[-1]>0);d=[]
    for H in holes:
        ring=_ring(ctx,H,1)
        if not len(ring):d.append([np.nan]*7);continue
        closed=[np.all(fields[layer,ring]>0)for layer in range(3)]
        closure=next((layer for layer,value in enumerate(closed)if value),3)
        inner=fields[:,H].mean(axis=1);outer=fields[:,ring].mean(axis=1)
        distance=[]
        for layer in range(3):
            a=unit(layers[layer,H].mean(axis=0));b=unit(layers[layer,ring].mean(axis=0));distance.append(1-a@b)
        d.append([np.mean(fields[0,H]>0),np.mean(fields[1,H]>0),closure,
                  (inner[-1]-inner[0])-(outer[-1]-outer[0]),distance[1]-distance[0],distance[2]-distance[1],inner[-1]])
    return holes,np.asarray(d).reshape(-1,7)


def _c131(ep):
    z,info=dynamic_region_run(ep,_c131_samples)
    return result(ep,z,'C131',info)


register('C131',_c131,
         ['Hole candidates are final predicted BG components enclosed by one physical four-connected FG component, excluding the real image border.',
          'Closure time is the first observed layer with a wholly positive one-grid outside ring; embedding-distance changes use actual per-layer hole/ring mean cosines, with each layer independently source-scaled.'],
         ('final_unit_dino','q_layer_tokens_half_threequarter_final','r_layer_tokens_half_threequarter_final'))
CONTROLS['control_C131_final_hole_profile']=lambda ep:result(ep,*dynamic_region_run(ep,lambda c,p:(lambda regions,d:(regions,d[:,[-1]]))(*_c131_samples(c,p)))[:1],'control_C131_final_hole_profile',{})
CONTROLS['control_C131_all_layer_point_fields']=point_method('control_C131_all_layer_point_fields',lambda c,p:_layer_fields(c,p,'layer_tokens_half_threequarter_final')[0].T)


def _edge_bands(ctx,box=None):
    y0,x0,y1,x1=(0,0,ctx.hw[0],ctx.hw[1])if box is None else box
    grid=np.arange(len(ctx.x)).reshape(ctx.hw);lines=[]
    for side in range(4):
        line=[]
        if side in(0,1):
            for x in range(x0,x1):
                ys=range(y0,min(y0+3,y1))if side==0 else range(max(y0,y1-3),y1)
                ids=np.array([grid[y,x]for y in ys],int);ids=ids[ctx.valid[ids]>0]
                if len(ids):line.append(ids)
        else:
            for y in range(y0,y1):
                xs=range(x0,min(x0+3,x1))if side==2 else range(max(x0,x1-3),x1)
                ids=np.array([grid[y,x]for x in xs],int);ids=ids[ctx.valid[ids]>0]
                if len(ids):line.append(ids)
        if line:lines.append(line)
    return lines


def _band_rows(ctx,line):
    out=[];previous=None
    for ids in line:
        profile=ctx.profile[ids].mean(axis=0)
        change=0. if previous is None else np.linalg.norm(profile-previous)
        out.append([ctx.u[ids].mean(),ctx.u[ids].min(),ctx.u[ids].max(),change]);previous=profile
    return np.asarray(out)


def _boundary_library(ctx,p):
    rc=p.context(True,ctx.fold);h,w=rc.hw;ym,xm=(h+1)//2,(w+1)//2
    boxes=[None,(0,0,ym,xm),(0,xm,ym,w),(ym,0,h,xm),(ym,xm,h,w)]
    data=[];soft=[]
    for box in boxes:
        for line in _edge_bands(rc,box):
            rows=_band_rows(rc,line)
            for k in range(1,len(line)):
                if ctx.fold is not None and(np.any(p.rb[line[k-1]]==ctx.fold)or np.any(p.rb[line[k]]==ctx.fold)):continue
                data.append(np.r_[rows[k-1],rows[k]])
                soft.append([np.sum(p.ep.wf[g])/max(p.rv[g].sum(),1e-12)for g in(line[k-1],line[k])])
    return _PairStates(np.asarray(data).reshape(-1,8),np.asarray(soft).reshape(-1,2),emission_columns=(0,4))


def _c134(ep,independent=False):
    from dataclasses import replace
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C134',{'degenerate':deg[1]})
    ctx=p.context();library=_boundary_library(ctx,p);sumfield=np.zeros(len(ctx.x));count=np.zeros(len(ctx.x))
    for line in _edge_bands(ctx):
        rows=_band_rows(ctx,line);ids=np.array([g[0]for g in line]);proxy=replace(ctx,u=ctx.u.copy());proxy.u[ids]=rows[:,0]
        if len(line)>1 and not independent:
            costs=-library.logs(np.array([np.r_[rows[k-1],rows[k]]for k in range(1,len(line))]))[:,[3,2,1,0]]
            label=_first_order(proxy,ids,costs)
        else:label=rows[:,0]>0
        for station,foreground in zip(line,label):
            sumfield[station]+=1 if foreground else-1;count[station]+=1
    z=ctx.u.copy();covered=count>0;z[covered]=sumfield[covered]/count[covered];z[ctx.valid<=0]=-4
    return result(ep,z,'C134',{'scope':'physical edge three-grid inward bands only','free_initial_final_states':True,'source_edges':'actual four image edges and four fixed quadrant crop edges'})


register('C134',_c134,
         ['Four fixed reference crop boxes are the absolute 2×2 image quadrants; their borders are pseudo-edges without re-encoding.',
          'Every decoded station assigns its signed FG/BG label to its three inward tokens; overlapping corner stations are averaged, and all interior tokens retain u.'])
CONTROLS['control_C134_independent_edge_stations']=lambda ep:_c134(ep,True)


def _working_rgb(ctx,p):
    from ics.cpu100.common import rgb_view
    role='r'if ctx.source else'q'
    try:image=rgb_view(p.ep,role).astype(np.float64)/255
    except ValueError as error:raise ArtifactUnavailable('Original '+role+' RGB and exact recorded physical canvas required')from error
    if image.shape[:2]!=(ctx.hw[0]*16,ctx.hw[1]*16):raise ValueError('Actual RGB canvas and native patch geometry differ')
    return image


def _encode_changed(ctx,p,image):
    callback=artifact(p.ep,'frozen_encode_rgb');role='r'if ctx.source else'q'
    feature=np.asarray(callback(role,image,working_canvas=True,side=image.shape[0]))
    if feature.shape!=ctx.hw+(ctx.x.shape[1],)or not np.isfinite(feature).all():raise ValueError('Changed RGB encoder must return actual native patch features, not cached final features')
    return unit(feature.reshape(ctx.x.shape))


def _margin_changed(ctx,feature):
    profile=dot(feature,ctx.bank)
    return np.clip((profile[:,ctx.fg].max(axis=1)-profile[:,ctx.bg].max(axis=1))/ctx.scale,-4,4)


def _pixel_mask(ctx,ids,shape):
    mask=np.zeros(ctx.hw,bool);mask.ravel()[ids]=True
    return np.repeat(np.repeat(mask,shape[0]//ctx.hw[0],axis=0),shape[1]//ctx.hw[1],axis=1)


def _inner_boundary(ctx,c):
    from scipy.ndimage import binary_erosion
    mask=np.zeros(ctx.hw,bool);mask.ravel()[c]=True
    inner=binary_erosion(mask,np.array([[0,1,0],[1,1,1],[0,1,0]]),border_value=0)
    return np.flatnonzero(inner.ravel()),np.flatnonzero((mask&~inner).ravel())


def _c103_descriptor(ctx,p):
    from scipy.ndimage import label
    original=_working_rgb(ctx,p);out=[]
    for c in ctx.regions:
        top=c[np.lexsort((c,-ctx.u[c]))[:max(1,int(np.ceil(len(c)/4)))]]
        mask=np.zeros(ctx.hw,bool);mask.ravel()[top]=True
        cc,count=label(mask,np.array([[0,1,0],[1,1,1],[0,1,0]]))
        groups=[np.flatnonzero(cc.ravel()==k)for k in range(1,count+1)]
        H=min(groups,key=lambda g:(-len(g),int(g.min())))
        inner,boundary=_inner_boundary(ctx,c);preserved=np.r_[H,_ring(ctx,H,1)]
        editable=np.setdiff1d(inner,preserved,assume_unique=False)
        remaining=np.setdiff1d(c,H,assume_unique=True)
        if not len(editable)or not len(boundary)or not len(remaining):out.append([np.nan]*4);continue
        canvas=original.copy();pixels=_pixel_mask(ctx,editable,original.shape)
        color=np.median(original[_pixel_mask(ctx,boundary,original.shape)],axis=0);canvas[pixels]=color
        new=_margin_changed(ctx,_encode_changed(ctx,p,canvas));delta=ctx.u-new
        outside=np.setdiff1d(np.flatnonzero(ctx.valid>0),c,assume_unique=True)
        denominator=abs(delta[outside].mean())if len(outside)else 0.
        out.append([ctx.u[remaining].mean(),delta[remaining].mean(),abs(delta[H].mean())/(denominator+1e-12),ctx.u[c].mean()])
    return np.asarray(out)


register('C103',region_method('C103',_c103_descriptor),
         ['The largest four-connected component of the highest-u ceil(|C|/4) tokens is H; ties use the smallest row ID.',
          'Only one-grid interior pixels outside H plus its one-grid buffer are replaced; the original candidate boundary median RGB fills them, and the complete changed image is genuinely re-encoded.'],
         ('final_unit_dino','original_q_rgb','original_r_rgb','frozen_encode_rgb'))
CONTROLS['control_C103_original_candidate_mean']=CONTROLS['control_C102_region_mean']


def _c149_descriptor(ctx,p,static=False):
    original=_working_rgb(ctx,p);out=[]
    for c in ctx.regions:
        inner,boundary=_inner_boundary(ctx,c)
        if len(inner)<4:out.append([np.nan]*4);continue
        ordered=np.sort(inner);shift=len(ordered)//2;changes=[]
        for direction in(1,-1):
            if static:
                feature=ctx.x.copy();feature[ordered]=ctx.x[np.roll(ordered,direction*shift)]
            else:
                canvas=original.copy()
                source=np.roll(ordered,direction*shift)
                for target,origin in zip(ordered,source):
                    ty,tx=np.unravel_index(target,ctx.hw);sy,sx=np.unravel_index(origin,ctx.hw)
                    canvas[ty*16:(ty+1)*16,tx*16:(tx+1)*16]=original[sy*16:(sy+1)*16,sx*16:(sx+1)*16]
                feature=_encode_changed(ctx,p,canvas)
            changes.append(ctx.u-_margin_changed(ctx,feature))
        out.append([ctx.u[c].mean(),np.mean([d[inner].mean()for d in changes]),
                    np.mean([d[boundary].mean()for d in changes]),changes[0][inner].mean()-changes[1][inner].mean()])
    return np.asarray(out)


register('C149',region_method('C149',_c149_descriptor),
         ['Candidate interior patches are cyclically shifted by floor(interior_count/2) in both signs, keeping every one-grid boundary patch and each patch internal RGB arrangement unchanged.',
          'Interior/boundary response drops are averaged across the two actual changed-image encodings; their signed difference is retained.'],
         ('final_unit_dino','original_q_rgb','original_r_rgb','frozen_encode_rgb'))
CONTROLS['control_C149_static_descriptor_permutation']=region_method('control_C149_static_descriptor_permutation',lambda c,p:_c149_descriptor(c,p,True))


def _encode_changed_with_attention(ctx,p,image):
    from .internal_encoder import CaptureSession
    callback=artifact(p.ep,'frozen_encode_rgb')
    bound=getattr(callback,'__self__',None)
    model=getattr(bound,'model',None)
    if model is None:model=artifact(p.ep,'frozen_model')
    if not hasattr(model,'blocks'):raise ArtifactUnavailable('Actual bound frozen DINO model required for changed-image attention capture')
    block=len(model.blocks);capture=CaptureSession(model,attention_layers=(block,))
    with capture.hooks():feature=_encode_changed(ctx,p,image)
    if block not in capture.attention:raise ArtifactUnavailable('Changed-image encoder callback did not execute the supplied actual model')
    return feature,capture.attention[block]


def _replica_positions(ctx,c):
    y,x=np.unravel_index(c,ctx.hw);height=int(y.max()-y.min()+1);width=int(x.max()-x.min()+1)
    original_center=np.array([y.mean(),x.mean()]);grid=np.arange(len(ctx.x)).reshape(ctx.hw);options=[]
    for row in range(ctx.hw[0]-height+1):
        for col in range(ctx.hw[1]-width+1):
            ids=grid[row:row+height,col:col+width].ravel()
            if not np.all(ctx.valid[ids]>=1)or np.intersect1d(c,ids,assume_unique=True).size:continue
            center=np.array([row+(height-1)/2,col+(width-1)/2]);distance=np.linalg.norm(center-original_center)
            options.append((-distance,row,col,ids))
    options.sort(key=lambda v:v[:3]);selected=[]
    for option in options:
        if any(np.intersect1d(option[3],old[3],assume_unique=True).size for old in selected):continue
        selected.append(option)
        if len(selected)==2:break
    return selected,(int(y.min()),int(x.min()),height,width)


def _c126_descriptor(ctx,p):
    original=_working_rgb(ctx,p);A=_attention(ctx,p).mean(axis=0);out=[]
    for c in ctx.regions:
        positions,box=_replica_positions(ctx,c)
        if len(positions)<2:out.append([np.nan]*7);continue
        row,col,height,width=box;source=original[row*16:(row+height)*16,col*16:(col+width)*16].copy()
        ring=_ring(ctx,c,1);copied=np.r_[positions[0][3],positions[1][3]]
        ring=np.setdiff1d(ring,copied,assume_unique=False)
        if not len(ring):out.append([np.nan]*7);continue
        canvas=original.copy();changes=[];attention_drop=[]
        oldmass=A[c][:,c].sum(axis=1).mean()
        for _,dest_y,dest_x,ids in positions:
            canvas[dest_y*16:(dest_y+height)*16,dest_x*16:(dest_x+width)*16]=source
            feature,observed=_encode_changed_with_attention(ctx,p,canvas)
            margin=_margin_changed(ctx,feature);changes.append(margin)
            current=observed.weights(c+observed.prefix,c+observed.prefix).sum(axis=-1).mean()
            attention_drop.append((oldmass-current)/max(oldmass,1e-12))
        out.append([ctx.u[c].mean(),changes[0][c].mean()-ctx.u[c].mean(),changes[1][c].mean()-ctx.u[c].mean(),
                    changes[0][ring].mean()-ctx.u[ring].mean(),changes[1][ring].mean()-ctx.u[ring].mean(),attention_drop[0],attention_drop[1]])
    return np.asarray(out)


register('C126',region_method('C126',_c126_descriptor),
         ['Replica boxes are placed on the native patch lattice; legal boxes have fully physical support, overlap neither original C nor another replica, and maximize distance from C centroid.',
          'Attention diversion is the signed relative drop of original-C sender attention to original-C receivers under the actual changed-image head capture; both one/two-copy responses are recorded.'],
         ('final_unit_dino','original_q_rgb','original_r_rgb','frozen_encode_rgb','frozen_model','q_attention_final','r_attention_final'))
CONTROLS['control_C126_margin_curve_only']=region_method('control_C126_margin_curve_only',lambda c,p:_c126_descriptor(c,p)[:,:5])


def _exclude_folds(ctx,p,support):
    excluded=()if ctx.fold is None else((ctx.fold,)if isinstance(ctx.fold,(int,np.integer))else tuple(ctx.fold))
    return bool(excluded and np.any(np.isin(p.rb[support],excluded)))


def _relation_kernel(ctx,p,sample_fn,dimension):
    """Reference FF versus FB relations; BB contributes to neither library."""
    rc=p.context(True,ctx.fold);relations,descriptors=sample_fn(rc,p)
    data=[];labels=[];weights=[]
    for (A,B),d in zip(relations,descriptors):
        if _exclude_folds(ctx,p,np.r_[A,B])or not np.isfinite(d).any():continue
        a=p.ep.wf[A].sum()/max(p.rv[A].sum(),1e-12);b=p.ep.wf[B].sum()/max(p.rv[B].sum(),1e-12)
        fg=a*b;bg=a*(1-b)+(1-a)*b;mass=fg+bg
        if mass<=0:continue
        data.append(d);labels.append(fg/mass);weights.append(mass*np.mean(p.rv[np.r_[A,B]]))
    return Kernel(np.asarray(data).reshape(-1,dimension),np.asarray(labels),np.asarray(weights))


def _cross_band_relations(ctx):
    from scipy.ndimage import label
    structure=np.array([[0,1,0],[1,1,1],[0,1,0]])
    negative=ctx.negative.reshape(ctx.hw);band,nb=label(negative,structure)
    sides,_=label((ctx.valid.reshape(ctx.hw)>0)&~negative,structure)
    eligible=[]
    for C in ctx.regions:
        identifiers=np.unique(sides.ravel()[C])
        if len(identifiers)!=1 or identifiers[0]==0:continue
        around=_ring(ctx,C,1);contacts=set(band.ravel()[around]);contacts.discard(0)
        if contacts:eligible.append((C,int(identifiers[0]),contacts))
    return [(A,B)for (A,a,aa),(B,b,bb)in itertools.combinations(eligible,2)if a!=b and aa&bb]


def _region_relation_rows(ctx,p,relations,attention=True):
    A=_attention(ctx,p).mean(axis=0)if attention else None
    centers={c.tobytes():np.column_stack(np.unravel_index(c,ctx.hw)).mean(axis=0)for c in ctx.regions}
    negative=[c for c in ctx.regions if np.mean(ctx.negative[c])>=.9]
    rows=[]
    for left,right in relations:
        ca=centers[left.tobytes()];cb=centers[right.tobytes()]
        distance=int(np.floor(np.linalg.norm(ca-cb)));cos=float(unit(ctx.x[left].mean(axis=0))@unit(ctx.x[right].mean(axis=0)))
        if not attention:rows.append([ctx.u[left].mean(),ctx.u[right].mean(),cos]);continue
        ab=A[np.ix_(left,right)].sum()/len(left);ba=A[np.ix_(right,left)].sum()/len(right)
        controls=[]
        for source,center in((left,ca),(right,cb)):
            for bg in negative:
                if np.intersect1d(source,bg,assume_unique=True).size:continue
                if int(np.floor(np.linalg.norm(center-centers[bg.tobytes()])))==distance:
                    controls.append(A[np.ix_(source,bg)].sum()/len(source)+A[np.ix_(bg,source)].sum()/len(bg))
        rows.append([ab,ba,ab+ba-(np.mean(controls)if controls else np.nan),ctx.u[left].mean(),ctx.u[right].mean(),cos])
    return np.asarray(rows).reshape(-1,6 if attention else 3)


def _c132_samples(ctx,p,attention=True):
    relations=_cross_band_relations(ctx)
    return relations,_region_relation_rows(ctx,p,relations,attention)


def _c132(ep,attention=True):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C132',{'degenerate':deg[1]})
    ctx=p.context();fn=lambda c,p:_c132_samples(c,p,attention)
    model=_relation_kernel(ctx,p,fn,6 if attention else 3);pairs,d=fn(ctx,p);scores=np.maximum(model(d),0)
    total=np.zeros(len(ctx.x));norm=np.zeros(len(ctx.x))
    for (A,B),score in zip(pairs,scores):
        if score<=0:continue
        for C in(A,B):
            points=C[ctx.broad[C]&~ctx.negative[C]]
            total[points]+=score/len(C);norm[points]+=1/len(C)
    h=np.divide(total,norm,out=np.zeros_like(total),where=norm>0);z=ctx.u+h;z[ctx.valid<=0]=-4
    return result(ep,z,'C132',{'pairs':len(pairs),'source_pair_kernel_active':bool(model.active),'modified_existing_weak_points':int(np.sum(norm>0)),'BG_band_fill':False})


register('C132',_c132,
         ['A strong-BG band is a physical four-connected negative-anchor component; two candidates must lie entirely in different components after deleting the band and touch the same band.',
          'Same-distance controls use floor(Euclidean centroid distance) buckets and candidates with at least .9 public negative-anchor fraction; missing controls are explicit missing coordinates.',
          'Existing weak points mean frozen public weak-A membership without a strong-BG anchor; only positive FF-versus-FB kernel scores write there.'],
         ('final_unit_dino','q_attention_final','r_attention_final'))
CONTROLS['control_C132_endpoint_cosine']=lambda ep:_c132(ep,False)


def _atom_interfaces(ctx):
    owner=np.full(len(ctx.x),-1,int)
    for k,C in enumerate(ctx.atoms):owner[C]=k
    i,j=edges4(ctx.hw,ctx.valid);pairs={tuple(sorted((int(a),int(b))))for a,b in zip(owner[i],owner[j])if a>=0 and b>=0 and a!=b}
    return [(ctx.atoms[a],ctx.atoms[b])for a,b in sorted(pairs)]


def _c137_samples(ctx,p):
    fields,layers=_layer_fields(ctx,p,'layer_tokens_mid_final');pairs=_atom_interfaces(ctx)
    if len(fields)!=2:raise ValueError('C137 requires actual middle/final states')
    final_cos=[float(unit(layers[-1,A].mean(axis=0))@unit(layers[-1,B].mean(axis=0)))for A,B in pairs]
    cutoff=float(np.median(final_cos))if final_cos else -np.inf;chosen=[];rows=[]
    for (A,B),cosine in zip(pairs,final_cos):
        if cosine>=cutoff:continue
        midcos=float(unit(layers[0,A].mean(axis=0))@unit(layers[0,B].mean(axis=0)))
        owner=np.zeros(len(ctx.x),bool);owner[A]=True;i,j=edges4(ctx.hw,ctx.valid)
        boundary_edges=(owner[i]&np.isin(j,B))|(owner[j]&np.isin(i,B))
        boundary=np.unique(np.r_[i[boundary_edges],j[boundary_edges]])
        change=np.mean(np.linalg.norm(layers[1,boundary]-layers[0,boundary],axis=1))if len(boundary)else np.nan
        chosen.append((A,B));rows.append([(midcos-cosine)/max(1-midcos,.01),fields[1,A].mean()-fields[0,A].mean(),fields[1,B].mean()-fields[0,B].mean(),change,fields[1,A].mean(),fields[1,B].mean()])
    return chosen,np.asarray(rows).reshape(-1,6)


def _c137_descriptor(ctx,p,constant=False,mid_graph=False):
    pairs,descriptors=_c137_samples(ctx,p);model=_relation_kernel(ctx,p,_c137_samples,6);scores=np.maximum(model(descriptors),0)
    W=ctx.W.copy().tolil();count=0
    if mid_graph:
        from .group_076_150_common import graph4
        _,layers=_layer_fields(ctx,p,'layer_tokens_mid_final');W=graph4(layers[0],ctx.hw,ctx.valid).tolil()
    else:
        for (A,B),score in zip(pairs,scores):
            if score<=0:continue
            cosine=dot(ctx.x[A],ctx.x[B]);right=np.argmax(cosine,axis=1);left=np.argmax(cosine,axis=0)
            for k,v in enumerate(right):
                if left[v]!=k:continue
                a,b=int(A[k]),int(B[v]);weight=1. if constant else min(score,4.)
                W[a,b]=float(W[a,b])+weight;W[b,a]=float(W[b,a])+weight;count+=1
    z,code=G(ctx,W.tocsr())
    return np.column_stack((ctx.u,z))


register('C137',point_method('C137',_c137_descriptor),
         ['Interface candidates are spatially adjacent atom pairs whose final feature-centroid cosine lies strictly below the median of all adjacent atom-pair cosines.',
          'Pair descriptor uses relative middle-to-final cosine-distance change, each endpoint margin change, mean actual interface-token state change and final endpoint margins; only mutual nearest endpoint pairs receive positive calibrated jump weights.'],
         ('final_unit_dino','q_layer_tokens_mid_final','r_layer_tokens_mid_final'))
CONTROLS['control_C137_constant_jump']=point_method('control_C137_constant_jump',lambda c,p:_c137_descriptor(c,p,True))
CONTROLS['control_C137_middle_graph']=point_method('control_C137_middle_graph',lambda c,p:_c137_descriptor(c,p,mid_graph=True))
CONTROLS['control_C137_endpoint_layer_fields']=point_method('control_C137_endpoint_layer_fields',lambda c,p:_layer_fields(c,p,'layer_tokens_mid_final')[0].T)


def _gap_relations(ctx):
    grid=np.arange(len(ctx.x)).reshape(ctx.hw);out=[];seen=set()
    foreground=[C for C in ctx.regions if ctx.u[C].mean()>0]
    for A,B in itertools.combinations(foreground,2):
        if np.intersect1d(A,B,assume_unique=True).size:continue
        ya,xa=np.unravel_index(A,ctx.hw);yb,xb=np.unravel_index(B,ctx.hw)
        if np.min(np.abs(ya[:,None]-yb[None])+np.abs(xa[:,None]-xb[None]))>3:continue
        y0=min(ya.min(),yb.min());y1=max(ya.max(),yb.max())+1;x0=min(xa.min(),xb.min());x1=max(xa.max(),xb.max())+1
        H=np.setdiff1d(grid[y0:y1,x0:x1].ravel(),np.r_[A,B]);H=H[ctx.valid[H]>0]
        if not len(H):continue
        # Keep only gap components that actually border both visible fragments.
        from scipy.ndimage import label
        mask=np.zeros(ctx.hw,bool);mask.ravel()[H]=True;cc,n=label(mask,np.array([[0,1,0],[1,1,1],[0,1,0]]))
        for k in range(1,n+1):
            h=np.flatnonzero(cc.ravel()==k);ring=_ring(ctx,h,1)
            if not np.intersect1d(ring,A).size or not np.intersect1d(ring,B).size:continue
            key=(A.tobytes(),B.tobytes(),h.tobytes())
            if key not in seen:out.append((A,B,h));seen.add(key)
    return out


def _c140_samples(ctx,p,profile_only=False):
    pairs=_gap_relations(ctx);attention=_attention(ctx,p).mean(axis=0);norm=_head_message_norm(ctx,p);regions=[];rows=[]
    for A,B,H in pairs:
        regions.append(H)
        if profile_only:rows.append(np.r_[ctx.u[H].mean(),dot(ctx.x[H],p.r).mean(axis=0)]);continue
        rows.append([attention[np.ix_(H,A)].sum()/len(H),attention[np.ix_(H,B)].sum()/len(H),
                     attention[np.ix_(A,H)].sum()/len(A),attention[np.ix_(B,H)].sum()/len(B),norm[H].mean(),
                     ctx.u[H].mean(),ctx.u[A].mean(),ctx.u[B].mean(),float(unit(ctx.x[A].mean(axis=0))@unit(ctx.x[B].mean(axis=0)))])
    return regions,np.asarray(rows).reshape(-1,1+len(p.r)if profile_only else 9)


def _c140(ep,profile_only=False):
    z,info=dynamic_region_run(ep,lambda c,p:_c140_samples(c,p,profile_only));return result(ep,z,'C140',info)


register('C140',_c140,
         ['Neighboring positive-mean candidates are nonoverlapping and separated by at most three Manhattan grid steps; their union container is their axis-aligned bounding box.',
          'Only physical gap components touching both fragments receive a descriptor and correction. Value-message norm is the actual full-AV per-head output norm, not rerouted or masked attention.'],
         ('final_unit_dino','q_attention_final','r_attention_final','q_head_output_final','r_head_output_final'))
CONTROLS['control_C140_gap_full_profile']=lambda ep:_c140(ep,True)
CONTROLS['control_C140_head_point_readout']=point_method('control_C140_head_point_readout',lambda c,p:np.column_stack((c.u,_head_message_norm(c,p))))


def _dct_inputs(shape,random=False):
    height,width=shape;y=np.arange(height)+.5;x=np.arange(width)+.5
    frequencies=sorted(((a,b)for a in range(9)for b in range(9)if a+b>0),key=lambda k:(sum(k),k))[:8]
    directions=np.array([np.cos(np.pi*a*y[:,None]/height)*np.cos(np.pi*b*x[None,:]/width)for a,b in frequencies])
    directions/=np.sqrt(np.mean(directions**2,axis=(1,2)))[:,None,None]
    if random:
        # An orthogonal rotation of the same eight-dimensional input subspace.
        orthogonal,_=np.linalg.qr(np.random.default_rng(135).normal(size=(8,8)))
        directions=np.einsum('ij,jyx->iyx',orthogonal,directions,optimize=False)
    return directions


def _c135_observations(p,source,random=False):
    key=(source,random)
    if not hasattr(p,'c135_observations'):p.c135_observations={}
    if key in p.c135_observations:return p.c135_observations[key]
    contexts=[p.context(source,fold)for fold in([None]+list(range(16))if source else[None])]
    contexts=[c for c in contexts if len(c.fg)and len(c.bg)]
    original=_working_rgb(contexts[0],p);directions=_dct_inputs(original.shape[:2],random)
    partial={ctx.fold:[]for ctx in contexts};means={ctx.fold:np.zeros(len(ctx.x))for ctx in contexts}
    # Only fields are retained; no 32-forward feature pool, caps or fake JVP.
    for direction in directions:
        fields={ctx.fold:[]for ctx in contexts}
        for epsilon in(1/255,1/510):
            for sign in(1,-1):
                image=np.clip(original+sign*epsilon*direction[:,:,None],0,1)
                feature=_encode_changed(contexts[0],p,image)
                for ctx in contexts:
                    field=_margin_changed(ctx,feature);fields[ctx.fold].append(field);means[ctx.fold]+=field/32
        for ctx in contexts:
            a,b,c,d=fields[ctx.fold];derivative=(a-b)/(2/255);fine=(c-d)/(2/510)
            stable=np.abs(derivative-fine)/np.maximum(np.abs(fine),.01)<=.2
            partial[ctx.fold].append((np.where(stable,fine,0),stable))
    observation={}
    for ctx in contexts:
        derivative=np.column_stack([d for d,v in partial[ctx.fold]]);valid=np.column_stack([v for d,v in partial[ctx.fold]])
        observation[ctx.fold]=(np.column_stack((derivative,valid.astype(float))),means[ctx.fold],valid)
    p.c135_observations[key]=observation
    return observation


def _c135_descriptor(ctx,p,point_only=False,original=False,average=False,random=False):
    from .group_076_150_common import mutual_graph
    fingerprint,mean,valid=_c135_observations(p,ctx.source,random)[ctx.fold]
    if point_only:return np.column_stack((ctx.u,fingerprint))
    if average:return np.column_stack((ctx.u,mean))
    W=mutual_graph(ctx.x,20);i,j=W.nonzero();yi,xi=np.unravel_index(i,ctx.hw);yj,xj=np.unravel_index(j,ctx.hw)
    eligible=((yi-yj)**2+(xi-xj)**2<=64)&(ctx.valid[i]>0)&(ctx.valid[j]>0)
    i,j=i[eligible],j[eligible];f=unit(fingerprint)
    weight=np.maximum(np.einsum('id,id->i',f[i],f[j]),0)
    # Tokens with no valid perturbation direction have no fingerprint edges.
    weight[~valid[i].any(axis=1)|~valid[j].any(axis=1)]=0
    Wj=sparse.csr_matrix((weight,(i,j)),shape=ctx.W.shape);z,code=G(ctx,Wj);baseline,_=G(ctx,W)
    return np.column_stack((ctx.u,baseline,baseline if original else z))


register('C135',point_method('C135',_c135_descriptor),
         ['The first eight non-DC DCT modes use ascending (frequency sum,row frequency,column frequency), with RMS one; the same luminance scalar is added to all three RGB channels.',
          'Exactly 32 actual floating-point forward views per image are encoded; both finite-difference steps and the .2 relative-stability test are applied before retaining fine-step derivatives.',
          'Fingerprint cosine is on the literal 16-dimensional derivative-plus-validity vector, clipped below at zero; no-valid-direction tokens have no fingerprint edges. Native physical working canvas is required.'],
         ('final_unit_dino','original_q_rgb','original_r_rgb','frozen_encode_rgb'))
CONTROLS['control_C135_point_fingerprint']=point_method('control_C135_point_fingerprint',lambda c,p:_c135_descriptor(c,p,point_only=True))
CONTROLS['control_C135_view_mean']=point_method('control_C135_view_mean',lambda c,p:_c135_descriptor(c,p,average=True))
CONTROLS['control_C135_original_graph']=point_method('control_C135_original_graph',lambda c,p:_c135_descriptor(c,p,original=True))
CONTROLS['control_C135_orthogonal_DCT_inputs']=point_method('control_C135_orthogonal_DCT_inputs',lambda c,p:_c135_descriptor(c,p,random=True))


def _event_descriptor(ctx,nodes,node):
    a,b=node['children'];A=nodes[a]['support'];B=nodes[b]['support'];H=node['bridge']
    groups=(A,B,H);centers=[unit(ctx.x[C].mean(axis=0))if len(C)else None for C in groups]
    cosine=[float(centers[i]@centers[j])if centers[i]is not None and centers[j]is not None else np.nan for i,j in((0,1),(0,2),(1,2))]
    return np.r_[[ctx.u[C].mean()if len(C)else np.nan for C in groups],cosine,[np.mean(ctx.negative[C])if len(C)else np.nan for C in groups]]


def _c133_library(ctx,p):
    from .c_tree_076_150 import max_tree
    rc=p.context(True,ctx.fold);nodes,_=max_tree(rc);data=[];labels=[]
    for node in nodes:
        if not node['event']:continue
        groups=[nodes[c]['support']for c in node['children']]+[node['bridge']]
        if _exclude_folds(ctx,p,np.concatenate(groups)):continue
        coverage=[p.ep.wf[C].sum()/max(p.rv[C].sum(),1e-12)for C in groups]
        labels.append(tuple(1 if c>=.9 else 0 if c<=.1 else 2 for c in coverage));data.append(_event_descriptor(rc,nodes,node))
    model=Kernel(np.asarray(data).reshape(-1,9),np.full(len(data),.5));labels=np.asarray(labels).reshape(-1,3)
    def cost(d):
        standardized=model.standardize(np.asarray(d).reshape(1,-1));logk=-np.sum((model.d-standardized[0])**2,axis=1)/(2*model.sigma**2)
        table=np.zeros((3,3,3))
        for state in itertools.product(range(3),repeat=3):
            at=np.all(labels==state,axis=1)
            if at.any():table[state]=-(logsumexp(logk[at])-np.log(at.sum()))
        return table
    return cost,len(data)


def _c133(ep,independent=False):
    from .c_tree_076_150 import max_tree,decode
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C133',{'degenerate':deg[1]})
    ctx=p.context();nodes,roots=max_tree(ctx);kernel,count=_c133_library(ctx,p);costs={}
    if not independent:
        for index,node in enumerate(nodes):
            if node['event']:costs[index]=kernel(_event_descriptor(ctx,nodes,node))
    mask,info=decode(ctx,nodes,roots,costs);mask[ctx.valid<=0]=False
    return result(ep,2*mask.astype(float)-1,'C133',dict(info,source_event_samples=count,unseen_states='zero joint cost; independent leaf unary remains'))


register('C133',_c133,
         ['Equal-u plateaus enter together; on a merge with more than two old components the first row-ordered pair receives the actual new bridge, subsequent unions have no invented bridge potential.',
          'Three hard reference states use actual whole-support coverage; an absent source joint state receives zero interaction cost, retaining independent -u_i*y_i leaf costs.',
          'Every bridge is fully decoded as a constrained BG/FG/MIX leaf sequence and every original vertex is backtraced; energy ties first minimize S0 edits and then row labels.'])
CONTROLS['control_C133_unary_tree_DP']=lambda ep:_c133(ep,True)


def _widest_arm(ctx,domain,start,targets,foreground):
    import heapq
    if not len(targets):return np.nan
    allowed=np.zeros(len(ctx.x),bool);allowed[domain]=True;role=expit(ctx.u if foreground else-ctx.u)
    W=ctx.W.tocsr();strength=np.full(len(ctx.x),-np.inf);strength[start]=1.;queue=[(-1.,int(start))];wanted=set(map(int,targets))
    while queue:
        minus,node=heapq.heappop(queue);value=-minus
        if value!=strength[node]:continue
        if node in wanted:return value
        for at in range(W.indptr[node],W.indptr[node+1]):
            other=int(W.indices[at])
            if not allowed[other]:continue
            weight=float(W.data[at])*role[node]*role[other];proposal=min(value,weight)
            if proposal>strength[other]:strength[other]=proposal;heapq.heappush(queue,(-proposal,other))
    return np.nan


def _c136_events(ctx,p,mode='both'):
    from .c_tree_076_150 import max_tree
    nodes,_=max_tree(ctx);events=[];rows=[]
    for node in nodes:
        if not node['event']or not len(node['bridge']):continue
        A,B=[nodes[c]['support']for c in node['children']];peaks=[int(C[np.lexsort((C,-ctx.u[C]))[0]])for C in(A,B)]
        h=int(node['bridge'].min());hy,hx=np.unravel_index(h,ctx.hw);yy,xx=np.indices(ctx.hw);domain=np.flatnonzero(((np.abs(yy-hy)+np.abs(xx-hx)<=2)&(ctx.valid.reshape(ctx.hw)>0)).ravel())
        ay,ax=np.unravel_index(peaks[0],ctx.hw);by,bx=np.unravel_index(peaks[1],ctx.hw);axis=np.array([by-ay,bx-ax],float)
        if np.linalg.norm(axis)==0:continue
        dy,dx=np.unravel_index(domain,ctx.hw);coordinate=np.column_stack((dy-hy,dx-hx));projection=coordinate@axis;perpendicular=coordinate@np.array([-axis[1],axis[0]])
        f=[]
        for peak,C,sign in zip(peaks,(A,B),(-1,1)):
            if peak in domain:targets=np.array([peak])
            else:
                target=domain[np.isin(domain,C)&(sign*projection>0)]
                if len(target):
                    distance=np.abs(np.unravel_index(target,ctx.hw)[0]-hy)+np.abs(np.unravel_index(target,ctx.hw)[1]-hx)
                    targets=target[distance==distance.max()]
                else:targets=np.empty(0,int)
            f.append(_widest_arm(ctx,domain,h,targets,True))
        b=[_widest_arm(ctx,domain,h,domain[ctx.negative[domain]&(sign*perpendicular>0)],False)for sign in(-1,1)]
        fstrength=min(f)if np.isfinite(f).all()else np.nan;bstrength=min(b)if np.isfinite(b).all()else np.nan
        identity=float(np.linalg.norm(ctx.profile[peaks[0]]-ctx.profile[peaks[1]]));row=[ctx.u[h],fstrength,bstrength,identity]
        if mode=='F':row[2]=np.nan
        if mode=='u':row[1:]=[np.nan]*3
        if mode=='profile':row=np.r_[ctx.u[h],dot(ctx.x[domain],p.r).mean(axis=0)]
        events.append((h,domain));rows.append(row)
    return events,np.asarray(rows).reshape(-1,1+len(p.r)if mode=='profile'else 4)


def _c136(ep,mode='both'):
    p=prepare(ep);deg=p.degenerate()
    if deg is not None:return result(ep,deg[0],'C136',{'degenerate':deg[1]})
    data=[];labels=[];weights=[];dimension=1+len(p.r)if mode=='profile'else 4
    for fold in range(16):
        ctx=p.context(True,fold)
        if not len(ctx.fg)or not len(ctx.bg):continue
        events,rows=_c136_events(ctx,p,mode)
        for (h,domain),row in zip(events,rows):
            if p.rb[h]!=fold:continue
            data.append(row);labels.append(p.c[h]);weights.append(p.rv[h])
    model=Kernel(np.asarray(data).reshape(-1,dimension),np.asarray(labels),np.asarray(weights));ctx=p.context();events,d=_c136_events(ctx,p,mode);h=model(d)
    total=np.zeros(len(ctx.x));norm=np.zeros(len(ctx.x))
    for (_,domain),score in zip(events,h):total[domain]+=score;norm[domain]+=1
    z=ctx.u+np.divide(total,norm,out=np.zeros_like(total),where=norm>0);z[ctx.valid<=0]=-4
    return result(ep,z,'C136',{'events':len(events),'source_event_samples':len(data),'kernel_active':bool(model.active),'missing_arms':'NaN with common missing flags, no forced cut'})


register('C136',_c136,
         ['Peak/saddle events are genuine upper-level-set merges in the maximum vertex-u bottleneck forest; joint plateaus use their lowest-row new bridge vertex as h.',
          'Local arms use exact widest spatial paths in the Manhattan-radius-two neighborhood, with capacities w_ij*sigma(role*u_i)*sigma(role*u_j); out-of-window peaks use the furthest local points in their actual premerge component.',
          'BG arms must reach actual negative anchors on opposite perpendicular sides of the peak axis; source labels are h coverage, and overlapping event-neighborhood increments are averaged uniformly.'])
CONTROLS['control_C136_foreground_arms']=lambda ep:_c136(ep,'F')
CONTROLS['control_C136_saddle_unary']=lambda ep:_c136(ep,'u')
CONTROLS['control_C136_local_full_profile']=lambda ep:_c136(ep,'profile')


def _triangles(graph,ids=None):
    allowed=set(range(graph.shape[0]))if ids is None else set(map(int,ids));neighbors={i:set(map(int,graph.indices[graph.indptr[i]:graph.indptr[i+1]]))&allowed for i in allowed}
    triples=[];wedges=[]
    for center in sorted(allowed):
        around=sorted(neighbors[center])
        for a,b in itertools.combinations(around,2):
            wedges.append((a,center,b))
            if center<a<b and b in neighbors[a]:triples.append((center,a,b))
    return np.asarray(triples,int).reshape(-1,3),np.asarray(wedges,int).reshape(-1,3)


def _triangle_descriptor(ctx,triples):
    if not len(triples):return np.empty((0,6))
    a,b,c=triples.T
    return np.column_stack((ctx.u[a],ctx.u[b],ctx.u[c],np.einsum('id,id->i',ctx.x[a],ctx.x[b]),np.einsum('id,id->i',ctx.x[b],ctx.x[c]),np.einsum('id,id->i',ctx.x[a],ctx.x[c])))


def _c143_kernels(ctx,p):
    from .group_076_150_common import mutual_graph
    rc=p.context(True,ctx.fold);graph=mutual_graph(rc.x,20);triangles,_=_triangles(graph,np.flatnonzero(rc.valid>0));d=[];labels=[];weights=[]
    for t in triangles:
        if _exclude_folds(ctx,p,t):continue
        coverage=p.c[t];fg=np.prod(coverage);mixed=1-fg-np.prod(1-coverage);mass=fg+mixed
        if mass<=0:continue
        for order in itertools.permutations(t):
            d.append(_triangle_descriptor(rc,np.array([order]))[0]);labels.append(fg/mass);weights.append(mass/6)
    tri=Kernel(np.asarray(d).reshape(-1,6),np.asarray(labels),np.asarray(weights))
    i,j=graph.nonzero();rows=[];labels=[];weights=[]
    for a,b in zip(i,j):
        if _exclude_folds(ctx,p,np.array([a,b])):continue
        ca,cb=p.c[a],p.c[b];fg=ca*cb;mix=ca*(1-cb)+(1-ca)*cb;mass=fg+mix
        if mass<=0:continue
        rows.append([rc.u[a],rc.u[b],float(rc.x[a]@rc.x[b])]);labels.append(fg/mass);weights.append(mass)
    pair=Kernel(np.asarray(rows).reshape(-1,3),np.asarray(labels),np.asarray(weights))
    return tri,pair


def _swap_graph(ctx,graph,seed,strict=False):
    rng=np.random.default_rng(seed);i,j=sparse.triu(graph,k=1).nonzero();edges=list(zip(map(int,i),map(int,j)));existing=set(edges);accepted=0
    if len(edges)<2:return graph.copy(),0
    affinity=[float(ctx.x[a]@ctx.x[b])for a,b in edges];cuts=np.quantile(affinity,np.linspace(0,1,11))
    def signature(a,b):return(int(np.searchsorted(cuts,float(ctx.x[a]@ctx.x[b]),side='right')),int((ctx.u[a]>0)==(ctx.u[b]>0)))
    for _ in range(8*len(edges)):
        a,b=sorted(rng.choice(len(edges),2,replace=False));(u,v),(x,y)=edges[a],edges[b]
        if rng.integers(2):x,y=y,x
        if len({u,v,x,y})<4:continue
        left=tuple(sorted((u,y)));right=tuple(sorted((x,v)))
        if left==right or left in existing or right in existing:continue
        if strict and sorted((signature(u,v),signature(x,y)))!=sorted((signature(*left),signature(*right))):continue
        existing.remove(edges[a]);existing.remove(edges[b]);existing.add(left);existing.add(right);edges[a]=left;edges[b]=right;accepted+=1
    rows=np.array(edges,int);out=sparse.csr_matrix((np.ones(2*len(rows)),(np.r_[rows[:,0],rows[:,1]],np.r_[rows[:,1],rows[:,0]])),shape=graph.shape)
    assert np.array_equal(np.diff(out.indptr),np.diff(graph.indptr))
    return out,accepted


def _triangle_residual(ctx,graph,C,tri,pair,wedge_only=False):
    triangles,wedges=_triangles(graph,C);edge_score=[]
    if len(wedges):
        a,b,c=wedges.T
        for i,j in((a,b),(b,c)):
            d=np.column_stack((ctx.u[i],ctx.u[j],np.einsum('id,id->i',ctx.x[i],ctx.x[j])));edge_score.append(pair(d))
        independent=np.mean(edge_score[0]+edge_score[1])
    else:independent=0.
    if wedge_only:return float(independent)
    if not len(triangles):return -float(independent)
    scores=[]
    for permutation in itertools.permutations(range(3)):scores.append(tri(_triangle_descriptor(ctx,triangles[:,permutation])))
    return float(np.mean(scores)-independent)


def _c143_descriptor(ctx,p,strict=False,wedge_only=False):
    from .group_076_150_common import mutual_graph
    graph=mutual_graph(ctx.x,20);tri,pair=_c143_kernels(ctx,p);permuted=[];accepted=[]
    for seed in range(14300,14308):
        changed,count=_swap_graph(ctx,graph,seed,strict);permuted.append(changed);accepted.append(count)
    if not any(accepted):return np.full((len(ctx.regions),4),np.nan)
    rows=[]
    for C in ctx.regions:
        residual=_triangle_residual(ctx,graph,C,tri,pair,wedge_only)
        null=np.array([_triangle_residual(ctx,g,C,tri,pair,wedge_only)for g in permuted])
        rows.append([ctx.u[C].mean(),residual,residual-null.mean(),null.var()])
    return np.asarray(rows)


register('C143',region_method('C143',_c143_descriptor),
         ['The triple kernel averages all six endpoint orders. Independent wedge prediction is the sum of two FF-versus-mixed pair log ratios; residual is mean triangle score minus mean over all centered two-edge wedges.',
          'Each of eight fixed-seed simple-graph nulls attempts exactly eight swaps per undirected edge; every accepted double-edge swap preserves all vertex degrees and fixed endpoint features. No legal swaps make the increment inactive.'])
CONTROLS['control_C143_wedge_only']=region_method('control_C143_wedge_only',lambda c,p:_c143_descriptor(c,p,wedge_only=True))
CONTROLS['control_C143_degree_homophily_null']=region_method('control_C143_degree_homophily_null',lambda c,p:_c143_descriptor(c,p,strict=True))
ASSUMPTIONS['control_C143_degree_homophily_null']=['The stricter null additionally preserves the multiset of original decile cosine-homophily bins and current unary-sign homophily for each swapped edge pair; it does not preserve exact continuous cosine sums.']
CONTROLS['control_C143_joint_three_attributes']=CONTROLS['control_C105_no_permutation']

# B has its own legal four-fold calibration and exact FoRIS host edit contract.
# It never uses the C kernel or substitutes the MEAN host for original FoRIS.
from . import b_methods_076_100 as _b_originals
METHODS.update(_b_originals.METHODS)
ASSUMPTIONS.update(_b_originals.ASSUMPTIONS)
REQUIREMENTS.update(_b_originals.REQUIREMENTS)
for _name,_fn in _b_originals.CONTROLS.items():CONTROLS[_name]=_fn
