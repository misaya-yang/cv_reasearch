"""Original B076--B100 algorithms, exact B host/edit contract.

Only explicit implementations enter METHODS. Controls and inactive calibration
are not additional methods. Frozen encoder and FoRIS host are real resources.
"""
from __future__ import annotations
import itertools,math
import numpy as np
from scipy import sparse
from scipy.special import expit,logsumexp
from .common import Result,artifact,ArtifactUnavailable
from .b_helpers_076_100 import (prepare_b,calibrate,finish_b,partial_match,owner,choices,role_neighbors,Frame,ward,role_cluster,logmean)
from .group_076_150_common import unit,dot,edges4
METHODS={};CONTROLS={};ASSUMPTIONS={};REQUIREMENTS={}


def register(mid,fn,kind,assumptions=(),assets=()):
    METHODS[mid]=fn;ASSUMPTIONS[mid]=list(assumptions)
    REQUIREMENTS[mid]=['final_unit_dino']+([]if kind=='D'else['foris_p0','foris_mask0','foris_producer','foris_renderer'])+list(assets)


def run(ep,mid,kind,fn,extra=None):
    p=prepare_b(ep);deg=p.degenerate()
    if deg is not None:
        return Result(deg[0].reshape(ep.q_hw),0.,info={'method_id':mid,'degenerate':deg[1],'qgt_access':False})
    if not p.structure_active():
        raw=p.frame().ell if kind=='D'else np.full(len(p.q),np.nan)if kind=='W'else np.zeros(len(p.q))
        return finish_b(ep,raw,mid,kind,0,{'structure_active':False,'inactive_reason':'four spatial leaveout banks with one-grid buffer do not retain both roles'})
    scale=calibrate(p,fn,'iqr')if kind in('H','Z')else None
    raw=fn(p.frame());info={'structure_active':True,'source_IQR':scale,'implementation_assumptions':ASSUMPTIONS.get(mid,[]),'source_folds_share_encoder_context':True}
    if extra:info.update(extra)
    return finish_b(ep,raw,mid,kind,scale,info)


def _dual_candidate(f,C,edge_weight=1.):
    ef,mf,vf=partial_match(f,C,True,edge_weight);eb,mb,vb=partial_match(f,C,False,edge_weight)
    # Matched signed evidence; both role graphs search at the same budget.
    return ef+eb,np.asarray(vf)+np.asarray(vb),(ef,mf,eb,mb)


def _dual_field(f,edge_weight=1.,direct=False):
    energies=[];fields=[];counts=[]
    for C in f.regions:
        energy,v,state=_dual_candidate(f,C,edge_weight);energies.append(energy);fields.append(v);counts.append(len(set(q for m in(state[1],state[3])for q in m if q>=0)))
    z=owner(f,energies,fields,counts)
    if direct:z=np.where(z!=0,z,f.ell)
    return z


def _reciprocal_options(f,C,foreground):
    own=f.role_f if foreground else f.role_b;options,rival=choices(f,C,foreground);keep=[]
    for role,points in enumerate(options):
        keep.append(tuple(q for q in points if np.argmax(own[q])==role))
    return keep,rival


def _b076(f,onepass=False,point_union=False):
    energies=[];fields=[];counts=[]
    for C in f.regions:
        sides=[];energy=0.;mapped=set()
        for foreground in(True,False):
            options,rival=_reciprocal_options(f,C,foreground);own=f.role_f if foreground else f.role_b
            if not options:sides.append(np.zeros(len(C)));continue
            candidates=[(r,q)for r,points in enumerate(options)for q in points if own[q,r]>rival[q]]
            if not candidates:sides.append(np.zeros(len(C)));continue
            seed=min(candidates,key=lambda v:(-(own[v[1],v[0]]-rival[v[1]]),v));active={seed[0]};forced={seed[0]:seed[1]}
            ef,mapping,field=partial_match(f,C,foreground,allowed_roles=active,forced=forced)
            adj=role_neighbors(f,foreground)
            for _ in range(len(options)):
                if onepass:active=set(range(len(options)))
                else:
                    frontier={r for r in range(len(options))if r not in active and any(adj[r,a]for a in active)}
                    if not frontier:break
                    # A semantic path must lie in this visible candidate, not in a reference hallucination.
                    from scipy.sparse.csgraph import connected_components
                    ids=np.asarray(C);lookup={int(q):k for k,q in enumerate(ids)};i,j=edges4(f.hw,f.valid)
                    keep=np.isin(i,C)&np.isin(j,C)&(f.role_f[i].max(axis=1)>=f.role_b[i].max(axis=1)if foreground else f.role_b[i].max(axis=1)>=f.role_f[i].max(axis=1))&(f.role_f[j].max(axis=1)>=f.role_b[j].max(axis=1)if foreground else f.role_b[j].max(axis=1)>=f.role_f[j].max(axis=1))
                    graph=sparse.csr_matrix((np.ones(2*keep.sum()),(np.r_[[lookup[int(a)]for a in i[keep]],[lookup[int(a)]for a in j[keep]]],np.r_[[lookup[int(a)]for a in j[keep]],[lookup[int(a)]for a in i[keep]]])),shape=(len(C),len(C)))
                    _,components=connected_components(graph,directed=False);roots={components[lookup[q]]for q in mapping if q>=0}
                    reachable={r for r in frontier if any(own[q,r]>rival[q]and components[lookup[q]]in roots for q in options[r])}
                    if not reachable:break
                    active|=reachable
                old_energy,old_map=ef,mapping
                # Full refit after every accepted role; old inconsistent correspondences may disappear.
                ef,mapping,field=partial_match(f,C,foreground,allowed_roles=active)
                if onepass or mapping==old_map:break
            energy+=ef;sides.append(field);mapped.update(q for q in mapping if q>=0)
        energies.append(energy);fields.append(sum(sides));counts.append(len(mapped))
    if point_union:
        z=np.zeros(len(f.x))
        for C,v in zip(f.regions,fields):
            at=np.abs(v)>np.abs(z[C]);z[C[at]]=v[at]
        return z
    return owner(f,energies,fields,counts)


register('B076',lambda ep:run(ep,'B076','H',_b076),'H',
         ['A local reciprocal seed is the strongest own-role winner also selecting that reference family as its nearest same-class family.',
          'Missing-role frontier follows reference role adjacency; its query witness must remain connected to current mapped points by same-class direct-advantage physical edges within the candidate. Every accepted frontier is followed by an unforced complete refit.'])
CONTROLS['control_B076_all_roles_onepass']=lambda ep:run(ep,'B076','H',lambda f:_b076(f,True))
CONTROLS['control_B076_same_list_point_union']=lambda ep:run(ep,'B076','H',lambda f:_b076(f,point_union=True))


def _b079(f,static=False,no_edges=False):
    energies=[];fields=[];counts=[]
    for C in f.regions:
        ef,mf,vf=partial_match(f,C,True,0 if no_edges else 1);eb,mb,vb=partial_match(f,C,False,0 if no_edges else 1)
        fields.append((vf+vb).copy());energies.append(ef+eb);counts.append(len(set(q for m in(mf,mb)for q in m if q>=0)))
    output=[np.zeros_like(v)for v in fields]
    for left,right in itertools.combinations(range(len(f.regions)),2):
        A,B=f.regions[left],f.regions[right]
        if np.intersect1d(A,B).size:continue
        ya,xa=np.unravel_index(A,f.hw);yb,xb=np.unravel_index(B,f.hw)
        if np.min(np.abs(ya[:,None]-yb[None])+np.abs(xa[:,None]-xb[None]))!=1:continue
        for foreground in(True,False):
            weight=0 if no_edges else 1;ea,ma,va=partial_match(f,A,foreground,weight);eb,mb,vb=partial_match(f,B,foreground,weight)
            for role,(a,b)in enumerate(zip(ma,mb)):
                if a<0 or b<0:continue
                # Cross-witness domains retain all original points plus the swapped witness.
                AA=np.sort(np.r_[A[A!=a],b]);BB=np.sort(np.r_[B[B!=b],a])
                ca,_,_=partial_match(f,AA,foreground,weight,forced={role:b});cb,_,_=partial_match(f,BB,foreground,weight,forced={role:a})
                if static:
                    ca,cb=ea,eb
                    own=f.role_f if foreground else f.role_b;op=f.role_b if foreground else f.role_f
                    ca+=op[b].max()-own[b,role]-(op[a].max()-own[a,role]);cb+=op[a].max()-own[a,role]-(op[b].max()-own[b,role])
                gain=max(ca+cb-ea-eb,0.)
                if no_edges:gain=0. # Source requirement: purely additive edge model has no structural advantage.
                # Only the swapped point gets this witness; no whole-object foreground grant.
                if a in A:output[left][np.searchsorted(A,a)]+=gain*va[np.searchsorted(A,a)]
                if b in B:output[right][np.searchsorted(B,b)]+=gain*vb[np.searchsorted(B,b)]
    return owner(f,energies,output,counts)


register('B079',lambda ep:run(ep,'B079','H',_b079),'H',
         ['Neighbor candidates are disjoint Ward regions with one physical adjacency; each role-witness exchange substitutes a point into the opposite domain and forces that exchanged role, then refits all remaining roles.',
          'Positive cross-versus-original complete-energy loss multiplies the original signed witness margin; FG and BG searches are symmetric. The zero-edge ablation is explicitly inactive.'])
CONTROLS['control_B079_frozen_correspondence_swap']=lambda ep:run(ep,'B079','H',lambda f:_b079(f,True))
CONTROLS['control_B079_zero_adjacency']=lambda ep:run(ep,'B079','H',lambda f:_b079(f,no_edges=True))


def _b080(f,ungated=False):
    from dataclasses import replace
    observed=[];fields=[];energies=[];counts=[]
    for C in f.regions:
        e,v,state=_dual_candidate(f,C);count=len(set(q for m in(state[1],state[3])for q in m if q>=0));counts.append(count)
        energies.append(e);fields.append(v);observed.append(-e/max(count,1))
    if ungated:return owner(f,energies,fields,counts)
    maxima=[]
    for replication in range(32):
        rng=np.random.default_rng(80000+replication);ff=f.role_f.copy();bb=f.role_b.copy();profile=f.profile.copy()
        for matrix,indices,labels in((ff,f.fg,f.flabels),(bb,f.bg,f.blabels)):
            for role in range(matrix.shape[1]):
                shift=(int(rng.integers(f.hw[0])),int(rng.integers(f.hw[1])))
                matrix[:,role]=np.roll(matrix[:,role].reshape(f.hw),shift,(0,1)).ravel()
                columns=indices[labels==role]
                profile[:,columns]=np.roll(profile[:,columns].reshape(f.hw+(len(columns),)),shift,(0,1)).reshape(len(f.x),len(columns))
        tf=np.argsort(-profile[:,f.fg],axis=1,kind='stable')[:,:4];tb=np.argsort(-profile[:,f.bg],axis=1,kind='stable')[:,:4]
        null=replace(f,role_f=ff,role_b=bb,profile=profile,top_f=tf,top_b=tb);stat=[]
        for C in f.regions:
            e,v,state=_dual_candidate(null,C);count=len(set(q for m in(state[1],state[3])for q in m if q>=0));stat.append(-e/max(count,1))
        maxima.append(max(stat,default=0.))
    threshold=float(np.quantile(maxima,.95))
    kept=[v if score>threshold else np.zeros_like(v)for score,v in zip(observed,fields)]
    return owner(f,energies,kept,counts)


register('B080',lambda ep:run(ep,'B080','H',_b080),'H',
         ['Each of 32 fixed-seed nulls independently applies a two-dimensional cyclic shift to each reference-family response and all corresponding member-token response columns, then rebuilds both classes top-four correspondences. Each null records the maximum dual conjunction over the entire fixed candidate pool.',
          'The conjunction statistic is negative complete dual match energy per explained point; only a strict exceedance of the null-max .95 quantile permits its original signed point margins.'],)
CONTROLS['control_B080_no_null_gate']=lambda ep:run(ep,'B080','H',lambda f:_b080(f,True))


def _b088(f,point=False):
    from dataclasses import replace
    from scipy.ndimage import label
    mask=np.zeros(f.ep.r_hw,bool);mask.ravel()[f.r_ids[f.bg]]=True
    components,count=label(mask,np.array([[0,1,0],[1,1,1],[0,1,0]]));banks=[]
    for component in range(1,count+1):
        ids=f.bg[components.ravel()[f.r_ids[f.bg]]==component]
        if len(ids)<len(f.fcenters):continue
        centers,labels=role_cluster(f.r[ids],min(len(f.fcenters),len(ids)))
        if len(centers)<len(f.fcenters):continue
        top=np.argsort(-f.profile[:,ids],axis=1,kind='stable')[:,:4]
        banks.append(replace(f,bg=ids,bcenters=centers,blabels=labels,role_b=dot(f.x,centers),top_b=top))
    energies=[];fields=[];counts=[]
    for C in f.regions:
        ef,mf,vf=partial_match(f,C,True,0 if point else 1)
        if point or not banks:eb,mb,vb=partial_match(f,C,False,0)
        else:
            matches=[partial_match(bank,C,False,1)for bank in banks]
            eb,mb,vb=min(matches,key=lambda state:(state[0]/max(sum(q>=0 for q in state[1]),1),state[1]))
        energies.append(ef+eb);field=vf+vb;counts.append(len(set(q for m in(mf,mb)for q in m if q>=0)))
        # Unit costs are localized to actual matched points, not the whole candidate.
        fields.append(field)
    z=owner(f,energies,fields,counts);return np.where(z!=0,z,f.ell)


register('B088',lambda ep:run(ep,'B088','D',_b088),'D',
         ['Actual pure-background four-connected components each supply a local role graph at the same K/top-four/null/search budget as foreground; components with fewer distinct families cannot supply an automatic foreground win and use the equal-budget point control.',
          'Complete FG+BG explanation energy chooses the point owner; each observed correspondence contributes its own BG-minus-FG unit cost, while unmatched points retain complete two-class ell.'])
CONTROLS['control_B088_equal_budget_point_correspondence']=lambda ep:run(ep,'B088','D',lambda f:_b088(f,True))


def _b094(f,random=False,appearance_only=False):
    from scipy.sparse.csgraph import connected_components
    profile=f.profile;n=len(f.r);groups=[]
    for role in(f.fg,f.bg):
        if not len(role):return np.zeros(len(f.x))
        similarity=dot(f.r[role],f.r[role]);similarity[np.diag_indices_from(similarity)]=0
        # Fixed near-duplicate criterion, not query-label tuning.
        related=similarity>=.99
        if not appearance_only:
            ids=f.r_ids[role];ry,rx=np.unravel_index(ids,f.ep.r_hw)
            related&=(ry[:,None]//max(1,f.ep.r_hw[0]//4)==ry[None]//max(1,f.ep.r_hw[0]//4))&(rx[:,None]//max(1,f.ep.r_hw[1]//4)==rx[None]//max(1,f.ep.r_hw[1]//4))
        _,label=connected_components(sparse.csr_matrix(related),directed=False)
        partition=[role[label==c]for c in np.unique(label)]
        if random:
            perm=np.random.default_rng(94).permutation(role);sizes=list(map(len,partition));partition=[];at=0
            for size in sizes:partition.append(perm[at:at+size]);at+=size
        if len(partition)<=1:return np.zeros(len(f.x))
        groups.append(partition)
    grouped=profile[:,f.fg].max(axis=1) # Allocate exact complete point field.
    if not any(len(g)>1 for partition in groups for g in partition):return np.zeros(len(f.x))
    fg=np.mean([profile[:,g].max(axis=1)-profile[:,f.bg].max(axis=1)for g in groups[0]],axis=0)
    bg=np.mean([profile[:,g].max(axis=1)-profile[:,f.fg].max(axis=1)for g in groups[1]],axis=0)
    independent=profile[:,f.fg].mean(axis=1)-profile[:,f.bg].mean(axis=1)
    return .5*(fg-bg)-independent


register('B094',lambda ep:run(ep,'B094','H',_b094),'H',
         ['Near-repeated reference evidence is joined only when cosine is at least .99 and both tokens share the same absolute4-by4 source block; connected groups are frozen independently in each legal source bank.',
          'Each class group contributes its maximum signed own-class-versus-opposite-class token margin once; equal class-group averages are symmetrized, then subtract the all-token class mean margin.'])
CONTROLS['control_B094_random_same_counts']=lambda ep:run(ep,'B094','H',lambda f:_b094(f,True))
CONTROLS['control_B094_appearance_only']=lambda ep:run(ep,'B094','H',lambda f:_b094(f,appearance_only=True))
CONTROLS['control_B_source_ell']=lambda ep:run(ep,'B088','D',lambda f:f.ell)


def _matching_problem(f,C):
    families=[];options=[];unaries=[];labels=[];adj=[]
    for foreground in(True,False):
        own=f.role_f if foreground else f.role_b;opts,rival=choices(f,C,foreground);matrix=role_neighbors(f,foreground)
        offset=len(options)
        for role,points in enumerate(opts):
            options.append((-1,)+points);unaries.append({-1:0.,**{int(q):float(rival[q]-own[q,role])for q in points}});labels.append(int(foreground));families.append(role)
        adj.append((offset,matrix))
    relation=np.zeros((len(options),len(options)),bool)
    for offset,matrix in adj:relation[offset:offset+len(matrix),offset:offset+len(matrix)]=matrix
    return options,unaries,np.asarray(labels),relation


def _ranked_matches(f,C,limit=16,ambiguity=None,node_budget=None):
    """A* with a valid relaxed unary lower bound; never a beam top-L claim."""
    import heapq
    opts,unary,labels,adj=_matching_problem(f,C);K=len(opts)
    suffix=np.zeros(K+1)
    for role in range(K-1,-1,-1):suffix[role]=suffix[role+1]+min(unary[role].values())
    heap=[(float(suffix[0]),0,(),0.)];complete=[];seen=set();nodes=0;best=None;incomplete=False
    while heap:
        bound,depth,mapping,energy=heapq.heappop(heap)
        if ambiguity is not None and best is not None and bound>best+ambiguity:heapq.heappush(heap,(bound,depth,mapping,energy));break
        if node_budget is not None and nodes>=node_budget:
            heapq.heappush(heap,(bound,depth,mapping,energy));incomplete=True;break
        nodes+=1
        if depth==K:
            key=tuple(sorted((int(q),int(labels[r]),int(r))for r,q in enumerate(mapping)if q>=0))
            if key in seen:continue
            seen.add(key);complete.append((energy,mapping));best=energy if best is None else best
            if ambiguity is None and len(complete)>=limit:break
            continue
        for q in opts[depth]:
            if q>=0 and q in mapping:continue
            value=energy+unary[depth][q]
            if q>=0:
                for role,old in enumerate(mapping):
                    if old>=0 and adj[depth,role]:value+=max(np.linalg.norm(f.geometry_q[q]-f.geometry_q[old])-2/max(f.hw),0)
            heapq.heappush(heap,(float(value+suffix[depth+1]),depth+1,mapping+(int(q),),float(value)))
    tail_bound=heap[0][0]if heap else np.inf
    return complete,float(tail_bound),5**K,labels,{'expanded_nodes':nodes,'unfinished_certified_relaxation':incomplete,'complete_exact_topL':not incomplete and ambiguity is None}


def _state_labels(C,mapping,labels,unknown=.5):
    p=np.full(len(C),unknown,dtype=float);location={int(q):i for i,q in enumerate(C)}
    for role,q in enumerate(mapping):
        if q>=0:p[location[q]]=labels[role]
    return p


def _b087(f,map_only=False,ignore_tail=False):
    energies=[];fields=[];counts=[]
    for C in f.regions:
        states,bound,total,labels,status=_ranked_matches(f,C,1 if map_only else 16)
        if not states:energies.append(0.);fields.append(f.ell[C].copy());counts.append(len(C));continue
        E=np.array([e for e,m in states]);logmass=-E/.07;origin=float(logmass.max());mass=np.exp(logmass-origin)
        posterior=np.sum(np.array([_state_labels(C,m,labels)for e,m in states])*mass[:,None],axis=0)/mass.sum()
        remaining=max(0,total-len(states));tail=0. if remaining==0 or not np.isfinite(bound)else np.exp(np.clip(np.log(remaining)-bound/.07-origin,-745,709))
        low=posterior*mass.sum()/(mass.sum()+tail);high=(posterior*mass.sum()+tail)/(mass.sum()+tail)
        permitted=(low>.5)|(high<.5)
        if ignore_tail:permitted[:]=True
        fields.append(np.where(permitted,2*posterior-1,f.ell[C]));energies.append(float(E[0]));counts.append(len(C))
    return owner(f,energies,fields,counts)


register('B087',lambda ep:run(ep,'B087','D',_b087),'D',
         ['A partial state jointly assigns distinct observed points to K_F+K_B colored families (at most16); equivalent role-member permutations collapse before weights. A* returns the exact first16 distinct complete states using nonnegative relation fees and relaxed remaining-unary lower bounds.',
          'The literal (b+1)^K count upper bound uses K=K_F+K_B; tail mass upper bound uses the smallest remaining A* frontier bound. Unassigned state points have .5 responsibility; no tail-stable sign falls back to complete ell.'])
CONTROLS['control_B087_single_MAP']=lambda ep:run(ep,'B087','D',lambda f:_b087(f,True,True))
CONTROLS['control_B087_ignore_tail']=lambda ep:run(ep,'B087','D',lambda f:_b087(f,ignore_tail=True))


def _b086_field(f,delta,map_only=False):
    energies=[];fields=[];counts=[]
    for C in f.regions:
        states,bound,total,labels,status=_ranked_matches(f,C,1,ambiguity=delta,node_budget=100000)
        if not states or status['unfinished_certified_relaxation']:
            # Unexpanded branches may assign either class to any point: [0,1]
            # is a certified outer bound, not a claim that the beam is complete.
            energies.append(states[0][0]if states else 0.);fields.append(np.zeros(len(C)));counts.append(len(C));continue
        responsibilities=np.array([_state_labels(C,m,labels)for e,m in states]);lo=responsibilities.min(axis=0);hi=responsibilities.max(axis=0)
        if map_only:lo=hi=responsibilities[0]
        signed=np.where(lo>.5,2*(lo-.5),np.where(hi<.5,2*(hi-.5),0.))
        energies.append(states[0][0]);fields.append(signed);counts.append(len(C))
    return owner(f,energies,fields,counts)


def _b086(ep,map_only=False):
    p=prepare_b(ep);deg=p.degenerate()
    if deg is not None:return Result(deg[0].reshape(ep.q_hw),0.,info={'method_id':'B086','degenerate':deg[1]})
    if not p.structure_active():return finish_b(ep,np.zeros(len(p.q)),'B086','H',0,{'structure_active':False})
    errors=[];frames=[]
    for fold in range(4):
        f=p.frame(True,fold);frames.append(f);held=(p.rb==fold)&(p.rv>0);errors.extend(np.maximum(-(2*p.c[held]-1)*f.ell[held],0))
    thresholds=np.unique(np.quantile(errors,[.1,.25,.5,.75,.9]));loss=[]
    for delta in thresholds:
        predictions=[];coverage=[];weights=[]
        for fold,f in enumerate(frames):
            held=(p.rb==fold)&(p.rv>0);predictions.extend(_b086_field(f,float(delta),map_only)[held]>0);coverage.extend(p.c[held]);weights.extend(p.rv[held])
        y=np.array(predictions);c=np.array(coverage);w=np.array(weights);risk=.5*(np.sum(w*c*~y)/np.sum(w*c)+np.sum(w*(1-c)*y)/np.sum(w*(1-c)))
        loss.append((float(risk),-float(delta)))
    delta=float(thresholds[min(range(len(loss)),key=lambda i:loss[i])]);fn=lambda f:_b086_field(f,delta,map_only)
    return run(ep,'B086','H',fn,{'reference_error_tolerance':delta,'ambiguity_bounds':'exact complete state set or certified [0,1] outer bounds under the100000-node resource budget'})


register('B086',_b086,'H',
         ['Reference error candidates are quantiles of the nonnegative wrong-sign heldout ell residual; choose the finite candidate minimizing class-balanced four-fold error, with larger ambiguity tolerance conservative on ties.',
          'Ambiguity A* enumerates every state within best cost plus delta; after100000 expanded nodes its unexpanded branches conservatively enlarge every point interval to[0,1], so no unsupported correction is issued.'])
CONTROLS['control_B086_single_best_responsibility']=lambda ep:_b086(ep,True)


def _b098_nodes(f,C):
    nodes=[]
    for foreground in(True,False):
        own=f.role_f if foreground else f.role_b;options,rival=choices(f,C,foreground)
        for role,points in enumerate(options):
            for q in points:nodes.append((q,int(foreground),role,float(own[q,role]-rival[q])))
    # No unmentioned cap: larger candidate sets use the prescribed32 beam.
    conflict=np.zeros((len(nodes),len(nodes)),bool)
    for a,b in itertools.combinations(range(len(nodes)),2):
        qa,la,ra,_=nodes[a];qb,lb,rb,_=nodes[b]
        incompatible=(qa==qb)or(la==lb and ra==rb)
        if la==lb and ra!=rb:
            roleids=f.fg if la else f.bg;labels=f.flabels if la else f.blabels
            ca=f.geometry_r[roleids[labels==ra]].mean(axis=0);cb=f.geometry_r[roleids[labels==rb]].mean(axis=0)
            # Local orientation order, axes with no reference displacement abstain.
            expected=cb-ca;actual=f.geometry_q[qb]-f.geometry_q[qa]
            incompatible|=bool(np.any((np.abs(expected)>1/max(f.ep.r_hw))&(expected*actual<0)))
        conflict[a,b]=conflict[b,a]=incompatible
    return nodes,conflict


def _independent_set(nodes,conflict,removed=None,greedy=False):
    weights=np.array([max(n[3],0)for n in nodes]);eligible=[i for i in range(len(nodes))if i!=removed and weights[i]>0]
    if greedy:
        chosen=[]
        for i in sorted(eligible,key=lambda i:(-weights[i],i)):
            if not any(conflict[i,j]for j in chosen):chosen.append(i)
        return tuple(chosen),float(weights[chosen].sum())
    states=[(0.,())]
    for i in eligible:
        extended=states+[(energy+weights[i],chosen+(i,))for energy,chosen in states if not any(conflict[i,j]for j in chosen)]
        extended.sort(key=lambda s:(-s[0],s[1]));states=extended if len(nodes)<=16 else extended[:32]
    energy,chosen=states[0];return chosen,float(energy)


def _b098(f,greedy=False,no_stability=False):
    energies=[];fields=[];counts=[]
    for C in f.regions:
        nodes,conflicts=_b098_nodes(f,C);selected,value=_independent_set(nodes,conflicts,greedy=greedy);stable=set(selected)
        if not no_stability:
            # Removing each selected correspondence tests stability of the others.
            for removed in selected:
                chosen,other=_independent_set(nodes,conflicts,removed,greedy);stable&=(set(chosen)|{removed})
        field=np.zeros(len(C));location={int(q):i for i,q in enumerate(C)}
        for i in stable:
            q,label,role,weight=nodes[i];field[location[q]]=(1 if label else-1)*max(weight,0.)
        energies.append(-value);fields.append(field);counts.append(len({nodes[i][0]for i in selected}))
    return owner(f,energies,fields,counts)


register('B098',lambda ep:run(ep,'B098','H',_b098),'H',
         ['Conflict edges express same observed point, same-class family capacity or reversed nonzero reference-axis ordering; roles may be reused in other local Ward domains.',
          'At most16 nodes use complete independent-set enumeration; larger sets use the original32 beam without a hidden node cap. A selected edge is stable if removing every other selected edge preserves it in the recomputed optimum.'])
CONTROLS['control_B098_greedy_independent_set']=lambda ep:run(ep,'B098','H',lambda f:_b098(f,True))
CONTROLS['control_B098_no_stability']=lambda ep:run(ep,'B098','H',lambda f:_b098(f,no_stability=True))


def _simplex_projection(vertices,point):
    # Exact active-set simplex least squares, including all boundary faces.
    s=len(vertices);best=None
    if np.linalg.matrix_rank(vertices[1:]-vertices[:1],tol=1e-10)<s-1:return None
    for mask in range(1,1<<s):
        ids=np.flatnonzero([(mask>>i)&1 for i in range(s)]);V=vertices[ids]
        if len(ids)==1:a=np.ones(1)
        else:
            base=V[-1];solution=np.linalg.lstsq((V[:-1]-base).T,point-base,rcond=1e-10)[0];a=np.r_[solution,1-solution.sum()]
        if np.min(a)<-1e-10:continue
        alpha=np.zeros(s);alpha[ids]=np.maximum(a,0);alpha/=alpha.sum();residual=float(np.sum((alpha@vertices-point)**2))
        key=(residual,tuple(alpha))
        if best is None or key<best[0]:best=(key,alpha)
    return None if best is None else(best[0][0],best[1])


def _b085_arrays(f,threshold=np.inf):
    mixed=np.flatnonzero((f.coverage>.1)&(f.coverage<.9));tm=np.argsort(-f.profile[:,mixed],axis=1,kind='stable')[:,:4]if len(mixed)else np.empty((len(f.x),0),int)
    score=f.ell.copy();residual=np.full(len(f.x),np.inf)
    for C in f.regions:
        for q in C:
            anchors=np.unique(np.r_[f.fg[f.top_f[q]],f.bg[f.top_b[q]],mixed[tm[q]]]);best=None
            for s in(3,4):
                for ids in itertools.combinations(anchors,s):
                    ids=np.asarray(ids);cov=f.coverage[ids]
                    if not np.any(cov>=.9)or not np.any(cov<=.1):continue
                    reconstructed=_simplex_projection(f.r[ids],f.x[q])
                    if reconstructed is None:continue
                    error,alpha=reconstructed;key=(error,tuple(ids))
                    if best is None or key<best[0]:best=(key,2*float(alpha@cov)-1)
            if best is not None and best[0][0]<residual[q]:
                residual[q]=best[0][0]
                if residual[q]<=threshold:score[q]=best[1]
    return score,residual


def _b085(ep,unrestricted=False):
    p=prepare_b(ep);deg=p.degenerate()
    if deg is not None:return Result(deg[0].reshape(ep.q_hw),0.,info={'method_id':'B085','degenerate':deg[1]})
    if not p.structure_active():return finish_b(ep,p.frame().ell,'B085','D',info={'structure_active':False})
    observations=[];all_errors=[]
    for fold in range(4):
        f=p.frame(True,fold);proposed,error=_b085_arrays(f);held=(p.rb==fold)&(p.rv>0);observations.append((f,proposed,error,held));all_errors.extend(error[held&np.isfinite(error)])
    threshold=None
    if all_errors:
        choices=np.unique(np.quantile(all_errors,[.1,.25,.5,.75,.9]));loss=[]
        for t in choices:
            numerator=np.zeros(2);denominator=np.zeros(2)
            for f,s,e,held in observations:
                pred=np.where(e<=t,s,f.ell)>0;c=p.c[held];w=p.rv[held]
                numerator+=np.array([np.sum(w*c*~pred[held]),np.sum(w*(1-c)*pred[held])]);denominator+=np.array([np.sum(w*c),np.sum(w*(1-c))])
            loss.append(float(np.mean(numerator/denominator)))
        threshold=float(choices[min(range(len(choices)),key=lambda i:(loss[i],choices[i]))])
    if unrestricted:threshold=np.inf
    f=p.frame();z=f.ell if threshold is None else _b085_arrays(f,threshold)[0]
    return finish_b(ep,z,'B085','D',info={'residual_threshold':threshold,'source_folds_share_encoder_context':True,'implementation_assumptions':ASSUMPTIONS['B085']})


register('B085',_b085,'D',
         ['Each candidate point uses its actual top-four pure-FG, pure-BG and mixed-anchor correspondences; all mixed-role3/4-vertex combinations are enumerated and exact active-set nonnegative sum-one least squares selects the smallest residual, ties stable anchor IDs.',
          'Degenerate affine vertex sets abstain. Residual gate is chosen from the five source-heldout residual quantiles by complete class-balanced classification risk; smaller acceptance radius wins ties. No-anchor or gate-missing points retain ell.'])
CONTROLS['control_B085_unrestricted_local_simplex']=lambda ep:_b085(ep,True)


def _b091(f,only_pairs=False):
    # Fixed reference background tuples that survive independent source blocks.
    ids=f.bg;labels=f.blabels;ry,rx=np.unravel_index(f.r_ids[ids],f.ep.r_hw);sourceblock=(ry*2//f.ep.r_hw[0])*2+rx*2//f.ep.r_hw[1]
    templates={};lookup={int(f.r_ids[i]):(int(label),int(block))for i,label,block in zip(ids,labels,sourceblock)}
    i,j=edges4(f.ep.r_hw);neighbors={int(r):set()for r in lookup}
    for a,b in zip(i,j):
        if int(a)in lookup and int(b)in lookup:neighbors[int(a)].add(int(b));neighbors[int(b)].add(int(a))
    for center,around in neighbors.items():
        triples=[(center,a)for a in around]
        if not only_pairs:triples += [(a,center,b)for a,b in itertools.combinations(sorted(around),2)]
        for tuple_ids in triples:
            mode=tuple(sorted(lookup[r][0]for r in tuple_ids));templates.setdefault(mode,set()).add(lookup[center][1])
    permitted={mode for mode,seen in templates.items()if len(seen)>=2}
    purefg=f.fg;response=dot(f.r[purefg],f.bcenters)
    fg_mode=np.argmax(response,axis=1)if len(purefg)else np.empty(0,int)
    fglookup={int(f.r_ids[r]):int(l)for r,l in zip(purefg,fg_mode)}
    fgneighbors={int(r):set()for r in fglookup}
    for a,b in zip(i,j):
        if int(a)in fglookup and int(b)in fglookup:fgneighbors[int(a)].add(int(b));fgneighbors[int(b)].add(int(a))
    veto=set()
    for center,around in fgneighbors.items():
        for path in[(center,a)for a in around]+([]if only_pairs else[(a,center,b)for a,b in itertools.combinations(sorted(around),2)]):
            veto.add(tuple(sorted(fglookup[r]for r in path)))
    permitted-=veto
    if not permitted:return np.zeros(len(f.x))
    fglabels=np.argmax(f.role_f,axis=1);bglabels=np.argmax(f.role_b,axis=1);adv=f.role_b.max(axis=1)-f.role_f.max(axis=1)
    result=np.zeros(len(f.x));ii,jj=edges4(f.hw,f.valid);qneighbors=[[]for _ in f.x]
    for a,b in zip(ii,jj):qneighbors[int(a)].append(int(b));qneighbors[int(b)].append(int(a))
    for q,around in enumerate(qneighbors):
        for points in [(q,a)for a in around]+([]if only_pairs else[(a,q,b)for a,b in itertools.combinations(around,2)]):
            mode=tuple(sorted(bglabels[list(points)]))
            if mode in permitted and np.all(adv[list(points)]>0):
                for point in points:result[point]=min(result[point],-adv[point])
    return result


register('B091',lambda ep:run(ep,'B091','Z',_b091),'Z',
         ['Forbidden two/three-mode background motifs are physical reference paths present in at least two absolute2-by2 source blocks; query motifs also require direct background advantage at every affected vertex.',
          'A reference foreground occurrence vetoes a forbidden pair; unmatched motifs have zero correction. Three-vertex foreground veto is evaluated by the same physical path shape.'])
CONTROLS['control_B091_pairs_only']=lambda ep:run(ep,'B091','Z',lambda f:_b091(f,True))


def _b089(f,single=False):
    valid=np.flatnonzero(f.valid>0);reverse=dot(f.r,f.x[valid]);bestq=valid[np.argsort(-reverse,axis=1,kind='stable')[:,:2]]
    back=np.argmax(f.profile,axis=1);background=f.profile[:,f.bg].max(axis=1)-f.profile[:,f.fg].max(axis=1);out=np.zeros(len(f.x))
    ry,rx=np.unravel_index(f.r_ids,f.ep.r_hw);rb=ry*2//f.ep.r_hw[0]*2+rx*2//f.ep.r_hw[1]
    for q in valid:
        if background[q]<=0:continue
        origins=np.argsort(-f.profile[q],kind='stable')[:min(8,len(f.r))];witness={}
        for r in origins:
            if f.coverage[r]<.9:continue
            for other in bestq[r]:
                returned=back[other]
                if f.coverage[returned]<=.1 and background[other]>0:witness.setdefault(int(rb[r]),set()).add(int(other))
        independent=len(witness)>= (1 if single else 2)and len(set().union(*witness.values()))>=(1 if single else 2)
        if independent:out[q]=-background[q]
    return out


register('B089',lambda ep:run(ep,'B089','Z',_b089),'Z',
         ['Chains are actual Q-to-reference then reference-to-top-two-Q then Q-to-reference nearest correspondences; reference-foreground to returned-background role conflict requires direct background advantage on both observed query endpoints.',
          'At least two different reference spatial blocks and two distinct returned query endpoints must witness the conflict; four-fold reconstruction additionally removes each held source block and its buffer. Pure FG cycles add no positive bonus.'])
CONTROLS['control_B089_single_conflict_chain']=lambda ep:run(ep,'B089','Z',lambda f:_b089(f,True))


def _b092(f,no_pose=False):
    # Frozen two-class books and units; no query-derived dictionary entries.
    families=[f.fcenters,f.bcenters];labels=[f.flabels,f.blabels];roles=[f.fg,f.bg]
    from .b_helpers_076_100 import boundary_excluded_fold
    residual_units=[]
    for role in roles:
        residual=[]
        for block in range(4):
            held=np.isin(f.r_ids[role],np.flatnonzero((np.indices(f.ep.r_hw)[0]*2//f.ep.r_hw[0]*2+np.indices(f.ep.r_hw)[1]*2//f.ep.r_hw[1]).ravel()==block))
            train=role[~boundary_excluded_fold(f.ep.r_hw,block)[f.r_ids[role]]]
            if held.any()and len(train):
                centers,_=role_cluster(f.r[train],min(8,len(train)))
                residual.extend(np.maximum(1-dot(f.r[role[held]],centers).max(axis=1),0))
        residual_units.append(max(float(np.median(residual))if residual else 0.,1e-12))
    energies=[];fields=[];counts=[]
    for C in f.regions:
        costs=[]
        for side,foreground in enumerate((True,False)):
            own=f.role_f if foreground else f.role_b;k=own.shape[1]
            role_ids=f.fg if foreground else f.bg;rolelabels=f.flabels if foreground else f.blabels
            reference=np.array([f.geometry_r[role_ids[rolelabels==r]].mean(axis=0)for r in range(k)])
            initial=np.argmax(own[C],axis=1)
            shift=np.mean(f.geometry_q[C]-reference[initial],axis=0)
            posefee=0. if no_pose else np.log(2+np.linalg.norm(shift)*max(f.hw))
            residual=np.maximum(1-own[C],0)
            descriptor=np.log1p(np.floor(residual/residual_units[side]+.5))
            displacement=f.geometry_q[C,None]-reference[None]-shift
            geometric=np.zeros_like(descriptor)if no_pose else np.log1p(np.floor(np.abs(displacement)*np.array(f.hw)+.5)).sum(axis=-1)
            unitcost=np.log(max(k,1))+descriptor+geometric+posefee/len(C)
            costs.append(unitcost.min(axis=1))
        a,b=costs;fields.append(b-a);energies.append(float(np.sum(a+b)));counts.append(len(C))
    return owner(f,energies,fields,counts)


register('B092',lambda ep:run(ep,'B092','D',_b092),'D',
         ['Two fixed pure-reference books code role identity with log K and quantized residual/displacement with log(1+integer bin); residual units are the median genuine nested-reference spatial-leaveout prototype residual, floored only at1e-12, and displacement units are native patch spacings.',
          'Every local explanation pays its own translation pose code, split equally across actual mapped vertices; complete two-class unit code differences are emitted only where both classes explain the point, and all other points retain ell.'])
CONTROLS['control_B092_no_pose_code']=lambda ep:run(ep,'B092','D',lambda f:_b092(f,True))


def _b090_frame(p,source=False,fold=None):
    # BPrepared's no-profile path does not perform any cross-part dot products.
    f=p.frame(source,fold,compute_profile=False)
    return f


def _b090_field(f,selector='disagreement',tolerance=.0):
    foreground=unit(np.average(f.r,axis=0,weights=f.coverage*f.weights));background=unit(np.average(f.r,axis=0,weights=(1-f.coverage)*f.weights))
    initial=dot(f.x,np.array([foreground,background]));base=initial[:,0]-initial[:,1]
    internal,label=role_cluster(f.x,np.minimum(16,len(f.x)));H=max(1,min(16,len(internal)))
    # Hypotheses are partial query-mode identities, predicted only from the two
    # observed mean responses and internal query membership, not missing QxR.
    hypotheses=np.ones(H,bool);reference_class=(f.coverage>=.5).astype(int);geometry=f.geometry_r
    responses=[];queried=[];rng=np.random.default_rng(90)
    for step in range(min(32,len(f.r))):
        candidates=[r for r in range(len(f.r))if r not in queried]
        if not candidates:break
        if selector=='random':chosen=candidates[int(rng.integers(len(candidates)))]
        elif selector=='confidence':
            scores=dot(f.r[candidates],np.array([foreground,background]));chosen=candidates[int(np.argmax(np.abs(scores[:,0]-scores[:,1])))]
        elif selector=='farthest':
            if queried:distance=1-dot(f.r[candidates],f.r[queried]).max(axis=1);chosen=candidates[int(np.argmax(distance))]
            else:chosen=candidates[0]
        else:
            active=np.flatnonzero(hypotheses)
            if not len(active):active=np.arange(H)
            predicted=[]
            for h in active:
                mode=(label==h).astype(float);own=initial[:,0]if reference_class[candidates[0]]else initial[:,1]
                # A reference part's source location changes its predicted support
                # among candidate local explanations; no actual missing response.
                centroid=f.geometry_q[label==h].mean(axis=0)
                amplitude=np.exp(-np.sum((geometry[candidates]-centroid)**2,axis=1))
                predicted.append(amplitude*np.mean(np.where(mode>0,initial[:,0],initial[:,1])))
            disagreement=np.var(predicted,axis=0);chosen=candidates[int(np.argmax(disagreement))]
        observed=dot(f.x,f.r[[chosen]])[:,0];queried.append(chosen);responses.append(observed)
        # Reject only hypotheses whose mean-response prediction is outside the
        # observed reference reconstruction scale; keep a zero version set legal.
        for h in np.flatnonzero(hypotheses):
            predicted=np.where(label==h,initial[:,0],initial[:,1])
            if np.mean((observed-predicted)**2)>tolerance**2:hypotheses[h]=False
    matrix=np.column_stack(responses)if responses else np.empty((len(f.x),0));seen=np.array(queried,int);score=base.copy()
    if len(seen):
        fw=f.coverage[seen]*f.weights[seen];bw=(1-f.coverage[seen])*f.weights[seen]
        if fw.sum()>0 and bw.sum()>0:score=logmean(matrix,fw)-logmean(matrix,bw)
    return score,{'actual_cross_part_queries':len(queried),'actual_query_reference_dot_products':len(f.x)*(2+len(queried)),'full_cross_affinity_precomputed':False,'retained_response_values':len(f.x)*len(queried),'surviving_hypotheses':int(hypotheses.sum())}


def _b090(ep,selector='disagreement'):
    p=prepare_b(ep)
    # Degenerate masks still use their precisely specified ordinary direct chain;
    # only normal active input has the32-part cost claim.
    deg=p.degenerate()
    if deg is not None:return Result(deg[0].reshape(ep.q_hw),0.,info={'method_id':'B090','degenerate':deg[1]})
    if not p.structure_active():
        z=p.frame().ell
        return Result(z.reshape(ep.q_hw),0.,info={'method_id':'B090','structure_active':False,'inactive_full_ell_cost':'O(NMd); no32-part cost claim'})
    frames=[_b090_frame(p,True,fold)for fold in range(4)];errors=[]
    for fold,frame in enumerate(frames):
        fg=unit(np.average(frame.r,axis=0,weights=frame.coverage*frame.weights));bg=unit(np.average(frame.r,axis=0,weights=(1-frame.coverage)*frame.weights))
        initial=dot(frame.x,np.array([fg,bg]));held=(p.rb==fold)&(p.rv>0)
        errors.extend(np.maximum(-(2*p.c[held]-1)*(initial[held,0]-initial[held,1]),0))
    thresholds=np.unique(np.quantile(errors,[.1,.25,.5,.75,.9]));loss=[]
    for t in thresholds:
        mistakes=np.zeros(2);total=np.zeros(2)
        for fold,frame in enumerate(frames):
            score,_=_b090_field(frame,selector,float(t));held=(p.rb==fold)&(p.rv>0);y=score[held]>0;c=p.c[held];w=p.rv[held]
            mistakes += [np.sum(w*c*~y),np.sum(w*(1-c)*y)];total += [np.sum(w*c),np.sum(w*(1-c))]
        loss.append(float(np.mean(mistakes/total)))
    tolerance=float(thresholds[min(range(len(thresholds)),key=lambda i:(loss[i],-thresholds[i]))])
    f=_b090_frame(p);z,info=_b090_field(f,selector,tolerance);info.update(method_id='B090',qgt_access=False,source_error_tolerance=tolerance,source_calibration='five finite source residual quantiles by four-fold class-balanced complete-field risk',implementation_assumptions=ASSUMPTIONS['B090'])
    return Result(z.reshape(ep.q_hw),0.,info=info)


register('B090',_b090,'D',
         ['Reference parts are actual bank tokens. Up to16 hypotheses assign foreground identity to one internal query mode each; missing response predictions use only the two observed reference-class-mean fields and source/query normalized spatial geometry.',
          'Prediction disagreement chooses the next unobserved actual part; observed part responses reject hypotheses exceeding the source-only error tolerance selected from the five heldout wrong-sign mean-field residual quantiles by class-balanced error. If both seen class masses are available, their normalized log-mean responses give the complete field; otherwise use the two initial means.'])
CONTROLS['control_B090_random32']=lambda ep:_b090(ep,'random')
CONTROLS['control_B090_confidence32']=lambda ep:_b090(ep,'confidence')
CONTROLS['control_B090_farthest32']=lambda ep:_b090(ep,'farthest')


def _triplet_fit(f,A,B,roles):
    optionsA,_=choices(f,A,True);optionsB,_=choices(f,B,True);states=[(0.,(),())]
    for role in roles:
        expanded=[]
        for cost,left,right in states:
            for a,b in itertools.product(optionsA[role],optionsB[role]):
                if a in left or b in right:continue
                value=(2-f.role_f[a,role]-f.role_f[b,role]+1-f.x[a]@f.x[b])/3
                expanded.append((float(cost+value),left+(a,),right+(b,)))
        expanded.sort(key=lambda state:(state[0],state[1],state[2]));states=expanded[:32]
        if not states:return None
    return states[0]


def _b100(f,no_holdout=False,two_way=False):
    energies=[];fields=[];counts=[]
    for C in f.regions:
        e,v,state=_dual_candidate(f,C);energies.append(e);fields.append(v.copy());counts.append(len(C))
    field=f.ell.copy();best=np.full(len(f.x),np.inf)
    for left,right in itertools.combinations(range(len(f.regions)),2):
        A,B=f.regions[left],f.regions[right]
        if np.intersect1d(A,B).size:continue
        ef,ma,va=partial_match(f,A,True);eg,mb,vb=partial_match(f,B,True)
        common=[r for r,(a,b)in enumerate(zip(ma,mb))if a>=0 and b>=0 and va[np.searchsorted(A,a)]>0 and vb[np.searchsorted(B,b)]>0]
        if len(common)<(2 if no_holdout else 3):continue
        paircost=dot(f.x[A],f.x[B]);ab=np.argmax(paircost,axis=1);ba=np.argmax(paircost,axis=0)
        cycle=[r for r in common if B[ab[np.searchsorted(A,ma[r])]]==mb[r]and A[ba[np.searchsorted(B,mb[r])]]==ma[r]]
        if len(cycle)<2:continue
        for held in common:
            used=[r for r in cycle if r!=held]
            if len(used)<2 and not no_holdout:continue
            if two_way:continue
            # Joint3-way fit estimates a feature translation only on cycle roles,
            # then evaluates a genuinely excluded family before writing its point.
            joint=_triplet_fit(f,A,B,used)
            if joint is None:continue
            joint_cost,fitA,fitB=joint
            direct_cost=np.mean([(2-f.role_f[ma[r],r]-f.role_f[mb[r],r])/2 for r in used])if used else np.inf
            if joint_cost/max(len(used),1)>=direct_cost:continue
            differences=np.array([f.x[b]-f.x[a]for a,b in zip(fitA,fitB)]);delta=differences.mean(axis=0)
            qA,qB=ma[held],mb[held];direct=np.linalg.norm(f.fcenters[held]-f.x[qB])**2
            translated=np.linalg.norm(unit(f.x[qA]+delta)-f.x[qB])**2
            improvement=direct-translated
            if improvement<=0:continue
            energy=ef+eg+sum(1-f.x[ma[r]]@f.x[mb[r]]for r in used)
            for q in(qA,qB):
                if energy<best[q]:
                    margin=f.role_f[q,held]-f.role_b[q].max();field[q]=margin*improvement;best[q]=energy
    if two_way:return _dual_field(f,direct=True)
    return field


register('B100',lambda ep:run(ep,'B100','D',_b100),'D',
         ['Each disjoint query-candidate pair must have at least two mutually nearest cross-query role witnesses with independent positive direct-reference margins; a third family is needed for the ordinary heldout test.',
          'The three-way32-beam joint fit minimizes the mean of both reference-family residuals and actual cross-query feature residual under distinct-point capacities on each side, only on independently cycle-consistent training roles. It must beat the direct two-reference-leg mean cost and lower excluded-family reconstruction residual before writing heldout points. All other points retain ell.'])
CONTROLS['control_B100_direct_two_way']=lambda ep:run(ep,'B100','D',lambda f:_b100(f,two_way=True))
CONTROLS['control_B100_no_heldout_family']=lambda ep:run(ep,'B100','D',lambda f:_b100(f,no_holdout=True))
