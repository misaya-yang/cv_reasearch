"""Spatial reference A-card kernels; no query label or surrogate encoder."""
from __future__ import annotations

import heapq
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from .a_helpers_001_050 import (EPS,Frame,b0,fps,nearest_mean_distance,rank_basis,
    spatial_blocks,sqdist,threshold,unit)


def _plain(frame):
    return lambda q,ids,role:b0(frame,q),{"mechanism":"A_B0_5NN"}


def _minimal_transport(vector,ta,tb):
    """Principal-angle minimal rotations on tangent span, implicit D×D map."""
    rank=min(ta.shape[1],tb.shape[1])
    if not rank:return vector.copy()
    left,cosine,right=np.linalg.svd(ta.T@tb,full_matrices=False)
    aa=ta@left[:,:rank];bb=tb@right.T[:,:rank]
    out=vector.copy()
    for a,b,c in zip(aa.T,bb.T,cosine[:rank]):
        c=float(np.clip(c,0,1));s=np.sqrt(max(0,1-c*c))
        if s<1e-8:continue
        orth=(b-c*a)/s
        va,vb=out@a,out@orth
        out+=((c-1)*va-s*vb)*a+(s*va+(c-1)*vb)*orth
    return out


def fit_a023(frame,transport=True):
    f,b,fi,bi,_=frame.banks()
    if len(f)<3 or not len(b) or np.any(fi<0) or np.any(bi<0):return _plain(frame)
    distance=sqdist(f,f);np.fill_diagonal(distance,np.inf)
    nearest=np.argsort(distance,axis=1,kind="stable")[:,:min(8,len(f)-1)]
    adjacency=[[] for _ in f]
    for i,neighbors in enumerate(nearest):
        for j in neighbors:
            if i in nearest[j]:adjacency[i].append(int(j))
    tangents=[rank_basis(f[near]-f[i],4) for i,near in enumerate(nearest)]
    yy,xx=np.indices(frame.ep.r_hw);coords=np.c_[yy.ravel(),xx.ravel()]
    normals=[None for _ in f]
    radius=np.array([float(np.quantile(distance[i,nearest[i]],.9)) for i in range(len(f))])
    for i,index in enumerate(fi):
        eligible=np.flatnonzero(np.sum((coords[bi]-coords[index])**2,1)<=4)
        if len(eligible):
            j=eligible[np.argmin(sqdist(f[i:i+1],b[eligible])[0])]
            n=f[i]-b[j]
            if np.linalg.norm(n)>EPS:
                held=frame.train & ~np.isin(frame.blocks,[frame.blocks[index],frame.blocks[bi[j]]])
                wf,wb=frame.wf[held],frame.wb[held]
                if wf.sum()>0 and wb.sum()>0:
                    response=frame.x[held]@n;positive=response>0
                    error=.5*(wf[~positive].sum()/wf.sum()+wb[positive].sum()/wb.sum())
                    if error<.5:normals[i]=unit(n)
    known=[i for i,v in enumerate(normals) if v is not None]
    if not known:return _plain(frame)
    transported=0;loop_rejections=0
    if transport:
        for target in range(len(f)):
            if normals[target] is not None:continue
            cost=np.full(len(f),np.inf);previous=np.full(len(f),-1,int)
            queue=[]
            for source in known:cost[source]=0.;heapq.heappush(queue,(0.,source))
            while queue:
                value,i=heapq.heappop(queue)
                if value!=cost[i]:continue
                if i==target:break
                for j in adjacency[i]:
                    proposed=value+np.sqrt(distance[i,j])
                    if proposed<cost[j]:cost[j]=proposed;previous[j]=i;heapq.heappush(queue,(proposed,j))
            if not np.isfinite(cost[target]):continue
            path=[target]
            while previous[path[-1]]>=0:path.append(int(previous[path[-1]]))
            path.reverse();normal=normals[path[0]].copy();pathnormals={path[0]:normal.copy()}
            for i,j in zip(path,path[1:]):
                normal=_minimal_transport(normal,tangents[i],tangents[j]);pathnormals[j]=normal.copy()
            # Available one-edge detours create an actual holonomy consistency
            # test; not an automatic identity product from a preset global map.
            valid=True
            for i,j in zip(path,path[1:]):
                common=sorted(set(adjacency[i])&set(adjacency[j]))
                for k in common:
                    local=pathnormals[i]
                    direct=_minimal_transport(local,tangents[i],tangents[j])
                    detour=_minimal_transport(_minimal_transport(local,tangents[i],tangents[k]),tangents[k],tangents[j])
                    cosine=float(unit(direct)@unit(detour))
                    if cosine<np.cos(np.deg2rad(30)):valid=False;break
                if not valid:break
            if valid:normals[target]=unit(normal);transported+=1
            else:loop_rejections+=1
    active=np.array([i for i,n in enumerate(normals) if n is not None],int)
    vectors=np.array([normals[i] for i in active])
    def predict(q,ids,role):
        base,info=b0(frame,q);route=np.argmin(sqdist(q,f[active]),1)
        selected=active[route];hit=sqdist(q,f)[np.arange(len(q)),selected]<=radius[selected]
        base[hit]=np.sum(q[hit]*vectors[route[hit]],1)
        return base,dict(info,normal_domain_points=int(hit.sum()),points=len(q),_fallback_mask=~hit)
    return predict,{"graph_nodes":len(f),"known_spatial_BG_normals":len(known),"transported_normals":transported,
        "loop_angle_rejections":loop_rejections,"tangent_rank_cap":4,"transport":transport}


def _involution(frame,debug=None):
    debug={} if debug is None else debug
    f,b,fi,bi,_=frame.banks()
    if len(f)<8 or not len(b) or np.any(fi<0):
        debug["degeneration"]="insufficient_observed_pure_role_anchors";return None
    yy,xx=np.indices(frame.ep.r_hw);coords=np.c_[yy.ravel(),xx.ravel()]
    center=(coords[frame.fids].min(0)+coords[frame.fids].max(0))/2
    feature_distance=sqdist(f,f);np.fill_diagonal(feature_distance,np.inf)
    feature_neighbors=np.argsort(feature_distance,axis=1,kind="stable")[:,:min(8,len(f)-1)]
    pairs=[]
    for axis in (0,1):
        reflected=coords[fi].copy().astype(float);reflected[:,axis]=2*center[axis]-reflected[:,axis]
        positional=sqdist(reflected,coords[fi]);nearest=np.argmin(positional,1)
        for i,j in enumerate(nearest):
            if i>=j or nearest[j]!=i or positional[i,j]>1 or j not in feature_neighbors[i] or i not in feature_neighbors[j]:continue
            pairs.append((i,int(j)))
    pairs=sorted(set(pairs))
    debug["mirror_pairs"]=len(pairs)
    if len(pairs)<4 or len(set(frame.blocks[fi[np.array(pairs).ravel()]]))<2:
        debug["degeneration"]="fewer_than_four_mutual_mirror_pairs_or_two_blocks";return None
    delta=np.array([f[i]-f[j] for i,j in pairs]);u=rank_basis(delta)
    if not u.shape[1]:debug["degeneration"]="zero_pair_difference_span";return None
    p=np.array([f[i]@u for i,j in pairs]);q=np.array([f[j]@u for i,j in pairs])
    covariance=(p.T@q+q.T@p)/2
    value,vectors=np.linalg.eigh(covariance)
    signs=np.where(value<0,-1.,1.)
    involution=(vectors*signs)@vectors.T
    used={i for pair in pairs for i in pair};unpaired=[i for i in range(len(f)) if i not in used]
    virtual=f[unpaired]+f[unpaired]@u@(involution-np.eye(u.shape[1]))@u.T
    if not len(virtual):debug["degeneration"]="no_unpaired_anchor_to_generate";return None
    # Actual negative-distance guard; generated endpoints do not enter as their
    # own positive evidence and cannot outrun a closer original BG explanation.
    safe=nearest_mean_distance(virtual,f,1)<nearest_mean_distance(virtual,b,1)
    virtual=virtual[safe]
    return f,b,u,involution,virtual,pairs


def fit_a024(frame,random_involution=False):
    debug={};packet=_involution(frame,debug)
    if packet is None:
        predict,info=_plain(frame)
        return predict,dict(info,card_id="A024",**debug)
    f,b,u,t,virtual,pairs=packet
    errors=[]
    # Every internal validation recreates pairs/T/virtuals without held labels.
    for block in range(4):
        held=frame.train&(frame.blocks==block)
        wf,wb=frame.wf[held],frame.wb[held]
        if wf.sum()<=0 or wb.sum()<=0:continue
        inner=Frame.make(frame.ep,frame.train&(frame.blocks!=block));p=_involution(inner)
        if p is None or not len(p[4]):continue
        ff,bb,_,_,vv,_=p
        raw=(nearest_mean_distance(frame.x[held],bb)-nearest_mean_distance(frame.x[held],ff))/2
        extra=(nearest_mean_distance(frame.x[held],bb)-nearest_mean_distance(frame.x[held],np.r_[ff,vv]))/2
        def error(score):return .5*(wf[score<=0].sum()/wf.sum()+wb[score>0].sum()/wb.sum())
        errors.append((float(error(raw)),float(error(extra))))
    if not errors or any(new>old+1e-12 for old,new in errors):
        predict,info=_plain(frame)
        return predict,dict(info,virtual_validation_failed=True,validation_errors=errors,candidate_virtuals=len(virtual))
    if random_involution:
        original=virtual+virtual@u@(t-np.eye(u.shape[1]))@u.T
        signs=np.random.default_rng(0).choice([-1.,1.],u.shape[1]);rot=np.linalg.qr(np.random.default_rng(0).normal(size=(u.shape[1],u.shape[1])))[0]
        t=(rot*signs)@rot.T
        virtual=original+original@u@(t-np.eye(u.shape[1]))@u.T
    if not len(virtual):
        predict,info=_plain(frame)
        return predict,dict(info,card_id="A024",degeneration="all_virtuals_rejected_by_BG_guard")
    ff=np.r_[f,virtual]
    def predict(q,ids,role):return (nearest_mean_distance(q,b)-nearest_mean_distance(q,ff))/2,{"virtual_FG":len(virtual)}
    return predict,{"mirror_pairs":len(pairs),"difference_span_rank":u.shape[1],"virtual_FG":len(virtual),
        "orthogonal_error":float(np.linalg.norm(t.T@t-np.eye(len(t)))),"involution_error":float(np.linalg.norm(t@t-np.eye(len(t)))),
        "cross_block_validation_errors":errors,"random_involution":random_involution}


def fit_a025(frame,remove=True):
    f,b,fi,bi,_=frame.banks()
    if len(frame.fids)<3 or len(frame.bids)<3 or np.any(fi<0) or np.any(bi<0):return _plain(frame)
    yy,xx=np.indices(frame.ep.r_hw);coords=np.c_[yy.ravel(),xx.ravel()]
    # Distance to nearest KNOWN opposite side, not held-out cells treated as BG.
    depth=np.zeros(len(frame.x))
    depth[frame.fids]=np.sqrt(cdist(coords[frame.fids],coords[frame.bids],'sqeuclidean').min(1))
    depth[frame.bids]=-np.sqrt(cdist(coords[frame.bids],coords[frame.fids],'sqeuclidean').min(1))
    def curve_coeff(ids):
        d=depth[ids];design=np.c_[np.ones(len(ids)),d,d*d]
        return np.linalg.solve(design.T@design+np.eye(3),design.T@frame.x[ids])
    cf,cb=curve_coeff(frame.fids),curve_coeff(frame.bids)
    shared=np.zeros_like(cf);accepted=[]
    for power in (1,2):
        cosine=float(unit(cf[power])@unit(cb[power]))
        if cosine>=.95 and remove:shared[power]=(cf[power]+cb[power])/2;accepted.append(power)
    def trend(d):return np.c_[np.ones(len(d)),d,d*d]@shared
    ff=f-trend(depth[fi]);bb=b-trend(depth[bi])
    samples=[np.quantile(depth[ids],[.1,.3,.5,.7,.9]) for ids in (frame.fids,frame.bids)]
    def predict(q,ids,role):
        scores=[]
        for bank,quantiles in zip((ff,bb),samples):
            cost=np.array([nearest_mean_distance(q-trend(np.full(len(q),d)),bank,1) for d in quantiles]).T
            scores.append(.07*(logsumexp(-cost/.07,axis=1)-np.log(5)))
        return scores[0]-scores[1],{"integrated_depth_values":5,"shared_curve_powers":accepted}
    return predict,{"shared_curve_powers":accepted,"shared_coeff_norms":np.linalg.norm(shared,axis=1).tolist(),
        "FG_depth_quantiles":samples[0].tolist(),"BG_depth_quantiles":samples[1].tolist(),
        "shared_intercept_always_zero":True,"remove_shared_term":remove}


def _pure_branch_filtration(bank,labels,blocks):
    """R merge forest: pure branch death is its first opposite-role contact."""
    n=len(bank);distance=np.sqrt(sqdist(bank,bank));ii,jj=np.triu_indices(n,1)
    order=np.lexsort((jj,ii,distance[ii,jj]))
    parent=list(range(2*n-1));nodes=[dict(members=[i],label=int(labels[i]),birth=0.,death=np.inf,children=[]) for i in range(n)]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def contact(i,t):
        stack=[i]
        while stack:
            j=stack.pop()
            if nodes[j]['label'] in (0,1) and not np.isfinite(nodes[j]['death']):nodes[j]['death']=t
            stack.extend(nodes[j]['children'])
    for k in order:
        a,b=root(int(ii[k])),root(int(jj[k]))
        if a==b:continue
        t=float(distance[ii[k],jj[k]]);la,lb=nodes[a]['label'],nodes[b]['label'];role=la if la==lb else -1
        if role==-1:contact(a,t);contact(b,t)
        j=len(nodes);nodes.append(dict(members=nodes[a]['members']+nodes[b]['members'],label=role,birth=t,death=np.inf,children=[a,b]))
        parent[a]=j;parent[b]=j
    safe=[]
    for node in nodes:
        if node['label'] not in (0,1) or not np.isfinite(node['death']) or node['death']<=node['birth']:continue
        members=np.array(node['members'],int)
        if len(set(blocks[members]))<2:continue
        safe.append((node['label'],members,node['birth'],node['death']))
    return distance,safe


def fit_a027(frame,multiradius=False,shuffle=False):
    f,b,fi,bi,_=frame.banks()
    if not len(f) or not len(b) or np.any(fi<0) or np.any(bi<0):return _plain(frame)
    bank=np.r_[f,b];labels=np.r_[np.ones(len(f),int),np.zeros(len(b),int)];blocks=frame.blocks[np.r_[fi,bi]]
    if shuffle:labels=np.random.default_rng(0).permutation(labels)
    distance,branches=_pure_branch_filtration(bank,labels,blocks)
    fg=[v for v in branches if v[0]==1];bg=[v for v in branches if v[0]==0]
    if not fg or not bg:
        predict,info=_plain(frame)
        return predict,dict(info,card_id='A027',degeneration='no_two_role_positive_crossblock_persistent_branches',FG_branches=len(fg),BG_branches=len(bg))
    radii=np.quantile(distance[np.triu_indices(len(bank),1)],[.1,.3,.5,.7,.9])
    def predict(q,ids,role):
        d=np.sqrt(sqdist(q,bank));nearest=[d[:,labels==c].min(1) for c in (0,1)]
        if multiradius:
            margins=[(d[:,labels==1]<=r).mean(1)-(d[:,labels==0]<=r).mean(1) for r in radii]
            return np.mean(margins,axis=0),{'radii':radii.tolist()}
        durations=[]
        for role,branchbank in ((1,fg),(0,bg)):
            out=np.zeros(len(q))
            for _,members,birth,death in branchbank:
                join=d[:,members].min(1)
                exit_=np.minimum(death,nearest[1-role])
                out=np.maximum(out,np.maximum(exit_-np.maximum(birth,join),0))
            durations.append(out)
        return durations[0]-durations[1],{'FG_positive_stay_points':int((durations[0]>0).sum()),'BG_positive_stay_points':int((durations[1]>0).sum()),'query_query_edges':0}
    return predict,{'anchors':len(bank),'FG_safe_branches':len(fg),'BG_safe_branches':len(bg),
        'lifespan_convention':'pure_component_formation_until_first_mixed_ancestor_contact',
        'reference_filtration_fixed_before_query':True,'label_shuffle_control':shuffle,'multiradius_control':multiradius}


def install(register,requirements):
    register("A023",fit_a023,(
        "64FPS FG mutual8NN graph, PCA tangent rank<=4; BGnormal nearestpureBG within2patch andbalancederror<.5 inknownblocks containing neitherendpoint.",
        "Implicit principal-angle minimalrotations, shortestnonrepeat graphpath; triangle-detour holonomy normalangle>30deg rejects; querynearestnormal onlywithinFGlocalNNradiusq.9.",),
        (("no_transport_direct_normals",lambda f:fit_a023(f,transport=False)),))
    register("A024",fit_a024,(
        "Horizontal/vertical mirrors aboutFGcoordinatebbox center; mutualpositionpairs tolerance1patch AND mutualfeature8NN,>=4pairs across>=2blocks; orthogonalinvolution spectral sign ofsymcrosscov inpairdiffspan.",
        "Generate onlyunpairedFGanchor transforms, discardcloserBGexplanation; innerfourblock rebuildpairs/T/virtuals and permitonlynonincreasingzero-cut balanced error inEVERYeffectiveblock.",),
        (("same_rank_random_involution",lambda f:fit_a024(f,random_involution=True)),("original_5NN",_plain)))
    register("A025",fit_a025,(
        "Maskdepth isnearestKNOWN opposite-pure-side grid distance withFGpositivesign; noheldout-asBG EDT; rolecurve degree2 ridge1.",
        "Shared powers1/2 onlyif rolecoefficientcos>=.95, meanvector, nointercept removal; integrateq.1/.3/.5/.7/.9 eachroledepth byequalmeanexp(-distance2/.07), outputlogratio.",),
        (("same_depth_no_shared_removal",lambda f:fit_a025(f,remove=False)),("original_5NN",_plain)))
    register('A027',fit_a027,(
        'CompleteEuclidean pairdistance zero-dimensional unionfind on<=128 observedroleanchors; purecomponent birth=formationradius, death=firstmixedancestorcontact.',
        'Bothrolebranches requiremembersfrom>=2Rblocks andpositivelife. Qstay=max(0,min(branchdeath,nearestoppositeedge)−max(branchbirth,nearestmemberedge)); maxrole durationdifference.',
        'Rbranchforestfixed,noQ-Q. Puredescendants persist toBGcontact insteadordinaryH0 same-label eldermerge death; this semantic lifetime convention is an explicitimplementationassumption.',),
        (('same_anchors_multiradius_vote',lambda f:fit_a027(f,multiradius=True)),('same_count_labels_shuffled',lambda f:fit_a027(f,shuffle=True))))
