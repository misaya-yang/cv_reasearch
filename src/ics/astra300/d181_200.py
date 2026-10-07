"""Faithful original Astra D181--D200 complete CPU algorithms.

Method registries contain supplied IDs only. Controls are separately registered.
The companion evidence records unspecified constants and genuine-resource needs.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import genpareto
from .common import artifact, ArtifactUnavailable
from . import d_helpers_151_200 as dh
from . import d_helpers_181_200 as h


def _open_background(ep,cfg):
    # The card explicitly uses D151 as the no-new-subtype complete model.
    from .group_151_225 import _d151_builder
    scorer,info=_d151_builder(ep,cfg)
    return (dh.b0_field(ep) if scorer is None else scorer(ep.q)),dict(info,D151_unavailable=scorer is None)


def _profile_control(ep,cfg):
    return h.rbf_control(ep,cfg)


def _d181(ep,cfg,fixed=False):
    k,rank,strength=cfg;xr,xq,_=dh.project(ep);m=h.modes(ep,k,original=False)
    if m is None:return None,dict(reason='insufficient_source_modes')
    ((fc,fi,f),(bc,bi,b)),dirty=m;valid=ep.q_valid>0
    if valid.sum()<8:return None,dict(reason='insufficient_query_samples')
    shared,_=dh.source_modes(xr[ep.wvalid>0],rank,spherical=False,weights=ep.wvalid[ep.wvalid>0])
    qbg,_=dh.source_modes(xq[valid],min(8,k),spherical=False)
    if not len(shared) or not len(qbg):return None,dict(reason='insufficient_factor_support')
    # Duplicate soft role mass rather than hard-assigning a mixed R token.
    fid=np.flatnonzero(ep.wf>0);bid=np.flatnonzero(ep.wb>0)
    x=np.vstack((xr[fid],xr[bid],xq[valid]));mass=np.r_[ep.wf[fid]/ep.wf.sum(),ep.wb[bid]/ep.wb.sum(),np.full(valid.sum(),1./valid.sum())]
    na,nf=len(shared),len(fc);atoms=np.vstack((shared,fc,qbg));initial=atoms.copy();nt=len(atoms)
    forbidden=np.zeros((len(x),nt),bool);forbidden[len(fid):len(fid)+len(bid),na:na+nf]=True
    residual_scale=max(float(np.median(np.min(h.sqdist(xr[f],fc),axis=1))),1e-5)
    sparse=.1*residual_scale;best=(np.inf,atoms.copy());visited=[]
    for _ in range(20):
        codes,cost=h.nnls_codes(x,atoms,sparse,forbid=forbidden,steps=1)
        # Each source FG mode has an actual nonzero target factor coefficient.
        for j in range(nf):
            here=np.flatnonzero(fi[fid]==j)
            if len(here) and np.max(codes[here,na+j])<=1e-8:
                index=here[np.argmin(h.sqdist(x[here],atoms[na+j:na+j+1]).ravel())]
                codes[index,na+j]=1e-6
        objective=float(cost@mass+strength*np.sum((atoms[na:na+nf]-initial[na:na+nf])**2))
        visited.append(objective)
        if objective<best[0]:best=(objective,atoms.copy())
        gram=dh.mm(codes.T,mass[:,None]*codes)+1e-6*np.eye(nt)
        rhs=dh.mm(codes.T,mass[:,None]*x)
        for j in range(na,na+nf):gram[j,j]+=strength;rhs[j]+=strength*initial[j]
        updated=np.linalg.solve(gram,rhs)
        # Source-only bounded target drift, retaining original identity support.
        bound=np.sqrt(max(residual_scale,1e-5));delta=updated[na:na+nf]-initial[na:na+nf]
        delta*=np.minimum(1.,bound/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),h.EPS))
        updated[na:na+nf]=initial[na:na+nf]+delta
        if fixed:updated[na+nf:]=initial[na+nf:]
        atoms=updated
    atoms=best[1];without=np.r_[np.arange(na),np.arange(na+nf,nt)]
    _,cost0=h.nnls_codes(xq,atoms[without],sparse);_,cost1=h.nnls_codes(xq,atoms,sparse)
    return cost0-cost1,dict(iterations=20,best_objective=best[0],visited_objectives=visited,
        signed_dictionary=True,nonnegative_codes=True,source_BG_target_coefficients_zero=True,
        source_FG_each_mode_nonzero=True,target_factors=nf,shared_factors=na,query_BG_factors=len(qbg),
        query_BG_joint_update=not fixed,soft_coverage_fallback=dirty)


def d181(ep):return h.calibrate(ep,'D181',_d181)

def d181_control(ep):return h.calibrate(ep,'D181_control_fixed_BG_same_dictionary',lambda e,c:_d181(e,c,True))


def _top_fraction(score,positive,valid):
    ids=np.flatnonzero(valid & ((score>0) if positive else (score<0)))
    count=min(len(ids),max(1,int(np.ceil(.1*np.count_nonzero(valid)))))
    order=np.argsort(-score[ids] if positive else score[ids],kind='stable')
    return ids[order[:count]]


def _d182(ep,cfg,uniform=False):
    k,_,strength=cfg;k=min(k,8);m=h.modes(ep,k)
    if m is None:return None,dict(reason='insufficient_source_modes')
    data,dirty=m;xr,xq,_=dh.project(ep);head=dh.head(ep,xr)
    if head is None:return None,{}
    oof=np.full(len(ep.r),np.nan)
    for train,held in dh.folds(ep):
        model=dh.head(dh.restricted(ep,train),xr)
        if model is not None:oof[held]=dh.predict(model,xr[held])
    qmargin=dh.predict(head,xq);base=dh.b0_field(ep);block=dh.blocks(ep.r_hw)
    updated=[];audit=[]
    for positive,(centers,assignment,selected) in zip((True,False),data):
        centers=centers.copy();qids=_top_fraction(qmargin,positive,ep.q_valid>0)
        qids=qids[(base[qids]>0) if positive else (base[qids]<0)]
        qmode=np.argmax(dh.mm(ep.q[qids],centers.T),axis=1) if len(qids) else np.empty(0,int)
        picked=np.zeros(len(ep.r),bool)
        for group in np.unique(block):
            ids=_top_fraction(oof,positive,(block==group)&np.isfinite(oof))
            picked[ids]=True
        for j in range(len(centers)):
            mode=selected&(assignment==j);per=[int(np.sum(mode&(block==z))) for z in np.unique(block)]
            legal=sum(n>=8 for n in per)>=2
            probability=(int(np.sum(mode&picked))+1.)/(int(mode.sum())+2.) if legal else 1.
            ids=qids[qmode==j]
            if legal and len(ids):
                inverse=1. if uniform else min(10.,1./max(probability,.1))
                soft=expit(qmargin[ids]) if positive else expit(-qmargin[ids])
                weights=inverse*soft/max(len(qids),1)*strength
                centers[j]=h.unit(centers[j]+np.sum(ep.q[ids]*weights[:,None],axis=0))
            audit.append(dict(role='FG' if positive else 'BG',mode=j,selection_probability=probability,
                              independent_blocks_ge8=sum(n>=8 for n in per),updated_points=len(ids) if legal else 0))
        updated.append(centers)
    scale=max(h.bandwidth(ep),1e-5)
    ff=logsumexp(dh.mm(ep.q,updated[0].T)/scale,axis=1)-np.log(len(updated[0]))
    bb=logsumexp(dh.mm(ep.q,updated[1].T)/scale,axis=1)-np.log(len(updated[1]))
    return ff-bb,dict(selection_audit=audit,updates=1,equal_mode_class_mass=True,
                      inverse_probability=not uniform,soft_coverage_fallback=dirty)


def d182(ep):return h.calibrate(ep,'D182',_d182)

def d182_control(ep):return h.calibrate(ep,'D182_control_same_seeds_uniform_weights',lambda e,c:_d182(e,c,True))


def _d183(ep,cfg,global_weights=False):
    k,_,penalty=cfg;m=h.modes(ep,k)
    if m is None:return None,dict(reason='insufficient_source_modes')
    ((fc,_,f),(bc,_,b)),dirty=m;scale=h.bandwidth(ep)
    flog=-h.sqdist(ep.q,fc)/scale;blog=-h.sqdist(ep.q,bc)/scale
    leaf=np.max(flog,axis=1)-np.max(blog,axis=1);regions,info=h.candidate_regions(ep)
    updates=[];y,x=np.indices(ep.q_hw);parity=((y+x)%2).ravel();support=dh.source_radius(ep,b)
    bg_distance=dh.nearest_distance(dh.pair(ep),b)
    supported=(bg_distance<=support)&(dh.b0_field(ep)<0) if np.isfinite(support) else np.zeros(len(ep.q),bool)
    global_w=h.mixture_weights(flog[ep.q_valid>0],penalty)
    for ids in regions:
        if len(ids)<2:continue
        outer=h.ring(ep,ids);outer=outer[supported[outer]]
        negative=bc
        if len(outer)>=8:
            extra,_=dh.source_modes(ep.q[outer],min(k,8),weights=ep.q_valid[outer])
            if len(extra):negative=np.vstack((bc,extra))
        local_blog=-h.sqdist(ep.q[ids],negative)/scale
        field=leaf[ids].copy()
        for side in (0,1):
            fit=ids[parity[ids]==side];testmask=parity[ids]!=side
            if not len(fit) or not testmask.any():continue
            weights=global_w if global_weights else h.mixture_weights(flog[fit],penalty)
            field[testmask]=logsumexp(flog[ids[testmask]]+np.log(np.maximum(weights,1e-12)),axis=1)-logsumexp(local_blog[testmask],axis=1)+np.log(len(negative))
        updates.append((ids,field))
    return h.merge_field(leaf,updates),dict(**info,candidate_crossfit='checkerboard_half_fit_other_half_score_swapped',
        optional_part_weights_zero_allowed=True,local_BG_only_direct_source_supported=True,instances_share_weights=global_weights,
        soft_coverage_fallback=dirty)


def d183(ep):return h.calibrate(ep,'D183',_d183)

def d183_control(ep):return h.calibrate(ep,'D183_control_global_free_part_mixture',lambda e,c:_d183(e,c,True))


def _d184_types(ep,cfg):
    k,_,strength=cfg;m=h.modes(ep,k)
    if m is None:return None
    ((fc,_,f),(bc,_,b)),_=m;scale=h.bandwidth(ep);atts=h.attributes(ep,4)
    if atts is None:return None
    ar,aq,_,_,_=atts;g=aq.shape[1];fr=ar[f].mean(0);br=ar[b].mean(0)
    sf=np.maximum(np.var(ar[f],axis=0),1e-5);sb=np.maximum(np.var(ar[b],axis=0),1e-5)
    factor=(-h.sqdist(ep.q,h.unit(fc.mean(0))[None])+h.sqdist(ep.q,h.unit(bc.mean(0))[None])).ravel()/scale
    parts,part_info=_d183(ep,cfg)
    fg_per=(aq-fr)**2/sf;bg_per=(aq-br)**2/sb
    conjunction=np.min(bg_per-fg_per,axis=1)
    # Exactly the same quantized K*d parameter charge for all three families.
    cost=float(k*ep.r.shape[1]*np.log(256))*strength
    fields=[factor,parts,conjunction]
    reconstruction=[float(np.minimum(h.sqdist(ep.q,fc).min(1),h.sqdist(ep.q,bc).min(1)).sum()),
                    float(np.minimum(-logsumexp(-h.sqdist(ep.q,fc)/scale,axis=1)+np.log(len(fc)),
                                     -logsumexp(-h.sqdist(ep.q,bc)/scale,axis=1)+np.log(len(bc))).sum()*scale),
                    float(np.minimum(np.max(fg_per,axis=1),np.max(bg_per,axis=1)).sum()*scale)]
    return fields,np.asarray(reconstruction)+cost,dict(equal_parameter_charge=cost,code_precision_bits=8,types=['common_factor','optional_parts','attribute_conjunction'],**part_info)


def _d184(ep,cfg,fixed=False,average=False):
    result=_d184_types(ep,cfg)
    if result is None:return None,dict(reason='insufficient_explanation_types')
    fields,costs,info=result;source_risks=np.zeros(3);count=0
    for train,held in dh.folds(ep):
        reduced=h.pseudo_query(ep,train);row=_d184_types(reduced,cfg)
        if row is None:return None,dict(reason='source_type_unavailable')
        for j,field in enumerate(row[0]):source_risks[j]+=dh.risk(ep,field,held)
        count+=1
    winner=int(np.argmin(source_risks));near=np.flatnonzero(costs<=costs.min()+.01*max(abs(float(costs.min())),1.))
    if not fixed:
        winner=int(min(near,key=lambda j:(source_risks[j],j)))
    if average:
        field=np.mean([f/max(float(np.std(f)),h.EPS) for f in fields],axis=0)
    else:field=fields[winner]
    return field,dict(**info,selected_type=winner,query_code_objectives=costs.tolist(),
                       source_type_errors=source_risks.tolist(),near_tie_source_preferred=True)


def d184(ep):return h.calibrate(ep,'D184',_d184)

def d184_control(ep):return h.calibrate(ep,'D184_control_source_fixed_type',lambda e,c:_d184(e,c,True))


def _d185(ep,cfg,unconstrained=False):
    k,_,penalty=cfg;base,base_info=_open_background(ep,cfg);m=h.modes(ep,k)
    if m is None:return base,dict(base_info,no_subtype='insufficient_modes')
    ((fc,_,f),(bc,_,b)),_=m;atts=h.attributes(ep,2)
    if atts is None:return base,dict(base_info,no_subtype='insufficient_attributes')
    ar,aq,_,_,_=atts;gates=[h.high_precision_gates(ep,ar[:,j:j+1])[0] for j in range(2)];valid=ep.q_valid>0
    qcenters,qassignment=dh.source_modes(ep.q[valid],min(16,k));qids=np.flatnonzero(valid)
    if not len(qcenters):return base,dict(base_info,no_subtype='no_proposals')
    models=[dh.head(ep,ar[:,j:j+1]) for j in range(2)]
    # Two attribute supports are measured on true source FG modes.
    modeattr=np.asarray([np.average(ar[f][np.argmax(dh.mm(ep.r[f],fc.T),axis=1)==j],axis=0,weights=ep.wf[f][np.argmax(dh.mm(ep.r[f],fc.T),axis=1)==j]) for j in range(len(fc))])
    spread=np.maximum(np.std(ar[f],axis=0),.05);bg_radius=dh.source_radius(ep,b);fg_radius=dh.source_radius(ep,f)
    born=[];audit=[];blocks=dh.blocks(ep.q_hw)
    for j,center in enumerate(qcenters):
        ids=qids[qassignment==j];attr=aq[ids].mean(0)
        agree=all(models[t] is not None and gates[t][1] and float(dh.predict(models[t],attr[t:t+1][None])[0])>gates[t][0] for t in range(2))
        modesupport=np.all(np.abs(modeattr-attr)<=2*spread,axis=1)
        bg=float(np.min(np.sqrt(h.sqdist(center[None],bc))))
        legal=unconstrained or (agree and modesupport.sum()>=2 and np.isfinite(bg_radius) and bg>bg_radius)
        replicated=len(np.unique(blocks[ids]))>=2
        gains=[]
        for z in np.unique(blocks[ids]):
            fitids=ids[blocks[ids]!=z];heldids=ids[blocks[ids]==z]
            if len(fitids)<8 or not len(heldids):continue
            new=h.unit(ep.q[fitids].mean(0));old=h.sqdist(ep.q[heldids],fc).min(1)
            gains.append(float(np.sum(old-np.minimum(old,h.sqdist(ep.q[heldids],new[None]).ravel())))-penalty*ep.r.shape[1]*np.log(256))
        accept=legal and replicated and gains and min(gains)>0 and len(born)<2
        audit.append(dict(cluster=j,two_F_modes=int(modesupport.sum()),two_attribute_gate=agree,replicated=replicated,held_code_gains=gains,accepted=bool(accept)))
        if accept:born.append(center)
    if not born:return base,dict(base_info,subtype_births=0,proposal_audit=audit)
    new=np.asarray(born);scale=h.bandwidth(ep);fresh=logsumexp(-h.sqdist(ep.q,np.vstack((fc,new)))/scale,axis=1)-logsumexp(-h.sqdist(ep.q,bc)/scale,axis=1)
    return fresh,dict(base_info,subtype_births=len(born),proposal_audit=audit,max_births=2,recursive_self_training=False)


def d185(ep):return h.calibrate(ep,'D185',_d185)

def d185_control(ep):return h.calibrate(ep,'D185_control_same_clusters_unrestricted_prototypes',lambda e,c:_d185(e,c,True))


def _discretize(ep,ar,aq):
    active=ep.wvalid>0;block=dh.blocks(ep.r_hw);cuts=[]
    for j in range(ar.shape[1]):
        thresholds=np.unique(np.quantile(ar[active,j],[.25,.5,.75]))
        if len(thresholds)!=3:return None
        assignments=np.searchsorted(thresholds,ar[:,j],side='right')
        if any(len(np.unique(block[active&(assignments==z)]))<2 for z in range(4)):
            return None
        cuts.append(thresholds)
    br=np.column_stack([np.searchsorted(t,ar[:,j],side='right') for j,t in enumerate(cuts)])
    bq=np.column_stack([np.searchsorted(t,aq[:,j],side='right') for j,t in enumerate(cuts)])
    return br,bq,cuts


def _tree_tables(ep,bins,edges,smooth):
    tables=[]
    for weight in (ep.wf,ep.wb):
        marg=[(np.bincount(bins[:,j],weights=weight,minlength=4)+smooth)/(weight.sum()+4*smooth) for j in range(bins.shape[1])]
        conditional={}
        for a,b in edges:
            counts=np.zeros((4,4));np.add.at(counts,(bins[:,a],bins[:,b]),weight)
            conditional[(a,b)]=(counts+smooth)/(counts.sum(1,keepdims=True)+4*smooth)
        tables.append((marg,conditional))
    return tables


def _tree_score(bins,tables,edges):
    incoming={b:a for a,b in edges};logs=[]
    for marg,conditional in tables:
        score=np.zeros(len(bins))
        for j in range(bins.shape[1]):
            if j in incoming:
                a=incoming[j];score+=np.log(conditional[(a,j)][bins[:,a],bins[:,j]])
            else:score+=np.log(marg[j][bins[:,j]])
        logs.append(score)
    return logs[0]-logs[1]


def _tree_model(ep,cfg,*,no_edges=False):
    _,_,smooth=cfg;atts=h.attributes(ep,4)
    if atts is None:return None
    ar,aq,p,a,_=atts;discrete=_discretize(ep,ar,aq)
    if discrete is None:return None
    br,bq,cuts=discrete;g=br.shape[1];edge_scores=[]
    for src in range(g):
        for dest in range(src+1,g):
            gains=[]
            for train,held in dh.folds(ep):
                reduced=h.pseudo_query(ep,train);features=h.attributes(reduced,4)
                if features is None:continue
                disc=_discretize(reduced,features[0],features[1])
                if disc is None:continue
                before=_tree_score(disc[1],_tree_tables(reduced,disc[0],[],smooth),[])
                after=_tree_score(disc[1],_tree_tables(reduced,disc[0],[(src,dest)],smooth),[(src,dest)])
                gains.append(dh.risk(ep,before,held)-dh.risk(ep,after,held))
            improvement=sum(gains) if gains and min(gains)>=-1e-12 else 0.
            edge_scores.append((improvement,src,dest,gains))
    edges=[];connected=np.arange(g)
    def root(j):
        while connected[j]!=j:j=connected[j]
        return j
    for gain,src,dest,_ in sorted(edge_scores,reverse=True):
        if no_edges or gain<=1e-12:continue
        if root(src)==root(dest) or any(b==dest for _,b in edges):continue
        connected[root(dest)]=root(src);edges.append((src,dest))
    if not no_edges and not edges:return None
    tables=_tree_tables(ep,br,edges,smooth)
    return _tree_score(bq,tables,edges),dict(edges=edges,edge_source_gains=edge_scores,attribute_groups=g,
        quantile_bins=4,two_source_blocks_per_bin=True,trees_use_R_only=True), (ar,aq,p,a,cuts,tables,edges)


def _d186(ep,cfg,independent=False):
    out=_tree_model(ep,cfg,no_edges=independent)
    return (None,dict(reason='no_legal_predictive_tree')) if out is None else out[:2]


def d186(ep):return h.calibrate(ep,'D186',_d186)

def d186_control(ep):return h.calibrate(ep,'D186_control_same_attributes_naive_Bayes',lambda e,c:_d186(e,c,True))


def _d187(ep,cfg,centroid=False):
    k,rank,penalty=cfg;m=h.modes(ep,k,original=False)
    if m is None:return None,dict(reason='insufficient_dictionaries')
    ((fc,_,f),(bc,_,b)),_=m;xr,xq,_=dh.project(ep)
    source_bg=np.average(xr[b],axis=0,weights=ep.wb[b]);centered=xr[b]-source_bg
    _,_,vt=np.linalg.svd(centered*np.sqrt(ep.wb[b,None]),full_matrices=False);loads=vt[:min(rank,len(vt))]
    scale=max(float(np.median(np.min(h.sqdist(xr[f],fc),axis=1))),1e-5);sparse=.1*scale
    _,ferr=h.nnls_codes(xq,fc,sparse);_,berr=h.nnls_codes(xq,bc,sparse);leaf=berr-ferr
    regions,info=h.candidate_regions(ep);updates=[];bg_radius=dh.source_radius(ep,b)
    supported=dh.b0_field(ep)<0
    supported &=dh.nearest_distance(dh.pair(ep),b)<=bg_radius if np.isfinite(bg_radius) else False
    audit=[]
    for ids in regions:
        outer=h.ring(ep,ids);outer=outer[supported[outer]]
        if len(outer)<8:continue
        mean=np.average(xq[outer],axis=0,weights=ep.q_valid[outer])
        shared=mean if centroid else source_bg+dh.mm(dh.mm(mean-source_bg,loads.T),loads)
        residual=xq[ids]-shared;target=fc-source_bg
        background=bc-source_bg
        _,no=h.nnls_codes(residual,background,sparse)
        _,yes=h.nnls_codes(residual,np.vstack((background,target)),sparse)
        updates.append((ids,no-yes-penalty*scale));audit.append(dict(size=len(ids),BG_ring_samples=len(outer),shared_norm=float(np.linalg.norm(shared))))
    return h.merge_field(leaf,updates),dict(**info,shared_texture_audit=audit,shared_parameter_same_both_hypotheses=True,
        interior_not_used_to_fit_texture=True,rank=len(loads),control_raw_ring_centroid=centroid)


def d187(ep):return h.calibrate(ep,'D187',_d187)

def d187_control(ep):return h.calibrate(ep,'D187_control_same_ring_negative_centroid',lambda e,c:_d187(e,c,True))


def _d188(ep,cfg,average=False):
    k,_,penalty=cfg;gr=h.groups(ep,4);m=h.modes(ep,k,original=False)
    if gr is None or m is None:return None,dict(reason='insufficient_attribute_groups')
    xr,xq,_,partition=gr;((fc,_,f),(bc,_,b)),_=m;g=len(partition)
    if g<2:return None,dict(reason='fewer_than_two_groups')
    subsets=[np.asarray(v,bool) for v in itertools.product((False,True),repeat=g) if sum(v)>=2]
    fg=np.column_stack([h.sqdist(xq[:,ids],fc[:,ids]).min(1) for ids in partition])
    bg=np.column_stack([h.sqdist(xq[:,ids],bc[:,ids]).min(1) for ids in partition])
    # Source leave-block prediction costs define the missing-group code charge.
    charges=np.zeros(g);count=0
    for train,held in dh.folds(ep):
        reduced=dh.restricted(ep,train);f2,b2,_=dh.pure(reduced)
        if not f2.any() or not b2.any():continue
        for j,ids in enumerate(partition):
            fd=h.sqdist(xr[held][:,ids],xr[f2][:,ids]).min(1);bd=h.sqdist(xr[held][:,ids],xr[b2][:,ids]).min(1)
            selected=np.flatnonzero(held);charges[j]+=float((fd@ep.wf[selected]+bd@ep.wb[selected])/max(ep.wvalid[selected].sum(),h.EPS))
        count+=1
    charges/=max(count,1);charges=np.maximum(charges,1e-5)
    fields=[(bg[:,v]-fg[:,v]).sum(1) for v in subsets];full=np.ones(g,bool);full_index=next(j for j,v in enumerate(subsets) if v.all())
    leaf=fields[full_index];regions,info=h.candidate_regions(ep);updates=[];audit=[]
    yy,xx=np.indices(ep.q_hw);parity=((yy+xx)%2).ravel()
    for ids in regions:
        objectives=[]
        for v in subsets:
            losses=[]
            for side in (0,1):
                fit=ids[parity[ids]==side];held=ids[parity[ids]!=side]
                if not len(fit) or not len(held):continue
                # One shared missing-bit vector; only the mixing frequency is
                # fitted on the opposite checkerboard, never separate F/B bits.
                prior=float(np.clip(expit((bg[fit][:,v]-fg[fit][:,v]).sum(1)).mean(),.01,.99))
                fl=-fg[held][:,v].sum(1)+np.log(prior);bl=-bg[held][:,v].sum(1)+np.log1p(-prior)
                losses.append(float(-logsumexp(np.vstack((fl,bl)),axis=0).mean()))
            objectives.append((float(np.mean(losses)) if losses else np.inf)+penalty*float(charges[~v].sum()))
        winner=int(np.argmin(objectives));improved=objectives[winner]<objectives[full_index]-1e-12
        value=np.mean([field[ids] for field in fields],axis=0) if average else fields[winner if improved else full_index][ids]
        updates.append((ids,value));audit.append(dict(size=len(ids),available_bits=subsets[winner].astype(int).tolist(),improved=improved,held_objectives=objectives))
    return h.merge_field(leaf,updates),dict(**info,missing_bit_audit=audit,subsets=len(subsets),enumerated_bit_patterns=2**g,
        all_available_retained=True,same_missing_bits_FG_BG=True,minimum_available_groups=2,source_missing_charges=charges.tolist())


def d188(ep):return h.calibrate(ep,'D188',_d188)

def d188_control(ep):return h.calibrate(ep,'D188_control_same_subsets_fixed_average',lambda e,c:_d188(e,c,True))


def _quantized_field(ep,centers,assignment):
    nr=len(ep.r);rassign=assignment[:nr];f=np.bincount(rassign,weights=ep.wf,minlength=len(centers));b=np.bincount(rassign,weights=ep.wb,minlength=len(centers))
    # Class-normalized soft counts, with unlabeled cells using the distance head.
    fn=f/max(ep.wf.sum(),h.EPS);bn=b/max(ep.wb.sum(),h.EPS);mass=fn+bn
    cell=fn/np.maximum(mass,h.EPS);qid=assignment[nr:];distance=expit(dh.b0_field(ep)/max(np.sqrt(h.bandwidth(ep)),.05))
    quant=cell[qid];unlabeled=mass[qid]<=h.EPS;quant[unlabeled]=distance[unlabeled]
    return .5*(quant+distance)-.5


def _quant_fit(ep,cfg,target=None,guard=True):
    _,_,bound=cfg;valid=ep.q_valid>0
    if valid.sum()<8:return None
    combined=np.vstack((ep.r,ep.q));weights=np.r_[ep.wvalid,ep.q_valid]
    active=weights>0;centers,local=dh.source_modes(combined[active],32,weights=weights[active]);assignment=np.argmin(h.sqdist(combined,centers),axis=1) if len(centers) else np.empty(0,int)
    if len(centers)<2:return None
    nr=len(ep.r);initial_entropy=h.balanced_entropy(ep,assignment[:nr],len(centers));audit=[];merges=0
    def source_error(count):
        score=np.zeros(nr);selected=np.zeros(nr,bool)
        for train,held in dh.folds(ep):
            reduced=h.pseudo_query(ep,train);out=_quant_fit(reduced,cfg,target=count,guard=False)
            if out is None:return np.inf
            score[held]=out[0][held];selected|=held
        return dh.risk(ep,score,selected)
    initial_error=source_error(len(centers)) if guard else np.nan
    while len(centers)>max(target or 2,2) and merges<20:
        distance=h.sqdist(centers,centers);np.fill_diagonal(distance,np.inf)
        pairs=np.dstack(np.unravel_index(np.argsort(distance.ravel(),kind='stable'),distance.shape))[0]
        accepted=False
        for left,right in pairs:
            if left>=right:continue
            trial=assignment.copy();trial[trial==right]=left;trial[trial>right]-=1
            f=np.bincount(assignment[:nr],weights=ep.wf,minlength=len(centers));b=np.bincount(assignment[:nr],weights=ep.wb,minlength=len(centers))
            entropy=h.balanced_entropy(ep,trial[:nr],len(centers)-1)
            if entropy-initial_entropy>bound*.1:continue
            counts=np.bincount(assignment,weights=weights,minlength=len(centers));new=centers.copy()
            new[left]=h.unit((counts[left]*new[left]+counts[right]*new[right])/max(counts[left]+counts[right],h.EPS));new=np.delete(new,right,axis=0)
            error=source_error(len(new)) if guard else np.nan
            if guard and error>initial_error+1e-12:continue
            audit.append(dict(before_cells=len(centers),conditional_label_entropy=entropy,
                              entropy_increment=entropy-initial_entropy,out_of_fold_error=float(error),
                              joint_reconstruction_cost=float(np.sum(weights*np.min(h.sqdist(combined,new),axis=1)))))
            centers=new;assignment=trial;merges+=1;accepted=True;break
        if not accepted:break
    return _quantized_field(ep,centers,assignment),dict(codewords=len(centers),codebook=centers.tolist(),merges=merges,
        merge_audit=audit,source_initial_OOF_error=float(initial_error),query_counts_used_as_FG_votes=False),centers


def _d189(ep,cfg,no_merge=False):
    out=_quant_fit(ep,cfg,target=32 if no_merge else None)
    if out is None:return None,dict(reason='insufficient_quantizer_support')
    if not no_merge and out[1]['merges']==0:return None,dict(reason='no_compression_gain',**out[1])
    return out[:2]


def d189(ep):return h.calibrate(ep,'D189',_d189)

def d189_control(ep):return h.calibrate(ep,'D189_control_joint_kmeans_labels_no_merging',lambda e,c:_d189(e,c,True))


def _principal_directions(x,weights,maximum=4):
    from scipy.sparse.linalg import LinearOperator,eigsh
    mean=np.average(x,axis=0,weights=weights);z=(x-mean)*np.sqrt(weights[:,None]/max(weights.sum(),h.EPS));d=x.shape[1]
    count=min(maximum,len(x)-1,d)
    if count<=0:return np.empty((0,d))
    if d<=16 or count==d:
        _,_,vt=np.linalg.svd(z,full_matrices=False);return vt[:count]
    operator=LinearOperator((d,d),matvec=lambda v:dh.mm(z.T,dh.mm(z,v)),dtype=float)
    vals,vec=eigsh(operator,k=count,which='LM',v0=np.ones(d)/np.sqrt(d),tol=1e-6,maxiter=500)
    return vec[:,np.argsort(vals)[::-1]].T


def _moment_problem(ep,strength,only_fg=False):
    f=ep.wf>0;b=ep.wb>0
    if min(f.sum(),b.sum())<2:return None,dict(reason='insufficient_moments')
    fm=np.average(ep.r[f],axis=0,weights=ep.wf[f]);bm=np.average(ep.r[b],axis=0,weights=ep.wb[b])
    candidates=[h.unit(fm-bm)]+list(_principal_directions(ep.r[f],ep.wf[f]))+list(_principal_directions(ep.r[b],ep.wb[b]))
    directions=[]
    for v in candidates:
        if np.linalg.norm(v)<h.EPS:continue
        if all(abs(float(v@c))<1.-1e-6 for c in directions):directions.append(v)
        if len(directions)==8:break
    a=np.asarray(directions).T;rp=dh.mm(ep.r,a);qp=dh.mm(ep.q,a);block=dh.blocks(ep.r_hw)
    constraints=[];means=[];epsilon=[]
    for weight in (ep.wf,ep.wb):
        mean=np.average(rp,axis=0,weights=weight);rows=[]
        for z in np.unique(block):
            selected=(block==z)&(weight>0)
            if selected.any():rows.append(np.average(rp[selected],axis=0,weights=weight[selected]))
        spread=np.max(np.abs(np.asarray(rows)-mean),axis=0) if rows else np.zeros(len(directions))
        means.append(mean);epsilon.append(spread)
    # |sum p*(phi-mu)| <= epsilon*sum p, and the B counterpart.
    af=qp-means[0];ab=qp-means[1];epsf,epsb=epsilon
    matrix=np.vstack(((af-epsf).T,(-af-epsf).T))
    offset=np.zeros(len(matrix))
    if not only_fg:
        bgmatrix=np.vstack(((ab-epsb).T,(-ab-epsb).T));matrix=np.vstack((matrix,-bgmatrix));offset=np.r_[offset,bgmatrix.sum(1)]
    valid=ep.q_valid>0;matrix=matrix[:,valid];n=valid.sum()
    oof=np.full(len(ep.r),np.nan)
    for train,held in dh.folds(ep):oof[held]=dh.b0_field(dh.restricted(ep,train),ep.r[held])
    values=oof[np.isfinite(oof)];iqr=float(np.subtract(*np.quantile(values,[.75,.25]))) if len(values) else 0.
    p0=np.clip(expit(dh.b0_field(ep)[valid]/max(iqr,.05)),.01,.99);log0=np.log(p0)-np.log1p(-p0)
    visited=[]
    def fun(p):
        violation=dh.mm(matrix,p)+offset;slack=np.maximum(violation,0.)
        objective=float(np.sum(p*np.log(p/p0)+(1-p)*np.log((1-p)/(1-p0)))+strength*slack.sum())
        gradient=np.log(p)-np.log1p(-p)-log0+strength*matrix[violation>0].sum(0)
        visited.append(objective);return objective,gradient
    fit=minimize(fun,p0,jac=True,method='L-BFGS-B',bounds=[(1e-8,1-1e-8)]*n,
                 options=dict(maxiter=200,maxls=20,ftol=1e-10,gtol=1e-7))
    out=np.zeros(len(ep.q));out[valid]=fit.x
    return out,dict(optimizer='convex_KL_plus_exact_eliminated_nonnegative_slacks_LBFGSB',
        iterations=int(fit.nit),function_evaluations=int(fit.nfev),optimizer_success=bool(fit.success),message=str(fit.message),
        directions=len(directions),FG_mass=float(out.sum()),BG_mass=float(n-out.sum()),fixed_area=False,
        slacks=np.maximum(dh.mm(matrix,fit.x)+offset,0.).tolist(),objective=float(fit.fun),
        natural_probability_threshold=.5,source_IQR=iqr,moment_tolerances=[e.tolist() for e in epsilon])


def d190(ep):return h.calibrate(ep,'D190',_moment_problem,(0.,1.,4.),probability=True)

def d190_control(ep):return h.calibrate(ep,'D190_control_only_FG_moments',lambda e,c:_moment_problem(e,c,True),(0.,1.,4.),probability=True)
