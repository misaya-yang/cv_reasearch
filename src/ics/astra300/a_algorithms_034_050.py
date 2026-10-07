"""Remaining supplied A-token rules; reference labels only."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from itertools import combinations

from .a_helpers_001_050 import (EPS,Frame,b0,fit_rbf,fit_ridge,fps,infer_a,
    nearest_mean_distance,rank_basis,sqdist,threshold,unit)


def _plain(frame):return lambda q,ids,role:b0(frame,q),{'mechanism':'A_B0_5NN'}


def fit_a042(frame,permuted=False,all_pca=False):
    f,b,*_=frame.banks()
    if frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    yy,xx=np.indices(frame.ep.r_hw)
    y=(yy.ravel()+.5)*2/frame.ep.r_hw[0]-1;x=(xx.ravel()+.5)*2/frame.ep.r_hw[1]-1
    design=np.c_[np.ones(len(x)),x,y,x*x,x*y,y*y]
    if permuted:design=design[np.random.default_rng(0).permutation(len(design))]
    coefficients=[]
    for weights in (frame.wf,frame.wb):
        coefficients.append(np.linalg.solve(design.T@(weights[:,None]*design)+np.eye(6),design.T@(weights[:,None]*frame.x)))
    shared=[]
    for j in range(1,6):
        if float(unit(coefficients[0][j])@unit(coefficients[1][j]))>=.95:
            shared.append((coefficients[0][j]+coefficients[1][j])/2)
    u=rank_basis(np.array(shared).reshape(-1,frame.x.shape[1]))
    if all_pca:
        ids=np.flatnonzero(frame.train);u=rank_basis(frame.x[ids]-frame.x[ids].mean(0),u.shape[1])
    def project(z):return unit(z-z@u@u.T)
    ff,bb=project(f),project(b)
    def predict(q,ids,role):
        qq=project(q)
        return (nearest_mean_distance(qq,bb)-nearest_mean_distance(qq,ff))/2,{}
    return predict,{'position_polynomial':['1','x','y','x2','xy','y2'],'shared_direction_rank':u.shape[1],
        'coordinate_fit_permuted':permuted,'all_PCA_control':all_pca,'query_coordinate_not_used':True}


def _weighted_rank(values,weights,query):
    order=np.argsort(values,kind='stable');v=values[order];w=weights[order]
    cumulative=np.r_[0.,np.cumsum(w)];left=np.searchsorted(v,query,side='left');right=np.searchsorted(v,query,side='right')
    return (cumulative[left]+.5*(cumulative[right]-cumulative[left]))/max(cumulative[-1],EPS)


def fit_a040(frame,quantile=.25,profile=False):
    f,b,*_=frame.banks()
    if frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    directions=unit(f-b[np.argmin(sqdist(f,b),1)])[:16]
    pc=[]
    for ids in (frame.fids,frame.bids):
        if len(ids)>1:pc.extend(rank_basis(frame.x[ids]-frame.x[ids].mean(0),4).T)
    all_directions=np.r_[directions,np.array(pc).reshape(-1,frame.x.shape[1])]
    keep=[]
    for v in all_directions:
        if np.linalg.norm(v)>EPS and (not keep or np.max(np.abs(np.array(keep)@v))<.99):keep.append(unit(v))
    if not keep:return _plain(frame)
    u=np.array(keep).T;reference=frame.x@u
    def ranks(z):
        projection=z@u;roles=[]
        for weights in (frame.wf,frame.wb):
            selected=weights>0
            roles.append(np.column_stack([_weighted_rank(reference[selected,j],weights[selected],projection[:,j]) for j in range(u.shape[1])]))
        return roles
    if profile:
        rr=np.concatenate(ranks(frame.x),axis=1);model=fit_rbf(frame,rr)
        if model is None:return _plain(frame)
        bank,coef,width=model
        return lambda q,ids,role:(np.exp(-sqdist(np.concatenate(ranks(q),axis=1),bank)/width)@coef,{}),{
            'control':'all_identical_direction_role_rank_profiles_RBF','directions':u.shape[1]}
    support=[]
    for block in range(4):
        src=Frame.make(frame.ep,frame.train&(frame.blocks!=block));held=frame.fids[frame.blocks[frame.fids]==block]
        if len(held) and src.wf.sum()>0:
            ff,*_=src.banks();support.extend(nearest_mean_distance(frame.x[held],ff).tolist())
    radius=max(float(np.quantile(support,.95)) if support else 2.,EPS)
    def predict(q,ids,role):
        fg,bg=ranks(q)
        df=np.quantile(np.minimum(fg,1-fg),quantile,axis=1)
        db=np.quantile(np.minimum(bg,1-bg),quantile,axis=1)
        score=np.minimum(df-db,(radius-nearest_mean_distance(q,f))/radius)
        return score,{'empirical_direction_count':u.shape[1]}
    return predict,{'directions':u.shape[1],'projection_tail_quantile':quantile,'soft_role_weighted_empirical_midrank':True,
                    'FG_support_radius2':radius}


def _attributes(frame,cap=8):
    f,b,*_=frame.banks()
    if not len(f) or not len(b):return np.empty((frame.x.shape[1],0))
    candidate=unit(f-b[np.argmin(sqdist(f,b),1)])
    selected=[]
    for v in candidate:
        if np.linalg.norm(v)>EPS and (not selected or np.max(np.abs(np.array(selected)@v))<.95):selected.append(v)
        if len(selected)>=cap:break
    return np.array(selected).T if selected else np.empty((frame.x.shape[1],0))


def fit_attribute_rbf(frame):
    u=_attributes(frame,8)
    if not u.shape[1]:return _plain(frame)
    model=fit_rbf(frame,frame.x@u)
    if model is None:return _plain(frame)
    bank,coef,width=model
    return lambda q,ids,role:(np.exp(-sqdist(q@u,bank)/width)@coef,{}),{'control':'same_continuous_attributes_RBF','dimensions':u.shape[1]}


def fit_attribute_tree(frame):
    u=_attributes(frame,8)
    if not u.shape[1] or frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    x=frame.x@u;fw=frame.wf/frame.wf.sum();bw=frame.wb/frame.wb.sum()
    def build(ids,depth):
        pf,pb=float(fw[ids].sum()),float(bw[ids].sum());leaf=(pf-pb)/max(pf+pb,EPS)
        if depth>=3 or len(ids)<2:return leaf
        current=min(pf,pb);best=None
        for j in range(u.shape[1]):
            for cut in np.unique(np.quantile(x[ids,j],[.1,.3,.5,.7,.9])):
                left=ids[x[ids,j]<=cut];right=ids[x[ids,j]>cut]
                if not len(left) or not len(right):continue
                error=min(float(fw[left].sum()),float(bw[left].sum()))+min(float(fw[right].sum()),float(bw[right].sum()))
                if error<current-1e-12:current=error;best=(j,float(cut),left,right)
        if best is None:return leaf
        j,cut,left,right=best;return (j,cut,build(left,depth+1),build(right,depth+1))
    tree=build(np.flatnonzero(frame.train),0)
    def predict(q,ids,role):
        p=q@u;out=np.zeros(len(q))
        def walk(node,indices):
            if not isinstance(node,tuple):out[indices]=node;return
            j,cut,left,right=node;selection=p[indices,j]<=cut
            walk(left,indices[selection]);walk(right,indices[~selection])
        walk(tree,np.arange(len(q)));return out,{}
    return predict,{'control':'same_attribute_reference_Cart_depth3','tree':tree,'depth_max':3}


def fit_a036(frame):
    u=_attributes(frame,8)
    if not u.shape[1] or frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    ids=np.flatnonzero(frame.train);projection=frame.x[ids]@u
    literals=[]
    for j in range(u.shape[1]):
        for cut in np.unique(np.quantile(projection[:,j],[.1,.3,.5,.7,.9])):
            for sign in (1.,-1.):literals.append((j,float(cut),sign))
    masks=np.array([sign*(projection[:,j]-cut)>0 for j,cut,sign in literals])
    wf=frame.wf[ids]/frame.wf.sum();wb=frame.wb[ids]/frame.wb.sum()
    # Enumerate EVERY feasible conjunction under this explicitly fixed <=8
    # attribute recipe, not a silently chosen beam or sample of triples.
    feasible=[]
    for size in (1,2,3):
        for clause in combinations(range(len(literals)),size):
            if len({literals[k][0] for k in clause})<size:continue
            covered=np.logical_and.reduce(masks[list(clause)],axis=0)
            if np.sum(wf[covered])>0:feasible.append((clause,covered))
    chosen=[];union=np.zeros(len(ids),bool)
    for _ in range(8):
        best=None;gain=0.
        for clause,covered in feasible:
            if clause in chosen:continue
            added=covered&~union
            value=float(np.sum(wf[added])-np.sum(wb[added]))
            if value>gain+1e-12:gain=value;best=(clause,covered)
        if best is None:break
        chosen.append(best[0]);union|=best[1]
    if not chosen:
        pred,info=_plain(frame);return pred,dict(info,card_id='A036',degeneration='no_positive_gain_source_DNF_clause',feasible_clauses=len(feasible))
    def predict(q,ids,role):
        p=q@u;out=np.full(len(q),-np.inf)
        for clause in chosen:
            margin=np.array([literals[k][2]*(p[:,literals[k][0]]-literals[k][1]) for k in clause]).min(0)
            out=np.maximum(out,margin)
        return out,{'clauses':len(chosen)}
    return predict,{'attributes':u.shape[1],'literal_count':len(literals),'feasible_clauses_enumerated':len(feasible),
        'clauses':[[literals[k] for k in clause] for clause in chosen],'BG_increment_penalty':1.}


def fit_a037(frame,exceptions=True):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    from .group_001_075 import _maximum_margin
    broad=fit_ridge(frame);reference=frame.x@broad[:-1]+broad[-1]
    rules=[];false_positive=frame.bids[reference[frame.bids]>0]
    if exceptions and len(false_positive):
        negatives=fps(frame.x,false_positive,32);positives=fps(frame.x,frame.fids,32)
        coef,good,_=_maximum_margin(frame.x[np.r_[positives,negatives]],np.r_[np.ones(len(positives)),-np.ones(len(negatives))])
        if good:
            center=frame.x[negatives].mean(0);radius=max(float(np.quantile(sqdist(frame.x[negatives],center[None])[:,0],.95)),EPS)
            rules.append((coef,center,radius,'negative_exception'))
            hit=sqdist(frame.x,center[None])[:,0]<=radius
            exception=frame.x@coef[:-1]+coef[-1]
            erroneous=frame.fids[(reference[frame.fids]>0)&hit[frame.fids]&(exception[frame.fids]<=0)]
            if len(erroneous):
                pos=fps(frame.x,erroneous,32)
                restoration,good,_=_maximum_margin(frame.x[np.r_[pos,negatives]],np.r_[np.ones(len(pos)),-np.ones(len(negatives))])
                if good:
                    pcenter=frame.x[pos].mean(0);pradius=max(float(np.quantile(sqdist(frame.x[pos],pcenter[None])[:,0],.95)),EPS)
                    rules.append((restoration,pcenter,pradius,'positive_restoration'))
    def predict(q,ids,role):
        score=q@broad[:-1]+broad[-1]
        applied=[]
        for coef,center,radius,kind in rules:
            domain=sqdist(q,center[None])[:,0]<=radius
            use=domain&(score>0 if kind=='negative_exception' else score<=0)
            score[use]=(q@coef[:-1]+coef[-1])[use];applied.append(int(use.sum()))
        return score,{'ordered_rule_application_counts':applied}
    return predict,{'ordered_depth':1+len(rules),'reference_false_positive_count':len(false_positive),
        'rule_types':[v[3] for v in rules],'ridge_L2':1.,'exceptions_enabled':exceptions,
        'description_length_tie_prefers_fewer_rules':True}


def infer_a037(ep):
    candidates=[]
    for exceptions in (False,True):
        result=infer_a(ep,'A037',lambda frame:fit_a037(frame,exceptions=exceptions),
            ('Twojointconfigs broadonly/orderedexceptions; literalC_R loss+.001pereextra rule choosesdescriptionlength, Qneverselects.',))
        loss=result.info['calibration']['balanced_coverage_error']
        complexity=result.info['fit'].get('ordered_depth',1)-1
        candidates.append((np.inf if loss is None else loss+.001*complexity,complexity,exceptions,result))
    _,_,choice,result=min(candidates,key=lambda v:(v[0],v[1],v[2]))
    result.info['calibration']['configurations']=2
    result.info['description_length_selected_exceptions']=choice
    result.info['source_config_objectives']=[None if not np.isfinite(v[0]) else v[0] for v in candidates]
    return result


def fit_a035(frame,both_gates=True):
    from .a_helpers_001_050 import spherical_modes
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    centers,_,groups=spherical_modes(frame.x,frame.fids,8)
    _,b,*_=frame.banks();rules=[]
    for k,center in enumerate(centers):
        representative=int(groups[k][np.argmin(sqdist(frame.x[groups[k]],center[None])[:,0])])
        ids=groups[k][frame.blocks[groups[k]]!=frame.blocks[representative]]
        if not len(ids):continue
        presence=frame.x[ids]@center
        contrast=presence-np.max(frame.x[ids]@b.T,axis=1)
        rules.append((center,float(np.quantile(presence,.05)),float(np.quantile(contrast,.05))))
    if not rules:return _plain(frame)
    def predict(q,ids,role):
        bg=np.max(q@b.T,axis=1);out=np.full(len(q),-np.inf)
        for center,presence_cut,contrast_cut in rules:
            p=q@center;first=p-presence_cut;second=p-bg-contrast_cut
            out=np.maximum(out,np.minimum(first,second) if both_gates else second)
        return out,{'mode_two_gate_count':len(rules)}
    return predict,{'modes':len(centers),'qualified_modes':len(rules),'both_gates':both_gates,
                    'FG_presence_and_BG_contrast_thresholds':[[v[1],v[2]] for v in rules]}


def fit_a038(frame,linear_control=False,random_delta=False):
    from .a_helpers_001_050 import spherical_modes
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    centers,_,groups=spherical_modes(frame.x,frame.fids,8);f,b,*_=frame.banks()
    deltas=[]
    for k,center in enumerate(centers):
        local=b[np.argsort(sqdist(b,center[None])[:,0],kind='stable')[:min(8,len(b))]]
        candidate=[np.zeros(frame.x.shape[1])]
        for i,j in combinations(range(len(local)),2):
            candidate.extend((local[i]-local[j],local[j]-local[i]))
            if len(candidate)>=16:break
        amplitude=[]
        for block in range(4):
            held=groups[k][frame.blocks[groups[k]]==block];source=groups[k][frame.blocks[groups[k]]!=block]
            if len(held) and len(source):amplitude.extend(np.sqrt(nearest_mean_distance(frame.x[held],frame.x[source],1)).tolist())
        bound=float(np.quantile(amplitude,.9)) if amplitude else 0.
        vectors=np.array(candidate[:16]);norm=np.linalg.norm(vectors,axis=1)
        vectors*=np.minimum(1,bound/np.maximum(norm,EPS))[:,None]
        if random_delta:
            random=np.random.default_rng(k).normal(size=vectors.shape);random=unit(random)*np.linalg.norm(vectors,axis=1)[:,None];random[0]=0;vectors=random
        deltas.append(vectors)
    def predict(q,ids,role):
        route=np.argmin(sqdist(q,centers),1);out=np.empty(len(q))
        for k,delta in enumerate(deltas):
            selected=route==k
            if not selected.any():continue
            z=q[selected];scores=[]
            for change in delta:
                altered=z+change if linear_control else unit(z+change)
                if linear_control:
                    axis=unit(f.mean(0))-unit(b.mean(0));scores.append(altered@axis)
                else:scores.append(np.max(altered@f.T,1)-np.max(altered@b.T,1))
            out[selected]=np.min(scores,axis=0)
        return out,{'routed_mode_counts':np.bincount(route,minlength=len(centers)).tolist()}
    return predict,{'modes':len(centers),'perturbation_counts':[len(v) for v in deltas],
        'max_delta_norms':[float(np.linalg.norm(v,axis=1).max()) for v in deltas],
        'includes_exact_zero_delta':True,'linear_control':linear_control,'random_delta':random_delta}


def fit_a039(frame,omit=True):
    # Coordinate dependence is intentional and receives an explicit rotation/
    # no-omission countercontrol rather than an orthogonal-invariance claim.
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    from .a_helpers_001_050 import spherical_modes
    f,b,*_=frame.banks();gap=np.mean(np.abs(f-b[np.argmin(sqdist(f,b),1)]),axis=0)
    coords=np.argsort(-gap,kind='stable')[:min(16,frame.x.shape[1])]
    centers,_,groups=spherical_modes(frame.x,frame.fids,8)
    bgcenter,_,bggroups=spherical_modes(frame.x,frame.bids,8)
    forbidden=[np.quantile(frame.x[g][:,coords],[.05,.95],axis=0) for g in bggroups]
    boxes=[]
    for group in groups:
        interval=np.quantile(frame.x[group][:,coords],[.05,.95],axis=0)
        flips=np.zeros(len(coords),bool)
        for j,coordinate in enumerate(coords):
            signs=[]
            for block in range(4):
                fg=group[frame.blocks[group]==block];bg=frame.bids[frame.blocks[frame.bids]==block]
                if len(fg) and len(bg):signs.append(np.sign(frame.x[fg,coordinate].mean()-frame.x[bg,coordinate].mean()))
            flips[j]=1 in signs and -1 in signs
        boxes.append((interval,flips))
    def predict(q,ids,role):
        values=q[:,coords];out=np.full(len(q),-np.inf)
        for interval,flips in boxes:
            valid_omissions=[None]+(list(np.flatnonzero(flips)) if omit else [])
            for excluded in valid_omissions:
                use=np.ones(len(coords),bool)
                if excluded is not None:use[excluded]=False
                if not use.any():continue
                margin=np.minimum(values[:,use]-interval[0,use],interval[1,use]-values[:,use]).min(1)
                for forbidden_box in forbidden:
                    violation=np.maximum(forbidden_box[0,use]-values[:,use],values[:,use]-forbidden_box[1,use]).max(1)
                    margin=np.minimum(margin,violation)
                out=np.maximum(out,margin)
        return out,{'at_most_one_source_flip_coordinate_omitted':omit}
    return predict,{'selected_coordinates':coords.tolist(),'FG_boxes':len(boxes),'BG_forbidden_boxes':len(forbidden),
        'source_flip_coordinates_per_box':[np.flatnonzero(v[1]).tolist() for v in boxes],'omission_enabled':omit}


def install(register,requirements,methods,controls,recipes):
    register('A042',fit_a042,(
        'Coordinatepolynomialdegree2 onknowntrainingR, separatewf/wb weightedridge1; nointercept nuisance, retaincoefficientdirectionscos>=.95, thinSVDsharedspan.',
        'ProjectR/Q toorthogonalcomplement thenunitnormalize for5NNcos; neverinsertR coordinatesintoQ orsubtractaR positionfunction.',),
        (('same_coords_permuted_fit',lambda f:fit_a042(f,permuted=True)),('same_rank_global_PCA',lambda f:fit_a042(f,all_pca=True))))
    register('A040',fit_a040,(
        '16nearest-crossrole anchordirections +4intrarolePCeach, removeabs-cos>=.99 redundancy; allknownR softroleweighted empiricalmidranks, minleft/righttail.',
        'Fixedlowerquantile.25 isoneexplicitjointconfig; outer4blockCRrebuilds directions/distributions andcalibratescut, minFGcrossblock95%5NNsupportmargin.',),
        (('all_identical_rank_profiles_RBF',lambda f:fit_a040(f,profile=True)),('minimum_direction_depth',lambda f:fit_a040(f,quantile=0.))))
    register('A035',fit_a035,(
        '8FG sphericalmodes, centroidrepresentative sourceblockexcluded forpresence/contrast minq.05 estimates; signedmode minimumoftwogates thenORmax, nevermodequotas.',),
        (('same_modes_single_contrast_gate',lambda f:fit_a035(f,both_gates=False)),))
    register('A036',fit_a036,(
        'Explicit8nonredundant nearestFG−BG attributes (originalcap64), fivequantiles andbothsigns; enumerateALLfeasible1..3differentattributeconjunctions.',
        'Greedy<=8clauses maximizeincrementalFGrolecoverage−incrementalBGrolecoverage,λBG1, continuousbestclauseweakestliteral; noquerypruning.',),
        (('same_attributes_RBF',fit_attribute_rbf),('same_attributes_depth3_tree',fit_attribute_tree)))
    register('A037',fit_a037,(
        'BroadbalancedridgeL2=1; sourceBGscore>0 generatesnegativeexceptionSVM with32FPS perclass andlocal95%BG-source-distance ball.',
        'SourceFGerroneouslyremovedbyexception generatespositive restoration samebudget/localball; ordereddepth<=3, laterpositive cannotprecedenegative.',),
        (('same_broad_without_ordered_exceptions',lambda f:fit_a037(f,exceptions=False)),))
    methods['A037']=infer_a037
    recipes['A037']['configurations']=2
    recipes['A037']['implementation_assumption'].append('Two totaljointconfigs broad/ordered, C_Rloss+.001perextrarule picksparsimoniousbranch, unchangedrulemaximum3.')
    register('A038',fit_a038,(
        '8FGmodes routeoriginalq; eachnearest8BGinternaldiffs+zero up-to16vectors, boundnormsourcecrossblockmodeFGnearest-distanceq.9.',
        'For everydelta reunitnormalizeq+delta andallowwinnerchange inmaxFG−maxBG; actualworstperturbation, notgloballinearoffset.',),
        (('fixed_linear_axis_analytic_collapse',lambda f:fit_a038(f,linear_control=True)),('same_norm_random_perturbations',lambda f:fit_a038(f,random_delta=True))))
    register('A039',fit_a039,(
        'Top16fixedDINOcoordinates byRrolemeancontrast,8FG/8BGjointq.05/.95boxes; coordinateomission onlyifRcrossblockFG−BGmeancontrastchanges sign, atmostone.',
        'OutsideBGforbiddenCartesianjointboxes andinsideFGbox aftersameomission, worstcoordinatecontinuousmargin; allDINOcoord dependence disclosed.',),
        (('same_boxes_no_omission',lambda f:fit_a039(f,omit=False)),))
