"""The supplied leave-part witnesses and shared/private reference factors."""
from __future__ import annotations

from collections import deque
import numpy as np
from scipy import ndimage

from .a_helpers_001_050 import (EPS,Frame,b0,fit_ridge,fps,
    nearest_mean_distance,rank_basis,sqdist,unit)
from .a_context_034_049 import _support_radius


def _plain(frame):return lambda q,ids,role:b0(frame,q),{'mechanism':'A_B0_5NN'}


def spatial_parts(frame,random=False):
    """Known R foreground component BFS, then <=4 equal-coverage segments."""
    mask=(frame.train&(frame.c>.5)).reshape(frame.ep.r_hw)
    labels,count=ndimage.label(mask,ndimage.generate_binary_structure(2,1))
    h,w=frame.ep.r_hw;ordered=[]
    for label in range(1,count+1):
        vertices=np.flatnonzero(labels.ravel()==label);seen={int(vertices[0])};queue=deque([int(vertices[0])])
        while queue:
            v=queue.popleft();ordered.append(v);y,x=divmod(v,w)
            for yy,xx in ((y-1,x),(y,x-1),(y,x+1),(y+1,x)):
                z=yy*w+xx
                if 0<=yy<h and 0<=xx<w and labels[yy,xx]==label and z not in seen:
                    seen.add(z);queue.append(z)
    ordered=np.array(ordered,int)
    if random:ordered=ordered[np.random.default_rng(0).permutation(len(ordered))]
    if not len(ordered):return [],[],{'components':count}
    cumulative=np.cumsum(frame.wf[ordered]);total=cumulative[-1]
    assigned=np.minimum(((cumulative-.5*frame.wf[ordered])*min(4,len(ordered))/max(total,EPS)).astype(int),3)
    parts=[ordered[assigned==j] for j in np.unique(assigned)]
    # Mixed pixels determine spatial extents, while only legal pure FG is used
    # to build this card's discriminative identity witnesses/factors.
    parts=[part[np.isin(part,frame.fids)] for part in parts]
    parts=[part for part in parts if len(part)]
    if not parts:return [],[],{'components':count}
    y,x=np.indices(frame.ep.r_hw);coords=np.c_[y.ravel(),x.ravel()]
    distances=np.column_stack([np.min(sqdist(coords[frame.bids],coords[p]),axis=1) for p in parts])
    assignment=np.argmin(distances,axis=1) if len(frame.bids) else np.empty(0,int)
    backgrounds=[frame.bids[assignment==j] for j in range(len(parts))]
    return parts,backgrounds,{'components':count,'part_FG_counts':[len(p) for p in parts],
        'part_associated_BG_counts':[len(p) for p in backgrounds],'random_patch_parts_control':random,
        'partition':'stable component BFS concatenation and equal soft-FG-mass cuts; cuts may cross disconnected components'}


def _sparse_head(x,fg,bg,penalty,iterations=30):
    if not len(fg) or not len(bg):return None
    ids=np.r_[fg,bg];z=np.c_[x[ids],np.ones(len(ids))]
    weights=np.r_[np.full(len(fg),.5/len(fg)),np.full(len(bg),.5/len(bg))]
    target=np.r_[np.ones(len(fg)),-np.ones(len(bg))]
    coefficient=np.zeros(z.shape[1]);step=1/(float(np.sum(weights[:,None]*z*z))+.01)
    for _ in range(iterations):
        gradient=z.T@(weights*(z@coefficient-target))+.01*np.r_[coefficient[:-1],0.]
        coefficient-=step*gradient
        coefficient[:-1]=np.sign(coefficient[:-1])*np.maximum(np.abs(coefficient[:-1])-step*penalty,0)
    return coefficient


def _part_dro(frame,parts,backgrounds):
    groups=[(p,b) for p,b in zip(parts,backgrounds) if len(p) and len(b)]
    if not groups:return None
    coefficient=np.zeros(frame.x.shape[1]+1);probability=np.ones(len(groups))/len(groups)
    for _ in range(30):
        gradients=[];risks=[]
        for fg,bg in groups:
            ids=np.r_[fg,bg];z=np.c_[frame.x[ids],np.ones(len(ids))]
            y=np.r_[np.ones(len(fg)),-np.ones(len(bg))]
            weights=np.r_[np.full(len(fg),.5/len(fg)),np.full(len(bg),.5/len(bg))]
            residual=z@coefficient-y
            gradients.append(z.T@(weights*residual));risks.append(float(np.sum(weights*residual**2)/2))
        log=np.log(np.maximum(probability,EPS))+np.array(risks)
        probability=np.exp(log-log.max());probability/=probability.sum()
        gradient=probability@np.array(gradients)+np.r_[coefficient[:-1],0.]
        coefficient-=.2*gradient
    return coefficient,probability


def fit_a034(frame,random_parts=False,dro=False,ridge=False):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    parts,backgrounds,part_info=spatial_parts(frame,random_parts)
    if len(parts)<2 or any(not len(b) for b in backgrounds):return _plain(frame)
    if dro:
        result=_part_dro(frame,parts,backgrounds)
        if result is None:return _plain(frame)
        coefficient,p=result
        return lambda q,ids,role:(q@coefficient[:-1]+coefficient[-1],{}),dict(part_info,
            control='same_spatial_parts_group_DRO_ridge',L2=1.,group_risk_weights=p.tolist(),iterations=30)
    if ridge:
        coefficient=fit_ridge(frame)
        return lambda q,ids,role:(q@coefficient[:-1]+coefficient[-1],{}),dict(part_info,control='ordinary_ridge_outer_spatial_C_R',L2=1.)
    penalties=(.001,.003,.01,.02,.04,.08,.16,.32);retained=[];audit=[]
    for penalty in penalties:
        full=_sparse_head(frame.x,frame.fids,frame.bids,penalty)
        valid=True;folds=[]
        for j,(held_f,held_b) in enumerate(zip(parts,backgrounds)):
            source_f=np.concatenate([p for k,p in enumerate(parts) if k!=j])
            source_b=np.concatenate([p for k,p in enumerate(backgrounds) if k!=j])
            coefficient=_sparse_head(frame.x,source_f,source_b,penalty)
            if coefficient is None:valid=False;folds.append({'part':j,'effective':False});continue
            fg=frame.x[held_f]@coefficient[:-1]+coefficient[-1]
            bg=frame.x[held_b]@coefficient[:-1]+coefficient[-1]
            error=.5*float(np.mean(fg<=0))+.5*float(np.mean(bg>0))
            cosine=float(unit(coefficient[:-1])@unit(full[:-1]))
            passed=error<.5 and float(fg.mean()-bg.mean())>0 and cosine>=.8
            valid&=passed;folds.append(dict(part=j,effective=True,error=error,direction_full_fold_cosine=cosine,passed=passed))
        if valid and np.linalg.norm(full[:-1])>EPS:
            values=frame.x[frame.train]@full[:-1]+full[-1]
            q25,q75=np.quantile(values,[.25,.75]);scale=max(float(q75-q25),EPS)
            retained.append((full,scale))
        audit.append(dict(L1=penalty,passed=valid,folds=folds,nonzero_weights=int(np.sum(full[:-1]!=0))))
    if not retained:return _plain(frame)
    f,*_=frame.banks();radius=_support_radius(frame)
    coefficients=np.array([v[0] for v in retained]);scales=np.array([v[1] for v in retained])
    def predict(q,ids,role):
        scores=(q@coefficients[:,:-1].T+coefficients[:,-1])/scales
        return np.median(scores,axis=1)+(radius-nearest_mean_distance(q,f))/radius,{}
    return predict,dict(part_info,witness_count=len(retained),witness_protocols=audit,
        FG_support_radius2=radius,sparse_solver='30 proximal squared-loss steps, L2=.01, no bias penalty',
        readout='median source-IQR-normalized signed witness plus signed FG support')


def _shared_private(x,parts,bg,rounds=20):
    """Exact fixed-coefficient least-square updates, finite ALS visited minimum."""
    k=len(parts);shared_rank=min(2,x.shape[1]);private_rank=1 if shared_rank+k<=8 else 0
    if shared_rank<1:return None
    joined=np.concatenate(parts);u=rank_basis(x[joined],shared_rank)
    if u.shape[1]<shared_rank:return None
    private=[]
    for part in parts:
        residual=x[part]-x[part]@u@u.T
        private.append(rank_basis(residual,private_rank))
    # A fixed background penalty is solved exactly in its thin covariance basis;
    # no final-token proxy or iterative pseudo-solve stands in for the factor fit.
    _,singular,vt=np.linalg.svd(x[bg],full_matrices=False)
    bu=vt.T;be=singular**2/max(len(bg),1);lam=.25
    def coefficients(uu,vv):
        return [x[p]@np.linalg.pinv(np.c_[uu,v].T) for p,v in zip(parts,vv)]
    def objective(uu,vv,cc):
        value=sum(float(np.sum((x[p]-a@np.c_[uu,v].T)**2)) for p,a,v in zip(parts,cc,vv))/max(len(joined),1)
        return value+lam*float(np.sum((x[bg]@uu)**2))/max(len(bg),1)
    c=coefficients(u,private);best=(objective(u,private,c),u.copy(),[v.copy() for v in private]);trace=[best[0]]
    for _ in range(rounds):
        gram=sum(a[:,:shared_rank].T@a[:,:shared_rank] for a in c)/len(joined)
        rhs=sum((x[p]-a[:,shared_rank:]@v.T).T@a[:,:shared_rank] for p,a,v in zip(parts,c,private))/len(joined)
        eigen,rotation=np.linalg.eigh(gram);eigen=np.maximum(eigen,1e-10);rhs=rhs@rotation
        changed=[]
        for j,value in enumerate(eigen):
            r=rhs[:,j];changed.append(r/value+bu@((1/(value+lam*be)-1/value)*(bu.T@r)))
        proposed=np.array(changed).T@rotation.T;u=rank_basis(proposed.T,shared_rank)
        if u.shape[1]!=shared_rank:break
        c=coefficients(u,private)
        updated=[]
        for p,a,v in zip(parts,c,private):
            if not v.shape[1]:updated.append(v);continue
            residual=x[p]-a[:,:shared_rank]@u.T;priv=a[:,shared_rank:]
            new=residual.T@priv@np.linalg.pinv(priv.T@priv)
            new-=u@(u.T@new);updated.append(rank_basis(new.T,private_rank))
        private=updated;c=coefficients(u,private)
        value=objective(u,private,c);trace.append(value)
        if value<best[0]:best=(value,u.copy(),[v.copy() for v in private])
    return best[1],best[2],dict(ALS_visited_objectives=trace,selected_objective=best[0],
        shared_rank=shared_rank,private_ranks=[v.shape[1] for v in best[2]],background_penalty=.25,
        total_rank=shared_rank+sum(v.shape[1] for v in best[2]))


def _shared_features(x,u,private):
    shared=[];residual=[]
    for v in private:
        dictionary=np.c_[u,v];coefficient=x@np.linalg.pinv(dictionary.T)
        shared.append(coefficient[:,:u.shape[1]])
        residual.append(np.sum((x-coefficient@dictionary.T)**2,axis=1))
    return np.array(shared),np.array(residual).T


def _shared_head(frame,u,private):
    features,_=_shared_features(frame.x,u,private);n=len(features)
    # Every dictionary's shared coordinate receives identical source role mass.
    design=np.concatenate([np.c_[a,np.ones(len(a))] for a in features])
    source=np.tile(frame.train,n);wf=np.tile(frame.wf,n)/n;wb=np.tile(frame.wb,n)/n
    weights=.5*wf/max(wf.sum(),EPS)+.5*wb/max(wb.sum(),EPS)
    y=np.divide(wf/max(wf.sum(),EPS)-wb/max(wb.sum(),EPS),2*weights,out=np.zeros(len(weights)),where=weights>0)
    z=design[source]*np.sqrt(weights[source,None]);target=y[source]*np.sqrt(weights[source])
    return np.linalg.solve(z.T@z+np.eye(design.shape[1]),z.T@target)


def _ordinary_prototype(frame):
    f=unit(np.sum(frame.x*frame.wf[:,None],axis=0));b=unit(np.sum(frame.x*frame.wb[:,None],axis=0))
    return lambda q,ids,role:(q@(f-b),{}),{'mechanism':'A046_supplied_ordinary_role_prototypes','shared_guard_failed':True}


def fit_a046(frame,ordinary=False,src=False,prototype=False):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    parts,backgrounds,part_info=spatial_parts(frame)
    if len(parts)<2 or any(not len(b) for b in backgrounds):return _ordinary_prototype(frame)
    result=_shared_private(frame.x,parts,frame.bids)
    if result is None:return _ordinary_prototype(frame)
    u,private,fit_info=result
    if prototype:return _ordinary_prototype(frame)
    if ordinary:
        cap=fit_info['total_rank'];basis=rank_basis(frame.x[frame.fids],cap)
        coef=fit_ridge(frame,frame.x@basis)
        return lambda q,ids,role:(q@basis@coef[:-1]+coef[-1],{}),dict(part_info,control='same_total_rank_unconstrained_joint_factor_ridge',rank=basis.shape[1])
    if src:
        f,b,*_=frame.banks();dictionary=np.concatenate([u]+private+[b.T],axis=1)
        fgcols=u.shape[1]+sum(v.shape[1] for v in private)
        inverse=np.linalg.solve(dictionary.T@dictionary+1e-3*np.eye(dictionary.shape[1]),dictionary.T)
        def predict(q,ids,role):
            coefficient=q@inverse.T
            foreground=coefficient[:,:fgcols]@dictionary[:,:fgcols].T
            background=coefficient[:,fgcols:]@dictionary[:,fgcols:].T
            return np.sum((q-background)**2,axis=1)-np.sum((q-foreground)**2,axis=1),{}
        return predict,dict(part_info,control='same_shared_private_FG_plus_BG_SRC_ridge_reconstruction',FG_dictionary_columns=fgcols,BG_columns=len(b),L2=.001)
    head=_shared_head(frame,u,private);validation=[];residuals=[]
    for j,(held_f,held_b) in enumerate(zip(parts,backgrounds)):
        keep=frame.train.copy();keep[np.r_[held_f,held_b]]=False;fold=Frame.make(frame.ep,keep)
        remaining=[p for k,p in enumerate(parts) if k!=j]
        model=_shared_private(frame.x,remaining,fold.bids)
        if model is None:validation.append(dict(part=j,passed=False,effective=False));continue
        uu,vv,_=model;hh=_shared_head(fold,uu,vv)
        ffeatures,fr=_shared_features(frame.x[held_f],uu,vv);bfeatures,_=_shared_features(frame.x[held_b],uu,vv)
        fs=np.max(np.einsum('kni,i->kn',ffeatures,hh[:-1])+hh[-1],axis=0)
        bs=np.max(np.einsum('kni,i->kn',bfeatures,hh[:-1])+hh[-1],axis=0)
        error=.5*float(np.mean(fs<=0))+.5*float(np.mean(bs>0));gap=float(fs.mean()-bs.mean())
        passed=error<.5 and gap>0;validation.append(dict(part=j,passed=passed,effective=True,error=error,FG_BG_gap=gap))
        residuals.extend(np.min(fr,axis=1).tolist())
    if not validation or not all(v['passed'] for v in validation):
        predict,info=_ordinary_prototype(frame);info.update(part_info);info['leave_part_guard']=validation;return predict,info
    limit=max(float(np.quantile(residuals,.99)),EPS)
    def predict(q,ids,role):
        shared,residual=_shared_features(q,u,private)
        score=(np.einsum('kni,i->nk',shared,head[:-1])+head[-1])
        eligible=residual<=limit;fallback=~eligible.any(1)
        score=np.where(eligible,score,-np.inf).max(1)
        original,_=b0(frame,q);score[fallback]=original[fallback]
        return score,{'_fallback_mask':fallback,'no_admissible_part_dictionary':int(fallback.sum())}
    return predict,dict(part_info,**fit_info,leave_part_guard=validation,source_leave_part_residual_q99=limit,
        identity_head_L2=1.,identity_head_only_shared_coefficients=True)


def install(register,requirements,methods,controls,recipes):
    register('A034',fit_a034,(
        'Known R c>.5 four-neighbor components, stableBFS concatenate then<=4equalsoftFGmass segments; mixedobservations define extent, pureFG builds witnesses. Manycomponents may share a part: explicit original grouping ambiguity.',
        'BG assigned closest physicalFGpart; eight L1 penalties [.001,.003,.01,.02,.04,.08,.16,.32], each30proximal squaredloss steps L2=.01. Each leave-part refits without its associatedBG; require balancederror<.5, positiveheldFG−BGgap, full/fold directioncos>=.8.',
        'Keep<=8passed fullrefit witnesses; median signed sourceIQR scores +signedFG95% support. No common witness fixedB0cut0. Outer4spatialC_R separately rebuilds everypart/model.',),
        (('same_parts_group_DRO_ridge',lambda f:fit_a034(f,dro=True)),('ordinary_ridge_spatial_C_R',lambda f:fit_a034(f,ridge=True)),
         ('random_patch_part_guard',lambda f:fit_a034(f,random_parts=True))))
    register('A046',fit_a046,(
        'Same R-only spatialparts; sharedrank2 +private1each (total<=8). Objective meanFG reconstructionerror+.25meanBG sharedprojectionenergy; exact coefficient/pseudoinverse factor updates,20ALSrounds, orthogonal shared/private spans, select lowestvisited objective.',
        'Leaveonewholepart+associatedBG out, refitfactors/head for each. Sharedcoordinate+constant balancedL2=1 head; requireallleavepart FG/BG balancederror<.5 andpositivegap, otherwise suppliedordinaryFG−BGmeans branch (notcountedasfactoractivity).',
        'Sourceleavepart bestdictionaryFG residualq.99 controlseligible queryk; scoremax_k(w sharedcoeff+b), noeligiblepoint fixedB0cut0. Traininghead replicatesallpart dictionaries equalroleweight, neverprivatecoeff input.',),
        (('same_total_rank_joint_factor',lambda f:fit_a046(f,ordinary=True)),('same_dictionary_SRC_reconstruction',lambda f:fit_a046(f,src=True)),
         ('ordinary_role_prototypes',lambda f:fit_a046(f,prototype=True)),('same_parts_group_DRO_ridge',lambda f:fit_a034(f,dro=True))))
