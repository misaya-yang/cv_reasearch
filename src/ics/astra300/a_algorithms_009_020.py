"""Actual finite A009--A020 kernels, imported by the stable group registry."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .common import require_artifact
from .a_helpers_001_050 import (EPS, Frame, ObservationUnavailable, b0, fit_rbf,
    fit_ridge, fps, infer_a, nearest_mean_distance, rank_basis, spherical_modes,
    sqdist, threshold, unit)


def _plain(frame):
    return lambda q, ids, role: b0(frame, q), {"mechanism": "A_B0_5NN"}


def full_profile_rbf(frame):
    """Complete legal R-affinity profiles; stream Q, never compress columns."""
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    basis=frame.x[frame.train]
    fi=fps(frame.x,frame.fids,64);bi=fps(frame.x,frame.bids,64)
    ids=np.r_[fi,bi];profiles=frame.x[ids]@basis.T
    distance=sqdist(profiles,profiles)
    width=max(float(np.median(distance[np.triu_indices(len(ids),1)])),EPS)
    kernel=np.exp(-distance/width)
    coef=np.linalg.solve(kernel+np.eye(len(ids)),np.r_[np.ones(len(fi)),-np.ones(len(bi))])
    def predict(q,ids,role):
        out=np.empty(len(q))
        for start in range(0,len(q),128):
            p=q[start:start+128]@basis.T
            out[start:start+128]=np.exp(-sqdist(p,profiles)/width)@coef
        return out,{"complete_profile_columns":len(basis),"Q_block":128}
    return predict,{"mechanism":"complete_R_affinity_profile_RBF_ridge","profile_columns":len(basis),
        "supervised_head_support_rows":len(profiles),"width2":width,"L2":1.,"profile_columns_compressed":False}


def _sparse_group(q, dictionary, groups, lam1=.01, lamg=.01):
    """Sparse-group lasso prox with actual optimality mapping,<=30 steps."""
    if not len(dictionary):
        return .5 * np.sum(q*q, 1), np.ones(len(q), bool), np.zeros(len(q)), 0
    gram = dictionary @ dictionary.T
    lip = max(float(np.linalg.eigvalsh(gram)[-1]) + 1e-4, EPS)
    corr = q @ dictionary.T
    z = np.zeros((len(q), len(dictionary))); accelerated = z.copy(); momentum = 1.
    def prox(value):
        out = np.sign(value) * np.maximum(np.abs(value) - lam1/lip, 0)
        for group in groups:
            norm = np.linalg.norm(out[:, group], axis=1)
            shrink = np.maximum(1 - lamg*np.sqrt(len(group)) / (lip*np.maximum(norm, EPS)), 0)
            out[:, group] *= shrink[:, None]
        return out
    residual = np.full(len(q), np.inf)
    for iteration in range(30):
        gradient = accelerated @ gram - corr + 1e-4*accelerated
        new = prox(accelerated - gradient/lip)
        tnew = (1+np.sqrt(1+4*momentum*momentum))/2
        accelerated = new + ((momentum-1)/tnew)*(new-z)
        z, momentum = new, tnew
        mapping = lip*(z-prox(z-(z@gram-corr+1e-4*z)/lip))
        residual = np.max(np.abs(mapping), axis=1)
        if np.all(residual <= 1e-4):
            break
    error = q-z@dictionary
    objective = .5*np.sum(error*error, 1)+lam1*np.sum(np.abs(z), 1)+.5e-4*np.sum(z*z, 1)
    for group in groups:
        objective += lamg*np.sqrt(len(group))*np.linalg.norm(z[:, group], axis=1)
    return objective, residual <= 1e-4, residual, iteration+1


def fit_a009(frame, grouping=True, wrong=False, ridge=False):
    f, b, *_ = frame.banks()
    if not len(f) or not len(b) or frame.wf.sum() <= 0 or frame.wb.sum() <= 0:
        return _plain(frame)
    distance = sqdist(f, b)
    f_to_b, b_to_f = np.argmin(distance, 1), np.argmin(distance, 0)
    pairs = [(i, int(j)) for i, j in enumerate(f_to_b) if b_to_f[j] == i]
    if wrong and len(pairs) > 1:
        pairs = [(p[0], pairs[(k+1)%len(pairs)][1]) for k, p in enumerate(pairs)]
    pf = [i for i in range(len(f)) if i not in {p[0] for p in pairs}]
    pb = [j for j in range(len(b)) if j not in {p[1] for p in pairs}]
    shared = np.array([v for i, j in pairs for v in (f[i], b[j])]).reshape(-1, frame.x.shape[1])
    # Every original column remains; matched columns enter the identical shared
    # group in both removal solves, never a mean or a different penalty budget.
    d_no_f, d_no_b = np.r_[shared, b[pb]], np.r_[shared, f[pf]]
    groups = [np.array([2*k,2*k+1]) for k in range(len(pairs))] if grouping else []
    def predict(q, ids, role):
        score, base_info = b0(frame, q)
        failed = np.zeros(len(q), bool); residuals=[]; steps=[]
        for start in range(0, len(q), 128):
            z=q[start:start+128]
            if ridge:
                values=[]
                for dictionary in (d_no_f,d_no_b):
                    coef=np.linalg.solve(dictionary@dictionary.T+.01*np.eye(len(dictionary)), dictionary@z.T)
                    residual=z-coef.T@dictionary
                    values.append(.5*np.sum(residual*residual,1)+.005*np.sum(coef*coef,0))
                score[start:start+len(z)]=values[0]-values[1]
                continue
            vf,gf,rf,nf=_sparse_group(z,d_no_f,groups,lamg=.01 if grouping else 0.)
            vb,gb,rb,nb=_sparse_group(z,d_no_b,groups,lamg=.01 if grouping else 0.)
            good=gf&gb; out=score[start:start+len(z)]; out[good]=(vf-vb)[good]
            failed[start:start+len(z)]=~good
            residuals.append(float(max(rf.max(initial=0),rb.max(initial=0)))); steps.append(max(nf,nb))
        return score, dict(base_info, failed_two_solve_points=int(failed.sum()), points=len(q),
                          max_KKT=max(residuals,default=0),max_steps=max(steps,default=0),_fallback_mask=failed)
    return predict, {"shared_groups": len(pairs), "shared_columns": len(shared),
        "private_F_columns": len(pf),"private_B_columns":len(pb), "grouped":grouping,
        "wrong_groups":wrong,"two_objectives_not_three":True,"epsilon":1e-4}


def _coordinate_sparse_solver(q, dictionary, lambda_e, budget=10):
    """Alternating actual l1 atom and coordinate-error solves; not a linear head."""
    gram=dictionary@dictionary.T
    lip=max(float(np.linalg.eigvalsh(gram)[-1]),EPS)
    coef=np.zeros((len(q),len(dictionary))); error=np.zeros_like(q)
    for iteration in range(30):
        gradient=(coef@dictionary+error-q)@dictionary.T
        raw=coef-gradient/lip
        coef=np.sign(raw)*np.maximum(np.abs(raw)-.01/lip,0)
        raw=q-coef@dictionary
        error=np.sign(raw)*np.maximum(np.abs(raw)-lambda_e,0)
    count=np.count_nonzero(np.abs(error)>1e-10,axis=1)
    sparse=count<=int(np.floor(q.shape[1]*.1))
    # The actual contract does not increase e budget to salvage an explanation.
    error[~sparse]=0
    if np.any(~sparse):
        # Re-optimize the no-e route rather than reusing e-contaminated weights.
        index=np.flatnonzero(~sparse)
        cc=np.zeros((len(index),len(dictionary)))
        for _ in range(30):
            value=cc-(cc@dictionary-q[index])@dictionary.T/lip
            cc=np.sign(value)*np.maximum(np.abs(value)-.01/lip,0)
        coef[index]=cc
    clean=q-error
    residual=clean-coef@dictionary
    cost=(np.sum(residual*residual,1)+.02*np.sum(np.abs(coef),1))/np.maximum(np.sum(clean*clean,1),EPS)
    return cost,sparse,count


def fit_a010(frame, no_error=False):
    f,b,*_=frame.banks()
    if not len(f) or not len(b):
        return _plain(frame)
    if not no_error:
        evidence=require_artifact(frame.ep,"A010_coordinate_sparse_cross_image_evidence")
        if not isinstance(evidence,dict) or "passed" not in evidence or not evidence.get("source_binding"):
            raise ObservationUnavailable("A010 needs source-bound real cross-image residual sparsity vs rotation evidence, not source-only residuals")
        if not evidence["passed"]:
            return _plain(frame)
        lam=float(evidence.get("lambda_e",0.))
        if lam<=0 or not np.isfinite(lam):
            raise ObservationUnavailable("A010 positive finite lambda_e must come from reference-fold residual calibration")
    else:
        lam=2.
    def predict(q,ids,role):
        cf,sf,nf=_coordinate_sparse_solver(q,f,lam)
        cb,sb,nb=_coordinate_sparse_solver(q,b,lam)
        return cb-cf,{"FG_sparse_error_points":int(sf.sum()),"BG_sparse_error_points":int(sb.sum()),
            "FG_no_error_reopt_points":int((~sf).sum()),"BG_no_error_reopt_points":int((~sb).sum()),
            "coordinate_cap":int(q.shape[1]*.1),"max_iterations":30}
    return predict,{"lambda_atom":.01,"lambda_e":lam,"coordinate_cap_fraction":.1,"precheck_passed":not no_error}


def _matching_pursuit(q,bank,max_atoms=8,quantization=None,sigma=1.):
    residual=q.copy(); index=np.zeros((len(q),max_atoms),int); coef=np.zeros((len(q),max_atoms))
    for j in range(max_atoms):
        response=residual@bank.T
        idx=np.argmax(np.abs(response),1)
        value=response[np.arange(len(q)),idx]
        if quantization is not None:
            lo,hi=quantization
            value=lo+np.rint(np.clip((value-lo)/max(hi-lo,EPS),0,1)*255)*(hi-lo)/255
        index[:,j]=idx;coef[:,j]=value
        residual-=value[:,None]*bank[idx]
    cost=.5*np.sum(residual*residual,1)/(sigma*sigma*np.log(2))
    used=np.count_nonzero(np.abs(coef)>1e-8,1)
    cost+=used*(np.ceil(np.log2(max(len(bank),1)))+8)
    return cost,residual,coef,used


def fit_a011(frame, bits=True):
    f,b,*_=frame.banks()
    if not len(f) or not len(b) or frame.wf.sum()<=0 or frame.wb.sum()<=0:
        return _plain(frame)
    coefficients=[];errors=[]
    for block in range(4):
        source=Frame.make(frame.ep, frame.train&(frame.blocks!=block))
        sf,sb,*_=source.banks()
        for ids,bank in ((frame.fids[frame.blocks[frame.fids]==block],sf),(frame.bids[frame.blocks[frame.bids]==block],sb)):
            if len(ids) and len(bank):
                _,res,c,_=_matching_pursuit(frame.x[ids],bank)
                coefficients.extend(c.ravel().tolist());errors.extend(np.sum(res*res,1).tolist())
    if not coefficients or not errors:
        return _plain(frame)
    lo,hi=map(float,np.quantile(coefficients,[.005,.995]))
    sigma=max(float(np.sqrt(np.mean(errors)/frame.x.shape[1])),1e-4)
    def predict(q,ids,role):
        cf,rf,af,nf=_matching_pursuit(q,f,quantization=(lo,hi),sigma=sigma)
        cb,rb,ab,nb=_matching_pursuit(q,b,quantization=(lo,hi),sigma=sigma)
        if not bits:
            cf=np.sum(rf*rf,1);cb=np.sum(rb*rb,1)
        return cb-cf,{"FG_atoms_mean":float(nf.mean()) if len(q) else 0,
                      "BG_atoms_mean":float(nb.mean()) if len(q) else 0,"max_atoms":8}
    return predict,{"coefficient_99pct_range":[lo,hi],"quantization_bins":256,"pooled_Gaussian_sigma":sigma,
                    "description_length_bits":bits,"Gaussian_role_common_constant_cancels":True}


def _support_ball(frame):
    f,b,fi,bi,_=frame.banks()
    if len(f)<2 or not len(b) or np.any(fi<0):
        return None
    gram=f@f.T;diag=np.diag(gram);nu=.1;cap=1/(nu*len(f))
    def objective(a):
        return float(a@gram@a-a@diag),2*gram@a-diag
    solved=minimize(objective,np.full(len(f),1/len(f)),jac=True,method="SLSQP",
        bounds=[(0,cap)]*len(f),constraints={"type":"eq","fun":lambda a:a.sum()-1,"jac":lambda a:np.ones(len(a))},
        options={"maxiter":100,"ftol":1e-10})
    if not solved.success:
        return None
    center=solved.x@f
    distance=np.sum((f-center)**2,1)
    radius=float(np.quantile(distance,.95))
    # Cross-block observed support fixes the final covering radius; this is not
    # a radius extrapolated from query points.
    held_errors=[]
    for block in range(4):
        src=frame.fids[frame.blocks[frame.fids]!=block]
        held=frame.fids[frame.blocks[frame.fids]==block]
        if len(src) and len(held):
            local_center=np.sum(frame.x[src]*frame.wf[src,None],0)/frame.wf[src].sum()
            held_errors.extend(np.sum((frame.x[held]-local_center)**2,1).tolist())
    if held_errors:
        radius=max(radius,float(np.quantile(held_errors,.95)))
    return f,b,center,max(radius,1e-6),int(solved.nit)


def fit_a012(frame,cuts=True):
    packet=_support_ball(frame)
    if packet is None:
        return _plain(frame)
    f,b,center,radius,iterations=packet
    planes=[]
    for negative in b:
        delta=negative-center;length=np.linalg.norm(delta)
        if length*length>=radius:
            continue
        if length<=EPS:
            return _plain(frame)
        n=delta/length;limit=float(n@(center+negative)/2)
        planes.append((n,limit))
    if cuts and planes:
        retained=np.ones(len(f),bool)
        for n,limit in planes:
            retained&=f@n<=limit
        if not retained.any():
            predict,info=_plain(frame)
            return predict,dict(info,degeneration="BG_cuts_reject_all_training_FG",svdd_iterations=iterations)
    def predict(q,ids,role):
        score=(radius-np.sum((q-center)**2,1))/radius
        if cuts:
            for n,limit in planes:
                score=np.minimum(score,(limit-q@n)/np.sqrt(radius))
        return score,{"BG_halfspaces":len(planes) if cuts else 0}
    return predict,{"SVDD_soft_nu":.1,"radius2":radius,"center_norm":float(np.linalg.norm(center)),
                    "svdd_iterations":iterations,"BG_halfspaces":len(planes),"cuts":cuts}


def _bounded_bg_pieces(frame):
    f,b,fi,bi,_=frame.banks()
    pieces=[]
    for j,center in enumerate(b):
        near=np.argsort(sqdist(b,center[None])[:,0],kind="stable")[:min(16,len(b))]
        u=rank_basis(b[near]-center,4)
        if not u.shape[1] or bi[j]<0:
            continue
        held=frame.bids[frame.blocks[frame.bids]!=frame.blocks[bi[j]]]
        if not len(held):
            continue
        held=held[np.argsort(sqdist(frame.x[held],center[None])[:,0],kind="stable")[:16]]
        delta=frame.x[held]-center;t=delta@u
        axes=np.maximum(np.quantile(np.abs(t),.95,axis=0),1e-6)
        axes*=max(float(np.quantile(np.sqrt(np.sum((t/axes)**2,1)),.95)),1.)
        normal=max(float(np.quantile(np.linalg.norm(delta-t@u.T,axis=1),.95)),1e-6)
        pieces.append((center,u,axes,normal))
    return f,b,pieces


def fit_a013(frame,unbounded=False):
    from .group_001_075 import _ellipsoid_projection
    f,b,pieces=_bounded_bg_pieces(frame)
    if not pieces or not len(f):
        return _plain(frame)
    fg_source=[]
    for block in range(4):
        src=frame.fids[frame.blocks[frame.fids]!=block];held=frame.fids[frame.blocks[frame.fids]==block]
        if len(src) and len(held):
            fg_source.extend(nearest_mean_distance(frame.x[held],frame.x[fps(frame.x,src,64)]).tolist())
    radius=max(float(np.quantile(fg_source,.95)) if fg_source else 2.,EPS)
    def predict(q,ids,role):
        db=np.full(len(q),np.inf)
        for center,u,axes,normal in pieces:
            t=(q-center)@u;orth=q-center-t@u.T
            tangent=np.zeros(len(q)) if unbounded else np.sum((t-_ellipsoid_projection(t,axes))**2,1)
            normal_residual=np.linalg.norm(orth,axis=1) if unbounded else np.maximum(np.linalg.norm(orth,axis=1)-normal,0)
            db=np.minimum(db,tangent+normal_residual**2)
        df=nearest_mean_distance(q,f)
        return np.minimum((db-df)/2,(radius-df)/max(radius,EPS)),{"BG_pieces":len(pieces)}
    return predict,{"BG_pieces":len(pieces),"FG_support_radius2":radius,"unbounded":unbounded}


def _metric_pairs(frame, ids):
    pairs_same=[];pairs_cross=[]
    ids=np.asarray(ids,int)
    for role_ids,other in ((np.intersect1d(ids,frame.fids),np.intersect1d(ids,frame.bids)),
                           (np.intersect1d(ids,frame.bids),np.intersect1d(ids,frame.fids))):
        selected=fps(frame.x,role_ids,32)
        if len(selected)<2 or not len(other):
            continue
        d=sqdist(frame.x[selected],frame.x[selected]);np.fill_diagonal(d,np.inf)
        nn=np.argmin(d,1)
        pairs_same.extend((frame.x[selected]-frame.x[selected[nn]]).tolist())
        neg=fps(frame.x,other,32)
        cross=sqdist(frame.x[selected],frame.x[neg])
        pairs_cross.extend((frame.x[selected]-frame.x[neg[np.argmin(cross,1)]]).tolist())
    return np.asarray(pairs_same).reshape(-1,frame.x.shape[1]),np.asarray(pairs_cross).reshape(-1,frame.x.shape[1])


def _learn_metric(same,cross,u,diagonal=False):
    d=np.ones(same.shape[1]);c=np.zeros((u.shape[1],u.shape[1]))
    if not len(same) or not len(cross):
        return d,c,0.,0
    s2=same*same;b2=cross*cross;su=same@u;bu=cross@u
    target=max(float(np.median(np.sum(b2,1))),1e-3)
    best=(d.copy(),c.copy());bestloss=np.inf
    for iteration in range(30):
        dist=b2@d+np.einsum('ni,ij,nj->n',bu,c,bu)
        active=dist<target
        loss=float(np.mean(s2@d+np.einsum('ni,ij,nj->n',su,c,su))+np.mean(np.maximum(target-dist,0))
            +np.mean((d-1)**2)+np.sum(c*c))
        if loss<bestloss:
            bestloss=loss;best=(d.copy(),c.copy())
        gd=s2.mean(0)-np.mean(b2*active[:,None],0)+2*(d-1)/len(d)
        gc=su.T@su/len(su)-bu[active].T@bu[active]/len(bu)+2*c
        d=np.maximum(d-.01*gd,1e-4)
        if not diagonal:
            cc=c-.01*gc;value,basis=np.linalg.eigh((cc+cc.T)/2)
            c=(basis*np.maximum(value,0))@basis.T
    return *best,bestloss,iteration+1


def fit_a014(frame,global_metric=False,diagonal=False):
    if not len(frame.fids) or not len(frame.bids):
        return _plain(frame)
    ids=np.flatnonzero(frame.train)
    pc_ids=fps(frame.x,ids,64);u=rank_basis(frame.x[pc_ids]-frame.x[pc_ids].mean(0),16)
    centers,assignment,groups=spherical_modes(frame.x,ids,1 if global_metric else 8)
    metrics=[]
    for group in groups:
        same,cross=_metric_pairs(frame,group)
        metrics.append(_learn_metric(same,cross,u,diagonal))
    f,b,*_=frame.banks()
    def metric_distance(q,bank,d,c):
        # Exact diag+lowrank quadratic distance; no D×D materialization.
        sd=np.sqrt(d);qv=q@u;bv=bank@u
        out=sqdist(q*sd,bank*sd)
        out+=np.einsum('ni,ij,nj->n',qv,c,qv)[:,None]+np.einsum('ni,ij,nj->n',bv,c,bv)[None,:]-2*qv@c@bv.T
        return np.maximum(out,0)
    def predict(q,ids,role):
        route=np.argmin(sqdist(q,centers),1);score=np.zeros(len(q))
        for k,(d,c,loss,steps) in enumerate(metrics):
            selected=route==k
            if not selected.any():continue
            df=metric_distance(q[selected],f,d,c);db=metric_distance(q[selected],b,d,c)
            kf,kb=min(5,len(f)),min(5,len(b))
            score[selected]=(np.partition(db,kb-1,axis=1)[:,:kb].mean(1)-np.partition(df,kf-1,axis=1)[:,:kf].mean(1))/2
        return score,{"domain_point_counts":np.bincount(route,minlength=len(centers)).tolist()}
    return predict,{"domains":len(centers),"PCA_rank":u.shape[1],"objectives":[m[2] for m in metrics],
        "iterations":[m[3] for m in metrics],"diagonal":diagonal,"global_metric":global_metric}


def _hull_projection(target,points,initial):
    """Feasible Frank-Wolfe convex-combination projection,<=30 updates."""
    z=initial.copy()
    for _ in range(30):
        gradient=z-target
        vertex=points[np.argmin(points@gradient)]
        direction=vertex-z
        alpha=np.clip(-(gradient@direction)/max(direction@direction,EPS),0,1)
        new=z+alpha*direction
        if np.linalg.norm(new-z)<1e-6:
            z=new;break
        z=new
    return z


def fit_a015(frame,move=True,medoids=False):
    if not len(frame.fids) or not len(frame.bids):
        return _plain(frame)
    selected=[fps(frame.x,role,8) for role in (frame.fids,frame.bids)]
    prototypes=np.r_[frame.x[selected[0]],frame.x[selected[1]]]
    labels=np.r_[np.ones(len(selected[0])),-np.ones(len(selected[1]))]
    groups=[]
    for role,sel in zip((frame.fids,frame.bids),selected):
        route=np.argmin(sqdist(frame.x[role],frame.x[sel]),1)
        groups.extend([frame.x[role[route==j]] for j in range(len(sel))])
    train=np.r_[frame.fids,frame.bids];x=frame.x[train];y=np.r_[np.ones(len(frame.fids)),-np.ones(len(frame.bids))]
    weights=np.where(y>0,.5/len(frame.fids),.5/len(frame.bids))
    best=prototypes.copy();bestloss=np.inf;visited=[]
    for iteration in range(30 if move else 1):
        distance=sqdist(x,prototypes)
        # Explicit per-row class masks avoid NumPy's flattened boolean indexing.
        same=np.where(labels[None,:]==y[:,None],distance,np.inf)
        other=np.where(labels[None,:]!=y[:,None],distance,np.inf)
        ip,im=np.argmin(same,1),np.argmin(other,1)
        dp,dm=distance[np.arange(len(x)),ip],distance[np.arange(len(x)),im]
        denom=np.maximum(dp+dm,EPS);mu=(dp-dm)/denom
        phi=expit(mu);loss=float(weights@phi);visited.append(loss)
        if loss<bestloss:bestloss=loss;best=prototypes.copy()
        if not move:break
        slope=phi*(1-phi)*weights
        grad=np.zeros_like(prototypes)
        for j in range(len(prototypes)):
            pos=ip==j;neg=im==j
            grad[j]+=np.sum((2*slope[pos]*2*dm[pos]/denom[pos]**2)[:,None]*(prototypes[j]-x[pos]),axis=0)
            grad[j]-=np.sum((2*slope[neg]*2*dp[neg]/denom[neg]**2)[:,None]*(prototypes[j]-x[neg]),axis=0)
        for j,points in enumerate(groups):
            target=prototypes[j]-.1*grad[j]
            prototypes[j]=_hull_projection(target,points,prototypes[j])
    prototypes=best
    if medoids:
        prototypes=np.array([points[np.argmin(sqdist(points,points.mean(0)[None])[:,0])] for points in groups])
    f,b=prototypes[labels>0],prototypes[labels<0]
    def predict(q,ids,role):
        return (nearest_mean_distance(q,b,1)-nearest_mean_distance(q,f,1))/2,{}
    return predict,{"FG_prototypes":len(f),"BG_prototypes":len(b),"iterations":len(visited),
        "best_GL VQ_objective":bestloss,"objectives":visited,"move":move,"medoids":medoids,
        "convex_hull_feasibility_by_construction":True}


def fit_a016(frame,aggregation="worstthird",continuous=False):
    f,b,fi,bi,_=frame.banks()
    if not len(f) or not len(b):
        return _plain(frame)
    cross=sqdist(f,b);candidates=unit(f-b[np.argmin(cross,1)])
    directions=[]
    for direction in candidates:
        if directions and np.max(np.abs(np.array(directions)@direction))>.95:continue
        errors=[]
        for block in range(4):
            train=frame.train&(frame.blocks!=block);held=frame.train&(frame.blocks==block)
            if frame.wf[train].sum()<=0 or frame.wb[train].sum()<=0 or frame.wf[held].sum()<=0 or frame.wb[held].sum()<=0:continue
            cut,_=threshold(frame.x[train]@direction,frame.wf[train],frame.wb[train])
            positive=frame.x[held]@direction>cut
            error=.5*(frame.wf[held][~positive].sum()/frame.wf[held].sum()+frame.wb[held][positive].sum()/frame.wb[held].sum())
            errors.append(error)
        if errors and max(errors)<.5:directions.append(direction)
        if len(directions)>=32:break
    if not directions:return _plain(frame)
    u=np.array(directions).T
    pf,pb=frame.x[frame.fids]@u,frame.x[frame.bids]@u
    loF,hiF=np.quantile(pf,[.05,.95],axis=0);loB,hiB=np.quantile(pb,[.05,.95],axis=0)
    if continuous:
        model=fit_rbf(frame,frame.x@u)
        bank,coef,width=model
        return lambda q,ids,role:(np.exp(-sqdist(q@u,bank)/width)@coef,{}),{"directions":u.shape[1],"control":"same_direction_RBF"}
    support=[]
    for block in range(4):
        src=frame.fids[frame.blocks[frame.fids]!=block];held=frame.fids[frame.blocks[frame.fids]==block]
        if len(src) and len(held):support.extend(nearest_mean_distance(frame.x[held],frame.x[fps(frame.x,src,64)]).tolist())
    radius=max(float(np.quantile(support,.95)) if support else 2.,EPS)
    def predict(q,ids,role):
        p=q@u
        df=np.maximum(loF-p,0)+np.maximum(p-hiF,0)
        db=np.maximum(loB-p,0)+np.maximum(p-hiB,0)
        margin=db-df
        k=max(1,int(np.ceil(u.shape[1]/3)))
        score=np.partition(margin,k-1,axis=1)[:,:k].mean(1) if aggregation=="worstthird" else (margin.mean(1) if aggregation=="mean" else margin.max(1))
        score=np.minimum(score,(radius-nearest_mean_distance(q,f))/max(radius,EPS))
        return score,{}
    return predict,{"directions":u.shape[1],"interval_quantiles":[.05,.95],"FG_support_radius2":radius,"aggregation":aggregation}


def fit_a017(frame,domains=True):
    f,b,fi,bi,_=frame.banks()
    if not len(f) or not len(b) or np.any(fi<0) or np.any(bi<0):return _plain(frame)
    packets=[]
    for bank,indices,own,other in ((f,fi,frame.fids,frame.bids),(b,bi,frame.bids,frame.fids)):
        records=[]
        for anchor,index in zip(bank,indices):
            cross_own=own[frame.blocks[own]!=frame.blocks[index]]
            cross_other=other[frame.blocks[other]!=frame.blocks[index]]
            if not len(cross_own) or not len(cross_other):continue
            d=sqdist(frame.x[cross_own],anchor[None])[:,0]
            radius=float(np.quantile(np.sort(d)[:min(8,len(d))],.9))
            true_rate=float(np.mean(d<=radius))
            false_rate=float(np.mean(sqdist(frame.x[cross_other],anchor[None])[:,0]<=radius))
            reliability=(true_rate-false_rate)/max(true_rate+false_rate,EPS)
            if reliability>0:records.append((anchor,radius,reliability,int(index)))
        packets.append(records)
    if not packets[0] or not packets[1]:return _plain(frame)
    def predict(q,ids,role):
        base,info=b0(frame,q);role_scores=[];hits=[]
        for bank,records in zip((f,b),packets):
            anchors=np.array([r[0] for r in records]);radii=np.array([r[1] for r in records])
            hit=sqdist(q,anchors)<=radii if domains else np.ones((len(q),len(anchors)),bool)
            score=np.zeros(len(q));any_hit=hit.any(1)
            response=q@anchors.T
            for i in np.flatnonzero(any_hit):
                z=response[i,hit[i]]/.07
                score[i]=.07*(np.log(np.exp(z-z.max()).mean())+z.max())
            # One empty role uses its ORIGINAL fullbank response; only both
            # empty yields exactly B0. Reliability is never a global multiplier.
            z=q@bank.T/.07
            maximum=z.max(1)
            fallback=.07*(np.log(np.exp(z-maximum[:,None]).mean(1))+maximum)
            score[~any_hit]=fallback[~any_hit]
            role_scores.append(score);hits.append(any_hit)
        active=hits[0]|hits[1]
        base[active]=(role_scores[0]-role_scores[1])[active]
        return base,dict(info,qualified_domain_points=int(active.sum()),points=len(q),_fallback_mask=~active)
    return predict,{"qualified_FG":len(packets[0]),"qualified_BG":len(packets[1]),
        "reliabilities":[[r[2] for r in p] for p in packets],"domains":domains,
        "reliability_only_qualifies_not_multiplies_similarity":True}


def _curvature_profile(q,bank):
    """Actual Hessian weighted covariance, block power rank3<=10."""
    response=q@bank.T/.07
    maximum=response.max(1);prob=np.exp(response-maximum[:,None]);prob/=prob.sum(1,keepdims=True)
    score=.07*(maximum+np.log(np.exp(response-maximum[:,None]).mean(1)))
    top=np.zeros(len(q))
    rank=min(3,bank.shape[1],len(bank)-1)
    if rank<=0:return score,top
    initial=np.random.default_rng(0).standard_normal((bank.shape[1],rank))
    initial=np.linalg.qr(initial)[0]
    for i in range(len(q)):
        p=prob[i];mean=p@bank
        def multiply(v):return (bank.T@(p[:,None]*(bank@v))-mean[:,None]*(mean@v)[None,:])/.07
        vectors=initial.copy()
        for _ in range(10):vectors=np.linalg.qr(multiply(vectors))[0][:,:rank]
        small=vectors.T@multiply(vectors)
        top[i]=np.maximum(np.linalg.eigvalsh((small+small.T)/2),0).sum()
    return score,top


def fit_a018(frame,curvature=True):
    f,b,*_=frame.banks()
    if not len(f) or not len(b) or frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    def descriptors(x):
        sf,cf=_curvature_profile(x,f);sb,cb=_curvature_profile(x,b)
        return np.c_[sf-sb,cf-cb] if curvature else (sf-sb)[:,None]
    source=descriptors(frame.x[frame.train]);indices=np.flatnonzero(frame.train)
    full=np.zeros((len(frame.x),source.shape[1]));full[indices]=source
    coef=fit_ridge(frame,full)
    def predict(q,ids,role):
        z=descriptors(q)
        return z@coef[:-1]+coef[-1],{"Hessian_power_steps":10,"Hessian_top_rank":min(3,len(f)-1,len(b)-1,q.shape[1])}
    return predict,{"features":"LSE margin + sum top3 FG/BG Hessian eigenvalue difference" if curvature else "LSE_margin_only",
                    "T":.07,"head_L2":1.,"block_power_steps":10,"actual_Hessian_no_state_iteration":True}


def _edge_repairs(frame,random=False):
    valid=np.flatnonzero(frame.train)
    if len(valid)<3:return []
    x=frame.x[valid]
    distance=sqdist(x,x);np.fill_diagonal(distance,np.inf)
    nearest=np.argsort(distance,axis=1,kind="stable")[:,:min(8,len(valid)-1)]
    fg=set(map(int,frame.fids));bg=set(map(int,frame.bids));edges=[]
    for i,neighbors in enumerate(nearest):
        for j in neighbors:
            if i>=j or i not in nearest[j]:continue
            a,b=int(valid[i]),int(valid[j])
            if (a in fg and b in bg) or (b in fg and a in bg):edges.append((float(distance[i,j]),a,b,i,int(j)))
    edges.sort(key=lambda e:(e[0],e[1],e[2]));repairs=[]
    for _,a,b,i,j in edges[:64]:
        sameA=valid[nearest[i]][[int(v) in (fg if a in fg else bg) for v in valid[nearest[i]]]]
        sameB=valid[nearest[j]][[int(v) in (fg if b in fg else bg) for v in valid[nearest[j]]]]
        if not len(sameA) or not len(sameB):continue
        ua=rank_basis(frame.x[sameA]-frame.x[a],1);ub=rank_basis(frame.x[sameB]-frame.x[b],1)
        nuisance=rank_basis(np.r_[ua.T,ub.T])
        delta=frame.x[a]-frame.x[b];normal=delta-delta@nuisance@nuisance.T
        secondary=(ua[:,0] if ua.shape[1] else np.zeros_like(delta))-(ub[:,0] if ub.shape[1] else np.zeros_like(delta))
        u=rank_basis(np.array([normal,secondary]),2)
        if not u.shape[1]:continue
        if random:u=rank_basis(np.random.default_rng(a+b).standard_normal((u.shape[1],frame.x.shape[1])))
        actual=delta@u
        same=np.r_[frame.x[sameA]-frame.x[a],frame.x[sameB]-frame.x[b]]
        bound=.1*np.sum(same*same,1);projected=same@u
        g=np.zeros((u.shape[1],u.shape[1]))
        for _ in range(30):
            g+=.1*(np.outer(actual,actual)-.02*g)
            eigen,vectors=np.linalg.eigh((g+g.T)/2);g=(vectors*np.maximum(eigen,0))@vectors.T
            g/=max(np.linalg.norm(g,'fro'),1.)
            increments=np.einsum('ni,ij,nj->n',projected,g,projected)
            factor=min(1.,float(np.min(np.divide(bound,increments,out=np.ones(len(bound)),where=increments>EPS))))
            g*=max(factor,0.)
        ra=float(np.quantile(sqdist(frame.x[sameA],frame.x[a:a+1])[:,0],.9))
        rb=float(np.quantile(sqdist(frame.x[sameB],frame.x[b:b+1])[:,0],.9))
        repairs.append((a,b,u,g,ra,rb))
    return repairs


def fit_a019(frame,random=False,global_route=False):
    repairs=_edge_repairs(frame,random)
    f,b,*_=frame.banks()
    if not repairs or not len(f) or not len(b):
        predict,info=_plain(frame)
        return predict,dict(info,card_id="A019",degeneration="no_eligible_mutual_crossclass_edge_repair",contradicted_mutual_edges=len(repairs))
    def predict(q,ids,role):
        df=sqdist(q,f);db=sqdist(q,b);mass=np.zeros(len(q));inf=np.zeros_like(df);inb=np.zeros_like(db)
        for a,bb,u,g,ra,rb in repairs:
            hit=(sqdist(q,frame.x[a:a+1])[:,0]<=ra)|(sqdist(q,frame.x[bb:bb+1])[:,0]<=rb)
            if global_route:hit[:]=True
            selected=np.flatnonzero(hit)
            if not len(selected):continue
            v=q[selected]@u
            for bank,acc in ((f,inf),(b,inb)):
                z=bank@u
                increment=np.einsum('ni,ij,nj->n',v,g,v)[:,None]+np.einsum('ni,ij,nj->n',z,g,z)[None,:]-2*v@g@z.T
                acc[selected]+=np.maximum(increment,0)
            mass[selected]+=1
        df+=np.divide(inf,mass[:,None],out=np.zeros_like(inf),where=mass[:,None]>0)
        db+=np.divide(inb,mass[:,None],out=np.zeros_like(inb),where=mass[:,None]>0)
        kf,kb=min(5,len(f)),min(5,len(b))
        score=(np.partition(db,kb-1,axis=1)[:,:kb].mean(1)-np.partition(df,kf-1,axis=1)[:,:kf].mean(1))/2
        return score,{"local_repair_points":int((mass>0).sum()),"points":len(q),"outside_exact_original":not global_route}
    return predict,{"contradicted_mutual_edges":len(repairs),"ranks":[r[2].shape[1] for r in repairs],
        "PSD_frobenius_norms":[float(np.linalg.norm(r[3],'fro')) for r in repairs],"steps":30,
        "sameclass_increment_cap_fraction":.1,"random_directions":random,"global_route":global_route}


def fit_a020(frame,unconstrained=False):
    f,b,*_=frame.banks()
    if not len(f) or not len(b) or frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    anchors=np.r_[f,b];u=rank_basis(anchors-anchors.mean(0),16);rank=u.shape[1]
    if not rank:return _plain(frame)
    distances=[]
    for block in range(4):
        for own in (frame.fids,frame.bids):
            src=own[frame.blocks[own]!=block];held=own[frame.blocks[own]==block]
            if len(src) and len(held):distances.extend(np.sqrt(nearest_mean_distance(frame.x[held],frame.x[fps(frame.x,src,64)],1)).tolist())
    h=float(np.quantile(distances,.9)) if distances else 0.
    if h<=EPS:return _plain(frame)
    train=np.flatnonzero(frame.train)
    # Pair endpoints are real legal training observations, not query labels.
    pairs=[];weights=[];targets=[]
    for role,other in ((frame.fids,frame.bids),(frame.bids,frame.fids)):
        selected=fps(frame.x,role,32)
        if len(selected)<2:continue
        d=sqdist(frame.x[selected],frame.x[selected]);np.fill_diagonal(d,np.inf)
        opposite=fps(frame.x,other,32)
        neg=np.argmin(sqdist(frame.x[selected],frame.x[opposite]),1)
        for k,index in enumerate(selected):
            pairs.extend([(int(index),int(selected[np.argmin(d[k])])),(int(index),int(opposite[neg[k]]))])
            weights.extend([1.,-1.])
    if not pairs:return _plain(frame)
    pairs=np.array(pairs);type_=np.array(weights)
    xall=frame.x;z=xall@u
    support=np.maximum(0,1-np.sqrt(nearest_mean_distance(xall,anchors,1))/h)
    base=np.sum((xall[pairs[:,0]]-xall[pairs[:,1]])**2,1)
    goal=1.05*float(np.median(base[type_<0]))
    w1=.1*np.eye(rank);w2=np.zeros((rank,rank));best=(w1.copy(),w2.copy());bestloss=np.inf;history=[]
    for iteration in range(30):
        hidden=np.tanh(z@w1.T);residual=support[:,None]*(hidden@w2.T)
        dz=(z+residual)[pairs[:,0]]-(z+residual)[pairs[:,1]]
        old=z[pairs[:,0]]-z[pairs[:,1]]
        dist=base+np.sum(dz*dz-old*old,1)
        active=(type_<0)&(dist<goal)
        pairgradient=np.where(type_>0,1.,np.where(active,-1.,0.))+.2*(dist-base)
        loss=float(np.mean(np.where(type_>0,dist,np.maximum(goal-dist,0)))+.1*np.mean((dist-base)**2))
        history.append(loss)
        if loss<bestloss:bestloss=loss;best=(w1.copy(),w2.copy())
        gradz=np.zeros_like(z)
        v=2*pairgradient[:,None]*dz/len(pairs)
        np.add.at(gradz,pairs[:,0],v);np.add.at(gradz,pairs[:,1],-v)
        gw2=(support[:,None]*gradz).T@hidden
        gh=(support[:,None]*gradz)@w2
        gw1=(gh*(1-hidden*hidden)).T@z
        w2-=.01*gw2;w1-=.01*gw1
        if not unconstrained:
            factor=np.linalg.norm(w2,2)*(np.sqrt(rank)/h+np.linalg.norm(w1,2))
            if factor>=.5:w2*=.499/max(factor,EPS)
    w1,w2=best
    certificate=float(np.linalg.norm(w2,2)*(np.sqrt(rank)/h+np.linalg.norm(w1,2)))
    if not unconstrained and certificate>=.5:return _plain(frame)
    def transform(q):
        gate=np.maximum(0,1-np.sqrt(nearest_mean_distance(q,anchors,1))/h)
        return q+(gate[:,None]*(np.tanh((q@u)@w1.T)@w2.T))@u.T,gate
    ff,_=transform(f);bb,_=transform(b)
    def predict(q,ids,role):
        transformed,gate=transform(q)
        return (nearest_mean_distance(transformed,bb)-nearest_mean_distance(transformed,ff))/2,{
            "nonzero_support_points":int((gate>0).sum()),"points":len(q),"outside_exact_identity":True}
    return predict,{"rank":rank,"support_h":h,"whole_residual_Lipschitz_bound":certificate,
        "invertibility_constraint_active":not unconstrained,"steps":30,"objectives":history,
        "best_objective":bestloss,"W2_norm":float(np.linalg.norm(w2))}


def install(register,requirements):
    register("A009",fit_a009,(
        "64 FPS columns perrole; every mutualNN pair sharedtwo-column group, privatecolumns exclude paired originals.",
        "Fixedlambda1=.01,lambdag=.01,eps=1e-4; FISTA sparse-group prox<=30, infinitynorm proximal-gradient KKT mapping<=1e-4 perpoint.",),
        (("ungrouped_lasso",lambda f:fit_a009(f,grouping=False)),("wrong_shared_groups",lambda f:fit_a009(f,wrong=True)),
         ("same_columns_ridge",lambda f:fit_a009(f,ridge=True))))
    register("A010",fit_a010,(
        "Actual source-bound cross-image coordinate-residual-vs-rotation evidence is required; source-only residuals are not silently accepted as that observation.",
        "Atoml1=.01, errorl1lambda fromreference-fold evidence; alternatingprox30; e nonzero<=floor(.1D), else30-step no-e reoptimization.",),
        (("no_error_SRC",lambda f:fit_a010(f,no_error=True)),))
    requirements["A010"]=["A010_coordinate_sparse_cross_image_evidence"]
    register("A011",fit_a011,(
        "Ordered matchingpursuit8atoms; one-role source-fold coefficient pooledq.005/.995 range,256bins,sharedGaussian sigma frompooledcrossblock reconstruction residual, floor1e-4.",
        "Indexceil(log2K)+8bits eachnonzeroquantizedatom; Gaussian residual negative-log-density rolecommon constant cancels.",),
        (("same_quantized_dictionary_residual",lambda f:fit_a011(f,bits=False)),))
    register("A012",fit_a012,(
        "FG softSVDDnu=.1,SLSQP<=100; radius=max(trainingq.95,crossblockFG-to-trainingmean residualq.95), noqueryradius.",
        "EachBG anchor insideball cuts a center/BG Euclideanbisector halfspace; center-coincidentBG or allFG eliminated=>invalidballB0.",),
        (("SVDD_without_BG_cuts",lambda f:fit_a012(f,cuts=False)),))
    register("A013",fit_a013,(
        "64BG localpieces,nearest16PCA rank4, tangentellipse/normalradius nearest16othertrainingblockBG q.95; exact product-set distance.",
        "ScoreBGnormalizedreconstruction−FG5NNdistance minFGcrossblockq.95supportmargin; anomaly alone cannot pass FGgate.",),
        (("unbounded_BG_local_space",lambda f:fit_a013(f,unbounded=True)),("original_two_role_5NN",_plain)))
    register("A014",fit_a014,(
        "8sphericalreferencefeaturedomains, globalunlabeledPCA rank16 from64FPS trainingR; domainpairs32FPS perrole nearestsame/opposite.",
        "30projected-gradient steps.01, d>=1e-4,C PSD, L2=1 normalizeddiagonal andFrobeniusC; crosshingetarget=sourcecrossdistance median, minvisitedobjective.",),
        (("same_pairs_global_metric",lambda f:fit_a014(f,global_metric=True)),("local_diagonal_only",lambda f:fit_a014(f,diagonal=True))))
    register("A015",fit_a015,(
        "8FPS prototypes perrole, owninitialVoronoi cluster convexhull; balancedGLVQsigmoid relative-distance loss,30steps at.1, minvisitedobjective.",
        "Eachprojection uses30FrankWolfe line-search convexcombination updates, alwaysfeasible; does not claim an exact hullprojection optimum.",),
        (("same_initial_FPS",lambda f:fit_a015(f,move=False)),("same_cluster_medoids",lambda f:fit_a015(f,move=False,medoids=True))))
    register("A016",fit_a016,(
        "Nearestcrossrole64anchor directions, requireeachsourcefoldbalancederror<.5, removeabs-cos>.95 redundantdirs, cap32.",
        "Roleprojectionintervals q.05/.95, outside-interval distanceBG−FG; meanworstceil(B/3), minFGcrossblock95%5NNsupportmargin.",),
        (("same_directions_mean",lambda f:fit_a016(f,aggregation="mean")),("same_directions_max",lambda f:fit_a016(f,aggregation="max")),
         ("same_directions_RBF",lambda f:fit_a016(f,continuous=True))))
    register("A017",fit_a017,(
        "Each64 FPS roleanchor supportball uses nearest8 same-role othertrainingblock distanceq.9; qualification iff crossblock truehit-rate>falsehit-rate.",
        "Reliabilityqualifies notgloballyweights; two-role eligibleanchor logmeanexpT.07; emptyrole uses ordinaryfullrole LSE, bothempty exactlysourceB0.",),
        (("qualified_without_domain",lambda f:fit_a017(f,domains=False)),("original_5NN",_plain)))
    register("A018",fit_a018,(
        "Hessian=(softmax weighted secondmoment−mean outermean)/T,T=.07; deterministicrank3 blockpower10 perrole andquerypoint.",
        "2Dreadoutuses LSEmargin andSUM top3 eigenvalueFG−BGdifference; balancedridgeL2=1, fixedhead refitinsideeveryouterfold.",),
        (("same_LSE_margin_only",lambda f:fit_a018(f,curvature=False)),))
    register("A019",fit_a019,(
        "Actualall-valid R8NN reciprocalcrosspurelabel edges cap64 bydistance/ID. Removebothendpoint sameclasslocalPC directions fromedge contrast.",
        "Rank<=2 fromremainingcontrast andlocalPCdifference;30PSDsteps.1,Frobenius<=1, scaleall same-neighborincrements<=10%original, L2=.01.",
        "Endpointball radiiq.9 sameclassneighbor distances; overlap metricincrementsmean, outsideincrementexact0, recompute5NNdistance.",),
        (("same_edges_random_U",lambda f:fit_a019(f,random=True)),("same_edges_global_route",lambda f:fit_a019(f,global_route=True))))
    register("A020",fit_a020,(
        "Reference-spanrank16;W1=.1I,W2=0;30manualgradientsteps.01, samepairdistance+oppositehinge target1.05sourcecrossmedian and.1pairdistance-distortion.",
        "supporta=max(0,1−nearestanchorEuclideandist/h), hsourcecrossblock sameclass nearest-distanceq.9; exactidentityoutside.",
        "Eachstep projectW2 tobound||W2||2(√rank/h+||W1||2)<.5 via.499 margin; preservebothW1/W2 learning, minvisitedobjective.",),
        (("same_head_without_invertibility_constraint",lambda f:fit_a020(f,unconstrained=True)),("original_5NN",_plain)))
