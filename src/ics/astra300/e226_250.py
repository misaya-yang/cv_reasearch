"""Supplied E226--E250: original RGB algorithms, explicit legal MEAN host.

Implemented kernels are connected to the common Astra adapter below.  Extra
views and the host are required resources; neither is silently synthesized.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.special import expit

from . import e_helpers_226_250 as H


@dataclass
class PixelProblem:
    ep: object
    U: np.ndarray
    rU: np.ndarray | None
    u0: np.ndarray
    r0: np.ndarray
    oob: np.ndarray
    source_folds: list
    labels: np.ndarray
    rvalid: np.ndarray
    rtokens: np.ndarray
    qtokens: np.ndarray
    thresholds: tuple | None
    host_binding: dict


def prepare(ep,host_q=None,host_r=None,host_binding=None,direct_host=False):
    if ep.q_rgb is None or ep.r_rgb is None or ep.reference_mask is None:
        raise ValueError('Supplied E methods require actual original RGB and complete MR')
    q0,_=H.direct_u0(ep.q,ep.r,ep.wf,ep.wvalid)
    r0,_=H.direct_u0(ep.r,ep.r,ep.wf,ep.wvalid)
    oob,folds=H.source_block_predictions(ep.r,ep.r_hw,ep.wf,ep.wvalid)
    rt=H.pixel_tokens(ep.r_rgb.shape[:2],ep.r_hw,ep.reference_geometry)
    qt=H.pixel_tokens(ep.q_rgb.shape[:2],ep.q_hw,ep.query_geometry)
    valid=ep.wvalid[rt]>0
    cov=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0)
    thresholds=H.anchor_thresholds(oob,cov,ep.wvalid)
    uq=H.native_to_original(q0.reshape(ep.q_hw),ep.original_shape,ep.query_geometry)
    # Source self-prediction is not used for threshold/coverage calibration.
    rp=H.native_to_original(r0.reshape(ep.r_hw),ep.r_rgb.shape[:2],ep.reference_geometry)
    if direct_host:
        U,rU=uq,rp;binding=dict(kind='C_E_U0_pure_role_max_or_16nn_vote')
    else:
        if host_q is None or not host_binding:
            raise ValueError('C_E requires bound MEAN continuous q host field; no prototype substitution')
        U=np.asarray(host_q,float);rU=None if host_r is None else np.asarray(host_r,float);binding=dict(host_binding)
        if U.shape!=ep.original_shape or (rU is not None and rU.shape!=ep.r_rgb.shape[:2]):
            raise ValueError('MEAN host must match actual original q/r shapes')
    return PixelProblem(ep,U,rU,q0,r0,oob,folds,np.asarray(ep.reference_mask,bool).ravel(),
                        valid,rt,qt,thresholds,binding)


def _anchors(problem):
    ep=problem.ep
    q=H.native_to_original(problem.u0.reshape(ep.q_hw),ep.original_shape,ep.query_geometry)
    if problem.thresholds is None:return np.zeros(ep.original_shape,np.int8)
    ft,bt=problem.thresholds
    return np.where((q>=ft)&(problem.U>.5),1,np.where((q<=bt)&(problem.U<=.5),-1,0)).astype(np.int8)


def e236_affine_lift(problem,control=None):
    """Local simplex RGB reconstruction followed by the specified global lift."""
    if control=='guided':
        I=np.asarray(problem.ep.q_rgb,float)/255.;u=problem.U
        mu=np.stack([ndimage.uniform_filter(I[...,k],5,mode='nearest') for k in range(3)],-1)
        mup=ndimage.uniform_filter(u,5,mode='nearest')
        cov=np.empty((*u.shape,3,3))
        ip=np.empty((*u.shape,3))
        for k in range(3):
            ip[...,k]=ndimage.uniform_filter(I[...,k]*u,5,mode='nearest')-mu[...,k]*mup
            for j in range(3):
                cov[...,k,j]=ndimage.uniform_filter(I[...,k]*I[...,j],5,mode='nearest')-mu[...,k]*mu[...,j]
        a=np.linalg.solve(cov+1e-3*np.eye(3),ip[...,None])[...,0]
        b=mup-np.sum(a*mu,-1)
        out=sum(ndimage.uniform_filter(a[...,k],5,mode='nearest')*I[...,k] for k in range(3))
        out+=ndimage.uniform_filter(b,5,mode='nearest')
        return np.clip(out,0,1),dict(status='ok',control='same5x5_RGB_guided_filter',epsilon=1e-3)
    if control=='gaussian':
        from scipy import sparse
        rgb=problem.ep.q_rgb;shape=rgb.shape[:2];p=np.prod(shape);ids=np.arange(p).reshape(shape)
        offsets=[(y,x) for y in range(-2,3) for x in range(-2,3)];neighbours=[]
        for y,x in offsets:
            yy=np.clip(np.arange(shape[0])+y,0,shape[0]-1);xx=np.clip(np.arange(shape[1])+x,0,shape[1]-1)
            neighbours.append(ids[yy[:,None],xx[None,:]].ravel())
        ni=np.stack(neighbours,1);c=np.asarray(rgb,float).reshape(-1,3)/255.
        d=np.sum((c[:,None]-c[ni])**2,2);band=max(float(np.median(d)),1e-4)
        weights=np.exp(-d/band);weights/=weights.sum(1,keepdims=True)
        W=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(p),25),ni.ravel())),shape=(p,p))
        info=dict(control='same_5x5_gaussian_RGB',bandwidth=band)
    else:
        W,info=H.affine_colour_weights(problem.ep.q_rgb,_anchors(problem))
        if info['unconverged_rows']:
            return problem.U.copy(),dict(status='fallback_W_not_certified',weights=info)
    out,solve=H.affine_lift(problem.U,W)
    if not solve['converged']:out=problem.U.copy()
    return out,dict(status='ok' if solve['converged'] else 'fallback_lift_not_certified',
                    weights=info,solver=solve)


def _pixel_head_training_domain(problem,train_domain,available):
    ep=problem.ep;valid=np.asarray(available,bool).copy()
    cov=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0)
    mixed=valid&((cov>.1)&(cov<.9))[problem.rtokens]
    if train_domain=='mixed':
        valid=mixed
    elif np.any(mixed&problem.labels) and np.any(mixed&~problem.labels):
        # The supplied four-arm check requires equal pixel sample budgets.
        # Whole-reference positions remain eligible, but are sampled to the
        # mixed arm's actual budget. No query state or label enters this choice.
        budget=int(mixed.sum());fg=np.flatnonzero(valid&problem.labels);bg=np.flatnonzero(valid&~problem.labels)
        nf=min(len(fg),budget//2);nb=min(len(bg),budget-nf)
        nf=min(len(fg),budget-nb)
        ids=np.concatenate((_spread_rows(fg,nf),_spread_rows(bg,nb)))
        valid=np.zeros_like(valid);valid[ids]=True
    return valid


def e242_pixel_collision(problem,train_domain='all',apply_domain='gate'):
    ep=problem.ep;valid=_pixel_head_training_domain(problem,train_domain,problem.rvalid)
    phi=H.pixel_phi(ep.r_rgb);qphi=H.pixel_phi(ep.q_rgb)
    # Refit the actual pixel head (including every train-only scale) inside
    # each physical-token holdout, not just the U0 bank used for its gate.
    h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw);source_folds=[]
    for by in range(2):
        for bx in range(2):
            y0,y1=by*h//2,(by+1)*h//2;x0,x1=bx*w//2,(bx+1)*w//2
            held=((yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)).ravel()
            exclusion=((yy>=max(0,y0-1))&(yy<min(h,y1+1))&
                       (xx>=max(0,x0-1))&(xx<min(w,x1+1))).ravel()
            available=problem.rvalid&~exclusion[problem.rtokens]
            train=_pixel_head_training_domain(problem,train_domain,available)
            test=problem.rvalid&held[problem.rtokens]
            fitted=H.fit_pixel_logistic(phi,ep.r,problem.rtokens,problem.labels,train)
            record=dict(block=[by,bx],held_pixels=int(test.sum()),train_pixels=int(train.sum()))
            if fitted is None or not test.any():
                record['status']='invalid_fold_missing_roles_or_test';source_folds.append(record);continue
            fp=fitted.predict(phi,ep.r,problem.rtokens).reshape(ep.r_rgb.shape[:2])
            rp,_=H.direct_u0(ep.r,ep.r,np.where(exclusion,0,ep.wf),np.where(exclusion,0,ep.wvalid))
            ru=H.native_to_original(rp.reshape(ep.r_hw),ep.r_rgb.shape[:2],ep.reference_geometry)
            gate=H.semantic_gate(rp,ep.r_hw,ep.r_rgb.shape[:2],ep.reference_geometry)
            if apply_domain=='full':gate=np.ones_like(gate)
            held_output=np.where(gate,fp,ru).ravel()[test]>.5
            truth=problem.labels[test];intersection=int(np.sum(held_output&truth));union=int(np.sum(held_output|truth))
            record.update(status='ok',fit=fitted.info,held_intersection=intersection,held_union=union,
                          held_iou=intersection/max(union,1),held_binary_packbits_hex=np.packbits(held_output).tobytes().hex(),
                          held_output_encoding='np.packbits row-major test selector; final original binary')
            source_folds.append(record)
    head=H.fit_pixel_logistic(phi,ep.r,problem.rtokens,problem.labels,valid)
    if head is None:return problem.U.copy(),dict(status='fallback_missing_two_source_classes')
    pred=head.predict(qphi,ep.q,problem.qtokens).reshape(ep.original_shape)
    gate=H.semantic_gate(problem.u0,ep.q_hw,ep.original_shape,ep.query_geometry)
    if apply_domain=='full':gate=np.ones_like(gate)
    result=np.where(gate,pred,problem.U)
    return result,dict(status='ok',train_domain=train_domain,apply_domain=apply_domain,
                       gate_pixels=int(gate.sum()),fit=head.info,source_pixel_head_folds=source_folds,
                       total_head_fits=1+sum(f['status']=='ok' for f in source_folds),
                       gate_outside_exact_host=bool(np.array_equal(result[~gate],problem.U[~gate])))


def e245_footprint_inverse(problem,control=None):
    ep=problem.ep;cov=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0)
    known=np.isfinite(problem.oob)&(ep.wvalid>0)
    if known.sum()<4 or np.unique(cov[known]).size<2:
        return problem.U.copy(),dict(status='fallback_source_mapping_unidentifiable')
    # For each source test block the bank AND the monotone response fit exclude
    # its labels and one-patch buffer.  Fitting one map to all cross-fit targets
    # then calling its in-sample residual 'held-out RMSE' would be incorrect.
    h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw);squared=0.;mass=0.;valid_folds=0
    for by in range(2):
        for bx in range(2):
            y0,y1=by*h//2,(by+1)*h//2;x0,x1=bx*w//2,(bx+1)*w//2
            test=((yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)).ravel()&(ep.wvalid>0)
            exclusion=((yy>=max(0,y0-1))&(yy<min(h,y1+1))&
                       (xx>=max(0,x0-1))&(xx<min(w,x1+1))).ravel()
            train=(ep.wvalid>0)&~exclusion
            if train.sum()<4 or not test.any() or np.unique(cov[train]).size<2:continue
            prob,_=H.direct_u0(ep.r,ep.r,np.where(train,ep.wf,0),np.where(train,ep.wvalid,0))
            tx,ty=H.isotonic_fit(prob[train],cov[train],ep.wvalid[train])
            error=np.interp(prob[test],tx,ty)-cov[test]
            squared+=float(ep.wvalid[test]@(error*error));mass+=float(ep.wvalid[test].sum());valid_folds+=1
    if valid_folds<2:return problem.U.copy(),dict(status='fallback_insufficient_coverage_folds',valid_source_folds=valid_folds)
    x,y=H.isotonic_fit(problem.r0[known],cov[known],ep.wvalid[known])
    rmse=float(np.sqrt(squared/max(mass,H.EPS)))
    sigma=max(.05,rmse)
    a=np.interp(problem.u0,x,y)
    A=H.footprint(ep.original_shape,ep.q_hw,ep.query_geometry)
    if control=='interpolate':
        out=H.native_to_original(a.reshape(ep.q_hw),ep.original_shape,ep.query_geometry)
        return out,dict(status='ok',control='same_source_mapping_direct_interpolation',sigma_coverage=sigma)
    out,solve=H.coverage_tv(problem.U,A,a,sigma,ep.q_rgb,ep.q_valid)
    if not solve['converged']:out=problem.U.copy()
    return out,dict(status='ok' if solve['converged'] else 'fallback_inverse_not_certified',
                    solver=solve,source_rmse_coverage=rmse,source_mapping_points=len(x),
                    valid_source_folds=valid_folds,source_block_predictions=problem.source_folds)


def _reflect_model(c,mask,valid):
    if not np.any(mask&valid):return None
    v=c[mask&valid];direction=np.median(v,axis=0)
    direction/=max(np.linalg.norm(direction),H.EPS)
    return direction


def _illumination_candidates(r,valid):
    bright=r[valid&(r.mean(1)>=np.quantile(r[valid].mean(1),.9))]
    ebright=np.median(bright,0);ebright/=max(np.linalg.norm(ebright),H.EPS)
    candidates=[ebright,np.full(3,1/np.sqrt(3)),np.array([.299,.587,.114])]
    return [v/max(np.linalg.norm(v),H.EPS) for v in candidates]


def _source_illumination_selection(problem,r):
    ep=problem.ep;h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw);scores=np.zeros(3);folds=0
    for by in range(2):
        for bx in range(2):
            y0,y1=by*h//2,(by+1)*h//2;x0,x1=bx*w//2,(bx+1)*w//2
            test=((yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)).ravel()[problem.rtokens]&problem.rvalid
            excluded=((yy>=max(0,y0-1))&(yy<min(h,y1+1))&
                      (xx>=max(0,x0-1))&(xx<min(w,x1+1))).ravel()[problem.rtokens]
            train=problem.rvalid&~excluded
            f=_reflect_model(r,problem.labels,train);b=_reflect_model(r,~problem.labels,train)
            if f is None or b is None or not np.any(test&problem.labels) or not np.any(test&~problem.labels):continue
            for j,e in enumerate(_illumination_candidates(r,train)):
                rf=H.two_ray_residual(r[test],f,e);rb=H.two_ray_residual(r[test],b,e);lab=problem.labels[test]
                scores[j]+=.5*(rf[lab].mean()+rb[~lab].mean())
            folds+=1
    return int(np.argmin(scores)),scores.tolist(),folds


def e250_dichromatic(problem,control=None):
    """Exact nonnegative d,s given source-selected illumination, five label rounds."""
    ep=problem.ep;r=np.asarray(ep.r_rgb,float).reshape(-1,3)/255.
    q=np.asarray(ep.q_rgb,float).reshape(-1,3)/255.
    valid=problem.rvalid;lab=problem.labels
    cf=_reflect_model(r,lab,valid);cb=_reflect_model(r,~lab,valid)
    if cf is None or cb is None:return problem.U.copy(),dict(status='fallback_missing_source_role')
    # Strong simple controls do not require the main model's illuminant
    # calibration to be identifiable. Missing that information closes the ray
    # module, not an otherwise well-defined ordinary colour classifier.
    if control in {'gmm','chromaticity'}:
        from ics.methods.pro_paired_environment import exact_potts_cut
        saturated=np.any(np.asarray(ep.q_rgb).reshape(-1,3)>=254,axis=1)
        U=problem.U.ravel();uc=np.clip(U,H.EPS,1-H.EPS);base=np.log(uc/(1-uc))
        edges,cap=H.rgb_edges(ep.q_rgb)
        if control=='gmm':
            colour=H.diagonal_colour_evidence(ep.q_rgb,ep.r_rgb,lab,valid,k=1)
            if colour is None:return problem.U.copy(),dict(status='fallback_source_colour_role_missing')
            evidence,fit=colour
        else:
            qr=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),H.EPS)
            rr=r/np.maximum(np.linalg.norm(r,axis=1,keepdims=True),H.EPS)
            source=np.sum((rr-cb)**2,1)-np.sum((rr-cf)**2,1)
            evidence=np.sum((qr-cb)**2,1)-np.sum((qr-cf)**2,1)
            evidence,scale=H.normalize_evidence(evidence,source,valid);fit=dict(scale=scale)
        evidence[saturated]=0;labels,cert=exact_potts_cut(base+evidence,edges,cap)
        out=labels.reshape(ep.original_shape).astype(float);out.reshape(-1)[saturated]=U[saturated]
        return out,dict(status='ok',control=control,fit=fit,cut=cert)
    anchors=_anchors(problem).ravel()
    if not np.any(anchors==1) or not np.any(anchors==-1):
        return problem.U.copy(),dict(status='fallback_missing_query_double_anchors',
                initial_fg_anchors=int(np.sum(anchors==1)),initial_bg_anchors=int(np.sum(anchors==-1)))
    # Illumination candidate selection is source-only and excludes unknown pixels.
    index,scores,folds=_source_illumination_selection(problem,r)
    if folds<2:return problem.U.copy(),dict(status='fallback_insufficient_source_illumination_folds',valid_folds=folds)
    e=_illumination_candidates(r,valid)[index]
    saturated=np.any(np.asarray(ep.q_rgb).reshape(-1,3)>=254,axis=1)
    source=H.two_ray_residual(r,cb,e)-H.two_ray_residual(r,cf,e)
    if float(np.max(np.abs(source[valid]),initial=0))<=H.EPS:
        return problem.U.copy(),dict(status='fallback_reflection_roles_unidentifiable')
    edges,cap=H.rgb_edges(ep.q_rgb)
    from ics.methods.pro_paired_environment import exact_potts_cut
    U=problem.U.ravel();base=np.log(np.clip(U,H.EPS,1-H.EPS)/(1-np.clip(U,H.EPS,1-H.EPS)))
    labels=U>.5;anchors=_anchors(problem).ravel()
    activity=0
    for iteration in range(5):
        evidence=H.two_ray_residual(q,cb,e)-H.two_ray_residual(q,cf,e)
        evidence,scale=H.normalize_evidence(evidence,source,valid);evidence[saturated]=0
        labels,cert=exact_potts_cut(base+evidence,edges,cap)
        # Only initial direct anchors authorize appearance adaptation, never
        # newly flipped foreground.  RGB coefficients remain exact NNLS.
        for cls,sign in ((True,1),(False,-1)):
            ids=(anchors==sign)&(labels==cls)&~saturated
            if ids.sum()<3:continue
            colour=cf if cls else cb
            d,s,_=H.two_ray_fit(q[ids],colour,e)
            direction=np.sum(d[:,None]*(q[ids]-s[:,None]*e),axis=0)/max(float(d@d),H.EPS)
            direction=np.maximum(direction,0);direction/=max(np.linalg.norm(direction),H.EPS)
            if np.linalg.norm(direction)<=H.EPS:continue
            if cls:cf=direction
            else:cb=direction
            activity+=1
    out=labels.reshape(ep.original_shape).astype(float)
    out.reshape(-1)[saturated]=U[saturated]
    return out,dict(status='ok',illumination_candidate=index,valid_source_folds=folds,
                    source_candidate_loss=scores,iterations=5,appearance_updates=activity,
                    saturated_pixels=int(saturated.sum()),last_cut=cert,
                    nnls='exact_two_ray_interior_and_all_boundary_faces')


def _patch_features(rgb):
    """3x3 RGB plus the fixed seven-channel texture, at every original pixel."""
    rgb=np.asarray(rgb,float)/255.;shape=rgb.shape[:2]
    rows=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            yy=np.clip(np.arange(shape[0])+dy,0,shape[0]-1)
            xx=np.clip(np.arange(shape[1])+dx,0,shape[1]-1)
            rows.append(rgb[yy[:,None],xx[None,:]].reshape(-1,3))
    return np.concatenate(rows+[H.texture_phi(np.rint(rgb*255).astype(np.uint8))],axis=1)


def _spread_rows(ids,budget):
    ids=np.asarray(ids,int)
    if len(ids)<=budget:return ids
    return ids[np.linspace(0,len(ids)-1,budget).astype(int)]


def sparse_patch_residual(x,atoms,steps=3,chunk=512):
    """Fixed three-atom signed OMP, rank-aware LS, complete residual output."""
    x,atoms=np.asarray(x,float),np.asarray(atoms,float)
    atoms=atoms/np.maximum(np.linalg.norm(atoms,axis=1,keepdims=True),H.EPS)
    out=np.sum(x*x,1);selected=[]
    if not len(atoms):return out
    for start in range(0,len(x),chunk):
        v=x[start:start+chunk];res=v.copy();indices=np.zeros((len(v),min(steps,len(atoms))),int)
        for j in range(indices.shape[1]):
            score=np.abs(res@atoms.T)
            if j:
                score[np.arange(len(v))[:,None],indices[:,:j]]=-np.inf
            indices[:,j]=np.argmax(score,axis=1)
            A=atoms[indices[:,:j+1]]
            gram=np.einsum('bkd,bld->bkl',A,A)
            rhs=np.einsum('bkd,bd->bk',A,v)
            coef=np.einsum('bkl,bl->bk',np.linalg.pinv(gram,rcond=1e-10),rhs)
            res=v-np.einsum('bk,bkd->bd',coef,A)
        out[start:start+len(v)]=np.sum(res*res,1)
    return out


def e238_sparse_colour_dictionary(problem,control=None):
    ep=problem.ep;r=_patch_features(ep.r_rgb);q=_patch_features(ep.q_rgb)
    # A held-out pixel and the entire interpolation/texture support are removed.
    support=ndimage.binary_erosion(problem.rvalid.reshape(ep.r_rgb.shape[:2]),
                                   structure=np.ones((3,3)),border_value=0).ravel()
    f=_spread_rows(np.flatnonzero(support&problem.labels),128)
    b=_spread_rows(np.flatnonzero(support&~problem.labels),128)
    n=min(len(f),len(b))
    if not n:return problem.U.copy(),dict(status='fallback_missing_equal_source_dictionary')
    f,b=f[:n],b[:n];source_atoms=np.concatenate((r[f],r[b]))
    mean=source_atoms.mean(0);scale=np.maximum(source_atoms.std(0),1e-3)
    r=(r-mean)/scale;q=(q-mean)/scale
    fa,ba=r[f].copy(),r[b].copy()
    # One fixed addition from initial anchors; there is no iterative relabeling.
    anchors=_anchors(problem).ravel()
    qa=_spread_rows(np.flatnonzero(anchors==1),8);qb=_spread_rows(np.flatnonzero(anchors==-1),8)
    nquery=min(len(qa),len(qb))
    if nquery:fa=np.concatenate((fa,q[qa[:nquery]]));ba=np.concatenate((ba,q[qb[:nquery]]))
    if control=='nearest':
        def residual(v,a):
            norm=np.maximum(np.linalg.norm(a,axis=1),H.EPS);a=a/norm[:,None]
            return np.sum(v*v,1)-np.max((v@a.T)**2,axis=1)
    elif control=='subspace':
        def residual(v,a):
            _,s,basis=np.linalg.svd(a,full_matrices=False)
            basis=basis[s>max(s[0] if len(s) else 0.,H.EPS)*1e-10]
            return np.maximum(np.sum(v*v,1)-np.sum((v@basis.T)**2,1),0.)
    else:residual=sparse_patch_residual
    rs=residual(r,ba)-residual(r,fa);qs=residual(q,ba)-residual(q,fa)
    qs,normalizer=H.normalize_evidence(qs,rs,support)
    from ics.methods.pro_paired_environment import exact_potts_cut
    U=problem.U.ravel();base=np.log(np.clip(U,H.EPS,1-H.EPS)/(1-np.clip(U,H.EPS,1-H.EPS)))
    edges,cap=H.rgb_edges(ep.q_rgb);labels,certificate=exact_potts_cut(base+qs,edges,cap)
    # Dictionaries are fixed after the one authorized addition; label steps are
    # thus at a fixed point after the first exact binary minimization.
    return labels.reshape(ep.original_shape).astype(float),dict(status='ok',source_atoms_per_class=n,
                    query_atoms_per_class=nquery,sparse_budget=3,evidence_scale=normalizer,
                    label_iterations=1,max_allowed_outer_iterations=5,cut=certificate,
                    control=control,source_dictionary_support_eroded_one_original_pixel=True)


def _legal_problem(ep,method):
    from .common import require_artifact,host_baseline
    direct=method in {'E237','E238','E242','E243'}
    if direct:return prepare(ep,direct_host=True)
    host=host_baseline(ep)
    original=require_artifact(ep,'mean_original_field')
    return prepare(ep,host_q=original,host_binding=dict(producer=host.info['host_producer'],
                        renderer=host.info['host_renderer'],field_space='original'))


def _call(ep,method,core,**kwargs):
    from .common import Result,host_baseline
    if float(np.sum(ep.wf))<=0:
        return Result(field=np.zeros(ep.original_shape),threshold=.5,
                      mask_original=np.zeros(ep.original_shape,bool),
                      info=dict(method=method,field_space='original',status='empty_reference_foreground',quality='unknown'))
    problem=_legal_problem(ep,method)
    field,info=core(problem,**kwargs)
    if not np.isfinite(field).all():
        field=problem.U.copy();info=dict(status='fallback_nonfinite_numerical_result')
    info.update(method=method,field_space='original',host_binding=problem.host_binding,
                source_fit_only=True,quality='unknown',original_shape=list(ep.original_shape))
    direct=method in {'E237','E238','E242','E243'}
    mask0=problem.U>.5 if direct else host_baseline(ep).mask_original
    if np.array_equal(field,problem.U):
        mask=mask0.copy();info['zero_or_fallback_exact_host']=True
    elif method in {'E238','E250'} or info.get('binary_optimizer',False):
        # The E binary optimizer's disagreements are edits; it does not replace
        # the unmodified host renderer at all of its agreeing pixels.
        cut=field>.5;unary=problem.U>.5;mask=np.where(cut!=unary,cut,mask0)
        info['binary_readout']='original_cut_unary_disagreement_on_actual_host'
    else:mask=field>.5
    return Result(field=field,threshold=.5,mask_original=mask,info=info)


METHODS = {
    'E236':lambda ep:_call(ep,'E236',e236_affine_lift),
    'E238':lambda ep:_call(ep,'E238',e238_sparse_colour_dictionary),
    'E242':lambda ep:_call(ep,'E242',e242_pixel_collision),
    'E245':lambda ep:_call(ep,'E245',e245_footprint_inverse),
    'E250':lambda ep:_call(ep,'E250',e250_dichromatic),
}
CONTROLS = {
    'E236_gaussian5x5':lambda ep:_call(ep,'E236',e236_affine_lift,control='gaussian'),
    'E236_RGB_guided5x5':lambda ep:_call(ep,'E236',e236_affine_lift,control='guided'),
    'E238_nearest_same_atoms':lambda ep:_call(ep,'E238',e238_sparse_colour_dictionary,control='nearest'),
    'E238_subspace_same_atoms':lambda ep:_call(ep,'E238',e238_sparse_colour_dictionary,control='subspace'),
    'E242_all_full':lambda ep:_call(ep,'E242',e242_pixel_collision,apply_domain='full'),
    'E242_mixed_gate':lambda ep:_call(ep,'E242',e242_pixel_collision,train_domain='mixed'),
    'E242_mixed_full':lambda ep:_call(ep,'E242',e242_pixel_collision,train_domain='mixed',apply_domain='full'),
    'E245_same_mapping_interpolation':lambda ep:_call(ep,'E245',e245_footprint_inverse,control='interpolate'),
    'E250_chromaticity_same_colours':lambda ep:_call(ep,'E250',e250_dichromatic,control='chromaticity'),
    'E250_one_GMM_same_observations':lambda ep:_call(ep,'E250',e250_dichromatic,control='gmm'),
}
RESOURCES = {k:dict(final_native=True,original_rgb=True,complete_MR=True,
                   mean_host=k not in {'E238','E242'},extra_encoder_forwards=0)
             for k in METHODS}
IMPLEMENTATION_STATUS = {
    'E236':'implemented_pending_kernel_and_review_checks',
    'E238':'implemented_pending_kernel_and_review_checks',
    'E242':'implemented_pending_kernel_and_review_checks',
    'E245':'implemented_pending_kernel_and_review_checks',
    'E250':'implemented_pending_kernel_and_review_checks',
}
