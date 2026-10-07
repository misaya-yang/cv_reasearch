"""Faithful supplied E201--E225 recipes, complete original RGB class outputs.

METHODS contains 25 cards. Controls are separate registry rows. Scientific
quality remains unknown until the root's sealed fixed600 evaluation.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage, sparse, optimize
from scipy.sparse import linalg
from scipy.special import expit, logsumexp, gammaln
from . import e_helpers_201_225 as K
from . import e_helpers_226_250 as H
from .common import Result, host_baseline, ArtifactUnavailable


def _colour(p,k=2,features='rgb'):
    fn=H.normalized_rgb if features=='rgb' else H.texture_phi
    return K.gmm_pair(fn(p.ep.q_rgb),fn(p.ep.r_rgb),p.labels,p.rvalid,k)


def _finish(ep,method,core,control=None):
    if np.sum(ep.wf)<=0:
        return Result(np.zeros(ep.original_shape),.5,np.zeros(ep.original_shape,bool),
                      dict(method=method,status='invalid_empty_reference_F',field_space='original',quality='unknown'))
    p=K.prepare(ep,method)
    if control=='host':out=p.U.copy();info=dict(status='module_disabled_exact_host')
    elif control=='sameunary':out=p.U.copy();info=dict(status='sameunary_continuous_original_zero_arm',force_unary_renderer=True)
    elif control=='rgbcut':out,cert=K.cut(p.U,ep.q_rgb);out=out.astype(float);info=dict(status='ok',control='sameunary_RGB_Potts',cut=cert,binary_optimizer=True)
    elif control=='proposal':out,info=e201(p,control='global_one')
    else:out,info=core(p,control=control)
    out=np.asarray(out,float)
    if out.shape!=tuple(ep.original_shape) or not np.isfinite(out).all():raise RuntimeError(method+' produced an illegal complete field')
    if np.array_equal(out,p.U) and not info.get('force_unary_renderer'):
        mask=p.mask0.copy();info['zero_or_fallback_exact_host']=True
    else:mask=out>.5
    info.update(method=method,field_space='original',source_fit_only=True,quality='unknown',
                actual_DINO_episodes_verified=0,host_binding=p.binding,original_shape=list(ep.original_shape),
                control=control,source_anchor_folds=p.source_folds)
    return Result(out,.5,mask,info)


def e201(p,control=None):
    colour=_colour(p,k=1 if control=='global_one' else 2)
    a=p.anchors.ravel();u=p.U.ravel();shape=p.U.shape;c=H.normalized_rgb(p.ep.q_rgb)
    if colour is None or not np.any(a==1) or not np.any(a==-1):return p.U.copy(),dict(status='fallback_missing_double_colour_anchors')
    _,scale,sourceF,sourceB=colour;cc,n=ndimage.label(p.anchors==1)
    active=min(n,16);selected=(cc>0)&(cc<=active)
    if not active:return p.U.copy(),dict(status='fallback_no_F_island')
    _,nearest=ndimage.distance_transform_edt(~selected,return_indices=True);zone=cc[tuple(nearest)].ravel()
    dist=ndimage.distance_transform_edt(~selected).ravel();radius=4*max(shape[0]/p.ep.q_hw[0],shape[1]/p.ep.q_hw[1])
    covered=(dist<=radius)&~((cc.ravel()>16));zone[~covered]=0
    if control in {'global_one','global_two'}:zone[:]=1;active=1;covered[:]=True
    labels=u>.5;history=[];best=np.inf;bestout=p.U.copy();cuts=[]
    for iteration in range(5):
        evidence=np.zeros(len(u));models=[]
        for j in range(1,active+1):
            ids=zone==j;fi=ids&((a==1)|labels);bi=ids&(a==-1)
            fg=H.fit_gmm(c[fi],1 if control=='global_one' else 2) if fi.sum()>=2 else sourceF
            bg=H.fit_gmm(c[bi],1 if control=='global_one' else 2) if bi.sum()>=2 else sourceB
            evidence[ids]=(fg.log_density(c[ids])-bg.log_density(c[ids]))/scale;models.append((fg,bg))
        proposed,cert=K.cut(p.U,p.ep.q_rgb,evidence)
        candidate=proposed.ravel();candidate[~covered]=u[~covered]>.5
        logits=K.logit(u)+evidence;e,cap=H.rgb_edges(p.ep.q_rgb)
        objective=float(np.sum(np.where(candidate,np.logaddexp(0,-logits),np.logaddexp(0,logits)))+cap@(candidate[e[:,0]]!=candidate[e[:,1]]))
        history.append(objective);cuts.append(cert)
        if objective>=best-1e-10:break
        best=objective;labels=candidate;bestout=labels.reshape(shape).astype(float);bestout.ravel()[~covered]=u[~covered]
    return bestout,dict(status='ok',binary_optimizer=True,iterations=len(history),objective_history=history,
                source_absmedian_scale=scale,positive_islands=n,modelled_islands=active,uncovered_pixels=int((~covered).sum()),
                negative_training='fixed_initial_qualified_B_anchors_only',cut=cuts[-1])


def e202(p,control=None):
    edges,cap=H.rgb_edges(p.ep.q_rgb);L=K.laplacian(edges,cap,p.U.size)
    out,info=K.quadratic_box(p.U,L,None if control=='noanchors' else p.anchors)
    return (out if info['converged'] else p.U.copy()),dict(status='ok' if info['converged'] else 'fallback_harmonic_uncertified',solver=info)


def e203(p,control=None):
    tree=K.rgb_tree(p.ep.q_rgb);out,info=K.tree_dp(p.U,tree,code_fee=control!='nocode')
    return out.astype(float),dict(status='ok',binary_optimizer=True,**info)


def e204(p,control=None):
    tree=K.rgb_tree(p.ep.q_rgb);groups=[]
    for size in (16,64,256):groups.extend(K.tree_partition(tree,size))
    if control=='shuffled':
        permutation=np.random.default_rng(0).permutation(p.U.size);groups=[permutation[g] for g in groups]
    gamma=1.;lam=1.;base=K.logit(p.U).ravel();z=p.U.ravel()>.5;a=p.anchors.ravel();history=[]
    edges,cap=H.rgb_edges(p.ep.q_rgb)
    def energy(mask):
        unary=float(np.sum(np.where(mask,np.logaddexp(0,-base),np.logaddexp(0,base))))
        unary+=float(np.sum((a!=0)&(mask!=(a>0))))
        return unary+float(cap@(mask[edges[:,0]]!=mask[edges[:,1]]))+sum(min(gamma,lam*min(mask[g].sum(),len(g)-mask[g].sum())) for g in groups)
    best=energy(z);history.append(best)
    for _ in range(5):
        evidence=np.where(a!=0,a,0.).astype(float)
        for g in groups:
            choices=[lam*float(z[g].sum()),lam*float((~z[g]).sum()),gamma];state=int(np.argmin(choices))
            if state<2:evidence[g]+=lam*(1 if state==1 else -1)
        mask,cert=K.cut(p.U,p.ep.q_rgb,evidence);candidate=mask.ravel();value=energy(candidate)
        if value>=best-1e-10:break
        z=candidate;best=value;history.append(value)
    return z.reshape(p.U.shape).astype(float),dict(status='ok',binary_optimizer=True,solver='5_round_latent_region_move_making',
                   gamma=gamma,lambda_minority=lam,cliques=len(groups),total_members=sum(map(len,groups)),
                   memberships_per_pixel=3,objective_history=history,global_optimality=False)


def e205(p,control=None):
    if control!='notrimap' and (not np.any(p.anchors==1) or not np.any(p.anchors==-1)):
        return p.U.copy(),dict(status='fallback_missing_matting_double_anchors')
    L=K.matting_laplacian(p.ep.q_rgb);out,solve=K.quadratic_box(p.U,L,None if control=='notrimap' else p.anchors)
    return (out if solve['converged'] else p.U.copy()),dict(status='ok' if solve['converged'] else 'fallback_matting_uncertified',
          solver=solve,matting_window=3,matting_regularization=1e-5,matting_nonzeros=L.nnz)


def e206(p,control=None):
    mapping=K.coverage_mapping(p);pair=_colour(p)
    if mapping is None or pair is None:return p.U.copy(),dict(status='fallback_unidentified_source_coverage_or_colour')
    coverage,sigma,calibration=mapping;_,_,fg,bg=pair;c=H.normalized_rgb(p.ep.q_rgb);a=p.anchors.ravel();u=p.U.ravel()
    A=H.footprint(p.U.shape,p.ep.q_hw,p.ep.query_geometry);out=u.copy();active=0;fallback=0;allstates=[]
    for token in np.flatnonzero(p.ep.q_valid>0):
        row=A.getrow(token);ids=row.indices;weights=row.data
        if not len(ids):continue
        near=ndimage.binary_dilation((p.qtokens.reshape(p.U.shape)==token),iterations=2).ravel()
        cf=np.median(c[near&(a==1)],0) if np.any(near&(a==1)) else fg.means[fg.weights.argmax()]
        cb=np.median(c[near&(a==-1)],0) if np.any(near&(a==-1)) else bg.means[bg.weights.argmax()]
        variance=max(float(np.median(np.sum((c[ids]-np.where((u[ids]>.5)[:,None],cf,cb))**2,1))),1e-4)
        if np.linalg.norm(cf-cb)<=K.EPS:fallback+=1;continue
        x=c[ids];initial=np.clip(((x-cb)@(cf-cb))/max(float(np.sum((cf-cb)**2)),K.EPS),0,1);alpha=initial.copy()
        for _ in range(5):
            delta=cf-cb;diag=1+np.sum(delta*delta)/variance;rank=0 if control=='nocoverage' else 1/(sigma*sigma)
            rhs=u[ids]+((x-cb)@delta)/variance+rank*coverage[token]*weights
            Q=sparse.diags(np.full(len(ids),diag))+rank*sparse.csr_matrix(np.outer(weights,weights))
            # Here Q>=I; reuse the same certified finite box operator.
            trial,cert=K.quadratic_box(rhs.reshape(1,-1),Q-sparse.eye(len(ids)))
            alpha=trial.ravel()
            if not cert['converged']:alpha=initial;break
            design=np.stack((alpha,1-alpha),1);coef=np.linalg.solve(design.T@design+1e-4*np.eye(2),design.T@x+1e-4*np.stack((cf,cb)))
            cf,cb=np.clip(coef,0,1)
        def energy(v):
            reconstruction=cb+v[:,None]*(cf-cb)
            return float(np.sum((x-reconstruction)**2)/variance+np.sum((v-u[ids])**2)+(0 if control=='nocoverage' else (weights@v-coverage[token])**2/(sigma*sigma)))
        candidates=[np.zeros(len(ids)),np.ones(len(ids)),alpha];cost=[energy(v) for v in candidates];state=int(np.argmin(cost))
        residual=float(np.mean(np.sum((x-(cb+alpha[:,None]*(cf-cb)))**2,1))/variance)
        if residual>4:fallback+=1;continue
        if control=='rankquota':
            order=np.argsort(-initial,kind='stable');v=np.zeros(len(ids));v[order[:int(round(coverage[token]*len(ids)))]]=1
        else:v=candidates[state]
        # Pixel cells can straddle work patches; accumulate exact overlap later.
        out[ids]=v;active+=1;allstates.append(state)
    return out.reshape(p.U.shape),dict(status='ok',active_patches=active,fallback_patches=fallback,
        chosen_BG_FG_mixed=[allstates.count(j) for j in range(3)],coverage_sigma=sigma,coverage_calibration=calibration,
        scalar_observation='per_query_patch_coverage_only',whole_image_area_constraint=False)


def _sample_edges(e,states,valid,cap=128):
    ids=[]
    for state in np.unique(states[valid]):
        a=np.flatnonzero(valid&(states==state));ids.extend(a[np.linspace(0,len(a)-1,min(cap,len(a))).astype(int)])
    return np.asarray(ids,int)


def _softmax_head(x,state,valid,classes=3):
    x=np.asarray(x,float);valid=np.asarray(valid,bool);states=np.asarray(state,int);weights=np.zeros(len(x))
    for j in range(classes):
        ids=valid&(states==j)
        if not ids.any():return None
        weights[ids]=1/(classes*ids.sum())
    mean=weights@x;scale=np.sqrt(np.maximum(weights@((x-mean)**2),K.EPS**2));z=(x-mean)/scale
    L=1+.5*(weights@np.sum(z*z,1)+1);coef=np.zeros((x.shape[1],classes));bias=np.zeros(classes)
    target=np.eye(classes)[states]
    for _ in range(100):
        scores=z@coef+bias;prob=np.exp(scores-logsumexp(scores,1,keepdims=True));d=weights[:,None]*(prob-target)
        coef-=(z.T@d+coef)/L;bias-=d.sum(0)/L
    return mean,scale,coef,bias,dict(steps=100,regularization=1,lipschitz=float(L),class_balance=True)


def _predict_softmax(model,x):
    mean,scale,coef,bias,_=model;scores=(x-mean)/scale@coef+bias
    return np.exp(scores-logsumexp(scores,1,keepdims=True))


def e207(p,control=None):
    re=H.grid_edges(p.ep.r_rgb.shape[:2]);lab=p.labels;states=np.where(lab[re[:,0]]&~lab[re[:,1]],0,
           np.where(~lab[re[:,0]]&lab[re[:,1]],1,2));rgb=H.normalized_rgb(p.ep.r_rgb)
    difference=np.mean((rgb[re[:,0]]-rgb[re[:,1]])**2,1);strong=difference>=max(float(np.median(difference)),1e-4)
    valid=p.rvalid[re[:,0]]&p.rvalid[re[:,1]]&((states!=2)|strong)
    ids=_sample_edges(re,states,valid)
    if not len(ids) or len(np.unique(states[ids]))<3:return p.U.copy(),dict(status='fallback_missing_ownership_roles')
    x=K.halfdisk_features(p.ep.r_rgb,p.rtokens,p.ep.r,re[ids]);model=_softmax_head(x,states[ids],np.ones(len(ids),bool))
    if model is None:return p.U.copy(),dict(status='fallback_missing_ownership_roles')
    sourceprob=_predict_softmax(model,x);sourcepot=-np.log(np.maximum(sourceprob,K.EPS));scale=K.source_scale(sourcepot[:,0]-sourcepot[:,1])
    qe=H.grid_edges(p.U.shape);pots=np.empty((len(qe),4))
    for start in range(0,len(qe),256):
        features=K.halfdisk_features(p.ep.q_rgb,p.qtokens,p.ep.q,qe[start:start+256]);prob=_predict_softmax(model,features)
        cost=-np.log(np.maximum(prob,K.EPS))/scale
        pots[start:start+len(prob)]=np.stack((cost[:,2],cost[:,1],cost[:,0],cost[:,2]),1)
    if control=='independent':
        evidence=np.zeros(p.U.size);np.add.at(evidence,qe[:,0],pots[:,1]-pots[:,2]);np.add.at(evidence,qe[:,1],pots[:,2]-pots[:,1])
        out,cert=K.cut(p.U,p.ep.q_rgb,evidence)
    else:out,cert=K.cut(p.U,p.ep.q_rgb,directed=(qe,pots))
    return out.astype(float),dict(status='ok',binary_optimizer=True,fit=model[-1],source_edges=len(ids),
            halfdisk_radius_pixels=4,source_absmedian_scale=scale,cut=cert)


def e208(p,control=None):
    if not np.any(p.anchors==1) or not np.any(p.anchors==-1):return p.U.copy(),dict(status='fallback_missing_barrier_double_anchors')
    df=K.exact_barrier(p.ep.q_rgb,p.anchors==1);db=K.exact_barrier(p.ep.q_rgb,p.anchors==-1)
    ref=K.H.native_to_original(p.r0.reshape(p.ep.r_hw),p.ep.r_rgb.shape[:2],p.ep.reference_geometry)
    rf=K.exact_barrier(p.ep.r_rgb,(ref>.5)&p.labels.reshape(ref.shape));rb=K.exact_barrier(p.ep.r_rgb,(ref<=.5)&~p.labels.reshape(ref.shape))
    if not np.isfinite(rf).all() or not np.isfinite(rb).all():return p.U.copy(),dict(status='fallback_missing_source_barrier_roles')
    scale=K.source_scale((rb-rf).ravel(),p.rvalid);evidence=(db-df)/scale
    if np.max(np.abs(evidence))<=K.EPS:return p.U.copy(),dict(status='fallback_indistinguishable_barriers')
    if control=='linear':return expit(K.logit(p.U)+evidence),dict(status='ok',channel_interval_levels=16,source_absmedian_scale=scale)
    out,cert=K.cut(p.U,p.ep.q_rgb,evidence)
    return out.astype(float),dict(status='ok',binary_optimizer=True,channel_interval_levels=16,
          solver='all_intensity_intervals_8_connected_reachability',same_RGB_path_minimum=False,
          source_absmedian_scale=scale,finite_B_anchor_penalty=True,cut=cert)


def e209(p,control=None):
    if not np.any(p.anchors==1):return p.U.copy(),dict(status='fallback_no_geodesic_roots')
    distance,pre,sources=K.shortest_forest(p.ep.q_rgb,p.U,p.anchors==1)
    valid=pre>=0;child=np.flatnonzero(valid);parents=pre[valid]
    if not len(child):return p.U.copy(),dict(status='fallback_empty_forest')
    if control=='distance':
        scale=max(float(np.median(distance[np.isfinite(distance)])),K.EPS);out,cert=K.cut(p.U,p.ep.q_rgb,-distance/scale)
    else:
        de=np.stack((parents,child),1);pot=np.tile((0.,1.,0.,0.),(len(de),1));out,cert=K.cut(p.U,p.ep.q_rgb,directed=(de,pot))
    return out.astype(float),dict(status='ok',binary_optimizer=True,forest_edges=len(child),
        roots=int(np.sum(p.anchors==1)),finite_parent_BG_child_FG_cost=1.,cut=cert)


def _gradient(x):
    gx=np.zeros_like(x);gy=np.zeros_like(x);gx[:,:-1]=x[:,1:]-x[:,:-1];gy[:-1]=x[1:]-x[:-1]
    return gx,gy


def _adjoint(x,y):
    out=np.zeros_like(x);out[:,:-1]-=x[:,:-1];out[:,1:]+=x[:,:-1];out[:-1]-=y[:-1];out[1:]+=y[:-1]
    return out


def _level_energy_gradient(phi,d0,d1,curvature=1.):
    eps=1.;h=.5+np.arctan(phi/eps)/np.pi;delta=eps/(np.pi*(eps*eps+phi*phi));dd=-2*eps*phi/(np.pi*(eps*eps+phi*phi)**2)
    qx,qy=_gradient(phi);norm=np.sqrt(qx*qx+qy*qy+1e-6);nx,ny=qx/norm,qy/norm;k=-_adjoint(nx,ny)
    weight=1+curvature*k*k;energy=float(np.sum(h*d1+(1-h)*d0+delta*norm*weight))
    r=2*curvature*delta*norm*k;rx,ry=_gradient(r);rx=-rx;ry=-ry;dot=nx*rx+ny*ry
    px=delta*weight*nx+(rx-nx*dot)/norm;py=delta*weight*ny+(ry-ny*dot)/norm
    grad=delta*(d1-d0)+dd*norm*weight+_adjoint(px,py)
    return energy,grad


def e210(p,control=None):
    pair=_colour(p);evidence=np.zeros(p.U.size) if pair is None else pair[0];lg=K.logit(p.U)+evidence.reshape(p.U.shape)
    d0=np.logaddexp(0,lg);d1=np.logaddexp(0,-lg);cu=0. if control=='nocurvature' else 1.;candidates=[]
    direct=H.native_to_original(p.u0.reshape(p.ep.q_hw),p.U.shape,p.ep.query_geometry);logs=[]
    for threshold in (.25,.5,.75):
        mask=p.U>threshold;phi=ndimage.distance_transform_edt(mask)-ndimage.distance_transform_edt(~mask)
        energy,_=_level_energy_gradient(phi,d0,d1,cu);hist=[energy];births=deaths=0
        for _ in range(5):
            energy,g=_level_energy_gradient(phi,d0,d1,cu);g=np.where(np.abs(phi)<=3,g,0);step=.25/max(float(np.max(np.abs(g))),1.)
            for _ in range(20):
                trial=phi-step*g;value,_=_level_energy_gradient(trial,d0,d1,cu)
                if value<=energy:phi=trial;energy=value;break
                step*=.5
            # Topology changes are proposed from direct semantic evidence and
            # accepted only by the same FULL elastica/data energy.
            for positive in (True,False):
                domain=(direct>.5)&(phi<=0) if positive else (direct<=.5)&(phi>0)
                cc,n=ndimage.label(domain)
                for j in range(1,n+1):
                    ids=cc==j;trial=phi.copy();trial[ids]=.5 if positive else -.5;value,_=_level_energy_gradient(trial,d0,d1,cu)
                    if value<energy:phi=trial;energy=value;births+=int(positive);deaths+=int(not positive)
            hist.append(energy)
        candidates.append((energy,phi>0));logs.append(dict(initial_level=threshold,full_energy_history=hist,births=births,deaths=deaths))
    best=min(range(3),key=lambda j:candidates[j][0]);return candidates[best][1].astype(float),dict(status='ok',binary_optimizer=True,
        level_sets=logs,selected=best,curvature_weight=cu,global_optimality=False,finite_narrow_band_steps=5)
