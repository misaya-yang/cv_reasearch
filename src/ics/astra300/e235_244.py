"""Second batch of supplied E methods: explicit RGB evidence and source fits."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.special import expit,logsumexp
from . import e_helpers_226_250 as H
from . import e226_250 as B


def _cut(problem,evidence,pair_extra=None):
    from ics.methods.pro_paired_environment import exact_potts_cut
    U=problem.U.ravel();u=np.clip(U,H.EPS,1-H.EPS)
    unary=np.log(u/(1-u))+np.asarray(evidence,float).ravel()
    edges,cap=H.rgb_edges(problem.ep.q_rgb)
    if pair_extra is not None:cap=cap+np.maximum(pair_extra,0)
    labels,cert=exact_potts_cut(unary,edges,cap)
    return labels.reshape(problem.U.shape).astype(float),cert


def _native_fields(problem,role='q',wf=None,wvalid=None):
    ep=problem.ep
    wf=ep.wf if wf is None else wf;wvalid=ep.wvalid if wvalid is None else wvalid
    known=wvalid>0
    cov=np.divide(wf,wvalid,out=np.zeros_like(wf),where=known)
    f=known&(cov>=.9);b=known&(cov<=.1)
    if not f.any() or not b.any():return None
    z=ep.q if role=='q' else ep.r;hw=ep.q_hw if role=='q' else ep.r_hw
    shape=ep.original_shape if role=='q' else ep.r_rgb.shape[:2]
    g=ep.query_geometry if role=='q' else ep.reference_geometry
    ff=np.empty(len(z));bb=np.empty(len(z))
    for start in range(0,len(z),128):
        sim=z[start:start+128]@ep.r.T
        ff[start:start+128]=sim[:,f].max(1);bb[start:start+128]=sim[:,b].max(1)
    return H.native_to_original(ff.reshape(hw),shape,g),H.native_to_original(bb.reshape(hw),shape,g)


def _lifting_design(rgb,u,fsim,bsim,centers=None):
    shape=u.shape;ids=np.arange(u.size).reshape(shape)
    if centers is None:centers=np.arange(u.size)
    centers=np.asarray(centers,int);cy,cx=np.divmod(centers,shape[1])
    colour=np.asarray(rgb,float).reshape(-1,3)/255.;uf=u.ravel()
    f,b=fsim.ravel(),bsim.ravel();features=[];values=[]
    for dy in range(-2,3):
        for dx in range(-2,3):
            ni=ids[np.clip(cy+dy,0,shape[0]-1),np.clip(cx+dx,0,shape[1]-1)]
            features.append(np.stack((np.sum((colour[ni]-colour[centers])**2,1),
                uf[ni]-uf[centers],np.abs(uf[ni]-uf[centers]),f[ni]-f[centers],
                b[ni]-b[centers],np.full(len(centers),dy/2),np.full(len(centers),dx/2)),1))
            values.append(uf[ni])
    return np.stack(features,1).astype(np.float32),np.stack(values,1).astype(np.float32)


def e243_source_learned_lift(problem,control=None):
    ep=problem.ep;qfields=_native_fields(problem,'q')
    if qfields is None:return problem.U.copy(),dict(status='fallback_missing_pure_source_role')
    valid=problem.rvalid;ids=np.flatnonzero(valid)
    if not np.any(valid&problem.labels) or not np.any(valid&~problem.labels):
        return problem.U.copy(),dict(status='fallback_missing_two_source_targets')
    # Same per-token budget prevents P x 25 x F from becoming a hidden memory
    # assumption. Every reference token is represented; no query labels select.
    # A whole coarse field and every neighbour response of a training example
    # come from ONE bank excluding that example's spatial block. Stitching
    # per-token OOF fields first would leak held labels through neighbouring
    # predictions whose banks contained that block.
    h,w0=ep.r_hw;yy,xx=np.indices(ep.r_hw);designs=[];observations=[];groups=[];valid_folds=0
    for by in range(2):
        for bx in range(2):
            y0,y1=by*h//2,(by+1)*h//2;x0,x1=bx*w0//2,(bx+1)*w0//2
            test=((yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)).ravel()
            excluded=((yy>=max(0,y0-1))&(yy<min(h,y1+1))&
                      (xx>=max(0,x0-1))&(xx<min(w0,x1+1))).ravel()
            vv=np.where(excluded,0,ep.wvalid);ff=np.where(excluded,0,ep.wf)
            fields=_native_fields(problem,'r',ff,vv)
            if fields is None:continue
            native,_=H.direct_u0(ep.r,ep.r,ff,vv)
            coarse=H.native_to_original(native.reshape(ep.r_hw),ep.r_rgb.shape[:2],ep.reference_geometry)
            centers=[]
            for token in np.flatnonzero(test&(ep.wvalid>0)):
                points=np.flatnonzero(valid&(problem.rtokens==token))
                if len(points):centers.append(B._spread_rows(points,16))
            if not centers:continue
            centers=np.concatenate(centers);x,v=_lifting_design(ep.r_rgb,coarse,*fields,centers)
            groups.append(centers);designs.append(x);observations.append(v);valid_folds+=1
    if valid_folds<2:return problem.U.copy(),dict(status='fallback_insufficient_source_lifting_tasks',valid_source_folds=valid_folds)
    train=np.concatenate(groups);psi=np.concatenate(designs);values=np.concatenate(observations)
    w=H.balanced_weights(problem.labels[train],np.ones(len(train),bool),problem.rtokens[train])
    target=problem.labels[train].astype(float);theta=np.zeros(7)
    best=theta.copy();bestloss=np.inf;losses=[]
    linear=None
    if control=='linear':
        inputs=np.concatenate((psi.reshape(len(psi),-1),values),1)
        linear=H.fit_pixel_logistic(inputs,np.zeros((1,1)),np.zeros(len(inputs),int),
                                    target,np.ones(len(inputs),bool))
    for iteration in range(100):
        score=np.einsum('skf,f->sk',psi,theta)
        alpha=np.exp(score-logsumexp(score,axis=1,keepdims=True))
        prediction=np.sum(alpha*values,1);pc=np.clip(prediction,H.EPS,1-H.EPS)
        loss=float(-w@(target*np.log(pc)+(1-target)*np.log(1-pc))+theta@theta)
        if loss<bestloss:bestloss=loss;best=theta.copy()
        deriv=(prediction-target)/np.maximum(pc*(1-pc),H.EPS)
        feature_derivative=np.einsum('sk,sk,skf->sf',alpha,values-prediction[:,None],psi)
        gradient=feature_derivative.T@(w*deriv)+2*theta
        # Fixed disclosed step. This is bounded nonconvex fitting, not a
        # Lipschitz/global-optimization claim about the learned lift.
        theta-=.05*gradient;losses.append(loss)
    result=np.empty(problem.U.size);qU=problem.U
    if control=='uniform':best=np.zeros_like(best)
    for start in range(0,len(result),8192):
        ids=np.arange(start,min(start+8192,len(result)))
        x,v=_lifting_design(ep.q_rgb,qU,*qfields,ids)
        if linear is not None:
            inputs=np.concatenate((x.reshape(len(x),-1),v),1)
            result[ids]=linear.predict(inputs,np.zeros((1,1)),np.zeros(len(inputs),int))
            continue
        scores=np.einsum('skf,f->sk',x,best)
        alpha=np.exp(scores-logsumexp(scores,axis=1,keepdims=True))
        result[ids]=np.sum(alpha*v,1)
    return result.reshape(problem.U.shape),dict(status='ok',train_source_pixels=len(train),
                max_pixels_per_token=16,valid_source_folds=valid_folds,fit_steps=100,step=.05,best_visited_objective=bestloss,
                theta=best.tolist(),input_coarse_field='real_out_of_block_U0_not_downsampled_MR',
                control=control,nonconvex_not_global_optimum=True)


def _contour_chains(rgb):
    gray=np.asarray(rgb,float).mean(-1)/255.
    gy=ndimage.sobel(gray,0,mode='nearest')/8;gx=ndimage.sobel(gray,1,mode='nearest')/8
    mag=np.hypot(gx,gy);level=max(float(np.quantile(mag,.75)),1e-4)
    edge=(mag>=level)&(mag>1e-4)
    # Thin by the observed local gradient normal. This uses RGB only.
    yy,xx=np.indices(edge.shape);ny=gy/np.maximum(mag,H.EPS);nx=gx/np.maximum(mag,H.EPS)
    before=ndimage.map_coordinates(mag,[yy-ny,xx-nx],order=1,mode='nearest')
    after=ndimage.map_coordinates(mag,[yy+ny,xx+nx],order=1,mode='nearest')
    edge&=(mag>=before)&(mag>=after)
    n=edge.size;shape=edge.shape;ids=np.flatnonzero(edge);visited=np.zeros(n,bool);chains=[]
    for root in ids:
        if visited[root]:continue
        chain=[];current=int(root);previous=-1
        while not visited[current]:
            visited[current]=True;chain.append(current);y,x=divmod(current,shape[1]);near=[]
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if not (dy or dx):continue
                    yy0,xx0=y+dy,x+dx
                    if 0<=yy0<shape[0] and 0<=xx0<shape[1]:
                        j=yy0*shape[1]+xx0
                        if edge.ravel()[j] and not visited[j]:near.append(j)
            if not near:break
            # Geometric chain continuity, not a role/shape template.
            if previous>=0:
                py,px=divmod(previous,shape[1]);direction=np.array([y-py,x-px])
                near.sort(key=lambda j:(-np.dot(np.array(divmod(j,shape[1]))-np.array([y,x]),direction),j))
            current,previous=near[0],current
        if len(chain)>=3:chains.append(np.array(chain,int))
    return chains,ny,nx


def _chain_words(chain,rgb,semantic,ny,nx):
    shape=semantic.shape;y,x=np.divmod(chain,shape[1]);nY=ny.ravel()[chain];nX=nx.ravel()[chain]
    ly=np.clip(np.rint(y+2*nY).astype(int),0,shape[0]-1);lx=np.clip(np.rint(x+2*nX).astype(int),0,shape[1]-1)
    ry=np.clip(np.rint(y-2*nY).astype(int),0,shape[0]-1);rx=np.clip(np.rint(x-2*nX).astype(int),0,shape[1]-1)
    left=ly*shape[1]+lx;right=ry*shape[1]+rx
    direction=np.mod(np.rint(np.arctan2(np.gradient(y.astype(float)),np.gradient(x.astype(float)))*4/np.pi).astype(int),8)
    turn=np.mod(np.diff(direction,prepend=direction[0]),8)
    colour=np.asarray(rgb,float).mean(-1).ravel()/255.;dc=colour[left]-colour[right]
    colourcode=(dc>0).astype(int)+2*(np.abs(dc)>.1)
    ds=semantic.ravel()[left]-semantic.ravel()[right];semcode=np.where(ds<-.05,0,np.where(ds>.05,2,1))
    return turn*12+colourcode*3+semcode,left,right


def _markov_model(sequences,order=2):
    vocab=96
    if order==0:
        counts=np.ones(vocab)
        for words in sequences:np.add.at(counts,words,1)
        return np.log(counts/counts.sum())
    counts=np.ones((vocab,vocab,vocab))
    for words in sequences:
        if len(words)>=3:np.add.at(counts,(words[:-2],words[1:-1],words[2:]),1)
    return np.log(counts/counts.sum(2,keepdims=True))


def _chain_likelihood(words,model):
    if model.ndim==1:return float(model[words].mean())
    if len(words)<3:return -np.log(96.)
    a=float(model[words[:-2],words[1:-1],words[2:]].mean())
    rev=words[::-1];b=float(model[rev[:-2],rev[1:-1],rev[2:]].mean())
    return .5*(a+b)


def e239_ordered_contour(problem,control=None):
    ep=problem.ep;native=np.where(np.isfinite(problem.oob),problem.oob,problem.r0)
    rsem=H.native_to_original(native.reshape(ep.r_hw),ep.r_rgb.shape[:2],ep.reference_geometry)
    chains,ny,nx=_contour_chains(ep.r_rgb);seq=[[],[],[]];training_valid=problem.rvalid
    for chain in chains:
        words,l,r=_chain_words(chain,ep.r_rgb,rsem,ny,nx)
        valid=training_valid[l]&training_valid[r]
        left=problem.labels[l];right=problem.labels[r]
        roles=np.where(left!=right,0,np.where(left,1,2))
        # Only contiguous same-role runs make one sequence; unknown pixels
        # never lend their GT role across the source holdout support.
        for role in range(3):
            runs,count=ndimage.label(valid&(roles==role))
            for j in range(1,count+1):
                selected=words[runs==j]
                if len(selected)>=3:seq[role].append(selected)
    if not seq[0] or not (seq[1] or seq[2]):return problem.U.copy(),dict(status='fallback_missing_source_sequence_classes')
    if control=='shuffled':
        rng=np.random.default_rng(0);seq=[[rng.permutation(a) for a in group] for group in seq]
    models=[_markov_model(s,0 if control=='independent' else 2) for s in seq]
    qchains,qny,qnx=_contour_chains(ep.q_rgb);evidence=np.zeros(problem.U.size);support=0
    qsem=H.native_to_original(problem.u0.reshape(ep.q_hw),ep.original_shape,ep.query_geometry)
    for chain in qchains:
        words,l,r=_chain_words(chain,ep.q_rgb,qsem,qny,qnx)
        ll=[_chain_likelihood(words,m) for m in models]
        strength=ll[0]-max(ll[1:]);direction=np.sign(qsem.ravel()[l]-qsem.ravel()[r])
        usable=direction!=0
        if strength>0 and usable.any():
            np.add.at(evidence,l[usable],strength*direction[usable]);np.add.at(evidence,r[usable],-strength*direction[usable])
            support+=int(usable.sum())
    # Source log-ratio scale is computed from the same observed sequences.
    rs=[_chain_likelihood(a,models[0])-max(_chain_likelihood(a,models[1]),_chain_likelihood(a,models[2]))
        for a in seq[0]+seq[1]+seq[2]]
    evidence,scale=H.normalize_evidence(evidence,rs)
    if not support:return problem.U.copy(),dict(status='fallback_no_directed_chain_support',source_sequences=[len(s) for s in seq])
    out,cert=_cut(problem,evidence)
    return out,dict(status='ok',binary_optimizer=True,order=0 if control=='independent' else 2,
                    source_sequences=[len(s) for s in seq],query_chains=len(qchains),
                    directed_supported_positions=support,evidence_scale=scale,cut=cert,control=control)


def e244_fragment_coexplanation(problem,control=None):
    ep=problem.ep;components,count=ndimage.label(problem.U>.5)
    if not count:return problem.U.copy(),dict(status='fallback_no_positive_fragment')
    anchors=B._anchors(problem).ravel();x=H.texture_phi(ep.q_rgb);ids=[]
    for j in range(1,count+1):
        points=np.flatnonzero(components.ravel()==j)
        if len(points)>=8 and np.any(anchors[points]==1):ids.append(points)
    retained=ids[:128];unmodeled=ids[128:]
    if not retained:return problem.U.copy(),dict(status='fallback_no_directly_authenticated_fragment')
    groups=[p.copy() for p in retained];merges=0
    dim=x.shape[1]
    def mdl(points):
        model=H.fit_gmm(x[points],1)
        return -float(model.log_density(x[points]).sum())+dim*np.log(1+len(points)),model
    if control!='independent':
        while len(groups)>1:
            best=(0.,None)
            for i in range(len(groups)):
                ai,_=mdl(groups[i])
                for j in range(i+1,len(groups)):
                    aj,_=mdl(groups[j]);merged=np.concatenate((groups[i],groups[j]));am,_=mdl(merged)
                    gain=ai+aj-am
                    if gain>best[0]:best=(gain,(i,j,merged))
            if best[1] is None:break
            i,j,merged=best[1];groups[i]=merged;groups.pop(j);merges+=1
    if control is None and not merges:return problem.U.copy(),dict(status='fallback_no_shared_MDL_gain',eligible_fragments=len(retained))
    if control=='global':
        fg=H.fit_gmm(x[np.concatenate(retained)],min(len(groups),16))
        fglog=fg.log_density(x)
    else:
        models=[mdl(g)[1] for g in groups]
        fglog=logsumexp(np.stack([m.log_density(x) for m in models]),axis=0)-np.log(len(models))
    bgpoints=np.flatnonzero(anchors==-1)
    if len(bgpoints)<8:return problem.U.copy(),dict(status='fallback_no_initial_negative_anchors')
    bg=H.fit_gmm(x[bgpoints],2);evidence=fglog-bg.log_density(x)
    # Source training colour models determine the evidence units; query
    # adaptation never supplies a new semantic label or fills a gap by fiat.
    source=H.diagonal_colour_evidence(ep.q_rgb,ep.r_rgb,problem.labels,problem.rvalid,k=2)
    if source is None:return problem.U.copy(),dict(status='fallback_source_identity_missing')
    source_evidence,normalizer=source
    evidence/=max(float(normalizer['scale']),H.EPS)
    for points in unmodeled:evidence[points]=0
    out,cert=_cut(problem,evidence)
    for points in unmodeled:out.ravel()[points]=problem.U.ravel()[points]
    return out,dict(status='ok',binary_optimizer=True,eligible_fragments=len(ids),modeled_fragments=len(retained),
                    unmodeled_fragments_exact_host=len(unmodeled),shared_groups=len(groups),merges=merges,
                    sharing_only_parameters_not_labels=True,cut=cert,control=control)


def _phase_edge_design(rgb,U,hw,g):
    shape=U.shape;edges=H.grid_edges(shape);p=H.pixel_tokens(shape,hw,g)
    c=np.asarray(rgb,float).reshape(-1,3)/255.;u=U.ravel();a,b=edges.T
    colour=np.mean((c[a]-c[b])**2,1);phase=(p[a]!=p[b]).astype(float)
    phi=np.stack((colour,np.abs(u[a]-u[b]),np.minimum(np.abs(u[a]-.5),np.abs(u[b]-.5)),phase),1)
    predicted=(u[a]>.5)!=(u[b]>.5)
    return edges,phi,predicted


def e235_grid_phase_rgb(problem,control=None):
    ep=problem.ep;h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw);features=[];targets=[]
    if control=='plain_RGB':
        colour=H.diagonal_colour_evidence(ep.q_rgb,ep.r_rgb,problem.labels,problem.rvalid,k=2)
        if colour is None:return problem.U.copy(),dict(status='fallback_missing_source_colour_roles')
        evidence,fit=colour;out,cert=_cut(problem,evidence)
        return out,dict(status='ok',binary_optimizer=True,control='plain_RGB',source_colour=fit,cut=cert)
    for by in range(2):
        for bx in range(2):
            y0,y1=by*h//2,(by+1)*h//2;x0,x1=bx*w//2,(bx+1)*w//2
            held=((yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)).ravel()
            excluded=((yy>=max(0,y0-1))&(yy<min(h,y1+1))&
                      (xx>=max(0,x0-1))&(xx<min(w,x1+1))).ravel()
            p,_=H.direct_u0(ep.r,ep.r,np.where(excluded,0,ep.wf),np.where(excluded,0,ep.wvalid))
            field=H.native_to_original(p.reshape(ep.r_hw),ep.r_rgb.shape[:2],ep.reference_geometry)
            edges,phi,predicted=_phase_edge_design(ep.r_rgb,field,ep.r_hw,ep.reference_geometry)
            a,b=edges.T;test=held[problem.rtokens]&problem.rvalid
            keep=predicted&test[a]&test[b]
            if keep.any():
                features.append(phi[keep]);targets.append(problem.labels[a[keep]]!=problem.labels[b[keep]])
    if not features:return problem.U.copy(),dict(status='fallback_no_source_predicted_boundaries')
    features=np.concatenate(features);targets=np.concatenate(targets)
    if np.unique(targets).size<2:return problem.U.copy(),dict(status='fallback_no_source_correct_and_artifact_boundaries')
    fitfeatures=features.copy()
    if control=='no_phase':fitfeatures[:,-1]=0
    head=H.fit_pixel_logistic(fitfeatures,np.zeros((1,1)),np.zeros(len(features),int),targets,np.ones(len(features),bool))
    edges,qphi,predicted=_phase_edge_design(ep.q_rgb,problem.U,ep.q_hw,ep.query_geometry)
    if control=='no_phase':qphi[:,-1]=0
    confidence=head.predict(qphi,np.zeros((1,1)),np.zeros(len(qphi),int))
    # RGB quietness merely nominates a grid-aligned ambiguity. Both same-class
    # assignments and both oriented splits remain valid labels in the full cut.
    quiet=qphi[:,0]<=max(float(np.median(features[:,0])),1e-4)
    nominate=predicted&(qphi[:,-1]>.5)&quiet
    if control=='no_phase':nominate=predicted&quiet
    colour=H.diagonal_colour_evidence(ep.q_rgb,ep.r_rgb,problem.labels,problem.rvalid,k=2)
    if colour is None:return problem.U.copy(),dict(status='fallback_missing_source_colour_roles')
    evidence,fit=colour
    extra=nominate*(1-confidence)*np.abs(qphi[:,1])
    if control=='plain_RGB':extra[:]=0
    if not nominate.any() and control!='plain_RGB':return problem.U.copy(),dict(status='fallback_no_query_phase_ambiguity')
    out,cert=_cut(problem,evidence,extra)
    return out,dict(status='ok',binary_optimizer=True,source_prediction_boundary_samples=len(features),
                    source_true_interface_rate=float(targets.mean()),query_phase_ambiguities=int(nominate.sum()),
                    fit=head.info,source_colour=fit,cut=cert,control=control,
                    boundary_targets_are_not_used_to_nominate_source_samples=True)


def _profiles(U,hw,geometry,max_per_island=128):
    fg=U>.5;component,count=ndimage.label(fg)
    inside=ndimage.distance_transform_edt(fg);outside=ndimage.distance_transform_edt(~fg)
    signed=inside-outside;gy,gx=np.gradient(signed);mag=np.hypot(gy,gx)
    boundary=fg&~ndimage.binary_erosion(fg)
    if geometry:
        side=geometry['view_side'];sh,sw=geometry['resized_hw']
        physical=max(U.shape[0]*side/(hw[0]*sh),U.shape[1]*side/(hw[1]*sw))
    else:physical=max(U.shape[0]/hw[0],U.shape[1]/hw[1])
    length=max(1,int(np.ceil(4*physical)));profiles=[]
    for j in range(1,count+1):
        centers=B._spread_rows(np.flatnonzero(boundary&(component==j)),max_per_island)
        for center in centers:
            y,x=divmod(int(center),U.shape[1]);ny=-gy[y,x]/max(mag[y,x],H.EPS);nx=-gx[y,x]/max(mag[y,x],H.EPS)
            if abs(ny)+abs(nx)<=H.EPS:continue
            for direction in (1,-1):
                k=np.arange(1,length+1)*direction
                ys=np.clip(np.rint(y+k*ny).astype(int),0,U.shape[0]-1)
                xs=np.clip(np.rint(x+k*nx).astype(int),0,U.shape[1]-1)
                points=ys*U.shape[1]+xs
                # Consecutive repeated discretizations carry one observation.
                points=points[np.r_[True,points[1:]!=points[:-1]]]
                profiles.append((int(center),points))
    return profiles,length,count


def e240_sequential_boundary(problem,control=None):
    ep=problem.ep;h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw);upper_errors=[];lower_errors=[];valid_folds=0
    rphi=H.texture_phi(ep.r_rgb);qphi=H.texture_phi(ep.q_rgb)
    for by in range(2):
        for bx in range(2):
            y0,y1=by*h//2,(by+1)*h//2;x0,x1=bx*w//2,(bx+1)*w//2
            held=((yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)).ravel()[problem.rtokens]&problem.rvalid
            excluded=((yy>=max(0,y0-1))&(yy<min(h,y1+1))&
                      (xx>=max(0,x0-1))&(xx<min(w,x1+1))).ravel()
            train=problem.rvalid&~excluded[problem.rtokens]
            f=H.fit_gmm(rphi[train&problem.labels],2);b=H.fit_gmm(rphi[train&~problem.labels],2)
            if f is None or b is None:continue
            rp,_=H.direct_u0(ep.r,ep.r,np.where(excluded,0,ep.wf),np.where(excluded,0,ep.wvalid))
            rU=H.native_to_original(rp.reshape(ep.r_hw),ep.r_rgb.shape[:2],ep.reference_geometry)
            ll=f.log_density(rphi)-b.log_density(rphi)
            scale=max(float(np.median(np.abs(ll[train]))),H.EPS)
            increments=ll/scale+np.log(np.clip(rU.ravel(),H.EPS,1-H.EPS)/(1-np.clip(rU.ravel(),H.EPS,1-H.EPS)))
            profiles,_,_= _profiles(rU,ep.r_hw,ep.reference_geometry)
            for center,points in profiles:
                if not held[center]:continue
                # Calibration never reads a target outside this held block.
                prefix=np.cumprod(held[points]).astype(bool);points=points[prefix]
                if not len(points):continue
                cumulative=np.cumsum(increments[points])
                upper_errors.extend(cumulative[~problem.labels[points]].tolist())
                lower_errors.extend(cumulative[problem.labels[points]].tolist())
            valid_folds+=1
    if valid_folds<2 or len(upper_errors)<8 or len(lower_errors)<8:
        return problem.U.copy(),dict(status='fallback_unidentified_source_stopping_bounds',valid_source_folds=valid_folds)
    upper=max(float(np.quantile(upper_errors,.95)),H.EPS)
    lower=min(float(np.quantile(lower_errors,.05)),-H.EPS)
    f=H.fit_gmm(rphi[problem.rvalid&problem.labels],2);b=H.fit_gmm(rphi[problem.rvalid&~problem.labels],2)
    if f is None or b is None:return problem.U.copy(),dict(status='fallback_source_colour_roles')
    rs=f.log_density(rphi)-b.log_density(rphi);qs=f.log_density(qphi)-b.log_density(qphi)
    scale=max(float(np.median(np.abs(rs[problem.rvalid]))),H.EPS)
    u=np.clip(problem.U.ravel(),H.EPS,1-H.EPS);increment=qs/scale+np.log(u/(1-u))
    profiles,length,islands=_profiles(problem.U,ep.q_hw,ep.query_geometry)
    evidence=np.zeros(problem.U.size);decided=0
    for _,points in profiles:
        cumulative=np.cumsum(increment[points])
        if control=='fixed_length':
            sign=np.sign(cumulative[-1]);stop=len(points)-1
        else:
            hit=np.flatnonzero((cumulative>=upper)|(cumulative<=lower))
            if not len(hit):continue
            stop=int(hit[0]);sign=1 if cumulative[stop]>=upper else -1
        if sign:
            np.add.at(evidence,points[:stop+1],sign);decided+=1
    if not decided:return problem.U.copy(),dict(status='fallback_all_profiles_undecided',profiles=len(profiles))
    out,cert=_cut(problem,evidence)
    return out,dict(status='ok',binary_optimizer=True,source_stopping_upper=upper,source_stopping_lower=lower,
                    valid_source_folds=valid_folds,profiles=len(profiles),decided_profiles=decided,
                    query_positive_islands=islands,maximum_profile_length=length,max_centers_per_island=128,
                    cut=cert,control=control,correlated_samples_not_independent_risk_guarantee=True)


METHODS={
 'E235':lambda ep:B._call(ep,'E235',e235_grid_phase_rgb),
 'E239':lambda ep:B._call(ep,'E239',e239_ordered_contour),
 'E240':lambda ep:B._call(ep,'E240',e240_sequential_boundary),
 'E243':lambda ep:B._call(ep,'E243',e243_source_learned_lift),
 'E244':lambda ep:B._call(ep,'E244',e244_fragment_coexplanation),
}
CONTROLS={
 'E235_same_head_no_grid_phase':lambda ep:B._call(ep,'E235',e235_grid_phase_rgb,control='no_phase'),
 'E235_plain_RGB_cut':lambda ep:B._call(ep,'E235',e235_grid_phase_rgb,control='plain_RGB'),
 'E239_independent_words':lambda ep:B._call(ep,'E239',e239_ordered_contour,control='independent'),
 'E239_same_words_shuffled_order':lambda ep:B._call(ep,'E239',e239_ordered_contour,control='shuffled'),
 'E240_fixed_length_all_observations':lambda ep:B._call(ep,'E240',e240_sequential_boundary,control='fixed_length'),
 'E243_uniform_same5x5':lambda ep:B._call(ep,'E243',e243_source_learned_lift,control='uniform'),
 'E243_RGB_guided5x5':lambda ep:B._call(ep,'E243',B.e236_affine_lift,control='guided'),
 'E243_linear_same_neighbour_inputs':lambda ep:B._call(ep,'E243',e243_source_learned_lift,control='linear'),
 'E244_independent_fragments':lambda ep:B._call(ep,'E244',e244_fragment_coexplanation,control='independent'),
 'E244_same_components_global_GMM':lambda ep:B._call(ep,'E244',e244_fragment_coexplanation,control='global'),
}
RESOURCES={id:dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=id!='E243',extra_encoder_forwards=0) for id in METHODS}
