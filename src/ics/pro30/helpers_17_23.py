"""Exact finite Pro30 primitives. All labels are current reference labels."""
from __future__ import annotations
from collections import OrderedDict
import time
import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation
from scipy.sparse.linalg import eigsh, splu
from scipy.special import expit
from .common import unit, require_artifact, ArtifactUnavailable, readonly
from ics.cpu100.common import rgb_view, prototype_margin

EPS=1e-12
_CACHE=OrderedDict()
_CACHE_EP=None


def cached(ep,name,builder):
    global _CACHE_EP
    identity=(ep.source_id,id(ep.q),id(ep.r))
    if _CACHE_EP!=identity:_CACHE.clear();_CACHE_EP=identity
    key=(ep.source_id,id(ep.q),id(ep.r),name)
    if key in _CACHE:return _CACHE[key]
    value=builder();_CACHE[key]=value
    # These are episode computations, never cross-episode labels/model fitting.
    while len(_CACHE)>24:_CACHE.popitem(last=False)
    return value


def mm(a,b):
    # Some Accelerate builds set stale FP status bits despite finite operands.
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):out=np.asarray(a)@np.asarray(b)
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite matrix product')
    return out


def groups4(hw):
    yy,xx=np.indices(hw)
    return ((np.minimum(yy*4//hw[0],3)+2*np.minimum(xx*4//hw[1],3))%4).ravel()


def risk(score,fg,bg,cut):
    pred=np.asarray(score)>cut
    return .5*(np.sum(fg[~pred])/max(np.sum(fg),EPS)+np.sum(bg[pred])/max(np.sum(bg),EPS))


def best_cut(score,fg,bg):
    """Exact observed-score strict > cut balanced classification threshold."""
    score=np.asarray(score,float);fg=np.asarray(fg,float);bg=np.asarray(bg,float)
    keep=(fg+bg)>0;score=score[keep];fg=fg[keep];bg=bg[keep]
    if not len(score) or fg.sum()<=0 or bg.sum()<=0:return 0.,np.nan
    order=np.argsort(score,kind='stable');s=score[order];f=fg[order];b=bg[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    # Include all-positive, all-negative and exact observed ties.
    cuts=np.r_[np.nextafter(s[0],-np.inf),s[ends]]
    loss=.5*np.r_[1.,np.cumsum(f)[ends]/fg.sum()+(bg.sum()-np.cumsum(b)[ends])/bg.sum()]
    # all-positive risk is .5, not 1; first array is scaled by .5 above.
    idx=np.lexsort((np.arange(len(cuts)),np.abs(cuts),loss))[0]
    return float(cuts[idx]),float(loss[idx])


def fps_lloyd(x,weights,k,rounds=5):
    x=np.asarray(x,float);w=np.asarray(weights,float);ids=np.flatnonzero(w>0)
    if not len(ids):return np.empty((0,x.shape[1])),np.empty(0),np.full(len(x),-1,int)
    xx=unit(x[ids]);ww=w[ids];k=min(int(k),len(ids));chosen=[int(np.argmax(ww))]
    distance=1-mm(xx,xx[chosen[0]])
    for _ in range(1,k):
        nxt=int(np.argmax(distance*ww));
        if distance[nxt]<=1e-12:break
        chosen.append(nxt);distance=np.minimum(distance,1-mm(xx,xx[nxt]))
    centers=xx[chosen].copy()
    for _ in range(rounds):
        labels=np.argmax(mm(xx,centers.T),axis=1)
        mass=np.bincount(labels,weights=ww,minlength=len(centers));sums=np.zeros_like(centers)
        np.add.at(sums,labels,xx*ww[:,None]);nonempty=mass>0
        centers=unit(sums[nonempty])
    labels=np.argmax(mm(xx,centers.T),axis=1);mass=np.bincount(labels,weights=ww,minlength=len(centers))
    full=np.full(len(x),-1,int);full[ids]=labels
    return centers,mass,full


def proto(ep):
    if ep.wf.sum()<=0:return np.full(len(ep.q),-1.)
    if ep.wb.sum()<=0:raise ValueError('Use the common explicit single-class degeneration')
    return mm(ep.q,unit(np.sum(ep.r*ep.wf[:,None],axis=0))-unit(np.sum(ep.r*ep.wb[:,None],axis=0)))


def image_and_valid(ep,role):
    image=np.asarray(rgb_view(ep,role));g=ep.reference_geometry if role=='r' else ep.query_geometry
    valid=np.zeros(image.shape[:2],bool)
    if g:
        oy,ox=map(int,g['padding_top_left']);sh,sw=map(int,g['resized_hw']);valid[oy:oy+sh,ox:ox+sw]=True
    else:valid[:]=True
    return image,valid


def _area_resize_matrix(nin,nout):
    rows=[];cols=[];values=[];scale=nin/nout
    for j in range(nout):
        lo=j*scale;hi=(j+1)*scale
        for i in range(int(np.floor(lo)),min(nin,int(np.ceil(hi)))):
            overlap=max(0.,min(hi,i+1)-max(lo,i))
            if overlap:rows.append(j);cols.append(i);values.append(overlap/scale)
    return sparse.csr_matrix((values,(rows,cols)),shape=(nout,nin))


def mask_canvas(ep):
    """Exact source-pixel footprint area, not bilinear/nearest MR shrinkage."""
    if ep.reference_mask is None:raise ArtifactUnavailable('M17 needs the full original reference mask')
    mask=np.asarray(ep.reference_mask,float)
    if mask.ndim!=2 or np.any((mask!=0)&(mask!=1)):raise ValueError('Complete binary original MR required')
    image,valid=image_and_valid(ep,'r');g=ep.reference_geometry
    sh,sw=map(int,g.get('resized_hw',image.shape[:2]));oy,ox=map(int,g.get('padding_top_left',(0,0)))
    sy=_area_resize_matrix(mask.shape[0],sh);sx=_area_resize_matrix(mask.shape[1],sw)
    out=np.zeros(valid.shape,float);out[oy:oy+sh,ox:ox+sw]=(sx@(sy@mask).T).T
    return out


def reference_exact_areas(ep):
    canvas=mask_canvas(ep);_,valid=image_and_valid(ep,'r');h,w=canvas.shape;rh,rw=ep.r_hw
    if h%rh or w%rw:raise ValueError('Native patch footprints require integer physical pixel extent')
    ph,pw=h//rh,w//rw
    fg=canvas.reshape(rh,ph,rw,pw).sum(axis=(1,3))/(ph*pw)
    area=valid.reshape(rh,ph,rw,pw).sum(axis=(1,3))/(ph*pw)
    return fg.ravel(),area.ravel(),canvas


def _lab(rgb):
    rgb=np.asarray(rgb,float)/(255. if np.asarray(rgb).dtype==np.uint8 else 1.)
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    xyz=mm(linear,np.array([[.4124564,.2126729,.0193339],[.3575761,.7151522,.119192],[.1804375,.072175,.9503041]]))
    xyz/=np.array([.95047,1.,1.08883]);delta=6/29
    f=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta**2)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)


def slic_fixed(rgb,valid,segments=2048,compactness=10.,rounds=10):
    """Deterministic classic Lab SLIC with local 2S windows and connectivity.

    A dependency-free implementation of the supplied partition algorithm;
    no change to segment count/compactness to meet a runtime target.
    """
    from scipy.ndimage import label,find_objects
    lab=_lab(rgb);h,w=valid.shape;ys,xs=np.where(valid)
    if not len(ys):raise ValueError('No valid RGB canvas')
    y0,y1=ys.min(),ys.max()+1;x0,x1=xs.min(),xs.max()+1
    area=len(ys);segments=min(segments,area);step=np.sqrt(area/segments)
    ny=max(1,int(round((y1-y0)/step)));nx=max(1,int(round((x1-x0)/step)))
    yy=y0+(np.arange(ny)+.5)*(y1-y0)/ny;xx=x0+(np.arange(nx)+.5)*(x1-x0)/nx
    gy,gx=np.meshgrid(yy,xx,indexing='ij');cy=gy.ravel();cx=gx.ravel()
    colors=lab[np.clip(cy.astype(int),0,h-1),np.clip(cx.astype(int),0,w-1)].copy()
    labels=np.full((h,w),-1,int)
    # Nine nearby initialized grid cells cover the native SLIC local windows.
    for _ in range(rounds):
        counts=np.zeros(len(cy));sy=np.zeros_like(cy);sx=np.zeros_like(cx);sc=np.zeros_like(colors)
        for start in range(y0,y1,32):
            end=min(start+32,y1);py,px=np.indices((end-start,x1-x0));py=py+start;px=px+x0
            basey=np.minimum(((py-y0)*ny/(y1-y0)).astype(int),ny-1)
            basex=np.minimum(((px-x0)*nx/(x1-x0)).astype(int),nx-1)
            best=np.full(py.shape,np.inf);chosen=np.full(py.shape,-1,int)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    iy=basey+dy;ix=basex+dx;legal=(iy>=0)&(iy<ny)&(ix>=0)&(ix<nx)
                    ids=np.clip(iy,0,ny-1)*nx+np.clip(ix,0,nx-1)
                    ds=(py-cy[ids])**2+(px-cx[ids])**2
                    dc=np.sum((lab[start:end,x0:x1]-colors[ids])**2,axis=-1)
                    dist=dc+(compactness/step)**2*ds;dist[~legal]=np.inf
                    improve=dist<best;chosen[improve]=ids[improve];best[improve]=dist[improve]
            chosen[~valid[start:end,x0:x1]]=-1;labels[start:end,x0:x1]=chosen
            good=chosen>=0;ids=chosen[good]
            counts+=np.bincount(ids,minlength=len(cy));sy+=np.bincount(ids,weights=py[good],minlength=len(cy));sx+=np.bincount(ids,weights=px[good],minlength=len(cx))
            for c in range(3):sc[:,c]+=np.bincount(ids,weights=lab[start:end,x0:x1,c][good],minlength=len(cy))
        good=counts>0;cy[good]=sy[good]/counts[good];cx[good]=sx[good]/counts[good];colors[good]=sc[good]/counts[good,None]
    # Split disconnected islands, then merge only tiny components to a neighbor.
    out=np.full_like(labels,-1);nextid=0;minimum=max(1,int(.5*area/max(1,len(cy))))
    for i,box in enumerate(find_objects(labels+1)):
        if box is None:continue
        ysli,xsli=box;parts,nparts=label(labels[ysli,xsli]==i)
        region=out[ysli,xsli]
        for j in range(1,nparts+1):region[parts==j]=nextid;nextid+=1
    # Sequential union-find uses complete pixel adjacency, deterministic tie.
    sizes=np.bincount(out[valid],minlength=nextid);parent=np.arange(nextid)
    edges=np.r_[np.c_[out[:-1][valid[:-1]&valid[1:]],out[1:][valid[:-1]&valid[1:]]],np.c_[out[:,:-1][valid[:,:-1]&valid[:,1:]],out[:,1:][valid[:,:-1]&valid[:,1:]]]]
    adj=[set() for _ in range(nextid)]
    for a,b in np.unique(np.sort(edges,axis=1),axis=0):
        if a!=b:adj[a].add(int(b));adj[b].add(int(a))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a in np.argsort(sizes,kind='stable'):
        a=root(int(a))
        if sizes[a]>=minimum:continue
        targets={root(b) for b in adj[a]}-{a}
        if targets:
            b=min(targets,key=lambda b:(-sizes[b],b));parent[a]=b;sizes[b]+=sizes[a];adj[b]|=adj[a]
    # The supplied inverse caps its region system at2048. Connectivity can
    # split one SLIC cell into several islands; pool the smallest adjacent
    # components deterministically until the original count cap is restored.
    components=int(np.sum(parent==np.arange(nextid)))
    while components>segments:
        roots=np.flatnonzero(parent==np.arange(nextid));merged=False
        for a in roots[np.argsort(sizes[roots],kind='stable')]:
            targets={root(b) for b in adj[a]}-{int(a)}
            if targets:
                b=min(targets,key=lambda b:(-sizes[b],b));parent[a]=b;sizes[b]+=sizes[a];adj[b]|=adj[a];components-=1;merged=True;break
        if not merged:raise ValueError('Disconnected physical valid canvas cannot satisfy SLIC region cap')
    mapping=np.array([root(i) for i in range(nextid)]);_,mapping=np.unique(mapping,return_inverse=True)
    out[valid]=mapping[out[valid]]
    return out


def area_matrix(labels,valid,hw):
    """Strict integer-pixel footprints; no center assignment of partial patch."""
    h,w=valid.shape
    if h%hw[0] or w%hw[1]:raise ValueError('Integer physical pixels per native patch required')
    yy,xx=np.indices((h,w));patch=(yy//(h//hw[0])*hw[1]+xx//(w//hw[1])).ravel()
    good=valid.ravel();cols=labels.ravel()[good];rows=patch[good]
    count=np.bincount(rows,minlength=np.prod(hw)).astype(float);s=int(cols.max()+1)
    A=sparse.coo_matrix((1/count[rows],(rows,cols)),shape=(np.prod(hw),s)).tocsr()
    return A,count


def inverse_regions(ep,role):
    def build():
        start=time.perf_counter();image,valid=image_and_valid(ep,role);hw=ep.r_hw if role=='r' else ep.q_hw
        raw=np.asarray(require_artifact(ep,role+'_raw'),float).reshape(np.prod(hw),-1)
        native=ep.r if role=='r' else ep.q
        if raw.shape!=native.shape or not np.allclose(unit(raw),native,atol=2e-5,rtol=2e-5):raise ValueError('M17 actual preunit finalLN/native unit parity failed')
        labels=slic_fixed(image,valid);slic_seconds=time.perf_counter()-start
        A,W=area_matrix(labels,valid,hw);D=np.asarray(A.sum(0)).ravel();u0=(A.T@raw)/D[:,None]
        start=time.perf_counter();H=(A.T@sparse.diags(W)@A+.1*sparse.diags(D)).tocsc()
        rhs=A.T@(raw*W[:,None])+.1*D[:,None]*u0;factor=splu(H);u=factor.solve(rhs)
        residual=float(np.linalg.norm(A@u-raw)/max(np.linalg.norm(raw),EPS))
        equation_residual=float(np.linalg.norm(H@u-rhs)/max(np.linalg.norm(rhs),EPS))
        mass=np.bincount(labels[valid],minlength=len(D)).astype(float)
        return dict(labels=labels,valid=valid,A=A,W=W,D=D,u0=unit(u0),u=unit(u),mass=mass,
            info={'slic_seconds':slic_seconds,'inverse_seconds':time.perf_counter()-start,'segments':len(D),'A_nnz':int(A.nnz),
                  'A_row_sum_max_error':float(np.max(np.abs(np.asarray(A.sum(1)).ravel()[W>0]-1))),
                  'raw_reconstruction_relative_residual':residual,'normal_equation_relative_residual':equation_residual,
                  'W_units':'actual valid canvas pixel area; D=A.T@1, lambda=.1 literal source'})
    return cached(ep,'inverse_'+role,build)


def role_lse(x,centers,mass,tau=.07):
    from scipy.special import logsumexp
    if not len(centers):raise ValueError('Nonempty legal role prototypes required')
    return tau*(logsumexp(mm(x,centers.T)/tau+np.log(mass)[None],axis=1)-np.log(mass.sum()))


def feature_graph(x,v,hw,k=20,tau=.07):
    ids=np.flatnonzero(v>0);xx=np.asarray(x,float)[ids];vv=np.asarray(v,float)[ids];n=len(ids)
    rows=[];cols=[];values=[];knn=np.empty((n,min(k,max(n-1,0))),int)
    for start in range(0,n,128):
        sim=mm(xx[start:start+128],xx.T);sim[np.arange(len(sim)),np.arange(start,start+len(sim))]=-np.inf
        # Stable exact tie-breaking preserves identical graph between controls.
        order=np.argsort(-sim,axis=1,kind='stable')[:,:knn.shape[1]];knn[start:start+len(sim)]=order
    directed={ (i,int(j)) for i in range(n) for j in knn[i] }
    for i,j in sorted(directed):
        if (j,i) in directed:rows.append(i);cols.append(j);values.append(float(np.exp((xx[i]@xx[j]-1)/tau)))
    inverse=np.full(len(x),-1,int);inverse[ids]=np.arange(n)
    for old in ids:
        y,z=divmod(int(old),hw[1])
        for nxt in ((old+1 if z+1<hw[1] else -1),(old+hw[1] if y+1<hw[0] else -1)):
            if nxt>=0 and inverse[nxt]>=0:
                a,b=int(inverse[old]),int(inverse[nxt]);rows.extend([a,b]);cols.extend([b,a]);values.extend([.05,.05])
    W=sparse.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr();W=sparse.diags(vv)@W@sparse.diags(vv)
    degree=np.asarray(W.sum(1)).ravel();dbar=float(degree.sum()/vv.sum());L=(sparse.diags(degree)-W)/max(dbar,1e-8)
    return ids,vv,W.tocsr(),L.tocsr()


def spectrum(x,v,hw):
    ids,vv,W,L=feature_graph(x,v,hw);n=len(ids);k=min(32,n)
    if n<2:return ids,vv,W,L,np.ones((n,1))/np.sqrt(vv.sum()),np.zeros(1),{}
    invsqrt=1/np.sqrt(vv);S=sparse.diags(invsqrt)@L@sparse.diags(invsqrt)
    start=time.perf_counter()
    if n<=33:values,basis=np.linalg.eigh(S.toarray());values=values[:k];basis=basis[:,:k]
    else:
        values,basis=eigsh(S,k=k,which='SM',v0=np.linspace(1.,2.,n),tol=1e-8,maxiter=10000)
        order=np.argsort(values);values=values[order];basis=basis[:,order]
    constant=np.sqrt(vv/vv.sum());basis[:,0]=constant
    # Retain constant exactly, QR remaining eigenspace only (degeneracy safe).
    remaining=basis[:,1:]-constant[:,None]*mm(constant,basis[:,1:])[None]
    if remaining.shape[1]:remaining=np.linalg.qr(remaining,mode='reduced')[0]
    basis=np.c_[constant,remaining];phi=invsqrt[:,None]*basis
    values=np.einsum('ij,ij->j',basis,S@basis);values[0]=0
    return ids,vv,W,L,phi,values,{'eigensolver_seconds':time.perf_counter()-start,
        'D_orthogonality_error':float(np.linalg.norm(mm(phi.T,vv[:,None]*phi)-np.eye(k))),
        'generalized_eigen_residual':float(np.linalg.norm(L@phi-vv[:,None]*phi*values)/max(np.linalg.norm(L@phi),EPS))}


def anchor_calibration(ep):
    yy,xx=np.indices(ep.r_hw);side=64
    # Physical8-token blocks on64²; small fixture uses exact physical fractions.
    by=(yy*side//ep.r_hw[0])//8;bx=(xx*side//ep.r_hw[1])//8;groups=(by+2*bx)%4
    C=(groups>=2)&(ep.wvalid.reshape(ep.r_hw)>0)
    A=(groups<2)&~binary_dilation(C,np.ones((3,3),bool))&(ep.wvalid.reshape(ep.r_hw)>0)
    A=A.ravel();C=C.ravel();blocks=(by*8+bx).ravel()
    okay=all(len(np.unique(blocks[m]))>=2 and ep.wf[m].sum()>=8 and ep.wb[m].sum()>=8 for m in (A,C))
    return A,C,okay


def balanced_loss(logit,fg,bg):
    return float(np.sum(fg*np.logaddexp(0,-logit))/max(fg.sum(),EPS)+np.sum(bg*np.logaddexp(0,logit))/max(bg.sum(),EPS))


def logistic_newton(x,fg,bg):
    x=np.asarray(x,float);Z=np.c_[x,np.ones(len(x))];theta=np.zeros(Z.shape[1]);penalty=np.r_[np.ones(x.shape[1]),0.]
    weights=fg/max(fg.sum(),EPS)+bg/max(bg.sum(),EPS);target=fg/max(fg.sum(),EPS)
    history=[];converged=False
    for iteration in range(25):
        score=mm(Z,theta);p=expit(score);loss=balanced_loss(score,fg,bg)+.5*np.dot(theta*penalty,theta);history.append(loss)
        grad=mm(Z.T,weights*p-target)+penalty*theta
        if np.linalg.norm(grad)<1e-5:converged=True;break
        H=mm(Z.T,Z*(weights*p*(1-p))[:,None])+np.diag(penalty)+1e-6*np.eye(len(theta));direction=np.linalg.solve(H,grad)
        step=1.;accepted=False
        for _ in range(20):
            new=theta-step*direction;newloss=balanced_loss(mm(Z,new),fg,bg)+.5*np.dot(new*penalty,new)
            if newloss<=loss-1e-4*step*np.dot(grad,direction):theta=new;accepted=True;break
            step*=.5
        if not accepted:break
    return theta,dict(Newton_iterations=iteration+1,converged=converged,objective_history=history)


def standardize(r,q,valid):
    center=np.median(r[valid],axis=0);scale=np.maximum(np.median(np.abs(r[valid]-center),axis=0),1e-3)
    return np.clip((r-center)/scale,-5,5),np.clip((q-center)/scale,-5,5),center,scale


def residual_readout(ep,re,qe,rs,qs,A,C):
    re,qe,ec,es=standardize(re,qe,ep.wvalid>0);rs,qs,sc,ss=standardize(rs,qs,ep.wvalid>0)
    be,ei=logistic_newton(re[A],ep.wf[A],ep.wb[A]);bs,si=logistic_newton(rs[A],ep.wf[A],ep.wb[A])
    enh=balanced_loss(mm(np.c_[re[C],np.ones(C.sum())],be),ep.wf[C],ep.wb[C])
    sim=balanced_loss(mm(np.c_[rs[C],np.ones(C.sum())],bs),ep.wf[C],ep.wb[C]);active=enh<=sim-.01
    info=dict(calibration_enhanced_log_loss=enh,calibration_simple_log_loss=sim,active=bool(active),
        anchor_tokens=int(A.sum()),calibration_tokens=int(C.sum()),anchor_fit_enhanced=ei,anchor_fit_simple=si,
        semantic_descriptors_reselected_after_gate=False,enhanced_center=ec.tolist(),enhanced_MAD=es.tolist(),simple_center=sc.tolist(),simple_MAD=ss.tolist())
    if not active:return np.zeros(len(ep.q)),info
    valid=ep.wvalid>0;be,ei=logistic_newton(re[valid],ep.wf[valid],ep.wb[valid]);bs,si=logistic_newton(rs[valid],ep.wf[valid],ep.wb[valid])
    d=mm(np.c_[qe,np.ones(len(qe))],be)-mm(np.c_[qs,np.ones(len(qs))],bs)
    info.update(full_reference_fit_enhanced=ei,full_reference_fit_simple=si)
    return .20*np.tanh(d/2),info


def bounded_fusion(b,residual):
    """Protect source's strict immutable bands from rounded tanh saturation."""
    b=np.asarray(b,float);z=b-.5+np.asarray(residual,float);tiny=np.finfo(np.float32).eps
    z[b<=.3]=np.minimum(z[b<=.3],-tiny);z[b>=.7]=np.maximum(z[b>=.7],tiny)
    return z


def baseline_b(ep):
    try:value=require_artifact(ep,'mean.continuous')
    except ArtifactUnavailable:
        operator=require_artifact(ep,'pro30_B_operator')
        output=operator(ep,reference=None,query=None,apd='native',fixed_native_gate=True)
        if not isinstance(output,tuple) or len(output)!=2 or output[1].get('complete_FoRIS_MEAN16') is not True:
            raise ArtifactUnavailable('Actual full original B and execution binding required')
        value=output[0]
    if isinstance(value,dict):value=value['continuous']
    b=np.asarray(value,float)
    if b.size!=len(ep.q) or not np.isfinite(b).all():raise ValueError('Complete same-grid original FoRIS+MEAN continuous field required')
    return b.ravel()


def area_sample(ids,weights,maxpoints=64):
    ids=np.asarray(ids,int);weights=np.asarray(weights,float);positive=weights>0;ids=ids[positive];weights=weights[positive]
    if not len(ids):return ids,weights
    if len(ids)<=maxpoints:return ids,weights/weights.sum()
    cut=(np.arange(maxpoints)+.5)*weights.sum()/maxpoints
    selected=np.searchsorted(np.cumsum(weights),cut,side='left');indices,counts=np.unique(selected,return_counts=True)
    return ids[indices],counts/maxpoints


def ward_tree(ep):
    """Exact FP64 spatial Ward heap, frozen128 connected leaves + merge tree."""
    def build():
        # Consume the structural owner's common tree so M11--M18 use exactly
        # the same connected regions and native-index tie convention.
        from .structures_09_16 import tree as common_tree
        start=time.perf_counter();nodes,leaves,roots=common_tree(ep,role='q',target=128)
        return dict(members=[node['support'] for node in nodes],mass=np.asarray([node['mass'] for node in nodes]),
            centers=np.asarray([node['mean'] for node in nodes]),children={i:tuple(node['children']) for i,node in enumerate(nodes) if node['children']},
            leaves=set(leaves),roots=list(roots),nodes=list(range(len(nodes))),seconds=time.perf_counter()-start)
    return cached(ep,'ward128',build)
