"""Pixel and source-only numerical kernels for supplied E226--E250.

These are implementation kernels, not a substitute for the supplied methods or
their MEAN host.  No model is loaded on import and no query labels are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage, sparse
from scipy.special import expit, logsumexp

EPS = 1e-6


def native_to_original(field, shape, geometry):
    """One physical half-pixel bilinear sampling, including recorded padding."""
    from ics.methods.direct_dino_features import sample_grid
    field = np.asarray(field, float)
    if not geometry:
        geometry = dict(view_side=field.shape[0], resized_hw=field.shape,
                        padding_top_left=(0, 0))
    side = float(geometry['view_side'])
    sh, sw = geometry['resized_hw']; oy, ox = geometry['padding_top_left']
    yy = (oy + (np.arange(shape[0]) + .5) * sh / shape[0]) * field.shape[0] / side - .5
    xx = (ox + (np.arange(shape[1]) + .5) * sw / shape[1]) * field.shape[1] / side - .5
    x, y = np.meshgrid(xx, yy)
    return sample_grid(field, y, x)


def pixel_tokens(shape, hw, geometry):
    """Containing physical patch, not an interpolated identity or GT gate."""
    if not geometry:
        geometry = dict(view_side=hw[0], resized_hw=hw, padding_top_left=(0, 0))
    side = float(geometry['view_side'])
    sh, sw = geometry['resized_hw']; oy, ox = geometry['padding_top_left']
    yy = np.clip(((oy + (np.arange(shape[0]) + .5) * sh / shape[0]) * hw[0] / side).astype(int), 0, hw[0]-1)
    xx = np.clip(((ox + (np.arange(shape[1]) + .5) * sw / shape[1]) * hw[1] / side).astype(int), 0, hw[1]-1)
    return (yy[:, None] * hw[1] + xx[None, :]).ravel()


def footprint(shape, hw, geometry):
    """Exact cell-overlap area averaging from original pixels to native patches.

    The rows average only the real-image portion of partially padded patches;
    physical valid fractions remain separate weights.  A pixel straddling a
    patch edge contributes its exact intersection area to both rows.
    """
    if not geometry:
        geometry = dict(view_side=hw[0], resized_hw=hw, padding_top_left=(0, 0))
    side = float(geometry['view_side'])
    sh, sw = geometry['resized_hw']; oy, ox = geometry['padding_top_left']
    ay, ax = [], []
    for nh, out, resized, offset, store in ((hw[0],shape[0],sh,oy,ay), (hw[1],shape[1],sw,ox,ax)):
        for k in range(nh):
            lo = max(0., (k*side/nh-offset)*out/resized)
            hi = min(float(out), ((k+1)*side/nh-offset)*out/resized)
            if hi <= lo:
                store.append((np.empty(0,int), np.empty(0))); continue
            p = np.arange(int(np.floor(lo)), int(np.ceil(hi)))
            a = np.maximum(0., np.minimum(p+1.,hi)-np.maximum(p,lo))
            store.append((p, a/(hi-lo)))
    rows, cols, values = [], [], []
    for y,(iy,wy) in enumerate(ay):
        for x,(ix,wx) in enumerate(ax):
            if not len(iy) or not len(ix): continue
            ids = (iy[:,None]*shape[1]+ix[None,:]).ravel()
            rows.append(np.full(len(ids),y*hw[1]+x,int)); cols.append(ids)
            values.append((wy[:,None]*wx[None,:]).ravel())
    if not rows: return sparse.csr_matrix((np.prod(hw), np.prod(shape)))
    return sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows),np.concatenate(cols))),
                             shape=(np.prod(hw),np.prod(shape)))


def pixel_phi(rgb):
    """Fixed E242 RGB chromaticity, luminance and 3x3 gradients."""
    c = np.asarray(rgb,float)/255.
    l = c.mean(-1)
    chroma = c / np.maximum(c.sum(-1,keepdims=True),EPS)
    gx = ndimage.sobel(l,1,mode='nearest')/8.
    gy = ndimage.sobel(l,0,mode='nearest')/8.
    return np.concatenate((chroma,l[...,None],gx[...,None],gy[...,None],
                           np.hypot(gx,gy)[...,None]),axis=-1).reshape(-1,7)


def texture_phi(rgb):
    """Fixed RGB/local texture representation; never a newly fitted backbone."""
    c = np.asarray(rgb,float)/255.
    l = c.mean(-1)
    avg = ndimage.uniform_filter(l,3,mode='nearest')
    var = np.maximum(ndimage.uniform_filter(l*l,3,mode='nearest')-avg*avg,0)
    gx = ndimage.sobel(l,1,mode='nearest')/8.
    gy = ndimage.sobel(l,0,mode='nearest')/8.
    return np.concatenate((c,avg[...,None],np.sqrt(var)[...,None],
                           gx[...,None],gy[...,None]),axis=-1).reshape(-1,7)


def grid_edges(shape):
    ids = np.arange(np.prod(shape)).reshape(shape)
    return np.concatenate((np.stack((ids[:-1].ravel(),ids[1:].ravel()),1),
                           np.stack((ids[:,:-1].ravel(),ids[:,1:].ravel()),1)))


def rgb_edges(rgb):
    e = grid_edges(rgb.shape[:2]); c = np.asarray(rgb,float).reshape(-1,3)/255.
    d = np.mean((c[e[:,0]]-c[e[:,1]])**2,axis=1)
    scale = max(float(np.median(d)),1e-4)
    return e,np.exp(-d/scale)/8.


def normalize_evidence(q, source, valid=None):
    a = np.asarray(source,float)
    if valid is not None: a = a[np.asarray(valid,bool)]
    scale = max(float(np.median(np.abs(a))) if len(a) else 0.,EPS)
    return np.asarray(q,float)/scale,scale


def balanced_weights(labels, valid, token_ids=None):
    labels,valid = np.asarray(labels,bool),np.asarray(valid,bool)
    w=np.zeros(len(labels),float)
    base=np.ones(len(labels))
    if token_ids is not None:
        token_ids=np.asarray(token_ids,int)
        counts=np.bincount(token_ids,weights=valid)
        base=1./np.maximum(counts[token_ids],1)
    for cls in (False,True):
        ids=valid&(labels==cls)
        if not ids.any(): return None
        w[ids]=.5*base[ids]/base[ids].sum()
    return w


@dataclass
class PixelLogistic:
    """Linear colour + DINO head without materializing P by 1024 features."""
    a: np.ndarray
    b: np.ndarray
    bias: float
    p_mean: np.ndarray
    p_scale: np.ndarray
    z_mean: np.ndarray
    z_scale: np.ndarray
    info: dict

    def predict(self,phi,z,token_ids):
        colour=(phi-self.p_mean)/self.p_scale
        native=((z-self.z_mean)/self.z_scale)@self.b
        return expit(colour@self.a+native[token_ids]+self.bias)


def fit_pixel_logistic(phi,z,token_ids,labels,valid,max_steps=100):
    """C_E class-balanced L2 logistic, reg=1 and train-only scales.

    A conservative Lipschitz step uses the weighted feature Frobenius norm.
    Semantic gradients are grouped by physical token. The intercept is not
    penalized; it is the only additional scalar. All 100 steps are counted.
    """
    w=balanced_weights(labels,valid,token_ids)
    if w is None:return None
    phi,z=np.asarray(phi,float),np.asarray(z,float)
    labels=np.asarray(labels,float); token_ids=np.asarray(token_ids,int)
    pm=w@phi; ps=np.sqrt(np.maximum(w@((phi-pm)**2),EPS*EPS))
    tw=np.bincount(token_ids,weights=w,minlength=len(z))
    zm=tw@z; zs=np.sqrt(np.maximum(tw@((z-zm)**2),EPS*EPS))
    p=(phi-pm)/ps; zz=(z-zm)/zs
    lips=.25*(float(w@np.sum(p*p,1))+float(tw@np.sum(zz*zz,1))+1.)+1.
    step=1./lips
    a=np.zeros(p.shape[1]);b=np.zeros(z.shape[1]);bias=0.
    losses=[]
    for _ in range(max_steps):
        logits=p@a+(zz@b)[token_ids]+bias
        diff=w*(expit(logits)-labels)
        a-=step*(p.T@diff+a)
        b-=step*(zz.T@np.bincount(token_ids,weights=diff,minlength=len(z))+b)
        bias-=step*diff.sum()
        loss=float(w@(np.logaddexp(0,logits)-labels*logits)+.5*(a@a+b@b))
        losses.append(loss)
    logits=p@a+(zz@b)[token_ids]+bias
    loss=float(w@(np.logaddexp(0,logits)-labels*logits)+.5*(a@a+b@b))
    return PixelLogistic(a,b,float(bias),pm,ps,zm,zs,
                         dict(steps=max_steps,lipschitz_bound=lips,step=step,
                              class_balance=True,train_pixels=int(np.count_nonzero(w)),
                              token_balanced=True,final_objective=loss))


@dataclass
class DiagGMM:
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray

    def log_density(self,x):
        x=np.asarray(x,float)
        d=((x[:,None]-self.means)**2/self.variances).sum(2)
        norm=np.log(2*np.pi*self.variances).sum(1)
        return logsumexp(np.log(self.weights)[None,:]-.5*(d+norm),axis=1)


def fit_gmm(x,k=2,steps=20):
    """Deterministic diagonal mixture with fixed, disclosed numerical floor."""
    x=np.asarray(x,float)
    if not len(x):return None
    k=min(k,len(x)); centers=[0]
    for _ in range(1,k):
        dist=((x[:,None]-x[centers])**2).sum(2).min(1)
        centers.append(int(np.argmax(dist)))
    means=x[centers].copy();globalvar=np.maximum(x.var(0),1e-4)
    variances=np.tile(globalvar,(k,1));weights=np.full(k,1./k)
    for _ in range(steps):
        d=((x[:,None]-means)**2/variances).sum(2)
        scores=np.log(np.maximum(weights,EPS))[None,:]-.5*(d+np.log(2*np.pi*variances).sum(1))
        resp=np.exp(scores-logsumexp(scores,axis=1,keepdims=True))
        mass=resp.sum(0)
        means=(resp.T@x)/np.maximum(mass[:,None],EPS)
        variances=np.maximum((resp.T@(x*x))/np.maximum(mass[:,None],EPS)-means*means,1e-4)
        weights=np.maximum(mass,EPS);weights/=weights.sum()
    return DiagGMM(means,variances,weights)


def diagonal_colour_evidence(qrgb,rrgb,rlabels,rvalid,k=2):
    r=texture_phi(rrgb);q=texture_phi(qrgb)
    f=fit_gmm(r[rvalid&rlabels],k);b=fit_gmm(r[rvalid&~rlabels],k)
    if f is None or b is None:return None
    rs=f.log_density(r)-b.log_density(r);qs=f.log_density(q)-b.log_density(q)
    qs,scale=normalize_evidence(qs,rs,rvalid)
    return qs,dict(scale=scale,fg_components=len(f.weights),bg_components=len(b.weights))


def simplex(v):
    """Euclidean projection of rows onto the probability simplex."""
    a=np.sort(v,axis=1)[:,::-1]
    css=np.cumsum(a,axis=1)-1
    j=np.arange(1,v.shape[1]+1)
    count=np.sum(a-css/j>0,axis=1)
    theta=css[np.arange(len(v)),np.maximum(count-1,0)]/np.maximum(count,1)
    return np.maximum(v-theta[:,None],0)


def affine_colour_weights(rgb,semantic,offsets=None,regularization=1e-3,max_steps=2000):
    """E236 simplex-constrained affine reconstruction, bounded active-set audit.

    Each 5x5 window is solved in chunks; neighbours carrying a source-calibrated
    strong opposite semantic sign are excluded. Returned W is sparse and row
    stochastic. 'semantic' is a signed anchor map, not a label or unknown GT.
    """
    shape=rgb.shape[:2];p=np.prod(shape)
    if offsets is None:offsets=[(y,x) for y in range(-2,3) for x in range(-2,3)]
    c=np.asarray(rgb,float).reshape(-1,3)/255.;anchor=np.asarray(semantic,int).ravel()
    ids=np.arange(p).reshape(shape)
    neighbours=[]
    for y,x in offsets:
        yy=np.clip(np.arange(shape[0])+y,0,shape[0]-1)
        xx=np.clip(np.arange(shape[1])+x,0,shape[1]-1)
        neighbours.append(ids[yy[:,None],xx[None,:]].ravel())
    ni=np.stack(neighbours,1);weights=np.empty(ni.shape,float)
    iterationmax=0;resmax=0.;gapmax=0.;failed=0
    for start in range(0,p,2048):
        end=min(start+2048,p);nb=ni[start:end]
        diff=c[nb]-c[start:end,None,:]
        allowed=(anchor[start:end,None]*anchor[nb]>=0)
        # The center is always available, including strong anchors.
        w=allowed/np.maximum(allowed.sum(1,keepdims=True),1)
        free=allowed.copy();converged=np.zeros(len(w),bool)
        for iteration in range(max_steps):
            # Equality-constrained minimizer on the current free face.  The
            # rank-three Woodbury solve is algebraically the 25-variable QP,
            # not a Gaussian-weight substitute for its affine objective.
            df=diff*free[...,None]
            gram=np.einsum('bkd,bke->bde',df,df)+regularization*np.eye(3)[None]
            direction=np.linalg.solve(gram,df.sum(1)[...,None])[...,0]
            candidate=free*(1-np.einsum('bkd,bd->bk',diff,direction))
            candidate/=np.maximum(candidate.sum(1,keepdims=True),EPS)
            negative=(candidate< -1e-12)&free
            ratios=np.where(negative,w/np.maximum(w-candidate,EPS),np.inf)
            alpha=np.minimum(ratios.min(1),1.)
            nxt=w+alpha[:,None]*(candidate-w)
            nxt=np.maximum(nxt,0);nxt/=np.maximum(nxt.sum(1,keepdims=True),EPS)
            on_boundary=(nxt<1e-12)&negative
            free&=~on_boundary
            w=nxt
            grad=2*(np.einsum('bkd,bd->bk',diff,np.einsum('bk,bkd->bd',w,diff))+regularization*w)
            multiplier=np.sum(grad*free,1)/np.maximum(free.sum(1),1)
            reduced=grad-multiplier[:,None]
            bad=np.where(allowed&~free,reduced,np.inf)
            add=np.argmin(bad,1);add_needed=(bad.min(1)<-1e-10)&(alpha>=1-1e-12)
            free[np.flatnonzero(add_needed),add[add_needed]]=True
            # Projected-gradient norm is a diagnostic. The independent strong
            # convex minorant below is the convergence certificate.
            projected=simplex(np.where(allowed,w-grad,-1e30))
            residual=np.max(np.abs(projected-w),axis=1)
            # Strong-convex tangent minorant yields a feasible primal lower
            # bound, not a claim based on a small optimizer step alone.
            g=2*(np.einsum('bkd,bd->bk',diff,np.einsum('bk,bkd->bd',w,diff))+regularization*w)
            minor=simplex(np.where(allowed,w-g/(2*regularization),-1e30))
            change=minor-w
            gap=np.maximum(0.,-np.sum(g*change,1)-regularization*np.sum(change*change,1))
            objective=np.sum(np.einsum('bk,bkd->bd',w,diff)**2,1)+regularization*np.sum(w*w,1)
            converged=gap<=1e-5*np.maximum(objective,1e-12)
            if converged.all():break
        failed+=int(np.sum(~converged));iterationmax=max(iterationmax,iteration+1)
        resmax=max(resmax,float(residual.max(initial=0)))
        gapmax=max(gapmax,float(gap.max(initial=0)))
        weights[start:end]=w
    W=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(p),len(offsets)),ni.ravel())),shape=(p,p))
    W.sum_duplicates()
    return W,dict(max_iterations=iterationmax,max_projected_gradient=resmax,
                  max_certified_gap=gapmax,unconverged_rows=failed,regularization=regularization,
                  solver='bounded_active_set_rank3_exact_free_face',
                  row_mass_error=float(np.max(np.abs(np.asarray(W.sum(1)).ravel()-1))))


def affine_lift(U,W,max_steps=2000):
    """Box-constrained strictly convex E236 solve, projected-gradient audit."""
    u=np.asarray(U,float).ravel();p=len(u)
    L=sparse.eye(p,format='csr')-W
    # ||L||_2 <= sqrt(||L||_1 ||L||_inf), avoiding a hidden eigen solve.
    norm1=float(np.max(np.asarray(abs(L).sum(0))))
    norminf=float(np.max(np.asarray(abs(L).sum(1))))
    lips=2*(1+norm1*norminf);step=1/lips
    v=u.copy();accelerated=v.copy();t=1.
    for iteration in range(max_steps):
        grad=2*(accelerated-u+L.T@(L@accelerated))
        nxt=np.clip(accelerated-step*grad,0,1)
        nt=(1+np.sqrt(1+4*t*t))/2
        accelerated=nxt+(t-1)/nt*(nxt-v);v=nxt;t=nt
        g=2*(v-u+L.T@(L@v))
        res=float(np.max(np.abs(v-np.clip(v-step*g,0,1))/step))
        pdual=2*(L@v);vv=-(L.T@pdual);maximizer=np.clip(u+vv/2,0,1)
        dual=float(-vv@maximizer+np.sum((maximizer-u)**2)-(pdual@pdual)/4)
        objective=float(np.sum((v-u)**2)+np.sum((L@v)**2))
        gap=max(0.,objective-dual);tol=1e-5*max(1e-12,abs(objective),abs(dual))
        if gap<=tol:break
    objective=float(np.sum((v-u)**2)+np.sum((L@v)**2))
    return v.reshape(U.shape),dict(iterations=iteration+1,projected_gradient=res,
                                   converged=bool(gap<=tol),objective=objective,lipschitz=lips,
                                   dual=dual,gap=gap,gap_tolerance=tol)


def isotonic_fit(x,y,weight=None):
    """Weighted pool-adjacent-violators; duplicate abscissas share one fit."""
    x,y=np.asarray(x,float),np.asarray(y,float)
    if weight is None:weight=np.ones(len(x))
    order=np.argsort(x,kind='stable');x=x[order];y=y[order];w=np.asarray(weight,float)[order]
    unique,idx=np.unique(x,return_inverse=True)
    mass=np.bincount(idx,weights=w);sums=np.bincount(idx,weights=w*y)
    blocks=[]
    for j,(m,s) in enumerate(zip(mass,sums)):
        if m<=0:continue
        blocks.append([j,j,m,s])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            right=blocks.pop();left=blocks.pop()
            blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    values=np.zeros(len(unique))
    for lo,hi,m,s in blocks:values[lo:hi+1]=s/m
    return unique,values


def two_ray_residual(x,c,e):
    """Exact two-variable nonnegative LS for E250 dichromatic reflection."""
    x=np.asarray(x,float);c=np.asarray(c,float);e=np.asarray(e,float)
    cc=c@c;ee=e@e;ce=c@e;det=cc*ee-ce*ce
    candidates=[np.zeros_like(x)]
    d=np.maximum(x@c/max(cc,EPS),0);candidates.append(d[:,None]*c)
    s=np.maximum(x@e/max(ee,EPS),0);candidates.append(s[:,None]*e)
    if det>EPS:
        d=(ee*(x@c)-ce*(x@e))/det;s=(cc*(x@e)-ce*(x@c))/det
        valid=(d>=0)&(s>=0)
        both=d[:,None]*c+s[:,None]*e
        candidates.append(np.where(valid[:,None],both,candidates[0]))
    errors=np.stack([np.sum((x-v)**2,1) for v in candidates],1)
    return errors.min(1)


def two_ray_fit(x,c,e):
    """The same exact E250 NNLS faces, retaining the winning coefficients."""
    x=np.asarray(x,float);c=np.asarray(c,float);e=np.asarray(e,float)
    cc=c@c;ee=e@e;ce=c@e;det=cc*ee-ce*ce
    d0=np.maximum(x@c/max(cc,EPS),0);s0=np.maximum(x@e/max(ee,EPS),0)
    ds=[(np.zeros(len(x)),np.zeros(len(x))),(d0,np.zeros(len(x))),(np.zeros(len(x)),s0)]
    if det>EPS:
        d=(ee*(x@c)-ce*(x@e))/det;s=(cc*(x@e)-ce*(x@c))/det
        valid=(d>=0)&(s>=0);ds.append((np.where(valid,d,0),np.where(valid,s,0)))
    errors=np.stack([np.sum((x-d[:,None]*c-s[:,None]*e)**2,1) for d,s in ds],1)
    best=np.argmin(errors,1);rows=np.arange(len(x))
    return np.stack([a for a,b in ds],1)[rows,best],np.stack([b for a,b in ds],1)[rows,best],errors[rows,best]


def direct_u0(q,r,wf,wvalid,chunk=128):
    """C_E U0, including the specified all-reference 16-NN coverage vote."""
    q,r=np.asarray(q,float),np.asarray(r,float)
    wf,wvalid=np.asarray(wf,float),np.asarray(wvalid,float)
    known=wvalid>0;coverage=np.divide(wf,wvalid,out=np.zeros_like(wf),where=known)
    if not known.any() or wf.sum()<=0:return np.zeros(len(q)),dict(status='empty_reference_foreground')
    f=known&(coverage>=.9);b=known&(coverage<=.1)
    out=np.empty(len(q))
    for start in range(0,len(q),chunk):
        sim=q[start:start+chunk]@r[known].T
        if f.any() and b.any():
            s=sim[:,f[known]].max(1)-sim[:,b[known]].max(1)
            out[start:start+chunk]=.5+s/4.
        else:
            k=min(16,sim.shape[1])
            order=np.argpartition(-sim,k-1,axis=1)[:,:k]
            mass=wvalid[known][order]
            out[start:start+chunk]=(coverage[known][order]*mass).sum(1)/np.maximum(mass.sum(1),EPS)
    return out,dict(status='pure_role_max' if f.any() and b.any() else 'weighted_16nn_vote',
                    pure_fg=int(f.sum()),pure_bg=int(b.sum()))


def source_block_predictions(r,hw,wf,wvalid):
    """C_E source predictions: 2x2 holdout plus one-patch training buffer."""
    h,w=hw;yy,xx=np.indices(hw);result=np.full(len(r),np.nan);folds=[]
    for by in range(2):
        for bx in range(2):
            y0,y1=by*h//2,(by+1)*h//2;x0,x1=bx*w//2,(bx+1)*w//2
            test=((yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)).ravel()&(wvalid>0)
            exclusion=((yy>=max(0,y0-1))&(yy<min(h,y1+1))&
                       (xx>=max(0,x0-1))&(xx<min(w,x1+1))).ravel()
            valid=np.where(exclusion,0,wvalid); fg=np.where(exclusion,0,wf)
            prediction,info=direct_u0(r,r,fg,valid)
            result[test]=prediction[test]
            folds.append(dict(block=(by,bx),test_tokens=int(test.sum()),
                              train_tokens=int(np.count_nonzero(valid)),**info))
    return result,folds


def anchor_thresholds(oob,cov,valid):
    f=np.isfinite(oob)&(valid>0)&(cov>=.9)
    b=np.isfinite(oob)&(valid>0)&(cov<=.1)
    if not f.any() or not b.any():return None
    ft=float(np.quantile(oob[f],.1));bt=float(np.quantile(oob[b],.9))
    if ft<=bt:return None
    return ft,bt


def semantic_gate(prob,hw,shape,geometry):
    """E242 gate uses only predicted probability and adjacent predicted roles."""
    field=np.asarray(prob,float).reshape(hw);binary=field>.5
    edge=np.zeros(hw,bool)
    edge[:-1]|=binary[:-1]!=binary[1:];edge[1:]|=binary[:-1]!=binary[1:]
    edge[:,:-1]|=binary[:,:-1]!=binary[:,1:];edge[:,1:]|=binary[:,:-1]!=binary[:,1:]
    token=(np.abs(field-.5)<=.1)|edge
    return token.ravel()[pixel_tokens(shape,hw,geometry)].reshape(shape)


def coverage_tv(U,A,a,sigma,rgb,valid=None,max_steps=2000):
    """E245 known-area strictly convex inverse, CP with primal/dual gap.

    sigma is in fractional coverage units; a is never an inferred query area.
    Valid fractions weight observations of partially padded footprints.
    """
    u0=np.asarray(U,float).ravel();a=np.asarray(a,float)
    if valid is None:valid=np.ones(len(a))
    v=np.sqrt(np.asarray(valid,float));B=sparse.diags(v)@A;target=v*a
    edges,cap=rgb_edges(rgb);p=len(u0);m=len(edges)
    D=sparse.csr_matrix((np.tile((-1.,1.),m),(np.repeat(np.arange(m),2),edges.ravel())),shape=(m,p))
    # ||B||² <= ||B||_1 ||B||_inf and grid differences have norm²<=8.
    norm=float(np.max(np.asarray(abs(B).sum(0))))*float(np.max(np.asarray(abs(B).sum(1))))+8
    step=.99/np.sqrt(max(norm,EPS));u=np.clip(u0,0,1);bar=u.copy()
    pa=np.zeros(B.shape[0]);pt=np.zeros(m);gap=np.inf
    for iteration in range(max_steps):
        pa=(pa+step*(B@bar-target))/(1+step*sigma*sigma/2)
        pt=np.clip(pt+step*(D@bar),-cap,cap)
        old=u.copy();u=np.clip((u-step*(B.T@pa+D.T@pt)+2*step*u0)/(1+2*step),0,1)
        bar=2*u-old
        if iteration%10==0 or iteration==max_steps-1:
            residual=B@u-target
            primal=float(residual@residual/(sigma*sigma)+np.sum((u-u0)**2)+cap@np.abs(D@u))
            vv=-(B.T@pa+D.T@pt)
            maximizer=np.clip(u0+vv/2,0,1)
            conjugate=float(vv@maximizer-np.sum((maximizer-u0)**2))
            dual=float(-conjugate-target@pa-(sigma*sigma/4)*(pa@pa))
            gap=max(0.,primal-dual)
            tolerance=1e-5*max(1e-12,abs(primal),abs(dual))
            if gap<=tolerance:break
    return u.reshape(U.shape),dict(iterations=iteration+1,primal=primal,dual=dual,
                                   gap=gap,gap_tolerance=tolerance,converged=bool(gap<=tolerance),
                                   sigma_coverage=float(sigma),area_row_mass_error=float(np.max(
                                       np.abs(np.asarray(A.sum(1)).ravel()[np.asarray(valid)>0]-1),initial=0)))
