"""Exact numerical operators and source-only protocol for supplied E201--E225.

No query annotations, learned external model, or encoder approximation enter here.
Fixed unspecified choices are recorded in the companion immutable recipes.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg
from scipy.special import expit, logsumexp, gammaln
from . import e_helpers_226_250 as H
from .common import ArtifactUnavailable, host_baseline, artifact, validate
from .e_float_cut import exact_potts_cut

EPS=1e-6
DIRECT={'E207','E216','E218'}

@dataclass
class Problem:
    ep: object
    U: np.ndarray
    mask0: np.ndarray
    u0: np.ndarray
    r0: np.ndarray
    oob: np.ndarray
    qtokens: np.ndarray
    rtokens: np.ndarray
    labels: np.ndarray
    rvalid: np.ndarray
    anchors: np.ndarray
    thresholds: tuple | None
    source_folds: list
    binding: dict


def prepare(ep, method):
    validate(ep)
    if ep.q_rgb is None or ep.r_rgb is None or ep.reference_mask is None:
        raise ArtifactUnavailable('E RGB methods require actual original R/Q RGB and complete original MR')
    qr,rr=np.asarray(ep.q_rgb),np.asarray(ep.r_rgb)
    if qr.shape!=(*ep.original_shape,3) or rr.ndim!=3 or rr.shape[2]!=3 or np.asarray(ep.reference_mask).shape!=rr.shape[:2]:
        raise ValueError('Actual physical RGB and MR shapes must match recorded original geometry')
    if not np.isfinite(qr).all() or not np.isfinite(rr).all():raise ValueError('Finite original RGB required')
    u0,_=H.direct_u0(ep.q,ep.r,ep.wf,ep.wvalid)
    r0,_=H.direct_u0(ep.r,ep.r,ep.wf,ep.wvalid)
    oob,folds=H.source_block_predictions(ep.r,ep.r_hw,ep.wf,ep.wvalid)
    qt=H.pixel_tokens(ep.original_shape,ep.q_hw,ep.query_geometry)
    rt=H.pixel_tokens(rr.shape[:2],ep.r_hw,ep.reference_geometry)
    cov=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0)
    thresholds=H.anchor_thresholds(oob,cov,ep.wvalid)
    uq=H.native_to_original(u0.reshape(ep.q_hw),ep.original_shape,ep.query_geometry)
    if method in DIRECT:
        U=uq;mask0=U>.5;binding=dict(kind='C_E_U0_pure_role_max_or_16NN',forwards=0)
    else:
        host=host_baseline(ep)
        U=np.asarray(artifact(ep,'mean_original_field'),float)
        if U.shape!=tuple(ep.original_shape) or not np.isfinite(U).all():raise ValueError('Bound original MEAN continuous field required')
        mask0=host.mask_original;binding=dict(producer=host.info['host_producer'],renderer=host.info['host_renderer'])
    if thresholds is None:anchor=np.zeros(ep.original_shape,np.int8)
    else:
        ft,bt=thresholds
        anchor=np.where((uq>=ft)&(U>.5),1,np.where((uq<=bt)&(U<=.5),-1,0)).astype(np.int8)
    return Problem(ep,U,mask0,u0,r0,oob,qt,rt,np.asarray(ep.reference_mask,bool).ravel(),
                   ep.wvalid[rt]>0,anchor,thresholds,folds,binding)


def logit(u):
    u=np.clip(np.asarray(u,float),EPS,1-EPS);return np.log(u/(1-u))


def source_scale(evidence,valid=None):
    e=np.asarray(evidence,float)
    if valid is not None:e=e[np.asarray(valid,bool)]
    return max(float(np.median(np.abs(e))) if len(e) else 0.,EPS)


def cut(U, rgb, evidence=None, *, extra_edges=None, extra_capacity=None, directed=None, edge_capacity=None):
    """Binary submodular cut; arbitrary unary difference is exact up to a constant."""
    logits=logit(U).ravel()
    if evidence is not None:logits+=np.asarray(evidence,float).ravel()
    edges,cap=H.rgb_edges(rgb)
    if edge_capacity is not None:cap=np.asarray(edge_capacity,float)
    if extra_edges is not None:
        edges=np.vstack((edges,np.asarray(extra_edges,int).reshape(-1,2)));cap=np.r_[cap,extra_capacity]
    projection=0.
    if directed is not None:
        de,pot=directed;de=np.asarray(de,int).reshape(-1,2);pot=np.asarray(pot,float).reshape(-1,4).copy()
        # V00,V01,V10,V11. Euclidean projection on the submodular halfspace.
        violation=np.maximum(pot[:,0]+pot[:,3]-pot[:,1]-pot[:,2],0)/4
        pot[:,0]-=violation;pot[:,3]-=violation;pot[:,1]+=violation;pot[:,2]+=violation
        projection=float(np.sum(violation*4))
        w=(pot[:,1]+pot[:,2]-pot[:,0]-pot[:,3])/2
        a=pot[:,2]-pot[:,0]-w;b=pot[:,1]-pot[:,0]-w
        np.add.at(logits,de[:,0],-a);np.add.at(logits,de[:,1],-b)
        edges=np.vstack((edges,de));cap=np.r_[cap,np.maximum(w,0)]
    labels,cert=exact_potts_cut(logits,edges,cap)
    cert.update(submodular_projection_l1=projection)
    return labels.reshape(U.shape),cert


def laplacian(edges,cap,n):
    e=np.asarray(edges,int);w=np.asarray(cap,float)
    D=sparse.csr_matrix((np.tile((-1.,1.),len(e)),(np.repeat(np.arange(len(e)),2),e.ravel())),shape=(len(e),n))
    return D.T@sparse.diags(w)@D


def quadratic_box(U,L,anchor=None,max_steps=2000):
    """min ||v-U||²+v'L v+finite anchor loss, with a strong-convex bound.

    For PSD L, A>=I. A tangent plus ||delta||² is a global lower bound;
    minimizing it over the box gives a feasible primal/lower-bound gap.
    """
    u=np.asarray(U,float).ravel();n=len(u);a=np.zeros(n) if anchor is None else np.asarray(anchor).ravel()
    weight=(a!=0).astype(float);target=(a>0).astype(float)
    A=sparse.eye(n,format='csr')+L+sparse.diags(weight);rhs=u+weight*target
    norm=float(np.max(np.asarray(abs(A).sum(1))));step=1/(2*max(norm,EPS))
    const=float(u@u+weight@(target*target));cg_iterations=0
    def count(_):
        nonlocal cg_iterations
        cg_iterations+=1
    x,status=linalg.cg(A,rhs,x0=np.clip(u,0,1),rtol=1e-10,atol=1e-12,maxiter=max_steps,callback=count)
    x=np.clip(x,0,1);bar=x.copy();t=1.
    for it in range(max_steps):
        g=2*(A@x-rhs);v=np.clip(x-g/2,0,1);d=v-x
        primal=float(x@(A@x)-2*rhs@x+const)
        gap=max(0.,float(-g@d-d@d));dual=primal-gap
        tol=1e-5*max(1e-12,abs(primal),abs(dual))
        if gap<=tol:break
        nxt=np.clip(bar-step*2*(A@bar-rhs),0,1)
        nt=(1+np.sqrt(1+4*t*t))/2;bar=nxt+(t-1)/nt*(nxt-x);x=nxt;t=nt
    return x.reshape(U.shape),dict(converged=bool(gap<=tol),iterations=it+1,cg_iterations=cg_iterations,
              cg_status=int(status),primal=primal,dual_lower_bound=dual,gap=gap,gap_tolerance=tol,
              finite_anchor_weight=1.,solver='CG_then_box_FISTA_strong_convex_lower_bound')


def matting_laplacian(rgb,epsilon=1e-5):
    """Closed-form RGB local linear matting Laplacian, actual 3x3 windows.

    L_ij=delta_ij-[1+(ci-mu)'(cov+epsilon/n I)^-1(cj-mu)]/n.
    Boundary windows are clipped, never made up by texture or a Potts graph.
    """
    c=np.asarray(rgb,float)/255.;h,w=c.shape[:2];n=h*w;ids=np.arange(n).reshape(h,w)
    rows=[];cols=[];data=[]
    # Interior blocks are vectorized; boundary windows have their true size.
    if h>=3 and w>=3:
        centers=ids[1:-1,1:-1].ravel()
        offsets=np.array([dy*w+dx for dy in (-1,0,1) for dx in (-1,0,1)])
        for start in range(0,len(centers),4096):
            ni=centers[start:start+4096,None]+offsets;v=c.reshape(-1,3)[ni];d=v-v.mean(1,keepdims=True)
            cov=np.einsum('bni,bnj->bij',d,d)/9+epsilon/9*np.eye(3)
            inv=np.linalg.inv(cov)
            local=np.eye(9)[None]-(1+np.einsum('bni,bij,bmj->bnm',d,inv,d))/9
            rows.append(np.repeat(ni,9,axis=1).ravel());cols.append(np.tile(ni,(1,9)).ravel());data.append(local.ravel())
    for y,x in np.ndindex(h,w):
        if 0<y<h-1 and 0<x<w-1:continue
        ni=ids[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)].ravel();v=c.reshape(-1,3)[ni];k=len(ni)
        d=v-v.mean(0);cov=d.T@d/k+epsilon/k*np.eye(3)
        local=np.eye(k)-(1+d@np.linalg.inv(cov)@d.T)/k
        rows.append(np.repeat(ni,k));cols.append(np.tile(ni,k));data.append(local.ravel())
    out=sparse.csr_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n))
    out.sum_duplicates();return (out+out.T)*.5


def source_folds(problem,support_pixels=4):
    """Physical reference blocks plus support buffer; labels are never copied in."""
    ep=problem.ep;h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw)
    # One complete work patch already covers interpolation; enlarge when the
    # stated RGB support exceeds one physical reference patch.
    rh,rw=ep.r_rgb.shape[:2];g=ep.reference_geometry
    patch_y=rh/h if not g else rh*g['view_side']/(h*g['resized_hw'][0])
    patch_x=rw/w if not g else rw*g['view_side']/(w*g['resized_hw'][1])
    bybuf=max(1,int(np.ceil(support_pixels/max(patch_y,EPS))));bxbuf=max(1,int(np.ceil(support_pixels/max(patch_x,EPS))))
    for by,bx in np.ndindex(2,2):
        y0,y1=by*h//2,(by+1)*h//2;x0,x1=bx*w//2,(bx+1)*w//2
        test=((yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)).ravel()&(ep.wvalid>0)
        exclusion=((yy>=max(0,y0-bybuf))&(yy<min(h,y1+bybuf))&(xx>=max(0,x0-bxbuf))&(xx<min(w,x1+bxbuf))).ravel()
        train=(ep.wvalid>0)&~exclusion
        yield train[problem.rtokens]&problem.rvalid,test[problem.rtokens]&problem.rvalid,train,test


def coverage_mapping(problem):
    """Every fold rebuilds U0 bank AND scalar isotonic coverage mapping."""
    ep=problem.ep;cov=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0)
    mass=error=0.;folds=[]
    for _,_,train,test in source_folds(problem):
        if train.sum()<2 or np.unique(cov[train]).size<2 or not test.any():continue
        prob,_=H.direct_u0(ep.r,ep.r,np.where(train,ep.wf,0),np.where(train,ep.wvalid,0))
        x,y=H.isotonic_fit(prob[train],cov[train],ep.wvalid[train]);d=np.interp(prob[test],x,y)-cov[test]
        error+=float(ep.wvalid[test]@(d*d));mass+=float(ep.wvalid[test].sum());folds.append(dict(train=int(train.sum()),test=int(test.sum())))
    known=ep.wvalid>0
    if not folds or np.unique(cov[known]).size<2:return None
    x,y=H.isotonic_fit(problem.r0[known],cov[known],ep.wvalid[known])
    return np.interp(problem.u0,x,y),max(.05,np.sqrt(error/max(mass,EPS))),dict(folds=folds,source_rmse=np.sqrt(error/max(mass,EPS)))


def gmm_pair(qphi,rphi,labels,valid,k=2):
    fg=H.fit_gmm(np.asarray(rphi)[valid&labels],k);bg=H.fit_gmm(np.asarray(rphi)[valid&~labels],k)
    if fg is None or bg is None:return None
    r=fg.log_density(rphi)-bg.log_density(rphi);q=fg.log_density(qphi)-bg.log_density(qphi)
    scale=source_scale(r,valid);return q/scale,scale,fg,bg


def rgb_tree(rgb,extra=None):
    """Full pixel binary RGB hierarchy by stable single-linkage Kruskal.

    Edge priorities use original RGB gradient and adjacent region differences;
    after merging the recorded boundary includes current mean-colour contrast.
    All pixels remain leaves. No superclass, GT or learned edge detector.
    """
    c=np.asarray(rgb,float).reshape(-1,3)/255.;n=len(c);edges,_=H.rgb_edges(rgb)
    score=np.mean((c[edges[:,0]]-c[edges[:,1]])**2,1)
    if extra is not None:score+=np.asarray(extra,float)
    order=np.argsort(score,kind='stable');uf=np.arange(n);top=np.arange(n);count=np.ones(2*n-1,int)
    means=np.zeros((2*n-1,3));means[:n]=c;children=np.full((2*n-1,2),-1,int);boundary=np.zeros(2*n-1)
    def find(j):
        while uf[j]!=j:uf[j]=uf[uf[j]];j=int(uf[j])
        return j
    node=n
    for k in order:
        a,b=map(find,edges[k])
        if a==b:continue
        i,j=top[a],top[b];children[node]=(i,j);count[node]=count[i]+count[j]
        means[node]=(means[i]*count[i]+means[j]*count[j])/count[node]
        boundary[node]=score[k]+np.mean((means[i]-means[j])**2)
        uf[b]=a;top[a]=node;node+=1
    return children[:node],count[:node],boundary[:node],node-1


def tree_partition(tree,max_size):
    children,count,_,root=tree;n=(len(children)+1)//2;groups=[];stack=[root]
    while stack:
        j=stack.pop()
        if j<n or count[j]<=max_size:
            todo=[j];ids=[]
            while todo:
                k=todo.pop()
                if k<n:ids.append(k)
                else:todo.extend(children[k])
            groups.append(np.asarray(ids,int))
        else:stack.extend(children[j])
    return groups


def tree_dp(U,tree,evidence=None,code_fee=True):
    children,count,boundary,root=tree;n=np.asarray(U).size;m=len(children)
    lg=logit(U).ravel()+(0 if evidence is None else np.asarray(evidence).ravel())
    d0=np.zeros(m);d1=np.zeros(m);d0[:n]=np.logaddexp(0,lg);d1[:n]=np.logaddexp(0,-lg)
    cost=np.empty(m);action=np.zeros(m,int);fee=np.log(max(n,2)) if code_fee else 0.
    scale=max(float(np.median(boundary[n:])) if m>n else 0,1e-4)
    for j in range(m):
        if j>=n:
            a,b=children[j];d0[j]=d0[a]+d0[b];d1[j]=d1[a]+d1[b]
        values=[d0[j]+fee,d1[j]+fee]
        if j>=n:
            a,b=children[j]
            # The price of opening a contour node decreases at measured RGB
            # separation. This is a fully specified pruning energy, not cut.
            values.append(cost[a]+cost[b]+np.exp(-boundary[j]/scale)/8)
        action[j]=int(np.argmin(values));cost[j]=values[action[j]]
    mask=np.zeros(n,bool);stack=[root];opened=0
    while stack:
        j=stack.pop()
        if action[j]==2:stack.extend(children[j]);continue
        opened+=1;value=action[j]==1;todo=[j]
        while todo:
            k=todo.pop()
            if k<n:mask[k]=value
            else:todo.extend(children[k])
    return mask.reshape(np.asarray(U).shape),dict(tree_nodes=m,opened_nodes=opened,code_fee=fee,tree_objective=float(cost[root]))


def exact_barrier(rgb,seeds,levels=16):
    """Exact independently minimized channel range in the specified 16 levels.

    Enumerate EVERY [lo,hi] intensity interval. Its 8-connected components
    intersecting a seed are exactly the feasible paths within that interval.
    """
    c=np.minimum(np.asarray(rgb,np.uint16)*levels//256,levels-1);seed=np.asarray(seeds,bool);out=np.zeros(seed.shape,float)
    for channel in range(3):
        z=c[...,channel];best=np.full(seed.shape,np.inf)
        for width in range(levels):
            for lo in range(levels-width):
                allowed=(z>=lo)&(z<=lo+width);cc,n=ndimage.label(allowed,np.ones((3,3),bool))
                touched=np.zeros(n+1,bool);touched[np.unique(cc[seed&allowed])]=True;touched[0]=False
                hit=touched[cc];best[hit]=np.minimum(best[hit],width/(levels-1))
        out+=best
    return out


def shortest_forest(rgb,U,roots):
    e,cap=H.rgb_edges(rgb);n=np.asarray(U).size;c=np.asarray(rgb,float).reshape(-1,3)/255.;u=np.asarray(U).ravel()
    costs=1+np.sqrt(np.mean((c[e[:,0]]-c[e[:,1]])**2,1))+np.abs(u[e[:,0]]-u[e[:,1]])
    adjacency=sparse.csr_matrix((np.r_[costs,costs],(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(n,n))
    root=np.flatnonzero(np.asarray(roots).ravel());dist,pre,sources=sparse.csgraph.dijkstra(adjacency,indices=root,
                                    min_only=True,return_predecessors=True)
    return dist,pre,sources


def fit_logistic(x,labels,valid,max_steps=100):
    """C_E class-balanced L2 logistic reg1, Lipschitz training-only step."""
    x=np.asarray(x,float);labels=np.asarray(labels,bool);valid=np.asarray(valid,bool)
    weights=H.balanced_weights(labels,valid)
    if weights is None:return None
    mean=weights@x;scale=np.sqrt(np.maximum(weights@((x-mean)**2),EPS*EPS));z=(x-mean)/scale
    lips=1+.25*(weights@np.sum(z*z,1)+1);coef=np.zeros(x.shape[1]);bias=0.
    for _ in range(max_steps):
        d=weights*(expit(z@coef+bias)-labels);coef-=(z.T@d+coef)/lips;bias-=d.sum()/lips
    return mean,scale,coef,bias,dict(steps=max_steps,regularization=1.,lipschitz=float(lips),class_balanced=True)


def predict_head(model,x):
    mean,scale,coef,bias,_=model;return expit(((np.asarray(x)-mean)/scale)@coef+bias)


def texture_filters(rgb):
    """Fixed low/high-frequency bank, RGB gradients and 1/2/4 pixel LoG."""
    c=np.asarray(rgb,float)/255.;l=c.mean(-1);parts=[c]
    for sigma in (1.,2.,4.):
        smooth=ndimage.gaussian_filter(l,sigma,mode='reflect');parts.extend([smooth[...,None],
              (l-smooth)[...,None],ndimage.gaussian_laplace(l,sigma,mode='reflect')[...,None]])
    parts.extend([(ndimage.sobel(l,1)/8)[...,None],(ndimage.sobel(l,0)/8)[...,None]])
    return np.concatenate(parts,-1).reshape(-1,14)


def kmeans(x,k,steps=5):
    x=np.asarray(x,float)
    if not len(x):return np.empty((0,x.shape[1]))
    ids=[0];dist=np.sum((x-x[0])**2,1)
    for _ in range(1,min(k,len(x))):
        j=int(np.argmax(dist))
        if dist[j]<=1e-12:break
        ids.append(j);dist=np.minimum(dist,np.sum((x-x[j])**2,1))
    centers=x[ids].copy()
    for _ in range(steps):
        label=nearest_words(x,centers)
        for j in range(len(centers)):
            if np.any(label==j):centers[j]=x[label==j].mean(0)
    return centers


def nearest_words(x,centers):
    x=np.asarray(x,float);out=np.empty(len(x),int)
    for start in range(0,len(x),4096):
        v=x[start:start+4096];d=np.sum(v*v,1)[:,None]+np.sum(centers*centers,1)[None]-2*v@centers.T
        out[start:start+4096]=np.argmin(d,1)
    return out


def texture_model(problem,train=None):
    """32 source-only words with equal FG/BG capacity; smoothed multinomial."""
    ep=problem.ep;r=texture_filters(ep.r_rgb);q=texture_filters(ep.q_rgb)
    valid=problem.rvalid if train is None else np.asarray(train,bool)&problem.rvalid;lab=problem.labels
    weights=H.balanced_weights(lab,valid)
    if weights is None:return None
    mean=weights@r;scale=np.sqrt(np.maximum(weights@((r-mean)**2),EPS*EPS));rr=(r-mean)/scale;qq=(q-mean)/scale
    vocab=np.vstack([kmeans(rr[valid&(lab==cls)],16) for cls in (False,True)])
    if not len(vocab):return None
    rw=nearest_words(rr,vocab);qw=nearest_words(qq,vocab);k=len(vocab)
    freq=[]
    for cls in (False,True):
        counts=np.bincount(rw[valid&(lab==cls)],minlength=k)+1.;freq.append(counts/counts.sum())
    ratio=np.log(freq[1]/freq[0]);rs=ratio[rw];qs=ratio[qw];norm=source_scale(rs,valid)
    return dict(q=qs/norm,r=rs/norm,qwords=qw,rwords=rw,vocab=vocab,frequency=np.stack(freq),
                mean=mean,feature_scale=scale,scale=norm,valid=valid)


def line_dp(cost,transition):
    """All states Viterbi with explicit backpointers (including BG breaks)."""
    c=np.asarray(cost,float);t=np.asarray(transition,float);n,k=c.shape;value=c[0].copy();back=np.zeros((n,k),int)
    for j in range(1,n):
        v=value[:,None]+t;back[j]=np.argmin(v,0);value=c[j]+np.min(v,0)
    state=np.empty(n,int);state[-1]=int(np.argmin(value))
    for j in range(n-1,0,-1):state[j-1]=back[j,state[j]]
    return state,float(value[state[-1]])


def ridge_candidates(rgb,limit=128):
    """Observed Hessian ridges at fixed RGB scales1/2/4 with measured normals."""
    lum=np.asarray(rgb,float).mean(-1)/255.;h,w=lum.shape;strength=np.zeros((h,w));normal=np.zeros((h,w,2))
    for sigma in (1.,2.,4.):
        xx=ndimage.gaussian_filter(lum,sigma,order=(0,2));yy=ndimage.gaussian_filter(lum,sigma,order=(2,0));xy=ndimage.gaussian_filter(lum,sigma,order=(1,1))
        matrix=np.stack((np.stack((xx,xy),-1),np.stack((xy,yy),-1)),-2)
        values,vectors=np.linalg.eigh(matrix);which=np.argmax(np.abs(values),-1)
        strongest=np.take_along_axis(values,which[...,None],-1)[...,0]
        weakest=np.take_along_axis(values,(1-which)[...,None],-1)[...,0]
        response=np.maximum(np.abs(strongest)-np.abs(weakest),0)*sigma*sigma
        update=response>strength;strength[update]=response[update]
        normals=np.take_along_axis(vectors,which[...,None,None],-1)[...,0]
        normal[update]=normals[update]
    positive=strength>max(float(np.quantile(strength,.75)),EPS)
    maxima=positive&(strength>=ndimage.maximum_filter(strength,3))
    points=np.argwhere(maxima);order=np.argsort(-strength[maxima],kind='stable')[:limit]
    return points[order],normal[tuple(points[order].T)] if len(order) else np.empty((0,2)),strength


def cross_sections(rgb,points,normals,radius=None):
    """Find measured edge pair on opposite sides, no transferred width."""
    lum=np.asarray(rgb,float).mean(-1)/255.;h,w=lum.shape;gx=ndimage.sobel(lum,1)/8;gy=ndimage.sobel(lum,0)/8
    mag=np.hypot(gx,gy);threshold=max(float(np.median(mag)),EPS);radius=radius or min(16,max(h,w))
    sections=[]
    for point,normal in zip(points,normals):
        y,x=point;nx,ny=normal;ends=[]
        for sign in (-1,1):
            samples=[]
            for d in range(1,radius+1):
                yy=int(round(y+sign*d*ny));xx=int(round(x+sign*d*nx))
                if not(0<=yy<h and 0<=xx<w):break
                samples.append((mag[yy,xx],d))
            if not samples:break
            peak,d=max(samples)
            if peak<=threshold:break
            ends.append(sign*d)
        if len(ends)!=2:sections.append(np.empty(0,int));continue
        ids=[]
        for d in np.arange(ends[0],ends[1]+.5,.5):
            yy=int(round(y+d*ny));xx=int(round(x+d*nx))
            if 0<=yy<h and 0<=xx<w:ids.append(yy*w+xx)
        sections.append(np.unique(ids))
    return sections


def point_chains(points,normals,shape):
    """Deterministic sparse ridge adjacency, oriented local chains, all points."""
    if not len(points):return []
    from scipy.spatial import cKDTree
    tree=cKDTree(points);pairs=tree.query_pairs(5,output_type='ndarray');adj=[[] for _ in points]
    for i,j in pairs:
        delta=points[j]-points[i];delta=delta/max(np.linalg.norm(delta),EPS)
        # normals are(x,y); line directions must align with their tangents.
        if abs(np.dot(delta,normals[i][::-1]))<.6 and abs(np.dot(delta,normals[j][::-1]))<.6:
            adj[i].append(int(j));adj[j].append(int(i))
    visited=set();chains=[]
    for root in sorted(range(len(points)),key=lambda j:(len(adj[j])==2,j)):
        if root in visited:continue
        chain=[];j=root;prev=-1
        while j not in visited:
            chain.append(j);visited.add(j);nexts=[k for k in sorted(adj[j]) if k!=prev and k not in visited]
            if not nexts:break
            prev,j=j,nexts[0]
        chains.append(np.asarray(chain,int))
    return chains


def halfdisk_features(rgb,tokens,z,edges,radius=4):
    """Original-pixel paired halfdisks with full native DINO side averages."""
    rgb=np.asarray(rgb,float)/255.;h,w=rgb.shape[:2];phi=texture_filters((rgb*255).astype(float)).reshape(h,w,-1)
    e=np.asarray(edges,int);out=np.empty((len(e),2*z.shape[1]+2*phi.shape[-1]+3))
    for start in range(0,len(e),256):
        en=e[start:start+256];ya,xa=np.divmod(en[:,0],w);yb,xb=np.divmod(en[:,1],w)
        center_y=(ya+yb)/2;center_x=(xa+xb)/2;ny=yb-ya;nx=xb-xa
        sides=[]
        for sign in (-1,1):
            offsets=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1) if dy*dy+dx*dx<=radius*radius]
            ss=np.zeros((len(en),z.shape[1]));pp=np.zeros((len(en),phi.shape[-1]));cc=np.zeros((len(en),3));mass=np.zeros(len(en))
            for dy,dx in offsets:
                yy=np.rint(center_y+dy).astype(int);xx=np.rint(center_x+dx).astype(int)
                valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)&(sign*(dy*ny+dx*nx)>.0)
                yy=np.clip(yy,0,h-1);xx=np.clip(xx,0,w-1);ids=yy*w+xx
                ss+=z[tokens[ids]]*valid[:,None];pp+=phi[yy,xx]*valid[:,None];cc+=rgb[yy,xx]*valid[:,None];mass+=valid
            sides.append((ss/np.maximum(mass[:,None],1),pp/np.maximum(mass[:,None],1),cc/np.maximum(mass[:,None],1)))
        a,b=sides
        out[start:start+len(en)]=np.concatenate((a[0],b[0],a[1],b[1],b[2]-a[2]),1)
    return out


def edge_features(rgb,tokens,z,edges):
    c=np.asarray(rgb,float).reshape(-1,3)/255.;p=H.texture_phi(rgb);e=np.asarray(edges,int)
    chrom=c/np.maximum(c.sum(1,keepdims=True),EPS)
    return np.concatenate((np.abs(chrom[e[:,0]]-chrom[e[:,1]]),
           np.abs(c[e[:,0]].mean(1)-c[e[:,1]].mean(1))[:,None],
           np.abs(p[e[:,0]]-p[e[:,1]]),np.abs(z[tokens[e[:,0]]]-z[tokens[e[:,1]]])),1)


def poisson_reconstruction(shape,edge_grad,mean,regularization=1.):
    """Fixed-mean reconstruction whose RHS is PREDICTED gradient divergence."""
    edges=H.grid_edges(shape);n=np.prod(shape);m=len(edges)
    D=sparse.csr_matrix((np.tile((-1.,1.),m),(np.repeat(np.arange(m),2),edges.ravel())),shape=(m,n))
    L=D.T@D;A=L+regularization*sparse.eye(n)
    rhs=D.T@np.asarray(edge_grad,float)
    out=np.empty((n,rhs.shape[1]));statuses=[]
    for channel in range(rhs.shape[1]):
        v,status=linalg.cg(A,rhs[:,channel],rtol=1e-10,atol=1e-12,maxiter=2000)
        v-=v.mean();v+=mean[channel];out[:,channel]=v;statuses.append(int(status))
    return out,D,dict(statuses=statuses,rhs='divergence_of_class_dictionary_prediction',mean_fixed=list(mean),regularization=regularization)


def interval_dp(rgb,U,fixed=None):
    """Exact ALL-interval weighted segmentation of one scanline.

    Background is one-pixel action; every foreground interval[a,b] is an
    action, including one-pixel intervals. Prefix sums give exact RGB SSE.
    """
    c=np.asarray(rgb,float)/255.;u=np.asarray(U,float);n=len(u);base=logit(u)
    d0=np.logaddexp(0,base);d1=np.logaddexp(0,-base)
    if fixed is not None:d0+=(np.asarray(fixed,bool));d1+=(~np.asarray(fixed,bool))
    sums=np.vstack((np.zeros((1,3)),np.cumsum(c,0)));sq=np.r_[0,np.cumsum(np.sum(c*c,1))]
    colour_scale=max(float(np.median(np.mean(np.diff(c,axis=0)**2,1))) if n>1 else 0,1e-4)
    boundary=np.zeros(n+1);boundary[1:n]=np.exp(-np.mean(np.diff(c,axis=0)**2,1)/colour_scale)/8
    pref=np.r_[0,np.cumsum(d1)];value=np.full(n+1,np.inf);value[0]=0;back=np.zeros(n+1,int);fg=np.zeros(n+1,bool)
    for b in range(1,n+1):
        best=value[b-1]+d0[b-1];start=b-1;isfg=False
        a=np.arange(b);length=b-a;total=sums[b]-sums[a]
        sse=np.maximum(sq[b]-sq[a]-np.sum(total*total,1)/length,0)
        costs=value[a]+pref[b]-pref[a]+sse/colour_scale+boundary[a]+boundary[b]
        j=int(np.argmin(costs))
        if costs[j]<best:best=costs[j];start=j;isfg=True
        value[b]=best;back[b]=start;fg[b]=isfg
    mask=np.zeros(n,bool);b=n
    while b:
        a=back[b]
        if fg[b]:mask[a:b]=True
        b=a
    return mask,float(value[-1])


def interval_energy(rgb,U,labels,fixed=None):
    c=np.asarray(rgb,float)/255.;z=np.asarray(labels,bool);base=logit(U)
    e=float(np.sum(np.where(z,np.logaddexp(0,-base),np.logaddexp(0,base))))
    if fixed is not None:e+=float(np.sum(z!=fixed))
    scale=max(float(np.median(np.mean(np.diff(c,axis=0)**2,1))) if len(c)>1 else 0,1e-4)
    cc,n=ndimage.label(z)
    for k in range(1,n+1):
        ids=np.flatnonzero(cc==k);v=c[ids];e+=float(np.sum((v-v.mean(0))**2)/scale)
        for i in (ids[0],ids[-1]+1):
            if 0<i<len(c):e+=float(np.exp(-np.mean((c[i]-c[i-1])**2)/scale)/8)
    return e
