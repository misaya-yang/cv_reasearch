"""Region-shared RGB nuisance and independently anchored contour profiles."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.special import logsumexp
from . import e_helpers_226_250 as H
from . import e226_250 as B
from . import e235_244 as C
from .e226_234 import fit_gmm


def _nll_gradient(x,model,delta):
    value=x-delta
    difference=value[:,None]-model.means
    scores=np.log(model.weights)[None]-.5*(np.sum(difference*difference/model.variances,2)+np.log(2*np.pi*model.variances).sum(1))
    normal=logsumexp(scores,axis=1);resp=np.exp(scores-normal[:,None])
    gradient=np.sum(resp[...,None]*(-difference/model.variances),axis=1)
    return -normal,gradient


def _shared_adversary(x,labels,f,b,L):
    starts=[np.zeros(L.shape[1])]
    for j in range(L.shape[1]):
        for sign in (-1,1):
            v=np.zeros(L.shape[1]);v[j]=sign;starts.append(v)
    best_cost=-np.inf;best=np.zeros(3);evaluations=0;average=np.zeros((len(x),2))
    for start in starts:
        t=start.copy()
        for iteration in range(21):
            delta=L@t;fc,fg=_nll_gradient(x,f,delta);bc,bg=_nll_gradient(x,b,delta)
            objective=float(np.where(labels,fc,bc).mean());evaluations+=1
            average+=np.c_[fc,bc]
            if objective>best_cost:best_cost=objective;best=delta.copy()
            if iteration==20:break
            gradient=np.where(labels[:,None],fg,bg).mean(0)
            t+=.05*(L.T@gradient);t/=max(np.linalg.norm(t),1)
    return best,best_cost,evaluations,average/evaluations


def _pointwise_adversary(x,model,L):
    starts=[np.zeros(L.shape[1])]
    for j in range(L.shape[1]):
        for sign in (-1,1):
            v=np.zeros(L.shape[1]);v[j]=sign;starts.append(v)
    worst=np.full(len(x),-np.inf)
    for start in starts:
        t=np.broadcast_to(start,(len(x),L.shape[1])).copy()
        for iteration in range(21):
            value,gradient=_nll_gradient(x,model,t@L.T);worst=np.maximum(worst,value)
            if iteration==20:break
            t+=.05*(gradient@L);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1.)
    return worst


def e241_shared_nuisance(problem,control=None):
    ep=problem.ep;r=np.asarray(ep.r_rgb,float).reshape(-1,3)/255.;q=np.asarray(ep.q_rgb,float).reshape(-1,3)/255.
    valid=problem.rvalid;lab=problem.labels
    f=fit_gmm(r[valid&lab],2);b=fit_gmm(r[valid&~lab],2)
    if f is None or b is None:return problem.U.copy(),dict(status='fallback_source_colour_role_missing')
    f_single=np.max(np.linalg.norm(f.means-f.means[0],axis=1),initial=0)<H.EPS
    b_single=np.max(np.linalg.norm(b.means-b.means[0],axis=1),initial=0)<H.EPS
    if control is None and f_single and b_single and np.allclose(f.variances[0],b.variances[0],rtol=1e-5,atol=1e-8):
        return problem.U.copy(),dict(status='fallback_class_ratio_affine_no_nonlinear_GMM_mechanism')
    h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw);differences=[];means=[r[valid&lab].mean(0),r[valid&~lab].mean(0)]
    for by in range(2):
        for bx in range(2):
            held=((yy>=by*h//2)&(yy<(by+1)*h//2)&(xx>=bx*w//2)&(xx<(bx+1)*w//2)).ravel()
            excluded=ndimage.binary_dilation(held.reshape(ep.r_hw),structure=np.ones((3,3))).ravel()
            train=valid&~excluded[problem.rtokens]
            if (train&lab).sum()<8 or (train&~lab).sum()<8:continue
            differences.append(.5*((r[train&lab].mean(0)-means[0])+(r[train&~lab].mean(0)-means[1])))
    if len(differences)<3:return problem.U.copy(),dict(status='fallback_uncertain_ellipsoid_unidentified')
    covariance=np.cov(np.asarray(differences).T);eigen,V=np.linalg.eigh(covariance);keep=eigen>1e-8
    if not keep.any():return problem.U.copy(),dict(status='fallback_no_observed_source_nuisance_radius')
    L=V[:,keep]*np.sqrt(eigen[keep])[None,:]
    anchors=B._anchors(problem).ravel()
    if not np.any(anchors==1) or not np.any(anchors==-1):return problem.U.copy(),dict(status='fallback_query_double_anchors_missing')
    source=f.log_density(r)-b.log_density(r);scale=max(float(np.median(np.abs(source[valid]))),H.EPS)
    components,count=ndimage.label(problem.U>.5);regions=[];covered=np.zeros(problem.U.size,bool)
    for j in range(1,count+1):
        region=ndimage.binary_dilation(components==j,iterations=4).ravel();points=np.flatnonzero(region)
        if not np.any(anchors[points]==1) or not np.any(anchors[points]==-1):continue
        if len(regions)>=128:continue
        regions.append(points);covered[points]=True
    if not regions:return problem.U.copy(),dict(status='fallback_no_two_anchor_region')
    # Give an overlapping halo one owner, so a pixel's likelihood is counted
    # once; sharing a class prior is not an instance repetition reward.
    owner=np.full(len(q),-1,int);closest=np.full(len(q),np.inf);row,col=np.indices(problem.U.shape)
    coords=np.c_[row.ravel(),col.ravel()]
    for j,points in enumerate(regions):
        center=coords[points].mean(0);dist=np.sum((coords[points]-center)**2,1)
        take=dist<closest[points];owner[points[take]]=j;closest[points[take]]=dist[take]
    regions=[p for j in range(len(regions)) if len(p:=np.flatnonzero(owner==j))];covered=owner>=0
    labels=problem.U.ravel()>.5;evidence=np.zeros(len(q));evaluation_count=0;visited=[];best=problem.U.copy();best_energy=np.inf
    grid_edges,grid_capacity=H.rgb_edges(ep.q_rgb)
    U=np.clip(problem.U.ravel(),H.EPS,1-H.EPS)
    for outer in range(5):
        sums=np.zeros(len(q));mass=np.zeros(len(q));class_cost=np.zeros((len(q),2))
        for points in regions:
            if control=='pointwise':
                # Exact same bounded threat support, but each pixel may choose
                # a different visited perturbation. This is a stronger threat.
                cost=np.c_[_pointwise_adversary(q[points],f,L),_pointwise_adversary(q[points],b,L)]
                own=(cost[:,1]-cost[:,0])/scale
            elif control=='average':
                delta,loss,evals,cost=_shared_adversary(q[points],labels[points],f,b,L);evaluation_count+=evals
                own=(cost[:,1]-cost[:,0])/scale
            else:
                delta,loss,evals,_=_shared_adversary(q[points],labels[points],f,b,L);evaluation_count+=evals
                cost=np.c_[-f.log_density(q[points]-delta),-b.log_density(q[points]-delta)]
                own=(cost[:,1]-cost[:,0])/scale
                visited.append(dict(outer=outer,region_pixels=len(points),delta=delta.tolist(),maximum_visited_loss=loss))
            np.add.at(sums,points,own);np.add.at(mass,points,1)
            class_cost[points]=cost/scale
        evidence=np.divide(sums,mass,out=np.zeros_like(sums),where=mass>0)
        # Score the CURRENT complete label field using its own visited maximum
        # before the next cut. Dropping the BG NLL constant while comparing
        # different perturbations would change the specified robust objective.
        data=np.where(labels,-np.log(U),-np.log(1-U)).sum()
        colour_cost=np.where(labels[covered],class_cost[covered,0],class_cost[covered,1]).sum()
        boundary=grid_capacity@(labels[grid_edges[:,0]]!=labels[grid_edges[:,1]])
        true_visited=float(data+colour_cost+boundary)
        if true_visited<best_energy:
            best_energy=true_visited;best=labels.reshape(problem.U.shape).astype(float)
            best.ravel()[~covered]=problem.U.ravel()[~covered]
        output,certificate=C._cut(problem,evidence);output.ravel()[~covered]=problem.U.ravel()[~covered]
        labels=output.ravel()>.5
    return best,dict(status='ok',binary_optimizer=True,shared_regions=len(regions),ellipsoid_rank=L.shape[1],
                    ellipsoid_axes=np.sqrt(eigen[keep]).tolist(),inner_visits=evaluation_count,outer_rounds=5,
                    inner_starts=1+2*L.shape[1],inner_steps=20,step=.05,source_evidence_scale=scale,
                    perturbation_trace=visited,best_complete_visited_objective=best_energy,
                    overlapping_region_pixels_counted_once=True,
                    finite_visited_max_not_global_adversarial_guarantee=True,control=control)


def e232_independent_roundtrip(problem,control=None):
    ep=problem.ep;anchors=B._anchors(problem).ravel()
    if not np.any(anchors==1) or not np.any(anchors==-1):return problem.U.copy(),dict(status='fallback_missing_query_double_anchors')
    colour=H.diagonal_colour_evidence(ep.q_rgb,ep.r_rgb,problem.labels,problem.rvalid,k=2)
    if colour is None:return problem.U.copy(),dict(status='fallback_source_texture_roles')
    texture,source_fit=colour;semantic=H.native_to_original(problem.u0.reshape(ep.q_hw),ep.original_shape,ep.query_geometry).ravel()
    chains,ny,nx=C._contour_chains(ep.q_rgb)
    if not chains:return problem.U.copy(),dict(status='fallback_no_RGB_contour')
    centers=B._spread_rows(np.unique(np.concatenate(chains)),128);shape=ep.original_shape
    g=ep.query_geometry
    width=max(shape[0]/ep.q_hw[0],shape[1]/ep.q_hw[1]) if not g else max(shape[0]*g['view_side']/(ep.q_hw[0]*g['resized_hw'][0]),shape[1]*g['view_side']/(ep.q_hw[1]*g['resized_hw'][1]))
    radius=int(np.ceil(4*width));evidence=np.zeros(problem.U.size);agreements=0;disagreements=0;missing=0
    for center in centers:
        y,x=divmod(int(center),shape[1]);t=np.arange(-radius,radius+1)
        yy=np.clip(np.rint(y+t*ny.ravel()[center]).astype(int),0,shape[0]-1)
        xx=np.clip(np.rint(x+t*nx.ravel()[center]).astype(int),0,shape[1]-1)
        points=yy*shape[1]+xx;points=points[np.r_[True,points[1:]!=points[:-1]]]
        fg=np.flatnonzero(anchors[points]==1);bg=np.flatnonzero(anchors[points]==-1)
        if not len(fg) or not len(bg):missing+=1;continue
        # Distinct confidence-qualified ends, not one peak sent both ways.
        f,b=int(fg[0]),int(bg[-1])
        if f==b:missing+=1;continue
        if f>b:points=points[::-1];f=len(points)-1-f;b=len(points)-1-b
        segment=points[f:b+1];s=semantic[segment];rgb=texture[segment]
        forward=np.flatnonzero((s<=.5)&(rgb<=0));reverse=np.flatnonzero((s>.5)&(rgb>=0))
        if not len(forward) or not len(reverse):missing+=1;continue
        out=int(forward[0]);inside=int(reverse[-1]);distance=np.linalg.norm(np.array(divmod(int(segment[out]),shape[1]))-np.array(divmod(int(segment[inside]),shape[1])))
        if control=='oneway':inside=max(0,out-1);distance=0
        elif control=='best_change':
            best=int(np.argmax(np.abs(np.diff(s+rgb))));inside=best;out=best+1;distance=0
        if distance<=1+1e-12:
            np.add.at(evidence,segment[:inside+1],1);np.add.at(evidence,segment[out:],-1);agreements+=1
        else:
            # A finite inconsistency weakens the local interface; it does not
            # assert the entire query is BG or select a maximum component.
            np.add.at(evidence,segment[inside:out+1],-.25*np.sign(s[inside:out+1]-.5));disagreements+=1
    if not agreements and not disagreements:return problem.U.copy(),dict(status='fallback_no_two_anchor_contour_factor',missing_profiles=missing)
    out,cert=C._cut(problem,evidence)
    return out,dict(status='ok',binary_optimizer=True,contour_centers=len(centers),profile_radius=radius,
                    independent_anchor_agreements=agreements,inconsistent_profiles=disagreements,
                    missing_anchor_or_turn_profiles=missing,tolerance_original_pixels=1,
                    source_colour=source_fit,cut=cert,control=control)


METHODS={'E232':lambda ep:B._call(ep,'E232',e232_independent_roundtrip),
         'E241':lambda ep:B._call(ep,'E241',e241_shared_nuisance)}
CONTROLS={
 'E232_oneway_same_profile':lambda ep:B._call(ep,'E232',e232_independent_roundtrip,control='oneway'),
 'E232_best_change_same_profile':lambda ep:B._call(ep,'E232',e232_independent_roundtrip,control='best_change'),
 'E241_pointwise_ellipsoid_worst':lambda ep:B._call(ep,'E241',e241_shared_nuisance,control='pointwise'),
 'E241_same_ellipsoid_RGB_average':lambda ep:B._call(ep,'E241',e241_shared_nuisance,control='average'),
}
RESOURCES={id:dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=True,extra_encoder_forwards=0) for id in METHODS}
