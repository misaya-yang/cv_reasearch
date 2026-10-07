"""Visible RGB scene/boundary hypotheses for the original supplied E cards."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.special import logsumexp
from . import e_helpers_226_250 as H
from . import e226_250 as B
from . import e235_244 as C


class ChunkGMM(H.DiagGMM):
    def log_density(self,x):
        x=np.asarray(x,float);out=np.empty(len(x))
        for start in range(0,len(x),512):
            out[start:start+512]=super().log_density(x[start:start+512])
        return out


def fit_gmm(x,k=2,steps=20):
    """Same deterministic diagonal EM, streamed to avoid N*K*d allocation."""
    x=np.asarray(x,float)
    if not len(x):return None
    k=min(k,len(x));ids=[0];closest=np.sum((x-x[0])**2,1)
    for _ in range(1,k):
        j=int(np.argmax(closest));ids.append(j)
        closest=np.minimum(closest,np.sum((x-x[j])**2,1))
    means=x[ids].copy();variances=np.tile(np.maximum(x.var(0),1e-4),(k,1));weights=np.full(k,1/k)
    for _ in range(steps):
        mass=np.zeros(k);sx=np.zeros_like(means);sxx=np.zeros_like(means)
        for start in range(0,len(x),512):
            v=x[start:start+512]
            d=np.sum((v[:,None]-means)**2/variances,2)
            log=np.log(np.maximum(weights,H.EPS))[None]-.5*(d+np.log(2*np.pi*variances).sum(1))
            r=np.exp(log-logsumexp(log,axis=1,keepdims=True));mass+=r.sum(0)
            sx+=r.T@v;sxx+=r.T@(v*v)
        means=sx/np.maximum(mass[:,None],H.EPS)
        variances=np.maximum(sxx/np.maximum(mass[:,None],H.EPS)-means*means,1e-4)
        weights=np.maximum(mass,H.EPS);weights/=weights.sum()
    return ChunkGMM(means,variances,weights)


def patch_geometry(shape,hw,g,index):
    """Original-space physical patch and exact clipped original pixel cells."""
    if not g:g=dict(view_side=hw[0],resized_hw=hw,padding_top_left=(0,0))
    side=g['view_side'];sh,sw=g['resized_hw'];oy,ox=g['padding_top_left'];y,x=divmod(index,hw[1])
    y0=max(0.,(y*side/hw[0]-oy)*shape[0]/sh);y1=min(float(shape[0]),((y+1)*side/hw[0]-oy)*shape[0]/sh)
    x0=max(0.,(x*side/hw[1]-ox)*shape[1]/sw);x1=min(float(shape[1]),((x+1)*side/hw[1]-ox)*shape[1]/sw)
    if y1<=y0 or x1<=x0:return None
    iy=np.arange(int(np.floor(y0)),int(np.ceil(y1)));ix=np.arange(int(np.floor(x0)),int(np.ceil(x1)))
    py,px=np.meshgrid(iy,ix,indexing='ij');ids=(py*shape[1]+px).ravel()
    left=np.maximum(px.ravel(),x0);right=np.minimum(px.ravel()+1.,x1)
    top=np.maximum(py.ravel(),y0);bottom=np.minimum(py.ravel()+1.,y1)
    return dict(ids=ids,left=left,right=right,top=top,bottom=bottom,
                mass=(right-left)*(bottom-top),rectangle=(y0,y1,x0,x1),
                cx=(left+right)/2,cy=(top+bottom)/2)


def line_cell_fraction(normal,offset,left,right,top,bottom):
    """Exact integral of n.x <= offset over each possibly clipped pixel cell.

    This is the CDF of two uniform intervals, including the axial limit. It
    performs no supersampling and never interprets a token's GT mixing state.
    """
    nx,ny=normal;cx=(left+right)/2;cy=(top+bottom)/2
    a=abs(nx)*(right-left);b=abs(ny)*(bottom-top)
    t=offset-nx*cx-ny*cy+(a+b)/2
    if abs(nx)<1e-12:return np.clip(t/np.maximum(b,H.EPS),0,1)
    if abs(ny)<1e-12:return np.clip(t/np.maximum(a,H.EPS),0,1)
    positive=lambda x:np.maximum(x,0.)**2
    return np.clip((positive(t)-positive(t-a)-positive(t-b)+positive(t-a-b))/(2*np.maximum(a*b,1e-30)),0,1)


def _line_states(geometry):
    y0,y1,x0,x1=geometry['rectangle'];cx=(x0+x1)/2;cy=(y0+y1)/2
    states=[]
    for angle in range(16):
        theta=2*np.pi*angle/16;normal=np.array([np.cos(theta),np.sin(theta)])
        radius=(abs(normal[0])*(x1-x0)+abs(normal[1])*(y1-y0))/2
        for j in range(8):
            offset=normal[0]*cx+normal[1]*cy+radius*((j+.5)/4-1)
            states.append((normal,float(offset)))
    return states+[(None,1.),(None,0.)]


def _state_fraction(state,geometry):
    normal,offset=state
    if normal is None:return np.full(len(geometry['ids']),offset)
    return line_cell_fraction(normal,offset,geometry['left'],geometry['right'],geometry['top'],geometry['bottom'])


def _side_profile(state,geometry,side,points=16):
    y0,y1,x0,x1=geometry['rectangle'];n,offset=state
    if n is None:return np.full(points,offset)
    t=(np.arange(points)+.5)/points
    if side=='top':x=x0+(x1-x0)*t;y=np.full(points,y0)
    elif side=='bottom':x=x0+(x1-x0)*t;y=np.full(points,y1)
    elif side=='left':y=y0+(y1-y0)*t;x=np.full(points,x0)
    else:y=y0+(y1-y0)*t;x=np.full(points,x1)
    return (n[0]*x+n[1]*y<=offset).astype(float)


def _side_intersection(state,geometry,side):
    y0,y1,x0,x1=geometry['rectangle'];n,offset=state
    if n is None:return np.nan
    if side in ('top','bottom'):
        if abs(n[0])<1e-12:return np.nan
        value=((offset-n[1]*(y0 if side=='top' else y1))/n[0]-x0)/(x1-x0)
    else:
        if abs(n[1])<1e-12:return np.nan
        value=((offset-n[0]*(x0 if side=='left' else x1))/n[1]-y0)/(y1-y0)
    return float(value) if 0<=value<=1 else np.nan


def e229_subpatch_integral(problem,control=None):
    ep=problem.ep;native=problem.u0.reshape(ep.q_hw);gate=H.semantic_gate(problem.u0,ep.q_hw,ep.q_hw,{})
    selected=np.flatnonzero(gate.ravel()&(ep.q_valid>0));colour=np.asarray(ep.q_rgb,float).reshape(-1,3)/255.
    source=np.asarray(ep.r_rgb,float).reshape(-1,3)/255.;valid=problem.rvalid
    means=[]
    for label in (False,True):
        if not np.any(valid&(problem.labels==label)):return problem.U.copy(),dict(status='fallback_missing_source_colour_role')
        means.append(np.median(source[valid&(problem.labels==label)],0))
    source_error=np.sum((source-np.where(problem.labels[:,None],means[1],means[0]))**2,1)
    colour_scale=max(float(np.median(source_error[valid])),1e-4)
    coverage=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0)
    known=np.isfinite(problem.oob)&(ep.wvalid>0)
    sigma=max(.05,float(np.sqrt(np.average((problem.oob[known]-coverage[known])**2,weights=ep.wvalid[known])))) if known.any() else .25
    geometries={};states={};costs={};profiles={};intersections={}
    for index in selected:
        g=patch_geometry(ep.original_shape,ep.q_hw,ep.query_geometry,int(index))
        if g is None:continue
        geometries[index]=g;states[index]=_line_states(g);nodecost=[]
        observed=colour[g['ids']];mass=g['mass'];total=mass.sum();u=problem.U.ravel()[g['ids']]
        for state in states[index]:
            a=_state_fraction(state,g);design=np.stack((a,1-a),1)
            # Two-sided colour explanation has identical freedom for every
            # line orientation, including both foreground/background sides.
            gram=design.T@(mass[:,None]*design)
            coef=np.linalg.pinv(gram,rcond=1e-10)@(design.T@(mass[:,None]*observed))
            reconstruction=design@np.clip(coef,0,1)
            rgb=float(mass@np.sum((observed-reconstruction)**2,1)/(total*colour_scale))
            coverage_cost=float(((mass@a/total)-(mass@u/total))**2/(sigma*sigma))
            nodecost.append(rgb+coverage_cost)
        costs[index]=np.array(nodecost)
        profiles[index]={side:np.stack([_side_profile(s,g,side) for s in states[index]])
                         for side in ('top','bottom','left','right')}
        intersections[index]={side:np.array([_side_intersection(s,g,side) for s in states[index]])
                              for side in ('top','bottom','left','right')}
    if not geometries:return problem.U.copy(),dict(status='fallback_no_conflict_patch')
    choices={j:int(np.argmin(costs[j])) for j in geometries};neighbours={j:[] for j in geometries}
    for j in geometries:
        y,x=divmod(int(j),ep.q_hw[1])
        for dy,dx,side,other in ((-1,0,'top','bottom'),(1,0,'bottom','top'),(0,-1,'left','right'),(0,1,'right','left')):
            k=(y+dy)*ep.q_hw[1]+x+dx
            if 0<=y+dy<ep.q_hw[0] and 0<=x+dx<ep.q_hw[1] and k in geometries:
                neighbours[j].append((k,side,other))
    if control!='independent':
        for _ in range(5):
            for j in sorted(geometries):
                score=costs[j].copy()
                for k,side,other in neighbours[j]:
                    score+=np.mean(np.abs(profiles[j][side]-profiles[k][other][choices[k]]),axis=1)
                    a=intersections[j][side];b=intersections[k][other][choices[k]]
                    if np.isfinite(b):score+=np.where(np.isfinite(a),np.abs(a-b),0.)
                choices[j]=int(np.argmin(score))
    area=np.zeros(problem.U.size);covered=np.zeros(problem.U.size)
    for j,g in geometries.items():
        a=_state_fraction(states[j][choices[j]],g)
        if control=='pixel_quota':
            # Same inferred coverage and RGB observations, not an oracle area.
            quota=int(round(float(g['mass']@a)));bg,fg=means
            evidence=np.sum((colour[g['ids']]-bg)**2,1)-np.sum((colour[g['ids']]-fg)**2,1)
            order=np.argsort(-evidence,kind='stable');a=np.zeros(len(a));a[order[:quota]]=1
        np.add.at(area,g['ids'],g['mass']*a);np.add.at(covered,g['ids'],g['mass'])
    out=(area+(1-covered)*problem.U.ravel()).reshape(ep.original_shape)
    return np.clip(out,0,1),dict(status='ok',conflict_patches=len(geometries),states_per_patch=130,
                    coordinate_rounds=0 if control=='independent' else 5,colour_scale=colour_scale,
                    semantic_coverage_sigma=sigma,exact_clipped_pixel_cell_integrals=True,
                    max_pixel_partition_error=float(np.max(np.maximum(covered-1,0),initial=0)),control=control)


def _one_gaussian(x):
    return H.DiagGMM(np.mean(x,axis=0,keepdims=True),np.maximum(np.var(x,axis=0,keepdims=True),1e-4),np.ones(1))


def e226_visible_t_layers(problem,control=None):
    ep=problem.ep;chains,ny,nx=C._contour_chains(ep.q_rgb)
    if not chains:return problem.U.copy(),dict(status='fallback_no_RGB_T_junction')
    skeleton=np.zeros(problem.U.shape,bool);skeleton.ravel()[np.unique(np.concatenate(chains))]=True
    degree=ndimage.convolve(skeleton.astype(int),np.ones((3,3),int),mode='constant')-skeleton
    centers=B._spread_rows(np.flatnonzero(skeleton&(degree==3)),128)
    colour=np.asarray(ep.q_rgb,float).reshape(-1,3)/255.;nodes=[];links=[]
    for center in centers:
        y,x=divmod(int(center),problem.U.shape[1]);arms=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dy or dx):continue
                yy,xx=y+dy,x+dx
                if 0<=yy<problem.U.shape[0] and 0<=xx<problem.U.shape[1] and skeleton[yy,xx]:
                    arms.append(np.array([dy,dx],float)/np.hypot(dy,dx))
        if len(arms)!=3:continue
        pairs=[(a,b) for a in range(3) for b in range(a+1,3)];a,b=min(pairs,key=lambda t:float(arms[t[0]]@arms[t[1]]));stem=3-a-b
        # Two local front/back orders plus unordered. No order means FG by
        # definition; its semantic pixels are always free in the binary cut.
        ends=[]
        for arm in arms:
            yy=int(np.clip(np.rint(y+4*arm[0]),0,problem.U.shape[0]-1));xx=int(np.clip(np.rint(x+4*arm[1]),0,problem.U.shape[1]-1));ends.append(yy*problem.U.shape[1]+xx)
        bar=(ends[a],ends[b]);stem_pair=(int(center),ends[stem])
        continuity_bar=float(np.sum((colour[bar[0]]-colour[bar[1]])**2))
        continuity_stem=float(np.sum((colour[stem_pair[0]]-colour[stem_pair[1]])**2))
        if abs(continuity_bar-continuity_stem)<H.EPS:continue
        costs=np.array([0.,continuity_bar-continuity_stem,continuity_stem-continuity_bar])
        nodes.append(dict(center=int(center),bar=bar,stem=stem_pair,cost=costs,
                          direction=arms[stem],point=np.array([y,x],float)))
    if not nodes:return problem.U.copy(),dict(status='fallback_T_direction_ambiguous')
    for a in range(len(nodes)):
        for b in range(a+1,len(nodes)):
            if np.linalg.norm(nodes[a]['point']-nodes[b]['point'])<=8:links.append((a,b))
    source=np.asarray(ep.r_rgb,float).reshape(-1,3)/255.;scale=max(float(np.median(np.var(source[problem.rvalid],axis=0))),1e-4)
    for node in nodes:node['cost']/=scale
    grid_edges,cap=H.rgb_edges(ep.q_rgb);U=problem.U.ravel();uc=np.clip(U,H.EPS,1-H.EPS);unary=np.log(uc/(1-uc))
    from ics.methods.pro_paired_environment import exact_potts_cut
    def pair_cost(node,state,y):
        if state==0:return 0.
        i,j=node['bar'] if state==1 else node['stem']
        return float(y[i]!=y[j])
    def full_energy(y,states):
        value=float(np.sum(np.logaddexp(0,unary)-unary*y)+cap@(y[grid_edges[:,0]]!=y[grid_edges[:,1]]))
        value+=sum(float(n['cost'][s])+pair_cost(n,s,y) for n,s in zip(nodes,states))
        if control!='independent':
            value+=sum(float(states[a]!=states[b])*.25 for a,b in links if states[a] and states[b])
        return value
    best=U>.5;bestcost=np.inf;cut_calls=0
    for initial in range(3):
        states=np.full(len(nodes),initial,int);y=U>.5
        for _ in range(5):
            for j,node in enumerate(nodes):
                scores=node['cost'].copy()+np.array([pair_cost(node,k,y) for k in range(3)])
                if control!='independent':
                    for a,b in links:
                        other=b if a==j else a if b==j else None
                        if other is not None and states[other]:scores[1:]+=.25*(np.arange(1,3)!=states[other])
                states[j]=int(np.argmin(scores))
            extra=[]
            for node,state in zip(nodes,states):
                if state:extra.append(node['bar'] if state==1 else node['stem'])
            edges=np.concatenate((grid_edges,np.asarray(extra,int).reshape(-1,2)))
            capacity=np.r_[cap,np.ones(len(extra))]
            y,certificate=exact_potts_cut(unary,edges,capacity);cut_calls+=1
            value=full_energy(y,states)
            if value<bestcost:bestcost=value;best=y.copy()
    return best.reshape(problem.U.shape).astype(float),dict(status='ok',binary_optimizer=True,
            T_junctions=len(nodes),compatibility_links=len(links),local_orders=3,cut_calls=cut_calls,
            finite_outer_rounds=5,initializations=3,best_complete_energy=bestcost,
            paid_compatibility_breaks_cycles_allowed=True,only_visible_pixels=True,control=control)


def _laplacian_rgb(rgb):
    value=np.asarray(rgb,float)/255.
    scales=[ndimage.gaussian_filter(value,(s,s,0),mode='nearest') for s in (1,2,4,8)]
    return [scales[j]-scales[j+1] for j in range(3)]


def e234_laplacian_roles(problem,control=None):
    ep=problem.ep;anchors=B._anchors(problem).ravel()
    if not np.any(anchors==1) or not np.any(anchors==-1):return problem.U.copy(),dict(status='fallback_query_double_anchors_missing')
    rp=_laplacian_rgb(ep.r_rgb);qp=_laplacian_rgb(ep.q_rgb);semantic=H.native_to_original(problem.u0.reshape(ep.q_hw),ep.original_shape,ep.query_geometry).ravel()
    valid=problem.rvalid;lab=problem.labels;evidences=[];models=[];source_scales=[]
    for r,q in zip(rp,qp):
        rv=r.reshape(-1,3);qv=q.reshape(-1,3);f=fit_gmm(rv[valid&lab],2);b=fit_gmm(rv[valid&~lab],2)
        if f is None or b is None:evidences.append(np.zeros(len(qv)));models.append(None);source_scales.append(0.);continue
        rs=f.log_density(rv)-b.log_density(rv);scale=max(float(np.median(np.abs(rs[valid]))),H.EPS)
        evidence=f.log_density(qv)-b.log_density(qv)
        # Residuals with identical class explanations are not all added to FG.
        if np.max(np.abs(rs[valid]),initial=0)<=H.EPS:evidence[:]=0
        evidence/=scale
        permitted=(np.abs(semantic-.5)<=.1)|(np.sign(evidence)==np.sign(semantic-.5))
        evidence[~permitted]=0;evidences.append(evidence);models.append((f,b));source_scales.append(scale)
    if not any(np.any(a) for a in evidences):return problem.U.copy(),dict(status='fallback_no_class_explainable_pyramid_residual')
    if control=='direct_all_channels':
        r=np.concatenate([v.reshape(-1,3) for v in rp],1);q=np.concatenate([v.reshape(-1,3) for v in qp],1)
        head=H.fit_pixel_logistic(r,np.zeros((1,1)),np.zeros(len(r),int),lab,valid)
        if head is None:return problem.U.copy(),dict(status='fallback_direct_head_missing_roles')
        p=np.clip(head.predict(q,np.zeros((1,1)),np.zeros(len(q),int)),H.EPS,1-H.EPS)
        evidence=np.log(p/(1-p));out,cert=C._cut(problem,evidence)
        return out,dict(status='ok',binary_optimizer=True,control=control,fit=head.info,cut=cert)
    current=problem.U.copy();total=np.zeros(problem.U.size);trace=[]
    for j in (2,1,0):
        total+=evidences[j]
        out,cert=C._cut(problem,total);trace.append(dict(level=j,foreground_pixels=int(np.sum(out>.5)),cut_energy=cert['energy']))
        current=out
        if control=='fixed_models':continue
        # Local appearance is refit only from initial direct anchors whose
        # current labels agree. Newly changed pixels never authorize a class.
        if models[j] is None:continue
        q=qp[j].reshape(-1,3);fg=(anchors==1)&(out.ravel()>.5);bg=(anchors==-1)&(out.ravel()<=.5)
        if fg.sum()>=8 and bg.sum()>=8:
            f=fit_gmm(q[fg],2);b=fit_gmm(q[bg],2)
            candidate=(f.log_density(q)-b.log_density(q))/source_scales[j]
            permitted=(np.abs(semantic-.5)<=.1)|(np.sign(candidate)==np.sign(semantic-.5))
            candidate[~permitted]=0
            # Same level gets one bounded local-appearance update; all levels
            # share coefficient1 rather than a hidden scale search.
            total+=candidate-evidences[j];current,cert=C._cut(problem,total)
    return current,dict(status='ok',binary_optimizer=True,pyramid_levels=3,RGB_sigmas=[1,2,4,8],
            shared_residual_coefficient=1,source_evidence_scales=source_scales,coarse_to_fine_trace=trace,
            semantic_counterevidence_gate=True,control=control)


def e227_touching_split(problem,control=None):
    ep=problem.ep;labels,count=ndimage.label(problem.U>.5);x=H.texture_phi(ep.q_rgb)
    anchors=B._anchors(problem).ravel();groups=[];split=0;domains=[]
    for j in range(1,count+1):
        pixels=np.flatnonzero(labels.ravel()==j)
        direct=pixels[anchors[pixels]==1]
        if len(direct)<8:continue
        one=_one_gaussian(x[direct]);two=fit_gmm(x[direct],2)
        n=len(direct);parameters=3*x.shape[1]+1
        gain=float(one.log_density(x[direct]).sum()-two.log_density(x[direct]).sum())
        gain=-gain-.5*parameters*np.log(1+n)
        chosen=two if gain>0 else one;split+=int(gain>0)
        if control=='no_split':chosen=one
        groups.append(chosen);domains.append(ndimage.binary_dilation(labels==j,iterations=2).ravel())
    if not groups:return problem.U.copy(),dict(status='fallback_missing_authenticated_candidate_domain')
    bgids=np.flatnonzero(anchors==-1)
    if len(bgids)<8:return problem.U.copy(),dict(status='fallback_missing_negative_anchors')
    background=fit_gmm(x[bgids],2);bg=background.log_density(x);evidence=np.zeros(len(x));coverage=np.zeros(len(x),bool)
    if control=='global':
        total_k=sum(len(m.weights) for m in groups)
        shared=fit_gmm(x[np.concatenate([np.flatnonzero(d&(anchors==1)) for d in domains])],total_k)
        for domain in domains:coverage|=domain
        evidence[coverage]=shared.log_density(x[coverage])-bg[coverage]
    else:
        for model,domain in zip(groups,domains):
            own=model.log_density(x[domain])-bg[domain]
            evidence[domain]=np.where(coverage[domain],np.maximum(evidence[domain],own),own)
            coverage|=domain
    r=H.texture_phi(ep.r_rgb);f=fit_gmm(r[problem.rvalid&problem.labels],2);b=fit_gmm(r[problem.rvalid&~problem.labels],2)
    if f is None or b is None:return problem.U.copy(),dict(status='fallback_source_identity_missing')
    rs=f.log_density(r)-b.log_density(r);evidence,scale=H.normalize_evidence(evidence,rs,problem.rvalid)
    out,cert=C._cut(problem,evidence);out.ravel()[~coverage]=problem.U.ravel()[~coverage]
    return out,dict(status='ok',binary_optimizer=True,candidate_domains=len(groups),accepted_splits=split,
                    final_F_components=sum(len(m.weights) for m in groups),evidence_scale=scale,
                    split_activity_is_FG_BG_union_edits_not_instance_count=True,cut=cert,control=control)


def e228_grouped_adaptation(problem,control=None):
    ep=problem.ep;initial=problem.U>.5;direct=H.native_to_original(problem.u0.reshape(ep.q_hw),ep.original_shape,ep.query_geometry)
    threshold=problem.thresholds[0] if problem.thresholds is not None else 1.
    births=(~initial)&(direct>=threshold)&(direct>.5)
    region,count=ndimage.label(initial|births);anchors=B._anchors(problem).ravel();x=H.texture_phi(ep.q_rgb)
    # Newly observed islands carry original direct semantic residual; RGB
    # reconstruction alone never opens another foreground model.
    fg_initial=anchors==1;fg_initial|=births.ravel()
    models=[];domains=[];overflow=[];closed=0
    for j in range(1,count+1):
        island=region==j;points=np.flatnonzero(island.ravel())
        if len(models)>=16:overflow.append(points);continue
        fg=points[fg_initial[points]];halo=ndimage.binary_dilation(island,iterations=4)
        bg=np.flatnonzero((halo&~island).ravel()&(anchors==-1))
        if len(fg)<8 or len(bg)<8:closed+=1;continue
        models.append((fit_gmm(x[fg],2),fit_gmm(x[bg],2)))
        domains.append(halo.ravel())
    if not models:return problem.U.copy(),dict(status='fallback_no_two_anchor_island_model',closed_models=closed)
    r=H.texture_phi(ep.r_rgb);rf=fit_gmm(r[problem.rvalid&problem.labels],2);rb=fit_gmm(r[problem.rvalid&~problem.labels],2)
    if rf is None or rb is None:return problem.U.copy(),dict(status='fallback_source_identity_missing')
    scale=max(float(np.median(np.abs(rf.log_density(r)-rb.log_density(r))[problem.rvalid])),H.EPS)
    covered=np.any(np.stack(domains),axis=0);evidence=np.zeros(len(x))
    if control=='independent':
        masks=[]
        for (f,b),domain in zip(models,domains):
            ev=np.zeros(len(x));ev[domain]=(f.log_density(x[domain])-b.log_density(x[domain]))/scale
            m,_=C._cut(problem,ev);m.ravel()[~domain]=0;masks.append(m>.5)
        out=np.any(np.stack(masks),axis=0).astype(float)
        out.ravel()[~covered]=problem.U.ravel()[~covered];cert=dict(control='union_of_same_island_cuts')
    elif control=='global':
        allfg=np.flatnonzero(fg_initial);allbg=np.flatnonzero(anchors==-1)
        f=fit_gmm(x[allfg],2*len(models));b=fit_gmm(x[allbg],2*len(models))
        evidence[covered]=(f.log_density(x[covered])-b.log_density(x[covered]))/scale
        out,cert=C._cut(problem,evidence);out.ravel()[~covered]=problem.U.ravel()[~covered]
    else:
        # Equal normalized model priors on both roles. Duplicating a model is
        # not an extra foreground vote; there is no instance-count prior.
        fl=np.full((len(models),covered.sum()),-np.inf);bl=fl.copy();points=np.flatnonzero(covered)
        for j,((f,b),domain) in enumerate(zip(models,domains)):
            enabled=domain[points];fl[j,enabled]=f.log_density(x[points[enabled]])
            bl[j,enabled]=b.log_density(x[points[enabled]])
        active=np.stack(domains)[:,points].sum(0)
        evidence[points]=(logsumexp(fl,axis=0)-np.log(active)-logsumexp(bl,axis=0)+np.log(active))/scale
        out,cert=C._cut(problem,evidence);out.ravel()[~covered]=problem.U.ravel()[~covered]
    for points in overflow:out.ravel()[points]=problem.U.ravel()[points]
    return out,dict(status='ok',binary_optimizer=True,island_models=len(models),initial_or_direct_birth_islands=count,
                    disabled_weak_anchor_models=closed,unmodeled_islands_host_preserved=len(overflow),
                    birth_pixels=int(births.sum()),normalized_equal_role_model_priors=True,cut=cert,control=control)


METHODS={
 'E226':lambda ep:B._call(ep,'E226',e226_visible_t_layers),
 'E227':lambda ep:B._call(ep,'E227',e227_touching_split),
 'E228':lambda ep:B._call(ep,'E228',e228_grouped_adaptation),
 'E229':lambda ep:B._call(ep,'E229',e229_subpatch_integral),
 'E234':lambda ep:B._call(ep,'E234',e234_laplacian_roles),
}
CONTROLS={
 'E226_independent_T_orders':lambda ep:B._call(ep,'E226',e226_visible_t_layers,control='independent'),
 'E227_no_split_GrabCut':lambda ep:B._call(ep,'E227',e227_touching_split,control='no_split'),
 'E227_same_components_global_F_GMM':lambda ep:B._call(ep,'E227',e227_touching_split,control='global'),
 'E228_same_islands_independent_cuts':lambda ep:B._call(ep,'E228',e228_grouped_adaptation,control='independent'),
 'E228_same_components_global_GMM':lambda ep:B._call(ep,'E228',e228_grouped_adaptation,control='global'),
 'E229_independent_line_states':lambda ep:B._call(ep,'E229',e229_subpatch_integral,control='independent'),
 'E229_same_coverage_pixel_quota':lambda ep:B._call(ep,'E229',e229_subpatch_integral,control='pixel_quota'),
 'E234_all_pyramid_channels_direct_head':lambda ep:B._call(ep,'E234',e234_laplacian_roles,control='direct_all_channels'),
 'E234_source_models_no_local_adaptation':lambda ep:B._call(ep,'E234',e234_laplacian_roles,control='fixed_models'),
}
RESOURCES={id:dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=True,extra_encoder_forwards=0) for id in METHODS}
