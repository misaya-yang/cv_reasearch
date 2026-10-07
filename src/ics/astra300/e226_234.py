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


def e233_scale_region_tree(problem,control=None):
    ep=problem.ep;shape=problem.U.shape;p=problem.U.size;edges=H.grid_edges(shape)
    rgb=np.asarray(ep.q_rgb,float)/255.;persistence=np.zeros(len(edges),int);edge_orders=[];birth=[]
    for sigma in (1,2,4,8):
        smooth=ndimage.gaussian_filter(rgb,(sigma,sigma,0),mode='nearest').reshape(-1,3)
        distance=np.mean((smooth[edges[:,0]]-smooth[edges[:,1]])**2,1)
        threshold=max(float(np.median(distance)),1e-8)
        persistence+=(distance>threshold)
        ids=np.flatnonzero(distance<=threshold);edge_orders.append(ids[np.argsort(distance[ids],kind='stable')]);birth.append(sigma)
    semantic=H.native_to_original(problem.u0.reshape(ep.q_hw),shape,ep.query_geometry).ravel()>.5
    authentic=(semantic[edges[:,0]]!=semantic[edges[:,1]])&(persistence>=3)
    if not authentic.any() and control is None:return problem.U.copy(),dict(status='fallback_no_persistent_opposed_semantic_interface')
    # Successive observed RGB mergers create a strictly laminar tree. The
    # four independent RGB segmentations are never falsely assumed nested.
    parent=np.arange(p);component_node=np.arange(p);nodes=[];cue=np.zeros((2*p,2));sizes=np.ones(2*p)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    def join(edge_index,level):
        i,j=map(int,edges[edge_index]);a,b=find(i),find(j)
        if a==b:return
        left,right=int(component_node[a]),int(component_node[b]);node=p+len(nodes)
        nodes.append((left,right,int(level)));sizes[node]=sizes[left]+sizes[right]
        if authentic[edge_index] and control!='no_persistence':
            reward=persistence[edge_index]/4.
            cue[left,int(semantic[i])]=max(cue[left,int(semantic[i])],reward)
            cue[right,int(semantic[j])]=max(cue[right,int(semantic[j])],reward)
        parent[b]=a;component_node[a]=node
    for level,order in enumerate(edge_orders):
        for index in order:join(index,level)
    # A forest root per remaining component remains legitimate; joining all
    # leftover RGB edges only supplies a full-cover root, not another scale.
    for index in np.arange(len(edges)):join(index,4)
    root=int(component_node[find(0)]);n=p+len(nodes);u=np.clip(problem.U.ravel(),H.EPS,1-H.EPS)
    costs=np.zeros((n,2));costs[:p,0]=-np.log(1-u);costs[:p,1]=-np.log(u)
    for k,(a,b,_) in enumerate(nodes,start=p):costs[k]=costs[a]+costs[b]
    # Region complexity is an explicit bounded assumption; it is shared with
    # the same-tree nonpersistent control rather than hidden in a solver.
    uniform=costs+(np.log1p(sizes[:n])/8)[:,None]-cue[:n]
    best=uniform.min(1);choice=uniform.argmin(1).astype(np.int8)
    for k,(a,b,_) in enumerate(nodes,start=p):
        split=best[a]+best[b]
        if split<best[k]:best[k]=split;choice[k]=2
    output=np.zeros(p,bool);stack=[root]
    while stack:
        node=stack.pop();state=int(choice[node])
        if node<p:output[node]=state==1;continue
        a,b,_=nodes[node-p]
        if state==2:stack.extend((a,b));continue
        pending=[node]
        while pending:
            j=pending.pop()
            if j<p:output[j]=state==1
            else:pending.extend(nodes[j-p][:2])
    return output.reshape(shape).astype(float),dict(status='ok',binary_optimizer=True,RGB_scales=[1,2,4,8],
                tree_nodes=n,fine_pixel_candidates=p,all_fine_candidates_retained=True,
                persistent_opposed_edges=int(authentic.sum()),DP_additive_region_objective=float(best[root]),
                complexity_penalty='log(1+region_pixels)/8 shared with control',
                finite_region_cover_not_global_binary_Potts_optimum=True,control=control)


def e230_optical_boundary(problem,control=None):
    ep=problem.ep;r=np.asarray(ep.r_rgb,float)/255.;q=np.asarray(ep.q_rgb,float)/255.;rm=np.asarray(ep.reference_mask,bool)
    valid=problem.rvalid.reshape(rm.shape);lab=rm.ravel();flat=r.reshape(-1,3)
    if not np.any(valid.ravel()&lab) or not np.any(valid.ravel()&~lab):return problem.U.copy(),dict(status='fallback_source_colour_roles_missing')
    h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw);loss=np.zeros(4);folds=0
    boundary=rm!=ndimage.binary_erosion(rm)
    band=ndimage.binary_dilation(boundary,iterations=16)
    for by in range(2):
        for bx in range(2):
            held=((yy>=by*h//2)&(yy<(by+1)*h//2)&(xx>=bx*w//2)&(xx<(bx+1)*w//2))
            excluded=ndimage.binary_dilation(held,structure=np.ones((3,3))).ravel()[problem.rtokens]
            train=valid.ravel()&~excluded;test=held.ravel()[problem.rtokens]&valid.ravel()&band.ravel()
            if np.sum(train&lab)<8 or np.sum(train&~lab)<8 or not test.any():continue
            foreground=np.median(flat[train&lab],0);background=np.median(flat[train&~lab],0)
            for j,sigma in enumerate((0,1,2,4)):
                a=rm.astype(float) if sigma==0 else ndimage.gaussian_filter(rm.astype(float),sigma,mode='nearest')
                prediction=background+a[...,None]*(foreground-background)
                loss[j]+=float(np.mean(np.sum((r.reshape(-1,3)[test]-prediction.reshape(-1,3)[test])**2,1)))
            folds+=1
    if folds<2:return problem.U.copy(),dict(status='fallback_insufficient_whole_source_boundary_blocks')
    chosen=1+int(np.argmin(loss[1:]));sigma=(0,1,2,4)[chosen]
    if control is None and loss[chosen]>=loss[0]-1e-4:
        return problem.U.copy(),dict(status='fallback_no_source_optical_blur_evidence',source_PSFloss=loss.tolist())
    fg=np.median(flat[valid.ravel()&lab],0);bg=np.median(flat[valid.ravel()&~lab],0)
    noise=max(float(np.median(np.sum((flat-np.where(lab[:,None],fg,bg))**2,1)[valid.ravel()])),1e-4)
    if control is None and loss[chosen]/folds>4*noise:
        return problem.U.copy(),dict(status='fallback_source_two_colour_PSF_model_mismatch',source_PSFloss=loss.tolist(),source_RGB_noise=noise)
    initial=problem.U>.5;edge=initial!=ndimage.binary_erosion(initial)
    centers=B._spread_rows(np.flatnonzero(edge),128)
    if not len(centers):return problem.U.copy(),dict(status='fallback_no_current_boundary_segment')
    signed=ndimage.distance_transform_edt(initial)-ndimage.distance_transform_edt(~initial)
    gy,gx=np.gradient(signed);mag=np.hypot(gy,gx);radius=max(4,4*sigma);patches=[];options=[];costs=[]
    U=np.clip(problem.U,H.EPS,1-H.EPS)
    for center in centers:
        y,x=divmod(int(center),initial.shape[1]);y0,y1=max(0,y-radius),min(initial.shape[0],y+radius+1);x0,x1=max(0,x-radius),min(initial.shape[1],x+radius+1)
        ids=np.arange(initial.size).reshape(initial.shape)[y0:y1,x0:x1];local=initial[y0:y1,x0:x1]
        ny=gy[y,x]/max(mag[y,x],H.EPS);nx=gx[y,x]/max(mag[y,x],H.EPS);states=[];statecost=[]
        for displacement in (-2*sigma,-sigma,0,sigma,2*sigma):
            binary=ndimage.shift(local.astype(float),(displacement*ny,displacement*nx),order=0,mode='nearest')>.5
            # The same actual optical kernel evaluates every latent sharp
            # boundary. It is unrelated to the DINO feature footprint.
            alpha=binary.astype(float) if control=='no_blur' else ndimage.gaussian_filter(binary.astype(float),sigma,mode='nearest')
            prediction=bg+alpha[...,None]*(fg-bg)
            rgb_cost=float(np.mean(np.sum((q[y0:y1,x0:x1]-prediction)**2,2))/noise)
            unary=float(np.mean(np.where(binary,-np.log(U[y0:y1,x0:x1]),-np.log(1-U[y0:y1,x0:x1]))))
            states.append(binary);statecost.append(rgb_cost+unary)
        if control is None and min(statecost)>5:
            continue
        patches.append(ids);options.append(states);costs.append(np.array(statecost))
    if not patches:return problem.U.copy(),dict(status='fallback_query_PSF_model_mismatch_all_segments',source_PSFloss=loss.tolist())
    choices=np.full(len(patches),2,int)
    # Overlapping boundary windows are coordinated by the actual disagreement
    # on their shared original pixels, with no latent object shape completion.
    adjacency=[[] for _ in patches]
    for i in range(len(patches)):
        for j in range(i+1,len(patches)):
            common,ia,ib=np.intersect1d(patches[i].ravel(),patches[j].ravel(),return_indices=True)
            if len(common):adjacency[i].append((j,ia,ib));adjacency[j].append((i,ib,ia))
    for _ in range(5):
        for i in range(len(patches)):
            score=costs[i].copy()
            for j,a,b in adjacency[i]:
                target=options[j][choices[j]].ravel()[b]
                score+=np.array([np.mean(state.ravel()[a]!=target)/8 for state in options[i]])
            choices[i]=int(np.argmin(score))
    votes=np.zeros(initial.size);mass=np.zeros(initial.size)
    for ids,states,choice in zip(patches,options,choices):
        np.add.at(votes,ids.ravel(),states[choice].ravel());np.add.at(mass,ids.ravel(),1)
    out=problem.U.ravel().copy();covered=mass>0;out[covered]=votes[covered]/mass[covered]
    return out.reshape(initial.shape),dict(status='ok',source_PSFloss=loss.tolist(),valid_source_boundary_folds=folds,
                    optical_PSF_sigma_original_pixels=sigma,source_RGB_noise=noise,boundary_segments=len(patches),
                    latent_normal_displacements=[-2*sigma,-sigma,0,sigma,2*sigma],coordinate_rounds=5,
                    overlap_consistency=True,uncovered_exact_U=True,control=control)


def e234_laplacian_roles(problem,control=None):
    ep=problem.ep;anchors=B._anchors(problem).ravel()
    rp=_laplacian_rgb(ep.r_rgb);qp=_laplacian_rgb(ep.q_rgb);semantic=H.native_to_original(problem.u0.reshape(ep.q_hw),ep.original_shape,ep.query_geometry).ravel()
    valid=problem.rvalid;lab=problem.labels;evidences=[];models=[];source_scales=[]
    if control=='direct_all_channels':
        r=np.concatenate([v.reshape(-1,3) for v in rp],1);q=np.concatenate([v.reshape(-1,3) for v in qp],1)
        head=H.fit_pixel_logistic(r,np.zeros((1,1)),np.zeros(len(r),int),lab,valid)
        if head is None:return problem.U.copy(),dict(status='fallback_direct_head_missing_roles')
        p=np.clip(head.predict(q,np.zeros((1,1)),np.zeros(len(q),int)),H.EPS,1-H.EPS)
        evidence=np.log(p/(1-p));out,cert=C._cut(problem,evidence)
        return out,dict(status='ok',binary_optimizer=True,control=control,fit=head.info,cut=cert)
    if not np.any(anchors==1) or not np.any(anchors==-1):return problem.U.copy(),dict(status='fallback_query_double_anchors_missing')
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
 'E233':lambda ep:B._call(ep,'E233',e233_scale_region_tree),
 'E230':lambda ep:B._call(ep,'E230',e230_optical_boundary),
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
 'E233_same_tree_without_persistence':lambda ep:B._call(ep,'E233',e233_scale_region_tree,control='no_persistence'),
 'E230_same_RGB_without_PSF':lambda ep:B._call(ep,'E230',e230_optical_boundary,control='no_blur'),
 'E230_ordinary_RGB_GMM':lambda ep:B._call(ep,'E230',B.e250_dichromatic,control='gmm'),
}
RESOURCES={id:dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=True,extra_encoder_forwards=0) for id in METHODS}
