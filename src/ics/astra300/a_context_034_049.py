"""Supplied A spatial/RGB/source-compression rules, with explicit source folds."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.special import expit, logsumexp

from .a_helpers_001_050 import (EPS, Frame, ObservationUnavailable, b0,
    fit_rbf, fit_ridge, fps, nearest_mean_distance, rank_basis, sqdist, unit)


def _plain(frame):
    return lambda q,ids,role:b0(frame,q), {'mechanism':'A_B0_5NN'}


# Clockwise physical directions, fixed independently of labels and query scores.
OFFSETS=((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))


def ring_indices(hw,valid):
    h,w=hw; y,x=np.indices(hw); indices=np.full((h*w,8),-1,int)
    for k,(dy,dx) in enumerate(OFFSETS):
        yy=y.ravel()+dy;xx=x.ravel()+dx
        keep=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
        neighbor=yy[keep]*w+xx[keep]
        rows=np.flatnonzero(keep);keep2=np.asarray(valid)[neighbor]>0
        indices[rows[keep2],k]=neighbor[keep2]
    indices[np.asarray(valid)<=0]=-1
    return indices


def _role_grid(frame,role):
    if role=='r':return frame.x,frame.ep.r_hw,frame.ep.wvalid
    return np.asarray(frame.ep.q,float),frame.ep.q_hw,frame.ep.q_valid


def _rings(frame,role,ids,u):
    x,hw,valid=_role_grid(frame,role);nn=ring_indices(hw,valid)[ids]
    visible=nn>=0;center=x[ids]@u
    delta=(x[np.maximum(nn,0)]@u-center[:,None,:])*visible[:,:,None]
    return delta,visible


def _support_radius(frame):
    values=[]
    for block in range(4):
        held=frame.fids[frame.blocks[frame.fids]==block]
        train=frame.fids[frame.blocks[frame.fids]!=block]
        if len(held) and len(train):
            bank=frame.x[fps(frame.x,train,64)]
            values.extend(nearest_mean_distance(frame.x[held],bank).tolist())
    return max(float(np.quantile(values,.95)) if values else 2.,EPS)


def _bag_descriptor(frame,role,ids,u):
    x,_,_=_role_grid(frame,role);diff,visible=_rings(frame,role,ids,u)
    count=visible.sum(1);mean=diff.sum(1)/np.maximum(count[:,None],1)
    centered=(diff-mean[:,None])*visible[:,:,None]
    covariance=np.einsum('nki,nkj->nij',centered,centered)/np.maximum(count[:,None,None],1)
    return np.c_[x[ids],mean,covariance.reshape(len(ids),u.shape[1]**2)]


def fit_ring_bag_rbf(frame,all_profiles=False):
    ids=np.flatnonzero(frame.train)
    u=rank_basis(frame.x[ids]-frame.x[ids].mean(0),16)
    if not u.shape[1] or not len(frame.fids) or not len(frame.bids):return _plain(frame)
    rr=_bag_descriptor(frame,'r',np.arange(len(frame.x)),u)
    if all_profiles:
        # Every valid training center and unordered bag contributes a column;
        # support for the supervised readout remains <=64 per role.
        bank=rr[ids];width=max(float(np.median(sqdist(bank,bank))),EPS)
        features=np.exp(-sqdist(rr,bank)/width);model=fit_rbf(frame,features)
        if model is None:return _plain(frame)
        z,coef,w=model
        def predict(q,indices,role):
            qq=_bag_descriptor(frame,role,indices,u)
            profile=np.exp(-sqdist(qq,bank)/width)
            return np.exp(-sqdist(profile,z)/w)@coef,{}
    else:
        coef=fit_ridge(frame,rr)
        def predict(q,indices,role):
            qq=_bag_descriptor(frame,role,indices,u)
            return qq@coef[:-1]+coef[-1],{}
    return predict,{'control':'center_plus_same_neighbor_mean_covariance'+('_all_affinity_RBF' if all_profiles else '_ridge'),
                    'PCA_rank':u.shape[1]}


def fit_a043(frame,shuffle=False):
    f,b,fi,bi,_=frame.banks()
    ids=np.flatnonzero(frame.train)
    u=rank_basis(frame.x[ids]-frame.x[ids].mean(0),16)
    if not u.shape[1] or not len(frame.fids) or not len(frame.bids):return _plain(frame)
    anchor_ids=np.r_[fi,bi];delta,visible=_rings(frame,'r',anchor_ids,u)
    permutation=np.random.default_rng(0).permutation(8) if shuffle else np.arange(8)
    delta=delta[:,permutation];visible=visible[:,permutation]
    def predict(q,indices,role):
        query,qvisible=_rings(frame,role,indices,u)
        query=query[:,permutation];qvisible=qvisible[:,permutation]
        distance=sqdist(q,frame.x[anchor_ids]);best=np.full(distance.shape,np.inf)
        for rotation in range(8):
            ring=np.roll(query,rotation,axis=1);seen=np.roll(qvisible,rotation,axis=1)
            for j in range(len(anchor_ids)):
                common=seen&visible[j];count=common.sum(1)
                value=distance[:,j]+np.sum((ring-delta[j])**2*common[:,:,None],axis=(1,2))/np.maximum(count,1)
                best[count>0,j]=np.minimum(best[count>0,j],value[count>0])
        fg=np.min(best[:,:len(fi)],1);bg=np.min(best[:,len(fi):],1)
        fallback=~(np.isfinite(fg)&np.isfinite(bg))
        score=np.zeros(len(q));score[~fallback]=bg[~fallback]-fg[~fallback]
        original,_=b0(frame,q);score[fallback]=original[fallback]
        return score,{'_fallback_mask':fallback,'no_common_neighbor_points':int(fallback.sum())}
    return predict,{'PCA_rank':u.shape[1],'anchor_counts':[len(fi),len(bi)],
        'clockwise_offsets':OFFSETS,'rotations':8,'joint_visible_count_normalization':True,
        'same_fixed_non_cyclic_permutation_control':shuffle}


def angular_code(response):
    """Joint-anchor canonical cyclic order, rank bits plus adjacent-rise bits."""
    response=np.asarray(response,float) # [center, 8 directions, anchors]
    order=np.argsort(response,axis=1,kind='stable')
    ranks=np.argsort(order,axis=1,kind='stable').astype(np.uint8)
    # Equal values share their first ordinal position: a constant ring is not
    # assigned an artificial order by the stable sort.
    for i in range(8):
        for j in range(i):
            equal=response[:,i]==response[:,j]
            ranks[:,i]=np.where(equal,np.minimum(ranks[:,i],ranks[:,j]),ranks[:,i])
    bits=np.concatenate([((ranks[...,None]>>np.arange(3))&1),
                         (np.roll(response,-1,axis=1)>response)[...,None]],axis=-1)
    flat=bits.reshape(len(response),8,-1).astype(np.uint8)
    canonical=flat.copy()
    # Lexicographic minimization is joint across anchors, preserving a physical
    # rotation rather than independently aligning each anchor's ring.
    for shift in range(1,8):
        candidate=np.roll(flat,shift,axis=1)
        a=candidate.reshape(len(response),-1);z=canonical.reshape(len(response),-1)
        different=a!=z;first=np.argmax(different,axis=1)
        choose=different.any(1)&(a[np.arange(len(a)),first]<z[np.arange(len(z)),first])
        canonical[choose]=candidate[choose]
    return canonical.reshape(len(response),-1),np.max(np.ptp(response,axis=1),axis=1)<=1e-12


def fit_a044(frame,shuffle=False,bag=False):
    f,b,fi,bi,_=frame.banks()
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    rnn=ring_indices(frame.ep.r_hw,frame.ep.wvalid)
    anchor_ids=np.r_[fi,bi];permutation=np.random.default_rng(0).permutation(8) if shuffle else np.arange(8)
    def describe(role,indices):
        x,hw,valid=_role_grid(frame,role);nn=ring_indices(hw,valid)[indices]
        response=x[np.maximum(nn,0)]@f.T;response=response[:,permutation]
        if bag:
            return np.c_[x[indices]@f.T,response.mean(1),response.var(1),np.min(response,1),np.max(response,1)],np.zeros(len(indices),bool)
        code,constant=angular_code(response)
        return code,(nn<0).any(1)|constant
    rc,invalid=describe('r',anchor_ids);keep=~invalid
    fg=rc[:len(fi)][keep[:len(fi)]];bg=rc[len(fi):][keep[len(fi):]]
    if not len(fg) or not len(bg):return _plain(frame)
    if bag:
        descriptors,_=describe('r',np.arange(len(frame.x)));coef=fit_ridge(frame,descriptors)
        return lambda q,ids,role:(describe(role,ids)[0]@coef[:-1]+coef[-1],{}),{'control':'same_anchor_center_ring_bag_ridge'}
    def code_distance(qcode,bank):
        out=np.full(len(qcode),np.inf)
        for start in range(0,len(qcode),64):
            differences=qcode[start:start+64,None]!=bank[None]
            out[start:start+64]=differences.mean(2).min(1)
        return out
    radius=_support_radius(frame)
    def predict(q,indices,role):
        code,invalid=describe(role,indices)
        score=code_distance(code,bg)-code_distance(code,fg)
        score=np.minimum(score,(radius-nearest_mean_distance(q,f))/radius)
        original,_=b0(frame,q);score[invalid]=original[invalid]
        return score,{'_fallback_mask':invalid,'constant_or_incomplete_ring':int(invalid.sum())}
    return predict,{'FG_response_anchors':len(f),'code_bank_counts':[len(fg),len(bg)],
        'bit_weights':'equal 3 ordinal bits and one adjacent-rise bit per anchor/direction',
        'FG_support_radius2':radius,'non_cyclic_permutation_control':shuffle}


def _context_descriptor(frame,role,ids,u):
    x,_,_=_role_grid(frame,role);delta,visible=_rings(frame,role,ids,u)
    projected=x[ids]@u;neighbor=delta+projected[:,None,:]
    # Mean context plus four opposite-direction differences. This fixed
    # low-dimensional representation is identical in main and direct controls.
    mean=np.sum(neighbor*visible[:,:,None],1)/np.maximum(visible.sum(1)[:,None],1)
    opposite=neighbor[:,:4]-neighbor[:,4:]
    return np.c_[mean,opposite.reshape(len(ids),-1)],visible.all(1)


def fit_a045(frame,direct=False,anomaly=False,foreground_predictor=False,affinity=False):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    known=np.flatnonzero(frame.train);u=rank_basis(frame.x[known]-frame.x[known].mean(0),16)
    if u.shape[1]<2:return _plain(frame)
    rr,interior=_context_descriptor(frame,'r',np.arange(len(frame.x)),u)
    source=frame.fids if foreground_predictor else frame.bids;source=source[interior[source]]
    if len(source)<2 or np.linalg.matrix_rank(rr[source]-rr[source].mean(0))<2:return _plain(frame)
    design=np.c_[rr[source],np.ones(len(source))]
    coef=np.linalg.solve(design.T@design+np.eye(design.shape[1]),design.T@frame.x[source])
    residual=frame.x-np.c_[rr,np.ones(len(rr))]@coef
    ff=residual[frame.fids];bb=residual[frame.bids]
    radius=_support_radius(frame);f,*_=frame.banks()
    if direct:
        descriptors=np.c_[frame.x,rr];head=fit_ridge(frame,descriptors)
    elif affinity:
        training=residual[known];scale=max(float(np.median(sqdist(training,training))),EPS)
        descriptors=np.c_[frame.x,np.exp(-sqdist(residual,training)/scale)]
        head=fit_ridge(frame,descriptors)
    else:head=None
    anomaly_cut=float(np.quantile(np.sum(bb*bb,1),.95))
    def predict(q,ids,role):
        context,inside=_context_descriptor(frame,role,ids,u)
        e=q-np.c_[context,np.ones(len(q))]@coef
        if direct:score=np.c_[q,context]@head[:-1]+head[-1]
        elif affinity:score=np.c_[q,np.exp(-sqdist(e,training)/scale)]@head[:-1]+head[-1]
        elif anomaly:score=np.sum(e*e,1)-anomaly_cut
        else:score=nearest_mean_distance(e,bb)-nearest_mean_distance(e,ff)
        if not direct and not affinity:score=np.minimum(score,(radius-nearest_mean_distance(q,f))/radius)
        original,_=b0(frame,q);score[~inside]=original[~inside]
        return score,{'_fallback_mask':~inside,'edge_fallback_points':int((~inside).sum())}
    return predict,{'context_PCA_rank':u.shape[1],'predictor_role':'FG_control' if foreground_predictor else 'BG',
        'predictor_source_points':len(source),'predictor_L2':1.,'FG_support_radius2':radius,
        'direct_context_ridge_control':direct,'residual_anomaly_control':anomaly,'all_residual_affinity_control':affinity}


def fit_a049(frame,equal_fps=False):
    fi=fps(frame.x,frame.fids,32);bi=fps(frame.x,frame.bids,32)
    if not len(fi) or not len(bi):return _plain(frame)
    candidate=np.r_[fi,bi];labels=np.r_[np.ones(len(fi),bool),np.zeros(len(bi),bool)]
    distances=sqdist(frame.x,frame.x[candidate]);selected=np.arange(len(candidate))
    validation=[]
    for block in range(4):
        held=np.flatnonzero(frame.train&(frame.blocks==block)&np.isin(np.arange(len(frame.x)),candidate,invert=True))
        use=selected[frame.blocks[candidate]!=block]
        if len(held) and (frame.wf[held].sum()>0) and (frame.wb[held].sum()>0) and labels[use].any() and (~labels[use]).any():
            validation.append((block,held))
    def risks(chosen):
        out=[]
        for block,held in validation:
            use=chosen[frame.blocks[candidate[chosen]]!=block]
            f=use[labels[use]];b=use[~labels[use]]
            if not len(f) or not len(b):return None
            df=np.partition(distances[held][:,f],min(5,len(f))-1,axis=1)[:,:min(5,len(f))].mean(1)
            db=np.partition(distances[held][:,b],min(5,len(b))-1,axis=1)[:,:min(5,len(b))].mean(1)
            pred=db>df
            out.append(.5*float(np.sum(frame.wf[held]*(~pred)))/frame.wf[held].sum()+
                       .5*float(np.sum(frame.wb[held]*pred))/frame.wb[held].sum())
        return np.array(out)
    current=risks(selected);history=[];stop='no_effective_inner_validation'
    if len(validation)>=2 and current is not None:
        while len(selected)>2:
            options=[]
            for deleted in selected:
                proposed=selected[selected!=deleted]
                if not labels[proposed].any() or not (~labels[proposed]).any():continue
                risk=risks(proposed)
                if risk is not None:options.append((float((risk-current).sum()),int(candidate[deleted]),deleted,proposed,risk))
            if not options:stop='no_legal_role_preserving_deletion';break
            _,_,deleted,proposed,risk=min(options,key=lambda v:(v[0],v[1]))
            if np.any(risk>current+1e-12):stop='any_block_error_increased';break
            history.append({'removed_native_id':int(candidate[deleted]),'block_errors':risk.tolist()})
            selected=proposed;current=risk;stop='minimum_one_anchor_per_role'
    chosen=candidate[selected];fg=chosen[labels[selected]];bg=chosen[~labels[selected]]
    if equal_fps:
        fg=fps(frame.x,frame.fids,len(fg));bg=fps(frame.x,frame.bids,len(bg))
    def predict(q,ids,role):
        return (nearest_mean_distance(q,frame.x[bg])-nearest_mean_distance(q,frame.x[fg]))/2,{}
    return predict,{'initial_anchor_counts':[len(fi),len(bi)],'core_anchor_counts':[len(fg),len(bg)],
        'core_native_ids':[fg.tolist(),bg.tolist()],'effective_source_blocks':len(validation),
        'deletion_trace':history,'stopping_reason':stop,'same_final_count_FPS_control':equal_fps,
        'inner_validation_anchor_positions_excluded':True,'held_block_anchor_positions_excluded':True}


_RGB_CACHE={}


def rgb_statistics(ep,role):
    """Handcrafted patch statistics on the recorded encoder RGB view, no model."""
    from .common import array_hash
    from ics.cpu100.common import rgb_view
    image=ep.r_rgb if role=='r' else ep.q_rgb
    geometry=ep.reference_geometry if role=='r' else ep.query_geometry
    hw=ep.r_hw if role=='r' else ep.q_hw
    if image is None or not geometry:
        raise ObservationUnavailable('Original RGB plus physical encoder transform required for RGB statistics')
    key=(role,array_hash(image),str(sorted(geometry.items())),tuple(hw))
    if key in _RGB_CACHE:return _RGB_CACHE[key]
    canvas=rgb_view(ep,role).astype(float)/255
    side=int(geometry['view_side']);sh,sw=map(int,geometry['resized_hw']);oy,ox=map(int,geometry['padding_top_left'])
    real=np.zeros((side,side),bool);real[oy:oy+sh,ox:ox+sw]=True
    grey=canvas@np.array((.299,.587,.114));gy,gx=np.gradient(grey)
    magnitude=np.hypot(gx,gy);orientation=(np.arctan2(gy,gx)+2*np.pi)%(2*np.pi)
    safe=ndimage.binary_erosion(real,structure=np.ones((3,3)),border_value=0)
    output=np.zeros((int(np.prod(hw)),19))
    for y in range(hw[0]):
        for x in range(hw[1]):
            y0,y1=int(y*side/hw[0]),int((y+1)*side/hw[0]);x0,x1=int(x*side/hw[1]),int((x+1)*side/hw[1])
            region=np.s_[y0:y1,x0:x1];pixels=canvas[region][real[region]]
            if not len(pixels):continue
            color=np.quantile(pixels,[.1,.5,.9],axis=0).ravel()
            selected=safe[region];angle=orientation[region][selected];mag=magnitude[region][selected]
            hist=np.bincount(np.minimum((angle*8/(2*np.pi)).astype(int),7),weights=mag,minlength=8)
            hist/=max(float(hist.sum()),EPS)
            output[y*hw[1]+x]=np.r_[color,hist,float(mag.mean()) if len(mag) else 0.,float(mag.std()) if len(mag) else 0.]
    # Only the two current RGB descriptor arrays are cached, never image pools.
    if len(_RGB_CACHE)>=2:_RGB_CACHE.clear()
    output.setflags(write=False);_RGB_CACHE[key]=output
    return output


def _rgb_scaler(frame):
    rr=rgb_statistics(frame.ep,'r');source=rr[frame.train]
    center=np.median(source,axis=0);scale=np.median(np.abs(source-center),axis=0)
    # A constant source statistic cannot legitimately determine huge query
    # distance; use physical scale1 rather than dividing by a tiny arbitrary eps.
    scale=np.where(scale>1e-8,scale,1.)
    return rr,(rr-center)/scale,center,scale


def _pair_direction(frame,fg,bg):
    delta=frame.x[fg]-frame.x[bg]
    # Ridge on antisymmetric endpoint differences with a fixed +1 target.
    w=delta.T@np.linalg.solve(delta@delta.T+np.eye(len(delta)),np.ones(len(delta)))
    bias=-float(np.mean((frame.x[fg]+frame.x[bg])/2@w))
    return w,bias


def fit_a041(frame,dino_pairs=False,shuffled=False,direct=False,conditioned=False):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    raw,appearance,center,scale=_rgb_scaler(frame)
    fg=fps(frame.x,frame.fids,64);bgall=frame.bids
    if direct:
        head=fit_ridge(frame,np.c_[frame.x,appearance])
        def predict(q,ids,role):
            low=appearance[ids] if role=='r' else (rgb_statistics(frame.ep,'q')[ids]-center)/scale
            return np.c_[q,low]@head[:-1]+head[-1],{}
        return predict,{'control':'same_RGB19_plus_DINO_direct_ridge','RGB_statistic_dimensions':19}
    if conditioned:
        source=np.flatnonzero(frame.train);design=np.c_[appearance[source],np.ones(len(source))]
        nuisance=np.linalg.solve(design.T@design+np.eye(design.shape[1]),design.T@frame.x[source])
        rr=frame.x-np.c_[appearance,np.ones(len(appearance))]@nuisance
        _,singular,vt=np.linalg.svd(rr[source]-rr[source].mean(0),full_matrices=False)
        variance=singular**2/max(len(source),1);floor=max(float(variance.sum()/frame.x.shape[1])*.1,1e-6)
        axes=vt.T;factor=1/np.sqrt(variance+floor)-1/np.sqrt(floor)
        def whiten(value):return value/np.sqrt(floor)+(value@axes*factor)@axes.T
        head=fit_ridge(frame,whiten(rr))
        def predict(q,ids,role):
            low=appearance[ids] if role=='r' else (rgb_statistics(frame.ep,'q')[ids]-center)/scale
            residual=q-np.c_[low,np.ones(len(q))]@nuisance
            return whiten(residual)@head[:-1]+head[-1],{}
        return predict,{'control':'same_RGB_conditioned_nuisance_regression_covariance_whitening_DINO_ridge','L2':1.,'covariance_eigenvalue_floor':floor}
    distance=sqdist(appearance[fg],appearance[bgall]);matched=bgall[np.argmin(distance,axis=1)]
        # "Sufficiently close" is source within-role/cross-block appearance
        # variation. Its numerical q.9 specification is fixed without Q labels.
    local=[]
    for roleids in (frame.fids,frame.bids):
        for block in range(4):
            held=roleids[frame.blocks[roleids]==block];train=roleids[frame.blocks[roleids]!=block]
            if len(held) and len(train):local.extend(np.min(sqdist(appearance[held],appearance[train]),axis=1).tolist())
    if not local:return _plain(frame)
    cut=float(np.quantile(local,.9));keep=np.min(distance,axis=1)<=cut+1e-12
    fg,matched=fg[keep],matched[keep]
    if not len(fg):return _plain(frame)
    if dino_pairs:matched=bgall[np.argmin(sqdist(frame.x[fg],frame.x[bgall]),axis=1)]
    if shuffled and len(matched)>1:matched=np.roll(matched,1)
    w,bias=_pair_direction(frame,fg,matched);radius=_support_radius(frame);f,*_=frame.banks()
    def predict(q,ids,role):
        score=q@w+bias
        score=np.minimum(score,(radius-nearest_mean_distance(q,f))/radius)
        return score,{}
    return predict,{'appearance_matched_pairs':len(fg),'pair_native_ids':np.c_[fg,matched].tolist(),
        'appearance_radius2_q90':cut,'DINO_difference_ridge_L2':1.,'FG_support_radius2':radius,
        'query_RGB_used_by_main':False,'same_budget_DINO_pair_control':dino_pairs,'pair_correspondence_shuffled_control':shuffled}


def _euclidean_modes(x,ids,k=8):
    selected=fps(x,ids,min(k,len(ids)));centers=x[selected].copy()
    for _ in range(20):
        assignment=np.argmin(sqdist(x[ids],centers),axis=1)
        nonempty=[j for j in range(len(centers)) if np.any(assignment==j)]
        new=np.array([x[ids[assignment==j]].mean(0) for j in nonempty])
        if new.shape==centers.shape and np.max(np.abs(new-centers))<1e-8:centers=new;break
        centers=new
    assignment=np.argmin(sqdist(x[ids],centers),axis=1)
    return centers,[ids[assignment==j] for j in range(len(centers))]


def fit_a047(frame,global_only=False,material_only=False,swap=False,direct_rbf=False,ungated=False):
    if frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    raw,low,center,scale=_rgb_scaler(frame);ids=np.flatnonzero(frame.train)
    centers,groups=_euclidean_modes(low,ids,8)
    global_head=fit_ridge(frame)
    heads=[];role_counts=[]
    for group in groups:
        keep=np.zeros(len(frame.x),bool);keep[group]=True;local=Frame.make(frame.ep,frame.train&keep)
        two=local.wf.sum()>0 and local.wb.sum()>0
        heads.append(fit_ridge(local) if two else global_head)
        role_counts.append([float(local.wf.sum()),float(local.wb.sum())])
    heads=np.array(heads)
    if swap:heads=np.roll(heads,1,axis=0)
    width=max(float(np.median(sqdist(low[ids],centers))),EPS)
    axis=unit(np.sum(frame.x*frame.wf[:,None],axis=0))-unit(np.sum(frame.x*frame.wb[:,None],axis=0))
    if direct_rbf:
        model=fit_rbf(frame,np.c_[frame.x,low])
        if model is None:return _plain(frame)
        bank,coefficient,kwidth=model
    def predict(q,indices,role):
        z=low[indices] if role=='r' else (rgb_statistics(frame.ep,'q')[indices]-center)/scale
        weight=np.exp(-sqdist(z,centers)/width-logsumexp(-sqdist(z,centers)/width,axis=1,keepdims=True))
        if direct_rbf:return np.exp(-sqdist(np.c_[q,z],bank)/kwidth)@coefficient,{}
        if material_only:
            rates=np.array([(a-b)/max(a+b,EPS) for a,b in role_counts]);score=weight@rates
        elif global_only:score=q@global_head[:-1]+global_head[-1]
        else:score=np.sum(weight*(q@heads[:,:-1].T+heads[:,-1]),axis=1)
        if not material_only and not ungated:score*=expit((q@axis)/.07)
        return score,{'soft_routed_bucket_weight_mean':weight.mean(0).tolist() if len(q) else []}
    return predict,{'material_buckets':len(centers),'bucket_role_masses':role_counts,'RGB19_width2':width,
        'classifier_L2':1.,'bucket_without_two_roles_uses_same_global_DINO_head':True,
        'nonnegative_global_semantic_gate':'sigmoid(unit(meanFG)-unit(meanBG) cosine/.07)',
        'global_only_control':global_only,'material_only_negative_control':material_only,'swapped_heads_control':swap,
        'same_RGB_DINO_RBF_control':direct_rbf,'ungated_control':ungated}


def install(register,requirements,methods,controls,recipes):
    register('A043',fit_a043,(
        'Train-only unlabelled thin PCA rank<=16; physical clockwise8-neighbor projected differences,8 cyclic rotations; center squared distance plus average squared common-visible directional differences.',
        'R/Q invalid padding neighbors removed; no shared valid direction to either role => fixed B0 cut0; up-to64 anchors per role. No neighbor bag replaces ordered matching.',),
        (('center_neighbor_mean_covariance_ridge',fit_ring_bag_rbf),('all_center_bag_affinity_RBF',lambda f:fit_ring_bag_rbf(f,all_profiles=True)),('same_bag_noncircular_order',lambda f:fit_a043(f,shuffle=True))))
    register('A044',fit_a044,(
        '64 FG response anchors; each ring has tied ordinal ranks encoded3bits plus one adjacent-rise bit, joint-anchor lexicographic minimum over8 physical cyclic rotations; equal Hamming bit weights.',
        'Incomplete or constant ring B0 cut0; source FG support95%5NN distance is necessary signed gate; canonical reference FG/BG codes use same bank budgets.',),
        (('same_center_and_ring_bag_ridge',lambda f:fit_a044(f,bag=True)),('same_bag_noncircular_order',lambda f:fit_a044(f,shuffle=True))))
    register('A045',fit_a045,(
        'Train-only PCA<=16; predictor input ring mean+four opposite projected differences, target full center DINO; BG-only vector ridgeL2=1; rank<2 or incomplete rings fixed B0 cut0.',
        'Classify 5NN residual FG/BG distances with original FG support95% necessary gate; full residual reference roles preserved, no anomaly-only identity.',),
        (('same_center_context_direct_ridge',lambda f:fit_a045(f,direct=True)),('residual_anomaly_semantic_gate',lambda f:fit_a045(f,anomaly=True)),
         ('all_residual_affinity_ridge',lambda f:fit_a045(f,affinity=True)),('FG_context_predictor_asymmetry',lambda f:fit_a045(f,foreground_predictor=True))))
    register('A049',fit_a049,(
        'Initial32 FPS per pure role, precompute allR distances. Inner held-block validation excludes all candidate-anchor positions and excludes that block anchors from scoring; require>=2 effective role-balanced blocks.',
        'For every legal deletion recompute balanced0-cut5NN source errors, choose smallest total delta/ID; stop if any block increases. Preserve>=1 per role, no forced compression/no Q selection.',),
        (('same_final_role_counts_FPS',lambda f:fit_a049(f,equal_fps=True)),))
    register('A041',fit_a041,(
        'Recorded physical RGB view, valid pixels only: RGB .1/.5/.9 quantiles +8-bin magnitude-weighted gradient orientation +gradient mean/std,19 dimensions; train median/MAD scaling, constant statistic scale1.',
        'Up-to64FG sourceFPS, closestBG in RGB statistics, accept if squared appearance distance <=source within-role cross-block nearest-distanceq.9; difference ridgeL2=1, pair-midpoint bias, necessary sourceFG95% support.',
        'Main uses Q DINO only; no source appearance match returns fixedB0cut0. RGB+DINO readout/conditional nuisance controls have explicit Q RGB access.',),
        (('same_budget_DINO_nearest_pairs',lambda f:fit_a041(f,dino_pairs=True)),('direct_same_RGB_DINO_ridge',lambda f:fit_a041(f,direct=True)),
         ('RGB_conditioned_DINO_ridge',lambda f:fit_a041(f,conditioned=True)),('matched_BG_correspondence_shift',lambda f:fit_a041(f,shuffled=True))))
    register('A047',fit_a047,(
        'Same physicalRGB19, train median/MAD scaling;8 Euclidean FPS-initialized material buckets,20 Lloyd iterations; each bucket balancedDINO ridgeL2=1 when both softrole masses>0, else exactglobalhead.',
        'Q own RGB routes all buckets with stable exp(-squared_distance/median_train_to_bucket_distance); signedweightedDINO score times nonnegative sigmoid(globalmeanFG−meanBGcos/.07). Bucket neverdirectlylabels main.',),
        (('same_global_DINO_head',lambda f:fit_a047(f,global_only=True)),('same_RGB_DINO_RBF',lambda f:fit_a047(f,direct_rbf=True)),
         ('material_only_negative_control',lambda f:fit_a047(f,material_only=True)),('bucket_classifier_correspondence_shift',lambda f:fit_a047(f,swap=True))))
    from .a_algorithms_009_020 import full_profile_rbf
    for mid in ('A041','A043','A044','A045','A047','A049'):
        # These are independent complete supervised readouts, not counts.
        name=mid+'__complete_reference_affinity_RBF'
        from functools import partial
        from .a_helpers_001_050 import infer_a
        controls[name]=partial(infer_a,method_id=name,fit=full_profile_rbf,
            assumptions=('Strongsimplecontrol; allvalidR affinitycolumns; same4fold C_R andcompleteU.',))
    controls['A047__ungated_global_DINO_ridge']=partial(infer_a,method_id='A047__ungated_global_DINO_ridge',
        fit=lambda f:fit_a047(f,global_only=True,ungated=True),assumptions=('Control only; fullDINO balancedridgeL2=1 withoutRGBrouting orsemanticgate.',))
    for mid in ('A041','A047'):requirements[mid]=['original_RGB','physical_RGB_transform']
