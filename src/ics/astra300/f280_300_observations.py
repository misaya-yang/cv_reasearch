"""Actual observer/forward F280/282/285/288/294/297/300 kernels.

These functions require the bound CPU model. No endpoint descriptor, toy
callback or processed cache is accepted as an internal intervention.
"""
from __future__ import annotations
from itertools import combinations
from collections import OrderedDict
import hashlib,json
import numpy as np
from scipy import ndimage
from scipy.special import expit,logsumexp
from . import common
from . import f_protocol as P
from . import f276_300 as K
from . import e_views_246_249 as V
from . import e_helpers_226_250 as H
from . import f285_299_native as N
EPS=1e-6
_TAIL_CACHE=OrderedDict()


def backend(c):
    resources=c.resources or {};episode=resources.get('bound_episode',c.ep)
    callback=resources.get('tail_replay')
    if callback is None:callback=common.artifact(episode,'tail_replay')
    owner=getattr(callback,'__self__',None);binding=getattr(owner,'binding',{})
    if binding.get('execution_kind')!='real_frozen_cpu_model':raise common.ArtifactUnavailable('Actual bound frozen CPU tail observer required')
    expected=c.ep.producer.get('model_assets',c.ep.producer);actual=binding.get('model_assets',binding)
    for key in ('checkpoint_sha256','config_sha256'):
        if expected.get(key) and expected[key]!=actual.get(key):raise common.ArtifactUnavailable('Internal observer '+key+' differs from native producer')
    # Calibration turns the original reference into Q. Its features/RGB remain
    # frozen and the source callback reads only image/raw-state data, never MR.
    qrole='r' if c.ep.q_rgb is not None and episode.r_rgb is not None and common.array_hash(c.ep.q_rgb)==common.array_hash(episode.r_rgb) else 'q'
    return episode,owner,callback,qrole


def tail(c,role,**kwargs):
    episode,owner,callback,qrole=backend(c);role=qrole if role=='q' else 'r'
    identity=dict(role=role,image_sha256=common.array_hash(episode.r_rgb if role=='r' else episode.q_rgb),
                  binding=getattr(owner,'binding',{}))
    for name,value in kwargs.items():
        identity[name]=common.array_hash(value) if isinstance(value,np.ndarray) else value
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
    if key in _TAIL_CACHE:
        z,receipt=_TAIL_CACHE[key];_TAIL_CACHE.move_to_end(key)
        return z,dict(receipt,actual_tail_blocks=0,tail_cache_hit=True,tail_state_and_intervention_sha256=key)
    before=int(owner.receipt.get('actual_tail_blocks',0));out=callback(episode,role,return_attention=False,**kwargs)
    z=np.asarray(out['raw_final_ln'])
    if z.dtype!=np.float32 or not np.isfinite(z).all():raise ValueError('Actual raw FP32 final LayerNorm required')
    if out.get('refresh_qkv')!=kwargs.get('refresh_qkv',True):raise ValueError('Requested true QKV routing was not executed')
    receipt=dict(role=role,actual_tail_blocks=int(owner.receipt.get('actual_tail_blocks',0))-before,
       private_special_tokens=bool(out.get('private_special_tokens')),raw_final_ln_sha256=common.array_hash(z),
       producer=out.get('producer'),refresh_qkv=bool(out.get('refresh_qkv')))
    z=z.copy();z.setflags(write=False);_TAIL_CACHE[key]=(z,receipt)
    while len(_TAIL_CACHE)>16:_TAIL_CACHE.popitem(last=False)
    return z,dict(receipt,tail_cache_hit=False,tail_state_and_intervention_sha256=key)


def _role_pairs(c,FGFG=2,FGBG=2):
    edges=K.neighbors(c.rhw);result=[]
    for kind,count in (('FGFG',FGFG),('FGBG',FGBG)):
        candidates=[(int(a),int(b)) for a,b in edges if c.valid[a] and c.valid[b] and
          ((c.F[a] and c.F[b]) if kind=='FGFG' else (c.F[a] and c.B[b]) or (c.B[a] and c.F[b]))]
        for pair in candidates:
            if all(set(pair).isdisjoint(set(v[0])) for v in result):result.append((pair,kind))
            if sum(v[1]==kind for v in result)>=count:break
    return result


def _domain_pairs(c,maximum=4):
    pieces=[p for p in c.P if p.any()];pairs=[]
    for a,b in combinations(range(len(pieces)),2):
        if np.any(pieces[a]&pieces[b]):continue
        contrast=abs(float(c.s[pieces[a]].mean()-c.s[pieces[b]].mean()))
        distance=np.linalg.norm(np.array(np.nonzero(pieces[a].reshape(c.hw))).mean(1)-np.array(np.nonzero(pieces[b].reshape(c.hw))).mean(1))
        pairs.append((-contrast,float(distance),a,b))
    pairs.sort();return [(pieces[a],pieces[b]) for _,_,a,b in pairs[:maximum]]


def _two_domain_delta(c,role,a,b,control=False):
    episode,owner,callback,qrole=backend(c);actualrole=qrole if role=='q' else 'r';session=owner.observe(episode,actualrole)
    raw=np.asarray(session.raw_final_ln,np.float32);ai=np.flatnonzero(a);bi=np.flatnonzero(b)
    X=np.array([(int(i),int(j)) for i in ai for j in bi],int).reshape(-1,2);Y=X[:,::-1]
    zX,rX=tail(c,role,blocked_edges=X);zY,rY=tail(c,role,blocked_edges=Y);zXY,rXY=tail(c,role,blocked_edges=np.r_[X,Y])
    delta=(zX+zY-2*raw) if control=='first_order' else zXY-zX-zY+raw
    direction=K.unit(c.r[c.F].mean(0))-K.unit(c.r[c.B].mean(0))
    rsession=owner.observe(episode,'r');scale=max(float(np.median(np.linalg.norm(rsession.raw_final_ln,axis=1))),EPS)
    return delta@direction/scale,[rX,rY,rXY]


def _kernel_class_ratio(train,labels,test):
    train=np.asarray(train,float);labels=np.asarray(labels,bool);test=np.asarray(test,float)
    if not labels.any() or labels.all():return None
    scale=np.maximum(np.median(np.abs(train-np.median(train,0)),0),EPS);train=train/scale;test=test/scale
    distance=np.sum((train[:,None]-train[None])**2,2);bandwidth=max(float(np.median(distance[distance>EPS])) if np.any(distance>EPS) else 0.,EPS)
    lk=-np.sum((test[:,None]-train[None])**2,2)/(2*bandwidth)
    score=logsumexp(lk[:,labels],1)-np.log(labels.sum())-logsumexp(lk[:,~labels],1)+np.log((~labels).sum())
    return score,dict(source_feature_MAD=scale.tolist(),source_bandwidth=bandwidth)


def f280(c,weight=1.,control=False):
    reference=_role_pairs(c)
    if sum(v[1]=='FGFG' for v in reference)<2 or sum(v[1]=='FGBG' for v in reference)<2:
        return K.result(c,score=c.s,fallback='two_source_FGFG_and_FGBG_interfaces_unavailable')
    observations=[];labels=[];receipts=[]
    for (a,b),kind in reference:
        A=np.zeros(len(c.r),bool);B=A.copy();A[a]=True;B[b]=True
        response,r=_two_domain_delta(c,'r',A,B,control);receipts+=r;ids=np.flatnonzero(A|B)
        source_margin=(c.r[ids]@(K.unit(c.r[c.F].mean(0))-K.unit(c.r[c.B].mean(0))))/.1
        observations.extend(np.c_[source_margin,response[ids]].tolist());labels.extend((c.c[ids]>.5).tolist())
    pairs=_domain_pairs(c);score=c.s.copy();owner=np.full(c.n,-1,int);distance=np.full(c.n,np.inf);xy=np.c_[np.indices(c.hw)[0].ravel(),np.indices(c.hw)[1].ravel()]
    for index,(A,B) in enumerate(pairs):
        response,r=_two_domain_delta(c,'q',A,B,control);receipts+=r;inside=A|B;center=xy[inside].mean(0);d=np.sum((xy-center)**2,1)
        use=inside&(d<distance);ratio=_kernel_class_ratio(observations,labels,np.c_[c.s[use],response[use]])
        if ratio is not None:score[use]=ratio[0];owner[use]=index;distance[use]=d[use]
    probability=c.p0.copy();use=owner>=0;probability[use]=expit(weight*score[use]);
    return K.KernelResult(probability=probability,unary=score,info=dict(mechanism='F280_raw_finalLN_nonadditive_directed_tail_interaction',
       source_interfaces=reference,query_pairs=len(pairs),affected_points=int(use.sum()),actual_tail_receipts=receipts,
       base_raw_states_before_unit=True,maximum_additional_tail_blocks_without_component_groups=48,control=control))


def f282(c,weight=1.,control=False):
    episode,owner,callback,qrole=backend(c);redges=K.neighbors(c.rhw)
    pools=[];states=[];blocks=P.quarters(c.rhw)
    for a,b in redges:
        if not c.pure[a] or not c.pure[b]:continue
        state=int(c.F[a])*2+int(c.F[b]);pools.append((int(a),int(b)));states.append(state)
    if not pools:return K.result(c,score=c.s,fallback='no_pure_source_action_edges')
    edges=np.array(pools);states=np.array(states);selected=[]
    for state in range(4):
        ids=np.flatnonzero(states==state)
        if len(np.unique(blocks[edges[ids,0]]))<2:return K.result(c,score=c.s,fallback='source_action_four_states_not_in_two_spatial_blocks')
        selected.extend(ids[np.linspace(0,len(ids)-1,min(len(ids),128),dtype=int)].tolist())
    selected=np.array(selected);edges=edges[selected];states=states[selected]
    sr=owner.observe(episode,'r');sq=owner.observe(episode,qrole)
    rfeat=np.c_[sr.attention[23].edge_features(edges),sr.attention[24].edge_features(edges)]
    qfeat=np.c_[sq.attention[23].edge_features(c.edges),sq.attention[24].edge_features(c.edges)]
    if control=='endpoint_profile':
        rp=c.r@c.r[c.valid].T;qp=c.q@c.r[c.valid].T;rfeat=np.c_[rp[edges[:,0]],rp[edges[:,1]]];qfeat=np.c_[qp[c.edges[:,0]],qp[c.edges[:,1]]]
    elif control=='alpha':
        def alpha(session,E):
            observations=[]
            for number in (23,24):
                att=session.attention[number];values=[]
                for a,b in E:
                    ab=att.weights(np.array([a+5]),np.array([b+5]))[:,0,0];ba=att.weights(np.array([b+5]),np.array([a+5]))[:,0,0]
                    values.append([np.median(np.log(ab+EPS)),np.median(np.log(ba+EPS))])
                observations.append(values)
            return np.concatenate(observations,1)
        rfeat=alpha(sr,edges);qfeat=alpha(sq,c.edges)
    scale=np.maximum(np.median(np.abs(rfeat-np.median(rfeat,0)),0),EPS);rfeat=rfeat/scale;qfeat=qfeat/scale
    bandwidth=max(float(np.median(np.sum((rfeat-np.median(rfeat,0))**2,1))),EPS);densities=[]
    for state in range(4):
        bank=rfeat[states==state];lk=-np.sum((qfeat[:,None]-bank[None])**2,2)/(2*bandwidth);densities.append(logsumexp(lk,1)-np.log(len(bank)))
    logp=np.stack(densities,1);logp-=logsumexp(logp,1,keepdims=True);prob=np.exp(logp).reshape(-1,2,2)
    logjoint=logp.reshape(-1,2,2);logrow=logsumexp(logjoint,2);logcol=logsumexp(logjoint,1)
    V=-logjoint+logrow[:,:,None]+logcol[:,None,:]
    violation=np.maximum(V[:,0,0]+V[:,1,1]-V[:,0,1]-V[:,1,0],0)
    V[:,0,0]-=violation/4;V[:,1,1]-=violation/4;V[:,0,1]+=violation/4;V[:,1,0]+=violation/4
    coupling=np.maximum((V[:,0,1]+V[:,1,0]-V[:,0,0]-V[:,1,1])/2,0)
    linear0=V[:,1,0]-V[:,0,0]-coupling;linear1=V[:,0,1]-V[:,0,0]-coupling
    score=c.s.copy();np.add.at(score,c.edges[:,0],-weight*linear0);np.add.at(score,c.edges[:,1],-weight*linear1)
    labels=K.cut(c,score,weights=c.edge_weight+weight*coupling)
    return K.result(c,labels=labels,score=score,mechanism='F282_actual_attention_action_four_role_pairwise',
         source_states=np.bincount(states,minlength=4).tolist(),action_features=6 if not control else rfeat.shape[1],
         submodular_projection_edges=int(np.sum(violation>0)),maximum_submodular_violation_after_projection=float(np.max(V[:,0,0]+V[:,1,1]-V[:,0,1]-V[:,1,0])),
         source_feature_MAD=scale.tolist(),source_bandwidth=bandwidth,control=control,
         actual_attention_normalization_includes_private_prefix=True)


def _canvas_crop(c,role,box):
    episode,owner,callback,qrole=backend(c);role=qrole if role=='q' else 'r'
    canvas=V._base_canvas(episode,role);y,x,height,width=box
    crop=np.asarray(__import__('PIL.Image',fromlist=['Image']).fromarray(canvas[y:y+height,x:x+width]).resize((canvas.shape[1],canvas.shape[0]),__import__('PIL.Image',fromlist=['Image']).Resampling.BILINEAR))
    raw,receipt=V.encode(episode,role,crop);baseg=episode.reference_geometry if role=='r' else episode.query_geometry
    fy=canvas.shape[0]/height;fx=canvas.shape[1]/width;sh,sw=baseg['resized_hw'];oy,ox=baseg['padding_top_left']
    geometry=dict(baseg,resized_hw=[sh*fy,sw*fx],padding_top_left=[(oy-y)*fy,(ox-x)*fx])
    shape=episode.r_rgb.shape[:2] if role=='r' else episode.original_shape;A,valid=V._valid_operator(shape,raw.shape[:2],geometry)
    return dict(z=V._unit(raw).reshape(-1,raw.shape[-1]),A=A,valid=valid,geometry=geometry,shape=raw.shape[:2],receipt=receipt)


def _registered_reference_labels(c,view):
    # Unknown/buffer pixels are removed before any transformed coverage bank.
    original=N._known_original_target(c)
    if original is None:raise common.ArtifactUnavailable('Actual original known-reference pixel mask required for true-view calibration')
    target,known=original;valid=(view['A']@known.ravel().astype(float))*view['valid'];fg=(view['A']@(known&target).ravel().astype(float))*view['valid']
    return fg,valid


def _view_profile(q,view,fg,valid):
    z=view['z'];f=fg;b=valid-fg;profile=q@z.T/.1
    if f.sum()<=EPS or b.sum()<=EPS:return None
    return logsumexp(profile+np.where(f>0,np.log(np.maximum(f,EPS)),-np.inf)[None],1)-np.log(f.sum())-logsumexp(profile+np.where(b>0,np.log(np.maximum(b,EPS)),-np.inf)[None],1)+np.log(b.sum())


def _source_view_choice(c,views,option_fields):
    target_and_known=N._known_original_target(c)
    if target_and_known is None:return 0,dict(fallback='actual_reference_training_pixels_missing')
    target,known=target_and_known;losses=np.zeros(len(option_fields));folds=0
    for train,held in P.buffered_folds(c.rhw,c.valid):
        if not held.any() or not np.any(c.F&train) or not np.any(c.B&train):continue
        original_token=H.pixel_tokens(target.shape,c.rhw,c.ep.reference_geometry);train_pixels=train[original_token].reshape(target.shape)&known
        predicted=[]
        for view in views:
            v=(view['A']@train_pixels.ravel().astype(float))*view['valid'];f=(view['A']@(train_pixels&target).ravel().astype(float))*view['valid']
            s=_view_profile(c.r,view,f,v)
            if s is not None:predicted.append(s)
        if len(predicted)!=len(views):continue
        model=P._model(c.r,c.c,np.asarray(c.ep.wvalid)*train);old=P._scores(c.r,c.r,model)[:,2]
        fields=[old,np.mean([old]+predicted,0)]
        domain=common.U(P._reference_query(c.ep,train),held.reshape(c.rhw).astype(float),.5)&known
        if not domain.any():continue
        for index,field in enumerate(fields):
            mask=H.native_to_original(expit(field).reshape(c.rhw),target.shape,c.ep.reference_geometry)>.5
            I=np.count_nonzero(mask&target&domain);U=np.count_nonzero((mask|target)&domain);losses[index]+=1-(I/U if U else 1.)
        folds+=1
    if folds<2:return 0,dict(fallback='fewer_than_two_registered_source_view_tasks')
    return int(np.argmin(losses)),dict(source_final_original_mask_losses=(losses/folds).tolist(),source_folds=folds)


def f285(c,weight=1.,control=False):
    pairs=[(int(np.count_nonzero(a!=b)),i,j) for i,a in enumerate(c.Q) for j,b in enumerate(c.Q) if i<j]
    if not pairs:return K.result(c,score=c.s,fallback='no_competing_whole_masks')
    _,a,b=max(pairs);A=c.Q[a];B=c.Q[b];profile=c.q@c.r.T
    discrimination=np.abs(profile[A].mean(0)-profile[B].mean(0)) if A.any() and B.any() else np.abs((A.astype(float)-B.astype(float))@profile)/max(np.count_nonzero(A!=B),1)
    positions=[];side=int(c.ep.producer.get('model_input_side',1024));width=min(side,12*16)
    for role in (c.F,c.B):
        ids=np.flatnonzero(role)
        if not len(ids):continue
        if control=='fixed':j=int(ids[0])
        elif control=='random':j=int(ids[np.argmin((ids*2654435761)%4294967296)])
        else:j=int(ids[np.argmax(discrimination[ids])])
        y,x=np.divmod(j,c.rhw[1]);cy=(y+.5)*side/c.rhw[0];cx=(x+.5)*side/c.rhw[1]
        by=int(np.clip(round(cy-width/2),0,side-width));bx=int(np.clip(round(cx-width/2),0,side-width))
        if positions and abs(by-positions[0][0])+abs(bx-positions[0][1])<16:
            order=ids[np.argsort(-discrimination[ids],kind='stable')]
            for alternative in order:
                yy,xx=np.divmod(alternative,c.rhw[1]);ny=int(np.clip(round((yy+.5)*side/c.rhw[0]-width/2),0,side-width));nx=int(np.clip(round((xx+.5)*side/c.rhw[1]-width/2),0,side-width))
                if abs(ny-positions[0][0])+abs(nx-positions[0][1])>=16:by,bx=ny,nx;break
        positions.append((by,bx,width,width))
    views=[_canvas_crop(c,'r',box) for box in positions];fields=[]
    for view in views:
        f,v=_registered_reference_labels(c,view);s=_view_profile(c.q,view,f,v)
        if s is not None:fields.append(s)
    if not fields:return K.result(c,score=c.s,fallback='true_reference_crops_have_no_legal_two_roles')
    mixture=np.mean([c.heads[:,2]]+fields,0)
    if control=='mean':
        simple=[]
        for view in views:
            f,v=_registered_reference_labels(c,view);s=V._weighted_margin(c.q,view['z'],f,v)
            if s is not None:simple.append(s)
        mixture=np.mean([c.s]+simple,0)
    choice,receipt=_source_view_choice(c,views,[c.heads[:,2],mixture]);score=c.heads[:,2] if choice==0 and control is not True else mixture
    return K.result(c,score=weight*score,mechanism='F285_query_competition_selects_true_reference_recrops',
         reference_boxes=[list(v) for v in positions],source_view_choice=choice,source_view_calibration=receipt,
         actual_encoder_receipts=[v['receipt'] for v in views],extra_reference_views=len(views),control=control)


def _attention_head_candidates(c,owner,episode):
    session=owner.observe(episode,'r');F=np.flatnonzero(c.F)+5;B=np.flatnonzero(c.B)+5;quality=[]
    for layer in (23,24):
        att=session.attention[layer];Hn=int(att.q.shape[1]);mass=np.zeros(Hn)
        for rows,columns in ((F,B),(B,F)):
            for start in range(0,len(rows),64):mass+=att.weights(rows[start:start+64],columns).sum((1,2))
        mass/=max(2*len(F)*len(B),1)
        quality.extend([(float(-value),layer,head) for head,value in enumerate(mass)])
    quality.sort();return [(layer,head) for _,layer,head in quality[:4]]


def _gated_fields(c,gates):
    rz,rreceipt=tail(c,'r',head_gates=gates);qz,qreceipt=tail(c,'q',head_gates=gates)
    r=K.unit(rz);q=K.unit(qz);profile=K.kernel_field(q,r[c.F],r[c.B]);return profile,r,q,[rreceipt,qreceipt]


def f288(c,weight=1.,control=False):
    episode,owner,callback,qrole=backend(c);selected=_attention_head_candidates(c,owner,episode);Hn=owner.model.blocks[-1].attn.num_heads
    configurations=[()]+[(j,) for j in range(len(selected))]+list(combinations(range(len(selected)),2));fields=[];source_losses=[];receipts=[]
    target_and_known=N._known_original_target(c)
    if target_and_known is None:return K.result(c,score=c.s,fallback='original_source_gate_calibration_pixels_missing')
    target,known=target_and_known
    for config in configurations:
        gates=np.ones((2,Hn),np.float32)
        for j in config:layer,head=selected[j];gates[layer-23,head]=0
        s,r,q,actual=_gated_fields(c,gates);fields.append(s);receipts+=actual;loss=[]
        # Ranking is reselected on training labels for each buffered fold; g's
        # indices identify rank positions, not absolute full-fit head IDs.
        for train,held in P.buffered_folds(c.rhw,c.valid):
            if not held.any() or not np.any(c.F&train) or not np.any(c.B&train):continue
            original_valid=c.valid
            try:
                c.valid=train;ranked=_attention_head_candidates(c,owner,episode)
            finally:c.valid=original_valid
            local=np.ones((2,Hn),np.float32)
            for j in config:layer,head=ranked[j];local[layer-23,head]=0
            localraw,lreceipt=tail(c,'r',head_gates=local);receipts.append(lreceipt);unit=K.unit(localraw)
            margin=K.kernel_field(unit,unit[c.F&train],unit[c.B&train]);pred=H.native_to_original(expit(margin).reshape(c.rhw),target.shape,c.ep.reference_geometry)>.5
            domain=H.native_to_original(held.reshape(c.rhw).astype(float),target.shape,c.ep.reference_geometry)>.5;domain &=known
            I=np.count_nonzero(pred&target&domain);U=np.count_nonzero((pred|target)&domain);loss.append(1-(I/U if U else 1.))
        source_losses.append(float(np.mean(loss)) if len(loss)>=2 else np.inf)
    entropy=np.array([np.mean(np.logaddexp(0,-np.abs(s))) for s in fields]);off=np.array([len(v) for v in configurations]);cost=np.array(source_losses)+weight*entropy+.1*off
    if control=='source':cost=np.array(source_losses)+.1*off
    elif control=='random':
        best=min(range(len(cost)),key=lambda j:(cost[j],off[j],j));same=[j for j in range(len(cost)) if off[j]==off[best]];selected_index=same[-1]
    if control!='random':selected_index=min(range(len(cost)),key=lambda j:(cost[j],off[j],j))
    if not np.isfinite(cost[selected_index]):return K.result(c,score=c.s,fallback='no_two_source_gate_tasks',actual_tail_receipts=receipts)
    return K.result(c,score=fields[selected_index],mechanism='F288_whole_field_global_attention_head_retirement',
         head_candidates=selected,rank_configs=[list(v) for v in configurations],source_losses=source_losses,
         query_self_BCE=entropy.tolist(),costs=cost.tolist(),selected_configuration=selected_index,
         actual_tail_receipts=receipts,calibration_head_ranking_rebuilt_per_training_fold=True,
         actual_blocks_include_source_head_reselection=True,control=control)


def f294(c,weight=1.,control=False):
    episode,owner,callback,qrole=backend(c);canvas=V._base_canvas(episode,'r');original=N._known_original_target(c)
    if original is None:return K.result(c,score=c.s,fallback='known_original_R_pixels_unavailable')
    target,known=original;bg=known&~target
    if not bg.any():return K.result(c,score=c.s,fallback='no_same_reference_BG_RGB')
    fill=np.median(np.asarray(episode.r_rgb)[bg],axis=0).astype(np.uint8);fields=[c.heads[:,2]];receipts=[];blocks=P.quarters(c.rhw)
    for block in range(4):
        removed=(blocks==block);train=c.valid&~removed
        if np.count_nonzero(c.F&train)<2 or np.count_nonzero(c.B&train)<2:continue
        if control=='token_delete':s=K.kernel_field(c.q,c.r[c.F&train],c.r[c.B&train]);fields.append(s);continue
        changed=canvas.copy();native=removed.reshape(c.rhw);pixel=ndimage.zoom(native.astype(np.uint8),[canvas.shape[0]/c.rhw[0],canvas.shape[1]/c.rhw[1]],order=0)>0
        changed[pixel]=fill;raw,receipt=V.encode(episode,'r',changed);receipts.append(receipt);z=V._unit(raw).reshape(-1,raw.shape[-1])
        fields.append(K.kernel_field(c.q,z[c.F&train],z[c.B&train]))
    if len(fields)<2:return K.result(c,score=c.s,fallback='no_legal_reference_block_withdrawal')
    if control in ('average','median'):
        s=np.mean(fields,0) if control=='average' else np.median(fields,0);return K.result(c,score=s,actual_encoder_receipts=receipts,control=control)
    candidates=list(c.Q)
    for s in fields:
        y=K.cut(c,s)
        if not any(np.array_equal(y,v) for v in candidates):candidates.append(y)
    candidates=candidates[:16];energies=np.array([[c.energy(y,s=s) for y in candidates] for s in fields]);regret=energies-energies.min(1,keepdims=True)
    scale=np.array([max(float(np.median(np.abs(v-np.median(v)))),EPS) for v in regret]);worst=(regret/scale[:,None]).max(0)
    costs=np.array([c.energy(y) for y in candidates])+weight*worst;winner=min(range(len(costs)),key=lambda j:(costs[j],int(candidates[j].sum()),j))
    return K.result(c,labels=candidates[winner],mechanism='F294_reference_true_block_withdrawal_minimax_whole_masks',
         actual_encoder_receipts=receipts,withheld_regions_unknown_not_BG=True,view_count=len(fields),
         candidate_regret=(regret/scale[:,None]).tolist(),selected_candidate=winner,control=control)


def _paired_boxes(c,role):
    hw=c.hw if role=='q' else c.rhw;n=np.prod(hw);side=int(c.ep.producer.get('model_input_side',1024));pairs=[]
    if role=='r':
        F=np.flatnonzero(c.F);B=np.flatnonzero(c.B)
        if not len(F) or not len(B):return []
        for f in F[np.linspace(0,len(F)-1,min(2,len(F)),dtype=int)]:
            fy,fx=np.divmod(f,hw[1]);b=B[np.argmin(np.sum((np.c_[B//hw[1],B%hw[1]]-[fy,fx])**2,1))];a=np.zeros(n,bool);bb=a.copy();a[f]=True;bb[b]=True;pairs.append((a,bb))
    else:
        pairs=_domain_pairs(c,2)
    boxes=[]
    for A,B in pairs:
        y,x=np.nonzero((A|B).reshape(hw));y0,y1=y.min()*side/hw[0],(y.max()+1)*side/hw[0];x0,x1=x.min()*side/hw[1],(x.max()+1)*side/hw[1]
        dy=(y1-y0)*.1;dx=(x1-x0)*.1;top=int(max(0,np.floor(y0-dy)));left=int(max(0,np.floor(x0-dx)));bottom=int(min(side,np.ceil(y1+dy)));right=int(min(side,np.ceil(x1+dx)))
        boxes.append((top,left,bottom-top,right-left))
    return boxes[:2]


def f297(c,weight=1.,control=False):
    rboxes=_paired_boxes(c,'r');qboxes=_paired_boxes(c,'q')
    if not rboxes or not qboxes:return K.result(c,score=c.s,fallback='paired_true_FG_BG_reference_or_query_domains_unavailable')
    source=[];labels=[];receipts=[];reference_views=[];rbase=(c.r@(K.unit(c.r[c.F].mean(0))-K.unit(c.r[c.B].mean(0))))/.1
    for box in rboxes:
        view=_canvas_crop(c,'r',box);fg,valid=_registered_reference_labels(c,view);margin=V._weighted_margin(view['z'],view['z'],fg,valid)
        if margin is None:continue
        cropcover=np.divide(fg,valid,out=np.zeros_like(fg),where=valid>0);pure=(valid>0)&((cropcover>=.9)|(cropcover<=.1))
        # Relative crop margin is centered within the shared actual context,
        # without assigning query hypothesis labels to either bank.
        centered=margin-np.mean(margin[pure]);original=H.native_to_original(rbase.reshape(c.rhw),c.ep.r_rgb.shape[:2],c.ep.reference_geometry)
        old=np.asarray(view['A']@original.ravel());source.extend(np.c_[old[pure],centered[pure]].tolist());labels.extend((cropcover[pure]>.5).tolist());reference_views.append((view,fg,valid));receipts.append(view['receipt'])
    if not reference_views or not any(labels) or all(labels):return K.result(c,score=c.s,fallback='paired_crops_no_two_true_classes',actual_encoder_receipts=receipts)
    score=c.s.copy();assigned=np.zeros(c.n,bool);distance=np.full(c.n,np.inf);xy=np.c_[np.indices(c.hw)[0].ravel(),np.indices(c.hw)[1].ravel()]
    for box in qboxes:
        view=_canvas_crop(c,'q',box);receipts.append(view['receipt']);margins=[]
        for rview,fg,valid in reference_views:
            m=V._weighted_margin(view['z'],rview['z'],fg,valid)
            if m is not None:margins.append(m)
        if not margins:continue
        margin=np.mean(margins,0);margin-=margin.mean();prob=H.native_to_original(margin.reshape(view['shape']),c.ep.original_shape,view['geometry']);A=H.footprint(c.ep.original_shape,c.hw,c.ep.query_geometry);cropmargin=np.asarray(A@prob.ravel())
        observed=V._observed_pixels(c.ep.original_shape,view['geometry']);use=(A@observed.ravel().astype(float))>.5
        center=xy[use].mean(0) if use.any() else np.zeros(2);d=np.sum((xy-center)**2,1);use &=d<distance
        ratio=_kernel_class_ratio(source,labels,np.c_[c.s[use],cropmargin[use]])
        if ratio is None:continue
        score[use]=cropmargin[use] if control=='mean' else ratio[0];assigned[use]=True;distance[use]=d[use]
    probability=c.p0.copy();probability[assigned]=expit(weight*score[assigned])
    return K.KernelResult(probability=probability,unary=score,info=dict(mechanism='F297_joint_target_opponent_shared_context_recrop',
        reference_boxes=[list(v) for v in rboxes],query_boxes=[list(v) for v in qboxes],actual_encoder_receipts=receipts,
        reobserved_points=int(assigned.sum()),maximum_extra_views=4,control=control))


def component_groups(mask,hw,known=None):
    mask=np.asarray(mask,bool).reshape(hw);known=np.ones(hw,bool) if known is None else np.asarray(known,bool).reshape(hw);groups=np.full(hw,-1,int);nextid=0
    for role in (False,True):
        label,count=ndimage.label((mask==role)&known)
        for j in range(1,count+1):groups[label==j]=nextid;nextid+=1
    # Unknown source-held/padding pixels have their own components and do not
    # enter either class bank. They are never labelled background for replay.
    label,count=ndimage.label(~known)
    for j in range(1,count+1):groups[label==j]=nextid;nextid+=1
    return groups.ravel(),nextid


def f300(c,weight=1.,control=False):
    if control in ('fixed_features','group_mean'):
        r=c.r.copy();rr=[]
        if control=='group_mean':
            groups,number=component_groups(c.c>.5,c.rhw,c.valid)
            for j in np.unique(groups):r[groups==j]=K.unit(r[groups==j].mean(0))
    else:
        groups,number=component_groups(c.c>.5,c.rhw,c.valid);rz,rreceipt=tail(c,'r',patch_groups=groups,private_special_tokens=True,refresh_qkv=control!='fixed_QK');r=K.unit(rz);rr=[rreceipt]
    costs=[];receipts=list(rr);fields=[];group_counts=[]
    for y in c.Q:
        groups,count=component_groups(y,c.hw);group_counts.append(count)
        if control=='fixed_features':q=c.q
        elif control=='group_mean':
            q=c.q.copy()
            for j in np.unique(groups):q[groups==j]=K.unit(q[groups==j].mean(0))
        else:
            raw,receipt=tail(c,'q',patch_groups=groups,private_special_tokens=True,refresh_qkv=control!='fixed_QK');receipts.append(receipt);q=K.unit(raw)
        score=K.kernel_field(q,r[c.F],r[c.B]);fields.append(score);loss=np.logaddexp(0,score)-score*y
        fee=np.mean(loss)+count*np.log(max(c.n,2))/max(c.n,1);costs.append(c.energy(y)+weight*fee)
    if control=='same_all_views_average':
        # A complete candidate is still selected; per-pixel view cherry-pick
        # would be a different and unfair whole-hypothesis control.
        average=np.mean(fields,0);costs=[c.energy(y,s=average) for y in c.Q]
    winner=min(range(len(costs)),key=lambda j:(costs[j],int(c.Q[j].sum()),j))
    return K.result(c,labels=c.Q[winner],mechanism='F300_mask_hypotheses_change_true_frozen_tail_graph',
          selected_candidate=winner,candidate_costs=list(map(float,costs)),candidate_groups=group_counts,
          actual_tail_receipts=receipts,private_prefix_states_per_spatial_component=True,
          actual_block_count_includes_all_component_replays=True,control=control,
          confidence_is_not_accuracy=True)


KERNELS={280:f280,282:f282,285:f285,288:f288,294:f294,297:f297,300:f300}
CONTROLS={280:['first_order'],282:['endpoint_profile','alpha'],285:['mean','fixed','random'],288:['source','random'],
          294:['average','median','token_delete'],297:['mean'],300:['fixed_features','fixed_QK','group_mean','same_all_views_average']}
RESOURCES={280:['tail_replay'],282:['tail_replay'],285:['tail_replay','frozen_encode_rgb'],288:['tail_replay'],
           294:['tail_replay','frozen_encode_rgb'],297:['tail_replay','frozen_encode_rgb'],300:['tail_replay']}
