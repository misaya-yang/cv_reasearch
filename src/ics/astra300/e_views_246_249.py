"""Actual frozen-DINO extra-view kernels. Missing model resources are errors."""
from __future__ import annotations
from collections import OrderedDict
import hashlib,threading
import numpy as np
from scipy import sparse,ndimage
from scipy.special import expit,logsumexp
from PIL import Image
from . import common
from . import e_helpers_226_250 as H
from . import e226_250 as B
from . import e235_244 as C

_CACHE=OrderedDict();_LOCK=threading.RLock();MAX_CACHED_VIEWS=16


def _encoder(ep):
    callback=common.artifact(ep,'frozen_encode_rgb');owner=getattr(callback,'__self__',None)
    binding=getattr(owner,'binding',{})
    expected=ep.producer.get('model_assets',ep.producer)
    actual=binding.get('model_assets',binding)
    if not callable(callback) or binding.get('execution_kind')!='real_frozen_cpu_model':
        raise common.ArtifactUnavailable('Actual frozen CPU RGB encoder required; fake callbacks are not main-method evidence')
    for name in ('checkpoint_sha256','config_sha256'):
        if expected.get(name) and expected[name]!=actual.get(name):
            raise common.ArtifactUnavailable('Extra-view '+name+' differs from the native DINO producer')
    return callback,binding


def encode(ep,role,canvas):
    callback,binding=_encoder(ep);canvas=np.asarray(canvas)
    cp=binding.get('model_assets',binding)
    key=(cp.get('checkpoint_sha256'),cp.get('config_sha256'),common.array_hash(canvas))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key);return _CACHE[key],dict(input_sha256=key[2],cache_hit=True,new_encoder_forwards=0,side=canvas.shape[0])
        value=np.asarray(callback(role,canvas,working_canvas=True,side=canvas.shape[0],raw=True))
        if value.dtype!=np.float32 or value.shape[:2]!=(canvas.shape[0]//16,canvas.shape[1]//16) or value.ndim!=3:
            raise ValueError('Actual FP32 final-LN native grid required; no pooled endpoint or processed cache')
        if not np.isfinite(value).all():raise ValueError('Nonfinite actual frozen RGB view')
        value=value.copy();value.setflags(write=False);_CACHE[key]=value
        while len(_CACHE)>MAX_CACHED_VIEWS:_CACHE.popitem(last=False)
        return value,dict(input_sha256=key[2],cache_hit=False,new_encoder_forwards=1,
                          side=canvas.shape[0],native_shape=list(value.shape),binding=binding)


def _base_canvas(ep,role):
    from ics.cpu100.common import rgb_view
    return rgb_view(ep,role)


def _shift(canvas,dy=0,dx=0):
    out=np.broadcast_to(np.array((124,116,104),np.uint8),canvas.shape).copy()
    h,w=canvas.shape[:2]
    source_y=slice(max(0,-dy),min(h,h-dy));source_x=slice(max(0,-dx),min(w,w-dx))
    target_y=slice(max(0,dy),min(h,h+dy));target_x=slice(max(0,dx),min(w,w+dx))
    out[target_y,target_x]=canvas[source_y,source_x]
    return out


def _unit(raw):
    raw=np.asarray(raw,np.float32)
    return raw/np.maximum(np.linalg.norm(raw,axis=-1,keepdims=True),H.EPS)


def _valid_operator(shape,hw,g):
    from ics.cpu100.common import grid_coverage
    # Shifted observed image may be clipped by the encoder canvas. The actual
    # real-image portion of each footprint is computed, without declaring the
    # artificial fill to be a known background label.
    side=float(g['view_side']);sh,sw=g['resized_hw'];oy,ox=g['padding_top_left']
    y0=np.arange(hw[0])*side/hw[0];x0=np.arange(hw[1])*side/hw[1]
    cy=np.maximum(0,np.minimum(y0+side/hw[0],min(oy+sh,side))-np.maximum(y0,max(oy,0)))/(side/hw[0])
    cx=np.maximum(0,np.minimum(x0+side/hw[1],min(ox+sw,side))-np.maximum(x0,max(ox,0)))/(side/hw[1])
    valid=(cy[:,None]*cx[None]).ravel()
    return H.footprint(shape,hw,g),valid


def _phase_geometry(g,dy,dx):
    result=dict(g);oy,ox=g['padding_top_left'];result['padding_top_left']=[oy+dy,ox+dx]
    return result


def difference_tv(U,D,target,sigma,rgb,max_steps=2000,tv=True):
    """E248 exact normalized strictly-convex objective and its dual certificate."""
    u0=np.asarray(U,float).ravel();p=len(u0);k=max(len(target),1);target=np.asarray(target,float)
    edges,capacity=H.rgb_edges(rgb);capacity=capacity/max(len(edges),1)
    if not tv:capacity[:]=0
    G=sparse.csr_matrix((np.tile((-1.,1.),len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),p))
    K=sparse.vstack((D,G),format='csr');column=np.asarray(abs(K).sum(0)).ravel();row=np.asarray(abs(K).sum(1)).ravel()
    # Diagonal CP preconditioning with a fixed sqrt(P) balance. Every original
    # pixel and both phase constraints retain the document's normalization.
    balance=np.sqrt(p);tau=.99*balance/np.maximum(column,H.EPS);step=.99/(balance*np.maximum(row,H.EPS))
    sa=step[:D.shape[0]];st=step[D.shape[0]:];a=np.zeros(D.shape[0]);t=np.zeros(len(edges));u=u0.copy();bar=u.copy()
    for iteration in range(max_steps):
        a=(a+sa*(D@bar-target))/(1+sa*k*sigma*sigma)
        t=np.clip(t+st*(G@bar),-capacity,capacity)
        previous=u.copy();u=np.clip((u-tau*(D.T@a+G.T@t)+(tau/p)*u0)/(1+tau/p),0,1);bar=2*u-previous
        if iteration%10==0 or iteration==max_steps-1:
            residual=D@u-target
            primal=float(np.sum((u-u0)**2)/(2*p)+(residual@residual)/(2*k*sigma*sigma)+capacity@np.abs(G@u))
            v=-(D.T@a+G.T@t);opt=np.clip(u0+p*v,0,1)
            conjugate=float(v@opt-np.sum((opt-u0)**2)/(2*p))
            dual=float(-conjugate-target@a-(k*sigma*sigma/2)*(a@a))
            gap=max(0.,primal-dual);tolerance=1e-5*max(1e-12,abs(primal),abs(dual))
            if gap<=tolerance:break
    return u.reshape(U.shape),dict(iterations=iteration+1,primal=primal,dual=dual,gap=gap,gap_tolerance=tolerance,
                  converged=bool(gap<=tolerance),original_pixels=p,valid_phase_observations=k,
                  D_constant_residual=float(np.max(np.abs(D@np.ones(p)),initial=0)),
                  objective='||u-U||²/(2P)+||Du-delta||²/(2Ksigma²)+RGBTV/(8|E|)',
                  optimizer='diagonally_preconditioned_ChambollePock')


def e248_shift_difference(problem,control=None):
    ep=problem.ep
    if ep.producer.get('model_input_side')!=1024 or ep.q_hw!=(64,64) or ep.r_hw!=(64,64):
        raise common.ArtifactUnavailable('E248 specified actual base1024 and +8pixel H/V views, not a128 grid surrogate')
    canvases={role:_base_canvas(ep,role) for role in ('r','q')};raw={};receipts=[]
    for role in ('r','q'):
        for name,dy,dx in (('horizontal',0,8),('vertical',8,0)):
            value,receipt=encode(ep,role,_shift(canvases[role],dy,dx));raw[role,name]=_unit(value).reshape(-1,value.shape[-1]);receipts.append(dict(role=role,phase=name,**receipt))
    if control=='same_views_head':
        fields=[]
        for name,dy,dx in (('horizontal',0,8),('vertical',8,0)):
            rg=_phase_geometry(ep.reference_geometry,dy,dx);A,rv=_valid_operator(ep.r_rgb.shape[:2],ep.r_hw,rg)
            coverage=A@np.asarray(ep.reference_mask,float).ravel()
            probability,head_info=H.direct_u0(raw['q',name],raw['r',name],coverage*rv,rv)
            qg=_phase_geometry(ep.query_geometry,dy,dx)
            field=H.native_to_original(probability.reshape(ep.q_hw),ep.original_shape,qg)
            # A clipped new view has no evidence for the final 8 canvas pixels;
            # those original locations retain U rather than a clipped token.
            oy,ox=qg['padding_top_left'];sh,sw=qg['resized_hw'];side=qg['view_side']
            yy,xx=np.indices(ep.original_shape)
            observed=(oy+(yy+.5)*sh/ep.original_shape[0]<side)&(ox+(xx+.5)*sw/ep.original_shape[1]<side)
            field=np.where(observed,field,problem.U);fields.append(field)
        return (problem.U+fields[0]+fields[1])/3,dict(status='ok',
                control='same_real_six_views_matching_reference_query_heads_and_average',encoder_views=receipts,
                independent_of_inverse_beta_gate=True,phase_masks_aligned_to_actual_views=True)
    rshape=ep.r_rgb.shape[:2];r0,rvalid=_valid_operator(rshape,ep.r_hw,ep.reference_geometry)
    source_target=[];source_dm=[];betas=[];h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw)
    for by in range(2):
        for bx in range(2):
            held=((yy>=by*h//2)&(yy<(by+1)*h//2)&(xx>=bx*w//2)&(xx<(bx+1)*w//2))
            train=(ep.wvalid>0)&~ndimage.binary_dilation(held,structure=np.ones((3,3))).ravel()
            cov=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0)
            f=train&(cov>=.9);b=train&(cov<=.1)
            if not f.any() or not b.any():return problem.U.copy(),dict(status='fallback_missing_fold_base_pure_roles',encoder_views=receipts)
            pf=np.mean(ep.r[f],0);pf/=max(np.linalg.norm(pf),H.EPS);pb=np.mean(ep.r[b],0);pb/=max(np.linalg.norm(pb),H.EPS);direction=pf-pb
            rho=[];dm=[];interior=ndimage.binary_erosion(held,structure=np.ones((3,3)),border_value=0).ravel()
            for name,dy,dx in (('horizontal',0,8),('vertical',8,0)):
                shift,valid=_valid_operator(rshape,ep.r_hw,_phase_geometry(ep.reference_geometry,dy,dx))
                keep=interior&(rvalid>=1-1e-10)&(valid>=1-1e-10)
                if not keep.any():continue
                rho.append(np.asarray((shift-r0)@np.asarray(ep.reference_mask,float).ravel())[keep])
                dm.append(((raw['r',name]-ep.r)@direction)[keep])
            if not rho:return problem.U.copy(),dict(status='fallback_no_valid_source_phase_pair',encoder_views=receipts)
            rho=np.concatenate(rho);dm=np.concatenate(dm);den=float(rho@rho)
            beta=float(rho@dm/den) if den>H.EPS else 0.
            betas.append(beta);source_target.append(rho);source_dm.append(dm)
    if len(betas)!=4 or min(betas)<=0:return problem.U.copy(),dict(status='fallback_four_source_betas_not_positive',source_betas=betas,encoder_views=receipts)
    rho=np.concatenate(source_target);dm=np.concatenate(source_dm);beta=float(rho@dm/max(rho@rho,H.EPS))
    sigma=max(.05,float(np.sqrt(np.mean(((dm-beta*rho)/beta)**2))))
    cov=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0);known=ep.wvalid>0
    pf=np.mean(ep.r[known&(cov>=.9)],0);pf/=max(np.linalg.norm(pf),H.EPS)
    pb=np.mean(ep.r[known&(cov<=.1)],0);pb/=max(np.linalg.norm(pb),H.EPS);direction=pf-pb
    base,valid0=_valid_operator(ep.original_shape,ep.q_hw,ep.query_geometry);matrices=[];deltas=[]
    for name,dy,dx in (('horizontal',0,8),('vertical',8,0)):
        shifted,valid=_valid_operator(ep.original_shape,ep.q_hw,_phase_geometry(ep.query_geometry,dy,dx));keep=(valid>=1-1e-10)&(valid0>=1-1e-10)
        matrices.append((shifted-base)[keep]);deltas.append(((raw['q',name]-ep.q)@direction)[keep]/beta)
    D=sparse.vstack(matrices,format='csr');delta=np.concatenate(deltas)
    if control=='no_RGB':
        field,certificate=difference_tv(problem.U,D,delta,sigma,ep.q_rgb,tv=False)
    else:field,certificate=difference_tv(problem.U,D,delta,sigma,ep.q_rgb)
    if not certificate['converged']:field=problem.U.copy()
    return field,dict(status='ok' if certificate['converged'] else 'fallback_difference_inverse_uncertified',
                    source_betas=betas,pooled_beta=beta,sigma_coverage_difference=sigma,encoder_views=receipts,
                    solver=certificate,base_input_pair_cost_separately_accounted=2,maximum_extra_views=4,control=control)


def _weighted_margin(q,r,wf,valid):
    f=np.sum(r*np.asarray(wf)[:,None],0);b=np.sum(r*(np.asarray(valid)-wf)[:,None],0)
    if np.linalg.norm(f)<=H.EPS or np.linalg.norm(b)<=H.EPS:return None
    return (q@(f/np.linalg.norm(f)-b/np.linalg.norm(b)))/.1


def _resize_geometry(g,factor):
    result=dict(g);result['view_side']=int(g['view_side']*factor)
    result['resized_hw']=[int(v*factor) for v in g['resized_hw']]
    result['padding_top_left']=[int(v*factor) for v in g['padding_top_left']]
    return result


def e246_high_resolution_router(problem,control=None):
    ep=problem.ep
    if ep.producer.get('model_input_side')!=1024:raise common.ArtifactUnavailable('E246 requires actual1024+2048 views')
    features={};geometry={};receipts=[]
    for role in ('r','q'):
        canvas=np.asarray(Image.fromarray(_base_canvas(ep,role)).resize((2048,2048),Image.Resampling.BILINEAR))
        raw,receipt=encode(ep,role,canvas);features[role]=_unit(raw).reshape(-1,raw.shape[-1]);receipts.append(dict(role=role,**receipt))
        geometry[role]=_resize_geometry(ep.reference_geometry if role=='r' else ep.query_geometry,2)
    R,rv=_valid_operator(ep.r_rgb.shape[:2],(128,128),geometry['r'])
    full_fg=(R@np.asarray(ep.reference_mask,float).ravel())*rv
    low=_weighted_margin(ep.q,ep.r,ep.wf,ep.wvalid);high=_weighted_margin(features['q'],features['r'],full_fg,rv)
    if low is None or high is None:return problem.U.copy(),dict(status='fallback_scale_class_missing',encoder_views=receipts)
    qlow=H.native_to_original(low.reshape(ep.q_hw),ep.original_shape,ep.query_geometry)
    qhigh=H.native_to_original(high.reshape(128,128),ep.original_shape,geometry['q'])
    h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw);rlow=np.full(problem.rvalid.shape,np.nan);rhigh=rlow.copy();folds=0
    for by in range(2):
        for bx in range(2):
            held=((yy>=by*h//2)&(yy<(by+1)*h//2)&(xx>=bx*w//2)&(xx<(bx+1)*w//2)).ravel()
            excluded=ndimage.binary_dilation(held.reshape(ep.r_hw),structure=np.ones((3,3))).ravel()
            train=problem.rvalid&~excluded[problem.rtokens];test=problem.rvalid&held[problem.rtokens]
            lowm=_weighted_margin(ep.r,ep.r,np.where(excluded,0,ep.wf),np.where(excluded,0,ep.wvalid))
            # Every high-resolution label and partial-valid weight comes from
            # the same legal original training pixels; held labels are unknown.
            hv=(R@train.astype(float))*rv;hf=(R@(train*problem.labels).astype(float))*rv
            highm=_weighted_margin(features['r'],features['r'],hf,hv)
            if lowm is None or highm is None or not test.any():continue
            lowp=H.native_to_original(lowm.reshape(ep.r_hw),ep.r_rgb.shape[:2],ep.reference_geometry).ravel()
            highp=H.native_to_original(highm.reshape(128,128),ep.r_rgb.shape[:2],geometry['r']).ravel()
            rlow[test]=lowp[test];rhigh[test]=highp[test];folds+=1
    known=np.isfinite(rlow)&np.isfinite(rhigh)&problem.rvalid
    if folds<2:return problem.U.copy(),dict(status='fallback_no_legal_scale_router_tasks',encoder_views=receipts)
    target=problem.labels.astype(float)
    lowloss=np.logaddexp(0,rlow[known])-rlow[known]*target[known]
    highloss=np.logaddexp(0,rhigh[known])-rhigh[known]*target[known]
    prefer=np.zeros(len(target),bool);prefer[known]=highloss<lowloss
    rstructure=H.texture_phi(ep.r_rgb)[:,4];qstructure=H.texture_phi(ep.q_rgb)[:,4]
    rdesign=np.c_[rstructure,np.nan_to_num(rlow),np.nan_to_num(rhigh)]
    qdesign=np.c_[qstructure,qlow.ravel(),qhigh.ravel()]
    head=H.fit_pixel_logistic(rdesign,np.zeros((1,1)),np.zeros(len(target),int),prefer,known)
    if control=='average':route=np.full(ep.original_shape,.5)
    elif control=='maximum_margin':route=(np.abs(qhigh)>np.abs(qlow)).astype(float)
    elif control=='source_best' or head is None:
        route=np.full(ep.original_shape,float(highloss.mean()<lowloss.mean()))
    else:route=(head.predict(qdesign,np.zeros((1,1)),np.zeros(len(qdesign),int))>.5).reshape(ep.original_shape).astype(float)
    delta=qhigh-qlow;scale=max(float(np.median(np.abs(rhigh[known]-rlow[known]))),H.EPS)
    # The supplied MEAN remains the complete host; both raw same-scale heads
    # contribute the newly observed scale difference rather than silently
    # replacing this card by a U0 host.
    out,certificate=C._cut(problem,route*delta/scale)
    return out,dict(status='ok',binary_optimizer=True,encoder_views=receipts,actual_input_sides=[1024,2048],
                    source_router_folds=folds,source_high_preference_pixels=int(prefer[known].sum()),
                    router_fit=None if head is None else head.info,source_scale_delta_MAD=scale,
                    source_low_BCE=float(lowloss.mean()),source_high_BCE=float(highloss.mean()),
                    routed_high_pixels=int(np.sum(route>.5)),cut=certificate,control=control,
                    high_resolution_tokens_factor=4,attention_quadratic_factor=16)


def _mask_on_canvas(mask,g):
    side=g['view_side'];sh,sw=g['resized_hw'];oy,ox=g['padding_top_left']
    out=np.zeros((side,side),bool)
    value=np.asarray(Image.fromarray(np.asarray(mask,np.uint8)*255).resize((sw,sh),Image.Resampling.NEAREST))>0
    out[oy:oy+sh,ox:ox+sw]=value
    return out


def _crop_geometry(g,y,x,width):
    factor=g['view_side']/width;sh,sw=g['resized_hw'];oy,ox=g['padding_top_left']
    return dict(g,view_side=g['view_side'],resized_hw=[sh*factor,sw*factor],
                padding_top_left=[(oy-y)*factor,(ox-x)*factor])


def _crop_canvas(canvas,y,x,width):
    return np.asarray(Image.fromarray(canvas[y:y+width,x:x+width]).resize(canvas.shape[:2][::-1],Image.Resampling.BILINEAR))


def _observed_pixels(shape,g):
    yy,xx=np.indices(shape);sh,sw=g['resized_hw'];oy,ox=g['padding_top_left'];side=g['view_side']
    cy=oy+(yy+.5)*sh/shape[0];cx=ox+(xx+.5)*sw/shape[1]
    return (cy>=0)&(cy<side)&(cx>=0)&(cx<side)


def _tile_features(rgb,U,mask,token_ids):
    if not mask.any():return np.zeros(3)
    values=np.asarray(rgb,float)[mask]/255.
    return np.array([float(np.mean(.5-np.abs(U[mask]-.5))),float(np.mean(values.std(0))),
                     1/max(len(np.unique(token_ids[mask.ravel()])),1)])


def _tile_proposals(problem):
    """Fixed-grid plus observed small uncertain RGB/semantic domains; <=32 boxes."""
    ep=problem.ep;g=ep.query_geometry;side=int(g['view_side']);sh,sw=g['resized_hw'];oy,ox=g['padding_top_left']
    boxes={}
    def add(center,width,kind,mask=None):
        y=int(np.clip(round(center[0]-width/2),0,side-width));x=int(np.clip(round(center[1]-width/2),0,side-width))
        key=(y,x,width)
        cropg=_crop_geometry(g,y,x,width);covered=_observed_pixels(ep.original_shape,cropg)
        if covered.any():boxes[key]=(kind,covered,_tile_features(ep.q_rgb,problem.U,covered,problem.qtokens))
    for y in (side/4,3*side/4):
        for x in (side/4,3*side/4):add((y,x),side//2,'fixed_grid')
    Aq,_=_valid_operator(ep.original_shape,ep.q_hw,g)
    colours=Aq@(np.asarray(ep.q_rgb,float).reshape(-1,3)/255.)
    bins=np.minimum(3,np.floor(4*colours).astype(int));code=(bins[:,0]*16+bins[:,1]*4+bins[:,2]).reshape(ep.q_hw)
    coarse=np.asarray(Aq@problem.U.ravel()).reshape(ep.q_hw)
    proposed=[]
    for value in np.unique(code):
        components,n=ndimage.label((code==value)&(ep.q_valid.reshape(ep.q_hw)>0))
        for j in range(1,n+1):
            ids=np.flatnonzero(components.ravel()==j)
            if not len(ids) or len(ids)>16:continue
            conflict=float(np.mean(.5-np.abs(coarse.ravel()[ids]-.5)))
            proposed.append((-(conflict/np.sqrt(len(ids))),int(ids[0]),ids))
    components,n=ndimage.label((coarse>.4)&(coarse<.6)&(ep.q_valid.reshape(ep.q_hw)>0))
    for j in range(1,n+1):
        ids=np.flatnonzero(components.ravel()==j)
        if 0<len(ids)<=16:proposed.append((-.5/np.sqrt(len(ids)),int(ids[0]),ids))
    proposed.sort(key=lambda a:(a[0],a[1]))
    for _,_,ids in proposed[:28]:
        y,x=np.divmod(ids,ep.q_hw[1]);center=((y.mean()+.5)*side/ep.q_hw[0],(x.mean()+.5)*side/ep.q_hw[1])
        add(center,side//4 if len(ids)<=4 else side//2,'small_observed_domain')
    return boxes


def e247_adaptive_tiles(problem,control=None):
    ep=problem.ep
    if ep.producer.get('model_input_side')!=1024:raise common.ArtifactUnavailable('E247 requires physical1024 base and actual1024 encoded tiles')
    proposals=_tile_proposals(problem)
    if not proposals:return problem.U.copy(),dict(status='fallback_no_observed_tile')
    # Two label-independent center crops at magnifications2/4. The same RGB
    # inputs can be used for every held fold; their banks are rebuilt legally.
    rcanvas=_base_canvas(ep,'r');scales={};receipts=[]
    for width in (512,256):
        y=x=(1024-width)//2;raw,receipt=encode(ep,'r',_crop_canvas(rcanvas,y,x,width));receipts.append(dict(role='r',tile=[y,x,width],**receipt))
        g=_crop_geometry(ep.reference_geometry,y,x,width);A,valid=_valid_operator(ep.r_rgb.shape[:2],ep.r_hw,g)
        scales[width]=dict(z=_unit(raw).reshape(-1,raw.shape[-1]),A=A,valid=valid,g=g,observed=_observed_pixels(ep.r_rgb.shape[:2],g))
    source=[];scale_error={};scale_benefit={};fold_counts={};h,w=ep.r_hw;yy,xx=np.indices(ep.r_hw)
    source_logits=[]
    for width,view in scales.items():
        errors=[];gains=[];folds=0
        for by in range(2):
            for bx in range(2):
                held=((yy>=by*h//2)&(yy<(by+1)*h//2)&(xx>=bx*w//2)&(xx<(bx+1)*w//2)).ravel()
                excluded=ndimage.binary_dilation(held.reshape(ep.r_hw),structure=np.ones((3,3))).ravel()
                train=problem.rvalid&~excluded[problem.rtokens]
                valid=(view['A']@train.astype(float))*view['valid'];wf=(view['A']@(train*problem.labels).astype(float))*view['valid']
                high=_weighted_margin(view['z'],view['z'],wf,valid)
                old=H.direct_u0(ep.r,ep.r,np.where(excluded,0,ep.wf),np.where(excluded,0,ep.wvalid))[0]
                oldp=H.native_to_original(old.reshape(ep.r_hw),ep.r_rgb.shape[:2],ep.reference_geometry)
                test=problem.rvalid&held[problem.rtokens]&view['observed'].ravel()
                if high is None or not test.any():continue
                hp=H.native_to_original(high.reshape(ep.r_hw),ep.r_rgb.shape[:2],view['g']).ravel()
                oldlogit=np.log(np.clip(oldp.ravel()[test],H.EPS,1-H.EPS)/np.clip(1-oldp.ravel()[test],H.EPS,1-H.EPS))
                labels=problem.labels[test].astype(float)
                oldloss=np.logaddexp(0,oldlogit)-labels*oldlogit
                newloss=np.logaddexp(0,hp[test])-labels*hp[test]
                errors.append(float(newloss.mean()));gains.append(float(np.mean(oldloss-newloss)));folds+=1
                mask=test.reshape(ep.r_rgb.shape[:2]);phi=_tile_features(ep.r_rgb,oldp,mask,problem.rtokens)
                source.append((phi,gains[-1],width));source_logits.append(dict(width=width,fold=[by,bx],pixels=int(test.sum()),old_BCE=float(oldloss.mean()),tile_BCE=errors[-1]))
        fold_counts[width]=folds
        if folds>=2:scale_error[width]=max(float(np.mean(errors)),H.EPS);scale_benefit[width]=float(np.mean(gains))
    if not scale_error:return problem.U.copy(),dict(status='fallback_no_two_valid_tile_source_folds',encoder_views=receipts,source_folds=fold_counts)
    features=np.stack([s[0] for s in source]);gain=np.array([s[1] for s in source]);mean=features.mean(0);std=np.maximum(features.std(0),H.EPS)
    design=np.c_[(features-mean)/std,np.ones(len(features))];coef=np.linalg.solve(design.T@design+np.diag([1.,1.,1.,0.]),design.T@gain)
    ranked=[]
    for key,(kind,mask,phi) in proposals.items():
        width=key[2]
        if width not in scale_error:continue
        priority=float(np.r_[(phi-mean)/std,1.]@coef)
        # Source-calibrated gain plus conflict/smallness, no target-label or
        # predicted-reference instance count is used as a query area budget.
        ranked.append((-(priority*max(phi[0],0)*max(phi[2],H.EPS)),key,kind,mask,priority))
    if control=='uniform':ranked.sort(key=lambda v:(v[2]!='fixed_grid',v[1]))
    elif control=='random':ranked.sort(key=lambda v:hashlib.sha256(repr(v[1]).encode()).hexdigest())
    else:ranked.sort(key=lambda v:(v[0],v[1]))
    if control is None and (not ranked or max(v[4] for v in ranked)<=0):
        return problem.U.copy(),dict(status='fallback_reference_no_positive_tile_gain',encoder_views=receipts,source_fold_losses=source_logits)
    selected=ranked[:4];value=np.zeros(problem.U.size);mass=np.zeros(problem.U.size);qcanvas=_base_canvas(ep,'q')
    for _,key,kind,mask,priority in selected:
        y,x,width=key;view=scales[width]
        wf=(view['A']@problem.labels.astype(float))*view['valid'];raw,receipt=encode(ep,'q',_crop_canvas(qcanvas,y,x,width));receipts.append(dict(role='q',tile=list(key),kind=kind,**receipt))
        margin=_weighted_margin(_unit(raw).reshape(-1,raw.shape[-1]),view['z'],wf,view['valid'])
        if margin is None:continue
        g=_crop_geometry(ep.query_geometry,y,x,width);mapped=H.native_to_original(margin.reshape(ep.q_hw),ep.original_shape,g).ravel()
        weight=1. if control=='same_views_average' else 1/scale_error[width]
        value[mask.ravel()]+=weight*mapped[mask.ravel()];mass[mask.ravel()]+=weight
    measured=mass>0
    base=np.clip(problem.U.ravel(),H.EPS,1-H.EPS);old=np.log(base/(1-base))
    # Reference loss supplies fusion precision, with finite host precision1.
    fused=old.copy();fused[measured]=(old[measured]+value[measured])/(1+mass[measured])
    evidence=fused-old;out,certificate=C._cut(problem,evidence)
    out.ravel()[~measured]=problem.U.ravel()[~measured]
    return out,dict(status='ok',binary_optimizer=True,encoder_views=receipts,source_fold_losses=source_logits,
            source_folds=fold_counts,source_scale_error=scale_error,source_scale_gain=scale_benefit,
            priority_coefficients=coef.tolist(),tile_proposals=len(proposals),selected_tiles=[list(v[1]) for v in selected],
            measured_original_pixels=int(measured.sum()),maximum_extra_views=6,total_with_base_pair=8,
            source_crops_label_independent=True,cut=certificate,control=control,
            implementation_assumption='fixed_center_reference512/256crops; source_BCE_inverse_precision; priority_ridge3features')


def _replace_with_initial_bg(canvas,region,background):
    if not background.any():return None
    # Every replaced pixel comes from actual same-image initial BG. Outside
    # region is byte-identical; no uniform gray deletion canvas is substituted.
    _,indices=ndimage.distance_transform_edt(~background,return_indices=True)
    output=canvas.copy();output[region]=canvas[indices[0][region],indices[1][region]]
    return output


def e249_candidate_occlusion(problem,control=None):
    ep=problem.ep;anchors=B._anchors(problem).reshape(ep.original_shape)
    if not np.any(anchors==-1):return problem.U.copy(),dict(status='fallback_no_initial_query_BG_texture')
    cov=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf),where=ep.wvalid>0)
    f=(ep.wvalid>0)&(cov>=.9);b=(ep.wvalid>0)&(cov<=.1)
    if not f.any() or not b.any():return problem.U.copy(),dict(status='fallback_no_two_pure_reference_roles')
    pf=np.mean(ep.r[f],0);pf/=max(np.linalg.norm(pf),H.EPS);pb=np.mean(ep.r[b],0);pb/=max(np.linalg.norm(pb),H.EPS);direction=pf-pb
    reference=[]
    # Two equal-sized foreground and two background visible patches provide
    # class-conditioned original/inner/outer responses, not class labels on
    # the altered image. Labels refer to each original reference candidate.
    for cls,ids in ((True,np.flatnonzero(f)),(False,np.flatnonzero(b))):
        for index in B._spread_rows(ids,2):
            from .e226_234 import patch_geometry
            geometry=patch_geometry(ep.r_rgb.shape[:2],ep.r_hw,ep.reference_geometry,int(index))
            mask=np.zeros(ep.r_rgb.shape[:2],bool);mask.ravel()[geometry['ids']]=True
            reference.append((mask,cls))
    regions=[];components,count=ndimage.label(problem.U>.5)
    # Bound dense original-resolution masks as they are proposed, including
    # semantic islands. A checkerboard must not allocate one H*W mask per pixel.
    def propose(mask):
        nonlocal regions
        if mask.any():regions.append(mask)
        if len(regions)>64:
            regions.sort(key=lambda a:(float(np.mean(np.abs(problem.U[a]-.5))),int(np.flatnonzero(a)[0])))
            regions=regions[:32]
    for j in range(1,count+1):propose(components==j)
    yy,xx=np.indices(ep.original_shape)
    for gy in range(4):
        for gx in range(4):propose((yy>=gy*ep.original_shape[0]//4)&(yy<(gy+1)*ep.original_shape[0]//4)&
                  (xx>=gx*ep.original_shape[1]//4)&(xx<(gx+1)*ep.original_shape[1]//4))
    # Actual RGB domains add candidates independently of current foreground.
    # Native-grid colour grouping bounds proposal memory; its masks are then
    # mapped by physical cell identity to the complete original image.
    Aq,_=_valid_operator(ep.original_shape,ep.q_hw,ep.query_geometry)
    native_colour=Aq@(np.asarray(ep.q_rgb,float).reshape(-1,3)/255.)
    bins=np.minimum(3,np.floor(4*native_colour).astype(int));code=(bins[:,0]*16+bins[:,1]*4+bins[:,2]).reshape(ep.q_hw)
    for value in np.unique(code):
        component,n=ndimage.label((code==value)&(ep.q_valid.reshape(ep.q_hw)>0))
        for j in range(1,n+1):
            native_mask=component.ravel()==j
            mask=native_mask[problem.qtokens].reshape(ep.original_shape)
            propose(mask)
    regions=[v for v in regions if v.any()]
    regions.sort(key=lambda mask:(float(np.mean(np.abs(problem.U[mask]-.5))) if mask.any() else np.inf,int(np.flatnonzero(mask)[0]) if mask.any() else 0))
    query=regions[:4];responses={};receipts=[]
    for role,candidates in (('r',[v[0] for v in reference]),('q',query)):
        canvas=_base_canvas(ep,role);g=ep.reference_geometry if role=='r' else ep.query_geometry
        shape=ep.r_rgb.shape[:2] if role=='r' else ep.original_shape;hw=ep.r_hw if role=='r' else ep.q_hw
        A,_=_valid_operator(shape,hw,g);z=ep.r if role=='r' else ep.q;original=z@direction
        bg=(~np.asarray(ep.reference_mask,bool)) if role=='r' else (anchors==-1)
        bg_canvas=_mask_on_canvas(bg,g);role_response=[]
        for j,mask in enumerate(candidates):
            footprint=np.asarray(A@mask.ravel().astype(float));weights=footprint/np.maximum(footprint.sum(),H.EPS)
            before=float(weights@original);values=[before];after_means=[]
            interior=_mask_on_canvas(mask,g);outer=ndimage.binary_dilation(interior,iterations=16)&~interior
            for name,changed in (('interior',interior),('outer_ring',outer)):
                altered=_replace_with_initial_bg(canvas,changed,bg_canvas)
                if altered is None:break
                raw,receipt=encode(ep,role,altered);unit=_unit(raw).reshape(-1,raw.shape[-1]);after=float(weights@(unit@direction))
                values.append(before-after);after_means.append(after);receipts.append(dict(role=role,candidate=j,operation=name,**receipt))
            if len(values)==3:role_response.append(values)
        responses[role]=np.asarray(role_response,float).reshape(-1,3)
    if len(responses['r'])!=len(reference) or len(responses['q'])!=len(query):
        return problem.U.copy(),dict(status='fallback_occlusion_material_missing',encoder_views=receipts)
    target=np.array([v[1] for v in reference]);source=responses['r'];query_values=responses['q']
    if control=='original_only':source=source[:,:1];query_values=query_values[:,:1]
    elif control=='deleted_patch_mean':source=np.c_[source[:,0]-source[:,1],source[:,0]-source[:,2]];query_values=np.c_[query_values[:,0]-query_values[:,1],query_values[:,0]-query_values[:,2]]
    scale=np.maximum(source.std(0),H.EPS);source=source/scale;query_values=query_values/scale
    bandwidth=max(float(np.median(np.sum((source[:,None]-source[None])**2,2))),H.EPS)
    logkernel=-np.sum((query_values[:,None]-source[None])**2,2)/bandwidth
    margin=logsumexp(logkernel[:,target],axis=1)-np.log(target.sum())-logsumexp(logkernel[:,~target],axis=1)+np.log((~target).sum())
    evidence=np.zeros(problem.U.size);mass=np.zeros(problem.U.size)
    for mask,value in zip(query,margin):evidence[mask.ravel()]+=value;mass[mask.ravel()]+=1
    evidence=np.divide(evidence,mass,out=np.zeros_like(evidence),where=mass>0)
    output,certificate=C._cut(problem,evidence);output.ravel()[mass==0]=problem.U.ravel()[mass==0]
    return output,dict(status='ok',binary_optimizer=True,reference_candidates=len(reference),query_candidates=len(query),
                   encoder_views=receipts,extra_views_maximum=16,total_with_base_pair_maximum=18,
                   response_features='original_fixed_bank_margin,inner_drop,outer_drop',source_bandwidth=bandwidth,
                   no_unseen_pixels_filled=True,not_semantic_causal_proof=True,cut=certificate,control=control)


METHODS={'E248':lambda ep:B._call(ep,'E248',e248_shift_difference),
         'E246':lambda ep:B._call(ep,'E246',e246_high_resolution_router)}
METHODS['E249']=lambda ep:B._call(ep,'E249',e249_candidate_occlusion)
METHODS['E247']=lambda ep:B._call(ep,'E247',e247_adaptive_tiles)
CONTROLS={'E248_same_six_views_average_head':lambda ep:B._call(ep,'E248',e248_shift_difference,control='same_views_head'),
          'E248_same_delta_without_RGB':lambda ep:B._call(ep,'E248',e248_shift_difference,control='no_RGB')}
CONTROLS.update(E246_same_two_scales_average=lambda ep:B._call(ep,'E246',e246_high_resolution_router,control='average'),
                E246_same_two_scales_max_margin=lambda ep:B._call(ep,'E246',e246_high_resolution_router,control='maximum_margin'),
                E246_source_best_single_scale=lambda ep:B._call(ep,'E246',e246_high_resolution_router,control='source_best'))
CONTROLS.update(E249_original_same_candidates=lambda ep:B._call(ep,'E249',e249_candidate_occlusion,control='original_only'),
                E249_same_actual_deleted_patch_means=lambda ep:B._call(ep,'E249',e249_candidate_occlusion,control='deleted_patch_mean'))
RESOURCES={'E248':dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=True,
                      actual_artifacts=['frozen_encode_rgb'],extra_encoder_forwards=4,total_encoder_views=6)}
RESOURCES['E246']=dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=True,
                       actual_artifacts=['frozen_encode_rgb'],extra_encoder_forwards=2,
                       extra_view_side=2048,total_cost='2C(N)+2C(4N), not4ordinaryforwards')
RESOURCES['E249']=dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=True,
                       actual_artifacts=['frozen_encode_rgb'],extra_encoder_forwards_maximum=16,total_encoder_views_maximum=18)
CONTROLS.update(E247_same_selected_views_average=lambda ep:B._call(ep,'E247',e247_adaptive_tiles,control='same_views_average'),
                E247_same_budget_uniform_tiles=lambda ep:B._call(ep,'E247',e247_adaptive_tiles,control='uniform'),
                E247_same_budget_fixed_random_tiles=lambda ep:B._call(ep,'E247',e247_adaptive_tiles,control='random'))
RESOURCES['E247']=dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=True,
                      actual_artifacts=['frozen_encode_rgb'],extra_encoder_forwards_maximum=6,total_encoder_views_maximum=8)
