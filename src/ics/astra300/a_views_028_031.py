"""Actual R-only frozen-DINO view methods; no feature-vector pseudo augmentation."""
from __future__ import annotations

from collections import OrderedDict
from functools import partial
import time
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.special import logsumexp

from ics.cpu100.common import rgb_view
from .common import array_hash,require_artifact
from .a_helpers_001_050 import (EPS,Frame,ObservationUnavailable,b0,fps,infer_a,
    nearest_mean_distance,rank_basis,sqdist,unit)
from .a_observations_021_048 import _binding


_CACHE=OrderedDict()


def _plain(frame):return lambda q,ids,role:b0(frame,q),{'mechanism':'A_B0_5NN'}


def _canvas(frame):
    ep=frame.ep;image=rgb_view(ep,'r').astype(np.float32)/255
    g=ep.reference_geometry;side=g['view_side'];sh,sw=g['resized_hw'];oy,ox=g['padding_top_left']
    if ep.reference_mask is None or ep.reference_mask.shape!=ep.r_rgb.shape[:2]:
        raise ObservationUnavailable('Actual complete original MR required for mask-guided views')
    mask=np.zeros((side,side),bool)
    mask[oy:oy+sh,ox:ox+sw]=np.asarray(Image.fromarray(ep.reference_mask.astype(np.uint8)).resize((sw,sh),Image.Resampling.NEAREST))>0
    valid=np.zeros((side,side),bool);valid[oy:oy+sh,ox:ox+sw]=True
    yy=(np.arange(side)*frame.ep.r_hw[0]/side).astype(int);xx=(np.arange(side)*frame.ep.r_hw[1]/side).astype(int)
    train=frame.train.reshape(ep.r_hw)[yy[:,None],xx[None,:]]
    return image,mask,valid,train


def _encode(frame,image,tag):
    binding=_binding(frame.ep);callback=require_artifact(frame.ep,'frozen_encode_rgb')
    if not callable(callback):raise ObservationUnavailable('Actual frozen RGB encoder callback required')
    assets=binding.get('model_assets',binding)
    key=(frame.ep.source_id,array_hash(image),str(assets.get('checkpoint_sha256')),
         str(assets.get('config_sha256')),tuple(frame.ep.r_hw),frame.x.shape[1])
    if key in _CACHE:
        value=_CACHE.pop(key);_CACHE[key]=value;return value,True
    value=np.asarray(callback('r',image,working_canvas=True,side=image.shape[0],raw=False))
    if value.shape!=tuple(frame.ep.r_hw)+(frame.x.shape[1],) or not np.isfinite(value).all():
        raise ObservationUnavailable('Actual augmented RGB forward must return aligned native patch grid')
    value=unit(value.reshape(len(frame.x),frame.x.shape[1])).astype(np.float32)
    value.setflags(write=False);_CACHE[key]=value
    while len(_CACHE)>16:_CACHE.popitem(last=False)
    return value,False


def _profile_readout(frame,views):
    source=np.flatnonzero(frame.train);fi=fps(frame.x,frame.fids,64);bi=fps(frame.x,frame.bids,64)
    if not len(fi) or not len(bi):return _plain(frame)
    ids=np.r_[fi,bi];atoms=np.concatenate([view[source] for view in views])
    descriptor=frame.x[ids]@atoms.T
    width=max(float(np.median(sqdist(descriptor,descriptor))),EPS)
    coefficients=np.linalg.solve(np.exp(-sqdist(descriptor,descriptor)/width)+np.eye(len(ids)),
        np.r_[np.ones(len(fi)),-np.ones(len(bi))])
    def predict(q,indices,role):
        result=np.empty(len(q))
        for start in range(0,len(q),64):
            profile=q[start:start+64]@atoms.T
            result[start:start+64]=np.exp(-sqdist(profile,descriptor)/width)@coefficients
        return result,{}
    return predict,{'control':'all_actual_view_training_reference_affinity_columns_RBF',
        'view_count':len(views),'profile_columns':len(atoms),'supervised_support_counts':[len(fi),len(bi)]}


def _mean_prototypes(frame,reference,projection=None):
    project=(lambda x:x) if projection is None else (lambda x:x-x@projection@projection.T)
    fg=unit(project(np.sum(reference*frame.wf[:,None],axis=0)[None]))[0]
    bg=unit(project(np.sum(reference*frame.wb[:,None],axis=0)[None]))[0]
    return lambda q,ids,role:(unit(project(q))@(fg-bg),{})


def _protect_directions(frame,differences,reference=None):
    reference=frame.x if reference is None else reference
    f=np.sum(reference*frame.wf[:,None],axis=0)/max(frame.wf.sum(),EPS)
    b=np.sum(reference*frame.wb[:,None],axis=0)/max(frame.wb.sum(),EPS)
    protected=rank_basis(np.array([f-b]));diff=np.asarray(differences,float)
    return rank_basis(diff-diff@protected@protected.T)


def fit_a028(frame,mean_only=False,profiles=False,far_only=False,brightness=False):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    image,mask,valid,known=_canvas(frame);side=image.shape[0];h,w=frame.ep.r_hw
    # Background sources must be entirely known, valid, truly outside MR.
    eligible=[]
    for v in frame.bids:
        y,x=divmod(int(v),w);region=np.s_[int(y*side/h):int((y+1)*side/h),int(x*side/w):int((x+1)*side/w)]
        if np.all(valid[region]&known[region]&~mask[region]):eligible.append(int(v))
    if len(eligible)<3:return _plain(frame)
    yy,xx=np.indices(frame.ep.r_hw);coords=np.c_[yy.ravel(),xx.ravel()]
    chosen=fps(coords,np.array(eligible),3);views=[frame.x];hits=[]
    if far_only:
        near=ndimage.binary_dilation(mask,iterations=max(1,int(side/h)*2));replace=known&valid&~mask&~near
    else:replace=known&valid&~mask
    for k,v in enumerate(chosen):
        altered=image.copy()
        if brightness:altered[replace]=np.clip(altered[replace]*(.85+.1*k),0,1)
        else:
            y,x=divmod(int(v),w);tile=image[int(y*side/h):int((y+1)*side/h),int(x*side/w):int((x+1)*side/w)]
            tiled=np.tile(tile,(int(np.ceil(side/tile.shape[0])),int(np.ceil(side/tile.shape[1])),1))[:side,:side]
            altered[replace]=tiled[replace]
        view,hit=_encode(frame,altered,'A028');views.append(view);hits.append(hit)
    mean=np.mean(views,axis=0)
    if profiles:return _profile_readout(frame,views)
    fg=frame.x[fps(frame.x,frame.fids,64)]
    # FG temporal differences are actual corresponding native patch outputs.
    selected=fps(frame.x,frame.fids,64)
    differences=np.concatenate([view[selected]-mean[selected] for view in views])
    u=np.empty((frame.x.shape[1],0)) if mean_only else rank_basis(differences)
    # Unlike A031, the supplied A028 deliberately has no class-gap protection.
    # This is a serious source-to-query transfer risk, not silently repaired.
    reference=frame.x.copy();reference[frame.fids]=mean[frame.fids]
    predict=_mean_prototypes(frame,reference,u)
    return predict,{'actual_view_count':4,'new_reference_view_requests':3,'view_cache_hits':sum(hits),
        'background_tile_native_ids':chosen.tolist(),'context_change_rank':u.shape[1],
        'unknown_held_pixels_preserved_original':True,'mean_only_control':mean_only,
        'far_background_control':far_only,'brightness_control':brightness,
        'no_query_domain_parameter_fitting':True,'class_gap_guard_not_in_A028_original':True}


def _photo_views(frame):
    image,mask,valid,known=_canvas(frame);views=[frame.x];hits=[]
    grey=np.mean(image,axis=2,keepdims=True);mean=image[valid].mean(0)
    variants=(image*.9,image*1.1,(image-mean)*.9+mean,grey+.9*(image-grey))
    for altered in variants:
        altered=np.clip(altered,0,1);altered[~valid]=image[~valid]
        view,hit=_encode(frame,altered.astype(np.float32),'A029');views.append(view);hits.append(hit)
    return views,hits


def fit_a029(frame,mean_only=False,nearest=False,profiles=False):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    views,hits=_photo_views(frame)
    if profiles:return _profile_readout(frame,views)
    fi=fps(frame.x,frame.fids,64);bi=fps(frame.x,frame.bids,64)
    if mean_only:
        reference=unit(np.mean(views,axis=0));f,b=reference[fi],reference[bi]
        return lambda q,ids,role:((nearest_mean_distance(q,b)-nearest_mean_distance(q,f))/2,{}),{
            'control':'same5realforwards_mean_reference_5NN','view_cache_hits':sum(hits)}
    orbits=[];rejection=[]
    for selected,roleids in ((fi,frame.fids),(bi,frame.bids)):
        distances=[]
        for block in range(4):
            held=roleids[frame.blocks[roleids]==block];train=roleids[frame.blocks[roleids]!=block]
            if len(held) and len(train):distances.extend(nearest_mean_distance(frame.x[held],frame.x[train],1).tolist())
        radius=float(np.quantile(distances,.95)) if distances else 2.
        orbit=np.stack([v[selected] for v in views],axis=1);accepted=np.zeros(orbit.shape[:2],bool)
        for i,native in enumerate(selected):
            train=roleids[frame.blocks[roleids]!=frame.blocks[native]]
            accepted[i]=nearest_mean_distance(orbit[i],frame.x[train],1)<=radius if len(train) else True
        keep=accepted.any(1);orbits.append((orbit[keep],accepted[keep]));rejection.append(int((~accepted).sum()))
    if any(not len(orbit) for orbit,_ in orbits):return _plain(frame)
    def likelihood(q,orbit,accepted):
        distance=sqdist(q,orbit.reshape(-1,orbit.shape[-1])).reshape(len(q),len(orbit),5)
        if nearest:return -np.min(np.where(accepted[None],distance,np.inf),axis=(1,2))
        logs=np.where(accepted[None],-distance/.07,-np.inf)
        peranchor=logsumexp(logs,axis=2)-np.log(accepted.sum(1))[None]
        return .07*(logsumexp(peranchor,axis=1)-np.log(len(orbit)))
    def predict(q,ids,role):return likelihood(q,*orbits[0])-likelihood(q,*orbits[1]),{}
    return predict,{'actual_view_count':5,'view_cache_hits':sum(hits),'orbit_anchor_counts':[len(v[0]) for v in orbits],
        'rejected_view_atoms':rejection,'probability_width2':.07,'view_probability_average_not_max':not nearest,
        'photo_recipe':['identity','brightness*.9','brightness*1.1','contrast*.9','saturation*.9']}


def _crop_views(frame,no_zoom=False):
    image,mask,valid,known=_canvas(frame);side=image.shape[0]
    ys,xs=np.nonzero(mask)
    if not len(ys):return None
    tight=(int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1)
    dy=max(1,int(np.ceil((tight[1]-tight[0])*.2)));dx=max(1,int(np.ceil((tight[3]-tight[2])*.2)))
    border=(max(0,tight[0]-dy),min(side,tight[1]+dy),max(0,tight[2]-dx),min(side,tight[3]+dx))
    y,x=np.indices(frame.ep.r_hw);cy=(y.ravel()+.5)*side/frame.ep.r_hw[0];cx=(x.ravel()+.5)*side/frame.ep.r_hw[1]
    halfy=side/frame.ep.r_hw[0]/2;halfx=side/frame.ep.r_hw[1]/2
    views=[frame.x];common=frame.ep.wvalid>=1-1e-12;hits=[]
    for bbox in (tight,border):
        y0,y1,x0,x1=bbox;crop=image[y0:y1,x0:x1]
        # Physical float interpolation avoids an unrequested uint8 round trip.
        if no_zoom:
            canvas=np.broadcast_to(np.array((124,116,104),np.float32)/255,image.shape).copy()
            canvas[y0:y1,x0:x1]=crop
        else:
            oy=(np.arange(side)+.5)*(y1-y0)/side-.5;ox=(np.arange(side)+.5)*(x1-x0)/side-.5
            xx,yy=np.meshgrid(ox,oy)
            canvas=np.stack([ndimage.map_coordinates(crop[:,:,j],[yy,xx],order=1,mode='nearest') for j in range(3)],axis=-1)
        native,hit=_encode(frame,canvas.astype(np.float32),'A030');hits.append(hit)
        gy=(cy-y0)/(y1-y0)*frame.ep.r_hw[0]-.5;gx=(cx-x0)/(x1-x0)*frame.ep.r_hw[1]-.5
        if no_zoom:gy=cy*frame.ep.r_hw[0]/side-.5;gx=cx*frame.ep.r_hw[1]/side-.5
        native=native.reshape(tuple(frame.ep.r_hw)+(frame.x.shape[1],))
        aligned=np.stack([ndimage.map_coordinates(native[:,:,j],[gy,gx],order=1,mode='nearest') for j in range(frame.x.shape[1])],axis=1)
        visible=(cy-halfy>=y0)&(cy+halfy<=y1)&(cx-halfx>=x0)&(cx+halfx<=x1)
        common&=visible;views.append(unit(aligned))
    return views,common,tight,border,hits


def fit_a030(frame,mean_only=False,tight_only=False,no_zoom=False):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    observed=_crop_views(frame,no_zoom)
    if observed is None:return _plain(frame)
    views,common,tight,border,hits=observed
    # The full original MR is necessary to guarantee a complete-target tight
    # crop. Its labels cannot be legally reconstructed in a held spatial fold.
    # This card therefore uses the source's explicitly permitted fixed cut0.
    eligible_f=frame.fids[common[frame.fids]];eligible_b=frame.bids[common[frame.bids]]
    if not len(eligible_f) or not len(eligible_b):return _plain(frame)
    fi=fps(frame.x,eligible_f,64);bi=fps(frame.x,eligible_b,64)
    if tight_only:
        f,b=views[1][fi],views[1][bi]
        return lambda q,ids,role:((nearest_mean_distance(q,b)-nearest_mean_distance(q,f))/2,{}),{
            'control':'tight_real_view_only_common_visible_5NN','tight_bbox':tight,'view_cache_hits':sum(hits)}
    protected=rank_basis(np.array([unit(v[eligible_f].mean(0))-unit(v[eligible_b].mean(0)) for v in views]))
    selected=np.r_[fi,bi];difference=np.concatenate([v[selected]-views[0][selected] for v in views[1:]])
    u=np.empty((frame.x.shape[1],0)) if mean_only else rank_basis(difference-difference@protected@protected.T)
    reference=frame.x.copy();reference[common]=np.mean(views,axis=0)[common] if mean_only else reference[common]
    fids=fps(frame.x,frame.fids,64);bids=fps(frame.x,frame.bids,64)
    def project(x):return x-x@u@u.T
    f,b=project(reference[fids]),project(reference[bids])
    return lambda q,ids,role:((nearest_mean_distance(project(q),b)-nearest_mean_distance(project(q),f))/2,{}),{
        'actual_view_count':3,'tight_bbox':tight,'border_bbox':border,'common_full_patch_count':int(common.sum()),
        'protected_class_difference_rank':protected.shape[1],'context_rank':u.shape[1],
        'view_cache_hits':sum(hits),'mean_only_control':mean_only,'C_R_disabled_mask_guided_complete_crop':True,
        'same_physical_scale_crop_control':no_zoom}


def _deformation_views(frame,misaligned=False):
    image,mask,valid,known=_canvas(frame);side=image.shape[0];yy,xx=np.indices((side,side),dtype=float)
    h,w=frame.ep.r_hw;py,px=np.indices((h,w),dtype=float);cy=(py.ravel()+.5)*side/h-.5;cx=(px.ravel()+.5)*side/w-.5
    amplitude=.01*side;views=[frame.x];common=frame.ep.wvalid>=1-1e-12;hits=[]
    for transform in ('horizontal_flip','x_sinusoid','y_sinusoid'):
        if transform=='horizontal_flip':sy,sx=yy,side-1-xx;ty,tx=cy,side-1-cx
        elif transform=='x_sinusoid':sy,sx=yy,xx-amplitude*np.sin(2*np.pi*yy/side);ty,tx=cy,cx+amplitude*np.sin(2*np.pi*cy/side)
        else:sy,sx=yy-amplitude*np.sin(2*np.pi*xx/side),xx;ty,tx=cy+amplitude*np.sin(2*np.pi*cx/side),cx
        canvas=np.stack([ndimage.map_coordinates(image[:,:,j],[sy,sx],order=1,mode='constant',cval=(124,116,104)[j]/255) for j in range(3)],axis=-1)
        # The actual mask is transformed with nearest interpolation; it does not
        # choose RGB edits. Source labels remain attached to original positions.
        transformed_mask=ndimage.map_coordinates(mask.astype(float),[sy,sx],order=0,mode='constant',cval=0)>0
        native,hit=_encode(frame,canvas.astype(np.float32),'A031');hits.append(hit)
        gy=(ty+.5)*h/side-.5;gx=(tx+.5)*w/side-.5
        if misaligned:gx=np.roll(gx,1)
        native=native.reshape((h,w,frame.x.shape[1]))
        aligned=np.stack([ndimage.map_coordinates(native[:,:,j],[gy,gx],order=1,mode='nearest') for j in range(frame.x.shape[1])],axis=1)
        warped_valid=ndimage.map_coordinates(valid.astype(float),[sy,sx],order=0,mode='constant',cval=0)>0
        target_full=np.array([warped_valid[int(y*side/h):int((y+1)*side/h),int(x*side/w):int((x+1)*side/w)].all()
                              for y in range(h) for x in range(w)]).reshape(h,w)
        full=(gy>=0)&(gy<=h-1)&(gx>=0)&(gx<=w-1)
        fy=np.floor(np.clip(gy,0,h-1)).astype(int);fx=np.floor(np.clip(gx,0,w-1)).astype(int)
        cyi=np.ceil(np.clip(gy,0,h-1)).astype(int);cxi=np.ceil(np.clip(gx,0,w-1)).astype(int)
        full&=target_full[fy,fx]&target_full[fy,cxi]&target_full[cyi,fx]&target_full[cyi,cxi]
        common&=full;views.append(unit(aligned))
    return views,common,hits


def fit_a031(frame,mean_only=False,all_atoms=False,profiles=False,misaligned=False):
    if not len(frame.fids) or not len(frame.bids):return _plain(frame)
    views,common,hits=_deformation_views(frame,misaligned)
    if profiles:return _profile_readout(frame,views)
    fi=fps(frame.x,frame.fids[common[frame.fids]],64);bi=fps(frame.x,frame.bids[common[frame.bids]],64)
    if not len(fi) or not len(bi):return _plain(frame)
    if all_atoms:
        f=np.concatenate([v[fi] for v in views]);b=np.concatenate([v[bi] for v in views])
        return lambda q,ids,role:((nearest_mean_distance(q,b,1)-nearest_mean_distance(q,f,1))/2,{}),{
            'control':'all_actual4view_atoms_nearest','origin_anchor_counts':[len(fi),len(bi)],'view_cache_hits':sum(hits)}
    mean=unit(np.mean(views,axis=0));selected=np.r_[fi,bi]
    difference=np.concatenate([v[selected]-views[0][selected] for v in views[1:]])
    u=np.empty((frame.x.shape[1],0)) if mean_only else _protect_directions(frame,difference)
    def project(x):return x-x@u@u.T
    reference=mean if mean_only else frame.x;f,b=project(reference[fi]),project(reference[bi])
    return lambda q,ids,role:((nearest_mean_distance(project(q),b)-nearest_mean_distance(project(q),f))/2,{}),{
        'actual_view_count':4,'deformation_amplitude_fraction':.01,'deformations':['identity','horizontal_flip','x_sinusoid','y_sinusoid'],
        'inverse_map_common_full_patch_count':int(common.sum()),'protected_mean_class_gap_exact':True,
        'difference_rank':u.shape[1],'mean_only_control':mean_only,'misaligned_inverse_control':misaligned,
        'view_cache_hits':sum(hits),'mask_warp_is_nearest_and_not_RGB_intervention_guide':True}


def _receipt(fn):
    def read(ep):
        p=getattr(ep,'provider',None);nested=getattr(p,'receipt',{}).get('internal_encoder',{})
        return dict(nested)
    def execute(ep):
        before=read(ep);result=fn(ep);after=read(ep)
        delta={k:after.get(k,0)-before.get(k,0) for k in ('actual_extra_forwards','actual_tail_blocks','forward_wall_seconds','forward_cpu_seconds')}
        result.info['actual_view_observation_cost_delta']=delta
        result.info['new_encoder_forwards']=int(delta['actual_extra_forwards'])
        if not after:result.info['observation_cost_scope']='supplied callback upstream cost unverified; synthetic callback is not DINO'
        return result
    return execute


def install(register,requirements,methods,controls,recipes):
    register('A028',fit_a028,(
        'Three spatialFPS distinct fullyknown pureBGpatch RGB tiles repeated outsideknownMR only; unknownheld/padding pixels untouched. Each outerC_Rfold reconstructs labelguided views, up-to15actual extraR forwards plusfullfit.',
        'Original+threeactual unitviews givecorrespondingFGmean; full temporalchange span from<=64FG origins, without an unrequestedsemanticguard. FGmean/BGnativeprototypes in orthogonalcomplement, queryprojectedandunitnormalized.',),
        (('same_four_view_mean_no_projection',lambda f:fit_a028(f,mean_only=True)),('same_four_view_complete_profiles',lambda f:fit_a028(f,profiles=True)),
         ('only_far_background_intervention',lambda f:fit_a028(f,far_only=True)),('same_budget_background_brightness',lambda f:fit_a028(f,brightness=True))))
    register('A029',fit_a029,(
        'Original+.9/.1.1brightness, .9contrastaroundvalidimageRGBmean, .9saturationtowardchannelmean. RGB-transform independentlabels so4extraRforwards are legally reused acrossC_Rfolds.',
        'SourceoriginFPS64/role, fiveactualorbitpoints; reject eachviewatom outside originalsame-role crossblockNNdistance95%; equalprobabilitymeanwithin eachsurvivingorbit thenequalmeanoverorigins, width squared.07. No bestviewselection.',),
        (('same_five_view_mean_5NN',lambda f:fit_a029(f,mean_only=True)),('same_orbits_nearest_not_marginal',lambda f:fit_a029(f,nearest=True)),
         ('all_actual_orbit_profiles_RBF',lambda f:fit_a029(f,profiles=True))))
    assumptions=(
        'ActualfullRMR tightbbox and20%bordermargin, two resizedfloatRGB nativeforwards; nativecenter inversemap, onlyfullpatchfootprints inallthreeviews.',
        'Complete-target crop cannot be rebuilt withoutheldMRlabels, so originalcommon exception applies: noC_Rthreshold selection, fixedsigneddistance cut0 andexplicitdegeneration. NofullMRcrop usedinsideOOF.',
        'Protectspanofthreeconsistentunitrolemean differences, deleteorthogonalcontextchange span; originalR/Q projected Euclidean5NN. Needbothpure roles in commonvisiblearea, otherwisefixedB0.',)
    recipes['A030']={'implementation_assumption':list(assumptions),'threshold':0,'calibration':'C_Rdisabled: noleakagecomplete-targetfoldcropunavailable','renderer':'A_U_fixed0'}
    methods['A030']=partial(infer_a,method_id='A030',fit=fit_a030,assumptions=assumptions,fixed_threshold=0.)
    for name,fit in (('same_three_view_mean',partial(fit_a030,mean_only=True)),('tight_real_view_only',partial(fit_a030,tight_only=True)),
                     ('same_physical_scale_three_view_mean',partial(fit_a030,mean_only=True,no_zoom=True))):
        cid='A030__'+name;controls[cid]=partial(infer_a,method_id=cid,fit=fit,assumptions=assumptions+('controlonly',),fixed_threshold=0.)
    register('A031',fit_a031,(
        'Original,horizontalflip,andtwoinvertiblesinusoidbends x/y amplitude1%side; realfloatRGBencodes andnearestMRwarps, inversefeaturecenterbilinear. Fullpatchboundaryholes excluded, no feature-vector rotation.',
        'Difference span from<=64sourceoriginsperrole; removeitscomponentparallelrawweightedFGmean−BGmean so exactweightedmeanclassdifferenceisretained. ThreeextraRviews RGBindependentlabels reusedlegallyacrossC_Rfolds, directions/prototypesstillrebuilt.',),
        (('same_four_view_mean',lambda f:fit_a031(f,mean_only=True)),('all_actual_augmented_atoms_nearest',lambda f:fit_a031(f,all_atoms=True)),
         ('all_actual_view_affinity_RBF',lambda f:fit_a031(f,profiles=True)),('same_real_views_wrong_inverse_map',lambda f:fit_a031(f,misaligned=True))))
    for mid in ('A028','A029','A030','A031'):
        requirements[mid]=['frozen_encode_rgb','internal_encoder_binding','original_RGB_and_complete_MR','physical_RGB_transform']
        methods[mid]=_receipt(methods[mid])
        for cid in list(controls):
            if cid.startswith(mid+'__'):controls[cid]=_receipt(controls[cid])
