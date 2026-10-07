"""Legal source RGB/label patches for supplied E231 and E237; no whole shape."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.special import logsumexp
from . import e_helpers_226_250 as H
from . import e226_250 as B
from . import e235_244 as C


def patch_arrays(rgb,mask,centers,angle=0.,radius=2):
    shape=rgb.shape[:2];cy,cx=np.divmod(np.asarray(centers,int),shape[1])
    y,x=np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1),indexing='ij')
    dy=np.cos(angle)*y-np.sin(angle)*x;dx=np.sin(angle)*y+np.cos(angle)*x
    yy=cy[:,None]+dy.ravel();xx=cx[:,None]+dx.ravel()
    colours=np.stack([ndimage.map_coordinates(np.asarray(rgb,float)[...,k]/255.,[yy,xx],order=1,mode='nearest')
                      for k in range(3)],axis=-1)
    labels=None if mask is None else ndimage.map_coordinates(np.asarray(mask,float),[yy,xx],order=1,mode='nearest')
    return colours.reshape(len(centers),-1),labels


def _dictionary(problem,rotations=1):
    ep=problem.ep;shape=ep.r_rgb.shape[:2];labels=problem.labels.reshape(shape)
    support=ndimage.binary_erosion(problem.rvalid.reshape(shape),structure=np.ones((7,7)),border_value=0)
    coverage=ndimage.uniform_filter(labels.astype(float),5,mode='nearest')
    regions=[support&(coverage>=.9),support&(coverage<=.1),support&(coverage>.1)&(coverage<.9)]
    selected=[B._spread_rows(np.flatnonzero(region),128//rotations) for region in regions]
    centers=np.concatenate(selected) if any(len(v) for v in selected) else np.empty(0,int)
    colours=[];mask=[];tokens=[];roles=[]
    for rotation in range(rotations):
        rgb,mr=patch_arrays(ep.r_rgb,labels,centers,angle=2*np.pi*rotation/rotations)
        colours.append(rgb);mask.append(mr);tokens.append(problem.rtokens[centers]);roles.append(labels.ravel()[centers])
    if not len(centers):return None
    return dict(rgb=np.concatenate(colours),mask=np.concatenate(mask),tokens=np.concatenate(tokens),
                center_role=np.concatenate(roles).astype(float),raw_centers=centers,
                per_category=[len(v)*rotations for v in selected],rotations=rotations)


def _retrieval(problem,dictionary,stride=2,control=None):
    ep=problem.ep;shape=ep.original_shape
    yy,xx=np.meshgrid(np.arange(0,shape[0],stride),np.arange(0,shape[1],stride),indexing='ij')
    centers=(yy*shape[1]+xx).ravel();coords=np.stack((yy.ravel(),xx.ravel()),1)
    dino=ep.q@ep.r[dictionary['tokens']].T
    native_ids=problem.qtokens[centers];U=problem.U.ravel()[centers]
    rows=np.empty((len(centers),min(8,len(dictionary['rgb']))),int);costs=np.empty(rows.shape,float)
    supported=np.ones(len(centers),bool)
    # A fixed, source-only pixel variance floor gives finite RGB evidence units.
    raw=np.asarray(ep.r_rgb,float).reshape(-1,3)/255.;scale=max(float(np.median(np.var(raw[problem.rvalid],axis=0))),1e-4)
    source=dictionary['rgb'];norm=np.sum(source*source,1)
    # Source leave-patch support radius: other patches whose physical center
    # token differs, so a patch cannot certify its own RGB reconstruction.
    thresholds=[]
    for start in range(0,len(source),128):
        x=source[start:start+128]
        dist=np.maximum(np.sum(x*x,1)[:,None]+norm[None]-2*x@source.T,0)/source.shape[1]
        same=dictionary['tokens'][start:start+128,None]==dictionary['tokens'][None,:]
        dist[same]=np.inf;nearest=dist.min(1);thresholds.extend(nearest[np.isfinite(nearest)].tolist())
    support_radius=max(float(np.quantile(thresholds,.9)) if thresholds else 0.,1e-4)
    role=dictionary['center_role'];u=np.clip(U,H.EPS,1-H.EPS)
    for start in range(0,len(centers),512):
        pixels=centers[start:start+512];q,_=patch_arrays(ep.q_rgb,None,pixels)
        rgb=np.maximum(np.sum(q*q,1)[:,None]+norm[None]-2*q@source.T,0)/q.shape[1]
        semantic=-np.log(np.clip(dino[native_ids[start:start+len(q)]],H.EPS,1))
        rolecost=-(role[None]*np.log(u[start:start+len(q),None])+(1-role)[None]*np.log(1-u[start:start+len(q),None]))
        cost=rgb/scale+rolecost
        if control!='RGB_only':cost+=semantic
        k=rows.shape[1];order=np.argsort(cost,axis=1,kind='stable')[:,:k]
        rows[start:start+len(q)]=order;costs[start:start+len(q)]=np.take_along_axis(cost,order,1)
        supported[start:start+len(q)]=np.min(rgb,axis=1)<=support_radius
    return centers,coords,rows,costs,supported,dict(stride=stride,source_colour_scale=scale,source_support_radius=support_radius)


def _overlap_pairs(coords,shape):
    lookup={tuple(p):j for j,p in enumerate(coords)};pairs=[]
    for j,(y,x) in enumerate(coords):
        for dy,dx in ((0,2),(2,0)):
            k=lookup.get((y+dy,x+dx))
            if k is None:continue
            # Both 5x5 patches use matching original pixel positions. Replicate
            # border values are still observations of that same source pixel.
            a=[];b=[]
            for iy in range(-2,3):
                for ix in range(-2,3):
                    if -2<=iy-dy<=2 and -2<=ix-dx<=2:
                        a.append((iy+2)*5+ix+2);b.append((iy-dy+2)*5+ix-dx+2)
            pairs.append((j,k,np.array(a),np.array(b)))
    return pairs


def _vote(problem,centers,labels,weights=None):
    shape=problem.U.shape;value=np.zeros(problem.U.size);mass=np.zeros(problem.U.size)
    cy,cx=np.divmod(centers,shape[1]);dy,dx=np.meshgrid(np.arange(-2,3),np.arange(-2,3),indexing='ij')
    ids=(np.clip(cy[:,None]+dy.ravel(),0,shape[0]-1)*shape[1]+
         np.clip(cx[:,None]+dx.ravel(),0,shape[1]-1))
    if weights is None:weights=np.ones(len(centers))
    np.add.at(value,ids.ravel(),(labels*weights[:,None]).ravel())
    np.add.at(mass,ids.ravel(),np.broadcast_to(weights[:,None],ids.shape).ravel())
    coverage=mass>0;prediction=problem.U.ravel().copy()
    prediction[coverage]=.5*(value[coverage]/mass[coverage]+prediction[coverage])
    return prediction.reshape(shape),coverage


def e231_compatible_jigsaw(problem,control=None):
    dictionary=_dictionary(problem)
    if dictionary is None:return problem.U.copy(),dict(status='fallback_no_legal_source_patch')
    centers,coords,ids,costs,supported,info=_retrieval(problem,dictionary)
    choice=np.zeros(len(centers),int);pairs=_overlap_pairs(coords,problem.U.shape);adj=[[] for _ in centers]
    for left,right,a,b in pairs:
        if not supported[left] or not supported[right]:continue
        adj[left].append((right,a,b));adj[right].append((left,b,a))
    if control=='kernel':
        weight=np.exp(-costs-logsumexp(-costs,axis=1,keepdims=True))
        labels=np.einsum('wk,wkp->wp',weight,dictionary['mask'][ids])
    else:
        if control!='nearest':
            for _ in range(5):
                for j in range(len(centers)):
                    if not supported[j]:continue
                    score=costs[j].copy();candidates=dictionary['mask'][ids[j]]
                    for k,a,b in adj[j]:
                        score+=np.mean(np.abs(candidates[:,a]-dictionary['mask'][ids[k,choice[k]],b]),axis=1)
                    choice[j]=int(np.argmin(score))
        labels=dictionary['mask'][ids[np.arange(len(ids)),choice]]
    prediction,covered=_vote(problem,centers[supported],labels[supported])
    return prediction,dict(status='ok',source_patch_categories=dictionary['per_category'],query_windows=len(centers),
                    supported_windows=int(supported.sum()),overlap_edges=len(pairs),coordinate_rounds=5 if control is None else 0,
                    renderer='equal_patch_votes_and_U_probability_then_original_strict_.5',control=control,**info)


def e237_rotated_patch_kernel(problem,control=None):
    rotations=1 if control=='no_rotation' else 8;dictionary=_dictionary(problem,rotations)
    if dictionary is None:return problem.U.copy(),dict(status='fallback_no_legal_rotated_source_patch')
    centers,coords,ids,costs,supported,info=_retrieval(problem,dictionary,control='RGB_only' if control=='RGB_only' else None)
    if control=='nearest_RGB':
        labels=dictionary['mask'][ids[:,0]]
    else:
        weights=np.exp(-costs-logsumexp(-costs,axis=1,keepdims=True))
        labels=np.einsum('wk,wkp->wp',weights,dictionary['mask'][ids])
    field,covered=_vote(problem,centers[supported],labels[supported])
    # Source-labelled fragments provide an ordinary kernel label regression,
    # not another role module. Its probability joins U before a full RGB cut.
    prob=np.clip(field,H.EPS,1-H.EPS);base=np.clip(problem.U,H.EPS,1-H.EPS)
    evidence=np.log(prob/(1-prob))-np.log(base/(1-base))
    out,cert=C._cut(problem,evidence);out.ravel()[~covered]=problem.U.ravel()[~covered]
    return out,dict(status='ok',binary_optimizer=True,rotations=rotations,source_patch_categories=dictionary['per_category'],
                    query_windows=len(centers),supported_windows=int(supported.sum()),source_support_eroded3pixels=True,
                    kernel_label_regression_not_new_role_module=True,cut=cert,control=control,**info)


METHODS={'E231':lambda ep:B._call(ep,'E231',e231_compatible_jigsaw),
         'E237':lambda ep:B._call(ep,'E237',e237_rotated_patch_kernel)}
CONTROLS={
 'E231_same_patches_kernel':lambda ep:B._call(ep,'E231',e231_compatible_jigsaw,control='kernel'),
 'E231_nearest_without_overlap':lambda ep:B._call(ep,'E231',e231_compatible_jigsaw,control='nearest'),
 'E237_no_rotation_same384budget':lambda ep:B._call(ep,'E237',e237_rotated_patch_kernel,control='no_rotation'),
 'E237_RGB_only_same_rotated_patches':lambda ep:B._call(ep,'E237',e237_rotated_patch_kernel,control='RGB_only'),
 'E237_nearest_same_patches':lambda ep:B._call(ep,'E237',e237_rotated_patch_kernel,control='nearest_RGB'),
}
RESOURCES={id:dict(final_native=True,original_rgb=True,complete_MR=True,mean_host=id=='E231',extra_encoder_forwards=0) for id in METHODS}
