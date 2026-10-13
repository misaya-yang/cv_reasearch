"""Task-relative constellation registration, composed with unchanged Strong S.

Uses legal R Mask and prediction-derived S only, no query labels or encoder.
Per-episode correspondence/pose optimization does not update feature weights.
"""
import hashlib
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

ARMS=('task_constellation_complete','semantic_only_pose_complete',
      'shuffled_binding_pose_complete','no_reference_BG_pose_complete',
      'unconstrained_accepted_support_complete',
      'same_edit_budget_foris_margin_from_S')
FIELDS=('joint','semantic_only','shuffled','no_BG')
R_PARTICLES=512
Q_PARTICLES=128
BG_PARTICLES=128
JOINT_STEPS=5
AXIS=(np.arange(64,dtype=np.float64)+.5)/64
Y,X=np.meshgrid(AXIS,AXIS,indexing='ij')
COORD=np.stack((X.ravel(),Y.ravel()),axis=1)


def hash_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def particles(weights,n):
    w=np.asarray(weights,np.float64).reshape(4096)
    mass=float(w.sum())
    if not math.isfinite(mass) or mass<=0: raise ValueError('missing_particle_mass')
    return np.searchsorted(np.cumsum(w),(np.arange(n)+.5)*mass/n,side='right').astype(np.int64)


def normal_frame(weights,hw):
    w=np.asarray(weights,np.float64).reshape(4096)
    total=float(w.sum())
    if total<=0: raise ValueError('empty_normal_frame')
    if np.count_nonzero(w)>0 and np.count_nonzero(w)==1:raise ValueError('single_native_point_normal_frame')
    coordinates=COORD*np.asarray([hw[1],hw[0]],np.float64)
    centre=(w[:,None]*coordinates).sum(0)/total
    radius=math.sqrt(float((w*((coordinates-centre)**2).sum(1)).sum()/total))
    if not math.isfinite(radius) or radius<=0: raise ValueError('zero_radius_normal_frame')
    return centre,radius


def unit(raw,device):
    a=np.asarray(raw)
    if a.shape!=(4096,1024) or not np.isfinite(a).all():raise ValueError('invalid_frozen_O24')
    t=torch.as_tensor(a,dtype=torch.float32,device=device)
    norms=torch.linalg.vector_norm(t,dim=1)
    if not bool((torch.isfinite(norms).all() & (norms>0).all()).item()):raise ValueError('invalid_token_norm')
    return t/norms[:,None]


def pose_fit(x,y):
    """Closed-form orientation-preserving similarity from matched coordinates."""
    if np.all(x==x[0]):raise ValueError('matched_reference_one_exact_coordinate')
    if np.all(y==y[0]):raise ValueError('matched_query_one_exact_coordinate')
    mx,my=x.mean(0),y.mean(0)
    xx,yy=x-mx,y-my
    dot=float((xx*yy).sum())
    cross=float((xx[:,0]*yy[:,1]-xx[:,1]*yy[:,0]).sum())
    if dot==0 and cross==0:raise ValueError('zero_similarity_direction_covariance')
    denom=float((xx*xx).sum())
    if denom<=0:raise ValueError('matched_reference_zero_spread')
    u,s,vt=np.linalg.svd(xx.T@yy)
    sign=1. if np.linalg.det(u@vt)>=0 else -1.
    rotation=u@np.diag([1.,sign])@vt
    scale=float((s*np.asarray([1.,sign])).sum()/denom)
    if not math.isfinite(scale) or scale<=0:raise ValueError('nonpositive_pose_scale')
    translation=my-scale*mx@rotation
    assert np.linalg.det(rotation)>0 and np.allclose(rotation.T@rotation,np.eye(2),atol=1e-12)
    return scale,rotation,translation


def distance(x,y,pose):
    scale,rotation,translation=pose
    mapped=scale*x@rotation+translation
    return ((y[:,None,:]-mapped[None,:,:])**2).sum(2)


def _fit(cost,x,y,iterations,counts):
    rows,cols=linear_sum_assignment(cost)
    counts['assignment_steps']+=1
    assert np.array_equal(rows,np.arange(Q_PARTICLES)) and len(set(cols))==Q_PARTICLES
    counts['pose_fit_attempts']+=1
    pose=pose_fit(x[cols],y);counts['pose_fits']+=1
    energy=float((cost[rows,cols]+.5*distance(x,y,pose)[rows,cols]).sum())
    history=[energy]
    assignment_energies=[]
    assignments=[hash_array(cols)]
    for step in range(iterations):
        joint=cost+.5*distance(x,y,pose)
        rows,cols=linear_sum_assignment(joint)
        counts['assignment_steps']+=1
        after_assignment=float(joint[rows,cols].sum())
        assert after_assignment<=history[-1]+1e-9*max(1.,abs(history[-1]))
        counts['pose_fit_attempts']+=1
        pose=pose_fit(x[cols],y);counts['pose_fits']+=1
        energy=float((cost[rows,cols]+.5*distance(x,y,pose)[rows,cols]).sum())
        # Alternating exact assignment and least-squares pose cannot increase
        # the same energy. Tolerance is solely FP64 arithmetic.
        assert energy<=history[-1]+1e-9*max(1.,abs(history[-1])),(history,energy)
        assert energy<=after_assignment+1e-9*max(1.,abs(after_assignment))
        assignment_energies.append(after_assignment)
        history.append(energy);assignments.append(hash_array(cols))
    residual=float(distance(x,y,pose)[rows,cols].mean())
    return pose,dict(energy_history=history,energy_after_assignment=assignment_energies,assignments_sha256=assignments,
                     **counts,
                     final_source_particle_indices=cols.tolist(),geometric_MSE=residual,
                     final_feature_cost_mean=float(cost[rows,cols].mean()))


def fit(cost,x,y,iterations):
    counts=dict(assignment_steps=0,pose_fit_attempts=0,pose_fits=0)
    try:
        return _fit(cost,x,y,iterations,counts)
    except (ValueError,np.linalg.LinAlgError) as exc:
        return None,dict(**counts,available=False,reason=str(exc))


def physical_pose(pose,rframe,qframe):
    scale,rotation,translation=pose
    cr,rr=rframe;cq,rq=qframe
    a=scale*rq/rr
    offset=cq+rq*translation-a*cr@rotation
    return a,rotation,offset


def compute(raw_r,raw_q,p,s,reference_hw,query_hw,*,device='mps',validate=False):
    p,s=np.asarray(p,np.float64).reshape(4096),np.asarray(s,np.float64).reshape(4096)
    assert np.isfinite(p).all() and np.isfinite(s).all() and np.all((p>=0)&(p<=1)) and np.all((s>=0)&(s<=1))
    result=dict(poses=dict.fromkeys(FIELDS),stats=dict(device=device,R_FG_mass=float(p.sum()),R_BG_mass=float((1-p).sum()),Q_S_mass=float(s.sum()),arms={}))
    try:
        ri,qi,bi=particles(p,R_PARTICLES),particles(s,Q_PARTICLES),particles(1-p,BG_PARTICLES)
        rframe,qframe=normal_frame(p,reference_hw),normal_frame(s,query_hw)
    except ValueError as exc:
        result['stats']['reason']=str(exc)
        for name in FIELDS: result['stats']['arms'][name]=dict(available=False,reason=str(exc),assignment_steps=0,pose_fit_attempts=0,pose_fits=0)
        return result
    result['stats'].update(R_FG_particle_sha256=hash_array(ri),Q_S_particle_sha256=hash_array(qi),R_BG_particle_sha256=hash_array(bi),
                           R_FG_unique_ids=int(np.unique(ri).size),Q_S_unique_ids=int(np.unique(qi).size),R_BG_unique_ids=int(np.unique(bi).size),reason=None)
    r,q=unit(raw_r,device),unit(raw_q,device)
    ir,iq,ib=(torch.as_tensor(a,device=device) for a in (ri,qi,bi))
    rf,qf,bg=r[ir],q[iq],r[ib]
    cosine=(qf@rf.T).detach().to('cpu').double().numpy()
    rbg=(rf@bg.T).max(1).values.detach().to('cpu').double().numpy()
    qbg=(qf@bg.T).max(1).values.detach().to('cpu').double().numpy()
    # Both sides must explain their match relative to legal background. max
    # couples the row and column terms, rather than adding row-only offsets
    # that cannot affect the rectangular assignment.
    cost=1.-cosine+np.maximum(qbg[:,None],rbg[None,:])
    assert np.isfinite(cost).all()
    result['stats']['semantic_cost_sha256']=hash_array(cost)
    penalty=np.maximum(qbg[:,None],rbg[None,:])
    interaction=penalty-penalty[:,[0]]-penalty[[0],:]+penalty[0,0]
    result['stats']['BG_penalty']=dict(max_abs_nonadditive_interaction=float(np.abs(interaction).max()),mean_abs_nonadditive_interaction=float(np.abs(interaction).mean()),
                                      entirely_query_row_constant=bool(qbg.min()>=rbg.max()),entirely_reference_column_constant=bool(rbg.min()>=qbg.max()),
                                      not_a_hard_semantic_veto=True)
    if validate:
        nr=np.array(raw_r,dtype=np.float64,copy=True);nr/=np.linalg.norm(nr,axis=1,keepdims=True)
        nq=np.array(raw_q,dtype=np.float64,copy=True);nq/=np.linalg.norm(nq,axis=1,keepdims=True)
        ref=1.-nq[qi]@nr[ri].T+np.maximum((nq[qi]@nr[bi].T).max(1)[:,None],(nr[ri]@nr[bi].T).max(1)[None,:])
        result['stats']['CPU64_cost_reference']=dict(max_absolute_error=float(np.max(np.abs(cost-ref))),RMS_error=float(np.sqrt(np.mean((cost-ref)**2))),selection_changes_production=False)
    del r,q,rf,qf,bg
    x=(COORD[ri]*np.asarray([reference_hw[1],reference_hw[0]])-rframe[0])/rframe[1]
    y=(COORD[qi]*np.asarray([query_hw[1],query_hw[0]])-qframe[0])/qframe[1]
    reverse=np.arange(R_PARTICLES-1,-1,-1)
    result['stats']['shuffle_exact_feature_marginals']=bool(np.array_equal(np.sort(cost,axis=1),np.sort(cost[:,reverse],axis=1)))
    for name,cost_used,steps in (('joint',cost,JOINT_STEPS),('semantic_only',cost,0),('shuffled',cost[:,reverse],JOINT_STEPS),('no_BG',1.-cosine,JOINT_STEPS)):
        pose,stats=fit(cost_used,x,y,steps)
        if pose is not None:
            a,rot,t=physical_pose(pose,rframe,qframe)
            result['poses'][name]=(a,rot,t,reference_hw,query_hw)
            stats.update(available=True,reason=None,physical_scale=a,physical_rotation=rot.tolist(),physical_translation=t.tolist())
        result['stats']['arms'][name]=stats
    return result


def warp(mask,pose,shape):
    """Inverse similarity sample; coordinates refer to canonical square frames."""
    if pose is None:return np.zeros(shape,bool)
    scale,rotation,offset,reference_hw,query_hw=pose
    yy=(np.arange(shape[0],dtype=np.float64)+.5)/shape[0]
    xx=(np.arange(shape[1],dtype=np.float64)+.5)/shape[1]
    y,x=np.meshgrid(yy,xx,indexing='ij')
    target=np.stack((x,y),axis=2)*np.asarray([query_hw[1],query_hw[0]])
    source=((target-offset)@rotation.T)/scale
    source/=np.asarray([reference_hw[1],reference_hw[0]])
    grid=torch.from_numpy(2*source-1)[None]
    image=torch.as_tensor(np.ascontiguousarray(mask),dtype=torch.float64)[None,None]
    output=F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,0].numpy()
    return output>.5


def resize(a,shape,mode='bilinear'):
    x=torch.as_tensor(np.ascontiguousarray(a),dtype=torch.float64)[None,None]
    kw={} if mode=='nearest' else dict(align_corners=False)
    return F.interpolate(x,size=shape,mode=mode,**kw)[0,0].numpy()


def compose_frame(masks,source_score,dense,reference_mask,accepted_any):
    s,b=masks['S'],masks['B']
    size=s.shape
    # Preserve the historical accepted-support domain exactly. The new
    # continuous reference template warp above has its own CPU64 definition.
    accepted=F.interpolate(torch.as_tensor(accepted_any).float()[None,None],size=size,mode='bilinear',align_corners=False)[0,0].numpy()>.5
    domain=(~s)&b&accepted
    outputs={}
    details={}
    for arm,name in zip(ARMS[:4],FIELDS):
        template=warp(reference_mask,dense['poses'][name],size)
        outputs[arm]=s|(domain&template)
        details[arm]=dict(add=int((outputs[arm]&~s).sum()),delete=0,template_area=int(template.sum()),pose_available=dense['poses'][name] is not None)
    outputs[ARMS[4]]=s|domain
    details[ARMS[4]]=dict(add=int(domain.sum()),delete=0)
    k=int((outputs[ARMS[0]]&~s).sum())
    score=np.asarray(source_score,np.float64).reshape(64,64)
    # Ranking minmax/linear transforms preserve order; use the actual native
    # source FoRIS score interpolated CPU64, stable row-major ties.
    rank=resize(score,size).ravel()
    ids=np.flatnonzero(domain.ravel())
    order=np.lexsort((ids,-rank[ids]))
    budget=s.copy().ravel();budget[ids[order[:k]]]=True
    outputs[ARMS[5]]=budget.reshape(size)
    details[ARMS[5]]=dict(add=k,delete=0,pool=int(domain.sum()),pool_is_whole_selected_domain=bool(k==len(ids)))
    assert tuple(outputs)==ARMS
    for arm,out in outputs.items():
        assert not np.any(s&~out) and not np.any((out&~s)&~domain)
        assert int((out&~s).sum())==details[arm]['add']
    assert int((outputs[ARMS[5]]&~s).sum())==k
    return outputs,dict(D_add=int(domain.sum()),S=int(s.sum()),arms=details)
