"""A021 genuine frozen RGB VJP followed by FG-descriptor JVP (CPU only)."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from functools import partial
import time
import numpy as np

from .common import ArtifactUnavailable,array_hash,require_artifact
from .a_helpers_001_050 import EPS,Frame,b0,fps,nearest_mean_distance,rank_basis,unit
from .a_views_028_031 import _canvas,_encode,_receipt
from .a_observations_021_048 import _binding


_PRODUCT_CACHE=OrderedDict()


def _mask_and_weights(frame):
    image,mask,valid,known=_canvas(frame)
    background=valid&known&~mask
    weights=frame.wf/max(float(frame.wf.sum()),EPS)
    return image,background,weights


def _actual_product(frame,image,background,weights):
    engine=require_artifact(frame.ep,'internal_encoder');_binding(frame.ep)
    if not all(hasattr(engine,k) for k in ('model','torch','_input','_normalized','forward','receipt')):
        raise ArtifactUnavailable('Actual frozen InternalEncoder needed for genuine VJP/JVP')
    torch=engine.torch;engine.forward.check_frozen()
    if not hasattr(torch,'func') or not hasattr(torch.func,'jvp'):
        raise ArtifactUnavailable('Actual Torch forward-AD JVP is unavailable; finite differences cannot substitute A021')
    x=engine._input(image).detach();ww=torch.from_numpy(np.array(weights,copy=True)).to(dtype=x.dtype)
    bg=torch.from_numpy(np.array(background,copy=True)).to(dtype=x.dtype)[None]
    dimension=frame.x.shape[1]
    vectors=np.random.default_rng(0).choice((-1.,1.),(8,dimension))/np.sqrt(dimension)
    expected=np.sum(frame.x*weights[:,None],axis=0)
    def descriptor(value):
        raw=engine.model.forward_features(engine._normalized(value))[0,engine.model.num_prefix_tokens:]
        features=torch.nn.functional.normalize(raw,dim=-1)
        return torch.sum(features*ww[:,None],dim=0)
    def math_context():
        if hasattr(torch.nn,'attention') and hasattr(torch.nn.attention,'sdpa_kernel'):
            return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
        if hasattr(torch.backends,'cuda') and hasattr(torch.backends.cuda,'sdp_kernel'):
            return torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False)
        raise ArtifactUnavailable('An explicit derivative-compatible actual math SDPA backend is required')
    outputs=[];defects=[];wall,cpu=time.monotonic(),time.process_time()
    try:
        with math_context(),torch.autocast('cpu',enabled=False):
            for vector in vectors:
                value=x.detach().requires_grad_(True)
                with torch.enable_grad():
                    fg=descriptor(value);engine.receipt['actual_extra_forwards']+=1
                    defect=float(np.max(np.abs(fg.detach().cpu().numpy()-expected)));defects.append(defect)
                    if defect>1e-5:
                        raise ArtifactUnavailable('A021 math-SDPA foreground descriptor differs from sealed native unit source, max='+str(defect))
                    v=torch.from_numpy(vector.astype(np.float32));gradient=torch.autograd.grad(torch.sum(fg*v),value,only_inputs=True)[0]
                    engine.receipt['actual_vjp_passes']=engine.receipt.get('actual_vjp_passes',0)+1
                direction=gradient.detach()*bg
                del fg,gradient,value
                # Forward AD is genuine JVP, with no second-order reverse tape.
                # Math SDPA is required because native CPU fused attention may
                # not implement forward AD. This backend choice is disclosed.
                with torch.no_grad():
                    _,response=torch.func.jvp(descriptor,(x,),(direction,))
                engine.receipt['actual_extra_forwards']+=1
                engine.receipt['actual_jvp_passes']=engine.receipt.get('actual_jvp_passes',0)+1
                outputs.append(response.detach().cpu().numpy().copy())
    except (RuntimeError,NotImplementedError) as error:
        raise ArtifactUnavailable('A021 actual frozen CPU VJP/JVP execution failed: '+repr(error)) from error
    finally:
        engine.receipt['forward_wall_seconds']+=time.monotonic()-wall
        engine.receipt['forward_cpu_seconds']+=time.process_time()-cpu
        engine.forward.check_frozen()
    return np.asarray(outputs),dict(actual_product='J_BG @ J_BG.T @ fixed_Rademacher_v',
        derivative_SDPA_backend='MATH, same frozen architecture/weights; native field numerical parity checked',
        foreground_descriptor_parity_max=max(defects),VJP_count=8,JVP_count=8,successful_model_forwards=16,
        actual_BG_input_pixels=int(background.sum()),FG_descriptor='softFG-weighted mean of actual unit finalLN patch outputs')


def _responses(frame,finite=False):
    image,background,weights=_mask_and_weights(frame)
    assets=frame.ep.producer.get('model_assets',frame.ep.producer)
    key=(frame.ep.source_id,array_hash(image),array_hash(background),array_hash(weights),
         array_hash(frame.x),str(assets.get('checkpoint_sha256')),str(assets.get('config_sha256')),finite)
    if key in _PRODUCT_CACHE:
        value=_PRODUCT_CACHE.pop(key);_PRODUCT_CACHE[key]=value
        return value[0],dict(value[1],product_cache_hit=True)
    fixture=frame.ep.artifacts.get('A021_synthetic_product_fixture')
    if fixture is not None and not finite:
        if not str(frame.ep.producer.get('kind','')).startswith('synthetic'):
            raise ArtifactUnavailable('Synthetic Jacobian product fixture is forbidden for real DINO')
        response=np.asarray(fixture(frame,image,background,weights),float)
        info={'actual_product':'synthetic analytical fixture; not a DINO derivative','actual_DINO_execution':False}
    elif finite:
        # This is an explicit control, never substituted for the main Jacobian.
        response=[];baseline=np.sum(frame.x*weights[:,None],axis=0);rng=np.random.default_rng(0)
        for _ in range(8):
            perturbation=rng.choice((-1.,1.),image.shape)*(2/255)*background[:,:,None]
            view,_=_encode(frame,np.clip(image+perturbation,0,1).astype(np.float32),'A021_finite_control')
            response.append(np.sum(view*weights[:,None],axis=0)-baseline)
        response=np.array(response);info={'control':'eight_actual_small_RGB_BG_perturbations','requested_extra_forwards':8,'amplitude':2/255}
    else:response,info=_actual_product(frame,image,background,weights)
    if response.shape!=(8,frame.x.shape[1]) or not np.isfinite(response).all():
        raise ArtifactUnavailable('A021 must obtain eight finite actual FG-space Jacobian-product responses')
    response.setflags(write=False);_PRODUCT_CACHE[key]=(response,info)
    while len(_PRODUCT_CACHE)>16:_PRODUCT_CACHE.popitem(last=False)
    return response,dict(info,product_cache_hit=False)


def _class_gap(frame,held,projection):
    f,b,_,_,_=frame.banks()
    def project(x):return x-x@projection@projection.T
    z=project(frame.x[held]);score=(nearest_mean_distance(z,project(b))-nearest_mean_distance(z,project(f)))/2
    wf,wb=frame.ep.wf[held],frame.ep.wb[held]
    gap=float(wf@score/wf.sum()-wb@score/wb.sum())
    return gap


def fit_a021(frame,random_projection=False,finite=False):
    if not len(frame.fids) or not len(frame.bids):
        return lambda q,ids,role:b0(frame,q),{'mechanism':'A_B0_5NN'}
    response,production=_responses(frame,finite);basis=rank_basis(response,8)
    folds=[]
    for block in range(4):
        held=np.flatnonzero(frame.train&(frame.blocks==block))
        known=frame.train&(frame.blocks!=block);source=Frame.make(frame.ep,known)
        if (not len(held) or not len(source.fids) or not len(source.bids)
            or frame.ep.wf[held].sum()<=0 or frame.ep.wb[held].sum()<=0):continue
        inner,receipt=_responses(source,finite);inner_basis=rank_basis(inner,8)
        zero=np.empty((frame.x.shape[1],0));base=_class_gap(source,held,zero)
        folds.append((block,source,held,inner_basis,base,receipt))
    selected=[];audit=[]
    for j in range(basis.shape[1]):
        passed=len(folds)>=2;tests=[]
        for block,source,held,inner,base,receipt in folds:
            valid=j<inner.shape[1]
            cosine=float(abs(basis[:,j]@inner[:,j])) if valid else 0.
            if valid:
                use=[k for k in selected+[j] if k<inner.shape[1]];gap=_class_gap(source,held,inner[:,use])
                valid=cosine>=.8 and gap>=base-1e-6
            else:gap=None
            passed&=valid;tests.append(dict(block=block,direction_cosine_abs=cosine,original_gap=base,projected_gap=gap,passed=valid))
        if passed:selected.append(j)
        audit.append(dict(response_singular_direction=j,retained=passed,tests=tests))
    u=basis[:,selected]
    if random_projection and u.shape[1]:u=rank_basis(np.random.default_rng(0).normal(size=(u.shape[1],frame.x.shape[1])))
    f,b,*_=frame.banks()
    def project(x):return x-x@u@u.T
    f,b=project(f),project(b)
    def predict(q,ids,role):
        z=project(q);return (nearest_mean_distance(z,b)-nearest_mean_distance(z,f))/2,{}
    return predict,{'response_rank':basis.shape[1],'deleted_rank':u.shape[1],
        'actual_product_receipt':production,'directions_source_crossblock_audit':audit,
        'inner_product_rebuild_count':len(folds),'same_rank_random_control':random_projection,
        'finite_RGB_control':finite,'readout':'quotient Euclidean5NN, originalunit inputs, no addedrenormalization',
        'cost_bound':'nested guards+outer C_R can require 11 unique label-states: up to176 truemodel forwards/88VJP/88JVP; not8cheapcosines'}


def install(register,requirements,methods,controls,recipes):
    register('A021',fit_a021,(
        'Actual frozen CPU internalmodel, eight normalizedseed0Rademacher FGdescriptor scalars; VJP atphysicalfloatRGB, zero gradient onMRinside/unknownheld/padding, genuine torch.func forward-AD JVP ofFGmean alongthatBGgradient.',
        'MathSDPA derivativebackend recordedand nativeFGdescriptor maxabsoluteparity1e-5 checked; unsupported JVP/backend/memory isexplicitunavailable, finiteRGBnevermainfallback. No weight training or architecture change.',
        'Full8response thinSVD; each nested spatialleaveblock recomputesJacobians fromitsknownlabels. Rank-index directionalignmentabs-cos>=.8 and heldFG/BGmean5NNgap nondecrease1e-6; retain<=8 directions, fixedquotient5NN.',
        'Productcache16tiny8×D responses; canonical known-label states reused exactly. Outer4fold+fullfit+nested guards up-to11uniqueproductstates (176forward/88VJP/88JVP), notsource8-groupheadlinecost.',),
        (('same_retained_rank_random_projection',lambda f:fit_a021(f,random_projection=True)),
         ('eight_actual_BG_RGB_finite_perturbation_space',lambda f:fit_a021(f,finite=True))))
    controls['A021__A001_reference_difference_projection']=methods['A001']
    from .a_algorithms_009_020 import full_profile_rbf
    from .a_helpers_001_050 import infer_a
    controls['A021__full_original_reference_affinity_RBF']=partial(infer_a,method_id='A021__full_original_reference_affinity_RBF',
        fit=full_profile_rbf,assumptions=('Controlonly; originalR/Q actualnative allreferenceprofiles, fullsourceC_Randmask.',))
    requirements['A021']=['internal_encoder','actual_CPU_frozen_autograd_VJP_and_forward_AD_JVP','original_RGB_and_complete_MR','physical_RGB_transform']
    methods['A021']=_receipt(methods['A021'])
    for cid in ('A021__same_retained_rank_random_projection','A021__eight_actual_BG_RGB_finite_perturbation_space'):
        controls[cid]=_receipt(controls[cid])
