"""M08 actual last-MLP down-projection input, exact kernel coordinate readout."""
from __future__ import annotations

from functools import partial
from collections import OrderedDict
import time
import numpy as np
from scipy.linalg import eigh

from ics.cpu100.common import rgb_view
from .common import (EPS,validate,unit,blocks,br_margin,finish,degenerate_margin,
    require_artifact,ArtifactUnavailable,array_hash,readonly,source_contract)
from ics.cpu100.invariance_support import _mm

_PACKET=None
_PINV=OrderedDict()


def _source_binding(ep,binding):
    if str(ep.producer.get('kind','')).startswith('synthetic'):return
    if not isinstance(binding,dict) or binding.get('execution_kind')!='real_frozen_cpu_model':
        raise ArtifactUnavailable('M08 requires source-bound actual frozen CPU MLP observations')
    expected=ep.producer.get('model_assets',ep.producer);observed=binding.get('model_assets',binding)
    for key in ('checkpoint_sha256','config_sha256'):
        if expected.get(key) and expected[key]!=observed.get(key):raise ArtifactUnavailable('M08 model source mismatch: '+key)


def _actual_packet(ep):
    global _PACKET
    if _PACKET is not None and _PACKET['episode'] is ep:return _PACKET['packet'],True
    supplied=ep.artifacts.get('last_mlp_pre_down')
    if supplied is not None:
        if supplied.get('stage')!='actual_last_MLP_down_projection_input':raise ArtifactUnavailable('M08 cannot substitute final-LN or arbitrary midlayer features for actual pre-down activation')
        _source_binding(ep,supplied.get('binding',{}))
        if not str(ep.producer.get('kind','')).startswith('synthetic'):
            expected=ep.producer.get('source_image_hashes')
            if not expected or supplied.get('source_image_hashes')!=expected:
                raise ArtifactUnavailable('M08 supplied observations do not bind the exact R/Q image hashes')
            if supplied.get('physical_transforms')!={'r':ep.reference_geometry,'q':ep.query_geometry}:
                raise ArtifactUnavailable('M08 supplied observations do not bind the exact R/Q physical transforms')
            if not all(role in supplied.get('final_feature_parity_max',{}) and supplied['final_feature_parity_max'][role]<=1e-5 for role in ('r','q')):
                raise ArtifactUnavailable('M08 supplied pre-down observations lack actual normal-forward native-feature parity')
        packet=supplied
    else:
        engine=require_artifact(ep,'internal_encoder');binding=engine.binding;_source_binding(ep,binding)
        model,torch=engine.model,engine.torch;engine.forward.check_frozen()
        mlp=model.blocks[-1].mlp;dimension=ep.r.shape[1]
        candidates=[(name,module) for name,module in mlp.named_modules()
                    if isinstance(module,torch.nn.Linear) and module.out_features==dimension and module.in_features>dimension]
        if len(candidates)!=1:raise ArtifactUnavailable('M08 needs one actual lastMLP down Linear with hidden width>nativewidth; observed '+str([v[0] for v in candidates]))
        name,down=candidates[0];activations={};parity={};extra_before=engine.receipt['actual_extra_forwards']
        for role,expected in (('r',ep.r),('q',ep.q)):
            captured=[]
            def prehook(module,args):captured.append(args[0].detach().cpu().numpy().copy())
            hook=down.register_forward_pre_hook(prehook)
            try:
                out=engine.encode_rgb(role,rgb_view(ep,role),working_canvas=True,raw=False)
            finally:hook.remove()
            if len(captured)!=1 or captured[0].shape[0]!=1 or captured[0].shape[1]!=len(expected)+model.num_prefix_tokens:
                raise ArtifactUnavailable('M08 hook must observe one actual full prefix+patch down-input tensor')
            activations[role]=readonly(captured[0][0,model.num_prefix_tokens:])
            defect=float(np.max(np.abs(np.asarray(out).reshape(expected.shape)-expected)));parity[role]=defect
            if defect>1e-5:raise ArtifactUnavailable('M08 new actual normal forward differs from native final descriptor: '+str(parity))
        weight=readonly(down.weight.detach().cpu().numpy());bias=None if down.bias is None else readonly(down.bias.detach().cpu().numpy())
        packet=dict(activations,weight=weight,bias=bias,binding=binding,stage='actual_last_MLP_down_projection_input',
            module_path='blocks.'+str(len(model.blocks)-1)+'.mlp.'+name,MLP_class=type(mlp).__name__,
            weight_array_sha256=array_hash(weight),bias_present=bias is not None,activation_stage='exact tensor passed into actual down module, after real gating/activation/dropout',
            final_feature_parity_max=parity,source_image_hashes=ep.producer.get('source_image_hashes'),
            physical_transforms={'r':dict(ep.reference_geometry),'q':dict(ep.query_geometry)},
            actual_extra_forwards=engine.receipt['actual_extra_forwards']-extra_before)
        engine.forward.check_frozen()
    weight=np.asarray(packet['weight']);r,q=np.asarray(packet['r']),np.asarray(packet['q'])
    if weight.ndim!=2 or weight.shape[0]!=ep.r.shape[1] or weight.shape[1]<=weight.shape[0] or r.shape!=(len(ep.r),weight.shape[1]) or q.shape!=(len(ep.q),weight.shape[1]):
        raise ArtifactUnavailable('M08 actual down weight/activation shapes do not match native R/Q')
    if not all(np.isfinite(v).all() for v in (weight,r,q)):raise ValueError('Nonfinite actual MLP observations')
    if packet.get('weight_array_sha256') and packet['weight_array_sha256']!=array_hash(weight):raise ArtifactUnavailable('M08 down weight hash changed')
    _PACKET={'episode':ep,'packet':packet};return packet,False


def _projector(weight):
    key=array_hash(weight);start=time.perf_counter()
    if key in _PINV:
        inverse,info=_PINV.pop(key);_PINV[key]=(inverse,info)
        return inverse,dict(info,weight_factor_cache_hit=True,factor_seconds_this_call=0.)
    w=np.asarray(weight,float);gram=w@w.T;values,vectors=eigh(gram)
    tolerance=np.finfo(float).eps*max(w.shape)*max(float(values.max()),1.)
    active=values>tolerance;inverse=(vectors[:,active]/values[active])@vectors[:,active].T
    info=dict(weight_rank=int(active.sum()),nullity=w.shape[1]-int(active.sum()),
        gram_eigen_tolerance=tolerance,weight_factor_cache_hit=False,factor_seconds_this_call=time.perf_counter()-start,
        formula='Ndelta=delta-W.T@(W W.T)^+@Wdelta; no hiddenwidth² matrix')
    _PINV[key]=(inverse,info)
    while len(_PINV)>2:_PINV.popitem(last=False)
    return inverse,info


def _kernel(weight,inverse,delta):
    return delta-weight.T@(inverse@(weight@delta))


def _append(ep,rh,qh):
    selected=ep.wvalid>0;mean=float(np.mean(rh[selected]));scale=max(float(np.std(rh[selected])),1e-4)
    r=unit(np.c_[ep.r,.5*(rh-mean)/scale]);q=unit(np.c_[ep.q,.5*(qh-mean)/scale])
    score,fit=br_margin(ep,r,q)
    return score,dict(fit,auxiliary_reference_mean=mean,auxiliary_reference_std=scale,extra_scalar_scale=.5)


def mlp_null(ep,mode='kernel'):
    validate(ep);start=time.perf_counter();mid='PRO30_M08' if mode=='kernel' else 'PRO30_M08__'+mode
    degenerate=degenerate_margin(ep)
    if degenerate is not None:return finish(ep,degenerate[0],mid,degenerate[1])
    if mode=='final_nonlinear':
        axis=unit(np.sum(ep.r*ep.wf[:,None],axis=0))-unit(np.sum(ep.r*ep.wb[:,None],axis=0))
        score,fit=_append(ep,(ep.r@axis)**3,(ep.q@axis)**3)
        return finish(ep,score,mid,dict(fit,**source_contract(8),control='same_one_extra_scalar_native_cubic_readout',
            new_encoder_forwards=0,inactive=False,postprocess_seconds=time.perf_counter()-start))
    partition=blocks(ep.r_hw);possible=[j for j in range(4) if ep.wf[partition==j].sum()>=2 and ep.wb[partition==j].sum()>=2]
    ungated=('full_activation','full_activation_alone','rowspace_unconditional','MLP_output_unconditional')
    if len(possible)<2 and mode not in ungated:
        score,fit=br_margin(ep);return finish(ep,score,mid,dict(fit,inactive=True,inactive_reason='fewer_than_two_mass_valid_reference_blocks',
            valid_mass_blocks=possible,new_encoder_forwards=0,source_tensor_extraction_skipped_by_exact_known_gate=True,**source_contract(8)))
    packet,hit=_actual_packet(ep);r,q=np.asarray(packet['r']),np.asarray(packet['q']);w=np.asarray(packet['weight'],float)
    inverse,factor_info=_projector(w)
    def delta_of(mask):
        f=ep.wf*mask;b=ep.wb*mask
        return _mm(f/f.sum(),r)-_mm(b/b.sum(),r)
    delta=delta_of(np.ones(len(ep.r),bool));v=_kernel(w,inverse,delta);norm=float(np.linalg.norm(v));delta_norm=float(np.linalg.norm(delta))
    block_vectors=[]
    for j in possible:
        vv=_kernel(w,inverse,delta_of(partition==j))
        if np.linalg.norm(vv)>1e-8:block_vectors.append(vv/np.linalg.norm(vv))
    similarity=np.array(block_vectors)@np.array(block_vectors).T if block_vectors else np.zeros((0,0))
    pairs=similarity[np.triu_indices(len(block_vectors),1)]
    median=float(np.median(pairs)) if len(pairs) else -1.;ratio=norm/max(delta_norm,1e-8)
    active=len(block_vectors)>=2 and median>=0 and ratio>=.05
    gate=dict(valid_mass_blocks=possible,valid_nonzero_direction_blocks=len(block_vectors),median_block_direction_cosine=median,
        null_fraction=ratio,active=active,null_projection_relative_error=float(np.linalg.norm(w@v)/max(norm,1e-8)))
    if not active and mode not in ungated:
        score,fit=br_margin(ep);return finish(ep,score,mid,dict(fit,inactive=True,inactive_reason='source_kernel_stability_gate_failed',
            gate=gate,weight_factor=factor_info,observation_cache_hit=hit,new_encoder_forwards=0 if hit else packet.get('actual_extra_forwards',0),**source_contract(8)))
    if mode=='full_activation_alone':
        score,fit=br_margin(ep,unit(r),unit(q))
    elif mode=='full_activation':
        score,fit=br_margin(ep,unit(np.c_[ep.r,.5*unit(r)]),unit(np.c_[ep.q,.5*unit(q)]))
    else:
        if mode in ('rowspace','rowspace_unconditional'):direction=delta-v
        elif mode=='random_kernel':
            direction=_kernel(w,inverse,np.random.default_rng(0).normal(size=len(delta)))
            direction*=norm/max(float(np.linalg.norm(direction)),1e-8)
        elif mode in ('MLP_output','MLP_output_unconditional'):
            rr=_mm(r,w.T);qq=_mm(q,w.T)
            difference=_mm(ep.wf/ep.wf.sum(),rr)-_mm(ep.wb/ep.wb.sum(),rr)
            direction=unit(difference);rh=_mm(rr,direction);qh=_mm(qq,direction)
        elif mode=='kernel':direction=v
        else:raise ValueError(mode)
        if mode not in ('MLP_output','MLP_output_unconditional'):
            direction=direction/max(float(np.linalg.norm(direction)),1e-8);rh=_mm(r,direction);qh=_mm(q,direction)
        score,fit=_append(ep,rh,qh)
    info=dict(source_contract(8),mode=mode,gate=gate,weight_factor=factor_info,fit=fit,inactive=False,
        observation_cache_hit=hit,module_path=packet.get('module_path'),MLP_class=packet.get('MLP_class'),
        weight_array_sha256=array_hash(packet['weight']),bias_present=packet.get('bias_present'),bias_scope='source auxiliary scalar is centered; common output bias is removed by reference centering',
        observation_stage=packet['stage'],new_encoder_forwards=0 if hit else packet.get('actual_extra_forwards',0),
        unconditional_same_capacity_control=mode in ('rowspace_unconditional','MLP_output_unconditional'),
        postprocess_seconds=time.perf_counter()-start,quality='unmeasured new tensor; synthetic fixture is not reachableRGB evidence')
    return finish(ep,score,mid,info)


def install(methods,controls,requirements,contracts):
    methods['PRO30_M08']=mlp_null
    for mode in ('MLP_output','full_activation','full_activation_alone','rowspace','random_kernel','final_nonlinear',
                 'rowspace_unconditional','MLP_output_unconditional'):
        controls['PRO30_M08__'+mode]=partial(mlp_null,mode=mode)
    requirements['PRO30_M08']=['actual_last_MLP_down_input_RQ','actual_down_Linear_weight_and_sourcebinding','or boundinternal_encoder for genuine RGB re-extraction']
    contracts['PRO30_M08']=dict(source_contract(8),input_contract='N+X',host='B_R',
        constants={'source_mass_per_role_per_block_min':2,'mass_valid_blocks_min':2,'median_direction_cosine_min':0,'norm_ratio_min':.05,'auxiliary_scale':.5,'auxiliary_std_min':1e-4},
        renderer='CPU100 two-threshold',solver_stop='once-worker exact weightgram pseudoinverse; same fixed B_R12IRLS',
        controls=[c for c in controls if c.startswith('PRO30_M08__')]+['PRO30_B_R'],
        numerical_assumption='MoorePenrose weightgram eigenvalue cutoff eps64*max(weightshape)*max(largestEigen,1); disclose rank/error; no hiddenwidth² projector')
