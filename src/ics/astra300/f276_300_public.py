"""Complete F276--300 wrappers with the common source-IoU and final readout.

Implementation is not a gain claim or a scientific pass. Resource methods only
run with the actual producer-bound observer; native methods do not require it.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from . import common
from . import f_protocol as P
from . import f276_300 as K
from . import f285_299_native as N
from . import f280_300_observations as O

KERNELS={j:getattr(K,f'f{j}') for j in (276,277,278,279,281,283,284)}
KERNELS.update(N.KERNELS);KERNELS.update(O.KERNELS)
CONTROL_NAMES={276:['uniform_deletion'],277:['no_dominant_removal'],278:['mean_head_risk'],279:['fixed_private'],281:['fixed_residual'],
               283:['independent'],284:['fixed_core']}
CONTROL_NAMES.update(N.CONTROLS);CONTROL_NAMES.update(O.CONTROLS)


def _resources(ep,number,extra=None):
    resources=dict(extra or {})
    if ep.reference_mask is not None:
        pixels=np.asarray(ep.reference_mask,bool).copy();pixels.setflags(write=False);resources['reference_training_pixels']=pixels
    if number in O.RESOURCES:
        resources['bound_episode']=ep
        for name in O.RESOURCES[number]:resources[name]=common.artifact(ep,name)
    return resources


def _source_primitive_scale(c,number):
    known=c.source.oof_valid&c.valid
    margin=c.source.heads[known,0]
    if number in (276,283,284):samples=np.r_[0.,.25,1.,1-(c.r[c.valid]@c.fm.T).max(1),1-(c.r[c.valid]@c.bm.T).max(1)]
    elif number in (277,279,290,293,299):samples=np.r_[1-(c.r[c.valid]@c.fm.T).max(1),1-(c.r[c.valid]@c.bm.T).max(1)]
    elif number in (278,286,288,294,300):samples=np.logaddexp(0,-np.abs(margin))
    elif number in (280,282,285,287,289,291,292,295,296,297,298):samples=margin
    elif number==281:samples=c.source.heads[known,2]
    else:raise ValueError('Unspecified F source primitive')
    return P.mad(samples),dict(name='source_cosine_residual_and_explicit_unit_fees' if number in (276,283,284) else 'source_cosine_residual' if number in (277,279,290,293,299) else 'source_own_label_BCE' if number in (278,286,288,294,300) else 'source_OOF_reference_margin',samples=int(np.size(samples)))


def _dictionary(out):
    if isinstance(out,dict):return out
    # A kernel may expose its same-unary probability for audit alongside a
    # discrete optimizer. The discrete mask remains the card's final decision.
    return dict(probability=out.probability if out.labels is None else None,
                y=out.labels,unary=out.unary,info=out.info)


def _pixel_result(ep,c,out,method):
    # A true original-pixel optimizer has no second resize. F's same initial
    # token-unary disagreement is evaluated on that identical original domain.
    labels=np.asarray(out['original_labels'],bool)
    if labels.shape!=ep.original_shape:raise ValueError('F original-pixel optimizer output geometry differs')
    initial=common.U(ep,(c.s>0).reshape(c.hw).astype(float),.5)
    mask=np.where(labels!=initial,labels,c.M0)
    info=dict(c.trace,**out.get('info',{}),quality='unknown',query_GT_read=False,method=method,
              field_space='original',binary_optimizer_readout=True,optimizer_foreground=int(labels.sum()),
              optimizer_sha256=common.array_hash(labels),initial_same_unary_sha256=common.array_hash(initial))
    return common.Result(labels.astype(np.float32),.5,mask,info)


def run(number,ep,*,weight=None,control=None,resources=None):
    if number not in KERNELS:raise ValueError('This module owns F276--300')
    resources=_resources(ep,number,resources);c=P.build_context(ep,resources=resources)
    if c.trace['empty_reference_fg'] or c.trace['single_class'] or any(c.trace['missing_pure_roles']):
        return common.Result(c.p0.reshape(c.hw),.5,info=dict(c.trace,method=f'F{number}',quality='unknown',
                    query_GT_read=False,fallback='negative_or_pure_role_factor_disabled',selected_weight=0.))
    def kernel(context,w):
        if w==0:
            y,certificate=P.exact_e0(context)
            return dict(y=y,info=dict(primary_factor_disabled=True,exact_E0_certificate=certificate))
        scale,definition=_source_primitive_scale(context,number);context.factor_scale=scale
        value_control=bool(control) if number in (276,277,278,279,281,283,284) else control
        try:out=_dictionary(KERNELS[number](context,w/scale,value_control))
        except (ArithmeticError,np.linalg.LinAlgError) as error:
            return dict(probability=context.p0,info=dict(fallback='nonidentifiable_or_numeric_failure',
                       fallback_detail=str(error),source_factor_MAD=scale,source_factor_definition=definition))
        out.setdefault('info',{}).update(source_factor_MAD=scale,source_factor_definition=definition,
          primary_weight_before_MAD=float(w),effective_primary_weight=float(w/scale),quality='unknown',query_GT_read=False)
        return out
    if weight is None:
        # The shared calibrator accepts native dicts. Pixel kernels retain the
        # same once-rendered original readout through this local evaluator.
        if number==296:weight,calibration=_calibrate_pixels(ep,kernel,resources)
        else:weight,calibration=P.calibrate_weight(ep,kernel,resources=resources)
    else:
        if float(weight) not in P.WEIGHTS:raise ValueError('F primary weights exactly0,.25,1')
        calibration=dict(calibration='explicit_fixed_weight_trial',selected_weight=float(weight))
    out=kernel(c,float(weight));out.setdefault('info',{}).update(selected_weight=float(weight),calibration=calibration,control=control,
         implementation_assumptions_file='evidence/local/astra300_20261007/group_226_300/f276_300_implementation.json')
    return _pixel_result(ep,c,out,f'F{number}') if 'original_labels' in out else P.kernel_result(ep,c,out,method=f'F{number}')


def _calibrate_pixels(ep,kernel,resources):
    if ep.reference_mask is None:return 0.,dict(fallback='original_reference_mask_unavailable')
    target=np.asarray(ep.reference_mask,bool);rows=[];folds=P.buffered_folds(ep.r_hw,np.asarray(ep.wvalid)>0)
    valid=[(train,held) for train,held in folds if held.any() and np.any(ep.wf[train]>0) and np.any(ep.wb[train]>0)]
    if len(valid)<2:return 0.,dict(fallback='fewer_than_two_valid_buffered_folds',effective_folds=len(valid))
    for weight in P.WEIGHTS:
        scores=[]
        for train,held in valid:
            episode=P._reference_query(ep,train);c=P.build_context(episode,resources=resources);out=kernel(c,weight)
            result=_pixel_result(episode,c,out,'F296') if 'original_labels' in out else P.kernel_result(episode,c,out)
            mask=common.render(episode,result)['original'];domain=common.U(episode,held.reshape(ep.r_hw).astype(float),.5)
            I=np.count_nonzero(mask&target&domain);U=np.count_nonzero((mask|target)&domain);scores.append(I/U if U else 1.)
        rows.append(dict(weight=weight,fold_IoU=list(map(float,scores)),mean_IoU=float(np.mean(scores))))
    selected=max(rows,key=lambda v:(v['mean_IoU'],-v['weight']))['weight']
    return selected,dict(calibration='F_original_reference_IoU',selected_weight=selected,trials=rows,effective_folds=len(valid),
              held_labels_removed_from_all_banks=True,reference_GT_read_only_after_complete_mask=True)


def _public(number):
    def fn(ep,**kwargs):return run(number,ep,**kwargs)
    fn.__name__=f'F{number}'
    return fn


def _control(number,name):
    def fn(ep,**kwargs):return run(number,ep,control=name,**kwargs)
    fn.__name__=f'F{number}_{name}'
    return fn


METHODS={f'F{n}':_public(n) for n in range(276,301)}
CONTROLS={f'F{n}_{name}':_control(n,name) for n,names in CONTROL_NAMES.items() for name in names}
RESOURCES={f'F{n}':dict(final_native_FP32=True,original_RGB=n in (285,294,296,297),complete_original_MR=True,
        actual_artifacts=O.RESOURCES.get(n,[]),source_calibration='4buffered_original_R_IoU_folds_3weights',
        gain='unknown',native_only=n not in O.RESOURCES) for n in range(276,301)}
