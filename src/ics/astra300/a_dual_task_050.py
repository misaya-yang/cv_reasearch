"""Original A050: two actual complete locked-host tasks with opposite MR."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import time
import numpy as np

from .common import ArtifactUnavailable,Result,array_hash,readonly,validate
from .a_helpers_001_050 import Frame,b0,infer_a


def _actual_task(ep,mask,ablate=False):
    # Lazy import: local absence of Torch is an explicit resource limitation.
    # The original episode validates the raw cache; only the host's legal task
    # labels are derived. Native R/Q features do not depend on the chosen MR.
    try:from . import complete_baselines as baseline
    except ImportError as error:
        raise ArtifactUnavailable('A050 actual complete released CPU host dependency unavailable: '+repr(error)) from error
    bundle=baseline._episode_bundle(ep)
    is_complement=np.array_equal(mask,~ep.reference_mask)
    task_ep=replace(ep,reference_mask=readonly(mask),wf=ep.wb if is_complement else ep.wf)
    host,encoder,_,config=baseline._host(task_ep,'foris','crf',bundle['raw'],bundle['basis'])
    captured={};original=host._binarize_response
    def capture(score_hw,*,target_hw):
        score=score_hw-score_hw.min();score=score/score.max().clamp_min(1e-6)
        captured['field']=readonly(score.detach().cpu().numpy())
        return original(score_hw,target_hw=target_hw)
    host._binarize_response=capture
    if ablate:
        host._part3_clustering=lambda score,**kwargs:(score,None,None)
        host._part4_semantic_consistency_correction=lambda score,**kwargs:score
    start=time.monotonic()
    try:
        with baseline.torch.inference_mode(),baseline.torch.autocast('cpu',enabled=False):
            output=baseline._run_host(task_ep,host,encoder)
    except (RuntimeError,ImportError,OSError) as error:
        raise ArtifactUnavailable('A050 actual complete locked FoRIS task failed: '+repr(error)) from error
    finally:
        host._binarize_response=original
        host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None
    if 'field' not in captured or output.dtype!=baseline.torch.bool or tuple(output.shape)!=tuple(ep.original_shape):
        raise ArtifactUnavailable('A050 actual full host response/final original mask missing')
    info=baseline._info(bundle,'foris','crf',time.monotonic()-start,encoder.calls,config)
    info.update(task_reference_mask_array_sha256=array_hash(mask),
        task_reference_mask_is_complement=is_complement,
        original_reference_mask_array_sha256=array_hash(ep.reference_mask),
        actual_task_mask_source='derived exact logical complement of known complete original MR' if is_complement else 'known complete original MR',
        task_label_feature_independence='native same frozen R/Q raw cache reused; Part1 and remaining pipeline independently run',
        Part3_seed_and_Part4_rescore_disabled_control=ablate)
    return dict(field=captured['field'],mask_original=readonly(output.cpu().numpy()),producer=info)


def _task(ep,mask,ablate=False):
    fixture=ep.artifacts.get('A050_synthetic_complete_task_fixture')
    if fixture is not None:
        if not str(ep.producer.get('kind','')).startswith('synthetic'):
            raise ArtifactUnavailable('Synthetic task fixture cannot be attached to a native DINO episode')
        value=fixture(ep,mask,ablate=ablate)
    else:value=_actual_task(ep,mask,ablate)
    field=np.asarray(value['field'],float)
    if field.shape!=tuple(ep.q_hw) or not np.isfinite(field).all() or np.any((field<0)|(field>1)):
        raise ArtifactUnavailable('A050 full-task native continuous probability field required')
    return field,value.get('producer',{})


def _logit(p):
    p=np.clip(p,1e-4,1-1e-4);return np.log(p)-np.log1p(-p)


def execute_a050(ep,mode='dual'):
    validate(ep);start=time.perf_counter()
    if ep.reference_mask is None:raise ArtifactUnavailable('A050 actual complete original MR required')
    if ep.wf.sum()<=0 or ep.wb.sum()<=0:
        # Empty/complement-empty host input has the source-defined numerical
        # degeneration; no purported second task is fabricated.
        frame=Frame.make(ep);score,info=b0(frame,ep.q)
        return Result(score.reshape(ep.q_hw),info=dict(method_id='A050',fit=info,
            degeneration='empty_role_complete_dual_host_undefined_B0cut0',new_encoder_forwards=0,query_GT_read=False))
    sf,first=_task(ep,ep.reference_mask,ablate=mode=='no_seed_no_rescore')
    if mode=='direct_complement':sn=1-sf;second={'control':'algebraic1-minus-first; noactualsecondtask'}
    elif mode=='identical_tasks':sn,second=_task(ep,ep.reference_mask)
    else:sn,second=_task(ep,~ep.reference_mask,ablate=mode=='no_seed_no_rescore')
    score=_logit(sf)-_logit(sn);complement_defect=float(np.max(np.abs(sn-(1-sf))))
    score=np.asarray(score);score.reshape(-1)[ep.q_valid<=0]=-1.
    return Result(score,info=dict(method_id='A050' if mode=='dual' else 'A050__'+mode,
        mode=mode,first_task=first,second_task=second,
        exact_reference_complement_equivariance_max_defect=complement_defect,
        equivalent_to_single_first_task_probability_cut_half=complement_defect<=1e-7,
        source_renderer='A_U_continuous_dual_logit_native_field_original_then_strict0',
        actual_complete_host_tasks=0 if ep.artifacts.get('A050_synthetic_complete_task_fixture') is not None else (1 if mode=='direct_complement' else 2),
        task_invocations=1 if mode=='direct_complement' else 2,
        synthetic_task_wiring_fixture=ep.artifacts.get('A050_synthetic_complete_task_fixture') is not None,
        new_encoder_forwards=0,native_encoder_cost='same bound R/Q actual cache reused; producer extraction is C0',
        cpu_seconds=time.perf_counter()-start,query_GT_read=False))


def install(register,requirements,methods,controls,recipes):
    assumptions=(
        'Same locked released FoRIS source/config/native positionalbasis/actualraw R/Q pack; independently execute completePart1-4+CRF with originalMR and exact1−MR. Nativefeatures shared, taskPart1/seed/rescore decisions notshared.',
        'Both actual continuousfields are source minmaxPart4 responses capturedbeforebinaryreadout; strictlogitdifferencecut0, clip1e-4, originalphysicalU. Completehost finalmasks alsoactuallycomputed though finalA050readoutusescontinuousfields.',
        'ExactcomplementtaskMR isexplicitderivedlabelasset, neverrelabeledasoriginalcacheMRproducer. Complement-equivariance maxdefect isreported; algebraicdegeneration isnotextraevidence.',)
    recipes['A050']={'implementation_assumption':list(assumptions),'threshold':0,'calibration':'fixed original dual-task cut0','renderer':'A_U_strict0'}
    methods['A050']=execute_a050
    for mode in ('direct_complement','identical_tasks','no_seed_no_rescore'):
        controls['A050__'+mode]=partial(execute_a050,mode=mode)
    def prototype(frame):
        from .a_helpers_001_050 import unit
        if frame.wf.sum()<=0 or frame.wb.sum()<=0:return lambda q,ids,role:b0(frame,q),{'mechanism':'A_B0_5NN'}
        axis=unit(np.sum(frame.x*frame.wf[:,None],axis=0))-unit(np.sum(frame.x*frame.wb[:,None],axis=0))
        return lambda q,ids,role:(q@axis,{}),{'control':'original_FG_minus_BG_mean_cosine'}
    controls['A050__ordinary_role_prototype']=partial(infer_a,method_id='A050__ordinary_role_prototype',fit=prototype,
        assumptions=('Control only; fulloriginalMRsameDINO weightedmeanrolecosine, strictfixed0.',),fixed_threshold=0.)
    requirements['A050']=['complete_baseline_assets','actual_RQ_raw_pack','original_RGB_and_complete_MR','CPU_complete_released_FoRIS_CRFincludingdependencies']
