"""M01 exact-host MEAN16 system and conservative complete-mask CG stop.

The current implementation explicitly reports real-model certification plus
machine consistency checks. It does NOT assert a rigorous floating-point proof.
"""
from __future__ import annotations

from functools import partial
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.stats import rankdata

from .common import (EPS,validate,Result,readonly,require_artifact,ArtifactUnavailable,
    degenerate_margin,finish,source_contract,array_hash)

_SYSTEM=None


def _episode_identity(ep,*,renderer='sourceRCG_FP32_1024_threshold_half_then_B_noCRF_binary_original'):
    assets=ep.producer.get('model_assets',ep.producer)
    images=ep.producer.get('source_image_hashes')
    if not isinstance(images,list) or len(images)!=2 or not all(images) or ep.reference_mask is None:
        raise ArtifactUnavailable('M01 saved system requires exact original images and complete MR identity')
    checkpoint=assets.get('checkpoint_sha256');config=assets.get('config_sha256',assets.get('model_config_sha256'))
    if not checkpoint or not config:raise ArtifactUnavailable('M01 saved system requires exact native checkpoint/config identity')
    return dict(schema='PRO30_MEAN16_EPISODE_IDENTITY_V1',source_image_hashes=images,
        original_MR_array_sha256=array_hash(ep.reference_mask),wf_array_sha256=array_hash(ep.wf),
        valid_R_array_sha256=array_hash(ep.wvalid),valid_Q_array_sha256=array_hash(ep.q_valid),
        native_unit_R_array_sha256=array_hash(ep.r),native_unit_Q_array_sha256=array_hash(ep.q),
        physical_transforms={'r':ep.reference_geometry,'q':ep.query_geometry},
        checkpoint_sha256=checkpoint,config_sha256=config,
        locked_host_source_archive_sha256='0ab1e809f5a753b4af67b54d158b799e4b2c3beff217f8fc8b1c8b6e50fe294f',
        MEAN_constants={'alpha':.25,'mutual20NN':20,'lambda':16,'confidence_floor':.1},
        renderer=renderer)


def _system_integrity(system):
    h=system['H'].tocsr()
    return dict(H_data=array_hash(h.data),H_indices=array_hash(h.indices),H_indptr=array_hash(h.indptr),
        H_shape=list(h.shape),rhs=array_hash(system['rhs']),y=array_hash(system['y']),a=array_hash(system['a']),
        native_hw=list(system['native_hw']))


def _actual_system(ep):
    global _SYSTEM
    if _SYSTEM is not None and _SYSTEM['episode'] is ep:return _SYSTEM['system'],True
    supplied=ep.artifacts.get('pro30_MEAN16_system')
    if supplied is not None:
        if not str(ep.producer.get('kind','')).startswith('synthetic'):
            producer=supplied.get('producer',{})
            if (producer.get('system_binding')!=_episode_identity(ep)
                or producer.get('complete_locked_host') is not True
                or producer.get('system_payload_integrity')!=_system_integrity(supplied)):
                raise ArtifactUnavailable('M01 system must bind actual locked complete host/current RQ images')
        system=supplied
    else:
        try:from ics.astra300 import complete_baselines as base
        except ImportError as error:raise ArtifactUnavailable('M01 actual CPU host dependency unavailable: '+repr(error)) from error
        bundle=base._episode_bundle(ep);host,encoder,_,config=base._host(ep,'foris','crf',bundle['raw'],bundle['basis'])
        torch=base.torch;captured={};part1=host._part1_positional_debias;binarize=host._binarize_response;finalizer=host._finalize_mask
        def capture_part1(*args,**kwargs):
            value=part1(*args,**kwargs);captured['part1']=value.detach().clone();return value
        def capture_score(score_hw,*,target_hw):
            captured['score']=score_hw.detach().cpu().numpy().copy();return binarize(score_hw,target_hw=target_hw)
        host._part1_positional_debias=capture_part1;host._binarize_response=capture_score
        # MEAN's field belongs before the working threshold/finalizer. Defer the
        # one real finalizer until that field exists; do not run an unused CRF
        # on the unmodified FoRIS mask and then a second CRF on the MEAN mask.
        host._finalize_mask=lambda mask,tgt_image:mask
        start=time.monotonic()
        try:
            with torch.inference_mode(),torch.autocast('cpu',enabled=False):base._run_host(ep,host,encoder)
        finally:
            host._part1_positional_debias=part1;host._binarize_response=binarize;host._finalize_mask=finalizer
        if 'part1' not in captured or 'score' not in captured:raise ArtifactUnavailable('M01 actual Part1/Part4 capture missing')
        features=captured['part1'][0];r=features[0].permute(1,2,0).reshape(4096,1024);q=features[1].permute(1,2,0).reshape(4096,1024)
        r=torch.nn.functional.normalize(r,dim=1);q=torch.nn.functional.normalize(q,dim=1)
        if ep.q_hw!=(64,64) or ep.r_hw!=(64,64) or np.any(ep.q_valid<1-1e-12):
            raise ArtifactUnavailable('M01 current locked full-Foris/RCG path requires complete physical64×64 square warp; not a letterbox substitute')
        coverage=np.divide(ep.wf,ep.wvalid,out=np.zeros(len(ep.r)),where=ep.wvalid>0)
        pure=np.flatnonzero(coverage>=.9)
        if not len(pure):pure=np.flatnonzero(coverage==coverage.max())
        guide=(q@torch.nn.functional.normalize(r[pure].mean(0),dim=0)).cpu().numpy()
        score=np.asarray(captured['score'],np.float32);s=(score-score.min())/max(float(score.max()-score.min()),1e-6);s=s.ravel()
        def rank(x):return ((rankdata(x,method='average')-.5)/len(x)).astype(np.float32)
        y=(s+.25*(rank(guide)-rank(s))).astype(np.float64)
        # Literal retained rcg.py graph: nativeTorch topk and FP32 local-distance
        # weights, reciprocal product/sqrt, mean-degree normalization.
        similarity=q@q.T;similarity.fill_diagonal_(-2);value,index=similarity.topk(20,dim=1);del similarity
        distance=(1-value).clamp_min(0);weight=torch.exp(-distance/distance[:,-1:].clamp_min(1e-6)).cpu().numpy().ravel()
        w=sparse.csr_matrix((weight,(np.repeat(np.arange(4096),20),index.cpu().numpy().ravel())),shape=(4096,4096))
        w=w.multiply(w.T);w.data=np.sqrt(w.data);degree=np.asarray(w.sum(1)).ravel();w=w/max(float(degree.mean()),1e-8)
        degree=np.asarray(w.sum(1)).ravel();a=.1+np.abs(2*s-1);a=a/a.mean();a=a.astype(np.float64)
        matrix=sparse.diags(a)+16*(sparse.diags(degree)-w);rhs=a*y
        target=encoder.expected['q'][None]
        def work_continuous(z):
            t=torch.from_numpy(np.asarray(z,np.float32).reshape(64,64))[None,None]
            return torch.nn.functional.interpolate(t,(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy()
        def final_mask(work):
            host._orig_tgt_size=tuple(ep.original_shape)
            with torch.inference_mode():value=host._finalize_mask(torch.from_numpy(np.asarray(work,bool)),target)
            return value.cpu().numpy().copy()
        producer=base._info(bundle,'foris','crf',time.monotonic()-start,encoder.calls,config)
        producer.update(complete_locked_host=True,source_image_hashes=ep.producer.get('source_image_hashes'),
            graph_feature_stage='actual source Part1 result, native normalized again as rcg.py',
            guide='unit mean pureR_FG, alpha.25',actual_query_graph='unchangedTorchmutual20NN/localbandwidth/sqrt/mean_degree',
            actual_finalizer='released FoRIS source CRF+original resize, driven by MEAN working mask',
            original_FoRIS_field_finalizer_deferred_to_MEAN_field=True)
        system=dict(H=matrix.tocsr(),rhs=rhs,y=y,a=a,native_hw=(64,64),work_continuous=work_continuous,
            final_mask=final_mask,producer=producer,valid_work=np.ones((1024,1024),bool))
        producer.update(system_binding=_episode_identity(ep,renderer='sourceRCG_FP32_1024_threshold_half_then_locked_FoRIS_CRForiginal'),system_payload_integrity=_system_integrity(system))
    h=system['H'].tocsr();n=int(np.prod(system['native_hw']))
    if h.shape!=(n,n) or np.asarray(system['rhs']).shape!=(n,) or np.asarray(system['y']).shape!=(n,):
        raise ArtifactUnavailable('M01 exact system shapes must align the complete native field')
    asym=h-h.T
    if asym.nnz and np.max(np.abs(asym.data))>1e-12:raise ArtifactUnavailable('M01 certificate requires actual symmetric graph system')
    off=h-sparse.diags(h.diagonal());row_sum=np.asarray(h.sum(1)).ravel()
    if np.any(off.data>0) or np.any(row_sum<=0):raise ArtifactUnavailable('M01 actual stored H must be strictly dominant M-matrix')
    system=dict(system,H=h,row_sum_min=float(row_sum.min()))
    _SYSTEM={'episode':ep,'system':system};return system,False


class _Early(Exception):
    def __init__(self,value):self.value=value.copy()


def solve_certified(system,mode='certificate'):
    h,rhs,y=system['H'],np.asarray(system['rhs'],float),np.asarray(system['y'],float)
    amin=system['row_sum_min'];iterations=[0];history=[];accepted=[False]
    def callback(z):
        iterations[0]+=1
        if mode=='full_original_cg':return
        if iterations[0]%2:return
        residual=rhs-h@z
        if mode=='fixed20':
            if iterations[0]>=20:raise _Early(z)
            return
        if mode=='same_residual_early':
            if np.max(np.abs(residual))<1e-5:raise _Early(z)
            return
        epsilon=float(np.max(np.abs(residual))/amin)
        work=np.asarray(system['work_continuous'](z),float);valid=np.asarray(system.get('valid_work',np.ones(work.shape)),bool)
        # Conservative floating slacks make the machine consistency test harder
        # but are not a verified directed-rounding interval enclosure. Keep this
        # boundary explicit in every returned record, per the original card.
        cast=float(np.max(np.abs(z.astype(np.float32).astype(float)-z)))
        slack=float(16*np.finfo(np.float32).eps*max(float(np.max(np.abs(z))),1.))
        machine_epsilon=epsilon+cast+slack
        uncertain=int(np.count_nonzero(valid&(np.abs(work-.5)<=machine_epsilon)))
        history.append(dict(iteration=iterations[0],real_model_epsilon=epsilon,machine_consistency_slack=cast+slack,
            uncertified_pixels=uncertain,work_pixels=int(valid.sum())))
        if mode=='certificate' and uncertain==0:accepted[0]=True;raise _Early(z)
    started=time.perf_counter()
    try:z,status=cg(h,rhs,x0=y,rtol=1e-7,atol=1e-9,maxiter=300,callback=callback)
    except _Early as stop:z,status=stop.value,0
    if status!=0:raise RuntimeError('M01 actual original300-stepCG did not converge: '+str(status))
    return z,dict(iterations=iterations[0],early_stopped_by_complete_mask_test=accepted[0],history=history,
        certified_reference='exact solution of same storedH real-arithmetic model',
        machine_precision_strict_certificate=False,numerical_status='real-model certificate + conservative machine consistency validation; directed-rounding closure not implemented',
        solver_seconds=time.perf_counter()-started,CG={'rtol':1e-7,'atol':1e-9,'maxiter':300,'preconditioner':'originalidentity'})


def certified_mask(ep,mode='certificate'):
    validate(ep);start=time.perf_counter();mid='PRO30_M01' if mode=='certificate' else 'PRO30_M01__'+mode
    degeneration=degenerate_margin(ep)
    if degeneration is not None:return finish(ep,degeneration[0],mid,dict(degeneration[1],RCG_host_degenerate_exception=True))
    system,hit=_actual_system(ep);z,info=solve_certified(system,mode)
    work=np.asarray(system['work_continuous'](z))>.5;mask=np.asarray(system['final_mask'](work),bool)
    if mask.shape!=tuple(ep.original_shape):raise ArtifactUnavailable('M01 actual host original renderer output shape differs')
    return Result(z.reshape(system['native_hw']),.5,readonly(mask),dict(source_contract(1),**info,
        method_id=mid,actual_host=system['producer'],host_system_cache_hit=hit,
        renderer='actual source RCG FP32continuous→1024>.5 then lockedhost finalizer',
        query_GT_read=False,new_encoder_forwards=0,postprocess_seconds=time.perf_counter()-start,
        quality='unmeasured computational candidate; no semanticgain claim'))


def install(methods,controls,requirements,contracts):
    methods['PRO30_M01']=certified_mask
    for mode in ('full_original_cg','fixed20','same_residual_early'):
        controls['PRO30_M01__'+mode]=partial(certified_mask,mode=mode)
    requirements['PRO30_M01']=['actual complete locked FoRIS + exactMEAN16 system','nativePart1 features/rawPart4 response','actual source hostrenderer']
    contracts['PRO30_M01']=dict(source_contract(1),input_contract='N+actualhostprocessedinternal',host='complete FoRIS+MEAN16',
        constants={'alpha':.25,'mutual_query_knn':20,'lambda':16,'confidence_floor':.1,'certificate_frequency':2,'uncertified_budget':0},
        solver_stop={'rtol':1e-7,'atol':1e-9,'maxiter':300,'preconditioner':'originalidentity, not new Jacobi'},
        renderer='originalRCGworkingthreshold and actualhostCRF/originalmapping; never common directoriginal',
        controls=[c for c in controls if c.startswith('PRO30_M01__')],
        limitations='Strict machine interval proof incomplete: explicitly labeled realmodelcertificate+machineconsistency; source permits this label. No claimed rigorous machine certificate or measuredspeedup.')
