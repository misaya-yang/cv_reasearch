"""Source-defined Pro30 M01--M08; controls are complete uncounted methods."""
from __future__ import annotations

from functools import partial
import time
import numpy as np

from .common import (EPS,Episode,Result,ArtifactUnavailable,BR_CONSTANTS,
    validate,unit,blocks,br_margin,br_result,fit_br,finish,degenerate_margin,
    require_artifact,source_contract)
from ics.cpu100 import cross_image_matching_batch2 as inherited

METHODS={}
CONTROLS={'PRO30_B_R':br_result,'PRO30_B_R_same_samples_ridge':partial(br_result,method_id='PRO30_B_R_same_samples_ridge',huber=False)}
REQUIREMENTS={}
CONTRACTS={}


def _inherit(ep,mid,fn):
    validate(ep);start=time.perf_counter();degenerate=degenerate_margin(ep)
    if degenerate is not None:
        margin,info=degenerate
    else:
        result=fn(ep);margin=np.asarray(result.margin).ravel();info=dict(result.info)
        if 'fallback' in info:
            old=dict(info);margin,info=br_margin(ep)
            info=dict(info,inactive=True,inactive_reason=old['fallback'],historical_fallback=old,
                historical_prototype_fallback_replaced_by_source_defined_B_R=True)
        else:info['inactive']=False
    info.update(source_contract(int(mid.split('__')[0].rsplit('M',1)[-1])),
        postprocess_seconds=time.perf_counter()-start,new_encoder_forwards=0,
        inherited_source='CPU100 cross_image_matching_batch2.py; fixed unchanged kernel',
        quality='unmeasured Pro30 fixed version')
    return finish(ep,margin,mid,info,invalid_value=0.)


METHODS['PRO30_M03']=partial(_inherit,mid='PRO30_M03',fn=inherited.joint_sparse_role_removal)
METHODS['PRO30_M04']=partial(_inherit,mid='PRO30_M04',fn=inherited.source_opponent_matching)
for mid,items in (
    ('PRO30_M03',(('independent_OMP3',inherited.independent_sparse_control),
                 ('coefficient_role_vote',inherited.sparse_coefficient_vote_control),
                 ('dense_class_cone',inherited.dense_class_cone_control),
                 ('convex_hull',inherited.convex_hull_control))),
    ('PRO30_M04',(('nearest_opponent_median',inherited.nearest_opponent_median_control),
                 ('allpair_opponent_median',inherited.allpair_opponent_median_control),
                 ('random_opponent_median',inherited.random_opponent_median_control),
                 ('raw_matched_mean',inherited.raw_matched_mean_control)))):
    for name,fn in items:CONTROLS[mid+'__'+name]=partial(_inherit,mid=mid+'__'+name,fn=fn)
    REQUIREMENTS[mid]=['native finalLN unit R/Q','complete MR area weights','physical geometry']
    CONTRACTS[mid]=dict(source_contract(int(mid[-2:])),input_contract='N',host=None,
        constants={'reference_role_modes_max':32,'OMP_active_atoms':3} if mid.endswith('03') else {'common_role_K_max':32,'coincident_difference_cut':1e-10},
        solver_stop='inherited fixed kernel',renderer='CPU100 two-threshold',
        controls=[c for c in CONTROLS if c.startswith(mid+'__')]+['PRO30_B_R','PRO30_B_R_same_samples_ridge'])

from .differential_06 import install as _install06
_install06(METHODS,CONTROLS,REQUIREMENTS,CONTRACTS)
from .mlp_null_08 import install as _install08
_install08(METHODS,CONTROLS,REQUIREMENTS,CONTRACTS)
from .align_factor_02_07 import install as _install02_07
_install02_07(METHODS,CONTROLS,REQUIREMENTS,CONTRACTS)

def _m07_m03_control(ep):
    result=METHODS['PRO30_M03'](ep)
    result.info.update(method_id='PRO30_M07__M03_shared_source_dictionary',control_only=True,reused_original_method='M03')
    return result

def _m07_qp02_control(ep):
    degeneration=degenerate_margin(ep)
    if degeneration is not None:return finish(ep,degeneration[0],'PRO30_M07__historical_QP02',degeneration[1])
    from ics.cpu100.query_partition import qp02
    result=qp02(ep)
    return finish(ep,result.margin,'PRO30_M07__historical_QP02',dict(result.info,control_only=True,reused_original_method='CPU100_QP02'))

CONTROLS['PRO30_M07__M03_shared_source_dictionary']=_m07_m03_control
CONTROLS['PRO30_M07__historical_QP02']=_m07_qp02_control
CONTRACTS['PRO30_M07']['controls'].extend(['PRO30_M07__M03_shared_source_dictionary','PRO30_M07__historical_QP02'])
