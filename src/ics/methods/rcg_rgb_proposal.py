"""One fixed exploratory composition: locked current RCG + existing RGB readout.

No new evidence, constants or independent mechanism. Recompute current RCG;
use the same RGB128 Potts .25; keep its cut/unary disagreement proposal only.
The final binary1024 mask is not the original RGB Potts optimum.
"""
from __future__ import annotations
import time
import numpy as np

CONFIG=dict(method='locked_rcg_rgb_disagreement_proposal_v2',
            kind='fixed_exploratory_composition',independent_method_increment=0,
            base='current ics.methods.rcg.predict(q,r,cov,rawscore), device=cpu',
            rgb='unchanged mean_rgb_potts RGB128 energy, weight .25',
            proposal='D=cut1024 XOR same128unary1024; final=where(D,cut1024,currentRCG1024)',
            field='binary1024, not continuous probability',potts_optimality_claim=False,
            parameter_search=False,new_encoder_forwards=0,query_gt_used=False,
            exposure='reused public600 exploratory composition; not independent confirmation')


def predict(q,r,cov,rawscore,rgb,*,original_shape=(1024,1024)):
    from .rcg import predict as rcg_predict,CONFIG as RCG_CONFIG
    from .huber_graph import make_mean_inputs
    from .prepared_cpu_bundle import mean_base
    from .mean_rgb_potts import predict as rgb_predict
    from .mean_rgb_proposal import compose
    from .pro_paired_environment import render as render_complete
    from ..experiment import render
    started=time.perf_counter()
    base_started=time.perf_counter()
    rcg,rcg_info=rcg_predict(q,r,cov,rawscore,device='cpu')
    rcg_seconds=time.perf_counter()-base_started
    mean_started=time.perf_counter()
    inputs,mean_pregraph_info=make_mean_inputs(q,r,cov,rawscore)
    mean,mean_info=mean_base(inputs,mean_pregraph_info.get('graph_storage_dtype','float64'))
    mean_seconds=time.perf_counter()-mean_started
    rgb_started=time.perf_counter()
    source=rgb_predict(q,r,cov,rcg,rgb,original_shape=original_shape)
    source['info']['input_base_role']='current locked RCG field; generic RGB energy unchanged'
    rgb_seconds=time.perf_counter()-rgb_started
    baseline=render(rcg)
    final,region,proposal=compose(baseline,source['mask_work'],source['unary_mask_work'])
    field=final.astype(np.float32)
    work,original=render_complete(field,tuple(original_shape))
    if not np.array_equal(work,final):
        raise RuntimeError('Final binary1024 renderer changed the specified composition')
    info=dict(recipe=CONFIG,rcg=dict(config=RCG_CONFIG,solver=rcg_info),
              mean=dict(solver=mean_info,pregraph=mean_pregraph_info),
              rgb=source['info'],proposal_baseline='currentRCG',
              proposal={key.replace('mean','baseline'):value for key,value in proposal.items()},
              rcg_recompute_seconds=rcg_seconds,mean_recompute_seconds=mean_seconds,
              rgb_component_seconds=rgb_seconds,wall_seconds=time.perf_counter()-started,
              query_gt_used=False,new_encoder_forwards=0,independent_method_increment=0,
              real_complete_quality='unmeasured')
    return dict(field=field,mask_work=work,mask_original=original,
                rcg_field=rcg,rcg_mask_work=baseline,mean_field=mean,mean_mask_work=render(mean),
                full_cut_field=source['field'],full_cut_mask_work=source['mask_work'],
                same_unary_field=source['unary_control'],same_unary_mask_work=source['unary_mask_work'],
                disagreement=region,info=info)
