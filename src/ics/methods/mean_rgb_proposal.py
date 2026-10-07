"""Readout revision v2: keep MEAN outside the RGB-cut/unary disagreement.

This is an explicitly post-hoc simple control revision (independent increment
zero). It retains the already defined RGB Potts component without retaining its
unary128 renderer everywhere. The final 1024 binary mask is not a Potts optimum.
"""
from __future__ import annotations
import time
import numpy as np

CONFIG = dict(method='mean_rgb_disagreement_local_proposal_v2',
              kind='strong_simple_control_revision',independent_method_increment=0,
              proposal_region='D = RGBcut1024 XOR same128unary1024',
              finalizer='Hfinal = where(D, RGBcut1024, originalMEAN1024)',
              output='binary 1024 mask; original-size mask by bilinear/strict .5',
              zero_graph='cut=unary implies exact original MEAN1024',
              parameter_search=False,query_gt_used=False,new_encoder_forwards=0,
              potts_optimality_claim=False,
              exposure='posthoc exploratory replay on reused public600; no independent confirmation')


def compose(mean_mask, cut_mask, unary_mask):
    """Boolean local substitution; no scores, labels, confidence or parameters."""
    values = [np.asarray(x) for x in (mean_mask,cut_mask,unary_mask)]
    if (any(x.dtype != np.bool_ or x.ndim != 2 for x in values)
            or not values[0].size or any(x.shape != values[0].shape for x in values)):
        raise ValueError('Aligned nonempty 2D boolean masks required')
    mean,cut,unary = values
    disagreement = cut ^ unary
    final = np.where(disagreement,cut,mean)
    if not np.array_equal(final[~disagreement],mean[~disagreement]):
        raise RuntimeError('MEAN preservation outside disagreement failed')
    return final,disagreement,dict(disagreement_pixels=int(disagreement.sum()),
                                  changed_vs_mean_pixels=int(np.count_nonzero(final!=mean)),
                                  added_vs_mean_pixels=int(np.count_nonzero(final & ~mean)),
                                  deleted_vs_mean_pixels=int(np.count_nonzero(mean & ~final)),
                                  outside_disagreement_mean_xor=0,
                                  zero_disagreement_mean_identity=bool(np.array_equal(final,mean)) if not disagreement.any() else None)


def predict(q,r,cov,base,rgb,*,original_shape=(1024,1024)):
    """Same original RGB Potts component -> locally revised complete output."""
    from .mean_rgb_potts import predict as source_predict
    from .pro_paired_environment import render
    from ..experiment import render as render_mean
    started=time.perf_counter()
    source=source_predict(q,r,cov,base,rgb,original_shape=original_shape)
    mean=render_mean(base) if np.asarray(cov).any() else np.zeros((1024,1024),bool)
    final,disagreement,info=compose(mean,source['mask_work'],source['unary_mask_work'])
    field=final.astype(np.float32)
    work,original=render(field,tuple(original_shape))
    if not np.array_equal(work,final):
        raise RuntimeError('1024 binary field renderer changed the final mask')
    return dict(field=field,mask_work=work,mask_original=original,disagreement=disagreement,
                full_cut_control=source['mask_work'],same_unary_control=source['unary_mask_work'],
                mean_control=mean,info=dict(recipe=CONFIG,proposal=info,source_component=source['info'],
                                          wall_seconds=time.perf_counter()-started,
                                          inherited_MEAN_recompute_cost_included=False))
