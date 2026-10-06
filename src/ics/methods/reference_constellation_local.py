"""Constellation v2: preserve the base outside accepted warped FG support.

The original v1 computes modes, peaks, poses, held-out validation and NMS once.
Only its final fusion domain changes. This is a new version, not a correction
whose results may be attributed to v1. Real segmentation benefit is unmeasured.
"""
from __future__ import annotations

import time

import numpy as np

from .reference_constellation import Config, predict as global_predict


METHOD_ID = 'reference_constellation_local_v2'


def predict(q, r, coverage, base, cfg=Config()):
    """Fuse on union{warped reference coverage > 0}; keep all other values.

    The support includes positive fractional bilinear coverage. There is no
    dilation, extra confidence gate or new geometry parameter. Aspect-ratio
    handling remains exactly that of v1. A pose-unsupported false target is
    retained whenever the base retains it.
    """
    started = time.perf_counter()
    original = global_predict(q, r, coverage, base, cfg)
    base = np.asarray(base, dtype=np.float64)
    prior = original['prior']
    region = prior > 0
    field = base.copy()
    # Reuse v1's exact arithmetic within support; do not clip outside support.
    field[region] = original['field'][region]
    info = dict(original['info'])
    info.update(
        method_id=METHOD_ID,
        fusion_contract='v1 fusion where accepted union prior > 0; original base elsewhere',
        local_region_tokens=int(region.sum()),
        outside_region_tokens=int((~region).sum()),
        outside_region_maximum_field_difference=0.0,
        clipped_base_values=int((((base < 0) | (base > 1)) & region).sum()),
        added_tokens=int(((field > .5) & (base <= .5)).sum()),
        deleted_tokens=int(((field <= .5) & (base > .5)).sum()),
        global_v1_added_tokens=int(((original['field'] > .5) & (base <= .5)).sum()),
        global_v1_deleted_tokens=int(((original['field'] <= .5) & (base > .5)).sum()),
        global_v1_seconds=original['info']['wall_seconds'],
        wall_seconds=time.perf_counter() - started,
        unsupported_false_targets_rejected=False,
        segmentation_benefit='unmeasured',
        complete_dataset_minutes='unmeasured',
    )
    return dict(field=field, global_field=original['field'], prior=prior,
                bag_field=original['bag_field'], local_region=region, info=info)
