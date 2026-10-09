"""Experimental MEAN simplification, preserving the locked solver and parameters.

The active benchmark entry remains unchanged. Default FoRIS stage2/3 use raw
query features; stage4 normalizes each gated token. Only that redundant positive
scalar gate is bypassed. Finite-precision mask identity requires validation.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ics.methods import mean_control, rcg
from mean_rank_controls import average_rank_cdf

CDF_OPS = SimpleNamespace(minmax=rcg.minmax, rank=average_rank_cdf)


def predict(q, r, cov, score):
    """Same unary and original CG; CDF counts replace scipy average ranks."""
    return mean_control.mean_control(q, r, cov, score, CDF_OPS)


@contextmanager
def without_scalar_feature_gate(host):
    """Bypass stage1 gate only for the default raw-query FoRIS configuration."""
    if not (host.use_raw_target_clustering and host.use_raw_target_scoring):
        raise ValueError('Gate removal requires the default raw-query score/cluster paths')
    original = host._part2_stage1_feature_gating
    host._part2_stage1_feature_gating = lambda fmaps_norm, ref_masks, n_refs: fmaps_norm
    try:
        yield host
    finally:
        host._part2_stage1_feature_gating = original
