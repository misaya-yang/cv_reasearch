"""Stable user-supplied IDs E226--E250 and F251--F300.

This export only exposes actually implemented algorithms. Resources required
by a supplied card remain explicit, and implementation never establishes gain.
"""
from __future__ import annotations
from . import e226_250
from .common import Result,artifact,host_baseline

METHODS={**e226_250.METHODS}
CONTROLS={**e226_250.CONTROLS}
RESOURCES={**e226_250.RESOURCES}


def mean_actual_stored_zero(ep):
    host=host_baseline(ep)
    return Result(field=artifact(ep,'mean_original_field'),threshold=.5,
                  mask_original=host.mask_original,
                  info=dict(field_space='original',control='actual_stored_MEAN_zero',
                            host_producer=host.info['host_producer'],host_renderer=host.info['host_renderer']))


def mean_continuous_original_zero(ep):
    host=host_baseline(ep)
    return Result(field=artifact(ep,'mean_original_field'),threshold=.5,
                  info=dict(field_space='original',control='continuous_original_MEAN_zero',
                            host_producer=host.info['host_producer'],
                            host_renderer='continuous_field_once_to_actual_original_then_strict_.5'))


CONTROLS.update(E_MEAN_stored_renderer_zero=mean_actual_stored_zero,
                E_MEAN_continuous_original_renderer_zero=mean_continuous_original_zero)

# F helpers are added here after their owner completes the shared F protocol
# checks. There is no endpoint/fallback-only placeholder occupying an ID.
