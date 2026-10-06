"""Native-score normalization bugfix wrapper for the frozen latent-slot arms.

The original ``latent`` module and its results remain unchanged. This wrapper
changes only the response passed as the native observation: it uses FoRIS's
per-image min/max normalization rather than clipping the raw response directly.
No slot, iteration, temperature, reference-mass, or likelihood-weight setting is
changed. This is an interface repair, not a new mechanism or a renamed success.
"""
from __future__ import annotations

import numpy as np

from . import latent


CONFIG = dict(latent.CONFIG)
CONFIG.update({
    "name": "latent_reference_constrained_slots_native_normalization_bugfix",
    "base_method": latent.CONFIG["name"],
    "native_score_normalization": "(raw-min)/max(max-min,1e-6)",
    "normalization_epsilon": 1e-6,
    "mechanism_parameters_changed": False,
})


def _call(fn, q, r, cov, score, *, device, extras):
    raw = np.asarray(score, dtype=np.float32)
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError("score must be a nonempty HxW native response field")
    if not np.isfinite(raw).all():
        raise ValueError("score contains nonfinite values")
    raw_min, raw_max = float(raw.min()), float(raw.max())
    # Match the shared evaluator's float32 subtraction before scalar conversion.
    denominator = max(float(raw.max() - raw.min()), CONFIG["normalization_epsilon"])
    normalized = (raw - raw.min()) / denominator
    field, info = fn(q, r, cov, normalized, device=device, extras=extras)
    info["config"] = dict(CONFIG)
    info["native_score_interface_repair"] = {
        "applied": True,
        "kind": "FoRIS minmax normalization before unchanged latent likelihood",
        "raw_min": raw_min,
        "raw_max": raw_max,
        "denominator": denominator,
        "epsilon": CONFIG["normalization_epsilon"],
        "old_raw_clip_fraction": float(np.mean((raw < 0) | (raw > 1))),
        "base_module_unchanged": True,
        "mechanism_parameters_changed": False,
    }
    return field, info


def predict(q, r, cov, score, *, device="cpu", extras=None):
    return _call(latent.predict, q, r, cov, score, device=device, extras=extras)


def control(q, r, cov, score, *, device="cpu", extras=None):
    return _call(latent.control, q, r, cov, score, device=device, extras=extras)


def two_slot_control(q, r, cov, score, *, device="cpu", extras=None):
    return _call(latent.two_slot_control, q, r, cov, score, device=device, extras=extras)


additional_controls = {"two_slot_em": two_slot_control}
