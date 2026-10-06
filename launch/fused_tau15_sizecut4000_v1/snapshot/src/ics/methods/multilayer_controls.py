"""One frozen additional control for D: the normalized depth difference alone.

The v1 transition and concat recipes remain in multilayer.py unchanged.
All arms share one real pair extraction and one global black-image extraction.
No query label, fitted coefficient, layer search or per-episode selection.
"""
from contextlib import contextmanager
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import multilayer
from .layer_extract import prepare_bases as prepare_v1_bases


CONFIG = {
    "name": "multilayer.delta_only", "layers_one_based": [16, 24],
    "descriptor": "unit(unit_l24-unit_l16)", "positional_rank": 500,
    "readout": multilayer.CONFIG["readout"],
    "field_semantics": multilayer.CONFIG["field_semantics"],
    "query_gt_used": False, "fitted_parameters": 0,
}


def delta_embedding(layers, *, device):
    # Reuse v1 normalization/difference exactly; remove only the late branch.
    joined = multilayer.embedding(layers, "transition", device=device)
    return F.normalize(joined[:, joined.shape[1] // 2:], dim=1)


@contextmanager
def _capture_black_layers(encoder):
    """Observe v1's existing black forward without changing its outputs."""
    name = "get_intermediate_layers"
    had_instance_value = name in encoder.__dict__
    instance_value = encoder.__dict__.get(name)
    previous = getattr(encoder, name)
    captured = {}

    def observe(images, *args, **kwargs):
        if captured or images.shape[0] != 1 or kwargs.get("n") != [15, 23]:
            raise ValueError("Expected exactly the v1 two-layer black-image forward")
        outputs = previous(images, *args, **kwargs)
        if len(outputs) != 2:
            raise ValueError("Both true black-image layers are required")
        for layer, fmap in zip(("16", "24"), outputs):
            captured[layer] = fmap[0].permute(1, 2, 0).reshape(-1, fmap.shape[1])
        return outputs

    setattr(encoder, name, observe)
    try:
        yield captured
    finally:
        if had_instance_value:
            setattr(encoder, name, instance_value)
        else:
            delattr(encoder, name)


@torch.inference_mode()
def prepare_bases(host):
    """Unchanged v1 bases plus delta basis, sharing v1's one black forward."""
    started = time.perf_counter()
    with _capture_black_layers(host.encoder) as layers:
        bases, receipt = prepare_v1_bases(host)
    if set(layers) != {"16", "24"}:
        raise ValueError("No real black-layer features were captured")
    if receipt["black_forwards"] != 1:
        raise ValueError("The frozen v1 basis preparation must use one black forward")
    before_delta = time.perf_counter()
    x = delta_embedding(layers, device=host.device).T
    x = F.normalize(x, dim=0)
    x = x - x.mean(dim=1, keepdim=True)
    if min(x.shape) < CONFIG["positional_rank"]:
        raise ValueError("Delta descriptor cannot supply the frozen rank-500 basis")
    u, _, _ = torch.linalg.svd(x, full_matrices=False)
    bases["delta"] = u[:, :CONFIG["positional_rank"]].contiguous().cpu()
    receipt = dict(receipt)
    receipt["v1_seconds_total"] = receipt["seconds_total"]
    receipt["seconds_by_svd"] = dict(receipt["seconds_by_svd"], delta=time.perf_counter() - before_delta)
    receipt.update(svd_count=3, seconds_total=time.perf_counter() - started,
                   additional_black_forwards_for_delta=0, v1_output_mutated=False,
                   delta_source="the exact two outputs of v1's single black forward")
    return bases, receipt


@torch.inference_mode()
def predict(q, r, cov, score, *, device="cpu", extras=None):
    """Complete raw FoRIS score; finalize with the verified v1 finalizer."""
    required = ("host", "q_layers", "r_layers", "ref_mask", "query_tensor",
                "layer_metadata", "arm_bases")
    if extras is None or any(key not in extras for key in required):
        raise ValueError("Delta control requires the same real inputs as v1")
    if extras["layer_metadata"].get("space") != "encoder_intermediate_norm_true":
        raise ValueError("Delta control cannot use debiased final-layer cache substitutes")
    host = extras["host"]
    if torch.device(device) != host.device:
        raise ValueError("Reuse the existing host device")
    shape = tuple(np.shape(score))
    if len(shape) != 2 or np.prod(shape) != np.shape(q)[0]:
        raise ValueError("Score shape must specify the query grid")
    if np.prod(np.shape(cov)) != np.shape(r)[0]:
        raise ValueError("Reference mask/token count mismatch")
    started = time.perf_counter()
    qt = delta_embedding(extras["q_layers"], device=device)
    rt = delta_embedding(extras["r_layers"], device=device)
    basis = torch.as_tensor(extras["arm_bases"]["delta"], device=device, dtype=torch.float32)
    if basis.shape != (qt.shape[1], CONFIG["positional_rank"]):
        raise ValueError("Delta basis has the wrong representation or rank")
    raw, did_debias = multilayer._readout(
        host, qt, rt, extras["ref_mask"], shape, basis, extras["query_tensor"])
    if not torch.isfinite(raw).all():
        raise ValueError("Delta complete readout produced nonfinite scores")
    field = raw.float().cpu().numpy().astype(np.float32)
    return field, {
        "method": CONFIG["name"], "arm": "delta_only", "config": CONFIG,
        "parts_1_through_4_completed": True, "native_finalization_pending": True,
        "native_rgb_xy_query_state_restored": True, "debias_applied": did_debias,
        "representation_channels": int(qt.shape[1]), "positional_rank": int(basis.shape[1]),
        "seconds_readout": time.perf_counter() - started,
        "query_gt_used": False, "layers_source": dict(extras["layer_metadata"]),
    }
