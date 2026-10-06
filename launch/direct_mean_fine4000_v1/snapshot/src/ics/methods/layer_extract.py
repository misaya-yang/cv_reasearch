"""One shared frozen encoder, one paired forward, two layer taps.

This module never constructs a model, downloads an asset, or starts a job.
Use after the resource check. Call sequentially outside intervention attention
hooks; do not describe the intervened activations as this method's inputs.
"""
import time

import torch
import torch.nn.functional as F

from .multilayer import embedding


def _layers(encoder, images):
    maps = encoder.get_intermediate_layers(images, n=[15, 23], reshape=True)
    if len(maps) != 2 or any(m.ndim != 4 for m in maps):
        raise ValueError("Encoder must return actual NCHW block 16 and 24 features")
    if maps[0].shape != maps[1].shape:
        raise ValueError("Layer shapes differ; never interpolate across token grids")
    return {str(layer): fmap.permute(0, 2, 3, 1).reshape(fmap.shape[0], -1, fmap.shape[1])
            for layer, fmap in zip((16, 24), maps)}


@torch.inference_mode()
def extract_pair(host, reference_image, reference_mask, query_image, *, verify_native=False):
    """Use native image/mask transforms and reference-first pair ordering."""
    if host._ref_images is not None or host._tgt_image is not None:
        raise ValueError("Host already has an active episode; refusing to overwrite it")
    started = time.perf_counter()
    try:
        host.set_reference(reference_image, reference_mask)
        host.set_target(query_image)
        if host._ref_images.shape[0] != 1:
            raise ValueError("Exactly one labeled reference is permitted")
        images = torch.cat((host._ref_images, host._tgt_image[None]), dim=0)
        layers = _layers(host.encoder, images)
        native_error = None
        if verify_native:
            # Run once for the first real episode. The additional pair forward
            # is a replay-fidelity check, not a candidate resource advantage.
            direct = host.encoder.get_intermediate_layers(images, n=1, reshape=True)[0]
            direct = direct.permute(0, 2, 3, 1).reshape(direct.shape[0], -1, direct.shape[1])
            native_error = float((direct.float() - layers["24"].float()).abs().max())
            if not torch.equal(direct, layers["24"]):
                raise ValueError(f"Layer tap does not reproduce the native extraction: max error {native_error}")
        h = w = host.image_size // 16
        if layers["24"].shape[1] != h * w:
            raise ValueError("Unexpected DINOv3 patch grid")
        # CPU copies synchronize the feature pass for honest wall timing.
        q_layers = {key: val[1].float().cpu() for key, val in layers.items()}
        r_layers = {key: val[0].float().cpu() for key, val in layers.items()}
        ref_mask = host._ref_masks[0].clone()
        return {
            "host": host, "q_layers": q_layers, "r_layers": r_layers,
            "ref_mask": ref_mask, "query_tensor": host._tgt_image[None].clone(),
            "cov": F.interpolate(ref_mask[None, None].float(), (h, w), mode="area")[0, 0].cpu().numpy(),
            "layer_metadata": {
                "space": "encoder_intermediate_norm_true", "layers_one_based": [16, 24],
                "indices_zero_based": [15, 23], "batch_order": ["reference", "query"],
                "pair_forwards": 1 + int(verify_native), "intervention": False,
                "native_tap_max_abs_error": native_error,
                "seconds_feature_forward_and_cpu_copy": time.perf_counter() - started,
            },
        }
    finally:
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


@torch.inference_mode()
def prepare_bases(host, rank=500):
    """One shared normalized-black forward; two fixed arm-specific SVDs."""
    if rank != 500:
        raise ValueError("The predeclared experiment uses native rank 500")
    started = time.perf_counter()
    black = torch.zeros(1, 3, host.image_size, host.image_size, device=host.device)
    mean = black.new_tensor([.485, .456, .406])[None, :, None, None]
    std = black.new_tensor([.229, .224, .225])[None, :, None, None]
    layers = {key: val[0] for key, val in _layers(host.encoder, (black - mean) / std).items()}
    bases, times = {}, {}
    for arm in ("transition", "concat"):
        arm_started = time.perf_counter()
        x = embedding(layers, arm, device=host.device).T
        x = F.normalize(x, dim=0)
        x = x - x.mean(dim=1, keepdim=True)
        if min(x.shape) < rank:
            raise ValueError("Rank exceeds the black-image representation matrix")
        u, _, _ = torch.linalg.svd(x, full_matrices=False)
        bases[arm] = u[:, :rank].contiguous().cpu()
        times[arm] = time.perf_counter() - arm_started
    return bases, {
        "source_input": "normalized_black_image", "rank": rank,
        "black_forwards": 1, "svd_count": 2, "seconds_by_svd": times,
        "seconds_total": time.perf_counter() - started,
        "no_additional_natural_images": True,
    }
