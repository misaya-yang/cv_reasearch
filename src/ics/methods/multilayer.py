"""Frozen layer-transition representation with the complete FoRIS readout.

No classifier is fitted and no query labels are accepted. ``predict`` and
``control`` share the exact FoRIS parts 1--4; ``finalize`` runs its native
binarizer/CRF. The required per-arm black-image bases must be prepared once
with layer_extract.prepare_bases, using the same already-loaded encoder.
"""
from contextlib import contextmanager
import math
import time

import numpy as np
import torch
import torch.nn.functional as F


CONFIG = {
    "name": "layer_transition_foris",
    "layers_one_based": [16, 24],
    "main": "concat(unit_l24, unit(unit_l24-unit_l16))/sqrt(2)",
    "control": "concat(unit_l24, unit_l16)/sqrt(2)",
    "positional_rank": 500,
    "readout": "source FoRIS parts 1,2,3,4; native binarizer and CRF",
    "field_semantics": "raw FoRIS part-4 score; use finalize, never cut directly at 0.5",
    "new_pair_forwards": 1,
    "black_forwards_once": 1,
    "query_gt_used": False,
}


def embedding(layers, arm, *, device):
    """Return N x 2D descriptors; require aligned, actual layer tensors."""
    if arm not in ("transition", "concat"):
        raise ValueError(f"Unknown representation arm: {arm}")
    values = []
    for key in ("16", "24"):
        if key not in layers:
            raise ValueError(f"Missing true layer {key}; no last-layer fallback")
        x = torch.as_tensor(layers[key], device=device, dtype=torch.float32)
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError("Layer tensors must be finite N x D matrices")
        values.append(F.normalize(x, dim=1))
    early, late = values
    if early.shape != late.shape:
        raise ValueError("Layer tokens must share the same image grid and channels")
    second = F.normalize(late - early, dim=1) if arm == "transition" else early
    return torch.cat((late, second), dim=1) / math.sqrt(2)


@contextmanager
def _readout_scope(host, basis, query_tensor):
    # Native part 3 reads _tgt_image for RGB/xy clustering. Extraction clears
    # episode state, so restore the actual transformed image for this arm.
    # Restore all touched state even if a native stage raises.
    previous = host.positional_basis
    previous_decision = host.should_debiass
    previous_target = host._tgt_image
    host.positional_basis = basis
    host._tgt_image = query_tensor
    try:
        yield
    finally:
        host.positional_basis = previous
        host.should_debiass = previous_decision
        host._tgt_image = previous_target


def _readout(host, q, r, mask, shape, basis, query_tensor):
    h, w = shape
    if q.shape != r.shape or q.shape[0] != h * w:
        raise ValueError("FoRIS requires aligned square-resized reference/query grids")
    if basis.ndim != 2 or basis.shape[0] != q.shape[1]:
        raise ValueError("The positional basis belongs to a different representation")
    mask = torch.as_tensor(mask, device=q.device).float()
    if mask.ndim != 2:
        raise ValueError("ref_mask must be the full transformed two-dimensional mask")
    query_tensor = torch.as_tensor(query_tensor)
    if query_tensor.ndim == 4 and query_tensor.shape[0] == 1:
        query_tensor = query_tensor[0]
    if query_tensor.ndim != 3 or tuple(query_tensor.shape) != (3, host.image_size, host.image_size):
        raise ValueError("Native clustering requires the transformed 3 x H x W query image")
    if query_tensor.device != q.device:
        raise ValueError("Query RGB and feature tensors must remain on the same host device")
    ref_masks = mask[None, None]
    # Preserve the encoder's token-major/channel-contiguous layout before
    # native channel normalization; needless layout changes can perturb ties.
    fmaps = torch.stack((r, q)).reshape(1, 2, h, w, -1).permute(0, 1, 4, 2, 3)
    fmaps = F.normalize(fmaps, dim=2)
    with _readout_scope(host, basis, query_tensor):
        fmaps = host._part1_positional_debias(fmaps, ref_masks, 1)
        did_debias = bool(host.should_debiass)
        part2 = host._part2_background_suppression(
            fmaps_norm=fmaps, ref_masks=ref_masks, n_refs=1, h=h, w=w)
        if part2 is None:
            raise ValueError("No reference foreground survives native nearest mask resize")
        score, sf, sbn, mu_fg, tgt_feat_denoised = part2
        score, cand_soft, seed_prior = host._part3_clustering(
            score, sf=sf, mu_fg=mu_fg, ref_feats_raw=fmaps[:, :1],
            tgt_feat_raw=fmaps[:, 1], ref_masks=ref_masks, n_refs=1, h=h, w=w)
        score = host._part4_semantic_consistency_correction(
            score, sf=sf, sbn=sbn, cand_soft=cand_soft,
            seed_prior=seed_prior, tgt_feat=tgt_feat_denoised)
    return score, did_debias


@torch.inference_mode()
def _run(q, r, cov, score, *, device, extras, arm):
    started = time.perf_counter()
    required = ("host", "q_layers", "r_layers", "ref_mask", "query_tensor", "layer_metadata")
    if extras is None or any(k not in extras for k in required):
        raise ValueError(f"Actual multi-layer inputs required: {required}")
    meta = extras["layer_metadata"]
    if meta.get("space") != "encoder_intermediate_norm_true":
        raise ValueError("Do not substitute debiased final caches for raw intermediate layers")
    shape = tuple(np.shape(score))
    if len(shape) != 2 or np.shape(q)[0] != math.prod(shape):
        raise ValueError("score must identify the query token grid")
    if math.prod(np.shape(cov)) != np.shape(r)[0]:
        raise ValueError("Reference coverage and reference token count differ")
    host = extras["host"]
    if torch.device(device) != host.device:
        raise ValueError("Reuse the host on its existing device; do not copy its model")
    if arm == "native":
        # The public host normalizes exactly once, below in _readout.
        qt = torch.as_tensor(extras["q_layers"]["24"], device=device).float()
        rt = torch.as_tensor(extras["r_layers"]["24"], device=device).float()
        basis = host.positional_basis
    else:
        qt = embedding(extras["q_layers"], arm, device=device)
        rt = embedding(extras["r_layers"], arm, device=device)
        if "arm_bases" not in extras or arm not in extras["arm_bases"]:
            raise ValueError("Prepare representation-specific black bases before inference")
        basis = extras["arm_bases"][arm]
    basis = torch.as_tensor(basis, device=device, dtype=torch.float32)
    if basis.shape[1] != CONFIG["positional_rank"]:
        raise ValueError("This locked comparison requires rank 500 for every arm")
    raw, did_debias = _readout(host, qt, rt, extras["ref_mask"], shape, basis,
                              extras["query_tensor"])
    raw = raw.float()
    if not torch.isfinite(raw).all():
        raise ValueError("The complete FoRIS readout produced nonfinite scores")
    # Preserve the raw score. Pre-normalizing here and normalizing again in
    # the native binarizer changes degenerate spans below its epsilon floor.
    field = raw.cpu().numpy().astype(np.float32)
    return field, {
        "method": CONFIG["name"], "arm": arm, "parts_1_through_4_completed": True,
        "native_finalization_pending": True, "debias_applied": did_debias,
        "native_rgb_xy_query_state_restored": True,
        "representation_channels": int(qt.shape[1]), "positional_rank": int(basis.shape[1]),
        "seconds_readout": time.perf_counter() - started,
        "layers_one_based": [24] if arm == "native" else [16, 24],
        "query_gt_used": False, "layers_source": dict(meta),
    }


def predict(q, r, cov, score, *, device="cpu", extras=None):
    return _run(q, r, cov, score, device=device, extras=extras, arm="transition")


def control(q, r, cov, score, *, device="cpu", extras=None):
    return _run(q, r, cov, score, device=device, extras=extras, arm="concat")


def native(q, r, cov, score, *, device="cpu", extras=None):
    """Native feature/readout control from the same actual pair forward."""
    return _run(q, r, cov, score, device=device, extras=extras, arm="native")


@torch.inference_mode()
def finalize(field, extras):
    """Finish an arm at model size with the source FoRIS binarizer and CRF."""
    host = extras["host"]
    if host.mask_refiner != "crf":
        raise ValueError("The native comparison requires the same FoRIS CRF")
    if host.resize_to_orig_size:
        raise ValueError("This experiment evaluates 1024 native masks; set resize_to_orig_size=False")
    query = extras["query_tensor"]
    if query.ndim == 3:
        query = query[None]
    if tuple(query.shape[-2:]) != (1024, 1024):
        raise ValueError("Expected the native transformed 1024 x 1024 query image")
    raw = torch.as_tensor(field, device=host.device, dtype=torch.float32)
    binary = host._binarize_response(raw, target_hw=(1024, 1024))
    return host._finalize_mask(binary, query).bool().cpu().numpy()


def run_episode(host, episode, manifest, bases, *, verify_native=False):
    """Shared-runner entry; reads only the support annotation, never query GT.

    ``bases`` is prepare_bases(host)[0]. The caller owns cohort-level sealing,
    SHA receipts, native replay checks and a separate subsequent scoring step.
    ``verify_native=True`` on the first episode additionally verifies that the
    layer-24 tap is exactly the native extraction; its cost is recorded.
    """
    from pathlib import Path
    from PIL import Image
    from .layer_extract import extract_pair

    data = Path(manifest["data_root"])
    annotations = Path(manifest["annotation_root"])
    with Image.open(data / episode["support"]) as opened:
        reference_image = opened.convert("RGB")
    with Image.open(data / episode["query"]) as opened:
        query_image = opened.convert("RGB")
    support_label = annotations / Path(episode["support"]).with_suffix(".png")
    with Image.open(support_label) as opened:
        reference_mask = torch.from_numpy((np.asarray(opened) == episode["c"] + 1).copy())
    extras = extract_pair(host, reference_image, reference_mask, query_image,
                          verify_native=verify_native)
    extras["arm_bases"] = bases
    q, r = extras["q_layers"]["24"], extras["r_layers"]["24"]
    h = w = host.image_size // 16
    dummy_score = np.zeros((h, w), np.float32)
    masks, metadata = {}, {"extraction": extras["layer_metadata"]}
    for name, function in (("native", native), ("multilayer", predict),
                           ("multilayer.control", control)):
        field, info = function(q, r, extras["cov"], dummy_score,
                               device=str(host.device), extras=extras)
        finalize_start = time.perf_counter()
        masks[name] = finalize(field, extras)
        info["native_finalization_pending"] = False
        info["seconds_finalize_and_cpu_copy"] = time.perf_counter() - finalize_start
        info["output_shape"] = list(masks[name].shape)
        metadata[name] = info
    return masks, metadata
