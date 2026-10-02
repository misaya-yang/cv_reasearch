"""Two frozen-encoder input interventions; no query annotation interface.

Background counterfactuals alter ONLY annotated support-background RGB.
Lattice-origin views shift RGB before patchification, then geometrically align
full maps back to the original lattice. This is NOT a RoPE phase intervention
and does not assert that inverse sampling inverts a contextual encoder.

Density/matching and native query geometry remain caller-owned. The encoder
adapter receives RGB only, and all view-choice supervision is lawful support
labels in fixed disjoint spatial holdouts. Neither adapter loads any weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ViewBank:
    names: tuple[str, ...]
    rgb: tuple[Tensor, ...]                  # each [B,3,H,W]
    maps: tuple[Tensor, ...]                 # aligned full [B,C,Hf,Wf]
    valid: tuple[Tensor, ...]                # [B,1,Hf,Wf], interpolation interior
    offsets: tuple[tuple[int, int], ...]     # pixel (dy,dx), I_view(u)=I(u+offset)
    image_shape: tuple[int, int]
    metadata: dict


@dataclass
class SupportViewWeights:
    weights: Tensor                         # [B,V], frozen for query decoding
    uniform_weights: Tensor                 # same views/calls/budget [B,V]
    heldout_brier: Tensor                    # [B,V,R], nan when ineligible
    eligible_regions: Tensor                # [B,R]
    valid: Tensor                           # [B]
    reasons: tuple[str, ...]
    metadata: dict


class FrozenEncoderAdapter:
    """Callable RGB[B,3,H,W] -> full feature[B,C,Hf,Wf], eval only.

    No annotation or candidate is supplied to the encoder. Inference-mode
    suppresses autograd but does not change the caller's parameter flags.
    A module in train mode is rejected rather than silently changing state.
    Input copies isolate the stored RGB from an in-place caller callback.
    """
    def __init__(self, encode: Callable[[Tensor], Tensor]):
        self.encode = encode

    def __call__(self, image: Tensor) -> Tensor:
        _image(image)
        if isinstance(self.encode, nn.Module) and any(m.training for m in self.encode.modules()):
            raise ValueError('Frozen encoder must already be in eval mode')
        with torch.inference_mode():
            output = self.encode(image.clone())
        if not isinstance(output, Tensor) or output.ndim != 4 or output.shape[0] != image.shape[0]:
            raise ValueError('Encoder must return a full BCHW tensor')
        if not output.is_floating_point() or not torch.isfinite(output).all():
            raise ValueError('Encoder returned nonfinite/nonfloating full maps')
        return output.detach().clone()


def _image(image: Tensor):
    if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
        raise ValueError('RGB must be floating [B,3,H,W]')
    if not torch.isfinite(image).all():
        raise ValueError('RGB must be finite')


def _mask(mask: Tensor, image: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape != (image.shape[0], 1, *image.shape[-2:]):
        raise ValueError('Support annotation must be [B,1,H,W] or [B,H,W]')
    if mask.dtype != torch.bool:
        raise ValueError('Supply an explicitly binarized legal support annotation')
    return mask.to(image.device)


def support_background_views(image: Tensor, support_fg: Tensor,
                             encoder: Callable[[Tensor], Tensor], *,
                             protected_pixels: Tensor | None = None) -> ViewBank:
    """Exactly 3 deterministic support views: original, mean, BG permutation.

    Mean is the original image's per-channel global RGB mean, not an external
    image. Rearrangement cyclically permutes ONLY original BG positions by
    floor(N_BG/2); foreground RGB is bitwise preserved in every view.
    Requires each support to have both FG and at least two BG pixels.
    """
    _image(image)
    fg = _mask(support_fg, image)
    protected = torch.zeros_like(fg) if protected_pixels is None else _mask(protected_pixels, image)
    bg = ~fg & ~protected
    if (~fg.flatten(1).any(1)).any() or (bg.flatten(1).sum(1) < 2).any():
        raise ValueError('Background intervention requires nonempty FG and >=2 BG pixels')
    original = image.clone()
    mean = image.mean((-2, -1), keepdim=True)
    flat = image.flatten(2)
    rearranged = flat.clone()
    for b in range(image.shape[0]):
        ids = torch.where(bg[b, 0].flatten())[0]
        rearranged[b, :, ids] = flat[b, :, ids.roll(int(ids.numel() // 2))]
    rearranged = rearranged.reshape_as(image)
    mean_view = torch.where(fg | protected, original, mean)
    rgb = (original, mean_view, rearranged)
    for value in rgb:
        if not torch.equal(value[fg.expand_as(image)], image[fg.expand_as(image)]):
            raise RuntimeError('Foreground RGB changed')
    adapter = encoder if isinstance(encoder, FrozenEncoderAdapter) else FrozenEncoderAdapter(encoder)
    maps = tuple(adapter(value) for value in rgb)
    if any(value.shape != maps[0].shape for value in maps):
        raise ValueError('All support views must return the same full-map shape')
    valid = tuple(torch.ones((image.shape[0], 1, *m.shape[-2:]), dtype=torch.bool,
                             device=m.device) for m in maps)
    return ViewBank(('original', 'global_mean_background', 'permuted_original_background'),
                    rgb, maps, valid, ((0, 0),) * 3, tuple(image.shape[-2:]),
                    dict(method='support_background_counterfactual', fg_rgb_bit_exact=True,
                         external_images=False, query_labels_used=False, encoder_calls=3,
                         heldout_pixels_unmodified=protected_pixels is not None))


def background_holdout_factory(image: Tensor, support_fg: Tensor,
                               encoder: Callable[[Tensor], Tensor]) -> Callable[[Tensor], ViewBank]:
    """Re-encode counterfactuals while ALL held pixels remain original RGB.

    Otherwise a foreground-preserving renderer would expose held annotations
    in its pixels, invalidating support-label cross-fitting. Region boundaries
    here coincide with native patch boundaries (nearest upsampling only).
    """
    def make(held_feature_cells: Tensor) -> ViewBank:
        if held_feature_cells.ndim != 4 or held_feature_cells.shape[:2] != (image.shape[0], 1):
            raise ValueError('Held cells must be B1HfWf')
        hf, wf = held_feature_cells.shape[-2:]
        h, w = image.shape[-2:]
        if h % hf or w % wf:
            raise ValueError('Cross-fit renderer needs exact native patch-to-pixel regions')
        protected = F.interpolate(held_feature_cells.float(), size=(h, w), mode='nearest').bool()
        return support_background_views(image, support_fg, encoder, protected_pixels=protected)
    return make


def shift_rgb(image: Tensor, offset: tuple[int, int]) -> Tensor:
    """I_s[u,v]=reflect(I)[u+dy,v+dx]; same size, no interpolation."""
    _image(image)
    dy, dx = offset
    h, w = image.shape[-2:]
    if not (0 <= dy < h and 0 <= dx < w):
        raise ValueError('Nonnegative offsets must be smaller than image dimensions')
    if offset == (0, 0):
        return image.clone()
    padded = F.pad(image, (0, dx, 0, dy), mode='reflect')
    return padded[..., dy:dy + h, dx:dx + w].clone()


def shift_support_annotation(support_fg: Tensor, offset: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Forward-align lawful support labels; reflected RGB receives NO label.

    Returns shifted FG and label-valid masks at image resolution. Caller must
    exclude invalid reflected pixels from both FG and BG density fitting.
    """
    if support_fg.ndim != 4 or support_fg.shape[1] != 1 or support_fg.dtype != torch.bool:
        raise ValueError('Support FG must be bool [B,1,H,W]')
    dy, dx = offset
    h, w = support_fg.shape[-2:]
    if not (0 <= dy < h and 0 <= dx < w):
        raise ValueError('Offset outside support image')
    shifted = F.pad(support_fg, (0, dx, 0, dy), value=False)[..., dy:dy+h, dx:dx+w]
    valid = torch.zeros_like(support_fg)
    valid[..., :h-dy, :w-dx] = True
    return shifted, valid


def inverse_coordinates(shape: tuple[int, int], offset: tuple[int, int], *, device=None,
                        dtype=torch.float64) -> tuple[Tensor, Tensor]:
    """Original pixel(y,x) -> shifted pixel(y-dy,x-dx), exact translation.

    Returns [H,W,2] (y,x), and true-original-input visibility [H,W].
    This coordinate identity is exact; feature interpolation is not an
    algebraic inverse of patchification/self-attention.
    """
    h, w = shape
    dy, dx = offset
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing='ij')
    coords = torch.stack((yy - dy, xx - dx), -1)
    valid = (coords[..., 0] >= 0) & (coords[..., 1] >= 0)
    valid &= (coords[..., 0] < h-dy) & (coords[..., 1] < w-dx)
    return coords, valid


def align_map_to_original(feature: Tensor, image_shape: tuple[int, int],
                          offset: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Back-sample view tokens at original token centers (align_corners=False).

    For token row i, u=(i+.5)*H/Hf-.5-dy and normalized grid coordinate
    2*(u+.5)/H-1. Valid excludes missing original pixels AND out-of-map token
    interpolation. Invalid outputs are zero; contextual effects of reflected
    RGB can still reach valid tokens through attention, and are not hidden.
    """
    if feature.ndim != 4 or not feature.is_floating_point():
        raise ValueError('Feature must be floating BCHW')
    b, _, hf, wf = feature.shape
    h, w = image_shape
    dy, dx = offset
    if offset == (0, 0):
        return feature.clone(), torch.ones((b, 1, hf, wf), dtype=torch.bool, device=feature.device)
    yy = (torch.arange(hf, device=feature.device, dtype=torch.float64)+.5)*h/hf-.5-dy
    xx = (torch.arange(wf, device=feature.device, dtype=torch.float64)+.5)*w/wf-.5-dx
    gy, gx = torch.meshgrid(2*(yy+.5)/h-1, 2*(xx+.5)/w-1, indexing='ij')
    grid = torch.stack((gx, gy), -1).to(feature.dtype)[None].expand(b, -1, -1, -1)
    ph, pw = h/hf, w/wf
    # Require every interpolated token's entire local input patch to be real
    # original RGB. Attention may still communicate reflected tokens globally.
    last_y = ((h-dy)//ph-1+.5)*ph-.5
    last_x = ((w-dx)//pw-1+.5)*pw-.5
    interior = ((yy >= (ph-1)/2) & (yy <= last_y))[:, None]
    interior = interior & ((xx >= (pw-1)/2) & (xx <= last_x))[None, :]
    valid = interior[None, None].expand(b, 1, hf, wf)
    output = F.grid_sample(feature, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return output.masked_fill(~valid, 0), valid


def patch_lattice_views(image: Tensor, encoder: Callable[[Tensor], Tensor], *,
                        patch_size: int = 16) -> ViewBank:
    """Fixed four input-lattice origins for a patch16 encoder, support OR query.

    No annotation is needed to make query views. Shifts are pixels, not tokens:
    (0,0),(8,0),(0,8),(8,8). Encoder must return the original native patch grid.
    """
    _image(image)
    h, w = image.shape[-2:]
    if patch_size != 16 or h % 16 or w % 16 or min(h, w) <= 8:
        raise ValueError('Frozen lattice card requires patch16 and divisible image dimensions')
    offsets = ((0, 0), (8, 0), (0, 8), (8, 8))
    adapter = encoder if isinstance(encoder, FrozenEncoderAdapter) else FrozenEncoderAdapter(encoder)
    rgb = tuple(shift_rgb(image, offset) for offset in offsets)
    maps, valid = [], []
    for value, offset in zip(rgb, offsets):
        feature = adapter(value)
        if feature.shape[-2:] != (h//16, w//16):
            raise ValueError('Native encoder feature grid differs from declared patch16 grid')
        aligned, mask = align_map_to_original(feature, (h, w), offset)
        maps.append(aligned)
        valid.append(mask)
    return ViewBank(tuple(f'origin_{y}_{x}' for y, x in offsets), rgb, tuple(maps),
                    tuple(valid), offsets, (h, w),
                    dict(method='pixel_patch_lattice_origins', encoder_calls=4, patch_size=16,
                         rgb_shift_exact=True, coordinate_inverse_exact=True,
                         feature_inverse_is_interpolation=True, rope_phase_shift=False,
                         query_labels_used=False, reflected_labels_excluded=True))


def spatial_holdout_regions(shape: tuple[int, int], *, device=None) -> Tensor:
    """Four fixed disjoint spatial quadrants, selected without any annotation."""
    h, w = shape
    if min(h, w) < 2:
        raise ValueError('Need >=2 cells per axis for fixed holdouts')
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    return (yy >= h//2).long()*2 + (xx >= w//2).long()


def fit_support_view_weights(bank: ViewBank, support_fg: Tensor,
                             predict_leave_region_out: Callable, *, regions: Tensor | None = None,
                             region_view_factory: Callable[[Tensor], ViewBank] | None = None,
                             min_train_anchors: int = 1
                             ) -> SupportViewWeights:
    """Fit per-support view weights using ONLY disjoint support-label holdouts.

    Callback(feature BCHW, train_fg B1HW, train_bg B1HW, held_valid B1HW)
    -> heldout probability B1HW. Labels are on the aligned FEATURE grid and
    must be legally downsampled by the caller using the unchanged density
    recipe. Held labels are NEVER passed to this callback. For background
    interventions, region_view_factory MUST rerender with held pixels fully
    protected, avoiding an annotation leak through counterfactual RGB itself.
    Use background_holdout_factory; this adds 3*R frozen encoder calls. The
    same-budget uniform control shares all these calibration calls. It may read full
    RGB-derived features, so this is spatial label holdout, not independent
    image generalization. Each eligible region needs held/train FG and BG.

    Predeclared weight rule: softmax(-mean balanced Brier / 0.1), same four
    eligible holdouts for every view. No hyperparameter search/query fitting.
    Ineligible cases explicitly return uniform weights with valid=False.
    """
    first = bank.maps[0]
    if min_train_anchors < 1:
        raise ValueError('Minimum training anchor count must be positive')
    b, _, h, w = first.shape
    if support_fg.ndim == 3:
        support_fg = support_fg[:, None]
    if support_fg.shape != (b, 1, h, w) or support_fg.dtype != torch.bool:
        raise ValueError('Weights require bool lawful support label on feature grid')
    fg = support_fg.to(first.device)
    is_background = bank.metadata['method'] == 'support_background_counterfactual'
    if is_background and region_view_factory is None:
        raise ValueError('Background view scoring needs protected-region re-encoding; full-mask views leak held labels')
    regions = spatial_holdout_regions((h, w), device=first.device) if regions is None else regions.to(first.device)
    if regions.shape != (h, w) or regions.dtype not in (torch.int32, torch.int64):
        raise ValueError('Regions must be a fixed integer HW partition')
    ids = torch.unique(regions, sorted=True)
    if len(ids) < 2 or (ids < 0).any():
        raise ValueError('Need >=2 disjoint holdout regions; no unlabeled region IDs')
    common = torch.stack([v.to(first.device) for v in bank.valid]).all(0)
    losses = torch.full((b, len(bank.maps), len(ids)), float('nan'), device=first.device)
    eligible = torch.zeros((b, len(ids)), dtype=torch.bool, device=first.device)
    for ri, rid in enumerate(ids):
        held = common & (regions == rid)[None, None]
        train = common & ~held
        train_fg, train_bg = train & fg, train & ~fg
        held_fg, held_bg = held & fg, held & ~fg
        good = (train_fg.flatten(1).sum(1) >= min_train_anchors) & (train_bg.flatten(1).sum(1) >= min_train_anchors)
        good &= held_fg.flatten(1).any(1) & held_bg.flatten(1).any(1)
        eligible[:, ri] = good
        if not good.any():
            continue
        scored = bank if region_view_factory is None else region_view_factory(held.clone())
        if scored.names != bank.names or any(m.shape != first.shape for m in scored.maps):
            raise ValueError('Cross-fit maps must preserve the frozen view bank interface')
        if is_background and not scored.metadata.get('heldout_pixels_unmodified', False):
            raise ValueError('Background cross-fit did not protect held RGB')
        for vi, feature in enumerate(scored.maps):
            # Copies prevent accidental label mutation by a recipe adapter.
            prediction = predict_leave_region_out(feature, train_fg.clone(), train_bg.clone(), held.clone())
            if not isinstance(prediction, Tensor) or prediction.shape != fg.shape:
                raise ValueError('Leave-region-out recipe must return B1HW probability')
            prediction = prediction.to(first.device)
            if not torch.isfinite(prediction[held]).all() or ((prediction[held] < 0) | (prediction[held] > 1)).any():
                raise ValueError('Invalid heldout probability; do not silently fit weights')
            err = (prediction.float()-fg.float()).square()
            pos = (err*held_fg).flatten(1).sum(1)/held_fg.flatten(1).sum(1).clamp_min(1)
            neg = (err*held_bg).flatten(1).sum(1)/held_bg.flatten(1).sum(1).clamp_min(1)
            losses[good, vi, ri] = ((pos+neg)/2)[good]
    valid = eligible.sum(1) >= 2
    uniform = torch.full((b, len(bank.maps)), 1/len(bank.maps), device=first.device)
    weights = uniform.clone()
    mean = losses.nanmean(2)
    weights[valid] = torch.softmax(-mean[valid]/.1, 1)
    return SupportViewWeights(weights.detach(), uniform, losses.detach(), eligible, valid,
                              tuple('ok' if item else 'fewer_than_two_legal_balanced_holdouts' for item in valid.tolist()),
                              dict(query_labels_used=False, rule='softmax(-balanced_brier/0.1)',
                                   fitted_scope='support_annotation_spatial_label_holdout',
                                   independent_images=False, same_budget_uniform_control=True,
                                   region_ids=ids.tolist(), encoder_calls=bank.metadata['encoder_calls'],
                                   min_train_anchors=min_train_anchors,
                                   calibration_encoder_calls=(int(eligible.any(0).sum())*len(bank.maps)
                                                              if region_view_factory is not None else 0),
                                   background_held_pixels_protected=is_background))


def combine_view_outputs(outputs: tuple[Tensor, ...], bank: ViewBank,
                         weights: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Weighted aligned outputs with per-cell valid normalization.

    Can combine caller-computed density probabilities; does not change density
    estimation or native geometric decoding. No valid evidence => zero+False,
    which the caller must explicitly handle (not call it semantic background).
    """
    if len(outputs) != len(bank.maps) or any(o.shape != outputs[0].shape for o in outputs):
        raise ValueError('One equal-shape full output per view is required')
    b = outputs[0].shape[0]
    if outputs[0].ndim != 4 or outputs[0].shape[-2:] != bank.maps[0].shape[-2:]:
        raise ValueError('Outputs must be aligned full BCHW maps')
    weights = torch.full((b, len(outputs)), 1/len(outputs), device=outputs[0].device) if weights is None else weights.to(outputs[0].device)
    if weights.shape != (b, len(outputs)) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Weights must be finite nonnegative BV')
    valid = torch.stack([v.to(outputs[0].device) for v in bank.valid], 1)
    weighted = valid*weights[:, :, None, None, None]
    denominator = weighted.sum(1)
    output = (torch.stack(outputs, 1)*weighted).sum(1)/denominator.clamp_min(1e-12)
    return output.masked_fill(denominator == 0, 0), denominator > 0
