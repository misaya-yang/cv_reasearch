"""Fixed whole-R/Q grayscale DINO patch matching; bounded complete MEAN edit.

Four physical128 RGB/gray forwards per pair, no CLS, removal, ROI selection,
training, GT-dependent parameters or claims of semantic color invariance.
"""
from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np
from PIL import Image

from .object_crop_cls import unit, render, reference_coverage


@dataclass(frozen=True)
class Config:
    view_side: int = 128
    neutral_rgb: tuple = (124, 116, 104)
    temperature: float = .07
    residual_weight: float = .05


def grayscale(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Original uint8 RGB required')
    value = rgb.astype(np.uint32)
    luminance = ((299*value[..., 0]+587*value[..., 1]+114*value[..., 2]+500)//1000).astype(np.uint8)
    return np.repeat(luminance[..., None], 3, axis=-1)


def views(rgb, reference_mask=None, cfg=Config()):
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    scale = cfg.view_side/max(h, w)
    sh, sw = max(1, round(h*scale)), max(1, round(w*scale))
    oy, ox = (cfg.view_side-sh)//2, (cfg.view_side-sw)//2
    pair = {}
    for name, image in (('rgb', rgb), ('gray', grayscale(rgb))):
        canvas = np.empty((128, 128, 3), np.uint8);canvas[:] = cfg.neutral_rgb
        canvas[oy:oy+sh, ox:ox+sw] = np.asarray(Image.fromarray(image).resize((sw, sh), Image.Resampling.BILINEAR))
        pair[name] = canvas
    valid = np.zeros((128, 128), np.float64);valid[oy:oy+sh, ox:ox+sw] = 1
    fg = None
    if reference_mask is not None:
        mask = np.asarray(reference_mask, bool)
        if mask.shape != (h, w):
            raise ValueError('Aligned full original reference mask required')
        fg = np.zeros((128, 128), np.float64)
        fg[oy:oy+sh, ox:ox+sw] = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
            (sw, sh), Image.Resampling.NEAREST), np.float64)
    geometry = dict(original_hw=[h, w], resized_hw=[sh, sw], padding_top_left=[oy, ox],
        physical_aspect_preserved=True, RGB_source_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        gray_physical_transform='uint8 floor((299R+587G+114B+500)/1000), repeated3channels, BEFORE resize',
        gray_padding='same neutral(124,116,104) as RGB; only physical image pixels are grayscale',
        patch_grid=[8, 8], native_work_grid=[64, 64], source_crop='entire original image')
    weights = dict(valid=valid.reshape(8, 16, 8, 16).mean(axis=(1, 3)).ravel(),
                   foreground=None if fg is None else fg.reshape(8, 16, 8, 16).mean(axis=(1, 3)).ravel())
    return pair, weights, geometry


def brightness_texture(rgb_view):
    """Fixed unit5 descriptor [1,meanL,stdL,mean|dxL|,mean|dyL|] per physical patch."""
    value = np.asarray(rgb_view, np.float64)/255
    luma = .299*value[..., 0]+.587*value[..., 1]+.114*value[..., 2]
    cells = luma.reshape(8, 16, 8, 16).transpose(0, 2, 1, 3).reshape(64, 16, 16)
    return unit(np.stack((np.ones(64), cells.mean((1, 2)), cells.std((1, 2)),
        np.abs(np.diff(cells, axis=2)).mean((1, 2)), np.abs(np.diff(cells, axis=1)).mean((1, 2))), axis=1))


def match_margin(query, reference, foreground_weights, background_weights):
    query, reference = unit(query), unit(reference)
    wf, wb = np.asarray(foreground_weights, np.float64), np.asarray(background_weights, np.float64)
    if wf.shape != (len(reference),) or wb.shape != wf.shape or min(wf.sum(), wb.sum()) <= 0:
        raise ValueError('Both known reference classes require positive physical area')
    fg = unit(np.average(reference, axis=0, weights=wf))
    bg = unit(np.average(reference, axis=0, weights=wb))
    margin = np.einsum('nd,d->n', query, fg-bg, optimize=False)
    return margin, dict(reference_foreground_weight=float(wf.sum()), reference_background_weight=float(wb.sum()),
                        prototype_cosine=float(np.sum(fg*bg)), target_class_semantics_unverified=True)


def map_tokens_to_original_grid(tokens, geometry, shape=(64, 64)):
    """Sample8x8 feature map at original64-cell centers in actual128 canvas.

    canvas_x=padding_x+(j+.5)*resized_width/64, same for y. grid_sample
    align_corners=False thus maps normalized2*canvas_x/128-1 to patch centers.
    RGB/gray/context-controls use identical mapping; padding has no R BG label.
    """
    import torch
    import torch.nn.functional as functional
    tokens = np.asarray(tokens, np.float64)
    if tokens.ndim != 2 or len(tokens) != 64:
        raise ValueError('Exact8x8 physical patch descriptors required')
    sh, sw = geometry['resized_hw'];oy, ox = geometry['padding_top_left']
    x = ox+(np.arange(shape[1])+.5)*sw/shape[1]
    y = oy+(np.arange(shape[0])+.5)*sh/shape[0]
    xx, yy = np.meshgrid(2*x/128-1, 2*y/128-1)
    grid = torch.from_numpy(np.stack((xx, yy), axis=-1))[None]
    tensor = torch.from_numpy(tokens.reshape(8, 8, -1).transpose(2, 0, 1).copy())[None]
    mapped = functional.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return unit(mapped[0].permute(1, 2, 0).numpy().reshape(-1, tokens.shape[1]))


def bounded_field(base, new_margin, native_margin, cfg=Config()):
    base = np.asarray(base, np.float64)
    edit = cfg.residual_weight*(np.tanh(np.asarray(new_margin)/cfg.temperature)-
                               np.tanh(np.asarray(native_margin)/cfg.temperature))
    edit = edit.reshape(base.shape)
    if np.max(np.abs(edit)) > .1+1e-12:
        raise RuntimeError('Fixed absolute .1 complete residual bound violated')
    return base+edit


def predict(reference_rgb, full_reference_mask, query_rgb, q_native, r_native, base, encoder, *, producer_binding, q_h20=None, r_h20=None, cfg=Config()):
    if cfg != Config() or not producer_binding.get('producer'):
        raise ValueError('Fixed recipe and explicit actual/native binding required')
    started = time.perf_counter()
    base = np.asarray(base, np.float64)
    if base.shape != (64, 64):raise ValueError('Same complete64x64 MEAN required')
    rviews, rweights, rgeo = views(reference_rgb, full_reference_mask, cfg)
    qviews, _, qgeo = views(query_rgb, None, cfg)
    wf, wb = rweights['foreground'], rweights['valid']-rweights['foreground']
    if min(wf.sum(), wb.sum()) <= 0:
        kinds=('mean', 'gray', 'rgb', 'brightness_texture')+ (('h20',) if q_h20 is not None and r_h20 is not None else ())
        fields = {name: base.copy() for name in kinds}
        return dict(fields=fields, masks={name: dict(zip(('work', 'original'), render(f, query_rgb.shape[:2]))) for name, f in fields.items()},
            margins={}, info=dict(abstention=True, reason='no_valid_physical128_referenceFG_or_BG',new_encoder_forwards=0,
            independent_new_method_count=1,query_gt_used=False,reference_geometry=rgeo,query_geometry=qgeo,
            h20_control='actual_bound_rawH20' if q_h20 is not None else 'OMITTED_NO_ACTUAL_BOUND_H20; not simulated or MEAN alias'))
    cov = reference_coverage(full_reference_mask).ravel()
    native_margin, native_stat = match_margin(q_native, r_native, cov, 1-cov)
    calls, tokens, margins, stats = [], {}, {}, {}
    for role, physical in (('reference', rviews), ('query', qviews)):
        for kind in ('rgb', 'gray'):
            before = time.perf_counter();value = np.asarray(encoder(physical[kind]), np.float64)
            if value.ndim != 2 or len(value) != 64 or not np.isfinite(value).all():
                raise ValueError('Actual final-LN64patch descriptors required')
            tokens[role+'_'+kind] = unit(value)
            calls.append(dict(role=role,view=kind,seconds=time.perf_counter()-before,view_hw=[128,128],
                view_sha256=hashlib.sha256(physical[kind].tobytes()).hexdigest(),new_image_forwards=1))
    for kind in ('rgb', 'gray'):
        mapped = map_tokens_to_original_grid(tokens['query_'+kind], qgeo)
        margins[kind], stats[kind] = match_margin(mapped, tokens['reference_'+kind], wf, wb)
    mapped = map_tokens_to_original_grid(brightness_texture(qviews['rgb']), qgeo)
    margins['brightness_texture'], stats['brightness_texture'] = match_margin(mapped, brightness_texture(rviews['rgb']), wf, wb)
    if (q_h20 is None) != (r_h20 is None):
        raise ValueError('H20 control requires both explicitly bound classes, or explicit omission')
    if q_h20 is not None:
        margins['h20'], stats['h20'] = match_margin(q_h20, r_h20, cov, 1-cov)
    fields = {'mean':base.copy()}
    for kind in ('gray', 'rgb', 'brightness_texture')+ (('h20',) if q_h20 is not None else ()):
        fields[kind] = bounded_field(base, margins[kind], native_margin, cfg)
    masks = {name:dict(zip(('work','original'), render(f, query_rgb.shape[:2]))) for name,f in fields.items()}
    similarity = {}
    for role in ('reference', 'query'):
        cosine = np.einsum('nd,nd->n', tokens[role+'_rgb'], tokens[role+'_gray'], optimize=False)
        similarity[role] = dict(unit_RGB_gray_cosine_min=float(cosine.min()),mean=float(cosine.mean()),max=float(cosine.max()),
            maximum_patch_difference=float(np.abs(tokens[role+'_rgb']-tokens[role+'_gray']).max()),
            exact_equal=bool(np.array_equal(tokens[role+'_rgb'],tokens[role+'_gray'])))
    info=dict(config=asdict(cfg),new_encoder_forwards=len(calls),calls=calls,query_gt_used=False,
        independent_new_method_count=1,reference_geometry=rgeo,query_geometry=qgeo,
        FG_BG_padding_excluded=True,reference_valid_patch_coverage=rweights['valid'].tolist(),
        reference_foreground_patch_coverage=wf.tolist(),margin_stats=stats,native_margin_stats=native_stat,
        producer_binding=producer_binding,RGB_gray_representation=similarity,
        RGB_gray_margin_maximum_difference=float(np.abs(margins['gray']-margins['rgb']).max()),
        gray_vs_RGB_field_maximum_difference=float(np.abs(fields['gray']-fields['rgb']).max()),
        maximum_edit_vs_MEAN=max(float(np.abs(f-base).max()) for f in fields.values()),
        original_RGB_gray_exactly_equal=all(value['exact_equal'] for value in similarity.values()),
        gray_information_and_class_margin_meaning='unknown; direct prototype matching, not calibrated semantic probability',
        semantic_invariance_claim=False,wall_seconds=time.perf_counter()-started,abstention=False)
    info['h20_control']='actual_bound_rawH20' if q_h20 is not None else 'OMITTED_NO_ACTUAL_BOUND_H20; not simulated or MEAN alias'
    return dict(fields=fields,masks=masks,margins={'native':native_margin,**margins},tokens=tokens,info=info)


class ActualPatch128Encoder:
    def __init__(self, wrapper):
        import torch
        self.model = wrapper.m.float().eval().requires_grad_(False)
        if type(self.model).__module__!='timm.models.eva' or type(self.model).__name__!='Eva' or self.model.num_prefix_tokens!=5:
            raise ValueError('Explicit actual Eva5prefix interface required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

    def __call__(self, image):
        import torch
        x=torch.from_numpy(np.asarray(image,np.uint8).copy()).permute(2,0,1).float()/255
        mean=x.new_tensor((.485,.456,.406))[:,None,None];std=x.new_tensor((.229,.224,.225))[:,None,None]
        with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):tokens=self.model.forward_features(((x-mean)/std)[None])
        if tuple(tokens.shape)!=(1,69,1024) or tokens.dtype!=torch.float32:raise ValueError('Actual dynamic128 FP32 final norm shape mismatch')
        return tokens[0,5:].numpy().astype(np.float64)
