"""Standard full-query decoder for reference-conditioned response observations.

All feature maps are floating [B, D, H, W]. Final/immediate/response maps have
final_channels; middle maps may concatenate a native layer output and Q/K/V
and have independent middle_channels. Support final/middle maps share a grid;
all six query maps share another grid. Native final/middle projections are shared across roles.
The four immediate/final-response projections have independent parameters.
Required support/query prefix banks contain selected-layer normalized native
outputs plus ALL native Q/K/V channels, in the declared middle_channels.
Every prefix token is projected and attended to, never first mean-pooled.
Query prefix immediate plus/minus responses are appended; the support prefix
slots are exact zeros. Both arms consume the same prefix inputs. This covers
selected-layer patch and prefix QKV, not the full encoder hidden state.

Order: area-downsample legal SUPPORT mask once with ``mask_to_coverage``;
forward returns signed logits [B, 1, Hq, Wq] for EVERY query patch; bilinear
upsample signed logits with align_corners=False to the declared 1024 grid;
threshold strictly at zero; the caller performs its one original source
CRF/final-size operation with query RGB. This module executes no CRF, source
minmax, candidate gate, label-dependent output selection or encoder call.
Training loss upsamples logits to the supplied label grid before BOTH losses;
query annotations enter only that separate loss function, never forward.
Logits and attention contexts are learned readouts, not calibrated posteriors.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _grid(grid):
    if len(grid) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in grid):
        raise ValueError('Expected two positive integer grid dimensions')
    return tuple(grid)


def _finite_float(tensor, name, ndim):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim or not tensor.is_floating_point():
        raise ValueError(name + ' must be a floating tensor of rank ' + str(ndim))
    if any(v <= 0 for v in tensor.shape) or not torch.isfinite(tensor).all():
        raise ValueError(name + ' must be nonempty and finite')


def _finite_prefix(tensor, name):
    """A prefix bank may explicitly have zero tokens, but no empty batch/channels."""
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3
            or not tensor.is_floating_point() or tensor.shape[0] <= 0 or tensor.shape[2] <= 0
            or not torch.isfinite(tensor).all()):
        raise ValueError(name + ' must be finite floating [B,P,C], with B/C nonempty')


def mask_to_coverage(support_mask, support_grid):
    """Legal support [B,1,H,W] mask -> fractional coverage; no purity gate."""
    support_grid = _grid(support_grid)
    if not isinstance(support_mask, torch.Tensor) or support_mask.ndim != 4 or support_mask.shape[1] != 1:
        raise ValueError('Support mask must be [B,1,H,W]')
    if any(v <= 0 for v in support_mask.shape):
        raise ValueError('Support mask must be nonempty')
    values = support_mask if support_mask.is_floating_point() else support_mask.float()
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Support mask values must be finite in [0,1]')
    return F.interpolate(values, size=support_grid, mode='area')


def support_attention(query, support, coverage, *, query_chunk_size=256):
    """All-anchor FG/BG attention, with no full Nq-by-Ns score allocation.

    Inputs are query [B,Nq,C], support [B,Ns,C], coverage [B,Ns]. Every
    support token remains in the key/value bank. Exact zero role weights are
    excluded; a missing role has an exactly zero context. Query chunking
    changes neither support normalization nor coverage weights. During
    autograd, checkpointing recomputes each score chunk instead of retaining
    all Nq-by-Ns probabilities. It adds backward compute, not information.
    """
    _finite_float(query, 'Query tokens', 3)
    _finite_float(support, 'Support tokens', 3)
    _finite_float(coverage, 'Support coverage', 2)
    if query.shape[0] != support.shape[0] or query.shape[2] != support.shape[2]:
        raise ValueError('Query/support batch and channel dimensions must match')
    if coverage.shape != support.shape[:2]:
        raise ValueError('Coverage must match every support token')
    if any(t.device != query.device or t.dtype != query.dtype for t in (support, coverage)):
        raise ValueError('Attention inputs must share dtype and device')
    if (coverage < 0).any() or (coverage > 1).any():
        raise ValueError('Coverage must lie in [0,1]')
    if not isinstance(query_chunk_size, int) or isinstance(query_chunk_size, bool) or query_chunk_size <= 0:
        raise ValueError('Query chunk size must be a positive integer')

    role_logs, available = [], []
    for weights in (coverage, 1-coverage):
        present = weights.sum(dim=1) > 0
        logs = weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
        logs = logs.masked_fill(weights == 0, -torch.inf)
        # Avoid an all-minus-infinity softmax; its context is zeroed below.
        logs = torch.where(present[:, None], logs, torch.zeros_like(logs))
        role_logs.append(logs)
        available.append(present)

    def attend(q, s, fg_log, bg_log, fg_present, bg_present):
        scores = (q @ s.transpose(1, 2))/math.sqrt(q.shape[-1])
        contexts = []
        for logs, present in ((fg_log, fg_present), (bg_log, bg_present)):
            probabilities = (scores+logs[:, None]).softmax(dim=-1)
            context = probabilities @ s
            contexts.append(context*present[:, None, None].to(context.dtype))
        return tuple(contexts)

    foreground, background = [], []
    for start in range(0, query.shape[1], query_chunk_size):
        arguments = (query[:, start:start+query_chunk_size], support,
                     role_logs[0], role_logs[1], available[0], available[1])
        if torch.is_grad_enabled() and any(t.requires_grad for t in arguments):
            fg, bg = checkpoint(attend, *arguments, use_reentrant=False, preserve_rng_state=False)
        else:
            fg, bg = attend(*arguments)
        foreground.append(fg)
        background.append(bg)
    return torch.cat(foreground, dim=1), torch.cat(background, dim=1)


def prefix_attention(query, prefix, *, query_chunk_size=256):
    """Attention over EVERY prefix token; no FG/BG role or physical label.

    Query [B,Nq,C], prefix [B,P,C]. An explicitly empty bank returns zero
    context. Chunk scores are checkpointed during autograd as for patch
    attention; the prefix bank remains intact in every chunk.
    """
    _finite_float(query, 'Prefix-attention query', 3)
    _finite_prefix(prefix, 'Prefix bank')
    if query.shape[0] != prefix.shape[0] or query.shape[2] != prefix.shape[2]:
        raise ValueError('Prefix/query batch and channel dimensions must match')
    if prefix.device != query.device or prefix.dtype != query.dtype:
        raise ValueError('Prefix/query dtype and device must match')
    if not isinstance(query_chunk_size, int) or isinstance(query_chunk_size, bool) or query_chunk_size <= 0:
        raise ValueError('Query chunk size must be a positive integer')
    if prefix.shape[1] == 0:
        return torch.zeros_like(query)

    def attend(q, p):
        scores = (q @ p.transpose(1, 2))/math.sqrt(q.shape[-1])
        return scores.softmax(dim=-1) @ p

    result = []
    for start in range(0, query.shape[1], query_chunk_size):
        arguments = (query[:, start:start+query_chunk_size], prefix)
        if torch.is_grad_enabled() and any(t.requires_grad for t in arguments):
            context = checkpoint(attend, *arguments, use_reentrant=False, preserve_rng_state=False)
        else:
            context = attend(*arguments)
        result.append(context)
    return torch.cat(result, dim=1)


class _StreamProjection(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.LayerNorm(channels), nn.Linear(channels, 64), nn.GELU())


class _ResidualSpatialBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm, GELU and a residual connection."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 128)

    def forward(self, value):
        hidden = F.gelu(self.norm1(self.conv1(value)))
        return F.gelu(value+self.norm2(self.conv2(hidden)))


class ReferenceResponseDecoder(nn.Module):
    """Same 64-channel-stream architecture for response and static controls.

    ``static=True`` duplicates the supplied native immediate plus/minus maps
    into the final-response slots. Those slots still pass through their OWN
    trainable projections; they are not disabled or filled with dead zeros.
    Middle and final streams may have different declared channel counts.
    Required prefix inputs include every selected-layer native prefix QKV
    token and native query immediate prefix responses, identically for both
    arms. No prefix FG/BG label is inferred from its token position.
    The runtime adapter owns frozen
    DINO extraction, selected-layer/probe arithmetic, and original refinement.
    """
    def __init__(self, final_channels, middle_channels=None, *, query_chunk_size=256):
        super().__init__()
        if middle_channels is None:
            middle_channels = final_channels
        for channels in (final_channels, middle_channels):
            if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
                raise ValueError('Native channel counts must be positive integers')
        if not isinstance(query_chunk_size, int) or isinstance(query_chunk_size, bool) or query_chunk_size <= 0:
            raise ValueError('Query chunk size must be a positive integer')
        self.final_channels = final_channels
        self.middle_channels = middle_channels
        self.query_chunk_size = query_chunk_size
        self.native_final = _StreamProjection(final_channels)
        self.native_middle = _StreamProjection(middle_channels)
        self.immediate_plus = _StreamProjection(final_channels)
        self.immediate_minus = _StreamProjection(final_channels)
        self.response_plus = _StreamProjection(final_channels)
        self.response_minus = _StreamProjection(final_channels)
        self.prefix_projection = _StreamProjection(middle_channels+2*final_channels)
        self.prefix_query = nn.Linear(128, 64)
        # Old640 + full support/query prefix contexts64 each.
        self.fusion = nn.Linear(768, 128)
        self.spatial = nn.Sequential(*[_ResidualSpatialBlock() for _ in range(3)])
        self.output = nn.Conv2d(128, 1, 1)

    def forward(self, support_final, support_middle, support_coverage,
                query_final, query_middle, immediate_plus, immediate_minus,
                response_plus=None, response_minus=None, *, support_prefix,
                query_prefix, prefix_immediate_plus, prefix_immediate_minus,
                static=False):
        """Predict every Q patch. No query GT, class ID or target-area input."""
        if not isinstance(static, bool):
            raise ValueError('Static mode must be boolean')
        if static:
            if response_plus is not None or response_minus is not None:
                raise ValueError('Static mode takes native immediate maps, not response maps')
            response_plus, response_minus = immediate_plus, immediate_minus
        elif response_plus is None or response_minus is None:
            raise ValueError('Response mode needs both final downstream response maps')
        named = dict(support_final=support_final, support_middle=support_middle,
                     query_final=query_final, query_middle=query_middle,
                     immediate_plus=immediate_plus, immediate_minus=immediate_minus,
                     response_plus=response_plus, response_minus=response_minus)
        parameter = self.fusion.weight
        for name, tensor in named.items():
            _finite_float(tensor, name, 4)
            channels = self.middle_channels if name.endswith('_middle') else self.final_channels
            if tensor.shape[1] != channels or tensor.shape[0] != support_final.shape[0]:
                raise ValueError('Feature maps must share batch and have declared stream channels')
            if tensor.device != parameter.device or tensor.dtype != parameter.dtype:
                raise ValueError('Feature maps must match decoder parameter dtype/device')
        if support_middle.shape[-2:] != support_final.shape[-2:]:
            raise ValueError('Support final/middle grids must coincide')
        for tensor in (query_middle, immediate_plus, immediate_minus, response_plus, response_minus):
            if tensor.shape[-2:] != query_final.shape[-2:]:
                raise ValueError('All query observation grids must coincide')
        _finite_float(support_coverage, 'Support coverage', 4)
        if support_coverage.shape != (support_final.shape[0], 1, *support_final.shape[-2:]):
            raise ValueError('Coverage must be area-downsampled to the support grid')
        if support_coverage.device != parameter.device or (support_coverage < 0).any() or (support_coverage > 1).any():
            raise ValueError('Support coverage must share device and lie in [0,1]')
        prefixes = dict(support_prefix=support_prefix, query_prefix=query_prefix,
                        prefix_immediate_plus=prefix_immediate_plus,
                        prefix_immediate_minus=prefix_immediate_minus)
        for name, tensor in prefixes.items():
            _finite_prefix(tensor, name)
            channels = self.final_channels if name.startswith('prefix_immediate') else self.middle_channels
            if tensor.shape[0] != support_final.shape[0] or tensor.shape[2] != channels:
                raise ValueError('Prefix inputs must match batch and declared channel counts')
            if tensor.device != parameter.device or tensor.dtype != parameter.dtype:
                raise ValueError('Prefix inputs must match decoder dtype/device')
        if (prefix_immediate_plus.shape[:2] != query_prefix.shape[:2]
                or prefix_immediate_minus.shape[:2] != query_prefix.shape[:2]):
            raise ValueError('Immediate prefix responses must match all query prefix tokens')

        def tokens(value):
            return value.flatten(2).transpose(1, 2)

        support = torch.cat((self.native_final(tokens(support_final)),
                             self.native_middle(tokens(support_middle))), dim=-1)
        query = torch.cat((self.native_final(tokens(query_final)),
                           self.native_middle(tokens(query_middle))), dim=-1)
        fg, bg = support_attention(query, support, support_coverage.flatten(1).to(query.dtype),
                                   query_chunk_size=self.query_chunk_size)
        support_zero = support_prefix.new_zeros((*support_prefix.shape[:2], self.final_channels))
        support_bank = self.prefix_projection(torch.cat((support_prefix, support_zero, support_zero), dim=-1))
        query_bank = self.prefix_projection(torch.cat((query_prefix, prefix_immediate_plus,
                                                       prefix_immediate_minus), dim=-1))
        prefix_query = self.prefix_query(query)
        support_context = prefix_attention(prefix_query, support_bank, query_chunk_size=self.query_chunk_size)
        query_context = prefix_attention(prefix_query, query_bank, query_chunk_size=self.query_chunk_size)
        streams = (query, self.immediate_plus(tokens(immediate_plus)),
                   self.immediate_minus(tokens(immediate_minus)),
                   self.response_plus(tokens(response_plus)),
                   self.response_minus(tokens(response_minus)), fg, bg,
                   support_context, query_context)
        hidden = self.fusion(torch.cat(streams, dim=-1))
        batch, _, height, width = query_final.shape
        hidden = hidden.transpose(1, 2).reshape(batch, 128, height, width)
        return self.output(self.spatial(hidden))


def upsample_logits(logits, target_hw=(1024, 1024)):
    """Bilinear signed logits BEFORE threshold; no minmax or refinement."""
    _finite_float(logits, 'Logits', 4)
    if logits.shape[1] != 1:
        raise ValueError('Binary logits must have one channel')
    return F.interpolate(logits, size=_grid(target_hw), mode='bilinear', align_corners=False)


def logits_to_mask(logits, target_hw=(1024, 1024)):
    """Declared model-grid mask only; caller owns the one CRF/final-size step."""
    return upsample_logits(logits, target_hw) > 0


def segmentation_loss(logits, labels, *, valid_mask=None):
    """BCE + batch-mean soft-IoU loss; query labels are allowed ONLY here.

    Labels [B,1,H,W] may be boolean or finite [0,1] floating values. Logits
    are upsampled to that grid first; labels are not downsampled/thresholded.
    Optional valid_mask is boolean [B,1,H,W] on the label device. BCE is
    averaged over all valid pixels; soft-IoU is computed over valid pixels
    per image, then averaged across images. Every image must have at least
    one valid pixel. None preserves the original unmasked calculation.
    An all-background or all-foreground label remains finite. A zero-sized
    image is invalid. Epsilon is a fixed numerical guard, not an IoU prior.
    """
    _finite_float(logits, 'Logits', 4)
    if not isinstance(labels, torch.Tensor) or labels.ndim != 4 or labels.shape[:2] != logits.shape[:2]:
        raise ValueError('Labels must match logit batch/channel dimensions')
    if logits.shape[1] != 1 or any(v <= 0 for v in labels.shape) or labels.device != logits.device:
        raise ValueError('Nonempty binary labels must share the logit device')
    labels = labels.to(dtype=logits.dtype)
    if not torch.isfinite(labels).all() or (labels < 0).any() or (labels > 1).any():
        raise ValueError('Labels must be finite in [0,1]')
    resized = upsample_logits(logits, tuple(labels.shape[-2:]))
    if valid_mask is None:
        bce = F.binary_cross_entropy_with_logits(resized, labels)
        probability = resized.sigmoid()
        intersection = (probability*labels).sum(dim=(1, 2, 3))
        union = (probability+labels-probability*labels).sum(dim=(1, 2, 3))
    else:
        if (not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool
                or valid_mask.shape != labels.shape or valid_mask.device != labels.device):
            raise ValueError('Valid mask must be boolean and match label shape/device')
        if not valid_mask.flatten(1).any(dim=1).all():
            raise ValueError('Every image must contain at least one valid pixel')
        pixel_bce = F.binary_cross_entropy_with_logits(resized, labels, reduction='none')
        bce = pixel_bce.masked_select(valid_mask).mean()
        probability = resized.sigmoid()
        valid = valid_mask.to(dtype=probability.dtype)
        intersection = (probability*labels*valid).sum(dim=(1, 2, 3))
        union = ((probability+labels-probability*labels)*valid).sum(dim=(1, 2, 3))
    soft_iou = 1-((intersection+1e-6)/(union+1e-6)).mean()
    return dict(loss=bce+soft_iou, bce=bce, soft_iou_loss=soft_iou)
