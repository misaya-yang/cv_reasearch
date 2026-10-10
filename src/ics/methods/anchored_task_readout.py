"""Frozen spatial role prototypes and one anchored whole-canvas decision.

This is an experimental readout, not the default segmenter. It consumes actual
CPU FP32 O24 feature caches and an actual reference mask / a FoRIS pseudo mask;
it has no model, image loader, query-GT access, local FoRIS frontend or CRF.
Prototype construction uses at most 64 foreground and 64 pure-background cells.
All sign thresholds are zero. The fixed tanh scale is .07, not a probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F

_NORM_EPS = 1e-12
_SOURCE_EPS = 1e-6  # Same numeric denominator floor as source FoRIS min-max.
_SCALE = .07


def _fp32_cpu(value, name):
    tensor = torch.as_tensor(value)
    if tensor.dtype != torch.float32 or tensor.device.type != 'cpu':
        raise ValueError(f'{name} must be the actual CPU FP32 array')
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} contains nonfinite values')
    return tensor


@dataclass(frozen=True)
class RoleBank:
    valid: bool
    reason: str | None
    foreground: torch.Tensor
    background: torch.Tensor
    foreground_xy: torch.Tensor
    background_xy: torch.Tensor
    foreground_cell_ids: torch.Tensor
    background_cell_ids: torch.Tensor
    foreground_mass: torch.Tensor
    background_mass: torch.Tensor
    paired_bg_indices: torch.Tensor
    coverage64: torch.Tensor

    @property
    def paired_background(self):
        return self.background[self.paired_bg_indices]


@torch.inference_mode()
def build_bank(raw_FP32_O24, reference_mask) -> RoleBank:
    """Construct spatial role prototypes from a real binary role mask.

    O24 is [4096,D], D=1024 for the actual frozen model (D can be smaller for
    mathematical fixtures). The original binary mask is nearest-resized to
    1024, area-pooled to64, then grouped into8x8 cells of8x8 feature tokens.
    Fractional foreground coverage is retained; BG uses only exact c==0.
    Weighted xy coordinates are token centers in the normalized image frame.
    BG pairing minimizes squared spatial distance; ties use original cell order.
    """
    raw = _fp32_cpu(raw_FP32_O24, 'raw O24')
    if raw.ndim != 2 or raw.shape[0] != 4096 or raw.shape[1] < 1:
        raise ValueError('Require O24 [4096,D] on its actual64x64 token grid')
    mask = torch.as_tensor(reference_mask)
    if mask.device.type != 'cpu' or mask.ndim != 2 or not mask.numel() or not torch.isfinite(mask).all():
        raise ValueError('Require a finite nonempty CPU binary mask canvas')
    if (mask < 0).any():
        raise ValueError('Negative role-mask values are invalid')
    binary = (mask > 0).float()[None,None]
    canonical = F.interpolate(binary, (1024,1024), mode='nearest')
    coverage = F.adaptive_avg_pool2d(canonical, (64,64))[0,0]
    unit = F.normalize(raw, dim=1, eps=_NORM_EPS).reshape(64,64,-1)
    cells = unit.reshape(8,8,8,8,-1).permute(0,2,1,3,4).reshape(64,64,-1)
    fg_weights = coverage.reshape(8,8,8,8).permute(0,2,1,3).reshape(64,64)
    bg_weights = (fg_weights == 0).float()
    axis = (torch.arange(64, dtype=torch.float32)+.5)/64
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    xy = torch.stack((xx,yy),-1).reshape(8,8,8,8,2).permute(0,2,1,3,4).reshape(64,64,2)

    def prototypes(weights):
        mass = weights.sum(1)
        ids = torch.nonzero(mass>0, as_tuple=False).reshape(-1)
        active = weights[ids]
        means = torch.einsum('ct,ctd->cd', active, cells[ids])/mass[ids,None]
        centers = torch.einsum('ct,ctd->cd', active, xy[ids])/mass[ids,None]
        norms = means.norm(dim=1)
        return F.normalize(means,dim=1,eps=_NORM_EPS),centers,ids,mass[ids],norms

    fg,fg_xy,fg_ids,fg_mass,fg_norm = prototypes(fg_weights)
    bg,bg_xy,bg_ids,bg_mass,bg_norm = prototypes(bg_weights)
    # Active cells with a canceled / numerically zero mean have no direction.
    # Do not silently drop them, fabricate a unit prototype, or allow a partial
    # valid bank. The floor is the normalization safeguard, not a GT threshold.
    if not len(fg):
        reason = 'missing_reference_foreground'
    elif not len(bg):
        reason = 'missing_pure_reference_background'
    elif (fg_norm <= _NORM_EPS).any():
        reason = 'degenerate_reference_foreground_prototype'
    elif (bg_norm <= _NORM_EPS).any():
        reason = 'degenerate_pure_reference_background_prototype'
    else:
        reason = None
    valid = reason is None
    pairs = torch.empty(0,dtype=torch.int64)
    if valid:
        distance = (fg_xy[:,None]-bg_xy[None]).square().sum(-1)
        pairs = distance.argmin(1)
    return RoleBank(valid,reason,fg,bg,fg_xy,bg_xy,fg_ids,bg_ids,fg_mass,bg_mass,pairs,coverage)


@torch.inference_mode()
def role_field(q_unit, bank: RoleBank) -> torch.Tensor:
    """Nearest FG cosine, then contrast its spatially paired BG: tanh(m/.07).

    Invalid banks raise explicitly; the caller can disable that role evidence
    using bank.valid. A zero field is not fabricated as valid role identity.
    The largest semantic matrix is [N,K_F], K_F<=64, not an all-token kernel.
    """
    if not bank.valid:
        raise ValueError('Invalid role bank: '+str(bank.reason))
    q = _fp32_cpu(q_unit,'query unit features')
    if q.ndim != 2 or q.shape[1] != bank.foreground.shape[1]:
        raise ValueError('Query and reference prototype dimensions differ')
    if q.shape[0] and (q.norm(dim=1)-1).abs().max() > 1e-3:
        raise ValueError('role_field requires already normalized query tokens')
    closest = (q@bank.foreground.T).argmax(1)
    difference = bank.foreground[closest]-bank.paired_background[closest]
    margin = (q*difference).sum(1)
    return torch.tanh(margin/_SCALE)


@torch.inference_mode()
def local_evidence(h_ref, h_query_context=None) -> torch.Tensor:
    """Context adjusts task-evidence magnitude in [.5,1], never its sign."""
    reference = _fp32_cpu(h_ref,'reference role field')
    if (reference.abs()>1).any():
        raise ValueError('Reference field must be bounded in [-1,1]')
    if h_query_context is None:
        return reference.clone()
    context = _fp32_cpu(h_query_context,'query-context role field')
    if context.shape != reference.shape or (context.abs()>1).any():
        raise ValueError('Context field must match shape and be bounded')
    return reference*(.75+.25*reference*context)


@torch.inference_mode()
def resize_field(field64, shape: tuple[int,int]) -> np.ndarray:
    """Bilinear signed64 field to actual window/canvas, align_corners=False."""
    value = _fp32_cpu(field64,'role/readout field')
    if value.numel()!=4096 or value.ndim not in (1,2):
        raise ValueError('Require a64x64/4096-element field')
    if len(shape)!=2 or min(shape)<1:
        raise ValueError('Invalid destination geometry')
    result = F.interpolate(value.reshape(1,1,64,64), size=shape,
        mode='bilinear',align_corners=False)[0,0]
    return result.numpy().copy()


@dataclass(frozen=True)
class Anchor:
    baseline: np.ndarray
    g0: np.ndarray
    h0: np.ndarray
    base_D: np.ndarray


@torch.inference_mode()
def make_anchor(B1024, score64, h064=None) -> Anchor:
    """FoRIS output sign anchors absolute pre-threshold response confidence.

    B is the complete original FoRIS result rendered back to its1024 guide.
    Source score uses FoRIS min-max's numeric floor. Constant score gives g0=0.
    A missing legal task bank is declared by h064=None, not invented evidence.
    """
    baseline = np.asarray(B1024,dtype=bool)
    if baseline.shape!=(1024,1024):
        raise ValueError('Require the complete FoRIS guide on1024 canvas')
    source = _fp32_cpu(score64,'actual source pre-threshold score')
    if source.numel()!=4096 or source.ndim not in (1,2):
        raise ValueError('Require the actual64x64 source score')
    span = float(source.max()-source.min())
    if span==0:
        g0 = np.zeros(baseline.shape,dtype=np.float32)
    else:
        normalized = (source-source.min())/max(span,_SOURCE_EPS)
        magnitude = np.abs(2*resize_field(normalized,(1024,1024))-1)
        g0 = (2*baseline.astype(np.float32)-1)*magnitude
    h0 = np.zeros(baseline.shape,np.float32) if h064 is None else resize_field(h064,(1024,1024))
    if (np.abs(h0)>1).any():
        raise ValueError('Whole-image task role field must be bounded')
    # FP64 canvas accumulation prevents ordinary FP32 order sensitivity when
    # adaptive view order differs from the fixed all9 order. Field inputs stay
    # actual FP32. No numeric tolerance is used as a foreground threshold.
    base = g0.astype(np.float64)+.5*h0.astype(np.float64)
    return Anchor(baseline.copy(),g0,h0,base)


def _sum_and_coverage(anchor, local_sum, total_coverage):
    value = np.asarray(local_sum,dtype=np.float64)
    coverage = np.asarray(total_coverage)
    shape = anchor.baseline.shape
    if value.shape!=shape or coverage.shape!=shape or not np.isfinite(value).all() or not np.isfinite(coverage).all():
        raise ValueError('Local accumulation and coverage must share anchor canvas')
    if (coverage<0).any() or not np.array_equal(coverage,np.floor(coverage)):
        raise ValueError('Coverage must be nonnegative integer counts')
    if (np.abs(value)>coverage+1e-6).any() or (value[coverage==0]!=0).any():
        raise ValueError('Accumulation exceeds bounded local evidence/coverage')
    return value,coverage.astype(np.float64)


def _mask(D, baseline):
    result = D>0
    tied = D==0
    result[tied] = baseline[tied]
    return result


def decision(anchor: Anchor, local_sum, total_coverage):
    """One final D=g0+.5*h0+.5*sum(local)/full9coverage decision."""
    value,coverage = _sum_and_coverage(anchor,local_sum,total_coverage)
    mean = np.divide(value,coverage,out=np.zeros_like(value),where=coverage>0)
    D = anchor.base_D+.5*mean
    return _mask(D,anchor.baseline),D


@dataclass(frozen=True)
class Certificate:
    mask: np.ndarray
    D: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    unknown_budget: np.ndarray
    uncertain: np.ndarray
    total_coverage: np.ndarray


def stream_certificate(anchor: Anchor, partial_sum, total_coverage, unseen_coverage) -> Certificate:
    """Same final decision's interval; unseen views are bounded unknown fields.

    Lower>0 or upper<0 certifies a strict sign. If no budget remains, exact ties
    use B. View coefficients never renormalize to the observed subset.
    """
    partial,coverage = _sum_and_coverage(anchor,partial_sum,total_coverage)
    unseen = np.asarray(unseen_coverage)
    if unseen.shape!=coverage.shape or not np.isfinite(unseen).all() or (unseen<0).any() or (unseen>coverage).any() or not np.array_equal(unseen,np.floor(unseen)):
        raise ValueError('Unseen coverage must be an integer subset of fixed coverage')
    if (np.abs(partial)>coverage-unseen+1e-6).any():
        raise ValueError('Partial sum exceeds already observed evidence budget')
    mask,D = decision(anchor,partial,coverage)
    U = np.divide(.5*unseen,coverage,out=np.zeros_like(partial),where=coverage>0)
    lower,upper = D-U,D+U
    # Outward one-ulp rounding is only an enclosure safeguard, never a tuned
    # ambiguity/GT threshold. With no unseen budget the final sign is exact.
    has_unseen = U>0
    lower[has_unseen] = np.nextafter(lower[has_unseen],-np.inf)
    upper[has_unseen] = np.nextafter(upper[has_unseen],np.inf)
    definite = (lower>0)|(upper<0)|(~has_unseen&(D==0))
    return Certificate(mask,D,lower,upper,U,~definite,coverage)


def choose_next(certificate: Certificate, boxes: Sequence[tuple[int,int,int,int]], observed) -> int | None:
    """Max unresolved per-view budget using geometry only; ties lower index."""
    observed = set(observed)
    remaining = [i for i in range(len(boxes)) if i not in observed]
    if not remaining or not certificate.uncertain.any():
        return None
    relevant = np.divide(.5*certificate.uncertain,certificate.total_coverage,
        out=np.zeros_like(certificate.D),where=certificate.total_coverage>0)
    scores = {}
    height,width = certificate.D.shape
    for i in remaining:
        x0,y0,x1,y1 = boxes[i]
        if not (0<=x0<x1<=width and 0<=y0<y1<=height):
            raise ValueError('Window box lies outside the frozen canvas')
        scores[i] = float(relevant[y0:y1,x0:x1].sum())
    return min(remaining,key=lambda i:(-scores[i],i))
