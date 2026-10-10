"""Frozen-O24 observation and reference-role candidates for joint inference.

No model, loader, prediction mask, query label or FoRIS dependency. The caller
supplies the actual whole-reference, whole-query and four native-quadrant O24
arrays, and an already aligned reference mask. This module creates unary
probabilities; it does not solve the graph or claim segmentation quality.
"""
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

FOUR_BOXES = ((0, 0, 512, 512), (512, 0, 1024, 512),
              (0, 512, 512, 1024), (512, 512, 1024, 1024))
GRID_HW = (128, 128)
ROLE_NAMES = ('background', 'foreground')


class MissingReferenceRole(ValueError):
    """The required full-reference positive/negative role is absent."""

    def __init__(self, missing_roles: Sequence[str]):
        self.missing_roles = tuple(missing_roles)
        super().__init__('Missing full-reference role: ' + ', '.join(self.missing_roles))


def _raw(value: Any, name: str, feature_dim: int) -> torch.Tensor:
    x = torch.as_tensor(value)
    if x.device.type != 'cpu' or x.dtype != torch.float32 or tuple(x.shape) != (4096, feature_dim):
        raise ValueError(f'{name} must be CPU FP32 [4096,{feature_dim}]')
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f'{name} contains non-finite O24 values')
    norm = torch.linalg.vector_norm(x, dim=1)
    if not bool((torch.isfinite(norm) & (norm > 0)).all()):
        raise ValueError(f'{name} must have finite nonzero token norms')
    return x.detach()


def reference_role_atoms(reference_mask_canvas1024: Any) -> dict[str, torch.Tensor]:
    """Continuous role coverage/centroids in the full reference canvas.

    Pixel centres, not integer pixel corners, define centroids. Coordinates
    are in reference patch units (16 canonical pixels). Geometry uses FP64
    accumulation, then returns FP32; mixed roles share a visual patch index.
    """
    mask = torch.as_tensor(reference_mask_canvas1024)
    if mask.device.type != 'cpu' or tuple(mask.shape) != (1024, 1024):
        raise ValueError('reference mask must already align to CPU 1024x1024 canvas')
    if mask.dtype != torch.bool and not mask.dtype.is_floating_point:
        raise ValueError('reference mask must be bool or finite [0,1] floating coverage')
    if not bool(torch.isfinite(mask).all()) or not bool(((mask >= 0) & (mask <= 1)).all()):
        raise ValueError('reference mask coverage must be finite and within [0,1]')
    pixels = mask.detach().to(torch.float64).reshape(64, 16, 64, 16).permute(0, 2, 1, 3)
    axis = (torch.arange(16, dtype=torch.float64) + .5) / 16
    yy, xx = torch.meshgrid(torch.arange(64, dtype=torch.float64),
                            torch.arange(64, dtype=torch.float64), indexing='ij')
    coverage, positions = [], []
    for role_pixels in (1 - pixels, pixels):
        mass = role_pixels.sum(dim=(-2, -1))
        coverage.append((mass / 256).reshape(-1))
        safe_mass = mass.clamp_min(torch.finfo(torch.float64).tiny)
        local_x = (role_pixels * axis[None, None, None, :]).sum(dim=(-2, -1)) / safe_mass
        local_y = (role_pixels * axis[None, None, :, None]).sum(dim=(-2, -1)) / safe_mass
        xy = torch.stack((xx + local_x, yy + local_y), dim=-1).reshape(4096, 2)
        xy[mass.reshape(-1) == 0] = 0
        positions.append(xy)
    cover = torch.stack(coverage, dim=1)
    total = cover.sum(dim=0)
    missing = [ROLE_NAMES[y] for y in range(2) if float(total[y]) <= 0]
    if missing:
        raise MissingReferenceRole(missing)
    log_weight = torch.full_like(cover, -torch.inf)
    valid = cover > 0
    weights = cover / total
    log_weight[valid] = weights[valid].log()
    return dict(patch_index=torch.arange(4096, dtype=torch.int64),
                role=torch.arange(2, dtype=torch.int64), valid=valid,
                role_coverage=cover.float(), role_area_pixels=(cover * 256).float(),
                xy=torch.stack(positions, dim=1).float(),
                log_role_weight=log_weight.float(), role_weight=weights.float(),
                total_role_area_pixels=(total * 256).float())


def _sampling_plan() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly the align_corners=False 64->128 bilinear coordinate plan."""
    fine = torch.arange(128, dtype=torch.float32)
    source = (fine + .5) / 2 - .5
    lower = source.floor().long()
    weight = source - lower
    lo, hi = lower.clamp(0, 63), (lower + 1).clamp(0, 63)
    ylo, xlo = torch.meshgrid(lo, lo, indexing='ij')
    yhi, xhi = torch.meshgrid(hi, hi, indexing='ij')
    wy, wx = torch.meshgrid(weight, weight, indexing='ij')
    ids = torch.stack((ylo * 64 + xlo, ylo * 64 + xhi,
                       yhi * 64 + xlo, yhi * 64 + xhi), dim=-1).reshape(-1, 4)
    weights = torch.stack(((1-wy)*(1-wx), (1-wy)*wx, wy*(1-wx), wy*wx), dim=-1).reshape(-1, 4)
    y, x = torch.meshgrid(fine, fine, indexing='ij')
    query_xy = torch.stack(((x + .5) / 2, (y + .5) / 2), dim=-1).reshape(-1, 2)
    return ids, weights, query_xy


def _bank(n: int, k: int) -> dict[str, torch.Tensor]:
    return dict(rank=torch.full((n, 2, k), -torch.inf),
                joint=torch.full((n, 2, k), -torch.inf),
                patch=torch.full((n, 2, k), -1, dtype=torch.int64))


def _deterministic_k(rank: torch.Tensor, patch: torch.Tensor,
                     joint: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Descending score, patch-ID ascending on exact FP32 ties.

    topk only obtains a cutoff and the complete set strictly above it. Its
    unstable cutoff ties are replaced by the smallest patch IDs. Sorting at
    most2k candidates fixes ties without perturbing scores or sorting all R.
    """
    top_values, top_indices = torch.topk(rank, k, dim=1, sorted=True)
    cutoff = top_values[:, -1:]
    strict = top_values > cutoff
    above_ids = patch.gather(1, top_indices)
    above_joint = joint.gather(1, top_indices)
    above_rank = torch.where(strict, top_values, -torch.inf)
    above_ids = torch.where(strict, above_ids, -1)
    above_joint = torch.where(strict, above_joint, -torch.inf)
    at_cutoff = (rank == cutoff) & (patch >= 0) & torch.isfinite(rank)
    sentinel = 4096
    tie_ids = torch.where(at_cutoff, patch, sentinel)
    smallest, tie_indices = torch.topk(tie_ids, k, dim=1, largest=False, sorted=True)
    tie_valid = smallest < sentinel
    chosen_ids = torch.cat((above_ids, torch.where(tie_valid, smallest, -1)), dim=1)
    chosen_rank = torch.cat((above_rank, torch.where(tie_valid, rank.gather(1, tie_indices), -torch.inf)), dim=1)
    chosen_joint = torch.cat((above_joint, torch.where(tie_valid, joint.gather(1, tie_indices), -torch.inf)), dim=1)
    by_id = torch.argsort(torch.where(chosen_ids >= 0, chosen_ids, sentinel), dim=1, stable=True)
    chosen_ids = chosen_ids.gather(1, by_id)
    chosen_rank = chosen_rank.gather(1, by_id)
    chosen_joint = chosen_joint.gather(1, by_id)
    by_score = torch.argsort(chosen_rank, dim=1, descending=True, stable=True)[:, :k]
    return chosen_rank.gather(1, by_score), chosen_ids.gather(1, by_score), chosen_joint.gather(1, by_score)


def _update(bank: dict[str, torch.Tensor], start: int, end: int, role: int,
            ranks: torch.Tensor, joint: torch.Tensor, patches: torch.Tensor,
            eligible: torch.Tensor) -> None:
    new_rank = ranks.masked_fill(~eligible[None], -torch.inf)
    new_joint = joint.masked_fill(~eligible[None], -torch.inf)
    new_patch = torch.where(eligible, patches, -1)[None].expand(end-start, -1)
    old_rank = bank['rank'][start:end, role]
    old_patch = bank['patch'][start:end, role]
    old_joint = bank['joint'][start:end, role]
    rank, patch, score = _deterministic_k(torch.cat((old_rank, new_rank), 1),
                                         torch.cat((old_patch, new_patch), 1),
                                         torch.cat((old_joint, new_joint), 1), old_rank.shape[1])
    bank['rank'][start:end, role] = rank
    bank['patch'][start:end, role] = patch
    bank['joint'][start:end, role] = score


def _merge_candidate_banks(global_bank: dict[str, torch.Tensor],
                           local_bank: dict[str, torch.Tensor],
                           joint_bank: dict[str, torch.Tensor],
                           expected_counts: Sequence[int] | torch.Tensor,
                           merge_chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integer-only global4/local4/joint8 merge, retaining exact source scores.

    Only the first occurrence of each valid patch is selected, in discovery
    order; the first eight unique patches fill each role. Global/local flags
    are combined even when they discover the same patch. Duplicate joint
    discoveries add no flag. No floating-point arithmetic or reranking.
    """
    n = global_bank['patch'].shape[0]
    counts = torch.as_tensor(expected_counts, dtype=torch.int64)
    if tuple(counts.shape) != (2,) or not bool(((counts >= 0) & (counts <= 8)).all()) or merge_chunk < 1:
        raise ValueError('Require two valid role counts within8 and positive merge chunk')
    for bank,k in ((global_bank,4),(local_bank,4),(joint_bank,8)):
        if bank['patch'].device.type != 'cpu' or bank['patch'].dtype != torch.int64 or tuple(bank['patch'].shape) != (n,2,k):
            raise ValueError('Candidate patch bank must be CPU int64 [N,2,K]')
        if bank['joint'].device.type != 'cpu' or bank['joint'].dtype != torch.float32 or tuple(bank['joint'].shape) != (n,2,k):
            raise ValueError('Candidate score bank must be CPU FP32 [N,2,K]')
    selected_patch = torch.full((n,16),-1,dtype=torch.int64)
    selected_score = torch.full((n,16),-torch.inf)
    sources = torch.zeros((n,16),dtype=torch.uint8)
    earlier = torch.tril(torch.ones((16,16),dtype=torch.bool),diagonal=-1)
    position = torch.arange(16,dtype=torch.int64)[None]
    for role in range(2):
        for start in range(0,n,merge_chunk):
            end = min(start+merge_chunk,n)
            gp = global_bank['patch'][start:end,role]
            lp = local_bank['patch'][start:end,role]
            ids = torch.cat((gp,lp,joint_bank['patch'][start:end,role]),dim=1)
            values = torch.cat((global_bank['joint'][start:end,role],
                                local_bank['joint'][start:end,role],
                                joint_bank['joint'][start:end,role]),dim=1)
            duplicate = ((ids[:,:,None] == ids[:,None,:]) & earlier[None]).any(dim=2)
            first = (ids >= 0) & ~duplicate
            keep = first & (first.cumsum(dim=1) <= 8)
            # Valid positions are unique integers, so padding ties cannot
            # affect selected order. All padded values are canonicalized below.
            keys = torch.where(keep,position,16)
            kept_position,order = torch.topk(keys,8,dim=1,largest=False,sorted=True)
            valid = kept_position < 16
            if not bool((valid.sum(dim=1) == counts[role]).all()):
                raise AssertionError('Candidate union no longer matches role coverage')
            chosen = ids.gather(1,order)
            score = values.gather(1,order)
            from_global = (chosen[:,:,None] == gp[:,None,:]).any(dim=2)
            from_local = (chosen[:,:,None] == lp[:,None,:]).any(dim=2)
            flags = from_global.to(torch.uint8) | (from_local.to(torch.uint8)*2)
            flags = torch.where(flags > 0,flags,4).to(torch.uint8)
            output = slice(role*8,(role+1)*8)
            selected_patch[start:end,output] = torch.where(valid,chosen,-1)
            selected_score[start:end,output] = torch.where(valid,score,-torch.inf)
            sources[start:end,output] = torch.where(valid,flags,0)
    return selected_patch,selected_score,sources


@torch.inference_mode()
def build_observation(reference_o24: Any, reference_mask_canvas1024: Any,
                      query_global_o24: Any, native4_o24: Sequence[Any], *,
                      boxes: Sequence[Sequence[int]] = FOUR_BOXES,
                      tau: float = .07, query_chunk: int = 256,
                      reference_chunk: int = 512, feature_dim: int = 1024,
                      return_observation_features: bool = False) -> dict[str, Any]:
    """Create sixteen role-conditioned candidates and mass-preserving unary.

    Real profiles use feature_dim=1024. Explicit smaller dimensions permit
    synthetic mechanism fixtures, with the same4096/16384 coordinate grids.
    Rmask is already aligned by the caller; this function never loads/resizes
    labels. Native arrays are NW,NE,SW,SE actual inputs, not sampled global Q.

    Returned b0 is probability; unary_cost=-logb0, with +inf invalid padding.
    Eight background slots precede eight foreground slots. P0 preserves full
    weighted reference-role mass; alpha is retained candidate location mass.
    """
    if tuple(tuple(b) for b in boxes) != FOUR_BOXES or len(native4_o24) != 4:
        raise ValueError('Require exactly the fixed NW/NE/SW/SE native quadrants')
    if not isinstance(feature_dim, int) or feature_dim < 1 or not isinstance(query_chunk, int) or not isinstance(reference_chunk, int) or min(query_chunk, reference_chunk) < 1:
        raise ValueError('Feature dimension and chunk sizes must be positive integers')
    if tau != .07:
        raise ValueError('This first candidate freezes tau=.07')
    reference = _raw(reference_o24, 'reference_o24', feature_dim)
    global_raw = _raw(query_global_o24, 'query_global_o24', feature_dim)
    native = [_raw(value, f'native4_o24[{i}]', feature_dim) for i, value in enumerate(native4_o24)]
    atoms = reference_role_atoms(reference_mask_canvas1024)
    reference_unit = reference / torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    sampled_raw = F.interpolate(global_raw.reshape(64,64,feature_dim).permute(2,0,1)[None],
                                size=GRID_HW, mode='bilinear', align_corners=False)[0].permute(1,2,0).reshape(16384,feature_dim).contiguous()
    global_norm = torch.linalg.vector_norm(sampled_raw, dim=1)
    if not bool((torch.isfinite(global_norm) & (global_norm > 0)).all()):
        raise ValueError('Sampled raw global Q contains finite-norm failure or zero vector')
    global_unit = sampled_raw / global_norm[:, None] if return_observation_features else None
    del sampled_raw
    fine_grid = torch.empty((128,128,feature_dim), dtype=torch.float32)
    for value, (x0,y0,x1,y1) in zip(native, FOUR_BOXES):
        unit = value / torch.linalg.vector_norm(value, dim=1, keepdim=True)
        fine_grid[y0//8:y1//8,x0//8:x1//8] = unit.reshape(64,64,feature_dim)
    fine_unit = fine_grid.reshape(16384,feature_dim)
    sample_ids, sample_weights, query_xy = _sampling_plan()
    global_bank, local_bank, joint_bank = _bank(16384,4), _bank(16384,4), _bank(16384,8)
    full_role_lse = torch.full((16384,2), -torch.inf)
    for r0 in range(0,4096,reference_chunk):
        r1 = min(r0+reference_chunk,4096)
        ru = reference_unit[r0:r1]
        coarse_scores = global_raw @ ru.T
        patches = torch.arange(r0,r1,dtype=torch.int64)
        for q0 in range(0,16384,query_chunk):
            q1 = min(q0+query_chunk,16384)
            global_scores = (coarse_scores[sample_ids[q0:q1]] * sample_weights[q0:q1,:,None]).sum(dim=1) / global_norm[q0:q1,None]
            local_scores = fine_unit[q0:q1] @ ru.T
            joint_scores = .5*global_scores + .5*local_scores
            for role in range(2):
                eligible = atoms['valid'][r0:r1,role]
                if not bool(eligible.any()):
                    continue
                weighted = joint_scores/tau + atoms['log_role_weight'][r0:r1,role][None]
                full_role_lse[q0:q1,role] = torch.logaddexp(full_role_lse[q0:q1,role], torch.logsumexp(weighted,dim=1))
                _update(global_bank,q0,q1,role,global_scores,joint_scores,patches,eligible)
                _update(local_bank,q0,q1,role,local_scores,joint_scores,patches,eligible)
                _update(joint_bank,q0,q1,role,joint_scores,joint_scores,patches,eligible)
    selected_patch,selected_score,sources = _merge_candidate_banks(
        global_bank,local_bank,joint_bank,atoms['valid'].sum(dim=0).clamp_max(8))
    roles = torch.cat((torch.zeros(8,dtype=torch.int64),torch.ones(8,dtype=torch.int64)))[None].expand(16384,-1).clone()
    valid = selected_patch >= 0
    safe_patch = selected_patch.clamp_min(0)
    state_xy = atoms['xy'][safe_patch,roles]
    state_xy[~valid] = 0
    selected_log_mass = selected_score/tau + atoms['log_role_weight'][safe_patch,roles]
    selected_log_mass[~valid] = -torch.inf
    logP0 = F.log_softmax(full_role_lse,dim=1)
    P0 = logP0.exp()
    selected_lse = torch.logsumexp(selected_log_mass.reshape(16384,2,8),dim=2)
    log_b0 = logP0[:,:,None] + selected_log_mass.reshape(16384,2,8) - selected_lse[:,:,None]
    log_b0 = log_b0.reshape(16384,16)
    b0, unary_cost = log_b0.exp(), -log_b0
    alpha = (selected_lse-full_role_lse).exp()
    if not bool(torch.isfinite(full_role_lse).all()) or not bool(torch.isfinite(unary_cost[valid]).all()) or not bool((alpha <= 1+1e-5).all()):
        raise ValueError('Non-finite role mass or retained-mass numerical inconsistency')
    result = dict(b0=b0, unary_cost=unary_cost, log_b0=log_b0, role=roles,
                  refpatch=selected_patch, ref_xy=state_xy, valid=valid, query_xy=query_xy,
                  grid_hw=GRID_HW, P0=P0, alpha=alpha.clamp_max(1),
                  full_role_lse=full_role_lse, selected_role_lse=selected_lse,
                  appearance=selected_score, candidate_source=sources,
                  global_candidate_refpatch=global_bank['patch'],
                  local_candidate_refpatch=local_bank['patch'],
                  joint_candidate_refpatch=joint_bank['patch'],
                  reference_atoms=atoms, reference_unit=reference_unit,
                  work=dict(feature_dim=feature_dim, query_chunk=query_chunk,
                            reference_chunk=reference_chunk,
                            global_matching_mac=4096*4096*feature_dim,
                            local_matching_mac=16384*4096*feature_dim,
                            total_matching_mac=(4096+16384)*4096*feature_dim,
                            encoded_image_calls=0,
                            observation_inputs=6, candidate_budget_per_role=8,
                            tie_rule='FP32 score descending, patch ID ascending; merge global4/local4 then joint supplement',
                            candidate_source_bits='1 global discovery, 2 local discovery, 4 joint supplement',
                            limits='MAC excludes LSE/topk/indexing/geometry. No runtime speed or quality claim.'))
    if return_observation_features:
        result.update(global_unit=global_unit,fine_unit=fine_unit)
    return result
