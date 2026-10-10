"""Full reference-role unary on frozen whole/corner O24 observations.

This head retains the original joint-reference observation's normalization,
sampling, chunked FP32 matching and role-LSE reduction order. It omits all
candidate discovery, state truncation and geometry. There is no model, loader,
mask readout or query-label access. Optional role weights are a reference-only
hypothesis; accepting them does not establish segmentation improvement.
"""
from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import joint_reference_observation as observation


def _identity() -> dict[str, Any]:
    source = Path(__file__)
    dependency = Path(observation.__file__)
    return dict(head_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                observation_contract_source_sha256=hashlib.sha256(dependency.read_bytes()).hexdigest(),
                representation='CPU FP32 actual O24;64x64 reference/whole query and four native64x64 quadrant arrays',
                role_order=('background', 'foreground'),
                sampling='raw whole Q bilinear64->128 align_corners=False, then token L2; native token L2 before four-corner placement',
                matching='joint=.5 global cosine+.5 native cosine; original FP32 chunk and LSE order',
                outputs='continuous full-role fields only; no prediction mask')


def _weights(value: Any, atoms: dict[str, torch.Tensor]) -> torch.Tensor:
    if value is None:
        return atoms['log_role_weight']
    weights = torch.as_tensor(value)
    if weights.device.type != 'cpu' or weights.dtype != torch.float32 or tuple(weights.shape) != (4096, 2):
        raise ValueError('log_role_weight must be CPU FP32 [4096,2]')
    valid = atoms['valid']
    if not bool(torch.isfinite(weights[valid]).all()) or not bool(torch.isneginf(weights[~valid]).all()):
        raise ValueError('log_role_weight requires finite legal-role support and -inf outside it')
    normalizer = torch.logsumexp(weights.double(), dim=0)
    if not bool((normalizer.abs() <= 1e-5).all()):
        raise ValueError('log_role_weight must be separately normalized within each role')
    return weights.detach()


@torch.inference_mode()
def build_dense_role_unary(reference_o24: Any, reference_mask_canvas1024: Any,
                          query_global_o24: Any, native4_o24: Sequence[Any], *,
                          log_role_weight: Any = None,
                          boxes: Sequence[Sequence[int]] = observation.FOUR_BOXES,
                          tau: float = .07, query_chunk: int = 256,
                          reference_chunk: int = 512,
                          feature_dim: int = 1024) -> dict[str, Any]:
    """Return P0[N,2] and full_role_lse[N,2] on the fixed128x128 grid.

    All real inputs must be CPU FP32 [4096,1024], finite with nonzero token
    norms. Smaller feature_dim is only an explicit synthetic fixture contract.
    The mask is already aligned to the1024 reference canvas and must contain
    both roles. Supplied log weights preserve exactly that legal role support,
    with each role's exponentiated weights summing to1 within FP32 tolerance.
    Inputs are read only. Default weights use original reference-role areas.
    """
    if tuple(tuple(b) for b in boxes) != observation.FOUR_BOXES or len(native4_o24) != 4:
        raise ValueError('Require exactly the fixed NW/NE/SW/SE native quadrants')
    if not isinstance(feature_dim, int) or feature_dim < 1 or not isinstance(query_chunk, int) or not isinstance(reference_chunk, int) or min(query_chunk, reference_chunk) < 1:
        raise ValueError('Feature dimension and chunk sizes must be positive integers')
    if tau != .07:
        raise ValueError('This unary head freezes tau=.07')
    reference = observation._raw(reference_o24, 'reference_o24', feature_dim)
    global_raw = observation._raw(query_global_o24, 'query_global_o24', feature_dim)
    native = [observation._raw(value, f'native4_o24[{i}]', feature_dim)
              for i, value in enumerate(native4_o24)]
    atoms = observation.reference_role_atoms(reference_mask_canvas1024)
    weights = _weights(log_role_weight, atoms)
    reference_unit = reference / torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    sampled_raw = F.interpolate(global_raw.reshape(64, 64, feature_dim).permute(2, 0, 1)[None],
                                size=observation.GRID_HW, mode='bilinear', align_corners=False)[0].permute(1, 2, 0).reshape(16384, feature_dim).contiguous()
    global_norm = torch.linalg.vector_norm(sampled_raw, dim=1)
    if not bool((torch.isfinite(global_norm) & (global_norm > 0)).all()):
        raise ValueError('Sampled raw global Q contains finite-norm failure or zero vector')
    del sampled_raw
    fine_grid = torch.empty((128, 128, feature_dim), dtype=torch.float32)
    for value, (x0, y0, x1, y1) in zip(native, observation.FOUR_BOXES):
        unit = value / torch.linalg.vector_norm(value, dim=1, keepdim=True)
        fine_grid[y0//8:y1//8, x0//8:x1//8] = unit.reshape(64, 64, feature_dim)
    fine_unit = fine_grid.reshape(16384, feature_dim)
    sample_ids, sample_weights, query_xy = observation._sampling_plan()
    full_role_lse = torch.full((16384, 2), -torch.inf)
    for r0 in range(0, 4096, reference_chunk):
        r1 = min(r0 + reference_chunk, 4096)
        ru = reference_unit[r0:r1]
        coarse_scores = global_raw @ ru.T
        for q0 in range(0, 16384, query_chunk):
            q1 = min(q0 + query_chunk, 16384)
            global_scores = (coarse_scores[sample_ids[q0:q1]] * sample_weights[q0:q1, :, None]).sum(dim=1) / global_norm[q0:q1, None]
            local_scores = fine_unit[q0:q1] @ ru.T
            joint_scores = .5*global_scores + .5*local_scores
            for role in range(2):
                eligible = atoms['valid'][r0:r1, role]
                if not bool(eligible.any()):
                    continue
                weighted = joint_scores/tau + weights[r0:r1, role][None]
                full_role_lse[q0:q1, role] = torch.logaddexp(
                    full_role_lse[q0:q1, role], torch.logsumexp(weighted, dim=1))
    if not bool(torch.isfinite(full_role_lse).all()):
        raise ValueError('Non-finite full reference-role LSE')
    logP0 = F.log_softmax(full_role_lse, dim=1)
    P0 = logP0.exp()
    identity = _identity()
    identity['role_weight_source'] = 'original_reference_area' if log_role_weight is None else 'caller_reference_only_log_role_weight'
    work = dict(feature_dim=feature_dim, query_chunk=query_chunk,
                reference_chunk=reference_chunk,
                global_matching_mac=4096*4096*feature_dim,
                local_matching_mac=16384*4096*feature_dim,
                total_matching_mac=(4096+16384)*4096*feature_dim,
                observation_inputs=6, encoded_image_calls=0,
                candidate_bank_allocations=0, topk_calls=0,
                candidate_merge_calls=0, candidate_state_arrays=0, BP_calls=0,
                limits='MAC excludes LSE/indexing/normalization. No measured runtime speed or quality claim.')
    return dict(P0=P0, logP0=logP0, full_role_lse=full_role_lse,
                log_role_weight=weights, reference_role_coverage=atoms['role_coverage'],
                query_xy=query_xy, grid_hw=observation.GRID_HW,
                work=work, identity=identity)
