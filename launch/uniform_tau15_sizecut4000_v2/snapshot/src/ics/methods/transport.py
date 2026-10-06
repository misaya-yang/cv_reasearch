"""Reference-paired contrast, without transport mass constraints.

The historical module name records the assigned research perspective. This is
not Sinkhorn, balanced/unbalanced OT, a reference-coverage objective, or a native
mask gate. A first absolute-rejection construction rejected every token in the
first real label-free smoke and is retained explicitly for reproduction. V2
compares each witness against its most confusable opposite-role reference
patches. Query truth is never an input.
"""
from __future__ import annotations

import time
import numpy as np
import torch


CONFIG = {
    "name": "reference_paired_contrast_v2",
    "reference_foreground_coverage": 0.5,
    "same_role_neighborhood_exclusion": 2,
    "witness_neighbors": 4,
    "evidence_temperature": 0.05,
    "output_temperature": 0.1,
    "minimum_local_separation": 0.0,
    "additional_model_forwards": 0,
    "feature_layers": ["cached_final"],
    "uses_native_score": False,
    "uses_query_area_prior": False,
    "configuration_status": "v2_fixed_after_label_free_all_reject_v1_smoke",
}


def _tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device).detach()


def _normalize(x):
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("Features must be finite N x D arrays")
    norm = x.norm(dim=1, keepdim=True)
    if (norm < 1e-8).any():
        raise ValueError("Zero-norm features cannot define correspondence")
    return x / norm


def _top_mean(values, allowed, k):
    """Finite top-k mean per row; never let a masked entry enter the mean."""
    values = values.masked_fill(~allowed, -torch.inf)
    vals = values.topk(min(k, values.shape[1]), dim=1).values
    valid = torch.isfinite(vals)
    count = valid.sum(dim=1)
    means = vals.masked_fill(~valid, 0).sum(dim=1) / count.clamp_min(1)
    return means, count


def _source_calibration(r, roles, grid):
    nr = len(r)
    if grid[0] * grid[1] != nr:
        raise ValueError("Reference coverage grid does not match reference tokens")
    coords = torch.stack(torch.meshgrid(
        torch.arange(grid[0], device=r.device),
        torch.arange(grid[1], device=r.device), indexing="ij"), dim=-1).reshape(nr, 2)
    near = (coords[:, None] - coords[None]).abs().amax(dim=-1)
    separated = near > CONFIG["same_role_neighborhood_exclusion"]
    same = roles[:, None] == roles[None]
    sim = r @ r.T
    k = CONFIG["witness_neighbors"]
    positive, n_positive = _top_mean(sim, same & separated, k)
    negative, n_negative = _top_mean(sim, ~same, k)
    # Tiny foregrounds cannot supply a spatially independent same-role witness.
    # They remain explicitly unsupported instead of borrowing adjacent self-copy.
    valid = (n_positive > 0) & (n_negative > 0)
    separation = positive - negative
    valid &= separation > CONFIG["minimum_local_separation"]
    thresholds = (positive + negative) * 0.5
    return thresholds, valid, separation


def _run(q, r, cov, score, *, device, extras, local):
    start = time.perf_counter()
    if extras and any(k.lower() in {"gt", "query_gt", "class_id", "target_mask"} for k in extras):
        raise ValueError("Inference extras must not contain query labels or class identity")
    q = _normalize(_tensor(q, device))
    r = _normalize(_tensor(r, device))
    coverage = _tensor(cov, device)
    if coverage.ndim != 2 or not torch.isfinite(coverage).all():
        raise ValueError("Coverage must be a finite 2-D reference grid")
    if (coverage < 0).any() or (coverage > 1).any():
        raise ValueError("Reference coverage must lie in [0, 1]")
    query_shape = tuple(score.shape)
    if len(query_shape) != 2 or np.prod(query_shape) != len(q):
        raise ValueError("Score grid must match query tokens; its values are unused")
    if q.shape[1] != r.shape[1]:
        raise ValueError("Reference/query dimensions differ")
    roles = coverage.flatten() >= CONFIG["reference_foreground_coverage"]
    thresholds, eligible, separation = _source_calibration(r, roles, coverage.shape)
    sim = q @ r.T
    k = CONFIG["witness_neighbors"]
    evidence = []
    counts = {}
    for role, label in [(True, "foreground"), (False, "background")]:
        selected = eligible & (roles == role)
        counts[label] = int(selected.sum().item())
        if not selected.any():
            evidence.append(torch.zeros(len(q), device=q.device))
            continue
        # The control retains exactly the same eligible atoms, full similarity
        # matrix, top-k operator and null option, but uses one level per role.
        levels = thresholds[selected]
        if not local:
            levels = levels.mean().expand_as(levels)
        values = (sim[:, selected] - levels[None]) / CONFIG["evidence_temperature"]
        support = values.topk(min(k, values.shape[1]), dim=1).values.mean(dim=1)
        evidence.append(support.clamp_min(0))
    margin = evidence[0] - evidence[1]
    output = torch.sigmoid(margin / CONFIG["output_temperature"])
    field = output.reshape(query_shape).cpu().numpy().astype(np.float32, copy=False)
    rejected = (evidence[0] == 0) & (evidence[1] == 0)
    info = {
        "method": "reference_local_rejection_v1" if local else "reference_global_rejection_control_v1",
        "seconds": time.perf_counter() - start,
        "device": str(device),
        "reference_tokens": len(r),
        "query_tokens": len(q),
        "eligible_reference_atoms": counts,
        "eligible_reference_fraction": float(eligible.float().mean().item()),
        "null_fraction": float(rejected.float().mean().item()),
        "reference_separation_median": float(separation.median().item()),
        "missing_role_evidence": [label for label, count in counts.items() if not count],
        "null_tie_rule": "field=0.5; final strict >0.5 renderer assigns background",
        "uses_native_score_values": False,
        "query_labels_used": False,
        "additional_model_forwards": 0,
        "real_efficacy_validated": False,
    }
    return field, info


@torch.inference_mode()
def rejected_absolute_predict(q, r, cov, score, *, device="cpu", extras=None):
    """Reproduce v1, which rejected all tokens in the first label-free smoke."""
    return _run(q, r, cov, score, device=device, extras=extras, local=True)


@torch.inference_mode()
def rejected_absolute_control(q, r, cov, score, *, device="cpu", extras=None):
    """Reproduce the v1 global acceptance-level control."""
    return _run(q, r, cov, score, device=device, extras=extras, local=False)


def _paired(q, r, cov, score, *, device, extras, local):
    start = time.perf_counter()
    if extras and any(k.lower() in {"gt", "query_gt", "class_id", "target_mask"} for k in extras):
        raise ValueError("Inference extras must not contain query labels or class identity")
    q, r = _normalize(_tensor(q, device)), _normalize(_tensor(r, device))
    coverage = _tensor(cov, device)
    if coverage.ndim != 2 or not torch.isfinite(coverage).all():
        raise ValueError("Coverage must be a finite 2-D reference grid")
    if (coverage < 0).any() or (coverage > 1).any():
        raise ValueError("Reference coverage must lie in [0, 1]")
    shape = tuple(score.shape)
    if len(shape) != 2 or np.prod(shape) != len(q):
        raise ValueError("Query score grid must match tokens; values are unused")
    if q.shape[1] != r.shape[1]:
        raise ValueError("Reference/query dimensions differ")
    roles = coverage.flatten() >= CONFIG["reference_foreground_coverage"]
    _, eligible, separation = _source_calibration(r, roles, coverage.shape)
    source_sim, cross_sim = r @ r.T, q @ r.T
    k = CONFIG["witness_neighbors"]
    evidence, counts = [], {}
    for role, name in [(True, "foreground"), (False, "background")]:
        selected = eligible & (roles == role)
        opposite = torch.nonzero(roles != role, as_tuple=False).flatten()
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        counts[name] = len(indices)
        if not len(indices) or not len(opposite):
            evidence.append(torch.zeros(len(q), device=q.device))
            continue
        if local:
            # Each witness competes with its most confusable opposite-role
            # reference appearances. This is relative similarity, not a source
            # absolute distance cutoff transported across the instance gap.
            nearest = source_sim[indices][:, opposite].topk(
                min(k, len(opposite)), dim=1).indices
            counterparts = r[opposite[nearest]].mean(dim=1)
            contrast = cross_sim[:, indices] - q @ counterparts.T
        else:
            # Strong simple same-input control: each query chooses its own
            # nearest opposite-role witnesses, i.e. direct kNN FG/BG contrast.
            nearest_opposite = cross_sim[:, opposite].topk(
                min(k, len(opposite)), dim=1).values.mean(dim=1)
            contrast = cross_sim[:, indices] - nearest_opposite[:, None]
        evidence.append(contrast.topk(min(k, len(indices)), dim=1).values.mean(dim=1))
    margin = evidence[0] - evidence[1]
    field = torch.sigmoid(margin / CONFIG["evidence_temperature"]).reshape(shape)
    result = field.cpu().numpy().astype(np.float32, copy=False)
    return result, {
        "method": CONFIG["name"] if local else "query_nearest_contrast_control_v2",
        "seconds": time.perf_counter() - start,
        "device": str(device),
        "eligible_reference_atoms": counts,
        "eligible_reference_fraction": float(eligible.float().mean().item()),
        "reference_separation_median": float(separation.median().item()),
        "missing_role_evidence": [name for name, n in counts.items() if not n],
        "query_positive_evidence_fraction": float((evidence[0] > 0).float().mean().item()),
        "query_negative_evidence_fraction": float((evidence[1] > 0).float().mean().item()),
        "uses_native_score_values": False,
        "query_labels_used": False,
        "additional_model_forwards": 0,
        "real_efficacy_validated": False,
    }


@torch.inference_mode()
def predict(q, r, cov, score, *, device="cpu", extras=None):
    """Complete paired-reference contrast output, no absolute rejection cutoff."""
    return _paired(q, r, cov, score, device=device, extras=extras, local=True)


@torch.inference_mode()
def control(q, r, cov, score, *, device="cpu", extras=None):
    """Complete direct query kNN foreground/background contrast."""
    return _paired(q, r, cov, score, device=device, extras=extras, local=False)
