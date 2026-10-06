"""V3: choose query-nearest witnesses before reading source-paired contrast.

Only witness selection changes from frozen transport V2. This remains a complete
replacement readout; it does not retain FoRIS background suppression, clustering
or CRF scope inference. Fixing contrast maximization alone need not recover that
missing ability. The exact V2 kNN control is reused without parameter changes.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .transport import (
    CONFIG as V2_CONFIG,
    _normalize,
    _source_calibration,
    _tensor,
    control as _v2_control,
)


CONFIG = {
    **V2_CONFIG,
    "name": "reference_query_selected_paired_contrast_v3",
    "witness_selection": "query_cosine_before_paired_contrast",
    "configuration_status": "fixed_operator_after_first20_diagnosis_no_parameter_search",
    "parent_method": "reference_paired_contrast_v2",
    "retains_complete_foris_scope_inference": False,
}


@torch.inference_mode()
def predict(q, r, cov, score, *, device="cpu", extras=None):
    """Complete 64 x 64 field; query labels and native scores are not used."""
    start = time.perf_counter()
    if extras and any(str(k).lower() in {"gt", "query_gt", "class_id", "target_mask"} for k in extras):
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
    evidence, counts, match_audit = [], {}, {}
    for role, name in [(True, "foreground"), (False, "background")]:
        selected = eligible & (roles == role)
        opposite = torch.nonzero(roles != role, as_tuple=False).flatten()
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        counts[name] = len(indices)
        if not len(indices) or not len(opposite):
            evidence.append(torch.zeros(len(q), device=q.device))
            continue

        nearest = source_sim[indices][:, opposite].topk(
            min(k, len(opposite)), dim=1).indices
        counterparts = r[opposite[nearest]].mean(dim=1)
        query_role_sim = cross_sim[:, indices]

        # The only inference change: V2 chooses top-k of (query_role_sim -
        # counterpart_sim). V3 fixes top-k from query_role_sim first, so a
        # distant witness cannot win merely by having an even more distant
        # reference-conditioned opponent. Counterparts and arithmetic are exact
        # V2 definitions; the chosen witnesses need not yield positive contrast.
        witness_sim, witness_local_indices = query_role_sim.topk(
            min(k, len(indices)), dim=1)
        counterpart_sim = (q @ counterparts.T).gather(1, witness_local_indices)
        role_evidence = (witness_sim - counterpart_sim).mean(dim=1)
        evidence.append(role_evidence)
        match_audit[name] = {
            "selected_witness_mean_cosine": float(witness_sim.mean().item()),
            "selected_counterpart_mean_cosine": float(counterpart_sim.mean().item()),
        }

    margin = evidence[0] - evidence[1]
    field = torch.sigmoid(margin / CONFIG["evidence_temperature"]).reshape(shape)
    result = field.cpu().numpy().astype(np.float32, copy=False)
    return result, {
        "method": CONFIG["name"],
        "seconds": time.perf_counter() - start,
        "device": str(device),
        "eligible_reference_atoms": counts,
        "eligible_reference_fraction": float(eligible.float().mean().item()),
        "reference_separation_median": float(separation.median().item()),
        "missing_role_evidence": [name for name, n in counts.items() if not n],
        "query_positive_evidence_fraction": float((evidence[0] > 0).float().mean().item()),
        "query_negative_evidence_fraction": float((evidence[1] > 0).float().mean().item()),
        "witness_match_audit": match_audit,
        "witness_selection": CONFIG["witness_selection"],
        "uses_native_score_values": False,
        "retains_complete_foris_scope_inference": False,
        "query_labels_used": False,
        "additional_model_forwards": 0,
        "real_efficacy_validated": False,
    }


@torch.inference_mode()
def control(q, r, cov, score, *, device="cpu", extras=None):
    """Exact unchanged V2 direct query kNN contrast control."""
    return _v2_control(q, r, cov, score, device=device, extras=extras)
