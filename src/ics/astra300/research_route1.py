"""One frozen route-1 research method, separate from the supplied Astra300.

Reference class contrast -> inexpensive mean guide -> one query SPD solve.
No encoder/model construction, query labels, tuning, vote, seed prior or CRF.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import time
from types import MappingProxyType

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
from scipy.spatial.distance import squareform
from scipy.special import logsumexp
from scipy.stats import rankdata

from . import common

CONFIG = MappingProxyType(dict(work_side=1024, native_grid=(64, 64), native_dimension=1024,
              pure_fg=.9, pure_bg=.1, basis_rank=500, projection_gate=.8,
              foreground_merge_cosine=.6, hard_bg_fraction=.2, lse_temperature=.07,
              bg_weight=.55, guide_alpha=.25, query_neighbors=20, fidelity_floor=.1,
              graph_lambda=16., output_threshold=.5, norm_floor=1e-12,
              orthogonal_bg_floor=1e-8, flat_field_floor=1e-6,
              graph_bandwidth_floor=1e-6, graph_degree_floor=1e-8,
              cg_rtol=1e-7, cg_atol=1e-9, cg_maxiter=300, dot_block=256,
              single_class_quantile=.05))
METHOD_ID = "research_route1_simple"
HUBER_SOURCE_SHA256 = "63918543831c4c8a1db98033805a895559fcf77df5d07c2662d558bc3f22773a"
_NATIVE_KINDS = {"frozen_DINOv3_FP32_native_final_LN_patches",
                 "frozen_DINOv3_FP32_native_final_LN_patches_then_unit"}


def _unit(value):
    value = np.asarray(value, np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return np.divide(value, norm, out=np.zeros_like(value), where=norm > CONFIG["norm_floor"])


def _dot(left, right):
    # macOS BLAS can leave stale FP exception flags even for bounded unit
    # vectors. Keep fast BLAS, suppress those flags locally, and reject actual
    # nonfinite results before clipping, fitting or claiming a certificate.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.asarray(left) @ np.asarray(right)
    if not np.isfinite(result).all():
        raise ArithmeticError("Dense dot produced a nonfinite actual result")
    return result


def _weighted_mean(value, weight):
    weight = np.asarray(weight, np.float64)
    return np.sum(np.asarray(value, np.float64) * weight[:, None], axis=0) / max(float(weight.sum()), CONFIG["norm_floor"])


def _synthetic(ep):
    return ep.producer.get("kind") == "synthetic_unit_fixture" and ep.producer.get("test_only") is True


def _native(ep):
    common.validate(ep)
    if any(len(hw) != 2 or any(int(value) != value or value < 1 for value in hw) for hw in (ep.q_hw, ep.r_hw)):
        raise ValueError("Positive native grid dimensions required")
    if ep.q.dtype not in (np.dtype("float32"), np.dtype("float64")) or ep.r.dtype != ep.q.dtype:
        raise ValueError("FP32 native features or their explicit FP64 unit export required; no FP16")
    if not _synthetic(ep):
        if ep.producer.get("kind") not in _NATIVE_KINDS or ep.producer.get("FoRIS_Part1_applied") is not False:
            raise common.Unavailable("Route1 requires unprocessed native FP32 final-LN provenance")
        assets = ep.producer.get("model_assets", ep.producer)
        architecture = ep.producer.get("architecture", assets.get("architecture"))
        if architecture is not None and architecture != "vit_large_patch16_dinov3":
            raise common.Unavailable("The input is not the fixed DINOv3-L/16 architecture")
        digest = assets.get("checkpoint_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise common.Unavailable("Known native frozen checkpoint SHA256 required")
        if ep.q_hw != (64, 64) or ep.r_hw != (64, 64) or ep.q.shape[1] != 1024 or ep.producer.get("model_input_side") != 1024:
            raise ValueError("The fixed production protocol is DINOv3-L/16 at 1024")


def _axis_area(original, native, side=None, resized=None, offset=0):
    """Exact original-pixel overlap with each physical native patch interval."""
    if side is None:
        boundaries = np.arange(native + 1, dtype=float) * original / native
        valid = np.ones(native)
    else:
        canvas = np.arange(native + 1, dtype=float) * side / native
        clipped = np.clip(canvas - offset, 0, resized)
        boundaries = clipped * original / resized
        valid = np.diff(clipped) / (side / native)
    rows, cols, data = [], [], []
    for j, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if hi <= lo:
            continue
        ids = np.arange(max(0, int(np.floor(lo))), min(original, int(np.ceil(hi))))
        mass = np.maximum(0., np.minimum(ids + 1., hi) - np.maximum(ids, lo)) / (hi - lo)
        rows.extend([j] * len(ids)); cols.extend(ids.tolist()); data.extend(mass.tolist())
    return sparse.csr_matrix((data, (rows, cols)), shape=(native, original)), valid


def _geometry(hw, original, geometry):
    if not geometry:
        ay, vy = _axis_area(original[0], hw[0]); ax, vx = _axis_area(original[1], hw[1])
    else:
        side = float(geometry["view_side"])
        sh, sw = map(float, geometry["resized_hw"]); oy, ox = map(float, geometry["padding_top_left"])
        if (side != 1024 or any(value != int(value) for value in (sh, sw, oy, ox))
                or min(sh, sw) <= 0 or min(oy, ox) < 0 or oy + sh > side or ox + sw > side):
            raise ValueError("Recorded 1024 physical resize/pad geometry required")
        if "original_hw" in geometry and tuple(geometry["original_hw"]) != tuple(original):
            raise ValueError("Original pixels disagree with recorded geometry")
        ay, vy = _axis_area(original[0], hw[0], side, sh, oy)
        ax, vx = _axis_area(original[1], hw[1], side, sw, ox)
    return ay, ax, (vy[:, None] * vx[None, :]).ravel()


def reference_footprint(ep):
    """Use actual complete MR pixels, never infer a footprint from packed wf."""
    mask = ep.reference_mask
    if mask is None:
        mask = common.artifact(ep, "reference_mask")
    mask = np.asarray(mask)
    if mask.ndim != 2 or min(mask.shape) < 1 or not np.isin(mask, (0, 1, 255)).all():
        raise ValueError("Actual complete original binary reference mask required")
    mask = mask != 0
    if ep.r_rgb is not None and ep.r_rgb.shape[:2] != mask.shape:
        raise ValueError("Reference RGB and complete MR dimensions differ")
    ay, ax, valid = _geometry(ep.r_hw, mask.shape, ep.reference_geometry)
    coverage = np.asarray(ax @ (ay @ mask.astype(float)).T).T.ravel()
    coverage = np.clip(coverage, 0., 1.)
    foreground = coverage * valid
    _, _, query_valid = _geometry(ep.q_hw, ep.original_shape, ep.query_geometry)
    info = dict(reference_coverage_recomputed=True, reference_mask_array_sha256=common.array_hash(mask),
                actual_reference_valid_mass=float(valid.sum()), actual_reference_fg_mass=float(foreground.sum()),
                packed_foreground_weight_max_difference=float(np.max(np.abs(foreground - ep.wf))),
                packed_reference_valid_max_difference=float(np.max(np.abs(valid - ep.wvalid))),
                packed_query_valid_max_difference=float(np.max(np.abs(query_valid - ep.q_valid))))
    return coverage, foreground, valid, query_valid, info


@lru_cache(maxsize=4)
def _basis_defect(digest, shape, matrix_bytes):
    basis = np.frombuffer(matrix_bytes, np.float32).reshape(shape).astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(_dot(basis.T, basis))
    return float(np.max(np.abs(eigenvalues - 1.)))


def positional_basis(ep):
    basis, producer = common.require_artifacts(ep, "positional_basis", "positional_basis_producer")
    basis = np.asarray(basis)
    if basis.dtype != np.float32 or basis.shape != (ep.q.shape[1], CONFIG["basis_rank"]) or not np.isfinite(basis).all():
        raise ValueError("Unchanged native FP32 d-by-500 positional basis required")
    if not isinstance(producer, dict):
        raise common.Unavailable("Explicit positional basis producer binding required")
    digest = common.array_hash(basis)
    if producer.get("basis_array_sha256") != digest:
        raise common.Unavailable("Positional basis content does not match its producer")
    expected = dict(ep.producer.get("model_assets", ep.producer))
    expected["config_sha256"] = expected.get("config_sha256", expected.get("model_config_sha256"))
    for key in ("checkpoint_sha256", "config_sha256"):
        if not expected.get(key) or producer.get(key) != expected[key]:
            raise common.Unavailable("Positional basis model identity differs: " + key)
    if _synthetic(ep):
        if producer.get("synthetic_test_only") is not True or producer.get("real_encoder_execution") is not False:
            raise common.Unavailable("Synthetic fixture basis must explicitly deny real encoder execution")
    elif (producer.get("state") != "NATIVE_BASIS_FROZEN" or producer.get("source_input") != "normalized_black_image"
          or producer.get("real_encoder_execution") is not True or producer.get("model_input_side") != 1024
          or producer.get("synthetic_test_only", False)):
        raise common.Unavailable("A real checkpoint-bound normalized-black-image native basis is required")
    basis = np.ascontiguousarray(basis).copy()
    defect = _basis_defect(digest, basis.shape, basis.tobytes())
    if defect >= .01:
        raise ValueError("Native numerical basis is not near-unit/well-conditioned")
    return basis, dict(positional_basis_array_sha256=digest, positional_basis_producer=dict(producer),
                       basis_gram_spectral_defect=defect, basis_preserved_without_QR=True)


def foreground_prototypes(reference, pure, valid):
    rows = np.flatnonzero(pure)
    if not len(rows):
        raise ValueError("The caller must use coverage mean when pure F is absent")
    if len(rows) == 1:
        labels = np.zeros(1, int)
    else:
        # squareform avoids allocating two O(M_F^2) integer triangle arrays.
        similarity = _dot(np.asarray(reference[rows], np.float64), np.asarray(reference[rows], np.float64).T)
        distance = np.maximum(0., 1. - np.clip(similarity, -1., 1.)); np.fill_diagonal(distance, 0.)
        hierarchy = linkage(squareform(distance, checks=False), method="average")
        cutoff = np.nextafter(1. - CONFIG["foreground_merge_cosine"], -np.inf)
        labels = fcluster(hierarchy, t=cutoff, criterion="distance")
    centers = [_unit(_weighted_mean(reference[rows[labels == j]], valid[rows[labels == j]])) for j in np.unique(labels)]
    return np.stack(centers), dict(pure_fg_tokens=len(rows), reference_fg_clusters=len(centers),
                                   foreground_clustering="complete average-linkage, strict cosine > .6",
                                   reference_cluster_count_unbounded_by_K8=True)


def exact_query_graph(query, valid, block=256):
    """Full-query exact mutual20NN; ties choose smaller active/native row ID."""
    query = np.asarray(query, np.float32); n = len(query); k = min(20, max(0, n - 1))
    if not k:
        return sparse.csr_matrix((n, n)), dict(query_knn_k=k, graph_edges=0, graph_components=n, graph_isolated_tokens=n)
    indices = np.empty((n, k), int); values = np.empty((n, k), np.float64)
    for first in range(0, n, block):
        score = np.clip(_dot(query[first:first + block], query.T), -1., 1.)
        score[np.arange(len(score)), np.arange(first, first + len(score))] = -np.inf
        for offset, row in enumerate(score):
            selected = np.argpartition(-row, k - 1)[:k]
            cutoff = float(row[selected].min())
            greater = np.flatnonzero(row > cutoff); equal = np.flatnonzero(row == cutoff)
            selected = np.r_[greater, equal[:k - len(greater)]]
            selected = selected[np.lexsort((selected, -row[selected]))]
            indices[first + offset] = selected; values[first + offset] = row[selected]
    distance = np.maximum(0., 1. - values)
    directed = np.exp(-distance / np.maximum(distance[:, -1:], CONFIG["graph_bandwidth_floor"]))
    weights = sparse.csr_matrix((directed.ravel(), (np.repeat(np.arange(n), k), indices.ravel())), shape=(n, n))
    weights = weights.multiply(weights.T); weights.data = np.sqrt(weights.data)
    degree = np.asarray(weights.sum(1)).ravel()
    normalization = float(np.average(degree, weights=valid))
    weights = (weights / max(normalization, CONFIG["graph_degree_floor"])).tocsr()
    components, _ = connected_components(weights, directed=False)
    return weights, dict(query_knn_k=k, graph_edges=int(weights.nnz // 2), graph_components=int(components),
                         graph_isolated_tokens=int(np.count_nonzero(degree == 0)),
                         graph_degree_normalizer=normalization, exact_all_query_knn=True,
                         graph_ties="smaller native row ID", graph_block_rows=block)


def solve_spd(y, s, weights, valid, lam=16.):
    """Actual CG certificate for the stipulated quadratic; y on any failure."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    fidelity = .1 + np.abs(2 * s - 1); fidelity /= np.average(fidelity, weights=valid)
    degree = np.asarray(weights.sum(1)).ravel(); laplacian = sparse.diags(degree) - weights
    matrix = sparse.diags(fidelity) + lam * laplacian; rhs = fidelity * y
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    status, error = 0, None
    if lam == 0:
        solved = y.copy()
    else:
        try:
            solved, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
        except (ArithmeticError, RuntimeError, ValueError) as caught:
            status, error, solved = -1, str(caught), y.copy()
    residual = float(np.linalg.norm(matrix @ solved - rhs)) if np.isfinite(solved).all() else float("inf")
    tolerance = max(1e-7 * float(np.linalg.norm(rhs)), 1e-9)
    converged = status == 0 and np.isfinite(residual) and residual <= tolerance * (1 + 1e-5)
    z = solved if converged else y.copy()
    def energy(value):
        return float(.5 * np.sum(fidelity * (value - y) ** 2) + .5 * lam * _dot(value, laplacian @ value))
    components, labels = connected_components(weights, directed=False)
    component_error = np.bincount(labels, weights=fidelity * (z - y), minlength=components)
    asymmetry = matrix - matrix.T
    certificate = dict(cg_status=int(status), cg_iterations=iterations[0], converged=bool(converged),
                       absolute_residual=residual if np.isfinite(residual) else None,
                       relative_residual=residual / max(float(np.linalg.norm(rhs)), 1e-12) if np.isfinite(residual) else None,
                       stopping_tolerance=tolerance, minimum_fidelity=float(fidelity.min()),
                       matrix_symmetry_max_error=float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.,
                       nonnegative_graph_weights=bool(np.all(weights.data >= 0)), spd_by_positive_fidelity_and_laplacian=True,
                       initial_energy=energy(y), returned_energy=energy(z), fallback_y=not bool(converged),
                       component_weighted_mass_max_error=float(np.max(np.abs(component_error))) if components else 0.,
                       maximum_principle_error=float(max(0., z.max() - y.max(), y.min() - z.min())),
                       solver_error=error)
    return z, certificate


def _result(ep, field, threshold, info, begin):
    value = np.asarray(field, np.float32).reshape(ep.q_hw)
    if not np.isfinite(value).all():
        raise ArithmeticError("Finite complete route1 field required")
    return common.Result(value, threshold, info=dict(info, config=dict(CONFIG), method=METHOD_ID,
        route="claim1_simple_efficiency", counts_as_astra300=False, quality="unknown", query_GT_read=False,
        synthetic_test_only=_synthetic(ep),
        new_encoder_forwards=0, encoder_constructed_or_called=False, cached_method_wall_seconds=time.perf_counter() - begin,
        renderer="one continuous original-size physical bilinear, strict threshold",
        native_field_sha256=common.array_hash(value)))


def _run(ep, alpha=.25, lam=16., single_fg=False, control=None):
    begin = time.perf_counter(); _native(ep)
    coverage, wf, vr, vq, info = reference_footprint(ep)
    basis, receipt = positional_basis(ep); info.update(receipt)
    q, r = np.asarray(ep.q, np.float32).copy(), np.asarray(ep.r, np.float32).copy()
    qi = np.flatnonzero(vq > 0); ri = np.flatnonzero(vr > 0)
    info.update(input_unit_export_dtype=str(ep.q.dtype), reference_active_tokens=len(ri), query_active_tokens=len(qi),
                synthetic_test_only=_synthetic(ep), guide_alpha=alpha, graph_lambda=lam,
                single_fg_control=single_fg, control=control)
    full = np.zeros(len(q), float)
    if wf.sum() <= 0 or not len(qi):
        info["fallback"] = "empty_reference_foreground" if wf.sum() <= 0 else "empty_query_physical_support"
        return _result(ep, full, .5, info, begin)
    mu_raw = _unit(_weighted_mean(r, wf)); query_mean = _weighted_mean(q[qi], vq[qi])
    semantic = float(_dot(mu_raw, _unit(query_mean))); projected = semantic < .8
    if projected:
        b = basis.astype(np.float64)
        q = _unit(q.astype(float) - _dot(_dot(q.astype(float), b), b.T)).astype(np.float32)
        r = _unit(r.astype(float) - _dot(_dot(r.astype(float), b), b.T)).astype(np.float32)
    info.update(semantic_projection_gate=semantic, positional_projection_applied=projected)
    mu_f = _unit(_weighted_mean(r, wf)); wb = vr - wf
    if wb.sum() <= 0:
        ids = np.flatnonzero(wf > 0); total = np.sum(r[ids] * wf[ids, None], axis=0)
        if len(ids) < 2:
            info["fallback"] = "single_class_insufficient_leave_one_out"
            return _result(ep, full, .5, info, begin)
        loo = _unit(total[None] - r[ids] * wf[ids, None])
        support = np.sum(r[ids] * loo, axis=1); threshold = float(np.quantile(support, .05))
        full[qi] = _dot(q[qi], mu_f) - threshold
        info.update(fallback="single_class_source_LOO5", source_LOO_threshold=threshold)
        return _result(ep, full, 0., info, begin)
    mu_b = _unit(_weighted_mean(r, wb))
    pure_f = (vr > 0) & (coverage >= .9); pure_b = (vr > 0) & (coverage <= .1)
    if not pure_b.any():
        full[qi] = _dot(q[qi], mu_f - mu_b); info["fallback"] = "no_pure_background_mean_difference"
        return _result(ep, full, 0., info, begin)
    if single_fg or not pure_f.any():
        centers = mu_f[None]
        info.update(pure_fg_tokens=int(pure_f.sum()), reference_fg_clusters=1,
                    foreground_clustering="coverage-weighted single mean", missing_pure_F=not bool(pure_f.any()))
    else:
        centers, cluster_info = foreground_prototypes(r, pure_f, vr); info.update(cluster_info)
    bg = np.flatnonzero(pure_b); ordering = np.lexsort((bg, -_dot(r[bg], mu_f)))
    hard = bg[ordering[:max(1, int(.2 * len(bg)))]]
    mu_h = _unit(_weighted_mean(r[hard], wb[hard])); orthogonal = mu_h - float(_dot(mu_h, mu_f)) * mu_f
    orthogonal_norm = float(np.linalg.norm(orthogonal))
    ub = orthogonal / orthogonal_norm if orthogonal_norm > 1e-8 else np.zeros_like(mu_f)
    info.update(pure_background_tokens=len(bg), hard_background_tokens=len(hard),
                hard_background_native_ids=hard.tolist(), orthogonal_background_norm=orthogonal_norm,
                zero_orthogonal_background=orthogonal_norm <= 1e-8)
    response = .07 * logsumexp(_dot(q[qi], centers.T) / .07, axis=1) - .55 * _dot(q[qi], ub)
    span = float(response.max() - response.min())
    if span <= 1e-6:
        full[qi] = _dot(q[qi], mu_f - mu_b); info.update(fallback="flat_response_mean_difference", response_span=span)
        return _result(ep, full, 0., info, begin)
    s = (response - response.min()) / span; guide = _dot(q[qi], mu_f)
    rank = lambda value: (rankdata(value, method="average") - .5) / len(value)
    y = s + alpha * (rank(guide) - rank(s))
    weights, graph_info = exact_query_graph(q[qi], vq[qi]); info.update(graph_info)
    z, certificate = solve_spd(y, s, weights, vq[qi], lam=lam); full[qi] = z
    info.update(response_span=span, solver_certificate=certificate,
                response_sha256=common.array_hash(response), target_sha256=common.array_hash(y),
                query_graph_values_sha256=common.array_hash(weights.data),
                target_minimum=float(y.min()), target_maximum=float(y.max()), no_post_solve_minmax=True)
    return _result(ep, full, .5, info, begin)


def predict(ep):
    return _run(ep)


def alpha0_control(ep):
    return _run(ep, alpha=0., control="research_route1_alpha0")


def lambda0_control(ep):
    return _run(ep, lam=0., control="research_route1_lambda0")


def single_fg_control(ep):
    return _run(ep, single_fg=True, control="research_route1_single_fg")


def raw_reference_nn_control(ep):
    begin = time.perf_counter(); _native(ep)
    coverage, wf, vr, vq, info = reference_footprint(ep)
    query = np.asarray(ep.q, np.float32); reference = np.asarray(ep.r, np.float32)
    qids, rids = np.flatnonzero(vq > 0), np.flatnonzero(vr > 0); field = np.zeros(len(query))
    if wf.sum() > 0 and len(rids):
        for first in range(0, len(qids), 256):
            ids = qids[first:first + 256]; score = _dot(query[ids], reference[rids].T)
            field[ids] = coverage[rids[score.argmax(1)]] >= .5
    info.update(control="all_valid_reference_raw_cosine_NN", all_reference_tokens=len(rids),
                reference_label="actual footprint coverage >= .5", ties="first native reference row", positional_projection_applied=False)
    return _result(ep, field, .5, info, begin)


def raw_huber_control(ep):
    begin = time.perf_counter(); _native(ep)
    _, wf, vr, vq, info = reference_footprint(ep)
    from ics.cpu100 import invariance_support
    source = Path(invariance_support.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != HUBER_SOURCE_SHA256:
        raise common.Unavailable("Existing raw Huber helper changed from the frozen source binding")
    bound = replace(common.as_episode(ep), wf=wf, wvalid=vr, q_valid=vq)
    out = invariance_support.huber_reference_readout(bound)
    info.update(out.info, control="existing_raw_Huber_reference_readout", source_helper_sha256=digest,
                positional_projection_applied=False, source_renderer_replaced_by_Astra_once=True,
                raw_huber_preserves_input_unit_export_precision=True)
    return _result(ep, out.margin, 0., info, begin)


METHODS = {METHOD_ID: predict}
CONTROLS = {"research_route1_alpha0": alpha0_control, "research_route1_lambda0": lambda0_control,
            "research_route1_single_fg": single_fg_control, "research_route1_full_reference_NN_raw": raw_reference_nn_control,
            "research_route1_Huber_raw": raw_huber_control}
