"""Fixed600 source-role neighbour binding, complete saved-S conflict readout.

Pure per-episode math: no IO, encoder, query GT, task/domain selector or fitting.
Role-conditioned KDE and adjacency are standard constructions, not originality claims.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch


TAU = .07
ROLE_COUNT = 128
NATIVE_HW = (64, 64)
NODE_COUNT = 4096
CHUNK_NODES = 128                 # Memory blocking only, not an algorithm parameter.
VALIDATION_NODES = 32
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))
FIELDS = ("full", "factorized", "no_query_neighbor_role", "role_marginal_shuffled")
ARMS = ("source_role_edge_complete", "source_role_edge_factorized_complete",
        "source_role_edge_no_query_neighbor_role_complete",
        "source_role_edge_role_marginal_shuffled_complete",
        "same_edit_budget_foris_margin_from_S")


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _neighbors():
    ids = np.arange(NODE_COUNT).reshape(NATIVE_HW)
    rr, cc = np.divmod(ids.ravel(), NATIVE_HW[1])
    out = np.full((NODE_COUNT, len(OFFSETS)), -1, dtype=np.int64)
    for k, (dr, dc) in enumerate(OFFSETS):
        r, c = rr+dr, cc+dc
        legal = (r >= 0) & (r < 64) & (c >= 0) & (c < 64)
        out[legal, k] = r[legal]*64+c[legal]
    return out


def _midpoints(weights):
    mass = float(weights.sum())
    return np.searchsorted(np.cumsum(weights),
                           (np.arange(ROLE_COUNT)+.5)*mass/ROLE_COUNT,
                           side="right").astype(np.int64)


def _canonical_marginal(nf, nb, degree, p_integer, neighbor_ids):
    rows = np.column_stack((nf, nb, degree, p_integer, neighbor_ids)).astype(np.int64)
    order = np.lexsort(tuple(rows[:, k] for k in range(rows.shape[1]-1, -1, -1)))
    return rows[order]


def _source_edges(p, neighbors):
    mf, mb = float(p.sum()), float((1-p).sum())
    stats = {"FG_mass": mf, "BG_mass": mb, "role_samples_each": ROLE_COUNT,
             "available": mf > 0 and mb > 0}
    if not stats["available"]:
        stats["reason"] = "reference_both_roles_required"
        return None, stats
    fi, bi = _midpoints(p), _midpoints(1-p)
    nf, nb = np.bincount(fi, minlength=NODE_COUNT), np.bincount(bi, minlength=NODE_COUNT)
    center_ids = np.flatnonzero(nf+nb)
    center, neighbor, degree = [], [], []
    for i in center_ids:
        adjacent = neighbors[i][neighbors[i] >= 0]
        center.extend([int(i)]*len(adjacent))
        neighbor.extend(int(j) for j in adjacent)
        degree.extend([len(adjacent)]*len(adjacent))
    center, neighbor, degree = (np.asarray(v, dtype=np.int64) for v in (center, neighbor, degree))
    nfe, nbe = nf[center], nb[center]
    wf = nfe.astype(np.float64)/(ROLE_COUNT*degree)
    wb = nbe.astype(np.float64)/(ROLE_COUNT*degree)
    p_integer = np.rint(p[neighbor]*256).astype(np.int64)
    assert abs(float(wf.sum())-1) < 1e-14 and abs(float(wb.sum())-1) < 1e-14
    groups = {}
    for k, key in enumerate(zip(nfe, nbe, degree, p_integer)):
        groups.setdefault(tuple(int(v) for v in key), []).append(k)
    permutation = np.arange(len(center), dtype=np.int64)
    for members in groups.values():
        permutation[members] = members[::-1]
    shuffled_neighbor = neighbor[permutation]
    assert np.array_equal(nfe, nfe[permutation]) and np.array_equal(nbe, nbe[permutation])
    assert np.array_equal(degree, degree[permutation])
    assert np.array_equal(p_integer, p_integer[permutation])
    assert np.array_equal(wf, wf[permutation]) and np.array_equal(wb, wb[permutation])
    original_marginal = _canonical_marginal(nfe, nbe, degree, p_integer, neighbor)
    shuffled_marginal = _canonical_marginal(nfe, nbe, degree, p_integer, shuffled_neighbor)
    assert np.array_equal(original_marginal, shuffled_marginal)
    descriptor_changed = neighbor != shuffled_neighbor
    stats.update(
        reason=None, unique_centers=int(len(center_ids)), edge_count=int(len(center)),
        FG_index_sha256=_hash(fi), BG_index_sha256=_hash(bi),
        centers_sha256=_hash(center), neighbors_sha256=_hash(neighbor),
        FG_weight_sha256=_hash(wf), BG_weight_sha256=_hash(wb),
        FG_weight_sum_CPU64=float(wf.sum()), BG_weight_sum_CPU64=float(wb.sum()),
        neighbor_coverage_integer_sha256=_hash(p_integer),
        source_edge_order="row-major unique center, then OFFSETS order, legal edges only",
        mixed_role_centers=int(np.count_nonzero((nf[center_ids] > 0) & (nb[center_ids] > 0))),
        shuffle={"rule": "reverse within exact (nF,nB,degree,p_neighbor_integer_0to256) group",
                 "group_count": len(groups), "singleton_groups": sum(len(v) == 1 for v in groups.values()),
                 "permutation_sha256": _hash(permutation),
                 "descriptor_index_changed_edges": int(descriptor_changed.sum()),
                 "descriptor_index_preserved_edges": int((~descriptor_changed).sum()),
                 "permutation_changed_edges": int(np.count_nonzero(permutation != np.arange(len(center)))),
                 "FG_weight_mass_changed_descriptor": float(wf[descriptor_changed].sum()),
                 "BG_weight_mass_changed_descriptor": float(wb[descriptor_changed].sum()),
                 "original_conditional_role_marginal_sha256": _hash(original_marginal),
                 "shuffled_conditional_role_marginal_sha256": _hash(shuffled_marginal),
                 "conditional_role_marginals_preserved": True,
                 "center_role_weight_neighbor_label_unchanged": True})
    return {"center": center, "neighbor": neighbor, "shuffled_neighbor": shuffled_neighbor,
            "wf": wf, "wb": wb, "p_neighbor": p[neighbor]}, stats


def _unit(raw, device):
    values = np.asarray(raw)
    if values.shape != (NODE_COUNT, 1024) or not np.isfinite(values).all():
        raise ValueError("Invalid frozen raw O24 input")
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    norm = torch.linalg.vector_norm(tensor, dim=1)
    if not bool((torch.isfinite(norm).all() & (norm > 0).all()).item()):
        raise ValueError("Zero/nonfinite raw token")
    return tensor/norm[:, None]


def _sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def _take(kernel, rows, columns, device):
    ri = torch.as_tensor(rows, dtype=torch.long, device=device)
    ci = torch.as_tensor(columns, dtype=torch.long, device=device)
    return kernel[ri][:, ci]


def _densities(k1, k2, shuffled_k2, compatibility, wf, wb):
    """All sums/exp and products here are production float32 on the given backend."""
    def sums(v):
        return (v*wf).sum(dim=1), (v*wb).sum(dim=1)
    center_f, center_b = sums(k1)
    neighbor_f, neighbor_b = sums(k2*compatibility)
    full_f, full_b = sums(k1*k2*compatibility)
    no_role_f, no_role_b = sums(k1*k2)
    shuffled_f, shuffled_b = sums(k1*shuffled_k2*compatibility)
    ff = torch.stack((full_f, center_f*neighbor_f, no_role_f, shuffled_f))
    bb = torch.stack((full_b, center_b*neighbor_b, no_role_b, shuffled_b))
    return ff, bb


def _field_stats(f, b, bad_edges, bad_nodes):
    stats = {}
    for k, field in enumerate(FIELDS):
        finite_f, finite_b = f[k][np.isfinite(f[k])], b[k][np.isfinite(b[k])]
        stats[field] = {"reason": "nonpositive_or_nonfinite_edge_density" if bad_edges[k] else None,
                        "density_min_F": float(finite_f.min()) if len(finite_f) else None,
                        "density_min_B": float(finite_b.min()) if len(finite_b) else None,
                        "density_max_F": float(finite_f.max()) if len(finite_f) else None,
                        "density_max_B": float(finite_b.max()) if len(finite_b) else None,
                        "nonfinite_F_edges": int(np.count_nonzero(~np.isfinite(f[k]))),
                        "nonfinite_B_edges": int(np.count_nonzero(~np.isfinite(b[k]))),
                        "nonpositive_F_edges": int(np.count_nonzero(np.isfinite(f[k]) & (f[k] <= 0))),
                        "nonpositive_B_edges": int(np.count_nonzero(np.isfinite(b[k]) & (b[k] <= 0))),
                        "invalid_Q_edges": int(bad_edges[k]), "invalid_Q_nodes": int(bad_nodes[k].sum()),
                        "fallback_scope": "entire complete arm retains S" if bad_edges[k] else None}
    return stats


def _cpu64_validation(raw_r, raw_q, edges, s, neighbors, produced_f, produced_b, edge_order):
    """32 fixed row-major Q nodes only; report scalars, never choose precision/masks."""
    start = time.monotonic()
    source_ids = np.unique(np.concatenate((edges["center"], edges["neighbor"])))
    query_ids = np.unique(np.concatenate((np.arange(VALIDATION_NODES),
                                          neighbors[:VALIDATION_NODES][neighbors[:VALIDATION_NODES] >= 0])))
    r = np.asarray(raw_r, dtype=np.float64)[source_ids]
    q = np.asarray(raw_q, dtype=np.float64)[query_ids]
    r = r/np.linalg.norm(r, axis=1, keepdims=True)
    q = q/np.linalg.norm(q, axis=1, keepdims=True)
    kernel = np.exp((q @ r.T-1)/TAU)
    cc = np.searchsorted(source_ids, edges["center"])
    nc = np.searchsorted(source_ids, edges["neighbor"])
    sc = np.searchsorted(source_ids, edges["shuffled_neighbor"])
    ff, bb = [], []
    for i, j in edge_order:
        k1, k2, ks = kernel[np.searchsorted(query_ids, i), cc], kernel[np.searchsorted(query_ids, j), nc], kernel[np.searchsorted(query_ids, j), sc]
        compatibility = s[j]*edges["p_neighbor"]+(1-s[j])*(1-edges["p_neighbor"])
        wf, wb = edges["wf"], edges["wb"]
        ff.append((np.sum(wf*k1*k2*compatibility), np.sum(wf*k1)*np.sum(wf*k2*compatibility),
                   np.sum(wf*k1*k2), np.sum(wf*k1*ks*compatibility)))
        bb.append((np.sum(wb*k1*k2*compatibility), np.sum(wb*k1)*np.sum(wb*k2*compatibility),
                   np.sum(wb*k1*k2), np.sum(wb*k1*ks*compatibility)))
    rf, rb = np.asarray(ff, dtype=np.float64).T, np.asarray(bb, dtype=np.float64).T
    valid_prod = np.isfinite(produced_f) & np.isfinite(produced_b) & (produced_f > 0) & (produced_b > 0)
    valid_ref = np.isfinite(rf) & np.isfinite(rb) & (rf > 0) & (rb > 0)
    node_ids = np.asarray([i for i, _ in edge_order], dtype=np.int64)
    results = {}
    for k, field in enumerate(FIELDS):
        valid = valid_prod[k] & valid_ref[k]
        finite_f = np.isfinite(produced_f[k]) & np.isfinite(rf[k])
        finite_b = np.isfinite(produced_b[k]) & np.isfinite(rb[k])
        ref_log = np.full(len(node_ids), np.nan)
        prod_log = np.full(len(node_ids), np.nan)
        ref_log[valid] = np.log(rf[k, valid])-np.log(rb[k, valid])
        prod_log[valid] = np.log(produced_f[k, valid])-np.log(produced_b[k, valid])
        native_delta, sign_difference = [], 0
        for i in range(VALIDATION_NODES):
            selected = node_ids == i
            if np.all(valid[selected]):
                expected, actual = float(ref_log[selected].mean()), float(prod_log[selected].mean())
                native_delta.append(actual-expected)
                sign_difference += int((actual > 0) != (expected > 0) or (actual < 0) != (expected < 0))
        delta = np.asarray(native_delta, dtype=np.float64)
        results[field] = {"max_abs_F_density_error": float(np.max(np.abs(produced_f[k, finite_f]-rf[k, finite_f]))) if np.any(finite_f) else None,
                          "max_abs_B_density_error": float(np.max(np.abs(produced_b[k, finite_b]-rb[k, finite_b]))) if np.any(finite_b) else None,
                          "nonfinite_F_comparisons": int(np.count_nonzero(~finite_f)),
                          "nonfinite_B_comparisons": int(np.count_nonzero(~finite_b)),
                          "edge_validity_difference": int(np.count_nonzero(valid_prod[k] != valid_ref[k])),
                          "compared_native_nodes": len(delta),
                          "max_abs_native_logratio_error": float(np.max(np.abs(delta))) if len(delta) else None,
                          "RMS_native_logratio_error": float(np.sqrt(np.mean(delta*delta))) if len(delta) else None,
                          "native_strict_zero_sign_difference": sign_difference}
    return {"fixed_Q_nodes": VALIDATION_NODES, "node_index_sha256": _hash(np.arange(VALIDATION_NODES, dtype=np.int64)),
            "fields": results, "CPU64_seconds": time.monotonic()-start,
            "extra_mask_or_score_arm": False, "precision_selection": False}


def compute(raw_r, raw_q, p, s, *, device="mps", validate=False):
    """p=legal R softcoverage; s=whole saved-S CLI 16x16 area fraction.

    S may be entirely foreground/background. A missing R role returns all None.
    Any nonpositive/nonfinite density invalidates that entire field (fallback S).
    Actual backend exceptions propagate so the sole producer can record an
    identical CPU32 retry. No epsilon or log-space substitute changes the math.
    """
    begun = time.monotonic()
    p, s = (np.asarray(v, dtype=np.float64).reshape(NODE_COUNT) for v in (p, s))
    if (not all(np.isfinite(v).all() for v in (p, s))
            or np.any((p < 0) | (p > 1)) or np.any((s < 0) | (s > 1))):
        raise ValueError("Invalid reference/whole-S native area fraction")
    if not np.array_equal(p, np.rint(p*256)/256):
        raise ValueError("Reference p must be exact legal 16x16 integer area/256 for shuffle grouping")
    neighbors = _neighbors()
    edges, source_stats = _source_edges(p, neighbors)
    stats = {"tau": TAU, "role128": ROLE_COUNT, "device": str(device),
             "source": source_stats, "fields": {}, "validation_requested": bool(validate),
             "S_fraction_sha256": _hash(s), "S_fraction_min": float(s.min()),
             "S_fraction_max": float(s.max()), "Q_neighbor_edges": int(np.count_nonzero(neighbors >= 0)),
             "Q_roles_required": False, "query_context": "all saved S CLI native fractions, not pseudo-label fitting"}
    result = {"fields": dict.fromkeys(FIELDS), "stats": stats}
    if edges is None:
        stats["fields"] = {field: {"reason": "reference_both_roles_required", "fallback_scope": "entire complete arm retains S"}
                           for field in FIELDS}
        stats["total_seconds"] = time.monotonic()-begun
        return result
    tick = time.monotonic()
    uq, ur = _unit(raw_q, device), _unit(raw_r, device)
    source_ids = np.unique(np.concatenate((edges["center"], edges["neighbor"])))
    reference_bank = ur[torch.as_tensor(source_ids, dtype=torch.long, device=device)]
    cc = np.searchsorted(source_ids, edges["center"])
    nc = np.searchsorted(source_ids, edges["neighbor"])
    sc = np.searchsorted(source_ids, edges["shuffled_neighbor"])
    wf, wb, pn = (torch.as_tensor(edges[key], dtype=torch.float32, device=device)
                  for key in ("wf", "wb", "p_neighbor"))
    stats["source"]["FG_weight_sum_backend32"] = float(wf.sum().item())
    stats["source"]["BG_weight_sum_backend32"] = float(wb.sum().item())
    _sync(device)
    stats["unit_bank_seconds"] = time.monotonic()-tick
    tick = time.monotonic()
    edge_count = int(np.count_nonzero(neighbors >= 0))
    densities_f = np.empty((len(FIELDS), edge_count), dtype=np.float64)
    densities_b = np.empty_like(densities_f)
    log_sums = np.zeros((len(FIELDS), NODE_COUNT), dtype=np.float64)
    bad_nodes = np.zeros_like(log_sums, dtype=bool)
    bad_edges = np.zeros(len(FIELDS), dtype=np.int64)
    captured_f, captured_b, validation_order = [], [], []
    written = 0
    for begin in range(0, NODE_COUNT, CHUNK_NODES):
        centers = np.arange(begin, min(begin+CHUNK_NODES, NODE_COUNT), dtype=np.int64)
        q_neighbors = neighbors[centers]
        query_ids = np.unique(np.concatenate((centers, q_neighbors[q_neighbors >= 0])))
        q_bank = uq[torch.as_tensor(query_ids, dtype=torch.long, device=device)]
        kernel = torch.exp((q_bank @ reference_bank.T-1)/TAU)
        center_kernel = _take(kernel, np.searchsorted(query_ids, centers), cc, device)
        for slot in range(len(OFFSETS)):
            legal = q_neighbors[:, slot] >= 0
            qi, qj = centers[legal], q_neighbors[legal, slot]
            if not len(qi):
                continue
            select = torch.as_tensor(np.flatnonzero(legal), dtype=torch.long, device=device)
            k1 = center_kernel[select]
            k2 = _take(kernel, np.searchsorted(query_ids, qj), nc, device)
            shuffled_k2 = _take(kernel, np.searchsorted(query_ids, qj), sc, device)
            sj = torch.as_tensor(s[qj], dtype=torch.float32, device=device)[:, None]
            compatibility = sj*pn[None, :]+(1-sj)*(1-pn[None, :])
            tf, tb = _densities(k1, k2, shuffled_k2, compatibility, wf, wb)
            ff, bb = tf.detach().cpu().to(torch.float64).numpy(), tb.detach().cpu().to(torch.float64).numpy()
            stop = written+len(qi)
            densities_f[:, written:stop], densities_b[:, written:stop] = ff, bb
            valid = np.isfinite(ff) & np.isfinite(bb) & (ff > 0) & (bb > 0)
            for k in range(len(FIELDS)):
                good = valid[k]
                log_sums[k, qi[good]] += np.log(ff[k, good])-np.log(bb[k, good])
                bad_nodes[k, qi[~good]] = True
                bad_edges[k] += int((~good).sum())
            if validate:
                chosen = qi < VALIDATION_NODES
                if np.any(chosen):
                    captured_f.append(ff[:, chosen])
                    captured_b.append(bb[:, chosen])
                    validation_order.extend((int(i), int(j)) for i, j in zip(qi[chosen], qj[chosen]))
            written = stop
        del kernel, center_kernel, q_bank
    assert written == edge_count
    _sync(device)
    stats["blocked_kernel_density_seconds"] = time.monotonic()-tick
    tick = time.monotonic()
    stats["fields"] = _field_stats(densities_f, densities_b, bad_edges, bad_nodes)
    degrees = np.count_nonzero(neighbors >= 0, axis=1)
    for k, field in enumerate(FIELDS):
        if bad_edges[k]:
            continue
        native = log_sums[k]/degrees
        if not np.isfinite(native).all():
            stats["fields"][field].update(reason="nonfinite_native_neighbor_mean", fallback_scope="entire complete arm retains S")
            continue
        result["fields"][field] = native.reshape(NATIVE_HW)
        stats["fields"][field].update(native_min=float(native.min()), native_max=float(native.max()),
                                       native_field_sha256=_hash(native), strict_zero_nodes=int(np.count_nonzero(native == 0)))
    stats["CPU64_logs_neighbor_average_seconds"] = time.monotonic()-tick
    if validate:
        stats["truth_free_CPU64_reference"] = _cpu64_validation(
            raw_r, raw_q, edges, s, neighbors, np.concatenate(captured_f, axis=1),
            np.concatenate(captured_b, axis=1), validation_order)
    stats["memory_blocking"] = {"Q_center_nodes": CHUNK_NODES, "unique_source_descriptors": len(source_ids),
                                "no_full4096_by_sourceedge_kernel": True, "dense_archive": False}
    stats["total_seconds"] = time.monotonic()-begun
    return result


def _render(field, shape):
    import torch.nn.functional as F
    tensor = torch.as_tensor(np.ascontiguousarray(field), dtype=torch.float64)[None, None]
    return F.interpolate(tensor, size=tuple(shape), mode="bilinear", align_corners=False)[0, 0].numpy()


def compose_frame(masks, score64, dense):
    """Saved S host; direct signed field only in four-way disagreement."""
    strong, bb, gg, rr, nn = (np.asarray(masks[k], dtype=bool) for k in ("S", "B", "G", "R0", "N"))
    assert strong.ndim == 2 and all(v.shape == strong.shape for v in (bb, gg, rr, nn))
    union, intersection = bb | gg | rr, bb & gg & rr
    da, dd = ~strong & union, strong & ~intersection
    protected = (strong & intersection) | (~strong & ~union)
    outputs, counts, fallback = {}, {}, {}
    for arm, field in zip(ARMS[:-1], FIELDS):
        native = dense["fields"][field]
        if native is None:
            prediction = strong.copy()
            add, delete = np.zeros(strong.shape, bool), np.zeros(strong.shape, bool)
            fallback[arm] = True
        else:
            z = _render(np.asarray(native, dtype=np.float64).reshape(NATIVE_HW), strong.shape)
            assert np.isfinite(z).all()
            add, delete = da & (z > 0), dd & (z < 0)
            prediction = (strong & ~delete) | add
            rebuilt = strong.copy()
            rebuilt[da & (z > 0)] = True
            rebuilt[dd & (z < 0)] = False
            assert np.array_equal(prediction, rebuilt)
            fallback[arm] = False
        assert np.array_equal(prediction[protected], strong[protected])
        assert int((prediction & ~strong).sum()) == int(add.sum())
        assert int((strong & ~prediction).sum()) == int(delete.sum())
        outputs[arm] = prediction
        counts[arm] = {"add": int(add.sum()), "delete": int(delete.sum())}
    main = outputs[ARMS[0]]
    ka, kd = int((main & ~strong).sum()), int((strong & ~main).sum())
    score = np.asarray(score64, dtype=np.float64).reshape(NATIVE_HW)
    assert np.isfinite(score).all()
    margin = 2*_render((score-score.min())/max(float(np.ptp(score)), 1e-6), strong.shape)-1
    budget, rebuilt = strong.copy(), strong.copy()
    for domain, count, descending, value in ((da, ka, True, True), (dd, kd, False, False)):
        ids = np.flatnonzero(domain.ravel())
        assert 0 <= count <= len(ids)
        values = -margin.ravel()[ids] if descending else margin.ravel()[ids]
        rank = np.lexsort((ids, values))
        budget.ravel()[ids[rank[:count]]] = value
        stable = np.argsort(values, kind="stable")
        rebuilt.ravel()[ids[stable[:count]]] = value
    assert np.array_equal(budget, rebuilt)
    assert int((budget & ~strong).sum()) == ka and int((strong & ~budget).sum()) == kd
    assert np.array_equal(budget[protected], strong[protected])
    outputs[ARMS[-1]] = budget
    counts[ARMS[-1]] = {"add": ka, "delete": kd}
    return outputs, {"edit_counts": counts, "fallback": fallback, "edit_counts_relative_host": "S",
                     "D_add": int(da.sum()), "D_delete": int(dd.sum()),
                     "protected_four_way_pixels": int(protected.sum()), "strict_zero_tie_retains_S": True,
                     "no_minmax_tanh_confidence_or_alpha_readout": True,
                     "sameframe_S_budget_positions_verified": True}
