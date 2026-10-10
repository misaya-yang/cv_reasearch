"""Reference-neighbor dependence probe with exact marginal-preserving controls.

This module creates complete four-role joint/independent density contrasts and
their unary-free binary interaction on query neighbor edges. It does not create
segmentation masks, optimize a graph, use query labels or claim boundary accuracy.
Actual FF/BB pairs are distinct neighboring reference tokens, never artificial
same-atom pairs. All three controls share the same sampled endpoint marginals.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch


TEMPERATURE = .07
OCCURRENCES_PER_STATE = 64
SHUFFLE_SEED = 20261010
EDGE_CHUNK = 512
ROLE_ORDER = ('background', 'foreground')


def _features(value: Any, name: str) -> torch.Tensor:
    x = torch.as_tensor(value)
    if x.device.type != 'cpu' or x.dtype != torch.float32 or x.ndim != 2:
        raise ValueError(f'{name} must be caller-normalized CPU FP32 [tokens, channels]')
    if not x.shape[0] or not x.shape[1] or not bool(torch.isfinite(x).all()):
        raise ValueError(f'{name} must be nonempty and finite')
    norms = torch.linalg.vector_norm(x, dim=1)
    if not bool((norms > 0).all()) or float((norms-1).abs().max()) > 1e-3:
        raise ValueError(f'{name} must preserve caller unit normalization')
    return x.detach()


def _grid(value: Any, tokens: int, name: str) -> tuple[int, int]:
    if len(value) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in value):
        raise ValueError(f'{name} must contain two positive integers')
    hw = int(value[0]), int(value[1])
    if hw[0]*hw[1] != tokens:
        raise ValueError(f'{name} must match token count')
    return hw


def _neighbors(hw: tuple[int, int]) -> np.ndarray:
    index = np.arange(hw[0]*hw[1], dtype=np.int64).reshape(hw)
    horizontal = np.stack((index[:, :-1].reshape(-1), index[:, 1:].reshape(-1)), axis=1)
    vertical = np.stack((index[:-1, :].reshape(-1), index[1:, :].reshape(-1)), axis=1)
    edges = np.concatenate((horizontal, vertical), axis=0)
    if len(edges):
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return edges


def _mass_quantiles(pairs: np.ndarray, mass: np.ndarray, count: int) -> np.ndarray:
    total = float(mass.sum(dtype=np.float64))
    if total <= 0:
        return np.empty((0, 2), np.int64)
    valid = mass > 0
    pairs, mass = pairs[valid], mass[valid]
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, mass = pairs[order], mass[order]
    cumulative = np.cumsum(mass, dtype=np.float64)
    locations = (np.arange(count, dtype=np.float64)+.5)*cumulative[-1]/count
    ids = np.searchsorted(cumulative, locations, side='left')
    if (ids < 0).any() or (ids >= len(pairs)).any():
        raise RuntimeError('Reference pair mass quantile index failure')
    return pairs[ids].copy()


def _pair_counts(pairs: np.ndarray, tokens: int) -> tuple[np.ndarray, np.ndarray]:
    if not len(pairs):
        return np.zeros(tokens, np.int64), np.zeros(tokens, np.int64)
    return (np.bincount(pairs[:, 0], minlength=tokens),
            np.bincount(pairs[:, 1], minlength=tokens))


def _bank_diagnostic(true: np.ndarray, shuffled: np.ndarray, tokens: int, mass: float) -> dict:
    left, right = _pair_counts(true, tokens)
    null_left, null_right = _pair_counts(shuffled, tokens)
    if not np.array_equal(left, null_left) or not np.array_equal(right, null_right):
        raise RuntimeError('Fixed shuffle changed a reference endpoint marginal')
    if len(true) != len(shuffled) or (true[:, 0] == true[:, 1]).any():
        raise RuntimeError('Reference true pair bank must use distinct physical endpoints')
    true_keys = true[:, 0]*tokens+true[:, 1]
    null_keys = shuffled[:, 0]*tokens+shuffled[:, 1]
    unique, counts = np.unique(true_keys, return_counts=True)
    null_unique, null_counts = np.unique(null_keys, return_counts=True)
    common, true_index, null_index = np.intersect1d(unique, null_unique, return_indices=True)
    unchanged = int(np.minimum(counts[true_index], null_counts[null_index]).sum())
    n = len(true)
    return dict(source_oriented_state_mass=mass, occurrence_count=n,
        unique_physical_true_pairs=len(unique),
        effective_physical_true_pairs=float(n*n/np.square(counts).sum()) if n else 0.,
        true_pair_ids=true.tolist(), fixed_shuffle_pair_ids=shuffled.tolist(),
        left_endpoint_ids=np.flatnonzero(left).tolist(), left_endpoint_counts=left[left > 0].tolist(),
        right_endpoint_ids=np.flatnonzero(right).tolist(), right_endpoint_counts=right[right > 0].tolist(),
        endpoint_weight='each occurrence exactly1/64; selected empirical conditional marginals',
        shuffle_left_marginal_bit_exact=True, shuffle_right_marginal_bit_exact=True,
        pair_multiset_changed_fraction=1-unchanged/n if n else 0.,
        true_self_pair_count=0,
        shuffle_self_pair_count=int((shuffled[:, 0] == shuffled[:, 1]).sum()),
        missing_state_independence_fallback=n == 0)


def _banks(coverage: np.ndarray, hw: tuple[int, int]) -> dict:
    edges = _neighbors(hw)
    c = np.asarray(coverage, np.float64).reshape(-1)
    n = len(c)
    rng = np.random.default_rng(SHUFFLE_SEED)
    true, shuffled, masses = {}, {}, {}
    # Same-side banks have exactly balanced orientations and endpoint marginals.
    # Shuffling rematches the identical endpoint stub multiset into symmetric
    # pairs. Self-pairs can occur in the null, are disclosed, and are never used
    # as true observed reference neighbors.
    for role in (0, 1):
        role_mass = c if role == 1 else 1-c
        mass = role_mass[edges[:, 0]]*role_mass[edges[:, 1]]
        selected = _mass_quantiles(edges, mass, OCCURRENCES_PER_STATE//2)
        bank = np.concatenate((selected, selected[:, ::-1]), axis=0)
        if len(bank):
            stubs = bank[:, 0]
            permutation = rng.permutation(len(stubs))
            permuted = stubs[permutation]
            half = len(permuted)//2
            rematched = np.stack((permuted[:half], permuted[half:]), axis=1)
            null = np.concatenate((rematched, rematched[:, ::-1]), axis=0)
        else:
            null = bank.copy()
        true[role, role], shuffled[role, role] = bank, null
        masses[role, role] = 2*float(mass.sum(dtype=np.float64))
    # FG/BG banks encode both physical orientations; BF is exactly the transpose
    # of FB, including its fixed shuffle, to avoid an arbitrary query orientation.
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    cross_mass = c[directed[:, 0]]*(1-c[directed[:, 1]])
    fb = _mass_quantiles(directed, cross_mass, OCCURRENCES_PER_STATE)
    fb_null = fb.copy()
    if len(fb):
        fb_null[:, 1] = fb[rng.permutation(len(fb)), 1]
    true[1, 0], true[0, 1] = fb, fb[:, ::-1].copy()
    shuffled[1, 0], shuffled[0, 1] = fb_null, fb_null[:, ::-1].copy()
    masses[1, 0] = masses[0, 1] = float(cross_mass.sum(dtype=np.float64))
    state_diag = {f'{y}{z}': _bank_diagnostic(true[y, z], shuffled[y, z], n, masses[y, z])
                  for y in (0, 1) for z in (0, 1)}
    source_total = sum(masses.values())
    if abs(source_total-2*len(edges)) > 1e-10*max(1, 2*len(edges)):
        raise RuntimeError('Reference four-state coverage masses do not close')
    return dict(true=true, shuffled=shuffled, source_edges=edges,
                diagnostics=dict(states=state_diag, physical_reference_edges=len(edges),
                    oriented_reference_edge_mass=2*len(edges), four_state_mass_sum=source_total,
                    four_state_mass_closure_error=abs(source_total-2*len(edges)),
                    reference_neighbors='actual horizontal/vertical distinct tokens; no self loops',
                    query_pair_orientation_invariant=True,
                    sampling='64 equal-mass quantile occurrences per state; same-side32 then mirror',
                    full_mass_vs_sampled_marginals='state total mass is full; exact preserved endpoint marginals belong to the selected empirical bank',
                    null_self_pairs='possible under independent endpoint rematching; disclosed per state'))


@torch.inference_mode()
def _tables(query: torch.Tensor, reference: torch.Tensor, edges: np.ndarray,
            banks: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    arrays = [bank.reshape(-1) for bank in banks['true'].values() if len(bank)]
    active_reference = np.unique(np.concatenate(arrays)) if arrays else np.empty(0, np.int64)
    positions = np.full(len(reference), -1, np.int64)
    positions[active_reference] = np.arange(len(active_reference))
    # Shared FP32 cosines, then stable FP64 LSE and all dependence arithmetic.
    logits = (query @ reference[torch.from_numpy(active_reference)].T).double()/TEMPERATURE
    real = torch.zeros((len(edges), 2, 2), dtype=torch.float64)
    null = torch.zeros_like(real)
    edge_tensor = torch.from_numpy(edges)
    marginal_errors = []
    for y in (0, 1):
        for z in (0, 1):
            true = banks['true'][y, z]
            shuffled = banks['shuffled'][y, z]
            if not len(true):
                continue  # No observed role-pair relation: copula is exactly1.
            left, right = [torch.from_numpy(positions[true[:, j]]) for j in (0, 1)]
            null_left, null_right = [torch.from_numpy(positions[shuffled[:, j]]) for j in (0, 1)]
            # Canonical endpoint order provides the exact same factorized model
            # to both arms, even when an equivalent histogram has a new row order.
            canonical_left, canonical_right = left.sort().values, right.sort().values
            log_n = math.log(len(true))
            for start in range(0, len(edges), EDGE_CHUNK):
                stop = min(start+EDGE_CHUNK, len(edges))
                a, b = edge_tensor[start:stop, 0], edge_tensor[start:stop, 1]
                marginal = (torch.logsumexp(logits[a][:, canonical_left], dim=1)-log_n
                            +torch.logsumexp(logits[b][:, canonical_right], dim=1)-log_n)
                joint = torch.logsumexp(logits[a][:, left]+logits[b][:, right], dim=1)-log_n
                joint_null = torch.logsumexp(logits[a][:, null_left]+logits[b][:, null_right], dim=1)-log_n
                real[start:stop, y, z] = joint-marginal
                null[start:stop, y, z] = joint_null-marginal
            # Histogram identity is exact; this zero is not a floating-point
            # approximation of a separately reconstructed denominator.
            marginal_errors.append(0.)
    return real, null, dict(active_reference_tokens=len(active_reference),
        active_reference_ids=active_reference.tolist(), query_reference_matching_mac=len(query)*len(active_reference)*query.shape[1],
        logits_shape=list(logits.shape), matching_dtype='FP32 cosines, FP64 LSE',
        maximum_denominator_marginal_error=max(marginal_errors, default=0.),
        factorized_table='each state J_ind is exactly its common endpoint-product density; log ratio0')


def _delta(table: torch.Tensor) -> torch.Tensor:
    return table[:, 0, 0]+table[:, 1, 1]-table[:, 0, 1]-table[:, 1, 0]


def _field_diagnostic(value: torch.Tensor) -> dict:
    if not len(value):
        return dict(edge_count=0, minimum=None, maximum=None, mean=None,
                    absolute_mean=None, attractive_fraction=None, repulsive_fraction=None)
    return dict(edge_count=len(value), minimum=float(value.min()), maximum=float(value.max()),
        mean=float(value.mean()), absolute_mean=float(value.abs().mean()),
        attractive_fraction=float((value > 1e-12).double().mean()),
        repulsive_fraction=float((value < -1e-12).double().mean()),
        numerical_zero_fraction=float((value.abs() <= 1e-12).double().mean()))


@torch.inference_mode()
def probe(reference_features: Any, reference_coverage: Any, query_features: Any, *,
          reference_grid_hw: Any = (64, 64), query_grid_hw: Any = (64, 64)) -> dict:
    """Return unary-free true/shuffle/factorized interactions, no masks.

    Coverage is the caller's lawful continuous reference-mask area pooling;
    no thresholded role mask is created. 0=background, 1=foreground. For each
    state yz, L_yz=log joint-neighbor KDE minus log its endpoint-product KDE.
    delta=L00+L11-L01-L10. Unary-free pair energy is
    V(y,z)=-delta*(2y-1)*(2z-1)/4; positive delta is attractive/submodular,
    negative delta is repulsive/non-submodular in a fixed global role encoding.

    The factorized control has delta exactly0. The shuffle has identical
    occurrence budgets and endpoint marginals. Query features are observed
    evidence only; no query labels, object identities, reference-to-query
    coordinates, class-size constraints or optimizer are used.
    """
    started = time.perf_counter()
    reference = _features(reference_features, 'reference_features')
    query = _features(query_features, 'query_features')
    if reference.shape[1] != query.shape[1]:
        raise ValueError('Reference/query channels must match')
    reference_grid = _grid(reference_grid_hw, len(reference), 'reference_grid_hw')
    query_grid = _grid(query_grid_hw, len(query), 'query_grid_hw')
    coverage = np.asarray(reference_coverage).reshape(-1)
    if (len(coverage) != len(reference) or not np.isfinite(coverage).all()
            or (coverage < 0).any() or (coverage > 1).any()):
        raise ValueError('Reference coverage must match tokens and be finite in [0,1]')
    if coverage.sum(dtype=np.float64) <= 0 or (1-coverage).sum(dtype=np.float64) <= 0:
        raise ValueError('Both lawful reference roles require positive area mass')
    source_banks = _banks(coverage, reference_grid)
    edges = _neighbors(query_grid)
    real, shuffled, matching = _tables(query, reference, edges, source_banks)
    delta_true, delta_null = _delta(real), _delta(shuffled)
    factorized = torch.zeros_like(delta_true)
    if not bool(torch.isfinite(real).all()) or not bool(torch.isfinite(shuffled).all()):
        raise RuntimeError('Nonfinite reference neighbor dependence field')
    return dict(edge_index=edges, delta_true=delta_true.numpy(),
        delta_fixed_shuffle=delta_null.numpy(), delta_factorized=factorized.numpy(),
        log_copula_true=real.numpy(), log_copula_fixed_shuffle=shuffled.numpy(),
        diagnostics=dict(method='role_conditional_reference_neighbor_dependence_probe',
            status='information probe only; neither graph optimization nor segmentation performance',
            constants=dict(temperature=TEMPERATURE, occurrences_per_state=OCCURRENCES_PER_STATE,
                shuffle_seed=SHUFFLE_SEED, edge_chunk=EDGE_CHUNK), role_order=ROLE_ORDER,
            input=dict(reference_tokens=len(reference), query_tokens=len(query),
                channels=reference.shape[1], reference_grid_hw=reference_grid, query_grid_hw=query_grid,
                representation='caller preserved unit FP32; no implicit normalization/APD'),
            construction=source_banks['diagnostics'], matching=matching,
            delta_true=_field_diagnostic(delta_true),
            delta_fixed_shuffle=_field_diagnostic(delta_null),
            delta_factorized=_field_diagnostic(factorized),
            true_minus_shuffle=_field_diagnostic(delta_true-delta_null),
            interpretation='delta is a dependence likelihood contrast; not a boundary probability or role orientation',
            query_labels_used=False, encoded_images=0, FoRIS_fields_used=False,
            forced_unlike_labels=False, full_graph_solved=False,
            total_probe_seconds=time.perf_counter()-started))
