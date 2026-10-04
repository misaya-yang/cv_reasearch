#!/usr/bin/env python3
"""SERVER-only CPU FP64 parity checks for the frozen Torch PCF implementation.

Tiny synthetic numerical contracts only: no DINO/image encoding, QueryGT,
training, CUDA initialization, actual GPU timing or segmentation result.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def require_server():
    if not sys.platform.startswith("linux") or not str(ROOT).startswith("/root/"):
        raise RuntimeError("SERVER CPU only; no local numerical/model execution")


def run():
    require_server()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    from tics.partial_correspondence import (build_relation, build_query_tree,
        semantic_correspondence, tree_sum_product, binary_potts)
    from tics.partial_correspondence_torch import (TorchRelation,
        tree_sum_product_torch, binary_potts_torch,
        semantic_correspondence_torch, shape_memory_audit)
    torch.set_num_threads(1)
    checks = []
    records = []
    maximum_error = 0.
    started = time.monotonic()
    rng = np.random.default_rng(6030)
    role = np.array([[1, 1, 1, 1], [1, 0, 0, 1]], dtype=bool)
    relation = build_relation(role)
    raw = rng.normal(size=(6, 5))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    tree = build_query_tree(raw, (2, 3))
    h = np.array([[.2, .6, .85], [.7, .3, .52]], dtype=np.float64)
    pi = rng.uniform(.02, 1, size=(6, relation.m))
    pi /= pi.sum(axis=1, keepdims=True)

    def parity(label, unary=h, probability=pi, kernel=relation, query_tree=tree,
               rho=.1, strength=1., permutation=None, batch=2):
        nonlocal maximum_error
        reference = tree_sum_product(unary, probability, kernel, query_tree,
            rho=rho, pair_strength=strength, relation_permutation=permutation,
            return_state_probabilities=True)
        actual = tree_sum_product_torch(unary, probability, kernel, query_tree,
            rho=rho, pair_strength=strength, relation_permutation=permutation,
            return_state_probabilities=True, device="cpu", node_batch=batch)
        errors = {}
        for key in ("foreground", "matched_foreground", "unmatched_foreground",
                    "background", "matched_states"):
            value = actual[key].cpu().numpy()
            errors[key] = float(np.max(np.abs(value-reference[key]))) if value.size else 0.
            if not np.isfinite(value).all() or errors[key] > 2e-10:
                raise AssertionError(label+" mismatch in "+key+": "+str(errors[key]))
        maximum_error = max(maximum_error, *errors.values())
        records.append(dict(id=label, state="PASSED", maximum_abs_error=max(errors.values()),
                            field_errors=errors, node_batch=batch,
                            neutral=actual["metadata"].get("model_neutral", False)))
        return actual, reference

    for batch in (1, 2, 64):
        parity("branching_tree_batch"+str(batch), batch=batch)
    checks.append("branching_tree_two_pass_all_state_marginals_reference_parity_three_batch_sizes")
    _, _ = parity("noUF", rho=0.)
    _, _ = parity("noUF_lambda081", rho=0., strength=.81)
    checks.append("same_full_state_model_noUF_and_lambda081_control_parity")
    permutation = rng.permutation(relation.m)
    shuffled, _ = parity("R_only_permutation", permutation=permutation)
    # Joint coordinate permutation is a separate invariance test; the method's
    # actual shuffle control above intentionally leaves semantic pi unchanged.
    joint, _ = parity("joint_R_pi_permutation", probability=pi[:, permutation],
                      permutation=permutation)
    original, _ = parity("original_for_joint_invariance")
    np.testing.assert_allclose(joint["foreground"].numpy(), original["foreground"].numpy(), atol=2e-10, rtol=0)
    np.testing.assert_allclose(joint["matched_states"].numpy(), original["matched_states"].numpy()[:, permutation], atol=2e-10, rtol=0)
    checks.append("complete_relation_only_shuffle_and_joint_state_coordinate_invariance")
    for count in (0, 1, 2, 3):
        small = build_relation(np.arange(4).reshape(1, 4) < count)
        smallpi = np.full((6, count), 1/count) if count else np.empty((6, 0))
        result, _ = parity("M"+str(count), kernel=small, probability=smallpi)
        assert np.array_equal(result["foreground"].numpy(), h)
        assert result["metadata"]["native_identity_required"]
    checks.append("M0_explicit_abstention_and_M1_M2_M3_exact_h_degeneracy")
    uniform, _ = parity("uniform_pi", probability=np.full_like(pi, 1/relation.m))
    zero, _ = parity("lambda0", strength=0.)
    ones, _ = parity("Rones", kernel=replace(relation, degenerate=True))
    for result in (uniform, zero, ones):
        assert np.array_equal(result["foreground"].numpy(), h)
        assert result["metadata"]["native_identity_required"]
    disconnected = build_relation(np.array([[1, 0, 1, 0, 1, 0, 1]], dtype=bool))
    disconnected_pi = pi[:, :4].copy()
    disconnected_pi /= disconnected_pi.sum(axis=1, keepdims=True)
    empty_K, _ = parity("disconnected_K0_balanced_Rones", kernel=disconnected,
                        probability=disconnected_pi)
    assert disconnected.sparseK.nnz == 0
    assert np.array_equal(empty_K["foreground"].numpy(), h)
    assert empty_K["metadata"]["native_identity_required"]
    checks.append("theory_neutral_uniform_pi_lambda0_and_Rones_return_exact_h")
    endpoints = np.array([[0., 1., .5], [1., 0., .25]])
    result, _ = parity("h_endpoints", unary=endpoints)
    p = result["foreground"].numpy()
    assert (p[endpoints == 0] == 0).all() and (p[endpoints == 1] == 1).all()
    checks.append("h0_h1_exact_endpoint_foreground_mass")
    onehot = np.eye(relation.m, dtype=np.float64)
    result, _ = parity("zero_pi_entries", probability=onehot)
    checks.append("zero_semantic_state_mass_log_domain_without_epsilon_repair")
    result, _ = parity("mass_and_UF_odds")
    states = result["matched_states"].numpy()
    uf, bg = result["unmatched_foreground"].numpy(), result["background"].numpy()
    np.testing.assert_allclose(states.sum(1).reshape(h.shape)+uf+bg, 1., atol=2e-12, rtol=0)
    np.testing.assert_allclose(uf*(1-h), bg*.1*h, atol=2e-12, rtol=0)
    assert np.all(result["foreground"].numpy() >= .1*h/(.1*h+1-h)-2e-12)
    checks.append("all_state_probability_sums_nonnegative_UF_neutral_odds_and_foreground_lower_bound")
    # Zero edges disconnect only the model relation, not the supplied full tree.
    weights = np.zeros_like(tree.edgew)
    zerotree = replace(tree, edgew=weights)
    result, _ = parity("neutral_edges", query_tree=zerotree)
    np.testing.assert_allclose(result["foreground"].numpy(), h, atol=2e-12, rtol=0)
    assert result["metadata"]["R_matvec_calls"] == 0
    checks.append("zero_weight_edges_remain_tree_edges_but_send_exact_neutral_messages")
    for batch in (1, 64):
        naive = binary_potts_torch(h, tree, device="cpu", node_batch=batch).numpy()
        np.testing.assert_allclose(naive, binary_potts(h, tree), atol=2e-12, rtol=0)
    checks.append("same_tree_same_h_binaryPotts_reference_parity")
    src = rng.normal(size=(role.size, 7))
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    query = rng.normal(size=(6, 7))
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    semantic = semantic_correspondence_torch(src, query, role, device="cpu", query_batch=2).numpy()
    np.testing.assert_allclose(semantic, semantic_correspondence(src, query, role), atol=3e-15, rtol=0)
    assert semantic.shape == (6, int(role.sum()))
    checks.append("same_temperature_unit_features_all_Source_states_semantic_softmax")
    vectors = rng.uniform(0, 1, size=(3, relation.m))
    operator = TorchRelation(relation, "cpu", permutation)
    actual = operator.action(torch.tensor(vectors, dtype=torch.float64)).numpy()
    expected = np.stack([relation.matvec(row[np.argsort(permutation)])[permutation] for row in vectors])
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0)
    checks.append("sparse_CSR_plus_rank1_batch_R_action_and_permutation_exact_equations")
    invalid = [
        dict(h=np.full_like(h, np.nan)), dict(h=np.full_like(h, -.1)),
        dict(pi=-pi), dict(pi=pi*.9), dict(pi=pi[:, :-1]),
        dict(rho=1.), dict(rho=-.1), dict(pair_strength=1.01),
        dict(relation_permutation=np.zeros(relation.m, dtype=int)),
        dict(relation_permutation=np.arange(relation.m)+.1),
        dict(tree=replace(tree, parent=np.full(h.size, -1))),
        dict(tree=replace(tree, edgew=np.full(h.size, np.nan))),
        dict(relation=replace(relation, scale=np.zeros(relation.m))),
        dict(node_batch=0),
    ]
    for bad in invalid:
        arguments = dict(h=h, pi=pi, relation=relation, tree=tree, device="cpu")
        arguments.update(bad)
        try:
            tree_sum_product_torch(**arguments)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("Illegal inputs silently accepted: "+str(tuple(bad)))
    checks.append("illegal_domain_tree_probability_permutation_and_batch_fail_without_fallback")
    assert not torch.cuda.is_initialized()
    paths = [ROOT/"tics/partial_correspondence.py",
             ROOT/"tics/partial_correspondence_torch.py", Path(__file__)]
    return dict(state="SERVER_PCF_TORCH_CPU_REFERENCE_PARITY_PASSED",
                checks=len(checks), passed=checks, records=records,
                maximum_abs_error=maximum_error, acceptance_max_abs_error=2e-10,
                GPU_parity_target_max_abs_error=1e-8,
                seconds=time.monotonic()-started,
                CUDA_initialized=False, no_model=True, no_Query_GT=True,
                fixture_scope="Small synthetic unit features and full finite latent trees only",
                real_task_gain_verified=False, actual_GPU_profile_completed=False,
                shape_memory_bound=shape_memory_audit(4096, 4096, 312*4096),
                source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_server()
    if args.out.exists():
        raise ValueError("Keep prior evidence; choose a fresh output path")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in
                     ("state", "checks", "maximum_abs_error", "CUDA_initialized")}))


if __name__ == "__main__":
    main()
