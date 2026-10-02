#!/usr/bin/env python3
"""Ten small CPU correctness checks for proposed reference-clamped diffusion.

No images, models, weights, server, query GT input or CUDA initialization.
Synthetic labels below are diagnostic expectations, not performance evidence.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tics.reference_diffusion import (DiffusionConfig, build_system,
                                     reference_diffusion, solve_cg)


def run():
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(317)
    support = torch.randn(8, 5, generator=generator, dtype=torch.float64)
    query = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    geometry = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    fallback = torch.tensor([.1, -.2, .3, -.4, .5, -.6], dtype=torch.float64)
    config = DiffusionConfig(knn=3, cross_temperature=.4, geometry_temperature=.5,
                             kde_temperature=.5, cg_tolerance=1e-12, cg_max_iterations=80)
    cases = []

    def record(name, ok, **details):
        cases.append(dict(name=name, passed=bool(ok), **details))

    system = build_system(support, mask, query, (2, 3), geometry, config)
    # Dense operator is a separate edge-assembly reference, never used in real inference.
    dense = torch.eye(len(query), dtype=torch.float64) * config.fidelity
    for i, j, w in zip(system.spatial_i.tolist(), system.spatial_j.tolist(),
                       system.spatial_weight.tolist()):
        dense[i, i] += w
        dense[j, j] += w
        dense[i, j] -= w
        dense[j, i] -= w
    for i, w in zip(system.cross_query.tolist(), system.cross_weight.tolist()):
        dense[i, i] += w
    minimum = float(torch.linalg.eigvalsh(dense).min())
    probe = torch.randn(6, generator=generator, dtype=torch.float64)
    error = float((system.matvec(probe) - dense @ probe).abs().max())
    record("symmetric_spd_scatter_operator", torch.equal(dense, dense.T)
           and minimum >= config.fidelity - 1e-12 and error < 1e-12,
           minimum_eigenvalue=minimum, fidelity_lower_bound=config.fidelity,
           scatter_dense_error=error)

    value, solver = solve_cg(system, config.cg_tolerance, config.cg_max_iterations)
    expected = torch.linalg.solve(dense, system.rhs)
    error = float((value - expected).abs().max())
    record("preconditioned_cg_equals_dense_solve", solver["converged"] and error < 1e-10,
           max_error=error, **solver)

    def energy(z):
        spatial = (system.spatial_weight * (z[system.spatial_i] - z[system.spatial_j]).square()).sum()
        signed = mask.to(z.dtype) * 2 - 1
        cross = (system.cross_weight * (z[system.cross_query] - signed[system.cross_support]).square()).sum()
        return .5 * (spatial + cross + config.fidelity * (z - system.target).square().sum())

    step = 1e-6
    finite_difference = float((energy(probe + step * value) - energy(probe - step * value)) / (2 * step))
    derivative = float(torch.dot(system.matvec(probe) - system.rhs, value))
    record("dirichlet_energy_gradient", abs(finite_difference - derivative) < 1e-8,
           directional_error=abs(finite_difference - derivative))

    swapped = build_system(support, ~mask, query, (2, 3), geometry, config)
    swapped_value, _ = solve_cg(swapped, config.cg_tolerance, config.cg_max_iterations)
    swap_error = float((value + swapped_value).abs().max())
    record("foreground_background_swap_equivariance", swap_error < 1e-10
           and torch.equal(system.cross_weight, swapped.cross_weight), signed_score_error=swap_error)

    # Easily separated synthetic foreground/background anchors and query patches.
    separated_support = torch.tensor([[1., .03], [1., -.03], [-1., .03], [-1., -.03]], dtype=torch.float64)
    separated_query = torch.tensor([[1., .02], [1., -.02], [-1., .02], [-1., -.02]], dtype=torch.float64)
    separated_mask = torch.tensor([True, True, False, False])
    simple, audit = reference_diffusion(separated_support, separated_mask, separated_query,
        (1, 4), separated_query, fallback_score=torch.zeros(4, dtype=torch.float64), config=config)
    record("clamped_positive_and_negative_evidence", audit["state"] == "SOLVED"
           and bool((simple[:2] > 0).all()) and bool((simple[2:] < 0).all()),
           scores=simple.tolist(), cross_edges=audit["cross_edges"])

    boundary = build_system(support, mask, query, (2, 3), geometry, config, "boundary_only")
    kde = build_system(support, mask, query, (2, 3), geometry, config, "native_kde")
    kde_value, _ = solve_cg(kde, config.cg_tolerance, config.cg_max_iterations)
    record("boundary_and_native_kde_controls", len(boundary.cross_weight) == 0
           and len(kde.cross_weight) == 0 and bool((kde.spatial_weight == 0).all())
           and float((kde_value - system.target).abs().max()) < 1e-12,
           boundary_cross_edges=len(boundary.cross_weight))

    plain = build_system(support, mask, query, (2, 3), geometry, config, "plain_label_propagation")
    plain_value, _ = solve_cg(plain, config.cg_tolerance, config.cg_max_iterations)
    record("plain_label_propagation_same_graph_control", torch.equal(plain.cross_weight, system.cross_weight)
           and torch.equal(plain.spatial_weight, system.spatial_weight)
           and bool((plain.target == 0).all()) and float((plain_value - value).abs().max()) > 1e-5,
           fidelity_increment_difference=float((plain_value - value).abs().max()))

    degeneracies = []
    for absent_mask in (torch.zeros_like(mask), torch.ones_like(mask)):
        result, audit = reference_diffusion(support, absent_mask, query, (2, 3), geometry,
                                            fallback_score=fallback, config=config)
        degeneracies.append(torch.equal(result, fallback) and audit["state"] == "FALLBACK_MISSING_SUPPORT_CLASS")
    record("missing_support_class_exact_native_fallback", all(degeneracies))

    malformed = []
    for defective in (query * 0, query.clone()):
        if bool((defective != 0).any()):
            defective[0, 0] = float("nan")
        result, audit = reference_diffusion(support, mask, defective, (2, 3), geometry,
                                            fallback_score=fallback, config=config)
        malformed.append(torch.equal(result, fallback) and audit["state"].startswith("FALLBACK_"))
    record("zero_or_nonfinite_feature_exact_fallback", all(malformed))

    impossible = DiffusionConfig(knn=3, cross_temperature=.4, geometry_temperature=.5,
        kde_temperature=.5, cg_tolerance=1e-15, cg_max_iterations=1)
    failed_value, failed_audit = reference_diffusion(support, mask, query, (2, 3), geometry,
        fallback_score=fallback, config=impossible)
    # No evidence creates a unique zero signed field, without hallucinating foreground.
    identical_support = torch.ones(4, 3, dtype=torch.float64)
    identical_query = torch.ones(6, 3, dtype=torch.float64)
    no_cross = DiffusionConfig(cross_strength=0, cg_tolerance=1e-12)
    zero_value, zero_audit = reference_diffusion(identical_support, separated_mask,
        identical_query, (2, 3), identical_query, fallback_score=fallback, config=no_cross)
    record("nonconvergence_fallback_and_no_evidence_no_quota", torch.equal(failed_value, fallback)
           and failed_audit["state"] == "FALLBACK_NONCONVERGED"
           and zero_audit["state"] == "SOLVED" and float(zero_value.abs().max()) < 1e-12,
           failed_relative_residual=failed_audit["relative_residual"], zero_evidence_max=float(zero_value.abs().max()))

    return dict(method="reference_clamped_manifold_diffusion", cpu_cases=cases,
                passed=sum(x["passed"] for x in cases), total=len(cases),
                all_passed=all(x["passed"] for x in cases), seed=317,
                scope="Synthetic float64 CPU correctness only; no segmentation gain or novelty established",
                cuda_calls=0, query_gt_argument=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
