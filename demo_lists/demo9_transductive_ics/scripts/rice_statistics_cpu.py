#!/usr/bin/env python3
"""Ten synthetic CPU checks of RICE statistics; no images or task-gain claim."""
import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch


def direct_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = Path(__file__).resolve().parents[1]
    # Direct import avoids tics.__init__'s historical device discovery. Also
    # suppress discovery explicitly if any future import reaches that package.
    with patch('torch.cuda.is_available', return_value=False):
        kernel = direct_module(base / 'tics/rice_statistics.py', 'rice_statistics_cpu_kernel')
        numpy_kernel = direct_module(base / 'tics/rice_subspace.py', 'rice_statistics_numpy_control')
        results = []
        generator = torch.Generator(device='cpu').manual_seed(2049)
        x = torch.randn(3, 64, 9, generator=generator, dtype=torch.float64)
        # Same physical patch retains instance variation across background views.
        x[1:] = x[0:1] + .12 * x[1:]
        coverage = torch.cat((torch.ones(32), torch.zeros(32)))
        stats = kernel.build_reference_statistics(x, coverage, (8, 8))

        def check(name, function):
            detail = function() or {}
            results.append(dict(name=name, passed=True, **detail))

        def identity():
            error = stats['audit']['covariance_identity_max_error']
            assert error < 1e-12
            return dict(max_error=error)
        check('total_covariance_identity', identity)

        def pairing_invariance():
            error = stats['audit']['augmented_shuffle_max_error']
            assert error < 1e-12
            return dict(max_error=error)
        check('fg_aug_invariant_to_within_mode_pair_shuffle', pairing_invariance)

        def pairing_changes():
            difference = stats['audit']['paired_shuffle_max_difference']
            assert difference > .01
            return dict(max_difference=difference)
        check('paired_only_changes_under_shuffle', pairing_changes)

        def pooled_check():
            fg, bg = stats['foreground_indices'], stats['background_indices']
            labels = stats['foreground_labels']
            residuals = []
            for k in torch.unique(labels):
                block = x[:, fg[labels == k]].reshape(-1, x.shape[-1])
                residuals.append(block - block.mean(0))
            block = x[:, bg].reshape(-1, x.shape[-1])
            residuals.append(block - block.mean(0))
            residual = torch.cat(residuals)
            dense = residual.T @ residual / len(residual)
            assert torch.allclose(dense, stats['covariances']['pooled_fisher'], atol=1e-12, rtol=1e-12)
            assert abs(stats['ridge'] - .1 * float(torch.trace(dense)) / 9) < 1e-15
        check('pooled_fisher_and_common_ridge', pooled_check)

        def projections():
            max_error = 0.
            for covariance in stats['covariances'].values():
                w, audit = kernel.discriminant_projection(stats['contrasts'], covariance, stats['ridge'])
                assert 0 < w.shape[1] <= 16
                max_error = max(max_error, audit['whitening_max_error'], audit['generalized_relative_error'])
            return dict(max_error=max_error)
        check('all_five_generalized_eigen_and_whitening', projections)

        def numpy_agreement():
            covariance = stats['covariances']['paired_only'].numpy()
            values, vectors = np.linalg.eigh(covariance)
            nuisance = vectors * np.sqrt(np.maximum(values, 0))[None]
            d = stats['contrasts']
            w, _ = kernel.discriminant_projection(d, stats['covariances']['paired_only'], stats['ridge'])
            wn, _ = numpy_kernel.reference_subspace(d.numpy(), nuisance, stats['ridge'], retained_energy=1.)
            # Eigenvector sign is arbitrary; compare induced metric.
            error = float(np.max(np.abs(w.numpy() @ w.numpy().T - wn @ wn.T)))
            assert error < 1e-8
            return dict(metric_max_error=error)
        check('existing_numpy_subspace_metric_agreement', numpy_agreement)

        def degeneracy():
            constant = torch.ones(3, 64, 9, dtype=torch.float64)
            zero = kernel.build_reference_statistics(constant, coverage, (8, 8))
            assert zero['state'] == 'DEGENERATE_ZERO_POOLED_TRACE' and zero['ridge'] == 0
            w, audit = kernel.discriminant_projection(zero['contrasts'], zero['covariances']['paired_only'], 0)
            assert w.shape == (9, 0) and audit['state'] == 'DEGENERATE_NONPOSITIVE_RIDGE'
            w, audit = kernel.discriminant_projection(torch.zeros(9, 2), torch.eye(9), .1)
            assert w.shape == (9, 0) and audit['state'] == 'NO_REFERENCE_DISCRIMINANT'
            missing = kernel.build_reference_statistics(x, torch.ones(64), (8, 8))
            assert missing['state'] == 'DEGENERATE_MISSING_PURE_CLASS'
        check('zero_scatter_zero_contrast_missing_class_fallbacks', degeneracy)

        def determinism():
            other = kernel.build_reference_statistics(x, coverage, (8, 8))
            assert torch.equal(stats['contrasts'], other['contrasts'])
            assert stats['audit'] == other['audit']
            for arm in stats['covariances']:
                assert torch.equal(stats['covariances'][arm], other['covariances'][arm])
        check('deterministic_sampling_modes_and_shuffle', determinism)

        def sampling():
            large = torch.randn(3, 512, 5, generator=generator)
            cov = torch.cat((torch.ones(180), torch.zeros(300), torch.full((32,), .5)))
            result = kernel.build_reference_statistics(large, cov, (16, 32))
            assert len(result['foreground_indices']) == 128 and len(result['background_indices']) == 256
            assert (cov[result['foreground_indices']] >= .9).all() and (cov[result['background_indices']] <= .1).all()
            assert result['audit']['requested_foreground_modes'] == 8
            assert result['audit']['background_modes'] <= 4
            k, j = result['audit']['foreground_modes'], result['audit']['background_modes']
            expected = (result['foreground_means'][:, None] - result['background_means'][None]).reshape(k*j, -1).T / np.sqrt(k*j)
            assert torch.equal(expected, result['contrasts'])
        check('pure_patch_caps_modes_and_contrast_weights', sampling)

        def contract_and_rank():
            invalid = coverage.clone(); invalid[0] = 1.1
            for values, cov, grid in ((x, invalid, (8, 8)), (x, coverage, (4, 8)), (x[:2], coverage, (8, 8))):
                try:
                    kernel.build_reference_statistics(values, cov, grid)
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid contract accepted')
            d = torch.diag(torch.arange(1, 21, dtype=torch.float64))
            w, audit = kernel.discriminant_projection(d, torch.eye(20), .1)
            assert w.shape == (20, 16) and audit['rank'] == 16
            assert not audit['post_projection_normalization']
        check('invalid_inputs_and_fixed_top16_rank', contract_and_rank)
    assert not torch.cuda.is_initialized(), 'CPU checks must never initialize CUDA'
    report = dict(state='CPU_SYNTHETIC_CHECKS_PASSED', checks=len(results), results=results,
                  device='cpu', cuda_discovery_suppressed=True, cuda_initialization_requested=False,
                  real_images_used=False, task_gain_measured=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
