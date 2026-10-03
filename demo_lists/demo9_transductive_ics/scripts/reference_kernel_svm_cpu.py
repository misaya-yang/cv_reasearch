#!/usr/bin/env python3
"""Ten CPU fixtures for a standard RBF SVM control; no image/gain claim.

No Torch/CUDA imports or package installation. Missing sklearn leaves explicit
skipped solver checks and a PARTIAL receipt; mathematical checks still run.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--require-sklearn', action='store_true',
                        help='Production CPU preflight: exit nonzero if solver checks cannot run')
    args = parser.parse_args()
    if args.out and args.out.exists():
        raise ValueError('Preserve existing CPU receipt; select a fresh output')
    source = Path(__file__).resolve().parents[1] / 'tics/reference_kernel_svm.py'
    spec = importlib.util.spec_from_file_location('_reference_svm_cpu', source)
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernel
    spec.loader.exec_module(kernel)
    rng = np.random.default_rng(2063)
    fg = rng.normal(size=(13, 6)) * .2 + np.array([1, 0, 0, 0, 0, 0])
    bg = rng.normal(size=(23, 6)) * .3 - np.array([1, 0, 0, 0, 0, 0])
    original_fg, original_bg = fg.copy(), bg.copy()
    query = rng.normal(size=(31, 6))
    model, fit_audit = kernel.fit_reference_kernel_svm(fg, bg)
    solver_available = fit_audit['state'] != 'SKLEARN_UNAVAILABLE'
    if solver_available:
        assert model is not None, fit_audit
    results = []

    def check(name, function, needs_solver=False):
        if needs_solver and not solver_available:
            results.append(dict(name=name, state='SKIPPED_SKLEARN_UNAVAILABLE', passed=None))
            return
        detail = function() or {}
        results.append(dict(name=name, state='PASSED', passed=True, **detail))

    def bandwidth():
        tau, audit = kernel.reference_bandwidth(fg, bg)
        count = min(len(fg), len(bg), 128)
        sample = np.concatenate((fg[np.arange(count)*len(fg)//count], bg[np.arange(count)*len(bg)//count]))
        direct = np.array([((sample[i]-sample[j])**2).sum() for i in range(len(sample)) for j in range(i)])
        expected = np.median(direct[direct > 0]) / 2
        assert np.isclose(tau, expected, rtol=1e-14, atol=1e-14)
        assert audit['bandwidth_samples_per_class'] == 13
        return dict(bandwidth=tau, balanced_count=count)
    check('balanced_reference_bandwidth_direct_pair_median', bandwidth)

    def representer():
        exact = model.representer_margin(query)
        actual = model.decision_function(query)
        error = float(np.max(np.abs(exact-actual)))
        assert np.allclose(exact, actual, rtol=1e-12, atol=1e-12)
        return dict(max_error=error)
    check('explicit_representer_matches_sklearn_decision_function', representer, True)

    def reversal():
        reverse, audit = kernel.fit_reference_kernel_svm(bg, fg)
        assert reverse is not None, audit
        assert np.array_equal(model.reference_features, reverse.reference_features)
        assert np.array_equal(model.decision_function(query), -reverse.decision_function(query))
        conflict_bg = np.concatenate((bg,fg[:2]))
        partial, audit = kernel.fit_reference_kernel_svm(fg,conflict_bg)
        reverse_partial, reverse_audit = kernel.fit_reference_kernel_svm(conflict_bg,fg)
        assert partial is not None and reverse_partial is not None, (audit,reverse_audit)
        assert len(partial.reference_features) == len(fg)+len(conflict_bg)
        assert np.array_equal(partial.reference_features,reverse_partial.reference_features)
        assert np.array_equal(partial.decision_function(query),-reverse_partial.decision_function(query))
        return dict(bitwise_semantic_margin_reversal=True)
    check('foreground_background_label_reversal', reversal, True)

    def chunked():
        whole = model.decision_function(query, chunk_rows=len(query))
        for chunk in [1, 3, 8, 128]:
            assert np.array_equal(whole, model.decision_function(query, chunk_rows=chunk))
        assert model.decision_function(np.empty((0, 6))).shape == (0,)
        return dict(bitwise_chunk_equality=True)
    check('all_query_margins_chunk_exact_and_empty_query', chunked, True)

    def scale():
        tau, _ = kernel.reference_bandwidth(fg, bg)
        scaled_tau, _ = kernel.reference_bandwidth(7.5*fg, 7.5*bg)
        assert np.isclose(scaled_tau, 7.5**2*tau, rtol=1e-14)
        gram = kernel.rbf_kernel(query, np.concatenate((fg,bg)), tau)
        scaled = kernel.rbf_kernel(7.5*query, 7.5*np.concatenate((fg,bg)), scaled_tau)
        error = float(np.max(np.abs(gram-scaled)))
        assert error < 1e-13
        return dict(kernel_max_error=error, scale_squared_bandwidth=True)
    check('distance_bandwidth_scale_invariance', scale)

    def objective():
        labels = np.r_[np.ones(len(fg)), -np.ones(len(bg))]
        weights = kernel.class_balanced_weights(labels)
        losses = rng.random(len(labels))
        weighted = (weights*losses).sum()
        balanced = len(labels)/2*(losses[labels==1].mean()+losses[labels==-1].mean())
        assert np.isclose(weighted, balanced, rtol=1e-14)
        assert np.isclose(fit_audit['equivalent_lambda'], 2/len(labels))
        return dict(equivalent_lambda=fit_audit['equivalent_lambda'])
    check('class_balanced_SVC_objective_scale', objective)

    def conflict():
        partial_bg = np.concatenate((bg,fg[:2]))
        fitted, audit = kernel.fit_reference_kernel_svm(fg,partial_bg)
        reverse, reverse_audit = kernel.fit_reference_kernel_svm(partial_bg,fg)
        assert audit['conflict_count'] == 2 and audit['partial_conflicts_retained']
        assert audit['conflicting_foreground_anchors'] == 2 and audit['conflicting_background_anchors'] == 2
        assert not audit['class_distribution_identical']
        assert audit['semantic_orientation'] == -reverse_audit['semantic_orientation']
        def canonical_problem(positive,negative,orientation):
            x = np.concatenate((positive,negative))
            y = np.r_[np.ones(len(positive)), -np.ones(len(negative))]
            canonical = y*orientation
            order = np.lexsort((canonical,)+tuple(x[:,j] for j in range(x.shape[1]-1,-1,-1)))
            return x[order],canonical[order],kernel.class_balanced_weights(y)[order]
        forward_problem = canonical_problem(fg,partial_bg,audit['semantic_orientation'])
        reverse_problem = canonical_problem(partial_bg,fg,reverse_audit['semantic_orientation'])
        assert all(np.array_equal(a,b) for a,b in zip(forward_problem,reverse_problem))
        # First lexicographic descriptor has equal class mass: skip it rather
        # than obtaining orientation from the arbitrary first conflicting row.
        small_fg = np.array([[0.,0.],[1.,0.]])
        small_bg = np.array([[0.,0.],[2.,0.]])
        _, small_audit = kernel.fit_reference_kernel_svm(small_fg,small_bg)
        assert small_audit['orientation_descriptor_index'] == 1 and small_audit['semantic_orientation'] == 1
        if solver_available:
            assert fitted is not None and reverse is not None, (audit,reverse_audit)
            assert len(fitted.reference_features) == len(fg)+len(partial_bg)
            assert np.array_equal(fitted.decision_function(query),-reverse.decision_function(query))
            solver_check='PASSED'
        else:
            assert fitted is None and reverse is None
            assert audit['state'] == reverse_audit['state'] == 'SKLEARN_UNAVAILABLE'
            solver_check='SKIPPED_SKLEARN_UNAVAILABLE'
        return dict(conflict_count=2,all_conflicting_anchors_retained=True,
                    canonical_problem_flip_exact=True,
                    partial_conflict_solver_and_flip_check=solver_check)
    check('partial_conflicts_retained_normalized_mass_orientation_and_audit', conflict)

    def degenerate():
        tau, audit = kernel.reference_bandwidth(np.ones((4,2)), np.ones((5,2)))
        assert tau is None and audit['state'] == 'NO_POSITIVE_REFERENCE_BANDWIDTH'
        fitted, audit = kernel.fit_reference_kernel_svm(np.ones((4,2)), np.ones((5,2)))
        assert fitted is None and audit['state'] == 'NO_POSITIVE_REFERENCE_BANDWIDTH'
        fitted, audit = kernel.fit_reference_kernel_svm(fg, np.empty((0,6)))
        assert fitted is None and audit['state'] == 'MISSING_REFERENCE_CLASS'
        same_fg = np.array([[0.,0.],[1.,0.]])
        same_bg = np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.]])
        fitted, audit = kernel.fit_reference_kernel_svm(same_fg,same_bg)
        assert fitted is None and audit['state'] == 'IDENTICAL_CLASS_DISTRIBUTIONS'
        assert audit['conflict_count'] == 2 and audit['class_distribution_identical']
        fitted, audit = kernel.fit_reference_kernel_svm(same_bg,same_fg)
        assert fitted is None and audit['state'] == 'IDENTICAL_CLASS_DISTRIBUTIONS'
    check('zero_bandwidth_missing_class_identical_distribution_fallbacks', degenerate)

    def invalid():
        for bad in (np.full_like(fg,np.nan), np.ones((3,5)), np.ones(4)):
            try:
                kernel.fit_reference_kernel_svm(bad,bg)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid feature contract accepted')
        assert np.array_equal(fg,original_fg) and np.array_equal(bg,original_bg)
        gram = kernel.rbf_kernel(np.concatenate((fg,bg)),np.concatenate((fg,bg)),fit_audit['bandwidth'])
        assert np.allclose(gram,gram.T,atol=1e-14,rtol=1e-14)
        assert np.linalg.eigvalsh(gram).min() > -1e-12
    check('unchanged_source_features_finite_contract_and_PSD_gram', invalid)

    def far_and_deterministic():
        second, audit = kernel.fit_reference_kernel_svm(fg,bg)
        assert second is not None, audit
        assert np.array_equal(model.decision_function(query),second.decision_function(query))
        far = np.full((2,6),1e6)
        assert np.array_equal(model.decision_function(far),np.full(2,model.intercept))
        scaled, audit = kernel.fit_reference_kernel_svm(7.5*fg,7.5*bg)
        assert scaled is not None, audit
        error = float(np.max(np.abs(model.decision_function(query)-scaled.decision_function(7.5*query))))
        assert error < 1e-6
        return dict(scale_margin_max_error=error, far_limit_is_intercept_not_guaranteed_background=True)
    check('fit_determinism_scale_margin_and_far_query_limit', far_and_deterministic, True)
    passed = sum(x['passed'] is True for x in results)
    receipt = dict(state='CPU_REFERENCE_KERNEL_SVM_PASSED' if passed == 10 else 'CPU_REFERENCE_KERNEL_SVM_PARTIAL_SKLEARN_UNAVAILABLE',
                   fixtures=len(results), passed=passed, skipped=10-passed, results=results,
                   fit_audit=fit_audit, sklearn_available=solver_available,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   Torch_or_CUDA_import_requested=False, real_images=False, task_gain_measured=False)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    if args.require_sklearn and not solver_available:
        parser.exit(2, 'Existing sklearn required: full SVM CPU checks were not executed.\n')


if __name__=='__main__':
    main()
