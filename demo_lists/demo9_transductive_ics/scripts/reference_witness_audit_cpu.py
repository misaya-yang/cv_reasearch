#!/usr/bin/env python3
"""Ten CPU-only numerical checks; no model, image, query GT, or task gain."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out and args.out.exists():
        raise ValueError('Preserve existing receipt; choose a fresh output')
    source = Path(__file__).resolve().parents[1]/'tics/reference_witness_audit.py'
    spec = importlib.util.spec_from_file_location('witness_cpu', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    full_audit = module.reference_witness_audit
    def audit(*args, **kwargs):
        result = full_audit(*args, **kwargs)
        return {key.removeprefix('cosine_'): value for key, value in result.items()
                if key.startswith('cosine_')}
    def branch(result, prefix):
        return {key.removeprefix(prefix+'_'): value for key, value in result.items()
                if key.startswith(prefix+'_')}
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(2073)
    refs = torch.randn(3, 7, 3, 4, dtype=torch.float64, generator=generator)
    query = torch.randn(7, 3, 4, dtype=torch.float64, generator=generator)
    masks = torch.rand(3, 3, 4, generator=generator) > .5
    results = []
    started = time.monotonic()

    def check(name, fn):
        detail = fn() or {}
        results.append(dict(name=name, passed=True, **detail))

    def dense_exact():
        out = audit(refs, query, masks, chunk=5)
        rn = F.normalize(refs, dim=1).permute(0, 2, 3, 1).reshape(-1, 7)
        qn = F.normalize(query, dim=0).reshape(7, -1)
        dense = rn @ qn
        flat_mask = masks.reshape(-1)
        for key, expected in [('fg_max', dense[flat_mask].max(0).values),
                              ('bg_max', dense[~flat_mask].max(0).values)]:
            torch.testing.assert_close(out[key].flatten(), expected, rtol=1e-14, atol=1e-14)
        assert torch.equal(out['nearest_index'].flatten(), dense.argmax(0))
        assert torch.equal(out['nearest_label'].flatten(), flat_mask[dense.argmax(0)])
        source_out = full_audit(refs, query, masks, chunk=5)
        source_refs = F.normalize(refs, dim=2).permute(0, 2, 3, 1).reshape(-1, 7)
        source_dense = source_refs @ qn
        for label, key in ((flat_mask, 'source_fg_max'), (~flat_mask, 'source_bg_max')):
            torch.testing.assert_close(source_out[key].flatten(), source_dense[label].max(0).values,
                                       rtol=1e-14, atol=1e-14)
        assert torch.equal(source_out['source_nearest_index'].flatten(), source_dense.argmax(0))
        source_votes = torch.stack([masks[j].flatten()[source_dense[j*12:(j+1)*12].argmax(0)]
                                   for j in range(3)]).sum(0).reshape(3, 4)
        assert torch.equal(source_out['source_candidate'], source_votes >= 2)
        assert source_out['source_ref_normalization_axis'] == 'height'
        return dict(max_abs_error=float((out['fg_max'].flatten()-dense[flat_mask].max(0).values).abs().max()),
                    source_dense_max_abs_error=float((source_out['source_fg_max'].flatten()-source_dense[flat_mask].max(0).values).abs().max()))
    check('bruteforce_dense_maxima_and_global_nearest', dense_exact)

    def vote_exact():
        out = audit(refs, query, masks, chunk=3)
        expected = []
        for j in range(3):
            sim = F.normalize(refs[j], dim=0).reshape(7, -1).T @ F.normalize(query, dim=0).reshape(7, -1)
            expected.append(masks[j].flatten()[sim.argmax(0)].reshape(3, 4))
        assert torch.equal(out['per_ref_nearest_label'], torch.stack(expected))
        assert torch.equal(out['candidate'], torch.stack(expected).sum(0) >= 2)
        assert out['per_ref_margin'].shape == (3, 3, 4)
        source = branch(full_audit(refs, query, masks, chunk=3), 'source')
        expected_source = []
        for j in range(3):
            sim = F.normalize(refs[j:j+1], dim=2).reshape(7, -1).T @ F.normalize(query, dim=0).reshape(7, -1)
            expected_source.append(masks[j].flatten()[sim.argmax(0)].reshape(3, 4))
        assert torch.equal(source['per_ref_nearest_label'], torch.stack(expected_source))
        assert torch.equal(source['candidate'], torch.stack(expected_source).sum(0) >= 2)
    check('per_reference_vote_and_margins', vote_exact)

    def ties():
        rr = torch.ones(2, 3, 1, 2, dtype=torch.float64)
        qq = torch.ones(3, 1, 2, dtype=torch.float64)
        mm = torch.tensor([[[False, True]], [[True, False]]])
        both = full_audit(rr, qq, mm, chunk=1)
        for prefix in ('source', 'cosine'):
            out = branch(both, prefix)
            assert torch.equal(out['margin'], torch.zeros_like(out['margin']))
            assert not out['nearest_label'].any() and (out['nearest_index'] == 0).all()
            assert (out['per_ref_nearest_index'] == 0).all()
            assert out['candidate'].all() and out['majority'] == 1
            assert not torch.equal(out['candidate'], out['nearest_label'])
    check('zero_margin_ties_keep_source_argmax_and_even_vote_rule', ties)

    def reversal():
        aa, bb = full_audit(refs, query, masks), full_audit(refs, query, ~masks)
        for prefix in ('source', 'cosine'):
            a, b = branch(aa, prefix), branch(bb, prefix)
            assert torch.equal(a['nearest_index'], b['nearest_index'])
            assert torch.equal(a['nearest_label'], ~b['nearest_label'])
            assert torch.equal(a['margin'], -b['margin'])
            assert torch.equal(a['per_ref_margin'], -b['per_ref_margin'])
            assert torch.equal(a['candidate'], ~b['candidate'])  # odd R only
    check('label_reversal_signed_margin_and_odd_majority', reversal)

    def chunks():
        aa = full_audit(refs, query, masks, chunk=1)
        max_error = 0.
        for c in (2, 5, 12, 40):
            bb = full_audit(refs, query, masks, chunk=c)
            for prefix in ('source', 'cosine'):
                a, b = branch(aa, prefix), branch(bb, prefix)
                for key in ('fg_max', 'bg_max', 'margin', 'per_ref_margin'):
                    torch.testing.assert_close(a[key], b[key], rtol=1e-14, atol=1e-14)
                    max_error = max(max_error, float((a[key]-b[key]).abs().max()))
                for key in ('nearest_index', 'nearest_label', 'candidate', 'votes'):
                    assert torch.equal(a[key], b[key])
        return dict(max_abs_score_difference=max_error, tie_policy='first row-major maximum')
    check('query_chunk_invariance', chunks)

    def permutation():
        aa = full_audit(refs, query, masks)
        order = torch.tensor([2, 0, 1])
        bb = full_audit(refs[order], query, masks[order])
        for prefix in ('source', 'cosine'):
            a, b = branch(aa, prefix), branch(bb, prefix)
            for key in ('fg_max', 'bg_max', 'margin', 'nearest_label', 'candidate'):
                assert torch.equal(a[key], b[key])
            assert torch.equal(a['per_ref_margin'][order], b['per_ref_margin'])
            local_a = a['nearest_index'] % 12
            assert torch.equal(local_a, b['nearest_index'] % 12)
            assert torch.equal(a['nearest_ref_index'], order[b['nearest_ref_index']])
    check('reference_permutation_no_ties', permutation)

    def missing():
        for label in (False, True):
            mm = torch.full_like(masks, label)
            both = full_audit(refs, query, mm)
            for prefix in ('source', 'cosine'):
                out = branch(both, prefix)
                assert out['state'] == ('NO_BACKGROUND' if label else 'NO_FOREGROUND')
                assert out['state_class'] == 'DEGENERATE'
                assert out['margin'] is None and out['per_ref_margin'] is None
                assert out['bg_max' if label else 'fg_max'] is None
                assert torch.equal(out['candidate'], torch.full_like(masks[0], label))
        mixed = masks.clone()
        mixed[0] = False
        both = full_audit(refs, query, mixed)
        for prefix in ('source', 'cosine'):
            out = branch(both, prefix)
            assert out['state'] == 'BOTH_LABELS' and out['margin'] is not None
            assert out['per_ref'][0]['margin'] is None and out['per_ref_margin'] is None
            assert out['per_ref'][0]['state'] == 'NO_FOREGROUND'
    check('missing_label_explicit_no_fabricated_margin', missing)

    def global_not_majority():
        rr = torch.tensor([[[[0., 1.]], [[1., 0.]]], [[[1., 0.]], [[0., 1.]]],
                           [[[1., 0.]], [[0., 1.]]]], dtype=torch.float64)
        qq = torch.tensor([[[1., 1.]], [[0., 0.]]], dtype=torch.float64)
        mm = torch.tensor([[[False, True]], [[False, True]], [[False, True]]])
        out = audit(rr, qq, mm, chunk=1)
        assert out['nearest_label'].all()  # global first tied witness is FG
        assert not out['candidate'].any()  # two independent source BG votes
        assert (out['votes'] == 1).all()
        # This fixture distinguishes the real source height-axis branch from
        # channel-normalised cosine: normalized reference magnitudes differ.
        axis_refs = torch.tensor([[[[10., 1.], [1000., 1.]],
                                    [[1., .5], [1., .5]]]], dtype=torch.float64)
        axis_query = torch.tensor([[[1., 1.], [1., 1.]],
                                  [[.1, .1], [.1, .1]]], dtype=torch.float64)
        axis_masks = torch.tensor([[[True, False], [False, False]]])
        branches = full_audit(axis_refs, axis_query, axis_masks)
        # If this concrete example happens not to flip, search is NOT allowed;
        # the fixed fixture must exhibit the predicted differing NN location.
        assert not torch.equal(branches['source_nearest_index'], branches['cosine_nearest_index'])
        assert not torch.equal(branches['source_candidate'], branches['cosine_candidate'])
        return dict(source_cosine_candidate_flip_count=int((branches['source_candidate'] != branches['cosine_candidate']).sum()))
    check('multi_reference_global_witness_not_source_majority', global_not_majority)

    def immutable():
        copied = [x.clone() for x in (refs, query, masks)]
        full_audit(refs.requires_grad_(), query.requires_grad_(), masks)
        for actual, before in zip((refs, query, masks), copied):
            assert torch.equal(actual, before)
        assert refs.grad is None and query.grad is None
    check('input_values_unmodified_and_no_grad', immutable)

    def types():
        for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            both = full_audit(refs.to(dtype), query.to(dtype), masks, chunk=4)
            for prefix in ('source', 'cosine'):
                out = branch(both, prefix)
                for key in ('fg_max', 'bg_max', 'margin', 'per_ref_margin', 'nearest_score', 'vote_soft'):
                    assert out[key].dtype == dtype and torch.isfinite(out[key]).all()
        assert not torch.cuda.is_initialized()
        try:
            audit(refs, query, masks.float())
        except TypeError:
            pass
        else:
            raise AssertionError('Soft masks silently accepted')
    check('floating_dtypes_finite_and_cpu_only', types)
    report = dict(state='CPU_REFERENCE_WITNESS_AUDIT_PASSED', passed=len(results), failed=0,
                  checks=results, elapsed_seconds=time.monotonic()-started,
                  torch_version=torch.__version__, cuda_initialized=torch.cuda.is_initialized(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  cpu_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  claims='Function numerical/interface checks only; no task result or pretrained source equivalence')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
