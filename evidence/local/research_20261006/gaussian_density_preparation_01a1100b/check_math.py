"""Finite synthetic positive/negative checks, not a natural segmentation result."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from ics.experiment import render
from ics.methods.reference_gaussian_density import Config, predict
from ics.methods.reference_hull import predict as hull_predict


def run():
    torch.set_num_threads(1)
    angle = np.arange(8)*np.pi/4
    ideal = np.c_[np.cos(angle), np.sin(angle), np.zeros((8, 2))]
    points = ideal.astype(np.float32)
    f_ids = np.repeat(np.arange(8), np.where(np.arange(8) % 2 == 0, 3, 1))
    b_ids = np.repeat(np.arange(8), np.where(np.arange(8) % 2 == 0, 1, 3))
    r = points[np.r_[np.tile(f_ids, 2), np.tile(b_ids, 2)]]
    cov = np.r_[np.ones(32), np.zeros(32)].reshape(8, 8)
    query_ids = np.r_[np.tile(np.arange(0, 8, 2), 8), np.tile(np.arange(1, 8, 2), 8)]
    q = points[query_ids]
    base = np.r_[np.full(32, .49), np.full(32, .51)].reshape(8, 8)
    target = np.arange(64).reshape(8, 8) < 32
    result = predict(q, r, cov, base)
    assert result['info']['foreground_modes'] == result['info']['background_modes'] == 8
    assert np.array_equal(result['field'] > .5, target)
    for key in ('polynomial_control', 'quadratic_ridge_control', 'uniform_control', 'nearest_control', 'centroid_control'):
        assert np.array_equal(result[key] > .5, ~target)
    # Exact common convex support cannot distinguish these differently weighted distributions.
    hull = hull_predict(q, r, cov, base)
    assert np.array_equal(hull['field'] > .5, ~target)

    ideal_f, ideal_b = ideal[f_ids], ideal[b_ids]
    moments = []
    for degree in (1, 2, 3):
        if degree == 1:
            left, right = ideal_f.mean(0), ideal_b.mean(0)
        elif degree == 2:
            left = np.einsum('ni,nj->ij', ideal_f, ideal_f)/len(ideal_f)
            right = np.einsum('ni,nj->ij', ideal_b, ideal_b)/len(ideal_b)
        else:
            left = np.einsum('ni,nj,nk->ijk', ideal_f, ideal_f, ideal_f)/len(ideal_f)
            right = np.einsum('ni,nj,nk->ijk', ideal_b, ideal_b, ideal_b)/len(ideal_b)
        moments.append(float(np.max(np.abs(left-right))))
    assert max(moments) < 1e-15
    fourth_f, fourth_b = float(np.mean(ideal_f[:, 0]**4)), float(np.mean(ideal_b[:, 0]**4))
    assert abs(fourth_f-fourth_b-.125) < 1e-15

    complete_target = np.zeros((1024, 1024), dtype=bool)
    complete_target[:512] = True
    complete = render(result['field'])
    assert np.array_equal(complete, complete_target)
    # Identical observed inputs but shifted semantic assignment: weak correct base is harmed.
    negative_gt = ~complete_target
    negative_weak_errors = int(np.count_nonzero(complete != negative_gt))
    assert negative_weak_errors == 1024*1024
    assert np.array_equal(render(base), negative_gt)
    # Strong correct base survives where a replacement classifier still flips labels.
    strong_base = np.r_[np.full(32, .1), np.full(32, .9)].reshape(8, 8)
    strong = predict(q, r, cov, strong_base)
    assert np.array_equal(render(strong['field']), negative_gt)
    assert np.array_equal(render(strong['standalone_control']), complete_target)
    assert np.max(np.abs(strong['field']-strong_base)) <= Config().maximum_correction+1e-15

    # Translation of only the base and orthogonal feature transformations must behave as defined.
    shifted = predict(q, r, cov, base+.2)
    assert np.allclose(shifted['field']-result['field'], .2, atol=2e-16, rtol=0)
    flipped = predict(-q, -r, cov, base)
    assert np.allclose(flipped['field'], result['field'], atol=2e-15, rtol=0)
    missing = predict(q, r, np.ones_like(cov), base)
    assert np.array_equal(missing['field'], base)
    empty = predict(q, r, np.zeros_like(cov), base)
    assert not empty['field'].any()

    # A two-worker standalone run reads only authorized keys, preserves repetitions, and seals outputs.
    with tempfile.TemporaryDirectory(prefix='gaussian_density_small_') as tmp:
        folder = Path(tmp)
        source = folder/'episode.npz'
        np.savez(source, q=q, r=r, cov=cov, base=base,
                 original_hw=np.array([41, 59]), query_gt=np.array([object()], dtype=object))
        destination = folder/'run'
        subprocess.run([sys.executable, str(ROOT/'scripts/run_reference_gaussian_density.py'),
                        '--inputs', str(source), str(source), '--out', str(destination), '--workers', '2'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert (destination/'prediction_complete.json').exists()
        worker_receipt = json.loads((destination/'000000.json').read_text())
        assert worker_receipt['input_keys_read'] == ['q', 'r', 'cov', 'base', 'original_hw']
        assert worker_receipt['query_gt_used'] is False
        with np.load(destination/'000000.npz') as packet:
            assert packet['field'].shape == (131072,)
            assert packet['field.original'].size == (41*59+7)//8
        assert (destination/'000001.npz').exists()

    source = ROOT/'src/ics/methods/reference_gaussian_density.py'
    report = dict(status='passed', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='unit-feature finite synthetic examples; not natural segmentation', real_episodes=0,
                  reference_labels_first_three_raw_moment_differences=moments,
                  reference_fourth_x_moments=[fourth_f, fourth_b],
                  default_learned_modes=[result['info']['foreground_modes'], result['info']['background_modes']],
                  positive_token_targets=32, positive_token_false_regions_deleted=32,
                  positive_complete_work_mask_errors=0, polynomial_and_nearest_controls_wrong_tokens=64,
                  same_support_hull_control_wrong_tokens=64,
                  negative_same_inputs_weak_correct_base_new_errors=negative_weak_errors,
                  negative_shift_strong_base_preserved=True, replacement_control_wrong_complete=True,
                  fixed_correction_bound=Config().maximum_correction,
                  two_worker_runner_gt_sentinel_unread=True, original_geometry_readout_checked=True,
                  info=result['info'])
    Path(__file__).with_name('math_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'source_sha256', 'real_episodes',
                    'positive_complete_work_mask_errors', 'negative_same_inputs_weak_correct_base_new_errors')}))


if __name__ == '__main__':
    run()
