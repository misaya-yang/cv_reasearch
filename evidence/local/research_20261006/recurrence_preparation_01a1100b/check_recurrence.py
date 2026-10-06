"""Full-query recurrence construction and failure, using synthetic unit features."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from ics.methods.query_recurrence import Config, choose_star, predict


def fixture():
    q = np.zeros((4096, 1024), dtype=np.float32)
    r = np.zeros_like(q)
    q[:, 2] = 1
    r[:, 2] = 1
    cov = np.zeros((64, 64))
    cov[2:10, 2:10] = 1
    r.reshape(64, 64, 1024)[2:10, 2:10] = np.eye(1, 1024, dtype=np.float32)[0]
    base = np.full((64, 64), .1)
    true = np.zeros_like(base, dtype=bool)
    false = np.zeros_like(true)
    for index, (row, col) in enumerate(((16, 4), (16, 24), (36, 4))):
        true[row:row+8, col:col+8] = True
        base[row:row+8, col:col+8] = .8 if index < 2 else .45
        block = q.reshape(64, 64, 1024)[row:row+8, col:col+8]
        block[:] = 0
        block[..., 0] = .6
        block[..., 1] = .8
    false[36:44, 40:48] = True
    base[false] = .8
    q[false.ravel()] = 0
    q[false.ravel(), 0] = .8
    q[false.ravel(), 1] = -.6
    return q, r, cov, base, true, false


def main():
    q, r, cov, base, target, distractor = fixture()
    result = predict(q, r, cov, base)
    np.testing.assert_array_equal(result['field'] > .5, target)
    assert not result['info']['abstention'] and len(result['info']['selected_seed_labels']) == 2
    assert (result['all_seed_control'][distractor] > .5).all()
    assert not (result['single_seed_control'][36:44, 4:12] > .5).any()
    # Single trusted instance provides no recurrence certificate.
    only_one = base.copy()
    only_one[16:24, 24:32] = .45
    only_one[distractor] = .45
    np.testing.assert_array_equal(predict(q, r, cov, only_one)['field'], only_one)
    low = np.full_like(base, .49)
    np.testing.assert_array_equal(predict(q, r, cov, low)['field'], low)
    np.testing.assert_array_equal(predict(q, r, cov, base, Config(correction_weight=0))['field'], base)
    # Unit features under a common sign flip preserve all cosine decisions.
    transformed = predict(-q, -r, cov, base)
    np.testing.assert_allclose(transformed['field'], result['field'], atol=1e-7)
    # No target labels are used to prefer one of two cosine stars.
    desc = np.array([[1., 0], [1., 0], [0., 1], [0., 1], [0., 1]])
    _, members, _ = choose_star(desc, np.full(5, .6), np.full(5, .8), Config())
    np.testing.assert_array_equal(members, [2, 3, 4])
    # Fragmentation cannot create more than32 feature descriptors.
    fragmented_base = np.full_like(base, .1)
    fragmented_q = q.copy()
    for row in range(0, 64, 8):
        for col in range(0, 64, 8):
            fragmented_base[row:row+2, col:col+4] = .8
            block = fragmented_q.reshape(64, 64, 1024)[row:row+2, col:col+4]
            block[:] = 0
            block[..., 0] = .6
            block[..., 1] = .8
    capped = predict(fragmented_q, r, cov, fragmented_base)
    assert capped['info']['available_seed_components'] == 64
    assert capped['info']['seed_components'] == 32
    assert len(capped['info']['selected_seed_labels']) == 32
    # Actual negative complete example: a repeated wrong category wins the
    # majority gate. Its reference cosine is even higher than the true category.
    bad_q, bad_base = q.copy(), base.copy()
    for row, col in ((4, 40), (16, 40)):
        bad_base[row:row+8, col:col+8] = .8
        block = bad_q.reshape(64, 64, 1024)[row:row+8, col:col+8]
        block[:] = 0
        block[..., 0] = .8
        block[..., 1] = -.6
    bad = predict(bad_q, r, cov, bad_base)
    assert len(bad['info']['selected_seed_labels']) == 3
    assert (bad['field'][distractor] > .5).all()
    assert not (bad['field'][36:44, 4:12] > .5).any()
    report = dict(
        kind='synthetic_complete_algorithm_checks', real_episodes=0,
        complete_output=dict(true_tokens=int(target.sum()), false_tokens=0,
                             recovered_tokens=64, deleted_false_tokens=64),
        all_seed_control_keeps_isolated_distractor=True,
        reference_best_single_seed_fails_weak_target=True,
        single_instance_abstention=True, all_weak_abstention=True, zero_weight_exact=True,
        common_feature_sign_invariance=True,
        fragmented_query_64_components_capped_at_32=True,
        repeated_wrong_category_counterexample=dict(selected_wrong_seeds=3, weak_target_recovered=False,
                                                    wrong_region_deleted=False),
        dimensions=[4096, 1024], candidate_and_controls_seconds=result['info']['wall_seconds'],
        timing_scope='Synthetic input already loaded; excludes shared baseline construction, I/O, 1024 rendering and cohort scoring',
        candidate_info=result['info'], real_gain='unmeasured', real_runtime='unmeasured',
        no_gpu=True, no_server=True,
    )
    Path(__file__).with_name('recurrence_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(complete_output=report['complete_output'],
                          repeated_wrong_category_counterexample=report['repeated_wrong_category_counterexample'],
                          synthetic_seconds=report['candidate_and_controls_seconds'], real_episodes=0)))


if __name__ == '__main__':
    main()
