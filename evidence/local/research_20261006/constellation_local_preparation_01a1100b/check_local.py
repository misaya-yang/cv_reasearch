"""Existing matched/unmatched synthetic counterexample, v2's changed contract."""
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from ics.methods.reference_constellation import Config, predict as v1_predict
from ics.methods.reference_constellation_local import METHOD_ID, predict


def fixture(dimension=16):
    # check_geometry.py:55-66 plus audit_4_6.md:80's unmatched base component.
    q = np.zeros((4096, dimension), dtype=np.float32)
    r = np.zeros_like(q)
    q[:, 4] = 1
    r[:, 4] = 1
    cov = np.zeros((64, 64))
    base = np.full((64, 64), .2)
    matched = np.zeros_like(cov, dtype=bool)
    anchors = np.array([10+10j, 20+10j, 10+20j, 20+20j])
    for k, point in enumerate(anchors):
        i = int(point.imag) * 64 + int(point.real)
        query = point + 28 + 28j
        j = int(query.imag) * 64 + int(query.real)
        r[i] = 0
        r[i, k] = 1
        cov[int(point.imag), int(point.real)] = 1
        q[j] = 0
        q[j, k] = 1
        matched[int(query.imag), int(query.real)] = True
    unmatched = np.zeros_like(cov, dtype=bool)
    unmatched[2:6, 45:49] = True
    base[unmatched] = .9
    return q, r, cov, base, matched, unmatched


def main():
    q, r, cov, base, matched, unmatched = fixture()
    expected = v1_predict(q, r, cov, base)
    output = predict(q, r, cov, base)
    region = output['local_region']
    assert output['info']['retained_poses'] == 1
    assert not (region & unmatched).any()
    np.testing.assert_array_equal(output['field'][~region], base[~region])
    np.testing.assert_array_equal(output['field'][region], expected['field'][region])
    for key, value in (('global_field', expected['field']), ('prior', expected['prior']),
                       ('bag_field', expected['bag_field'])):
        np.testing.assert_array_equal(output[key], value)
    assert output['info']['poses'] == expected['info']['poses']
    assert int((output['field'][matched] > .5).sum()) == 4
    assert int((expected['field'][matched] > .5).sum()) == 4
    assert int((output['field'][unmatched] > .5).sum()) == 16
    assert int((expected['field'][unmatched] > .5).sum()) == 0
    # Values outside the region are preserved even when they are outside [0,1].
    unclipped = base.copy()
    unclipped[0, 0], unclipped[0, 1] = -.2, 1.2
    outside = predict(q, r, cov, unclipped)
    np.testing.assert_array_equal(outside['field'][~outside['local_region']],
                                  unclipped[~outside['local_region']])
    # The same unsupported .9 component is retained if it is actually a distractor.
    # Identity is an assumption in this synthetic example, never a query input.
    unsupported_false_target_kept = int((output['field'][unmatched] > .5).sum())
    one = np.zeros_like(r)
    one[:, 0] = 1
    no_pose = predict(q, one, cov, base)
    assert no_pose['info']['abstention'] and not no_pose['local_region'].any()
    np.testing.assert_array_equal(no_pose['field'], base)
    zero = predict(q, r, cov, base, Config(geometry_weight=0))
    np.testing.assert_array_equal(zero['field'], base)
    sources = ['src/ics/methods/reference_constellation_local.py',
               'src/ics/methods/reference_constellation.py',
               'src/ics/methods/reference_occupancy.py']
    report = dict(
        kind='existing_synthetic_counterexample_and_contract_checks_only', method=METHOD_ID,
        real_episodes=0, synthetic_targets_have_assumed_identity=True,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        matched_tokens=4, unmatched_tokens=16,
        global_v1_matched_kept=int((expected['field'][matched] > .5).sum()),
        local_v2_matched_kept=int((output['field'][matched] > .5).sum()),
        global_v1_unmatched_kept=int((expected['field'][unmatched] > .5).sum()),
        local_v2_unmatched_kept=int((output['field'][unmatched] > .5).sum()),
        retained_poses=output['info']['retained_poses'], local_region_tokens=int(region.sum()),
        outside_region_maximum_field_difference=float(np.max(np.abs(output['field'][~region] - base[~region]))),
        unchanged_v1_pose_prior_bag_controls=True, inside_region_identical_to_v1=True,
        outside_unclipped_values_preserved=True, no_pose_and_zero_weight_keep_base=True,
        unsupported_false_target_tokens_kept=unsupported_false_target_kept,
        local_info=output['info'], real_gain='unmeasured', real_runtime='unmeasured',
        no_gpu=True, no_server=True,
    )
    Path(__file__).with_name('local_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'method', 'global_v1_matched_kept', 'local_v2_matched_kept', 'global_v1_unmatched_kept',
        'local_v2_unmatched_kept', 'outside_region_maximum_field_difference',
        'unsupported_false_target_tokens_kept', 'real_episodes')}))


if __name__ == '__main__':
    main()
