"""M4 contract checks with a deterministic local non-DINO encoder; no performance claim."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from PIL import Image
from ics.methods import pro_paired_environment as method


class DeterministicEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        rgb = np.asarray(image, np.float64)/255
        local = rgb.reshape(64, 16, 64, 16, 3).mean(axis=(1, 3))
        global_mean = np.broadcast_to(rgb.mean(axis=(0, 1)), local.shape)
        raw = np.concatenate((local, global_mean, np.ones((64, 64, 1))), axis=-1)
        raw -= raw.mean(axis=-1, keepdims=True)
        raw /= np.sqrt(np.mean(raw**2, axis=-1, keepdims=True)+1e-6)
        return raw.astype(np.float32)


def make_encoder(*, binding, device):
    if binding['producer'] != 'deterministic_contract_fixture' or device != 'cpu':
        raise ValueError('This fixture is only a CPU contract check, never a DINO substitute')
    return DeterministicEncoder()


def brute_rectangle(mask):
    candidates = []
    height, width = mask.shape
    for top, bottom in itertools.combinations(range(height+1), 2):
        for left, right in itertools.combinations(range(width+1), 2):
            if not mask[top:bottom, left:right].any():
                candidates.append((-(bottom-top)*(right-left), (top, left, bottom, right)))
    return min(candidates)[1] if candidates else None


def fixture():
    reference = np.empty((1024, 1024, 3), np.uint8)
    reference[:] = [15, 120, 50]
    mask = np.zeros((1024, 1024), bool)
    mask[256:768, 256:768] = True
    reference[mask] = [230, 30, 50]
    yy, xx = np.indices((1024, 1024))
    query = np.stack((20+xx//16, 80+yy//16, 100+xx//32), axis=-1).astype(np.uint8)
    return reference, mask, query


def main():
    rng = np.random.RandomState(20261006)
    for _ in range(40):
        mask = rng.rand(4, 5) < .35
        assert method.maximum_background_rectangle(mask) == brute_rectangle(mask)
    maximum_gap = 0.
    for _ in range(32):
        logits = rng.normal(size=6)
        edges = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]])
        capacity = rng.uniform(0, 2, len(edges))
        actual, certificate = method.exact_potts_cut(logits, edges, capacity)
        optimum = min(method.potts_energy(bits, logits, edges, capacity)
                      for bits in itertools.product((False, True), repeat=6))
        maximum_gap = max(maximum_gap, abs(certificate['energy']-optimum))
        assert abs(certificate['energy']-optimum) < 1e-10
    bank = method.unit(rng.normal(size=(2, 4, 8, 11)))
    means, env, parts, ridge = method.covariance_factors(bank)
    difference = means[0]-means[1]
    errors = []
    for factor in (env, np.concatenate((env, parts), axis=1)):
        direction, receipt = method.inverse_direction(factor, difference, ridge)
        direct = np.linalg.solve(np.einsum('ik,jk->ij', factor, factor, optimize=False)+ridge*np.eye(11), difference)
        errors.append(float(np.max(np.abs(direction-direct))))
        assert np.allclose(direction, direct, rtol=1e-10, atol=1e-10)
        logits, info = method.normalized_logits(means, direction, means)
        assert np.allclose(logits, [1., -1.], atol=1e-12)
    # Class within-covariance decomposition uses exactly the main numerical ridge.
    residual = bank-means[:, None, None, :]
    flat_residual = residual.reshape(-1, 11)
    full_class_covariance = np.einsum('ki,kj->ij', flat_residual, flat_residual, optimize=False)/(2*4*8)
    assert np.allclose(full_class_covariance, np.einsum('ik,jk->ij', env, env, optimize=False)+
                       np.einsum('ik,jk->ij', parts, parts, optimize=False), atol=1e-14)
    # Pixel-to-patch-center mapping reproduces a known affine feature plane.
    yy, xx = np.indices((64, 64))
    feature_plane = np.stack((xx+.5, yy+.5, np.ones((64, 64))), axis=-1)
    positions = np.array([[160., 320.], [240., 560.]])
    expected = method.unit(np.column_stack((positions/16, np.ones(2))))
    assert np.allclose(method.sample_points(feature_plane, positions), expected, atol=1e-14)
    from scipy.optimize import brentq
    from scipy.special import expit
    ce_bank = np.zeros((2, 4, 8, 3), np.float64)
    ce_bank[0, ..., 0], ce_bank[1, ..., 0] = 1., -1.
    ce_direction, ce_bias, ce_receipt = method.fit_ce_direction(ce_bank)
    ce_expected = brentq(lambda value: value/3-expit(-value), 0, 3)
    assert abs(ce_direction[0]-ce_expected) < 1e-7 and abs(ce_bias) < 1e-10
    assert np.max(np.abs(ce_direction[1:])) < 1e-12 and ce_receipt['iterations'] <= 100
    assert ce_receipt['reached_gradient_target']
    irregular_mask = np.zeros((512, 512), bool)
    irregular_mask[64:448, 96:416] = True
    irregular_mask[200:240, 160:352] = False
    selected, coverage, small_shape = method.canonical_points(irregular_mask)
    assert len(selected) == 32 and np.all(coverage >= .9)
    small_mask = np.asarray(Image.fromarray(irregular_mask.astype(np.uint8)).resize(
        (small_shape[1], small_shape[0]), Image.Resampling.NEAREST))
    for x, y in selected:
        row, col = int(y*small_shape[0]), int(x*small_shape[1])
        assert irregular_mask[int(y*512), int(x*512)]
        assert small_mask[row-8:row+8, col-8:col+8].mean() >= .9
    reference, mask, query = fixture()
    geometry = method.prepare_geometry(reference, mask)
    assert len(geometry['conditions']) == 4 and len(geometry['points_xy']) == 32
    negative_box = geometry['negative_box']
    top, left, bottom, right = negative_box
    assert not mask[top:bottom, left:right].any()
    expected_conditions = [(128, [256, 256]), (128, [768, 768]),
                           (256, [256, 256]), (256, [768, 768])]
    assert [(c['long_edge'], c['center_xy']) for c in geometry['conditions']] == expected_conditions
    for condition in geometry['conditions']:
        top, left, bottom, right = condition['box_tlbr']
        outside = np.ones(mask.shape, bool)
        outside[top:bottom, left:right] = False
        for name in ('positive', 'negative'):
            edited = method.transplant(query, geometry[name], condition)
            assert np.array_equal(edited[outside], query[outside])
        xy = method.mapped_points(geometry['points_xy'], condition)
        assert np.all(xy[:, 0]-left >= 16) and np.all(right-xy[:, 0] >= 16)
        assert np.all(xy[:, 1]-top >= 16) and np.all(bottom-xy[:, 1] >= 16)
    native_binding = dict(method='full_foris_native', recipe_sha256='a'*64)
    encoder_binding = dict(producer='deterministic_contract_fixture')
    fallback_calls = []

    def native(_r, _m, _q, _features):
        fallback_calls.append(True)
        return dict(mask_work=np.zeros((1024, 1024), bool),
                    mask_original=np.zeros(_q.shape[:2], bool), info=dict(encoder_forwards=0, cached=True))

    query_identity = hashlib.sha256(query.tobytes()).hexdigest()
    encoder = DeterministicEncoder()
    result = method.predict(reference, mask, query, encoder, native,
                            native_binding=native_binding, encoder_binding=encoder_binding)
    assert encoder.calls == 17 and result['info']['primary_encoder_forwards'] == 9
    assert not fallback_calls
    assert hashlib.sha256(query.tobytes()).hexdigest() == query_identity
    assert result['info']['statistics']['paired']['ridge'] == result['info']['statistics']['class_lda']['ridge']
    assert all(record['untouched_query_pixels_labeled'] == 0 for record in result['info']['interventions'])
    for start in range(0, 16, 2):
        records = result['info']['interventions'][start:start+2]
        assert records[0]['labeled_points_xy'] == records[1]['labeled_points_xy']
        assert records[0]['training_label'] == 1 and records[1]['training_label'] == -1
    assert all(value['mask_work'].shape == (1024, 1024) and value['mask_original'].shape == query.shape[:2]
               for value in result['arms'].values())
    geometry_encoder = DeterministicEncoder()
    failed_geometry = method.predict(reference, np.zeros_like(mask), query, geometry_encoder, native,
                                     native_binding=native_binding, encoder_binding=encoder_binding)
    assert geometry_encoder.calls == 0 and all(v['info']['fallback_category'] == 'geometry'
                                               for v in failed_geometry['arms'].values())
    constant = np.zeros((64, 64, 2), np.float32)
    constant[..., 0] = 1
    failed_statistics = method.predict(reference, mask, query, lambda _: constant, native,
                                       native_binding=native_binding, encoder_binding=encoder_binding)
    assert all(v['info']['fallback_category'] == 'statistical'
               for key, v in failed_statistics['arms'].items() if key != 'ce_self')
    assert not failed_statistics['arms']['ce_self']['info']['fallback']
    assert len(fallback_calls) == 2  # Shared lazy native once per failed episode.
    # The real CLI accepts actual RGB files and seals all four arms, ignoring an unread GT sentinel path.
    with tempfile.TemporaryDirectory(prefix='m4_cli_', dir=Path(__file__).parent) as folder:
        folder = Path(folder)
        for filename, value in (('reference.png', reference), ('mask.png', mask.astype(np.uint8)), ('query.png', query)):
            Image.fromarray(value).save(folder/filename)
        np.savez_compressed(folder/'native.npz', mask_work=np.zeros(mask.shape, np.uint8),
                            mask_original=np.zeros(query.shape[:2], np.uint8))
        row = dict(id='fixture_0', reference_rgb=str(folder/'reference.png'), reference_mask=str(folder/'mask.png'),
                   query_rgb=str(folder/'query.png'), native_npz=str(folder/'native.npz'),
                   native_sha256=hashlib.sha256((folder/'native.npz').read_bytes()).hexdigest(),
                   query_gt='/nonexistent/DO_NOT_READ_QUERY_GT')
        (folder/'manifest.json').write_text(json.dumps([row]))
        (folder/'encoder_binding.json').write_text(json.dumps(encoder_binding))
        (folder/'native_binding.json').write_text(json.dumps(native_binding))
        envvars = dict(os.environ, PYTHONPATH=str(Path(__file__).parent)+os.pathsep+str(ROOT/'src'),
                       PYTHONDONTWRITEBYTECODE='1')
        completed = subprocess.run([sys.executable, str(ROOT/'scripts/run_pro_paired_environment.py'),
                                    '--manifest', str(folder/'manifest.json'), '--out', str(folder/'run'),
                                    '--encoder-factory', 'check_m4:make_encoder', '--expected', '1', '--threads', '1',
                                    '--encoder-binding', str(folder/'encoder_binding.json'),
                                    '--native-binding', str(folder/'native_binding.json')],
                                   env=envvars, check=True, text=True, capture_output=True)
        seal = json.loads((folder/'run/sealed.json').read_text())
        assert seal['state'] == 'ALL_PREDICTIONS_SEALED' and seal['n'] == 1
        assert 'query_gt' not in json.loads((folder/'run/inference_manifest.json').read_text())[0]
    report = dict(kind='deterministic_non_DINO_contract_check', real_episodes=0,
                  background_rectangle_enumeration_cases=40, exact_cut_enumeration_cases=32,
                  maximum_cut_energy_gap=maximum_gap, covariance_solve_errors=errors,
                  class_covariance_decomposition=True, class_lda_reuses_main_lambda=True,
                  CE_known_convex_optimum=True, CE_gradient_target_reached=ce_receipt['reached_gradient_target'],
                  six_complete_readout_arms=True,
                  fixed_four_conditions=True, paired_point_mapping=True,
                  pixel_to_feature_bilinear_mapping=True,
                  irregular_mask_foreground_and_coverage=True,
                  unchanged_query_bytes=True, untouched_query_pixels_never_labeled=True,
                  canonical_point_count=32, primary_encoder_calls=9, all_four_arm_encoder_calls=17,
                  geometry_fallback_before_encoder=True, statistical_fallback_bound_native=True,
                  cli_sealed=True, cli_query_gt_sentinel_unread=True, cli_stdout=completed.stdout.strip(),
                  encoder='deterministic RGB pooled features with per-token normalization; not DINOv3',
                  actual_DINO_validation='not_run', no_gpu=True, no_remote=True)
    Path(__file__).with_name('check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
