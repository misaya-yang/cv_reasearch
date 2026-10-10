"""Independent no-GT math and frozen-input audit for the scene method."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = REPO.parent / 'cv_data'
METHOD = REPO / 'src/ics/methods/autonomous_context_transport.py'
RUNNER = REPO / 'scripts/autonomous_scene_reconstruction.py'
EXPERIMENT = DATA / 'a/autonomous_scene_reconstruction600_20261010'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array_hash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(json.dumps([list(x.shape), x.dtype.str]).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def load_method():
    spec = importlib.util.spec_from_file_location('audited_scene_method', METHOD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_binary_primal(x, sample, strength):
    """Solve the ORIGINAL two-role objective including intercept in one system."""
    x = x.double()
    z = torch.cat((x, torch.ones((len(x), 1), dtype=torch.float64)), dim=1)
    p, n = sample['positive'], sample['negative']
    lhs = z.T @ ((.5*(p+n))[:, None]*z)
    lhs[:-1, :-1] += strength*torch.eye(x.shape[1], dtype=torch.float64)
    rhs = z.T @ (.5*(p-n))
    theta = torch.linalg.solve(lhs, rhs)
    return theta[:-1], theta[-1]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(8031)
    method = load_method()
    started = time.perf_counter()
    reference = F.normalize(torch.randn(96, 12), dim=1)
    query = F.normalize(torch.randn(80, 12), dim=1)
    # Binary-exact continuous coverages make mask complement sampling symmetric.
    coverage = (np.arange(96) % 17).astype(np.float32)/16
    sample = method._role_sample(coverage)
    results = {}
    for name, cloud, grid in (('scene', query, (8, 10)), ('source_kernel', reference, (8, 12))):
        mixture = method._mixture(cloud, grid)
        basis, responsibility = method._basis(reference, mixture)
        fit = method._ridge(basis[sample['ids']], sample)
        w, b = independent_binary_primal(basis[sample['ids']], sample, method.RIDGE_STRENGTH)
        pi = mixture['occupancy']
        u = fit['coefficient']/pi.sqrt()
        pred = basis[sample['ids']] @ w+b
        direct_binary = (.5*sample['positive']*(pred-1).square()
                         +.5*sample['negative']*(pred+1).square()).sum()
        collapsed = (sample['weights']*(pred-sample['target']).square()).sum()
        constant = (sample['weights']*(1-sample['target'].square())).sum()
        check = dict(primal_coefficient_max_error=float((w-fit['coefficient']).abs().max()),
                     primal_intercept_error=float((b-fit['bias']).abs()),
                     original_collapsed_loss_error=float((direct_binary-collapsed-constant).abs()),
                     occupancy_sum_error=float((pi.sum()-1).abs()),
                     responsibility_row_sum_max_error=float((responsibility.sum(1)-1).abs().max()),
                     scene_norm_identity_error=float((pi @ u.square()-fit['coefficient'].square().sum()).abs()),
                     min_occupancy=float(pi.min()), components=len(pi))
        assert max(check[k] for k in check if k.endswith('error')) < 1e-10, check
        results[name] = check
    fit = method._ridge(reference[sample['ids']], sample)
    w, b = independent_binary_primal(reference[sample['ids']], sample, method.RIDGE_STRENGTH)
    results['support_ridge'] = dict(primal_coefficient_max_error=float((w-fit['coefficient']).abs().max()),
                                    primal_intercept_error=float((b-fit['bias']).abs()))
    assert max(results['support_ridge'].values()) < 1e-10
    r_before, q_before, c_before = reference.clone(), query.clone(), coverage.copy()
    first = method.fit_predict(reference, coverage, query, (8, 10))
    repeat = method.fit_predict(reference, coverage, query, (8, 10))
    complement = method.fit_predict(reference, 1-coverage, query, (8, 10))
    results['field_properties'] = dict(
        byte_identical_repeat={arm: np.array_equal(first['fields'][arm], repeat['fields'][arm]) for arm in first['fields']},
        complement_max_sign_error={arm: float(np.max(np.abs(first['fields'][arm]+complement['fields'][arm]))) for arm in first['fields']},
        input_not_mutated=bool(torch.equal(reference, r_before) and torch.equal(query, q_before) and np.array_equal(coverage, c_before)),
        json_allow_nan_false=True,
        field_sha256={arm: array_hash(field) for arm, field in first['fields'].items()})
    json.dumps(first['diagnostics'], allow_nan=False)
    assert all(results['field_properties']['byte_identical_repeat'].values())
    assert max(results['field_properties']['complement_max_sign_error'].values()) < 1e-6
    assert results['field_properties']['input_not_mutated']

    # Synthetic renderer checks use the same released helper, no CRF backend.
    sys.path.insert(0, str(DATA / 'third_party/foris_official'))
    from utils.refinement import upsample_mask, crf_refine
    binary = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.bool)
    expected = F.interpolate(binary.float()[None, None], (31, 23), mode='bilinear', align_corners=False)[0, 0] > .5
    assert torch.equal(upsample_mask(binary, 31, 23), expected)
    class IdentitySolver:
        def __call__(self, image, logits):
            return logits
    normalized_image = torch.zeros((1, 3, 31, 23))
    refined = crf_refine(IdentitySolver(), 10, .95, normalized_image, expected)
    # Identity solver has a deterministic equal-logit (background) boundary.
    results['renderer'] = dict(original_size_helper_exact=True, crf_result_shape=list(refined.shape),
                               crf_result_bool=refined.dtype == torch.bool,
                               backend_executed=False,
                               strict_zero_interpolation='runner uses continuous signed field to1024 before >0; no minmax')

    # Real INPUT METADATA only: never open reference/query mask pixels or predictions.
    metadata = dict(experiment_prepared=bool((EXPERIMENT/'config.json').exists()), query_GT_reads=0,
                    real_prediction_arrays_read=0, real_feature_arrays_read=0)
    if metadata['experiment_prepared']:
        cfg = read(EXPERIMENT/'config.json')
        manifest, tasks = read(EXPERIMENT/'manifest.json'), read(EXPERIMENT/'tasks.json')
        fields = ('episode_id', 'reference_rgb_hash', 'query_rgb_hash', 'reference_mask_hash',
                  'reference_crop', 'query_crop', 'reference_size_hw', 'query_size_hw')
        roots = {
            'deepglobe': DATA/'a/joint_role_pilot200_20261010',
            'paco_part': DATA/'a/joint_role_pilot200_20261010',
            'coco': DATA/'a/coco_role_competition200_20261010',
            'lvis': DATA/'a/lvis_mean200_20261008/run'}
        archived = {key: {row['episode_id']: row for row in read(root/'manifest.json')} for key, root in roots.items()}
        foris_lvis = {row['episode_id']: row for row in read(DATA/'a/lvis_foris1400_score_reuse_20261009/run/manifest.json')}
        mismatches = []
        datasets = {}
        for row, task in zip(manifest, tasks):
            dataset = row['dataset']
            datasets[dataset] = datasets.get(dataset, 0)+1
            if dataset not in archived:
                # Preserve actual custom Deep naming without guessing its protocol.
                archived[dataset] = {r['episode_id']: r for r in read(roots['deepglobe']/'manifest.json')}
            comparisons = [('primary_source', archived[dataset][row['episode_id']])]
            if dataset == 'lvis':
                comparisons.append(('foris_source', foris_lvis[row['episode_id']]))
            for label, prior in comparisons:
                for field in fields:
                    if row.get(field) != prior.get(field):
                        mismatches.append(dict(episode_id=row['episode_id'], source=label, field=field))
            assert task['episode_id'] == row['episode_id']
        frozen_matches = {relative: sha(EXPERIMENT/'frozen'/relative) == digest for relative, digest in cfg['source_sha256'].items()}
        metadata.update(n=len(manifest), task_n=len(tasks), datasets=datasets,
                        baseline_manifest_fields=list(fields), baseline_input_mismatches=mismatches,
                        frozen_source_matches=frozen_matches,
                        live_method_matches_frozen=sha(METHOD) == cfg['source_sha256']['src/ics/methods/autonomous_context_transport.py'],
                        live_runner_matches_frozen=sha(RUNNER) == cfg['source_sha256']['scripts/autonomous_scene_reconstruction.py'],
                        manifest_hash_matches=sha(EXPERIMENT/'manifest.json') == cfg['manifest_sha256'],
                        task_hash_matches=sha(EXPERIMENT/'tasks.json') == cfg['tasks_sha256'],
                        profile_hash_matches=sha(cfg['profile_path']) == cfg['profile_sha256'],
                        basis_hash_matches=sha(cfg['basis_path']) == cfg['basis_sha256'])
        assert not mismatches, mismatches
        assert all(frozen_matches.values())
        assert all(metadata[key] for key in ('manifest_hash_matches', 'task_hash_matches', 'profile_hash_matches', 'basis_hash_matches'))
    report = dict(status='PASSED_NO_GT_INDEPENDENT_STATIC_AND_SYNTHETIC_REVIEW',
                  blocking_findings=[],
                  nonblocking_provenance_gap=dict(location='scripts/autonomous_scene_reconstruction.py:139',
                      fact='Config does not bind all imported external CRF Python/C++ sources or the compiled native binary.',
                      observed_output_defect=False,
                      correction='At output validation, bind imported CRF source/backend receipts; or extend a future runner freeze.'),
                  contrast_scope='Complete query-mixture representation versus source-mixture representation; centers and occupancies both change.',
                  complement_symmetry_scope='Fixed caller-normalized features; legal foreground-dependent APD may change after raw-mask complement.',
                  source_sha256={str(METHOD.relative_to(REPO)): sha(METHOD), str(RUNNER.relative_to(REPO)): sha(RUNNER)},
                  fixture_source_sha256=sha(__file__), math=results, prepared_input_metadata=metadata,
                  runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cpu_threads=2),
                  elapsed_seconds=time.perf_counter()-started,
                  limitations=['No real feature inference or scoring run by this validator.',
                               'CRF lattice parity and actual-sample determinism await sealed-output validation.',
                               'Synthetic checks do not establish novelty, generalization, accuracy, or practical end-to-end latency.'])
    (OUT/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
