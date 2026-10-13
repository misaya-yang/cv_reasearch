"""B4's same twelve fixed arms, replacing K64 means by A7 reciprocal graph.

Only sealed independent_c/addition_* fields are method evidence. Complete FoRIS
and MEAN controls are reused solely by score, after every natural mask is sealed.
No new parameter search, encoding, query-GT inference, or confirmation access.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg
import lab
import plan_common as pc
import plan_report as pr
import a2_exchange as workflow
import a7_generic as a7
import b4_minimal as b4
from p0_manifest import raw_metadata

REQUIRED = ('query_neighbors20',)
GEOMETRY_CACHE = OrderedDict()


def reciprocal_geometry(q, neighbors):
    """A7 W/degree, constructed once and shared by all twelve unary fields."""
    q = F.normalize(torch.as_tensor(q, dtype=torch.float32), dim=1)
    n = q.shape[0]
    ids = torch.as_tensor(neighbors, dtype=torch.long)
    assert ids.shape == (n, 20) and int(ids.min()) >= 0 and int(ids.max()) < n
    assert not bool((ids == torch.arange(n)[:, None]).any()), 'Self-neighbor in P0 graph'
    similarities = torch.empty((n, 20), dtype=torch.float32)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        similarities[start:stop] = (q[start:stop, None, :] * q[ids[start:stop]]).sum(-1)
    distance = (1 - similarities).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(n), 20), ids.numpy().ravel())), shape=(n, n))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel()
    w = w / max(float(degree.mean()), 1e-8)
    degree = np.asarray(w.sum(1)).ravel()
    return {'w': w, 'degree': degree, 'laplacian16': 16 * (sparse.diags(degree) - w),
            'n': n, 'reciprocal_undirected_edges': int(w.nnz // 2)}


def graph_from_geometry(field, geometry):
    """Same A7 minmax/A/lambda16/CG, with a field-specific confidence A."""
    s = lab.minmax(torch.as_tensor(field, dtype=torch.float32).reshape(-1)).numpy()
    assert s.size == geometry['n'] and np.isfinite(s).all()
    a = .1 + np.abs(2 * s - 1)
    a = (a / a.mean()).astype(np.float64)
    y = s.astype(np.float64)
    H = sparse.diags(a) + geometry['laplacian16']
    rhs = a * y
    iterations = [0]

    def callback(unused):
        iterations[0] += 1

    z, status = cg(H, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
    if status:
        raise RuntimeError('Independent graph CG failure: ' + str(status))
    assert np.isfinite(z).all()
    info = {'lambda': 16, 'neighbors': 20, 'neighbor_source': 'P0 query_neighbors20',
            'reciprocal_undirected_edges': geometry['reciprocal_undirected_edges'],
            'cg_iterations': iterations[0],
            'cg_relative_residual': float(np.linalg.norm(H @ z - rhs) / max(np.linalg.norm(rhs), 1e-12)),
            'rank_correction': False, 'FoRIS_score_consumed': False,
            'confidence': 's=minmax(c+sum(active additions)); A=(.1+abs(2*s-1))/mean(.1+abs(2*s-1))'}
    return z.reshape(64, 64).astype(np.float32), info


def compose_field(c, additions, active):
    correction = np.zeros_like(c, dtype=np.float32)
    for name in active:
        correction += additions[name]
    return np.asarray(c, np.float32) + correction


@lru_cache(8)
def query_features(payload):
    metadata = raw_metadata(str(payload))
    assert pc.sha(payload) == metadata['payload_sha256'], 'Cached query O24 changed'
    with np.load(payload, allow_pickle=False) as archive:
        raw = archive['O/24'].copy()
    assert raw.shape == (4096, 1024) and raw.dtype == np.float32 and np.isfinite(raw).all()
    return lab.debias(torch.from_numpy(raw)).cpu()


def shared_geometry(binding, neighbors):
    payload = binding['fit']['query_raw_identity']['payload_path']
    metadata = raw_metadata(str(payload))
    neighbor_sha = hashlib.sha256(np.ascontiguousarray(neighbors).tobytes()).hexdigest()
    identity = (metadata['input_sha256'], metadata['tensor_sha256'], neighbor_sha)
    if identity in GEOMETRY_CACHE:
        GEOMETRY_CACHE.move_to_end(identity)
        return GEOMETRY_CACHE[identity], True
    geometry = reciprocal_geometry(query_features(str(payload)), neighbors)
    GEOMETRY_CACHE[identity] = geometry
    if len(GEOMETRY_CACHE) > 8:
        GEOMETRY_CACHE.popitem(last=False)
    return geometry, False


def source_run(cohort):
    root = lab.ASSETS / 'a/x1012_B4' / cohort
    if not (root / 'sealed.json').is_file():
        raise RuntimeError('Wait for the original B4 natural seal: ' + str(root / 'sealed.json'))
    seal = pc.read(root / 'sealed.json')
    config = pc.read(root / 'config.json')
    method = pc.read(root / 'method.freeze.json')
    rows = pc.read(root / 'manifest.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED'
    assert pc.sha(root / 'manifest.json') == seal['manifest_sha256']
    assert pc.sha(root / 'config.json') == seal['config_sha256']
    assert seal['n'] == len(rows) == len(seal['records'])
    assert [r['index'] for r in seal['records']] == [pc.key(b) for b in rows]
    assert not any(b.get('split') == 'confirm' for b in rows)
    assert config['arms'] == list(b4.natural_arms(config['method']['with_crf']))
    assert method['candidates'] == {a: b4.budget(v) for a, v in b4.ACTIVE.items()}
    assert pc.sha(lab.BASIS) == method['basis_sha256']
    for source, expected in config['source_sha256'].items():
        assert pc.sha(source) == expected, 'Original frozen B4 source changed: ' + source
    return root, rows, config, seal, method


def prepare(cohort):
    root, rows, original, source_seal, source_method = source_run(cohort)
    out = lab.ASSETS / 'a/x1012_B4_graph' / cohort
    source_sha = dict(original['source_sha256'])
    for path in (Path(__file__), Path(a7.__file__), root / 'sealed.json', root / 'method.freeze.json',
                 root / 'config.json', root / 'manifest.json'):
        source_sha[str(path)] = pc.sha(path)
    method = copy.deepcopy(source_method)
    method.update(decision_basis='reciprocal_graph',
                  readout='A7 independent_graph(c+sum(active additions), q, P0 query_neighbors20); bilinear1024 -> minmax -> >.5',
                  zero_step='A7 independent_graph(c); same frozen c/neighbor identities and graph formula',
                  graph={'lambda': 16, 'neighbors': 20, 'minmax_before_graph': True,
                         'field_specific_confidence_A': True, 'rank_correction': False,
                         'CG': {'rtol': 1e-7, 'atol': 1e-9, 'maxiter': 300}},
                  source_sha256=source_sha,
                  controls='read original B4 separately sealed complete FoRIS/MEAN only in score; never natural inputs',
                  source_B4_method_path=str(root / 'method.freeze.json'),
                  source_B4_method_sha256=pc.sha(root / 'method.freeze.json'))
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'config.json'
    if not path.exists():
        for name in ('predictions', 'fields', 'records', 'frozen'):
            (out / name).mkdir(exist_ok=True)
        for code in (Path(__file__), Path(a7.__file__), Path(b4.__file__), Path(workflow.__file__), Path(pc.__file__), Path(pr.__file__), Path(lab.__file__)):
            shutil.copyfile(code, out / 'frozen' / code.name)
        pc.write(out / 'manifest.json', rows)
        pc.write(out / 'method.freeze.json', method)
        pc.write(path, {'state': 'FROZEN', 'task': 'B4_graph', 'cohort': cohort, 'n': len(rows),
                       'workers': 1, 'threads': 4, 'arms': original['arms'],
                       'field_sources': original['field_sources'], 'source_sha256': source_sha,
                       'method': {'with_crf': original['method']['with_crf'], 'decision_basis': 'reciprocal_graph'},
                       'source_B4_root': str(root), 'source_B4_seal_sha256': pc.sha(root / 'sealed.json'),
                       'source_B4_fields': {r['index']: {'path': r['field_path'], 'sha256': r['field_sha256']} for r in source_seal['records']},
                       'query_GT_in_inference': False, 'new_encoder_calls': 0})
    config = pc.read(path)
    assert config['cohort'] == cohort and config['source_sha256'] == source_sha
    assert config['arms'] == original['arms'] and config['field_sources'] == original['field_sources']
    assert pc.read(out / 'method.freeze.json') == method
    assert pc.read(out / 'manifest.json') == rows
    return out, rows, config, source_seal, method


def produce(cohort):
    pc.setup('cpu')
    out, rows, config, source_seal, method = prepare(cohort)
    if (out / 'sealed.json').exists():
        print({'task': 'B4_graph', 'state': 'ALREADY_SEALED', 'cohort': cohort}, flush=True)
        return
    with_crf = config['method']['with_crf']
    host, adapter = pc.host_cpu() if with_crf else (None, None)
    if with_crf:
        from ics.official_data import load_inputs
        from utils.refinement import crf_refine
        def forbidden(*args, **kwargs):
            raise RuntimeError('Graph B4 forbids complete FoRIS and encoder inference')
        host.predict = host.segment = host._extract_features = forbidden
    source_records = {r['index']: r for r in source_seal['records']}
    began, records = time.monotonic(), []
    for i, binding in enumerate(rows, 1):
        old = workflow.checked_record(out, binding, config)
        if old is not None:
            records.append(old)
            continue
        start = time.monotonic()
        original = source_records[pc.key(binding)]
        try:
            with pc.inference_only(binding):
                assert pc.sha(original['field_path']) == original['field_sha256']
                assert pc.sha(original['prediction_path']) == original['prediction_sha256']
                neighbors = workflow.load_fields(binding, config, REQUIRED)['query_neighbors20']
                with np.load(original['field_path'], allow_pickle=False) as archive:
                    c = archive['independent_c'].astype(np.float32)
                    additions = {k: archive['addition_' + k].astype(np.float32) for k in b4.DEPENDENCY_ORDER}
                assert c.shape == (64, 64) and np.isfinite(c).all()
                assert all(v.shape == c.shape and np.isfinite(v).all() for v in additions.values())
                geometry_start = time.monotonic()
                geometry, cache_hit = shared_geometry(binding, neighbors)
                geometry_seconds = time.monotonic() - geometry_start
                with np.load(original['prediction_path'], allow_pickle=False) as archive:
                    diagnostics = ('INDN', 'T') + (('INDN_crf', 'T_crf') if with_crf else ())
                    predictions = {a: archive[a].copy() for a in diagnostics}
                saved, costs, graph_details = {'independent_c': c}, {}, {}
                if with_crf:
                    _, _, query = load_inputs(binding['row'], lab.ASSETS)
                    host.set_target(query)
                    image = host._tgt_image.clone().unsqueeze(0)
                crf_geometries = None
                for arm, active in b4.ACTIVE.items():
                    arm_start = time.monotonic()
                    field, detail = graph_from_geometry(compose_field(c, additions, active), geometry)
                    graph_seconds = time.monotonic() - arm_start
                    readout_start = time.monotonic()
                    predictions[arm] = pc.pack(pc.readout(field))
                    saved['field_' + arm] = field
                    source_cost = original['candidate_budget_and_costs'][arm]
                    costs[arm] = {**copy.deepcopy(method['candidates'][arm]), 'graph_seconds': graph_seconds,
                                  'readout_seconds': time.monotonic() - readout_start,
                                  'shared_query_graph_geometry_seconds': geometry_seconds,
                                  'source_dependency_stage_seconds': source_cost['dependency_stage_seconds'],
                                  'source_feature_preprocessing_seconds': source_cost['shared_feature_preprocessing_seconds'],
                                  'latency_scope': 'graph replay measured; sealed B4 dependency costs separately retained; no encoding'}
                    graph_details[arm] = detail
                    if with_crf:
                        crf_start = time.monotonic()
                        refined = crf_refine(host._crf, host._crf_band_px, host._crf_p_core, image,
                                             torch.from_numpy(pc.unpack(predictions[arm])))
                        predictions[arm + '_crf'] = pc.pack(refined.numpy())
                        current = (host._crf.spatial_filter._prepared_geometry, host._crf.bilateral_filter._prepared_geometry)
                        if crf_geometries is None:
                            crf_geometries = current
                        else:
                            assert all(x is y for x, y in zip(crf_geometries, current)), 'CRF geometry reuse changed'
                        costs[arm + '_crf'] = dict(costs[arm], common_CRF=True, CRF_seconds=time.monotonic() - crf_start)
                assert set(predictions) == set(config['arms'])
                assert adapter is None or adapter.encoder is None and adapter.encoder_calls == 0
            record = workflow.save_record(out, binding, config, predictions, saved, start,
                       {'decision_basis': 'reciprocal_graph', 'graph_geometry_cache_hit': cache_hit,
                        'graph': graph_details, 'candidate_budget_and_costs': costs,
                        'source_B4_field_sha256': original['field_sha256'],
                        'FoRIS_score_consumed_by_natural_inference': False, 'complete_FoRIS_pipeline_calls': 0,
                        'complete_controls_consumed_by_natural_inference': False})
            records.append(record)
        finally:
            if host is not None:
                host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None
        if i <= 3 or i % 50 == 0:
            print({'task': 'B4_graph', 'cohort': cohort, 'completed': i, 'n': len(rows),
                   'seconds': round(time.monotonic() - began, 1)}, flush=True)
    workflow.seal_run(out, rows, records, began)


def score(cohort):
    pc.setup('cpu')
    out, rows, config, source_seal, method = prepare(cohort)
    seal = pc.read(out / 'sealed.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED'
    assert pc.sha(out / 'config.json') == seal['config_sha256']
    assert pc.sha(out / 'manifest.json') == seal['manifest_sha256']
    original_root = Path(config['source_B4_root'])
    control_seal = pc.read(original_root / 'controls/sealed.json')
    assert control_seal['state'] == 'ALL_COMPLETE_CONTROLS_SEALED'
    assert control_seal['natural_seal_sha256'] == config['source_B4_seal_sha256']
    assert control_seal['method_freeze_sha256'] == pc.sha(original_root / 'method.freeze.json')
    predictions_by_key = {r['index']: r for r in seal['records']}
    controls_by_key = {r['index']: r for r in control_seal['records']}
    assert len(rows) == len(predictions_by_key) == len(controls_by_key) == control_seal['n']
    # Validate every natural/control artifact before the first scoring GT read.
    for binding in rows:
        key = pc.key(binding)
        assert workflow.checked_record(out, binding, config) == predictions_by_key[key]
        control = controls_by_key[key]
        assert pc.sha(control['path']) == control['sha256']
    records = []
    arms = ('FoRIS', 'MEAN') + tuple(config['arms'])
    for binding in rows:
        key = pc.key(binding)
        with np.load(predictions_by_key[key]['prediction_path'], allow_pickle=False) as archive:
            predictions = {a: pc.unpack(archive[a]) for a in config['arms']}
        with np.load(controls_by_key[key]['path'], allow_pickle=False) as archive:
            predictions.update({a: pc.unpack(archive[a]) for a in ('FoRIS', 'MEAN')})
        records.append(pr.score_record(binding, predictions))
    costs = {}
    eligible_arms = tuple(b4.ACTIVE) + (tuple(a + '_crf' for a in b4.ACTIVE) if config['method']['with_crf'] else ())
    for arm in eligible_arms:
        values = [r['candidate_budget_and_costs'][arm] for r in seal['records']]
        elapsed = [v['graph_seconds'] + v['readout_seconds'] + v.get('CRF_seconds', 0.) for v in values]
        costs[arm] = {**method['candidates'][arm.removesuffix('_crf')],
                      'mean_new_graph_readout_seconds': float(np.mean(elapsed)),
                      'median_new_graph_readout_seconds': float(np.median(elapsed)),
                      'common_CRF': arm.endswith('_crf'), 'latency_scope': values[0]['latency_scope']}
    task = 'B4_graph' if cohort == 'explore' else 'B4_graph_seed150'
    result = pr.report(task, out, records, arms, [
        '原B4固定12个合法arm的系数、依赖闭包与≤2步预算完全保留；只替换K64区域均值为A7 reciprocal graph。',
        '图输入为独立c+sum(active additions)，每个arm先minmax并构造其自身confidence A；P0互惠20NN，lambda16，原CG常数与IND读出。',
        '零步IND0在本报告表示A7 independent_graph(c)自然基线；INDN/T仍是原B4封存诊断掩码，不参与候选选择。',
        '每例q及W/degree只构建一次并复用；confidence A随arm变化。原CRF开关/参数保持不变且不计两步，原INDN/T的CRF诊断掩码逐位复用。',
        '自然推理仅消费已封存独立c与B/V/P/K additions及查询缓存特征；原B4完整FoRIS/MEAN controls仅评分读取。全部新掩码先封存再读取query GT。',
        '报告是十域固定开发候选，decision_basis=reciprocal_graph供C1冻结识别；未访问confirm、不做域/例条件选择或参数扩展。',
    ], extra={'method': method, 'cohort': cohort, 'candidate_budgets_and_costs': costs,
              'complete_controls': ['FoRIS', 'MEAN'], 'source_B4_control_seal_sha256': pc.sha(original_root / 'controls/sealed.json')})
    eligible = {a: {d: s[a]['delta_to_foris_pp'] for d, s in result['summary'].items()} for a in eligible_arms}
    passing = [a for a, values in eligible.items() if len(values) == 10 and all(v >= -.5 for v in values.values())]
    pc.write(out / 'completion.json', {'state': 'DEVELOPMENT_CANDIDATES_REPORTED', 'n': len(rows), 'cohort': cohort,
              'decision_basis': 'reciprocal_graph', 'ten_domain_within_minus_05_candidates': passing,
              'candidate_delta_pp': eligible, 'new_encoder_calls': 0, 'confirm_scored': False,
              'per_case_or_domain_component_selection': False, 'method_freeze_sha256': pc.sha(out / 'method.freeze.json')})


def self_check():
    """Synthetic formula equivalence only, without any project data or GT."""
    pc.setup('cpu')
    rng = np.random.default_rng(0)
    q = torch.from_numpy(rng.normal(size=(4096, 8)).astype(np.float32))
    offsets = np.r_[np.arange(1, 11), -np.arange(1, 11)]
    neighbors = ((np.arange(4096)[:, None] + offsets[None]) % 4096).astype(np.uint16)
    geometry = reciprocal_geometry(q, neighbors)
    c = rng.uniform(size=(64, 64)).astype(np.float32)
    additions = {k: rng.normal(0, .1, (64, 64)).astype(np.float32) for k in b4.DEPENDENCY_ORDER}
    for active in ((), ('B', 'V')):
        field = compose_field(c, additions, active)
        expected, unused = a7.independent_graph(field, q, neighbors)
        actual, info = graph_from_geometry(field, geometry)
        assert np.array_equal(actual, expected)
        assert np.array_equal(pc.pack(pc.readout(actual)), pc.pack(pc.readout(expected)))
        assert info['lambda'] == 16 and not info['rank_correction']
    assert len(b4.ACTIVE) == 12 and all(b4.budget(a)['FoRIS_stage_count'] <= 2 for a in b4.ACTIVE.values())
    assert b4.budget(b4.ACTIVE['P_only'])['computed_dependencies'] == ['V', 'P']
    print({'state': 'B4_GRAPH_SELF_CHECK_PASS', 'A7_field_and_mask_bit_exact': True,
           'fixed_candidate_n': 12, 'project_data_read': False, 'query_GT_read': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--cohort', choices=('seed150', 'explore'), default='seed150')
    args = parser.parse_args()
    produce(args.cohort) if args.mode == 'produce' else score(args.cohort)
