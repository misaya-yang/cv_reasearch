"""A4 max-J evidence modulation plus the original twelve legal B4 arms.

Natural inputs are sealed A4/P0 fields and original B4 addition fields only.
The complete FoRIS/MEAN controls are opened exclusively by score after seal.
No new parameter search, encoder, reference/view, or confirmation data.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import torch
import lab
import plan_common as pc
import plan_report as pr
import a2_exchange as workflow
import a4_modulation as a4
import b4_minimal as b4
import b4_graph_basis as graph

A4_ROOT = lab.ASSETS / 'a/x1012_A4/modulation'
REQUIRED = ('c32', 'sim_max', 'regions64')
DECISION_BASIS = {'region': 'K64_region_mean', 'graph': 'reciprocal_graph'}


def modulated_evidence(c32, sim_max, selected_s):
    """Exact A4 FP16-novelty continuous field, including 0**0 at s=0."""
    c = np.asarray(c32, np.float32).reshape(-1)
    maximum = np.asarray(sim_max, np.float16).astype(np.float32).reshape(-1)
    novelty = (maximum - maximum.min()) / max(float(maximum.max() - maximum.min()), 1e-6)
    return (c * np.power(novelty, selected_s)).reshape(64, 64).astype(np.float32)


def prepare(basis, cohort):
    original_root, rows, original, source_seal, source_method = graph.source_run(cohort)
    a4_seal, a4_config = pc.read(A4_ROOT / 'sealed.json'), pc.read(A4_ROOT / 'config.json')
    assert a4_seal['state'] == 'ALL_J_MODULATION_PREDICTIONS_SEALED'
    assert pc.sha(A4_ROOT / 'config.json') == a4_seal['config_sha256']
    assert a4_config['s_grid'] == list(a4.STRENGTHS)
    a4_rows = pc.read(A4_ROOT / 'manifest.json')
    assert len(rows) == len(a4_rows) == len(a4_seal['records']) == a4_seal['n']
    assert all(workflow.same_inputs(before, after) for before, after in zip(rows, a4_rows))
    a4_records = {r['index']: r for r in a4_seal['records']}
    assert set(a4_records) == {pc.key(b) for b in rows}
    sources = dict(original['source_sha256'])
    for path in (Path(__file__), Path(a4.__file__), Path(graph.__file__), Path(pr.__file__),
                 original_root / 'sealed.json', original_root / 'method.freeze.json',
                 original_root / 'config.json', original_root / 'manifest.json',
                 A4_ROOT / 'sealed.json', A4_ROOT / 'config.json', A4_ROOT / 'manifest.json'):
        sources[str(path)] = pc.sha(path)
    arms = tuple(b4.ACTIVE)
    with_crf = original['method']['with_crf']
    if with_crf:
        arms += tuple(a + '_crf' for a in b4.ACTIVE)
    required = REQUIRED + (('query_neighbors20',) if basis == 'graph' else ())
    method = copy.deepcopy(source_method)
    method.update(
        decision_basis=DECISION_BASIS[basis], evidence_modulation='A4_max_J_continuous',
        main_evidence='c_mod = P0 c32 * minmax(P0 FP16 sim_max cast to FP32)**A4 sealed selected_s',
        readout=('A4 selected_field32 + same K64 region_mean(sum(active additions)); bilinear1024 -> minmax -> >.5'
                 if basis == 'region' else
                 'original A7 reciprocal graph(c_mod + sum(active additions)); bilinear1024 -> minmax -> >.5'),
        zero_step=('byte-exact A4 sealed J_continuous mask and selected_field32; original A4 endpoint overrides retained'
                   if basis == 'region' else 'same original reciprocal graph applied to c_mod'),
        evidence_selection={'s_grid': list(a4.STRENGTHS), 'selection': 'GT-free A4 quality J maximum; lower s wins ties',
                            'endpoint_J_masks': 's=0 uses FP32 original IND0 mask; s=1 uses FP32/raw-novelty original INDN mask',
                            'continuous_field': 'FP16 sim_max novelty for every s, including endpoints',
                            'query_GT_used': False, 'source_A4_root': str(A4_ROOT),
                            'source_A4_seal_sha256': pc.sha(A4_ROOT / 'sealed.json')},
        K_semantics={'prob': 'sf=original c32, sbn=1-original c32',
                     'minmax': 'sf=minmax(original c32), sbn=1-minmax(original c32)',
                     'note': 'Original sealed K additions are unchanged independent side information; neither c_mod nor B/V/P updates their adapter.'},
        controls='Only score reads original B4 separately sealed complete FoRIS/MEAN; never natural inputs',
        per_case_evidence_modulation=True, per_case_component_selection=False, dataset_branch=False,
        source_sha256=sources, source_B4_method_path=str(original_root / 'method.freeze.json'),
        source_B4_method_sha256=pc.sha(original_root / 'method.freeze.json'), C1_compatible=True)
    if basis == 'graph':
        method['graph'] = {'lambda': 16, 'neighbors': 20, 'minmax_before_graph': True,
                           'field_specific_confidence_A': True, 'rank_correction': False,
                           'CG': {'rtol': 1e-7, 'atol': 1e-9, 'maxiter': 300}}
    out = lab.ASSETS / 'a/x1012_B4_modulated' / basis / cohort
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'config.json'
    source_records = {r['index']: r for r in source_seal['records']}
    if not path.exists():
        for name in ('predictions', 'fields', 'records', 'frozen'):
            (out / name).mkdir(exist_ok=True)
        for code in (Path(__file__), Path(a4.__file__), Path(graph.__file__), Path(b4.__file__),
                     Path(workflow.__file__), Path(pc.__file__), Path(pr.__file__), Path(lab.__file__)):
            shutil.copyfile(code, out / 'frozen' / code.name)
        pc.write(out / 'manifest.json', rows)
        pc.write(out / 'method.freeze.json', method)
        pc.write(path, {'state': 'FROZEN', 'task': 'B4_modulated', 'basis': basis, 'cohort': cohort,
                 'n': len(rows), 'workers': 1, 'threads': 4, 'arms': list(arms),
                 'field_sources': original['field_sources'], 'source_sha256': sources,
                 'required_fields': list(required), 'method': {'with_crf': with_crf,
                 'decision_basis': DECISION_BASIS[basis], 'evidence_modulation': 'A4_max_J_continuous'},
                 'source_B4_root': str(original_root), 'source_B4_seal_sha256': pc.sha(original_root / 'sealed.json'),
                 'source_B4_fields': {k: {'path': r['field_path'], 'sha256': r['field_sha256'],
                                         'input_field_sha256': r['input_field_sha256']} for k, r in source_records.items()},
                 'source_A4_records': a4_records, 'C1_compatible': True,
                 'query_GT_in_inference': False, 'new_encoder_calls': 0})
    config = pc.read(path)
    assert config['basis'] == basis and config['cohort'] == cohort and config['arms'] == list(arms)
    assert config['source_sha256'] == sources and config['field_sources'] == original['field_sources']
    assert pc.read(out / 'method.freeze.json') == method and pc.read(out / 'manifest.json') == rows
    return out, rows, config, source_seal, method


def produce(basis, cohort):
    pc.setup('cpu')
    out, rows, config, source_seal, method = prepare(basis, cohort)
    if (out / 'sealed.json').exists():
        print({'task': 'B4_modulated', 'basis': basis, 'state': 'ALREADY_SEALED', 'cohort': cohort}, flush=True)
        return
    with_crf = config['method']['with_crf']
    host, adapter = pc.host_cpu() if with_crf else (None, None)
    if with_crf:
        from ics.official_data import load_inputs
        from utils.refinement import crf_refine
        def forbidden(*args, **kwargs):
            raise RuntimeError('Modulated B4 forbids complete FoRIS/encoder prediction')
        host.predict = host.segment = host._extract_features = forbidden
    began, records = time.monotonic(), []
    original_records = {r['index']: r for r in source_seal['records']}
    for i, binding in enumerate(rows, 1):
        old = workflow.checked_record(out, binding, config)
        if old is not None:
            records.append(old)
            continue
        start = time.monotonic()
        key = pc.key(binding)
        original = original_records[key]
        selected = config['source_A4_records'][key]
        try:
            with pc.inference_only(binding):
                fields = workflow.load_fields(binding, config, tuple(config['required_fields']))
                assert selected['field_source_sha256'] == original['input_field_sha256'] == config['field_sources'][key]['sha256']
                assert pc.sha(original['field_path']) == original['field_sha256']
                assert pc.sha(selected['path']) == selected['sha256']
                selected_s = float(selected['selected_s'])
                assert selected_s in a4.STRENGTHS
                with np.load(selected['path'], allow_pickle=False) as archive:
                    selected_field = archive['selected_field32'].astype(np.float32).reshape(64, 64)
                    zero_mask = archive['J_continuous'].copy()
                with np.load(original['field_path'], allow_pickle=False) as archive:
                    additions = {k: archive['addition_' + k].astype(np.float32) for k in b4.DEPENDENCY_ORDER}
                c = fields['c32'].reshape(64, 64).astype(np.float32)
                modulation_start = time.monotonic()
                c_mod = modulated_evidence(c, fields['sim_max'], selected_s)
                modulation_seconds = time.monotonic() - modulation_start
                assert np.isfinite(c_mod).all() and np.isfinite(selected_field).all()
                geometry, geometry_seconds, cache_hit = None, 0., False
                if basis == 'graph':
                    geometry_start = time.monotonic()
                    geometry, cache_hit = graph.shared_geometry(binding, fields['query_neighbors20'])
                    geometry_seconds = time.monotonic() - geometry_start
                predictions, graph_details, costs = {}, {}, {}
                saved = {'c32_original': c, 'c_mod': c_mod, 'selected_field32': selected_field,
                         'selected_s': np.array(selected_s, np.float32)}
                if with_crf:
                    _, _, query = load_inputs(binding['row'], lab.ASSETS)
                    host.set_target(query)
                    image = host._tgt_image.clone().unsqueeze(0)
                for arm, active in b4.ACTIVE.items():
                    arm_start = time.monotonic()
                    if basis == 'graph':
                        field, detail = graph.graph_from_geometry(graph.compose_field(c_mod, additions, active), geometry)
                        detail['confidence'] = 's=minmax(c_mod+sum(active additions)); original field-specific A and lambda16'
                        graph_details[arm] = detail
                    elif not active:
                        field = selected_field.copy()
                    else:
                        correction = sum((torch.from_numpy(additions[k]) for k in active), torch.zeros((64, 64), dtype=torch.float32))
                        field = selected_field + workflow.region_mean(correction.numpy(), fields['regions64'])
                    decision_seconds = time.monotonic() - arm_start
                    readout_start = time.monotonic()
                    predictions[arm] = zero_mask.copy() if basis == 'region' and not active else pc.pack(pc.readout(field))
                    saved['field_' + arm] = field
                    inherited = original['candidate_budget_and_costs'][arm]
                    costs[arm] = {**copy.deepcopy(method['candidates'][arm]),
                                  'decision_seconds': decision_seconds, 'readout_seconds': time.monotonic() - readout_start,
                                  'shared_modulation_seconds': modulation_seconds,
                                  'shared_query_graph_geometry_seconds': geometry_seconds,
                                  'source_dependency_stage_seconds': inherited['dependency_stage_seconds'],
                                  'source_feature_preprocessing_seconds': inherited['shared_feature_preprocessing_seconds'],
                                  'latency_scope': 'modulation/basis/readout replay measured; original sealed B4 dependency costs retained separately; A4 maxJ selection/cache encoding excluded'}
                    if with_crf:
                        crf_start = time.monotonic()
                        refined = crf_refine(host._crf, host._crf_band_px, host._crf_p_core, image,
                                             torch.from_numpy(pc.unpack(predictions[arm])))
                        predictions[arm + '_crf'] = pc.pack(refined.cpu().numpy())
                        costs[arm + '_crf'] = dict(costs[arm], common_CRF=True, CRF_seconds=time.monotonic() - crf_start)
                assert set(predictions) == set(config['arms'])
                assert adapter is None or adapter.encoder is None and adapter.encoder_calls == 0
            record = workflow.save_record(out, binding, config, predictions, saved, start,
                     {'decision_basis': DECISION_BASIS[basis], 'evidence_modulation': 'A4_max_J_continuous',
                      'selected_s': selected_s, 'source_A4_prediction_sha256': selected['sha256'],
                      'source_B4_field_sha256': original['field_sha256'],
                      'region_zero_step_A4_mask_reused': basis == 'region',
                      'K_adapter_uses_original_c': True, 'graph_geometry_cache_hit': cache_hit,
                      'graph': graph_details, 'candidate_budget_and_costs': costs,
                      'complete_controls_consumed_by_natural_inference': False,
                      'FoRIS_score_consumed_by_natural_inference': False, 'complete_FoRIS_pipeline_calls': 0})
            records.append(record)
        finally:
            if host is not None:
                host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None
        if i <= 3 or i % 50 == 0:
            print({'task': 'B4_modulated', 'basis': basis, 'cohort': cohort, 'completed': i,
                   'n': len(rows), 'seconds': round(time.monotonic() - began, 1)}, flush=True)
    workflow.seal_run(out, rows, records, began)


def score(basis, cohort):
    pc.setup('cpu')
    out, rows, config, unused, method = prepare(basis, cohort)
    seal = pc.read(out / 'sealed.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED'
    assert pc.sha(out / 'manifest.json') == seal['manifest_sha256'] and pc.sha(out / 'config.json') == seal['config_sha256']
    original_root = Path(config['source_B4_root'])
    control_seal = pc.read(original_root / 'controls/sealed.json')
    assert control_seal['state'] == 'ALL_COMPLETE_CONTROLS_SEALED'
    assert control_seal['natural_seal_sha256'] == config['source_B4_seal_sha256']
    assert control_seal['method_freeze_sha256'] == pc.sha(original_root / 'method.freeze.json')
    produced = {r['index']: r for r in seal['records']}
    controls = {r['index']: r for r in control_seal['records']}
    assert len(rows) == len(produced) == len(controls) == control_seal['n']
    # Complete validation of natural and benchmark artifacts precedes the first GT read.
    for binding in rows:
        key = pc.key(binding)
        assert workflow.checked_record(out, binding, config) == produced[key]
        assert pc.sha(controls[key]['path']) == controls[key]['sha256']
    records = []
    arms = ('FoRIS', 'MEAN') + tuple(config['arms'])
    for binding in rows:
        key = pc.key(binding)
        with np.load(produced[key]['prediction_path'], allow_pickle=False) as archive:
            predictions = {a: pc.unpack(archive[a]) for a in config['arms']}
        with np.load(controls[key]['path'], allow_pickle=False) as archive:
            predictions.update({a: pc.unpack(archive[a]) for a in ('FoRIS', 'MEAN')})
        records.append(pr.score_record(binding, predictions))
    costs = {}
    for arm in config['arms']:
        values = [r['candidate_budget_and_costs'][arm] for r in seal['records']]
        elapsed = [v['decision_seconds'] + v['readout_seconds'] + v.get('CRF_seconds', 0.) for v in values]
        costs[arm] = {**method['candidates'][arm.removesuffix('_crf')],
                      'mean_measured_basis_readout_seconds': float(np.mean(elapsed)),
                      'median_measured_basis_readout_seconds': float(np.median(elapsed)),
                      'mean_shared_modulation_seconds': float(np.mean([v['shared_modulation_seconds'] for v in values])),
                      'mean_shared_graph_geometry_seconds': float(np.mean([v['shared_query_graph_geometry_seconds'] for v in values])),
                      'source_mean_dependency_stage_seconds': float(np.mean([v['source_dependency_stage_seconds'] for v in values])),
                      'source_mean_feature_preprocessing_seconds': float(np.mean([v['source_feature_preprocessing_seconds'] for v in values])),
                      'latency_scope': values[0]['latency_scope']}
    label = f'B4_modulated_{basis}_{cohort}'
    result = pr.report(label, out, records, arms, [
        'A4已seal的GT-free maxJ九点selected_s调制主证据：c32*minmax(FP16 sim_max)**s；没有新增参数搜索或GT选择。',
        '原12个合法B4臂及系数/依赖预算不变；P_only仍计算V+P两步。K_prob/K_minmax继续使用原始c的独立侧信息，未改成c_mod。',
        'region零步逐位复用A4 J_continuous mask/selected_field32，保留A4端点掩码规则；其他region仅在该field加同K64区域均值的原additions。',
        'graph使用原reciprocal_geometry/graph_from_geometry，输入c_mod+同原additions；lambda16、20邻居和CG参数保持原值。',
        'FoRIS/MEAN完整强控仅score阶段读取原B4 controls，未进入自然输入。全部自然掩码先seal，再读取query GT。',
        '这是同1360例/十域开发组件组装；C1按相同maxJ重算selected_s。A6上界门槛未过的监督/LODO未恢复。',
    ], extra={'method': method, 'cohort': cohort, 'candidate_budgets_and_costs': costs,
              'complete_controls': ['FoRIS', 'MEAN'], 'C1_compatible': True,
              'source_B4_control_seal_sha256': pc.sha(original_root / 'controls/sealed.json')})
    eligible = {a: {d: table[a]['delta_to_foris_pp'] for d, table in result['summary'].items()} for a in config['arms']}
    passing = [a for a, values in eligible.items() if len(values) == 10 and all(v >= -.5 for v in values.values())]
    pc.write(out / 'completion.json', {'state': 'DEVELOPMENT_CANDIDATES_REPORTED', 'n': len(rows), 'cohort': cohort,
             'decision_basis': DECISION_BASIS[basis], 'evidence_modulation': 'A4_max_J_continuous',
             'ten_domain_within_minus_05_candidates': passing, 'candidate_delta_pp': eligible,
             'C1_compatible': True, 'new_encoder_calls': 0, 'confirm_scored': False,
             'per_case_or_domain_component_selection': False,
             'method_freeze_sha256': pc.sha(out / 'method.freeze.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--basis', choices=('region', 'graph'), required=True)
    parser.add_argument('--cohort', choices=('seed150',), default='seed150')
    args = parser.parse_args()
    {'produce': produce, 'score': score}[args.mode](args.basis, args.cohort)
