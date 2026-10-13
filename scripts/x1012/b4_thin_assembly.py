"""One root-selected B4 arm, with the plan's fixed rho>.9 raw-T fallback.

produce reads only sealed natural predictions and P0 reference statistics/T.
score reads query GT only after all assembly masks and original B4 controls have
been validated. No arm grid, new threshold, encoding, or confirm access.
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
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np
import lab
import plan_common as pc
import plan_report as pr
import a2_exchange as workflow

THRESHOLD = .9
REQUIRED = ('ref_stats', 'mask/T')
SOURCE_TASKS = ('B4_seed150', 'B4_graph_seed150', 'B4_modulated_region_seed150', 'B4_modulated_graph_seed150')


def candidate_definitions(source_method, arm):
    """Self-contained budgets for C1; no b4.ACTIVE name lookup is needed."""
    base = arm.removesuffix('_crf')
    source = source_method['candidates'].get(arm, source_method['candidates'].get(base))
    if source is None:
        raise ValueError('Source report does not define the specified candidate: ' + arm)
    budget = copy.deepcopy(source)
    assert 0 <= budget['FoRIS_stage_count'] <= 2
    budget.update(source_arm=arm, base_arm=base,
                  common_crf=source.get('common_crf', arm.endswith('_crf')))
    gated = copy.deepcopy(budget)
    gated.update(source_arm=arm + '_thin09', underlying_source_arm=arm,
                 reference_rho_gate=THRESHOLD, reference_rho_fallback='raw_T',
                 triggered_FoRIS_stage_count=0, triggered_common_crf=False)
    return {arm: budget, arm + '_thin09': gated}


def assemble(source_bits, raw_t_bits, rho):
    """Strict >.9, including the unmodified raw-T mask on the triggered branch."""
    assert np.isfinite(rho) and 0 <= rho <= 1
    triggered = rho > THRESHOLD
    return (raw_t_bits if triggered else source_bits).copy(), triggered


def prepare(source_report, arm, cohort):
    pc.setup('cpu')
    report_path = Path(source_report).resolve()
    report = pc.read(report_path)
    assert report['state'] == 'COMPLETE' and report['task'] in SOURCE_TASKS
    assert report['extra'].get('cohort') == cohort == 'seed150'
    assert arm in report['arms'], 'The root-specified arm must already have been scored'
    assert set(report['summary']) == set(lab.ALL_DOMAINS), 'Source report must contain all ten domains'
    source_out = report_path.parent
    source_seal = pc.read(source_out / 'sealed.json')
    source_config = pc.read(source_out / 'config.json')
    source_method = report['extra']['method']
    rows = pc.read(source_out / 'manifest.json')
    assert source_seal['state'] == 'ALL_PREDICTIONS_SEALED'
    assert pc.sha(source_out / 'config.json') == source_seal['config_sha256']
    assert pc.sha(source_out / 'manifest.json') == source_seal['manifest_sha256']
    assert len(rows) == source_seal['n'] == report['n']
    assert not any(b.get('split') == 'confirm' for b in rows)
    assert [r['index'] for r in source_seal['records']] == [pc.key(b) for b in rows]
    assert pc.sha(lab.BASIS) == source_method['basis_sha256']
    for path, expected in source_method['source_sha256'].items():
        assert pc.sha(path) == expected, 'Source method changed: ' + path
    candidates = candidate_definitions(source_method, arm)
    sha = dict(source_method['source_sha256'])
    for path in (Path(__file__), Path(pc.__file__), Path(pr.__file__), Path(workflow.__file__),
                 Path(lab.__file__), report_path, source_out / 'config.json', source_out / 'sealed.json',
                 source_out / 'manifest.json', source_out / 'method.freeze.json'):
        sha[str(path)] = pc.sha(path)
    method = copy.deepcopy(source_method)
    method.update(state='METHOD_FROZEN_BEFORE_ASSEMBLY_GT', candidates=candidates,
                  decision_basis=source_method.get('decision_basis', 'region_mean'),
                  evidence_modulation=source_method.get('evidence_modulation'), source_sha256=sha,
                  assembly='one root-selected source arm vs same arm with fixed reference-only raw-T fallback',
                  root_selected_source_arm=arm, source_report_path=str(report_path),
                  source_report_sha256=pc.sha(report_path),
                  reference_rho_definition='P0 FP32 ref_stats[2]: FG area mass on reference coverage<.9 tokens / total FG mass',
                  gated_readout='T from original P0 mask/T, no evidence modulation, additions, graph, or CRF when triggered',
                  dataset_branch=False, per_case_reference_only_fallback=True,
                  C1_candidate_schema='per-arm active_score_terms/computed_dependencies/common_crf; gate field exists only on thin09')
    out = lab.ASSETS / 'a/x1012_B4_assembly' / cohort
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    if not config_path.exists():
        for name in ('predictions', 'fields', 'records', 'frozen'):
            (out / name).mkdir(exist_ok=True)
        for path in (Path(__file__), Path(pc.__file__), Path(pr.__file__), Path(workflow.__file__), Path(lab.__file__)):
            shutil.copyfile(path, out / 'frozen' / path.name)
        pc.write(out / 'manifest.json', rows)
        pc.write(out / 'method.freeze.json', method)
        pc.write(config_path, {'state': 'FROZEN', 'task': 'B4_assembly', 'cohort': cohort, 'n': len(rows),
                   'arms': list(candidates), 'source_arm': arm, 'source_report_path': str(report_path),
                   'source_report_sha256': pc.sha(report_path), 'source_out': str(source_out),
                   'original_B4_root': source_config.get('source_B4_root', str(source_out)),
                   'source_natural_seal_sha256': pc.sha(source_out / 'sealed.json'),
                   'field_sources': source_config['field_sources'], 'source_sha256': sha,
                   'workers': 1, 'threads': 4, 'query_GT_in_inference': False, 'new_encoder_calls': 0})
    config = pc.read(config_path)
    assert config['source_report_path'] == str(report_path) and config['source_arm'] == arm
    assert config['source_report_sha256'] == pc.sha(report_path)
    assert config['source_sha256'] == sha and config['field_sources'] == source_config['field_sources']
    assert pc.read(out / 'method.freeze.json') == method and pc.read(out / 'manifest.json') == rows
    return out, rows, config, source_seal, method


def produce(source_report, arm, cohort):
    out, rows, config, source_seal, method = prepare(source_report, arm, cohort)
    if (out / 'sealed.json').exists():
        print({'task': 'B4_assembly', 'state': 'ALREADY_SEALED', 'source_arm': arm}, flush=True)
        return
    indexed = {r['index']: r for r in source_seal['records']}
    began, records = time.monotonic(), []
    for i, binding in enumerate(rows, 1):
        previous = workflow.checked_record(out, binding, config)
        if previous is not None:
            records.append(previous)
            continue
        start = time.monotonic()
        original = indexed[pc.key(binding)]
        with pc.inference_only(binding):
            assert pc.sha(original['prediction_path']) == original['prediction_sha256']
            fields = workflow.load_fields(binding, config, REQUIRED)
            stats = np.asarray(fields['ref_stats'])
            assert stats.shape == (3,) and stats.dtype == np.float32
            rho = float(stats[2])
            with np.load(original['prediction_path'], allow_pickle=False) as archive:
                source_bits = archive[arm].copy()
            thin_bits, triggered = assemble(source_bits, fields['mask/T'], rho)
            predictions = {arm: source_bits, arm + '_thin09': thin_bits}
        record = workflow.save_record(out, binding, config, predictions,
                    {'reference_rho': np.array([rho], np.float32)}, start,
                    {'source_prediction_sha256': original['prediction_sha256'], 'source_arm': arm,
                     'rho': rho, 'thin09_triggered': triggered,
                     'candidate_declared_maximum_stage_count': {a: d['FoRIS_stage_count'] for a, d in method['candidates'].items()},
                     'actual_FoRIS_stage_count': {arm: method['candidates'][arm]['FoRIS_stage_count'],
                                      arm + '_thin09': 0 if triggered else method['candidates'][arm]['FoRIS_stage_count']},
                     'actual_common_crf': {arm: method['candidates'][arm]['common_crf'],
                                      arm + '_thin09': False if triggered else method['candidates'][arm]['common_crf']},
                     'FoRIS_score_consumed_by_natural_inference': False,
                     'complete_controls_consumed_by_natural_inference': False,
                     'latency_scope': 'replay sealed source prediction and reference-only gate; source model cost not remeasured'})
        records.append(record)
        if i <= 3 or i % 100 == 0:
            print({'task': 'B4_assembly', 'completed': i, 'n': len(rows),
                   'seconds': round(time.monotonic() - began, 1)}, flush=True)
    workflow.seal_run(out, rows, records, began)


def score(source_report, arm, cohort):
    out, rows, config, source_seal, method = prepare(source_report, arm, cohort)
    seal = pc.read(out / 'sealed.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED'
    assert pc.sha(out / 'config.json') == seal['config_sha256']
    assert pc.sha(out / 'manifest.json') == seal['manifest_sha256']
    control_root = Path(config['original_B4_root'])
    controls = pc.read(control_root / 'controls/sealed.json')
    assert controls['state'] == 'ALL_COMPLETE_CONTROLS_SEALED'
    assert controls['natural_seal_sha256'] == pc.sha(control_root / 'sealed.json')
    assert controls['method_freeze_sha256'] == pc.sha(control_root / 'method.freeze.json')
    indexed = {r['index']: r for r in seal['records']}
    benchmarks = {r['index']: r for r in controls['records']}
    assert len(rows) == len(indexed) == len(benchmarks) == controls['n']
    for binding in rows:
        key = pc.key(binding)
        assert workflow.checked_record(out, binding, config) == indexed[key]
        assert pc.sha(benchmarks[key]['path']) == benchmarks[key]['sha256']
    arms = ('FoRIS', 'MEAN') + tuple(config['arms'])
    records = []
    for binding in rows:
        key = pc.key(binding)
        with np.load(indexed[key]['prediction_path'], allow_pickle=False) as archive:
            predictions = {a: pc.unpack(archive[a]) for a in config['arms']}
        with np.load(benchmarks[key]['path'], allow_pickle=False) as archive:
            predictions.update({a: pc.unpack(archive[a]) for a in ('FoRIS', 'MEAN')})
        record = pr.score_record(binding, predictions)
        record.update(rho=indexed[key]['rho'], thin09_triggered=indexed[key]['thin09_triggered'],
                      actual_FoRIS_stage_count=indexed[key]['actual_FoRIS_stage_count'])
        records.append(record)
    triggered = {d: {'n': sum(r['dataset'] == d for r in records),
                     'triggered_n': sum(r['dataset'] == d and r['thin09_triggered'] for r in records)} for d in lab.ALL_DOMAINS}
    result = pr.report('B4_assembly_seed150', out, records, arms, [
        'root仅指定一个已评分源arm；只比较原预测与同一arm附加计划原文固定rho>.9转原始T的fallback，没有扩大候选网格或阈值。',
        'rho仅来自P0 FP32参考ref_stats[2]，触发规则十域完全相同。触发时逐位复用P0 mask/T，不做源证据调制、FoRIS additions、图或CRF。',
        '未触发时逐位复用源arm，源decision_basis/evidence_modulation及系数/依赖预算保持不变；最大≤2步，触发分支0步。',
        '仅thin09候选元数据定义reference_rho_gate=.9，原候选没有该字段；C1按每候选自含active/deps/门控定义冻结，不依赖候选名猜测。',
        '全部组装自然预测先封存后读query GT；完整FoRIS/MEAN controls独立复用原B4封存产物，仅作评分。没有新编码或confirm访问。',
    ], extra={'method': method, 'cohort': cohort, 'complete_controls': ['FoRIS', 'MEAN'],
              'reference_only_trigger_counts': triggered, 'source_report_path': config['source_report_path'],
              'source_report_sha256': config['source_report_sha256'],
              'original_B4_control_seal_sha256': pc.sha(control_root / 'controls/sealed.json')})
    deltas = {a: {d: table[a]['delta_to_foris_pp'] for d, table in result['summary'].items()} for a in config['arms']}
    pc.write(out / 'completion.json', {'state': 'DEVELOPMENT_ASSEMBLY_REPORTED', 'source_arm': arm,
                   'n': len(rows), 'cohort': cohort, 'candidate_delta_pp': deltas,
                   'ten_domain_within_minus_05_candidates': [a for a, v in deltas.items() if len(v) == 10 and all(x >= -.5 for x in v.values())],
                   'reference_only_trigger_counts': triggered, 'new_encoder_calls': 0, 'confirm_scored': False,
                   'method_freeze_sha256': pc.sha(out / 'method.freeze.json')})


def self_check():
    """Boundary and branch-preservation fixtures, no data/GT/model access."""
    source = np.array([0, 255, 15], np.uint8)
    raw_t = np.array([255, 0, 240], np.uint8)
    assert np.array_equal(assemble(source, raw_t, .9)[0], source)
    assert not assemble(source, raw_t, .9)[1]
    assert np.array_equal(assemble(source, raw_t, .900001)[0], raw_t)
    assert assemble(source, raw_t, .900001)[1]
    method = {'candidates': {'VP': {'active_score_terms': ['V', 'P'], 'computed_dependencies': ['V', 'P'], 'FoRIS_stage_count': 2}}}
    candidates = candidate_definitions(method, 'VP')
    assert 'reference_rho_gate' not in candidates['VP']
    assert candidates['VP_thin09']['reference_rho_gate'] == .9
    assert candidates['VP_thin09']['triggered_FoRIS_stage_count'] == 0
    assert candidates['VP_thin09']['computed_dependencies'] == ['V', 'P']
    assert candidates['VP_thin09']['source_arm'] == 'VP_thin09'
    assert candidates['VP_thin09']['underlying_source_arm'] == 'VP'
    assert candidate_definitions(method, 'VP_crf')['VP_crf']['common_crf']
    print({'state': 'B4_THIN_ASSEMBLY_SELF_CHECK_PASS', 'strict_boundary': True,
           'raw_T_byte_preserved': True, 'source_budget_preserved': True, 'project_data_read': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--cohort', choices=('seed150',), default='seed150')
    args = parser.parse_args()
    globals()[args.mode](args.source_report, args.arm, args.cohort)
