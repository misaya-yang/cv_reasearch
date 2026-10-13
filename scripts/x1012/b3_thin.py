"""B3: full available-original explore cohort plus Fundus/Deep new explore pairs.

Only P0 FP32 independent fields/masks and reference statistics are consumed.
All fixed gates are produced/sealed before GT; one uniform gate is selected on
explore only. No confirm input, dataset branch, encoder, or new raw feature read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import lab
import plan_common as pc
import plan_report as pr
import b1_kshot as core

OUT = lab.ASSETS / 'a/x1012_B3'
P0 = lab.ASSETS / 'a/x1012_P0'
FIELDS = P0 / 'fields'
THRESHOLDS = (.5, .6, .7, .8, .9)
GATES = tuple(f'T_if_rho_gt_{v:.1f}' for v in THRESHOLDS)
STRICT = 'T_if_rho_gt_0.9_strict'
NATURAL_ARMS = ('IND0', 'T', *GATES, STRICT)
ARMS = ('FoRIS', *NATURAL_ARMS)


def source_field(binding):
    path = FIELDS / 'records' / (pc.key(binding) + '.json')
    if not path.is_file():
        raise FileNotFoundError('P0 field not yet available; resume B3 after P0: ' + str(path))
    receipt = pc.read(path)
    assert receipt['index'] == pc.key(binding) and receipt['episode_id'] == binding['row']['episode_id']
    assert pc.sha(receipt['path']) == receipt['sha256'], 'P0 field changed'
    return receipt


def infer(binding):
    with core.legal_inference(binding):
        source = source_field(binding)
        with np.load(source['path'], allow_pickle=False) as f:
            c, ind0 = f['c32'].copy(), f['ind0_32'].copy()
            stats = f['ref_stats'].copy()
            packed_t, packed_ind0 = f['mask/T'].copy(), f['mask/IND0'].copy()
        assert c.shape == ind0.shape == (4096,) and c.dtype == ind0.dtype == np.float32
        assert stats.shape == (3,) and np.isfinite(c).all() and np.isfinite(ind0).all() and np.isfinite(stats).all()
        # Saved P0 masks preserve its producer's interpolation arithmetic.
        # A CPU re-render must not introduce border-pixel changes to those controls.
        assert packed_t.dtype == packed_ind0.dtype == np.uint8
        assert packed_t.shape == packed_ind0.shape == (1024 * 1024 // 8,)
        rho, area = float(stats[2]), float(stats[0])
        assert 0 <= rho <= 1 and 0 <= area <= 1
        predictions = {'IND0': packed_ind0, 'T': packed_t}
        triggers = {}
        for threshold, arm in zip(THRESHOLDS, GATES):
            triggers[arm] = rho > threshold
            predictions[arm] = (packed_t if triggers[arm] else packed_ind0).copy()
        triggers[STRICT] = rho > .9
        predictions[STRICT] = (packed_t if triggers[STRICT] else packed_ind0).copy()
        assert np.array_equal(predictions[STRICT], predictions[GATES[-1]])
    return predictions, {'source_field_path': source['path'], 'source_field_sha256': source['sha256'],
                         'reference_area_fraction': area, 'rho': rho, 'triggers': triggers,
                         'new_encoder_calls': 0, 'query_GT_reads': 0}


def produce():
    pc.setup('cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / 'config.json'
    if not config_path.exists():
        explore = pc.read(P0 / 'explore.json')
        rows = [b for b in explore if b['cohort'] != 'x1012_P0_new_pair' or b['dataset'] in ('fundus', 'deepglobe_road')]
        assert rows and len({pc.key(b) for b in rows}) == len(rows)
        assert all(b.get('split') != 'confirm' for b in rows)
        pc.write(OUT / 'manifest.json', rows)
        pc.write(config_path, {'state': 'FROZEN', 'task': 'B3', 'n': len(rows), 'cpu_threads': 4,
                 'scope': 'all original tasks in P0 explore plus all Fundus/DeepGlobe new explore pairs; no stride subsample',
                 'per_domain_n': dict(Counter(b['dataset'] for b in rows)),
                 'new_pair_n': dict(Counter(b['dataset'] for b in rows if b['cohort'] == 'x1012_P0_new_pair')),
                 'thresholds': THRESHOLDS, 'strict_rule': 'rho > .9, exact alias of the .9 grid rule',
                 'rho': 'P0 FP32 ref_stats[2] = foreground area mass in coverage<.9 tokens / total foreground mass',
                 'natural_rule': 'T if rho > global threshold, else IND0; no dataset identity in gate',
                 'field_keys': ['c32', 'ind0_32', 'ref_stats', 'mask/T', 'mask/IND0'],
                 'explore_manifest_sha256_at_freeze': pc.sha(P0 / 'explore.json'),
                 'source_sha256': core.source_hashes(Path(__file__)),
                 'new_encoder_calls': 0, 'raw_feature_reads': 0, 'query_GT_reads': 0})
    config = pc.read(config_path)
    core.check_sources(config)
    if (OUT / 'sealed.json').exists():
        core.verify_seal(OUT, 'ALL_EXPLORE_THIN_GATE_PREDICTIONS_SEALED')
        print({'task': 'B3', 'state': 'ALREADY_SEALED', 'n': config['n']}, flush=True)
        return
    rows = pc.read(OUT / 'manifest.json')
    for name in ('predictions', 'records'):
        (OUT / name).mkdir(exist_ok=True)
    start, cpu = time.monotonic(), time.process_time()
    receipts = []
    missing = []
    for i, b in enumerate(rows, 1):
        name = pc.key(b)
        receipt_path = OUT / 'records' / (name + '.json')
        if receipt_path.exists():
            r = pc.read(receipt_path)
            assert pc.sha(r['prediction_path']) == r['prediction_sha256']
            assert pc.sha(r['source_field_path']) == r['source_field_sha256']
            receipts.append(r)
            continue
        if not (FIELDS / 'records' / (name + '.json')).exists():
            missing.append({'index': name, 'dataset': b['dataset'], 'episode_id': b['row']['episode_id']})
            continue
        path = OUT / 'predictions' / (name + '.npz')
        assert not path.exists(), 'Preserve incomplete B3 prediction'
        tick = time.monotonic()
        predictions, detail = infer(b)
        np.savez_compressed(path, **predictions)
        r = {'index': name, 'dataset': b['dataset'], 'episode_id': b['row']['episode_id'],
             'prediction_path': str(path), 'prediction_sha256': pc.sha(path),
             'cohort': b['cohort'], 'seconds': time.monotonic() - tick, **detail}
        pc.write(receipt_path, r)
        receipts.append(r)
        if i <= 3 or i % 100 == 0:
            print({'task': 'B3', 'produced_n': len(receipts), 'target_n': len(rows),
                   'missing_P0_fields_n': len(missing), 'seconds': round(time.monotonic() - start, 1)}, flush=True)
    cost = {'invocation_wall_seconds': time.monotonic() - start, 'invocation_cpu_seconds': time.process_time() - cpu,
            'episode_seconds_sum': sum(r['seconds'] for r in receipts), 'cpu_threads': 4,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'new_encoder_calls': 0, 'raw_feature_reads': 0, 'query_GT_reads': 0}
    if missing:
        pc.write(OUT / 'pending_fields.json', {'state': 'PARTIAL_WAITING_P0_FIELDS', 'target_n': len(rows),
                 'produced_n': len(receipts), 'missing_n': len(missing), 'missing_tasks': missing, **cost})
        print({'task': 'B3', 'state': 'PARTIAL_WAITING_P0_FIELDS', 'produced_n': len(receipts),
               'target_n': len(rows), 'missing_n': len(missing), 'score_enabled': False}, flush=True)
        return
    pc.write(OUT / 'sealed.json', {'state': 'ALL_EXPLORE_THIN_GATE_PREDICTIONS_SEALED', 'n': len(rows), 'records': receipts,
             'manifest_sha256': pc.sha(OUT / 'manifest.json'), 'config_sha256': pc.sha(config_path), **cost})


def baseline_binding(binding):
    arm = binding.get('old_arms', {}).get('B')
    if arm is None:
        path = P0 / 'foris/new/records' / (pc.key(binding) + '.json')
        if not path.exists():
            raise FileNotFoundError('P0 same-pair FoRIS baseline not ready; no B3 GT scoring before baseline: ' + str(path))
        r = pc.read(path)
        assert r['index'] == pc.key(binding) and r['episode_id'] == binding['row']['episode_id']
        arm = {'path': r['prediction_path'], 'sha256': r['prediction_sha256'], 'cli_key': 'cli/foris.crf'}
    actual = pc.sha(arm['path'])
    if arm.get('sha256'):
        assert actual == arm['sha256'], 'FoRIS baseline changed'
    return {'path': arm['path'], 'sha256': actual, 'cli_key': arm['cli_key']}


def score():
    pc.setup('cpu')
    seal, rows, receipts = core.verify_seal(OUT, 'ALL_EXPLORE_THIN_GATE_PREDICTIONS_SEALED')
    # Preflight every exact-pair control before the first query GT read.
    baseline = {pc.key(b): baseline_binding(b) for b in rows}
    pc.write(OUT / 'baseline_sources_sealed.json', {'state': 'ALL_SAME_PAIR_BASELINES_BOUND',
             'n': len(rows), 'sources': baseline, 'natural_seal_sha256': pc.sha(OUT / 'sealed.json'),
             'query_GT_reads': 0, 'new_encoder_calls': 0})
    records = []
    for i, b in enumerate(rows, 1):
        r = receipts[pc.key(b)]
        with np.load(r['prediction_path'], allow_pickle=False) as f:
            predictions = {a: pc.unpack(f[a]) for a in NATURAL_ARMS}
        source = baseline[pc.key(b)]
        with np.load(source['path'], allow_pickle=False) as f:
            predictions['FoRIS'] = pc.unpack(f[source['cli_key']])
        record = pr.score_record(b, predictions)
        record.update(rho=r['rho'], reference_area_fraction=r['reference_area_fraction'],
                      triggers=r['triggers'], cohort=b['cohort'])
        records.append(record)
        if i <= 3 or i % 100 == 0:
            print({'task': 'B3-score', 'completed': i, 'n': len(rows)}, flush=True)
    notes = ['全部P0 explore原任务及Fundus/Deep新explore配对；不以150/域速度探针替代完整结果。',
             '只复用P0 c32/ind0_32/ref_stats/mask/T/mask/IND0；核对字段SHA/shape，自然基线逐位沿用P0已封存mask。',
             '统一扫描rho>.5/.6/.7/.8/.9；额外严格>.9与网格>.9逐位相同，作为边界核对而非独立搜索臂。',
             '所有门控只读取参考统计；自然预测全部seal后才GT，候选选择仅explore，任何confirm结果留给C1。']
    report = pr.report('B3', OUT, records, ARMS, notes,
                       extra={'production_cost': {k: v for k, v in seal.items() if k != 'records'}})
    trigger_summary = {d: {arm: {'n': sum(r['dataset'] == d for r in records),
                              'triggered_n': sum(r['dataset'] == d and r['triggers'][arm] for r in records),
                              'trigger_rate': sum(r['dataset'] == d and r['triggers'][arm] for r in records) /
                                              max(sum(r['dataset'] == d for r in records), 1),
                              'delta_vs_IND0_pp': report['summary'][d][arm]['class_miou'] - report['summary'][d]['IND0']['class_miou']}
                           for arm in (*GATES, STRICT)}
                       for d in report['summary']}
    triggered_tables = {}
    for threshold, arm in zip(THRESHOLDS, GATES):
        triggered = [r for r in records if r['triggers'][arm]]
        if triggered:
            subset_arms = ('FoRIS', 'IND0', 'T', arm)
            subset_records = [dict(r, metrics={a: r['metrics'][a] for a in subset_arms}) for r in triggered]
            subreport = pr.report(f'B3-rho-gt{threshold:.1f}', OUT / f'rho_gt{threshold:.1f}', subset_records, subset_arms,
                       notes + [f'本表只列rho>{threshold:.1f}实际触发任务；各域n、FN/FP、面积误差和query-photo配对bootstrap均重新聚合。'])
            triggered_tables[arm] = {'threshold': threshold, 'n': len(triggered), 'summary': subreport['summary']}
    new_pairs = [r for r in records if r['cohort'] == 'x1012_P0_new_pair']
    if new_pairs:
        pr.report('B3-new-explore-pairs', OUT / 'new_explore_pairs', new_pairs, ARMS,
                  notes + ['仅Fundus/Deep新配对，参考/query图片仍是已有缓存；配对独立性不等同全新图片。'])
    diagnostics, feasible = {}, []
    for threshold, arm in zip(THRESHOLDS, GATES):
        gains = {d: table[arm]['class_miou'] - table['IND0']['class_miou'] for d, table in report['summary'].items()}
        nondecreasing = all(v >= 0 for v in gains.values())
        thin_gain = all(gains.get(d, float('-inf')) > 0 for d in ('fundus', 'deepglobe_road'))
        diagnostics[arm] = {'threshold': threshold, 'per_domain_delta_vs_IND0_pp': gains,
                            'nondecreasing_all_domains': nondecreasing, 'positive_Fundus_and_Deep': thin_gain,
                            'feasible': nondecreasing and thin_gain}
        if nondecreasing and thin_gain:
            feasible.append((float(np.mean(list(gains.values()))), threshold, arm))
    selected = max(feasible) if feasible else None
    frozen = {'state': 'FROZEN_UNIFORM_REFERENCE_ONLY_GATE' if selected else 'NO_GATE_SATISFIES_EXPLORE_CRITERION',
              'rule': 'T if rho > threshold else IND0' if selected else 'IND0; gate disabled',
              'threshold': selected[1] if selected else None, 'selected_arm': selected[2] if selected else 'IND0',
              'selection': 'Require all ten available domains nondecreasing vs IND0 and positive Fundus/Deep; maximise mean domain gain, larger threshold breaks ties',
              'failure': None if selected else 'No fixed threshold including strict>.9 improves thin domains without harming another domain; no dataset branch added.',
              'diagnostics': diagnostics, 'trigger_summary': trigger_summary, 'triggered_subset_tables': triggered_tables,
              'source_report_sha256': pc.sha(OUT / 'report.json'), 'natural_seal_sha256': pc.sha(OUT / 'sealed.json'),
              'confirm_used_for_selection': False, 'new_encoder_calls': 0}
    pc.write(OUT / 'frozen_gate.json', frozen)
    pc.write(OUT / 'completion.json', {'state': 'COMPLETE_EXPLORE_ONLY', 'n': len(records),
             'gate_state': frozen['state'], 'selected_arm': frozen['selected_arm'], 'threshold': frozen['threshold'],
             'confirm_scored': False, 'new_encoder_calls': 0})
    path = lab.ROOT / 'docs/research/x1012_tasks/B3.md'
    with path.open('a') as f:
        f.write('\n固定门控选择：' + frozen['rule'] + '; threshold=' + str(frozen['threshold']) + '。\n')
        if frozen['failure']:
            f.write('\n失败条件：' + frozen['failure'] + '\n')
        f.write('\n逐域触发率与对IND0影响：\n\n|域|门控|触发n/总n|触发率|对IND0 pp|\n|---|---|---:|---:|---:|\n')
        for d, table in trigger_summary.items():
            for arm, value in table.items():
                f.write(f"|{d}|{arm}|{value['triggered_n']}/{value['n']}|{value['trigger_rate']:.3f}|{value['delta_vs_IND0_pp']:+.3f}|\n")
    print({'task': 'B3', 'gate_state': frozen['state'], 'selected_arm': frozen['selected_arm']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    args = parser.parse_args()
    produce() if args.mode == 'produce' else score()
