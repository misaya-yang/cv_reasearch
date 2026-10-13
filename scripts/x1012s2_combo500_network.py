"""Frozen network-v2 replay on the same exposed 500, preserving v1 outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import time
import x1012s2_combo500 as old
import numpy as np

ROOT, pc, pr, lab = old.ROOT, old.pc, old.pr, old.lab
read, write, sha = old.read, old.write, old.sha
PARENT = old.OUT
OUT = PARENT/'network_v2'
SEARCH = old.ASSETS/'a/x1012s2_composition_search/boolean_network_v2'
RULES = {
    'network4': {'truth_table_hex': 'fce0fce0fce0fce0',
                 'expression': '(B&U)|(M&U)|(M&T)|(U&T)'},
    'network6': {'truth_table_hex': 'fee0fee0fee0fcc0',
                 'expression': '(M&U)|(M&T)|(U&T)|(B&U&I)|(B&T&I)|(B&U&J)|(B&T&J)'},
}
BASELINES = old.BASELINES+('r00097',)


def minimal_terms(table):
    return [p for p in range(64) if (table >> p) & 1 and
            all(not ((table >> (p ^ (1 << bit))) & 1) for bit in range(6) if p & (1 << bit))]


def apply_network(table, sources):
    result = np.zeros_like(sources['B'])
    for term in minimal_terms(table):
        conjunction = np.full_like(result, 255)
        for bit, symbol in enumerate(old.SYMBOLS):
            if term & (1 << bit):
                conjunction &= sources[symbol]
        result |= conjunction
    return result


def prepare():
    for folder in ('predictions', 'records', 'frozen'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    path = OUT/'config.json'
    if not path.exists():
        parent_config = read(PARENT/'config.json')
        parent_seal = read(PARENT/'sealed.json')
        assert parent_seal['config_sha256'] == sha(PARENT/'config.json')
        receipts = {r['index']: r for r in parent_seal['records']}
        rows = read(PARENT/'bindings.json')
        assert len(rows) == len(receipts) == 500
        for row in rows:
            source = receipts[row['index']]
            row['sources']['T'] = {'path': source['prediction_path'], 'sha256': source['prediction_sha256'], 'key': 'T'}
            row['original_rule'] = {'path': source['prediction_path'], 'sha256': source['prediction_sha256'], 'key': 'r00097'}
        write(OUT/'bindings.json', rows)
        sources = {s: ((np.arange(64) >> j) & 1).astype(np.uint8) for j, s in enumerate(old.SYMBOLS)}
        for rule in RULES.values():
            table = int(rule['truth_table_hex'], 16)
            expected = np.array([(table >> p) & 1 for p in range(64)], np.uint8)
            assert np.array_equal(apply_network(table, sources), expected)
        assert np.array_equal(apply_network(int(RULES['network4']['truth_table_hex'], 16), sources),
                              ((sources['B'] | sources['M'] | sources['T']) & sources['U']) | (sources['M'] & sources['T']))
        deps = [PARENT/'config.json', PARENT/'sealed.json', PARENT/'bindings.json',
                SEARCH/'config.json', SEARCH/'report.json',
                ROOT/'scripts/x1012s2_boolean_network_search.py', Path(old.__file__)]
        config = {
            'state': 'TWO_NETWORKS_FROZEN_BEFORE_NATURAL500_OUTPUTS', 'n': 500,
            'rules': {name: dict(rule, minimal_positive_terms=minimal_terms(int(rule['truth_table_hex'], 16)))
                      for name, rule in RULES.items()},
            'selection': 'Exactly two externally requested networks selected on exposed development1360. No R5-based modification or re-selection.',
            'source_symbols_LSB_first': old.SYMBOLS,
            'parent_primary_retained': 'r00097=((B|M|T)&U); parent files unchanged',
            'new4_branch_vs_r00097': 'M&T&~U only; no deleted pixels',
            'T': 'Reuse all500 T masks already frozen in parent seal; no new field readout or encoding',
            'exposure': 'Same already exposed R5_500; development reproduction, not new blind confirmation. New confirm1000 is out of scope and never accessed.',
            'all500_masks_sealed_before_GT': True, 'new_encoder_calls': 0, 'training': False,
            'resources': {'device': 'cpu', 'workers': 1, 'threads': 1},
            'source_sha256': sha(__file__), 'dependencies_sha256': {str(p): sha(p) for p in deps},
            'bindings_sha256': sha(OUT/'bindings.json'),
            'paired_CI': '1000 query-photo cluster resamples per domain, seed20261013; equal-weight domain macro; same fixed-rule descriptive procedure as parent',
            'comparisons': ['each network vs FoRIS/MEAN/UN_midrange/r00097', 'network6 vs network4'],
        }
        shutil.copyfile(__file__, OUT/'frozen/x1012s2_combo500_network.py')
        write(path, config)
    config = read(path)
    assert sha(__file__) == config['source_sha256']
    assert all(sha(p) == h for p, h in config['dependencies_sha256'].items())
    assert sha(OUT/'bindings.json') == config['bindings_sha256']
    return read(OUT/'bindings.json'), config


def produce():
    rows, config = prepare()
    assert not (OUT/'sealed.json').exists(), 'Preserve sealed network-v2 outputs'
    began, records = time.monotonic(), []
    for i, row in enumerate(rows, 1):
        key = row['index']; target = OUT/'predictions'/(key+'.npz')
        assert not target.exists(), 'Preserve partial natural prediction'
        with pc.inference_only(row['binding']):
            sources = old.load_existing(row)
            original = row['original_rule']
            assert sha(original['path']) == original['sha256']
            with np.load(original['path'], allow_pickle=False) as z:
                parent = z[original['key']].copy()
            predictions = {a: apply_network(int(rule['truth_table_hex'], 16), sources) for a, rule in config['rules'].items()}
            assert np.array_equal(predictions['network4'], parent | (sources['M'] & sources['T']))
            assert np.array_equal(predictions['network4'] ^ parent, sources['M'] & sources['T'] & ~sources['U'])
            np.savez_compressed(target, **predictions)
        r = {'index': key, 'dataset': row['dataset'], 'prediction_path': str(target),
             'prediction_sha256': sha(target), 'query_GT_reads': 0, 'new_encoder_calls': 0}
        records.append(r); write(OUT/'records'/(key+'.json'), r)
        if i <= 2 or i % 100 == 0:
            print({'phase': 'network-natural', 'completed': i, 'n': 500, 'seconds': round(time.monotonic()-began, 1)}, flush=True)
    write(OUT/'sealed.json', {'state': 'ALL500_TWO_NETWORK_NATURAL_OUTPUTS_SEALED', 'n': 500,
        'records': records, 'config_sha256': sha(OUT/'config.json'), 'bindings_sha256': sha(OUT/'bindings.json'),
        'parent_seal_sha256': sha(PARENT/'sealed.json'), 'query_GT_reads': 0, 'new_encoder_calls': 0,
        'training': False, 'wall_seconds': time.monotonic()-began,
        'latency_scope': 'saved-mask Boolean replay only; complete source computation excluded'})
    print({'state': 'ALL500_TWO_NETWORK_NATURAL_OUTPUTS_SEALED', 'seal_sha256': sha(OUT/'sealed.json')}, flush=True)


def score():
    rows, config = prepare()
    seal = read(OUT/'sealed.json')
    assert seal['n'] == 500 and seal['config_sha256'] == sha(OUT/'config.json')
    receipts = {r['index']: r for r in seal['records']}
    assert all(sha(r['prediction_path']) == r['prediction_sha256'] for r in receipts.values())
    parent_report = read(PARENT/'report.json')
    previous = {r['index']: r for r in parent_report['records']}
    arms = tuple(old.ARM.values())+('r00097',)+tuple(RULES)
    began, records = time.monotonic(), []
    for i, row in enumerate(rows, 1):
        key = row['index']; b = row['binding']
        sources = old.load_existing(row)
        predictions = {old.ARM[s]: pc.unpack(p) for s, p in sources.items()}
        with np.load(row['original_rule']['path'], allow_pickle=False) as z:
            predictions['r00097'] = pc.unpack(z['r00097'])
        with np.load(receipts[key]['prediction_path'], allow_pickle=False) as z:
            predictions.update({a: pc.unpack(z[a]) for a in RULES})
        gt, valid = pc.truth(b)
        r = {'index': key, 'dataset': b['dataset'], 'fold': b['row']['fold'], 'class_id': b['row']['global_class_id'],
             'query_photo': str((old.ASSETS/b['row']['query_path']).resolve()),
             'metrics': {a: pr.metric(predictions[a], gt, valid) for a in arms},
             'edits': {a: {base: old.edit_counts(predictions[a], predictions[base], gt, valid)
                           for base in BASELINES} for a in RULES}}
        for a in tuple(old.ARM.values())+('r00097',):
            assert all(r['metrics'][a][k] == previous[key]['metrics'][a][k] for k in ('I', 'U', 'TP', 'FP', 'FN', 'GT', 'predicted')), (key, a)
        assert r['edits']['network4']['r00097'][2:] == [0, 0]
        for a in RULES:
            for base in BASELINES:
                at, af, dt, df = r['edits'][a][base]
                assert r['metrics'][a]['TP'] == r['metrics'][base]['TP']+at-dt
                assert r['metrics'][a]['FP'] == r['metrics'][base]['FP']+af-df
        records.append(r)
        if i <= 2 or i % 100 == 0:
            print({'phase': 'network-exposed-score', 'completed': i, 'n': 500, 'seconds': round(time.monotonic()-began, 1)}, flush=True)
    summary = pr.summary(records, arms)
    # Use the already frozen parent bootstrap implementation with this process's
    # two requested rule labels. This changes no parent source/config or file.
    old.RULE_IDS = tuple(RULES)
    old.BASELINES = old.BASELINES+('network4',)
    paired = old.bootstrap(records, arms, summary)
    for table in summary.values():
        for value in table.values():
            value.pop('paired_query_photo_bootstrap95_pp', None)
    edits = {d: {a: {base: np.sum([r['edits'][a][base] for r in records if r['dataset'] == d], axis=0).tolist()
                       for base in BASELINES} for a in RULES} for d in lab.ALL_DOMAINS}
    result = {'state': 'COMPLETE_FIXED_NETWORK_EXPOSED500_REPRODUCTION', 'n': 500, 'rules': config['rules'],
        'summary': summary, 'macro10_class_miou': {a: float(np.mean([summary[d][a]['class_miou'] for d in lab.ALL_DOMAINS])) for a in arms},
        'paired_comparisons': paired, 'edit_columns': old.EDITS,
        'edit_columns_zh': ['补回TP', '加入FP', '删除TP', '移除FP'], 'edits': edits, 'records': records,
        'config_sha256': sha(OUT/'config.json'), 'seal_sha256': sha(OUT/'sealed.json'),
        'parent_report_sha256': sha(PARENT/'report.json'), 'parent_seal_sha256': sha(PARENT/'sealed.json'),
        'all_existing_controls_integer_identity': True, 'new_encoder_calls': 0, 'training': False,
        'post_score_rule_changes': False, 'new_blind_confirmation': False, 'confirm1000_accessed': False,
        'full_pipeline_latency_measured_here': False, 'exposure': config['exposure'],
        'score_seconds': time.monotonic()-began}
    write(OUT/'report.json', result)
    compact = {k: v for k, v in result.items() if k != 'records'}
    compact.update(report_path=str(OUT/'report.json'), report_sha256=sha(OUT/'report.json'))
    write(OUT/'completion.json', compact)
    write(ROOT/'evidence/research/x1012s2_combo500_network_v2.json', compact)
    lines = ['# 分支AND/OR网络在已曝光500配对上的固定复现', '',
      '**同一已曝光R5开发复现，不是新盲测；两条网络先冻结并封存全部自然输出，再读GT评分。不修改原r00097，不访问新confirm1000。**', '',
      '`network4=fce0fce0fce0fce0`：`(B&U)|(M&U)|(M&T)|(U&T)`，相对原r00097只新增`M&T&~U`。',
      '`network6=fee0fee0fee0fcc0`：`(M&U)|(M&T)|(U&T)|(B&U&I)|(B&T&I)|(B&U&J)|(B&T&J)`。', '',
      '|域|FoRIS|MEAN|UN|原r00097|network4|network6|network4−原|network6−原|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d,t in summary.items():
        values=[t[a]['class_miou'] for a in BASELINES+tuple(RULES)]
        values += [paired['domains'][d][a]['r00097']['delta_pp'] for a in RULES]
        lines.append('|'+d+'|'+'|'.join(f'{v:.4f}' for v in values)+'|')
    values=[result['macro10_class_miou'][a] for a in BASELINES+tuple(RULES)]
    values += [paired['macro10'][a]['r00097']['delta_pp'] for a in RULES]
    lines += ['|十域均值|'+'|'.join(f'{v:.4f}' for v in values)+'|', '',
      '## 对原r00097的真实增删代价', '',
      '|域|网络|补回TP|新增FP|删除TP|移除FP|对原分数pp|', '|---|---|---:|---:|---:|---:|---:|']
    for d,t in edits.items():
        for a in RULES:
            lines.append('|'+d+'|'+a+'|'+'|'.join(str(x) for x in t[a]['r00097'])+f"|{paired['domains'][d][a]['r00097']['delta_pp']:+.4f}|")
    lines += ['', '## 固定规则照片配对区间', '',
      '|范围|网络|对照|差值pp|95%区间|', '|---|---|---|---:|---|']
    for scope,t in [*paired['domains'].items(), ('macro10',paired['macro10'])]:
        for a in RULES:
            for base in BASELINES+('network4',):
                if a==base:continue
                v=t[a][base];lo,hi=v['paired_query_photo_bootstrap95_pp']
                lines.append(f"|{scope}|{a}|{base}|{v['delta_pp']:+.4f}|[{lo:+.4f},{hi:+.4f}]|")
    lines += ['', '全部对FoRIS/MEAN/UN/原规则的四项整数与逐例评分保留在JSON；CI为曝光数据上固定规则的描述性区间，不校正此前开发选择。',
      '[可复算结果](../../evidence/research/x1012s2_combo500_network_v2.json)。',
      f'封存SHA `{result["seal_sha256"]}`；配置SHA `{result["config_sha256"]}`。',
      '新增编码0、拟合0。执行时间只覆盖既有mask重放，不能冒充完整方法延迟。']
    (ROOT/'docs/research/COMPOSITION_NETWORK_500_REPLAY_20261013.md').write_text('\n'.join(lines)+'\n')
    print({'state': result['state'], 'macro10_class_miou': result['macro10_class_miou'],
           'macro_paired': paired['macro10'], 'seconds': round(time.monotonic()-began,1)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze','produce','score'))
    args = parser.parse_args()
    if args.mode == 'freeze':
        rows, config = prepare(); print({'state':config['state'],'n':len(rows)},flush=True)
    elif args.mode == 'produce':
        produce()
    else:
        score()
