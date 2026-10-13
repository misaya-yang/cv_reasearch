"""Exact exposed-development scoring of a frozen Boolean composition library."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import time
import numpy as np
import x1012s2_composition_library as lib


def boot_difference(rows, cand_i, cand_u, base_i, base_u, seed=20261013, repeats=1000):
    rng = np.random.default_rng(seed)
    groups = sorted({r['query_photo'] for r in rows})
    slots = sorted({(str(r['fold']), str(r['class_id'])) for r in rows})
    gm, sm = ({v: i for i, v in enumerate(values)} for values in (groups, slots))
    totals = np.zeros((4, len(groups), len(slots)), np.float64)
    for n, row in enumerate(rows):
        g = gm[row['query_photo']]
        s = sm[(str(row['fold']), str(row['class_id']))]
        totals[:, g, s] += (cand_i[n], cand_u[n], base_i[n], base_u[n])
    differences = []
    for _ in range(repeats):
        weight = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        ci, cu, bi, bu = np.einsum('g,ags->as', weight, totals)
        present = (cu + bu) > 0
        differences.append(float(np.mean(100 * (ci[present] / np.maximum(cu[present], 1)
                                      - bi[present] / np.maximum(bu[present], 1)))))
    return np.percentile(differences, [2.5, 97.5]).tolist()


def main(histogram_path):
    started = time.monotonic()
    config = lib.read(lib.OUT / 'config.json')
    assert lib.sha(lib.OUT / 'rules.json') == config['rules_sha256']
    assert lib.sha(lib.OUT / 'natural_bindings.json') == config['natural_bindings_sha256']
    rules = lib.read(lib.OUT / 'rules.json')
    bindings = {r['index']: r for r in lib.read(lib.OUT / 'natural_bindings.json')}
    hist = lib.read(histogram_path)
    rows = hist['records']
    assert hist['n'] == len(rows) == config['n_cases']
    assert hist['bit_order_LSB_first'] == list(lib.SYMBOLS)
    assert hist['manifest_sha256'] == config['manifest_sha256']
    assert hist['all_existing_path_integer_metrics_reproduced']
    for row in rows:
        expected = bindings[row['index']]
        for key in ('dataset', 'fold', 'class_id'):
            assert str(row[key]) == str(expected[key]), (row['index'], key)
        assert Path(row['query_photo']).resolve() == Path(expected['query_photo']).resolve()
        row['query_photo'] = expected['query_photo']
        for symbol in lib.SYMBOLS:
            source, bound = row['prediction_sources'][symbol], expected['sources'][symbol]
            assert source['path'] == bound['path']
            assert source['mask_key'] == bound['key']
            assert source['file_sha256'] == bound['sha256']
    lookup = np.array([[(int(r['truth_table_hex'], 16) >> p) & 1 for p in range(64)]
                       for r in rules], np.float64)
    fg = np.asarray([r['fg_count'] for r in rows], np.float64)
    bg = np.asarray([r['bg_count'] for r in rows], np.float64)
    assert np.array_equal((fg + bg).sum(1), np.array([r['valid_pixel_n'] for r in rows]))
    domains = list(dict.fromkeys(r['dataset'] for r in rows))
    slots = sorted({(r['dataset'], str(r['fold']), str(r['class_id'])) for r in rows})
    slot_index = {v: i for i, v in enumerate(slots)}
    fg_slot, bg_slot = np.zeros((len(slots), 64)), np.zeros((len(slots), 64))
    for k, row in enumerate(rows):
        s = slot_index[(row['dataset'], str(row['fold']), str(row['class_id']))]
        fg_slot[s] += fg[k]
        bg_slot[s] += bg[k]
    intersections = lookup @ fg_slot.T
    false_positive = lookup @ bg_slot.T
    unions = fg_slot.sum(1)[None, :] + false_positive
    slot_iou = 100 * intersections / np.maximum(unions, 1)
    scores = np.column_stack([slot_iou[:, [i for i, s in enumerate(slots) if s[0] == d]].mean(1)
                              for d in domains])
    macro = scores.mean(1)
    ids = {r['id']: i for i, r in enumerate(rules)}
    sources = {symbol: ids[rule_id] for symbol, rule_id in config['source_rule_ids'].items()}
    original = lib.read(lib.ASSETS / 'a/x1012_R1/report.json')['summary']
    for column, domain in enumerate(domains):
        for symbol, arm in (('B', 'FoRIS'), ('M', 'MEAN'), ('U', 'VP_UN'), ('T', 'T')):
            assert abs(scores[sources[symbol], column] - original[domain][arm]['class_miou']) < 1e-8
    rank = sorted(range(len(rules)), key=lambda i: (-macro[i], len(rules[i]['essential_sources']),
                                                   len(rules[i]['operations']), rules[i]['id']))
    best_per_dependency = {}
    for i in rank:
        dependency = ''.join(rules[i]['essential_sources'])
        best_per_dependency.setdefault(dependency, i)
    # Keep a score/dependency frontier without pretending sources have equal cost.
    frontier = [i for i in rank if not any(
        set(rules[j]['essential_sources']) < set(rules[i]['essential_sources'])
        and macro[j] >= macro[i] - 1e-12 for j in best_per_dependency.values())]
    frontier_best = [i for i in frontier if i in best_per_dependency.values()]
    base_bits = np.asarray([(p & 1) for p in range(64)], np.float64)

    def summarize(i):
        table = lookup[i]
        additions = table * (1 - base_bits)
        deletions = (1 - table) * base_bits
        by_domain = {}
        for column, domain in enumerate(domains):
            selected = [k for k, r in enumerate(rows) if r['dataset'] == domain]
            gf, gb = fg[selected].sum(0), bg[selected].sum(0)
            by_domain[domain] = {'n': len(selected), 'class_miou': float(scores[i, column]),
                'delta_to_FoRIS_pp': float(scores[i, column] - scores[sources['B'], column]),
                'edits_vs_FoRIS': {'add_TP': int(additions @ gf), 'add_FP': int(additions @ gb),
                                  'delete_TP': int(deletions @ gf), 'delete_FP': int(deletions @ gb)}}
        return {**rules[i], 'macro10_class_miou': float(macro[i]),
                'macro_delta_to_FoRIS_pp': float(macro[i] - macro[sources['B']]),
                'worst_domain_delta_to_FoRIS_pp': float((scores[i] - scores[sources['B']]).min()),
                'positive_domains_vs_FoRIS': sum(scores[i] > scores[sources['B']] + 1e-10),
                'negative_domains_vs_FoRIS': sum(scores[i] < scores[sources['B']] - 1e-10),
                'domains': by_domain}

    best = rank[0]
    best_record = summarize(best)
    selected_eval = sorted(set([best, *sources.values()]))
    case_i = lookup[selected_eval] @ fg.T
    case_u = fg.sum(1)[None, :] + lookup[selected_eval] @ bg.T
    eval_index = {i: n for n, i in enumerate(selected_eval)}
    paired = {}
    for symbol, other in sources.items():
        values = {}
        for col, domain in enumerate(domains):
            selected = [k for k, r in enumerate(rows) if r['dataset'] == domain]
            values[domain] = {'delta_pp': float(scores[best, col] - scores[other, col]),
                'paired_query_photo_bootstrap95_pp': boot_difference(
                    [rows[k] for k in selected], case_i[eval_index[best], selected],
                    case_u[eval_index[best], selected], case_i[eval_index[other], selected],
                    case_u[eval_index[other], selected])}
        paired[symbol] = values
    np.savez_compressed(lib.OUT / 'all_rule_scores.npz', scores=scores, macro=macro,
                        domains=np.asarray(domains), rule_ids=np.asarray([r['id'] for r in rules]))
    result = {'state': 'COMPLETE_EXPOSED_DEVELOPMENT_SEARCH', 'n_cases': len(rows),
        'n_rules': len(rules), 'operation_count_limit': None,
        'config_sha256': lib.sha(lib.OUT / 'config.json'),
        'joint_histogram_path': str(histogram_path.resolve()),
        'joint_histogram_sha256': lib.sha(histogram_path), 'scoring_source_sha256': lib.sha(__file__),
        'baselines': {s: summarize(i) for s, i in sources.items()},
        'best': best_record, 'top50': [summarize(i) for i in rank[:50]],
        'best_per_exact_dependency': {k: summarize(i) for k, i in best_per_dependency.items()},
        'score_dependency_frontier': [summarize(i) for i in frontier_best],
        'best_paired_comparisons': paired, 'score_seconds': time.monotonic() - started,
        'training': False, 'selection_uses_exposed_development_GT': True,
        'new_confirmation_done': False, 'full_pipeline_latency_verified': False,
        'limitations': 'Exact closure only of the six-source add/delete chain family. Development ranking is not held-out confirmation or a claim of optimality over other operators.'}
    # numpy scalar booleans/counts to JSON scalars without hiding nonfinite values.
    def clean(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    result = clean(result)
    lib.write(lib.OUT / 'report.json', result)
    best_metrics = []
    for k, r in enumerate(rows):
        metrics = {}
        for name, index in {'candidate': best, **sources}.items():
            intersection, union = (int(v[eval_index[index], k]) for v in (case_i, case_u))
            gt = int(fg[k].sum())
            metrics[name] = {'I': intersection, 'U': union, 'TP': intersection,
                            'FP': union - gt, 'FN': gt - intersection, 'GT': gt,
                            'predicted': intersection + union - gt}
        best_metrics.append({k2: r[k2] for k2 in ('index', 'dataset', 'fold', 'class_id', 'query_photo')}
                            | {'metrics': metrics})
    lib.write(lib.OUT / 'selected_case_metrics.json', best_metrics)
    header = ['# 六路自然预测的完整增删闭包搜索', '',
        f'全部{len(rows)}个已曝光开发任务，{len(rules)}个不同组合。操作长度未设限；这里只穷尽本组六路AND/OR链，不是所有算法。', '',
        f"开发最高：`{best_record['id']}`，`{best_record['expression']}`；十域均值{macro[best]:.3f}，相对FoRIS{macro[best]-macro[sources['B']]:+.3f}pp。", '',
        '最终选择还需要完整计算依赖的延迟和冻结后的新确认；不能把布尔运算耗时当完整方法耗时。', '',
        '|域|FoRIS|MEAN|UN|T|INS bilinear|INS CRF|组合|组合−FoRIS|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for col, d in enumerate(domains):
        vals = [scores[sources[s], col] for s in lib.SYMBOLS] + [scores[best, col], scores[best, col] - scores[sources['B'], col]]
        header.append('|' + d + '|' + '|'.join(f'{v:.3f}' for v in vals) + '|')
    header += ['', '## 分数与依赖集合前沿', '', '|规则|所需来源|十域均值|对FoRIS|', '|---|---|---:|---:|']
    for i in frontier_best:
        header.append(f"|`{rules[i]['expression']}`|{','.join(rules[i]['essential_sources'])}|{macro[i]:.3f}|{macro[i]-macro[sources['B']]:+.3f}|")
    header += ['', '完整逐域增删整数、全部分数、同照片配对区间和自然规则冻结见仓库外同名目录。']
    text = '\n'.join(header) + '\n'
    (lib.OUT / 'REPORT.md').write_text(text)
    (lib.REPO / 'docs/research/COMPOSITION_BOOLEAN_RESULTS_20261012.md').write_text(text)
    print({'state': result['state'], 'rules': len(rules), 'n': len(rows),
           'best': {k: best_record[k] for k in ('id', 'expression', 'essential_sources', 'macro10_class_miou',
                'macro_delta_to_FoRIS_pp', 'worst_domain_delta_to_FoRIS_pp')},
           'baselines': {s: float(macro[i]) for s, i in sources.items()},
           'score_seconds': result['score_seconds']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histogram', type=Path,
        default=lib.ASSETS / 'a/x1012s2_error_overlap/joint6_1360.json')
    args = parser.parse_args()
    main(args.histogram)
