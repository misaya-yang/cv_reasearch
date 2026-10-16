"""Reaggregate saved SH integers; no inference, GT files or external assets.

Confidence intervals are copied from W30's original report, never recomputed.
Use --isic for the first P1 development accounting, or --json for all results.
"""
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
import argparse
import gzip
import json
import math


def load_rows(folder, name):
    with gzip.open(folder / name, 'rt', encoding='utf-8') as stream:
        return list(map(json.loads, stream))


def indexed(rows):
    result = {}
    for row in rows:
        key = str(row['index'])
        if key in result:
            raise ValueError('Duplicate episode: ' + key)
        result[key] = row
    return result


def counts(row, arm):
    i, u, p = row[arm]
    g = row['gt']
    if any(type(v) is not int or v < 0 for v in (i, u, p, g)):
        raise ValueError('Nonnegative integer counts required')
    if i > min(p, g) or u != p + g - i:
        raise ValueError('Inconsistent I/U/predicted/GT counts')
    return i, u, p, g


def miou(rows, arm, official=False):
    if not rows:
        raise ValueError('Empty cohort')
    slots = defaultdict(lambda: [0, 0])
    for row in rows:
        i, u, _, _ = counts(row, arm)
        cls = row['cls'] if 'cls' in row else row['class_id']
        totals = slots[row['fold'], cls]
        totals[0] += i
        totals[1] += u
    folds = defaultdict(list)
    for (fold, _), (i, u) in slots.items():
        folds[fold].append(100 * i / max(u, 1))
    return (mean(mean(v) for v in folds.values()) if official else
            mean(v for values in folds.values() for v in values))


def domain_scores(rows, arms, official=False):
    return {d: {a: miou([r for r in rows if r['dataset'] == d], a, official)
                for a in arms} for d in sorted({r['dataset'] for r in rows})}


def matching(row, other, fields):
    if any(row[k] != other[k] for k in fields):
        raise ValueError('Episode identity mismatch: ' + str(row['index']))


def require_close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f'Saved report mismatch: {actual} versus {expected}')


def error_accounting(rows, arm):
    values = [counts(r, arm) for r in rows]
    return dict(n=len(values), FP=sum(p-i for i, u, p, g in values),
                FN=sum(g-i for i, u, p, g in values),
                over_area=sum(p > g for i, u, p, g in values),
                under_area=sum(p < g for i, u, p, g in values),
                median_predicted_over_GT=median(p/g for i, u, p, g in values if g) if any(g for i, u, p, g in values) else None)


def isic_accounting(folder, single_line):
    base = indexed(r for r in load_rows(folder, 'x1013_E1_iu.jsonl.gz') if r['split'] == 'S')
    diagnostics = indexed(load_rows(folder, 'w58_score_repair_cases.jsonl.gz'))
    rows = []
    for row in single_line:
        if row['dataset'] != 'isic':
            continue
        other = base[str(row['index'])]
        matching(row, other, ('dataset', 'fold', 'cls', 'photo', 'gt', 'split'))
        if row['split'] != 'S':
            raise ValueError('P1 uses development S only')
        diagnostic = diagnostics[str(row['index'])]
        matching(row, diagnostic, ('dataset', 'photo', 'gt'))
        require_close(row['SH+'][0] / max(row['SH+'][1], 1), diagnostic['shplus'])
        rows.append(dict(other, **row))
    catastrophic = [r for r in rows if r['SH+'][0] / max(r['SH+'][1], 1) < .3]
    arms = ('U0', 'Mprime', 'SH', 'SH+', 'foris_noCRF', 'foris', 'M', 'MFU')
    return dict(n=len(rows), query_photos=len({r['photo'] for r in rows}),
                metric='Pooled I/U per (fold,class); not episode-mean IoU',
                scores_pp={a: miou(rows, a) for a in arms},
                all={a: error_accounting(rows, a) for a in arms},
                SHplus_below_03={a: error_accounting(catastrophic, a) for a in arms},
                catastrophic_diagnostics=dict(Counter(diagnostics[str(r['index'])]['diagnostic'] for r in catastrophic)),
                scope='Saved S integers only. Marginal FP/FN cannot identify added/deleted TP/FP or the cause of a readout error.')


def summarize(root):
    folder = root / 'evidence/research'
    e1 = load_rows(folder, 'w30_E1_iu.jsonl.gz')
    with gzip.open(folder / 'photo_holdout_iu.json.gz', 'rt', encoding='utf-8') as stream:
        old = indexed(json.load(stream))
    if set(indexed(e1)) != set(old):
        raise ValueError('W30 and baseline episode sets differ')
    for row in e1:
        prior = old[str(row['index'])]
        matching(row, prior, ('dataset', 'fold', 'class_id'))
        for arm, source in (('FoRIS', 'FoRIS'), ('MFU', 'main')):
            v = prior['metrics'][source]
            if row['gt'] != v['GT']:
                raise ValueError('W30 and baseline GT counts differ')
            row[arm] = [v['I'], v['U'], v['predicted']]
    report = json.loads((folder / 'w30_E1_report.json').read_text())
    e1_scores = domain_scores(e1, ('SH', 'SH+', 'FoRIS', 'MFU'))
    macro = {a: mean(v[a] for v in e1_scores.values()) for a in ('SH', 'SH+', 'FoRIS', 'MFU')}
    if len(e1) != report['n']:
        raise ValueError('W30 E1 row count differs')
    for a in macro:
        name = 'main' if a == 'SH+' else a
        require_close(macro[a], report['macro'][name])
        for d, values in e1_scores.items():
            require_close(values[a], report['per_domain'][d][name])
    e2 = load_rows(folder, 'w30_E2_iu.jsonl.gz')
    indexed(e2)
    e2_scores = domain_scores(e2, ('SH', 'SH+'), official=True)
    e2_report = json.loads((folder / 'w30_E2_summary.json').read_text())
    if set(e2_scores) != set(e2_report):
        raise ValueError('W30 E2 completed domain sets differ')
    for d, values in e2_scores.items():
        if sum(r['dataset'] == d for r in e2) != e2_report[d]['n']:
            raise ValueError('W30 E2 row count differs: ' + d)
        for a, value in values.items():
            require_close(value, e2_report[d]['official_miou'][a])
    ablations = {}
    for name, filename in (('W27_S', 'w27_sh_single_line_S.jsonl.gz'), ('W28_S', 'w28_sh_second_batch_S.jsonl.gz')):
        rows = load_rows(folder, filename)
        indexed(rows)
        if any(r['split'] != 'S' for r in rows):
            raise ValueError('Ablations must use S only')
        scores = domain_scores(rows, ('U0+', 'Mprime+', 'SH+'))
        ablations[name] = dict(n=len(rows), domains=scores,
                              macro_pp={a: mean(v[a] for v in scores.values()) for a in ('U0+', 'Mprime+', 'SH+')})
        if name == 'W27_S':
            p1 = isic_accounting(folder, rows)
    return dict(scope='Saved integer reaggregation; no inference, query GT files, asset paths or network. Original CIs are quoted, not re-estimated.',
                W30_E1=dict(n=len(e1), macro_pp=macro, domains=e1_scores,
                            saved_comparisons=report['paired']['macro10']),
                W30_E2=dict(n=len(e2), domains=e2_scores,
                            scope='SH levels only; baseline integers and the full official protocol manifest are external. Completed-domain point estimates match the saved official report.'),
                **ablations, P1_ISIC_S=p1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--isic', action='store_true', help='Print only the P1 S-half accounting as JSON')
    args = parser.parse_args()
    if args.isic:
        folder = args.root / 'evidence/research'
        result = isic_accounting(folder, load_rows(folder, 'w27_sh_single_line_S.jsonl.gz'))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = summarize(args.root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print('Saved integers only; no new inference or confidence intervals.')
    print('W30 E1 (n=3211):', ', '.join(f'{a} {v:.6f}' for a, v in result['W30_E1']['macro_pp'].items()))
    print('SH+ - FoRIS: {:+.6f} pp'.format(result['W30_E1']['macro_pp']['SH+']-result['W30_E1']['macro_pp']['FoRIS']))
    for d, scores in result['W30_E2']['domains'].items():
        print(f'W30 E2 {d}: SH+ {scores["SH+"]:.6f}')
    for name in ('W27_S', 'W28_S'):
        scores = result[name]['macro_pp']
        print(f'{name} SH+ minus U0+/Mprime+: {scores["SH+"]-scores["U0+"]:+.6f} / {scores["SH+"]-scores["Mprime+"]:+.6f} pp')
    print('P1 ISIC S:', result['P1_ISIC_S']['n'], 'episodes; use --isic for error accounting.')


if __name__ == '__main__':
    main()
