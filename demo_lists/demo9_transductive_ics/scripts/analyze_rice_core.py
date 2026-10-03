#!/usr/bin/env python3
"""CPU-only paired analysis of the frozen RICE core experiment, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ('foris_crf', 'identity_density', 'plain_discriminant', 'fg_aug_fisher',
        'pooled_fisher', 'paired_only', 'paired_shuffle', 'foris_identity_density',
        'foris_pooled_fisher', 'foris_paired_only', 'foris_paired_shuffle')
LEDGER = ('recovered_fn', 'lost_tp', 'added_fp', 'removed_fp')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task(row):
    return tuple(int(row[k]) for k in ('fold', 'e', 'c'))


def counts(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Each I/U must contain two integer counts')
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in value):
        raise ValueError('I/U counts must be nonnegative integers')
    if value[0] > value[1]:
        raise ValueError('Intersection exceeds union')
    return np.asarray(value, dtype=np.float64)


def class_table(rows, arm, field):
    totals = {}
    for row in rows:
        key = (int(row['fold']), int(row['c']))
        totals[key] = totals.get(key, np.zeros(2)) + counts(row[field][arm])
    return {key: float(100 * val[0] / max(val[1], 1)) for key, val in totals.items()}, totals


def aggregate(table):
    folds = sorted({f for f, _ in table})
    per_fold = {str(f): float(np.mean([v for (ff, _), v in table.items() if ff == f])) for f in folds}
    return dict(fold_mean_class_miou=float(np.mean(list(per_fold.values()))), per_fold=per_fold)


def paired(rows, arm, baseline, field, repetitions=2000, seed=2053):
    a, _ = class_table(rows, arm, field)
    b, _ = class_table(rows, baseline, field)
    if a.keys() != b.keys():
        raise ValueError('Unpaired class tables')
    rng = np.random.default_rng(seed)
    samples = []
    by_fold = {}
    for fold in sorted({f for f, _ in a}):
        keys = sorted(k for k in a if k[0] == fold)
        differences = np.array([a[k] - b[k] for k in keys])
        draws = differences[rng.integers(0, len(keys), size=(repetitions, len(keys)))].mean(1)
        samples.append(draws)
        by_fold[str(fold)] = dict(delta_pp=float(differences.mean()), classes=len(keys),
            exploratory_delta95_pp=np.quantile(draws, [.025, .975]).tolist())
    pooled = np.mean(samples, axis=0)
    differences = [a[k] - b[k] for k in sorted(a)]
    return dict(delta_pp=aggregate(a)['fold_mean_class_miou']-aggregate(b)['fold_mean_class_miou'],
        exploratory_delta95_pp=np.quantile(pooled, [.025, .975]).tolist(), per_fold=by_fold,
        per_class=[dict(fold=k[0], c=k[1], delta_pp=a[k]-b[k]) for k in sorted(a)],
        class_signs=dict(positive=sum(v > 0 for v in differences), negative=sum(v < 0 for v in differences),
                         tie=sum(v == 0 for v in differences)),
        resampling='Classes within each fixed fold; equal fold mean', bootstrap_repetitions=repetitions,
        seed=seed, fitted_models_and_thresholds_fixed=True,
        qualification='Exploratory development subset; shared photos violate independent-class interpretation')


def analyze(report, manifest, report_sha=None, manifest_sha=None):
    if report.get('state') != 'COMPLETED':
        raise ValueError('Only the completed frozen core experiment is analyzed')
    if manifest.get('schema') != 'rice_core_assets_v1' or manifest.get('state') != 'PREPARED_ASSETS':
        raise ValueError('Frozen RICE assets manifest required')
    arms = tuple(manifest.get('arms', []))
    if len(arms) != 11 or len(set(arms)) != 11 or set(arms) != set(ARMS):
        raise ValueError('Frozen eleven-arm comparison, including full-pipeline insert controls, required')
    expected = {task(r): r for r in manifest['frozen_episodes']}
    rows = report['records']
    if len(expected) != 40 or len(rows) != 40 or len({task(r) for r in rows}) != 40:
        raise ValueError('Exactly forty unique prepared/observed tasks required')
    if set(expected) != {task(r) for r in rows}:
        raise ValueError('Actual task identities differ from frozen manifest')
    if any(sum(k[0] == f for k in expected) != 10 for f in range(4)):
        raise ValueError('Exactly ten tasks per fold0..3 required')
    photo_tasks = {}
    for row in rows:
        frozen = expected[task(row)]
        for role in ('support', 'query'):
            if row.get(role) != frozen[role]:
                raise ValueError('Actual support/query photo differs from frozen task')
            photo_tasks.setdefault(row[role], set()).add(task(row))
        for field in ('iu', 'original_iu'):
            if set(row.get(field, {})) != set(arms):
                raise ValueError('Missing/extra scored arm: ' + field)
            for arm in arms:
                counts(row[field][arm])
    repeated = [dict(photo=p, tasks=[list(k) for k in sorted(tasks)])
                for p, tasks in sorted(photo_tasks.items()) if len(tasks) > 1]
    scores, comparisons, oracle = {}, {}, {}
    for field in ('iu', 'original_iu'):
        scores[field] = {}
        for arm in arms:
            table, totals = class_table(rows, arm, field)
            scores[field][arm] = {**aggregate(table), 'per_class': [dict(fold=k[0], c=k[1],
                I=int(totals[k][0]), U=int(totals[k][1]), iou=table[k]) for k in sorted(table)]}
        comparisons[field] = {arm+' vs foris_crf': paired(rows, arm, 'foris_crf', field)
                             for arm in arms if arm != 'foris_crf'}
        for baseline in ('identity_density', 'plain_discriminant', 'fg_aug_fisher', 'pooled_fisher', 'paired_shuffle'):
            if baseline != 'paired_only':
                comparisons[field]['paired_only vs '+baseline] = paired(rows, 'paired_only', baseline, field)
        for baseline in ('foris_identity_density', 'foris_pooled_fisher', 'foris_paired_shuffle'):
            comparisons[field]['foris_paired_only vs '+baseline] = paired(rows, 'foris_paired_only', baseline, field)
        selected = []
        choices = []
        for row in rows:
            # Stable arm order resolves ties; GT selection is diagnostic only.
            arm = max(arms, key=lambda name: row[field][name][0] / max(row[field][name][1], 1))
            selected.append({**row, field: {'episode_oracle': row[field][arm]}})
            choices.append(dict(fold=row['fold'], e=row['e'], c=row['c'], arm=arm))
        table, _ = class_table(selected, 'episode_oracle', field)
        oracle[field] = {**aggregate(table), 'choices': choices,
            'scope': 'GT chooses best frozen arm per episode; finite bank diagnostic, NOT a class-mIoU ceiling or deployable method'}
    harm = {}
    for arm in arms:
        entries = []
        for row in rows:
            ledger = row.get('ledger', {}).get(arm)
            if ledger is None:
                raise ValueError('Missing native FoRIS harm ledger for ' + arm)
            if set(ledger) != set(LEDGER) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in ledger.values()):
                raise ValueError('Invalid harm ledger')
            current, anchor = row['iu'][arm], row['iu']['foris_crf']
            if current[0]-anchor[0] != ledger['recovered_fn']-ledger['lost_tp']:
                raise ValueError('Harm ledger disagrees with intersection change')
            if current[1]-anchor[1] != ledger['added_fp']-ledger['removed_fp']:
                raise ValueError('Harm ledger disagrees with union change')
            if arm == 'foris_crf' and any(ledger.values()):
                raise ValueError('Native host compared with itself must have zero harm ledger')
            entries.append(ledger)
        harm[arm] = {k: sum(r[k] for r in entries) for k in LEDGER}
        harm[arm]['per_fold'] = {str(f): {k: sum(row['ledger'][arm][k] for row in rows if row['fold'] == f)
                                              for k in LEDGER} for f in range(4)}
        harm[arm]['scope'] = 'Model1024 pixel counts relative to native foris_crf; not original-resolution counts'
    return dict(state='CPU_FROZEN_OUTPUT_ANALYSIS', report_sha256=report_sha, manifest_sha256=manifest_sha,
        records=40, arms=list(arms), aggregation='Within class sumI/sumU, within fold class mean, equal mean across four folds',
        model_and_original_resolution_separate=True, scores=scores, comparisons=comparisons,
        harm_ledger=harm, episode_oracle=oracle, repeated_all_role_photos=repeated,
        independent_class_bootstrap_photo_warning=bool(repeated),
        scope='Fixed forty-episode development evidence; no all-fold full benchmark claim; no new model/data/GPU work')


def self_check():
    manifest = dict(schema='rice_core_assets_v1', state='PREPARED_ASSETS', frozen_episodes=[], arms=list(ARMS))
    rows = []
    for fold in range(4):
        for e in range(10):
            row = dict(fold=fold, e=e, c=4*e+fold, support='s'+str(fold)+'_'+str(e), query='q'+str(fold)+'_'+str(e))
            manifest['frozen_episodes'].append(dict(row))
            row.update(iu={a: [2, 4] for a in ARMS}, original_iu={a: [1, 4] for a in ARMS},
                       ledger={a: {k: 0 for k in LEDGER} for a in ARMS})
            rows.append(row)
    report = dict(state='COMPLETED', records=rows)
    result = analyze(report, manifest)
    assert result['scores']['iu']['foris_crf']['fold_mean_class_miou'] == 50
    assert result['scores']['original_iu']['foris_crf']['fold_mean_class_miou'] == 25
    assert result['comparisons']['iu']['paired_only vs foris_crf']['exploratory_delta95_pp'] == [0., 0.]
    rows[0]['iu']['paired_only'] = [3, 4]
    rows[0]['ledger']['paired_only']['recovered_fn'] = 1
    changed = analyze(report, manifest)
    assert changed['episode_oracle']['iu']['choices'][0]['arm'] == 'paired_only'
    assert changed['comparisons']['iu']['paired_only vs foris_crf']['delta_pp'] == .625
    assert changed['comparisons']['original_iu']['paired_only vs foris_crf']['delta_pp'] == 0
    # Aggregation must use summed counts, not the mean of episode IoUs.
    extra = [dict(fold=0, c=0, iu={'a': [1, 1]}), dict(fold=0, c=0, iu={'a': [0, 9]})]
    assert class_table(extra, 'a', 'iu')[0][(0, 0)] == 10
    rows[1]['support'] = rows[0]['support']; manifest['frozen_episodes'][1]['support'] = rows[0]['support']
    assert analyze(report, manifest)['independent_class_bootstrap_photo_warning']
    rows[1]['query'] = 'wrong'
    try: analyze(report, manifest)
    except ValueError: pass
    else: raise AssertionError('Photo identity mismatch accepted')
    print(json.dumps(dict(state='CPU_RICE_ANALYSIS_PASSED', class_sumIU=True, resolutions_separate=True,
                         paired_bootstrap=True, shared_photo_warning=True, no_GPU_or_real_task=True)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path); p.add_argument('--manifest', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check: self_check(); return
    if any(v is None for v in (a.report, a.manifest, a.out)):
        p.error('--report, --manifest and --out are required')
    if a.out.exists(): raise ValueError('Preserve existing analysis; use a fresh output')
    result = analyze(json.loads(a.report.read_text()), json.loads(a.manifest.read_text()), digest(a.report), digest(a.manifest))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(state=result['state'], records=40, shared_photo_warning=result['independent_class_bootstrap_photo_warning'])))


if __name__ == '__main__': main()
