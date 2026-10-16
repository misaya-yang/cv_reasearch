"""Read sealed Fundus accounting only. No model, image, feature or GT loading.

Reconstruct confusion counts from existing I/U/predicted/GT counts. The older
layer report omits predicted area; count its already sealed prediction bits and
recover GT=U-predicted+I, cross-checking every arm. No new masks are generated.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent/'cv_data/a'
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, jsonl=False):
    path = Path(path)
    SOURCES[str(path)] = sha(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream] if jsonl else json.load(stream)


def counts(iu, gt):
    i, u, p = map(int, iu)
    assert 0 <= i <= min(p, gt) and u == p+gt-i
    return dict(TP=i, FP=p-i, FN=gt-i, GT=gt, predicted=p, I=i, U=u)


def aggregate(rows, arm):
    total = {key: sum(r[arm][key] for r in rows) for key in rows[0][arm]}
    tp, fp, fn, gt, pred = (total[k] for k in ('TP', 'FP', 'FN', 'GT', 'predicted'))
    return dict(n=len(rows), counts=total, pooled_IoU=100*tp/(gt+fp),
                precision=100*tp/max(pred, 1), recall=100*tp/gt,
                FP_per_GT=fp/gt, FN_per_GT=fn/gt, predicted_per_GT=pred/gt)


def cohort(rows, arms):
    assert len({r['index'] for r in rows}) == len(rows)
    out = dict(n=len(rows), query_photos=len({r['photo'] for r in rows}),
               arms={a: aggregate(rows, a) for a in arms})
    if 'SH' in arms and 'FoRIS' in arms:
        aa, bb = out['arms']['SH'], out['arms']['FoRIS']
        out['SH_minus_FoRIS'] = dict(
            IoU_pp=aa['pooled_IoU']-bb['pooled_IoU'],
            precision_pp=aa['precision']-bb['precision'], recall_pp=aa['recall']-bb['recall'],
            FP_reduction_percent=100*(1-aa['counts']['FP']/bb['counts']['FP']),
            fewer_FP_tasks=sum(r['SH']['FP'] < r['FoRIS']['FP'] for r in rows),
            fewer_TP_tasks=sum(r['SH']['TP'] < r['FoRIS']['TP'] for r in rows),
            both_fewer_FP_and_TP=sum(r['SH']['FP'] < r['FoRIS']['FP'] and r['SH']['TP'] < r['FoRIS']['TP'] for r in rows),
            IoU_wins=sum(r['SH']['I']/max(r['SH']['U'], 1) > r['FoRIS']['I']/max(r['FoRIS']['U'], 1) for r in rows))
    return out


def main():
    official = read(ROOT/'evidence/research/w30_E2_iu.jsonl.gz', True)
    official = [r for r in official if r['dataset'] == 'fundus']
    old = {r['index']: r for r in read(ASSETS/'x1013_official/scores/fundus.jsonl.gz', True)}
    rows = []
    for r in official:
        o = old[r['index']]
        assert (r['photo'], r['gt']) == (o['photo'], o['gt'])
        assert r['SH'] == r['SH+'] == o['main']
        rows.append(dict(index=r['index'], photo=r['photo'],
                         SH=counts(r['SH+'], r['gt']), FoRIS=counts(o['FoRIS'], r['gt'])))
    result = dict(scope='Existing sealed predictions and counts only; descriptive reanalysis, no new evaluation run',
                  cohorts=dict(FIVES_official=cohort(rows, ('SH', 'FoRIS'))))
    fresh = read(ROOT/'evidence/research/w54_fundus_fresh.jsonl.gz', True)
    for pool, name in (('A', 'Fundus_AVSeg'), ('B', 'FIVES_train')):
        rows = []
        for r in fresh:
            if r['pool'] != pool:
                continue
            assert r['SH'] == r['SH+'] == r['MFU'] == r['U']
            rows.append(dict(index=r['index'], photo=r['query'],
                             SH=counts(r['SH+'], r['gt']), FoRIS=counts(r['FoRIS'], r['gt'])))
        result['cohorts'][name] = cohort(rows, ('SH', 'FoRIS'))

    base = ASSETS/'fundus_layer_resolution50_20261013'
    report = read(base/'report.json')
    complete = read(base/'predictions_complete.json')
    indexed = {arm: {r['index']: r for r in records} for arm, records in report['records']['CLI1024'].items()}
    rows_per_layer = {layer: [] for layer in (19, 20, 23)}
    for r in complete['records']:
        assert r['size'] == 1024
        path = Path(r['prediction_path']); assert sha(path) == r['prediction_sha256']
        SOURCES[str(path)] = r['prediction_sha256']
        inferred_gt = set()
        with np.load(path, allow_pickle=False) as z:
            for layer in (19, 20, 23):
                row = dict(index=r['index'], photo=r['query_photo'])
                for method, name in (('T', 'SH'), ('FoRIS', 'FoRIS')):
                    record = indexed[f'{method}_1024_L{layer}'][r['index']]
                    assert record['photo'] == row['photo']
                    predicted = int(np.unpackbits(z[f'{method}_L{layer}'])[:1024*1024].sum())
                    gt = record['U']-predicted+record['I']; inferred_gt.add(gt)
                    row[name] = counts([record['I'], record['U'], predicted], gt)
                    if method == 'T':
                        assert np.array_equal(z[f'T_L{layer}'], z[f'U_L{layer}'])
                rows_per_layer[layer].append(row)
        assert len(inferred_gt) == 1, ('Inconsistent archived GT count', r['index'])
    result['matched_layers'] = {str(layer): cohort(rows, ('SH', 'FoRIS')) for layer, rows in rows_per_layer.items()}
    result['matched_layer_note'] = '50 existing development pairs / 47 query photos. SH here is the shared T fallback. Input and metric 1024; 2048 input never run.'

    # Natural outputs only. Historical grid oracles are not reused.
    r1 = read(ASSETS/'x1012_R1/report.json')
    arms = ('FoRIS', 'MEAN', 'T', 'IND0', 'raw_G0', 'raw_BG', 'raw_UN', 'raw_PERM')
    rows = []
    for r in r1['records']:
        if r['dataset'] != 'fundus':
            continue
        row = dict(index=r['index'], photo=r['query_photo'])
        for a in arms:
            m = r['metrics'][a]
            row[a] = counts([m['I'], m['U'], m['predicted']], m['GT'])
            assert all(row[a][k] == m[k] for k in ('TP', 'FP', 'FN'))
        rows.append(row)
    result['old_single_field_interventions'] = cohort(rows, arms)
    result['old_single_field_note'] = 'Read actual source before interpreting IND0/graph as causal interventions; these are not FoRIS-internal ablations.'
    # The user's abstention analogy motivates checking an existing strict oracle,
    # not generating/tuning a new threshold or opening query labels again.
    t1 = [r for r in read(ASSETS/'x1013_T1/iu.jsonl.gz', True) if r['dataset'] == 'fundus']
    sh = {str(r['index']): r for r in read(ROOT/'evidence/research/w23_simple_pipeline_S.jsonl.gz', True) if r['dataset'] == 'fundus'}
    j = lambda x: x[0]/max(x[1], 1)
    assert all(r['T'] == sh[r['index']]['SH'] == sh[r['index']]['SH+'] and r['gt'] == sh[r['index']]['gt'] for r in t1)
    assert all(j(r['oracle_foris']) >= j(r['foris_noCRF']) for r in t1)
    bindings = {str(r['index']): r for split in ('explore', 'confirm')
                for r in read(ASSETS/f'x1012_P0/{split}.json') if r['dataset'] == 'fundus'}
    dtypes = Counter()
    for r in t1:
        item = bindings[r['index']]['score64']; path = Path(item['path'])
        SOURCES[str(path)] = sha(path)
        if item.get('sha256'):
            assert item['sha256'] == SOURCES[str(path)]
        with np.load(path, allow_pickle=False) as z:
            dtypes[str(z[item['key']].dtype)] += 1
    result['existing_threshold_oracle'] = dict(
        n=len(t1), query_photos=len({r['photo'] for r in t1}),
        current_SH_integer_identity=len(t1), all_fallback=all(r['rho'] > .9 for r in t1),
        archived_field_dtypes=dict(dtypes),
        episode_mean_IoU={a: 100*sum(j(r[a]) for r in t1)/len(t1)
                          for a in ('foris', 'foris_noCRF', 'T', 'oracle_foris', 'oracle_candidate')},
        pooled_IoU={a: 100*sum(r[a][0] for r in t1)/sum(r[a][1] for r in t1)
                    for a in ('foris', 'foris_noCRF', 'T', 'oracle_foris', 'oracle_candidate')},
        natural_SH_exceeds_FoRIS_oracle=sum(j(r['T']) > j(r['oracle_foris']) for r in t1),
        limitations='Cached score field, bilinear 1024 readout, per-episode strict threshold oracle before CRF. Not a pooled-IoU upper bound, full PR dominance, or bound over all methods/layers/settings.')
    result['execution'] = dict(new_predictions=0, new_encoder_calls=0, feature_reads=0,
                               query_GT_file_reads=0, model_parameter_updates=0,
                               existing_layer_prediction_files=50, mps=0)
    result['sources_sha256'] = SOURCES
    result['script_sha256'] = sha(Path(__file__))
    target = ROOT/'evidence/research/w69_fundus_archive_accounting.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for name, value in result['cohorts'].items():
        print(name, json.dumps(value, ensure_ascii=False))
    print('matched layers', json.dumps({k: v['SH_minus_FoRIS'] for k, v in result['matched_layers'].items()}))
    print('old interventions', json.dumps({a: {k: v[k] for k in ('pooled_IoU', 'precision', 'recall', 'FP_per_GT')} for a, v in result['old_single_field_interventions']['arms'].items()}))
    print('existing oracle', json.dumps(result['existing_threshold_oracle']))
    print('Saved', target)


if __name__ == '__main__':
    main()
