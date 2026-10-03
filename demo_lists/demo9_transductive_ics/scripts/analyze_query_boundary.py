"""Paired original-resolution analysis of the finite stopping pilot, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ('foris_public', 'midpoint_replay', 'scalar_contrast', 'dino_boundary', 'rgb_boundary')


def metric(rows, arm):
    classes = {}
    for row in rows:
        i, u = row['original_iu'][arm]
        if type(i) is not int or type(u) is not int or not 0 <= i <= u or u <= 0:
            raise ValueError('Invalid original integer I/U')
        pair = classes.setdefault(row['c'], [0, 0])
        pair[0] += i; pair[1] += u
    return 100*np.mean([i/u for i, u in classes.values()])


def groups(rows):
    parents = list(range(len(rows)))
    def find(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for role in ('support', 'query'):
            uid = int(Path(row[role]).stem.rsplit('_', 1)[-1])
            if uid in seen: parents[find(i)] = find(seen[uid])
            else: seen[uid] = i
    result = {}
    for i in range(len(rows)): result.setdefault(find(i), []).append(i)
    return list(result.values())


def analyze(report):
    if report.get('schema') != 'query_boundary_experiment_v1' or report.get('state') != 'COMPLETED':
        raise ValueError('Completed finite stopping report required')
    rows = report['records']
    if len(rows) != report.get('expected_rows',40) or sorted({r['fold'] for r in rows}) != report.get('expected_folds',[0,1,2,3]):
        raise ValueError('Frozen row count/folds required')
    if any(set(r['original_iu']) != set(ARMS) or not r['query_GT_after_all_predictions']
           or not r['midpoint_replay_exact'] for r in rows):
        raise ValueError('Matched source/control/freeze boundary absent')
    scores = {arm: float(metric(rows, arm)) for arm in ARMS}
    components = groups(rows)
    rng = np.random.default_rng(31027)
    boot = []
    for _ in range(2000):
        indices = [i for block in rng.integers(len(components), size=len(components)) for i in components[block]]
        sample = [rows[i] for i in indices]
        boot.append([metric(sample, 'dino_boundary')-metric(sample, arm)
                     for arm in ('foris_public', 'scalar_contrast', 'rgb_boundary')])
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    comparisons = {arm: dict(delta_pp=scores['dino_boundary']-scores[arm],
                             exploratory_CI95_pp=intervals[:, k].tolist())
                   for k, arm in enumerate(('foris_public', 'scalar_contrast', 'rgb_boundary'))}
    return dict(state='CPU_FROZEN_OUTPUT_ANALYSIS', experiment='query_affinity_stopping',
        dataset='official COCO20i', seed=0, samples=len(rows),
        per_fold={str(f):sum(r['fold']==f for r in rows) for f in sorted({r['fold'] for r in rows})},
        original_class_miou=scores, comparisons=comparisons,
        by_fold={str(f): {arm: float(metric([r for r in rows if r['fold'] == f], arm))
                         for arm in ARMS} for f in sorted({r['fold'] for r in rows})},
        paired_resampling='whole support/query photo-connected components; macro average of classes present per draw',
        bootstrap_seed=31027, bootstrap_replicates=2000, photo_groups=components,
        same_continuous_upsampling_and_source_RGB_CRF=True,
        independent_confirmation=False, method_score_claim=False,
        further_expansion_signal=(comparisons['foris_public']['delta_pp'] >= 2
            and comparisons['scalar_contrast']['delta_pp'] > 0 and comparisons['rgb_boundary']['delta_pp'] > 0),
        qualification='Exploratory reusedDEV; test-fold subset cannot establish solid research or absence of information.')


def cpu_check():
    rows = []
    for f in range(4):
        for k in range(10):
            uid = 100+20*f+2*k
            rows.append(dict(fold=f, c=f+4*k, e=k,
                support=f'val2014/COCO_val2014_{uid:012d}.jpg', query=f'val2014/COCO_val2014_{uid+1:012d}.jpg',
                original_iu={arm:[3,5] for arm in ARMS},
                midpoint_replay_exact=True, query_GT_after_all_predictions=True))
    result = analyze(dict(schema='query_boundary_experiment_v1', state='COMPLETED', records=rows))
    assert all(c['delta_pp'] == 0 and c['exploratory_CI95_pp'] == [0,0] for c in result['comparisons'].values())
    rows[1]['support'] = rows[0]['query']
    assert any(len(g) == 2 for g in groups(rows))
    return dict(state='CPU_QUERY_BOUNDARY_ANALYSIS_CHECKED', matched_zero_exact=True,
                cross_role_photo_group=True, real_task_gain_measured=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    if args.out.exists(): raise ValueError('Preserve prior result')
    if args.cpu_check: value = cpu_check()
    else: value = analyze(json.loads(args.report.read_text()))
    value['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream: json.dump(value, stream)
    print(json.dumps(value))
