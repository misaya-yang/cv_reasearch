"""Class-pooled mIoU and paired episode intervals conditional on the image pool.

Resample within (fold, class), preserving every official draw, including
repeats. This does not estimate uncertainty over unseen photographs.
"""
from collections import defaultdict
import numpy as np


def counts(prediction, truth, ignore=None):
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    if p.shape != t.shape:
        raise ValueError('Prediction/GT geometry differs')
    valid = np.ones(t.shape, bool) if ignore is None else ~np.asarray(ignore, bool)
    if valid.shape != t.shape:
        raise ValueError('Ignore geometry differs')
    return [int((p & t & valid).sum()), int(((p | t) & valid).sum())]


def gross_edits(prediction, baseline, truth, ignore=None):
    p, b, t = (np.asarray(x, bool) for x in (prediction, baseline, truth))
    if p.shape != t.shape or b.shape != t.shape:
        raise ValueError('Edit geometry differs')
    valid = np.ones(t.shape, bool) if ignore is None else ~np.asarray(ignore, bool)
    if valid.shape != t.shape:
        raise ValueError('Ignore geometry differs')
    add, delete = p & ~b & valid, b & ~p & valid
    return [int((add & t).sum()), int((add & ~t).sum()),
            int((delete & t).sum()), int((delete & ~t).sum())]


def summarize(records, baselines=('foris.crf', 'rcg'), repetitions=100000,
              seed=0, expected_classes=None, batch_size=1000, unit='episode'):
    """Records: fold, class_id, iu={arm: [intersection, union]}.

    All arms use exactly the same paired resampling indices. Report both the
    official mean of fold mIoUs and the alternative all-class mean explicitly.
    Expected classes, when supplied, are {fold: [class_id, ...]}; absent official
    classes retain their zero in the metric and are reported as missing.
    query_photo resamples pooled photo groups within each (fold,class), keeping
    every reference/crop/repeated draw from that photo together. Point estimates
    always retain the original episode weights. The episode default preserves
    the archived exploratory protocol.
    """
    if not records or repetitions < 1 or batch_size < 1:
        raise ValueError('Require records and positive bootstrap sizes')
    if unit not in ('episode', 'query_photo'):
        raise ValueError('Unknown bootstrap unit')
    if unit == 'query_photo' and any(r.get('query_photo_id') is None for r in records):
        raise ValueError('Photo-group intervals require original query photo IDs')
    arms = tuple(sorted(records[0]['iu']))
    if not arms or any(tuple(sorted(r['iu'])) != arms for r in records):
        raise ValueError('Every paired record must contain exactly the same arms')
    if any(b not in arms for b in baselines):
        raise ValueError('Missing baseline arm')
    groups = defaultdict(list)
    for r in records:
        groups[(str(r['fold']), str(r['class_id']))].append(r)
    folds = sorted({f for f, _ in groups})
    classes = {f: sorted(c for ff, c in groups if ff == f) for f in folds}
    missing = {}
    if expected_classes is not None:
        expected = {str(f): sorted(map(str, cs)) for f, cs in expected_classes.items()}
        if set(folds) - set(expected):
            raise ValueError('Unexpected fold')
        for f, cs in expected.items():
            if set(classes.get(f, [])) - set(cs):
                raise ValueError('Unexpected class')
            missing[f] = sorted(set(cs) - set(classes.get(f, [])))
        folds, classes = sorted(expected), expected
    if any(not classes[f] for f in folds):
        raise ValueError('Empty evaluation fold')
    arrays = {}
    point = {f: np.zeros(len(arms)) for f in folds}
    class_points, singleton, photo_singleton = {}, 0, 0
    for f in folds:
        for c in classes[f]:
            g = groups.get((f, c), [])
            x = np.asarray([[r['iu'][a] for a in arms] for r in g], dtype=np.float64)
            if not g:
                pooled = np.zeros((len(arms), 2))
            else:
                if x.shape != (len(g), len(arms), 2) or not np.isfinite(x).all():
                    raise ValueError('Invalid I/U')
                if (x < 0).any() or (x[:, :, 0] > x[:, :, 1]).any():
                    raise ValueError('Invalid intersection/union bounds')
                pooled = x.sum(axis=0)
                singleton += len(g) == 1
                if unit == 'query_photo':
                    photos = defaultdict(list)
                    for row, value in zip(g, x):
                        photos[str(row['query_photo_id'])].append(value)
                    x = np.stack([np.sum(photos[p], axis=0) for p in sorted(photos)])
                    photo_singleton += len(x) == 1
                arrays[(f, c)] = x
            value = 100 * pooled[:, 0] / np.maximum(pooled[:, 1], 1)
            class_points[f+':'+c] = dict(zip(arms, value.tolist()))
            point[f] += value / len(classes[f])
    observed = np.mean(list(point.values()), axis=0)
    draws = np.zeros((repetitions, len(arms)), dtype=np.float64)
    rng = np.random.RandomState(seed)  # independent of official loader RNG
    for (f, c), x in sorted(arrays.items()):
        n = len(x)
        weight = 100 / (len(folds) * len(classes[f]))
        for start in range(0, repetitions, batch_size):
            end = min(start + batch_size, repetitions)
            # Multinomial counts are exactly bootstrap-with-replacement draws;
            # the same count matrix applies to every arm and both I/U columns.
            multiplicity = rng.multinomial(n, np.full(n, 1/n), size=end-start)
            pooled = (multiplicity @ x.reshape(n, -1)).reshape(end-start, len(arms), 2)
            draws[start:end] += weight * pooled[:, :, 0] / np.maximum(pooled[:, :, 1], 1)
    paired = {}
    for base in baselines:
        b = arms.index(base)
        paired[base] = {a: dict(delta_pp=float(observed[i]-observed[b]),
                                ci95_pp=np.quantile(draws[:, i]-draws[:, b], [.025, .975]).tolist())
                        for i, a in enumerate(arms)}
    all_class = np.mean([[v[a] for a in arms] for v in class_points.values()], axis=0)
    return dict(n=len(records), folds=folds, class_count=sum(map(len, classes.values())),
                query_photo_count=(len({str(r['query_photo_id']) for r in records})
                                   if all(r.get('query_photo_id') is not None for r in records) else None),
                singleton_photo_strata=photo_singleton if unit == 'query_photo' else None,
                fold_class_counts={f: len(classes[f]) for f in folds},
                missing_classes=missing, singleton_strata=int(singleton),
                miou=dict(zip(arms, observed.tolist())),
                per_fold={f: dict(zip(arms, v.tolist())) for f, v in point.items()},
                all_class_miou=dict(zip(arms, all_class.tolist())), per_class=class_points,
                paired=paired, statistic='per-class pooled I/U -> class mean -> fold mean',
                bootstrap=dict(unit=('query photo within fold and class' if unit == 'query_photo'
                                     else 'episode within fold and class'), repetitions=repetitions,
                               seed=seed, rng='RandomState', repeats='retained with draw weight',
                               scope='conditional on the given image pool; not unseen-photo uncertainty'))
