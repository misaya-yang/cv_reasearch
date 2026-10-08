"""Per-class IoU accounting from sealed parent/candidate pixel counts."""
from collections import defaultdict

import numpy as np


def summarize_edits(records, parent, candidate, expected_classes=None):
    groups = defaultdict(lambda: np.zeros(8, dtype=np.int64))
    for row in records:
        base, updated = row['iu'][parent], row['iu'][candidate]
        edits = row['edits'][candidate][parent]
        if len(edits) != 4 or updated != [base[0]+edits[0]-edits[2], base[1]+edits[1]-edits[3]]:
            raise ValueError('Gross edits do not reconstruct candidate I/U')
        groups[(str(row['fold']), str(row['class_id']))] += np.asarray(base+updated+edits, dtype=np.int64)
    classes = defaultdict(set)
    for fold, cls in groups:
        classes[fold].add(cls)
    if expected_classes is not None:
        declared = {str(f):set(map(str, cs)) for f, cs in expected_classes.items()}
        if any(f not in declared or not cs <= declared[f] for f, cs in classes.items()):
            raise ValueError('Unexpected fold/class in edits')
        classes = declared
    per_class, folds = {}, {}
    for fold, ids in sorted(classes.items()):
        if not ids:
            raise ValueError('Empty edit evaluation fold')
        values = []
        for cls in sorted(ids):
            i, u, ci, cu, at, af, dt, df = map(int, groups[(fold, cls)])
            j = i/max(u, 1)
            mode = dict(parent=100*j, add_only=100*(i+at)/max(u+af, 1),
                        delete_only=100*(i-dt)/max(u-df, 1), full=100*ci/max(cu, 1))
            per_class[fold+':'+cls] = dict(miou=mode, gross_edits=[at, af, dt, df],
                addition=dict(profit=at-j*af if at+af else None,
                              purity=at/(at+af) if at+af else None, required_purity=j/(1+j)),
                deletion=dict(profit=j*df-dt if dt+df else None,
                              purity=df/(dt+df) if dt+df else None, required_purity=1/(1+j)))
            values.append(mode)
        folds[fold] = {name:float(np.mean([v[name] for v in values])) for name in values[0]}
    if not folds:
        raise ValueError('No edit records')
    point = {name:float(np.mean([v[name] for v in folds.values()])) for name in next(iter(folds.values()))}
    return dict(miou=point, delta_pp={name:v-point['parent'] for name,v in point.items()},
                per_fold=folds, per_class=per_class,
                interpretation='only-add/only-delete reconstructed per class; no whole-pool purity substitution')
