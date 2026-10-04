#!/usr/bin/env python3
"""Exploratory replay of saved outputs; no inference, fitting, or new labels.

Question: does per-episode choice among two hosts/OR/AND exhaust disagreement?
Already inspected point values: 69.97 / 70.65 versus pixel oracle 82.08.
Persistence of the gap locates a finer-grained decision opportunity only.
A small gap would instead favour whole-output selection; neither outcome proves
an observable selector, novelty, or a new method. Bootstrap is descriptive after
inspection, not a preregistered confirmation. Choices maximize episode IoU and
are NOT exact maximizers of the reported class-summed mIoU.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    names = ['foris', 'sam', 'OR', 'AND', 'disagreement_oracle',
             'episode_best_two', 'episode_best_four']
    values = []
    for row in rows:
        iu = dict(row['original_iu'])
        ratio = lambda key: iu[key][0] / iu[key][1] if iu[key][1] else 0
        iu['episode_best_two'] = iu[max(names[:2], key=ratio)]
        iu['episode_best_four'] = iu[max(names[:4], key=ratio)]
        values.append([iu[key] for key in names])

    # Exact finite-family class-summed ratio optimization. At optimum r,
    # sum_e max_a (I_ea-r U_ea)=0. This separates across episodes within class.
    classes = defaultdict(list)
    for i, row in enumerate(rows):
        classes[(row['fold'], row['c'])].append(i)
    certificates = {}
    for name, arms in [('class_optimal_two', range(2)), ('class_optimal_four', range(4))]:
        choices = {}
        max_residual = 0.0
        for indices in classes.values():
            ratio = 0.0
            for _ in range(100):
                selected = {i: max(arms, key=lambda a: values[i][a][0]-ratio*values[i][a][1])
                            for i in indices}
                inter = sum(values[i][a][0] for i, a in selected.items())
                union = sum(values[i][a][1] for i, a in selected.items())
                new_ratio = inter / union
                if abs(new_ratio-ratio) < 1e-13:
                    break
                ratio = new_ratio
            else:
                raise RuntimeError('Fractional optimization did not converge')
            residual = sum(max(values[i][a][0]-new_ratio*values[i][a][1] for a in arms)
                           for i in indices)
            max_residual = max(max_residual, abs(residual)/union)
            choices.update(selected)
        assert max_residual < 1e-12
        for i in range(len(rows)):
            values[i].append(values[i][choices[i]])
        names.append(name)
        certificates[name] = dict(normalized_optimality_residual=max_residual,
                                  choice_indices=[choices[i] for i in range(len(rows))])

    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    photos = {}
    for i, row in enumerate(rows):
        for role in ('support', 'query'):
            photo = row[role]
            if photo in photos:
                parents[root(i)] = root(photos[photo])
            else:
                photos[photo] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    groups = list(groups.values())

    def scores(indices):
        totals = {}
        for i in indices:
            row = rows[i]
            key = (row['fold'], row['c'])
            acc = totals.setdefault(key, [[0, 0] for _ in names])
            for a, (inter, union) in enumerate(values[i]):
                acc[a][0] += inter
                acc[a][1] += union
        folds = defaultdict(list)
        for (fold, _), acc in totals.items():
            folds[fold].append([100 * inter / union if union else 0 for inter, union in acc])
        return [sum(sum(row[a] for row in items) / len(items) for items in folds.values())
                / len(folds) for a in range(len(names))]

    point = scores(range(len(rows)))
    expected = [59.783015987111895, 61.09084950950372, 62.82910706740774,
                56.86774457216066, 82.08458241741721]
    assert all(abs(a-b) < 1e-9 for a, b in zip(point, expected)), 'Baseline replay drift'
    contrasts = [('four_minus_two', 6, 5), ('pixel_minus_four', 4, 6),
                 ('two_minus_OR', 5, 2), ('OR_minus_foris', 2, 0),
                 ('optimal_four_minus_optimal_two', 8, 7),
                 ('pixel_minus_optimal_four', 4, 8)]
    samples = {key: [] for key, _, _ in contrasts}
    rng = random.Random(0)
    for _ in range(2000):
        indices = [i for _ in groups for i in groups[rng.randrange(len(groups))]]
        result = scores(indices)
        for key, a, b in contrasts:
            samples[key].append(result[a] - result[b])
    def quantile(items, q):
        items = sorted(items)
        x = (len(items)-1)*q
        low = int(x)
        return items[low] + (x-low)*(items[min(low+1, len(items)-1)]-items[low])
    report = dict(scope='COCO-20i val2014 seed0 oldCONF600; four folds; exploratory GT-assisted replay',
                  episodes=len(rows), metric='mean over folds of class-summed original-resolution IoU',
                  groups=len(groups), draws=2000, bootstrap_seed=0,
                  bootstrap='connected support/query photographs, random.Random resampling',
                  miou=dict(zip(names, point)),
                  contrasts={key: dict(delta=point[a]-point[b],
                                       ci95=[quantile(samples[key], q) for q in (.025, .975)])
                             for key, a, b in contrasts},
                  limitations=['GT selection is not deployable',
                               'episode-best is not the exact class-mIoU-optimal selector',
                               'class-optimal GT choices frozen before descriptive bootstrap, not reoptimized per draw',
                               'no evidence that any legal feature can realize the gap',
                               'bootstrap after inspection; exposed data, no independent confirmation'],
                  input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  script=__file__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report['exact_class_optimum_certificates'] = certificates
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'exact_class_optimum_certificates'}, indent=2))


if __name__ == '__main__':
    main()
