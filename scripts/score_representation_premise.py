#!/usr/bin/env python3
"""Select once on query-photo half 0, verify only that choice on half 1."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from ics.official_data import array_hash, file_hash


def photo_key(row):
    stem = Path(str(row['query_photo_id'])).stem
    if row['dataset'] in ('coco', 'lvis', 'paco_part'):
        match = re.search(r'(\d{12})$', stem)
        if match:
            return 'coco/'+match.group(1)
    return row['dataset']+'/'+stem


def photo_half(row):
    # Hash assignment is independent of GT and stable when another dataset is
    # added. The same underlying COCO photo keeps its half across part crops.
    return int(hashlib.sha256(photo_key(row).encode()).digest()[0] >= 128)


def macro(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if field in row['auc']:
            groups[(row['fold'], row['class_id'])].append(row['auc'][field])
    folds = defaultdict(list)
    for (fold, cls), values in groups.items():
        folds[fold].append(float(np.mean(values)))
    value = float(np.mean([np.mean(v) for v in folds.values()])) if folds else None
    return dict(auc=value, valid_episodes=sum(len(v) for v in groups.values()),
                valid_classes=len(groups), valid_folds=len(folds))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    seal = json.loads((a.run/'sealed.json').read_text())
    if seal['state'] != 'ALL_PREDICTIONS_SEALED':
        raise ValueError('All predictions must be sealed before premise query labels are read')
    rows = json.loads((a.run/'manifest.json').read_text())
    config = json.loads((a.run/'config.json').read_text())
    if config['split_role'] != 'dev' or config['prepared_protocol'].get('seed') != 1:
        raise ValueError('Prerequisite selection requires the fresh seed1 development split')
    if file_hash(a.run/'manifest.json') != seal['manifest_sha256']:
        raise ValueError('Sealed manifest changed')
    index = {r['episode_id']: r for r in map(json.loads, (a.run/'inference.jsonl').read_text().splitlines())}
    if set(index) != {r['episode_id'] for r in rows}:
        raise ValueError('Require the complete original episode cohort')
    names = ('coco', 'pascal_part', 'paco_part')
    if set(r['dataset'] for r in rows) != set(names):
        raise ValueError('Representation selection requires exactly the three prerequisite datasets')
    a.out.mkdir(parents=True, exist_ok=True)
    score_config = dict(run=str(a.run.resolve()), prediction_seal_sha256=file_hash(a.run/'sealed.json'),
                        scorer_sha256=file_hash(Path(__file__)), manifest_sha256=seal['manifest_sha256'])
    saved_config = a.out/'config.json'
    if saved_config.exists() and json.loads(saved_config.read_text()) != score_config:
        raise ValueError('Premise output belongs to different code or predictions')
    saved_config.write_text(json.dumps(score_config, indent=2)+'\n')
    if (a.out/'decision.json').exists():
        print('Existing frozen selection retained; no second selection', flush=True)
        return
    all_fields = sorted(index[rows[0]['episode_id']]['representation_validity'])
    baseline = 'O/24/raw'
    candidates = [k for k in all_fields if not k.startswith('C2concat/')]
    tables, records = [], {0: [], 1: []}

    def evaluate(row, fields):
        record = index[row['episode_id']]
        paths = [('predictions', 'prediction_file', 'prediction_sha256'),
                 ('representations', 'representation_file', 'representation_sha256'),
                 ('fields', 'prediction_file', 'fields_sha256')]
        for directory, key, digest in paths:
            if file_hash(a.run/directory/record[key]) != record[digest]:
                raise ValueError('Sealed field/prediction changed')
        with Image.open(row['query_mask_path']) as image:
            truth = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
        if array_hash(truth) != row['query_mask_hash']:
            raise ValueError('Frozen query mask changed')
        truth = F.interpolate(torch.from_numpy(truth)[None, None].float(),
                              (1024, 1024), mode='nearest')
        target = (F.interpolate(truth, (64, 64), mode='area')[0, 0].numpy() >= .5).ravel()
        with np.load(a.run/'predictions'/record['prediction_file']) as saved:
            parent = np.unpackbits(saved['cli/rcg.fine'], count=1024**2).reshape(1024, 1024)
        circle = (F.interpolate(torch.from_numpy(parent)[None, None].float(),
                                (64, 64), mode='area')[0, 0].numpy() >= .5).ravel()
        result = dict(episode_id=row['episode_id'], dataset=row['dataset'],
                      fold=row['fold'], class_id=row['loader_class_id'],
                      photo_key=photo_key(row), half=photo_half(row), auc={})
        if len(np.unique(target[circle])) != 2:
            return result  # Per-episode no positives or negatives is N/A.
        with np.load(a.run/'representations'/record['representation_file']) as saved:
            for key in fields:
                if record['representation_validity'][key]:
                    result['auc'][key] = float(roc_auc_score(target[circle], saved[key][circle]))
        with np.load(a.run/'fields'/record['prediction_file']) as saved:
            result['auc']['FoRIS/score'] = float(roc_auc_score(target[circle], saved['score'].ravel()[circle]))
        return result

    def table(half, fields):
        gains = {}
        for field in fields:
            gains[field] = []
            for name in names:
                # Paired valid episodes give candidate and control the same
                # classes/weights, including when a reference mean degenerates.
                paired = [r for r in records[half] if r['dataset'] == name and
                          baseline in r['auc'] and field in r['auc']]
                value, control = macro(paired, field), macro(paired, baseline)
                delta = value['auc']-control['auc'] if value['auc'] is not None else None
                tables.append(dict(half=half, dataset=name, representation=field,
                                   **value, delta_auc=delta))
                gains[field].append(delta)
        return gains

    for row in rows:
        if photo_half(row) == 0:
            records[0].append(evaluate(row, all_fields))
    gains = table(0, all_fields+['FoRIS/score'])

    def choose(fields):
        valid = [f for f in fields if all(v is not None for v in gains[f])]
        return min(valid, key=lambda f: (-min(gains[f]), f)) if valid else None

    chosen = choose(candidates)
    context = choose(['C2concat/raw', 'C2concat/deb'])
    selected = [f for f in (chosen, context) if f is not None]
    for row in rows:
        if photo_half(row) == 1:
            records[1].append(evaluate(row, sorted(set(selected+[baseline]))))
    confirmation = table(1, selected+[baseline, 'FoRIS/score'])
    decision = dict(state='COMPLETE', role='development prerequisite, not a segmentation gain result',
                    n=len(rows), photo_half_rule='canonical photo key SHA256 first byte <128 selects half0',
                    circle='RCG+fine CLI final mask area-projected to64 with >=0.5',
                    labels='raw GT nearest to1024, area to64 with >=0.5; per-episode single-label circles are N/A',
                    aggregation='episode AUC -> class mean -> fold mean; paired valid episodes',
                    candidates={})
    for label, choice in [('C1', chosen), ('C2', context)]:
        passed = choice is not None and all(v >= .02 for v in gains[choice]+confirmation[choice])
        decision['candidates'][label] = dict(representation=choice, passed=passed,
                                           selection_delta=None if choice is None else gains[choice],
                                           verification_delta=None if choice is None else confirmation[choice])
    with (a.out/'representation.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(tables[0]))
        writer.writeheader(); writer.writerows(tables)
    for half in (0, 1):
        (a.out/f'half{half}_episode_auc.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records[half]))
    (a.out/'decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
