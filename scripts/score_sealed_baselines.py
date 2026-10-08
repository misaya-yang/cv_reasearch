#!/usr/bin/env python3
"""Paired dual-frame statistics from completed runs with identical frozen inputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from ics.official_data import FOLDS, array_hash, file_hash
from ics.metrics import counts, gross_edits, summarize

IDENTITY = ('dataset', 'fold', 'loader_class_id', 'reference_rgb_hash', 'reference_mask_hash',
            'query_rgb_hash', 'query_mask_hash', 'reference_size_hw', 'query_size_hw',
            'reference_crop', 'query_crop', 'reference_object_id', 'query_object_id')
PAPER = dict(coco=60.9, lvis=42.8, pascal_part=55.8, paco_part=42.3, suim=59.1)


def choose_metric(original, cli, target, tolerance=.5):
    original_matches, cli_matches = abs(original-target) <= tolerance, abs(cli-target) <= tolerance
    return dict(primary='original' if original_matches else 'cli',
                published_target=target, original_delta_pp=original-target, cli_delta_pp=cli-target,
                reproduction_matches=original_matches or cli_matches,
                final_protocol_claim_on_hold=not (original_matches or cli_matches))


def open_run(path):
    seal = json.loads((path/'sealed.json').read_text())
    if seal['state'] != 'ALL_PREDICTIONS_SEALED':
        raise ValueError('Every input run must finish before labels are opened')
    for file, key in [('manifest.json', 'manifest_sha256'), ('config.json', 'config_sha256'),
                      ('inference.jsonl', 'inference_index_sha256')]:
        if file_hash(path/file) != seal[key]:
            raise ValueError('Sealed run input changed: '+str(path/file))
    rows = json.loads((path/'manifest.json').read_text())
    index = {r['episode_id']: r for r in map(json.loads, (path/'inference.jsonl').read_text().splitlines())}
    if len(rows) != seal['n'] or len(index) != len(rows) or set(index) != {r['episode_id'] for r in rows}:
        raise ValueError('Incomplete or duplicate run identity')
    return dict(path=path, seal=seal, rows={r['episode_id']:r for r in rows}, index=index,
                config=json.loads((path/'config.json').read_text()))


def score_runs(paths, out, unit='query_photo', metric_policy=None):
    out.mkdir(parents=True, exist_ok=True)
    runs = [open_run(path.resolve()) for path in paths]
    first = runs[0]
    for run in runs[1:]:
        if set(first['rows']) != set(run['rows']) or first['config']['split_role'] != run['config']['split_role']:
            raise ValueError('Paired runs need the same complete split/cohort')
        for key, row in first['rows'].items():
            if any(row.get(k) != run['rows'][key].get(k) for k in IDENTITY):
                raise ValueError('Paired frozen inputs differ: '+key)
    arms = [arm for run in runs for arm in run['config']['arms']]
    if len(set(arms)) != len(arms):
        raise ValueError('Ambiguous duplicated method names')
    baselines = tuple(a for a in ('foris.crf', 'rcg.fine', 'rcg', 'insid3') if a in arms)
    if not baselines:
        baselines = (arms[0],)
    config = dict(runs={str(r['path']):file_hash(r['path']/'sealed.json') for r in runs},
                  scorer_sha256=file_hash(Path(__file__)), bootstrap_unit=unit,
                  metric_policy=metric_policy, repetitions=10000)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
        raise ValueError('Output belongs to different predictions/scoring policy')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    if (out/'report.json').exists():
        return json.loads((out/'report.json').read_text())
    items = []
    for key, row in first['rows'].items():
        with Image.open(row['query_mask_path']) as image:
            raw = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
        if array_hash(raw) != row['query_mask_hash']:
            raise ValueError('Frozen annotation changed')
        tensor = torch.from_numpy(raw)[None, None].float()
        frames = {'original': tuple(row['query_size_hw']), 'cli': (1024, 1024)}
        # Raw mask goes directly to each frame. Source geometry mismatches are
        # retained exactly as in both upstream public evaluators.
        truth = {f: F.interpolate(tensor, shape, mode='nearest')[0, 0].numpy() > .5
                 for f, shape in frames.items()}
        predictions = {f:{} for f in frames}
        for run in runs:
            record = run['index'][key]
            file = run['path']/'predictions'/record['prediction_file']
            if file_hash(file) != record['prediction_sha256']:
                raise ValueError('Sealed prediction changed')
            with np.load(file) as saved:
                if tuple(saved['original_hw']) != frames['original']:
                    raise ValueError('Prediction geometry differs from frozen RGB')
                for frame, shape in frames.items():
                    for arm in run['config']['arms']:
                        predictions[frame][arm] = np.unpackbits(saved[frame+'/'+arm], count=int(np.prod(shape))).reshape(shape).astype(bool)
        item = dict(episode_id=key, dataset=row['dataset'], fold=row['fold'],
                    class_id=row['loader_class_id'], query_photo_id=row.get('query_photo_id'), frames={})
        for frame in frames:
            item['frames'][frame] = dict(truth_pixels=int(truth[frame].sum()),
                iu={arm:counts(mask, truth[frame]) for arm, mask in predictions[frame].items()},
                edits={arm:{b:gross_edits(mask, predictions[frame][b], truth[frame]) for b in baselines}
                       for arm, mask in predictions[frame].items()})
        items.append(item)
    report = dict(split_role=first['config']['split_role'], n=len(items), arms=arms, datasets={},
                  edit_order=['add_TP', 'add_FP', 'delete_TP', 'delete_FP'],
                  label_frames='raw annotation nearest-resized directly to each prediction frame',
                  official_goal_reached=False)
    grouped = defaultdict(list)
    for item in items:
        grouped[item['dataset']].append(item)
    prepared = first['config'].get('prepared_protocol') or {}
    for name, records in grouped.items():
        expected = {k.split('/')[1]:v['expected_class_ids'] for k,v in prepared.get('folds', {}).items() if k.startswith(name+'/')} or None
        summaries = {}
        for frame in ('original', 'cli'):
            metric_rows = [dict(r, iu=r['frames'][frame]['iu']) for r in records]
            summaries[frame] = summarize(metric_rows, baselines=baselines, repetitions=10000,
                                        expected_classes=expected, unit=unit)
            if unit == 'query_photo':
                summaries[frame]['episode_bootstrap_appendix'] = summarize(
                    metric_rows, baselines=baselines, repetitions=10000, expected_classes=expected)['paired']
            area = sum(r['frames'][frame]['truth_pixels'] for r in records)
            summaries[frame]['gross_edits'] = {}
            for arm in arms:
                summaries[frame]['gross_edits'][arm] = {}
                for base in baselines:
                    pixels = np.sum([r['frames'][frame]['edits'][arm][base] for r in records], axis=0).tolist()
                    summaries[frame]['gross_edits'][arm][base] = dict(pixels=pixels,
                        percent_of_gt_area=[100*p/area if area else None for p in pixels])
        policy = (metric_policy or {}).get(name, dict(primary='cli', reason='CLI development fallback pending complete official FoRIS reproduction', final_protocol_claim_on_hold=True))
        report['datasets'][name] = dict(primary_policy=policy, primary=summaries[policy['primary']], frames=summaries)
    (out/'episode_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
    temporary = out/'report.tmp';temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(out/'report.json')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--unit', choices=['query_photo', 'episode'], default='query_photo')
    p.add_argument('--metric-policy', type=Path)
    p.add_argument('--freeze-official-metric-policy', action='store_true')
    a = p.parse_args()
    policy = json.loads(a.metric_policy.read_text()) if a.metric_policy else None
    report = score_runs(a.runs, a.out, a.unit, policy)
    if a.freeze_official_metric_policy:
        if report['split_role'] != 'official' or 'foris.crf' not in report['arms']:
            raise ValueError('Metric choice requires complete official FoRIS reproduction')
        original_run = open_run(a.runs[0].resolve())
        prepared = original_run['config'].get('prepared_protocol') or {}
        if prepared.get('state') != 'OFFICIAL_MANIFEST_FROZEN':
            raise ValueError('Require an original complete official preparation')
        for name in report['datasets']:
            metadata = {int(k.split('/')[1]):v for k,v in prepared.get('folds', {}).items() if k.startswith(name+'/')}
            if set(metadata) != set(FOLDS[name]):
                raise ValueError('Cannot choose a dataset metric from partial folds')
            for fold, info in metadata.items():
                n = sum(r['dataset'] == name and r['fold'] == fold for r in original_run['rows'].values())
                if n != info['official_length'] or n != info['selected_length']:
                    raise ValueError('Cannot choose a dataset metric from partial fold length')
        frozen = {n:choose_metric(d['frames']['original']['miou']['foris.crf'],
                                  d['frames']['cli']['miou']['foris.crf'], PAPER[n])
                  for n,d in report['datasets'].items()}
        (a.out/'metric_policy.json').write_text(json.dumps(frozen, indent=2)+'\n')
    print(json.dumps({n:d['primary']['miou'] for n,d in report['datasets'].items()}), flush=True)


if __name__ == '__main__':
    main()
