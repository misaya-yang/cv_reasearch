"""E1: fixed K64 linear descriptor-capacity diagnostic from sealed A6 assets.

This is a supervised diagnostic, never a legal training-free segmentation result.
It does not call the old A6 entry points, change their gate, regenerate regions,
open RGB/raw feature payloads, or read heldout target counts before predictions seal.
All output is external to the repository. Run only after the E1 plan is saved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / 'cv_data'
SOURCE = ASSETS / 'a/x1012_A6'
DEFAULT_OUT = ASSETS / 'a/x1012s_E1'
SEED = 20261012
GEOMETRY = 'kmeans64'
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import a6_helpers as ah
import plan_common as pc
import plan_report as pr


def sha(path):
    return pc.sha(path)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    pc.write(path, value)


def checked(path, expected):
    if sha(path) != expected:
        raise RuntimeError('Input changed: ' + str(path))


def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **values)
    temp.replace(path)


def photo_groups(tasks):
    """Join identity aliases of one photo, never join a reference/query pair.

    Original paths/source hashes group all crops. RGB hashes catch identical
    copied views; COCO image numbers join copies across derived datasets; iSAID
    tile names group their shared original scene. No annotations or scores used.
    """
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    roles = []
    for binding in tasks:
        row, pair = binding['row'], []
        for role in ('query', 'reference'):
            path = row[role + '_path']
            aliases = ['path:' + str(Path(path))]
            if path in row.get('source_files', {}):
                aliases.append('source_sha:' + row['source_files'][path])
            if row.get(role + '_rgb_hash'):
                aliases.append('rgb_sha:' + row[role + '_rgb_hash'])
            if binding['dataset'] == 'isaid':
                aliases.append('isaid_source:' + Path(path).stem.split('_')[0])
            if binding['dataset'] in ('coco', 'lvis', 'paco_part'):
                match = re.search(r'(\d{12})\.(?:jpg|png)$', path)
                if match:
                    aliases.append('coco_photo:' + match.group(1))
            for alias in aliases:
                union(aliases[0], alias)
            pair.append(aliases[0])
        roles.append(pair)
    return [[find(role) for role in pair] for pair in roles]


def split_tasks(tasks):
    groups = photo_groups(tasks)

    def side(photo):
        value = hashlib.sha256(f'{SEED}\0{photo}'.encode()).digest()
        return int.from_bytes(value[:8], 'big') % 2

    train, test, dropped = [], [], []
    counts = defaultdict(Counter)
    for index, (binding, pair) in enumerate(zip(tasks, groups)):
        q, r = map(side, pair)
        split = 'train' if q == r == 0 else 'test' if q == r == 1 else 'cross_drop'
        {'train': train, 'test': test, 'cross_drop': dropped}[split].append(index)
        counts[binding['dataset']][split] += 1
    train_photos = {p for i in train for p in groups[i]}
    test_photos = {p for i in test for p in groups[i]}
    if train_photos & test_photos:
        raise RuntimeError('A photograph crosses the train/test split')
    if any(v['train'] == 0 or v['test'] == 0 for v in counts.values()):
        raise RuntimeError('An empty domain split; do not change seed after outcomes')
    return dict(seed=SEED, train_indices=train, test_indices=test, dropped_indices=dropped,
                photo_groups=groups, counts=counts, train_photo_n=len(train_photos),
                test_photo_n=len(test_photos), all_role_photo_overlap=0,
                rule='hash original photo identity; retain episodes only when R and Q are on same side',
                lodo='remove every training episode from heldout domain; photo isolation is already global')


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    source_config = read(SOURCE / 'config.json')
    checked(SOURCE / 'manifest.json', source_config['manifest_sha256'])
    checked(Path(ah.__file__), source_config['source_sha256'][str(Path(ah.__file__))])
    descriptor_seal = read(SOURCE / 'descriptors_sealed.json')
    if descriptor_seal['state'] != 'ALL_GT_FREE_DESCRIPTORS_SEALED':
        raise RuntimeError('Unsealed A6 descriptors')
    checked(SOURCE / 'config.json', descriptor_seal['config_sha256'])
    tasks = read(SOURCE / 'manifest.json')
    if any(b.get('split') == 'confirm' for b in tasks):
        raise RuntimeError('E1 accepts only the sealed explore manifest')
    descriptors = {r['index']: r for r in descriptor_seal['records']}
    if len(tasks) != 11477 or set(descriptors) != {pc.key(b) for b in tasks}:
        raise RuntimeError('A6 episode identity/count mismatch')
    split = split_tasks(tasks)
    # Metadata presence audit only; no target values are read at this stage.
    for binding in tasks:
        key = pc.key(binding)
        for path in (Path(descriptors[key]['path']), SOURCE / 'targets' / f'{key}.npz',
                     SOURCE / 'target_records' / f'{key}.json'):
            if not path.is_file():
                raise FileNotFoundError(path)
    config = dict(task='E1', geometry=GEOMETRY, features=list(ah.FEATURES), seed=SEED,
                  models=['full13', 'mean_c'], train_objective='soft BCE', l2=1e-4,
                  optimizer='frozen A6 LBFGS max_iter100', natural_threshold_probability=.5,
                  weighting='valid region area / episode valid area, episodes equally weighted',
                  modes=['holdout', 'LODO'], source=str(SOURCE),
                  source_sha256={str(p): sha(p) for p in (
                      SOURCE / 'manifest.json', SOURCE / 'config.json',
                      SOURCE / 'descriptors_sealed.json', SOURCE / 'targets_sealed.json',
                      Path(__file__), Path(ah.__file__), Path(pc.__file__), Path(pr.__file__))},
                  n=len(tasks), train_n=len(split['train_indices']), test_n=len(split['test_indices']),
                  cross_split_dropped_n=len(split['dropped_indices']),
                  query_GT_role='training half region targets only; heldout counts after prediction seal',
                  new_encoder_calls=0, deployment_method=False, confirm_scored=False,
                  rationale='A6 K64 oracle exceeds FoRIS on matched LVIS small objects; supervised head was never fit',
                  old_70_gate='not consulted or modified; this is a separately planned capacity diagnostic')
    if (out / 'config.json').exists():
        if read(out / 'config.json') != config or read(out / 'split.json') != split:
            raise RuntimeError('Preserve existing E1 run with different inputs/configuration')
    else:
        write(out / 'config.json', config)
        write(out / 'split.json', split)
    print(json.dumps(dict(stage='prepared', train_n=config['train_n'], test_n=config['test_n'],
                         counts=split['counts']), ensure_ascii=False), flush=True)
    return tasks, descriptors, split, config


def load_features(tasks, descriptors, selected):
    features, lengths = [], []
    for position, index in enumerate(selected):
        receipt = descriptors[pc.key(tasks[index])]
        checked(receipt['path'], receipt['sha256'])
        with np.load(receipt['path'], allow_pickle=False) as archive:
            value = archive['features/' + GEOMETRY].astype(np.float32)
        if value.ndim != 2 or value.shape[1] != 13 or not 0 < len(value) <= 64 or not np.isfinite(value).all():
            raise RuntimeError('Invalid sealed K64 descriptors')
        features.append(value)
        lengths.append(len(value))
    return np.concatenate(features), np.asarray(lengths, np.int64)


def target(binding, descriptor_sha):
    key = pc.key(binding)
    receipt = read(SOURCE / 'target_records' / (key + '.json'))
    if receipt['index'] != key or receipt['descriptor_sha256'] != descriptor_sha:
        raise RuntimeError('A6 target binding mismatch')
    checked(receipt['path'], receipt['sha256'])
    with np.load(receipt['path'], allow_pickle=False) as archive:
        n = archive['n/' + GEOMETRY].astype(np.int64)
        tp = archive['tp/' + GEOMETRY].astype(np.int64)
    if n.ndim != 1 or not 0 < len(n) <= 64 or tp.shape != n.shape or n.sum() != receipt['valid'] or tp.sum() != receipt['GT']:
        raise RuntimeError('A6 region target geometry mismatch')
    if np.any(tp < 0) or np.any(tp > n):
        raise RuntimeError('Invalid TP/valid region counts')
    return n, tp, receipt


def fit(out, tasks, descriptors, split, config):
    seal_path = out / 'predictions_sealed.json'
    if seal_path.exists():
        checked(out / 'predictions.npz', read(seal_path)['prediction_sha256'])
        return
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    train, test = split['train_indices'], split['test_indices']
    x_train, train_lengths = load_features(tasks, descriptors, train)
    x_test, test_lengths = load_features(tasks, descriptors, test)
    targets, weights = [], []
    for position, index in enumerate(train):
        binding = tasks[index]
        n, tp, _ = target(binding, descriptors[pc.key(binding)]['sha256'])
        if len(n) != train_lengths[position]:
            raise RuntimeError('Training descriptor and count lengths differ')
        targets.append((tp / np.maximum(n, 1)).astype(np.float32))
        weights.append((n / max(n.sum(), 1)).astype(np.float32))
    y, weight = np.concatenate(targets), np.concatenate(weights)
    train_domain = np.repeat([tasks[i]['dataset'] for i in train], train_lengths)
    test_domain = np.repeat([tasks[i]['dataset'] for i in test], test_lengths)
    predictions = {'test_indices': np.asarray(test, np.int64),
                   'offsets': np.r_[0, np.cumsum(test_lengths)],
                   'train_indices': np.asarray(train, np.int64),
                   'train_offsets': np.r_[0, np.cumsum(train_lengths)]}
    models = {}
    for feature_arm, columns in [('full13', list(range(13))), ('mean_c', [0])]:
        for mode in ('holdout', 'LODO'):
            logits = np.full(len(x_test), np.nan, np.float32)
            domains = [None] if mode == 'holdout' else sorted(set(train_domain))
            for domain in domains:
                selected = np.ones(len(x_train), bool) if domain is None else train_domain != domain
                to_predict = np.ones(len(x_test), bool) if domain is None else test_domain == domain
                label = feature_arm + '/' + mode + '/' + (domain or 'all')
                model_path = out / 'models' / (label.replace('/', '__') + '.json')
                if model_path.exists():
                    saved = read(model_path)
                    if saved['config_sha256'] != sha(out / 'config.json'):
                        raise RuntimeError('Existing E1 model uses another configuration')
                    model = saved['model']
                else:
                    model = ah.fit_logistic(x_train[selected][:, columns], y[selected], weight[selected])
                    write(model_path, dict(model=model, features=[ah.FEATURES[k] for k in columns],
                          training_domains=sorted(set(train_domain[selected])), heldout_domain=domain,
                          config_sha256=sha(out / 'config.json'), train_region_n=int(selected.sum())))
                logits[to_predict] = (x_test[to_predict][:, columns] @ np.asarray(model['coef'], np.float32)
                                      + model['intercept'])
                if domain is None:
                    train_logits = x_train[:, columns] @ np.asarray(model['coef'], np.float32) + model['intercept']
                    predictions['logits/train/' + feature_arm] = train_logits
                    predictions['probability/train/' + feature_arm] = torch.sigmoid(torch.from_numpy(train_logits).double()).numpy()
                    predictions['natural/train/' + feature_arm] = train_logits > 0
                models[label] = dict(path=str(model_path), sha256=sha(model_path))
                print(json.dumps(dict(stage='fit', model=label, seconds=round(time.monotonic() - start, 2))), flush=True)
            if not np.isfinite(logits).all():
                raise RuntimeError('Missing or nonfinite predictions')
            arm = mode + '/' + feature_arm
            predictions['logits/' + arm] = logits
            # Double probabilities preserve strict same-score groups for the diagnostic oracle.
            predictions['probability/' + arm] = torch.sigmoid(torch.from_numpy(logits).double()).numpy()
            predictions['natural/' + arm] = logits > 0
    save_npz(out / 'predictions.npz', **predictions)
    write(seal_path, dict(state='ALL_HELDOUT_PREDICTIONS_SEALED', n=len(test),
                         prediction_sha256=sha(out / 'predictions.npz'),
                         config_sha256=sha(out / 'config.json'), split_sha256=sha(out / 'split.json'),
                         models=models, seconds=time.monotonic() - start,
                         heldout_target_files_read=0, new_encoder_calls=0, deployment_method=False))


def strict_oracle(probability, n, tp, gt_area):
    """Best feasible strict probability threshold; never split a tied score group."""
    order = np.argsort(-probability, kind='stable')
    sorted_p = probability[order]
    endpoints = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1, len(order)]
    endpoints = np.r_[0, endpoints]
    inter = np.r_[0, np.cumsum(tp[order])]
    false = np.r_[0, np.cumsum((n - tp)[order])]
    cut = endpoints[np.argmax(inter[endpoints] / np.maximum(gt_area + false[endpoints], 1))]
    result = np.zeros(len(n), bool)
    result[order[:cut]] = True
    return result


def paired_summary(records, candidate, control):
    projected = []
    for record in records:
        entry = dict(record, metrics={'FoRIS': record['metrics'][control], candidate: record['metrics'][candidate]})
        projected.append(entry)
    table = pr.summary(projected, ['FoRIS', candidate])
    return {d: dict(gain_pp=s[candidate]['delta_to_foris_pp'],
                    paired_query_photo_bootstrap95_pp=s[candidate]['paired_query_photo_bootstrap95_pp'])
            for d, s in table.items()}


def score(out, tasks, descriptors, split):
    start = time.monotonic()
    seal = read(out / 'predictions_sealed.json')
    checked(out / 'predictions.npz', seal['prediction_sha256'])
    checked(out / 'config.json', seal['config_sha256'])
    checked(out / 'split.json', seal['split_sha256'])
    with np.load(out / 'predictions.npz', allow_pickle=False) as archive:
        predictions = {name: archive[name].copy() for name in archive.files}
    if predictions['test_indices'].tolist() != split['test_indices']:
        raise RuntimeError('Prediction order mismatch')
    records = {'train': [], 'holdout': [], 'LODO': []}
    mode_counts, mode_receipts = {}, {}
    arms = ['FoRIS', 'IND0', 'full13', 'mean_c', 'full13_threshold_oracle',
            'mean_c_threshold_oracle', 'K64_episode_oracle', 'K64_class_oracle']
    for phase, phase_modes in [('train', ['train']), ('test', ['holdout', 'LODO'])]:
        counts, target_receipts = [], []
        offsets = predictions['train_offsets'] if phase == 'train' else predictions['offsets']
        for position, index in enumerate(split[phase + '_indices']):
            binding = tasks[index]
            n, tp, receipt = target(binding, descriptors[pc.key(binding)]['sha256'])
            counts.append((n, tp))
            target_receipts.append(receipt)
            span = slice(offsets[position], offsets[position + 1])
            if len(n) != span.stop - span.start:
                raise RuntimeError('Scoring descriptor and count lengths differ')
            for mode in phase_modes:
                record = copy.deepcopy(receipt['record'])
                # The resampling cluster is the same original-photo group used for splitting.
                record['query_photo'] = split['photo_groups'][index][0]
                for arm in ('full13', 'mean_c'):
                    natural = predictions['natural/' + mode + '/' + arm][span]
                    record['metrics'][arm] = ah.metric_counts(natural, n, tp, receipt['GT'], receipt['valid'])
                    oracle = strict_oracle(predictions['probability/' + mode + '/' + arm][span], n, tp, receipt['GT'])
                    record['metrics'][arm + '_threshold_oracle'] = ah.metric_counts(oracle, n, tp, receipt['GT'], receipt['valid'])
                oracle = ah.selection(tp, n - tp, receipt['GT'])
                record['metrics']['K64_episode_oracle'] = ah.metric_counts(oracle, n, tp, receipt['GT'], receipt['valid'])
                records[mode].append(record)
        for mode in phase_modes:
            mode_counts[mode], mode_receipts[mode] = counts, target_receipts
        save_npz(out / (phase + '_region_counts.npz'), n=np.concatenate([x[0] for x in counts]),
                 tp=np.concatenate([x[1] for x in counts]), offsets=offsets,
                 indices=predictions[phase + '_indices'])
    for mode in records:
        counts, target_receipts = mode_counts[mode], mode_receipts[mode]
        groups = defaultdict(list)
        for i, record in enumerate(records[mode]):
            groups[(record['dataset'], str(record['fold']), str(record['class_id']))].append(i)
        for indices in groups.values():
            n = np.concatenate([counts[i][0] for i in indices])
            tp = np.concatenate([counts[i][1] for i in indices])
            keep = ah.selection(tp, n - tp, sum(target_receipts[i]['GT'] for i in indices))
            offset = 0
            for i in indices:
                n_i, tp_i = counts[i]
                receipt = target_receipts[i]
                metric = ah.metric_counts(keep[offset:offset + len(n_i)], n_i, tp_i, receipt['GT'], receipt['valid'])
                records[mode][i]['metrics']['K64_class_oracle'] = metric
                offset += len(n_i)
    summaries, contrasts = {}, {}
    for mode, rows in records.items():
        summaries[mode] = pr.summary(rows, arms)
        contrasts[mode] = {f'full13_minus_{control}': paired_summary(rows, 'full13', control)
                           for control in ('mean_c', 'IND0', 'FoRIS')}
        write(out / f'{mode}_records.json', rows)
    notes = [
        'Supervised diagnostic only: fixed K64 and pre-existing 13-dimensional GT-free regional descriptors.',
        'Both reference and query original-photo identities are isolated globally; mixed-side episodes are omitted.',
        'LODO excludes the heldout domain from the same training half; test set is identical across both modes.',
        'Train-only standardization, fixed soft BCE/L2 1e-4, fixed probability >.5; no model/threshold grid.',
        'Train table is in-sample fit description; only holdout and LODO assess label-isolated transfer.',
        'Full descriptor arm and mean_c-only arm use the same supervision/geometry/optimizer.',
        'CLI1024 metrics use exact nearest region counts; class mIoU averages observed (fold,class) pooled I/U.',
        'Threshold and region oracles read heldout labels only after predictions seal; not natural results.',
        'Failure constrains this descriptor/linear head/readout only, not DINO or single-reference information limits.',
        'No new encoding, raw feature loads, RGB loads, or confirm evaluation. Cached labels are for diagnostics only.',
    ]
    report = dict(state='COMPLETE_SUPERVISED_DIAGNOSTIC', n=len(split['test_indices']),
                  train_n=len(split['train_indices']), arms=arms, summaries=summaries,
                  contrasts=contrasts, notes=notes, new_encoder_calls=0, deployment_method=False,
                  confirm_scored=False, score_seconds=time.monotonic() - start,
                  config_sha256=sha(out / 'config.json'), prediction_seal_sha256=sha(out / 'predictions_sealed.json'))
    write(out / 'report.json', report)
    lines = ['# E1 fixed K64 region descriptor capacity', '', *notes, '']
    for mode in summaries:
        lines.extend([f'## {mode}', '', '|Domain|n|Arm|class mIoU|episode mean|delta FoRIS pp|95% query-photo CI|',
                      '|---|---:|---|---:|---:|---:|---|'])
        for domain, table in summaries[mode].items():
            for arm, value in table.items():
                ci = value['paired_query_photo_bootstrap95_pp']
                lines.append(f"|{domain}|{value['n']}|{arm}|{value['class_miou']:.3f}|{value['episode_mean_iou']:.3f}|"
                             f"{value['delta_to_foris_pp']:+.3f}|[{ci[0]:+.3f},{ci[1]:+.3f}]|")
        lines.append('')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    write(out / 'completion.json', {k: report[k] for k in ('state', 'n', 'train_n', 'new_encoder_calls',
                                                         'deployment_method', 'confirm_scored', 'score_seconds')})
    print(json.dumps(dict(stage='complete', n=report['n'], report=str(out / 'report.md'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'fit', 'score', 'all'), default='prepare')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if ROOT == out or ROOT in out.parents or not (ASSETS / 'a').resolve() in out.parents:
        raise ValueError('Output must be below the external cv_data/a directory')
    tasks, descriptors, split, config = prepare(out)
    if args.stage in ('fit', 'all'):
        fit(out, tasks, descriptors, split, config)
    if args.stage in ('score', 'all'):
        score(out, tasks, descriptors, split)


if __name__ == '__main__':
    main()
