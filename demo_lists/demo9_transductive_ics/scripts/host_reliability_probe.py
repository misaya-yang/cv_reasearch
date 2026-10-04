#!/usr/bin/env python3
"""Prepare an annotation-free packet and a bounded CPU measuring experiment.

Preparation reads existing native masks and visual-prompt metadata only.
Evaluation is separately opt-in, uses old scored I/U, and fits supervised
measuring devices. It is neither a training-free method nor fresh confirmation.
No torch/model import, GPU call, dataset download, or machine shutdown exists.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zlib

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

ARMS = ('foris', 'sam', 'OR', 'AND')
SOURCE_FEATURES = ('source_iou', 'source_area', 'source_components_log')
FEATURES = SOURCE_FEATURES + (
    'sam_presence_logit', 'sam_kept_log', 'query_proposal_conf_max',
    'query_proposal_conf_mean', 'query_proposal_count_log',
    'sam_area', 'sam_box_area', 'sam_box_fill', 'sam_edge_fraction',
    'foris_area', 'foris_box_area', 'foris_box_fill', 'foris_edge_fraction',
    'host_iou', 'intersection_over_sam', 'intersection_over_foris',
    'sam_only_area', 'foris_only_area', 'log_area_ratio',
)
CARD = [
    'Assumption tested: the frozen legal packet contains cross-class information about which complete host output to use, beyond a constant host or OR.',
    'Null prediction: no gain over the best fixed control and no advantage over test-packet shuffling; baseline replay difference0. Alternative decision target: >=2pp over the strongest fixed control, not a forecast derived from oracle headroom.',
    'If the null is refuted with positive aggregate gains and a matched signal contrast, identify which legal block carries the effect; this remains a supervised diagnostic and does not authorize a new method, fresh labels or GPU.',
    'If the packet fails, stop this packet/decoder combination; do not sweep thresholds or reopen old selectors, and do not claim all model representations are insufficient. Any join, feature leakage, or baseline replay error invalidates the run.',
]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def key(row):
    return (int(row['fold']), int(row['e']), int(row['c']), row['support'], row['query'])


def unique(items):
    result = {key(row): row for row in items}
    if len(result) != len(items):
        raise ValueError('Duplicate episode')
    return result


def mask_features(mask):
    area = int(mask.sum())
    yy, xx = np.nonzero(mask)
    box = int((yy.max()-yy.min()+1)*(xx.max()-xx.min()+1)) if area else 0
    edges = int(np.count_nonzero(mask[1:] != mask[:-1]) + np.count_nonzero(mask[:, 1:] != mask[:, :-1]))
    return [area/mask.size, box/mask.size, area/max(box, 1), edges/mask.size]


def packet(rec, sam, foris):
    """Explicit allowlist: no category, text arm, QueryGT, scored I/U, or timing."""
    if sam.shape != foris.shape or list(sam.shape) != rec['query_shape']:
        raise ValueError('Original mask geometry mismatch')
    inter, sa, fa = int((sam & foris).sum()), int(sam.sum()), int(foris.sum())
    si, su = rec['reference_union_iu']  # legal SOURCE supervision only
    probs = [float(p[0]) for p in rec['proposal_metadata'] if p[1] > 0]
    presence = np.clip(float(rec['presence']['visual']), 1e-6, 1-1e-6)
    values = [si/max(su, 1), float(rec['reference_area']), np.log1p(rec['components']),
              np.log(presence/(1-presence)), np.log1p(rec['kept']['visual']),
              max(probs, default=0), float(np.mean(probs)) if probs else 0, np.log1p(len(probs)),
              *mask_features(sam), *mask_features(foris),
              inter/max(sa+fa-inter, 1), inter/max(sa, 1), inter/max(fa, 1),
              (sa-inter)/sam.size, (fa-inter)/sam.size, np.log((sa+1)/(fa+1))]
    if len(values) != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError('Invalid legal feature vector')
    return [float(value) for value in values]


def split_indices(records, fold):
    test = [i for i, r in enumerate(records) if r['fold'] == fold]
    forbidden = {r[role] for i in test for r in [records[i]] for role in ('support', 'query')}
    train = [i for i, r in enumerate(records) if r['fold'] != fold
             and not ({r['support'], r['query']} & forbidden)]
    train_classes = {records[i]['c'] for i in train}
    test_classes = {records[i]['c'] for i in test}
    if train_classes & test_classes or not train or not test:
        raise ValueError('Class-disjoint nonempty split required')
    return np.asarray(train), np.asarray(test)


def prepare(args):
    if (args.out/'manifest.json').exists():
        raise ValueError('Preparation already frozen; choose a new output directory')
    start = time.monotonic()
    records, inputs = [], {}
    for split, expected in (('dev', 241), ('confirm', 600)):
        a = args.root/'results/sam3_handover_v3'/('A_'+split)
        b = args.root/'results/sam3_handover_v3'/('B_'+split)
        ledger = a/'predictions_shard0.jsonl'
        preds = rows(ledger)
        freeze_a = read(a/'prediction_freeze_shard0.json')
        freeze_b = read(b/'freeze_report.json')
        if len(preds) != expected or freeze_a['predictions_jsonl_sha256'] != sha(ledger):
            raise ValueError('Native prediction ledger incomplete or changed')
        if freeze_a['query_annotation_opened'] is not False or freeze_b['query_GT_pixels_opened'] is not False:
            raise ValueError('Prediction-before-label contract absent')
        unique(preds)
        frozen = {r['index']: r for r in freeze_b['frozen_cases']}
        assets = {r['path']: r for r in freeze_a['prediction_files']}
        inputs[str(ledger)] = sha(ledger)
        for name in ('prediction_freeze_shard0.json',):
            inputs[str(a/name)] = sha(a/name)
        inputs[str(b/'freeze_report.json')] = sha(b/'freeze_report.json')
        for index, rec in enumerate(preds):
            if time.monotonic()-start > args.seconds:
                raise TimeoutError('Preparation wall-clock cap')
            case_path = Path(frozen[index]['path'])
            with gzip.open(case_path, 'rt') as stream:
                case = json.load(stream)
            if key(case['row']) != key(rec) or case['query_GT_pixels_opened'] is not False:
                raise ValueError('Native mask episode mismatch')
            packed = case['original_masks']['native']
            if packed['codec'] != 'zlib_np_packbits_big':
                raise ValueError('Unknown original native mask codec')
            bits = zlib.decompress(base64.b64decode(packed['data']))
            if hashlib.sha256(bits).hexdigest() != packed['bits_sha256']:
                raise ValueError('Native bitmap corruption')
            shape = tuple(packed['shape'])
            pixels = int(np.prod(shape))
            fm = np.unpackbits(np.frombuffer(bits, np.uint8))[:pixels].reshape(shape).astype(bool)
            sam_path = a/rec['prediction_file']
            if sam_path.stat().st_size != assets[rec['prediction_file']]['bytes']:
                raise ValueError('SAM saved mask size changed')
            with np.load(sam_path, allow_pickle=False) as masks:
                if tuple(masks['shape']) != shape:
                    raise ValueError('SAM original dimensions mismatch')
                sm = np.unpackbits(masks['visual'])[:pixels].reshape(shape).astype(bool)
            metadata = {name: rec[name] for name in ('fold', 'e', 'c', 'support', 'query')}
            records.append(dict(**metadata, cohort=split, features=packet(rec, sm, fm),
                                source_mask_digest=packed['bits_sha256']))
            if (index+1) % 50 == 0:
                print(json.dumps(dict(state='PREPARING', cohort=split, count=index+1)), flush=True)
    unique(records)
    for fold in range(4):
        split_indices(records, fold)
    args.out.mkdir(parents=True, exist_ok=True)
    feature_path = args.out/'features.jsonl'
    with feature_path.open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
    manifest = dict(state='CPU_PREPARED_NOT_EVALUATED', features=str(feature_path),
                    features_sha256=sha(feature_path), feature_names=FEATURES, source_features=SOURCE_FEATURES,
                    records=len(records), input_ledgers=inputs, script_sha256=sha(__file__),
                    scorer_sha256=sha(Path(__file__).with_name('paired_frozen_stats.py')),
                    target_ledgers={s: str(args.root/'results/fixed_host_union_v1'/(s+'_episodes.jsonl'))
                                    for s in ('dev', 'confirm')},
                    targets_opened=False, query_annotation_pixels_opened=False, GPU_used=False,
                    partition=[dict(fold=f, train=len(split_indices(records, f)[0]),
                                    test=len(split_indices(records, f)[1])) for f in range(4)],
                    card=CARD, seconds=time.monotonic()-start,
                    controls=list(ARMS)+['train_best_constant', 'random_four'],
                    models=['source_linear', 'full_linear', 'full_rbf64'],
                    primary='full_rbf64',
                    caveats=['Only recorded top20 visual proposal metadata used',
                             'All841 episodes previously exposed; cross-fitting is not fresh confirmation',
                             'No model class is an information upper bound'])
    write(args.out/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def fit_predict(x, y, weights, test, kind):
    mean, sd = np.mean(x, axis=0), np.std(x, axis=0)
    x = np.clip((x-mean)/np.maximum(sd, 1e-6), -8, 8)
    test = np.clip((test-mean)/np.maximum(sd, 1e-6), -8, 8)
    if kind == 'rbf':
        centers = x[np.random.default_rng(0).permutation(len(x))[:min(64, len(x))]]
        kernel = lambda z: np.exp(-np.maximum((z*z).sum(1)[:, None] + (centers*centers).sum(1)[None, :]
                                               - 2*z@centers.T, 0)/x.shape[1])
        test, x = kernel(test), kernel(x)
    x, test = np.column_stack((np.ones(len(x)), x)), np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
    coefficient = np.linalg.solve(x.T@(weights[:, None]*x)+penalty, x.T@(weights[:, None]*y))
    return test@coefficient


def evaluate(args):
    if not args.execute_diagnostic:
        raise ValueError('Evaluation requires explicit --execute-diagnostic; preparation never fits')
    if (args.out/'report.json').exists():
        raise ValueError('Preserve completed diagnostic; choose a fresh output directory')
    start = time.monotonic()
    manifest = read(args.prepared/'manifest.json')
    if sha(manifest['features']) != manifest['features_sha256'] or sha(__file__) != manifest['script_sha256']:
        raise ValueError('Frozen preparation/source changed')
    if sha(Path(__file__).with_name('paired_frozen_stats.py')) != manifest['scorer_sha256']:
        raise ValueError('Frozen paired scorer changed')
    records = rows(manifest['features'])
    targets = unique([r for path in manifest['target_ledgers'].values() for r in rows(path)])
    if set(targets) != set(unique(records)):
        raise ValueError('Feature/target scope mismatch')
    x = np.asarray([r['features'] for r in records])
    iu = np.asarray([[targets[key(r)]['original_iu'][a] for a in ARMS] for r in records], float)
    y = iu[..., 0]/np.maximum(iu[..., 1], 1)
    choices = {a: np.full(len(records), i, int) for i, a in enumerate(ARMS)}
    rng = np.random.default_rng(0)
    choices['random_four'] = rng.integers(0, 4, len(records))
    choices['train_best_constant'] = np.zeros(len(records), int)
    for name in manifest['models']:
        for suffix in ('', '_shuffled_test_packet', '_inverted'):
            choices[name+suffix] = np.zeros(len(records), int)
    fold_notes = []
    for fold in range(4):
        if time.monotonic()-start > args.seconds:
            raise TimeoutError('Finite diagnostic cap')
        train, test = split_indices(records, fold)
        classes = np.asarray([records[i]['c'] for i in train])
        weights = np.asarray([1/np.count_nonzero(classes == c) for c in classes])
        weights *= len(weights)/weights.sum()
        class_scores = [np.mean([iu[train[classes == c], a, 0].sum()/max(iu[train[classes == c], a, 1].sum(), 1)
                                for c in np.unique(classes)]) for a in range(4)]
        choices['train_best_constant'][test] = int(np.argmax(class_scores))
        for name in manifest['models']:
            cols = slice(0, len(SOURCE_FEATURES)) if name.startswith('source') else slice(None)
            order = np.random.default_rng(fold).permutation(len(test))
            both = np.concatenate((x[test, cols], x[test[order], cols]))
            pred = fit_predict(x[train, cols], y[train], weights, both, 'rbf' if name.endswith('rbf64') else 'linear')
            real, shuffled = pred[:len(test)], pred[len(test):]
            choices[name][test] = real.argmax(1)
            choices[name+'_shuffled_test_packet'][test] = shuffled.argmax(1)
            choices[name+'_inverted'][test] = real.argmin(1)
        fold_notes.append(dict(fold=fold, train=len(train), test=len(test), best_constant=ARMS[int(np.argmax(class_scores))]))
    # Only saved per-case I/U enter this stage, never query annotation images.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paired_frozen_stats import paired, clear_cache
    prediction_iu = {name: iu[np.arange(len(records)), select] for name, select in choices.items()}
    report = dict(state='COMPLETED_EXPLORATORY_MEASURING_DEVICE', card=CARD, folds=fold_notes,
                  cohorts={}, runtime_seconds=None, GPU_used=False, novelty_claim=False,
                  source_sha256=sha(__file__), prepared_sha256=sha(args.prepared/'manifest.json'))
    for cohort in ('dev', 'confirm'):
        ids = [i for i, r in enumerate(records) if r['cohort'] == cohort]
        scored = [dict(**records[i], original_iu={name: values[i].tolist() for name, values in prediction_iu.items()})
                  for i in ids]
        for row in scored:
            row['original_iu']['GT_pixel_disagreement'] = targets[key(row)]['original_iu']['disagreement_oracle']
        clear_cache()
        results = {}
        for name in (*choices, 'GT_pixel_disagreement'):
            results[name] = paired(scored, lambda r, name=name: r['original_iu'][name],
                                   lambda r: r['original_iu']['OR'], draws=2000)
            results[name]['per_fold_miou'] = {}
            for fold in range(4):
                selected = [r for r in scored if r['fold'] == fold]
                sums = {}
                for row in selected:
                    target = sums.setdefault(row['c'], [0., 0.])
                    inter, union = row['original_iu'][name]
                    target[0] += inter; target[1] += union
                results[name]['per_fold_miou'][str(fold)] = float(100*np.mean([i/max(u, 1) for i, u in sums.values()]))
        contrasts = {}
        for name in manifest['models']:
            controls = ('train_best_constant', name+'_shuffled_test_packet', name+'_inverted')
            if name != 'source_linear':
                controls += ('source_linear',)
            for control in controls:
                contrasts[name+'__minus__'+control] = paired(scored, lambda r, name=name: r['original_iu'][name],
                                                           lambda r, control=control: r['original_iu'][control], draws=2000)
        original_report = read(Path(manifest['target_ledgers'][cohort]).with_name(cohort+'_report.json'))
        for arm in ARMS:
            value = original_report['class_sum_original_mIoU'][arm]
            if abs(results[arm]['miou']-value) > 1e-9:
                raise ValueError('Frozen original-resolution baseline replay differs: '+cohort+'/'+arm)
        report['cohorts'][cohort] = dict(episodes=len(ids), results=results, contrasts=contrasts,
                                        scope='old exposed data; original-resolution class-summed IoU')
        args.out.mkdir(parents=True, exist_ok=True)
        write(args.out/(cohort+'_choices.json'), [dict(fold=records[i]['fold'], e=records[i]['e'], c=records[i]['c'],
                                                      choices={k: ARMS[int(v[i])] for k, v in choices.items()}) for i in ids])
    report['runtime_seconds'] = time.monotonic()-start
    write(args.out/'report.json', report)
    print(json.dumps(dict(state=report['state'], seconds=report['runtime_seconds'])))


def self_test(args):
    start = time.monotonic()
    rec = dict(query_shape=[8, 9], reference_union_iu=[8, 10], reference_area=.3, components=2,
               presence={'visual': .8, 'text': .999}, kept={'visual': 3, 'text': 8},
               proposal_metadata=[[.8, 12, 8, 6], [.4, 0, 6, 4]])
    sam = np.zeros((8, 9), bool); sam[1:6, 2:7] = True
    fm = np.zeros_like(sam); fm[2:7, 1:6] = True
    first = packet(rec, sam, fm)
    contaminated = dict(rec, c=999, original_iu={'visual': [999, 999]}, query_GT='FORBIDDEN')
    contaminated['presence'] = dict(visual=.8, text=0)
    assert first == packet(contaminated, sam, fm)
    assert np.isfinite(packet(rec, np.zeros_like(sam), np.zeros_like(fm))).all()
    records = [dict(fold=f, e=i, c=f, support=f's{f}_{i}', query=f'q{f}_{i}') for f in range(4) for i in range(10)]
    records[10]['support'] = records[0]['query']
    tr, te = split_indices(records, 0)
    assert 10 not in tr and len(te) == 10
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(FEATURES))); y = rng.random((40, 4))
    for kind in ('linear', 'rbf'):
        pred = fit_predict(x[tr], y[tr], np.ones(len(tr)), x[te], kind)
        assert pred.shape == (10, 4) and np.isfinite(pred).all()
        altered = y.copy(); altered[te] = 1000
        assert np.array_equal(pred, fit_predict(x[tr], altered[tr], np.ones(len(tr)), x[te], kind))
    from paired_frozen_stats import paired
    scored = [dict(c=i % 4, support=f's{i}', query=f'q{i}', a=[i+1, i+10]) for i in range(10)]
    identity = paired(scored, lambda r: r['a'], lambda r: r['a'], draws=50)
    assert identity['gain'] == 0 and identity['ci95'] == [0, 0]
    # Exercise the SAME complete evaluator on synthetic data, including joins,
    # baseline reproduction, fold purging, model controls and report writing.
    with tempfile.TemporaryDirectory(prefix='host_reliability_fixture_') as tmp:
        folder = Path(tmp)
        legal, targets = [], {'dev': [], 'confirm': []}
        for index, record in enumerate(records):
            cohort = 'dev' if record['e'] < 5 else 'confirm'
            truth = rng.random((8, 9)) > .6
            sm, fm = rng.random((8, 9)) > .5, rng.random((8, 9)) > .5
            masks = dict(foris=fm, sam=sm, OR=fm|sm, AND=fm&sm,
                         disagreement_oracle=(fm&sm)|(truth&(fm^sm)))
            legal.append(dict(**record, cohort=cohort, features=packet(rec, sm, fm)))
            targets[cohort].append(dict(**record, original_iu={a: [int((m&truth).sum()), int((m|truth).sum())]
                                                              for a, m in masks.items()}))
        feature_path = folder/'features.jsonl'
        feature_path.write_text(''.join(json.dumps(row)+'\n' for row in legal))
        target_paths = {}
        for cohort, target_rows in targets.items():
            path = folder/(cohort+'_episodes.jsonl')
            path.write_text(''.join(json.dumps(row)+'\n' for row in target_rows))
            target_paths[cohort] = str(path)
            baseline = {a: paired(target_rows, lambda r, a=a: r['original_iu'][a],
                                  lambda r, a=a: r['original_iu'][a], draws=1)['miou'] for a in ARMS}
            write(folder/(cohort+'_report.json'), dict(class_sum_original_mIoU=baseline))
        write(folder/'manifest.json', dict(features=str(feature_path), features_sha256=sha(feature_path),
                                          script_sha256=sha(__file__), target_ledgers=target_paths,
                                          scorer_sha256=sha(Path(__file__).with_name('paired_frozen_stats.py')),
                                          models=['source_linear', 'full_linear', 'full_rbf64']))
        evaluate(argparse.Namespace(execute_diagnostic=True, prepared=folder, out=folder/'evaluated', seconds=120))
        full = read(folder/'evaluated/report.json')
        assert all(full['cohorts'][cohort]['episodes'] == 20 for cohort in targets)
    result = dict(state='CPU_SELF_TEST_PASSED', cases=10, feature_count=len(FEATURES),
                  checks=['text_and_QueryGT_do_not_enter_packet', 'empty_masks', 'cross_role_photo_purge',
                          'linear_and_rbf_finite', 'heldout_targets_do_not_change_predictions', 'paired_identity',
                          'complete_evaluator40_synthetic_rows'],
                  seconds=time.monotonic()-start, real_model_test=False, scientific_evaluation=False,
                  source_sha256=sha(__file__))
    write(args.out/'self_test.json', result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'self-test', 'evaluate'])
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/demo9_transductive_ics'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--execute-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.seconds > 900 or args.seconds < 1:
        parser.error('CPU cap must be1..900 seconds')
    if args.mode == 'evaluate' and not args.prepared:
        parser.error('--prepared is required')
    {'prepare': prepare, 'self-test': self_test, 'evaluate': evaluate}[args.mode](args)


if __name__ == '__main__':
    main()
