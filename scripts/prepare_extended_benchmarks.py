#!/usr/bin/env python3
"""Freeze released seed-0 extended FoRIS inputs; no encoder or predictions.

iSAID defaults to the first 20 sequential episodes of every official fold.
Single-fold datasets retain the released loader's default episode count.
Unavailable pools/protocols are recorded and do not replace ready datasets.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '2')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify_draws_and_inputs(rows, assets):
    """Replay all author sampling draws and all legal inputs, without scoring."""
    from PIL import Image
    from ics.extended_benchmark_data import build_dataset, load_inputs, reset_sampling_seed
    counts = Counter()
    for name, fold in sorted({(r['dataset'], r['fold']) for r in rows}):
        reset_sampling_seed(0)
        ds = build_dataset(name, fold, assets)
        for row in (r for r in rows if r['dataset'] == name and r['fold'] == fold):
            target, references, class_id = ds.sample_episode(row['official_index'])
            expected = row['official_selection']['sample_episode']
            if (str(target) != expected['tgt_name'] or list(map(str, references)) != expected['ref_names']
                    or int(class_id) != row['loader_class_id']):
                raise AssertionError('Manifest altered the official random draw')
            forbidden = {str(Path(row[k]).resolve()) for k in ('query_mask_path', 'query_ignore_mask_path') if k in row}
            forbidden.add(str((assets/row['query_annotation_path']).resolve()))
            opened, original_open = [], Image.open
            def observed_open(path, *args, **kwargs):
                opened.append(str(Path(path).resolve()))
                if opened[-1] in forbidden:
                    raise AssertionError('Inference adapter opened query labels')
                return original_open(path, *args, **kwargs)
            try:
                Image.open = observed_open
                reference, mask, query = load_inputs(row, assets)
            finally:
                Image.open = original_open
            if len(opened) != 3 or list(mask.shape) != row['reference_mask_size_hw']:
                raise AssertionError('Inference adapter contract changed')
            if not mask.numpy().any():
                raise AssertionError('Official reference mask is empty')
            counts[name] += 1
    return dict(state='PASSED', episodes=dict(counts),
                checks=['all sequential official seed0 draws identical',
                        'all frozen RGB/reference masks replay and hash identically',
                        'all inference calls open only R/R-mask/Q; query masks and source GT denied'],
                encodes=0, predictions=0, query_scoring=0)


def main():
    from ics.extended_benchmark_data import (EXTENDED_FOLDS, METRIC_CONVENTIONS,
        canonical_name, source_receipt, verify_pool, build_dataset, record_episode,
        reset_sampling_seed, file_hash)
    import torch
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=REPO.parent/'cv_data')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--datasets', default='isaid,isic,lung,fundus,deepglobe')
    parser.add_argument('--per-fold', type=int, default=20)
    args = parser.parse_args()
    args.assets, args.out = args.assets.resolve(), args.out.resolve()
    args.datasets = [canonical_name(n) for n in args.datasets.split(',')]
    if args.per_fold < 1 or len(set(args.datasets)) != len(args.datasets) or set(args.datasets)-set(EXTENDED_FOLDS):
        parser.error('Require positive per-fold count and unique extended datasets')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Preparation requires a fresh output directory')
    source = source_receipt(args.assets, args.datasets)
    args.out.mkdir(parents=True, exist_ok=True)
    for rel in source['files']:
        dest = args.out/'source/foris'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.assets/'third_party/foris_official'/rel, dest)
    for path in (Path(__file__), REPO/'src/ics/extended_benchmark_data.py', REPO/'src/ics/official_data.py'):
        dest = args.out/'source/local'/path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    rows, pools, folds, pending = [], {}, {}, {}
    for name in args.datasets:
        try:
            pool = verify_pool(name, args.assets)
            write(args.out/'pools'/(name+'.json'), pool)
            hashes = dict(pool['asset_sha256'])
            dataset_rows, dataset_folds = [], {}
            for fold in EXTENDED_FOLDS[name]:
                reset_sampling_seed(0)
                ds = build_dataset(name, fold, args.assets)
                n = min(len(ds), args.per_fold) if len(EXTENDED_FOLDS[name]) > 1 else len(ds)
                selected = []
                for index in range(n):
                    selected.append(record_episode(ds, index, fold, args.assets, args.out/'masks', hashes))
                    if (index+1) % 100 == 0:
                        print(json.dumps(dict(dataset=name, fold=fold, prepared=index+1, total=n)), flush=True)
                dataset_rows.extend(selected)
                dataset_folds[f'{name}/{fold}'] = dict(official_length=len(ds), selected_length=n,
                    expected_class_ids=list(ds.class_ids), selected_class_counts=dict(Counter(r['loader_class_id'] for r in selected)),
                    protocol='first sequential official seed0 draws' if n < len(ds) else 'released default full episode count')
                print(json.dumps(dict(dataset=name, fold=fold, prepared=n, official_length=len(ds))), flush=True)
            for fold in EXTENDED_FOLDS[name]:
                dest = args.out/'manifests'/name/f'fold_{fold}.jsonl'
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(''.join(json.dumps(r)+'\n' for r in dataset_rows if r['fold'] == fold))
            rows.extend(dataset_rows)
            folds.update(dataset_folds)
            pools[name] = dict(state=pool['state'], files=pool['files'], bytes=pool['bytes'],
                               inventory_sha256=file_hash(args.out/'pools'/(name+'.json')))
        except FileNotFoundError as error:
            pending[name] = str(error)
            print(json.dumps(dict(dataset=name, state='BLOCKED', reason=str(error))), flush=True)
    write(args.out/'manifest.json', rows)
    validation = verify_draws_and_inputs(rows, args.assets)
    write(args.out/'validation.json', validation)
    # Mask file paths are location-dependent. This second digest captures the
    # complete sampled identities independently of the preparation directory.
    identity = [{k: v for k, v in r.items() if not k.endswith('_mask_path')} for r in rows]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write(args.out/'prepared.json', dict(state='EXTENDED_PILOT_FROZEN_WITH_BLOCKERS' if pending else 'EXTENDED_PILOT_FROZEN',
        seed=0, shots=1, shuffle=False, num_workers=0, n=len(rows), folds=folds, pools=pools,
        pending_datasets=pending, source=source, metric_conventions=METRIC_CONVENTIONS,
        manifest_sha256=file_hash(args.out/'manifest.json'), episode_identity_sha256=digest,
        validation_sha256=file_hash(args.out/'validation.json'),
        query_label_role='protocol preparation only; no predictions or scoring',
        pilot_scope='iSAID prefix is not a full benchmark result; ISIC uses released default 600 draws',
        duplicates='official draws retained', inference_interface='R, reference mask, Q only'))
    print(json.dumps(dict(state='FROZEN', n=len(rows), pending=pending,
                          manifest_sha256=file_hash(args.out/'manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
