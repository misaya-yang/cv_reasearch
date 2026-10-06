#!/usr/bin/env python3
"""Pro M3 cached CPU infer/score; only q/r/cov/score and lazy native read.

Manifest schema matches run_cpu_feature_candidates.py. Infer seals all four
complete 1024 masks before score reads GT. No DINO encoder calls occur here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def initialize(threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def one(job):
    import numpy as np
    import resource
    from ics.experiment import load_inputs, render, sha, unpack
    from ics.methods.pro_role_prediction import predict
    row, directory = job
    directory = Path(directory)
    started = time.perf_counter()
    (q, r, cov, score), source = load_inputs('/', row)
    if tuple(q.shape) != (4096, 1024) or tuple(r.shape) != (4096, 1024) or cov.shape != (64, 64) or score.shape != cov.shape:
        raise ValueError('Batch v0 requires4096x1024 feature matrices and64x64 coverage/score')
    native_read = []

    def native():
        with np.load(row['packet_export'], allow_pickle=False) as packet:
            mask = unpack(packet['native'])
        native_read.append(True)
        return mask

    result = predict(q, r, cov, score, native_fallback=native)
    masks = result.get('masks', {arm: render(field) for arm, field in result['fields'].items()})
    occurrence = row['occurrence_id']
    prediction_path = directory / 'predictions' / f'{occurrence}.npz'
    field_path = directory / 'fields' / f'{occurrence}.npz'
    np.savez_compressed(prediction_path, **{arm: np.packbits(mask) for arm, mask in masks.items()})
    np.savez_compressed(field_path, **result['fields'],
                        **{'diagnostic_raw_' + key: value for key, value in result.get('diagnostics', {}).items()})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = dict(occurrence_id=occurrence, inputs=source, arms=result['info'],
                   prediction_sha256=sha(prediction_path), field_sha256=sha(field_path),
                   process_peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024),
                   wall_seconds=time.perf_counter() - started, query_gt_opened=False,
                   input_keys_read=['q', 'r', 'cov', 'score'] + (['native'] if native_read else []),
                   full_native_read_only_for_input_condition_fallback=bool(native_read),
                   new_encoder_forwards=0, renderer='FP32 bilinear64_to1024 align_corners=False strict_gt0.5')
    write(directory / 'receipts' / f'{occurrence}.json', receipt)
    return dict(occurrence_id=occurrence, seconds=receipt['wall_seconds'],
                fallback=result['info']['fallback'], hungarian_calls=result['info']['hungarian_calls'],
                process_peak_rss_bytes=receipt['process_peak_rss_bytes'])


def infer(args):
    from ics.experiment import sha
    if args.workers < 1 or args.threads < 1 or args.workers * args.threads > 30:
        raise ValueError('Require positive worker/threads and totalCPU<=30')
    raw = json.loads(args.manifest.read_text())
    rows = raw if isinstance(raw, list) else raw['episodes']
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Empty manifest')
    evaluation, inference = [], []
    for i, original in enumerate(rows):
        row = dict(original)
        for key in ('c', 'fold', 'support', 'query', 'feature_export', 'packet_export'):
            if key not in row:
                raise ValueError('Missing manifest field: ' + key)
        row['occurrence_id'] = f'{i:06d}'
        row['c'], row['fold'] = int(row['c']), int(row['fold'])
        for key in ('feature_export', 'packet_export'):
            path = Path(row[key])
            row[key] = str((path if path.is_absolute() else args.root / path).resolve(strict=True))
        for spec in row.get('evaluation_controls', {}).values():
            path = Path(spec['path'])
            spec['path'] = str((path if path.is_absolute() else args.root / path).resolve(strict=True))
            spec['sha256'] = sha(spec['path'])
        evaluation.append(row)
        # Query truth/class/fold/photos never enter the worker descriptor.
        inference.append({key: row[key] for key in ('occurrence_id', 'feature_export', 'packet_export')})
    args.out.mkdir(parents=True, exist_ok=False)
    for subdir in ('predictions', 'fields', 'receipts'):
        (args.out / subdir).mkdir()
    write(args.out / 'evaluation_manifest.json', evaluation)
    write(args.out / 'inference_manifest.json', inference)
    write(args.out / 'config.json', dict(
        method='Pro_M3_v0', primary='pro_m3_heldout', exposure=args.exposure,
        workers=args.workers, threads=args.threads, encoder_forwards=0,
        source_manifest_sha256=sha(args.manifest), real_gain='unmeasured',
        primary_endpoint='work1024_class_summed_mIoU; original_size_endpoint_not_scored_by_this_cache_adapter',
        native_fallback='lazy bound fullFoRIS native1024 only; never replace with preCRF',
        code_sha256={str(path.relative_to(REPO)): sha(path) for path in
                     (REPO / 'src/ics/methods/pro_role_prediction.py', Path(__file__),
                      REPO / 'src/ics/experiment.py', REPO / 'scripts/run_cpu_feature_candidates.py')}))
    started = time.perf_counter()
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.threads,),
                             mp_context=mp.get_context('spawn')) as executor:
        for result in executor.map(one, [(row, str(args.out.resolve())) for row in inference]):
            receipts.append(result)
            print(json.dumps(dict(done=len(receipts), total=len(rows), **result)), flush=True)
    # Not a sampled simultaneous RSS peak: explicitly label this conservative
    # sum of worker process high-water marks for compatibility with the scorer.
    maximum_rss = max(receipt['process_peak_rss_bytes'] for receipt in receipts)
    write(args.out / 'sealed.json', dict(
        state='ALL_PREDICTIONS_SEALED', n=len(rows),
        **{name + '_sha256': sha(args.out / (name + '.json')) for name in
           ('config', 'inference_manifest', 'evaluation_manifest')},
        receipts={row['occurrence_id']: sha(args.out / 'receipts' / (row['occurrence_id'] + '.json')) for row in rows_with_ids(evaluation)},
        elapsed_seconds=time.perf_counter() - started,
        peak_owned_rss_bytes=maximum_rss * args.workers,
        peak_owned_rss_semantics='maximum_worker_RSS_high_water_times_worker_count; conservative estimate not measured concurrentRSS',
        query_gt_opened=False, encoder_forwards=0))


def rows_with_ids(rows):
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('infer')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--root', type=Path, default=Path('.'))
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--threads', type=int, default=1)
    run.add_argument('--limit', type=int)
    run.add_argument('--exposure', default='exposed_design_cache; not independent confirmation')
    score = commands.add_parser('score')
    score.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'infer':
        infer(args)
    else:
        from run_cpu_feature_candidates import score as common_score
        common_score(args)


if __name__ == '__main__':
    main()
