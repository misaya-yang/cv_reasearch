#!/usr/bin/env python3
"""Fixed reference convex-hull candidate on cached q/r/cov/base, CPU only.

Infer: python3 scripts/run_reference_hull.py --inputs episode*.npz --out new_run --workers 2
Verify: python3 scripts/run_reference_hull.py --verify new_run
Input contract: frozen aligned q/r:4096x1024, cov/base:64x64. Only these
four NPZ keys are read. GT scoring is a separate operation after sealed.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from run_reference_adjacency import initialize, sha

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def one(job):
    import numpy as np
    import resource
    from ics.experiment import render
    from ics.methods.reference_hull import predict
    from ics.methods.reference_quadratic import predict as quadratic_predict
    occurrence, source, expected_sha, directory = job
    source, directory = Path(source), Path(directory)
    started, cpu_started = time.perf_counter(), time.process_time()
    if sha(source) != expected_sha:
        raise ValueError('Input changed before inference')
    with np.load(source, allow_pickle=False) as data:
        q, r, cov, base = (data[key].copy() for key in ('q', 'r', 'cov', 'base'))
    if q.shape != (4096, 1024) or r.shape != q.shape or cov.shape != (64, 64) or base.shape != cov.shape:
        raise ValueError('Require frozen features4096x1024 and fields64x64')
    result = predict(q, r, cov, base)
    quadratic = quadratic_predict(q, r, cov, base)
    fields = {'reference_hull': result['field'], 'base.control': base,
              'nearest.control': result['nearest_control'], 'centroid.control': result['centroid_control'],
              'linear_span.control': result['subspace_control'], 'affine_span.control': result['affine_control'],
              'reference_quadratic.control': quadratic['field']}
    masks = {name: np.packbits(render(field)) for name, field in fields.items()}
    name = f'{occurrence:06d}'
    prediction = directory / 'predictions' / (name + '.npz')
    field_path = directory / 'fields' / (name + '.npz')
    np.savez_compressed(prediction, **masks)
    np.savez_compressed(field_path, **fields, **result['bounds'])
    if sha(source) != expected_sha:
        raise ValueError('Input changed during inference')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = dict(occurrence=occurrence, input_path=str(source), input_sha256=expected_sha,
                   input_keys_read=['q', 'r', 'cov', 'base'], prediction_sha256=sha(prediction),
                   field_sha256=sha(field_path), method=result['info'], quadratic_control=quadratic['info'],
                   wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu_started,
                   process_peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024),
                   query_gt_opened=False, new_encoder_forwards=0, pid=os.getpid(),
                   finalizer='all fields float32 bilinear1024 align_corners=False then >0.5')
    write(directory / 'receipts' / (name + '.json'), receipt)
    return dict(occurrence=occurrence, seconds=receipt['wall_seconds'], pid=os.getpid())


def verify(directory):
    import numpy as np
    directory = Path(directory)
    seal = json.loads((directory / 'sealed.json').read_text())
    if seal['state'] != 'ALL_PREDICTIONS_SEALED':
        raise ValueError('Predictions are not sealed')
    if sha(directory / 'contract.json') != seal['contract_sha256']:
        raise ValueError('Contract changed after seal')
    contract = json.loads((directory / 'contract.json').read_text())
    expected = {f'{i:06d}' for i in range(len(contract['inputs']))}
    if set(seal['occurrences']) != expected:
        raise ValueError('Seal does not cover all declared occurrences')
    for occurrence, entry in seal['occurrences'].items():
        for category in ('predictions', 'fields', 'receipts'):
            suffix = '.json' if category == 'receipts' else '.npz'
            path = directory / category / (occurrence + suffix)
            if sha(path) != entry[category + '_sha256']:
                raise ValueError('Changed sealed artifact: ' + str(path))
        receipt = json.loads((directory / 'receipts' / (occurrence + '.json')).read_text())
        declared = contract['inputs'][int(occurrence)]
        if (receipt['input_sha256'] != declared['sha256'] or receipt['input_path'] != declared['path']
                or receipt['input_keys_read'] != ['q', 'r', 'cov', 'base'] or receipt['query_gt_opened']):
            raise ValueError('Receipt input isolation or identity mismatch')
        with np.load(directory / 'predictions' / (occurrence + '.npz'), allow_pickle=False) as packet:
            if sorted(packet.files) != sorted(contract['arms']):
                raise ValueError('Unpaired prediction arms')
            if any(packet[key].dtype != np.uint8 or packet[key].shape != (131072,) for key in packet.files):
                raise ValueError('Require packed complete1024 masks')
    return dict(state='SEAL_VERIFIED', occurrences=len(expected), query_gt_opened=False,
                sealed_sha256=sha(directory / 'sealed.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.inputs or args.out:
            parser.error('--verify cannot be combined with inference arguments')
        print(json.dumps(verify(args.verify)), flush=True)
        return
    if not args.inputs or args.out is None:
        parser.error('--inputs and --out are required for inference')
    if not 1 <= args.workers <= 30:
        parser.error('workers must be 1..30, one numerical thread each')
    sys.path.insert(0, str(ROOT / 'src'))
    from ics.methods.reference_hull import Config
    sources = [path.resolve(strict=True) for path in args.inputs]
    if any(not path.is_file() for path in sources):
        parser.error('All input paths must be files')
    inputs = [dict(path=str(path), sha256=sha(path)) for path in sources]
    args.out.mkdir(parents=True, exist_ok=False)
    for category in ('predictions', 'fields', 'receipts'):
        (args.out / category).mkdir()
    code = ('scripts/run_reference_hull.py', 'scripts/run_reference_adjacency.py',
            'src/ics/methods/reference_hull.py', 'src/ics/methods/reference_quadratic.py',
            'src/ics/methods/reference_occupancy.py', 'src/ics/experiment.py')
    contract = dict(method='reference_hull', config=asdict(Config()), independent_methods=1,
                    workers=args.workers, threads_per_worker=1, inputs=inputs, repeats_preserved=True,
                    code_sha256={path: sha(ROOT / path) for path in code},
                    arms=['reference_hull', 'base.control', 'nearest.control', 'centroid.control',
                          'linear_span.control', 'affine_span.control', 'reference_quadratic.control'],
                    input_contract='q/r4096x1024, cov/base64x64; existing frozen MEAN baseline',
                    complete_mask_size=[1024, 1024], query_gt_opened=False, new_encoder_forwards=0,
                    real_gain='unmeasured', real_dataset_minutes='unmeasured')
    write(args.out / 'contract.json', contract)
    jobs = [(i, item['path'], item['sha256'], str(args.out.resolve())) for i, item in enumerate(inputs)]
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            for result in executor.map(one, jobs):
                print(json.dumps(result), flush=True)
        for relative, digest in contract['code_sha256'].items():
            if sha(ROOT / relative) != digest:
                raise ValueError('Inference code changed during run: ' + relative)
        entries = {}
        for i in range(len(jobs)):
            name = f'{i:06d}'
            entries[name] = {category + '_sha256': sha(args.out / category / (name + suffix))
                             for category, suffix in (('predictions', '.npz'), ('fields', '.npz'), ('receipts', '.json'))}
        write(args.out / 'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', occurrences=entries,
              contract_sha256=sha(args.out / 'contract.json'), wall_seconds=time.perf_counter() - started,
              query_gt_opened=False, new_encoder_forwards=0))
        print(json.dumps(verify(args.out)), flush=True)
    except Exception as error:
        write(args.out / 'failed.json', dict(state='INFERENCE_FAILED', error_type=type(error).__name__,
                                             detail=str(error), sealed=False))
        raise


if __name__ == '__main__':
    main()
