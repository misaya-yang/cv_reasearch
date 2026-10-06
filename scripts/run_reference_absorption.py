#!/usr/bin/env python3
"""CPU reference-absorption residual and fixed simple controls; read q/r/cov/base."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing
from pathlib import Path
import sys

from run_reference_adjacency import initialize, sha

ROOT = Path(__file__).resolve().parents[1]


def one(job):
    import numpy as np
    import resource
    import time
    from ics.experiment import render
    from ics.methods.reference_absorption import predict
    index, source, destination = job
    source, destination = Path(source), Path(destination)
    started = time.perf_counter()
    with np.load(source, allow_pickle=False) as data:
        q, r, cov, base = (data[key].copy() for key in ('q', 'r', 'cov', 'base'))
    if q.shape != (4096, 1024) or r.shape != q.shape or base.shape != (64, 64) or cov.shape != base.shape:
        raise ValueError('Require frozen q/r4096x1024 and cov/base64x64')
    result = predict(q, r, cov, base)
    fields = {'reference_absorption': result['field'], 'mean.control': base}
    for key in ('nearest_control', 'kernel_control', 'one_hop_control', 'one_step_control',
                'component_control', 'full_harmonic_control'):
        fields['absorption_' + key.replace('_control', '.control')] = result[key]
    path = destination / f'{index:06d}.npz'
    np.savez_compressed(path, **{key: np.packbits(render(value)) for key, value in fields.items()},
                        **{'field_' + key: value for key, value in fields.items()})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = dict(occurrence=index, input_path=str(source), input_sha256=sha(source),
                   input_keys_read=['q', 'r', 'cov', 'base'], prediction_sha256=sha(path),
                   method=result['info'], query_gt_used=False, new_encoder_forwards=0,
                   wall_seconds=time.perf_counter()-started,
                   process_peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
                   finalizer='all fields bilinear1024 align_corners=False then >0.5')
    receipt_path = path.with_suffix('.json')
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return dict(occurrence=index, prediction_sha256=receipt['prediction_sha256'],
                receipt_sha256=sha(receipt_path), seconds=receipt['wall_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 30:
        parser.error('workers must be 1..30 with one numerical thread each')
    inputs = [path.resolve(strict=True) for path in args.inputs]
    args.out.mkdir(parents=True, exist_ok=False)
    contract = dict(method='reference_anchor_absorption_residual_v1', independent_method_count=1,
                    workers=args.workers, threads_per_worker=1,
                    source_sha256=sha(ROOT/'src/ics/methods/reference_absorption.py'),
                    compression_sha256=sha(ROOT/'src/ics/methods/reference_occupancy.py'),
                    renderer_sha256=sha(ROOT/'src/ics/experiment.py'),
                    initializer_sha256=sha(ROOT/'scripts/run_reference_adjacency.py'),
                    runner_sha256=sha(__file__), inputs=[str(path) for path in inputs],
                    repeats_preserved=True, query_gt_used=False, real_gain='unmeasured')
    (args.out/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    jobs = [(i, str(source), str(args.out.resolve())) for i, source in enumerate(inputs)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        for result in executor.map(one, jobs):
            results.append(result)
            print(json.dumps(result), flush=True)
    (args.out/'prediction_complete.json').write_text(json.dumps(
        dict(method=contract['method'], occurrences=len(jobs), query_gt_used=False,
             contract_sha256=sha(args.out/'contract.json'), predictions=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
