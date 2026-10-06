#!/usr/bin/env python3
"""CPU cached-feature candidate, fixed contract; predicts before any GT scoring.

Each input NPZ provides q, r, cov, base. q/r are aligned H*W,D features;
cov/base are aligned H,W fields. base is the existing frozen MEAN field.
No query labels are loaded. Extra NPZ keys are not opened.
Example:
  python scripts/run_reference_adjacency.py --inputs episode_*.npz --out new_run --workers 6
A new output directory is required. This command starts no server or encoder.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def initialize():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def one(job):
    import numpy as np
    import resource
    import torch
    from ics.experiment import render
    from ics.methods.reference_adjacency import predict
    occurrence, source, directory = job
    source, directory = Path(source), Path(directory)
    with np.load(source, allow_pickle=False) as data:
        q, r, cov, base = (data[key].copy() for key in ('q', 'r', 'cov', 'base'))
    result = predict(q, r, cov, base)
    # Default exact MAP readout: nearest-neighbor expansion of the token solution.
    # Bilinear max-marginal readout is an explicitly named control, not a new method.
    import torch.nn.functional as F
    exact = F.interpolate(torch.from_numpy(result['token_mask'].astype(np.float32))[None, None],
                          (1024, 1024), mode='nearest')[0, 0].numpy() > .5
    masks = dict(adjacency_map=exact, base_control=render(base),
                 adjacency_bilinear_control=render(result['field']),
                 base_nearest_control=F.interpolate(torch.from_numpy((base > .5).astype(np.float32))[None, None],
                                                    (1024, 1024), mode='nearest')[0,0].numpy() > .5)
    path = directory / f'{occurrence:06d}.npz'
    np.savez_compressed(path, **{key: np.packbits(value) for key, value in masks.items()},
                        token_map=result['token_mask'], field=result['field'])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = dict(occurrence=occurrence, input_path=str(source), input_sha256=sha(source),
                   input_keys_read=['q', 'r', 'cov', 'base'], prediction_sha256=sha(path),
                   peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024),
                   finalizer='exact token MAP -> nearest1024; bilinear readout is a control',
                   method=result['info'])
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return dict(occurrence=occurrence, path=str(path), seconds=result['info']['wall_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 30:
        parser.error('workers must be 1..30; each worker uses one numerical thread')
    sources = [p.resolve(strict=True) for p in args.inputs]
    args.out.mkdir(parents=True, exist_ok=False)
    contract = dict(method='reference_adjacency', independent_methods=1,
                    source_sha256=sha(ROOT / 'src/ics/methods/reference_adjacency.py'),
                    shared_tree_source_sha256=sha(ROOT / 'src/ics/methods/reference_occupancy.py'),
                    render_source_sha256=sha(ROOT / 'src/ics/experiment.py'),
                    runner_sha256=sha(Path(__file__)),
                    inputs=[str(p) for p in sources], workers=args.workers, threads_per_worker=1,
                    base_requirement='existing frozen MEAN field, same producer/protocol for all rows',
                    query_gt_used=False, new_encoder_forwards=0,
                    repeats_preserved=True, real_runtime='unmeasured before this execution')
    (args.out / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    jobs = [(i, str(source), str(args.out.resolve())) for i, source in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        for completed in executor.map(one, jobs):
            print(json.dumps(completed), flush=True)
    (args.out / 'prediction_complete.json').write_text(json.dumps(dict(occurrences=len(jobs), query_gt_used=False)) + '\n')


if __name__ == '__main__':
    main()
