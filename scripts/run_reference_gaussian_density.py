#!/usr/bin/env python3
"""CPU density-line revision; NPZ inputs q,r,cov,base only; predictions sealed.

base is a bound frozen MEAN field. Optional original_hw is observed geometry.
No GT keys are opened, and no encoder or remote resource is started.
"""
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
    from ics.methods.reference_gaussian_density import predict
    from ics.methods.pro_reference_relations import render_original
    occurrence, source, directory = job
    source, directory = Path(source), Path(directory)
    start = time.perf_counter()
    with np.load(source, allow_pickle=False) as packet:
        q, r, cov, base = (packet[key].copy() for key in ('q', 'r', 'cov', 'base'))
        original_hw = packet['original_hw'].copy() if 'original_hw' in packet.files else None
    result = predict(q, r, cov, base)
    fields = {key: value for key, value in result.items() if key != 'info'}
    fields['base_control'] = base
    masks = {key: np.packbits(render(value)) for key, value in fields.items()}
    if original_hw is not None:
        if (original_hw.shape != (2,) or not np.isfinite(original_hw).all()
                or np.any(original_hw <= 0) or np.any(original_hw != original_hw.astype(int))):
            raise ValueError('original_hw must contain positive integer H/W')
        for key, value in fields.items():
            original, _ = render_original(value, original_hw.astype(int))
            masks[key+'.original'] = np.packbits(original)
    path = directory/f'{occurrence:06d}.npz'
    np.savez_compressed(path, **masks)
    np.savez_compressed(directory/f'{occurrence:06d}.fields.npz', **fields)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = dict(occurrence=occurrence, input_path=str(source), input_sha256=sha(source),
                   input_keys_read=['q', 'r', 'cov', 'base']+(['original_hw'] if original_hw is not None else []),
                   prediction_sha256=sha(path), method=result['info'], query_gt_used=False,
                   complete_prediction_seconds=time.perf_counter()-start,
                   peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
                   original_hw=None if original_hw is None else original_hw.astype(int).tolist(),
                   finalizer='bilinear1024 strict>.5; optional binary-work-mask bilinear-original strict>.5')
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return dict(occurrence=occurrence, seconds=receipt['complete_prediction_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 30:
        parser.error('workers must be 1..30, each worker uses one numerical thread')
    sources = [path.resolve(strict=True) for path in args.inputs]
    args.out.mkdir(parents=True, exist_ok=False)
    contract = dict(method='reference_density_full_feature_bounded_v2', independent_method_increment=0,
                    revised_family='reference_score_density_query_prior_v1', workers=args.workers,
                    threads_per_worker=1, source_sha256=sha(ROOT/'src/ics/methods/reference_gaussian_density.py'),
                    prototype_source_sha256=sha(ROOT/'src/ics/methods/reference_occupancy.py'),
                    quadratic_control_source_sha256=sha(ROOT/'src/ics/methods/reference_quadratic.py'),
                    original_renderer_source_sha256=sha(ROOT/'src/ics/methods/pro_reference_relations.py'),
                    render_source_sha256=sha(ROOT/'src/ics/experiment.py'),
                    worker_initializer_sha256=sha(ROOT/'scripts/run_reference_adjacency.py'),
                    runner_sha256=sha(__file__), inputs=[str(path) for path in sources], repeats_preserved=True,
                    query_gt_used=False, new_encoder_forwards=0, real_segmentation_gain='unmeasured')
    (args.out/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    jobs = [(i, str(source), str(args.out.resolve())) for i, source in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        for result in executor.map(one, jobs):
            print(json.dumps(result), flush=True)
    (args.out/'prediction_complete.json').write_text(json.dumps(dict(occurrences=len(jobs), query_gt_used=False))+'\n')


if __name__ == '__main__':
    main()
