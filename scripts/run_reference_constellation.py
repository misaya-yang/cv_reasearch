#!/usr/bin/env python3
"""Fixed CPU landmark candidate; NPZ keys q,r,cov,base only.

Frozen aligned DINO features q/r:4096x1024, complete reference coverage cov:64x64,
existing MEAN field base:64x64. No query GT is opened.
Example: python scripts/run_reference_constellation.py --inputs inputs/*.npz --out new_run --workers 6
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing
from pathlib import Path
import sys

from run_reference_adjacency import initialize,sha

ROOT=Path(__file__).resolve().parents[1]


def one(job):
    import numpy as np
    import resource
    import time
    from ics.experiment import render
    from ics.methods.reference_constellation import predict
    occurrence,source,directory=job
    source,directory=Path(source),Path(directory)
    started=time.perf_counter()
    with np.load(source,allow_pickle=False) as packet:
        q,r,cov,base=(packet[key].copy() for key in ('q','r','cov','base'))
    if q.shape!=(4096,1024) or r.shape!=q.shape or cov.shape!=(64,64) or base.shape!=cov.shape:
        raise ValueError('Primary contract requires frozen features4096x1024 and fields64x64')
    result=predict(q,r,cov,base)
    fields=dict(constellation=result['field'],base_control=base,
                clipped_base_control=np.clip(base,0,1),bag_mode_control=result['bag_field'],
                aligned_prior_control=result['prior'])
    path=directory/f'{occurrence:06d}.npz'
    np.savez_compressed(path,**{key:np.packbits(render(value)) for key,value in fields.items()},
                        output_field=result['field'],aligned_prior=result['prior'])
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt=dict(occurrence=occurrence,input_path=str(source),input_sha256=sha(source),
                 input_keys_read=['q','r','cov','base'],prediction_sha256=sha(path),method=result['info'],
                 complete_prediction_seconds=time.perf_counter()-started,
                 peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024),
                 query_gt_used=False,new_encoder_forwards=0,
                 finalizer='all fields bilinear1024 align_corners=False then >0.5')
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return dict(occurrence=occurrence,seconds=receipt['complete_prediction_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=30:parser.error('workers must be 1..30, one numerical thread each')
    sources=[p.resolve(strict=True) for p in args.inputs]
    args.out.mkdir(parents=True,exist_ok=False)
    contract=dict(method='reference_constellation',independent_methods=1,
                  workers=args.workers,threads_per_worker=1,
                  source_sha256=sha(ROOT/'src/ics/methods/reference_constellation.py'),
                  clustering_source_sha256=sha(ROOT/'src/ics/methods/reference_occupancy.py'),
                  worker_initializer_sha256=sha(ROOT/'scripts/run_reference_adjacency.py'),
                  render_source_sha256=sha(ROOT/'src/ics/experiment.py'),runner_sha256=sha(__file__),
                  inputs=[str(p) for p in sources],repeats_preserved=True,query_gt_used=False,
                  real_dataset_gain='unmeasured',real_dataset_minutes='unmeasured')
    (args.out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    jobs=[(i,str(source),str(args.out.resolve())) for i,source in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        for result in executor.map(one,jobs):print(json.dumps(result),flush=True)
    (args.out/'prediction_complete.json').write_text(json.dumps(dict(occurrences=len(jobs),query_gt_used=False))+'\n')


if __name__=='__main__':main()
