#!/usr/bin/env python3
"""Fixed CPU graph candidate; NPZ keys y,a,edge_i,edge_j,edge_w only.

y is the frozen MEAN pre-graph reference-guided target, a its fidelity,
edge_i<edge_j and edge_w are its unique undirected normalized graph edges.
Alternatively NPZ keys q,r,cov,score reconstruct the same pre-graph inputs.
The input is not the already-smoothed MEAN field. Query GT is never opened.
Example: python scripts/run_huber_graph.py --inputs inputs/*.npz --out new_run --workers 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

from run_reference_adjacency import initialize, sha

ROOT = Path(__file__).resolve().parents[1]


def one(job):
    import numpy as np
    import resource
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.experiment import render
    from ics.methods.huber_graph import Config, make_mean_inputs, predict, quadratic_control
    import time
    occurrence, source, directory = job
    source, directory = Path(source), Path(directory)
    whole_start=time.perf_counter()
    with np.load(source,allow_pickle=False) as packet:
        if 'y' in packet.files:
            keys=['y','a','edge_i','edge_j','edge_w']
            y,a,left,right,weights = (packet[key].copy() for key in keys)
            origin=dict(source='existing_pregraph_packet')
        else:
            keys=['q','r','cov','score']
            inputs,origin=make_mean_inputs(*(packet[key].copy() for key in keys))
            y,a,left,right,weights=inputs
    field, info = predict(y,a,left,right,weights)
    boxed, boxed_info = quadratic_control(y,a,left,right,weights)
    # Original MEAN quadratic solve is unboxed; include it to isolate the box constraint.
    n = y.size
    costs = Config().graph_lambda * weights
    degree = np.bincount(np.r_[left,right],weights=np.r_[costs,costs],minlength=n)
    matrix = sparse.coo_matrix((np.r_[a.ravel()+degree,-costs,-costs],
                               (np.r_[np.arange(n),left,right],np.r_[np.arange(n),right,left])),
                              shape=(n,n)).tocsr()
    original,status = cg(matrix,a.ravel()*y.ravel(),x0=y.ravel(),rtol=1e-7,atol=1e-9,maxiter=300)
    if status:
        raise RuntimeError(f'Original quadratic control failed: {status}')
    fields = dict(huber=field,boxed_quadratic_control=boxed,
                  original_quadratic_control=original.reshape(y.shape),pregraph_control=y)
    path = directory/f'{occurrence:06d}.npz'
    np.savez_compressed(path,**{key:np.packbits(render(value)) for key,value in fields.items()},
                        huber_field=field,boxed_field=boxed,original_field=original.reshape(y.shape))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = dict(occurrence=occurrence,input_path=str(source),input_sha256=sha(source),
                   input_keys_read=keys,input_origin=origin,
                   prediction_sha256=sha(path),method=info,boxed_control=boxed_info,
                   original_control_relative_residual=float(np.linalg.norm(matrix@original-a.ravel()*y.ravel())
                                                             /max(np.linalg.norm(a.ravel()*y.ravel()),1e-12)),
                   peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024),
                   query_gt_used=False,new_encoder_forwards=0,
                   complete_prediction_seconds=time.perf_counter()-whole_start,
                   finalizer='all fields: bilinear1024 align_corners=False then >0.5')
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return dict(occurrence=occurrence,solver_seconds=info['wall_seconds'],gap=info['gap'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args = parser.parse_args()
    if not 1<=args.workers<=30:
        parser.error('workers must be 1..30, one numerical thread each')
    sources=[p.resolve(strict=True) for p in args.inputs]
    args.out.mkdir(parents=True,exist_ok=False)
    contract=dict(method='huber_graph',independent_methods=1,workers=args.workers,threads_per_worker=1,
                  source_sha256=sha(ROOT/'src/ics/methods/huber_graph.py'),
                  worker_initializer_sha256=sha(ROOT/'scripts/run_reference_adjacency.py'),
                  render_source_sha256=sha(ROOT/'src/ics/experiment.py'),runner_sha256=sha(__file__),
                  inputs=[str(p) for p in sources],repeats_preserved=True,query_gt_used=False,
                  input_requirement='same frozen MEAN pre-graph y/a/undirected graph; no already-smoothed base',
                  real_dataset_runtime='unmeasured before execution',real_segmentation_gain='unmeasured')
    (args.out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    jobs=[(i,str(source),str(args.out.resolve())) for i,source in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        for result in executor.map(one,jobs):
            print(json.dumps(result),flush=True)
    (args.out/'prediction_complete.json').write_text(json.dumps(dict(occurrences=len(jobs),query_gt_used=False))+'\n')


if __name__=='__main__':
    main()
