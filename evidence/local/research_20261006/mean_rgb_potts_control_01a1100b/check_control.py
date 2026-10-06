"""Independent small-cut enumeration/networkx checks plus two CPU workers."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))


def worker(sign):
    import numpy as np
    import torch
    from ics.methods.mean_rgb_potts import predict
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    q=np.ones((4096,8),np.float16);r=q.copy();cov=np.ones((64,64),np.float32)
    base=np.full((64,64),.7 if sign>0 else .3,np.float32)
    # Both signs include the opposite-sign weak one-cell island, so this checks
    # removal and addition. Synthetic expected signs are not query GT fitting.
    base[30:34,30:34]=.49 if sign>0 else .51
    rgb=np.full((128,128,3),127,np.uint8)
    began=time.perf_counter()
    output=predict(q,r,cov,base,rgb,original_shape=(75,109))
    assert output['field'].shape==(128,128)
    assert output['mask_work'].shape==(1024,1024)
    assert output['mask_original'].shape==(75,109)
    assert np.all(output['field']==(1 if sign>0 else 0))
    assert output['info']['grid_added_pixels' if sign>0 else 'grid_deleted_pixels']>0
    return dict(sign=sign,pid=os.getpid(),wall_seconds=time.perf_counter()-began,
                graph_seconds=output['info']['graph_seconds'],
                cut_seconds=output['info']['cut_seconds'],
                render_seconds=output['info']['render_seconds'],
                certificate=output['info']['cut'],graph=output['info']['graph'],
                grid_added_pixels=output['info']['grid_added_pixels'],
                grid_deleted_pixels=output['info']['grid_deleted_pixels'],
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)))


def small_checks():
    import numpy as np
    import networkx as nx
    from ics.methods.mean_rgb_potts import exact_potts_cut,potts_energy,rgb_edge_capacities
    rng=np.random.RandomState(106)
    checks=[]
    for case in range(24):
        rgb=rng.randint(0,256,(2,3,3),dtype=np.uint8)
        edges,cap,info=rgb_edge_capacities(rgb)
        logits=rng.normal(0,.2,6)
        if case==0:logits=np.ones(6)
        if case==1:logits=-np.ones(6)
        if case==2:logits=np.zeros(6)
        mask,cert=exact_potts_cut(logits,edges,cap)
        exhaustive=[]
        for code in range(64):
            z=((code>>np.arange(6))&1).astype(bool)
            exhaustive.append(potts_energy(z,logits,edges,cap))
        independent=nx.DiGraph()
        for i,ell in enumerate(logits):
            independent.add_edge(6,i,capacity=float(np.logaddexp(0,ell)))
            independent.add_edge(i,7,capacity=float(np.logaddexp(0,-ell)))
        for (a,b),weight in zip(edges,cap):
            independent.add_edge(int(a),int(b),capacity=float(weight))
            independent.add_edge(int(b),int(a),capacity=float(weight))
        value,partition=nx.minimum_cut(independent,6,7)
        actual=potts_energy(mask,logits,edges,cap)
        assert abs(actual-min(exhaustive))<1e-10
        assert abs(actual-value)<1e-10
        assert abs(actual-cert['maximum_flow'])<1e-10
        if case==0:assert mask.all()
        if case==1:assert not mask.any()
        if case==2:assert not mask.any()  # Empty foreground resolves exact zero ties.
        checks.append(dict(case=case,energy=actual,enumeration_gap=actual-min(exhaustive),
                           networkx_gap=actual-value,maxflow_certificate_gap=cert['certificate_gap']))
    # A strong RGB edge removes pairwise coupling, preserving both labels.
    rgb=np.zeros((2,3,3),np.uint8);rgb[:,2]=255
    edges,cap,_=rgb_edge_capacities(rgb)
    logits=np.array([.02,.02,-.02,.02,.02,-.02])
    mask,_=exact_potts_cut(logits,edges,cap)
    assert np.array_equal(mask,logits>0)
    return checks


if __name__=='__main__':
    from ics.experiment import sha
    destination=Path(__file__).resolve().parent/'check.json'
    small=small_checks()
    began=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        costs=list(pool.map(worker,[1,-1]))
    receipt=dict(state='SMALL_EXACT_CUT_AND_TWO_WORKER_SYNTHETIC_CONTRACTS_PASSED',
                 workers=2,threads_per_worker=1,independent_small_graph_cases=small,
                 synthetic_rgb128_cases=costs,total_parallel_seconds=time.perf_counter()-began,
                 query_gt_opened=False,new_encoder_forwards=0,
                 real_segmentation_gain='not evaluated',native_MEAN_recompute_cost_included=False,
                 module_sha256=sha(ROOT/'src/ics/methods/mean_rgb_potts.py'),
                 potts_source_sha256=sha(ROOT/'src/ics/methods/pro_paired_environment.py'))
    destination.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,indent=2,allow_nan=False))
