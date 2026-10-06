"""Synthetic raw-feature pre-graph reconstruction versus existing MEAN producer."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import cg
from ics.methods.huber_graph import make_mean_inputs
from ics.methods.mean_graph import predict


def main():
    rng=np.random.RandomState(20261006)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    q=rng.normal(size=(4096,1024)).astype(np.float32)
    r=rng.normal(size=(4096,1024)).astype(np.float32)
    cov=np.zeros((64,64));cov[20:44,20:44]=1
    score=rng.uniform(0,1,size=(64,64)).astype(np.float32)
    (y,a,left,right,w),origin=make_mean_inputs(q,r,cov,score)
    costs=16*w;n=y.size
    degree=np.bincount(np.r_[left,right],weights=np.r_[costs,costs],minlength=n)
    matrix=sparse.coo_matrix((np.r_[a.ravel()+degree,-costs,-costs],
                             (np.r_[np.arange(n),left,right],np.r_[np.arange(n),right,left])),shape=(n,n)).tocsr()
    z,status=cg(matrix,a.ravel()*y.ravel(),x0=y.ravel(),rtol=1e-7,atol=1e-9,maxiter=300)
    assert status==0
    old,_=predict(q,r,cov,score,device='cpu')
    error=float(np.max(np.abs(z.reshape(64,64)-old)))
    assert error<1e-6,error
    (constant,*_),constant_origin=make_mean_inputs(q,r,cov,np.ones((64,64),dtype=np.float32))
    assert not constant.any() and constant_origin['source_constant_empty_rule']
    report=dict(synthetic_cache_shape=[4096,1024],reference_modes_not_required=True,
                original_mean_maximum_field_difference=error,exact_bitwise_parity_claimed=False,
                pregraph=origin,constant_source_empty_checked=True,real_episodes=0,no_gpu=True,no_server=True)
    Path(__file__).with_name('raw_input_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
