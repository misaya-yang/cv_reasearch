"""Exact historical MEAN_a0.25_l16 control, restored without other families.

Source: 82df916:evidence/local/research_20261005/astra_portable/components.py.
The mean_control function body and signature are preserved byte-for-byte.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg
from . import rcg as locked_rcg


@torch.inference_mode()
def mean_control(q,r,cov,score,rcg):
    """Only MEAN_a0.25_l16 from the locked reciprocal_controls implementation."""
    q=F.normalize(torch.as_tensor(q).float(),dim=1);r=F.normalize(torch.as_tensor(r).float(),dim=1)
    fi=np.flatnonzero(np.asarray(cov).ravel()>=.9)
    if not len(fi):fi=np.flatnonzero(np.asarray(cov).ravel()==np.max(cov))
    guide=(q@F.normalize(r[fi].mean(0),dim=0)).numpy()
    s=rcg.minmax(score).ravel();y=(s+.25*(rcg.rank(guide)-rcg.rank(s))).astype(np.float64)
    sim=q@q.T;sim.fill_diagonal_(-2);values,idx=sim.topk(20,dim=1);del sim
    distance=(1-values).clamp_min(0)
    weights=torch.exp(-distance/distance[:,-1:].clamp_min(1e-6)).numpy().ravel()
    w=sparse.csr_matrix((weights,(np.repeat(np.arange(4096),20),idx.numpy().ravel())),shape=(4096,4096))
    w=w.multiply(w.T);w.data=np.sqrt(w.data)
    degree=np.asarray(w.sum(1)).ravel();w=w/max(float(degree.mean()),1e-8);degree=np.asarray(w.sum(1)).ravel()
    a=.1+np.abs(2*s-1);a=(a/a.mean()).astype(np.float64)
    H=sparse.diags(a)+16*(sparse.diags(degree)-w);rhs=a*y;iterations=[0]
    def cb(_):iterations[0]+=1
    z,status=cg(H,rhs,x0=y,rtol=1e-7,atol=1e-9,maxiter=300,callback=cb)
    if status:raise RuntimeError('Locked mean control CG failure: '+str(status))
    info=dict(source_pure_tokens=int(len(fi)),graph_undirected_edges=int(w.nnz//2),cg_iterations=iterations[0],
              cg_relative_residual=float(np.linalg.norm(H@z-rhs)/max(np.linalg.norm(rhs),1e-12)))
    return z.reshape(64,64).astype(np.float32),info


def predict(q, r, cov, score):
    return mean_control(q, r, cov, score, locked_rcg)
