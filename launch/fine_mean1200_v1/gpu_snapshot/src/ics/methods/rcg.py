"""Locked RCG equations, with dense matrix operations on the selected device.

Source: retained validation121_locked/locked_methods/rcg_readout.py.
Sparse graph construction/CG remain CPU. CUDA reduction parity is unverified;
this is not a claim of bit-identical CUDA and original CPU outputs.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.stats import rankdata
from scipy.sparse.csgraph import connected_components

CONFIG={'arm':'RCG','alpha':0.5,'cross_image_k':10,'query_k':20,'source_purity':0.9,
        'lambda':16.0,'confidence_floor':0.1,'query_gt_in_inference':False,
        'native_mask_in_inference':False,'finalizer':'bilinear 1024, align_corners=False, >0.5; no re-minmax',
        'parameter_provenance':'historical development-selected constants; fixed here; no new fold selection performed',
        'dense_device':'caller-selected; GPU numerical parity not yet verified',
        'cg_rtol':1e-7,'cg_atol':1e-9,'cg_maxiter':300}

def rank(x: np.ndarray) -> np.ndarray:
    return ((rankdata(x.ravel(),method='average')-0.5)/x.size).astype(np.float32)

def minmax(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float32)
    return (x-x.min())/max(float(x.max()-x.min()),1e-6)

def mask_from_field(z: np.ndarray) -> np.ndarray:
    x=torch.from_numpy(np.ascontiguousarray(z,dtype=np.float32))[None,None]
    return F.interpolate(x,(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy()>.5

def unpack(x: np.ndarray) -> np.ndarray:
    if x.dtype!=np.uint8 or x.size!=131072:raise ValueError('Expected 1024x1024 np.packbits uint8 mask')
    return np.unpackbits(x).reshape(1024,1024).astype(bool)

@torch.inference_mode()
def predict(q: torch.Tensor|np.ndarray,r: torch.Tensor|np.ndarray,
            cov: np.ndarray,score: np.ndarray, *, device='cpu', extras=None) -> tuple[np.ndarray,dict[str,Any]]:
    """Return a 64x64 float32 field and solver metadata. No evaluation inputs."""
    q=torch.as_tensor(q,device=device).float();r=torch.as_tensor(r,device=device).float()
    cov=np.asarray(cov);score=np.asarray(score)
    if q.shape!=(4096,1024) or r.shape!=(4096,1024) or cov.shape!=(64,64) or score.shape!=(64,64):
        raise ValueError('Invalid cache shapes; regular 64x64 / 1024x1024 mapping is required')
    if not (torch.isfinite(q).all() and torch.isfinite(r).all() and np.isfinite(cov).all() and np.isfinite(score).all()):
        raise ValueError('Nonfinite inputs')
    if cov.min()<0 or cov.max()>1:raise ValueError('Reference coverage must lie in [0,1]')
    if (q.norm(dim=1)==0).any() or (r.norm(dim=1)==0).any():raise ValueError('Zero-norm tokens')
    q=F.normalize(q,dim=1);r=F.normalize(r,dim=1)
    fi=np.flatnonzero(cov.ravel()>=CONFIG['source_purity'])
    if not len(fi):fi=np.flatnonzero(cov.ravel()==cov.max())
    sim=q@r.T
    dq=sim.topk(CONFIG['cross_image_k'],dim=1).values.mean(1)
    dr=sim.topk(CONFIG['cross_image_k'],dim=0).values.mean(0)
    guide=((2*sim[:,fi]-dr[fi][None,:]).max(1).values-dq).cpu().numpy()
    del sim,dr,dq,r
    s=minmax(score).ravel()
    y=(s+CONFIG['alpha']*(rank(guide)-rank(s))).astype(np.float64)
    # Directed adaptive 20-NN query graph, then retain reciprocal edges only.
    sim=q@q.T;sim.fill_diagonal_(-2)
    values,idx=sim.topk(CONFIG['query_k'],dim=1);del sim,q
    distance=(1-values).clamp_min(0)
    weights=torch.exp(-distance/distance[:,-1:].clamp_min(1e-6)).cpu().numpy().ravel()
    w=sparse.csr_matrix((weights,(np.repeat(np.arange(4096),CONFIG['query_k']),idx.cpu().numpy().ravel())),shape=(4096,4096))
    w=w.multiply(w.T);w.data=np.sqrt(w.data)
    degree=np.asarray(w.sum(1)).ravel();w=w/max(float(degree.mean()),1e-8)
    degree=np.asarray(w.sum(1)).ravel()
    a=CONFIG['confidence_floor']+np.abs(2*s-1);a=a/a.mean();a=a.astype(np.float64)
    H=sparse.diags(a)+CONFIG['lambda']*(sparse.diags(degree)-w)
    rhs=a*y;iterations=[0]
    def callback(_:np.ndarray)->None:iterations[0]+=1
    z,status=cg(H,rhs,x0=y,rtol=CONFIG['cg_rtol'],atol=CONFIG['cg_atol'],maxiter=CONFIG['cg_maxiter'],callback=callback)
    if status!=0:raise RuntimeError(f'CG did not converge: status={status}')
    residual=float(np.linalg.norm(H@z-rhs)/max(np.linalg.norm(rhs),1e-12))
    ncomp,labels=connected_components(w,directed=False)
    info={'source_pure_tokens':int(len(fi)),'graph_undirected_edges':int(w.nnz//2),'graph_components':int(ncomp),
          'graph_isolated_tokens':int((degree==0).sum()),'cg_iterations':iterations[0],'cg_relative_residual':residual,
          'weighted_mean_error':float(abs(np.dot(a,z-y))),
          'maximum_principle_error':float(max(0,z.max()-y.max(),y.min()-z.min()))}
    return z.reshape(64,64).astype(np.float32),info
