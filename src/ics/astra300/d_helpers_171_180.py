"""Private utilities for the literal D171--D180 temporary source models."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from .common import EPS
from . import d_helpers_151_200 as dh


def augmented(x):
    return np.c_[x, np.ones(len(x))]


def parameter(model):
    return np.r_[model[0], model[1]]


def balance(ep, fg=None, bg=None):
    fm=ep.wf.copy() if fg is None else ep.wf*np.asarray(fg)
    bm=ep.wb.copy() if bg is None else ep.wb*np.asarray(bg)
    if min(fm.sum(),bm.sum())<=EPS:
        return None
    f=.5*fm/fm.sum(); b=.5*bm/bm.sum(); weights=f+b
    return f/np.maximum(weights,EPS),weights


def loss_gradient(theta,z,y,weights,penalty=0.):
    score=dh.mm(z,theta); residual=weights*(expit(score)-y)
    loss=float(np.sum(weights*(np.logaddexp(0.,score)-y*score)))
    grad=dh.mm(z.T,residual)
    if penalty:
        loss+=.5*penalty*float(theta[:-1]@theta[:-1]);grad[:-1]+=penalty*theta[:-1]
    return loss,grad


def source_penalty(x,weights):
    return .01*max(float(np.sum(weights*np.sum(x*x,axis=1))/max(x.shape[1],1)),EPS)


def oof(ep,xr,fit=None):
    scores=np.full(len(ep.r),np.nan); records=[]
    for train,held in dh.folds(ep):
        reduced=dh.restricted(ep,train)
        model=dh.head(reduced,xr) if fit is None else fit(reduced)
        if model is None:
            continue
        scores[held]=dh.predict(model,xr[held]);records.append((reduced,held,model))
    return scores,records


def mode_ids(ep,k):
    centers,assignment=dh.foreground_modes(ep,k)
    if len(centers):
        assignment=np.argmax(dh.mm(ep.r,centers.T),axis=1)
    return centers,assignment


def recall_safe(ep,before,after,assignment,held):
    return dh.fg_recall_non_decrease(ep,before,after,assignment,held)


def sqdist(x,y):
    return np.maximum(0.,np.sum(x*x,axis=1)[:,None]+np.sum(y*y,axis=1)-2*dh.mm(x,y.T))


def median_width(x,maximum=256):
    points=dh.fps_rows(np.asarray(x,float),min(maximum,len(x)))
    d=sqdist(points,points); values=d[np.triu_indices(len(points),1)]
    values=values[values>EPS*EPS]
    return max(float(np.median(values)) if len(values) else 1.,EPS)


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),EPS)


def trust(value,origin,radius):
    delta=value-origin;norm=float(np.linalg.norm(delta))
    return origin+delta*min(1.,radius/max(norm,EPS))


def profile(ep):
    """Every valid training-R affinity column, retaining original D identity."""
    atoms=ep.r[ep.wvalid>0]
    return (lambda x:dh.mm(x,atoms.T)),len(atoms)


def ridge(x,y,weights,penalty=.01):
    z=augmented(x); gram=dh.mm(z.T,weights[:,None]*z)
    reg=np.eye(z.shape[1])*penalty;reg[-1,-1]=EPS
    return np.linalg.solve(gram+reg,dh.mm(z.T,weights*y))


def lowrank_profile(profile_rows,weights,rank):
    """R-only weighted SVD; no Q label or private held-R column."""
    active=weights>0
    mean=np.average(profile_rows[active],axis=0,weights=weights[active])
    centered=(profile_rows[active]-mean)*np.sqrt(weights[active,None])
    # Economy SVD is a supervised-sample sized computation, never N x N Q.
    _,_,vt=np.linalg.svd(centered,full_matrices=False)
    basis=vt[:min(rank,len(vt))].T
    return mean,basis


def rff(x,width,dimension=64):
    rng=np.random.default_rng(0)
    omega=rng.normal(size=(x.shape[1],dimension))/np.sqrt(width)
    phase=rng.uniform(0.,2*np.pi,size=dimension)
    return lambda rows:np.sqrt(2./dimension)*np.cos(dh.mm(rows,omega)+phase)


def semantic_score(ep,raw_predict,extra=None):
    """Stable row IDs for R avoid matching identical held rows to wrong state."""
    def score(x):
        return raw_predict(x)
    score.source_score=lambda ids:raw_predict(ep.r[ids])
    if extra:
        for key,value in extra.items():
            setattr(score,key,value)
    return score
