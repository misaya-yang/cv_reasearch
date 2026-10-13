"""Finite last-MLP tangent features of a frozen, reference-defined scalar task.

Let s_i=a^T unit(LN(z_i)), with a fixed per-reference cotangent. Parameters
are the last MLP's fc1/fc2 weights and biases; actual model weights never move.
The empirical gradient kernel is evaluated by four small feature Gram factors,
without materializing one multi-million-parameter Jacobian per image token.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def task_axis(output,coverage,kind='task'):
    e=F.normalize(torch.as_tensor(output).double(),dim=1)
    if kind=='whole_reference':vector=e.mean(0)
    elif kind=='task':
        c=torch.as_tensor(coverage,dtype=torch.float64)
        if c.shape!=(len(e),) or not c.sum()>0 or not (1-c).sum()>0:
            raise ValueError('Reference task requires both area roles')
        vector=(e*c[:,None]).sum(0)/c.sum()-(e*(1-c)[:,None]).sum(0)/(1-c).sum()
    else:raise ValueError('Unknown cotangent')
    if not vector.norm()>0:raise ValueError('Degenerate reference cotangent')
    return F.normalize(vector,dim=0).float()


@torch.inference_mode()
def factors(data,axis,params,device='mps',dtype=torch.float32):
    def t(x):return torch.as_tensor(x,dtype=dtype,device=device)
    z=t(data['block_output']);o=t(data['O24']);u=t(data['mlp_input'])
    v=t(data['mlp_preactivation']);hidden=t(data['mlp_hidden']);a=t(axis)
    gamma=t(params['norm_weight']);scale=t(params['gamma2']);w2=t(params['fc2_weight'])
    eps=float(params['norm_eps'])
    centered=z-z.mean(1,keepdim=True)
    sigma=torch.sqrt(centered.square().mean(1,keepdim=True)+eps)
    standardized=centered/sigma
    unit=F.normalize(o,dim=1);norm=o.norm(dim=1,keepdim=True)
    g=(a[None]-(unit@a)[:,None]*unit)/norm
    g=g*gamma[None]
    through_norm=(g-g.mean(1,keepdim=True)-standardized*(g*standardized).mean(1,keepdim=True))/sigma
    c=through_norm*scale[None]
    gelu_prime=.5*(1+torch.erf(v/math.sqrt(2)))+v*torch.exp(-.5*v.square())/math.sqrt(2*math.pi)
    eta=(c@w2)*gelu_prime
    diagonal=c.square().sum(1)*(hidden.square().sum(1)+1)+eta.square().sum(1)*(u.square().sum(1)+1)
    if not torch.isfinite(diagonal).all() or not (diagonal>0).all():
        raise ValueError('Nonfinite or exactly zero task tangent norm')
    return dict(c=c,eta=eta,hidden=hidden,u=u,diagonal=diagonal)


def subset(factors,ids):
    return {name:value[ids] for name,value in factors.items()}


@torch.inference_mode()
def kernel(left,right,normalize=True):
    value=(left['c']@right['c'].T)*(left['hidden']@right['hidden'].T+1)
    value+=(left['eta']@right['eta'].T)*(left['u']@right['u'].T+1)
    if normalize:value/=torch.sqrt(left['diagonal'][:,None]*right['diagonal'][None,:])
    return value


def kernel_ridge(gram,query_gram,target,weights,regularizer=.01):
    """Weighted centered RKHS regression, unpenalized intercept."""
    g=torch.as_tensor(gram).cpu().double();q=torch.as_tensor(query_gram).cpu().double()
    y=torch.as_tensor(target).cpu().double();a=torch.as_tensor(weights).cpu().double()
    a=a/a.sum();g=(g+g.T)/2
    row_mean=g@a;total=a@row_mean;ym=a@y
    centered=g-row_mean[:,None]-row_mean[None,:]+total
    matrix=centered+torch.diag(regularizer/a)
    coefficient=torch.linalg.solve(matrix,y-ym)
    qcenter=q-(q@a)[:,None]-row_mean[None,:]+total
    result=qcenter@coefficient+ym
    info=dict(normal_residual=float((matrix@coefficient-(y-ym)).norm()/(y-ym).norm().clamp_min(1e-20)),
              reference_mean_target=float(ym),regularizer=regularizer,reference_samples=len(y))
    return result.numpy().astype(np.float32),info
