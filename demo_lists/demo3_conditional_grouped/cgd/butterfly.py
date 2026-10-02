import math
import torch

def butterfly(rgb, angles, inverse=False):
    """rgb [..., P, 3]; angles [..., log2(P), P/2]; colors share rotations."""
    p=rgb.shape[-2];levels=int(math.log2(p))
    if p!=2**levels or angles.shape[-2:]!=(levels,p//2):raise ValueError('Power-of-two P and matching angles required')
    x=rgb
    for level in (range(levels-1,-1,-1) if inverse else range(levels)):
        stride=2**level
        blocks=x.reshape(*x.shape[:-2],p//(2*stride),2,stride,3)
        a,b=blocks.unbind(-3)
        theta=angles[...,level,:].reshape(*angles.shape[:-2],p//(2*stride),stride,1)
        if inverse:theta=-theta
        c,s=theta.cos(),theta.sin()
        x=torch.stack((c*a-s*b,s*a+c*b),dim=-3).reshape_as(x)
    return x

def group_indices(p,g,device):
    if p%g:raise ValueError('Groups must partition the patch')
    i=torch.arange(p,device=device)[:,None];j=torch.arange(g,device=device)[None,:]
    return (i//g)*g+(i%g+j)%g
