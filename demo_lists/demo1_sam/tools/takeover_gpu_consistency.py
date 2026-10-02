"""Same full-resolution six-permutation matching on GPU, without exporting views."""
import itertools
import torch


def gpu_consistency(original, scores, views):
    # original[P,4,H,W], views[V,P,4,H,W]. Binary dot products are exact FP32
    # counts for these small images; divide and rank in FP64 like NumPy.
    if original.dtype != torch.bool or views.dtype != torch.bool:
        raise ValueError("binary masks required")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("disable TF32 to retain exact binary intersection counts")
    if original.shape[-1]*original.shape[-2] > 2**24:
        raise ValueError("image too large for exact FP32 intersection counts")
    v,p=views.shape[:2]
    a=original[:,1:].flatten(2).float()
    b=views[:,:,1:].permute(1,0,2,3,4).reshape(p,v*3,-1).float()
    intersections=torch.bmm(a,b.transpose(1,2)).reshape(p,3,v,3).permute(0,2,1,3).double()
    areas=a.sum(-1).double()
    view_areas=b.sum(-1).reshape(p,v,3).double()
    union=areas[:,None,:,None]+view_areas[:,:,None,:]-intersections
    matrix=torch.where(union>0,intersections/union.clamp_min(1),torch.ones_like(union))
    perms=list(itertools.permutations(range(3)))
    objectives=torch.stack([sum(matrix[:,:,j,q[j]] for j in range(3)) for q in perms],-1)
    matched=torch.tensor(perms,device=original.device)[objectives.argmax(-1)]
    consistency=matrix.gather(-1,matched[...,None]).squeeze(-1).mean(1)
    chosen=consistency.argmax(-1)+1
    baseline=scores[:,1:].argmax(-1)+1
    informative=(consistency.max(-1).values-consistency.min(-1).values)>1e-8
    return torch.where(informative,chosen,baseline),consistency
