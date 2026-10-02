"""Original-view candidate features and task metrics; labels never enter features."""
import torch
from torch.nn import functional as F


def candidate_features(low, scores, tokens, sparse, full, regime):
    z=low[:,1:].float();m=z>0;p=z.shape[0]
    prob=z.sigmoid()
    occ=torch.stack([(z>t).float().mean((-2,-1)) for t in (-2,-1,0,1,2)],-1)
    flat=m.flatten(2).float()
    inter=torch.bmm(flat,flat.transpose(1,2))
    areas=flat.sum(-1)
    union=areas[:,:,None]+areas[:,None,:]-inter
    pair=torch.where(union>0,inter/union.clamp_min(1),torch.ones_like(union))
    full_area=full[:,1:].float().mean((-2,-1))
    stability=occ[...,3]/occ[...,1].clamp_min(1e-6)
    stats=torch.stack([z.mean((-2,-1))/10,z.std((-2,-1))/10,
                       prob.mean((-2,-1)),(prob*(1-prob)).mean((-2,-1)),
                       stability,full_area,full_area/full_area.max(-1,keepdim=True).values.clamp_min(1e-6),
                       scores[:,1:],scores[:,1:]-scores[:,1:].max(-1,keepdim=True).values],-1)
    pool=F.adaptive_avg_pool2d(prob.reshape(p*3,1,256,256),(4,4)).reshape(p,3,16)
    onehot=torch.zeros(p,3,3,device=z.device);onehot[:,:,('central','near_boundary','box').index(regime)]=1
    geometry=torch.cat([occ,stats,pool,pair.sort(-1).values,onehot],-1)
    # Same scorer applies to all candidates. No explicit candidate index, GT size,
    # category, erosion depth, or quality labels are supplied.
    latent=torch.cat([tokens[:,2:5],tokens[:,0:1].expand(-1,3,-1),sparse.mean(1)[:,None].expand(-1,3,-1)],-1)
    return torch.cat([geometry,latent],-1),geometry.shape[-1]


def boundary(mask,radius):
    x=F.pad(mask.float(),(radius,radius,radius,radius),value=0)
    inv=1-x
    inv=F.max_pool2d(inv,(1,2*radius+1),stride=1)
    inv=F.max_pool2d(inv,(2*radius+1,1),stride=1)
    return mask & (inv>0)


def quality(masks,gt,valid=None):
    # Full original-resolution task metrics; image/sequence grouping is external.
    a=masks[:,1:];b=gt[:,None]
    if valid is not None:
        a=a&valid[None,None];b=b&valid[None,None]
    intersection=(a&b).sum((-2,-1)).double();union=(a|b).sum((-2,-1)).double()
    values=torch.where(union>0,intersection/union.clamp_min(1),torch.ones_like(union))
    radius=max(1,round(.02*(gt.shape[-2]**2+gt.shape[-1]**2)**.5))
    aa=boundary(a,radius);bb=boundary(b,radius)
    bi=(aa&bb).sum((-2,-1)).double();bu=(aa|bb).sum((-2,-1)).double()
    return values,torch.where(bu>0,bi/bu.clamp_min(1),torch.ones_like(bu))
