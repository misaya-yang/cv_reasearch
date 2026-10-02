"""Deployment head: coefficient contraction plus real selected-parent upsampling."""
import math
import torch
from torch import nn
from takeover_head_fallback import grid


def load_student(path,device):
    state=torch.load(path,weights_only=True,map_location=device)
    assert state['normalization_folded']
    student=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,64)).to(device).eval()
    student.load_state_dict(state['state'])
    return student,state['feature_center'],state['basis']


def head(model,student,x,hyper,center,basis,budget):
    p=len(x);coeff=student(x)
    weights=torch.einsum('pmc,ucr->pmur',hyper,basis.reshape(16,32,64)).reshape(p,64,64)
    bias=torch.einsum('pmc,uc->pmu',hyper,center.reshape(16,32)).reshape(p,64)
    ph=(torch.bmm(coeff,weights.transpose(1,2))+bias[:,None]).reshape(p,4096,4,16)
    if budget:
        margin=(ph.abs()/hyper.norm(dim=-1)[:,None,:,None].clamp_min(1e-6)).amin((-2,-1))
        indices=margin.topk(math.ceil(4096*budget),dim=-1,largest=False).indices
        batch=torch.arange(p,device=x.device)[:,None]
        patches=model.output_upscaling(x[batch,indices].reshape(-1,256,1,1)).flatten(2)
        h=hyper[:,None].expand(-1,indices.shape[1],-1,-1).reshape(-1,4,32)
        ph[batch,indices]=torch.bmm(h,patches).reshape(p,indices.shape[1],4,16)
    return grid(ph)


def decode(model,student,image,pe,dense,sparse,center,basis,budget,phase_cache=None):
    # Official SAM1 two-way transformer and IoU/hypernetworks; only the
    # upsampling/mask contraction is replaced. No full upscaled grid is built.
    output=torch.cat([model.iou_token.weight,model.mask_tokens.weight],dim=0)
    tokens=torch.cat([output[None].expand(len(sparse),-1,-1),sparse],dim=1)
    src=torch.repeat_interleave(image,len(sparse),dim=0)+dense
    pos=torch.repeat_interleave(pe,len(sparse),dim=0)
    hs,x=model.transformer(src,pos,tokens)
    scores=model.iou_prediction_head(hs[:,0])
    hyper=torch.stack([mlp(hs[:,i+1]) for i,mlp in enumerate(model.output_hypernetworks_mlps)],1)
    if phase_cache is not None:
        from takeover_compact_phase_fallback import head as phase_head
        return phase_head(model,phase_cache,student,x,hyper,center,basis,budget),scores
    return head(model,student,x,hyper,center,basis,budget),scores
