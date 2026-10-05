"""Compose the previously frozen deletion component with conditional additions.

Deletion del[p,w0,t0] is replayed exactly, including its existing C base.
Addition families can recover deleted RCG pixels or complete omitted extent.
All choices use reference-only p1 expected counts. Fixed compositions/quotas and
the unrestricted same-field addition optimum are independent strong controls.
"""
import run_directional_operator as prior
from pixel_budget_control import optimize

CONFIG = dict(name='frozen_delete_p_with_conditional_addition_families',
    fixed_deletion_recipe=['del[p,w0,t0]', 'none'], max_edit_fraction=.05,
    budget_origin='RCG foreground area', budget_fractions=[.25,.5,1.],
    addition_families=['restore_RCG','mean','foris','extent','agreement','union'],
    objective='maximize p1 expected I/U after complete fixed deletion',
    query_gt_in_inference=False, extra_encoder_forwards=0,
    exposure='new construction after inspected1200; development only')


def replay_delete(q, masks, field):
    import torch
    import torch.nn.functional as F
    origin=masks['rcg'];base=masks['Cbase.control'];seed=masks['conservative.control']
    if bool((base & ~origin).any()):raise ValueError('Deletion base must be a subset of RCG')
    cover=lambda mask:mask.reshape(64,16,64,16).float().mean((1,3)).reshape(-1)
    positive=cover(seed)>.5;outside=cover(origin)==0
    scores=torch.as_tensor(field,device='cuda').reshape(-1)
    ids=torch.nonzero(outside)[:,0];ids=ids[torch.argsort(-scores[ids],stable=True)]
    background=torch.zeros_like(positive);background[ids[:min(int(positive.sum()),len(ids))]]=True
    features=torch.as_tensor(q,device='cuda',dtype=torch.float32)
    features=(features/features.norm(dim=1,keepdim=True).clamp_min(1e-12)).double()
    def centroid(bank):
        weights=bank.double();sums=(features*weights[:,None]).sum(0)[None].expand(4096,-1)
        norm=sums.norm(dim=1);valid=(weights.sum()>.5)&(norm>1e-12)
        return ((features*sums).sum(1)/norm.clamp_min(1e-12)).float(),valid
    pos,vpos=centroid(positive);neg,vneg=centroid(background);valid=vpos&vneg
    valid=valid & bool(positive.any()) & bool(background.any())
    up=lambda value:F.interpolate(value.reshape(1,1,64,64).float(),(1024,1024),mode='bilinear',align_corners=False)[0,0].reshape(-1)
    margin=up(torch.where(valid,pos-neg,torch.zeros_like(pos)))
    eligible=up(valid.float())>=1-1e-6
    removed=base & eligible & (margin<0)
    mask=base & ~removed
    # Exact original same-count control: lowest bilinear RCG scores inside C.
    count=int(removed.sum());ids=torch.nonzero(base)[:,0]
    chosen=ids[torch.argsort(up(scores)[ids],stable=True)[:count]]
    matched=base.clone();matched[chosen]=False
    return mask,matched,dict(positive_tokens=int(positive.sum()),background_tokens=int(background.sum()),
                            base_removed=int((origin & ~base).sum()),query_removed=count)


def predict(q,r,cov,comparisons,field):
    import numpy as np
    import torch
    from ics.experiment import unpack
    masks={name:torch.as_tensor(unpack(value).reshape(-1),device='cuda') for name,value in comparisons.items()}
    rcg=masks['rcg'];deleted,matched,deletion_info=replay_delete(q,masks,field)
    budget=int(int(rcg.sum())*CONFIG['max_edit_fraction'])
    p1,p2,info=prior.value_fields(q,r,cov,comparisons,field)
    extent=(masks['native']|masks['mean.control'])&~rcg
    restore=rcg&~deleted
    additions={'restore_RCG':restore,'mean':masks['mean.control']&~deleted,
               'foris':masks['native']&~deleted,'extent':extent,
               'agreement':masks['native']&masks['mean.control']&~deleted}
    additions['union']=(rcg|masks['native']|masks['mean.control'])&~deleted
    z=torch.as_tensor(field,device='cuda').repeat_interleave(16,0).repeat_interleave(16,1).reshape(-1)
    def quota(domain):
        ids=torch.nonzero(domain)[:,0];ids=ids[torch.argsort(-z[ids],stable=True)[:budget]]
        mask=deleted.clone();mask[ids]=True;return mask
    output={'frozen.delete_p.control':deleted,'frozen.delete_p.same_count.control':matched,
            'composed.union_quota.control':quota(additions['union']),
            'composed.restore_quota.control':quota(restore),
            'composed.extent_quota.control':quota(extent)}
    if p1 is None or not budget:
        output.update({'conditional.joint':deleted,'composed.blind.control':deleted,'composed.pixel.control':deleted})
        info.update(budget=budget,add_pixels=0,delete_pixels=int((rcg&~deleted).sum()),deletion=deletion_info)
    else:
        zero={'none':torch.zeros_like(rcg)}
        selected,audit,_,_=prior.solve(deleted,[p1,p2],additions,zero,budget,dual=False)
        # Blind composition retains the previously working addition rule based on RCG.
        old_adds={'mean':masks['mean.control']&~rcg,'foris':masks['native']&~rcg}
        old_adds['agreement']=old_adds['mean']&old_adds['foris']
        old_adds['union']=old_adds['mean']|old_adds['foris']
        old,_,_,_=prior.solve(rcg,[p1,p2],old_adds,zero,budget,dual=False)
        exact,certificate=optimize(p1,deleted,additions['union'],torch.zeros_like(rcg),budget)
        chosen=selected['joint'];intersection=(p1*chosen).sum();union=p1.sum()+chosen.sum()-intersection
        if certificate['expected_iou']+1e-6<float(intersection/union.clamp_min(1e-8)):
            raise ValueError('Pixel control must contain conditional addition family')
        output.update({'conditional.joint':chosen,'composed.blind.control':deleted|(old['joint']&~rcg),
                       'composed.pixel.control':exact})
        info.update(audit,deletion=deletion_info,pixel_control=certificate)
        info['restored_RCG_pixels']=int((chosen&~deleted&rcg).sum())
        info['new_extent_pixels']=int((chosen&~rcg).sum())
    info['fixed_delete_pixels']=int((rcg&~deleted).sum())
    info['final_delete_pixels_vs_RCG']=int((rcg&~output['conditional.joint']).sum())
    output={name:np.packbits(mask.cpu().numpy()) for name,mask in output.items()}
    return output,np.zeros(4096,np.float32) if p1 is None else p1.reshape(64,16,64,16).mean((1,3)).reshape(-1).cpu().numpy(),info
