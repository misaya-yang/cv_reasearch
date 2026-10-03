"""Diagnostic V->U reread in the SAME native value coordinate system.

Input is a complete_pair_evidence_extract package, or its tensor dictionary with
_sdpa_metadata attached. U is recomputed from captured post-RoPE native Q/K/V,
not a saved original SDPA output; backend/stride differences can change bits.
No encoder intervention, cross-image QK attention, new head, or stored NxN matrix.
"""
from contextlib import nullcontext
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

_LAST_AUDIT={}


def get_last_audit():
    return dict(_LAST_AUDIT)


def _inputs(package,layer,device):
    features=package.get('features',package)
    metadata=package.get('metadata',{}).get('sdpa',features.get('_sdpa_metadata'))
    if metadata is None:raise ValueError('recorded native SDPA metadata required; pass complete cache package')
    index=layer-1;record=metadata.get(index,metadata.get(str(index)))
    if record is None:raise ValueError('native SDPA record missing for block%d'%layer)
    if record.get('mask_shape') is not None or record.get('is_causal',False):
        raise ValueError('this fixed diagnostic requires original unmasked noncausal image attention')
    if float(record.get('dropout_p',0))!=0:raise ValueError('original inference dropout must be zero')
    q,k,v=[features['block%d_%s_postRoPE'%(layer,name)].to(device) for name in ('q','k','v')]
    if q.ndim!=4 or q.shape!=k.shape or q.shape!=v.shape or q.shape[0]!=2:
        raise ValueError('full SQ batch2 native QKV [2,H,N,d] required')
    return q,k,v,record.get('scale')


def _sdpa(q,k,v,scale):
    context=nullcontext()
    if q.is_cuda:
        # Never allow a full-NxN MATH fallback at real resolution.
        from torch.nn.attention import sdpa_kernel,SDPBackend
        backends=[SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]
        if hasattr(SDPBackend,'CUDNN_ATTENTION'):backends.append(SDPBackend.CUDNN_ATTENTION)
        context=sdpa_kernel(backends=backends)
    with context:
        return F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False,scale=scale)


def _concat(heads,prefix):
    b,h,n,d=heads.shape
    if not 0<=prefix<n:raise ValueError('invalid native prefix')
    return heads.transpose(1,2).reshape(b,n,h*d)[:,prefix:].contiguous()


def derive_attention_representations(features,prefix,device):
    """Return 4 raw tensors [2,P,C]: block{12,24}_{V,U}; no query/reference L2norm."""
    out={};_LAST_AUDIT.clear()
    with torch.inference_mode():
        for layer in (12,24):
            q,k,v,scale=_inputs(features,layer,device)
            u=_sdpa(q,k,v,scale)
            out['block%d_V'%layer]=_concat(v,prefix)
            out['block%d_U'%layer]=_concat(u,prefix)
    _LAST_AUDIT.update(scope='two fixed native blocks; each image independently attends to its OWN full sequence including prefix',
        U='reread SDPA, not bitexact original-U claim',no_cross_image_QK=True,no_NxN_matrix=True)
    return out


def signed_value_average(features,prefix,fg,device):
    """4 scalar fields [P], fixed reference-V contrast w, query never L2norm.

    Multihead identity is SUM_h SUM_j A_hij (w_h dot V_hj), NOT averaging a
    single global w dot V by mean-head attention. Prefix values contribute to U.
    fg is reader-supplied LEGAL reference patch coverage in [0,1], no query GT.
    """
    out={};audit={}
    with torch.inference_mode():
        for layer in (12,24):
            q,k,v,scale=_inputs(features,layer,device);vp=_concat(v,prefix)
            coverage=torch.as_tensor(fg,device=v.device,dtype=v.dtype).reshape(-1)
            if coverage.numel()!=vp.shape[1] or not torch.isfinite(coverage).all() or (coverage<0).any() or (coverage>1).any():
                raise ValueError('legal reference coverage must match ALL patch tokens')
            foreground=coverage.sum();background=(1-coverage).sum()
            if float(foreground)<=0 or float(background)<=0:raise ValueError('reference FG and BG both required')
            w=(vp[0]*coverage[:,None]).sum(0)/foreground-(vp[0]*(1-coverage[:,None])).sum(0)/background
            u=_sdpa(q,k,v,scale);up=_concat(u,prefix)
            value=vp[1]@w;aggregate=up[1]@w
            # Pack head-local signed values into same-width native V so GPU
            # fused SDPA can verify the linear bridge without any NxN logits.
            wh=w.reshape(v.shape[1],v.shape[-1])
            scalar=(v*wh[None,:,None,:]).sum(-1)
            packed=torch.zeros_like(v);packed[...,0]=scalar
            reread=_sdpa(q,k,packed,scale)[1,:,prefix:,0].sum(0)
            out['block%d_value_linear_no_query_norm'%layer]=value
            out['block%d_aggregate_linear_same_w_no_query_norm'%layer]=aggregate
            audit[str(layer)]=dict(fixed_w='raw referenceV weighted FG mean minus BG mean; identical w for query V/U',
                headwise_average_max_abs=float((reread-aggregate).abs().max()),
                query_normalized=False,native_scale_recorded=scale,
                average_identity='sum_h sum_j A_hij * (w_h dot V_hj)',original_U_saved=False)
    _LAST_AUDIT.clear();_LAST_AUDIT.update(fixed_reference_V_linear=audit,no_query_GT=True,no_new_training=True,no_NxN_matrix=True)
    return out


def cpu_check(path):
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU check initialized CUDA')
    cases=[]
    with torch.inference_mode():
        for seed in range(10):
            torch.manual_seed(seed);tensors={};records={};prefix=5
            for layer in (12,24):
                for name in ('q','k','v'):tensors['block%d_%s_postRoPE'%(layer,name)]=torch.randn(2,4,21,8,dtype=torch.float64)
                records[layer-1]=dict(scale=None if seed%2 else .2,mask_shape=None,is_causal=False,dropout_p=0.)
            package=dict(features=tensors,metadata=dict(sdpa=records));got=derive_attention_representations(package,prefix,'cpu')
            errors=[];separate=[]
            for layer in (12,24):
                q,k,v,scale=_inputs(package,layer,'cpu');s=v.shape[-1]**-.5 if scale is None else scale
                manual=((q@k.transpose(-2,-1))*s).softmax(-1)@v
                errors.append(float((got['block%d_U'%layer]-_concat(manual,prefix)).abs().max()))
                singles=torch.cat([_sdpa(q[i:i+1],k[i:i+1],v[i:i+1],scale) for i in range(2)])
                separate.append(float((_concat(singles,prefix)-got['block%d_U'%layer]).abs().max()))
                assert torch.equal(got['block%d_V'%layer],_concat(v,prefix))
            fg=torch.cat([torch.ones(8),torch.zeros(8)])
            fields=signed_value_average(package,prefix,fg,'cpu');proof=get_last_audit()
            assert len(fields)==4 and all(t.shape==(16,) for t in fields.values())
            assert max(errors)<1e-12 and max(separate)<1e-12
            assert all(a['headwise_average_max_abs']<1e-12 for a in proof['fixed_reference_V_linear'].values())
            cases.append(dict(seed=seed,manual_softmax_max_abs=max(errors),batch_independence_max_abs=max(separate),
                              headwise_linear_max_abs=max(a['headwise_average_max_abs'] for a in proof['fixed_reference_V_linear'].values())))
    result=dict(state='PASSED',cases_count=10,cases=cases,CUDA_initialized=False,no_pretrained_weights=True,
                manual_NxN_only_tiny_CPU_fixture=True,real_resolution_no_MATH_fallback=True,
                no_original_U_bitexact_claim=True,no_task_gain_claim=True)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-check',required=True);cpu_check(p.parse_args().cpu_check)
