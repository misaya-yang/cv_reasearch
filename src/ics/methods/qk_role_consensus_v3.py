"""QK family v3: fixed pre-RoPE directed role margins, bounded MEAN residual.

No head/layer selection, encoder replacement, query supervision or inverse RoPE.
This is a substantial revision of the existing QK family: independent count=0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    block_one_based: int = 21
    role_temperature: float = .07
    response_temperature: float = .07
    residual_bound: float = .1
    query_chunk: int = 256


def _config(cfg):
    if (cfg.block_one_based != 21 or cfg.role_temperature != .07
            or cfg.response_temperature != .07 or cfg.residual_bound != .1 or cfg.query_chunk < 1):
        raise ValueError('QK v3 fixed recipe changed')


@torch.inference_mode()
def extract_pre_rope(actual_block, paired_raw_h20, prefix=5):
    """Actual norm1/qkv/q_norm/k_norm at raw H20; never invert saved RoPE.

    The bound production Eva has a single unbiased qkv Linear. Unsupported bias
    layouts fail closed instead of silently reconstructing another teacher.
    """
    if (actual_block.training or any(p.requires_grad for p in actual_block.parameters())
            or paired_raw_h20.ndim != 3 or paired_raw_h20.shape[0] != 2
            or paired_raw_h20.dtype != torch.float32 or not torch.isfinite(paired_raw_h20).all()):
        raise ValueError('Require frozen FP32 actual block and paired [R,Q] raw H20')
    attn=actual_block.attn
    if (not isinstance(attn.qkv,torch.nn.Linear) or attn.qkv.bias is not None
            or getattr(attn,'q_bias',None) is not None or getattr(attn,'v_bias',None) is not None):
        raise ValueError('Bound Eva unbiased combined actual QKV required')
    if prefix<0 or prefix>=paired_raw_h20.shape[1]:raise ValueError('Invalid prefix mapping')
    normalized=actual_block.norm1(paired_raw_h20)
    projected=attn.qkv(normalized)
    batch,tokens,width=projected.shape
    heads=int(attn.num_heads);head_width=width//(3*heads)
    if width!=3*heads*head_width:raise ValueError('Actual QKV/head layout mismatch')
    q,k,_=projected.reshape(batch,tokens,3,heads,head_width).permute(2,0,3,1,4).unbind(0)
    q,k=attn.q_norm(q),attn.k_norm(k)
    return q[:,:,prefix:].permute(0,2,1,3).contiguous(), k[:,:,prefix:].permute(0,2,1,3).contiguous(), {
        'source':'actual_block21.norm1/qkv/q_norm/k_norm before any RoPE',
        'heads':heads,'head_width':head_width,'prefix':prefix,'native_teacher_parameters_unchanged':True,
        'inverse_RoPE':False,'head_selection':False,'layer_selection':False}


def _head_embedding(value):
    value=torch.as_tensor(value,dtype=torch.float32,device='cpu')
    if value.ndim!=3 or not torch.isfinite(value).all():raise ValueError('Finite [patch,head,head_width] teacher states required')
    # Dot of these flattened rows is the equal mean of per-head cosine logits.
    return F.normalize(value,dim=-1).reshape(len(value),-1)/math.sqrt(value.shape[1])


def _unit(value):
    value=torch.as_tensor(value,dtype=torch.float32,device='cpu')
    if value.ndim!=2 or not torch.isfinite(value).all():raise ValueError('Finite token features required')
    return F.normalize(value,dim=-1)


def _role_margin(similarity, log_fg, log_bg, temperature):
    return temperature*(torch.logsumexp(similarity/temperature+log_fg,dim=1)
                        -torch.logsumexp(similarity/temperature+log_bg,dim=1))


def _weights(coverage, device):
    cov=torch.as_tensor(coverage,dtype=torch.float32,device=device).reshape(-1)
    if not torch.isfinite(cov).all() or bool((cov<0).any() or (cov>1).any()):raise ValueError('Reference mask coverage must be [0,1]')
    foreground=float(cov.sum());background=float((1-cov).sum())
    if foreground<=0 or background<=0:return None,dict(empty_reference=foreground<=0,missing_background=background<=0)
    fg=torch.where(cov>0,cov.log(),-torch.inf)-math.log(foreground)
    bg=torch.where(cov<1,(1-cov).log(),-torch.inf)-math.log(background)
    return (fg,bg),dict(foreground_mass=foreground,background_mass=background)


def agreement(a,b,temperature=.07):
    x,y=torch.tanh(a/temperature),torch.tanh(b/temperature)
    return torch.where(x*y>0,torch.sign(x+y)*torch.minimum(x.abs(),y.abs()),torch.zeros_like(x))


@torch.inference_mode()
def evidence(r_q,r_k,q_q,q_k,coverage,r_h20,q_h20,r_final,q_final,cfg=Config()):
    _config(cfg)
    rq,rk,qq,qk=(_head_embedding(x) for x in (r_q,r_k,q_q,q_k))
    if rq.shape!=rk.shape or qq.shape!=qk.shape or rq.shape[1]!=qq.shape[1]:raise ValueError('Teacher views do not share head dimensions')
    rh,qh,rf,qf=(_unit(x) for x in (r_h20,q_h20,r_final,q_final))
    if any(len(x)!=len(rq) for x in (rh,rf)) or any(len(x)!=len(qq) for x in (qh,qf)):
        raise ValueError('Direct control features must share exact token mapping')
    weights,edge=_weights(coverage,rq.device)
    if np.asarray(coverage).size!=len(rq):raise ValueError('Reference coverage alignment changed')
    if weights is None:return None,edge
    log_fg,log_bg=weights
    collected={name:[] for name in ('forward','reverse','symmetric_logits','direct_H20','direct_final')}
    started=time.perf_counter()
    for start in range(0,len(qq),cfg.query_chunk):
        end=min(len(qq),start+cfg.query_chunk)
        forward=qq[start:end]@rk.T
        reverse=qk[start:end]@rq.T
        for name,similarity in (('forward',forward),('reverse',reverse),('symmetric_logits',(forward+reverse)/2),
                               ('direct_H20',qh[start:end]@rh.T),('direct_final',qf[start:end]@rf.T)):
            collected[name].append(_role_margin(similarity,log_fg,log_bg,cfg.role_temperature))
    fields={name:torch.cat(values) for name,values in collected.items()}
    a,b=fields['forward'],fields['reverse']
    fields['agreement']=agreement(a,b,cfg.response_temperature)
    fields['symmetric_margin']=torch.tanh((a+b)/(2*cfg.response_temperature))
    info=dict(config=asdict(cfg),weights=edge,all_heads_equal=True,query_gt_used=False,
              joint_positive=int(((a>0)&(b>0)).sum()),joint_negative=int(((a<0)&(b<0)).sum()),
              disagreement=int((a*b<0).sum()),zero_direction=int(((a==0)|(b==0)).sum()),
              margin_forward_reverse_maxabs=float((a-b).abs().max()),
              evidence_seconds=time.perf_counter()-started,external_semantic_information_added=False)
    return fields,info


def render(field, original_hw, working_size=1024):
    x=torch.as_tensor(np.ascontiguousarray(field),dtype=torch.float32)[None,None]
    work=F.interpolate(x,(working_size,working_size),mode='bilinear',align_corners=False)>.5
    original=F.interpolate(work.float(),tuple(original_hw),mode='bilinear',align_corners=False)>.5
    return work[0,0].numpy(),original[0,0].numpy()


@torch.inference_mode()
def predict(r_q,r_k,q_q,q_k,coverage,base,r_h20,q_h20,r_final,q_final,original_hw,cfg=Config(),*,working_size=1024):
    _config(cfg)
    base=np.asarray(base,dtype=np.float32)
    if base.ndim!=2 or base.size!=len(q_q) or not np.isfinite(base).all():raise ValueError('Exact finite MEAN host grid required')
    ev,info=evidence(r_q,r_k,q_q,q_k,coverage,r_h20,q_h20,r_final,q_final,cfg)
    arm_names=('agreement','forward','reverse','symmetric_logits','symmetric_margin','direct_H20','direct_final')
    fields={'mean.control':base.copy()}
    if ev is None:
        for arm in arm_names:fields[arm]=np.zeros_like(base) if info['empty_reference'] else base.copy()
        changes={arm:float(np.max(np.abs(fields[arm]-base))) for arm in arm_names}
    else:
        changes={}
        for arm in arm_names:
            response=ev[arm] if arm in ('agreement','symmetric_margin') else torch.tanh(ev[arm]/cfg.response_temperature)
            residual=(cfg.residual_bound*response).numpy().reshape(base.shape)
            fields[arm]=base+residual
            changes[arm]=float(np.max(np.abs(fields[arm]-base)))
            if changes[arm]>cfg.residual_bound+1e-6:raise RuntimeError('Small residual budget violated')
    masks={arm:render(field,original_hw,working_size) for arm,field in fields.items()}
    info.update(field_max_changes=changes,budget_applies=not info.get('empty_reference',False),
                full_output=True,clamp=False,new_graph_solve=False,
                family='QK substantial revision v3',independent_method_increment=0,
                real_quality='unmeasured',renderer='bilinear grid->1024 >.5 then bilinear binary->original >.5')
    return {'fields':fields,'masks':masks,'evidence':None if ev is None else {k:v.numpy() for k,v in ev.items()},'info':info}
