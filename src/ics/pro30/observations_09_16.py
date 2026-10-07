"""One genuine R/Q observation for Pro30 M09 and M10, no N²H cache.

Only final block Q/K/V and the actual pre-final-LN sequence are retained. Source
statistics use full native softmax denominators in row tiles. This is an extra
encoder observation when not already sealed; its cost is never called free.
"""
from __future__ import annotations
import time
import numpy as np
from .common import artifact,ArtifactUnavailable,readonly,array_hash
_LAST_EP=None
_LAST=None


def prepare_last(ep):
    global _LAST_EP,_LAST
    if _LAST_EP is ep:return _LAST
    engine=artifact(ep,'internal_encoder')
    if not hasattr(engine,'observe')or not hasattr(engine,'model'):raise ArtifactUnavailable('M09/M10 need the actual frozen native InternalEncoder, no final-token surrogate')
    from ics.cpu100.common import rgb_view
    identity=(ep.source_id,tuple(ep.producer.get('source_image_hashes',())))
    roles={};before=dict(engine.receipt);started=time.perf_counter()
    for role in('r','q'):
        session=None;image_hash=array_hash(rgb_view(ep,role))
        if engine._episode_identity==identity:
            for key,existing in engine.sessions.items():
                if key[0]==role and key[1]==image_hash and len(engine.model.blocks)in existing.attention and existing.pre_final_ln is not None:
                    session=existing;break
        if session is None:session=engine.observe(ep,role,layers=(len(engine.model.blocks),),attention_layers=(len(engine.model.blocks),))
        observed=session.attention[len(engine.model.blocks)];prefix=int(observed.prefix)
        # Pinned producer prefix layout is one CLS then native register tokens.
        if prefix!=engine.model.num_prefix_tokens or prefix<1:raise ArtifactUnavailable('Native CLS/register prefix layout unavailable')
        if not hasattr(engine.model,'cls_token'):raise ArtifactUnavailable('Native model does not bind CLS-first prefix layout')
        import torch
        with torch.inference_mode(),torch.autocast('cpu',enabled=False):
            final=engine.model.norm(session.pre_final_ln)
        raw=final[0,prefix:].detach().cpu().numpy();expected=ep.r if role=='r'else ep.q
        normalized=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
        if normalized.shape!=expected.shape or not np.allclose(normalized,expected,atol=2e-5,rtol=2e-5):raise ArtifactUnavailable('M09/M10 actual final sequence does not match native cache producer')
        register=readonly(final[0,1:prefix].detach().cpu().numpy())
        roles[role]={'observation':observed,'register':register,'producer':dict(session.producer,prefix_order='actual pinned CLS at0, native registers at1:prefix',register_state='actual model.finalLN(post-final-block), no CLS'),'statistics':{}}
    roles['receipt']={'extra_encoder_forwards_this_entry':engine.receipt['actual_extra_forwards']-before.get('actual_extra_forwards',0),'observation_seconds_this_entry':time.perf_counter()-started,'forward_seconds_total':engine.receipt['forward_wall_seconds'],'minimal_attention_layers':[len(engine.model.blocks)],'no_full_N_squared_H_cache':True,'parity':dict(engine.receipt['parity']),'shared_by_methods':['PRO30_M09','PRO30_M10']}
    _LAST_EP=ep;_LAST=roles
    return roles


def source_statistics(ep,role,*,random_groups=False,chunk=32):
    """Output messages[N,4,D], mean-head masses[N,4] and closure audit."""
    shared=prepare_last(ep);data=shared[role];cache=data['statistics'];key=bool(random_groups)
    if key in cache:return cache[key]
    import torch
    observation=data['observation'];q,k,v=observation.q,observation.k,observation.v
    if q.shape[0]!=1 or k.shape[0]!=1 or v.shape[0]!=1:raise ArtifactUnavailable('Batch-one exact native attention observation required')
    if observation.causal:raise ArtifactUnavailable('DINO noncausal native observation required')
    prefix=observation.prefix;hw=ep.r_hw if role=='r'else ep.q_hw;valid=np.asarray(ep.wvalid if role=='r'else ep.q_valid)>0;N=len(valid);S=N+prefix
    if q.shape[-2]!=S or k.shape[-2]!=S or v.shape[-2]!=S:raise ValueError('Actual prefix/patch sequence length differs from native grid')
    H=q.shape[1];D=observation.projection_weight.shape[0];messages=np.empty((N,4,D),np.float32);mass=np.empty((N,4),np.float32);started=time.perf_counter();max_residual=0.;max_mass_error=0.
    physical=np.flatnonzero(valid)+prefix;padding=np.flatnonzero(~valid)+prefix
    with torch.inference_mode():
        for start in range(0,N,chunk):
            stop=min(start+chunk,N);indices=np.arange(start,stop);rows=torch.as_tensor(indices+prefix,dtype=torch.long)
            logits=(q.index_select(-2,rows)@k.transpose(-2,-1))*observation.scale
            if observation.mask is not None:
                mask=observation.mask
                if mask.ndim>=2 and mask.shape[-2]!=1:mask=mask.index_select(-2,rows)
                logits=logits.masked_fill(~mask,-float('inf'))if mask.dtype==torch.bool else logits+mask
            A=logits.softmax(-1)[0]
            groups=np.full((len(indices),S),3,np.int8);groups[:,:prefix]=2;groups[:,physical]=1
            for local,row in enumerate(indices):
                y,x=divmod(int(row),hw[1]);neighbors=[]
                for yy in range(max(0,y-1),min(hw[0],y+2)):
                    for xx in range(max(0,x-1),min(hw[1],x+2)):
                        j=yy*hw[1]+xx
                        if valid[j]:neighbors.append(j+prefix)
                groups[local,neighbors]=0
                if random_groups:
                    permutation=np.random.default_rng(900000+int(row)).permutation(S)
                    groups[local]=groups[local,permutation]
            components=[];masses=[]
            for group in range(4):
                mask=torch.from_numpy(groups==group).to(A.dtype)
                grouped=A*mask[None];components.append(grouped@v[0]);masses.append(grouped.sum(dim=-1).mean(dim=0))
            raw_total=sum(components);native=observation.head_output[0].index_select(-2,rows)
            residual=float(torch.max(torch.abs(raw_total-native)));max_residual=max(max_residual,residual)
            if not torch.allclose(raw_total,native,atol=2e-5,rtol=2e-5):raise ArithmeticError('Native source-split AV sum does not match actual complete native AV output')
            for group,component in enumerate(components):
                concatenated=component.transpose(0,1).reshape(len(indices),-1)
                output=concatenated@observation.projection_weight.T
                messages[start:stop,group]=output.cpu().numpy();mass[start:stop,group]=masses[group].cpu().numpy()
            max_mass_error=max(max_mass_error,float(np.max(np.abs(mass[start:stop].sum(axis=1)-1))))
    audit={'source_order':['self_and_physical3x3','other_physical_patches','all_prefix','physical_padding_patches'],'observed_block':int(observation.block),'head_count':int(H),'prefix_tokens':int(prefix),'actual_mask':None if observation.mask is None else {'shape':list(observation.mask.shape),'dtype':str(observation.mask.dtype)},'native_QK_scaling':float(observation.scale),'output_bias_assigned_to_any_source':False,'LayerScale_applied':False,'native_AV_closure_max_abs_error':max_residual,'mean_head_mass_sum_max_abs_error':max_mass_error,'row_tile':chunk,'full_attention_stored':False,'attention_recompute_and_projection_seconds':time.perf_counter()-started,'source_messages_bytes':messages.nbytes,'random_key_group_null':random_groups,'producer':data['producer']}
    output=(readonly(messages),readonly(mass),audit);cache[key]=output
    return output
