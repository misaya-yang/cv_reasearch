#!/usr/bin/env python3
"""Exact mixed global/patch content attention for LIVE timm EvaAttention (DINOv3).

No weights, projections, patch RoPE, V, output norm/gate/projection, MLP or residual
are changed. The gamma=1 logits use native rotated Q/K for patch--patch and raw
post-qk_norm Q/K whenever at least one endpoint is a CLS/register prefix token.
A three-head-dimension lift computes ONE unified softmax with original scale.
On CUDA force Flash SDPA (native d64 -> lifted d192), never silently allocate a
full N*N math-attention matrix. Only actual Flash unavailability enables explicit
query-chunk256 fallback, recorded in the audit. CPU uses SDPA math for unit tests.

Usage: with global_content_attention(timm_backbone) as audit: features=encode(...)
No-op: enabled=False makes NO forward assignments; calls original forward exactly.
All assignments are instance-local and rolled back in finally, including errors.
"""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import sys
from types import MethodType
import warnings

import torch
import torch.nn.functional as F


def _validate(q,k,v,qr,kr,n_global):
    if q.ndim!=4 or any(x.shape!=q.shape for x in (k,v,qr,kr)):
        raise ValueError('Expected equal [B,H,N,d] Q/K/V and rotated Q/K shapes')
    if not 0<=n_global<=q.shape[-2]:raise ValueError('Global prefix outside sequence')
    if len({x.dtype for x in (q,k,v,qr,kr)})!=1 or len({x.device for x in (q,k,v,qr,kr)})!=1:
        raise ValueError('Q/K/V dtypes/devices must match native value dtype/device')


def lift_qkv(q,k,v,qr,kr,n_global):
    """PP: qr.kr; PG/GP/GG: q.k; values unchanged in first d coordinates."""
    _validate(q,k,v,qr,kr,n_global);zero=torch.zeros_like(q)
    # Global Q=[0,0,q], patch Q=[qr,q,0].
    qa=torch.cat([zero[...,:n_global,:],qr[...,n_global:,:]],dim=-2)
    qb=torch.cat([zero[...,:n_global,:],q[...,n_global:,:]],dim=-2)
    qc=torch.cat([q[...,:n_global,:],zero[...,n_global:,:]],dim=-2)
    # Global K=[0,k,k], patch K=[kr,0,k].
    ka=torch.cat([zero[...,:n_global,:],kr[...,n_global:,:]],dim=-2)
    kb=torch.cat([k[...,:n_global,:],zero[...,n_global:,:]],dim=-2)
    return torch.cat([qa,qb,qc],dim=-1),torch.cat([ka,kb,k],dim=-1),torch.cat([v,zero,zero],dim=-1)


def explicit_piecewise_reference(q,k,v,qr,kr,n_global,scale=None,attn_mask=None,is_causal=False):
    """Tiny CPU mathematical reference ONLY, full logits prohibited in real encoder."""
    _validate(q,k,v,qr,kr,n_global);scale=q.shape[-1]**-.5 if scale is None else scale
    logits=(qr@kr.transpose(-2,-1))*scale
    if n_global:
        logits[...,:n_global,:]=(q[...,:n_global,:]@k.transpose(-2,-1))*scale
        logits[...,n_global:,:n_global]=(q[...,n_global:,:]@k[...,:n_global,:].transpose(-2,-1))*scale
    if is_causal:
        mask=torch.ones(q.shape[-2],q.shape[-2],device=q.device,dtype=torch.bool).tril()
        logits=logits.masked_fill(~mask,float('-inf'))
    if attn_mask is not None:
        logits=logits.masked_fill(~attn_mask,float('-inf')) if attn_mask.dtype==torch.bool else logits+attn_mask
    return logits.softmax(dim=-1)@v


def _chunk_attention(q,k,v,qr,kr,n_global,scale,attn_mask,is_causal,chunk=256):
    """Bounded real fallback, single softmax across ALL keys per query chunk."""
    if chunk!=256:raise ValueError('Only the preregistered256-query fallback is allowed')
    n=q.shape[-2];parts=[]
    for lo in range(0,n,chunk):
        hi=min(n,lo+chunk);logits=(qr[...,lo:hi,:]@kr.transpose(-2,-1))*scale
        end=min(hi,n_global)
        if lo<end:logits[...,:end-lo,:]=(q[...,lo:end,:]@k.transpose(-2,-1))*scale
        start=max(lo,n_global)
        if start<hi:logits[...,start-lo:,:n_global]=(q[...,start:hi,:]@k[...,:n_global,:].transpose(-2,-1))*scale
        if is_causal:
            allowed=torch.arange(n,device=q.device)[None,:]<=torch.arange(lo,hi,device=q.device)[:,None]
            logits=logits.masked_fill(~allowed,float('-inf'))
        if attn_mask is not None:
            selected=attn_mask[...,lo:hi,:] if attn_mask.ndim>=2 and attn_mask.shape[-2]!=1 else attn_mask
            logits=logits.masked_fill(~selected,float('-inf')) if selected.dtype==torch.bool else logits+selected
        # Native eval SDPA stabilizes softmax; use FP32 for low-precision fallback,
        # preserving FP64 for the algebraic CPU check. No independent key softmax.
        dtype=torch.float64 if logits.dtype==torch.float64 else torch.float32
        probability=logits.to(dtype).softmax(dim=-1).to(v.dtype)
        probability=torch.nan_to_num(probability,nan=0.) # Fully masked row: SDPA zero semantics.
        parts.append(probability@v)
    return torch.cat(parts,dim=-2)


def lifted_attention(q,k,v,qr,kr,n_global,scale=None,attn_mask=None,is_causal=False,audit=None,mode='global_content'):
    _validate(q,k,v,qr,kr,n_global);d=q.shape[-1];scale=d**-.5 if scale is None else scale
    if mode=='global_content':ql,kl,vl=lift_qkv(q,k,v,qr,kr,n_global)
    elif mode=='padded_native':
        zero=torch.zeros_like(v)
        ql,kl,vl=[torch.cat([t,zero,zero],dim=-1) for t in (qr,kr,v)]
    else:raise ValueError('Only global_content or the fixed padded_native numerical control is supported')
    if q.is_cuda:
        try:
            from torch.nn.attention import sdpa_kernel,SDPBackend
            # Explicit backend guard: no silent math SDPA/N*N allocation on CUDA.
            with warnings.catch_warnings(record=True) as diagnostics:
                warnings.simplefilter('always')
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    out=F.scaled_dot_product_attention(ql,kl,vl,attn_mask=attn_mask,
                          dropout_p=0.,is_causal=is_causal,scale=scale)[...,:d]
            backend='flash_sdpa_lift3d';reason=None
        except (ImportError,NotImplementedError,RuntimeError) as exc:
            # Never hide OOM, data errors, or unrelated failures as backend issues.
            message=str(exc).lower()
            unavailable=isinstance(exc,(ImportError,NotImplementedError)) or any(x in message for x in
                ('no available kernel','no viable backend','not supported','does not support','no kernel found'))
            if not unavailable:raise
            reason=str(exc)
            if 'diagnostics' in locals():reason+=' | '+' | '.join(str(w.message) for w in diagnostics)
            if mode=='padded_native':out=_chunk_attention(qr,kr,v,qr,kr,0,scale,attn_mask,is_causal,chunk=256)
            else:out=_chunk_attention(q,k,v,qr,kr,n_global,scale,attn_mask,is_causal,chunk=256)
            backend='explicit_query_chunk256_fallback'
    else:
        out=F.scaled_dot_product_attention(ql,kl,vl,attn_mask=attn_mask,
                    dropout_p=0.,is_causal=is_causal,scale=scale)[...,:d]
        backend='cpu_sdpa_lift3d';reason=None
    if audit is not None:
        audit['calls']+=1;audit['backend_counts'][backend]=audit['backend_counts'].get(backend,0)+1
        if reason is not None and reason not in audit['fallback_reasons']:audit['fallback_reasons'].append(reason)
    return out


def _eva_forward(self,x,rope=None,attn_mask=None,is_causal=False,*,rope_apply,audit,mode='global_content'):
    if self.training:raise RuntimeError('Global-content intervention is strictly frozen eval-only')
    B,N,C=x.shape
    if not 0<=self.num_prefix_tokens<=N:raise ValueError('Invalid native CLS/register prefix')
    gate=self.gate(x).sigmoid() if self.gate is not None else None
    if self.qkv is not None:
        if self.q_bias is None:qkv=self.qkv(x)
        else:
            bias=torch.cat((self.q_bias,self.k_bias,self.v_bias))
            if self.qkv_bias_separate:qkv=self.qkv(x);qkv+=bias
            else:qkv=F.linear(x,weight=self.qkv.weight,bias=bias)
        qkv=qkv.reshape(B,N,3,self.num_heads,-1).permute(2,0,3,1,4);q,k,v=qkv.unbind(0)
    else:
        q=self.q_proj(x).reshape(B,N,self.num_heads,-1).transpose(1,2)
        k=self.k_proj(x).reshape(B,N,self.num_heads,-1).transpose(1,2)
        v=self.v_proj(x).reshape(B,N,self.num_heads,-1).transpose(1,2)
    q,k=self.q_norm(q),self.k_norm(k);qr,kr=q,k
    if rope is not None:
        npt=self.num_prefix_tokens;half=getattr(self,'rotate_half',False)
        qr=torch.cat([q[:,:,:npt,:],rope_apply(q[:,:,npt:,:],rope,half=half)],dim=2).type_as(v)
        kr=torch.cat([k[:,:,:npt,:],rope_apply(k[:,:,npt:,:],rope,half=half)],dim=2).type_as(v)
        q,k=q.type_as(v),k.type_as(v)
    # Same original scale, never lifted-head default1/sqrt(3d).
    output=lifted_attention(q,k,v,qr,kr,self.num_prefix_tokens,scale=self.scale,
                           attn_mask=attn_mask,is_causal=is_causal,audit=audit,mode=mode)
    output=output.transpose(1,2).reshape(B,N,C)
    output=self.norm(output)
    if gate is not None:output=output*gate
    return self.proj_drop(self.proj(output))


@contextmanager
def global_content_attention(model,enabled=True,*,rope_apply=None,mode='global_content'):
    """Patch actual EvaAttention instances; finally restore exact original fields.

    audit is a JSON-compatible dictionary. Disabled context does not wrap forward
    and leaves all parameters, modules, precision and original backend untouched.
    """
    if mode not in ('global_content','padded_native'):raise ValueError('Unknown fixed attention mode')
    modules=[(name,module) for name,module in model.named_modules() if type(module).__name__=='EvaAttention']
    if not modules:raise ValueError('No live timm EvaAttention found; do not guess another attention implementation')
    audit=dict(enabled=bool(enabled),mode=mode,gamma=(1 if mode=='global_content' else 0) if enabled else 0,calls=0,backend_counts={},fallback_reasons=[],
               modules=[],rollback_complete=False,scale='Original native head_dim**-.5, not lifted width scale')
    original=[]
    try:
        for name,module in modules:
            try:source_sha=hashlib.sha256(inspect.getsource(type(module).forward).encode()).hexdigest()
            except (OSError,TypeError):source_sha=None
            audit['modules'].append(dict(name=name,class_name=type(module).__module__+'.'+type(module).__name__,
                heads=module.num_heads,head_dim=module.head_dim,prefix=module.num_prefix_tokens,
                fused_attn=bool(module.fused_attn),native_scale=float(module.scale),native_forward_sha256=source_sha,
                qkv_fused=module.qkv is not None,qkv_bias_separate=bool(getattr(module,'qkv_bias_separate',False)),
                q_norm=type(module.q_norm).__name__,k_norm=type(module.k_norm).__name__,
                output_norm=type(module.norm).__name__,gate_present=module.gate is not None,
                rotate_half=bool(module.rotate_half),attn_dropout_p=float(module.attn_drop.p),
                projection_dropout_p=float(module.proj_drop.p)))
            if not enabled:continue # Native call path completely untouched.
            if module.training:raise RuntimeError('All patched attention modules must already be eval')
            apply=rope_apply
            if apply is None:
                owner=sys.modules[type(module).__module__]
                apply=getattr(owner,'apply_rot_embed_cat',None)
            if apply is None:raise ValueError('Native Eva rotary helper unavailable')
            required=('qkv','q_bias','k_bias','v_bias','qkv_bias_separate','q_norm','k_norm','norm','gate','proj','proj_drop','scale')
            if any(not hasattr(module,key) for key in required):raise ValueError('Unknown EvaAttention interface')
            if float(module.scale)!=module.head_dim**-.5:raise ValueError('Nonstandard native scale needs a separate contract')
            had_instance='forward' in module.__dict__;old_instance=module.__dict__.get('forward');old=module.forward
            original.append((module,had_instance,old_instance,old))
            def patched(this,x,rope=None,attn_mask=None,is_causal=False,_apply=apply):
                return _eva_forward(this,x,rope,attn_mask,is_causal,rope_apply=_apply,audit=audit,mode=mode)
            module.forward=MethodType(patched,module)
        yield audit
    finally:
        for module,had_instance,old_instance,old in reversed(original):
            if had_instance:module.forward=old_instance
            else:delattr(module,'forward')
            current=module.forward
            if getattr(current,'__func__',current) is not getattr(old,'__func__',old):raise AssertionError('Native forward rollback failed')
        audit['rollback_complete']=True


def cuda_flash_probe(*,device='cuda',dtype=torch.bfloat16):
    """EXPLICIT future-authorized tiny GPU hook; never called by CPU self-check.

    Tests d64->192 actual Flash support using17 tokens, not a performance benchmark
    or a license to start CUDA. Returns backend/fallback and padded-native drift.
    Caller must apply its own live GPU/memory/authorization checks first.
    """
    if not str(device).startswith('cuda'):raise ValueError('CUDA hook requires an explicitly authorized CUDA device')
    tensors=[torch.randn(1,2,17,64,device=device,dtype=dtype) for _ in range(5)]
    q,k,v,qr,kr=tensors
    audits=[];outs=[]
    for mode in ('global_content','padded_native'):
        audit=dict(calls=0,backend_counts={},fallback_reasons=[])
        outs.append(lifted_attention(q,k,v,qr,kr,5,scale=.125,audit=audit,mode=mode));audits.append(audit)
    native=F.scaled_dot_product_attention(qr,kr,v,dropout_p=0.,scale=.125)
    return dict(device=str(device),dtype=str(dtype),native_head_dim=64,lift_head_dim=192,
                tokens=17,global_prefix=5,audits=audits,
                padded_native_max_abs=float((outs[1].float()-native.float()).abs().max()),
                semantic_output_finite=bool(torch.isfinite(outs[0]).all()),scope='Tiny actual backend check only')


def self_check():
    torch.set_num_threads(1);worst=0.;fallback_worst=0.
    for seed in range(10):
        generator=torch.Generator().manual_seed(seed);B=1+seed%2;H=1+seed%3;N=7+seed;d=4+2*(seed%3)
        q,k,v=[torch.randn(B,H,N,d,generator=generator,dtype=torch.float64) for _ in range(3)]
        qr,kr=[torch.randn(B,H,N,d,generator=generator,dtype=torch.float64) for _ in range(2)]
        ng=[0,1,5,N][seed%4];scale=d**-.5
        mask=None
        if seed%3==1:mask=torch.ones(N,N,dtype=torch.bool);mask[:,0]=False
        if seed%3==2:mask=torch.randn(N,N,generator=generator,dtype=torch.float64)*.1
        expected=explicit_piecewise_reference(q,k,v,qr,kr,ng,scale,mask)
        got=lifted_attention(q,k,v,qr,kr,ng,scale,mask)
        chunked=_chunk_attention(q,k,v,qr,kr,ng,scale,mask,False)
        error=float((got-expected).abs().max());worst=max(worst,error)
        fallback_worst=max(fallback_worst,float((chunked-expected).abs().max()))
        assert error<=1e-12 and fallback_worst<=1e-12
        padded=lifted_attention(q,k,v,qr,kr,ng,scale,mask,mode='padded_native')
        native=F.scaled_dot_product_attention(qr,kr,v,attn_mask=mask,scale=scale)
        assert float((padded-native).abs().max())<=1e-12
        causal=lifted_attention(q,k,v,qr,kr,ng,scale,is_causal=True)
        causal_ref=explicit_piecewise_reference(q,k,v,qr,kr,ng,scale,is_causal=True)
        assert float((causal-causal_ref).abs().max())<=1e-12
    print(json.dumps(dict(cpu_fp64_cases=10,max_lift_output_error=worst,max_fallback_output_error=fallback_worst,
                         prefix0_and_allglobal=True,additive_boolean_and_causal_masks=True,scale='original d',gamma=1)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:self_check()
    else:parser.error('Import the reversible context in the native runner, or use --self-check')
