#!/usr/bin/env python3
"""Read-only native-trajectory attention traces and LOCAL patch->global readouts.

Original EvaAttention.forward executes once and its EXACT tensor object is returned
unchanged to the encoder. Side QKV/readouts are never residual/MLP inputs. Outputs
are native projected/gated/normed attention contributions -> cloned native block
LayerScale/drop_path(eval) -> actual backbone final norm -> remove prefix.

API:
  with native_attention_readouts(backbone, mode='local_pg') as trace:
      native_outputs = backbone.forward_intermediates(images, ...)
  trace.raw_maps('R0', grid=(64,64), pass_index=0)  # [B,L,C,H,W]
  trace.raw_maps('Rcf', ...); trace.raw_maps('Rpad', ...)
  trace.feature_streams(pass_index=0)             # [L,B,P,C] CPU FP32 by default
  trace.native_scales(pass_index=0)               # [B,L,1,1,1] exact native FP32 std

Modes: native_only reuses actual return (no QKV recompute); local_pg computes PG-only
counterfactual + matched patch2d padded-native kernel control; observe_modified reads
an already modified outer forward without renativizing or asserting native replica.
For observe_modified, fusion scales/weights MUST come from a separate ORIGINAL run.
No layer weights or density decoder are invented here. Default CPU FP32 preserves tiny readout deltas; FP16 is an explicit storage option. Ordinary SDPA exposes no row
LSE: this conservative code recomputes attention and does not claim cheap delta/LSE.
"""
from contextlib import contextmanager
import sys
from types import MethodType

import torch
import torch.nn.functional as F


def eva_components(attn,x,rope,rope_apply):
    """Same live Eva qkv/bias/qknorm/RoPE/order; no attn.forward recursion."""
    B,N,C=x.shape;gate=attn.gate(x).sigmoid() if attn.gate is not None else None
    if attn.qkv is not None:
        if attn.q_bias is None:qkv=attn.qkv(x)
        else:
            bias=torch.cat((attn.q_bias,attn.k_bias,attn.v_bias))
            if attn.qkv_bias_separate:qkv=attn.qkv(x);qkv+=bias
            else:qkv=F.linear(x,weight=attn.qkv.weight,bias=bias)
        qkv=qkv.reshape(B,N,3,attn.num_heads,-1).permute(2,0,3,1,4);q,k,v=qkv.unbind(0)
    else:
        q=attn.q_proj(x).reshape(B,N,attn.num_heads,-1).transpose(1,2)
        k=attn.k_proj(x).reshape(B,N,attn.num_heads,-1).transpose(1,2)
        v=attn.v_proj(x).reshape(B,N,attn.num_heads,-1).transpose(1,2)
    q,k=attn.q_norm(q),attn.k_norm(k);qr,kr=q,k
    if rope is not None:
        ng=attn.num_prefix_tokens;half=attn.rotate_half
        qr=torch.cat([q[:,:,:ng,:],rope_apply(q[:,:,ng:,:],rope,half=half)],dim=2).type_as(v)
        kr=torch.cat([k[:,:,:ng,:],rope_apply(k[:,:,ng:,:],rope,half=half)],dim=2).type_as(v)
        q,k=q.type_as(v),k.type_as(v)
    return q,k,v,qr,kr,gate


def project_heads(attn,heads,x,gate):
    B,N,C=x.shape;out=attn.norm(heads.transpose(1,2).reshape(B,N,C))
    if gate is not None:out=out*gate
    return attn.proj_drop(attn.proj(out))


def native_heads(attn,qr,kr,v,mask=None,is_causal=False):
    # Native fused path keeps original implicit d-scale/backend, exactly original.
    if attn.fused_attn:
        return F.scaled_dot_product_attention(qr,kr,v,attn_mask=mask,dropout_p=0.,is_causal=is_causal)
    if qr.is_cuda:raise RuntimeError('Unsupported non-fused CUDA native replica; no silent NxN side allocation')
    logits=(qr*attn.scale)@kr.transpose(-2,-1)
    if is_causal:logits=logits.masked_fill(~torch.ones(qr.shape[-2],qr.shape[-2],dtype=torch.bool,device=qr.device).tril(),float('-inf'))
    elif mask is not None:logits=logits.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else logits+mask
    return logits.softmax(-1)@v


def _patch_mask(mask,n_global,n,is_causal,device):
    selected=mask
    if selected is not None and selected.ndim>=2 and selected.shape[-2]!=1:
        selected=selected[...,n_global:,:]
    if is_causal:
        allowed=torch.arange(n,device=device)[None,:]<=torch.arange(n_global,n,device=device)[:,None]
        if selected is None:selected=allowed
        elif selected.dtype==torch.bool:selected=selected&allowed
        else:selected=selected.masked_fill(~allowed,float('-inf'))
    return selected


def explicit_pg_reference(q,k,v,qr,kr,n_global,scale=None,mask=None,is_causal=False):
    """Tiny CPU reference: only patch-query/global-key logits replaced."""
    scale=q.shape[-1]**-.5 if scale is None else scale
    logits=(qr[...,n_global:,:]@kr.transpose(-2,-1))*scale
    if n_global:logits[...,:n_global]=(q[...,n_global:,:]@k[...,:n_global,:].transpose(-2,-1))*scale
    selected=_patch_mask(mask,n_global,q.shape[-2],is_causal,q.device)
    if selected is not None:logits=logits.masked_fill(~selected,float('-inf')) if selected.dtype==torch.bool else logits+selected
    return logits.softmax(-1)@v


def _pg_chunk(q,k,v,qr,kr,ng,scale,mask,is_causal,mode):
    n=q.shape[-2];outputs=[]
    for lo in range(ng,n,256):
        hi=min(n,lo+256);logits=(qr[...,lo:hi,:]@kr.transpose(-2,-1))*scale
        if mode=='local_pg' and ng:logits[...,:ng]=(q[...,lo:hi,:]@k[...,:ng,:].transpose(-2,-1))*scale
        selected=mask
        if selected is not None and selected.ndim>=2 and selected.shape[-2]!=1:selected=selected[...,lo:hi,:]
        if is_causal:
            allowed=torch.arange(n,device=q.device)[None,:]<=torch.arange(lo,hi,device=q.device)[:,None]
            logits=logits.masked_fill(~allowed,float('-inf'))
        if selected is not None:logits=logits.masked_fill(~selected,float('-inf')) if selected.dtype==torch.bool else logits+selected
        dtype=torch.float64 if logits.dtype==torch.float64 else torch.float32
        probabilities=logits.to(dtype).softmax(-1).to(v.dtype);probabilities=torch.nan_to_num(probabilities,nan=0.)
        outputs.append(probabilities@v)
    return torch.cat(outputs,-2)


def patch_readout_heads(q,k,v,qr,kr,n_global,scale=None,mask=None,is_causal=False,mode='local_pg',audit=None):
    if q.ndim!=4 or any(t.shape!=q.shape for t in (k,v,qr,kr)):raise ValueError('Equal [B,H,N,d] native QKV required')
    n,d=q.shape[-2:];scale=d**-.5 if scale is None else scale
    if not 0<=n_global<=n:raise ValueError('Invalid native prefix')
    if mode not in ('local_pg','padded_native'):raise ValueError('Unknown fixed side-kernel mode')
    if n_global==n:return v[...,:0,:]
    zero=torch.zeros_like(v)
    if mode=='local_pg':
        qlift=torch.cat([qr[...,n_global:,:],q[...,n_global:,:]],-1)
        ka=torch.cat([zero[...,:n_global,:],kr[...,n_global:,:]],-2)
        kb=torch.cat([k[...,:n_global,:],zero[...,n_global:,:]],-2)
        klift=torch.cat([ka,kb],-1)
    else:
        qlift=torch.cat([qr[...,n_global:,:],zero[...,n_global:,:]],-1)
        klift=torch.cat([kr,zero],-1)
    vlift=torch.cat([v,zero],-1);selected=_patch_mask(mask,n_global,n,is_causal,q.device)
    reason=None
    if q.is_cuda:
        try:
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                result=F.scaled_dot_product_attention(qlift,klift,vlift,attn_mask=selected,
                    dropout_p=0.,is_causal=False,scale=scale)[...,:d]
            backend='flash_patch_lift2d'
        except (ImportError,NotImplementedError,RuntimeError) as exc:
            message=str(exc).lower()
            unavailable=isinstance(exc,(ImportError,NotImplementedError)) or any(t in message for t in
                ('no available kernel','no viable backend','not supported','does not support','no kernel found'))
            if not unavailable:raise
            reason=str(exc);result=_pg_chunk(q,k,v,qr,kr,n_global,scale,mask,is_causal,mode);backend='query_chunk256_fallback'
    else:
        result=F.scaled_dot_product_attention(qlift,klift,vlift,attn_mask=selected,
                    dropout_p=0.,is_causal=False,scale=scale)[...,:d];backend='cpu_patch_lift2d'
    if audit is not None:
        key=mode+':'+backend;audit['backend_counts'][key]=audit['backend_counts'].get(key,0)+1
        if reason is not None and reason not in audit['fallback_reasons']:audit['fallback_reasons'].append(reason)
    return result


class NativeReadoutCapture:
    def __init__(self,mode,layers,output_dtype,sink,store):
        self.mode=mode;self.layers=layers;self.output_dtype=output_dtype;self.sink=sink;self.store=store
        self.frames={i:[] for i in layers}
        self.audit=dict(mode=mode,native_calls=0,side_records=0,native_replica_exact=[],backend_counts={},fallback_reasons=[],
            side_errors=[],rollback_complete=False,native_trajectory_return='Original tensor object, never detached/replaced/modified',
            readout='Projected/gated/output-normed attention -> cloned native LayerScale/drop_path(eval) -> full-sequence final norm -> remove prefix',
            scale='Native FP32 patch x channel std, correction1; before any optional storage quantization. Not RMS.',
            weights='External actual RSRM support weights only; this module computes no layer weights',
            no_row_lse_reuse=True,whole_model_origin_invariance_claim=False,
            matched_scale_source='ORIGINAL native capture; never observe_modified scales')
    def _record(self,index,r0,rcf,rpad,n_global,record):
        patch0=r0[:,n_global:];patchcf=rcf[:,n_global:];patchpad=rpad[:,n_global:]
        if patch0.shape[1]==0:raise ValueError('No patch tokens available for a segmentation readout')
        scale=patch0.float().std(dim=(1,2),keepdim=True).detach().cpu()
        first=patch0.detach().to(device='cpu',dtype=self.output_dtype).clone()
        arrays=dict(R0=first)
        arrays['Rcf']=first if rcf is r0 else patchcf.detach().to(device='cpu',dtype=self.output_dtype).clone()
        arrays['Rpad']=first if rpad is r0 else patchpad.detach().to(device='cpu',dtype=self.output_dtype).clone()
        entry=dict(arrays,native_std=scale,metadata=record)
        pass_index=len(self.frames[index])
        if self.sink is not None:self.sink(index,pass_index,entry)
        self.frames[index].append(entry if self.store else dict(metadata=record))
        self.audit['side_records']+=1
    def _validate(self,pass_index):
        if not self.store:raise RuntimeError('Sink-only capture did not retain CPU streams')
        missing=[i for i in self.layers if len(self.frames[i])<=pass_index]
        failed=[e for e in self.audit['side_errors'] if e['pass_index']==pass_index]
        if missing or failed:raise RuntimeError(f'Invalid/incomplete side readouts: missing={missing}, errors={failed}')
    def feature_streams(self,pass_index=0):
        self._validate(pass_index)
        return {name:torch.stack([self.frames[i][pass_index][name] for i in self.layers]) for name in ('R0','Rcf','Rpad')}
    def raw_maps(self,which='R0',grid=None,pass_index=0,device=None,dtype=None):
        if grid is None:raise ValueError('Caller must provide actual native patch grid, e.g.(64,64)')
        self._validate(pass_index)
        if which not in ('R0','Rcf','Rpad'):raise ValueError('Unknown readout arm')
        array=torch.stack([self.frames[i][pass_index][which] for i in self.layers]);L,B,P,C=array.shape;h,w=grid
        if h*w!=P:raise ValueError('Patch grid does not match native trace')
        output=array.permute(1,0,3,2).reshape(B,L,C,h,w)
        return output.to(device=device,dtype=dtype) if device is not None or dtype is not None else output
    def native_scales(self,pass_index=0):
        if self.mode=='observe_modified':raise RuntimeError('Freeze scales from ORIGINAL capture, not full-GL competitor')
        self._validate(pass_index)
        return torch.stack([self.frames[i][pass_index]['native_std'] for i in self.layers],dim=1).unsqueeze(-1)


def _native_transform(block,final_norm,value):
    # detach alone still aliases original storage! clone before any in-place ls1.
    output=value.clone()
    if hasattr(block,'ls1'):output=block.ls1(output)
    elif hasattr(block,'gamma_1'):
        if block.gamma_1 is not None:output=block.gamma_1*output
    else:raise ValueError('Unknown native block LayerScale; caller readout_transform required')
    if hasattr(block,'drop_path1'):output=block.drop_path1(output)
    return final_norm(output.clone())


@contextmanager
def native_attention_readouts(backbone,mode='native_only',*,layers=None,output_dtype=torch.float32,
                              sink=None,store=True,rope_apply=None,readout_transform=None):
    """Instance-local hooks; use actual Eva backbone with .norm, not encoder wrapper.

    readout_transform(index, CLONED projected attention tensor) can adapt another
    verified native RSRM interface. It must not alter modules/weights/buffers. Default
    uses actual parent.ls1 OR learned gamma_1, drop_path1(eval), backbone.norm.
    Side exceptions are recorded and original encoder trajectory continues. Calling
    feature_streams/raw_maps then rejects missing/invalid features, never falls back
    to guessed zeros. Multiple native passes (SQ B2 then FlipS B1) use pass_index.
    """
    if mode not in ('native_only','local_pg','observe_modified'):raise ValueError('Unknown fixed readout mode')
    if any(module.training for module in backbone.modules()):raise RuntimeError('All backbone/readout modules must be frozen eval, including LayerScale/drop_path/norm')
    named=dict(backbone.named_modules());attention=[(name,m) for name,m in named.items() if type(m).__name__=='EvaAttention']
    if not attention:raise ValueError('No verified live EvaAttention modules')
    indices=list(range(len(attention))) if layers is None else sorted(set(int(i) for i in layers))
    if not indices or any(i<0 or i>=len(attention) for i in indices):raise ValueError('Invalid selected native layers')
    if readout_transform is None and not hasattr(backbone,'norm'):raise ValueError('Pass actual backbone .m or explicit verified transform')
    capture=NativeReadoutCapture(mode,indices,output_dtype,sink,store);original=[]
    try:
        for index in indices:
            name,module=attention[index]
            if module.training:raise RuntimeError('Native trace is strictly frozen eval-only')
            owner_path=name.rsplit('.',1)[0] if '.' in name else ''
            block=named[owner_path] if owner_path in named else backbone
            apply=rope_apply or getattr(sys.modules[type(module).__module__],'apply_rot_embed_cat',None)
            if mode=='local_pg' and apply is None:raise ValueError('Native rotary helper missing')
            native=module.forward
            native_function=getattr(native,'__func__',native)
            if mode!='observe_modified' and getattr(native_function,'__module__','').rsplit('.',1)[-1]=='global_content_attention':
                raise RuntimeError('Outer all-layer intervention detected; use observe_modified and ORIGINAL frozen scales')
            had='forward' in module.__dict__;old_instance=module.__dict__.get('forward')
            original.append((module,had,old_instance,native))
            transform=(lambda value,_idx=index,_block=block:readout_transform(_idx,value.clone())) if readout_transform else (lambda value,_block=block:_native_transform(_block,backbone.norm,value))
            def observed(this,x,rope=None,attn_mask=None,is_causal=False,_index=index,_native=native,_apply=apply,_transform=transform):
                # EXACT once. Side branches never supply the return value.
                actual=_native(x,rope=rope,attn_mask=attn_mask,is_causal=is_causal)
                capture.audit['native_calls']+=1;pass_index=len(capture.frames[_index])
                try:
                    before=actual.detach().clone();r0=_transform(actual)
                    metadata=dict(layer=_index,prefix=this.num_prefix_tokens,input_shape=list(x.shape),mode=mode)
                    if mode!='local_pg':
                        rcf=rpad=r0
                        metadata['native_replica_checked']=False
                    else:
                        q,k,v,qr,kr,gate=eva_components(this,x,rope,_apply)
                        u0=native_heads(this,qr,kr,v,attn_mask,is_causal)
                        replica=project_heads(this,u0,x,gate);exact=bool(torch.equal(replica,actual))
                        capture.audit['native_replica_exact'].append(dict(layer=_index,pass_index=pass_index,exact=exact,
                            max_abs=float((replica.float()-actual.float()).abs().max())))
                        if not exact:raise RuntimeError('Native SDPA/projected replica is not byte-exact; reject counterfactual readout')
                        upg=patch_readout_heads(q,k,v,qr,kr,this.num_prefix_tokens,this.scale,attn_mask,is_causal,
                                               mode='local_pg',audit=capture.audit)
                        upad=patch_readout_heads(q,k,v,qr,kr,this.num_prefix_tokens,this.scale,attn_mask,is_causal,
                                                mode='padded_native',audit=capture.audit)
                        # FULL-N norm/gate/proj. Keep native global rows, so only PG
                        # patch readout changes; original full output remains return.
                        merged_cf=torch.cat([u0[...,:this.num_prefix_tokens,:],upg],dim=-2)
                        merged_pad=torch.cat([u0[...,:this.num_prefix_tokens,:],upad],dim=-2)
                        cf_projected=project_heads(this,merged_cf,x,gate);pad_projected=project_heads(this,merged_pad,x,gate)
                        rcf=_transform(cf_projected);rpad=_transform(pad_projected)
                        metadata.update(native_replica_checked=True,
                            projected_global_rows_exact=bool(torch.equal(cf_projected[:,:this.num_prefix_tokens],actual[:,:this.num_prefix_tokens])))
                    if not torch.equal(actual,before):raise AssertionError('Side readout mutated original attention tensor')
                    capture._record(_index,r0,rcf,rpad,this.num_prefix_tokens,metadata)
                except Exception as exc:
                    capture.audit['side_errors'].append(dict(layer=_index,pass_index=pass_index,error=repr(exc)))
                    capture.frames[_index].append(dict(error=repr(exc)))
                return actual # SAME object, even if side branch failed.
            module.forward=MethodType(observed,module)
        yield capture
    finally:
        for module,had,old_instance,native in reversed(original):
            if had:module.forward=old_instance
            else:delattr(module,'forward')
            restored=module.forward
            if getattr(restored,'__func__',restored) is not getattr(native,'__func__',native):raise AssertionError('Native attention pointer rollback failed')
        capture.audit['rollback_complete']=True


def self_check():
    torch.set_num_threads(1);error=0.;padded_error=0.
    for seed in range(10):
        generator=torch.Generator().manual_seed(seed);B=1+seed%2;H=1+seed%3;N=8+seed;d=4+2*(seed%3)
        q,k,v,qr,kr=[torch.randn(B,H,N,d,generator=generator,dtype=torch.float64) for _ in range(5)]
        ng=[0,1,5,N][seed%4];mask=None
        if seed%3==1:mask=torch.zeros(N,N,dtype=torch.float64);mask[:,0]=-.2
        if seed%3==2:mask=torch.ones(N,N,dtype=torch.bool);mask[:,-1]=False
        expected=explicit_pg_reference(q,k,v,qr,kr,ng,mask=mask)
        actual=patch_readout_heads(q,k,v,qr,kr,ng,mask=mask)
        if actual.numel():error=max(error,float((actual-expected).abs().max()))
        assert error<=1e-12
        padded=patch_readout_heads(q,k,v,qr,kr,ng,mask=mask,mode='padded_native')
        selected=_patch_mask(mask,ng,N,False,q.device)
        native=F.scaled_dot_product_attention(qr[...,ng:,:],kr,v,attn_mask=selected,scale=d**-.5)
        if padded.numel():padded_error=max(padded_error,float((padded-native).abs().max()))
        assert padded_error<=1e-12
        if ng<N:
            causal=patch_readout_heads(q,k,v,qr,kr,ng,is_causal=True)
            reference=explicit_pg_reference(q,k,v,qr,kr,ng,is_causal=True)
            assert float((causal-reference).abs().max())<=1e-12
    import json
    print(json.dumps(dict(cpu_fp64_cases=10,max_pg_fullsoftmax_error=error,max_padded_native_patch_error=padded_error,
                         prefix0_and_allglobal=True,causal_prefix_offset=True,no_cuda_calls=True,no_lse_shortcut=True)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:self_check()
    else:parser.error('Import native_attention_readouts in the caller, or use --self-check')
