#!/usr/bin/env python3
"""Local CPU verification of live-source EvaAttention forward integration.

This machine has no installed timm. The fixture below uses the forward operations
read verbatim from live timm1.0.30 EvaAttention (eva.py222--301), with constructed
small CPU projection/norm modules. It is NOT a claim that installed whole-Eva or
CUDA Flash execution was tested. No package/model download or CUDA initialization.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F
from global_content_attention import global_content_attention,explicit_piecewise_reference


def apply_rot_embed_cat(x,emb,half=False):
    # Exact helper operations from live pos_embed_sincos.py281--297.
    sin,cos=emb.chunk(2,-1)
    if half:
        a,b=x.chunk(2,-1);rotated=torch.cat([-b,a],dim=-1)
    else:rotated=torch.stack([-x[...,1::2],x[...,::2]],dim=-1).flatten(-2)
    return x*cos+rotated*sin


class EvaAttention(nn.Module):
    def __init__(self,dim=24,heads=3,prefix=5,fused=True,bias=False,separate=False,qk_norm=False,scale_norm=False,gated=False,rotate_half=True):
        super().__init__();self.num_heads=heads;self.head_dim=dim//heads;self.num_prefix_tokens=prefix
        self.scale=self.head_dim**-.5;self.fused_attn=True;self.qkv_bias_separate=separate;self.rotate_half=rotate_half
        if fused:
            self.qkv=nn.Linear(dim,dim*3,bias=False);self.q_proj=self.k_proj=self.v_proj=None
            if bias:
                self.q_bias=nn.Parameter(torch.randn(dim));self.register_buffer('k_bias',torch.zeros(dim),persistent=False)
                self.v_bias=nn.Parameter(torch.randn(dim))
            else:self.q_bias=self.k_bias=self.v_bias=None
        else:
            self.qkv=None;self.q_bias=self.k_bias=self.v_bias=None
            self.q_proj=nn.Linear(dim,dim,bias=bias);self.k_proj=nn.Linear(dim,dim,bias=False);self.v_proj=nn.Linear(dim,dim,bias=bias)
        self.q_norm=nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm=nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.norm=nn.LayerNorm(dim) if scale_norm else nn.Identity()
        self.gate=nn.Linear(dim,dim,bias=bias) if gated else None
        self.proj=nn.Linear(dim,dim);self.attn_drop=nn.Dropout(.2);self.proj_drop=nn.Dropout(.3)
    def components(self,x,rope):
        B,N,C=x.shape
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
            npt=self.num_prefix_tokens;half=self.rotate_half
            qr=torch.cat([q[:,:,:npt,:],apply_rot_embed_cat(q[:,:,npt:,:],rope,half=half)],dim=2).type_as(v)
            kr=torch.cat([k[:,:,:npt,:],apply_rot_embed_cat(k[:,:,npt:,:],rope,half=half)],dim=2).type_as(v)
        return q,k,v,qr,kr
    def finish(self,output,x):
        B,N,C=x.shape;output=self.norm(output.transpose(1,2).reshape(B,N,C))
        if self.gate is not None:output=output*self.gate(x).sigmoid()
        return self.proj_drop(self.proj(output))
    def forward(self,x,rope=None,attn_mask=None,is_causal=False):
        # Same live native operations, factored only to avoid duplicate fixture code.
        q,k,v,qr,kr=self.components(x,rope)
        output=F.scaled_dot_product_attention(qr,kr,v,attn_mask=attn_mask,dropout_p=self.attn_drop.p if self.training else 0.,is_causal=is_causal)
        return self.finish(output,x)


def main():
    torch.set_num_threads(1);maximum=0.;padded_error=0.;noop_cases=0;rollback_cases=0
    for seed in range(10):
        torch.manual_seed(seed);prefix=[0,1,5][seed%3];N=11
        attn=EvaAttention(prefix=prefix,fused=seed%3!=0,bias=seed%2==0,separate=seed%3==1,
                          qk_norm=seed%2==1,scale_norm=seed%3==2,gated=seed%3==0,rotate_half=seed%2==0).double().eval().requires_grad_(False)
        x=torch.randn(2,N,24,dtype=torch.float64);angle=torch.randn(N-prefix,8,dtype=torch.float64)
        rope=torch.cat([angle.sin(),angle.cos()],-1)
        mask=torch.zeros(N,N,dtype=torch.float64);mask[:,0]=-.2
        parameters=[(name,parameter.data_ptr(),parameter.detach().clone()) for name,parameter in attn.named_parameters()]
        baseline=attn(x,rope,mask);native_function=attn.forward.__func__
        with global_content_attention(attn,enabled=False,rope_apply=apply_rot_embed_cat) as noop:
            assert torch.equal(baseline,attn(x,rope,mask));assert noop['calls']==0
            assert attn.forward.__func__ is native_function
        noop_cases+=1
        q,k,v,qr,kr=attn.components(x,rope)
        expected=attn.finish(explicit_piecewise_reference(q,k,v,qr,kr,prefix,attn.scale,mask),x)
        with global_content_attention(attn,rope_apply=apply_rot_embed_cat) as audit:
            actual=attn(x,rope,mask);error=float((actual-expected).abs().max());maximum=max(maximum,error)
            assert error<=1e-12 and audit['calls']==1
            assert audit['backend_counts']=={'cpu_sdpa_lift3d':1}
        assert audit['rollback_complete'] and attn.forward.__func__ is native_function
        assert torch.equal(baseline,attn(x,rope,mask))
        with global_content_attention(attn,rope_apply=apply_rot_embed_cat,mode='padded_native') as padded:
            value=attn(x,rope,mask);error=float((value-baseline).abs().max());padded_error=max(padded_error,error)
            assert error<=1e-12 and padded['gamma']==0
        try:
            with global_content_attention(attn,rope_apply=apply_rot_embed_cat) as interrupted:
                raise RuntimeError('deliberate body failure')
        except RuntimeError:pass
        assert interrupted['rollback_complete'] and attn.forward.__func__ is native_function;rollback_cases+=1
        for name,address,original in parameters:
            parameter=dict(attn.named_parameters())[name]
            assert parameter.data_ptr()==address and torch.equal(parameter,original)
    # Training mode must be rejected, not silently apply stochastic attention.
    bad=EvaAttention().train()
    try:
        with global_content_attention(bad,rope_apply=apply_rot_embed_cat):raise AssertionError('Training accepted')
    except RuntimeError:pass
    # Partial-entry validation failure must restore earlier patched modules.
    bundle=nn.Module();bundle.good=EvaAttention().eval();bundle.bad=EvaAttention().train()
    before=bundle.good.forward.__func__
    try:
        with global_content_attention(bundle,rope_apply=apply_rot_embed_cat):raise AssertionError('Mixed training state accepted')
    except RuntimeError:pass
    assert bundle.good.forward.__func__ is before and 'forward' not in bundle.good.__dict__
    print(json.dumps(dict(cpu_cases=10,fixture='Live-source operations; local timm unavailable, not installed-whole-model validation',
        max_modified_piecewise_output_error=maximum,max_padded_native_output_error=padded_error,
        exact_original_noop_cases=noop_cases,exception_rollback_cases=rollback_cases,
        parameter_storage_and_values_unchanged=True,fused_and_unfused_qkv=True,
        qkv_bias_and_separate_addition=True,qk_norm_output_norm_gate_and_dropout_eval=True,
        prefix0_1_5=True,training_rejected=True,partial_entry_rollback=True,cuda_calls=0)))


if __name__=='__main__':main()
