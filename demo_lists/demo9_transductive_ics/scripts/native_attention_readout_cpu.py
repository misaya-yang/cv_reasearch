#!/usr/bin/env python3
"""Local CPU integration fixtures; NO installed-timm/whole-GPU claims.

Small four-block Eva-source attention, learned gamma_1 and in-place ls1 branches,
MLP/residual/final norm verify the original trajectory even on side exceptions.
Actual live-source attention operations come from global_content_attention_cpu.
"""
import json
from types import MethodType
import torch
from torch import nn
from global_content_attention_cpu import EvaAttention,apply_rot_embed_cat
from global_content_attention import global_content_attention
from native_attention_readout import native_attention_readouts


class InplaceLayerScale(nn.Module):
    def __init__(self,dim):
        super().__init__();self.gamma=nn.Parameter(torch.linspace(.1,.9,dim));self.inplace=True
    def forward(self,x):return x.mul_(self.gamma)


class InplaceFinalNorm(nn.Module):
    def forward(self,x):
        x.sub_(x.mean(-1,keepdim=True));x.div_(torch.sqrt(x.square().mean(-1,keepdim=True)+1e-5));return x


class Block(nn.Module):
    def __init__(self,index):
        super().__init__();self.norm1=nn.LayerNorm(24);self.norm2=nn.LayerNorm(24)
        self.attn=EvaAttention(prefix=5,gated=index%2==0,qk_norm=index%2==1)
        self.attn.native_invocations=0;native=self.attn.forward
        def counted(this,x,rope=None,attn_mask=None,is_causal=False):
            this.native_invocations+=1
            value=native(x,rope,attn_mask,is_causal);this.native_return=value
            this.native_return_snapshot=value.detach().clone();return value
        self.attn.forward=MethodType(counted,self.attn)
        if index%2==0:self.ls1=InplaceLayerScale(24)
        else:self.gamma_1=nn.Parameter(torch.linspace(.15,.75,24))
        self.drop_path1=nn.Identity();self.mlp=nn.Sequential(nn.Linear(24,40),nn.GELU(),nn.Linear(40,24))
    def forward(self,x,rope):
        update=self.attn(self.norm1(x),rope);self.native_return_identity=update is self.attn.native_return
        update=self.ls1(update) if hasattr(self,'ls1') else self.gamma_1*update
        x=x+self.drop_path1(update);return x+self.mlp(self.norm2(x))


class Backbone(nn.Module):
    def __init__(self,inplace_norm=False):
        super().__init__();self.blocks=nn.ModuleList([Block(i) for i in range(4)])
        self.norm=InplaceFinalNorm() if inplace_norm else nn.LayerNorm(24)
    def forward(self,x,rope):
        for block in self.blocks:x=block(x,rope)
        return self.norm(x)


def main():
    torch.set_num_threads(1);count=0;replicas=0
    for seed in range(10):
        torch.manual_seed(seed);model=Backbone(inplace_norm=seed%2==0).double().eval().requires_grad_(False)
        x=torch.randn(2,14,24,dtype=torch.float64);angles=torch.randn(9,8,dtype=torch.float64);rope=torch.cat([angles.sin(),angles.cos()],-1)
        parameters={n:p.detach().clone() for n,p in model.named_parameters()}
        original_pointers=[block.attn.forward for block in model.blocks]
        baseline=model(x.clone(),rope)
        for block in model.blocks:block.attn.native_invocations=0
        with native_attention_readouts(model,mode='native_only',rope_apply=apply_rot_embed_cat,output_dtype=torch.float64) as native:
            identity=model(x.clone(),rope)
        assert torch.equal(identity,baseline) and native.audit['native_calls']==4
        assert all(block.attn.native_invocations==1 and block.native_return_identity for block in model.blocks)
        arrays=native.feature_streams();assert torch.equal(arrays['R0'],arrays['Rcf']) and torch.equal(arrays['R0'],arrays['Rpad'])
        assert native.raw_maps('R0',(3,3)).shape==(2,4,24,3,3)
        scale=native.native_scales();expected=arrays['R0'].float().std((2,3),keepdim=True).permute(1,0,2,3).unsqueeze(-1)
        assert torch.equal(scale,expected) and scale.shape==(2,4,1,1,1)
        for block in model.blocks:block.attn.native_invocations=0
        with native_attention_readouts(model,mode='local_pg',rope_apply=apply_rot_embed_cat,output_dtype=torch.float64) as local:
            output=model(x.clone(),rope)
            flip=model(x[:1].clone(),rope) # One context, second independent B1 native pass.
        assert torch.equal(output,baseline) and torch.equal(flip,model(x[:1].clone(),rope))
        assert all(block.native_return_identity for block in model.blocks)
        assert not local.audit['side_errors'] and all(r['exact'] for r in local.audit['native_replica_exact'])
        replicas+=len(local.audit['native_replica_exact'])
        assert local.raw_maps('R0',(3,3),pass_index=0).shape==(2,4,24,3,3)
        assert local.raw_maps('Rcf',(3,3),pass_index=1).shape==(1,4,24,3,3)
        assert local.audit['rollback_complete']
        # Arbitrary verified transform may throw or mutate its INPUT, but that input
        # is a clone, and the original tensor must still reach every native block.
        def exploding(index,value):
            if index==1:value.zero_();raise RuntimeError('deliberate in-place side callback failure')
            value.add_(.5);return model.norm(value)
        with native_attention_readouts(model,mode='local_pg',rope_apply=apply_rot_embed_cat,readout_transform=exploding) as bad:
            original_despite_error=model(x.clone(),rope)
        assert torch.equal(original_despite_error,baseline) and bad.audit['side_errors']
        try:bad.feature_streams();raise AssertionError('Invalid side trace accepted')
        except RuntimeError:pass
        # Observe outer all-layer GL intervention: collect ITS true forward return,
        # never recompute/replace it with d-native SDPA or assert a native replica.
        with global_content_attention(model,rope_apply=apply_rot_embed_cat):
            full_gl_baseline=model(x.clone(),rope)
            with native_attention_readouts(model,mode='observe_modified',rope_apply=apply_rot_embed_cat) as gl:
                observed=model(x.clone(),rope)
        assert torch.equal(observed,full_gl_baseline) and not gl.audit['native_replica_exact']
        try:gl.native_scales();raise AssertionError('Modified scales allowed as original match')
        except RuntimeError:pass
        for block,old in zip(model.blocks,original_pointers):
            assert block.attn.forward.__func__ is old.__func__
        for name,param in model.named_parameters():assert torch.equal(param,parameters[name])
        assert torch.equal(model(x.clone(),rope),baseline)
        count+=1
    print(json.dumps(dict(cpu_backbone_fixtures=count,blocks_per_fixture=4,byte_exact_native_trajectory_cases=count,
        inplace_layer_scale_and_final_norm=True,native_output_same_object=True,native_only_raw_identity=True,
        native_replica_byte_exact_records=replicas,side_error_trajectory_byte_exact=True,
        missing_or_invalid_readouts_rejected=True,layer_std_correction1=True,two_native_passes_B2_then_B1=True,
        observe_modified_not_renativized=True,original_parameter_values_unchanged=True,original_forward_pointers_restored=True,
        fixture_scope='Locally constructed small CPU blocks using known live source operations; no installed whole-DINO/GPU validation',cuda_calls=0)))


if __name__=='__main__':main()
