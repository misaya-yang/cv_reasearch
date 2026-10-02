#!/usr/bin/env python3
"""CPU-only installed-timm EvaAttention/EvaBlock integration test, no DINO weights.

Run on a host with the already installed actual timm package:
 CUDA_VISIBLE_DEVICES='' PYTHONPATH=/root/demo4_cache/env /root/miniconda3/bin/python \
   scripts/native_attention_readout_installed_cpu.py --out /tmp/native_readout_installed_cpu.json

This is real installed Eva modules in a small32-channel four-block container,
not a locally copied Eva forward, a pretrained whole-model reproduction, or GPU
Flash validation. No timm.create_model, pretrained weight loader or download.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import argparse
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn
import timm
from timm.models.eva import EvaAttention,EvaBlock
from timm.layers import LayerNorm
from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3,apply_rot_embed_cat
from global_content_attention import global_content_attention
from native_attention_readout import (native_attention_readouts,eva_components,native_heads,
    project_heads,explicit_pg_reference)


DIM=32;HEADS=4;PREFIX=5;GRID=(4,4);TOKENS=PREFIX+GRID[0]*GRID[1]
NORM=partial(LayerNorm,eps=1e-5)


def make_rope(half=True):
    module=RotaryEmbeddingDinoV3(dim=DIM//HEADS,temperature=100,rotate_half=half).cpu().eval()
    value=module.get_embed(shape=GRID)
    assert tuple(value.shape)==(GRID[0]*GRID[1],2*(DIM//HEADS))
    return module,value


def make_attention(seed):
    attention=EvaAttention(dim=DIM,num_heads=HEADS,qkv_bias=seed%2==0,
        qkv_fused=seed%3!=0,qkv_bias_separate=seed%3==1,num_prefix_tokens=PREFIX,
        qk_norm=seed%2==1,scale_norm=seed%3==2,norm_layer=NORM,
        gated=seed%3==0,rotate_half=True,attn_drop=0.,proj_drop=0.,device='cpu').double().eval().requires_grad_(False)
    with torch.no_grad():
        if getattr(attention,'q_bias',None) is not None:
            attention.q_bias.copy_(torch.linspace(-.2,.1,DIM,dtype=torch.float64))
            attention.v_bias.copy_(torch.linspace(.15,-.05,DIM,dtype=torch.float64))
    return attention


class InstalledBackboneContainer(nn.Module):
    """Calls REAL EvaBlock.forward; no handwritten block/attention replacement."""
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([EvaBlock(dim=DIM,num_heads=HEADS,qkv_bias=False,qkv_fused=True,
            mlp_ratio=1.5,num_prefix_tokens=PREFIX,attn_type='eva',rotate_half=True,
            norm_layer=NORM,init_values=.2,attn_drop=0.,proj_drop=0.,drop_path=0.,device='cpu') for _ in range(4)])
        self.norm=NORM(DIM)
        self.double().eval().requires_grad_(False)
        # A learned-shaped nonconstant gamma vector proves code reads real tensors,
        # not the documented constructor's scalar initialization value.
        with torch.no_grad():
            for i,block in enumerate(self.blocks):
                assert isinstance(block.gamma_1,nn.Parameter) and block.gamma_1.shape==(DIM,)
                block.gamma_1.copy_(torch.linspace(.08,.6,DIM,dtype=torch.float64)+(i*.01))
        self.call_counts=[0]*len(self.blocks)
        self.assert_native_return_identity=True
        for i,block in enumerate(self.blocks):
            original=block.attn.forward
            def counted(this,x,rope=None,attn_mask=None,is_causal=False,_i=i,_original=original):
                self.call_counts[_i]+=1
                result=_original(x,rope=rope,attn_mask=attn_mask,is_causal=is_causal)
                this.test_native_return=result
                return result
            block.attn.forward=MethodType(counted,block.attn)
            def same_return(this,args,result):
                if self.assert_native_return_identity:
                    assert result is this.test_native_return,'Side readout replaced native output object'
            block.attn.register_forward_hook(same_return)
    def forward(self,x,rope):
        for block in self.blocks:x=block(x,rope=rope)
        return self.norm(x)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out');parser.add_argument('--cases',type=int,default=10);a=parser.parse_args()
    if a.cases!=10:parser.error('The frozen CPU acceptance card uses ten cases')
    if torch.cuda.is_initialized() or torch.cuda.is_available():raise RuntimeError('This installed-module acceptance must be strictly CPU')
    torch.set_num_threads(1)
    _,rope=make_rope();global_max=0.;pg_max=0.;native_padded_max=0.;replicas=0
    report=dict(state='RUNNING',torch_version=torch.__version__,timm_version=timm.__version__,
        module_scope='Real installed EvaAttention and EvaBlock, small32-dimensional four-block container; no pretrained ViT-L or GPU claim',
        dimension=DIM,heads=HEADS,prefix=PREFIX,grid=GRID,rope_shape=list(rope.shape),cases=[],
        source_sha256={name:hashlib.sha256(inspect.getsource(obj).encode()).hexdigest() for name,obj in
                       [('EvaAttention.forward',EvaAttention.forward),('EvaBlock.forward',EvaBlock.forward),('apply_rot_embed_cat',apply_rot_embed_cat)]})
    with torch.inference_mode():
        for seed in range(10):
            torch.manual_seed(seed);x=torch.randn(2,TOKENS,DIM,dtype=torch.float64)
            attn=make_attention(seed);parameters={name:p.detach().clone() for name,p in attn.named_parameters()}
            native=attn(x,rope=rope);original_forward=attn.forward.__func__
            with global_content_attention(attn,enabled=False) as noop:
                assert torch.equal(attn(x,rope=rope),native) and noop['calls']==0
                assert attn.forward.__func__ is original_forward
            q,k,v,qr,kr,gate=eva_components(attn,x,rope,apply_rot_embed_cat)
            heads=native_heads(attn,qr,kr,v);replica=project_heads(attn,heads,x,gate)
            assert torch.equal(replica,native)
            pg=explicit_pg_reference(q,k,v,qr,kr,PREFIX,attn.scale)
            expected=project_heads(attn,torch.cat([heads[...,:PREFIX,:],pg],-2),x,gate)
            # Actual installed attention enclosed in a genuine real EvaBlock parent.
            model=InstalledBackboneContainer();pointers=[b.attn.forward for b in model.blocks]
            baseline=model(x.clone(),rope);model.call_counts=[0]*4
            with native_attention_readouts(model,mode='native_only',output_dtype=torch.float64) as only:
                original=model(x.clone(),rope)
            assert torch.equal(original,baseline) and model.call_counts==[1]*4
            r0=only.feature_streams()['R0'];assert torch.equal(r0,only.feature_streams()['Rcf'])
            assert only.raw_maps('R0',GRID).shape==(2,4,DIM,*GRID)
            scales=r0.float().std((2,3),keepdim=True).permute(1,0,2,3).unsqueeze(-1)
            assert torch.equal(scales,only.native_scales())
            model.call_counts=[0]*4
            with native_attention_readouts(model,mode='local_pg',output_dtype=torch.float64) as local:
                unchanged=model(x.clone(),rope)
                flip=model(x[:1].clone(),rope)
            assert model.call_counts==[2]*4 and local.audit['native_calls']==8
            assert torch.equal(unchanged,baseline)
            assert torch.equal(flip,model(x[:1].clone(),rope))
            assert not local.audit['side_errors']
            assert all(item['exact'] for item in local.audit['native_replica_exact'])
            replicas+=len(local.audit['native_replica_exact'])
            assert local.raw_maps('Rcf',GRID,pass_index=1).shape==(1,4,DIM,*GRID)
            # Readout callback failure MUST not stop or mutate native blocks.
            def bad_transform(index,value):
                if index==1:value.zero_();raise RuntimeError('deliberate side failure')
                return model.norm(value)
            with native_attention_readouts(model,mode='local_pg',readout_transform=bad_transform) as failed:
                still_native=model(x.clone(),rope)
            assert torch.equal(still_native,baseline) and failed.audit['side_errors']
            try:failed.raw_maps('Rcf',GRID);raise AssertionError('Invalid trace accepted')
            except RuntimeError:pass
            # Full GL competitor: observe its actual returned update, never use
            # native d-head SDPA to replace/renativize the modified trajectory.
            model.assert_native_return_identity=False
            try:
                with global_content_attention(model):
                    full_gl=model(x.clone(),rope)
                    with native_attention_readouts(model,mode='observe_modified') as observed:
                        same_gl=model(x.clone(),rope)
            finally:model.assert_native_return_identity=True
            assert torch.equal(full_gl,same_gl) and not observed.audit['native_replica_exact']
            try:observed.native_scales();raise AssertionError('Modified scales accepted')
            except RuntimeError:pass
            for block,pointer in zip(model.blocks,pointers):assert block.attn.forward.__func__ is pointer.__func__
            assert torch.equal(model(x.clone(),rope),baseline)
            # Pure installed Eva integration of the all-global lift is checked
            # independently of the block trajectory/capture machinery.
            from global_content_attention import explicit_piecewise_reference
            expected_global=project_heads(attn,explicit_piecewise_reference(q,k,v,qr,kr,PREFIX,attn.scale),x,gate)
            with global_content_attention(attn) as global_audit:actual_global=attn(x,rope=rope)
            err=float((actual_global-expected_global).abs().max());global_max=max(global_max,err);assert err<=1e-12
            with global_content_attention(attn,mode='padded_native'):padded=attn(x,rope=rope)
            err=float((padded-native).abs().max());native_padded_max=max(native_padded_max,err);assert err<=1e-12
            # Local PG integration against explicit full-key softmax, using the
            # real installed module and its exact post-QKV output transforms.
            from native_attention_readout import patch_readout_heads
            actual_pg=patch_readout_heads(q,k,v,qr,kr,PREFIX,attn.scale)
            actual_projected=project_heads(attn,torch.cat([heads[...,:PREFIX,:],actual_pg],-2),x,gate)
            err=float((actual_projected-expected).abs().max());pg_max=max(pg_max,err);assert err<=1e-12
            for name,param in attn.named_parameters():assert torch.equal(param,parameters[name])
            assert attn.forward.__func__ is original_forward and torch.equal(attn(x,rope=rope),native)
            report['cases'].append(dict(seed=seed,whole_small_block_trajectory_byte_exact=True,side_error_trajectory_byte_exact=True,
                local_replica_records=len(local.audit['native_replica_exact']),native_only_same_update=True,
                observe_GL_actual_output_byte_exact=True,original_pointers_restored=True))
    report.update(state='PASSED',cases_count=10,native_replica_exact_records=replicas,
                  max_pg_projected_error=pg_max,max_global_lift_projected_error=global_max,
                  max_padded_native_projected_error=native_padded_max,
                  raw_std_correction1=True,actual_learned_gamma_vector_read=True,
                  native_calls_once=True,two_passes_B2_then_B1=True,cuda_initialized=torch.cuda.is_initialized(),
                  no_pretrained_model_or_weights_loaded=True,no_downloads=True)
    if a.out:
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
