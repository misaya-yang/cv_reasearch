#!/usr/bin/env python3
"""Ten actual public-source CPU fixtures, not a pretrained segmentation score.

Only the constructor SVD uses a D8/rank2 fixture. Prediction, RGB grouping,
reference gate, all source hooks and postprocessor remain actual source code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['HF_HUB_OFFLINE']='1'


def make_fixture(foris_root, seed=6010, image_size=64):
    """Reusable actual source composition for independent worker CPU smokes.

    Returns a label-free query context, synthetic encoder and audit. New-view
    callbacks re-enter the same actual source, with one shared finite budget.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root));sys.path.insert(0,str(foris_root))
    from models.foris import FoRIS
    from tics.ten_direction_context import TenDirectionContext,ViewBudget,capture_exact_native_and_middle
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
            self.num_prefix_tokens=1
    class Encoder(torch.nn.Module):
        def __init__(self):super().__init__();self.m=Backbone();self.calls=0
        def get_intermediate_layers(self,x,n=1,reshape=True):
            assert n==1 and reshape
            self.calls+=1;v=F.avg_pool2d(x,16)
            v=torch.cat([v,v.square(),v[:,:2]-v[:,1:3]],1)
            tokens=v.flatten(2).transpose(1,2)
            tokens=torch.cat([tokens.mean(1,keepdim=True),tokens],1)
            for block in self.m.blocks:tokens=block(tokens)
            return [tokens[:,1:].transpose(1,2).reshape(v.shape)]
    g=torch.Generator().manual_seed(seed)
    s=Image.fromarray(torch.randint(0,256,(image_size,image_size,3),generator=g,dtype=torch.uint8).numpy())
    q=Image.fromarray(torch.randint(0,256,(image_size,image_size,3),generator=g,dtype=torch.uint8).numpy())
    mask=torch.zeros(image_size,image_size,dtype=torch.bool)
    mask[image_size//4:3*image_size//4,image_size//4:3*image_size//4]=True
    encoder=Encoder().eval()
    with patch.object(FoRIS,'_build_positional_basis',lambda this,device:torch.eye(8)[:,:2]):
        host=FoRIS(encoder,image_size=image_size,svd_components=2,tau=.6,
                   mask_refiner='bilinear',resize_to_orig_size=False,device='cpu').eval()
    budget=ViewBudget(max_extra_b2=8,native_b2_calls=1)
    def construct(s_rgb,s_mask,q_rgb,shared_budget,metadata):
        observed={}
        try:
            host.set_reference(s_rgb,s_mask);host.set_target(q_rgb)
            with capture_exact_native_and_middle(host,encoder) as observed:
                uncached=host.segment().reshape(image_size,image_size).bool()
        finally:
            shared_budget.actual_b2_calls+=observed.get('encoder_forward_calls',0)
            host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None
        ctx=TenDirectionContext(host,encoder,s_rgb,q_rgb,s_mask,
            raw_native=observed['raw_native'],layer_maps={6:observed[6],12:observed[12]},
            budget=shared_budget,new_view_callback=construct,coordinate_metadata=metadata)
        got=ctx.initialize_native()
        assert torch.equal(uncached,got['mask'])
        got['uncached_public_exact']=True
        return ctx
    with torch.inference_mode():ctx=construct(s,mask,q,budget,None)
    return ctx,encoder,dict(actual_public_source=True,fixture_D=8,image_size=image_size,
                            CRF_exercised=False,CUDA_initialized=torch.cuda.is_initialized())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foris-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve old CPU receipt; fresh output required')
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU context audit initialized CUDA')
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root));sys.path.insert(0,str(a.foris_root))
    from models.foris import FoRIS
    from tics.ten_direction_context import TenDirectionContext,ViewBudget,capture_exact_native_and_middle
    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
            self.num_prefix_tokens=1
    class SyntheticEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.m=TinyBackbone();self.calls=0
        def get_intermediate_layers(self,x,n=1,reshape=True):
            if n!=1 or not reshape:raise ValueError('Actual native n=1 extraction only')
            self.calls+=1;pool=F.avg_pool2d(x,16)
            features=torch.cat([pool,pool.square(),pool[:,:2]-pool[:,1:3]],1)
            tokens=features.flatten(2).transpose(1,2)
            tokens=torch.cat([tokens.mean(1,keepdim=True),tokens],1)
            for block in self.m.blocks:tokens=block(tokens)
            return [tokens[:,1:].transpose(1,2).reshape(features.shape)]
    def basis(this,device):return torch.eye(8)[:,:2]
    records=[]
    for seed in range(10):
        g=torch.Generator().manual_seed(6010+seed)
        s=Image.fromarray(torch.randint(0,256,(64,64,3),generator=g,dtype=torch.uint8).numpy())
        q=Image.fromarray(torch.randint(0,256,(64,64,3),generator=g,dtype=torch.uint8).numpy())
        mask=torch.zeros(64,64,dtype=torch.bool);mask[16:48,16:48]=True
        encoder=SyntheticEncoder().eval()
        with torch.inference_mode(),patch.object(FoRIS,'_build_positional_basis',basis):
            host=FoRIS(encoder,image_size=64,svd_components=2,tau=.6,
                mask_refiner='bilinear',resize_to_orig_size=False,device='cpu').eval()
            host.set_reference(s,mask);host.set_target(q)
            with capture_exact_native_and_middle(host,encoder) as observed:
                uncached=host.segment().reshape(64,64).bool()
            ctx=TenDirectionContext(host,encoder,s,q,mask,raw_native=observed['raw_native'],
                layer_maps={6:observed[6],12:observed[12]},budget=ViewBudget())
            got=ctx.initialize_native()
            assert torch.equal(uncached,got['mask'])
            assert encoder.calls==1
            same=ctx.replay(ref_mask=mask)
            assert torch.equal(same['mask'],got['mask']) and same['stages']['part1_gate_recomputed']
            alternative=ctx.replay(ref_mask=~mask)
            assert alternative['stages']['part1_gate_recomputed'] and encoder.calls==1
            scope=host._build_seed_cluster_prior.__func__.__globals__
            cluster=scope['agglomerative_clustering']
            def fail(**kwargs):raise ValueError('expected own stage error')
            try:ctx.replay(part2_override=fail)
            except ValueError as e:assert str(e)=='expected own stage error'
            else:raise AssertionError('Callback error swallowed')
            assert scope['agglomerative_clustering'] is cluster
            assert host._ref_images is None and host._tgt_image is None
            recovered=ctx.replay()
            assert torch.equal(recovered['mask'],got['mask']) and encoder.calls==1
            saved=ctx.expected_raw_input.clone();ctx.expected_raw_input=ctx.expected_raw_input+1
            try:ctx.replay()
            except RuntimeError as e:assert 'cache requested' in str(e)
            else:raise AssertionError('Changed cached B2 input accepted')
            finally:ctx.expected_raw_input=saved
            assert not torch.cuda.is_initialized()
        records.append(dict(seed=6010+seed,uncached_public_vs_cached_exact=True,
            all_identity_callbacks_exact=True,reference_hypothesis_Part1_recomputed=True,
            changed_B2_cache_rejected=True,callback_error_restores_source=True,
            native_encoder_calls=encoder.calls,middle_blocks=[6,12],
            public_RGB_position_path=True,CRF_exercised=False))
    value=dict(state='CPU_TEN_CONTEXT_PASSED',cases=10,records=records,
        CUDA_initialized=False,fixture='Actual source functions, D8/image64/rank2 synthetic encoder; not production DINO or segmentation accuracy',
        actual_source_execution=True,
        source_sha256=hashlib.sha256((a.foris_root/'models/foris.py').read_bytes()).hexdigest(),
        context_sha256=hashlib.sha256((root/'tics/ten_direction_context.py').read_bytes()).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(value,indent=1)+'\n')
    print(json.dumps(dict(state=value['state'],cases=10,CUDA_initialized=False)),flush=True)


if __name__=='__main__':main()
