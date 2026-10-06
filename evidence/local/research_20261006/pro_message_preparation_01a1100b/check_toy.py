"""Small CPU-only independent dense-attention checks; not actual DINO results."""
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch import nn
from ics.methods.pro_message_extrapolation import (
    Config,StandardPreNormBlockAdapter,EvaBlockAdapter,build_native_cache,grouped_message,
    replay_region,descriptor,ward_partitions,predict,fuse,render)


class ToyAttention(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.num_heads,self.scale=2,(width//2)**-.5
        self.qkv=nn.Linear(width,3*width,bias=True)
        self.q_norm,self.k_norm=nn.LayerNorm(width//2),nn.LayerNorm(width//2)
        self.norm=nn.Identity();self.proj=nn.Linear(width,width);self.proj_drop=nn.Dropout(.2)


class ToyBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.norm1,self.norm2=nn.LayerNorm(width),nn.LayerNorm(width)
        self.attn=ToyAttention(width)
        self.ls1,self.ls2=nn.Linear(width,width,bias=False),nn.Linear(width,width,bias=False)
        self.drop_path1,self.drop_path2=nn.Identity(),nn.Identity()
        self.mlp=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.native_qkv_calls=0

    def qkv_native(self,hidden):
        self.native_qkv_calls+=1
        tokens,width=hidden.shape
        q,k,v=self.attn.qkv(self.norm1(hidden)).reshape(tokens,3,2,width//2).permute(1,2,0,3)
        q,k=self.attn.q_norm(q),self.attn.k_norm(k)
        # This toy rotation is explicit and fixed; not actual DINO RoPE.
        angle=torch.arange(tokens,dtype=hidden.dtype,device=hidden.device)[None,:,None]/7
        def rotate(z):
            left,right=z.chunk(2,dim=-1)
            return torch.cat((left*torch.cos(angle)-right*torch.sin(angle),
                              left*torch.sin(angle)+right*torch.cos(angle)),dim=-1)
        return rotate(q),rotate(k),v

    def forward_native(self,hidden):
        q,k,v=self.qkv_native(hidden)
        # Independent ungrouped dense softmax reference.
        message=torch.softmax((q@k.transpose(-2,-1))*self.attn.scale,dim=-1)@v
        joined=message.transpose(0,1).reshape(len(hidden),-1)
        h=hidden+self.ls1(self.attn.proj_drop(self.attn.proj(joined)))
        return h+self.ls2(self.mlp(self.norm2(h)))


def toy_rope(tensor,angle):
    left,right=tensor.chunk(2,dim=-1)
    return torch.cat((left*torch.cos(angle)-right*torch.sin(angle),
                      left*torch.sin(angle)+right*torch.cos(angle)),dim=-1)


class ToyEvaBlock(nn.Module):
    def __init__(self,width,prefix):
        super().__init__()
        self.prefix=prefix
        self.norm1,self.norm2=nn.LayerNorm(width),nn.LayerNorm(width)
        self.attn=ToyAttention(width)
        self.attn.qkv=nn.Linear(width,3*width,bias=False)
        self.gamma_1,self.gamma_2=nn.Parameter(torch.full((width,),.3)),nn.Parameter(torch.full((width,),.2))
        self.drop_path1,self.drop_path2=nn.Identity(),nn.Identity()
        self.mlp=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))

    def forward(self,hidden,rope):
        batch,tokens,width=hidden.shape
        q,k,v=self.attn.qkv(self.norm1(hidden)).reshape(batch,tokens,3,2,width//2).permute(2,0,3,1,4)
        q,k=self.attn.q_norm(q),self.attn.k_norm(k)
        q=torch.cat((q[:,:,:self.prefix],toy_rope(q[:,:,self.prefix:],rope)),2)
        k=torch.cat((k[:,:,:self.prefix],toy_rope(k[:,:,self.prefix:],rope)),2)
        message=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        joined=message.transpose(1,2).reshape(batch,tokens,width)
        h=hidden+self.drop_path1(self.gamma_1*self.attn.proj_drop(self.attn.proj(self.attn.norm(joined))))
        return h+self.drop_path2(self.gamma_2*self.mlp(self.norm2(h)))


def main():
    torch.set_num_threads(1);torch.manual_seed(5)
    rng=np.random.RandomState(5)
    dtype=torch.float64
    inside_logits=torch.randn(2,3,4,dtype=dtype)
    outside_logits=torch.randn(2,3,5,dtype=dtype)
    vi,vo=torch.randn(2,4,4,dtype=dtype),torch.randn(2,5,4,dtype=dtype)
    grouped,mass=grouped_message((inside_logits,outside_logits),vi,vo,1.)
    direct=torch.softmax(torch.cat((inside_logits,outside_logits),-1),-1)@torch.cat((vi,vo),1)
    torch.testing.assert_close(grouped,direct,atol=1e-12,rtol=1e-12)
    in_mass=torch.exp(torch.logsumexp(inside_logits,-1)-torch.logsumexp(torch.cat((inside_logits,outside_logits),-1),-1))
    torch.testing.assert_close(in_mass+mass,torch.ones_like(mass),atol=1e-12,rtol=1e-12)
    qs=torch.randn(2,3,4,dtype=dtype);ks=torch.randn(2,9,4,dtype=dtype)
    native_logits=(qs@ks.transpose(-2,-1))*.5
    formal,_=grouped_message((native_logits[:,:,:4],native_logits[:,:,4:]),vi,vo,.75)
    mixed=.75*torch.nn.functional.scaled_dot_product_attention(qs[None],ks[None],torch.cat((vi,vo),1)[None],scale=.5)[0]
    mixed+=.25*torch.nn.functional.scaled_dot_product_attention(qs[None],ks[:,:4][None],vi[None],scale=.5)[0]
    torch.testing.assert_close(formal,mixed,atol=1e-12,rtol=1e-12)
    empty,zero_mass=grouped_message((inside_logits,outside_logits[:,:,:0]),vi,vo[:,:0],.75)
    torch.testing.assert_close(empty,torch.softmax(inside_logits,-1)@vi)
    assert torch.count_nonzero(zero_mass)==0
    # Extremely asymmetric finite logits need no denominator floor.
    extreme,ext_mass=grouped_message((inside_logits+1000,outside_logits-1000),vi,vo,1.)
    torch.testing.assert_close(extreme,torch.softmax(inside_logits,-1)@vi,atol=1e-12,rtol=1e-12)
    assert torch.count_nonzero(ext_mass)==0
    width,prefix,shape=8,2,(4,4)
    blocks=[ToyBlock(width).eval().to(dtype) for _ in range(4)]
    adapters=[StandardPreNormBlockAdapter(block,native_qkv=block.qkv_native,
               native_forward=block.forward_native,
               contract='pre_norm_attention_ls1_residual_pre_norm_mlp_ls2_residual') for block in blocks]
    final_norm=nn.LayerNorm(width).eval().to(dtype)
    h20=torch.randn(prefix+16,width,dtype=dtype)
    patch_ids=torch.arange(prefix,prefix+16)
    cache=build_native_cache(h20,patch_ids,adapters,final_norm,lambda x:x,
                             producer={'kind':'synthetic_dense_reference_only'})
    calls=[b.native_qkv_calls for b in blocks]
    errors=[]
    for region in ([0],[1,2,3,4],list(range(16))):
        _,audit=replay_region(cache,region,1.,audit_native=True)
        errors.append(audit['native_maximum_error'])
    assert [b.native_qkv_calls for b in blocks]==calls # No QK recomputation per branch.
    eva_blocks=[ToyEvaBlock(width,prefix).eval().to(dtype) for _ in range(4)]
    rope=torch.arange(16,dtype=dtype)[None,None,:,None]/9
    eva_adapters=[EvaBlockAdapter(b,rope=rope,apply_rope=toy_rope,num_prefix_tokens=prefix) for b in eva_blocks]
    eva_cache=build_native_cache(h20,patch_ids,eva_adapters,final_norm,lambda x:x,
                                 producer={'kind':'synthetic_Eva_interface_only'})
    _,eva_audit=replay_region(eva_cache,[0,3,4,6],1.,audit_native=True)
    parts,audit=ward_partitions(rng.normal(size=(16,width)),shape,(4,8,16,32))
    assert sum(sum(map(len,p)) for p in parts.values())==64
    from ics.methods.pro_role_prediction import ward_tree,unit as m3_unit
    for sample in (rng.normal(size=(16,width)),np.zeros((16,width))):
        own,_=ward_partitions(sample,shape,(4,8,16,32))
        children,_,minimum,_,_=ward_tree(m3_unit(sample),shape)
        active=set(range(16));snapshots={16:sorted(active,key=lambda i:minimum[i])}
        for offset,(left,right) in enumerate(children):
            active.remove(int(left));active.remove(int(right));active.add(16+offset)
            if len(active) in (4,8):snapshots[len(active)]=sorted(active,key=lambda i:minimum[i])
        def leaves(node):
            stack,out=[node],[]
            while stack:
                current=stack.pop()
                if current<16:out.append(current)
                else:stack.extend(children[current-16])
            return np.array(sorted(out))
        for count,partition in own.items():
            expected=[leaves(v) for v in snapshots[min(count,16)]]
            for a,b in zip(partition,expected):np.testing.assert_array_equal(a,b)
    disconnected=np.zeros(shape,bool);disconnected[::2,::2]=1
    _,disconnected_audit=ward_partitions(rng.normal(size=(16,width)),shape,(2,),disconnected)
    assert disconnected_audit['nonlocal_component_merges']==2
    bounded=descriptor(np.array([1.,0]),np.array([1.,-1]))
    assert bounded['eta']<1
    assert not np.allclose(bounded['extrapolate'],np.array([1.,-4])/np.sqrt(17))
    base=rng.uniform(.1,.9,size=shape).astype(np.float32)
    fused=fuse(base,np.zeros(shape))
    np.testing.assert_array_equal(fused,base)
    for left,right in zip(render(base,(31,47)),render(fused,(31,47))):
        np.testing.assert_array_equal(left,right)
    coverage=np.zeros(shape);coverage[1:3,1:3]=1
    features=cache.projected_native.cpu().numpy()
    result=predict(cache,cache,features,features,coverage,base,(31,47),reference_has_foreground=True)
    assert result['info']['query_processed_area']==64
    assert result['info']['reference_processed_area']==16
    np.testing.assert_array_equal(result['fields']['zero'],base)
    assert not np.array_equal(result['fields']['native_region'],base)
    assert all(v.shape==(31,47) and v.dtype==bool for v in result['original_masks'].values())
    report=dict(kind='CPU_toy_only',actual_DINO_episodes=0,native_grouped_softmax_maxabs=float((grouped-direct).abs().max()),
                native_SDPA_algebra_equals_formal_beta075_maxabs=float((formal-mixed).abs().max()),
                beta1_layer_maxabs=max(errors),group_probability_mass_equal_one=True,empty_outside_exact=True,
                eva_gamma_residual_adapter_toy_maxabs=eva_audit['native_maximum_error'],
                extreme_logits_no_denominator_floor=True,native_QK_frozen_across_branch_states=True,
                query_four_partition_area=64,reference_partition_area=16,disconnected_BG_closest_mean_merges=2,
                ward_partitions_match_current_M3_random_and_ties=True,
                norm_cap_eta=float(bounded['eta']),norm_cap_does_not_claim_beta0=True,
                zero_delta_field_and_both_masks_exact=True,native_region_is_not_zero_control=True,
                all_arms_full_original_bool_masks=True,new_encoder_calls=0,no_GPU=True,no_SSH=True)
    Path(__file__).with_name('toy_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
