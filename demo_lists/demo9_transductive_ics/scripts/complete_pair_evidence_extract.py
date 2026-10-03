"""Finite identity-evidence capture; no new decoder, QKV recompute or NxN matrix.

Two original SDPA calls are observed AFTER native Q/K norm and RoPE. Every
original function returns its original tensor. Native dtype is saved losslessly.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def source_debias_decision(host):
    names=('_should_apply_positional_debias','_part1_positional_debias')
    saved={n:(n in host.__dict__,host.__dict__.get(n),getattr(host,n)) for n in names}
    original=saved[names[0]][2];part1=saved[names[1]][2]
    trace=dict(decision=[],pre_debias=None,input_output_same_tensor=None)
    def tap(*args,**kwargs):
        result=original(*args,**kwargs);trace['decision'].append(bool(result));return result
    def input_tap(*args,**kwargs):
        incoming=args[0] if args else kwargs['fmaps_norm']
        trace['pre_debias']=incoming.detach().clone()
        result=part1(*args,**kwargs);trace['input_output_same_tensor']=result is incoming
        return result
    try:
        setattr(host,names[0],tap);setattr(host,names[1],input_tap);yield trace
    finally:
        for name,(had,old,_) in saved.items():
            setattr(host,name,old) if had else delattr(host,name)


@contextmanager
def original_evidence(backbone, indices=(11,23)):
    """No replacement of attn.forward; hooks scope the ACTUAL native SDPA call."""
    indices=tuple(indices)
    if len(backbone.blocks)<=max(indices):raise RuntimeError('required native blocks absent')
    old=F.scaled_dot_product_attention;active=[];handles=[]
    trace=dict(tokens={},qkv={},sdpa_metadata={},calls={},restored=False)
    def enter(index):
        def hook(module,args):active.append(index)
        return hook
    def leave(module,args,result):
        if not active:raise RuntimeError('attention scope imbalance')
        active.pop()
    def output(index):
        def hook(module,args,result):
            if not isinstance(result,torch.Tensor) or result.ndim!=3:raise RuntimeError('native full-sequence block output contract')
            if index in trace['tokens']:raise RuntimeError('selected native block ran twice')
            trace['tokens'][index]=result.detach().clone()
        return hook
    def sdpa(q,k,v,*args,**kwargs):
        if active:
            index=active[-1];trace['calls'][index]=trace['calls'].get(index,0)+1
            if trace['calls'][index]!=1:raise RuntimeError('selected scope has multiple SDPA calls')
            if q.ndim!=4 or q.shape!=k.shape or k.shape!=v.shape:raise RuntimeError('native full Q/K/V shape contract')
            trace['qkv'][index]=dict(q=q.detach().clone(),k=k.detach().clone(),v=v.detach().clone())
            trace['sdpa_metadata'][index]=dict(shape=list(q.shape),native_dtype=str(q.dtype),
                scale=kwargs.get('scale'),dropout_p=kwargs.get('dropout_p',0.),
                is_causal=bool(kwargs.get('is_causal',False)),
                mask_shape=None if kwargs.get('attn_mask') is None else list(kwargs['attn_mask'].shape),
                stage='actual SDPA inputs after native Q/K normalization and RoPE; no side recomputation')
        return old(q,k,v,*args,**kwargs)
    try:
        for index in indices:
            block=backbone.blocks[index]
            handles.append(block.attn.register_forward_pre_hook(enter(index)))
            handles.append(block.attn.register_forward_hook(leave,always_call=True))
            handles.append(block.register_forward_hook(output(index)))
        F.scaled_dot_product_attention=sdpa
        yield trace
        if active or set(trace['tokens'])!=set(indices) or set(trace['qkv'])!=set(indices):
            raise RuntimeError('native capture incomplete or fused SDPA unavailable; no fallback')
    finally:
        F.scaled_dot_product_attention=old
        for handle in handles:handle.remove()
        trace['restored']=True


def quantize(name,tensor,storage,audit):
    cpu=tensor.detach().cpu().contiguous()
    saved=cpu.clone() if storage=='native' else cpu.to(torch.float16).contiguous()
    if not torch.isfinite(saved).all():raise RuntimeError('storage overflow/nonfinite '+name)
    audit[name]=dict(shape=list(cpu.shape),native_dtype=str(cpu.dtype),storage_dtype=str(saved.dtype),
        explicitly_rounded=cpu.dtype!=saved.dtype,
        max_rounding_abs=float((cpu.double()-saved.double()).abs().max()) if cpu.numel() else 0.)
    return saved


def cpu_check(out):
    from functools import partial
    from timm.models.eva import EvaBlock
    from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU acceptance initialized CUDA')
    class Tiny(torch.nn.Module):
        def __init__(self,seed):
            super().__init__()
            self.blocks=torch.nn.ModuleList([EvaBlock(dim=32,num_heads=4,mlp_ratio=1.5,
                num_prefix_tokens=5,attn_type='eva',rotate_half=True,qkv_bias=bool(seed%2),
                norm_layer=partial(torch.nn.LayerNorm,eps=1e-5),attn_drop=0.,proj_drop=0.,drop_path=0.) for _ in range(4)])
            self.double().eval()
            for block in self.blocks:block.attn.fused_attn=True
        def forward(self,x,rope):
            for block in self.blocks:x=block(x,rope=rope)
            return x
    cases=[];old=F.scaled_dot_product_attention
    with torch.inference_mode():
        for seed in range(10):
            torch.manual_seed(seed);m=Tiny(seed);x=torch.randn(2,21,32,dtype=torch.float64)
            rope=RotaryEmbeddingDinoV3(dim=8,rotate_half=True).get_embed(shape=(4,4))
            plain=m(x,rope)
            with original_evidence(m,(1,3)) as trace:observed=m(x,rope)
            assert torch.equal(plain,observed) and F.scaled_dot_product_attention is old
            assert trace['calls']=={1:1,3:1} and trace['restored']
            assert all(v.shape==(2,4,21,8) for fields in trace['qkv'].values() for v in fields.values())
            assert all(v.shape==(2,21,32) for v in trace['tokens'].values())
            assert not any(t.ndim==4 and t.shape[-2:]==(21,21) for fields in trace['qkv'].values() for t in fields.values())
            audit={};quantize('fixture',trace['tokens'][1],'native',audit)
            assert audit['fixture']['max_rounding_abs']==0
            class SourcePrimitive:
                def _should_apply_positional_debias(self,z):return bool(seed%2)
                def _part1_positional_debias(self,z):return z*2 if self._should_apply_positional_debias(z) else z
            source=SourcePrimitive();before=source._part1_positional_debias(x)
            with source_debias_decision(source) as st:after=source._part1_positional_debias(x)
            assert torch.equal(before,after) and torch.equal(st['pre_debias'],x)
            assert st['input_output_same_tensor']==(not bool(seed%2)) and st['decision']==[bool(seed%2)]
            cases.append(dict(seed=seed,native_output_bitexact=True,one_original_SDPA_per_selected_scope=True,
                              full_prefix_and_patch_preserved=True,source_input_tap_primitive_exact=True,rounding=audit['fixture']))
    report=dict(state='PASSED',cases=cases,cases_count=10,CUDA_initialized=torch.cuda.is_initialized(),
                no_pretrained_model_loaded=True,no_NxN_attention=True,no_QKV_recompute=True,
                real_DINO_GPU_hook_and_mask_gate='UNTESTED_RUNTIME_GATE')
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=2));print(json.dumps(report))


def run(a):
    from PIL import Image
    sys.path.insert(0,a.public_scripts);import extent_experiment as public
    man=json.loads(Path(a.manifest).read_text());rows=man['episodes']
    if len(rows)!=12 or any(sum(r['fold']==f for r in rows)!=3 for f in range(4)):raise RuntimeError('exact12/3perfold contract')
    if os.environ.get('DEMO9_CUDA_GUARD')!='1':raise RuntimeError('root-owned finite guard required')
    root=Path(a.out)
    if root.exists() and any(root.iterdir()):raise RuntimeError('fresh own output required')
    root.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.4,device=0)
    # No model is instantiated by --cpu-check. Real loading is only root's run.
    from types import SimpleNamespace
    host=public.build_host(SimpleNamespace(foris_root=None,demo4_root=a.demo4_root,fixture=False),man,'cuda')
    backbone=host.encoder.m;prefix=int(backbone.num_prefix_tokens)
    report=dict(state='RUNNING',pairs=0,storage=a.storage,storage_policy='native dtype lossless; no precision change',gpu_memory_fraction_cap=.4,
        cache_max_bytes=int(4.5*1024**3),reserve_bytes=5*1024**3,feature_bytes=0,no_queryGT_in_features=True,
        no_attention_matrix=True,no_extra_model_or_decoder=True,premise='IDENTITY_INFORMATION_AUDIT_UNTESTED')
    def save(): (root/'report.json').write_text(json.dumps(report,indent=2))
    save();start=time.monotonic();label_jobs=[]
    try:
        with torch.inference_mode():
            for index,row in enumerate(rows):
                if time.monotonic()-start>600:raise RuntimeError('finite600s exceeded')
                if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('5GiB disk reserve violated')
                sp=Image.open(Path(man['data_root'])/row['support']).convert('RGB');qp=Image.open(Path(man['data_root'])/row['query']).convert('RGB')
                ref_label=np.asarray(Image.open(Path(man['annotation_root'])/Path(row['support']).with_suffix('.png')))==row['c']+1
                ref=torch.from_numpy(ref_label.copy())
                torch.manual_seed(0);np.random.seed(0)
                plain=public.run_foris(host,sp,ref,qp)[0] if index<2 else None
                torch.manual_seed(0);np.random.seed(0)
                with original_evidence(backbone) as captured,source_debias_decision(host) as source_decision:
                    native,got,ref_model,tgt=public.run_foris(host,sp,ref,qp)
                if len(source_decision['decision'])!=1 or source_decision['pre_debias'] is None:raise RuntimeError('actual source Part1 input/decision not observed exactly once')
                if plain is not None and not torch.equal(native,plain):raise RuntimeError('unobserved/observed native mask mismatch')
                previous=Path(man['matched_packet_root'])/('%d_%d_%d.npz'%(row['fold'],row['e'],row['c']))
                # NPZ is lazy: only the old native mask is accessed, never truth.
                with np.load(previous) as packet:
                    old_mask=np.unpackbits(packet['native']).reshape(native.shape).astype(bool)
                if not np.array_equal(native.cpu().numpy(),old_mask):raise RuntimeError('old matched native mask mismatch')
                raw=got['raw'];actual=got['deb'];audit={}
                public_row={k:row[k] for k in ('fold','e','c','support','query')}
                package=dict(metadata=dict(row=public_row,prefix_tokens=prefix,grid=list(raw.shape[-2:]),
                    feature_roles={'raw_final':'actual source _extract_features return',
                      'actual_debiased':'actual source Part1 returned normalized/projected maps',
                      'pre_debias':'actual source Part1 input, no CPU reconstruction',
                      'block12':'native full block output after residual/MLP, before final backbone norm',
                      'block24':'native full block output after residual/MLP, before final backbone norm'},
                    actual_source_SAFR_applied=source_decision['decision'][0],
                    Part1_input_output_same_tensor=source_decision['input_output_same_tensor'],
                    derived_middle_normalization='saved native finalNorm parameters may be applied on CPU, explicitly DERIVED representation, not native block12 output',
                    pre_RoPE_QK_saved=False,pre_RoPE_reason='optional fields omitted to retain complete required native evidence within finite4.5GiB budget',
                    sdpa=captured['sdpa_metadata'],native_unobserved_exact=None if plain is None else torch.equal(native,plain),matched_old_mask_exact=True),features={},parameters={})
                norm=backbone.norm
                for name in ('weight','bias'):
                    tensor=getattr(norm,name,None)
                    package['parameters']['final_norm_'+name]=None if tensor is None else tensor.detach().cpu().clone()
                package['metadata']['final_norm_spec']=dict(type=type(norm).__module__+'.'+type(norm).__qualname__,eps=getattr(norm,'eps',None))
                for name,tensor in [('raw_final',raw),('pre_debias',source_decision['pre_debias']),('actual_debiased',actual),('block12',captured['tokens'][11]),('block24',captured['tokens'][23]),('original_score',got['score']),('reference_native_mask',ref_model)]:
                    package['features'][name]=quantize(name,tensor,a.storage,audit)
                for layer in (11,23):
                    for name,tensor in captured['qkv'][layer].items():
                        key='block%d_%s_postRoPE'%(layer+1,name);package['features'][key]=quantize(key,tensor,a.storage,audit)
                package['metadata']['rounding']=audit
                estimate=sum(v.numel()*v.element_size() for v in package['features'].values())+1024**2
                if report['feature_bytes']+estimate>int(4.5*1024**3):raise RuntimeError('4.5GiB exact cache budget would be exceeded')
                dest=root/('pair_%02d.pt'%index);torch.save(package,dest)
                with dest.open('rb') as f:os.fsync(f.fileno())
                report['feature_bytes']+=dest.stat().st_size
                if report['feature_bytes']>int(4.5*1024**3):raise RuntimeError('serialized cache budget exceeded')
                # No query-label access until ALL12 evidence packages are committed.
                label_jobs.append((index,public_row,ref_label.copy(),np.packbits(native.cpu().numpy()),tuple(native.shape)))
                item=dict(index=index,fold=row['fold'],e=row['e'],c=row['c'],features=str(dest),feature_bytes=dest.stat().st_size,
                          source_native_exact=True,first2_unobserved_comparison=index<2,prefix_tokens=prefix,
                          captured_native_QKV_after_RoPE=True,all_patch_tokens=True,rounding=audit)
                with (root/'pairs.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
                report['pairs']=index+1;save();print(json.dumps(item),flush=True)
                del package,captured,source_decision,got,raw,actual,native,tgt,ref_model,plain
        report.update(state='FEATURES_FROZEN',all12_features_fsynced=True);save()
        for index,row,ref_label,native_bits,model_shape in label_jobs:
            query_label=np.asarray(Image.open(Path(man['annotation_root'])/Path(row['query']).with_suffix('.png')))==row['c']+1
            np.savez_compressed(root/('labels_%02d.npz'%index),reference=ref_label,query=query_label,
                                native=native_bits,model_shape=np.asarray(model_shape))
        report['query_labels_opened_after_all_feature_commits']=True
        report.update(state='COMPLETED',seconds=time.monotonic()-start);save()
    except BaseException as error:
        report.update(state='ERROR',error=repr(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-check');p.add_argument('--manifest');p.add_argument('--out')
    p.add_argument('--storage',choices=['native'],default='native')
    p.add_argument('--public-scripts',default='/root/autodl-tmp/demo9_extent/scripts');p.add_argument('--demo4-root',default='/root/autodl-tmp/demo4')
    args=p.parse_args()
    if args.cpu_check:cpu_check(args.cpu_check)
    elif not args.manifest or not args.out:p.error('--manifest/--out required')
    else:run(args)
