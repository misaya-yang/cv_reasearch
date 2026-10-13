"""Read-only frozen-DINO last-MLP observations for the existing60 RGB pairs.

No query mask or candidate scoring is accessed. Cache by exact existing input
key, retain all actual FP32 intermediates, and compare native O24 to its older
input-bound payload instead of assuming different batch executions are equal.
"""
from __future__ import annotations
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_n,'2')
import argparse
from pathlib import Path
import json
import importlib.metadata
import shutil
import sys
import time
import numpy as np
import torch
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import opus55_reference_mask_trace as base
from ics.official_data import decoded_rgb,array_hash
from raw_feature_cache import tensor_hash

OUT=base.ASSETS/'a/task_tangent_features60_20261010'


def read(path):return json.loads(Path(path).read_text())


def prepare(out):
    if (out/'config.json').exists():return
    pairs=read(base.OUT/'pairs.json');jobs={}
    for pair in pairs:
        for role in ('reference','query'):
            b=pair['raw_binding'][role];row=pair['row'];key=b['key']
            if key not in jobs:
                jobs[key]=dict(key=key,path=row[role+'_path'],crop=row.get(role+'_crop'),
                    rgb_hash=row[role+'_rgb_hash'],binding=b,first_pair_id=pair['id'],role=role)
    out.mkdir(parents=True,exist_ok=True)
    for folder in ('raw','records','frozen'):(out/folder).mkdir(exist_ok=True)
    shutil.copyfile(__file__,out/'frozen'/Path(__file__).name)
    base.write(out/'pairs.json',pairs);base.write(out/'jobs.json',list(jobs.values()))
    weights=base.ASSETS/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'
    base.write(out/'config.json',dict(n_pairs=len(pairs),n_unique_inputs=len(jobs),
        source_sha256=base.sha(__file__),pairs_sha256=base.sha(out/'pairs.json'),jobs_sha256=base.sha(out/'jobs.json'),
        weights=str(weights),weights_sha256=base.sha(weights),original_profile_sha256=base.sha(base.PROFILE),
        extractor_source_sha256=base.sha(base.REPO/'src/ics/data.py'),
        transform_source_sha256=base.sha(base.ASSETS/'third_party/foris_official/utils/data.py'),
        observations=['O24','mlp_input','mlp_preactivation','mlp_hidden','block_output'],
        model='frozen DINOv3-L/16, actual last block24, prefix5 excluded',
        precision='actual FP32, B1, MPS; no autocast; no encoder gradients or parameter updates',
        purpose='test a frozen-model task-functional prior unavailable from the old O24-only cache',
        scope='same60 existing RGB pairs and same1024 transforms; no new photos or datasets',
        estimated_disk_gib_upper=22,query_GT_reads=0,
        runtime=dict(torch=str(torch.__version__),timm=importlib.metadata.version('timm'))))


def verify(out):
    c=read(out/'config.json')
    assert base.sha(__file__)==c['source_sha256']==base.sha(out/'frozen'/Path(__file__).name)
    assert base.sha(out/'pairs.json')==c['pairs_sha256'] and base.sha(out/'jobs.json')==c['jobs_sha256']
    assert base.sha(base.REPO/'src/ics/data.py')==c['extractor_source_sha256']
    assert base.sha(base.ASSETS/'third_party/foris_official/utils/data.py')==c['transform_source_sha256']
    return c


def run(out,smoke=False):
    prepare(out);cfg=verify(out);jobs=read(out/'jobs.json')
    if (out/'COMPLETE.json').exists():return
    if smoke:jobs=jobs[:2]
    if shutil.disk_usage(out).free<25*2**30:raise RuntimeError('Insufficient space for the declared FP32 branch cache')
    from ics.data import TimmDINOv3
    sys.path.insert(0,str(base.ASSETS/'third_party/foris_official'))
    from utils.data import build_transform
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.mps.set_per_process_memory_fraction(20e9/torch.mps.recommended_max_memory())
    before=time.monotonic();encoder=TimmDINOv3(Path(cfg['weights']).parent).eval().requires_grad_(False).to('mps')
    model=encoder.m;block=model.blocks[-1];transform=build_transform(1024)
    assert len(model.blocks)==24 and model.num_prefix_tokens==5 and isinstance(model.norm,torch.nn.LayerNorm)
    assert isinstance(block.mlp.act,torch.nn.GELU) and block.mlp.act.approximate=='none'
    assert isinstance(block.mlp.norm,torch.nn.Identity) and block.mlp.drop1.p==block.mlp.drop2.p==0
    assert all(not p.requires_grad for p in encoder.parameters())
    params=dict(fc2_weight=block.mlp.fc2.weight.detach().cpu().numpy(),gamma2=block.gamma_2.detach().cpu().numpy(),
        norm_weight=model.norm.weight.detach().cpu().numpy(),norm_bias=model.norm.bias.detach().cpu().numpy(),norm_eps=np.asarray(model.norm.eps))
    pp=out/'parameters.npz'
    if pp.exists():
        with np.load(pp) as z:
            for k,v in params.items():assert np.array_equal(z[k],v)
    else:np.savez(pp,**params)
    initialized=time.monotonic()-before;records=[];start=time.monotonic()
    for index,job in enumerate(jobs):
        rp=out/'records'/(job['key']+'.json')
        if rp.exists():
            rec=read(rp);assert base.sha(out/'raw'/rec['file'])==rec['sha256'];records.append(rec);continue
        begin=time.monotonic();captured={};handles=[];memory=[]
        def save(name):
            def hook(module,args,output):
                if name=='block_output':captured['_full_block_output']=output.detach().float().cpu().contiguous()
                captured[name]=output[:,model.num_prefix_tokens:].detach().float().cpu().contiguous()
                memory.append(torch.mps.driver_allocated_memory())
            return hook
        handles=[block.norm2.register_forward_hook(save('mlp_input')),block.mlp.fc1.register_forward_hook(save('mlp_preactivation')),
                 block.mlp.act.register_forward_hook(save('mlp_hidden')),block.register_forward_hook(save('block_output'))]
        attempts=[]
        try:
            with base.deny_labels(base.OUT,attempts),torch.inference_mode():
                image=decoded_rgb(base.ASSETS/job['path'],job['crop']);assert array_hash(np.asarray(image))==job['rgb_hash']
                x=transform(image)[None];assert tensor_hash(x[0].numpy())==job['binding']['input_tensor_hash']
                t=time.monotonic();output=encoder.get_intermediate_layers(x.to('mps'),n=1,reshape=True)[0]
                torch.mps.synchronize();forward_seconds=time.monotonic()-t
                captured['O24']=output.permute(0,2,3,1).reshape(1,4096,1024).float().cpu().contiguous()
                # The actual final normalization must match captured Z.
                replay=model.norm(captured.pop('_full_block_output').to('mps'))[:,model.num_prefix_tokens:].cpu()
                assert torch.equal(replay,captured['O24']), 'Native final-normalization identity failed'
                assert not attempts and all(not p.requires_grad for p in encoder.parameters())
            values={k:v[0].numpy() for k,v in captured.items()}
            assert set(values)==set(cfg['observations'])
            assert all(np.isfinite(a).all() and a.dtype==np.float32 for a in values.values())
            assert base.sha(job['binding']['entry'])==job['binding']['entry_sha256']
            entry=read(job['binding']['entry']);oldpath=Path(job['binding']['entry']).parent/entry['file']
            assert base.sha(oldpath)==entry['file_sha256']
            with np.load(oldpath) as z:old=z['O/24']
            difference=float(np.max(np.abs(values['O24']-old)));old_equal=np.array_equal(values['O24'],old)
            file=job['key']+'.npz';np.savez(out/'raw'/file,**values)
            rec=dict(index=index,key=job['key'],file=file,sha256=base.sha(out/'raw'/file),
                tensors={k:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=tensor_hash(a)) for k,a in values.items()},
                input_tensor_sha256=job['binding']['input_tensor_hash'],old_O24_bit_exact=old_equal,old_O24_max_error=difference,
                forward_seconds=forward_seconds,seconds=time.monotonic()-begin,
                sampled_mps_driver_peak=max(memory),query_label_access_attempts=attempts,encoder_parameter_updates=0,
                bytes=(out/'raw'/file).stat().st_size)
            assert max(memory)<20e9
            base.write(rp,rec);records.append(rec)
            base.write(out/'activity.json',dict(state='ENCODING',pid=os.getpid(),completed=len(records),total=len(jobs)))
            print(json.dumps(dict(completed=len(records),total=len(jobs),forward_seconds=forward_seconds,old_O24_bit_exact=old_equal,driver_gib=max(memory)/2**30)),flush=True)
        finally:
            for h in handles:h.remove()
        del output,captured,values;torch.mps.empty_cache()
    if smoke:
        base.write(out/'SMOKE.json',dict(records=records,parameters_sha256=base.sha(pp),initialization_seconds=initialized))
    else:
        base.write(out/'COMPLETE.json',dict(state='FROZEN_LAST_MLP_FEATURES_COMPLETE',n_unique_inputs=len(records),records=records,
            config_sha256=base.sha(out/'config.json'),parameters_sha256=base.sha(pp),initialization_seconds=initialized,
            seconds=time.monotonic()-start,query_GT_reads=0,encoder_updates=0))
        base.write(out/'activity.json',dict(state='COMPLETE',n_unique_inputs=len(records)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','smoke','run']);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if a.mode=='prepare':prepare(a.out)
    else:run(a.out,a.mode=='smoke')


if __name__=='__main__':main()
