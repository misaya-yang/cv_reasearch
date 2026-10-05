#!/usr/bin/env python3
"""One new complete A/B construction on exposed DEV241.

B excludes reference-NN foreground votes from its background bank. A completes
omitted extent outside RCG; it does not restore pixels removed by B. Preserve
original B, same-bank-size controls and independent addition controls.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import run_directional_operator as prior
from pixel_budget_control import optimize


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def predict(q,r,cov,comparisons,field,nn_margin):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.experiment import unpack
    masks={name:torch.as_tensor(unpack(value).reshape(-1),device='cuda') for name,value in comparisons.items()}
    rcg=masks['rcg'];base=masks['Cbase.control'];seed=masks['conservative.control']
    cover=lambda value:value.reshape(64,16,64,16).float().mean((1,3)).reshape(-1)
    positive=cover(seed)>.5;outside=cover(rcg)==0
    z=torch.as_tensor(field,device='cuda').reshape(-1)
    ids=torch.nonzero(outside)[:,0];ids=ids[torch.argsort(-z[ids],stable=True)]
    filtered=ids[torch.as_tensor(nn_margin,device='cuda')[ids]<0]
    count=min(int(positive.sum()),len(filtered));clean_ids=filtered[:count];same_ids=ids[:count]
    features=torch.as_tensor(q,device='cuda',dtype=torch.float32)
    features=(features/features.norm(dim=1,keepdim=True).clamp_min(1e-12)).double()
    def centroid(bank):
        weights=bank.double();sums=(features*weights[:,None]).sum(0)[None].expand(4096,-1)
        norm=sums.norm(dim=1);valid=(weights.sum()>.5)&(norm>1e-12)
        return ((features*sums).sum(1)/norm.clamp_min(1e-12)).float(),valid
    up=lambda value:F.interpolate(value.reshape(1,1,64,64).float(),(1024,1024),mode='bilinear',align_corners=False)[0,0].reshape(-1)
    p,vp=centroid(positive)
    def deletion(chosen):
        bank=torch.zeros(4096,dtype=torch.bool,device='cuda');bank[chosen]=True
        b,vb=centroid(bank);valid=vp&vb
        margin=up(torch.where(valid,p-b,torch.zeros_like(p)))
        return base&~((up(valid.float())>=1-1e-6)&(margin<0))
    clean=deletion(clean_ids);same=deletion(same_ids)
    p1,p2,info=prior.value_fields(q,r,cov,comparisons,field)
    budget=int(int(rcg.sum())*.05)
    add={'mean':masks['mean.control']&~rcg,'foris':masks['native']&~rcg}
    add['agreement']=add['mean']&add['foris'];add['union']=add['mean']|add['foris']
    output={'bgclean.B.control':clean,'bgclean.same_bank_size.B.control':same}
    zero={'none':torch.zeros_like(rcg)}
    if p1 is None or not budget:
        selected=clean;same_composed=same;old_extent=masks['frozen.delete_p.control'];pixel=clean
    else:
        result,audit,_,_=prior.solve(clean,[p1,p2],add,zero,budget,dual=False);selected=result['joint'];info.update(audit)
        result,_,_,_=prior.solve(same,[p1,p2],add,zero,budget,dual=False);same_composed=result['joint']
        result,_,_,_=prior.solve(masks['frozen.delete_p.control'],[p1,p2],add,zero,budget,dual=False);old_extent=result['joint']
        pixel,certificate=optimize(p1,clean,add['union'],torch.zeros_like(rcg),budget);info['pixel_certificate']=certificate
    output.update({'bgclean.joint':selected,'bgclean.same_bank_size.joint.control':same_composed,
                   'bgclean.old_B_extent.control':old_extent,'bgclean.pixel.control':pixel})
    zup=z.reshape(64,64).repeat_interleave(16,0).repeat_interleave(16,1).reshape(-1)
    chosen=torch.nonzero(add['union'])[:,0];chosen=chosen[torch.argsort(-zup[chosen],stable=True)[:budget]]
    quota=clean.clone();quota[chosen]=True;output['bgclean.extent_quota.control']=quota
    info.update(clean_bank_tokens=count,original_bank_tokens=min(int(positive.sum()),len(ids)),
        reference_NN_definition='cached debiased fg_max-bg_max<0',restore_domain_disabled=True)
    return {name:np.packbits(value.cpu().numpy()) for name,value in output.items()},info


def infer(a):
    import numpy as np
    import torch
    from ics.experiment import sha,load_inputs
    if a.out.exists():raise FileExistsError('Fresh output required')
    parent=json.loads((a.source/'sealed.json').read_text());rows=json.loads((a.source/'manifest.json').read_text())
    if parent['state']!='ALL_PREDICTIONS_SEALED' or len(rows)!=241 or sha(a.source/'manifest.json')!=parent['manifest_sha256']:raise ValueError('Require sealedDEV241')
    providers=json.loads(a.providers.read_text());identity=lambda r:(r['c'],Path(r['support']).name,Path(r['query']).name)
    matches={identity(r):r for r in providers}
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.1)
    a.out.mkdir();(a.out/'predictions').mkdir()
    protocol=dict(n=241,exposure='New construction after inspectedDEV241/1200;development only',
        origin='sealed raw unprojected DINOv3 l24 nearest-reference-token mask',replay_mask_differences={},
        query_gt_in_inference=False,additional_encoder_forwards=0,
        source_seal_sha256=sha(a.source/'sealed.json'),source_code={str(REPO/'scripts'/name):sha(REPO/'scripts'/name) for name in
            ['run_background_clean_dev.py','score_composed_dev.py','run_directional_operator.py','run_calibrated_operator.py','pixel_budget_control.py']},input_receipts={})
    old_protocol=json.loads((a.source/'protocol.json').read_text())
    original_candidate=Path(old_protocol['sources']['candidate']['path'])
    original_receipts=json.loads((original_candidate/'protocol.json').read_text())['input_receipts']
    source_seals={};hashes={};choices={};begin=time.monotonic()
    for n,row in enumerate(rows,1):
        key=row['key'];source=matches[identity(row)];pred=a.source/'predictions'/(key+'.npz')
        if sha(pred)!=parent['predictions'][key]:raise ValueError('Changed source mask')
        with np.load(pred,allow_pickle=False) as z:comparisons={name:z[name].copy() for name in z.files}
        comparisons['composed.original.control']=comparisons['conditional.joint'].copy()
        (q,r,cov,score),receipt=load_inputs(Path('/'),source)
        if receipt['feature_sha256']!=original_receipts[source['key']]['feature_sha256'] or receipt['packet_sha256']!=original_receipts[source['key']]['packet_sha256']:
            raise ValueError('Original frozen input changed')
        field=Path(source['recheck_run'])/'fields'/(source['key']+'.npz')
        run=source['recheck_run']
        if run not in source_seals:source_seals[run]=json.loads((Path(run)/'sealed.json').read_text())
        if sha(field)!=source_seals[run]['fields'][source['key']]:raise ValueError('Changed sealed RCG field')
        with np.load(field,allow_pickle=False) as z:values=z['rcg'].copy()
        with np.load(source['packet_export'],allow_pickle=False) as z:margin=z['fg_max'].reshape(-1).copy()-z['bg_max'].reshape(-1).copy()
        output,info=predict(q,r,cov,comparisons,values,margin)
        destination=a.out/'predictions'/(key+'.npz');np.savez_compressed(destination,**comparisons,**output)
        hashes[key]=sha(destination);choices[key]=info
        protocol['input_receipts'][key]=dict(receipt,field_sha256=sha(field),prediction_sha256=sha(pred))
        if n%25==0:print(json.dumps(dict(n=n,total=241,seconds=round(time.monotonic()-begin,1))),flush=True)
    write(a.out/'manifest.json',rows);write(a.out/'protocol.json',protocol);write(a.out/'choices.json',choices)
    write(a.out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=241,manifest_sha256=sha(a.out/'manifest.json'),
        protocol_sha256=sha(a.out/'protocol.json'),predictions=hashes,choices_sha256=sha(a.out/'choices.json'),
        query_gt_in_inference=False,seconds=time.monotonic()-begin,cuda_peak_bytes=torch.cuda.max_memory_allocated()))


def score(a):
    from score_composed_dev import score as score_existing
    score_existing(a)
    report=json.loads((a.out/'report.json').read_text());report['primary']='bgclean.joint';write(a.out/'report.json',report)
    print(json.dumps(dict(primary='bgclean.joint',scores=report['scores'],comparisons=report['contrasts']['bgclean.joint'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['infer','score']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--providers',type=Path);p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'))
    a=p.parse_args()
    if a.phase=='infer' and (a.source is None or a.providers is None):p.error('Sealed source andexisting providers required')
    {'infer':infer,'score':score}[a.phase](a)


if __name__=='__main__':main()
