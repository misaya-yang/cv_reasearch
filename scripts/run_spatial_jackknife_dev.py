#!/usr/bin/env python3
"""Spatial-block jackknife of the original query-background deletion evidence.

Preserve originalBpooling;delete only when every leave-block-out pool agrees.
Keep fixed existing extent additions;compare original-margin same-count/slack.
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
    ids=torch.nonzero(outside)[:,0];ids=ids[torch.argsort(-z[ids],stable=True)[:min(int(positive.sum()),len(ids))]]
    features=torch.as_tensor(q,device='cuda',dtype=torch.float32)
    features=(features/features.norm(dim=1,keepdim=True).clamp_min(1e-12)).double()
    up=lambda value:F.interpolate(value.reshape(1,1,64,64).float(),(1024,1024),mode='bilinear',align_corners=False)[0,0].reshape(-1)
    pmean=features[positive].sum(0);pnorm=pmean.norm()
    full_bg=features[ids].sum(0);bnorm=full_bg.norm()
    valid=bool(positive.any()) and len(ids)>0 and float(pnorm)>1e-12 and float(bnorm)>1e-12
    if valid:
        fg=(features@pmean/pnorm).float();full=(features@full_bg/bnorm).float();margin=up(fg-full)
        minimum=full.double().clone();max_shift=torch.zeros(4096,dtype=torch.float64,device='cuda')
        blocks=(ids//64//4)*16+(ids%64//4)
        occupied=torch.unique(blocks)
        all_valid=True
        for block in occupied:
            drop=blocks==block;remain=full_bg-features[ids[drop]].sum(0);norm=remain.norm()
            if int(drop.sum())==len(ids) or float(norm)<=1e-12:all_valid=False;break
            score=features@remain/norm;minimum=torch.minimum(minimum,score)
            max_shift=torch.maximum(max_shift,(score-full).abs())
        gone=base&(margin<0)
        robust_gone=gone&(up(fg-minimum.float())<0) if all_valid else torch.zeros_like(base)
        restored=int((gone&~robust_gone).sum())
        # Same number of saved pixels, ranked by original deletion margin only.
        simple_gone=gone.clone();locations=torch.nonzero(gone)[:,0]
        chosen=locations[torch.argsort(margin[locations],descending=True,stable=True)[:restored]]
        simple_gone[chosen]=False
        slack_gone=base&(margin<-.02)
        old=base&~gone
        if not torch.equal(old,masks['frozen.delete_p.control']):raise ValueError('Original frozenBmask replay changed')
        robust=base&~robust_gone;matched=base&~simple_gone;slack=base&~slack_gone
        info=dict(block_tokens=4,occupied_blocks=len(occupied),bank_tokens=len(ids),saved_pixels=restored,
                  original_removed=int(gone.sum()),robust_removed=int(robust_gone.sum()),all_jackknifes_valid=all_valid,
                  mean_max_BG_cosine_shift=float(max_shift.mean()),query_gt_in_inference=False)
    else:
        robust=matched=slack=base.clone();info=dict(block_tokens=4,occupied_blocks=0,bank_tokens=len(ids),saved_pixels=0,query_gt_in_inference=False)
    extent=masks['conditional.joint']&~rcg
    output={'jackknife.joint':robust|extent,'jackknife.B.control':robust,
            'jackknife.original_margin_same_count.control':matched|extent,
            'jackknife.fixed_slack.control':slack|extent,
            'jackknife.original_B_same_extent.control':masks['frozen.delete_p.control']|extent}
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
            ['run_spatial_jackknife_dev.py','score_composed_dev.py','run_directional_operator.py','run_calibrated_operator.py','pixel_budget_control.py']},input_receipts={})
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
    report=json.loads((a.out/'report.json').read_text());report['primary']='jackknife.joint';write(a.out/'report.json',report)
    print(json.dumps(dict(primary='jackknife.joint',scores=report['scores'],comparisons=report['contrasts']['jackknife.joint'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['infer','score']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--providers',type=Path);p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'))
    a=p.parse_args()
    if a.phase=='infer' and (a.source is None or a.providers is None):p.error('Sealed source andexisting providers required')
    {'infer':infer,'score':score}[a.phase](a)


if __name__=='__main__':main()
