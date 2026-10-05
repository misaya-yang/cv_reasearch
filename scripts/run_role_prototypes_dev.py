#!/usr/bin/env python3
"""Complete easy/hard query-background prototypes with a reference-FG guard.

A can recover targets outside prior mask unions using two positive margins.
These are cosine confidence rules, not calibrated membership probabilities.
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
    margin_ref=torch.as_tensor(nn_margin,device='cuda').reshape(-1)
    easy=ids[margin_ref[ids]<0];hard=ids[margin_ref[ids]>=0]
    features=F.normalize(torch.as_tensor(q,device='cuda',dtype=torch.float32),dim=1).double()
    refs=F.normalize(torch.as_tensor(r,device='cuda',dtype=torch.float32),dim=1).double()
    def query_score(selected):
        if len(selected)==0:return torch.full((4096,),-torch.inf,device='cuda'),False
        mean=features[selected].sum(0);norm=mean.norm()
        if float(norm)<=1e-12:return torch.full((4096,),-torch.inf,device='cuda'),False
        return ((features*mean).sum(1)/norm).float(),True
    pos,valid_pos=query_score(torch.nonzero(positive)[:,0])
    easy_score,valid_easy=query_score(easy);hard_score,valid_hard=query_score(hard);pooled,_=query_score(ids)
    weights=torch.as_tensor(cov,device='cuda').reshape(-1).double();refmean=(refs*weights[:,None]).sum(0);refnorm=refmean.norm()
    reference=((features*refmean).sum(1)/refnorm.clamp_min(1e-12)).float()
    foreground=torch.maximum(pos,reference) if float(refnorm)>1e-12 else pos
    background=torch.maximum(easy_score,hard_score)
    up=lambda v:F.interpolate(v.reshape(1,1,64,64).float(),(1024,1024),mode='bilinear',align_corners=False)[0,0].reshape(-1)
    valid=valid_pos and (valid_easy or valid_hard)
    gap=foreground-background if valid else torch.zeros_like(z)
    role_B=base&~(up(gap)<0) if valid else base.clone()
    split_B=base&~(up(pos-background)<0) if valid else base.clone()
    guard_B=base&~(up(foreground-pooled)<0) if valid and len(ids) else base.clone()
    budget=int(int(rcg.sum())*.05)
    candidate=~rcg & (up(gap)>.1) & (up(margin_ref)>0) if valid else torch.zeros_like(rcg)
    priority=torch.minimum(up(gap)-.1,up(margin_ref))
    chosen=torch.nonzero(candidate)[:,0];chosen=chosen[torch.argsort(priority[chosen],descending=True,stable=True)[:budget]]
    addition=torch.zeros_like(rcg);addition[chosen]=True
    current_A=masks['conditional.joint']&~rcg
    output={'roleproto.joint':role_B|addition,'roleproto.B.control':role_B,
            'roleproto.query_only_split_B.control':split_B,'roleproto.unsplit_guard_B.control':guard_B,
            'roleproto.old_B_new_A.control':masks['frozen.delete_p.control']|addition,
            'roleproto.new_B_old_A.control':role_B|current_A}
    info=dict(easy_tokens=len(easy),hard_tokens=len(hard),foreground_query_tokens=int(positive.sum()),
              added_pixels=int(addition.sum()),budget=budget,whole_query_addition_domain=True,
              reference_guard='max query-seed/reference foreground means',negative_rule='max easy/hard BG prototypes',
              addition_rule='prototype margin>.1 and referenceNN margin>0; rank minimum margin;fixed5percentRCGbudget',
              probabilities_claimed=False,query_gt_in_inference=False)
    return {name:np.packbits(mask.cpu().numpy()) for name,mask in output.items()},info


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
            ['run_role_prototypes_dev.py','score_composed_dev.py','run_directional_operator.py','run_calibrated_operator.py','pixel_budget_control.py']},input_receipts={})
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
    report=json.loads((a.out/'report.json').read_text());report['primary']='roleproto.joint';write(a.out/'report.json',report)
    print(json.dumps(dict(primary='roleproto.joint',scores=report['scores'],comparisons=report['contrasts']['roleproto.joint'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['infer','score']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--providers',type=Path);p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'))
    a=p.parse_args()
    if a.phase=='infer' and (a.source is None or a.providers is None):p.error('Sealed source andexisting providers required')
    {'infer':infer,'score':score}[a.phase](a)


if __name__=='__main__':main()
