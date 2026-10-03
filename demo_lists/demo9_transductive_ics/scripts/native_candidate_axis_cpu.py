#!/usr/bin/env python3
"""Ten exact extracted-source axis controls, CPU only; no task benefit claim."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tics.native_candidate_axis import candidate_normalization_axis


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Fresh result, preserve previous checks')
    torch.set_num_threads(1)
    source=a.source.read_text();tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FoRIS')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_locate_candidates')
    segment=ast.get_source_segment(source,method)
    def downsample_mask(x,h,w):return F.interpolate(x.float(),(h,w),mode='nearest').squeeze(1)>0.5
    scope=dict(torch=torch,F=F,math=math,downsample_mask=downsample_mask)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(a.source),'exec'),scope)
    fn=scope['_locate_candidates'];fn._fixture_source=segment
    class Fixture:_locate_candidates=fn
    records=[]
    for i in range(10):
        torch.manual_seed(2111+i);n=1+i%2;h,w,d=3,4,8
        refs=F.normalize(torch.randn(1,n,d,h,w),dim=2);query=F.normalize(torch.randn(1,d,h,w),dim=1)
        masks=torch.zeros(n,1,h,w,dtype=torch.bool);masks[:,:,0,:]=True
        args=dict(ref_feats=refs,tgt_feat=query,ref_masks=masks,ref_prototype=torch.ones(d),n_refs=n,h=h,w=w)
        original_refs=refs.clone();host=Fixture();native=host._locate_candidates(**args)
        with candidate_normalization_axis(host,2):replica=host._locate_candidates(**args)
        assert all(torch.equal(x,y) for x,y in zip(native,replica))
        with candidate_normalization_axis(host,1):channel=host._locate_candidates(**args)
        votes=torch.zeros(h,w,dtype=torch.int32)
        for j in range(n):
            r=F.normalize(refs[0,j].flatten(1).T,dim=1);q=F.normalize(query[0].flatten(1).T,dim=1)
            ids=(q@r.T).argmax(dim=1);votes+=masks[j,0].flatten()[ids].reshape(h,w).int()
        assert torch.equal(channel[0],votes>=math.ceil(n/2))
        assert torch.equal(channel[1],votes.to(query.dtype)/n)
        assert torch.equal(refs,original_refs) and '_locate_candidates' not in host.__dict__
        try:
            with candidate_normalization_axis(host,1):raise RuntimeError('fixture')
        except RuntimeError:pass
        assert '_locate_candidates' not in host.__dict__
        records.append(dict(case=i,source_replica_exact=True,channel_matches_dense_cosine=True,changed_candidate_pixels=int((native[0]!=channel[0]).sum()),restored_on_success_and_error=True))
    assert any(x['changed_candidate_pixels'] for x in records)
    paths=[Path(__file__),a.source,Path(__file__).resolve().parents[1]/'tics/native_candidate_axis.py']
    result=dict(state='CPU_NATIVE_CANDIDATE_AXIS_PASSED',cases=records,source_hashes={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},CUDA_initialized=torch.cuda.is_initialized(),fixture='exact extracted source method, synthetic features, downsample helper scoped to nearest grids',real_segmentation_gain_measured=False)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(state=result['state'],cases=len(records),changed_candidate_pixels=sum(x['changed_candidate_pixels'] for x in records),CUDA_initialized=result['CUDA_initialized'])))


if __name__=='__main__':main()
