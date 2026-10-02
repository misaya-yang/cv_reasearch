#!/usr/bin/env python3
"""CPU native-source operation fixtures, not pretrained-host/GPU validation."""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tics.host_signed_field import native_signed_field


class Host:
    mask_refiner='bilinear';resize_to_orig_size=False
    def _binarize_response(self,score_hw,*,target_hw):
        score=score_hw-score_hw.min();score=score/score.max().clamp_min(1e-6)
        return F.interpolate(score[None,None],target_hw,mode='bilinear',align_corners=False)[0,0]>.5
    def _finalize_mask(self,mask,tgt_image):
        return F.interpolate(mask.reshape(1,1,*mask.shape[-2:]).float(),tgt_image.shape[-2:],mode='bilinear',align_corners=False)[0,0]>.5


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args()
    torch.manual_seed(4200);host=Host();tests=0
    for dtype in [torch.float32,torch.float64]:
        for constant in [False,True]:
            score=torch.ones(4,5,dtype=dtype) if constant else torch.randn(4,5,dtype=dtype)
            native=host._binarize_response(score,target_hw=(19,23))
            with native_signed_field(host,'foris') as c:
                result=host._binarize_response(score,target_hw=(19,23))
            assert torch.equal(native,result) and torch.equal(c.only_field()>0,result) and c.restored
            tests+=1
    for zero in [False,True]:
        mask=torch.zeros(4,5,dtype=torch.bool) if zero else torch.rand(4,5)>.5
        image=torch.randn(1,3,19,23);native=host._finalize_mask(mask,image)
        with native_signed_field(host,'insid3') as c:result=host._finalize_mask(mask,image)
        assert torch.equal(native,result) and torch.equal(c.only_field()>0,result) and c.restored
        tests+=1
    assert '_finalize_mask' not in host.__dict__ and '_binarize_response' not in host.__dict__
    report=dict(state='PASSED',cases=tests,native_mask_exact=True,restored=True,
                scope='CPU source-operation fixtures; real pretrained host parity remains required')
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report))
    print(json.dumps(report))


if __name__=='__main__':main()
