#!/usr/bin/env python3
"""One actual source fixture executes all ten methods and matched controls.

This is integration readiness only. No real query annotations, pretrained
encoder, GPU, downloads, scientific score or gain are produced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['HF_HUB_OFFLINE']='1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foris-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Fresh readiness receipt required')
    import torch
    torch.set_num_threads(1)
    from ten_direction_context_cpu import make_fixture
    from ten_direction_experiment import registry,write_json,sha
    methods=registry()
    records=[];start=time.monotonic()
    with torch.inference_mode():
        ctx,encoder,fixture=make_fixture(a.foris_root,seed=6033,image_size=128)
        for item in methods:
            roles=[('naive',item['naive']),('method',item['function'])]
            roles += list(item.get('controls',{}).items())
            for role,fn in roles:
                tick=time.monotonic()
                try:
                    value=fn(ctx)
                    assert isinstance(value,dict) and isinstance(value['mask'],torch.Tensor)
                    assert value['mask'].shape==ctx.work_hw and value['mask'].dtype==torch.bool
                    assert not torch.cuda.is_initialized()
                    record=dict(id=item['id'],role=role,state='PASSED',elapsed_s=time.monotonic()-tick,
                                actual_full_source_native_exact=ctx.native['identity_exact'],
                                encoder_B2_calls=encoder.calls)
                except Exception as exc:
                    record=dict(id=item['id'],role=role,state='ERROR',error_type=type(exc).__name__,
                                error=str(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-tick)
                records.append(record)
                print(json.dumps({k:v for k,v in record.items() if k!='traceback'}),flush=True)
                state='RUNNING_CPU_INTEGRATION'
                write_json(a.out,dict(state=state,records=records,CUDA_initialized=False,fixture=fixture))
    success=all(r['state']=='PASSED' for r in records)
    root=Path(__file__).resolve().parents[1]
    hashes={str(path):sha(path) for path in list((root/'tics').glob('ten_*.py'))+[Path(__file__)]}
    receipt=dict(state='CPU_ALL_TEN_METHODS_PASSED' if success else 'CPU_ALL_TEN_METHODS_ERROR',
        records=records,methods=10,CUDA_initialized=False,elapsed_s=time.monotonic()-start,
        fixture=fixture,source_hashes=hashes,extra_B2_calls=ctx.budget.extra_b2_calls,
        no_query_GT=True,not_a_task_quality_result=True)
    write_json(a.out,receipt)
    return 0 if success else 2


if __name__=='__main__':raise SystemExit(main())
