"""Specified v2 readout: truth-table identity and complete positive/negative examples."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from ics.methods.mean_rgb_proposal import compose,predict,CONFIG
from ics.experiment import render,sha

torch.set_num_threads(1)
rows=[]
for code in range(8):
    mean=np.full((1024,1024),bool(code&1))
    cut=np.full_like(mean,bool(code&2));unary=np.full_like(mean,bool(code&4))
    final,region,info=compose(mean,cut,unary)
    expected=cut if not np.array_equal(cut,unary) else mean
    assert np.array_equal(final,expected)
    assert np.array_equal(final[~region],mean[~region])
    zero,_,_=compose(mean,unary,unary)
    assert np.array_equal(zero,mean)
    rows.append(dict(bits=code,outside_region_xor=0,zero_graph_mean_xor=0))

q=np.ones((4096,8),np.float16);r=q.copy();cov=np.ones((64,64),np.float32)
rgb=np.full((128,128,3),127,np.uint8)
complete=[]
for sign in (1,-1):
    base=np.full((64,64),.7 if sign>0 else .3,np.float32)
    base[30:34,30:34]=.49 if sign>0 else .51
    output=predict(q,r,cov,base,rgb,original_shape=(75,109))
    direct=render(base)
    assert output['field'].shape==(1024,1024)
    assert np.array_equal(output['mean_control'],direct)
    assert np.array_equal(output['mask_work'],render(output['field']))
    region=output['disagreement']
    assert np.array_equal(output['mask_work'][~region],direct[~region])
    assert np.array_equal(output['mask_work'][region],output['full_cut_control'][region])
    assert output['mask_original'].shape==(75,109)
    info=output['info']['proposal']
    assert info['added_vs_mean_pixels' if sign>0 else 'deleted_vs_mean_pixels']>0
    complete.append(dict(sign=sign,**info,work_shape=list(output['mask_work'].shape),
                         original_shape=list(output['mask_original'].shape),
                         direct_mean_xor=0,renderer_xor=0))
check=dict(state='SPECIFIED_V2_PROPERTIES_AND_COMPLETE_SIGNED_CASES_PASSED',
           recipe=CONFIG,truth_tables=rows,complete_examples=complete,
           module_sha256=sha(ROOT/'src/ics/methods/mean_rgb_proposal.py'),
           inherited_rgb_module_sha256=sha(ROOT/'src/ics/methods/mean_rgb_potts.py'),
           query_gt_opened=False,new_encoder_forwards=0)
(Path(__file__).resolve().parent/'check.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps(check,indent=2))
