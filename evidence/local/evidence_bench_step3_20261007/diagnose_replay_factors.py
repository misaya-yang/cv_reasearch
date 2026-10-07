"""Three-case CPU attribution of stored-score precision and repeated normalization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

CODE=Path('/root/autodl-tmp/evidence_bench_step3_20261007_25142/code')
sys.path.insert(0,str(CODE/'scripts'))
sys.path.insert(0,str(CODE/'src'))
from run_rcg2 import solve
from ics.methods import rcg
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
BASE=Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs')
OUT=Path('/root/autodl-tmp/evidence_bench_step3_20261007_25142/diagnosis_v2')
OUT.mkdir(exist_ok=False)
rows=json.loads((BASE/'claude_order_fresh600/rows.json').read_text())
manifest={r['key']:r for r in json.loads((BASE/'confirm1200_conditional_v1/inputs/manifest.json').read_text())}
receipts=json.loads((CODE.parent/'acceptance1/receipts.json').read_text())
chosen=list(dict.fromkeys([0,max(range(len(receipts)),key=lambda i:receipts[i]['max_abs_error']),next(i for i,r in enumerate(receipts) if r['raw_cut_mask_different_tokens'])]))
with np.load(BASE/'claude_order_fresh600/tokens.npz',allow_pickle=False) as z:
    scores=z['score'].astype(np.float32);stored=z['rcg'].astype(np.float32)
result=dict(scope='numerical attribution only, three fixed diagnostic cases; no new method, parameter fitting or full600 rerun', selection='first case, largest previously observed field drift, first previously observed mask drift',cases=[])
for index in chosen:
    row=manifest[rows[index]['key']]
    cache=torch.load(row['feature_export'],map_location='cpu',weights_only=True)
    q,r=cache['q'].float(),cache['r'].float()
    with np.load(row['packet_export'],allow_pickle=False) as packet:
        cov=packet['cov'].copy();source_score=packet['score'].copy()
    fields={};times={}
    for norm in (False,True):
        for full_precision in (False,True):
            name=('renormalized' if norm else 'retained_qr')+('_FP32_score' if full_precision else '_FP16_score')
            started=time.monotonic()
            fields[name]=solve(F.normalize(q,dim=1) if norm else q,F.normalize(r,dim=1) if norm else r,cov,source_score if full_precision else scores[index].reshape(64,64),16.,'cpu')
            times[name]=time.monotonic()-started
    started=time.monotonic();fields['canonical_API_FP16_score'],info=rcg.predict(q,r,cov,scores[index].reshape(64,64),device='cpu');times['canonical_API_FP16_score']=time.monotonic()-started
    expected=stored[index].reshape(64,64)
    case=dict(key=row['key'],index=index,source_score_FP32_vs_stored_score_max_abs=float(np.abs(source_score-scores[index].reshape(64,64)).max()),statistics={},solver=info)
    for name,field in fields.items():
        case['statistics'][name]=dict(max_abs_vs_saved=float(np.abs(field-expected).max()),mean_abs_vs_saved=float(np.abs(field-expected).mean()),mask_difference=int(np.count_nonzero((field>.5)!=(expected>.5))),FP16_rounded_field_difference=int(np.count_nonzero(field.astype(np.float16)!=expected.astype(np.float16))),seconds=times[name])
    def change(a,b):
        return dict(max_abs=float(np.abs(fields[a]-fields[b]).max()),mean_abs=float(np.abs(fields[a]-fields[b]).mean()),mask_difference=int(np.count_nonzero((fields[a]>.5)!=(fields[b]>.5))))
    case['controlled_changes']=dict(score_precision_at_retained_qr=change('retained_qr_FP16_score','retained_qr_FP32_score'),renormalization_at_FP32_score=change('renormalized_FP32_score','retained_qr_FP32_score'),normalization_at_FP16_score=change('renormalized_FP16_score','retained_qr_FP16_score'),canonical_vs_legacy_same_normalized_inputs=change('canonical_API_FP16_score','renormalized_FP16_score'))
    np.savez_compressed(OUT/(row['key']+'.npz'),**fields)
    result['cases'].append(case)
result['legacy_source_sha256']=hashlib.sha256((CODE/'scripts/run_rcg2.py').read_bytes()).hexdigest()
result['canonical_source_sha256']=hashlib.sha256((CODE/'src/ics/methods/rcg.py').read_bytes()).hexdigest()
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
