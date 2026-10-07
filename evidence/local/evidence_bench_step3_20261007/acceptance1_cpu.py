"""CPU-only PLAN step3 acceptance1; no decomposition or parameter changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np

BASE = Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs')
OUT = Path('/root/autodl-tmp/evidence_bench_step3_20261007_25142/acceptance1')
DATA = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def init():
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with np.load(BASE/'claude_order_fresh600/tokens.npz', allow_pickle=False) as source:
        DATA.update(score=source['score'].astype(np.float32), cov=source['ref'].astype(np.float32), stored=source['rcg'].astype(np.float32))

def replay(task):
    import torch
    from ics.methods import rcg
    index, row = task
    begin = time.monotonic()
    path = Path(row['feature_export'])
    cache = torch.load(path, map_location='cpu', weights_only=True)
    field, solver = rcg.predict(cache['q'], cache['r'], DATA['cov'][index].reshape(64,64), DATA['score'][index].reshape(64,64), device='cpu')
    stored = DATA['stored'][index].reshape(64,64)
    delta = np.abs(field-stored)
    half_ulp = np.abs(np.spacing(stored.astype(np.float16)).astype(np.float32))*.5
    receipt = dict(key=row['key'], feature_path=str(path), feature_sha256=sha(path),
        source_cache_precision=str(cache['q'].dtype), debiased=bool(cache.get('debiased', False)),
        wall_seconds=time.monotonic()-begin, solver=solver,
        max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
        storage_bound_exceeded_tokens=int((delta>half_ulp+1e-6).sum()),
        raw_cut_mask_different_tokens=int(((field>.5)!=(stored>.5)).sum()),
        FP16_cut_mask_different_tokens=int(((field.astype(np.float16)>.5)!=(stored>.5)).sum()))
    return index, field, receipt

def metric(counts, classes):
    return float(np.mean([counts[classes==c,0].sum()/max(counts[classes==c,1].sum(),1) for c in np.unique(classes)])*100)

if __name__ == '__main__':
    from ics.methods import rcg
    started = time.monotonic()
    rows = json.loads((BASE/'claude_order_fresh600/rows.json').read_text())
    manifest = json.loads((BASE/'confirm1200_conditional_v1/inputs/manifest.json').read_text())
    index = {row['key']:row for row in manifest}
    assert len(rows)==600 and len({row['key'] for row in rows})==600
    bound = []
    for row in rows:
        source = index[row['key']]
        assert all(source[key]==row[key] for key in ('fold','c','support','query'))
        bound.append(source)
    OUT.mkdir(parents=True, exist_ok=False)
    fields = np.zeros((600,64,64),np.float32)
    receipts = [None]*600
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool:
        pending = [pool.submit(replay, (i,row)) for i,row in enumerate(bound)]
        for number, future in enumerate(concurrent.futures.as_completed(pending),1):
            i, field, receipt = future.result()
            fields[i] = field
            receipts[i] = receipt
            if number%25==0:
                print(json.dumps(dict(finished=number,total=600,seconds=time.monotonic()-started)),flush=True)
    np.savez_compressed(OUT/'replayed_fields.npz',rcg=fields)
    (OUT/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    with np.load(BASE/'claude_order_fresh600/tokens.npz',allow_pickle=False) as source:
        truth = source['truth'].astype(np.float32)>.5
        stored = source['rcg'].astype(np.float32)
    classes = np.array([row['c'] for row in rows])
    masks = dict(stored=stored>.5,replay=fields.reshape(600,4096)>.5,
                 replay_FP16_compatibility=fields.reshape(600,4096).astype(np.float16)>.5)
    counts = {name:np.stack(((mask&truth).sum(1),(mask|truth).sum(1)),1) for name,mask in masks.items()}
    scores = {name:metric(value,classes) for name,value in counts.items()}
    report = dict(n=600,classes=len(np.unique(classes)),score=scores,wall_seconds=time.monotonic()-started,
        workers=16,threads=2,CPU_only=True,query_GT_first_read='after all600 replay fields',
        decomposition_started=False,rcg_config=rcg.CONFIG,
        token_source_sha256=sha(BASE/'claude_order_fresh600/tokens.npz'),
        rows_sha256=sha(BASE/'claude_order_fresh600/rows.json'),
        feature_manifest_sha256=sha(BASE/'confirm1200_conditional_v1/inputs/manifest.json'),
        rcg_source_sha256=sha(rcg.__file__),runner_sha256=sha(__file__),
        raw_mask_different_tokens=int(sum(r['raw_cut_mask_different_tokens'] for r in receipts)),
        FP16_mask_different_tokens=int(sum(r['FP16_cut_mask_different_tokens'] for r in receipts)),
        episodes_raw_mask_different=int(sum(r['raw_cut_mask_different_tokens']>0 for r in receipts)),
        maximum_field_error=max(r['max_abs_error'] for r in receipts))
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'counts.npz',**counts)
    print(json.dumps(report),flush=True)
