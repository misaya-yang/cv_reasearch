"""CPU-only native Part2 acceptance from actual original reference annotations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path('/root/autodl-tmp/evidence_bench_step3_20261007_25142/code')
sys.path.insert(0,str(CODE/'scripts'))
import bench_evidence as bench
BASE=Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs')
ANNOTATION=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')
OUT=CODE.parent/'acceptance2_original_annotation_v2'
DATA={}

def init():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    prefix,prefix_hash=bench.part2_prefix()
    DATA.update(prefix=prefix,prefix_hash=prefix_hash)

def replay(task):
    index,row=task;begin=time.monotonic()
    q,r,features=bench.features(Path(row['feature_export']))
    packet,packet_hash=bench.packet(Path(row['packet_export']))
    fg,reference=bench.native_foreground(row,ANNOTATION,packet['cov'])
    evidence=bench.native_evidence(fg,DATA['prefix'])
    field=evidence(q,r,packet['cov'])
    return index,field,dict(key=row['key'],features=features,reference=reference,
        feature_path=row['feature_export'],packet_path=row['packet_export'],packet_sha256=packet_hash,
        prefix_hash=DATA['prefix_hash'],part2=evidence.last_info,wall_seconds=time.monotonic()-begin,query_GT_read=False)

if __name__=='__main__':
    started=time.monotonic();fresh_path=BASE/'claude_order_fresh600/rows.json'
    manifest_path=BASE/'confirm1200_conditional_v1/inputs/manifest.json'
    fresh=bench.rows(fresh_path);joined=bench.exact_join(fresh,bench.rows(manifest_path))
    OUT.mkdir(exist_ok=False);fields=np.zeros((600,64,64),np.float32);receipts=[None]*600
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        futures=[pool.submit(replay,(i,r)) for i,r in enumerate(joined)]
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            i,field,receipt=future.result();fields[i]=field;receipts[i]=receipt
            if n%25==0:print(json.dumps(dict(finished=n,total=600,seconds=time.monotonic()-started)),flush=True)
    np.savez_compressed(OUT/'replayed_fields.npz',s2=fields)
    (OUT/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    with np.load(BASE/'claude_order_fresh600/tokens.npz',allow_pickle=False) as source:
        stored=source['s2'].copy().reshape(600,64,64);truth=source['truth'].astype(np.float32).reshape(600,64,64)>.5
    classes=np.array([r['c'] for r in fresh]);counts={}
    replay_masks=np.stack([bench.mask(field,'raw') for field in fields])
    stored_masks=np.stack([bench.mask(field,'raw') for field in stored])
    for name,masks in (('stored',stored_masks),('replay',replay_masks)):
        counts[name]=np.stack(((masks&truth).sum((1,2)),(masks|truth).sum((1,2))),1)
    scores={name:bench.miou(value,classes) for name,value in counts.items()}
    fraction=float(np.mean(replay_masks!=stored_masks));gain=scores['replay']-scores['stored']
    report=dict(n=600,classes=len(np.unique(classes)),scores=scores,gain_pp=gain,
        mask_different_tokens=int(np.count_nonzero(replay_masks!=stored_masks)),mask_disagreement_fraction=fraction,
        maximum_field_error=float(np.max(np.abs(fields-stored.astype(np.float32)))),
        strict_tolerance_pass=abs(gain)<=.3 and fraction<=.01,
        tier='passed' if abs(gain)<=.3 and fraction<=.01 else 'recomputed_s2_endpoint' if abs(gain)<=2 else 'stop',
        annotation_root=str(ANNOTATION),original_cov_exact_all600=True,CPU_only=True,workers=16,threads=2,
        wall_seconds=time.monotonic()-started,query_GT_first_read='after all600 inference fields',
        token_source_sha256=bench.sha(BASE/'claude_order_fresh600/tokens.npz'),rows_sha256=bench.sha(fresh_path),
        feature_manifest_sha256=bench.sha(manifest_path),runner_sha256=bench.sha(__file__),
        helper_sha256=bench.sha(bench.__file__),stage_bank_sha256=bench.sha(CODE/'src/ics/methods/stage_bank.py'),
        fields_sha256=bench.sha(OUT/'replayed_fields.npz'),receipts_sha256=bench.sha(OUT/'receipts.json'),
        no_new_encoder=True,decomposition_started=False)
    np.savez_compressed(OUT/'counts.npz',**counts)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
