#!/usr/bin/env python3
"""Post-hoc16-state accounting of pre/native/RCG/truth on sealed4000draws.

This is a GT diagnostic, not inference or a deployable selector. All counts use
existing packed masks; no feature, encoder, new component or parameter search.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
from ics.experiment import sha,unpack,summarize


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def one(job):
    from scipy.ndimage import distance_transform_edt
    packet,pred,expected_hash,expected_iu,key=job
    if sha(pred)!=expected_hash:raise ValueError('Changed sealed prediction '+key)
    with np.load(packet,allow_pickle=False) as z:
        pre,native,truth=(unpack(z[k]) for k in ('pre','native','truth'))
    with np.load(pred,allow_pickle=False) as z:rcg=unpack(z['RCG'])
    for name,mask in (('native',native),('rcg',rcg)):
        iu=[int((mask&truth).sum()),int((mask|truth).sum())]
        if iu!=expected_iu[name]:raise ValueError('Previously verified4000I/U changed '+name+' '+key)
    if truth.any() and not truth.all():
        distance=np.where(truth,distance_transform_edt(truth),distance_transform_edt(~truth))
        band=np.where(distance<=8,0,np.where(distance<=16,1,2)).astype(np.uint8)
    else:band=np.full(truth.shape,2,dtype=np.uint8)
    state=pre.astype(np.uint8)+2*native.astype(np.uint8)+4*rcg.astype(np.uint8)+8*truth.astype(np.uint8)
    counts=np.bincount((state+16*band).ravel(),minlength=48).reshape(3,16)
    if int(counts.sum())!=1024**2:raise ValueError('Nonexhaustive accounting')
    return counts,dict(packet_sha256=sha(packet),prediction_sha256=expected_hash)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--verified',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    previous=[json.loads(s) for s in a.verified.read_text().splitlines()]
    if len(previous)!=4000:raise ValueError('Require all4000preserved draws')
    runs={};rows=[];jobs=[]
    for row in previous:
        block=row['public_batch'];run=a.root/f'outputs/claude_official/run{block}'
        if block not in runs:
            seal=json.loads((run/'sealed.json').read_text())
            if seal['state']!='ALL_PREDICTIONS_SEALED' or sha(run/'manifest.json')!=seal['manifest_sha256']:
                raise ValueError('Unsealed source')
            runs[block]=dict(seal=seal,seal_sha256=sha(run/'sealed.json'))
        source_key=row['key'].split(':',1)[-1]
        packet=a.root/f'outputs/claude_official/root{block}/results/extent_v1/run/packets/{source_key}.npz'
        pred=run/'predictions'/(source_key+'.npz')
        jobs.append((str(packet),str(pred),runs[block]['seal']['predictions'][source_key],row['iu'],row['key']))
        rows.append(dict(row,batch=str(block)))
    begin=time.monotonic();results=[]
    with ProcessPoolExecutor(a.workers) as pool:
        for n,result in enumerate(pool.map(one,jobs,chunksize=4),1):
            results.append(result)
            if n%400==0:print(json.dumps(dict(completed=n,total=4000,seconds=time.monotonic()-begin)),flush=True)
    counts=np.stack([r[0] for r in results]);pooled=counts.sum(1)
    states=np.arange(16);P=(states&1)>0;N=(states&2)>0;R=(states&4)>0;T=(states&8)>0
    def iu(mask):return np.stack((pooled[:,mask&T].sum(1),pooled[:,mask|T].sum(1)),axis=1)
    arrays={'native':iu(N),'foris_pre.control':iu(P),'rcg':iu(R)}
    report,draws=summarize(rows,arrays,{})
    definitions={
        'native_to_rcg.add_TP':R&~N&T,'native_to_rcg.add_FP':R&~N&~T,
        'native_to_rcg.delete_TP':N&~R&T,'native_to_rcg.delete_FP':N&~R&~T,
        'pre_to_native.add_TP':N&~P&T,'pre_to_native.add_FP':N&~P&~T,
        'pre_to_native.delete_TP':P&~N&T,'pre_to_native.delete_FP':P&~N&~T,
        'pre_to_rcg.add_TP':R&~P&T,'pre_to_rcg.add_FP':R&~P&~T,
        'pre_to_rcg.delete_TP':P&~R&T,'pre_to_rcg.delete_FP':P&~R&~T,
        'shared_pre_add.TP':~P&N&R&T,'shared_pre_add.FP':~P&N&R&~T,
        'shared_pre_delete.TP':P&~N&~R&T,'shared_pre_delete.FP':P&~N&~R&~T,
        'rcg_cancels_native_add.TP':~P&N&~R&T,'rcg_cancels_native_add.FP':~P&N&~R&~T,
        'rcg_cancels_native_delete.TP':P&~N&R&T,'rcg_cancels_native_delete.FP':P&~N&R&~T,
    }
    edits={name:counts[:,:,selection].sum(2) for name,selection in definitions.items()}
    report['edit_counts']={name:dict(total=int(value.sum()),GTdistance_0to8=int(value[:,0].sum()),
        GTdistance_8to16=int(value[:,1].sum()),GTdistance_over16=int(value[:,2].sum())) for name,value in edits.items()}
    # Correct set-accounting recomposes each exact mask's I/U.
    for first,last,prefix in (('native','rcg','native_to_rcg'),('foris_pre.control','native','pre_to_native'),
                              ('foris_pre.control','rcg','pre_to_rcg')):
        base=arrays[first].copy()
        base[:,0]+=edits[prefix+'.add_TP'].sum(1)-edits[prefix+'.delete_TP'].sum(1)
        base[:,1]+=edits[prefix+'.add_FP'].sum(1)-edits[prefix+'.delete_FP'].sum(1)
        if not np.array_equal(base,arrays[last]):raise ValueError('Set accounting failure')
    report.update(scope='GTdiagnostics on4000benchmark reuse; not method inference or independent confirmation',
        state_bits={'pre':1,'native':2,'RCG':4,'truth':8},distance='nearest opposite GT label at1024; bands<=8,(8,16],>16',
        exact_draw_IU_parity=True,source_code_sha256=sha(Path(__file__)),seconds=time.monotonic()-begin,
        native_mask_sealed=True,query_GT_used_for_diagnostics=True,raw_DINO_edits='unavailable beyond DEV241')
    a.out.mkdir(parents=True);write(a.out/'manifest.json',rows);write(a.out/'report.json',report)
    write(a.out/'receipt.json',dict(verified_IU_sha256=sha(a.verified),source_seals={str(k):v['seal_sha256'] for k,v in runs.items()},
        inputs={row['key']:result[1] for row,result in zip(rows,results)},source_code_sha256=sha(Path(__file__))))
    np.savez_compressed(a.out/'states.npz',counts=counts,**{k:v for k,v in arrays.items()})
    np.save(a.out/'bootstrap_photo_draws.npy',draws)
    print(json.dumps(dict(scores=report['scores'],edit_counts=report['edit_counts'])),flush=True)


if __name__=='__main__':main()
