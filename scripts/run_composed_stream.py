#!/usr/bin/env python3
"""Frozen composed predictor on public blocks4/5, then combined2400 readout.

Consume live provider features as masks/fields become ready. Preserve compact
packets/masks; never delete or recreate features. No query GT before own sealing.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import run_composed_operator as method
from ics.experiment import sha


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def source_live(freeze):
    p=Path('/proc')/str(freeze['source_chain']['pid'])/'stat'
    if not p.exists():return False
    s=p.read_text().rsplit(')',1)[1].split()
    return s[0] not in ('Z','X') and int(s[19])==freeze['source_chain']['start_ticks']


def infer(a):
    import numpy as np
    import torch
    from ics.experiment import load_inputs
    if a.out.exists():raise FileExistsError('Fresh output required')
    freeze=json.loads(a.freeze.read_text())
    for name,digest in freeze['method_source_sha256'].items():
        if sha(REPO/'scripts'/name)!=digest:raise ValueError('Frozen predictor changed')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.1)
    a.out.mkdir();(a.out/'predictions').mkdir();(a.out/'packets').mkdir()
    rows=[];parents={}
    for block in a.blocks:
        source=a.public/('batch'+str(block)+'.json');data=json.loads(source.read_text())
        run=a.public/('run'+str(block));cache=a.public/('root'+str(block))
        parents[str(run)]=dict(block=block,manifest_sha256=sha(source),captured={})
        for item in data['episodes']:
            old=dict(item);key=f"{old['fold']}_{old['e']}_{old['c']}"
            rows.append(dict(old,key=f'public{block}__{key}',source_key=key,public_batch=block,
                batch='public'+str(block),recheck_run=str(run),
                feature_export=str(cache/'cache/evidence_v1/feat'/(key+'.pt')),
                source_packet=str(cache/'results/extent_v1/run/packets'/(key+'.npz')),
                packet_export=str(a.out/'packets'/(f'public{block}__{key}'+'.npz'))))
    if len(rows)!=1200 or len({r['key'] for r in rows})!=1200:raise ValueError('Require1200new draw IDs')
    protocol=dict(n=1200,freeze_sha256=sha(a.freeze),freeze_path=str(a.freeze),config=method.CONFIG,
        base_run=str(a.base),base_seal_sha256=sha(a.base/'sealed.json'),
        base_episode_sha256=sha(a.base/'episodes.jsonl'),base_report_sha256=sha(a.base/'report.json'),
        resolution=1024,seed=0,blocks=a.blocks,query_gt_in_inference=False,additional_encoder_forwards=0,
        exposure='Frozen after original1200; new public draw blocks4/5; benchmark reuse and photo overlap retained',
        feature_policy='Read provider features in RAM; no feature deletion/recreation; protect existing1200cache',
        source_code={str(REPO/'scripts'/name):sha(REPO/'scripts'/name) for name in
            ['run_composed_stream.py','run_composed_operator.py','run_directional_operator.py','run_calibrated_operator.py','pixel_budget_control.py']},
        input_receipts={},parents={})
    write(a.out/'manifest.json',rows);write(a.out/'protocol.json',protocol)
    pending=list(rows);hashes={};choices={};begin=time.monotonic();last_wait=None
    while pending:
        progress=False
        for row in list(pending):
            run=Path(row['recheck_run']);key=row['source_key'];draw=row['key']
            prediction=run/'predictions'/(key+'.npz');field=run/'fields'/(key+'.npz')
            if not prediction.exists() or not field.exists():continue
            if not Path(row['feature_export']).exists() or not Path(row['source_packet']).exists():
                raise RuntimeError('Provider inputs expired; no cache reconstruction: '+draw)
            try:
                with np.load(prediction,allow_pickle=False) as z:
                    comparisons={'rcg':z['RCG'].copy(),'astra.control':z['external_mean__delete'].copy(),
                        'mean.control':z['MEAN_CONTROL'].copy(),'Cbase.control':z['RCG_count_matched_delete'].copy(),
                        'conservative.control':z['conservative_delete'].copy()}
                with np.load(field,allow_pickle=False) as z:values=z['rcg'].copy()
            except (OSError,EOFError,ValueError,zipfile.BadZipFile):continue
            packet_digest=sha(row['source_packet']);shutil.copy2(row['source_packet'],row['packet_export'])
            if sha(row['packet_export'])!=packet_digest:raise ValueError('Packet changed during copy')
            inputs=dict(row,packet_export=row['source_packet'])
            (q,r,cov,score),receipt=load_inputs(Path('/'),inputs)
            with np.load(row['packet_export'],allow_pickle=False) as z:comparisons['native']=z['native'].copy()
            output,_,info=method.predict(q,r,cov,comparisons,values)
            info.pop('raw_origin_unchanged',None);choices[draw]=info
            destination=a.out/'predictions'/(draw+'.npz');np.savez_compressed(destination,**comparisons,**output)
            hashes[draw]=sha(destination);captured=dict(prediction=sha(prediction),field=sha(field))
            parents[str(run)]['captured'][key]=captured
            protocol['input_receipts'][draw]=dict(receipt,copied_packet_sha256=packet_digest,**captured)
            pending.remove(row);progress=True
            if len(hashes)%25==0:print(json.dumps(dict(n=len(hashes),total=1200,seconds=round(time.monotonic()-begin,1))),flush=True)
        if not progress:
            if not source_live(freeze):raise RuntimeError('Verified source chain stopped with missing draws')
            state=(len(hashes),pending[0]['public_batch'])
            if state!=last_wait:
                print(json.dumps(dict(state='WAIT_SOURCE_FIELDS',n=len(hashes),block=state[1],source_chain=freeze['source_chain'])),flush=True);last_wait=state
            time.sleep(1)
    for name,record in parents.items():
        run=Path(name)
        while not (run/'sealed.json').exists():
            if not source_live(freeze):raise RuntimeError('Source stopped before seal')
            time.sleep(1)
        parent=json.loads((run/'sealed.json').read_text());source_rows=json.loads((run/'manifest.json').read_text())
        if parent['state']!='ALL_PREDICTIONS_SEALED' or parent.get('query_gt_in_inference') is not False or sha(run/'manifest.json')!=parent['manifest_sha256']:
            raise ValueError('Invalid source seal')
        by_key={r['key']:r for r in source_rows}
        for row in [r for r in rows if r['recheck_run']==name]:
            key=row['source_key'];old=by_key[key];capture=record['captured'][key]
            if any(old[k]!=row[k] for k in ['c','support','query']):raise ValueError('Source identity changed')
            if capture['prediction']!=parent['predictions'][key] or capture['field']!=parent['fields'][key]:raise ValueError('Source payload changed')
        protocol['parents'][name]=dict(seal_sha256=sha(run/'sealed.json'),block=record['block'])
    write(a.out/'protocol.json',protocol);write(a.out/'choices.json',choices)
    write(a.out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=1200,query_gt_in_inference=False,
        manifest_sha256=sha(a.out/'manifest.json'),protocol_sha256=sha(a.out/'protocol.json'),
        choices_sha256=sha(a.out/'choices.json'),predictions=hashes,seconds=time.monotonic()-begin,
        cuda_peak_bytes=torch.cuda.max_memory_allocated()))


def score(a):
    import itertools
    import numpy as np
    from ics.experiment import unpack,summarize
    seal=json.loads((a.out/'sealed.json').read_text());protocol=json.loads((a.out/'protocol.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED' or (a.out/'report.json').exists():raise ValueError('Incomplete or already scored')
    for field,name in [('manifest_sha256','manifest.json'),('protocol_sha256','protocol.json'),('choices_sha256','choices.json')]:
        if sha(a.out/name)!=seal[field]:raise ValueError('Changed scoring input')
    for name,digest in protocol['source_code'].items():
        if sha(name)!=digest:raise ValueError('Changed code')
    if sha(a.base/'sealed.json')!=protocol['base_seal_sha256']:raise ValueError('Changed original1200')
    if sha(a.base/'episodes.jsonl')!=protocol['base_episode_sha256'] or sha(a.base/'report.json')!=protocol['base_report_sha256']:
        raise ValueError('Changed original1200readout')
    rows=json.loads((a.out/'manifest.json').read_text());details=[];edits={};count_cache={}
    banks=('astra','p','all','far')+tuple(f'top{f:g}' for f in (.02,.05,.1,.2))+('split:astra','split:p','split:top0.05','split:top0.1')+('p0.5','p2','p4')
    index=list(itertools.product(banks,(0,5),(-.1,-.05,-.02,0.,.02,.05))).index(('p',0,0.))
    freeze=json.loads(a.freeze.read_text())
    for name in protocol['parents']:
        run=Path(name);path=run/'sweep_counts_wide.npz'
        if sha(run/'sealed.json')!=protocol['parents'][name]['seal_sha256']:raise ValueError('Changed source seal')
        while True:
            try:
                with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in ['iu_c','iu_rcg','iu_mean','iu_astra','d_rem','d_ctl']}
                break
            except (OSError,EOFError,ValueError,zipfile.BadZipFile):
                if not source_live(freeze):raise RuntimeError('Source stopped before count parity')
                time.sleep(1)
        source_rows=json.loads((run/'manifest.json').read_text())
        count_cache[name]=({r['key']:i for i,r in enumerate(source_rows)},data)
    for row in rows:
        key=row['key'];packet=Path(row['packet_export']);pred=a.out/'predictions'/(key+'.npz')
        if sha(packet)!=protocol['input_receipts'][key]['copied_packet_sha256'] or sha(pred)!=seal['predictions'][key]:raise ValueError('Changed prediction or GT packet')
        with np.load(packet,allow_pickle=False) as z:truth=unpack(z['truth'])
        with np.load(pred,allow_pickle=False) as z:masks={name:unpack(z[name]) for name in z.files}
        record=dict(row,iu={})
        for name,mask in masks.items():
            record['iu'][name]=[int((mask&truth).sum()),int((mask|truth).sum())]
            plus,minus=mask&~masks['rcg'],masks['rcg']&~mask
            values=dict(add_TP=int((plus&truth).sum()),add_FP=int((plus&~truth).sum()),delete_TP=int((minus&truth).sum()),delete_FP=int((minus&~truth).sum()))
            total=edits.setdefault(name,{k:0 for k in values})
            for k,v in values.items():total[k]+=v
        ids,data=count_cache[row['recheck_run']];i=ids[row['source_key']]
        for arm,expected in [('frozen.delete_p.control',data['iu_c'][i]-data['d_rem'][index,i]),
                             ('frozen.delete_p.same_count.control',data['iu_c'][i]-data['d_ctl'][index,i]),
                             ('rcg',data['iu_rcg'][i]),('mean.control',data['iu_mean'][i]),('astra.control',data['iu_astra'][i])]:
            if not np.array_equal(record['iu'][arm],expected):raise ValueError('Original fixed-rule/count parity failed: '+key+' '+arm)
        details.append(record)
    old=[json.loads(line) for line in (a.base/'episodes.jsonl').read_text().splitlines()]
    combined=old+details;arms=set(old[0]['iu'])&set(details[0]['iu'])
    arrays=lambda data:{name:np.array([r['iu'][name] for r in data]) for name in sorted(arms)}
    report,_=summarize(combined,arrays(combined),{})
    new_report,_=summarize(details,arrays(details),{})
    old_report=json.loads((a.base/'report.json').read_text());combined_edits={}
    for arm in arms:combined_edits[arm]={k:old_report['edits_vs_RCG'][arm][k]+edits[arm][k] for k in edits[arm]}
    report.update(primary='conditional.joint',new_draws_readout=new_report,edits_vs_RCG=combined_edits,
        new_draws_edits_vs_RCG=edits,exposure=protocol['exposure'],source_blocks=[0,1]+a.blocks,
        fixed_deletion_replay_parity=2400,parameters_updated=False,
        raw_DINO='original DEV241 accounting retained; no raw masks fabricated for2400',
        INSID3='released logic onDEV241; full2400 INSID3 absent')
    write(a.out/'report.json',report)
    (a.out/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in combined))
    print(json.dumps(dict(n=2400,scores=report['scores'],primary=report['contrasts']['conditional.joint'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['infer','score'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    p.add_argument('--blocks',type=int,nargs='+',default=[4,5]);a=p.parse_args()
    {'infer':infer,'score':score}[a.phase](a)


if __name__=='__main__':main()
