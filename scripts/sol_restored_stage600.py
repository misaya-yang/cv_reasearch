"""Saved failed restorations versus original S observation/gate/fusion evidence."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/local/restored_stage600_20261011'
METHODS={'graph':('query_task_graph600_20261011','query_task_graph_complete'),
         'discriminant':('task_discriminant600_20261011','joint_task_discriminant_complete')}

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'report.json').exists()
    start,cpu=time.monotonic(),time.process_time();torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('stage',ROOT/'scripts/sol_strong_stage600.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    h=s.helper_module();inv=s.read(s.OUT/'inventory.json');protocol=s.read(s.OUT/'protocol.json')
    assert s.sha(s.OUT/'inventory.json')==protocol['inventory_sha256']
    for p,digest in protocol['source_sha256'].items():assert s.sha(p)==digest
    previous=s.read(ROOT/'evidence/local/strong_veto_reversal600_20261011/report.json')
    members=s.read(s.ALIGN/'membership.json');bindings=s.lines(s.OLD/'score_bindings.jsonl')
    seals={k:s.read(ROOT/'evidence/local'/d/'sealed.json') for k,(d,_) in METHODS.items()}
    receipts={}
    for e in inv['records']:
        for asset in e['assets'].values():assert s.sha(asset['path'])==asset['sha256']
        if e['cohort'] not in receipts:receipts[e['cohort']]={r['episode_id']:r for r in s.lines(Path(e['cohort'])/'inference.jsonl')}
        assert s.canonical_sha(receipts[e['cohort']][e['episode_id']])==e['region_receipt_sha256']
    for name,(d,_) in METHODS.items():
        for i,m in enumerate(members):
            rec=seals[name]['records'][i];assert rec['episode_id']==m['episode_id']
            assert s.sha(ROOT/'evidence/local'/d/'predictions'/rec['filename'])==rec['prediction_sha256']
    checks=Counter();totals=Counter();records=[];hashed=set()
    for i,(m,e) in enumerate(zip(members,inv['records'])):
        assert e['index']==i and e['episode_id']==m['episode_id']
        shape,base,stages,raw,votes,_=s.reconstruct(e,receipts[e['cohort']][e['episode_id']],h,checks)
        strong=stages['S'];binding=bindings[m['source_index']]
        b=h.old_support(binding,(1024,1024),'cli',hashed,checks)['B'];negative=b&~strong
        # Disjoint earliest visible failure groups; gate rejection can coexist with raw minority.
        groups={'no_native_support':raw==0,
                'native_support_no_accepted_support':(raw>0)&(votes==0)&~stages['native_majority'],
                'accepted_support_raw_majority_failed':(votes>0)&~stages['native_majority'],
                'raw_majority_gate_changes_final':stages['native_majority']&~strong}
        coverage=sum(mask.astype(np.uint8) for mask in groups.values())
        assert np.all(coverage[negative]==1);checks['stage_partition']+=1
        target=h.resize(h.truth(binding['row'],checks),(1024,1024),'nearest')>.5;counts={}
        for group,mask in groups.items():
            for role,t in [('TP',target),('FP',~target)]:counts[f'denominator/{group}/{role}']=int((negative&mask&t).sum())
        for name,(d,arm) in METHODS.items():
            rec=seals[name]['records'][i]
            with np.load(ROOT/'evidence/local'/d/'predictions'/rec['filename'],allow_pickle=False) as z:pred=h.unpack(z,'cli/'+arm,(1024,1024),checks)
            for group,mask in groups.items():
                for role,t in [('TP',target),('FP',~target)]:counts[f'{name}/{group}/{role}']=int((negative&pred&mask&t).sum())
            for role in ('TP','FP'):
                assert sum(counts[f'{name}/{group}/{role}'] for group in groups)==previous['records'][i]['frames']['cli']['methods'][name]['reverse_S_B_'+role];checks['prior_restore_reconciliation']+=1
        totals.update(counts);records.append(dict(index=i,episode_id=m['episode_id'],counts=counts))
        if (i+1)%100==0:print(json.dumps({'completed':i+1}),flush=True)
    report=dict(n=600,frame='CLI1024',totals=dict(totals),records=records,checks=dict(checks),wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                raw_reads=0,new_predictions=0,new_fits=0,query_GT_reads=600,
                scope='Saved S stage/failed-restoration join, observational. No accepted support is a fixed old gate fact, not semantic ground truth; raw majority failed may coexist with gate rejection. No new method.')
    s.write(OUT/'report.json',report);print(json.dumps({'totals':dict(totals),'wall_seconds':report['wall_seconds']}))

if __name__=='__main__':main()
