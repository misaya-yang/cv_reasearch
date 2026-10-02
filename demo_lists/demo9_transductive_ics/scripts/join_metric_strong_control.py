#!/usr/bin/env python3
"""CPU-only exact role/task join, add a separately frozen strong host control."""
import argparse
import hashlib
import json
from pathlib import Path


def identity(r):return (r['e'],r['c'],r['support'],r['query'])


def join(metric,strong):
    if metric.get('state')!='COMPLETED' or strong.get('state')!='COMPLETED':raise ValueError('Both predictions must already be complete')
    controls={identity(r):r for r in strong['records']}
    if len(controls)!=len(strong['records']) or set(controls)!={identity(r) for r in metric['records']}:raise ValueError('Frozen task/ordered RGB roles differ')
    for row in metric['records']:
        extra=controls[identity(row)]
        for key in ('iu','original_iu'):
            if set(row[key])&set(extra[key]):raise ValueError('Do not overwrite any scored method')
            row[key].update(extra[key])
        row['arm_states'].update(extra['armstate'])
    metric['scope']='Frozen learned metrics plus full native FoRIS CRF on exactly the same held-out tasks; no new prediction or GT selection'
    return metric


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metric',type=Path);p.add_argument('--strong',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:
        r=dict(e=0,c=4,support='s',query='q')
        metric=dict(state='COMPLETED',records=[dict(r,iu={'learner':[2,4]},original_iu={'learner':[2,4]},arm_states={'learner':'SOLVED'})])
        control=dict(state='COMPLETED',records=[dict(r,iu={'foris_crf':[3,4]},original_iu={'foris_crf':[3,4]},armstate={'foris_crf':'COMPLETED'})])
        assert join(metric,control)['records'][0]['iu']['foris_crf']==[3,4]
        print('CPU_EXACT_TASK_JOIN_PASSED');return
    if not all((a.metric,a.strong,a.out)):p.error('Explicit frozen source reports required')
    if a.out.exists():raise ValueError('Fresh joined output required')
    result=join(json.loads(a.metric.read_text()),json.loads(a.strong.read_text()))
    result['joined_source_sha256']={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in (a.metric,a.strong)}
    a.out.write_text(json.dumps(result,indent=2));print('COMPLETED_EXACT_FROZEN_TASK_JOIN')


if __name__=='__main__':main()
