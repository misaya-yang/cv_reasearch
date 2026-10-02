#!/usr/bin/env python3
"""Join already frozen identity and .75 strong-host predictions by exact roles."""
import argparse
import hashlib
import json
from pathlib import Path
from join_metric_strong_control import identity


def join(base,changed):
    if base.get('state')!='COMPLETED' or changed.get('state')!='COMPLETED':
        raise ValueError('Only completed frozen predictions may be joined')
    if base['host']!='foris' or changed['host']!='foris' or base['refiner']!='crf' or changed['refiner']!='crf':
        raise ValueError('Full native FoRIS CRF control required')
    if base['args'].get('query_brightness',1.)!=1. or changed['args']['query_brightness']!=.75:
        raise ValueError('Fixed identity and .75 controls required; no parameter selection')
    rows={identity(r):r for r in base['records']}
    if len(rows)!=len(base['records']) or set(rows)!={identity(r) for r in changed['records']}:
        raise ValueError('Exact ordered task and RGB role identity required')
    for row in changed['records']:
        native=rows[identity(row)]
        for key in ('iu','original_iu'):
            if set(row[key]) & set(native[key]):raise ValueError('Cannot overwrite an arm')
            row[key].update(native[key])
        row['arm_states']={**native['armstate'],**row['armstate']}
    changed['scope']='Fixed .75 brightness stress on complete FoRIS CRF, reused exact-task identity outputs; development diagnostic, not a new method'
    changed['baseline']='foris_crf'
    changed['naive']='foris_crf_brightness075'
    changed['oracle_scope']='No model or GT-selected transform; this is a strong-host robustness control'
    return changed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path);p.add_argument('--changed',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:
        task=dict(e=0,c=4,support='s',query='q')
        b=dict(state='COMPLETED',host='foris',refiner='crf',args={},records=[dict(task,iu={'foris_crf':[2,4]},original_iu={'foris_crf':[2,4]},armstate={'foris_crf':'COMPLETED'})])
        c=dict(state='COMPLETED',host='foris',refiner='crf',args={'query_brightness':.75},records=[dict(task,iu={'foris_crf_brightness075':[3,4]},original_iu={'foris_crf_brightness075':[3,4]},armstate={'foris_crf_brightness075':'COMPLETED'})])
        assert join(b,c)['records'][0]['iu']['foris_crf']==[2,4]
        print('CPU_BRIGHTNESS_EXACT_JOIN_PASSED');return
    if not all((a.base,a.changed,a.out)):p.error('Explicit completed source reports and fresh output required')
    if a.out.exists():raise ValueError('Preserve prior outputs')
    d=join(json.loads(a.base.read_text()),json.loads(a.changed.read_text()))
    d['joined_source_sha256']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.base,a.changed)}
    a.out.write_text(json.dumps(d));print('COMPLETED_EXACT_BRIGHTNESS_CONTROL_JOIN')


if __name__=='__main__':main()
