"""Supplement paired macro intervals after a window experiment is sealed and scored.

Same observed-slot photo bootstrap convention as x1013/score.py, with one RNG
stream across domains rather than resetting a seed per domain. Development
intervals condition on the fixed reference configurations; no new inference.
"""
from pathlib import Path
import gzip
import json
import sys
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/x1012'),str(ROOT/'scripts/x1013')]
import common as c
import score


def draws(records,arms,rng):
    photos=list(dict.fromkeys(r['photo'] for r in records))
    slots=list(dict.fromkeys((r['fold'],r['cls']) for r in records))
    pi={v:i for i,v in enumerate(photos)};si={v:i for i,v in enumerate(slots)}
    p=[pi[r['photo']] for r in records];s=[si[(r['fold'],r['cls'])] for r in records]
    shape=(len(photos),len(slots))
    w=np.stack([np.bincount(rng.integers(len(photos),size=len(photos)),minlength=len(photos)) for _ in range(500)])
    presence=sparse.csr_matrix((np.ones(len(records)),(p,s)),shape=shape)
    represented=w@presence>0
    result={}
    for a in arms:
        ii=sparse.csr_matrix(([r[a][0] for r in records],(p,s)),shape=shape)
        uu=sparse.csr_matrix(([r[a][1] for r in records],(p,s)),shape=shape)
        j=100*(w@ii)/np.maximum(w@uu,1)
        result[a]=(j*represented).sum(1)/represented.sum(1)
    return result


def analyze(name):
    paths={'W59':'w59_normalization_causal_S.jsonl.gz','W60':'w60_pure_background_S.jsonl.gz','W61':'w61_reference_stability_S.jsonl.gz','W62':'w62_background_channels_S.jsonl.gz'}
    out=c.task_path(name)
    assert (out/'complete.json').exists() and (out/'scored.json').exists()
    with gzip.open(ROOT/'evidence/research'/paths[name],'rt') as f:records=list(map(json.loads,f))
    if name=='W59':pairs=[(a+s,'A0'+s) for a in ('Nout','AbsFix','AbsWeight') for s in ('','+')]
    elif name=='W60':pairs=[('PureBG'+s,'A0'+s) for s in ('','+')]
    elif name=='W61':pairs=[('pooled_Fixed'+s,'pooled_SH'+s) for s in ('','+')]
    else:pairs=[(a+s,b+s) for a,b in (('K','A0'),('Mclean','A0'),('Joint','A0'),('Joint','K'),('Joint','Mclean')) for s in ('','+')]
    arms=sorted({a for pair in pairs for a in pair});rng=np.random.default_rng(2026101607)
    all_draws={};point={}
    for d in c.DOMAINS:
        sel=[r for r in records if r['dataset']==d]
        assert sel,d
        all_draws[d]=draws(sel,arms,rng)
        point[d]={a:score.miou(sel,a) for a in arms}
    result={'run':name,'n':len(records),'repeats':500,'seed':2026101607,
            'scope':'original S development; paired query photos within each fixed domain; fixed reference configurations; observed slots per bootstrap draw',
            'limitations':'not independent confirmation; no guarantee for new random references or unmodelled cross-domain/cross-role photo dependence',
            'comparisons':{}}
    for a,b in pairs:
        dd=np.mean([all_draws[d][a]-all_draws[d][b] for d in c.DOMAINS],axis=0)
        key=a+' - '+b if name=='W62' else a
        result['comparisons'][key]={'reference':b,'delta':float(np.mean([point[d][a]-point[d][b] for d in c.DOMAINS])),
                                  'ci':np.percentile(dd,[2.5,97.5]).tolist()}
    (out/'macro_intervals.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'evidence/research'/f'{name.lower()}_macro_intervals.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):analyze(sys.argv[1])
