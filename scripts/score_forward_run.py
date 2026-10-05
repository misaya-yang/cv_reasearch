#!/usr/bin/env python3
"""Score a sealed complete-native forward run against its exact cached controls."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--rcg-run',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    from ics.experiment import sha,unpack,packet,summarize
    seal=json.loads((a.out/'sealed.json').read_text())
    if sha(a.out/'manifest.json')!=seal['manifest_sha256']:raise ValueError('Manifest changed')
    rows=json.loads((a.out/'manifest.json').read_text())
    rcg_seal=json.loads((a.rcg_run/'sealed.json').read_text())
    if sha(a.rcg_run/'manifest.json')!=rcg_seal['manifest_sha256']:raise ValueError('RCG manifest changed')
    rcg_rows={r['key']:r for r in json.loads((a.rcg_run/'manifest.json').read_text())}
    arrays={}; corrections={}; details=[]
    for row in rows:
        key=row['key'];path=a.out/'predictions'/f'{key}.npz';rcg_path=a.rcg_run/'predictions'/f'{key}.npz'
        if sha(path)!=seal['predictions'][key]:raise ValueError('Forward prediction changed')
        if sha(rcg_path)!=rcg_seal['predictions'][key]:raise ValueError('RCG prediction changed')
        if any(row[k]!=rcg_rows[key][k] for k in ('fold','c','support','query')):raise ValueError('RCG cohort mismatch')
        pp=packet(a.cache_root,row)
        if sha(pp)!=rcg_seal['inputs'][key]['packet_sha256']:raise ValueError('Cached control packet changed')
        with np.load(pp,allow_pickle=False) as f: truth,native=(unpack(f[k]) for k in ('truth','native'))
        with np.load(path,allow_pickle=False) as f: masks={k:unpack(f[k]) for k in f.files}
        mismatch=int(np.count_nonzero(masks['native']!=native))
        if mismatch:
            raise ValueError(f'{key}: {mismatch} native pixels differ. Stop same-protocol comparison; diagnose replay first.')
        with np.load(rcg_path,allow_pickle=False) as f: masks['rcg']=unpack(f['rcg'])
        for arm,m in masks.items():
            iu=[int((m&truth).sum()),int((m|truth).sum())]
            arrays.setdefault(arm,[]).append(iu)
            add,delete=m&~native,native&~m
            corr=dict(key=key,c=row['c'],fold=row['fold'],batch=str(row.get('batch','unspecified')),
                      add_TP=int((add&truth).sum()),delete_FP=int((delete&~truth).sum()),
                      delete_TP=int((delete&truth).sum()),add_FP=int((add&~truth).sum()))
            corrections.setdefault(arm,[]).append(corr)
            details.append(dict(corr,arm=arm,intersection=iu[0],union=iu[1]))
    arrays={k:np.asarray(v,dtype=np.int64) for k,v in arrays.items()}
    if any(len(v)!=len(rows) for v in arrays.values()):raise ValueError('Incomplete arm cohort')
    report,draws=summarize(rows,arrays,corrections)
    report.update(native_replay_mismatched_pixels=0,forward_seal_sha256=sha(a.out/'sealed.json'),rcg_seal_sha256=sha(a.rcg_run/'sealed.json'))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'episode_metrics.json').write_text(json.dumps(details,indent=2)+'\n')
    np.save(a.out/'bootstrap_photo_draws.npy',draws)
    print(json.dumps(report['scores'],indent=2))

if __name__=='__main__':main()
