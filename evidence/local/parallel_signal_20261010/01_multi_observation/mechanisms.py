"""Paired Deep localCRF/no-localCRF and exact recurrence accounting."""
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
import study as s
from fusion import load_fields, gross, counts


def main():
    start, cpu = time.monotonic(), time.process_time()
    diagnostic = json.loads((s.OUT/'diagnostic_report.json').read_text())
    coverage, raw, accepted, guide, truth = np.indices((5, 5, 5, 2, 2))
    recurrence = {}
    for group, z in diagnostic['groups'].items():
        h = np.asarray(z['joint_counts'])
        cells = []
        for c in (1,2,4):
            for support_name, v in [('raw',raw),('accepted',accepted)]:
                for role,t in [('target',1),('background',0)]:
                    mask = (coverage==c)&(truth==t)
                    def n(condition):
                        return int(h[mask&condition].sum())
                    cells.append(dict(coverage=c,support=support_name,role=role,
                        canvas_pixels=n(np.ones_like(mask)), any_support=n(v>0),
                        support_histogram={str(k):n(v==k) for k in range(c+1)},
                        repeated_atleast2=n(v>=2), strict_majority=n(2*v>c),
                        minority=n((v>0)&(2*v<=c)),unanimous=n(v==c),
                        minority_guide_fg=n((v>0)&(2*v<=c)&(guide==1))))
        recurrence[group] = cells
    seal=json.loads((s.OUT/'reconstructed_seal.json').read_text())
    old={r['episode_id']:r for r in seal['files'] if r['group']=='deep_original100'}
    new={r['episode_id']:r for r in seal['files'] if r['group']=='deep_fast100'}
    assert old.keys()==new.keys() and len(old)==100
    rows=[]
    for eid in old:
        a,shape=load_fields(old[eid]); b,newshape=load_fields(new[eid])
        assert shape==newshape
        assert np.array_equal(a['base_original'],b['base_original'])
        gt=s.nearest(s.load_truth(old[eid]['row']),s.COVER.shape)
        masks_a=dict(raw_union=a['raw']>0,accepted_union=a['accepted']>0,
                     raw_majority=a['unfiltered'],accepted_majority=a['region'])
        masks_b=dict(raw_union=b['raw']>0,accepted_union=b['accepted']>0,
                     raw_majority=b['unfiltered'],accepted_majority=b['region'])
        rows.append(dict(episode_id=eid,edits={k:gross(masks_a[k],masks_b[k],gt) for k in masks_a},
                         old={k:counts(v,gt) for k,v in masks_a.items()},
                         fast={k:counts(v,gt) for k,v in masks_b.items()}))
    (s.OUT/'deep_crf_paired_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    paired={}
    for stage in rows[0]['edits']:
        paired[stage]={'edits':{k:sum(r['edits'][stage][k] for r in rows) for k in rows[0]['edits'][stage]}}
        for version in ['old','fast']:
            pooled={k:sum(r[version][stage][k] for r in rows) for k in rows[0][version][stage]}
            pooled['pooled_iou']=100*pooled['intersection']/pooled['union']
            paired[stage][version]=pooled
    examples=[]
    fixed=[('paco_fast600','dev_s1/paco_part/0/0'),('paco_fast600','dev_s1/paco_part/0/14'),('deep_fast100','dg18-0000')]
    lookup={(r['group'],r['episode_id']):r for r in seal['files']}
    for key in fixed:
        r=lookup[key];f,shape=load_fields(r)
        gt=s.nearest(s.load_truth(r['row']),s.COVER.shape)
        regions={'all_gt_or_bg':np.ones(s.COVER.shape,bool), 'raw_support':f['raw']>0,
                 'raw_majority':f['unfiltered'], 'accepted_majority':f['region'],
                 'accepted_minority':(f['accepted']>0)&~f['region'],
                 'accepted_minority_whole_fg':(f['accepted']>0)&~f['region']&f['guide'],
                 'raw_unanimous_coverage4':(f['raw']==4)&(s.COVER==4),
                 'accepted_unanimous_coverage4':(f['accepted']==4)&(s.COVER==4)}
        examples.append(dict(group=key[0],episode_id=key[1],counts={k:dict(target=int((m & gt).sum()),background=int((m & ~gt).sum())) for k,m in regions.items()},
                             original_size=[int(v) for v in shape],field_file=str(s.OUT/'reconstructed'/r['filename'])))
    result=dict(recurrence=recurrence, deep_crf_paired=paired, examples=examples,
                source_diagnostic_sha=s.sha(s.OUT/'diagnostic_report.json'),
                source_fields_seal_sha=s.sha(s.OUT/'reconstructed_seal.json'),
                code_sha=s.sha(__file__), seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,cpu_threads=1,
                note='All recurrence statistics condition on coverage. Deep union stages are explanatory sets, not new candidate predictions. LocalCRF changes source topology and downstream gates simultaneously.')
    s.write(s.OUT/'mechanisms.json',result)
    print(json.dumps(dict(seconds=result['seconds'],deep_crf_paired=paired,examples=examples)),flush=True)


if __name__=='__main__':
    main()
