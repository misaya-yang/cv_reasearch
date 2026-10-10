#!/usr/bin/env python3
"""Check preserved source ordering, baseline identity, and outer-fold photo overlap."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ROOT=REPO/'evidence/local/research_20261005/pipeline_verified'
S=ROOT/'mean_fine_residual_transfer4000_v1'
C=ROOT/'exact_family_selection_v1/public4000_v3_crossfold'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[json.loads(x) for x in (S/'episodes.jsonl').read_text().splitlines()]
seal=json.loads((C/'sealed.json').read_text())
assert list(seal['predictions'])==[r['key'] for r in rows]
with np.load(S/'counts.npz') as s,np.load(C/'counts.npz') as c:
    for arm in ['native','rcg','mean.control','rcg64.control','fine.rcg16.control','fine.rcg64']:
        assert np.array_equal(s['iu:'+arm],c['frozen_subtoken4000_v1::'+arm])
    assert np.array_equal(s['iu:mean_fine_residual_transfer_v1'],c['mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1'])
selection=json.loads((C/'selection.json').read_text())
result={'state':'COMPLETE','n':len(rows),'ordered_draw_keys_match_crossfold_seal':True,'all_seven_ordered_IU_arrays_identical':True,'source_manifest_available_locally':False,'folds':{},'sources':{str(p):sha(p) for p in [S/'episodes.jsonl',S/'counts.npz',C/'counts.npz',C/'selection.json',C/'sealed.json',REPO/'scripts/run_exact_family_crossfold.py']}}
for fold in range(4):
    tr=[r for r in rows if r['fold']!=fold];te=[r for r in rows if r['fold']==fold]
    assert len(tr)==selection['folds'][str(fold)]['training_n']==3000
    assert len(te)==selection['folds'][str(fold)]['held_n']==1000
    ts={r[k] for r in tr for k in ('query','support')};hs={r[k] for r in te for k in ('query','support')}
    result['folds'][fold]={'training_episodes':len(tr),'held_episodes':len(te),'shared_unique_photos':len(ts&hs),'held_any_photo_in_training':sum(any(r[k] in ts for k in ('query','support')) for r in te),'held_query_photo_in_training':sum(r['query'] in ts for r in te),'held_support_photo_in_training':sum(r['support'] in ts for r in te)}
result['interpretation']='Class-fold heldout with source-library exposure and cross-fold photo overlap, not photograph-isolated confirmation. Shared photos are not same-target labels, and no query GT is read during mask construction.'
(HERE/'provenance_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
