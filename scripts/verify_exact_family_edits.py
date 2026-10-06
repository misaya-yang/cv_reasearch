#!/usr/bin/env python3
"""Reconcile four edit categories with sealed exact-family finalist I/U."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_exact_family_selection import edit_attribution, recipe_lut, write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.source/'manifest.json').read_text());rows=manifest['rows'];arms=json.loads((a.source/'arms.json').read_text());L=len(arms)
    h=np.load(a.source/'patterns.npy',mmap_mode='r');native_index=next(i for i,x in enumerate(arms) if x['npz_key']=='native');on=(np.arange(1<<L)&(1<<native_index))!=0
    report=json.loads((a.run/'report.json').read_text());crossfold='selection' in report
    counts_path=a.run/('counts.npz' if crossfold else 'final_counts.npz')
    with np.load(counts_path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    recipes={'heldfold_recipe':None} if crossfold else {name:report['recipes'][int(name[6:])] for name in report['selected_global_DEV_recipes']}
    corrections={}
    for name,recipe in recipes.items():
        edits={k:np.zeros(len(rows),dtype=np.int64) for k in ['add_TP','add_FP','delete_TP','delete_FP']}
        selections=[(np.ones(len(rows),dtype=bool),recipe)] if not crossfold else [(np.array([str(r['fold'])==fold for r in rows]),v['selected']) for fold,v in report['selection'].items()]
        for ix,recipe in selections:
            lut=recipe_lut(recipe,L);sub=h[ix]
            expected=np.stack([sub[:,lut,1].sum(1,dtype=np.int64),sub[:,:,1].sum(1,dtype=np.int64)+sub[:,lut,0].sum(1,dtype=np.int64)],axis=1)
            if not np.array_equal(expected,arrays[name][ix]):raise ValueError('Saved actual I/U mismatch')
            for key,sel,t in [('add_TP',lut&~on,1),('add_FP',lut&~on,0),('delete_TP',~lut&on,1),('delete_FP',~lut&on,0)]:edits[key][ix]=sub[:,sel,t].sum(1,dtype=np.int64)
        if not np.array_equal(arrays[name][:,0]-arrays['native'][:,0],edits['add_TP']-edits['delete_TP']):raise ValueError('I edit identity failed')
        if not np.array_equal(arrays[name][:,1]-arrays['native'][:,1],edits['add_FP']-edits['delete_FP']):raise ValueError('U edit identity failed')
        corrections[name]=[dict(c=r['c'],**{k:int(v[i]) for k,v in edits.items()}) for i,r in enumerate(rows)]
    result=dict(n=len(rows),per_draw_IU_and_edit_identities_exact=True,macro=edit_attribution(rows,arrays,corrections),totals={name:{key:sum(r[key] for r in rec) for key in ['add_TP','add_FP','delete_TP','delete_FP']} for name,rec in corrections.items()},source='GT-labelled membership histogram with per-draw I/U parity to independently sealed actual finalist masks; pixel counts are descriptive, not causal')
    write(a.run/'edit_attribution.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
