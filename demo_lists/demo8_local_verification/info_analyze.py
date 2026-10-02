"""Compare frozen evidence arms on exactly the same held development episodes.

Bootstrap connected support/query photo components, conditional on each fitted
model. Published scores, new-image generalization and information-theoretic
claims are deliberately outside this report's contract.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,'/root/demo4_cache/env')
import numpy as np


def run(root):
    root=Path(root);out=root/'information_readout_v1'
    d=json.loads((out/'report.json').read_text());split=json.loads((out/'split.json').read_text())
    test=np.array(split['test']);rows=[split['rows'][i] for i in test]
    classes=np.array([r['c'] for r in rows]);counts=[]
    for r in rows:
        with np.load(root/'dense_observation_v1'/f"f{r['fold']}_e{r['e']:04d}.npz") as a:counts.append(a['counts'])
    counts=np.stack(counts)
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for r in rows:
        a,b=find(r['query']),find(r['reference'])
        if a!=b:parent[max(a,b)]=min(a,b)
    _,groups=np.unique([find(r['query']) for r in rows],return_inverse=True);ng=int(groups.max()+1)
    rng=np.random.default_rng(2044)
    weights=rng.multinomial(ng,np.full(ng,1/ng),size=1500)[:,groups]
    def metric(a,w=None):
        if w is None:w=np.ones(len(rows))
        i=np.bincount(classes,weights=a[:,0]*w,minlength=80)
        u=np.bincount(classes,weights=a[:,1]*w,minlength=80)
        return 100*float(np.mean(i[u>0]/u[u>0]))
    predictions={}
    for k,r in d['results'].items():
        assert r['rows']==test.tolist(),'Never compare unaligned prediction rows'
        p=np.array(r['choices']);predictions[k]=[counts[np.arange(len(rows)),p[:,m],m] for m in [0,1]]
    controls={'native':[counts[:,0,0],counts[:,0,1]]}
    for res in [512,1024]:
        f=root/f'foris{res}_v1/report.json';b=json.loads(f.read_text())
        lookup={(r['fold'],r['e']):r for r in b['records']}
        values=[]
        for r in rows:
            t=lookup[(r['fold'],r['e'])]
            assert (r['query'],r['reference'],r['c'])==(t['query'],t['reference'],t['c'])
            values.append(t['counts'][f'official_crf{res}_fp32'])
        controls[f'FoRIS{res}_CRF']=np.array(values)
    # The best previous fitted scalar control can exceed FoRIS on this fold;
    # do not compare a fold-only result to an all-1200 aggregate or omit it.
    legacy=json.loads((root/'readouts_v1/report.json').read_text())
    for seed in [2037,2038]:
        pick=np.array(legacy['predictions'][f'scalar_task_head_seed{seed}'])[test]
        for m,name in enumerate(['replace','add_missing']):
            controls[f'legacy_scalar_s{seed}_{name}']=counts[np.arange(len(rows)),pick[:,m],m]
    effects=[]
    for seed in [2042,2043]:
        base=f"f{split['fold']}_R0_final_s{seed}"
        for arm in ['R1_middle','R2_pixels']:
            method=f"f{split['fold']}_{arm}_s{seed}"
            for m,name in enumerate(['replace','add_missing']):
                a,b=predictions[method][m],predictions[base][m]
                diffs=[metric(a,w)-metric(b,w) for w in weights]
                effects.append(dict(method=method,control=base,task=name,delta_pp=metric(a)-metric(b),
                     paired_photo_component_ci=np.quantile(diffs,[.025,.975]).tolist()))
    baseline={k:([metric(x) for x in a] if k=='native' else metric(a)) for k,a in controls.items()}
    result=dict(state='COMPLETED',episodes=len(rows),classes=len(np.unique(classes)),image_components=ng,
                fitted_seed_effects=effects,matched_cohort_controls=baseline,
                protocol='old held-fold development, class and both image roles excluded from training; fixed fitted-model bootstrap',
                precision='our same-BF16-autocast encoder with FP16 storage; FoRIS FP32 encoder+official CRF, practical control not pure evidence attribution',
                legacy_scope='previous scalar control shares class/image split; different seeds and smaller capacity, practical reference not pure causal attribution',
                interpretation='R0 improves legacy: usable final evidence/readout problem. R1/R2 exceed matched R0: usable evidence increment under this reader. Neither outcome alone proves irreversible information loss; negative is construct-specific.')
    (out/'paired_effects.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();run(args.root)
