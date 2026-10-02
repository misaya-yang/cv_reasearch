"""Paired development effects with image-connected episode bootstrap.

Conditional on the fitted models; training seed variation is shown separately.
Never interpret the per-image candidate oracle as a class-mIoU upper bound.
"""
import argparse,json
from pathlib import Path
import numpy as np


def run(root):
    root=Path(root);d=json.loads((root/'readouts_v1/report.json').read_text())
    split=json.loads((root/'readouts_v1/split_manifest.json').read_text());meta=split['rows']
    counts=[];added=[]
    for r in meta:
        with np.load(root/f"dense_observation_v1/f{r['fold']}_e{r['e']:04d}.npz") as a:
            counts.append(a['counts']);added.append(a['added_counts'])
    counts=np.array(counts);added=np.array(added);classes=np.array([r['c'] for r in meta])
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for r in meta:
        a,b=find(r['query']),find(r['reference'])
        if a!=b:parent[max(a,b)]=min(a,b)
    _,groups=np.unique([find(r['query']) for r in meta],return_inverse=True);ng=groups.max()+1
    chosen={k:np.array(v) for k,v in d['predictions'].items()}
    picked={k:counts[np.arange(len(meta)),p[:,mode],mode] for k,p in chosen.items() for mode in []}
    comparisons=[]
    for seed in [2037,2038]:
        comparisons.extend([(f'paired_crop_simple_stats_seed{seed}',f'scalar_task_head_seed{seed}'),
                            (f'paired_crop_dense_seed{seed}',f'global_dense_seed{seed}'),
                            (f'paired_crop_simple_stats_seed{seed}',f'global_simple_stats_seed{seed}'),
                            (f'scalar_task_head_seed{seed}',f'global_no_crop_stats_seed{seed}')])
    rng=np.random.default_rng(2041);weights=rng.multinomial(ng,np.full(ng,1/ng),size=1500)[:,groups]
    def metric(a,w=None):
        if w is None:w=np.ones(len(meta))
        i=np.bincount(classes,weights=a[:,0]*w,minlength=80);u=np.bincount(classes,weights=a[:,1]*w,minlength=80)
        return 100*np.mean(i[u>0]/u[u>0])
    results=[]
    for a,b in comparisons:
        for mode,name in enumerate(['replace','add_missing']):
            aa=counts[np.arange(len(meta)),chosen[a][:,mode],mode];bb=counts[np.arange(len(meta)),chosen[b][:,mode],mode]
            diffs=[metric(aa,w)-metric(bb,w) for w in weights]
            results.append({'method':a,'control':b,'task':name,'delta_pp':metric(aa)-metric(bb),
                            'paired_image_component_bootstrap_ci':np.quantile(diffs,[.025,.975]).tolist()})
    # Correct the diagnostic no-change rate: zero added pixels can occur at ANY
    # candidate index, not only at the explicit native index zero.
    corrected={}
    for k,p in chosen.items():
        a=added[np.arange(len(meta)),p[:,1]]
        corrected[k]=float(np.mean(a.sum(1)==0))
    out={'state':'COMPLETED','n':len(meta),'image_connected_groups':int(ng),
         'bootstrap':'paired support/query connected image components; fixed fitted models; no novel-test claim',
         'comparisons':results,'actual_unchanged_pixel_fraction':corrected,
         'correction':'Original report index==0 rate is a candidate-index statistic, not actual no-change; use corrected zero-added-pixel rate.',
         'decision':'No independent dense-map benefit over legacy-crop scalar task head established; do not develop a budget scheduler from this experiment.'}
    (root/'readouts_v1/paired_effects.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();run(args.root)
