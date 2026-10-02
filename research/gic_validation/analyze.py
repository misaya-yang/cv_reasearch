"""Paired whole-scene bootstrap on frozen real-pose outputs, including failures."""
import argparse,json,time
from pathlib import Path
import numpy as np
from pose_eval import auc,ARMS

def analyze(directory,repetitions=2000):
    directory=Path(directory);report=json.loads((directory/'report.json').read_text())
    rows=report['results'];scenes=sorted({r['scene'] for r in rows})
    values=np.array([[r['arms'][a]['error']['pose'] for a in ARMS] for r in rows])
    groups=[np.array([i for i,r in enumerate(rows) if r['scene']==s]) for s in scenes]
    rng=np.random.default_rng(2044)
    comparisons={}
    # All methods share each replicate; resample complete scenes, not points.
    bootstrap=np.empty((repetitions,len(ARMS),3))
    for b in range(repetitions):
        ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        for a in range(len(ARMS)):
            for j,t in enumerate([5,10,20]):bootstrap[b,a,j]=auc(values[ix,a],t)
    for a,arm in enumerate(ARMS):
        comparisons[arm]=dict(
            auc=report['summary'][arm],
            fallback_count=sum('fallback' in r['arms'][arm]['details'] or 'failure' in r['arms'][arm]['details'] for r in rows),
            cpu_seconds_median=float(np.median([r['arms'][arm]['seconds'] for r in rows])),
            cpu_seconds_p95=float(np.percentile([r['arms'][arm]['seconds'] for r in rows],95)),
            seconds_scope='CPU implementation under concurrent jobs; not exclusive performance comparison')
    contrasts={}
    for baseline in ['native_poselib','native_extra8','diagonal','shuffled','inverse_occupancy','one_per_cell']:
        ai=ARMS.index('gic');bi=ARMS.index(baseline)
        dd=values[:,ai]-values[:,bi]
        contrasts['gic_minus_'+baseline]=dict(
            auc_delta={f'AUC@{t}':report['summary']['gic'][f'AUC@{t}']-report['summary'][baseline][f'AUC@{t}'] for t in [5,10,20]},
            paired_scene_ci95={f'AUC@{t}':np.percentile(bootstrap[:,ai,j]-bootstrap[:,bi,j],[2.5,97.5]).tolist() for j,t in enumerate([5,10,20])},
            pose_improved=int((dd< -1e-6).sum()),pose_worse=int((dd>1e-6).sum()),
            regressions_over_1deg=int((dd>1).sum()),improvements_over_1deg=int((dd< -1).sum()),
            worst_regressions=[dict(key=rows[i]['key'],scene=rows[i]['scene'],delta_degrees=float(dd[i]),gic=float(values[i,ai]),baseline=float(values[i,bi])) for i in np.argsort(dd)[-5:][::-1]])
    rho=[r['arms']['gic']['details'].get('rho',0) for r in rows]
    result=dict(state='COMPLETED',pairs=len(rows),scenes=len(scenes),summary=comparisons,contrasts=contrasts,
        rho=dict(median=float(np.median(rho)),upper_boundary_fraction=float(np.mean(np.array(rho)>.949)),zero_fraction=float(np.mean(np.array(rho)==0))),
        fatal_failures=report['fatal_failures'],
        inference_warning='Only two MegaDepth scenes: scene bootstrap cannot establish broad outdoor-scene generalization' if len(scenes)<10 else 'Single matcher/seed/isotropic protocol; not RoMaV2 or full DMS comparison',
        bootstrap=dict(seed=2044,replicates=repetitions,unit='entire scene, shared replicates across methods'),
        scene_results={s:{a:{f'AUC@{t}':auc(values[ix,ai],t) for t in [5,10,20]} for ai,a in enumerate(ARMS)} for s,ix in zip(scenes,groups)})
    (directory/'paired_analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['state','pairs','scenes','rho','fatal_failures','inference_warning']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directories',nargs='+');args=ap.parse_args()
    for d in args.directories:
        while not (Path(d)/'report.json').exists():time.sleep(10)
        analyze(d)
