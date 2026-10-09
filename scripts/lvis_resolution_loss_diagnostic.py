#!/usr/bin/env python3
"""Label-side geometry and graph-loss diagnosis of sealed LVIS1400.

No feature encoding, new prediction, coefficient choice or new mIoU. Pixel
regions are intersected before token pooling; coverage is not a score ceiling.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import multiprocessing
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
SOURCE = REPO.parent/'cv_data/a/lvis_atomic1400_20261009'
OUT = REPO.parent/'cv_data/a/lvis_resolution_loss1400_20261009'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def init():
    import torch
    torch.set_num_threads(2)


def pool(mask):
    return mask.reshape(64,16,64,16).sum((1,3),dtype=np.float64).reshape(-1)


def one(task):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from scipy.stats import rankdata
    from ics.official_data import array_hash
    from ics.methods.rcg import minmax
    row, rec = task
    pred = SOURCE/'predictions'/rec['filename']
    field = SOURCE/'fields'/rec['filename']
    assert sha(pred) == rec['prediction_sha256'] and sha(field) == rec['field_sha256']
    with Image.open(row['query_mask_path']) as im:
        raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
    assert array_hash(raw) == row['query_mask_hash']
    truth=F.interpolate(torch.from_numpy(raw)[None,None].float(),(1024,1024),mode='nearest')[0,0].numpy()>.5
    with np.load(pred) as z:
        masks={a:np.unpackbits(z['cli/'+a],count=1024**2).reshape(1024,1024).astype(bool)
            for a in ('diag.raw_nn','diag.apd_nn','foris.fg','foris.bg','foris.vote','foris.prior',
                      'foris.penalty','foris.pre','foris.crf','mean.rank','mean.graph_only','mean')}
    with np.load(field) as z:
        s=minmax(z['score']).reshape(-1)
        y=z['mean_unary'].reshape(-1).astype(np.float64)
        solved=z['mean'].reshape(-1).astype(np.float64)
        guide=z['mean_guide'].reshape(-1)
    fg=pool(truth); total=int(truth.sum()); cov=fg/256
    # Exact regions of existing pixel predictions, never a generated proposal.
    regions={
        'graph_deleted_TP': truth & masks['mean.rank'] & ~masks['mean'],
        'graph_added_TP': truth & ~masks['mean.rank'] & masks['mean'],
        'graph_added_FP': ~truth & ~masks['mean.rank'] & masks['mean'],
        'mean_final_FN': truth & ~masks['mean'],
    }
    guide_rank=(rankdata(guide,method='average')-.5)/4096
    details={}
    for name,region in regions.items():
        mass=pool(region); n=float(mass.sum())
        assert int(n)==int(region.sum())
        details[name]=dict(pixels=int(n),
            token_mass_s_gt_half=float(mass[s>.5].sum()),
            token_mass_y_gt_half=float(mass[y>.5].sum()),
            token_mass_z_gt_half=float(mass[solved>.5].sum()),
            token_mass_query_cov_ge_09=float(mass[cov>=.9].sum()),
            mass_weighted_s=None if not n else float(mass@s/n),
            mass_weighted_y=None if not n else float(mass@y/n),
            mass_weighted_z=None if not n else float(mass@solved/n),
            mass_weighted_guide_global_rank=None if not n else float(mass@guide_rank/n))
    intersections={a:int((truth&m).sum()) for a,m in masks.items()}
    return dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],
        pilot=rec['pilot'],truth_pixels=total,query_fraction=total/1024**2,
        query_token_geometry=dict(equivalent_full_tokens=total/256,n_overlap_tokens=int((fg>0).sum()),
            n_pure_ge_09=int((cov>=.9).sum()),max_coverage=float(cov.max()),
            fg_mass_in_pure_ge_09=float(fg[cov>=.9].sum()),
            fg_mass_in_mixed_below_09=float(fg[(cov>0)&(cov<.9)].sum())),
        intersections=intersections,regions=details,encoder_calls=0,new_predictions=0)


def summary(rows):
    result=dict(n=len(rows),truth_pixels=sum(r['truth_pixels'] for r in rows),
        query_without_pure_ge_09=sum(r['query_token_geometry']['n_pure_ge_09']==0 for r in rows),
        geometry_quantiles={},regions={},zero_intersections={},zero_mean_breakdown={})
    if not rows:return result
    for key in ('equivalent_full_tokens','n_overlap_tokens','n_pure_ge_09','max_coverage'):
        result['geometry_quantiles'][key]=np.quantile([r['query_token_geometry'][key] for r in rows],[0,.1,.5,.9,1]).tolist()
    result['fg_mass_mixed_fraction']=sum(r['query_token_geometry']['fg_mass_in_mixed_below_09'] for r in rows)/max(result['truth_pixels'],1)
    for arm in rows[0]['intersections']:
        result['zero_intersections'][arm]=sum(r['intersections'][arm]==0 for r in rows)
    zero=[r for r in rows if not r['intersections']['mean']]
    # Disjoint categories: rank input had a hit, source pre had a hit but rank
    # had none, earlier foreground prefix had a hit but source pre had none,
    # or that earliest FG prefix had none. These are diagnoses, not routing.
    for r in zero:
        i=r['intersections']
        label=('lost_at_graph_after_rank_hit' if i['mean.rank'] else
               'lost_at_rank_after_pre_hit' if i['foris.pre'] else
               'lost_in_frontend_after_FG_hit' if i['foris.fg'] else 'no_FG_prefix_hit')
        result['zero_mean_breakdown'][label]=result['zero_mean_breakdown'].get(label,0)+1
    assert sum(result['zero_mean_breakdown'].values())==len(zero)
    for region in rows[0]['regions']:
        rr=[r['regions'][region] for r in rows];n=sum(x['pixels'] for x in rr)
        values=dict(pixels=n)
        for key in rr[0]:
            if key.startswith('token_mass_'):
                values[key]=sum(x[key] for x in rr)
                values[key+'_fraction']=values[key]/max(n,1)
            elif key.startswith('mass_weighted_'):
                values[key]=None if not n else sum(x[key]*x['pixels'] for x in rr if x['pixels'])/n
        result['regions'][region]=values
    return result


def main():
    began=time.monotonic();seal=json.loads((SOURCE/'sealed.json').read_text())
    for filename,key in [('manifest.json','manifest_sha256'),('config.json','config_sha256'),('inference.jsonl','inference_index_sha256')]:
        assert sha(SOURCE/filename)==seal[key]
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    index={r['episode_id']:r for r in map(json.loads,(SOURCE/'inference.jsonl').read_text().splitlines())}
    prior={r['episode_id']:r for r in map(json.loads,(SOURCE/'episode_metrics.jsonl').read_text().splitlines())}
    assert len(manifest)==len(index)==len(prior)==1400
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=init) as executor:
        rows=list(executor.map(one,[(r,index[r['episode_id']]) for r in manifest],chunksize=10))
    for r in rows:
        previous=prior[r['episode_id']]['frames']['cli']
        assert r['truth_pixels']==previous['truth_pixels']
        assert all(r['intersections'][arm]==previous['iu'][arm][0] for arm in r['intersections'])
        for region,key in [('graph_deleted_TP',2),('graph_added_TP',0),('graph_added_FP',1)]:
            assert r['regions'][region]['pixels']==previous['edits']['mean']['mean.rank'][key]
        assert r['regions']['mean_final_FN']['pixels']==r['truth_pixels']-r['intersections']['mean']
    groups={'all1400':rows,'design1300':[r for r in rows if not r['pilot']],
        'exposed100':[r for r in rows if r['pilot']],
        'query_lt1pct':[r for r in rows if r['query_fraction']<.01],
        'query_no_pure_token':[r for r in rows if not r['query_token_geometry']['n_pure_ge_09']],
        'query_has_pure_token':[r for r in rows if r['query_token_geometry']['n_pure_ge_09']]}
    report=dict(state='COMPLETE_LABEL_DIAGNOSTIC_ONLY',n=1400,encoder_calls=0,new_prediction_masks=0,new_miou=0,
        source_seal_sha256=sha(SOURCE/'sealed.json'),source_episode_metrics_sha256=sha(SOURCE/'episode_metrics.jsonl'),
        script_sha256=sha(__file__),definition='Original 1024 nearest GT; actual pixel regions intersected before 16x16 pooling; geometry and mass-weighted saved-field values only',
        limitations='Token purity is not a representation or mIoU upper bound. Token centers can disagree with bilinear pixel decisions. GT groups are not inference rules. No candidate chosen from these labels.',
        groups={k:summary(v) for k,v in groups.items()},elapsed_seconds=time.monotonic()-began)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
