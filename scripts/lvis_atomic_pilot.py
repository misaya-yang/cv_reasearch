#!/usr/bin/env python3
"""Three preregistered interventions on the fixed100, never on the other1300.

The candidate is reference-role-conditioned graph smoothing; uniform mass
attenuation and full FoRIS foreground-anchored normalization are controls.
All inference uses frozen source fields/features and query RGB, no query GT.
"""
import concurrent.futures
import fcntl
import json
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from lvis_atomic_study import ASSETS, DEFAULT_OUT, REPO, point, return_taps, sha, write

ROOT=DEFAULT_OUT/'pilot100'
ARMS=['foris.crf','mean','mean.graph_only','mean.role_graph','mean.uniform_graph','foris.fg_anchor.crf']


def prepare():
    source_config=json.loads((DEFAULT_OUT/'config.json').read_text())
    rows=json.loads((DEFAULT_OUT/'manifest.json').read_text());selected=set(source_config['pilot_ids'])
    rows=[r for r in rows if r['episode_id'] in selected];assert len(rows)==100
    config=dict(state='FROZEN_100_CASE_EXPLORATORY_INTERVENTIONS',n=100,arms=ARMS,
        original_pilot_config_sha256=sha(DEFAULT_OUT/'config.json'),
        diagnostic_design_n=1300,query_GT_in_inference=False,encoder_calls=0,
        candidate='mean.role_graph',
        role_graph='complete reference coverage-weighted FG/BG kernel densities; tau .07; W*=W*(1-|p_i-p_j|), no renormalization',
        uniform_control='same y/A; W_uniform=(sum W*/sum W)*W, identical total edge mass; lambda16',
        normalization_control='full original final FoRIS continuous score read using actual FG LSE min/max, then original CRF; other fields unchanged',
        parameter_search=False,metric='fixed CLI1024 primary; original reported independently',
        source_sha256={str(p):sha(p) for p in [Path(__file__),REPO/'src/ics/methods/reference_role_graph.py',
             REPO/'src/ics/methods/mean_control.py',ASSETS/'third_party/foris_official/models/foris.py',
             ASSETS/'third_party/foris_official/utils/refinement.py',REPO/'scripts/lvis_atomic_study.py']},
        interpretation='100-case initial paired test; baseline GT exposed; not official/full or independent confirmation',
        novelty='FG/BG kernel density is existing machinery; conditional edge regularization is a candidate, not established new relationship information')
    if (ROOT/'config.json').exists():
        assert json.loads((ROOT/'config.json').read_text())==config;return
    write(ROOT/'config.json',config);write(ROOT/'manifest.json',rows)
    (ROOT/'predictions').mkdir();(ROOT/'fields').mkdir()
    write(ROOT/'activity.json',dict(state='FROZEN',n=100))


def initialize():
    global NP,TORCH,F,IMAGE,MEAN,ROLE,CRF,BAND,CORE,TRANSFORM,REFINE,RENDER
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from ics.methods import mean_control,reference_role_graph
    from ics.m4_crf import install
    from run_m4_baselines import render
    torch.set_num_threads(2);torch.manual_seed(0)
    sys.path.insert(0,str(ASSETS/'third_party/foris_official'))
    with (ROOT/'crf_install.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        install(ASSETS/'third_party/crf_source',ASSETS/'runtime/macos/crf')
    from utils.refinement import init_crf,crf_refine
    from utils.data import build_transform
    CRF,BAND,CORE=init_crf(1024,'cpu');CRF.eval().requires_grad_(False)
    NP,TORCH,F,IMAGE,MEAN,ROLE,TRANSFORM,REFINE,RENDER=(np,torch,f,Image,mean_control,
            reference_role_graph,build_transform(1024),crf_refine,render)


def one(task):
    row,record=task;name=record['filename'];started=time.monotonic()
    assert record['pilot'] and record['feature_sha256']
    feature=DEFAULT_OUT/'pilot_features'/name;assert sha(feature)==record['feature_sha256']
    with NP.load(feature) as source:q,r,cov=(source[k].copy() for k in ('q','r','cov'))
    fields_file=DEFAULT_OUT/'fields'/name;assert sha(fields_file)==record['field_sha256']
    with NP.load(fields_file) as source:
        score,mean_field,fg=(source[k].copy() for k in ('score','mean','fg'))
    with TORCH.inference_mode(),return_taps(dict(mean=(MEAN.mean_control,['q','r','y','a','w']))) as tap:
        old,_=MEAN.predict(q,r,cov,score)
        assert NP.array_equal(old,mean_field)
        at=time.monotonic();m=tap['mean']
        result=ROLE.predict(m['q'],m['r'],cov,m['w'],m['a'],m['y'])
        new={'mean.role_graph':result['role_graph'],'mean.uniform_graph':result['uniform_graph']}
        info,p=result['info'],result['p']
        role_seconds=time.monotonic()-at
        masks={arm:__import__('ics.methods.rcg',fromlist=['mask_from_field']).mask_from_field(value)
               for arm,value in new.items()}
        parent=TORCH.from_numpy(fg);child=TORCH.from_numpy(score)
        # Only terminal source minmax changes. The original semantic/prefix
        # computations and the complete original CRF suffix remain intact.
        normalized=(child-parent.min())/(parent-parent.min()).max().clamp_min(1e-6)
        initial=F.interpolate(normalized[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5
        with IMAGE.open(ASSETS/row['query_path']) as opened:image=opened.convert('RGB')
        from ics.official_data import array_hash
        assert array_hash(NP.asarray(image))==row['query_rgb_hash']
        target=TRANSFORM(image)[None]
        at=time.monotonic();controlled=REFINE(CRF,BAND,CORE,target,initial)
        crf_seconds=time.monotonic()-at
        masks['foris.fg_anchor.crf']=controlled.numpy()
    pred=dict(original_hw=NP.asarray(row['query_size_hw']))
    baseline=DEFAULT_OUT/'predictions'/name;assert sha(baseline)==record['prediction_sha256']
    with NP.load(baseline) as saved:
        for arm in ['foris.crf','mean','mean.graph_only']:
            for frame in ['cli','original']:pred[frame+'/'+arm]=saved[frame+'/'+arm].copy()
    for arm,mask in masks.items():
        pred['cli/'+arm]=NP.packbits(mask)
        pred['original/'+arm]=NP.packbits(RENDER(mask,tuple(row['query_size_hw'])))
    destination=ROOT/'predictions'/name;NP.savez_compressed(destination,**pred)
    field=ROOT/'fields'/name;NP.savez_compressed(field,**new,role_probability=p,
                                              foreground_anchor_normalized_score=normalized.numpy())
    return dict(episode_id=row['episode_id'],filename=name,prediction_sha256=sha(destination),
                fields_sha256=sha(field),source_mean_field_bit_exact=True,encoder_calls=0,
                role_graph_seconds=role_seconds,normalization_crf_seconds=crf_seconds,
                seconds=time.monotonic()-started,diagnostics=info)


def infer():
    prepare();config=json.loads((ROOT/'config.json').read_text())
    for path,digest in config['source_sha256'].items():assert sha(path)==digest
    if (ROOT/'sealed.json').exists():print('Pilot already sealed; not restarting');return
    rows=json.loads((ROOT/'manifest.json').read_text())
    index={r['episode_id']:r for r in map(json.loads,(DEFAULT_OUT/'inference.jsonl').read_text().splitlines())}
    ledger=ROOT/'inference.jsonl';done={}
    if ledger.exists():
        for r in map(json.loads,ledger.read_text().splitlines()):
            assert sha(ROOT/'predictions'/r['filename'])==r['prediction_sha256'];done[r['episode_id']]=r
    start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),
          initializer=initialize) as pool,ledger.open('a',buffering=1) as log:
        pending=[pool.submit(one,(r,index[r['episode_id']])) for r in rows if r['episode_id'] not in done]
        for future in concurrent.futures.as_completed(pending):
            rec=future.result();log.write(json.dumps(rec)+'\n');done[rec['episode_id']]=rec
            write(ROOT/'activity.json',dict(state='INFERENCE',n=100,completed=len(done),
                  elapsed_seconds=time.monotonic()-start,worker_pids=[p.pid for p in pool._processes.values()]))
            if len(done)%10==0:print(json.dumps(dict(completed=len(done),n=100,seconds=time.monotonic()-start)),flush=True)
    assert len(done)==100
    write(ROOT/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=100,
        config_sha256=sha(ROOT/'config.json'),manifest_sha256=sha(ROOT/'manifest.json'),
        inference_index_sha256=sha(ledger),query_GT_in_inference=False,encoder_calls=0,
        new_query_photos=0,all_source_mean_fields_bit_exact=True))
    write(ROOT/'activity.json',dict(state='INFERENCE_COMPLETE',n=100,completed=100,worker_pids=[]))


def score():
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts,gross_edits
    torch.set_num_threads(2)
    seal=json.loads((ROOT/'sealed.json').read_text());assert seal['n']==100
    for name,key in [('config.json','config_sha256'),('manifest.json','manifest_sha256'),('inference.jsonl','inference_index_sha256')]:
        assert sha(ROOT/name)==seal[key]
    rows=json.loads((ROOT/'manifest.json').read_text());index={r['episode_id']:r for r in map(json.loads,(ROOT/'inference.jsonl').read_text().splitlines())}
    previous={r['episode_id']:r for r in map(json.loads,(DEFAULT_OUT/'episode_metrics.jsonl').read_text().splitlines())}
    results=[]
    for row in rows:
        rec=index[row['episode_id']];path=ROOT/'predictions'/rec['filename'];assert sha(path)==rec['prediction_sha256']
        with Image.open(row['query_mask_path']) as image:raw=(np.asarray(image.convert('L'))>0).astype(np.uint8)
        assert array_hash(raw)==row['query_mask_hash']
        result=dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],frames={})
        with np.load(path) as saved:
            for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                truth=f.interpolate(torch.from_numpy(raw)[None,None].float(),shape,mode='nearest')[0,0].numpy()>.5
                masks={a:np.unpackbits(saved[frame+'/'+a],count=int(np.prod(shape))).reshape(shape).astype(bool) for a in ARMS}
                iu={a:counts(p,truth) for a,p in masks.items()}
                for a in ARMS[:3]:assert iu[a]==previous[row['episode_id']]['frames'][frame]['iu'][a]
                edits={a:{b:gross_edits(p,masks[b],truth) for b in ['foris.crf','mean','mean.uniform_graph']}
                       for a,p in masks.items()}
                result['frames'][frame]=dict(iu=iu,edits=edits,truth_pixels=int(truth.sum()))
        results.append(result)
    (ROOT/'episode_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    report=dict(state='COMPLETE',n=100,encoder_calls=0,frames={},pilot_only=True,
                inference_sealed_before_candidate_labels=True,original_baseline_GT_previously_exposed=True,
                conditional_class_bootstrap='100 observed classes each have one query; within-class resampling degenerates; no generalization CI claim')
    for frame in ['cli','original']:
        flat=[dict(r,**r['frames'][frame]) for r in results]
        p=point(flat,ARMS);pairs={}
        for a in ARMS:
            for b in ['foris.crf','mean','mean.uniform_graph']:
                delta=[100*(r['iu'][a][0]/max(r['iu'][a][1],1)-r['iu'][b][0]/max(r['iu'][b][1],1)) for r in flat]
                pixels=np.sum([r['edits'][a][b] for r in flat],axis=0).tolist()
                pairs[a+' vs '+b]=dict(delta_pp=p[a]-p[b],cases_up=sum(x>1e-10 for x in delta),
                  cases_down=sum(x< -1e-10 for x in delta),cases_equal=sum(abs(x)<=1e-10 for x in delta),
                  edits=pixels,edit_order=['add_TP','add_FP','delete_TP','delete_FP'])
        report['frames'][frame]=dict(miou=p,pairs=pairs)
    write(ROOT/'report.json',report);write(ROOT/'activity.json',dict(state='COMPLETE',n=100,completed=100,worker_pids=[]))
    print(json.dumps(report['frames']),flush=True)


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ['infer','score']:raise SystemExit('Use infer or score')
    infer() if sys.argv[1]=='infer' else score()
