#!/usr/bin/env python3
"""Fixed100 reference-conditioned calibration and equal-budget control.

Consumes already sealed role/coordinate fields. New inference opens query RGB
only for the unchanged CRF. All query labels are confined to the sealed scorer.
"""
import concurrent.futures
import fcntl
import json
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from lvis_atomic_study import ASSETS, DEFAULT_OUT, REPO, point, sha, write

SOURCE=DEFAULT_OUT/'pilot100'
ROOT=ASSETS/'a/lvis_reference_protected100_20261009'
ARMS=['foris.crf','mean','foris.fg_anchor.crf','calibration.reference.crf','calibration.uniform.crf']


def prepare():
    seal=json.loads((SOURCE/'sealed.json').read_text());assert seal['n']==100
    for file,key in [('manifest.json','manifest_sha256'),('config.json','config_sha256'),('inference.jsonl','inference_index_sha256')]:
        assert sha(SOURCE/file)==seal[key]
    config=dict(n=100,arms=ARMS,source_seal_sha256=sha(SOURCE/'sealed.json'),
        cohort='same previously frozen first10 per LVIS fold; no new cases',
        hypothesis='reference FG/BG tendency can preserve some true foreground during calibration',
        formula='bilinear1024 native/anchor/p first; low=min(native,anchor),gap=abs(native-anchor); role=low+p*gap',
        control='uniform=low+gamma*gap; gamma=sum(p*gap)/sum(gap), same pixel L1 upward correction budget',
        source_role='previous sealed reference coverage-weighted FG/BG kernel density tau .07; no re-encoding',
        prediction='>0.5 then original10-iterationCRF, original bool-to-image interpolation',
        source_sha256={str(p):sha(p) for p in [Path(__file__),REPO/'src/ics/methods/reference_protected_calibration.py',
             ASSETS/'third_party/foris_official/utils/refinement.py',REPO/'src/ics/m4_crf.py']},
        exposure='continued development on the exposed100; not independent confirmation',
        query_GT_in_inference=False,encoder_calls=0,parameter_search=False)
    if (ROOT/'config.json').exists():
        assert json.loads((ROOT/'config.json').read_text())==config;return
    write(ROOT/'config.json',config)
    write(ROOT/'manifest.json',json.loads((SOURCE/'manifest.json').read_text()))
    (ROOT/'predictions').mkdir();(ROOT/'fields').mkdir()
    write(ROOT/'activity.json',dict(state='FROZEN',n=100))


def initialize():
    global NP,TORCH,IMAGE,PROTECTED,RCG,RENDER,CRF,BAND,CORE,TRANSFORM,REFINE
    import numpy as np
    import torch
    from PIL import Image
    from ics.methods import reference_protected_calibration,rcg
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
    NP,TORCH,IMAGE,PROTECTED,RCG,RENDER,TRANSFORM,REFINE=(np,torch,Image,
        reference_protected_calibration,rcg,render,build_transform(1024),crf_refine)


def one(task):
    row,old,atomic=task;name=old['filename'];start=time.monotonic()
    field=SOURCE/'fields'/name;assert sha(field)==old['fields_sha256']
    with NP.load(field) as saved:
        anchor,p=(saved[k].copy() for k in ['foreground_anchor_normalized_score','role_probability'])
    original=DEFAULT_OUT/'fields'/name;assert sha(original)==atomic['field_sha256']
    with NP.load(original) as saved:native=RCG.minmax(saved['score'])
    at=time.monotonic();result=PROTECTED.predict(native,anchor,p)
    readout_seconds=time.monotonic()-at
    with TORCH.inference_mode():
        from ics.official_data import array_hash
        with IMAGE.open(ASSETS/row['query_path']) as opened:image=opened.convert('RGB')
        assert array_hash(NP.asarray(image))==row['query_rgb_hash']
        target=TRANSFORM(image)[None];masks={};times={}
        for arm,key in [('calibration.reference.crf','protected'),('calibration.uniform.crf','uniform')]:
            initial=TORCH.from_numpy(result[key]>.5)
            at=time.monotonic();masks[arm]=REFINE(CRF,BAND,CORE,target,initial).numpy()
            times[arm]=time.monotonic()-at
    packed=dict(original_hw=NP.asarray(row['query_size_hw']))
    old_prediction=SOURCE/'predictions'/name;assert sha(old_prediction)==old['prediction_sha256']
    with NP.load(old_prediction) as saved:
        for frame in ['cli','original']:
            for arm in ARMS[:3]:packed[frame+'/'+arm]=saved[frame+'/'+arm].copy()
    for arm,mask in masks.items():
        packed['cli/'+arm]=NP.packbits(mask)
        packed['original/'+arm]=NP.packbits(RENDER(mask,tuple(row['query_size_hw'])))
    destination=ROOT/'predictions'/name;NP.savez_compressed(destination,**packed)
    fields=ROOT/'fields'/name;NP.savez_compressed(fields,reference=result['protected'],uniform=result['uniform'])
    return dict(episode_id=row['episode_id'],filename=name,prediction_sha256=sha(destination),
        fields_sha256=sha(fields),encoder_calls=0,source_role_sha256=old['fields_sha256'],
        readout_seconds=readout_seconds,crf_seconds=times,diagnostics=result['info'],seconds=time.monotonic()-start)


def infer():
    prepare();config=json.loads((ROOT/'config.json').read_text())
    for file,digest in config['source_sha256'].items():assert sha(file)==digest
    if (ROOT/'sealed.json').exists():print('Already sealed; no restart');return
    old={r['episode_id']:r for r in map(json.loads,(SOURCE/'inference.jsonl').read_text().splitlines())}
    atomic={r['episode_id']:r for r in map(json.loads,(DEFAULT_OUT/'inference.jsonl').read_text().splitlines())}
    rows=json.loads((ROOT/'manifest.json').read_text());assert len(rows)==100
    ledger=ROOT/'inference.jsonl';done={}
    if ledger.exists():
        for r in map(json.loads,ledger.read_text().splitlines()):
            assert sha(ROOT/'predictions'/r['filename'])==r['prediction_sha256'];done[r['episode_id']]=r
    started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),
          initializer=initialize) as pool,ledger.open('a',buffering=1) as log:
        jobs=[pool.submit(one,(r,old[r['episode_id']],atomic[r['episode_id']])) for r in rows if r['episode_id'] not in done]
        for future in concurrent.futures.as_completed(jobs):
            rec=future.result();done[rec['episode_id']]=rec;log.write(json.dumps(rec)+'\n')
            write(ROOT/'activity.json',dict(state='INFERENCE',n=100,completed=len(done),
                controller_pid=__import__('os').getpid(),worker_pids=[p.pid for p in pool._processes.values()],
                elapsed_seconds=time.monotonic()-started))
            if len(done)%10==0:print(json.dumps(dict(n=len(done),total=100,seconds=time.monotonic()-started)),flush=True)
    assert len(done)==100
    write(ROOT/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=100,
         config_sha256=sha(ROOT/'config.json'),manifest_sha256=sha(ROOT/'manifest.json'),
         inference_index_sha256=sha(ledger),query_GT_in_inference=False,encoder_calls=0))
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
    for file,key in [('config.json','config_sha256'),('manifest.json','manifest_sha256'),('inference.jsonl','inference_index_sha256')]:
        assert sha(ROOT/file)==seal[key]
    rows=json.loads((ROOT/'manifest.json').read_text())
    index={r['episode_id']:r for r in map(json.loads,(ROOT/'inference.jsonl').read_text().splitlines())}
    source={r['episode_id']:r for r in map(json.loads,(SOURCE/'episode_metrics.jsonl').read_text().splitlines())}
    metrics=[];bases=['foris.crf','foris.fg_anchor.crf','calibration.uniform.crf']
    for row in rows:
        rec=index[row['episode_id']];path=ROOT/'predictions'/rec['filename'];assert sha(path)==rec['prediction_sha256']
        with Image.open(row['query_mask_path']) as image:raw=(np.asarray(image.convert('L'))>0).astype(np.uint8)
        assert array_hash(raw)==row['query_mask_hash']
        item=dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],frames={})
        with np.load(path) as saved:
            for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                truth=f.interpolate(torch.from_numpy(raw)[None,None].float(),shape,mode='nearest')[0,0].numpy()>.5
                masks={a:np.unpackbits(saved[frame+'/'+a],count=int(np.prod(shape))).reshape(shape).astype(bool) for a in ARMS}
                iu={a:counts(mask,truth) for a,mask in masks.items()}
                for a in ARMS[:3]:assert iu[a]==source[row['episode_id']]['frames'][frame]['iu'][a]
                edits={a:{b:gross_edits(mask,masks[b],truth) for b in bases} for a,mask in masks.items()}
                item['frames'][frame]=dict(iu=iu,edits=edits,truth_pixels=int(truth.sum()),
                    empty={a:bool(not mask.any()) for a,mask in masks.items()})
        metrics.append(item)
    (ROOT/'episode_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metrics))
    report=dict(state='COMPLETE',n=100,frames={},encoder_calls=0,development_exposed=True,
        claim_scope='same100 continued exploration; no unseen-class/photo or dataset generalization claim')
    for frame in ['cli','original']:
        flat=[dict(r,**r['frames'][frame]) for r in metrics];p=point(flat,ARMS);pairs={}
        for arm in ARMS:
            for base in bases:
                deltas=[100*(r['iu'][arm][0]/max(r['iu'][arm][1],1)-r['iu'][base][0]/max(r['iu'][base][1],1)) for r in flat]
                pixels=np.sum([r['edits'][arm][base] for r in flat],axis=0).tolist()
                pairs[arm+' vs '+base]=dict(delta_pp=p[arm]-p[base],cases_up=sum(d>1e-10 for d in deltas),
                    cases_down=sum(d< -1e-10 for d in deltas),cases_equal=sum(abs(d)<=1e-10 for d in deltas),
                    edits=pixels,edit_order=['add_TP','add_FP','delete_TP','delete_FP'])
        report['frames'][frame]=dict(miou=p,pairs=pairs,
            zero_intersection={a:sum(r['iu'][a][0]==0 for r in flat) for a in ARMS},
            empty_predictions={a:sum(r['empty'][a] for r in flat) for a in ARMS},
            per_fold={str(fold):point([r for r in flat if r['fold']==fold],ARMS) for fold in range(10)})
    write(ROOT/'report.json',report);write(ROOT/'activity.json',dict(state='COMPLETE',n=100,completed=100,worker_pids=[]))
    print(json.dumps({f:{k:v for k,v in x.items() if k in ['miou','zero_intersection','empty_predictions']} for f,x in report['frames'].items()}),flush=True)


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ['infer','score']:raise SystemExit('Use infer or score')
    infer() if sys.argv[1]=='infer' else score()
