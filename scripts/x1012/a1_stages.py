"""A1 actual FoRIS prefix staircase, with unchanged source computation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import os
from pathlib import Path
import shutil
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='4'
import numpy as np
import torch
import lab
import plan_common as pc
OUT=lab.ASSETS/'a/x1012_A1'
ARMS=('fg','bg','vote','prior','penalty','pre','FoRIS')


def rows():
    return lab.subset(200,domains=('coco',))+lab.subset(300,domains=('pascal_part','paco_part','suim'))+lab.ext_subset(300,domains=('isic','isaid'))


def init():
    global HOST,ADAPTER,RUN,LOAD,TAPS
    pc.setup('cpu');HOST,ADAPTER=pc.host_cpu()
    from ics.foris import run_foris
    from ics.official_data import load_inputs
    from lvis_atomic_study import return_taps
    RUN,LOAD,TAPS=run_foris,load_inputs,return_taps


def infer(b):
    name=pc.key(b);record=OUT/'records'/(name+'.json')
    if record.exists():
        r=pc.read(record);assert pc.sha(r['prediction_path'])==r['prediction_sha256'];return r
    start=time.monotonic();ADAPTER.bind(b)
    functions={'stage2':(HOST._part2_stage2_contrastive_score,['sim_fg_hw']),
               'stage3':(HOST._part3_clustering,['vote_soft','prior']),
               'penalty':(HOST._semantic_disagreement_penalty,['penalty']),
               'delta':(HOST._semantic_cluster_reweight_map,['labels_t','delta_cluster'])}
    with pc.inference_only(b),TAPS(functions) as taps:
        reference,mask,query=LOAD(b['row'],lab.ASSETS)
        final,got,_,_=RUN(HOST,reference,mask,query)
        s2,s3=got['s2'],got['s3']; vote=taps['stage3']['vote_soft'];prior=taps['stage3']['prior']
        voted=s2+HOST.candidate_boost*(vote-.5)
        assert torch.equal(voted+HOST.seed_cluster_boost*(prior-.5),s3)
        penalty=taps['penalty']['penalty'].clamp_max(HOST.semantic_penalty_max)
        delta=taps['delta']['delta_cluster'][taps['delta']['labels_t']].reshape(64,64)
        assert torch.equal(s3-penalty+delta,got['score'])
        prefixes={'fg':taps['stage2']['sim_fg_hw'],'bg':s2,'vote':voted,'prior':s3,'penalty':s3-penalty,'pre':got['score']}
        predictions={a:pc.pack(HOST._binarize_response(v,target_hw=(1024,1024)).numpy()) for a,v in prefixes.items()}
        assert np.array_equal(predictions['pre'],pc.pack(got['pre'].numpy()))
        predictions['FoRIS']=pc.pack(final.numpy())
        assert np.array_equal(predictions['FoRIS'],pc.pack(lab.foris_cli(b).numpy())),'Source output changed'
    pred=OUT/'predictions'/(name+'.npz');field=OUT/'fields'/(name+'.npz')
    assert not pred.exists() and not field.exists(),'Preserve partial output'
    np.savez_compressed(pred,**predictions);np.savez_compressed(field,**{a:v.numpy() for a,v in prefixes.items()})
    r={'index':name,'episode_id':b['row']['episode_id'],'dataset':b['dataset'],'prediction_path':str(pred),
       'prediction_sha256':pc.sha(pred),'field_path':str(field),'field_sha256':pc.sha(field),
       'seconds':time.monotonic()-start,'new_encoder_calls':0,'query_GT_reads':0,'archive_bit_exact':True}
    pc.write(record,r);ADAPTER.memo.clear();return r


def produce():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('predictions','fields','records'): (OUT/name).mkdir(exist_ok=True)
    tasks=rows(); config=OUT/'config.json'
    if not config.exists():
        sources=(Path(__file__),Path(pc.__file__),Path(lab.__file__),lab.ROOT/'scripts/lvis_atomic_study.py',lab.ASSETS/'third_party/foris_official/models/foris.py')
        (OUT/'frozen').mkdir()
        for i,p in enumerate(sources):shutil.copyfile(p,OUT/'frozen'/f'{i}_{p.name}')
        pc.write(config,{'state':'FROZEN','n':len(tasks),'workers':1,'threads':4,'arms':ARMS,'new_encoder_calls':0,
                         'source_sha256':{str(p):pc.sha(p) for p in sources},'query_GT_in_inference':False})
        pc.write(OUT/'manifest.json',tasks)
    assert all(pc.sha(p)==h for p,h in pc.read(config)['source_sha256'].items())
    if (OUT/'sealed.json').exists():print('A1 already sealed',flush=True);return
    start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        futures=[pool.submit(infer,b) for b in tasks]
        for i,f in enumerate(as_completed(futures),1):
            records.append(f.result())
            if i<=3 or i%50==0:print({'completed':i,'total':len(tasks),'seconds':round(time.monotonic()-start,1)},flush=True)
    pc.write(OUT/'sealed.json',{'state':'ALL_PREFIX_PREDICTIONS_SEALED','n':len(tasks),'seconds':time.monotonic()-start,
                              'records':records,'manifest_sha256':pc.sha(OUT/'manifest.json'),'config_sha256':pc.sha(config),'new_encoder_calls':0})
    print({'state':'ALL_PREFIX_PREDICTIONS_SEALED','n':len(tasks)},flush=True)


def score():
    import plan_report as pr
    seal=pc.read(OUT/'sealed.json');assert seal['state']=='ALL_PREFIX_PREDICTIONS_SEALED'
    indexed={r['index']:r for r in seal['records']};records=[]
    for b in pc.read(OUT/'manifest.json'):
        r=indexed[pc.key(b)];assert pc.sha(r['prediction_path'])==r['prediction_sha256']
        with np.load(r['prediction_path'],allow_pickle=False) as z:
            predictions={a:pc.unpack(z[a]) for a in ARMS}
        records.append(pr.score_record(b,predictions))
    result=pr.report('A1',OUT,records,ARMS,['七个前缀均来自实际FoRIS原函数局部变量；原完整预测逐位复现。','不同前缀各自用原FoRIS minmax/bilinear/>.5读出，观察逐阶段关联，不冒称各步独立因果。','每一步只读缓存O24，CPU单进程4线程；全部预测先封存后读取query GT。'])
    steps=list(ARMS);effects=[]
    for d in ('coco','isic','suim'):
        for band in ('small','medium'):
            rs=[r for r in records if r['dataset']==d and ((r['metrics']['FoRIS']['target_fraction']<.02) if band=='small' else (.02<=r['metrics']['FoRIS']['target_fraction']<.2))]
            if not rs:continue
            gt=sum(r['metrics']['FoRIS']['GT'] for r in rs)
            for a,b in zip(steps,steps[1:]):
                effects.append({'dataset':d,'band':band,'from':a,'to':b,'FP_GT_change':sum(r['metrics'][b]['FP']-r['metrics'][a]['FP'] for r in rs)/max(gt,1),'FN_GT_change':sum(r['metrics'][b]['FN']-r['metrics'][a]['FN'] for r in rs)/max(gt,1),'n':len(rs)})
    pc.write(OUT/'borrow_step_effects.json',effects)
    path=lab.ROOT/'docs/research/x1012_tasks/A1.md'
    with path.open('a') as f:
        f.write('\n## 借用步骤候选：大小档FP/GT变化\n\n|域|档|步骤|FP/GT变化|FN/GT变化|n|\n|---|---|---|---:|---:|---:|\n')
        for x in sorted(effects,key=lambda v:v['FP_GT_change']):f.write(f"|{x['dataset']}|{x['band']}|{x['from']}→{x['to']}|{x['FP_GT_change']:+.4f}|{x['FN_GT_change']:+.4f}|{x['n']}|\n")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('produce','score'));a=p.parse_args();produce() if a.mode=='produce' else score()
