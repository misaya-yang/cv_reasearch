"""Apply frozen reference-only R2 geometry to the selected R1 VP_UN field.

No refit, threshold search, new encoder, query GT, or dataset branch in inference.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import shutil
import time
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='2'
import numpy as np
import torch
import lab
import plan_common as pc
import plan_report as pr
import r2_within_patch as geometry

OUT=lab.ASSETS/'a/x1012_R3'
R1=lab.ASSETS/'a/x1012_R1'
R2=lab.ASSETS/'a/x1012_R2'
ARMS=('UN_identity','UN_pixel','UN_pixel_fixed1','UN_pixel_permuted')


def initialize():
    geometry.initialize()


def prepare():
    for name in ('predictions','records','frozen'):(OUT/name).mkdir(parents=True,exist_ok=True)
    if not (OUT/'config.json').exists():
        assert pc.read(R1/'completion.json')['candidate_rank'][0]=='VP_UN'
        assert pc.read(R2/'sealed.json')['n']==1360
        sources=[Path(__file__),Path(geometry.__file__),Path(pc.__file__),Path(pr.__file__),Path(lab.__file__),geometry.PIXEL_SOURCE]
        config={'state':'FROZEN_BEFORE_COMBINED_OUTPUTS','n':1360,'arms':ARMS,
                'selected_R1':'VP_UN','selection':'predeclared worst-domain development gap then equal mean',
                'R2_selection':'reuse each reference beta/alpha unchanged, no refit/reselection',
                'reference_count':1,'new_encoder_calls':0,'query_GT_in_inference':False,
                'source_sha256':{str(p):pc.sha(p) for p in sources},
                'parent_seal_sha256':{str(p):pc.sha(p) for p in [R1/'sealed.json',R2/'sealed.json']}}
        pc.write(OUT/'config.json',config);shutil.copyfile(R1/'manifest.json',OUT/'manifest.json')
        for i,p in enumerate(sources):shutil.copyfile(p,OUT/'frozen'/f'{i}_{p.name}')
    c=pc.read(OUT/'config.json');assert all(pc.sha(p)==h for p,h in c['source_sha256'].items())
    return pc.read(OUT/'manifest.json')


def produce_one(binding):
    name=pc.key(binding);record=OUT/'records'/f'{name}.json'
    if record.exists():
        r=pc.read(record);assert pc.sha(r['prediction_path'])==r['prediction_sha256'];return r
    start=time.monotonic()
    with pc.inference_only(binding):
        r1=pc.read(R1/'records'/f'{name}.json');r2=pc.read(R2/'records'/f'{name}.json')
        assert pc.sha(r1['field_path'])==r1['field_sha256']
        with np.load(r1['field_path'],allow_pickle=False) as z:field=z['VP_UN'].copy()
        with np.load(r1['prediction_path'],allow_pickle=False) as z:baseline=pc.unpack(z['VP_UN'])
        semantic=geometry.standardize_within(lab.up(torch.from_numpy(field)).numpy())
        counts=geometry.blocks(baseline).sum(1)
        descriptor=geometry.pixel.descriptor(geometry.pixel.rgb(binding['row'],'query'))
        residual=geometry.standardize_within(descriptor@np.asarray(r2['beta'],np.float32))
        residual_perm=geometry.standardize_within(descriptor@np.asarray(r2['beta_permuted'],np.float32))
        predictions={'UN_identity':geometry.rerank(semantic,residual,counts,0.),
                     'UN_pixel':geometry.rerank(semantic,residual,counts,r2['selection']['selected']),
                     'UN_pixel_fixed1':geometry.rerank(semantic,residual,counts,1.),
                     'UN_pixel_permuted':geometry.rerank(semantic,residual_perm,counts,r2['selection_permuted']['selected'])}
        assert np.array_equal(predictions['UN_identity'],baseline)
        assert all(np.array_equal(geometry.blocks(p).sum(1),counts) for p in predictions.values())
        target=OUT/'predictions'/f'{name}.npz';assert not target.exists()
        np.savez_compressed(target,**{a:pc.pack(p) for a,p in predictions.items()})
    r={'index':name,'dataset':binding['dataset'],'prediction_path':str(target),'prediction_sha256':pc.sha(target),
       'R1_field_sha256':r1['field_sha256'],'R2_reference_fit_record_sha256':pc.sha(R2/'records'/f'{name}.json'),
       'seconds':time.monotonic()-start,'reference_count':1,'new_encoder_calls':0,'inference_query_GT_reads':0,
       'all_patch_areas_preserved':True,'identity_xor':0,'alpha':r2['selection']['selected']}
    pc.write(record,r);return r


def produce(workers):
    rows=prepare();start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,r in enumerate(pool.map(produce_one,rows,chunksize=1),1):
            records.append(r)
            if i<=3 or i%100==0:print({'phase':'R3-produce','completed':i,'n':len(rows),'seconds':round(time.monotonic()-start,1)},flush=True)
    pc.write(OUT/'sealed.json',{'state':'ALL_NATURAL_OUTPUTS_SEALED','n':len(rows),'records':records,
                              'manifest_sha256':pc.sha(OUT/'manifest.json'),'config_sha256':pc.sha(OUT/'config.json'),
                              'wall_seconds':time.monotonic()-start,'case_seconds_sum':sum(r['seconds'] for r in records),
                              'workers':workers,'new_encoder_calls':0,'inference_query_GT_reads':0})


def score_one(job):
    b,r=job
    with np.load(r['prediction_path'],allow_pickle=False) as z:pred={a:pc.unpack(z[a]) for a in ARMS}
    result=pr.score_record(b,pred);gt,valid=pc.truth(b);baseline=pred['UN_identity']
    result['edits']={a:{'add_TP':int((p&~baseline&gt&valid).sum()),'add_FP':int((p&~baseline&~gt&valid).sum()),
                       'delete_TP':int((~p&baseline&gt&valid).sum()),'delete_FP':int((~p&baseline&~gt&valid).sum())} for a,p in pred.items()}
    return result


def score(workers):
    initialize();rows=prepare();seal=pc.read(OUT/'sealed.json');assert seal['n']==len(rows)
    assert pc.sha(OUT/'config.json')==seal['config_sha256'] and pc.sha(OUT/'manifest.json')==seal['manifest_sha256']
    receipts={r['index']:r for r in seal['records']}
    for r in receipts.values():assert pc.sha(r['prediction_path'])==r['prediction_sha256']
    start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,r in enumerate(pool.map(score_one,((b,receipts[pc.key(b)]) for b in rows),chunksize=1),1):
            records.append(r)
            if i<=3 or i%100==0:print({'phase':'R3-score','completed':i,'n':len(rows),'seconds':round(time.monotonic()-start,1)},flush=True)
    original={r['index']:r for r in pc.read(R1/'report.json')['records']}
    for r in records:
        o=original[r['index']];assert r['metrics']['UN_identity']==o['metrics']['VP_UN']
        for a in ('FoRIS','MEAN','VP_previous'):r['metrics'][a]=o['metrics'][a]
    arms=('FoRIS','MEAN','VP_previous')+ARMS
    report=pr.report('R3',OUT,records,arms,[
        'R1开发最差域差值选择VP_UN；R2每例reference-only beta/alpha原封不动组合，不再拟合或选强度。',
        '单参考、无新编码，全部组合自然掩码封存后评分。UN_identity逐位等于R1 VP_UN；每16块面积保持。',
        '固定alpha1与参考标签置换是完整对照；不使用FoRIS输出作为推理输入。'],extra={'production':{k:v for k,v in seal.items() if k!='records'}})
    contrasts={}
    for baseline in ('UN_identity','UN_pixel_permuted','UN_pixel_fixed1'):
        projected=[dict(r,metrics={'FoRIS':r['metrics'][baseline],'UN_pixel':r['metrics']['UN_pixel']}) for r in records]
        contrasts['pixel_minus_'+baseline]=pr.summary(projected,('FoRIS','UN_pixel'))
    pc.write(OUT/'mechanism_contrasts.json',contrasts)
    eligible=('UN_identity','UN_pixel','UN_pixel_fixed1')
    passing=[a for a in eligible if all(t[a]['delta_to_foris_pp']>=-.5 for t in report['summary'].values())]
    ranked=sorted(eligible,key=lambda a:(min(t[a]['delta_to_foris_pp'] for t in report['summary'].values()),np.mean([t[a]['delta_to_foris_pp'] for t in report['summary'].values()])),reverse=True)
    pc.write(OUT/'completion.json',{'state':'COMPLETE','n':len(rows),'candidate_rank':ranked,'ten_domain_gate_passing':passing,
                                  'score_seconds':time.monotonic()-start,'new_encoder_calls':0})
    print({'phase':'R3-complete','passing':passing,'rank':ranked},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('produce','score'))
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    produce(args.workers) if args.mode=='produce' else score(args.workers)
