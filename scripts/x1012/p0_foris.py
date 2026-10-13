"""P0-2 original FoRIS on cached new pairs; 20-case exact replay gate, no GT."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import os
from pathlib import Path
import shutil
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='4'
import numpy as np
import torch
import lab
import plan_common as pc

OUT=lab.ASSETS/'a/x1012_P0/foris'


def init():
    global HOST,ADAPTER,RUN,LOAD
    pc.setup('cpu')
    HOST,ADAPTER=pc.host_cpu()
    from ics.foris import run_foris
    from ics.official_data import load_inputs
    RUN,LOAD=run_foris,load_inputs


def infer(b,probe=False):
    start=time.monotonic(); name=pc.key(b); root=OUT/('probe' if probe else 'new')
    record=root/'records'/(name+'.json')
    if record.exists():
        r=pc.read(record)
        assert pc.sha(r['prediction_path'])==r['prediction_sha256'] and pc.sha(r['field_path'])==r['field_sha256']
        return r
    ADAPTER.bind(b)
    with pc.inference_only(b):
        reference,mask,query=LOAD(b['row'],lab.ASSETS)
        final,got,_,_=RUN(HOST,reference,mask,query)
        score=got['score'].cpu().numpy().copy()
        cli=final.cpu().numpy()
        original=torch.nn.functional.interpolate(final.float()[None,None],size=tuple(b['row']['query_size_hw']),mode='bilinear',align_corners=False)[0,0]>.5
        archive_xor=None; field_abs=None
        if probe:
            old=lab.foris_cli(b).numpy(); archive_xor=int(np.count_nonzero(cli!=old))
            sf=b['score64']
            with np.load(sf['path'],allow_pickle=False) as z:
                previous=np.asarray(z[sf['key']],dtype=np.float32).reshape(64,64)
            field_abs=float(np.max(np.abs(score-previous)))
    pred=root/'predictions'/(name+'.npz'); field=root/'fields'/(name+'.npz')
    assert not pred.exists() and not field.exists(),'Preserve incomplete output '+name
    np.savez_compressed(pred,**{'cli/foris.crf':pc.pack(cli),'original/foris.crf':pc.pack(original.numpy()),'original_hw':np.array(b['row']['query_size_hw'],np.int32)})
    np.savez_compressed(field,score=score)
    r={'index':name,'episode_id':b['row']['episode_id'],'dataset':b['dataset'],
       'prediction_path':str(pred),'prediction_sha256':pc.sha(pred),'field_path':str(field),'field_sha256':pc.sha(field),
       'archive_cli_xor':archive_xor,'archive_score_max_abs':field_abs,'seconds':time.monotonic()-start,
       'new_encoder_calls':ADAPTER.encoder_calls,'query_gt_reads':0}
    pc.write(record,r); ADAPTER.memo.clear(); return r


def produce(probe):
    OUT.mkdir(parents=True,exist_ok=True)
    for part in ('probe','new'):
        for name in ('predictions','fields','records'):
            (OUT/part/name).mkdir(parents=True,exist_ok=True)
    config_path=OUT/'config.json'
    if not config_path.exists():
        sources=(Path(__file__),Path(pc.__file__),Path(lab.__file__),lab.ASSETS/'third_party/foris_official/models/foris.py',lab.ROOT/'src/ics/foris.py')
        (OUT/'frozen').mkdir()
        for i,p in enumerate(sources):shutil.copyfile(p,OUT/'frozen'/f'{i}_{p.name}')
        pc.write(config_path,{'state':'FROZEN','workers':2,'threads_per_worker':4,'new_encoder_calls':0,
                             'source_sha256':{str(p):pc.sha(p) for p in sources},'query_GT_in_inference':False})
    config=pc.read(config_path)
    assert all(pc.sha(p)==h for p,h in config['source_sha256'].items()),'Execution source changed'
    if probe:
        rows=[b for d in lab.ALL_DOMAINS for b in [x for x in pc.seed_rows(3) if x['dataset']==d][:2]]
        assert len(rows)==20
    else:
        identity=pc.read(OUT/'probe.json')
        assert identity['n']==20 and identity['archive_mismatch_tasks']==0,'Stop at replay mismatch'
        rows=[b for part in ('explore','confirm') for b in pc.read(lab.ASSETS/f'a/x1012_P0/{part}.json') if b['cohort']=='x1012_P0_new_pair']
        pc.write(OUT/'manifest.json',rows)
    start=time.monotonic(); records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        futures={pool.submit(infer,b,probe):b for b in rows}
        for i,future in enumerate(as_completed(futures),1):
            r=future.result();records.append(r)
            if i<=20 or i%100==0:print({'completed':i,'total':len(rows),'dataset':r['dataset'],'seconds':round(time.monotonic()-start,1),'archive_xor':r['archive_cli_xor']},flush=True)
            if probe and r['archive_cli_xor']!=0:raise RuntimeError('P0-2 exact replay failed; preserved mismatch '+r['episode_id'])
    summary={'state':'PROBE_EXACT' if probe else 'NEW_PAIR_PREDICTIONS_SEALED','n':len(rows),
             'seconds':time.monotonic()-start,'archive_mismatch_tasks':sum(r['archive_cli_xor'] not in (0,None) for r in records),
             'maximum_probe_field_abs':max((r['archive_score_max_abs'] or 0) for r in records),'new_encoder_calls':0,'query_GT_reads':0}
    if probe:
        pc.write(OUT/'probe.json',summary)
    else:
        indexed={r['index']:r for r in records}
        for part in ('explore','confirm'):
            path=lab.ASSETS/f'a/x1012_P0/{part}.json'; entries=pc.read(path)
            shutil.copyfile(path,OUT/'frozen'/f'{part}.before_baseline.json')
            for b in entries:
                if pc.key(b) not in indexed:continue
                r=indexed[pc.key(b)]
                b['old_arms']['B']={'path':r['prediction_path'],'sha256':r['prediction_sha256'],'cli_key':'cli/foris.crf','original_key':'original/foris.crf'}
                b['score64']={'path':r['field_path'],'sha256':r['field_sha256'],'key':'score'}
            pc.write(path,entries)
        pc.write(OUT/'sealed.json',dict(summary,records=records,
                 manifest_sha256=pc.sha(OUT/'manifest.json'),config_sha256=pc.sha(config_path)))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('probe','produce'));a=parser.parse_args();produce(a.mode=='probe')
