"""Necessary uniform-rejection controls and complete saved-mask reconstruction."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('control_support',ROOT/'candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
np=c.np;read,lines,sha,write,resize=c.read,c.lines,c.sha,c.write,c.resize
PILOT=c.PILOT;V=ROOT/'paco_remaining500';OUT=ROOT/'uniform_controls'


def input_records():
    pilot=read(PILOT/'manifest.json');p_source={r['episode_id']:r for r in lines(PILOT/'inference.jsonl')}
    diag={r['episode_id']:r for r in lines(c.DATA/'a/cache_mechanism_study_20261010/frontend_fields_episodes.jsonl')}
    cand={r['episode_id']:r for r in read(c.OUT/'sealed.json')['receipts']}
    raw={r['episode_id']:r for r in read(c.RAW/'sealed.json')['receipts']}
    records=[]
    for row in pilot:
        if row['dataset']!='deepglobe_road':continue
        eid=row['episode_id'];r=cand[eid];src=p_source[eid];d=diag[eid]
        records.append(dict(row=row,calibration=r['calibration'],ridge_prediction_path=str(c.OUT/'predictions'/r['filename']),
            ridge_prediction_sha256=r['prediction_sha256'],field_path=str(c.RAW/'fields'/raw[eid]['filename']),
            field_sha256=raw[eid]['sha256'],field_prefix='source_apd.ridge.',baseline_path=str(PILOT/'predictions'/src['filename']),
            baseline_sha256=src['prediction_sha256'],score_path=d['source_score_path'],score_sha256=d['source_score_sha256'],score_key='score'))
    index=read(V/'source_index.json')
    for row,r in zip(read(V/'manifest.json'),read(V/'sealed.json')['receipts']):
        eid=row['episode_id'];src=index[eid];s=src['source_score']
        if r['field_sha256']:
            field_path=V/'fields'/r['filename'];field_sha=r['field_sha256'];prefix=''
        else:
            rr=raw[eid];field_path=c.RAW/'fields'/rr['filename'];field_sha=rr['sha256'];prefix='source_apd.ridge.'
        records.append(dict(row=row,calibration=r['calibration'],ridge_prediction_path=str(V/'predictions'/r['filename']),
            ridge_prediction_sha256=r['prediction_sha256'],field_path=str(field_path),field_sha256=field_sha,field_prefix=prefix,
            baseline_path=src['baseline_path'],baseline_sha256=src['baseline_sha256'],score_path=s['path'],score_sha256=s['sha256'],score_key=s['score_key']))
    assert len(records)==700
    return records


def generate():
    OUT.mkdir(exist_ok=True);assert not (OUT/'sealed.json').exists()
    (OUT/'predictions').mkdir(exist_ok=True)
    inputs=input_records();write(OUT/'inputs.json',inputs)
    write(OUT/'config.json',dict(generator_sha256=sha(__file__),inputs_sha256=sha(OUT/'inputs.json'),
        controls={'uniform.full_bg':'h=-1: full allowed background perturbation, no reference needed',
                  'uniform.matched_global':'h=-mean_original(abs(h_global)): same per-image original-pixel L1 evidence budget',
                  'uniform.matched_local4':'h=-mean_original(abs(h_local4)): same per-image original-pixel L1 evidence budget'},
        readout='same g0+.5*h, strict>0, exact tie inherits FoRIS; no parameter sweep/query GT',
        purpose='Separate task-direction benefit from simply reducing uncertain foreground.'))
    (OUT/'frozen_generator.py').write_bytes(Path(__file__).read_bytes())
    receipts=[]
    for i,item in enumerate(inputs):
        row=item['row'];shape=tuple(row['query_size_hw']);cal=item['calibration']
        for name in ('field','baseline','score','ridge_prediction'):assert sha(item[name+'_path'])==item[name+'_sha256']
        with np.load(item['field_path'],allow_pickle=False) as z:
            fields={view:z[item['field_prefix']+view].copy() for view in ('global','local4')}
        with np.load(item['baseline_path'],allow_pickle=False) as z:
            baseline=np.unpackbits(z['original/foris.crf'],count=np.prod(shape)).reshape(shape).astype(bool)
        with np.load(item['score_path'],allow_pickle=False) as z:s=z[item['score_key']].reshape(64,64).astype(float)
        span=float(s.max()-s.min());norm=(s-s.min())/max(span,1e-6)
        confidence=np.abs(2*resize(norm,shape)-1) if span else np.zeros(shape)
        g0=(2*baseline.astype(float)-1)*confidence
        evidence={}
        for view,value in fields.items():
            h=np.clip((value-cal['threshold'])/cal['separation'],-1,1) if cal['separation']>np.finfo(float).eps else np.zeros_like(value)
            evidence[view]=resize(h,shape)
        with np.load(item['ridge_prediction_path'],allow_pickle=False) as z:
            for view,h in evidence.items():
                d=g0+.5*h;replay=np.packbits((d>0)|((d==0)&baseline))
                assert np.array_equal(replay,z['anchor.'+view]),(row['episode_id'],view,'reconstruction')
        amplitudes={'uniform.full_bg':1.,'uniform.matched_global':float(np.abs(evidence['global']).mean()),
                    'uniform.matched_local4':float(np.abs(evidence['local4']).mean())}
        predictions={}
        for arm,amplitude in amplitudes.items():
            d=g0-.5*amplitude;predictions[arm]=np.packbits((d>0)|((d==0)&baseline))
        filename=f'{i:06d}.npz';path=OUT/'predictions'/filename
        np.savez_compressed(path,original_hw=np.array(shape),**predictions)
        receipts.append(dict(episode_id=row['episode_id'],filename=filename,sha256=sha(path),amplitudes=amplitudes,
                             both_ridge_masks_rebuilt_exact=True))
    write(OUT/'sealed.json',dict(n=700,receipts=receipts,config_sha256=sha(OUT/'config.json'),query_GT_read=False,
        new_encoder_calls=0,raw_feature_reads=0,reconstructed_ridge_masks=1400))
    print(json.dumps(dict(state='SEALED',n=700,reconstructed_ridge_masks=1400)),flush=True)


def score():
    import sys
    sys.path.insert(0,str(PILOT/'frozen/src'));from ics.metrics import summarize
    inputs=read(OUT/'inputs.json');seal=read(OUT/'sealed.json')
    assert sha(OUT/'inputs.json')==read(OUT/'config.json')['inputs_sha256']
    assert sha(OUT/'config.json')==seal['config_sha256']
    old={r['episode_id']:r for r in lines(c.OUT/'scored_episodes.jsonl') if r['dataset']=='deepglobe_road'}
    old.update({r['episode_id']:dict(r,dataset='paco_part') for r in lines(V/'scored_episodes.jsonl')})
    rows=[]
    for item,rec in zip(inputs,seal['receipts']):
        row=item['row'];assert row['episode_id']==rec['episode_id'];out=old[row['episode_id']]
        with c.math.Image.open(row['query_mask_path']) as im:truth=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert c.math.array_sha(truth)==row['query_mask_hash'];shape=tuple(row['query_size_hw']);truth=resize(truth,shape,'nearest')>.5
        with np.load(item['baseline_path'],allow_pickle=False) as z:baseline=np.unpackbits(z['original/foris.crf'],count=truth.size).reshape(shape).astype(bool)
        p=OUT/'predictions'/rec['filename'];assert sha(p)==rec['sha256']
        with np.load(p,allow_pickle=False) as z:
            for arm in rec['amplitudes']:
                pred=np.unpackbits(z[arm],count=truth.size).reshape(shape).astype(bool)
                out['iu'][arm]=[int((pred&truth).sum()),int((pred|truth).sum())]
                out['edits'][arm]=[int((pred&~baseline&truth).sum()),int((pred&~baseline&~truth).sum()),
                    int((~pred&baseline&truth).sum()),int((~pred&baseline&~truth).sum())]
                assert out['edits'][arm][:2]==[0,0]
        for arm,edits in out['edits'].items():
            if arm not in out['iu']:continue
            if arm.startswith(('anchor.','uniform.')):
                i,u=out['iu']['foris.crf'];at,af,dt,df=edits
                assert out['iu'][arm]==[i+at-dt,u+af-df]
        rows.append(out)
    (OUT/'scored_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    reports={}
    for ds in ('deepglobe_road','paco_part'):
        expected=read(V/'config.json')['expected_classes'] if ds=='paco_part' else None
        report=summarize([r for r in rows if r['dataset']==ds],expected_classes=expected,
                         baselines=('foris.crf','uniform.full_bg','uniform.matched_global','uniform.matched_local4'),repetitions=10000,unit='query_photo')
        reports[ds]=report
    write(OUT/'report.json',dict(frame='original',groups=reports,source_seal_sha256=sha(OUT/'sealed.json')))
    for ds,report in reports.items():
        print(ds,{k:v for k,v in report['miou'].items() if k.startswith(('anchor.','uniform.')) or k=='foris.crf'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','score']);args=p.parse_args()
    start=time.monotonic();generate() if args.stage=='generate' else score()
    print(json.dumps(dict(seconds=time.monotonic()-start)),flush=True)
