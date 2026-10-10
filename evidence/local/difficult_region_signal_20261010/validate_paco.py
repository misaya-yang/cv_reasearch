"""Frozen reference-ridge rules on the remaining existing PACO500, pooled600."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_candidate',ROOT/'candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
np,torch,F=c.np,c.torch,c.F
read,lines,sha,write,resize=c.read,c.lines,c.sha,c.write,c.resize
DATA=c.DATA;PILOT=c.PILOT;SOURCE=DATA/'a/paco_role_competition600_20261010'
OUT=ROOT/'paco_remaining500'


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'config.json').exists(),'Already prepared'
    manifest=read(SOURCE/'manifest.json');assert len(manifest)==600
    pilot={r['episode_id']:r for r in read(PILOT/'manifest.json') if r['dataset']=='paco_part'}
    candidate_seal=read(c.OUT/'sealed.json')
    assert sha(ROOT/'candidate.py')==candidate_seal['generator_sha256']
    assert sha(c.SUPPORT)==read(c.RAW/'sealed.json')['script_sha256']
    profile=Path(read(PILOT/'config.json')['raw_profile'])
    catalogue_path=DATA/'a/pro_joint_correspondence_review_20261010/assets_availability.json'
    catalogue={r['episode_id']:r for r in read(catalogue_path)['datasets']['PACO600']['records']}
    index={};old={r['episode_id']:r for r in lines(SOURCE/'inference.jsonl')}
    scores=read(DATA/'a/paco_anchored_task600_20261010/source_score_index.json')
    means={}
    for source_root in sorted({r['source_root'] for r in scores.values()}):
        p=Path(source_root)
        records={r['episode_id']:r for r in lines(p/'inference.jsonl')}
        rows={r['episode_id']:r for r in read(p/'manifest.json')}
        for eid,r in records.items():
            means[eid]=dict(path=str(p/'predictions'/r['prediction_file']),sha256=r['prediction_sha256'],manifest=rows[eid])
    for row in manifest:
        eid=row['episode_id'];a=catalogue[eid]
        raw_keys=[a['roles']['reference_full']['metadata_candidates'][0],a['roles']['query_full']['metadata_candidates'][0]]+[v['key'] for v in a['corners']]
        requests=[]
        for key in raw_keys:
            info=read(profile.parent/key/'entry.json')
            assert info['profile_id']==profile.parent.name
            requests.append(dict(key=key,payload_sha256=info['file_sha256'],tensor_sha256=info['features']['O/24']['tensor_sha256'],input_sha256=info['input_tensor_hash']))
        rec=old[eid];pred=SOURCE/'predictions'/rec['filename'];assert sha(pred)==rec['prediction_sha256']
        mean=means[eid]
        for key in ('reference_rgb_hash','query_rgb_hash','reference_mask_hash','query_mask_hash','reference_crop','query_crop','loader_class_id'):
            assert row.get(key)==mean['manifest'].get(key),(eid,key)
        del mean['manifest']
        assert sha(mean['path'])==mean['sha256']
        assert sha(scores[eid]['path'])==scores[eid]['sha256']
        index[eid]=dict(raw=requests,baseline_path=str(pred),baseline_sha256=rec['prediction_sha256'],source_score=scores[eid],mean=mean)
    write(OUT/'manifest.json',manifest);write(OUT/'source_index.json',index)
    frozen=OUT/'frozen';frozen.mkdir(exist_ok=True)
    for name,path in [('validate_paco.py',Path(__file__)),('candidate.py',ROOT/'candidate.py'),('signal_math.py',c.SUPPORT)]:
        (frozen/name).write_bytes(path.read_bytes())
    write(OUT/'config.json',dict(manifest_sha256=sha(OUT/'manifest.json'),source_index_sha256=sha(OUT/'source_index.json'),
        generator_sha256=sha(__file__),candidate_sha256=sha(ROOT/'candidate.py'),signal_math_sha256=sha(c.SUPPORT),
        profile_path=str(profile),profile_sha256=sha(profile),basis_sha256=sha(DATA/'native_assets/positional_basis.pt'),
        pilot_episode_ids=sorted(pilot),validation_n=500,total_n=600,candidate_source_seal_sha256=sha(c.OUT/'sealed.json'),
        recipe='exact candidate_reference_v1 anchor.local4, with anchor.global as simpler matched view control; no selector/new parameters',
        exposure='Remaining500 not used for this signal/candidate design, but historical development scores/analyses were exposed; not untouched confirmation',
        expected_classes=read(SOURCE/'experiment.json')['expected_classes']))
    for directory in ('predictions','fields','records'):(OUT/directory).mkdir(exist_ok=True)
    print(json.dumps(dict(state='PREPARED',validation_n=500,pool_n=600)),flush=True)


def generate():
    cfg=read(OUT/'config.json');assert sha(__file__)==cfg['generator_sha256']
    assert sha(ROOT/'candidate.py')==cfg['candidate_sha256'] and sha(c.SUPPORT)==cfg['signal_math_sha256']
    assert sha(OUT/'manifest.json')==cfg['manifest_sha256'] and sha(OUT/'source_index.json')==cfg['source_index_sha256']
    assert not (OUT/'sealed.json').exists(),'Completed validation must not be overwritten'
    profile=Path(cfg['profile_path']);assert sha(profile)==cfg['profile_sha256']
    index=read(OUT/'source_index.json');manifest=read(OUT/'manifest.json')
    pilots={r['episode_id']:r for r in read(c.OUT/'sealed.json')['receipts']}
    pilot_ids=set(cfg['pilot_episode_ids'])
    sys.path[:0]=[str(PILOT/'frozen/src'),str(PILOT/'frozen/scripts'),str(DATA/'third_party/foris_official')]
    from ics.official_data import load_inputs
    from raw_feature_cache import canonical_hash,tensor_hash
    from utils.data import build_transform
    transform=build_transform(1024)
    basis_path=DATA/'native_assets/positional_basis.pt';assert sha(basis_path)==cfg['basis_sha256']
    basis=torch.load(basis_path,map_location='cpu',weights_only=True)['basis'].float();projection=torch.eye(1024)-basis@basis.T
    sentinel=cfg['pilot_episode_ids'][0];receipts=[]
    for i,row in enumerate(manifest):
        eid=row['episode_id'];name=f'{i:06d}.npz';dest=OUT/'predictions'/name;record_file=OUT/'records'/f'{i:06d}.json'
        if record_file.exists():
            receipt=read(record_file);assert sha(dest)==receipt['prediction_sha256'];receipts.append(receipt);continue
        started=time.monotonic();source=index[eid];predictions={};fields={}
        if eid in pilot_ids and eid!=sentinel:
            old=pilots[eid];path=c.OUT/'predictions'/old['filename'];assert sha(path)==old['prediction_sha256']
            with np.load(path,allow_pickle=False) as z:
                predictions={key:z[key].copy() for key in ('anchor.global','anchor.local4')}
            calibration=old['calibration'];apd=None;io_seconds=0.;origin='copied_sealed_pilot'
        else:
            reference,reference_mask,query=load_inputs(row,DATA);canonical=transform.transforms[0](query)
            images=[reference,query]+[canonical.crop(box) for box in ((0,0,512,512),(512,0,1024,512),(0,512,512,1024),(512,512,1024,1024))]
            arrays=[]
            for image,request in zip(images,source['raw']):
                inputs=transform(image).numpy();key=canonical_hash(dict(profile=profile.parent.name,input_tensor_hash=tensor_hash(inputs)))
                assert key==request['key'] and tensor_hash(inputs)==request['input_sha256']
                folder=profile.parent/key
                with (folder/'entry.lock').open('rb') as lock:
                    fcntl.flock(lock,fcntl.LOCK_SH);info=read(folder/'entry.json')
                    assert info['file_sha256']==request['payload_sha256'] and sha(folder/info['file'])==request['payload_sha256']
                    with np.load(folder/info['file'],allow_pickle=False) as z:raw=z['O/24'].copy()
                    assert raw.shape==(4096,1024) and raw.dtype==np.float32 and tensor_hash(raw)==request['tensor_sha256']
                    fcntl.flock(lock,fcntl.LOCK_UN)
                arrays.append(torch.from_numpy(raw))
            io_seconds=time.monotonic()-started
            r,q=(F.normalize(x,dim=1) for x in arrays[:2]);binary=resize(reference_mask.numpy(),(64,64),'nearest').ravel()>.5
            mu=q.mean(0);semantic=float(F.normalize(r[binary].mean(0),dim=0)@mu/(mu.norm()+1e-6)) if binary.any() else None
            apd=semantic is None or semantic<.8
            global_raw=F.interpolate(arrays[1].reshape(64,64,1024).permute(2,0,1)[None],(128,128),mode='bilinear',align_corners=False)[0].permute(1,2,0).reshape(-1,1024)
            qg=F.normalize(global_raw,dim=1)
            ql=torch.cat((torch.cat((arrays[2].reshape(64,64,1024),arrays[3].reshape(64,64,1024)),1),torch.cat((arrays[4].reshape(64,64,1024),arrays[5].reshape(64,64,1024)),1)),0).reshape(-1,1024)
            ql=F.normalize(ql,dim=1);del arrays,global_raw
            if apd:r,qg,ql=(F.normalize(x@projection.T,dim=1) for x in (r,qg,ql))
            coverage=c.math.mask(row,'reference').reshape(64,16,64,16).mean(axis=(1,3)).ravel()
            fits,_=c.math.reference_fit(r,coverage);coefficient,bias=fits['ridge']
            calibration=c.reference_threshold((r.double()@coefficient+bias).numpy(),coverage,c.quantile_ids(coverage))
            fields={v:(x.double()@coefficient+bias).numpy().reshape(128,128) for v,x in [('global',qg),('local4',ql)]}
            shape=tuple(row['query_size_hw']);assert sha(source['baseline_path'])==source['baseline_sha256']
            with np.load(source['baseline_path'],allow_pickle=False) as z:
                baseline=np.unpackbits(z['original/foris.crf'],count=np.prod(shape)).reshape(shape).astype(bool)
            score_source=source['source_score'];assert sha(score_source['path'])==score_source['sha256']
            with np.load(score_source['path'],allow_pickle=False) as z:s=z[score_source['score_key']].reshape(64,64).astype(float)
            span=float(s.max()-s.min());norm=(s-s.min())/max(span,1e-6)
            confidence=np.abs(2*resize(norm,shape)-1) if span else np.zeros(shape)
            g0=(2*baseline.astype(float)-1)*confidence
            assert np.array_equal((g0>0)|((g0==0)&baseline),baseline)
            for view,score in fields.items():
                scale=calibration['separation'];h=np.clip((score-calibration['threshold'])/scale,-1,1) if scale>np.finfo(float).eps else np.zeros_like(score)
                decision=g0+.5*resize(h,shape);predictions['anchor.'+view]=np.packbits((decision>0)|((decision==0)&baseline))
            origin='new_cached_inference'
            if eid==sentinel:
                old=pilots[eid]
                with np.load(c.OUT/'predictions'/old['filename'],allow_pickle=False) as z:
                    for key,value in predictions.items():assert np.array_equal(value,z[key]),'Frozen-rule sentinel changed'
                assert calibration==old['calibration'];origin='sentinel_exact_replay'
        np.savez_compressed(dest,original_hw=np.array(row['query_size_hw']),**predictions)
        field_sha=None
        if fields:
            f=OUT/'fields'/name;np.savez_compressed(f,**fields);field_sha=sha(f)
        receipt=dict(episode_id=eid,filename=name,prediction_sha256=sha(dest),field_sha256=field_sha,
            origin=origin,calibration=calibration,apd_applied=apd,input_and_raw_seconds=io_seconds,seconds=time.monotonic()-started)
        write(record_file,receipt);receipts.append(receipt)
        write(OUT/'activity.json',dict(completed=len(receipts),total=600,last=eid))
        if (i+1)%50==0:print(json.dumps(dict(completed=i+1,total=600)),flush=True)
    write(OUT/'sealed.json',dict(n=600,receipts=receipts,config_sha256=sha(OUT/'config.json'),
        query_GT_read=False,new_encoder_calls=0,new_raw_writes=0,sentinel_exact=True))


def score():
    cfg=read(OUT/'config.json');seal=read(OUT/'sealed.json');assert sha(OUT/'config.json')==seal['config_sha256']
    index=read(OUT/'source_index.json');manifest=read(OUT/'manifest.json');pilot=set(cfg['pilot_episode_ids'])
    sys.path.insert(0,str(PILOT/'frozen/src'));from ics.metrics import summarize
    rows=[]
    for row,rec in zip(manifest,seal['receipts']):
        assert row['episode_id']==rec['episode_id'];source=index[row['episode_id']]
        with c.math.Image.open(row['query_mask_path']) as im:truth=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert c.math.array_sha(truth)==row['query_mask_hash']
        shape=tuple(row['query_size_hw']);truth=resize(truth,shape,'nearest')>.5
        predictions={};assert sha(source['baseline_path'])==source['baseline_sha256']
        with np.load(source['baseline_path'],allow_pickle=False) as z:
            for arm in ('foris.crf','region.fast','task.competitive'):
                predictions[arm]=np.unpackbits(z['original/'+arm],count=truth.size).reshape(shape).astype(bool)
        mean=source['mean'];assert sha(mean['path'])==mean['sha256']
        with np.load(mean['path'],allow_pickle=False) as z:predictions['mean']=np.unpackbits(z['original/mean'],count=truth.size).reshape(shape).astype(bool)
        p=OUT/'predictions'/rec['filename'];assert sha(p)==rec['prediction_sha256']
        with np.load(p,allow_pickle=False) as z:
            for arm in ('anchor.global','anchor.local4'):predictions[arm]=np.unpackbits(z[arm],count=truth.size).reshape(shape).astype(bool)
        baseline=predictions['foris.crf']
        r=dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],query_photo_id=row['query_photo_id'],
            pilot=row['episode_id'] in pilot,iu={},edits={})
        for arm,pred in predictions.items():
            r['iu'][arm]=[int((pred&truth).sum()),int((pred|truth).sum())]
            r['edits'][arm]=[int((pred&~baseline&truth).sum()),int((pred&~baseline&~truth).sum()),int((~pred&baseline&truth).sum()),int((~pred&baseline&~truth).sum())]
        rows.append(r)
    (OUT/'scored_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    reports={}
    for name,subset in [('remaining500',[r for r in rows if not r['pilot']]),('pilot100',[r for r in rows if r['pilot']]),('all600',rows)]:
        report=summarize(subset,baselines=('foris.crf','mean','region.fast','task.competitive'),repetitions=10000,unit='query_photo',expected_classes=cfg['expected_classes'])
        report['observed_classes_miou']=summarize(subset,baselines=('foris.crf',),repetitions=1)['miou']
        reports[name]=report
    assert abs(reports['all600']['miou']['foris.crf']-38.4608684255599)<1e-5
    write(OUT/'report.json',dict(groups=reports,frame='original',exposure=cfg['exposure'],source_seal_sha256=sha(OUT/'sealed.json')))
    print(json.dumps({k:v['miou'] for k,v in reports.items()},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate','score']);args=p.parse_args()
    begin=time.monotonic();{'prepare':prepare,'generate':generate,'score':score}[args.stage]()
    print(json.dumps(dict(seconds=time.monotonic()-begin)),flush=True)
