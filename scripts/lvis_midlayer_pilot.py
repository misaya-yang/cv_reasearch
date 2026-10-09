#!/usr/bin/env python3
"""One fixed O12 role-contrast experiment on the original frozen LVIS100.

Missing O12 requires real MPS FP32 R/Q encoding. New raw features are stored
separately; original raw caches are never extended or replaced. Query labels
are read only by score after all100 predictions have been sealed.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from raw_feature_cache import RawFeatureCache,file_hash,tensor_hash,canonical_hash
from lvis_atomic_study import write,point
from lvis_tail_rank_pilot import verify_seal
import lvis_reference_discriminant_pilot as original

ASSETS=REPO.parent/'cv_data'
SOURCE=ASSETS/'a/lvis_reference_erasure100_20261009'
DISCRIM=ASSETS/'a/lvis_reference_discriminant100_20261009'
PILOT=ASSETS/'a/lvis_reference_fidelity100_20261009'
DEFAULT_ROOT=ASSETS/'a/lvis_midlayer12_100_20261009'
BASIS_ROOT=ASSETS/'a/representation_premise_20261008/branch_basis'
BASELINES=['foris.crf','mean','mean.graph_only','foris.fg_anchor.crf','mean.role_graph',
    'mean.uniform_graph','mean.context_rank_graph','mean.context_rank_matched',
    'mean.reference_fidelity','mean.reference_fidelity_matched','mean.raw_reference_contrast']
NEW_ARMS={'mean.midlayer12_reference_contrast':'mid12'}
ARMS=BASELINES+list(NEW_ARMS)


def index(path):return {r['episode_id']:r for r in map(json.loads,Path(path).read_text().splitlines())}


def paths():
    return [Path(__file__).resolve(),REPO/'scripts/lvis_reference_discriminant_pilot.py',
        REPO/'scripts/lvis_reference_erasure_pilot.py',REPO/'scripts/lvis_tail_rank_pilot.py',
        REPO/'scripts/raw_feature_cache.py',REPO/'scripts/cached_dino.py',REPO/'scripts/run_m4_baselines.py',
        REPO/'scripts/lvis_atomic_study.py',REPO/'src/ics/data.py',REPO/'src/ics/representations.py',
        REPO/'src/ics/methods/reference_erasure_guide.py',REPO/'src/ics/methods/rcg.py',
        REPO/'src/ics/native_basis.py',REPO/'src/ics/official_data.py',REPO/'src/ics/metrics.py']


def own_basis():
    import torch
    meta=json.loads((BASIS_ROOT/'complete.json').read_text())
    p=BASIS_ROOT/meta['bases']['O/12']['path']
    assert file_hash(p)==meta['bases']['O/12']['sha256']
    doc=torch.load(p,map_location='cpu',weights_only=True)
    u=doc['basis']
    assert doc['branch']=='O/12' and u.dtype==torch.float32 and u.shape==(1024,500) and torch.isfinite(u).all()
    return u,p,meta


def prepare(root):
    from lvis_reference_erasure_pilot import original_profile
    profile_path,profile=original_profile()
    parents=(SOURCE,DISCRIM,PILOT)
    for p in parents:verify_seal(p)
    original.validate(DISCRIM)
    rows=json.loads((SOURCE/'manifest.json').read_text());assert len(rows)==100
    assert rows==json.loads((PILOT/'manifest.json').read_text())
    tasks=json.loads((DISCRIM/'tasks.json').read_text());prior=index(PILOT/'inference.jsonl')
    for t in tasks:
        r=prior[t['episode_id']];p=PILOT/'predictions'/r['filename']
        assert file_hash(p)==r['prediction_sha256']
        t.update(baseline_prediction=str(p),baseline_prediction_sha256=r['prediction_sha256'])
    u,basis_path,basis_meta=own_basis();assert basis_meta['config']['weights_sha256']==profile['weights_sha256']
    sources={str(p):file_hash(p) for p in paths()}
    extra=dict(profile,acquisition=dict(name='fixed_O12_original_pair',branches=['O/12','O/24'],
        batch=2,layer_numbers='one-based explicit hooks, not n=12',observer_sha256=sources[str(REPO/'src/ics/representations.py')]))
    root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():
        old=json.loads((root/'config.json').read_text())
        if old['source_sha256']==sources:validate(root);return
        if (root/'inference.jsonl').exists() or (root/'raw_cache').exists():raise ValueError('Launched inputs/source are frozen')
    write(root/'manifest.json',rows);write(root/'tasks.json',tasks)
    cfg=dict(state='PREPARED_ONLY',n=100,arms=ARMS,source_sha256=sources,
        parent_seal_sha256={str(p):file_hash(p/'sealed.json') for p in parents},
        original_profile_path=str(profile_path),original_profile_sha256=file_hash(profile_path),
        original_profile=profile,new_raw_profile=extra,new_raw_profile_id=canonical_hash(extra),
        basis12_path=str(basis_path),basis12_sha256=file_hash(basis_path),
        basis12_complete_sha256=file_hash(BASIS_ROOT/'complete.json'),
        original_discriminant_config_sha256=file_hash(DISCRIM/'config.json'),
        original_discriminant_tasks_sha256=file_hash(DISCRIM/'tasks.json'),
        manifest_sha256=file_hash(root/'manifest.json'),tasks_sha256=file_hash(root/'tasks.json'),
        candidate='same O12 R/Q, complete coverage FG-BG FP32 means and unit delta; Q processed FP16 then FP32 unit',
        APD='original parent flag, own O12 basis; immutable native U24 for matched old control',
        control='bit-exact original O24 raw contrast guide/field/masks',
        readout='actual original H/A/s; rank.25 graph16 CG x0=y rtol1e-7 atol1e-9 max300; original renderer',
        producer='strict frozen checkpoint, original1024RGB preprocessing, MPS FP32 B2, only missing O12',
        cache='independent root/profile, no original cache merge',
        new_query_photos=0,new_candidates=1,layer_grid=False,extra_CRF=False,
        query_GT_in_inference=False,exposure='same exposed developmental100; no generalization claim')
    write(root/'config.json',cfg)
    for p in paths():
        dest=root/'source'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in parents:
        dest=root/'producer_snapshots'/p.name;dest.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','manifest.json','sealed.json','inference.jsonl'):shutil.copy2(p/name,dest/name)
    write(root/'activity.json',dict(state='PREPARED_ONLY',n=100,completed=0,encoder_calls=0))


def validate(root):
    cfg=json.loads((root/'config.json').read_text())
    for p,h in cfg['source_sha256'].items():assert file_hash(p)==h,p
    for p,h in cfg['parent_seal_sha256'].items():assert file_hash(Path(p)/'sealed.json')==h;verify_seal(Path(p))
    for name in ('manifest','tasks'):assert file_hash(root/(name+'.json'))==cfg[name+'_sha256']
    assert file_hash(cfg['basis12_path'])==cfg['basis12_sha256']
    assert file_hash(cfg['original_profile_path'])==cfg['original_profile_sha256']
    assert file_hash(DISCRIM/'config.json')==cfg['original_discriminant_config_sha256']
    assert file_hash(DISCRIM/'tasks.json')==cfg['original_discriminant_tasks_sha256']
    return cfg


def old_readout(task):
    import numpy as np
    from ics.methods.reference_erasure_guide import build_guides,parent_graph_readout
    from ics.methods.rcg import mask_from_field
    from run_m4_baselines import render
    q,r,cov,source,h,chain=original.inputs(task)
    raw=build_guides(q,r,r,r,cov,source['parent_guide'])['arrays']['guide_raw']
    assert np.array_equal(raw,source['guide_raw'])
    s=source['source_s'].reshape(-1);a=source['parent_a'].reshape(-1)
    parent,y,_=parent_graph_readout(source['parent_guide'],s,a,h)
    control,_,_=parent_graph_readout(raw,s,a,h)
    assert np.array_equal(parent.reshape(64,64),source['parent_mean'])
    assert np.array_equal(y.reshape(64,64),source['parent_unary'])
    assert np.array_equal(control.reshape(64,64),source['field_raw'])
    assert file_hash(task['erasure_predictions'])==task['erasure_prediction_sha256']
    with np.load(task['erasure_predictions']) as z:
        shape=tuple(z['original_hw'])
        for arm,field in [('mean',parent),('mean.raw_reference_contrast',control)]:
            mask=mask_from_field(field.reshape(64,64))
            assert np.array_equal(np.packbits(mask),z['cli/'+arm])
            assert np.array_equal(np.packbits(render(mask,shape)),z['original/'+arm])
    return cov,source,h,chain


def infer(root,limit=None):
    import numpy as np,torch,torch.nn.functional as F
    from cached_dino import CachedDINO
    from ics.data import TimmDINOv3
    from ics.official_data import load_inputs
    from ics.methods.reference_erasure_guide import process_reference,build_guides,parent_graph_readout
    from ics.methods.rcg import mask_from_field
    from lvis_reference_erasure_pilot import setup_data
    from run_m4_baselines import render
    cfg=validate(root)
    if (root/'sealed.json').exists():print('Already sealed100; no restart');return
    original.initialize(DISCRIM);transform=setup_data();basis,_,_=own_basis()
    old_cache=RawFeatureCache(Path(cfg['original_profile_path']).parent.parent,cfg['original_profile'])
    extra=RawFeatureCache(root/'raw_cache',cfg['new_raw_profile'])
    adapter=CachedDINO(extra,None,'mps',('O/12','O/24'))
    rows=json.loads((root/'manifest.json').read_text());tasks={t['episode_id']:t for t in json.loads((root/'tasks.json').read_text())}
    done=index(root/'inference.jsonl') if (root/'inference.jsonl').exists() else {}
    for rec in done.values():
        assert file_hash(root/'fields'/rec['filename'])==rec['fields_sha256']
        assert file_hash(root/'predictions'/rec['filename'])==rec['prediction_sha256']
    if limit and len(done)>=limit:
        print('Requested first case already completed; no additional inference');return
    for name in ('fields','predictions'):(root/name).mkdir(exist_ok=True)
    began=time.monotonic();total_calls=sum(r['encoder_calls'] for r in done.values())
    with (root/'inference.jsonl').open('a',buffering=1) as ledger:
        for row in rows:
            eid=row['episode_id']
            if eid in done:continue
            started=time.monotonic();task=tasks[eid];cov,source,h,chain=old_readout(task)
            reference,reference_mask,query=load_inputs(row,ASSETS)
            pair=torch.stack((transform(reference),transform(query)))
            mask=F.interpolate(reference_mask[None,None].float(),(1024,1024),mode='nearest')
            fresh_cov=F.interpolate(mask,(64,64),mode='area')[0,0].numpy()
            assert np.array_equal(fresh_cov,cov)
            old_raw=[old_cache.read(x.numpy(),('O/24',))['O/24'] for x in pair]
            q24=F.normalize(process_reference(old_raw[1],original.BASIS,task['apd_applied']).half().float(),dim=1)
            with np.load(task['feature']) as z:qparent=F.normalize(torch.from_numpy(z['q']).float(),dim=1)
            assert torch.equal(q24,qparent),'Original Q processing mismatch'
            missing=any(not (extra.folder/extra.key(x.numpy())/'entry.json').exists() for x in pair)
            if missing and adapter.encoder is None:
                assert torch.backends.mps.is_available()
                adapter.encoder=TimmDINOv3(str(ASSETS/'demo4_cache/models/dinov3-vitl16-timm')).to('mps').float().eval().requires_grad_(False)
            before=adapter.encoder_calls;at=time.monotonic()
            bank=adapter.raw(pair,('O/12','O/24'),dict(episode_id=eid,source_config_sha256=file_hash(root/'config.json')))
            calls=adapter.encoder_calls-before;encode_seconds=time.monotonic()-at;total_calls+=calls
            for j in range(2):assert np.array_equal(bank['O/24'][j].numpy(),old_raw[j]),'Synchronous O24 differs from immutable cache'
            r12=process_reference(bank['O/12'][0].numpy(),basis,task['apd_applied'])
            q12=F.normalize(process_reference(bank['O/12'][1].numpy(),basis,task['apd_applied']).half().float(),dim=1)
            result=build_guides(q12,r12,r12,r12,cov,source['parent_guide']);guide=result['arrays']['guide_raw']
            field,unary,solver=parent_graph_readout(guide,source['source_s'].reshape(-1),source['parent_a'].reshape(-1),h)
            arrays=dict(guide_mid12=guide,unary_mid12=unary,field_mid12=field,
                coverage=cov,foreground_mean_mid12=result['arrays']['foreground_mean_raw'],
                background_mean_mid12=result['arrays']['background_mean_raw'],contrast_mid12=result['arrays']['contrast_raw'],
                unit_contrast_mid12=result['arrays']['unit_contrast_raw'],
                source_s=source['source_s'],parent_a=source['parent_a'],parent_guide=source['parent_guide'],parent_unary=source['parent_unary'])
            assert file_hash(task['baseline_prediction'])==task['baseline_prediction_sha256']
            with np.load(task['baseline_prediction']) as z:
                packed={k:z[k].copy() for k in z.files};shape=tuple(z['original_hw'])
            with np.load(task['erasure_predictions']) as z:
                for frame in ('cli','original'):packed[frame+'/mean.raw_reference_contrast']=z[frame+'/mean.raw_reference_contrast'].copy()
            mask=mask_from_field(field.reshape(64,64));arm='mean.midlayer12_reference_contrast'
            packed['cli/'+arm]=np.packbits(mask);packed['original/'+arm]=np.packbits(render(mask,shape))
            name=task['filename'];np.savez_compressed(root/'fields'/name,**arrays);np.savez_compressed(root/'predictions'/name,**packed)
            raw_entries=[]
            for x in pair:
                key=extra.key(x.numpy());p=extra.folder/key/'entry.json';entry=json.loads(p.read_text())
                raw_entries.append(dict(key=key,entry_sha256=file_hash(p),payload_sha256=entry['file_sha256'],features=entry['features']))
            rec=dict(episode_id=eid,filename=name,fields_sha256=file_hash(root/'fields'/name),prediction_sha256=file_hash(root/'predictions'/name),
                raw_entries=raw_entries,original_reference_chain=chain,
                synchronous_O24_both_bit_exact=True,parent_mean_and_raw24_both_frames_bit_exact=True,
                encoder_calls=calls,encoder_batch=2 if calls else 0,encode_seconds=encode_seconds,
                seconds=time.monotonic()-started,solver=solver,guide_diagnostics=result['diagnostics']['guides']['raw'],
                q12_processed_tensor_sha256=tensor_hash(q12.numpy()),r12_processed_tensor_sha256=tensor_hash(r12.numpy()),
                controller_pid=os.getpid(),query_GT_in_inference=False)
            ledger.write(json.dumps(rec)+'\n');done[eid]=rec
            write(root/'activity.json',dict(state='INFERENCE',n=100,completed=len(done),encoder_calls=total_calls,
                controller_pid=os.getpid(),elapsed_seconds=time.monotonic()-began))
            if len(done)%10==0:print(json.dumps(dict(completed=len(done),encoder_calls=total_calls,seconds=time.monotonic()-began)),flush=True)
            if limit and len(done)>=limit:return
    assert len(done)==100;validate(root)
    write(root/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=100,config_sha256=file_hash(root/'config.json'),
        manifest_sha256=file_hash(root/'manifest.json'),inference_index_sha256=file_hash(root/'inference.jsonl'),
        tasks_sha256=file_hash(root/'tasks.json'),encoder_calls=total_calls,query_GT_in_inference=False))
    write(root/'activity.json',dict(state='INFERENCE_COMPLETE',n=100,completed=100,encoder_calls=total_calls,controller_pid=os.getpid()))


def score(root):
    import numpy as np,torch,torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts,gross_edits
    torch.set_num_threads(2);config=validate(root);verify_seal(root)
    prior=index(PILOT/'episode_metrics.jsonl');raw_prior=index(SOURCE/'episode_metrics.jsonl');ii=index(root/'inference.jsonl');records=[]
    for row in json.loads((root/'manifest.json').read_text()):
        rec=ii[row['episode_id']];path=root/'predictions'/rec['filename'];assert file_hash(path)==rec['prediction_sha256'] and file_hash(root/'fields'/rec['filename'])==rec['fields_sha256']
        with Image.open(row['query_mask_path']) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert array_hash(raw)==row['query_mask_hash'];item=dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],query_photo_id=row['query_photo_id'],frames={})
        with np.load(path) as z:
            for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                truth=f.interpolate(torch.from_numpy(raw)[None,None].float(),shape,mode='nearest')[0,0].numpy()>.5
                masks={arm:np.unpackbits(z[frame+'/'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool) for arm in ARMS};iu={arm:counts(mask,truth) for arm,mask in masks.items()}
                for arm in BASELINES:assert iu[arm]==(raw_prior if arm=='mean.raw_reference_contrast' else prior)[row['episode_id']]['frames'][frame]['iu'][arm]
                item['frames'][frame]=dict(iu=iu,edits={arm:{b:gross_edits(masks[arm],masks[b],truth) for b in ARMS} for arm in NEW_ARMS},truth_pixels=int(truth.sum()))
        records.append(item)
    (root/'episode_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));report=dict(state='COMPLETE',n=100,frames={},encoder_calls=sum(x['encoder_calls'] for x in ii.values()),exposure=config['exposure'],new_information_or_gain_presumed=False)
    for frame in ('cli','original'):
        flat=[dict(r,**r['frames'][frame]) for r in records];points=point(flat,ARMS);pairs={}
        for arm in NEW_ARMS:
            for b in ARMS:
                d=[100*(r['iu'][arm][0]/max(r['iu'][arm][1],1)-r['iu'][b][0]/max(r['iu'][b][1],1)) for r in flat]
                pairs[arm+' vs '+b]=dict(delta_pp=points[arm]-points[b],cases_up=sum(v>1e-10 for v in d),cases_down=sum(v< -1e-10 for v in d),cases_equal=sum(abs(v)<=1e-10 for v in d),
                    edits=np.sum([r['edits'][arm][b] for r in flat],axis=0).tolist(),edit_order=['add_TP','add_FP','delete_TP','delete_FP'])
        report['frames'][frame]=dict(miou=points,pairs=pairs)
    write(root/'report.json',report);write(root/'activity.json',dict(state='COMPLETE',controller_pid=os.getpid(),n=100,completed=100,encoder_calls=sum(x['encoder_calls'] for x in ii.values())));print(json.dumps({f:report['frames'][f]['miou'] for f in ('cli','original')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','first','infer','score','run']);p.add_argument('--out',type=Path,default=DEFAULT_ROOT);a=p.parse_args()
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='first':infer(a.out,limit=1)
    elif a.mode=='infer':infer(a.out)
    elif a.mode=='score':score(a.out)
    else:infer(a.out);score(a.out)


if __name__=='__main__':main()
