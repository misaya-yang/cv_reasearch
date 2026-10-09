#!/usr/bin/env python3
"""One fixed100 full-reference ridge discriminant; no encoder/image path.

Preparation/probe are label-free and produce no candidate masks. Infer reads
the original sealed raw-cache keys, preserves original processing and MEAN H,
and adds one candidate. Score reads query GT only after all100 are sealed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys
import time
import concurrent.futures
import multiprocessing

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from raw_feature_cache import file_hash,tensor_hash,canonical_hash
from lvis_atomic_study import write,point

ASSETS=REPO.parent/'cv_data';SOURCE=ASSETS/'a/lvis_reference_erasure100_20261009'
ATOMIC=ASSETS/'a/lvis_atomic1400_20261009';PILOT=ATOMIC/'pilot100'
DEFAULT_ROOT=ASSETS/'a/lvis_reference_discriminant100_20261009'
BASELINES=['foris.crf','mean','foris.fg_anchor.crf','mean.raw_reference_contrast']
CANDIDATE='mean.reference_discriminant';ARMS=BASELINES+[CANDIDATE]


def index(path):return {r['episode_id']:r for r in map(json.loads,Path(path).read_text().splitlines())}


def verify_seal(root):
    seal=json.loads((root/'sealed.json').read_text())
    assert seal['state']=='ALL_PREDICTIONS_SEALED' and seal['n']==100
    for name,key in [('config.json','config_sha256'),('manifest.json','manifest_sha256'),('inference.jsonl','inference_index_sha256')]:
        assert file_hash(root/name)==seal[key],str(root/name)
    return seal


@contextmanager
def cache_entry(profile_path,profile,key):
    if not re.fullmatch('[0-9a-f]{64}',key):raise ValueError('Invalid sealed raw-cache key')
    folder=Path(profile_path).parent/key
    with (folder/'entry.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        try:
            info=json.loads((folder/'entry.json').read_text())
            pid=canonical_hash(profile)
            assert info['profile_id']==pid and Path(profile_path).parent.name==pid
            assert canonical_hash(dict(profile=pid,input_tensor_hash=info['input_tensor_hash']))==key
            assert info['file']==info['file_sha256']+'.npz' and Path(info['file']).name==info['file']
            yield folder,info
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def read_raw(task,config):
    import numpy as np
    path=Path(config['raw_profile_path'])
    assert file_hash(path)==config['raw_profile_sha256']
    assert json.loads(path.read_text())==config['raw_profile']
    with cache_entry(path,config['raw_profile'],task['raw_cache_key']) as (folder,info):
        assert file_hash(folder/'entry.json')==task['raw_entry_sha256']
        assert info['file_sha256']==task['raw_payload_sha256']
        assert file_hash(folder/info['file'])==info['file_sha256']
        with np.load(folder/info['file'],allow_pickle=False) as z:raw=z['O/24'].copy()
        receipt=info['features']['O/24'];digest=tensor_hash(raw)
        assert raw.dtype==np.float32 and np.isfinite(raw).all()
        assert list(raw.shape)==receipt['shape'] and str(raw.dtype)==receipt['dtype']
        assert digest==receipt['tensor_sha256']==task['erasure_raw_FP32_tensor_sha256']
    return raw,dict(profile_id=canonical_hash(config['raw_profile']),profile_sha256=config['raw_profile_sha256'],
        cache_key=task['raw_cache_key'],entry_sha256=task['raw_entry_sha256'],payload_sha256=info['file_sha256'],
        input_tensor_hash=info['input_tensor_hash'],raw_O24_tensor_sha256=digest,
        sealed_key_from_erasure_original_input=True,images_decoded=0,encoder_calls=0)


def paths():
    return [Path(__file__).resolve(),REPO/'src/ics/methods/reference_discriminant.py',
        REPO/'src/ics/methods/reference_erasure_guide.py',REPO/'src/ics/methods/rcg.py',REPO/'src/ics/native_basis.py',
        REPO/'src/ics/metrics.py',REPO/'src/ics/official_data.py',REPO/'scripts/lvis_atomic_study.py',
        REPO/'scripts/run_m4_baselines.py',REPO/'scripts/raw_feature_cache.py']


def prepare(root):
    verify_seal(SOURCE);verify_seal(PILOT)
    source_config=json.loads((SOURCE/'config.json').read_text())
    rows=json.loads((SOURCE/'manifest.json').read_text());assert rows==json.loads((PILOT/'manifest.json').read_text())
    source=index(SOURCE/'inference.jsonl');pilot=index(PILOT/'inference.jsonl');atomic=index(ATOMIC/'inference.jsonl')
    profile_path=Path(source_config['original_raw_profile_path']);profile=source_config['original_raw_profile']
    assert file_hash(profile_path)==source_config['original_raw_profile_sha256'] and json.loads(profile_path.read_text())==profile
    tasks=[]
    for row in rows:
        eid=row['episode_id'];s=source[eid];p=pilot[eid];a=atomic[eid];name=s['filename']
        field=SOURCE/'fields'/name;epred=SOURCE/'predictions'/name;ppred=PILOT/'predictions'/p['filename'];feature=ATOMIC/'pilot_features'/a['filename']
        assert file_hash(field)==s['fields_sha256'] and file_hash(epred)==s['prediction_sha256']
        assert file_hash(ppred)==p['prediction_sha256'] and file_hash(feature)==a['feature_sha256']
        with cache_entry(profile_path,profile,s['original_reference_cache_key']) as (folder,entry):
            task=dict(episode_id=eid,filename=name,source_field=str(field),source_field_sha256=s['fields_sha256'],
                erasure_predictions=str(epred),erasure_prediction_sha256=s['prediction_sha256'],
                old_predictions=str(ppred),old_prediction_sha256=p['prediction_sha256'],
                feature=str(feature),feature_sha256=a['feature_sha256'],raw_cache_key=s['original_reference_cache_key'],
                raw_entry_sha256=file_hash(folder/'entry.json'),raw_payload_sha256=entry['file_sha256'],
                erasure_raw_FP32_tensor_sha256=s['original_reference_raw_FP32_tensor_sha256'],
                erasure_processed_FP32_tensor_sha256=s['processing_hashes']['full_processed_FP32'],
                erasure_parent_q_FP32_tensor_sha256=s['processing_hashes']['parent_q_FP32'],
                erasure_record_sha256=canonical_hash(s),apd_applied=s['apd_applied'])
        tasks.append(task)
    assert len(tasks)==100
    sources={str(p):file_hash(p) for p in paths()};root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():
        old=json.loads((root/'config.json').read_text())
        if old['source_sha256']==sources:validate(root);return old
        if (root/'inference.jsonl').exists() or (root/'sealed.json').exists():raise ValueError('Launched source is frozen')
    write(root/'manifest.json',rows);write(root/'tasks.json',tasks)
    config=dict(state='PREPARED_ONLY',n=100,arms=ARMS,source_sha256=sources,
        producer_seal_sha256={str(p):file_hash(p/'sealed.json') for p in (SOURCE,PILOT)},
        manifest_sha256=file_hash(root/'manifest.json'),tasks_sha256=file_hash(root/'tasks.json'),
        raw_profile_path=str(profile_path),raw_profile_sha256=file_hash(profile_path),raw_profile=profile,
        basis_path=source_config['native_basis_path'],basis_sha256=source_config['native_basis_sha256'],
        reference='direct sealed original O24 cache key; profile/entry/payload/tensor verified; original APD processing',
        query='original cached FP16 q to FP32 F.normalize; erasure parent_q tensor hash exact',
        means='same original erasure FP32 complete-cov FG/BG weighted means/delta',
        covariance='.5*(weighted within-FG covariance + weighted within-BG covariance); FP64 around original FP32 means',
        ridge='trace(Sigma)/C, C=1024; no epsilon event selection',
        direction='solve FP64 (Sigma+ridge I)v=delta.double; unit FP64 then FP32 query dot',
        readout='unchanged actual parent H/A/s, alpha.25 lambda_graph16 locked rank/CG and original rendering',
        controls='four sealed masks reused, raw FG-BG source guide/field/masks replay bit exact',
        new_candidates=1,encoder_calls=0,new_image_decodes=0,query_GT_in_inference=False,
        parameter_search=False,extra_CRF=False,tail_or_omega=False,
        exposure='same exposed frozen100; development only',claim='classical regularized reference discriminant, not new relation or optimal-LDA claim')
    write(root/'config.json',config)
    for p in paths():
        dest=root/'source'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name,parent in [('erasure',SOURCE),('old_pilot',PILOT)]:
        dest=root/'producer_snapshots'/name;dest.mkdir(parents=True,exist_ok=True)
        for filename in ('config.json','manifest.json','sealed.json','inference.jsonl'):shutil.copy2(parent/filename,dest/filename)
        if (parent/'source').exists():shutil.copytree(parent/'source',dest/'source',dirs_exist_ok=True)
    write(root/'activity.json',dict(state='PREPARED_ONLY',n=100,completed=0,encoder_calls=0))
    return config


def validate(root):
    config=json.loads((root/'config.json').read_text())
    for p,d in config['source_sha256'].items():assert file_hash(p)==d,p
    for p,d in config['producer_seal_sha256'].items():assert file_hash(Path(p)/'sealed.json')==d;verify_seal(Path(p))
    for name in ('manifest','tasks'):assert file_hash(root/(name+'.json'))==config[name+'_sha256']
    assert file_hash(config['basis_path'])==config['basis_sha256']
    assert file_hash(config['raw_profile_path'])==config['raw_profile_sha256']
    return config


def initialize(root):
    global CFG,BASIS
    import torch
    from ics.native_basis import load_native_basis
    torch.set_num_threads(2)
    CFG=validate(Path(root));BASIS,_=load_native_basis(CFG['basis_path'])


def inputs(task):
    import numpy as np,torch,torch.nn.functional as f
    from scipy import sparse
    from ics.methods.reference_erasure_guide import process_reference
    raw,chain=read_raw(task,CFG)
    full=process_reference(raw,BASIS,task['apd_applied'])
    assert tensor_hash(full.numpy())==task['erasure_processed_FP32_tensor_sha256']
    assert file_hash(task['feature'])==task['feature_sha256']
    assert file_hash(task['source_field'])==task['source_field_sha256']
    with np.load(task['feature']) as z:q_half=z['q'].copy();cov=z['cov'].copy();r_half=z['r'].copy()
    assert q_half.dtype==np.float16 and np.array_equal(full.half().numpy(),r_half)
    q=f.normalize(torch.as_tensor(q_half).float(),dim=1)
    assert tensor_hash(q.numpy())==task['erasure_parent_q_FP32_tensor_sha256']
    with np.load(task['source_field']) as z:
        source={k:z[k].copy() for k in ('coverage','source_s','parent_guide','parent_unary','parent_a','parent_mean',
            'guide_raw','field_raw','foreground_mean_raw','background_mean_raw','contrast_raw')}
        h=sparse.csr_matrix((z['parent_H_data'].copy(),z['parent_H_indices'].copy(),z['parent_H_indptr'].copy()),shape=tuple(z['parent_H_shape']))
    assert np.array_equal(cov,source['coverage']) and full.shape==(4096,1024)
    return q,full,cov,source,h,chain


def guides_and_replay(task):
    import numpy as np
    from ics.methods.reference_discriminant import predict
    from ics.methods.reference_erasure_guide import parent_graph_readout
    from ics.methods.rcg import mask_from_field
    from run_m4_baselines import render
    q,full,cov,source,h,chain=inputs(task)
    start=time.monotonic();result=predict(q,full,cov,source['parent_guide']);covariance_seconds=time.monotonic()-start
    for key,expected in [('raw_guide','guide_raw'),('mean_fg','foreground_mean_raw'),('mean_bg','background_mean_raw'),('delta','contrast_raw')]:
        assert np.array_equal(result[key].reshape(-1),source[expected].reshape(-1)),key
    s=source['source_s'].reshape(-1);a=source['parent_a'].reshape(-1)
    parent,parent_y,_=parent_graph_readout(source['parent_guide'],s,a,h)
    raw,_,_=parent_graph_readout(result['raw_guide'],s,a,h)
    assert np.array_equal(parent.reshape(64,64),source['parent_mean']) and np.array_equal(parent_y.reshape(64,64),source['parent_unary'])
    assert np.array_equal(raw.reshape(64,64),source['field_raw'])
    assert file_hash(task['erasure_predictions'])==task['erasure_prediction_sha256']
    with np.load(task['erasure_predictions']) as z:
        shape=tuple(z['original_hw'])
        for name,field in [('mean',parent),('mean.raw_reference_contrast',raw)]:
            mask=mask_from_field(field.reshape(64,64))
            for frame,got,hw in [('cli',mask,(1024,1024)),('original',render(mask,shape),shape)]:
                old=np.unpackbits(z[frame+'/'+name],count=int(np.prod(hw))).reshape(hw).astype(bool)
                assert np.array_equal(got,old),name+'/'+frame
    return result,source,h,chain,covariance_seconds


def probe(root):
    import numpy as np,torch
    from ics.methods.reference_discriminant import predict,solve_direction
    initialize(root);task=json.loads((root/'tasks.json').read_text())[0]
    result,_,_,chain,seconds=guides_and_replay(task)
    q=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],dtype=torch.float32)
    r=torch.tensor([[1.,0.],[.6,.8],[-1.,0.],[0.,-1.]],dtype=torch.float32)
    cov=np.array([[1.,.75],[0.,.25]],np.float32);old=np.zeros(4,np.float32)
    got=predict(q,r,cov,old);x=r.numpy().astype(np.float64);cc=cov.ravel().astype(np.float64)
    sigma=np.zeros((2,2),np.float64)
    for weights,mean in [(cc,got['mean_fg']),(1-cc,got['mean_bg'])]:
        for i in range(4):sigma+=.5*weights[i]*np.outer(x[i]-mean.astype(np.float64),x[i]-mean.astype(np.float64))/weights.sum()
    np.testing.assert_allclose(got['covariance'],sigma,rtol=1e-14,atol=1e-14)
    v=np.linalg.solve(sigma+np.trace(sigma)/2*np.eye(2),got['delta'].astype(np.float64))
    np.testing.assert_allclose(got['vector'],v,rtol=1e-12,atol=1e-12)
    order=np.array([3,1,0,2]);perm=predict(q,r[order],cov.ravel()[order].reshape(2,2),old)
    np.testing.assert_allclose(perm['unit_vector_FP64'],got['unit_vector_FP64'],rtol=2e-6,atol=2e-6)
    angle=.4;rotation=torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],dtype=torch.float32)
    rotated=predict(q@rotation,r@rotation,cov,old)
    np.testing.assert_allclose(rotated['unit_vector_FP64'],got['unit_vector_FP64']@rotation.double().numpy(),rtol=3e-6,atol=3e-6)
    np.testing.assert_allclose(rotated['guide'],got['guide'],rtol=3e-6,atol=3e-6)
    delta=torch.tensor([1.,2.],dtype=torch.float64);iv,ii=solve_direction(2*torch.eye(2,dtype=torch.float64),delta)
    np.testing.assert_allclose((iv/iv.norm()).numpy(),(delta/delta.norm()).numpy(),rtol=1e-14,atol=1e-14)
    zv,zi=solve_direction(torch.zeros(2,2,dtype=torch.float64),delta);assert zi['fallback']=='zero_trace_reuse_raw_guide'
    empty=predict(q,r,np.zeros((2,2),np.float32),np.arange(4,dtype=np.float32));assert np.array_equal(empty['guide'],np.arange(4,dtype=np.float32))
    constant=predict(q,torch.ones(4,2),cov,old);assert constant['diagnostics']['fallback']=='zero_delta_reuse_raw_guide'
    folder=root.with_name(root.name+'_preparation');folder.mkdir(parents=True,exist_ok=True)
    independent=folder/'independent_first_case_vectors.npz';independent_check=None
    if independent.exists():
        with np.load(independent) as z:
            for k,e in [('delta','delta'),('mean_fg','mean_fg'),('mean_bg','mean_bg')]:assert np.array_equal(result[k],z[e])
            error=float(np.max(np.abs(result['unit_vector_FP64']-z['fisher_unit_FP64'])))
            np.testing.assert_allclose(result['unit_vector_FP64'],z['fisher_unit_FP64'],rtol=1e-10,atol=1e-11)
        independent_check=dict(state='PASS',unit_FP64_max_error=error,source_sha256=file_hash(independent))
    write(folder/'receipt.json',dict(state='PREPARATION_CHECKS_PASS_NO_INFERENCE',n_fixed=100,episode_id=task['episode_id'],
        config_sha256=file_hash(root/'config.json'),checks=dict(source_profile_entry_payload_tensor_chain=True,full_processed_and_parent_q_sha_exact=True,
            raw_guide_means_delta_bit_exact=True,parent_and_raw_fields_and_both_masks_bit_exact=True,
            independent_scalar_covariance_directsolve=True,rotation_permutation_tolerance=True,isotropic_raw_direction=True,
            zero_trace_raw_fallback=True,empty_role_parent_fallback=True,zero_delta_raw_fallback=True),
        source_chain=chain,first_case=result['diagnostics'],covariance_solve_seconds=seconds,independent_first_case=independent_check,
        encoder_calls=0,new_masks=0,new_miou=0,query_GT_opened=False))
    print(json.dumps(dict(state='PREPARATION_CHECKS_PASS_NO_INFERENCE',receipt=str(folder/'receipt.json'))),flush=True)


def one(task,root):
    import numpy as np
    from ics.methods.reference_erasure_guide import parent_graph_readout
    from ics.methods.rcg import mask_from_field
    from run_m4_baselines import render
    began=time.monotonic();result,source,h,chain,covariance_seconds=guides_and_replay(task)
    z,y,solver=parent_graph_readout(result['guide'],source['source_s'].reshape(-1),source['parent_a'].reshape(-1),h)
    fields={k:v for k,v in result.items() if k!='diagnostics'};fields.update(unary=y.reshape(64,64),field=z.reshape(64,64))
    packed={};assert file_hash(task['old_predictions'])==task['old_prediction_sha256']
    with np.load(task['erasure_predictions']) as saved:
        shape=tuple(saved['original_hw']);packed['original_hw']=saved['original_hw'].copy()
        for arm in ('foris.crf','mean','mean.raw_reference_contrast'):
            for frame in ('cli','original'):packed[frame+'/'+arm]=saved[frame+'/'+arm].copy()
    with np.load(task['old_predictions']) as saved:
        for frame in ('cli','original'):
            for name in ('foris.crf','mean'):assert np.array_equal(saved[frame+'/'+name],packed[frame+'/'+name])
            packed[frame+'/foris.fg_anchor.crf']=saved[frame+'/foris.fg_anchor.crf'].copy()
    mask=mask_from_field(z.reshape(64,64));packed['cli/'+CANDIDATE]=np.packbits(mask);packed['original/'+CANDIDATE]=np.packbits(render(mask,shape))
    name=task['filename'];np.savez_compressed(root/'fields'/name,**fields);np.savez_compressed(root/'predictions'/name,**packed)
    return dict(episode_id=task['episode_id'],filename=name,fields_sha256=file_hash(root/'fields'/name),prediction_sha256=file_hash(root/'predictions'/name),
        source_chain=chain,full_processed_FP32_sha256=task['erasure_processed_FP32_tensor_sha256'],parent_q_FP32_sha256=task['erasure_parent_q_FP32_tensor_sha256'],
        raw_guide_means_delta_bit_exact=True,parent_and_raw_fields_and_masks_bit_exact=True,apd_policy_preserved=True,
        diagnostics=result['diagnostics'],solver=solver,covariance_solve_seconds=covariance_seconds,seconds=time.monotonic()-began,
        encoder_calls=0,new_image_decodes=0,query_GT_in_inference=False)


def infer(root,workers):
    validate(root)
    if (root/'sealed.json').exists():print('Already sealed100; no restart');return
    for name in ('fields','predictions'):(root/name).mkdir(exist_ok=True)
    tasks=json.loads((root/'tasks.json').read_text());done={}
    if (root/'inference.jsonl').exists():
        done=index(root/'inference.jsonl')
        for r in done.values():assert file_hash(root/'fields'/r['filename'])==r['fields_sha256'] and file_hash(root/'predictions'/r['filename'])==r['prediction_sha256']
    started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(root,)) as pool,(root/'inference.jsonl').open('a',buffering=1) as ledger:
        futures=[pool.submit(one,t,root) for t in tasks if t['episode_id'] not in done]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();ledger.write(json.dumps(r)+'\n');done[r['episode_id']]=r
            write(root/'activity.json',dict(state='INFERENCE',n=100,completed=len(done),encoder_calls=0,elapsed_seconds=time.monotonic()-started))
    assert len(done)==100;validate(root)
    write(root/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=100,config_sha256=file_hash(root/'config.json'),manifest_sha256=file_hash(root/'manifest.json'),
        inference_index_sha256=file_hash(root/'inference.jsonl'),tasks_sha256=file_hash(root/'tasks.json'),encoder_calls=0,new_image_decodes=0,
        query_GT_in_inference=False,all_raw_and_parent_replays_bit_exact=True))
    write(root/'activity.json',dict(state='INFERENCE_COMPLETE',n=100,completed=100,encoder_calls=0))


def score(root):
    import numpy as np,torch,torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts,gross_edits
    torch.set_num_threads(2);config=validate(root);verify_seal(root)
    source=index(SOURCE/'episode_metrics.jsonl');pilot=index(PILOT/'episode_metrics.jsonl');ii=index(root/'inference.jsonl');records=[]
    for row in json.loads((root/'manifest.json').read_text()):
        rec=ii[row['episode_id']];path=root/'predictions'/rec['filename'];assert file_hash(path)==rec['prediction_sha256'] and file_hash(root/'fields'/rec['filename'])==rec['fields_sha256']
        with Image.open(row['query_mask_path']) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert array_hash(raw)==row['query_mask_hash'];item=dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],query_photo_id=row['query_photo_id'],frames={})
        with np.load(path) as saved:
            for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                truth=f.interpolate(torch.from_numpy(raw)[None,None].float(),shape,mode='nearest')[0,0].numpy()>.5
                masks={a:np.unpackbits(saved[frame+'/'+a],count=int(np.prod(shape))).reshape(shape).astype(bool) for a in ARMS};iu={a:counts(p,truth) for a,p in masks.items()}
                for a in BASELINES:
                    prior=pilot if a=='foris.fg_anchor.crf' else source
                    assert iu[a]==prior[row['episode_id']]['frames'][frame]['iu'][a]
                item['frames'][frame]=dict(iu=iu,edits={b:gross_edits(masks[CANDIDATE],masks[b],truth) for b in BASELINES},truth_pixels=int(truth.sum()))
        records.append(item)
    (root/'episode_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));report=dict(state='COMPLETE',n=100,frames={},encoder_calls=0,exposure=config['exposure'],new_candidates=1,novel_relation_or_optimal_LDA_claim=False)
    for frame in ('cli','original'):
        flat=[dict(r,**r['frames'][frame]) for r in records];points=point(flat,ARMS);pairs={}
        for b in BASELINES:
            d=[100*(r['iu'][CANDIDATE][0]/max(r['iu'][CANDIDATE][1],1)-r['iu'][b][0]/max(r['iu'][b][1],1)) for r in flat]
            pairs[b]=dict(delta_pp=points[CANDIDATE]-points[b],cases_up=sum(v>1e-10 for v in d),cases_down=sum(v< -1e-10 for v in d),cases_equal=sum(abs(v)<=1e-10 for v in d),
                edits=np.sum([r['edits'][b] for r in flat],axis=0).tolist(),edit_order=['add_TP','add_FP','delete_TP','delete_FP'])
        report['frames'][frame]=dict(miou=points,candidate_pairs=pairs)
    write(root/'report.json',report);write(root/'activity.json',dict(state='COMPLETE',n=100,completed=100,encoder_calls=0));print(json.dumps({f:report['frames'][f]['miou'] for f in ('cli','original')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','probe','infer','score','run']);p.add_argument('--out',type=Path,default=DEFAULT_ROOT);p.add_argument('--workers',type=int,default=8);p.add_argument('--probe',action='store_true');a=p.parse_args()
    if a.probe and a.mode!='prepare':p.error('--probe is only for prepare')
    if a.mode=='prepare':
        prepare(a.out)
        if a.probe:probe(a.out)
    elif a.mode=='probe':probe(a.out)
    elif a.mode=='infer':infer(a.out,a.workers)
    elif a.mode=='score':score(a.out)
    else:infer(a.out,a.workers);score(a.out)


if __name__=='__main__':main()
