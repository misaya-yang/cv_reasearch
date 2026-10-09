#!/usr/bin/env python3
"""Fixed100 reference-supported fidelity and degree-weighted mass control.

Only sealed fields/masks are inputs. No q/R features, images or encoder are read
by infer. Query GT is opened only by score after all100 predictions are sealed.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from raw_feature_cache import file_hash
from lvis_atomic_study import write,point
from lvis_tail_rank_pilot import verify_seal,parent_replay

ASSETS=REPO.parent/'cv_data';SOURCE=ASSETS/'a/lvis_reference_erasure100_20261009'
PILOT=ASSETS/'a/lvis_context_rank_graph100_20261009'
DEFAULT_ROOT=ASSETS/'a/lvis_reference_fidelity100_20261009'
BASELINES=['foris.crf','mean','mean.graph_only','foris.fg_anchor.crf','mean.role_graph','mean.uniform_graph','mean.context_rank_graph','mean.context_rank_matched']
NEW_ARMS={'mean.reference_fidelity':'candidate','mean.reference_fidelity_matched':'matched'}
ARMS=BASELINES+list(NEW_ARMS)


def index(path):return {r['episode_id']:r for r in map(json.loads,Path(path).read_text().splitlines())}


def paths():
    return [Path(__file__).resolve(),REPO/'src/ics/methods/reference_fidelity.py',
        REPO/'src/ics/methods/tail_rank_control.py',REPO/'src/ics/methods/rcg.py',REPO/'src/ics/metrics.py',
        REPO/'src/ics/official_data.py',REPO/'scripts/lvis_tail_rank_pilot.py',REPO/'scripts/lvis_atomic_study.py',
        REPO/'scripts/run_m4_baselines.py',REPO/'scripts/raw_feature_cache.py']


def prepare(root):
    verify_seal(SOURCE);verify_seal(PILOT)
    rows=json.loads((SOURCE/'manifest.json').read_text());assert rows==json.loads((PILOT/'manifest.json').read_text()) and len(rows)==100
    source=index(SOURCE/'inference.jsonl');prior=index(PILOT/'inference.jsonl');tasks=[]
    for row in rows:
        eid=row['episode_id'];s=source[eid];p=prior[eid];fp=SOURCE/'fields'/s['filename'];pp=PILOT/'predictions'/p['filename']
        assert file_hash(fp)==s['fields_sha256'] and file_hash(pp)==p['prediction_sha256']
        tasks.append(dict(episode_id=eid,filename=s['filename'],source_field=str(fp),source_field_sha256=s['fields_sha256'],
                          baseline_prediction=str(pp),baseline_prediction_sha256=p['prediction_sha256']))
    sources={str(p):file_hash(p) for p in paths()};root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():
        old=json.loads((root/'config.json').read_text())
        if old['source_sha256']==sources:validate(root);return old
        if (root/'inference.jsonl').exists() or (root/'sealed.json').exists():raise ValueError('Launched source is frozen')
    write(root/'manifest.json',rows);write(root/'tasks.json',tasks)
    config=dict(state='PREPARED_ONLY',n=100,arms=ARMS,source_sha256=sources,
        producer_seal_sha256={str(p):file_hash(p/'sealed.json') for p in (SOURCE,PILOT)},
        manifest_sha256=file_hash(root/'manifest.json'),tasks_sha256=file_hash(root/'tasks.json'),
        group='G=(original s>.5)&(original y>.5)',
        conditional_rank='locked FP32 half-tie guide rank in G; FP64 arithmetic',
        fallback='nG<2, constant G guide, or zero G degree: explicit parent identity',
        candidate='deltaA=16*original_degree*conditional_rank only G',
        matched='deltaA=16*original_degree*eta only G; eta matches total deltaA, not RHS',
        actual_H='Hnew=Hparent+diag(deltaA); RHS=(original A+deltaA)*original y',
        graph_unchanged=True,source_guide_and_unary_unchanged=True,outside_G_A_unchanged=True,
        readout='original unary, updated fidelity A/H; original CG x0=y rtol1e-7 atol1e-9 max300; no extra CRF; original rendering',
        parameter_search=False,encoder_calls=0,new_image_decodes=0,query_GT_in_inference=False,
        exposure='same exposed frozen100; development only',claim='classical fidelity reweighting, not new information or presumed gain')
    write(root/'config.json',config)
    for p in paths():
        dest=root/'source'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name,parent in [('erasure',SOURCE),('old_pilot',PILOT)]:
        dest=root/'producer_snapshots'/name;dest.mkdir(parents=True,exist_ok=True)
        for f in ('config.json','manifest.json','sealed.json','inference.jsonl'):shutil.copy2(parent/f,dest/f)
        if (parent/'source').exists():shutil.copytree(parent/'source',dest/'source',dirs_exist_ok=True)
    write(root/'activity.json',dict(state='PREPARED_ONLY',n=100,completed=0,encoder_calls=0))
    return config


def validate(root):
    cfg=json.loads((root/'config.json').read_text())
    for p,d in cfg['source_sha256'].items():assert file_hash(p)==d,p
    for p,d in cfg['producer_seal_sha256'].items():assert file_hash(Path(p)/'sealed.json')==d;verify_seal(Path(p))
    for name in ('manifest','tasks'):assert file_hash(root/(name+'.json'))==cfg[name+'_sha256']
    return cfg


def probe(root):
    import numpy as np,torch
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.methods.reference_fidelity import build
    torch.set_num_threads(2);validate(root)
    task=json.loads((root/'tasks.json').read_text())[0]
    s,g,y,a,h,field,solver=parent_replay(task);result=build(h,a,s,g,y)
    expected=DEFAULT_ROOT.parent/(DEFAULT_ROOT.name+'_preparation')/'independent_first_case.npz'
    with np.load(expected) as z:
        for key in z.files:assert np.array_equal(z[key],result['arrays'][key]),key
    # Constant guide deliberately invokes parent identity; CDF itself is .5.
    identity=build(h,a,s,np.zeros_like(g),y)
    assert identity['candidate_Hdiff'].nnz==identity['matched_Hdiff'].nnz==0
    assert np.array_equal(identity['candidate_a'],a) and np.array_equal(identity['matched_a'],a)
    # Three-node direct dense fixture tests both H and synchronous RHS changes.
    aa=np.array([1.,2.,1.]);yy=np.array([.7,.8,.1]);ss=np.array([.6,.8,.1],np.float32);gg=np.array([.2,.9,.1],np.float32)
    ww=np.array([[0.,.1,0.],[.1,0.,.2],[0.,.2,0.]])
    hh=sparse.csr_matrix(np.diag(aa)+16*(np.diag(ww.sum(1))-ww));small=build(hh,aa,ss,gg,yy)
    for kind in ('candidate','matched'):
        da=small['arrays'][kind+'_delta_a'];new=small[kind+'_H']
        assert np.array_equal(new.toarray(),hh.toarray()+np.diag(da))
        rhs=small[kind+'_a']*yy
        zz,status=cg(new,rhs,x0=yy,rtol=1e-7,atol=1e-9,maxiter=300)
        assert status==0 and np.max(np.abs(zz-np.linalg.solve(new.toarray(),rhs)))<1e-10
    for source in (np.zeros_like(ss),np.array([.8,.1,.1],np.float32)):
        empty=build(hh,aa,source,gg,yy)
        assert empty['candidate_Hdiff'].nnz==empty['matched_Hdiff'].nnz==0
    isolated=build(sparse.diags(aa).tocsr(),aa,ss,gg,yy)
    assert isolated['candidate_Hdiff'].nnz==0
    prep=root.parent/(root.name+'_preparation');prep.mkdir(exist_ok=True)
    write(prep/'receipt.json',dict(state='PREPARATION_CHECKS_PASS_NO_INFERENCE',n_fixed=100,
        checks=dict(parent_field_and_both_masks_bit_exact=True,independent_first_case_arrays_bit_exact=True,
            constant_empty_singleton_zero_degree_identity=True,sameG_total_increment_and_outside_A=True,
            small_dense_H_and_synchronous_RHS_CG=True),first_real_diagnostics=result['diagnostics'],
        config_sha256=file_hash(root/'config.json'),parent_solver=solver,encoder_calls=0,new_masks=0,new_miou=0,query_GT_opened=False))

def initialize(root):
    import torch
    torch.set_num_threads(2);validate(Path(root))


def one(task,root):
    import numpy as np
    from ics.methods.reference_fidelity import build
    from ics.methods.tail_rank_control import solve_parent
    from ics.methods.rcg import mask_from_field
    from run_m4_baselines import render
    began=time.monotonic();s,g,y,a,h,old,parent_solver=parent_replay(task);result=build(h,a,s,g,y)
    fields=dict(result['arrays'],parent_unary=y.reshape(64,64),parent_a=a.reshape(64,64),parent_guide=g.reshape(64,64),source_s=s.reshape(64,64))
    solvers={};packed={}
    with np.load(task['baseline_prediction']) as z:
        shape=tuple(z['original_hw']);packed['original_hw']=z['original_hw'].copy()
        for arm in BASELINES:
            for frame in ('cli','original'):packed[frame+'/'+arm]=z[frame+'/'+arm].copy()
    for arm,kind in NEW_ARMS.items():
        z,info=solve_parent(y,result[kind+'_a'],result[kind+'_H']);fields[arm]=z.reshape(64,64)
        info['graph']='Original W; actual Hparent+diag(deltaA) with synchronously updated A';solvers[arm]=info
        diff=result[kind+'_Hdiff']
        for key,value in [('indptr',diff.indptr),('indices',diff.indices),('data',diff.data),('shape',np.asarray(diff.shape))]:fields[kind+'_actual_Hdiff_'+key]=value
        mask=mask_from_field(z.reshape(64,64));packed['cli/'+arm]=np.packbits(mask);packed['original/'+arm]=np.packbits(render(mask,shape))
    name=task['filename'];np.savez_compressed(root/'fields'/name,**fields);np.savez_compressed(root/'predictions'/name,**packed)
    return dict(episode_id=task['episode_id'],filename=name,fields_sha256=file_hash(root/'fields'/name),prediction_sha256=file_hash(root/'predictions'/name),
        parent_field_and_masks_bit_exact=True,diagnostics=result['diagnostics'],solvers=solvers,parent_solver=parent_solver,seconds=time.monotonic()-began,
        source_guide_and_unary_unchanged=True,original_graph_unchanged=True,outside_G_A_unchanged=True,
        worker_pid=os.getpid(),encoder_calls=0,new_image_decodes=0,query_GT_in_inference=False)


def infer(root,workers):
    validate(root)
    if (root/'sealed.json').exists():print('Already sealed100; no restart');return
    write(root/'activity.json',dict(state='INFERENCE_START',controller_pid=os.getpid(),n=100,completed=0,encoder_calls=0))
    for d in ('fields','predictions'):(root/d).mkdir(exist_ok=True)
    tasks=json.loads((root/'tasks.json').read_text());done={}
    if (root/'inference.jsonl').exists():
        done=index(root/'inference.jsonl')
        for r in done.values():assert file_hash(root/'fields'/r['filename'])==r['fields_sha256'] and file_hash(root/'predictions'/r['filename'])==r['prediction_sha256']
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(root,)) as pool,(root/'inference.jsonl').open('a',buffering=1) as ledger:
        futures=[pool.submit(one,t,root) for t in tasks if t['episode_id'] not in done]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();ledger.write(json.dumps(r)+'\n');done[r['episode_id']]=r
            write(root/'activity.json',dict(state='INFERENCE',controller_pid=os.getpid(),n=100,completed=len(done),encoder_calls=0))
    assert len(done)==100;validate(root)
    write(root/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=100,config_sha256=file_hash(root/'config.json'),manifest_sha256=file_hash(root/'manifest.json'),inference_index_sha256=file_hash(root/'inference.jsonl'),
        tasks_sha256=file_hash(root/'tasks.json'),encoder_calls=0,query_GT_in_inference=False,all_parent_replays_bit_exact=True))
    write(root/'activity.json',dict(state='INFERENCE_COMPLETE',n=100,completed=100,encoder_calls=0))


def score(root):
    import numpy as np,torch,torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts,gross_edits
    torch.set_num_threads(2);config=validate(root);verify_seal(root)
    prior=index(PILOT/'episode_metrics.jsonl');ii=index(root/'inference.jsonl');records=[]
    for row in json.loads((root/'manifest.json').read_text()):
        rec=ii[row['episode_id']];path=root/'predictions'/rec['filename'];assert file_hash(path)==rec['prediction_sha256'] and file_hash(root/'fields'/rec['filename'])==rec['fields_sha256']
        with Image.open(row['query_mask_path']) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert array_hash(raw)==row['query_mask_hash'];item=dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],query_photo_id=row['query_photo_id'],frames={})
        with np.load(path) as z:
            for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                truth=f.interpolate(torch.from_numpy(raw)[None,None].float(),shape,mode='nearest')[0,0].numpy()>.5
                masks={arm:np.unpackbits(z[frame+'/'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool) for arm in ARMS};iu={arm:counts(mask,truth) for arm,mask in masks.items()}
                for arm in BASELINES:assert iu[arm]==prior[row['episode_id']]['frames'][frame]['iu'][arm]
                item['frames'][frame]=dict(iu=iu,edits={arm:{b:gross_edits(masks[arm],masks[b],truth) for b in ARMS} for arm in NEW_ARMS},truth_pixels=int(truth.sum()))
        records.append(item)
    (root/'episode_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));report=dict(state='COMPLETE',n=100,frames={},encoder_calls=0,exposure=config['exposure'],new_information_or_gain_presumed=False)
    for frame in ('cli','original'):
        flat=[dict(r,**r['frames'][frame]) for r in records];points=point(flat,ARMS);pairs={}
        for arm in NEW_ARMS:
            for b in ARMS:
                d=[100*(r['iu'][arm][0]/max(r['iu'][arm][1],1)-r['iu'][b][0]/max(r['iu'][b][1],1)) for r in flat]
                pairs[arm+' vs '+b]=dict(delta_pp=points[arm]-points[b],cases_up=sum(v>1e-10 for v in d),cases_down=sum(v< -1e-10 for v in d),cases_equal=sum(abs(v)<=1e-10 for v in d),
                    edits=np.sum([r['edits'][arm][b] for r in flat],axis=0).tolist(),edit_order=['add_TP','add_FP','delete_TP','delete_FP'])
        report['frames'][frame]=dict(miou=points,pairs=pairs)
    write(root/'report.json',report);write(root/'activity.json',dict(state='COMPLETE',controller_pid=os.getpid(),n=100,completed=100,encoder_calls=0));print(json.dumps({f:report['frames'][f]['miou'] for f in ('cli','original')}),flush=True)


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
