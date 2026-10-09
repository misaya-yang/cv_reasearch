#!/usr/bin/env python3
"""Fixed100 graph-only conditional-CDF gating and localized mass control.

Only sealed fields/masks are inputs. No q/R features, images or encoder are read
by infer. Query GT is opened only by score after all100 predictions are sealed.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import multiprocessing
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
PILOT=ASSETS/'a/lvis_atomic1400_20261009/pilot100'
DEFAULT_ROOT=ASSETS/'a/lvis_context_rank_graph100_20261009'
BASELINES=['foris.crf','mean','mean.graph_only','foris.fg_anchor.crf','mean.role_graph','mean.uniform_graph']
NEW_ARMS={'mean.context_rank_graph':'candidate','mean.context_rank_matched':'matched'}
ARMS=BASELINES+list(NEW_ARMS)


def index(path):return {r['episode_id']:r for r in map(json.loads,Path(path).read_text().splitlines())}


def paths():
    return [Path(__file__).resolve(),REPO/'src/ics/methods/context_rank_graph.py',
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
        group='original token G=(s>.5)',cdf='conditional G guide distribution; half ties; FP32 exactly locked rank on G, FP64 edge arithmetic',
        fallback='nG<2 or constant guide on G: all-graph identity',affected='E=(G_i OR G_j); all other edges preserved',
        parent_weights='FP64 W_ij=-Hparent_ij/16 for i!=j',candidate='W*=W*(1-abs(u_i-u_j)) only E',
        matched='same E multiplied eta=sum_E W*/sum_E W; same localized edge mass; no degree renormalization',
        actual_H='Hnew=Hparent+16*(diag(sum(deltaW))-deltaW), preserving parent diagonal rounding residual',
        source_guide_and_unary_unchanged=True,parent_a_unchanged=True,
        readout='saved unary and A; original CG x0=y rtol1e-7 atol1e-9 max300; no extra CRF; original rendering',
        parameter_search=False,encoder_calls=0,new_image_decodes=0,query_GT_in_inference=False,
        exposure='same exposed frozen100; development only',claim='classical bilateral rank gating, not new information or presumed gain')
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
    from ics.methods.context_rank_graph import build,conditional_cdf
    from ics.methods.tail_rank_control import solve_parent
    torch.set_num_threads(2);validate(root)
    task=json.loads((root/'tasks.json').read_text())[0]
    s,g,y,a,h,field,solver=parent_replay(task)
    # Original-field replay only; no real candidate solve or masks in probe.
    real=build(h,s,g)
    g0=np.array([1,1,2,3,0,4],np.float32);group=np.array([1,1,1,1,0,0],bool)
    expected=np.array([(sum(v<x for v in g0[group])+.5*sum(v==x for v in g0[group]))/sum(group) for x in g0],np.float32)
    assert np.array_equal(conditional_cdf(g0,group),expected)
    w=np.array([[0,.1,.2,0],[.1,0,.3,.1],[.2,.3,0,.15],[0,.1,.15,0]],np.float64)
    aa=np.array([.4,.7,.5,.8],np.float64);residual=np.array([1e-6,-2e-7,3e-7,-4e-8])
    dense=np.diag(aa+16*w.sum(1)+residual)-16*w;hh=sparse.csr_matrix(dense)
    original_residual=hh.diagonal()-(aa+16*w.sum(1))
    yy=np.array([.4,.8,.2,.6],np.float64)
    for ss,gg in [(np.zeros(4,np.float32),np.arange(4,dtype=np.float32)),
                  (np.array([.8,.9,.1,.2],np.float32),np.ones(4,np.float32))]:
        identity=build(hh,ss,gg)
        for kind in ('candidate','matched'):
            got=identity[kind+'_H'];assert np.array_equal(got.data,hh.data) and np.array_equal(got.indices,hh.indices) and np.array_equal(got.indptr,hh.indptr)
            assert np.array_equal(solve_parent(yy,aa,got)[0],solve_parent(yy,aa,hh)[0])
    ss=np.array([.7,.2,.8,.1],np.float32);gg=np.array([.3,.6,.2,.8],np.float32);new=build(hh,ss,gg)
    for kind in ('candidate','matched'):
        arrays=new['arrays'];ww=np.zeros((4,4),np.float64);ww[arrays['edge_rows'],arrays['edge_cols']]=arrays[kind+'_weights']
        delta=ww-w;expected_h=dense+16*(np.diag(delta.sum(1))-delta)
        np.testing.assert_allclose(new[kind+'_H'].toarray(),expected_h,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(ww,ww.T,rtol=0,atol=1e-14);assert (ww>=0).all()
        cg,_=solve_parent(yy,aa,new[kind+'_H']);direct=np.linalg.solve(expected_h,aa*yy).astype(np.float32)
        np.testing.assert_allclose(cg,direct,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(new[kind+'_H'].diagonal()-(aa+16*ww.sum(1)),original_residual,rtol=0,atol=4e-15)
    assert new['diagnostics']['outside_E_bit_exact'] and new['diagnostics']['edge_mass_absolute_error']<1e-12
    folder=root.with_name(root.name+'_preparation');folder.mkdir(parents=True,exist_ok=True)
    independent=folder/'independent_first_case.npz';independent_check=None
    if independent.exists():
        with np.load(independent) as z:
            for key,expected in [('group','G'),('u','u'),('edge_rows','row'),('edge_cols','col'),
                                 ('parent_weights','w'),('affected_edges','affected'),('factors','factor')]:
                assert np.array_equal(real['arrays'][key],z[expected]),key
            uniform=np.ones_like(real['arrays']['parent_weights']);uniform[real['arrays']['affected_edges']]=real['diagnostics']['eta']
            np.testing.assert_allclose(uniform,z['uniform_factor'],rtol=1e-14,atol=1e-14)
        independent_check=dict(state='PASS_GRAPH_ARRAYS_NO_CANDIDATE_SOLVE',sha256=file_hash(independent))
    write(folder/'receipt.json',dict(state='PREPARATION_CHECKS_PASS_NO_INFERENCE',n_fixed=100,episode_id=task['episode_id'],config_sha256=file_hash(root/'config.json'),
        checks=dict(parent_field_and_both_masks_bit_exact=True,empty_and_constant_G_H_and_field_bit_exact=True,
            CDF_ties_match_independent_enumeration=True,localized_mass_and_outside_edges_preserved=True,
            symmetry_nonnegative_weights=True,small_directdense_H_and_CG=True,parent_diagonal_residual_preserved=True),
        first_real_graph=real['diagnostics'],independent_first_case=independent_check,parent_solver=solver,encoder_calls=0,new_masks=0,new_miou=0,query_GT_opened=False))
    print(json.dumps(dict(state='PREPARATION_CHECKS_PASS_NO_INFERENCE',receipt=str(folder/'receipt.json'))),flush=True)


def initialize(root):
    import torch
    torch.set_num_threads(2);validate(Path(root))


def one(task,root):
    import numpy as np
    from ics.methods.context_rank_graph import build
    from ics.methods.tail_rank_control import solve_parent
    from ics.methods.rcg import mask_from_field
    from run_m4_baselines import render
    began=time.monotonic();s,g,y,a,h,old,parent_solver=parent_replay(task);result=build(h,s,g)
    fields=dict(result['arrays'],parent_unary=y.reshape(64,64),parent_a=a.reshape(64,64),parent_guide=g.reshape(64,64),source_s=s.reshape(64,64))
    solvers={};packed={}
    with np.load(task['baseline_prediction']) as z:
        shape=tuple(z['original_hw']);packed['original_hw']=z['original_hw'].copy()
        for arm in BASELINES:
            for frame in ('cli','original'):packed[frame+'/'+arm]=z[frame+'/'+arm].copy()
    for arm,kind in NEW_ARMS.items():
        z,info=solve_parent(y,a,result[kind+'_H']);fields[arm]=z.reshape(64,64);solvers[arm]=info
        diff=result[kind+'_Hdiff']
        for key,value in [('indptr',diff.indptr),('indices',diff.indices),('data',diff.data),('shape',np.asarray(diff.shape))]:fields[kind+'_actual_Hdiff_'+key]=value
        mask=mask_from_field(z.reshape(64,64));packed['cli/'+arm]=np.packbits(mask);packed['original/'+arm]=np.packbits(render(mask,shape))
    name=task['filename'];np.savez_compressed(root/'fields'/name,**fields);np.savez_compressed(root/'predictions'/name,**packed)
    return dict(episode_id=task['episode_id'],filename=name,fields_sha256=file_hash(root/'fields'/name),prediction_sha256=file_hash(root/'predictions'/name),
        parent_field_and_masks_bit_exact=True,diagnostics=result['diagnostics'],solvers=solvers,parent_solver=parent_solver,seconds=time.monotonic()-began,
        source_guide_unary_and_A_unchanged=True,encoder_calls=0,new_image_decodes=0,query_GT_in_inference=False)


def infer(root,workers):
    validate(root)
    if (root/'sealed.json').exists():print('Already sealed100; no restart');return
    for d in ('fields','predictions'):(root/d).mkdir(exist_ok=True)
    tasks=json.loads((root/'tasks.json').read_text());done={}
    if (root/'inference.jsonl').exists():
        done=index(root/'inference.jsonl')
        for r in done.values():assert file_hash(root/'fields'/r['filename'])==r['fields_sha256'] and file_hash(root/'predictions'/r['filename'])==r['prediction_sha256']
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(root,)) as pool,(root/'inference.jsonl').open('a',buffering=1) as ledger:
        futures=[pool.submit(one,t,root) for t in tasks if t['episode_id'] not in done]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();ledger.write(json.dumps(r)+'\n');done[r['episode_id']]=r
            write(root/'activity.json',dict(state='INFERENCE',n=100,completed=len(done),encoder_calls=0))
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
