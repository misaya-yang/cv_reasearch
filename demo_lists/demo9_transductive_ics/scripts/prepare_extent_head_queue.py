#!/usr/bin/env python3
"""Prepare the finite image-isolated T1 plan. Standard library only.

Writing this plan does NOT authorize GPU, start a process or arm shutdown.
Actual numerical fixtures are root-owned and must have passed before generation.
PCA is built from TRAIN prefixes inside the first live stage before any cache;
confirmation cache has no query truth. A separate confirmation stage consumes
DEV-frozen checkpoints/variant and freezes predictions before its single GT read.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

OWNER=Path('/root/autodl-tmp/demo9_transductive_ics')
READONLY=Path('/root/autodl-tmp/demo9_extent')
ENV_PATH='/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('preserve existing metadata: '+str(path))
    path.write_text(json.dumps(value,indent=2)+'\n')

def owned(path):
    p=Path(path).resolve()
    if p!=OWNER and OWNER not in p.parents:raise ValueError('output outside own namespace: '+str(p))
    return p

def readonly(path):
    p=Path(path).resolve()
    if READONLY not in p.parents:raise ValueError('DEV input must be readonly Claude namespace: '+str(p))
    return p

def uid(name):
    if not isinstance(name,str):raise ValueError('one string image per role required')
    m=re.search(r'(\d{12})\.(?:jpg|jpeg|png)$',Path(name).name,re.I)
    if m is None:raise ValueError('COCO UID absent from image name: '+name)
    return int(m.group(1))

def photos(man):
    return {uid(r[k]) for r in man['episodes'] for k in ('support','query')}

def exposed(man):
    values=man.get('protocol',{}).get('exposed_pool_UIDs',[])
    if not isinstance(values,list):raise ValueError('protocol.exposed_pool_UIDs must be a list')
    return {int(x) for x in values}

def validate_manifest(path,role):
    man=read(path)
    if man.get('state')!='PREPARED':raise ValueError(role+' manifest is not PREPARED')
    rows=man.get('episodes',[])
    if not rows:raise ValueError(role+' manifest is empty')
    identities=[]
    for r in rows:
        f,c,e=int(r['fold']),int(r['c']),int(r['e'])
        if f not in range(4) or c not in range(80) or c%4!=f:
            raise ValueError('collection fold/class mismatch in '+role)
        if uid(r['support'])==uid(r['query']):raise ValueError('same support/query in '+role)
        identities.append((f,e,c))
    if len(identities)!=len(set(identities)):raise ValueError('duplicate case key in '+role)
    return man

def literal_assignment(path,name):
    tree=ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('literal schema declaration missing: '+name)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OWNER)
    p.add_argument('--python',default='/root/miniconda3/bin/python')
    p.add_argument('--train-manifest',type=Path,required=True)
    p.add_argument('--test-manifest',type=Path,default=READONLY/'results/extent_v1/episodes.json')
    p.add_argument('--confirm-manifest',type=Path,required=True)
    p.add_argument('--extent-run',type=Path,default=READONLY/'results/extent_v1/run')
    p.add_argument('--evidence-cache',type=Path,default=READONLY/'cache/evidence_v1')
    p.add_argument('--selfcheck',type=Path,required=True)
    p.add_argument('--fixture-report',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--sampling-receipt',type=Path,default=OWNER/'results/extent_head_t1_isolated_v1/sampling_receipt.json')
    p.add_argument('--prior-failure',type=Path,action='append',default=[])
    p.add_argument('--cache-cap-seconds',type=int,default=5400)
    p.add_argument('--replay-cap-seconds',type=int,default=900)
    p.add_argument('--development-cap-seconds',type=int,default=5400)
    p.add_argument('--confirmation-cap-seconds',type=int,default=900)
    a=p.parse_args()
    root=owned(a.root)
    if root!=OWNER:raise ValueError('only this own execution root is allowed')
    relation_names=literal_assignment(root/'tics/relations.py','NAMES')
    if len(relation_names)!=16:raise ValueError('actual relation schema must contain16 channels')
    if literal_assignment(root/'scripts/extent_head.py','MODEL_CHANNELS')!=32:raise ValueError('legacy25-channel head is SUPERSEDED; common16relations+16PCA requires32')
    tm,cm=owned(a.train_manifest),owned(a.confirm_manifest)
    sm=Path(a.test_manifest).resolve()
    if READONLY not in sm.parents:sm=owned(sm)
    source_dev=READONLY/'results/extent_v1/episodes.json'
    run,ev=readonly(a.extent_run),readonly(a.evidence_cache)
    train=validate_manifest(tm,'training');dev=validate_manifest(sm,'development');confirm=validate_manifest(cm,'confirmation')
    if confirm.get('protocol',{}).get('confirmation_query_mask_pixels_read') is not False:raise ValueError('confirmation query-GT embargo not attested by sampling metadata')
    if dev['episodes']!=read(source_dev)['episodes']:raise ValueError('DEV copy must preserve exact readonly old241 cases')
    for key in ('data_root','annotation_root','foris_root','projection_basis'):
        if any(m.get(key)!=dev.get(key) for m in (train,confirm)):raise ValueError('shared source coordinate mismatch: '+key)
    role_uids={k:photos(v) for k,v in [('train',train),('dev',dev),('confirm',confirm)]}
    overlaps={k:sorted(role_uids[x]&role_uids[y]) for k,x,y in [('train_dev','train','dev'),('train_confirm','train','confirm'),('dev_confirm','dev','confirm')]}
    if any(overlaps.values()):raise ValueError('all-role COCO UID leakage: '+str({k:len(v) for k,v in overlaps.items()}))
    pool_uids=exposed(train)|exposed(confirm)
    if role_uids['confirm']&pool_uids:raise ValueError('confirmation includes prior exposed pool UID')
    for path,state in ((a.selfcheck,'PASSED'),(a.fixture_report,'COMPLETED'),(ev/'report.json','COMPLETED')):
        if read(path).get('state')!=state:raise ValueError('CPU/input artifact not ready: '+str(path))
    name=lambda r:f"{r['fold']}_{r['e']}_{r['c']}"
    for r in dev['episodes']:
        if not (ev/'feat'/(name(r)+'.pt')).is_file() or not (run/'packets'/(name(r)+'.npz')).is_file():
            raise ValueError('readonly DEV feature/packet missing: '+name(r))
    counts={k:{str(f):sum(int(r['fold'])==f for r in m['episodes']) for f in range(4)} for k,m in [('train',train),('dev',dev),('confirm',confirm)]}
    if min(counts['train'].values())<400:raise ValueError('not enough collection rows for independent100/200/400 learning curves')
    if not all(counts['dev'].values()) or not all(counts['confirm'].values()):raise ValueError('all four held folds required')
    cache,out,plan=owned(a.cache),owned(a.out),owned(a.plan)
    prep=root/'results/extent_head_t1_preparation'
    if prep not in out.parents and out!=prep:raise ValueError('fit output must be under preparation prefix')
    if cache.exists() or (out/'report.json').exists():raise ValueError('fresh own cache/report required')
    receipt=owned(a.receipt or plan.with_name('queue_validation.json'))
    cap=[a.cache_cap_seconds,a.replay_cap_seconds,a.development_cap_seconds,a.confirmation_cap_seconds]
    if any(n<=0 or n>43200 for n in cap):raise ValueError('finite positive caps <=43200 required')
    foris=Path(train['foris_root'])
    code=[root/x for x in ('scripts/extent_train_cache.py','scripts/extent_head.py','scripts/extent_experiment.py','scripts/analyze_extent.py','tics/relations.py','tics/extent_cut.py','tics/native_assets.py','tics/__init__.py')]
    frozen=[foris/x for x in ('models/foris.py','utils/clustering.py','utils/refinement.py','utils/data.py')]+[Path(train['projection_basis'])]+code+[source_dev]
    for path in frozen:
        if not path.is_file():raise ValueError('prepared source/input absent: '+str(path))
    sampling=None
    if a.sampling_receipt:
        s=owned(a.sampling_receipt);sampling=dict(path=str(s),sha256=sha(s),metadata=read(s))
    historical=[dict(path=str(owned(x)),sha256=sha(x),metadata=read(x)) for x in a.prior_failure]
    env=dict(PYTHONPATH=ENV_PATH,DEMO9_CUDA_GUARD='1',DEMO4_GPU_FRAC='.4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    done=lambda path,state='COMPLETED':dict(path=str(path),json_equals=dict(state=state))
    manifests=[dict(path=str(x),json_equals=dict(state='PREPARED'),sha256=sha(x)) for x in (tm,sm,cm)]
    ready=[done(a.selfcheck.resolve(),'PASSED'),done(a.fixture_report.resolve()),done(ev/'report.json')]
    freeze=[dict(path=str(x),sha256=sha(x)) for x in frozen]
    built=dict(path=str(cache/'report.json'),json_equals=dict(state='COMPLETED',pca_state='PCA_READY',fourfold_train_prefix_verified=True,PCA_prior_pool_features_used=False,native_checks=8,native_bit_identical=8))
    pca=dict(path=str(cache/'pca.pt'))
    replayed=done(cache/'replay_report.json')
    development_out=out/'development';confirmation_out=out/'confirmation'
    selected=done(development_out/'selection.json','SELECTION_FROZEN')
    developed=dict(path=str(development_out/'report.json'),json_equals=dict(state='DEVELOPMENT_SELECTED_CONFIRM_SEALED',training_epochs=40,seeds=2,score_control_capacity_matched=True,channel_count=32))
    completed=done(confirmation_out/'report.json')
    py=a.python
    def stage(name,timeout,argv,requires,produces):
        return dict(name=name,kind='gpu',cwd=str(root),timeout_seconds=timeout,argv=argv,requires=requires,cpu_artifacts=freeze,code_files=[str(x) for x in code],env=env,produces=produces,success_checks=produces)
    stages=[
        stage('T1_train_prefix_PCA_then_cache',cap[0],[py,str(root/'scripts/extent_train_cache.py'),'--manifest',str(tm),'--confirm-manifest',str(cm),'--test-manifest',str(sm),'--packets',str(run/'packets'),'--components','16','--out',str(cache)],manifests+ready,[built,pca]),
        stage('T1_readonly_DEV_replay',cap[1],[py,str(root/'scripts/extent_train_cache.py'),'--replay','--test-manifest',str(sm),'--features',str(ev/'feat'),'--packets',str(run/'packets'),'--out',str(cache)],[built,pca],[replayed]),
        stage('T1_base_fit_DEV_selection',cap[2],[py,str(root/'scripts/extent_head.py'),'--cache',str(cache),'--run',str(run),'--out',str(development_out/'report.json'),'--phase','development','--train-manifest',str(tm),'--epochs','40','--seeds','2','--batch','64','--curve','100,200,400'],[built,replayed,pca]+manifests,[developed,selected]),
        stage('T1_one_selected_confirmation',cap[3],[py,str(root/'scripts/extent_head.py'),'--cache',str(cache),'--out',str(confirmation_out/'report.json'),'--phase','confirmation','--choice',str(development_out/'selection.json'),'--confirm-manifest',str(cm)],[developed,selected,pca,manifests[2]],[completed,dict(path=str(confirmation_out/'confirmation_prediction_freeze.json')),dict(path=str(confirmation_out/'confirmation_predictions.npz'))])]
    recipe=dict(relation_names=list(relation_names),relation_channel_count=16,pca_channel_count=16,common_input_channel_count=32,epochs=40,ensemble_seeds=2,batch=64,pca_components=16,independent_curve_collection_sizes=[100,200,400],curve_nominal_model_training_sizes=[300,600,1200],independent_curve_model_fits=48,full_model_fits=56,total_model_fits=104,confirmation_refits=0,selected_variant_frozen_on_DEV=True,matched_score_control_same_head_and_capacity=True,scope='T1 patch64/noCRF premise; no T2/T3 or benchmark claim')
    exposure=dict(state='VALIDATED_METADATA_ONLY',actual_collection_counts=counts,actual_total_counts={k:sum(v.values()) for k,v in counts.items()},actual_model_training_counts={str(f):sum(n for g,n in counts['train'].items() if int(g)!=f) for f in range(4)},collection_class_counts={str(f):dict(Counter(int(r['c']) for r in train['episodes'] if int(r['fold'])==f)) for f in range(4)},UID_role_overlaps=overlaps,photo_UIDs={k:sorted(v) for k,v in role_uids.items()},previously_exposed_pool_UIDs=sorted(pool_uids),pool_confirm_overlap=[],old241_DEV_scope='already analysed development, NOT fresh',confirmation_GT_policy='unopened during cache/fitting; selected checkpoint+variant freeze, then prediction SHA freeze, then one GT read',sampling_receipt=sampling,prior_fixed_range_failure_metadata=historical,no_prior_pool_PCA=True,relation_names=list(relation_names),input_channel_count=32,legacy25_receipts='SUPERSEDED: cannot authorize current schema32 runtime',four_model_train_prefix_PCA='first32TRAIN rows/collection; model f only other3 collections; mean[4,1024],V[4,1024,16]',recipe=recipe)
    write(receipt,exposure)
    value=dict(state='PREPARED_NOT_AUTHORIZED',platform='autodl',cuda_python=py,stages=stages,protocol='finite cache -> readonly DEV replay -> DEV-only selection -> one selected confirmation; no retries or sweeps',autostart=False,shutdown_armed=False,current_mode='NO_CARD_CPU_PREPARATION',GPU_execution_requires='new explicit user authorization; generating or preflighting this file does not grant it',finite_stage_caps_seconds=dict(zip([s['name'] for s in stages],cap)),finite_total_cap_seconds=sum(cap),cap_is_measured_runtime=False,recipe=recipe,validation_receipt=dict(path=str(receipt),sha256=sha(receipt)),inputs=dict(train=str(tm),DEV=str(sm),confirm=str(cm),readonly_DEV_manifest_source=str(source_dev),readonly_DEV_run=str(run),readonly_evidence_cache=str(ev),development_report=str(development_out/'report.json'),selection=str(development_out/'selection.json'),confirmation_report=str(confirmation_out/'report.json')),no_old_pool_dependency=True)
    write(plan,value)
    print(json.dumps(dict(state='PREPARED_NOT_AUTHORIZED',plan=str(plan),validation_receipt=str(receipt),actual_counts=exposure['actual_total_counts'],finite_total_cap_seconds=sum(cap),total_independent_fits=104,autostart=False)))

if __name__=='__main__':main()
