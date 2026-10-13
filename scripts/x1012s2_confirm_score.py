"""One frozen primary candidate on confirm1000; metadata inspection is the default.

`inspect`, `template`, `preflight`, and `self-test` never open query GT. `score`
requires a final-main freeze descriptor and all natural output seals. It scores
the declared primary plus fixed controls and explicitly declared ablations;
it never ranks candidates or changes the primary using confirmation scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import builtins
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT.parent/'cv_data'
MANIFEST=ASSETS/'a/x1012s2_confirm1000/manifest.json'
EXPECTED_MANIFEST='0810bfd1dff670a5a62da8ca8a8bd7dfcac0efcb768c35a574b9f3ed1842c718'
BASELINE_SEAL=ASSETS/'a/x1012s2_confirm1000/baseline/sealed.json'
SOURCES_SEAL=ASSETS/'a/x1012s2_runtime/confirm1000_sources/sealed.json'
DEFAULT_OUT=ASSETS/'a/x1012s2_confirm1000_scores'
DOMAINS=('coco','lvis','pascal_part','paco_part','suim','isaid','isic','lung','fundus','deepglobe_road')
CONTROLS=('FoRIS','MEAN','U','T','INSID3_bilinear','INSID3_CRF640')
EDIT_COLUMNS=('recovered_TP','added_FP','deleted_TP','removed_FP')
CI_SEED=2026101302
CI_REPEATS=1000
FREEZE_STATE='FINAL_MAIN_FROZEN_BEFORE_CONFIRMATION_SCORING'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def key(binding):
    value=binding['index'];return value if isinstance(value,str) else f'{value:06d}'


def manifest_metadata():
    assert MANIFEST.is_file(),'Missing pinned confirmation manifest'
    assert sha(MANIFEST)==EXPECTED_MANIFEST,'Confirmation manifest identity differs'
    rows=read(MANIFEST)
    assert isinstance(rows,list) and len(rows)==1000
    assert len({key(b) for b in rows})==1000
    assert Counter(b['dataset'] for b in rows)==dict.fromkeys(DOMAINS,100)
    assert len({b['inference_task_signature'] for b in rows})==1000
    return rows


@contextmanager
def deny_query_gt(rows):
    """Protect every query foreground/ignore file during non-scoring modes."""
    forbidden={Path(b['row'][k]).resolve() for b in rows for k in ('query_mask_path','query_ignore_mask_path') if b['row'].get(k)}
    saved=builtins.open,io.open
    def wrap(fn):
        def guarded(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)) and Path(os.fsdecode(path)).resolve() in forbidden:
                raise RuntimeError('Confirmation query GT is forbidden in metadata/preflight mode')
            return fn(path,*args,**kwargs)
        return guarded
    builtins.open,io.open=map(wrap,saved)
    try:yield
    finally:builtins.open,io.open=saved


def canonical_photos(rows):
    """Metadata-only alias union; whole-photo groups ignore query crop and class.

    Same real RGB path, original RGB-file SHA, or official dataset/photo ID is
    one photo. Original file SHA also joins shared COCO/LVIS/PACO photographs
    across domains. Road reference/query copies join by their official photo ID.
    No image or annotation contents are opened here.
    """
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    tokens={}
    for b in rows:
        r=b['row'];raw=r['query_path'];resolved=str((ASSETS/raw).resolve())
        ts=['path:'+resolved]
        photo_id=r.get('query_photo_id')
        if photo_id is not None:ts.append('photo_id:'+b['dataset']+':'+str(photo_id))
        h=r.get('source_files',{}).get(raw)
        if h:ts.append('original_RGB_file_sha256:'+h)
        for token in ts:union(ts[0],token)
        tokens[key(b)]=ts
    components=defaultdict(set)
    for ts in tokens.values():
        for token in ts:components[find(ts[0])].add(token)
    canonical={}
    for root,ts in components.items():
        hashes=sorted(t for t in ts if t.startswith('original_RGB_file_sha256:'))
        ids=sorted(t for t in ts if t.startswith('photo_id:'))
        canonical[root]=(hashes or ids or sorted(ts))[0]
    mapping={k:canonical[find(ts[0])] for k,ts in tokens.items()}
    groups=defaultdict(list)
    for b in rows:groups[mapping[key(b)]].append(b)
    info={'canonical_photo_n':len(groups),
          'canonical_photo_n_by_domain':{d:len({mapping[key(b)] for b in rows if b['dataset']==d}) for d in DOMAINS},
          'cross_domain_photo_clusters':[{'canonical_photo':p,'domains':sorted({b['dataset'] for b in bs}),
              'case_n':len(bs)} for p,bs in groups.items() if len({b['dataset'] for b in bs})>1],
          'metadata_policy':'Union real RGB path, original RGB file SHA, and dataset/official photo ID; do not split crops/classes of one photo.',
          'query_GT_reads':0,'RGB_contents_read':0}
    return mapping,info


def freeze_template():
    return {'state':FREEZE_STATE,'manifest_sha256':EXPECTED_MANIFEST,
        'main':{'id':'REPLACE_WITH_ONE_DEVELOPMENT_SELECTED_ID',
                'definition':'REPLACE_WITH_THE_FIXED_FORMULA_AND_POSTPROCESSING',
                'config_path':str(ASSETS/'a/x1012s2_confirm1000/candidate_sources/config.json'),'config_sha256':'REPLACE',
                'seal_path':str(ASSETS/'a/x1012s2_confirm1000/candidate_sources/sealed.json'),'seal_sha256':'REPLACE',
                'mask_key':'REPLACE_WITH_NATURAL_MAIN_MASK_KEY',
                'path_field':'prediction_path','sha_field':'prediction_sha256',
                'production_timing_scope':'Describe whether generation computes the complete cached-O24 pipeline or only composes existing fields/masks.'},
        'controls':{'baseline_seal_path':str(BASELINE_SEAL),'sources_seal_path':str(SOURCES_SEAL)},
        'ablations':[],
        'ablation_policy':'Optional explicit frozen descriptors, same shape as main plus purpose; mechanistic only, never used to change main after scoring.',
        'timing_evidence':[],
        'timing_evidence_schema':{'path':'/absolute/path/to/benchmark.json','sha256':'REPLACE','label':'Matched benchmark label',
                                  'scope':'Explicit cached-O24 versus RGB/encoder-inclusive scope','applies_to':['exact_candidate_id']},
        'development_selection_evidence':[],
        'confirmation_rule_selection_forbidden':True}


def index_seal(path,expected_sha=None):
    p=Path(path);assert p.is_file(),'Natural output seal is not complete: '+str(p)
    actual=sha(p)
    if expected_sha:assert actual==expected_sha,'Seal digest mismatch: '+str(p)
    seal=read(p)
    assert seal['n']==1000 and len(seal['records'])==1000
    assert seal.get('manifest_sha256',EXPECTED_MANIFEST)==EXPECTED_MANIFEST
    records={str(r['index']):r for r in seal['records']}
    assert len(records)==1000
    return seal,records,actual


def natural_spec(record,path_field,hash_field,mask_key):
    path=record.get(path_field)
    value=record.get(hash_field)
    if path is None and path_field=='prediction_path':path=record.get('path')
    if value is None and hash_field=='prediction_sha256':value=record.get('sha256')
    assert path and value and len(value)==64
    return {'path':str(Path(path).resolve()),'sha256':value,'key':mask_key}


def candidate_descriptor(descriptor):
    assert descriptor['id'] and not descriptor['id'].startswith('REPLACE')
    assert descriptor['definition'] and not str(descriptor['definition']).startswith('REPLACE')
    assert sha(descriptor['config_path'])==descriptor['config_sha256']
    config=read(descriptor['config_path'])
    seal,records,h=index_seal(descriptor['seal_path'],descriptor['seal_sha256'])
    declared=[v for v in (seal.get('manifest_sha256'),config.get('manifest_sha256'),config.get('source_manifest_sha256')) if v]
    assert declared and all(v==EXPECTED_MANIFEST for v in declared),'Main/ablation producer must bind the pinned original manifest'
    if seal.get('config_sha256'):assert seal['config_sha256']==descriptor['config_sha256']
    return seal,records,h


def inspection(out):
    rows=manifest_metadata()
    with deny_query_gt(rows):
        _,photos=canonical_photos(rows)
        sources={}
        for name,path in (('FoRIS_MEAN',BASELINE_SEAL),('U_T_INSID3',SOURCES_SEAL)):
            sources[name]={'seal_path':str(path),'sealed':path.is_file()}
            if path.is_file():
                seal=read(path);sources[name].update(n=seal.get('n'),state=seal.get('state'),seal_sha256=sha(path))
        result={'state':'SCORER_PREPARED_METADATA_ONLY_MAIN_NOT_SELECTED','manifest_path':str(MANIFEST),
            'manifest_sha256':EXPECTED_MANIFEST,'n':1000,'per_domain_n':dict(Counter(b['dataset'] for b in rows)),
            'canonical_query_photo_policy':photos,'controls':sources,
            'primary_metric':'Pool I/U within every observed (fold,global_class), then equal mean of those slots; ten-domain macro is equal-domain mean.',
            'CI_protocol':'One global canonical-photo cluster resample per replicate, shared weights across all domains, refs, masks and class episodes of that photo; then recompute domain slot scores and macro.',
            'bootstrap_seed':CI_SEED,'bootstrap_repeats':CI_REPEATS,
            'scope':'New unused inference tasks from preexisting photographs, not a new-photo holdout.',
            'query_GT_reads':0,'scoring_executed':False,'scorer_sha256':sha(__file__),
            'next_step':'Wait for the explicit final-main freeze notification. Only then supply the pinned freeze descriptor to score.'}
        write(out/'preparation.json',result)
        write(out/'final_freeze.template.json',freeze_template())
    return result


def preflight(freeze_path,out,verify_masks=False):
    rows=manifest_metadata()
    assert freeze_path is not None,'A final-main freeze descriptor is required; no confirmation GT was read'
    with deny_query_gt(rows):
        frozen=read(freeze_path)
        assert frozen['state']==FREEZE_STATE
        assert frozen['manifest_sha256']==EXPECTED_MANIFEST
        assert frozen.get('confirmation_rule_selection_forbidden') is True
        main=frozen['main'];controls=frozen.get('controls',{})
        bm,bm_rows,bm_sha=index_seal(controls.get('baseline_seal_path',BASELINE_SEAL),controls.get('baseline_seal_sha256'))
        uj,uj_rows,uj_sha=index_seal(controls.get('sources_seal_path',SOURCES_SEAL),controls.get('sources_seal_sha256'))
        assert bm.get('foris_complete')==bm.get('mean_complete')==1000
        assert bm.get('query_GT_scored') is False
        assert not bm.get('query_GT_reads') and not uj.get('query_GT_reads')
        assert set(uj.get('providers',('U','T','I','J')))=={'U','T','I','J'}
        main_seal,main_rows,main_sha=candidate_descriptor(main)
        main_label='main';arm_order=[main_label,*CONTROLS]
        descriptors={main_label:main};candidate_maps={main_label:main_rows};candidate_seals={main_label:main_seal}
        for descriptor in frozen.get('ablations',[]):
            assert descriptor.get('purpose'),'Every scored ablation needs a frozen mechanistic purpose'
            label='ablation/'+descriptor['id'];assert label not in arm_order and descriptor['id']!=main['id']
            seal,mapping,_=candidate_descriptor(descriptor)
            descriptors[label]=descriptor;candidate_maps[label]=mapping;candidate_seals[label]=seal;arm_order.append(label)
        expected={key(b) for b in rows}
        for mapping in (bm_rows,uj_rows,*candidate_maps.values()):assert set(mapping)==expected
        canonical,photo_info=canonical_photos(rows)
        jobs=[];file_hashes={}
        for b in rows:
            k=key(b);br,sr=bm_rows[k],uj_rows[k]
            assert br.get('mean_status')=='COMPLETE'
            sources={
                'FoRIS':natural_spec(br,'prediction_path','prediction_sha256','cli/foris.crf'),
                'MEAN':natural_spec(br,'mean_prediction_path','mean_prediction_sha256','cli/mean')}
            for alias,actual in (('U','U'),('T','T'),('INSID3_bilinear','I'),('INSID3_CRF640','J')):
                sources[alias]=natural_spec(sr,'prediction_path','prediction_sha256',actual)
            for label,descriptor in descriptors.items():
                r=candidate_maps[label][k]
                sources[label]=natural_spec(r,descriptor.get('path_field','prediction_path'),descriptor.get('sha_field','prediction_sha256'),descriptor['mask_key'])
                if r.get('dataset') is not None:assert r['dataset']==b['dataset']
            if verify_masks:
                import numpy as np
                paths=defaultdict(list)
                for label,spec in sources.items():paths[spec['path']].append((label,spec))
                for path,items in paths.items():
                    if path not in file_hashes:file_hashes[path]=sha(path)
                    assert all(file_hashes[path]==spec['sha256'] for _,spec in items)
                    with np.load(path,allow_pickle=False) as z:
                        for label,spec in items:
                            mask=z[spec['key']]
                            assert mask.dtype==np.uint8 and mask.shape==(131072,),(k,label,'Expected packed1024 mask')
            jobs.append({'index':k,'binding':b,'canonical_query_photo':canonical[k],
                'query_photo_path':str((ASSETS/b['row']['query_path']).resolve()),'sources':sources})
        timing=[]
        for evidence in frozen.get('timing_evidence',[]):
            assert evidence['scope'] and evidence['label'] and sha(evidence['path'])==evidence['sha256']
            timing.append(evidence)
        protocol={'manifest_sha256':EXPECTED_MANIFEST,'freeze_path':str(Path(freeze_path).resolve()),
            'freeze_sha256':sha(freeze_path),'scorer_sha256':sha(__file__),
            'baseline_seal_sha256':bm_sha,'source_seal_sha256':uj_sha,'main_seal_sha256':main_sha,
            'ablation_seal_sha256':{k:sha(d['seal_path']) for k,d in descriptors.items() if k!='main'},
            'canonical_photos_sha256':digest(canonical),'main_id':main['id'],'arm_order':arm_order,
            'primary_metric':'Observed(fold,class) pooledI/U, equal-slot mean, equal-domain macro',
            'global_photo_bootstrap':{'seed':CI_SEED,'repeats':CI_REPEATS,'shared_cross_domain_weights':True},
            'scope':'Unused R-input/reference-mask1024/Q-input tasks on old photographs; not global photo holdout',
            'timing_evidence':timing}
        result={'state':'ALL_FROZEN_INPUTS_READY_NO_GT_READ','n':len(jobs),'protocol':protocol,
            'protocol_sha256':digest(protocol),'canonical_photos':photo_info,
            'prediction_files_verified':len(file_hashes),'query_GT_reads':0,
            'controls_seal_state':{'FoRIS_MEAN':bm['state'],'UTIJ':uj['state']},
            'main_id':main['id'],'ablation_ids':[d['id'] for k,d in descriptors.items() if k!='main']}
        write(out/'preflight.json',result)
        write(out/'evaluation_bindings.json',jobs)
    return rows,jobs,frozen,result,{'FoRIS_MEAN':bm,'UTIJ':uj,**candidate_seals}


def score_helpers(out):
    import torch
    modules={}
    for name,source in [('lab',ASSETS/'a/x1012_R1/frozen/3_lab.py'),('plan_common',ASSETS/'a/x1012_R1/frozen/1_plan_common.py')]:
        target=out/'runtime'/(name+'.py');target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(source.read_bytes())
        assert target.read_bytes()==source.read_bytes()
        spec=importlib.util.spec_from_file_location(name,target);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
        if name=='lab':module.ROOT,module.ASSETS,module.DEV=ROOT,ASSETS,torch.device('cpu')
        modules[name]=module
    torch.set_num_threads(1)
    return modules['plan_common']


def popcount(bits):
    import numpy as np
    table=np.asarray([i.bit_count() for i in range(256)],np.uint8)
    return int(table[bits].sum(dtype=np.uint64))


def packed_metric(pred,g,v,ng,nv):
    import numpy as np
    p=pred&v;tp,n=popcount(p&g),popcount(p);fp,fn=n-tp,ng-tp
    err=float(abs(np.log2(n/ng))) if n and ng else (0. if not n and not ng else None)
    return {'I':tp,'U':tp+fp+fn,'TP':tp,'FP':fp,'FN':fn,'GT':ng,'predicted':n,
        'IoU':100*tp/max(tp+fp+fn,1),'target_fraction':ng/max(nv,1),'abs_log2_area_error':err,
        'infinite_area_error':err is None,'zero_predicted_area':n==0}


def edit_vector(pred,base,g,v):
    p,b=pred&v,base&v;add,delete=p&~b,b&~p;at,dt=popcount(add&g),popcount(delete&g)
    return [at,popcount(add)-at,dt,popcount(delete)-dt]


def point_summary(records,arms):
    import numpy as np
    result={}
    for d in DOMAINS:
        rs=[r for r in records if r['dataset']==d]
        if not rs:continue
        table={}
        for a in arms:
            slots=defaultdict(lambda:[0,0])
            for r in rs:
                m=r['metrics'][a];s=slots[(str(r['fold']),str(r['class_id']))];s[0]+=m['I'];s[1]+=m['U']
            slot_scores={f'{f}/{c}':100*i/max(u,1) for (f,c),(i,u) in slots.items()}
            folds=defaultdict(list)
            for (f,c),(i,u) in slots.items():folds[f].append(100*i/max(u,1))
            ms=[r['metrics'][a] for r in rs];area=[float('inf') if m['abs_log2_area_error'] is None else m['abs_log2_area_error'] for m in ms]
            median=float(np.median(area))
            size={label:[m for m in ms if lo<=m['target_fraction']<hi] for label,lo,hi in (('<2%',0,.02),('2-20%',.02,.2),('>20%',.2,1.00001))}
            table[a]={'n':len(rs),'canonical_query_photo_n':len({r['query_photo'] for r in rs}),
                'observed_class_slot_n':len(slots),'class_miou':float(np.mean(list(slot_scores.values()))),
                'episode_mean_iou':float(np.mean([m['IoU'] for m in ms])),
                'per_fold_observed_class_miou':{f:float(np.mean(v)) for f,v in folds.items()},
                'fold_equal_miou_secondary':float(np.mean([np.mean(v) for v in folds.values()])),
                'FN_GT':sum(m['FN'] for m in ms)/max(sum(m['GT'] for m in ms),1),
                'FP_GT':sum(m['FP'] for m in ms)/max(sum(m['GT'] for m in ms),1),
                'median_abs_log2_area_error':median if np.isfinite(median) else 'Infinity',
                'zero_prediction_n':sum(m['zero_predicted_area'] for m in ms),
                'size_bands':{label:{'n':len(v),'episode_mean_iou':float(np.mean([m['IoU'] for m in v])) if v else None} for label,v in size.items()},
                'class_slot_I_U':{f'{f}/{c}':iu for (f,c),iu in slots.items()}}
        result[d]=table
    return result


def global_photo_bootstrap(records,arms,summary,repeats=CI_REPEATS,seed=CI_SEED,debug=False):
    """Every canonical photo gets one weight shared across all its domains."""
    import numpy as np
    from scipy.sparse import csr_matrix
    photos=sorted({r['query_photo'] for r in records});pm={p:i for i,p in enumerate(photos)}
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(photos),np.full(len(photos),1/len(photos)),size=repeats)
    observed=list(summary);samples={};out={}
    for d in observed:
        rs=[r for r in records if r['dataset']==d];slots=sorted({(str(r['fold']),str(r['class_id'])) for r in rs});sm={s:i for i,s in enumerate(slots)}
        pi=[pm[r['query_photo']] for r in rs];si=[sm[(str(r['fold']),str(r['class_id']))] for r in rs]
        shape=(len(photos),len(slots));presence=csr_matrix((np.ones(len(rs)),(pi,si)),shape=shape)
        seen=(weights@presence)>0
        assert np.all(seen.sum(1)>0),'A global bootstrap draw omitted an entire domain; inspect the declared sampling population before changing protocol'
        samples[d]={}
        for a in arms:
            ii=csr_matrix(([r['metrics'][a]['I'] for r in rs],(pi,si)),shape=shape)
            uu=csr_matrix(([r['metrics'][a]['U'] for r in rs],(pi,si)),shape=shape)
            samples[d][a]=((100*(weights@ii)/np.maximum(weights@uu,1))*seen).sum(1)/seen.sum(1)
        out[d]={}
        for a in arms:
            if a=='main':continue
            delta=samples[d]['main']-samples[d][a]
            out[d][a]={'main_minus_arm_pp':summary[d]['main']['class_miou']-summary[d][a]['class_miou'],
                'paired_canonical_photo_bootstrap95_pp':np.quantile(delta,[.025,.975]).tolist()}
    macro={}
    for a in arms:
        if a=='main':continue
        delta=np.mean([samples[d]['main']-samples[d][a] for d in observed],axis=0)
        macro[a]={'main_minus_arm_pp':float(np.mean([summary[d]['main']['class_miou']-summary[d][a]['class_miou'] for d in observed])),
            'paired_canonical_photo_bootstrap95_pp':np.quantile(delta,[.025,.975]).tolist()}
    result={'domains':out,'macro10':macro,'seed':seed,'repeats':repeats,'canonical_photo_n':len(photos),
        'resampling':'global canonical-photo multinomial; a photo has identical weight in every domain/episode/arm; observed class slots reaggregated each draw',
        'intervals':'unadjusted fixed-primary paired95 percent bootstrap intervals, not simultaneous multiple-comparison intervals'}
    if debug:result['_samples']=samples;result['_weights']=weights;result['_photos']=photos
    return result


def numeric_summary(values):
    import numpy as np
    x=np.asarray([v for v in values if isinstance(v,(float,int)) and np.isfinite(v)],np.float64)
    if not len(x):return {'n':0,'available':False}
    return {'n':len(x),'mean_seconds':float(x.mean()),'p50_seconds':float(np.median(x)),
        'p95_seconds':float(np.quantile(x,.95)),'sum_seconds':float(x.sum()),'available':True}


def timing_summary(seals,frozen):
    b,s=seals['FoRIS_MEAN'],seals['UTIJ']
    result={'FoRIS_cached_pipeline':numeric_summary([r.get('foris_seconds') for r in b['records']]),
        'MEAN_increment_after_FoRIS_not_standalone':numeric_summary([r.get('mean_seconds') for r in b['records']]),
        'FoRIS_plus_MEAN_serial_source_run':numeric_summary([r.get('seconds') for r in b['records']]),
        'UTIJ_shared_source_bundle':numeric_summary([r.get('complete_cached_seconds') for r in s['records']]),
        'source_runs':{},'candidate_production':{},'external_evidence':[],
        'scope':'Production clocks are cached-O24/source or mask-generation measurements. Never sum independent source clocks into a claimed final latency; use matched complete-pipeline evidence, distinguishing cached-O24 from actual RGB encoding.'}
    for label,seal in seals.items():
        wall=seal.get('wall_seconds',seal.get('seconds'))
        result['source_runs'][label]={'seal_reported_wall_seconds':wall,'case_n':seal.get('n'),
            'seal_run_cases_per_second':seal['n']/wall if isinstance(wall,(int,float)) and wall>0 else None,
            'note':'Reported seal-run throughput; resume/revalidation and concurrency boundaries may differ from per-case method latency.'}
        if label=='main' or label.startswith('ablation/'):
            result['candidate_production'][label]={k:numeric_summary([r.get(k) for r in seal['records']]) for k in ('complete_cached_seconds','seconds_including_artifacts','seconds') if any(k in r for r in seal['records'])}
            descriptor=frozen['main'] if label=='main' else next(d for d in frozen.get('ablations',[]) if 'ablation/'+d['id']==label)
            result['candidate_production'][label]['declared_scope']=descriptor.get('production_timing_scope','Natural artifact-production clocks only; not certified full selected-pipeline latency')
    for e in frozen.get('timing_evidence',[]):
        report=read(e['path']);result['external_evidence'].append({**e,'evidence_state':report.get('state'),
            'reported_summary':report.get('summary',report.get('latency_summary')),
            'ratio_of_mean_times':report.get('ratio_of_mean_times'),'xor_sum':report.get('xor_sum'),
            'limits':report.get('limits'),'new_encoder_calls':report.get('new_encoder_calls')})
    return result


def render_report(report,out):
    main=report['frozen_main'];summary=report['summary'];arms=report['arm_order'];paired=report['paired_comparisons']
    lines=['# 固定主方法的1000任务确认评分','',
        f"主方法：`{main['id']}`。定义：`{json.dumps(main['definition'],ensure_ascii=False) if isinstance(main['definition'],dict) else main['definition']}`。",'',
        '主方法在确认GT读取前冻结；全部主方法、对照和显式消融自然输出封存后统一评分。主方法保持固定，消融只作机制分析，不能据本表重选主公式。',
        '任务是未使用的R输入/参考mask1024/Q输入组合，照片均为已有照片；不能宣称全新照片留出。',
        '主指标：各观察到的(fold,global_class)槽池化I/U，再槽等权；总体为十域等权均值。折等权次指标单列在JSON，不能混用。',
        '置信区间：全局canonical query-photo簇重采；同一照片跨域共享一次采样权重，每次重算域分数与总体。', '',
        '|域|'+'|'.join(arms)+'|','|---|'+'|'.join(['---:']*len(arms))+'|']
    for d,t in summary.items():lines.append('|'+d+'|'+'|'.join(f"{t[a]['class_miou']:.4f}" for a in arms)+'|')
    lines.append('|十域均值|'+'|'.join(f"{report['macro10_class_miou'][a]:.4f}" for a in arms)+'|')
    lines+=['','|范围|主方法−对照|差值pp|照片配对95%区间|','|---|---|---:|---|']
    for scope,t in [*paired['domains'].items(),('macro10',paired['macro10'])]:
        for a,v in t.items():
            lo,hi=v['paired_canonical_photo_bootstrap95_pp'];lines.append(f"|{scope}|{a}|{v['main_minus_arm_pp']:+.4f}|[{lo:+.4f},{hi:+.4f}]|")
    lines+=['','|域|对照|补回TP|新增FP|删除TP|移除FP|','|---|---|---:|---:|---:|---:|']
    for d,t in report['main_edits'].items():
        for a,v in t.items():lines.append('|'+d+'|'+a+'|'+'|'.join(str(x) for x in v)+'|')
    lines+=['','## 延迟和吞吐证据','',report['timing']['scope'],'']
    for label,v in report['timing'].items():
        if isinstance(v,dict) and v.get('available'):
            lines.append(f"- {label}: n={v['n']}, mean={v['mean_seconds']:.4f}s, p50={v['p50_seconds']:.4f}s, p95={v['p95_seconds']:.4f}s。")
    lines+=['','|自然产物流程|seal报告wall秒|seal运行吞吐例/秒|','|---|---:|---:|']
    for label,v in report['timing']['source_runs'].items():
        wall,rate=v['seal_reported_wall_seconds'],v['seal_run_cases_per_second']
        lines.append('|'+label+'|'+(f'{wall:.4f}' if isinstance(wall,(float,int)) else '未提供')+'|'+(f'{rate:.4f}' if rate is not None else '未提供')+'|')
    lines+=['','这些吞吐属于对应完整产物流程；若一个自然库同时生产多臂，不能当作单个主方法的在线吞吐。']
    for label,v in report['timing']['candidate_production'].items():
        lines.append(f"- {label} production: {v['declared_scope']}")
    for e in report['timing']['external_evidence']:
        lines.append(f"- {e['label']}: {e['scope']}；适用方法 {e.get('applies_to',[])}；来源 `{e['path']}`，SHA `{e['sha256']}`。")
        if e.get('reported_summary') is not None:lines.append('  `'+json.dumps(e['reported_summary'],ensure_ascii=False)+'`')
    lines+=['',f"确认manifest SHA `{EXPECTED_MANIFEST}`；评分协议SHA `{report['protocol_sha256']}`。",f"完整逐例整数、canonical照片分组、折/大小/FN/FP及全部时间口径：`{out/'report.json'}`。"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def attach_timing(out,evidence_path):
    """Append completed timing evidence without rereading GT or changing scores."""
    assert evidence_path is not None,'Provide a JSON list of timing evidence descriptors'
    rows=manifest_metadata()
    with deny_query_gt(rows):
        report=read(out/'report.json');additions=read(evidence_path)
        assert isinstance(additions,list) and additions
        before_metrics=digest({'main':report['frozen_main'],'summary':report['summary'],
                               'paired':report['paired_comparisons'],'edits':report['main_edits']})
        known={(e['path'],e['sha256']) for e in report['timing']['external_evidence']}
        for evidence in additions:
            assert evidence['scope'] and evidence['label'] and sha(evidence['path'])==evidence['sha256']
            if (evidence['path'],evidence['sha256']) in known:continue
            source=read(evidence['path'])
            report['timing']['external_evidence'].append({**evidence,'evidence_state':source.get('state'),
                'reported_summary':source.get('summary',source.get('latency_summary')),
                'ratio_of_mean_times':source.get('ratio_of_mean_times'),'xor_sum':source.get('xor_sum'),
                'limits':source.get('limits'),'new_encoder_calls':source.get('new_encoder_calls')})
        report.setdefault('timing_only_updates',[]).append({'evidence_descriptor_sha256':sha(evidence_path),
            'main_and_accuracy_sha256':before_metrics,'query_GT_reads':0,'accuracy_changed':False})
        assert before_metrics==digest({'main':report['frozen_main'],'summary':report['summary'],
                                       'paired':report['paired_comparisons'],'edits':report['main_edits']})
        write(out/'report.json',report);render_report(report,out)
        compact={k:v for k,v in report.items() if k!='records'}
        compact.update(report_path=str(out/'report.json'),report_sha256=sha(out/'report.json'))
        write(out/'completion.json',compact)
    return {'state':'TIMING_EVIDENCE_ATTACHED_ACCURACY_UNCHANGED','main_id':report['frozen_main']['id'],'query_GT_reads':0}


def score(freeze_path,out):
    # No caller should invoke this before the root's explicit rule-freeze notice.
    rows,jobs,frozen,ready,seals=preflight(freeze_path,out,verify_masks=True)
    protocol=ready['protocol'];fingerprint=ready['protocol_sha256']
    started=out/'score_started.json'
    if started.exists():
        assert read(started)['protocol_sha256']==fingerprint,'Confirmation protocol/main changed after scoring began'
    else:
        write(started,{'state':'FIXED_CONFIRMATION_SCORING_STARTED','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'protocol':protocol,'protocol_sha256':fingerprint,'query_GT_reads_before_this_marker':0})
    if (out/'report.json').exists():
        result=read(out/'report.json');assert result['protocol_sha256']==fingerprint
        print({'state':'ALREADY_COMPLETE_UNCHANGED_MAIN','main_id':frozen['main']['id']},flush=True);return
    pc=score_helpers(out);arms=protocol['arm_order'];records=[];began=time.monotonic()
    import numpy as np
    for i,job in enumerate(jobs,1):
        path=out/'records'/(job['index']+'.json')
        if path.exists():
            record=read(path);assert record['protocol_sha256']==fingerprint
        else:
            predictions={};paths=defaultdict(list)
            for a,spec in job['sources'].items():paths[spec['path']].append((a,spec))
            for source,items in paths.items():
                assert sha(source)==items[0][1]['sha256']
                with np.load(source,allow_pickle=False) as z:
                    for a,spec in items:predictions[a]=z[spec['key']].copy()
            gt,valid=pc.truth(job['binding']);g,v=pc.pack(gt),pc.pack(valid);ng,nv=int(gt.sum()),int(valid.sum())
            b=job['binding'];metrics={a:packed_metric(predictions[a],g,v,ng,nv) for a in arms}
            edits={a:edit_vector(predictions['main'],predictions[a],g,v) for a in arms if a!='main'}
            for a,(at,af,dt,df) in edits.items():
                assert metrics['main']['TP']==metrics[a]['TP']+at-dt and metrics['main']['FP']==metrics[a]['FP']+af-df
            record={'index':job['index'],'dataset':b['dataset'],'fold':b['row']['fold'],'class_id':b['row']['global_class_id'],
                'query_photo':job['canonical_query_photo'],'query_photo_path':job['query_photo_path'],
                'inference_task_signature':b['inference_task_signature'],'metrics':metrics,'main_edits':edits,
                'protocol_sha256':fingerprint,'prediction_source_identity':digest(job['sources'])}
            write(path,record)
        records.append(record)
        if i<=2 or i%100==0:print({'phase':'fixed-confirm1000-score','completed':i,'n':1000,'main_id':frozen['main']['id']},flush=True)
    summary=point_summary(records,arms)
    point={'state':'FIXED_MAIN_POINT_SCORES_COMPLETE_CI_PENDING','main_id':frozen['main']['id'],
        'n':len(records),'summary':summary,'arm_order':arms,'protocol_sha256':fingerprint,
        'macro10_class_miou':{a:float(np.mean([summary[d][a]['class_miou'] for d in DOMAINS])) for a in arms}}
    write(out/'point_scores.json',point)
    print({'phase':'fixed-confirm1000-point-complete','main_id':frozen['main']['id'],
        'macro10':point['macro10_class_miou'],'point_scores':str(out/'point_scores.json')},flush=True)
    paired=global_photo_bootstrap(records,arms,summary)
    edits={d:{a:np.sum([r['main_edits'][a] for r in records if r['dataset']==d],axis=0).tolist() for a in arms if a!='main'} for d in DOMAINS}
    report={'state':'COMPLETE_ONE_FROZEN_MAIN_CONFIRM1000','n':1000,'frozen_main':frozen['main'],
        'frozen_ablations':frozen.get('ablations',[]),'ablation_use':'mechanistic reporting only; primary never reselected',
        'arm_order':arms,'summary':summary,'macro10_class_miou':{a:float(np.mean([summary[d][a]['class_miou'] for d in DOMAINS])) for a in arms},
        'paired_comparisons':paired,'edit_columns':EDIT_COLUMNS,'main_edits':edits,'records':records,
        'canonical_photos':ready['canonical_photos'],'protocol':protocol,'protocol_sha256':fingerprint,
        'timing':timing_summary(seals,frozen),'score_seconds':time.monotonic()-began,
        'query_GT_role':'Only scoring after final-main and all natural outputs are frozen; no model/threshold selection',
        'primary_changed_after_confirmation':False,'new_encoder_calls':0,'fitted_models':0,
        'scope':protocol['scope']}
    write(out/'report.json',report);render_report(report,out)
    compact={k:v for k,v in report.items() if k!='records'}
    compact.update(report_path=str(out/'report.json'),report_sha256=sha(out/'report.json'))
    write(out/'completion.json',compact)
    print({'state':report['state'],'main_id':frozen['main']['id'],'macro10':report['macro10_class_miou'],
        'paired_macro':paired['macro10'],'report':str(out/'REPORT.md')},flush=True)


def self_test():
    """Synthetic-only checks; no real manifest, GT, masks, or predictions read."""
    import numpy as np
    toy=[{'index':'a','dataset':'coco','row':{'query_path':'synthetic/alias1.jpg','query_photo_id':'10','source_files':{'synthetic/alias1.jpg':'same_rgb_hash'}}},
         {'index':'b','dataset':'lvis','row':{'query_path':'synthetic/alias2.jpg','query_photo_id':'20','source_files':{'synthetic/alias2.jpg':'same_rgb_hash'}}},
         {'index':'c','dataset':'deepglobe_road','row':{'query_path':'synthetic/ref/30.jpg','query_photo_id':'30','source_files':{}}},
         {'index':'d','dataset':'deepglobe_road','row':{'query_path':'synthetic/query/30.jpg','query_photo_id':'30','source_files':{}}}]
    photo,meta=canonical_photos(toy)
    assert photo['a']==photo['b'] and photo['c']==photo['d'] and photo['a']!=photo['c']
    synthetic_gt=ROOT/'synthetic-confirm-gt-never-opened.png'
    with deny_query_gt([{'row':{'query_mask_path':str(synthetic_gt)}}]):
        try:synthetic_gt.read_bytes()
        except RuntimeError as error:assert 'GT is forbidden' in str(error)
        else:raise AssertionError('Metadata mode did not deny query GT')
    g=np.packbits(np.array([1,1,0,0,1,0,0,0],bool));v=np.packbits(np.array([1,1,1,1,0,0,0,0],bool))
    p=np.packbits(np.array([1,0,1,0,1,1,1,1],bool));m=packed_metric(p,g,v,2,4)
    assert (m['TP'],m['FP'],m['FN'],m['GT'],m['predicted'])==(1,1,1,2,2)
    vals=[(1,2),(1,10),(8,10)];rs=[]
    for d in ('coco','lvis'):
        for j,(inter,union) in enumerate(vals):
            metric={'I':inter,'U':union,'TP':inter,'FP':union-inter,'FN':0,'GT':inter,'predicted':union,'IoU':100*inter/union,
                    'target_fraction':.1,'abs_log2_area_error':0.,'zero_predicted_area':False}
            rs.append({'index':d+str(j),'dataset':d,'fold':0,'class_id':0 if j<2 else 1,
                'query_photo':'shared'+str(j),'metrics':{'main':metric,'FoRIS':dict(metric)}})
    s=point_summary(rs,('main','FoRIS'));assert abs(s['coco']['main']['class_miou']-(100*2/12+80)/2)<1e-12
    boot=global_photo_bootstrap(rs,('main','FoRIS'),s,repeats=100,debug=True)
    assert np.array_equal(boot['_samples']['coco']['main'],boot['_samples']['lvis']['main'])
    assert boot['macro10']['FoRIS']['paired_canonical_photo_bootstrap95_pp']==[0.,0.]
    return {'state':'SYNTHETIC_TESTS_PASS','checks':['canonical_cross_domain_photo_alias','road_reference_query_copy_alias',
        'metadata_mode_denies_query_GT','ignore_aware_packed_counts','class_slot_pooling_not_episode_average','global_shared_cross_domain_bootstrap_weights','paired_identity_zero_CI'],
        'real_query_GT_reads':0,'real_mask_reads':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inspect','template','preflight','score','attach-timing','self-test'),nargs='?',default='inspect')
    parser.add_argument('--freeze',type=Path);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--verify-predictions',action='store_true',help='Preflight only: verify all natural mask hashes/keys without query GT')
    parser.add_argument('--timing-evidence',type=Path,help='attach-timing only: JSON list of completed timing report descriptors')
    args=parser.parse_args();args.out=args.out.resolve()
    if args.mode=='inspect':print(json.dumps(inspection(args.out),ensure_ascii=False),flush=True)
    elif args.mode=='template':print(json.dumps(freeze_template(),indent=2,ensure_ascii=False),flush=True)
    elif args.mode=='self-test':print(json.dumps(self_test(),ensure_ascii=False),flush=True)
    elif args.mode=='preflight':print(json.dumps(preflight(args.freeze,args.out,args.verify_predictions)[3],ensure_ascii=False),flush=True)
    elif args.mode=='attach-timing':print(json.dumps(attach_timing(args.out,args.timing_evidence),ensure_ascii=False),flush=True)
    else:score(args.freeze,args.out)
