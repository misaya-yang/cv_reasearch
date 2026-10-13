"""Complete second exposed-development scoring of the original35 soft formulas.

The two development cohorts remain distinct. No new grid, formula, fitted
parameter, domain-specific choice, or access to the unexposed confirmation set.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import shutil
import time
import x1012s2_soft500 as one
import numpy as np
import torch

ROOT, ASSETS, pc, pr, lab = one.ROOT, one.ASSETS, one.pc, one.pr, one.lab
read, write, sha = one.read, one.write, one.sha
PARENT = one.OUT
OUT = PARENT/'all35'
DEVELOP = ASSETS/'a/x1012s2_soft_composition'
SYMBOLS = ('B','M','U','T')
VERTICES={'B':'b4m0u0t0','M':'b0m4u0t0','U':'b0m0u4t0','T':'b0m0u0t4'}
ALIASES={'B':'FoRIS','M':'MEAN','U':'UN_midrange','T':'T'}
EDITS=('recovered_TP','added_FP','deleted_TP','removed_FP')
LUT=np.asarray([i.bit_count() for i in range(256)],np.uint8)


def initialize():
    torch.set_num_threads(1)
    assert torch.get_num_threads()==1 and lab.DEV.type=='cpu'


def prepare():
    for folder in ('predictions','records','frozen'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    path=OUT/'config.json'
    if not path.exists():
        old=read(DEVELOP/'config.json')
        assert old['source_order']==list(SYMBOLS) and old['mixtures']==35
        formulas=old['formulas']
        assert len(formulas)==35
        for a,w in formulas.items():
            assert set(w)==set(SYMBOLS) and sum(w.values())==1 and all(x in (0.,.25,.5,.75,1.) for x in w.values())
        source_check=read(PARENT/'sources_sealed.json')
        assert source_check['n']==500 and source_check['xor_total']==0 and source_check['source_comparisons']==1500
        rows=read(PARENT/'bindings.json')
        prior={r['index']:r for r in read(PARENT/'sealed.json')['records']}
        for row in rows:row['previous_single_formula']=prior[row['index']]
        write(OUT/'bindings.json',rows)
        deps=[DEVELOP/'config.json',DEVELOP/'report.json',DEVELOP/'sealed.json',
              PARENT/'source_contract.json',PARENT/'sources_sealed.json',PARENT/'bindings.json',
              PARENT/'config.json',PARENT/'sealed.json',PARENT/'report.json',
              one.NETWORK/'sealed.json',Path(one.__file__)]
        config={'state':'ORIGINAL35_FORMULAS_FROZEN_FOR_SECOND_EXPOSED_DEVELOPMENT_COHORT',
            'n':500,'formula_n':35,'source_order':SYMBOLS,'formulas':formulas,
            'grid_modified':False,'source_contract':'exactly prior1500 verified M/U/T frames; B is existing complete FoRIS binary0/1',
            'arithmetic':'FP64 integer quarter weights0..4, strict sum>2, no new minmax/clip/CRF',
            'singletons':'All B/M/U/T pure masks must reproduce existing natural masks; prior fixed b0m1u2t1 must reproduce the preserved single-formula500 output.',
            'evaluation_scope':'Both1360 and500 are exposed development. This is complete development-family evaluation, not another independent validation.',
            'reporting':'Separate equal-ten-domain means and domain distributions for both cohorts; equal-cohort mean is descriptive, not a new confirmation statistic. No final deployment selection in this task.',
            'all17500_natural_masks_sealed_before_GT':True,'new_encoder_calls':0,'training':False,
            'domain_specific_formula':False,'confirm1000_accessed':False,
            'resources':{'device':'cpu','workers':2,'threads_per_worker':1},
            'source_sha256':sha(__file__),'input_sha256':{str(p):sha(p) for p in deps},
            'bindings_sha256':sha(OUT/'bindings.json')}
        write(path,config);shutil.copyfile(__file__,OUT/'frozen/x1012s2_soft500_all35.py')
    config=read(path)
    assert sha(__file__)==config['source_sha256']
    assert all(sha(p)==h for p,h in config['input_sha256'].items())
    assert sha(OUT/'bindings.json')==config['bindings_sha256']
    return read(OUT/'bindings.json'),config


def produce_one(job):
    row,formulas=job;key=row['index'];began=time.monotonic()
    path=OUT/'predictions'/(key+'.npz');assert not path.exists(),'Preserve partial all35 output'
    names=list(formulas)
    weights=np.asarray([[round(formulas[a][s]*4) for s in SYMBOLS] for a in names],np.float64)
    with pc.inference_only(row['binding']):
        maps,natural=one.frames(row)
        arrays=np.stack([pc.unpack(natural['B']).astype(np.float32),maps['M'],maps['U'],maps['T']]).reshape(4,-1)
        masks=np.empty((35,1024*1024//8),np.uint8)
        for start in range(0,1024*1024,65536):
            stop=min(start+65536,1024*1024)
            masks[:,start//8:stop//8]=np.packbits((weights@arrays[:,start:stop].astype(np.float64))>2.,axis=1)
        for s,a in VERTICES.items():assert np.array_equal(masks[names.index(a)],natural[s]),(key,s,'pure source identity')
        previous=row['previous_single_formula'];assert sha(previous['prediction_path'])==previous['prediction_sha256']
        with np.load(previous['prediction_path'],allow_pickle=False) as z:
            assert np.array_equal(masks[names.index(one.WINNER)],z[one.WINNER]),(key,'Preserved single-formula result differs')
        np.savez_compressed(path,**{a:masks[i] for i,a in enumerate(names)})
    record={'index':key,'dataset':row['dataset'],'prediction_path':str(path),'prediction_sha256':sha(path),
            'source_singletons_mask_xor':dict.fromkeys(SYMBOLS,0),'previous_single_formula_mask_xor':0,
            'seconds':time.monotonic()-began,'query_GT_reads':0,'new_encoder_calls':0}
    write(OUT/'records'/(key+'.json'),record);return record


def produce():
    rows,config=prepare();assert not (OUT/'sealed.json').exists(),'Already sealed'
    began,records=time.monotonic(),[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,r in enumerate(pool.map(produce_one,((row,config['formulas']) for row in rows),chunksize=1),1):
            records.append(r)
            if i<=2 or i%50==0:print({'phase':'all35-natural','completed':i,'n':500,'seconds':round(time.monotonic()-began,1)},flush=True)
    seal={'state':'ALL500_ALL35_NATURAL_MASKS_SEALED','n':500,'formula_n':35,'mask_n':17500,
        'records':records,'config_sha256':sha(OUT/'config.json'),'bindings_sha256':sha(OUT/'bindings.json'),
        'source_parent_seal_sha256':sha(PARENT/'sources_sealed.json'),'seconds':time.monotonic()-began,
        'workers':2,'threads_per_worker':1,'query_GT_reads':0,'new_encoder_calls':0,
        'previous_single_formula_total_xor':sum(r['previous_single_formula_mask_xor'] for r in records)}
    write(OUT/'sealed.json',seal)
    print({'state':seal['state'],'mask_n':17500,'seal_sha256':sha(OUT/'sealed.json')},flush=True)


def popcount(x):return int(LUT[x].sum(dtype=np.uint64))


def metric(bits,g,v,ng,nv):
    p=bits&v;tp,n=popcount(p&g),popcount(p);fp,fn=n-tp,ng-tp
    error=float(abs(np.log2(n/ng))) if n and ng else (0. if not ng and not n else None)
    return {'I':tp,'U':tp+fp+fn,'TP':tp,'FP':fp,'FN':fn,'GT':ng,'predicted':n,
            'IoU':100*tp/max(tp+fp+fn,1),'target_fraction':ng/max(nv,1),
            'abs_log2_area_error':error,'infinite_area_error':error is None,'zero_predicted_area':n==0}


def edits(pred,baseline,g,v):
    p,b=pred&v,baseline&v;add,delete=p&~b,b&~p
    at,dt=popcount(add&g),popcount(delete&g)
    return [at,popcount(add)-at,dt,popcount(delete)-dt]


def score_one(job):
    row,receipt,names=job;b=row['binding'];key=row['index']
    assert sha(receipt['prediction_path'])==receipt['prediction_sha256']
    gt,valid=pc.truth(b);g,v=pc.pack(gt),pc.pack(valid);ng,nv=int(gt.sum()),int(valid.sum())
    controls={}
    for a in ('network4','r00097'):
        source=row[a];assert sha(source['path'])==source['sha256']
        with np.load(source['path'],allow_pickle=False) as z:controls[a]=z[source['key']].copy()
    metrics,edit_b,edit_n={},{},{}
    with np.load(receipt['prediction_path'],allow_pickle=False) as z:
        baseline=z[VERTICES['B']].copy()
        for a in names:
            p=z[a];metrics[a]=metric(p,g,v,ng,nv)
            edit_b[a]=edits(p,baseline,g,v);edit_n[a]=edits(p,controls['network4'],g,v)
        for s,a in VERTICES.items():metrics[ALIASES[s]]=dict(metrics[a])
    for a,p in controls.items():metrics[a]=metric(p,g,v,ng,nv)
    return {'index':key,'dataset':b['dataset'],'fold':b['row']['fold'],'class_id':b['row']['global_class_id'],
        'query_photo':str((ASSETS/b['row']['query_path']).resolve()),'metrics':metrics,
        'edits_vs_FoRIS':edit_b,'edits_vs_network4':edit_n}


def dependency_info(weights):
    direct=[s for s in SYMBOLS if weights[s]>0]
    components=set()
    for s in direct:
        if s=='B':components.update(('FoRIS_score64','FoRIS_final_CRF'))
        elif s=='M':components.update(('FoRIS_score64','MEAN_rank_and_query_graph'))
        elif s=='U':components.update(('independent_c32','VP_component_signals','UN_reliability_graph'))
        elif s=='T':components.add('independent_c32')
    return {'direct_sources':direct,'source_components_with_shared_work_deduplicated':sorted(components),
        'requires_complete_FoRIS_CRF':weights['B']>0,
        'requires_FoRIS_score64':weights['B']>0 or weights['M']>0,
        'raw_DINO_source':'same cached R/Q O24, no extra encoder calls',
        'note':'M depends on FoRIS score even when B mixture weight is0. Listed components are dependency identifiers, not summed wall-time estimates.'}


def score():
    rows,config=prepare();names=list(config['formulas']);seal=read(OUT/'sealed.json')
    assert seal['n']==500 and seal['mask_n']==17500 and seal['config_sha256']==sha(OUT/'config.json')
    receipts={r['index']:r for r in seal['records']}
    assert all(sha(r['prediction_path'])==r['prediction_sha256'] for r in receipts.values())
    began,records=time.monotonic(),[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,r in enumerate(pool.map(score_one,((row,receipts[row['index']],names) for row in rows),chunksize=1),1):
            records.append(r)
            if i<=2 or i%50==0:print({'phase':'all35-dev500-score','completed':i,'n':500,'seconds':round(time.monotonic()-began,1)},flush=True)
    prior={r['index']:r for r in read(PARENT/'report.json')['records']}
    for r in records:
        for a in ('FoRIS','MEAN','UN_midrange','T','r00097','network4',one.WINNER):
            assert all(r['metrics'][a][k]==prior[r['index']]['metrics'][a][k] for k in ('I','U','TP','FP','FN','GT','predicted')),(r['index'],a,'preserved score identity')
    all_arms=('FoRIS','MEAN','UN_midrange','T','r00097','network4')+tuple(names)
    summary=pr.summary(records,all_arms)
    for t in summary.values():
        for val in t.values():val.pop('paired_query_photo_bootstrap95_pp',None)
    means500={a:float(np.mean([t[a]['class_miou'] for t in summary.values()])) for a in all_arms}
    dev=read(DEVELOP/'report.json')
    comparison=[]
    for a in names:
        m1360=dev['equal_domain_means'][a];m500=means500[a]
        comparison.append({'arm':a,'weights':config['formulas'][a],**dependency_info(config['formulas'][a]),
            'macro1360':m1360,'macro500':m500,'equal_cohort_average_macro':.5*(m1360+m500),
            'delta_to_B1360':m1360-dev['equal_domain_means']['FoRIS'],'delta_to_B500':m500-means500['FoRIS'],
            'delta_to_U1360':m1360-dev['equal_domain_means']['U'],'delta_to_U500':m500-means500['UN_midrange'],
            'delta_to_network4_500':m500-means500['network4'],
            'domains':{d:{'class_miou1360':dev['summary'][d][a]['class_miou'],
                'class_miou500':summary[d][a]['class_miou'],
                'delta_to_B1360':dev['summary'][d][a]['class_miou']-dev['summary'][d]['FoRIS']['class_miou'],
                'delta_to_B500':summary[d][a]['class_miou']-summary[d]['FoRIS']['class_miou'],
                'delta_to_network4_500':summary[d][a]['class_miou']-summary[d]['network4']['class_miou']}
                for d in lab.ALL_DOMAINS}})
    rank1360=sorted(names,key=lambda a:(-dev['equal_domain_means'][a],a))
    rank500=sorted(names,key=lambda a:(-means500[a],a))
    ranked_both=sorted(comparison,key=lambda r:(-r['equal_cohort_average_macro'],r['arm']))
    sums={key:{d:{a:np.sum([r[key][a] for r in records if r['dataset']==d],axis=0).tolist() for a in names}
                   for d in lab.ALL_DOMAINS} for key in ('edits_vs_FoRIS','edits_vs_network4')}
    result={'state':'COMPLETE_ORIGINAL35_SECOND_EXPOSED_DEVELOPMENT_SEARCH','n':500,'formula_n':35,
        'summary500':summary,'records500':records,'macro500':means500,'comparison35':comparison,
        'rank1360':rank1360,'rank500':rank500,'rank_equal_cohort_average':[r['arm'] for r in ranked_both],
        'equal_cohort_average_definition':'Arithmetic mean of two separate equal-ten-domain development means; descriptive ranking, not pooled1860 or independent confirmation.',
        'prior_single_formula_preserved':True,'source_and_previous_singleton_score_identity':True,
        'edit_columns':EDITS,'edit_columns_zh':['补回TP','新增FP','删除TP','移除FP'],**sums,
        'config_sha256':sha(OUT/'config.json'),'seal_sha256':sha(OUT/'sealed.json'),
        'development1360_report_sha256':sha(DEVELOP/'report.json'),
        'score_seconds':time.monotonic()-began,'new_encoder_calls':0,'training':False,
        'domain_specific_formula':False,'new_blind_confirmation':False,'confirm1000_accessed':False,
        'final_method_selected_here':False,'full_pipeline_latency_measured_here':False,
        'exposure':config['evaluation_scope']}
    write(OUT/'report.json',result)
    compact={k:v for k,v in result.items() if k!='records500'}
    compact.update(report_path=str(OUT/'report.json'),report_sha256=sha(OUT/'report.json'))
    write(OUT/'completion.json',compact);write(ROOT/'evidence/research/x1012s2_soft500_all35.json',compact)
    lines=['# 原35个连续公式：两批曝光开发的完整并表','',
        '**1360和500均是曝光开发队列。本次完整评价相同35个预封存公式；不新增网格、不按域选权重、不改来源，不是再次独立验证。先封存全部17,500自然mask，再统一评分；新confirm1000未访问。**','',
        '此前只复现第一批赢家b0m1u2t1的结果按原字节保留；本次所有四个纯源及该先前公式的自然mask和评分整数全部复现。','',
        '|公式|B/M/U/T|直接来源|总组件依赖|1360十域均值|500十域均值|两批等权均值|500−network4|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for r in ranked_both:
        lines.append('|'+r['arm']+'|'+'/'.join(f'{r["weights"][s]:g}' for s in SYMBOLS)+'|'+','.join(r['direct_sources'])+'|'+','.join(r['source_components_with_shared_work_deduplicated'])+f"|{r['macro1360']:.6f}|{r['macro500']:.6f}|{r['equal_cohort_average_macro']:.6f}|{r['delta_to_network4_500']:+.6f}|")
    lines += ['', '**依赖不等于混合权重：M即使在B权重0时仍依赖FoRIS score64。各来源共享DINO与部分计算，不能把分支旧耗时直接相加；最终选择还需实测完整管线延迟。**',
        '', '两批等权均值仅将两个开发macro平均，不是把1860例池化，也不是新确认统计。完整每公式每域两批分数、对FoRIS/U/network4差值和500逐例增删整数均在JSON。',
        '', '## 三种开发排序首位的逐域取舍', '']
    picked=list(dict.fromkeys((rank1360[0],rank500[0],ranked_both[0]['arm'])))
    lookup={r['arm']:r for r in comparison}
    lines += ['|公式|域|1360分数|1360−B|500分数|500−B|500−network4|','|---|---|---:|---:|---:|---:|---:|']
    for a in picked:
        for d,t in lookup[a]['domains'].items():
            lines.append(f"|{a}|{d}|{t['class_miou1360']:.4f}|{t['delta_to_B1360']:+.4f}|{t['class_miou500']:.4f}|{t['delta_to_B500']:+.4f}|{t['delta_to_network4_500']:+.4f}|")
    lines += ['', '[全部35公式完整结果](../../evidence/research/x1012s2_soft500_all35.json)。',
        f'封存SHA `{result["seal_sha256"]}`；配置SHA `{result["config_sha256"]}`。',
        '本任务不自行冻结最终部署方法；两批开发结果与完整延迟应共同用于之后的选择。']
    (ROOT/'docs/research/SOFT35_TWO_DEVELOPMENT_COHORTS_20261013.md').write_text('\n'.join(lines)+'\n')
    print({'state':result['state'],'n':500,'formula_n':35,'best1360':rank1360[0],
        'best500':rank500[0],'best500_macro':means500[rank500[0]],
        'best_equal_cohort_average':ranked_both[0], 'seconds':round(time.monotonic()-began,1)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('freeze','produce','score'))
    args=parser.parse_args();initialize()
    if args.mode=='freeze':
        rows,config=prepare();print({'state':config['state'],'n':len(rows)},flush=True)
    elif args.mode=='produce':produce()
    else:score()
