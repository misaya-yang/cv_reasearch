"""Optional fixed reference-rho gate on the original35 exposed soft programs.

Freeze routing from reference statistics, reaggregate existing per-case scores,
count edit budgets on existing masks, then materialize/check the joint-development
winner. No threshold search, fitting, encoding, or confirmation-data access.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import x1012s2_soft500 as shared
import numpy as np

ROOT, ASSETS, pc, pr, lab = shared.ROOT, shared.ASSETS, shared.pc, shared.pr, shared.lab
read, write, sha = shared.read, shared.write, shared.sha
OUT=ASSETS/'a/x1012s2_reference_gate_composition'
DEV=ASSETS/'a/x1012s2_soft_composition'
FIVE=shared.OUT/'all35'
NETWORK_DEV=ASSETS/'a/x1012s2_composition_search/boolean_network_v2/materialized/fce0fce0fce0fce0_fee0fee0fee0fcc0'
NETWORK_FIVE=shared.NETWORK
T_ARM='b0m0u0t4'
U_ARM='b0m0u4t0'
EDITS=('recovered_TP','added_FP','deleted_TP','removed_FP')
LUT=np.asarray([i.bit_count() for i in range(256)],np.uint8)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'frozen').mkdir(exist_ok=True)
    path=OUT/'config.json'
    if not path.exists():
        formulas=read(DEV/'config.json')['formulas'];assert len(formulas)==35
        cohorts={}
        dependencies=[DEV/'config.json',DEV/'report.json',FIVE/'report.json',FIVE/'config.json',Path(shared.__file__)]
        for cohort,root,ref_root,network in (
            ('development1360',DEV,shared.base.R1,NETWORK_DEV),
            ('development500',FIVE,shared.base.R5C,NETWORK_FIVE)):
            source_seal=read(root/'sealed.json')
            src={r['index']:r for r in source_seal['records']}
            refs={r['index']:r for r in read(ref_root/'sealed.json')['records']}
            net_seal=read(network/'sealed.json');nets={r['index']:r for r in net_seal['records']}
            manifest=read(DEV/'manifest.json') if cohort=='development1360' else [r['binding'] for r in read(FIVE/'bindings.json')]
            rows=[]
            for b in manifest:
                key=pc.key(b);p,n=src[key],nets[key];rho=float(refs[key]['reference_rho'])
                rows.append({'index':key,'dataset':b['dataset'],'reference_rho':rho,'gate_true':rho>.9,
                    'source_prediction_path':p['prediction_path'],'source_prediction_sha256':p['prediction_sha256'],
                    'network_prediction_path':n.get('path',n.get('prediction_path')),
                    'network_prediction_sha256':n.get('sha256',n.get('prediction_sha256')),
                    'network_keys':({'network4':'fce0fce0fce0fce0','network6':'fee0fee0fee0fcc0'} if cohort=='development1360' else {'network4':'network4','network6':'network6'}),
                    'binding':b})
            cohorts[cohort]=rows
            dependencies += [root/'sealed.json',ref_root/'sealed.json',network/'sealed.json']
        write(OUT/'bindings.json',cohorts)
        config={'state':'ALL35_REFERENCE_GATE_PROGRAMS_FROZEN_BEFORE_SCORE_REAGGREGATION',
            'formula_n':35,'formulas':formulas,'gate':'existing reference_rho > .9; strict, no search',
            'routing':'if gate true use existing original T mask, else use the unchanged parent soft formula',
            'T_identity':'Each cohort keeps its already sealed soft-source T. Development1360 original P0 T can have known CPU/MPS one-pixel differences from R1 U fallback; record controls, never patch masks.',
            'optional_operator':True,'ungated_programs_preserved':True,'domain_branch':False,
            'source_of_gate':'reference statistics already sealed by R1/R5, no query label or score used',
            'ranking_for_materialization':'maximum arithmetic average of the two separate equal-ten-domain development means; deterministic lexical parent-arm tie-break; no per-domain selection',
            'scope':'Both cohorts exposed development; no new blind validation; confirm1000 not accessed',
            'source_cost':'gate true: compute reference rho and independent c32/T only; skip MEAN, FoRIS score/final CRF, VP and UN graph. Gate false: parent dependencies, shared work counted once.',
            'paired_CI':'1000 paired query-photo cluster bootstrap per cohort/domain; seed20261013; fixed-program development description, not selection-adjusted inference',
            'new_encoder_calls':0,'training':False,'rho_grid_expansion':False,
            'source_sha256':sha(__file__),'input_sha256':{str(p):sha(p) for p in dependencies},
            'bindings_sha256':sha(OUT/'bindings.json'),'resources':{'device':'cpu','workers':1,'threads':1}}
        write(path,config);shutil.copyfile(__file__,OUT/'frozen/x1012s2_reference_gate_composition.py')
    config=read(path)
    assert sha(__file__)==config['source_sha256']
    assert all(sha(p)==h for p,h in config['input_sha256'].items())
    assert sha(OUT/'bindings.json')==config['bindings_sha256']
    return read(OUT/'bindings.json'),config


def get_reports():
    a,b=read(DEV/'report.json'),read(FIVE/'report.json')
    return {'development1360':{r['index']:r for r in a['records']},
            'development500':{r['index']:r for r in b['records500']}}


def point(rows,arm):
    slots=defaultdict(lambda:[0,0])
    for r in rows:
        m=r['metrics'][arm];s=slots[(str(r['fold']),str(r['class_id']))];s[0]+=m['I'];s[1]+=m['U']
    metrics=[r['metrics'][arm] for r in rows]
    return {'n':len(rows),'class_slot_n':len(slots),
        'class_miou':float(np.mean([100*i/max(u,1) for i,u in slots.values()])),
        'episode_mean_iou':float(np.mean([m['IoU'] for m in metrics])),
        'FN_GT':sum(m['FN'] for m in metrics)/max(sum(m['GT'] for m in metrics),1),
        'FP_GT':sum(m['FP'] for m in metrics)/max(sum(m['GT'] for m in metrics),1)}


def paired(rows,names,summary):
    from scipy.sparse import csr_matrix
    rng=np.random.default_rng(20261013);out={};draws=defaultdict(list)
    for d in lab.ALL_DOMAINS:
        rs=[r for r in rows if r['dataset']==d]
        photos=sorted({r['query_photo'] for r in rs});slots=sorted({(str(r['fold']),str(r['class_id'])) for r in rs})
        pm,sm={p:i for i,p in enumerate(photos)},{s:i for i,s in enumerate(slots)}
        pi=[pm[r['query_photo']] for r in rs];si=[sm[(str(r['fold']),str(r['class_id']))] for r in rs]
        shape=(len(photos),len(slots));presence=csr_matrix((np.ones(len(rs)),(pi,si)),shape=shape)
        weights=rng.multinomial(len(photos),np.full(len(photos),1/len(photos)),size=1000)
        present=(weights@presence)>0;out[d]={}
        for a in names:
            samples=[]
            for label in (a,'G_'+a):
                ii=csr_matrix(([r['metrics'][label]['I'] for r in rs],(pi,si)),shape=shape)
                uu=csr_matrix(([r['metrics'][label]['U'] for r in rs],(pi,si)),shape=shape)
                samples.append(((100*(weights@ii)/np.maximum(weights@uu,1))*present).sum(1)/present.sum(1))
            delta=samples[1]-samples[0];draws[a].append(delta)
            out[d][a]={'delta_pp':summary[d]['G_'+a]['class_miou']-summary[d][a]['class_miou'],
                'paired_query_photo_bootstrap95_pp':np.quantile(delta,[.025,.975]).tolist()}
    macro={a:{'delta_pp':float(np.mean([out[d][a]['delta_pp'] for d in lab.ALL_DOMAINS])),
        'paired_query_photo_bootstrap95_pp':np.quantile(np.mean(draws[a],axis=0),[.025,.975]).tolist()} for a in names}
    return {'domains':out,'macro10':macro,'seed':20261013,'repeats':1000}


def popcount(x):return int(LUT[x].sum(dtype=np.uint64))


def count_edits_and_controls(bindings,names):
    totals={};records=[];identity=[];gt_reads=0
    for cohort,rows in bindings.items():
        sums={d:{a:[0,0,0,0] for a in names} for d in lab.ALL_DOMAINS}
        for row in rows:
            if not row['gate_true']:continue
            assert sha(row['source_prediction_path'])==row['source_prediction_sha256']
            assert sha(row['network_prediction_path'])==row['network_prediction_sha256']
            with np.load(row['source_prediction_path'],allow_pickle=False) as z:
                t=z[T_ARM].copy();u=z[U_ARM].copy()
                with np.load(row['network_prediction_path'],allow_pickle=False) as n:
                    xor={a:popcount(n[k]^t) for a,k in row['network_keys'].items()}
                ut=popcount(u^t)
                if ut==0:assert all(x==0 for x in xor.values()),(cohort,row['index'],'Boolean T-identity failure when U=T')
                identity.append({'cohort':cohort,'index':row['index'],'dataset':row['dataset'],
                    'U_vs_T_xor':ut,'network_vs_T_xor':xor})
                gt,valid=pc.truth(row['binding']);gt_reads+=1;g,v=pc.pack(gt),pc.pack(valid);target=t&v
                for a in names:
                    p=z[a]&v;add,delete=target&~p,p&~target
                    at,dt=popcount(add&g),popcount(delete&g)
                    vector=[at,popcount(add)-at,dt,popcount(delete)-dt]
                    sums[row['dataset']][a]=[old+x for old,x in zip(sums[row['dataset']][a],vector)]
                    records.append({'cohort':cohort,'index':row['index'],'parent_arm':a,'edits':vector})
        totals[cohort]=sums
    write(OUT/'gate_case_edit_counts.json',records)
    write(OUT/'boolean_identity_controls.json',{'n':len(identity),'records':identity,
        'network4_xor':sum(r['network_vs_T_xor']['network4'] for r in identity),
        'network6_xor':sum(r['network_vs_T_xor']['network6'] for r in identity),
        'U_T_xor':sum(r['U_vs_T_xor'] for r in identity),
        'exceptions_preserved_not_repaired':True})
    return totals,gt_reads


def cost(weights,n,gate_n):
    direct=[s for s in ('B','M','U','T') if weights[s]>0]
    return {'n':n,'gate_true_n':gate_n,'gate_false_n':n-gate_n,'gate_true_fraction':gate_n/n,
        'gate_true_direct_sources':['T'],'gate_false_direct_sources':direct,
        'direct_source_invocations':{s:(gate_n if s=='T' else 0)+(n-gate_n if s in direct else 0) for s in ('B','M','U','T')},
        'component_invocations':{
            'reference_rho':n,'independent_c32':gate_n+(n-gate_n if weights['U']>0 or weights['T']>0 else 0),
            'FoRIS_score64':n-gate_n if weights['B']>0 or weights['M']>0 else 0,
            'FoRIS_final_CRF':n-gate_n if weights['B']>0 else 0,
            'MEAN_rank_and_query_graph':n-gate_n if weights['M']>0 else 0,
            'VP_component_signals':n-gate_n if weights['U']>0 else 0,
            'UN_reliability_graph':n-gate_n if weights['U']>0 else 0},
        'latency_note':'Logical source/component invocation counts, not an end-to-end latency measurement; sharing and actual early routing must be implemented and timed.'}


def score():
    bindings,config=prepare();names=list(config['formulas']);reports=get_reports()
    began=time.monotonic();cohorts={};all_rows={}
    for cohort,bs in bindings.items():
        old=reports[cohort];rs=[]
        for b in bs:
            r=old[b['index']];metrics={a:r['metrics'][a] for a in names}
            metrics.update({'G_'+a:r['metrics'][T_ARM if b['gate_true'] else a] for a in names})
            metrics.update(FoRIS=r['metrics']['FoRIS'],MEAN=r['metrics']['MEAN'],
                U=r['metrics'].get('U',r['metrics'].get('UN_midrange')),T=r['metrics']['T'])
            rs.append({k:r[k] for k in ('index','dataset','fold','class_id','query_photo')}|{'metrics':metrics})
        summary={d:{a:point([r for r in rs if r['dataset']==d],a) for a in ('FoRIS','MEAN','U','T')+tuple(names)+tuple('G_'+a for a in names)} for d in lab.ALL_DOMAINS}
        means={a:float(np.mean([t[a]['class_miou'] for t in summary.values()])) for a in summary[lab.ALL_DOMAINS[0]]}
        paired_result=paired(rs,names,summary)
        gate_counts=Counter(b['dataset'] for b in bs if b['gate_true'])
        cohorts[cohort]={'n':len(bs),'gate_true_n':sum(gate_counts.values()),'gate_true_by_domain':dict(gate_counts),
            'summary':summary,'macro10':means,'paired_to_parent':paired_result,
            'parent_rank':sorted(names,key=lambda a:(-means[a],a)),
            'gated_rank':sorted(names,key=lambda a:(-means['G_'+a],a)),
            'source_costs':{a:cost(config['formulas'][a],len(bs),sum(gate_counts.values())) for a in names}}
        all_rows[cohort]=rs
    compare=[]
    for a in names:
        c={'parent_arm':a,'gated_arm':'G_'+a,'weights':config['formulas'][a]}
        for cohort in cohorts:
            m=cohorts[cohort]['macro10'];c[cohort]={'parent_macro':m[a],'gated_macro':m['G_'+a],
                'delta_to_parent_pp':m['G_'+a]-m[a],
                'paired_query_photo_bootstrap95_pp':cohorts[cohort]['paired_to_parent']['macro10'][a]['paired_query_photo_bootstrap95_pp']}
        c['equal_cohort_parent_macro']=float(np.mean([c[k]['parent_macro'] for k in cohorts]))
        c['equal_cohort_gated_macro']=float(np.mean([c[k]['gated_macro'] for k in cohorts]));compare.append(c)
    compare.sort(key=lambda r:(-r['equal_cohort_gated_macro'],r['parent_arm']))
    winner=compare[0]['parent_arm']
    print({'phase':'existing-integer-score-complete','winner':compare[0]},flush=True)
    edit_totals,gt_reads=count_edits_and_controls(bindings,names)
    for cohort,rs in all_rows.items():
        for d in lab.ALL_DOMAINS:
            ds=[r for r in rs if r['dataset']==d]
            for a in names:
                at,af,dt,df=edit_totals[cohort][d][a]
                assert sum(r['metrics']['G_'+a]['TP']-r['metrics'][a]['TP'] for r in ds)==at-dt
                assert sum(r['metrics']['G_'+a]['FP']-r['metrics'][a]['FP'] for r in ds)==af-df
    expected={cohort:[{k:r[k] for k in ('index','dataset','fold','class_id','query_photo')}|
        {'metrics':{'gated':r['metrics']['G_'+winner],'parent':r['metrics'][winner],'T':r['metrics']['T']}} for r in rs]
        for cohort,rs in all_rows.items()}
    write(OUT/'winner_case_metrics.json',expected)
    result={'state':'COMPLETE_FIXED35_REFERENCE_GATE_DEVELOPMENT_REAGGREGATION','formula_n':35,
        'winner_parent_arm':winner,'winner_gated_arm':'G_'+winner,'winner':compare[0],
        'comparison35':compare,'cohorts':cohorts,'edit_columns':EDITS,
        'edit_columns_zh':['补回TP','新增FP','删除TP','移除FP'],'edits_vs_parent':edit_totals,
        'config_sha256':sha(OUT/'config.json'),'bindings_sha256':sha(OUT/'bindings.json'),
        'winner_metrics_sha256':sha(OUT/'winner_case_metrics.json'),
        'identity_controls':read(OUT/'boolean_identity_controls.json'),
        'raw_GT_reads_for_offline_edit_counts':gt_reads,'score_metrics_from_existing_integers':True,
        'new_encoder_calls':0,'training':False,'rho_threshold_searched':False,
        'domain_specific_formula':False,'confirm1000_accessed':False,'independent_confirmation':False,
        'ungated_programs_preserved':True,'score_seconds':time.monotonic()-began,
        'exposure':config['scope'],'full_pipeline_latency_measured_here':False}
    write(OUT/'report.json',result)
    write(ROOT/'evidence/research/x1012s2_reference_gate_composition.json',result)
    print({'state':result['state'],'winner':compare[0],'seconds':round(time.monotonic()-began,1)},flush=True)


def materialize():
    bindings,config=prepare();report=read(OUT/'report.json');winner=report['winner_parent_arm']
    assert report['config_sha256']==sha(OUT/'config.json')
    records=[];start=time.monotonic()
    for cohort,rows in bindings.items():
        for row in rows:
            dest=OUT/'materialized'/cohort/'predictions'/(row['index']+'.npz')
            dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            with pc.inference_only(row['binding']):
                assert sha(row['source_prediction_path'])==row['source_prediction_sha256']
                key=T_ARM if row['gate_true'] else winner
                with np.load(row['source_prediction_path'],allow_pickle=False) as z:mask=z[key].copy()
                np.savez_compressed(dest,gated=mask)
            records.append({'cohort':cohort,'index':row['index'],'gate_true':row['gate_true'],
                'prediction_path':str(dest),'prediction_sha256':sha(dest),
                'source_path':row['source_prediction_path'],'source_sha256':row['source_prediction_sha256'],
                'selected_source_key':key,'new_encoder_calls':0,'query_GT_reads':0})
    seal={'state':'JOINT_DEVELOPMENT_WINNER_ACTUAL_MASKS_SEALED','n':len(records),
        'parent_arm':winner,'gated_arm':'G_'+winner,'records':records,
        'config_sha256':sha(OUT/'config.json'),'search_report_sha256':sha(OUT/'report.json'),
        'query_GT_reads_during_materialization':0,'seconds':time.monotonic()-start}
    write(OUT/'materialized/sealed.json',seal)
    expected=read(OUT/'winner_case_metrics.json')
    lookup={(c,r['index']):r for c,rows in bindings.items() for r in rows}
    e={(c,r['index']):r['metrics']['gated'] for c,rows in expected.items() for r in rows}
    checked=[]
    for r in records:
        assert sha(r['prediction_path'])==r['prediction_sha256']
        with np.load(r['prediction_path'],allow_pickle=False) as z:p=pc.unpack(z['gated'])
        gt,valid=pc.truth(lookup[(r['cohort'],r['index'])]['binding'])
        actual=pc.pixel_metrics(p,gt,valid)
        assert all(actual[k]==e[(r['cohort'],r['index'])][k] for k in ('I','U','TP','FP','FN','GT','predicted')),(r['cohort'],r['index'])
        checked.append({'cohort':r['cohort'],'index':r['index'],'I':actual['I'],'U':actual['U']})
    verification={'state':'ALL1860_ACTUAL_MASKS_MATCH_REAGGREGATED_INTEGER_METRICS','n':len(checked),
        'records':checked,'seal_sha256':sha(OUT/'materialized/sealed.json'),'new_encoder_calls':0,
        'training':False,'confirm1000_accessed':False,'raw_GT_role':'post-materialization verification only'}
    write(OUT/'materialized/verification.json',verification)
    write(OUT/'completion.json',{'state':'COMPLETE','winner':report['winner'],'materialized_n':len(records),
        'report_sha256':sha(OUT/'report.json'),'materialized_seal_sha256':verification['seal_sha256'],
        'verification_sha256':sha(OUT/'materialized/verification.json'),'config_sha256':sha(OUT/'config.json'),
        'confirm1000_accessed':False,'new_encoder_calls':0,'training':False})
    print({'state':verification['state'],'n':len(checked),'winner':winner,
           'seconds':round(time.monotonic()-start,1)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('freeze','score','materialize'))
    args=parser.parse_args()
    if args.mode=='freeze':
        rows,config=prepare();print({'state':config['state'],'n':sum(map(len,rows.values()))},flush=True)
    else:globals()[args.mode]()
