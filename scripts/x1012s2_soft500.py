"""One fixed continuous winner on exposed500, with exact natural-source parity."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import time
import x1012s2_combo500 as base
import numpy as np
import torch

ROOT, ASSETS, pc, pr, lab = base.ROOT, base.ASSETS, base.pc, base.pr, base.lab
read, write, sha = base.read, base.write, base.sha
OUT = ASSETS/'a/x1012s2_soft500'
PARENT = base.OUT
NETWORK = PARENT/'network_v2'
SOURCE_CONTRACT = ROOT/'scripts/x1012s2_soft_composition.py'
WINNER = 'b0m1u2t1'
COMPARE = ('FoRIS', 'MEAN', 'UN_midrange', 'network4')
LUT = np.asarray([i.bit_count() for i in range(256)], np.uint8)


def prepare_sources():
    for folder in ('source_records', 'predictions', 'records', 'frozen'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    path = OUT/'source_contract.json'
    if not path.exists():
        rows = read(PARENT/'bindings.json')
        controls = {r['index']: r for r in read(base.R5B/'baseline/sealed.json')['records']}
        parents = {r['index']: r for r in read(PARENT/'sealed.json')['records']}
        networks = {r['index']: r for r in read(NETWORK/'sealed.json')['records']}
        assert len(rows) == len(controls) == len(parents) == len(networks) == 500
        for row in rows:
            name=row['index']; c,p,n=controls[name],parents[name],networks[name]
            row['M_field']={'path':c['mean_field_path'],'sha256':c['mean_field_sha256'],'key':'field'}
            row['sources']['T']={'path':p['prediction_path'],'sha256':p['prediction_sha256'],'key':'T'}
            row['r00097']={'path':p['prediction_path'],'sha256':p['prediction_sha256'],'key':'r00097'}
            row['network4']={'path':n['prediction_path'],'sha256':n['prediction_sha256'],'key':'network4'}
        write(OUT/'bindings.json',rows)
        parents_to_pin=[PARENT/'config.json',PARENT/'sealed.json',PARENT/'bindings.json',
                        NETWORK/'config.json',NETWORK/'sealed.json',base.R5B/'baseline/sealed.json',
                        base.R5C/'sealed.json',SOURCE_CONTRACT,Path(base.__file__)]
        contract={
            'state':'SOURCE_ONLY_READOUT_IDENTITY_CONTRACT_BEFORE_ANY_MIXTURE', 'n':500,
            'M':'saved complete R5 MEAN64 field -> CPU bilinear1024 float32 align_corners=False; no normalization or clipping',
            'U':'saved R5 VP_UN64 -> original CPU bilinear1024 then original minmax; .5 matches existing UN_midrange',
            'T':'saved R5 c32 -> original CPU bilinear1024 then original minmax; .5 matches parent-frozen original T',
            'identity_requirement':'All500 x 3 source strict>.5 masks must bitwise match existing natural M/U/T masks before formula freeze; no GT, mask repair, calibration or alternate readout selection.',
            'resources':{'device':'cpu','workers':1,'threads':1,'MPS':False},
            'query_GT_reads':0,'new_encoder_calls':0,'training':False,
            'source_sha256':sha(__file__),'bindings_sha256':sha(OUT/'bindings.json'),
            'parent_sha256':{str(p):sha(p) for p in parents_to_pin}}
        write(path,contract)
        shutil.copyfile(__file__,OUT/'frozen/x1012s2_soft500.py')
    contract=read(path)
    assert contract['source_sha256']==sha(__file__)
    assert all(sha(p)==h for p,h in contract['parent_sha256'].items())
    assert contract['bindings_sha256']==sha(OUT/'bindings.json')
    return read(OUT/'bindings.json'),contract


def frames(row):
    natural=base.load_existing(row)
    m=row['M_field']
    assert sha(m['path'])==m['sha256']
    assert sha(row['T_field_path'])==row['T_field_sha256']
    with np.load(m['path'],allow_pickle=False) as z:
        mean=lab.up(torch.as_tensor(z[m['key']],dtype=torch.float32).reshape(64,64)).numpy()
    with np.load(row['T_field_path'],allow_pickle=False) as z:
        un=lab.minmax(lab.up(torch.as_tensor(z['VP_UN'],dtype=torch.float32).reshape(64,64))).numpy()
        transfer=lab.minmax(lab.up(torch.as_tensor(z['c32'],dtype=torch.float32).reshape(64,64))).numpy()
    maps={'M':mean,'U':un,'T':transfer}
    for symbol,value in maps.items():
        assert value.shape==(1024,1024) and value.dtype==np.float32 and np.isfinite(value).all()
        xor=int(LUT[pc.pack(value>.5)^natural[symbol]].sum())
        assert xor==0,(row['index'],symbol,'pure-source readout differs',xor)
    return maps,natural


def verify():
    rows,contract=prepare_sources()
    assert not (OUT/'sources_sealed.json').exists(),'Sources already verified and sealed'
    began,records=time.monotonic(),[]
    for i,row in enumerate(rows,1):
        with pc.inference_only(row['binding']):
            maps,natural=frames(row)
        r={'index':row['index'],'dataset':row['dataset'],'pure_source_mask_xor':{s:0 for s in ('M','U','T')},
           'M_field':row['M_field'],'U_T_field_path':row['T_field_path'],'U_T_field_sha256':row['T_field_sha256'],
           'M_range':[float(maps['M'].min()),float(maps['M'].max())],
           'query_GT_reads':0,'new_encoder_calls':0}
        records.append(r);write(OUT/'source_records'/(row['index']+'.json'),r)
        if i<=2 or i%100==0:print({'phase':'source-identity','completed':i,'n':500,'seconds':round(time.monotonic()-began,1)},flush=True)
    write(OUT/'sources_sealed.json',{'state':'ALL1500_PURE_SOURCE_MASKS_BITWISE_IDENTICAL','n':500,
        'source_comparisons':1500,'xor_total':0,'records':records,'query_GT_reads':0,'new_encoder_calls':0,
        'source_contract_sha256':sha(OUT/'source_contract.json'),'bindings_sha256':sha(OUT/'bindings.json'),
        'wall_seconds':time.monotonic()-began})
    print({'state':'ALL1500_PURE_SOURCE_MASKS_BITWISE_IDENTICAL','n':500},flush=True)


def freeze():
    rows,contract=prepare_sources()
    sources=read(OUT/'sources_sealed.json')
    assert sources['n']==500 and sources['xor_total']==0
    assert sources['source_contract_sha256']==sha(OUT/'source_contract.json')
    config={'state':'ONE_WINNER_FROZEN_AFTER_SOURCE_IDENTITY_BEFORE_NATURAL_MIXTURE','n':500,
        'winner':WINNER,'weights':{'B':0.,'M':.25,'U':.5,'T':.25},
        'operation':'source_order B,M,U,T; integer weights0,1,2,1 multiplied into source float32 values after FP64 cast; strict sum>2 (equivalent weighted sum>.5); no new minmax, clipping or CRF',
        'selection':'Only the one requested1360-development winner; no new grid and no parameter changes using500',
        'exposure':'Already exposed R5_500 development reproduction; not blind confirmation; confirm1000 never accessed',
        'source_contract_sha256':sha(OUT/'source_contract.json'),'sources_seal_sha256':sha(OUT/'sources_sealed.json'),
        'bindings_sha256':sha(OUT/'bindings.json'),'source_sha256':sha(__file__),
        'query_GT_in_inference':False,'new_encoder_calls':0,'training':False,
        'paired_CI':'1000 same-query-photo cluster bootstrap per domain, seed20261013; equal-domain macro; fixed-rule descriptive intervals',
        'all500_natural_predictions_sealed_before_GT':True,'resources':contract['resources']}
    path=OUT/'config.json'
    if path.exists():assert read(path)==config
    else:write(path,config)
    return rows,config


def produce():
    rows,config=freeze()
    assert not (OUT/'sealed.json').exists(),'Preserve already sealed mixture'
    began,records=time.monotonic(),[]
    for i,row in enumerate(rows,1):
        path=OUT/'predictions'/(row['index']+'.npz')
        assert not path.exists(),'Preserve partial natural output'
        with pc.inference_only(row['binding']):
            maps,natural=frames(row)
            arrays=np.stack([pc.unpack(natural['B']).astype(np.float32),maps['M'],maps['U'],maps['T']]).reshape(4,-1)
            weights=np.asarray([0,1,2,1],np.float64)
            packed=np.empty(1024*1024//8,np.uint8)
            for start in range(0,1024*1024,65536):
                stop=min(start+65536,1024*1024)
                weighted=weights@arrays[:,start:stop].astype(np.float64)
                packed[start//8:stop//8]=np.packbits(weighted>2.)
            np.savez_compressed(path,**{WINNER:packed})
        r={'index':row['index'],'dataset':row['dataset'],'prediction_path':str(path),'prediction_sha256':sha(path),
           'source_record_sha256':sha(OUT/'source_records'/(row['index']+'.json')),'query_GT_reads':0,'new_encoder_calls':0}
        records.append(r);write(OUT/'records'/(row['index']+'.json'),r)
        if i<=2 or i%100==0:print({'phase':'natural-mixture','completed':i,'n':500,'seconds':round(time.monotonic()-began,1)},flush=True)
    write(OUT/'sealed.json',{'state':'ALL500_FIXED_SOFT_WINNER_NATURAL_MASKS_SEALED','n':500,'records':records,
        'config_sha256':sha(OUT/'config.json'),'sources_seal_sha256':sha(OUT/'sources_sealed.json'),
        'bindings_sha256':sha(OUT/'bindings.json'),'query_GT_reads':0,'new_encoder_calls':0,'training':False,
        'wall_seconds':time.monotonic()-began,'latency_scope':'cached source readout and one mixture only; not complete pipeline latency'})
    print({'state':'ALL500_FIXED_SOFT_WINNER_NATURAL_MASKS_SEALED','seal_sha256':sha(OUT/'sealed.json')},flush=True)


def score():
    rows,config=freeze();seal=read(OUT/'sealed.json')
    assert seal['n']==500 and seal['config_sha256']==sha(OUT/'config.json')
    receipts={r['index']:r for r in seal['records']}
    assert all(sha(r['prediction_path'])==r['prediction_sha256'] for r in receipts.values())
    prior={r['index']:r for r in read(PARENT/'report.json')['records']}
    networks={r['index']:r for r in read(NETWORK/'report.json')['records']}
    arms=('FoRIS','MEAN','UN_midrange','T','r00097','network4',WINNER)
    began,records=time.monotonic(),[]
    for i,row in enumerate(rows,1):
        key=row['index'];b=row['binding'];sources=base.load_existing(row)
        predictions={base.ARM[s]:pc.unpack(sources[s]) for s in ('B','M','U','T')}
        for arm in ('r00097','network4'):
            source=row[arm];assert sha(source['path'])==source['sha256']
            with np.load(source['path'],allow_pickle=False) as z:predictions[arm]=pc.unpack(z[source['key']])
        with np.load(receipts[key]['prediction_path'],allow_pickle=False) as z:predictions[WINNER]=pc.unpack(z[WINNER])
        gt,valid=pc.truth(b)
        r={'index':key,'dataset':b['dataset'],'fold':b['row']['fold'],'class_id':b['row']['global_class_id'],
           'query_photo':str((ASSETS/b['row']['query_path']).resolve()),
           'metrics':{a:pr.metric(predictions[a],gt,valid) for a in arms},
           'edits':{base_arm:base.edit_counts(predictions[WINNER],predictions[base_arm],gt,valid) for base_arm in COMPARE}}
        for a in arms[:-1]:
            expected=(networks if a=='network4' else prior)[key]['metrics'][a]
            assert all(r['metrics'][a][k]==expected[k] for k in ('I','U','TP','FP','FN','GT','predicted')),(key,a)
        for a in COMPARE:
            at,af,dt,df=r['edits'][a]
            assert r['metrics'][WINNER]['TP']==r['metrics'][a]['TP']+at-dt
            assert r['metrics'][WINNER]['FP']==r['metrics'][a]['FP']+af-df
        records.append(r)
        if i<=2 or i%100==0:print({'phase':'exposed500-score','completed':i,'n':500,'seconds':round(time.monotonic()-began,1)},flush=True)
    summary=pr.summary(records,arms)
    # Reuse the preserved bootstrap implementation; these module globals are
    # local to this scoring process and do not modify the parent source/files.
    base.RULE_IDS=(WINNER,);base.BASELINES=COMPARE
    paired=base.bootstrap(records,arms,summary)
    for table in summary.values():
        for value in table.values():value.pop('paired_query_photo_bootstrap95_pp',None)
    edits={d:{a:np.sum([r['edits'][a] for r in records if r['dataset']==d],axis=0).tolist() for a in COMPARE} for d in lab.ALL_DOMAINS}
    result={'state':'COMPLETE_ONE_FIXED_SOFT_WINNER_EXPOSED500_REPRODUCTION','n':500,'winner':WINNER,
        'formula':config['weights'],'summary':summary,
        'macro10_class_miou':{a:float(np.mean([summary[d][a]['class_miou'] for d in lab.ALL_DOMAINS])) for a in arms},
        'paired_comparisons':paired,'edit_columns':base.EDITS,'edit_columns_zh':['补回TP','新增FP','删除TP','移除FP'],
        'edits':edits,'records':records,'source_parity_comparisons':1500,'source_parity_total_xor':0,
        'config_sha256':sha(OUT/'config.json'),'seal_sha256':sha(OUT/'sealed.json'),
        'sources_seal_sha256':sha(OUT/'sources_sealed.json'),'existing_controls_integer_identity':True,
        'new_encoder_calls':0,'training':False,'post_score_parameter_changes':False,'new_blind_confirmation':False,
        'confirm1000_accessed':False,'full_pipeline_latency_measured_here':False,
        'score_seconds':time.monotonic()-began,'exposure':config['exposure']}
    write(OUT/'report.json',result)
    compact={k:v for k,v in result.items() if k!='records'}
    compact.update(report_path=str(OUT/'report.json'),report_sha256=sha(OUT/'report.json'))
    write(OUT/'completion.json',compact);write(ROOT/'evidence/research/x1012s2_soft500.json',compact)
    lines=['# 唯一连续赢家在已曝光500配对上的固定复现','',
        '**已曝光R5发展复现，不是新盲测；未访问confirm1000。全部500×3纯源读出逐位匹配既有自然mask后冻结唯一公式，再封存全部500混合输出，最后评分。**','',
        '`p=.25*M+.50*U+.25*T`，使用原1024连续读出，FP64加权、strict>.5；B权重0，无新minmax/clip/CRF，无新编码或训练。M不归一化，U/T保留各自原读出归一化。','',
        '|域|FoRIS|MEAN|UN|network4|soft35|对network4 pp|', '|---|---:|---:|---:|---:|---:|---:|']
    for d,t in summary.items():
        vals=[t[a]['class_miou'] for a in COMPARE+(WINNER,)]
        vals.append(paired['domains'][d][WINNER]['network4']['delta_pp'])
        lines.append('|'+d+'|'+'|'.join(f'{v:.4f}' for v in vals)+'|')
    vals=[result['macro10_class_miou'][a] for a in COMPARE+(WINNER,)]+[paired['macro10'][WINNER]['network4']['delta_pp']]
    lines+=['|十域均值|'+'|'.join(f'{v:.4f}' for v in vals)+'|','','|范围|对照|差值pp|照片配对95%区间|','|---|---|---:|---|']
    for scope,t in [*paired['domains'].items(),('macro10',paired['macro10'])]:
        for a in COMPARE:
            v=t[WINNER][a];lo,hi=v['paired_query_photo_bootstrap95_pp']
            lines.append(f"|{scope}|{a}|{v['delta_pp']:+.4f}|[{lo:+.4f},{hi:+.4f}]|")
    lines+=['','|域|对照|补回TP|新增FP|删除TP|移除FP|','|---|---|---:|---:|---:|---:|']
    for d,t in edits.items():
        for a in COMPARE:lines.append('|'+d+'|'+a+'|'+'|'.join(str(x) for x in t[a])+'|')
    lines+=['','CI为既定公式的曝光数据描述性区间，不校正此前1360选式。所有权重、来源定义与读出均未按500结果修改。',
        f'1500纯源比较XOR=0；源封存SHA `{result["sources_seal_sha256"]}`。',
        f'自然预测封存SHA `{result["seal_sha256"]}`；配置SHA `{result["config_sha256"]}`。',
        '[完整逐域及增删JSON](../../evidence/research/x1012s2_soft500.json)。',
        '缓存读出/混合耗时不代表完整管线延迟。']
    (ROOT/'docs/research/SOFT_COMPOSITION_500_REPLAY_20261013.md').write_text('\n'.join(lines)+'\n')
    print({'state':result['state'],'macro10':result['macro10_class_miou'],'paired_macro':paired['macro10'],
           'seconds':round(time.monotonic()-began,1)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('verify','freeze','produce','score'))
    args=parser.parse_args()
    if args.mode=='freeze':
        rows,config=freeze();print({'state':config['state'],'n':len(rows)},flush=True)
    else:globals()[args.mode]()
