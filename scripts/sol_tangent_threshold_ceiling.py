#!/usr/bin/env python3
"""Exact post-seal whole-pixel threshold ceilings; never inference thresholds."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_task_tangent_kernel as p
from insid3_factor_evidence import components
from opus55_mask_trace_readout_audit import full_iou_oracle

np,torch,F,base=p.np,p.torch,p.F,p.base
OUT=base.REPO/'evidence/local/sol_validation_20261010'


def main():
    torch.set_num_threads(2);start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    pairs=p.producer.read(p.OUT/'pairs.json');truth=p.producer.read(base.OUT/'query_truth.json')
    groups=components(pairs);seal=p.producer.read(p.OUT/'sealed.json');report=p.producer.read(p.OUT/'report.json')
    assert report['seal_sha256']==base.sha(p.OUT/'sealed.json')
    assert base.sha(base.OUT/'query_truth.json')==p.producer.read(base.OUT/'prepared.json')['query_truth_sha256']
    cfg=dict(kind='Post-seal exact best full-pixel threshold oracle, no inference or new candidate',
        arms=p.ARMS,frame='CLI1024, same bilinear raw score as native zero renderer',
        statistic='Two real task IoUs averaged within each query photo; connected R/Q-photo group bootstrap',
        warning='Query GT chooses each threshold; never deployable, no official class mIoU or independent confirmation',
        source_sha256=base.sha(Path(__file__)),parent_seal_sha256=base.sha(p.OUT/'sealed.json'),
        parent_report_sha256=base.sha(p.OUT/'report.json'),identity_helper_sha256=base.sha(Path(__file__).with_name('insid3_factor_evidence.py')),
        helper_sha256=base.sha(Path(__file__).with_name('opus55_mask_trace_readout_audit.py')),new_encodings=0,new_predictions=0)
    base.write(OUT/'threshold_ceiling_config.json',cfg)
    reported={r['id']:r for r in report['records']};records=[];checks=0
    for pair,rec in zip(pairs,seal['records']):
        assert pair['id']==rec['id'];path=p.OUT/'fields'/rec['file'];assert base.sha(path)==rec['fields_sha256'];checks+=1
        rows=[]
        with np.load(path) as fields:
            for v in ['A','B']:
                binding=truth[pair['id']]['masks'][v];assert base.sha(binding['path'])==binding['sha256'];checks+=1
                target=F.interpolate(torch.from_numpy(base.read_mask(binding['path']).astype('float32'))[None,None],(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
                parent=next(t for t in reported[pair['id']]['tasks'] if t['variant']==v)['metrics']['cli1024'];arms={}
                for arm in p.ARMS:
                    full=F.interpolate(torch.from_numpy(fields[v+'/'+arm].copy())[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy()
                    oracle=full_iou_oracle(full,target);pred=full>=oracle['cut_inclusive']
                    hit=int((pred&target).sum());fp=int((pred&~target).sum());gt=int(target.sum());union=gt+fp
                    assert hit==oracle['TP'] and fp==oracle['FP'];checks+=1
                    assert abs(100*hit/max(union,1)-oracle['iou'])<1e-10;checks+=1
                    native=parent[arm]['iou'];assert oracle['iou']+1e-9>=native;checks+=1
                    arms[arm]=dict(**oracle,FN=gt-hit,GT=gt,recall=hit/max(gt,1),precision=hit/max(hit+fp,1),
                        native_iou=native,oracle_minus_native=oracle['iou']-native,oracle_minus_complete_foris=oracle['iou']-parent['foris']['iou'])
                rows.append(dict(variant=v,arms=arms,foris=parent['foris']))
        records.append(dict(id=pair['id'],dataset=pair['dataset'],tasks=rows))
        base.write(OUT/'threshold_ceiling_activity.json',dict(state='POST_SEAL_THRESHOLD_ORACLE',completed=len(records),total=60,seconds=time.monotonic()-start,new_encodings=0))
    datasets={}
    for name in sorted({r['dataset'] for r in records}):
        sub=[r for r in records if r['dataset']==name];arms={}
        for arm in p.ARMS:
            averages=lambda key:[np.mean([t['arms'][arm][key] for t in r['tasks']]) for r in sub]
            arms[arm]=dict(native_iou=float(np.mean(averages('native_iou'))),oracle_iou=float(np.mean(averages('iou'))),
                oracle_minus_native=p.group_bootstrap(sub,averages('oracle_minus_native'),groups),
                oracle_minus_complete_foris=p.group_bootstrap(sub,averages('oracle_minus_complete_foris'),groups),
                task_cases_ceiling_below_foris=sum(t['arms'][arm]['oracle_minus_complete_foris']<0 for r in sub for t in r['tasks']))
        for control in p.ARMS:
            arms['task_tangent']['oracle_minus_'+control]=p.group_bootstrap(sub,[np.mean([t['arms']['task_tangent']['iou']-t['arms'][control]['iou'] for t in r['tasks']]) for r in sub],groups)
        datasets[name]=dict(n_pairs=len(sub),photo_components=len({groups[r['id']] for r in sub}),arms=arms)
    result=dict(state='COMPLETE_POST_SEAL_THRESHOLD_CEILING',datasets=datasets,records=records,checks=checks,
        config_sha256=base.sha(OUT/'threshold_ceiling_config.json'),seconds=time.monotonic()-start,new_encodings=0,new_inference=0,
        limits='Oracle chooses threshold using query GT; excludes CRF or arbitrary spatial masks. Best level-set ability is not a usable calibration method or generalization proof.')
    base.write(OUT/'tangent_threshold_ceiling.json',result)
    lines=['# Sol：任务核完整阈值上界','',
        '全部源字段先封存后GT。每个字段独立取GT最佳完整像素IoU阈值，仅估计固定排序的level-set能力，不用于推理、调参或算法成绩。未执行encoder、核拟合或CRF。','',
        '|域|原任务核完整IoU|任务核GT阈值上界|去位置O24上界|隐藏激活上界|无mask梯度上界|任务核上界−完整FoRIS|','|---|---:|---:|---:|---:|---:|---|']
    for dataset,s in datasets.items():
        arms=s['arms'];a=arms['task_tangent'];change=a['oracle_minus_complete_foris']
        lines.append(f"|{dataset}|{a['native_iou']:.3f}|{a['oracle_iou']:.3f}|{arms['o24_position_ridge']['oracle_iou']:.3f}|{arms['hidden_ridge']['oracle_iou']:.3f}|{arms['mean_tangent']['oracle_iou']:.3f}|{change['mean']:+.3f} {change['ci95']}|")
    lines+=['','若任务核GT上界仍弱于简单对照，就不能把该差距全部归给零点；若某些任务连上界都低于完整FoRIS，单纯阈值不能修复这些任务。oracle在完整像素含所有背景评分，区别于只区分A/B两个目标的阈值上界。']
    (OUT/'TANGENT_THRESHOLD_CEILING.md').write_text('\n'.join(lines)+'\n')
    base.write(OUT/'threshold_ceiling_activity.json',dict(state='COMPLETE',n_pairs=60,checks=checks,seconds=time.monotonic()-start,new_encodings=0))
    print(json.dumps({d:{a:round(v['oracle_iou'],3) for a,v in s['arms'].items()} for d,s in datasets.items()}),flush=True)


if __name__=='__main__':
    main()
