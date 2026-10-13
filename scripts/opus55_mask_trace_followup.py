#!/usr/bin/env python3
"""One selected causal deletion and a separately marked threshold oracle.

Deleting the saved additive semantic-cluster delta is exactly the source
Part4 output before its final addition. All other fields, source binarization
and CRF are unchanged. Oracle thresholds are diagnosis only, never inference.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import opus55_reference_mask_trace as base

OUT=base.OUT/'semantic_delta_followup'


def prepare(out):
    diagnosis=json.loads((base.OUT/'diagnosis.json').read_text())
    assert not (out/'config.json').exists()
    pairs=json.loads((base.OUT/'pairs.json').read_text());index={r['id']:r for r in map(json.loads,(base.OUT/'inference.jsonl').read_text().splitlines())}
    config=dict(hypothesis='The additive query-cluster semantic reweight map dilutes mask-specified task discrimination',
        intervention='Remove delta_map only: native score=s3-penalty+delta; intervention=s3-penalty. All upstream quantities frozen, same original binarization and CRF',
        selection_reason='Mean penalty->score switch AUC drops in all three20-pair development domains; only COCO paired CI excludes zero, so this is a candidate, not established cross-domain law',
        n_pairs=len(pairs),new_encodings=0,query_GT_in_inference=False,thresholds_swept=0,
        source_sha256={str(p):base.sha(p) for p in [Path(__file__),Path(base.__file__),base.ASSETS/'third_party/foris_official/models/foris.py',base.REPO/'src/ics/foris.py']},
        parent_sha256={str(base.OUT/name):base.sha(base.OUT/name) for name in ['sealed.json','pairs.json','inference.jsonl','diagnosis.json']},
        oracle='After prediction seal only, optimize binary task-discrimination threshold on exclusive query GT; a ceiling, never a candidate or usable test threshold',
        metric='Same task switch AUC plus paired mean IoU across two tasks per photograph; this is not official class mIoU',
        exposure='Operator chosen from this study; paired development ablation, no independence claim')
    out.mkdir(parents=True,exist_ok=True);base.write(out/'config.json',config)
    base.write(out/'tasks.json',[dict(pair=p,trace=index[p['id']],identity_probe=i%20==0) for i,p in enumerate(pairs)])


def initialize(out):
    global NP,TORCH,F,HOST,ROOT
    import numpy as np
    import torch
    import torch.nn.functional as f
    from cached_dino import CachedDINO,cache_host
    from raw_feature_cache import RawFeatureCache
    torch.set_num_threads(2);torch.manual_seed(0)
    cache=RawFeatureCache(base.PROFILE.parent.parent,json.loads(base.PROFILE.read_text()))
    host=cache_host(base.ASSETS,CachedDINO(cache),mask_refiner='crf')
    NP,TORCH,F,HOST,ROOT=np,torch,f,host,Path(out)


def infer_one(task):
    from ics.official_data import decoded_rgb,array_hash
    start=time.monotonic();p=task['pair'];trace=task['trace'];assert base.sha(trace['file'])==trace['sha256']
    attempts=[];packed={};shapes={}
    with base.deny_labels(base.OUT,attempts),TORCH.inference_mode():
        q=decoded_rgb(base.ASSETS/p['row']['query_path'],p['row'].get('query_crop'));assert array_hash(NP.asarray(q))==p['row']['query_rgb_hash']
        tgt=HOST._transform(q).unsqueeze(0)
        with NP.load(trace['file']) as z:
            for v in ['A','B']:
                score=TORCH.from_numpy(z[v+'/penalty'].copy())
                pre=HOST._binarize_response(score,target_hw=(1024,1024));crf=HOST._finalize_mask(pre,tgt)
                packed[v+'/no_delta/pre']=NP.packbits(pre.numpy());packed[v+'/no_delta/crf']=NP.packbits(crf.numpy())
                for stage in ['pre','crf']:packed[v+'/native/'+stage]=z[v+'/'+stage].copy()
                if task['identity_probe']:
                    native=HOST._binarize_response(TORCH.from_numpy(z[v+'/score'].copy()),target_hw=(1024,1024))
                    native_crf=HOST._finalize_mask(native,tgt)
                    assert NP.array_equal(NP.packbits(native.numpy()),packed[v+'/native/pre'])
                    assert NP.array_equal(NP.packbits(native_crf.numpy()),packed[v+'/native/crf'])
        assert not attempts and HOST.encoder.encoder is None and HOST.encoder.encoder_calls==0
    file=ROOT/'predictions'/f"{p['id']}.npz";file.parent.mkdir(parents=True,exist_ok=True);NP.savez_compressed(file,**packed)
    return dict(id=p['id'],dataset=p['dataset'],file=str(file),sha256=base.sha(file),encoder_calls=0,
                native_renderer_identity_probe=task['identity_probe'],native_renderer_bit_exact=task['identity_probe'],
                query_label_access_attempts=attempts,seconds=time.monotonic()-start,worker_pid=os.getpid())


def infer(out,workers):
    config=json.loads((out/'config.json').read_text())
    for p,digest in config['source_sha256'].items():assert base.sha(p)==digest
    for p,digest in config['parent_sha256'].items():assert base.sha(p)==digest
    if (out/'sealed.json').exists():return
    tasks=json.loads((out/'tasks.json').read_text());done={};ledger=out/'inference.jsonl';start=time.monotonic()
    if ledger.exists():
        for r in map(json.loads,ledger.read_text().splitlines()):assert base.sha(r['file'])==r['sha256'];done[r['id']]=r
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(out),)) as pool,ledger.open('a',buffering=1) as handle:
        futures=[pool.submit(infer_one,t) for t in tasks if t['pair']['id'] not in done]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();handle.write(json.dumps(r)+'\n');done[r['id']]=r
            base.write(out/'activity.json',dict(state='CRF_INTERVENTION',completed=len(done),n=len(tasks),controller_pid=os.getpid(),
                worker_pids=[p.pid for p in pool._processes.values()],seconds=time.monotonic()-start,new_encodings=0))
    assert len(done)==len(tasks)
    base.write(out/'sealed.json',dict(state='ALL_INTERVENTION_MASKS_SEALED',n_pairs=len(tasks),encoder_calls=0,query_GT_in_inference=False,
        config_sha256=base.sha(out/'config.json'),tasks_sha256=base.sha(out/'tasks.json'),inference_sha256=base.sha(ledger)))


def oracle_binary_auc(score,positive,negative):
    import numpy as np
    s=np.asarray(score).ravel();p=np.asarray(positive,dtype=float).ravel();n=np.asarray(negative,dtype=float).ravel()
    keep=(p+n)>0;s=s[keep];p=p[keep];n=n[keep]
    order=np.argsort(-s,kind='stable');s=s[order];p=p[order];n=n[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    values=.5+.5*(np.cumsum(p)[ends]/p.sum()-np.cumsum(n)[ends]/n.sum())
    best=int(np.argmax(values))
    return dict(auc=float(max(.5,values[best])),cut_inclusive=float(s[ends[best]]))


def diagnose(out):
    import numpy as np
    import torch
    import torch.nn.functional as f
    torch.set_num_threads(2)
    seal=json.loads((out/'sealed.json').read_text())
    for name,key in [('config.json','config_sha256'),('tasks.json','tasks_sha256'),('inference.jsonl','inference_sha256')]:assert base.sha(out/name)==seal[key]
    records=[];parents=json.loads((base.OUT/'diagnosis.json').read_text());parent_index={r['id']:r for r in parents['records']}
    truth=json.loads((base.OUT/'query_truth.json').read_text());tasks={t['pair']['id']:t for t in json.loads((out/'tasks.json').read_text())}
    for r in map(json.loads,(out/'inference.jsonl').read_text().splitlines()):
        assert base.sha(r['file'])==r['sha256'];task=tasks[r['id']];trace=task['trace'];assert base.sha(trace['file'])==trace['sha256']
        masks=[]
        for v in ['A','B']:
            binding=truth[r['id']]['masks'][v];assert base.sha(binding['path'])==binding['sha256']
            native=base.read_mask(binding['path']);masks.append(f.interpolate(torch.from_numpy(native.astype('float32'))[None,None],(1024,1024),mode='nearest')[0,0].numpy().astype(bool))
        ex=[masks[0]&~masks[1],masks[1]&~masks[0]];keep=ex[0]|ex[1];measures={};oracle=[];interpolated=[];invariants=[]
        with np.load(r['file']) as z,np.load(trace['file']) as field:
            for arm in ['native','no_delta']:
                measures[arm]={}
                for stage in ['pre','crf']:
                    values=[];auc=[]
                    for i,v in enumerate(['A','B']):
                        pred=np.unpackbits(z[v+'/'+arm+'/'+stage],count=1024*1024).reshape(1024,1024).astype(bool)
                        values.append(dict(I=int((pred&masks[i]).sum()),U=int((pred|masks[i]).sum()),
                            target=int(masks[i].sum()),TP=int((pred&masks[i]).sum()),FP=int((pred&~masks[i]).sum()),FN=int((~pred&masks[i]).sum())))
                        auc.append(base.weighted_auc(pred[keep],ex[i][keep],ex[1-i][keep]))
                    measures[arm][stage]=dict(switch_auc=float(np.mean(auc)),per_task_auc=auc,IU=values,
                        task_mean_iou=float(np.mean([100*x['I']/max(x['U'],1) for x in values])))
            for i,v in enumerate(['A','B']):
                score=torch.from_numpy(field[v+'/score'].copy());normalized=score-score.min();normalized=normalized/normalized.max().clamp_min(1e-6)
                full=f.interpolate(normalized[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy()
                assert np.array_equal(np.packbits(full>.5),z[v+'/native/pre'])
                interpolated.append(base.weighted_auc(full[keep],ex[i][keep],ex[1-i][keep]))
                oracle.append(oracle_binary_auc(full,ex[i],ex[1-i]))
                mass=[e.reshape(64,16,64,16).sum((1,3)) for e in ex]
                before=base.weighted_auc(score.numpy(),mass[i],mass[1-i]);after=base.weighted_auc(normalized.numpy(),mass[i],mass[1-i]);invariants.append(abs(before-after))
        records.append(dict(id=r['id'],dataset=r['dataset'],classes=tasks[r['id']]['pair']['classes'],measures=measures,
            interpolated_switch_auc=float(np.mean(interpolated)),oracle_binary_switch_auc=float(np.mean([o['auc'] for o in oracle])),
            oracle_cuts=oracle,max_minmax_auc_error=max(invariants),source=parent_index[r['id']]['scores']['score']['switch_auc']))
    report={}
    for d in sorted({r['dataset'] for r in records}):
        sub=[r for r in records if r['dataset']==d];report[d]=dict(n=len(sub),
            causal_crf_switch_delta=base.bootstrap([r['measures']['no_delta']['crf']['switch_auc']-r['measures']['native']['crf']['switch_auc'] for r in sub]),
            causal_crf_task_mean_iou_delta=base.bootstrap([r['measures']['no_delta']['crf']['task_mean_iou']-r['measures']['native']['crf']['task_mean_iou'] for r in sub]),
            native_task_mean_iou=float(np.mean([r['measures']['native']['crf']['task_mean_iou'] for r in sub])),
            no_delta_task_mean_iou=float(np.mean([r['measures']['no_delta']['crf']['task_mean_iou'] for r in sub])),
            minmax_max_auc_error=max(r['max_minmax_auc_error'] for r in sub),
            interpolation_delta=base.bootstrap([r['interpolated_switch_auc']-r['source'] for r in sub]),
            binarization_delta=base.bootstrap([r['measures']['native']['pre']['switch_auc']-r['interpolated_switch_auc'] for r in sub]),
            threshold_oracle_minus_native=base.bootstrap([r['oracle_binary_switch_auc']-r['measures']['native']['pre']['switch_auc'] for r in sub]),
            native_binary_switch_auc=float(np.mean([r['measures']['native']['pre']['switch_auc'] for r in sub])),
            oracle_binary_switch_auc=float(np.mean([r['oracle_binary_switch_auc'] for r in sub])))
    base.write(out/'diagnosis.json',dict(state='POST_SEAL_CAUSAL_AND_ORACLE_DIAGNOSIS',datasets=report,records=records,
        verdict='Determine from all three domains; deletion is only a development intervention, oracle cuts cannot become inference thresholds',
        seal_sha256=base.sha(out/'sealed.json'),new_encodings=0))
    lines=['# 语义聚类修正移除：因果干预与阈值诊断','',
        '仅删除最后的additive delta_map，原上游、原minmax/bilinear/>.5及原CRF全部保持；零编码，先封存后GT。',
        '算子由60对开发轨迹选择，结果是开发因果检验，不是独立泛化确认。均值IoU是每照片两任务的配对均值，不是官方class mIoU。','',
        '| 数据集 | 原完整任务均值IoU | 去delta | 点差及95%区间 | 完整任务切换AUC点差 |','|---|---:|---:|---|---:|']
    for d,v in report.items():
        change=v['causal_crf_task_mean_iou_delta'];ci=change['ci95']
        lines.append(f"| {d} | {v['native_task_mean_iou']:.3f} | {v['no_delta_task_mean_iou']:.3f} | {change['mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {v['causal_crf_switch_delta']['mean']:+.5f} |")
    lines+=['','归一化保持排序的误差、插值影响、二值化影响与GT最优二值阈值上界在`diagnosis.json`。',
            'Oracle只估计当前连续场允许的任务区分上限；测试GT不可用于设阈值，oracle差距本身不证明某可部署方法能恢复。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    base.write(out/'activity.json',dict(state='DIAGNOSED',n_pairs=len(records),new_encodings=0))
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','infer','diagnose','run']);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=a.out.resolve()
    if a.phase=='prepare':prepare(out)
    if a.phase in ['infer','run']:infer(out,a.workers)
    if a.phase in ['diagnose','run']:diagnose(out)


if __name__=='__main__':main()
