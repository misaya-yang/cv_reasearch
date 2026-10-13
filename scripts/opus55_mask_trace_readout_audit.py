#!/usr/bin/env python3
"""Post-seal readout ceilings and affine-component audit; zero inference.

Both oracles use query GT and are explicitly diagnostic. The task oracle
does not score background; its full IoU reveals that missing constraint.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import opus55_reference_mask_trace as base
import opus55_mask_trace_followup as follow


def full_iou_oracle(score,target):
    import numpy as np
    s=score.ravel();truth=target.ravel();order=np.argsort(-s,kind='stable');s=s[order];truth=truth[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(truth,dtype=np.int64)[ends];selected=ends+1;union=target.sum()+selected-tp
    iou=tp/np.maximum(union,1);best=int(np.argmax(iou))
    return dict(iou=float(100*iou[best]),cut_inclusive=float(s[ends[best]]),TP=int(tp[best]),FP=int(selected[best]-tp[best]))


def main():
    import numpy as np
    import torch
    import torch.nn.functional as f
    torch.set_num_threads(2)
    root=base.OUT;prepared=json.loads((root/'prepared.json').read_text());assert base.sha(root/'query_truth.json')==prepared['query_truth_sha256']
    original_seal=json.loads((root/'sealed.json').read_text());assert base.sha(root/'inference.jsonl')==original_seal['inference_sha256']
    follow_seal=json.loads((follow.OUT/'sealed.json').read_text());assert base.sha(follow.OUT/'inference.jsonl')==follow_seal['inference_sha256']
    truth=json.loads((root/'query_truth.json').read_text());traces={r['id']:r for r in map(json.loads,(root/'inference.jsonl').read_text().splitlines())}
    diagnosis=json.loads((follow.OUT/'diagnosis.json').read_text());records=[]
    for r in diagnosis['records']:
        trace=traces[r['id']];assert base.sha(trace['file'])==trace['sha256'];targets=[]
        for v in ['A','B']:
            b=truth[r['id']]['masks'][v];assert base.sha(b['path'])==b['sha256']
            targets.append(f.interpolate(torch.from_numpy(base.read_mask(b['path']).astype('float32'))[None,None],(1024,1024),mode='nearest')[0,0].numpy().astype(bool))
        ex=[targets[0]&~targets[1],targets[1]&~targets[0]];keep=ex[0]|ex[1];tasks=[]
        with np.load(trace['file']) as z:
            a=z['A/score'].astype(float).ravel();b=z['B/score'].astype(float).ravel();ac=a-a.mean();bc=b-b.mean()
            alpha=float(ac@bc/max(ac@ac,1e-20));beta=float(b.mean()-alpha*a.mean());difference=b-a;residual=b-(alpha*a+beta)
            affine=dict(alpha=alpha,beta=beta,positive_scale=alpha>0,
                difference_energy_explained=float(np.clip(1-residual@residual/max(difference@difference,1e-20),0,1)),
                pearson=float(ac@bc/max(np.linalg.norm(ac)*np.linalg.norm(bc),1e-20)))
            for i,v in enumerate(['A','B']):
                score=torch.from_numpy(z[v+'/score'].copy());norm=score-score.min();norm=norm/norm.max().clamp_min(1e-6)
                full=f.interpolate(norm[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy()
                oracle=r['oracle_cuts'][i];pred=full>=oracle['cut_inclusive'];gt=targets[i]
                I=int((pred&gt).sum());U=int((pred|gt).sum());native=r['measures']['native']['pre']['IU'][i]
                full_oracle=full_iou_oracle(full,gt);best=full>=full_oracle['cut_inclusive']
                tasks.append(dict(role=v,native_pre_iou=100*native['I']/max(native['U'],1),
                    task_oracle_pre_iou=100*I/max(U,1),task_oracle_TP=I,task_oracle_FP=int((pred&~gt).sum()),
                    native_pre_TP=native['TP'],native_pre_FP=native['FP'],target_area=int(gt.sum()),task_oracle=oracle,
                    full_iou_oracle=full_oracle,full_iou_oracle_task_auc=base.weighted_auc(best[keep],ex[i][keep],ex[1-i][keep])))
        records.append(dict(id=r['id'],dataset=r['dataset'],classes=r['classes'],affine=affine,tasks=tasks))
    summary={}
    for d in sorted({r['dataset'] for r in records}):
        sub=[r for r in records if r['dataset']==d]
        def mean_task(rec,key):return np.mean([x[key] for x in rec['tasks']])
        def iou_best(rec):return np.mean([x['full_iou_oracle']['iou'] for x in rec['tasks']])
        positive=[r for r in sub if r['affine']['positive_scale']]
        summary[d]=dict(n=len(sub),native_pre_task_mean_iou=float(np.mean([mean_task(r,'native_pre_iou') for r in sub])),
            task_oracle_pre_task_mean_iou=float(np.mean([mean_task(r,'task_oracle_pre_iou') for r in sub])),
            task_oracle_full_iou_delta=base.bootstrap([mean_task(r,'task_oracle_pre_iou')-mean_task(r,'native_pre_iou') for r in sub]),
            full_iou_oracle_pre_task_mean_iou=float(np.mean([iou_best(r) for r in sub])),
            full_iou_oracle_delta=base.bootstrap([iou_best(r)-mean_task(r,'native_pre_iou') for r in sub]),
            task_oracle_FP_per_GT=float(sum(t['task_oracle_FP'] for r in sub for t in r['tasks'])/sum(t['target_area'] for r in sub for t in r['tasks'])),
            native_pre_FP_per_GT=float(sum(t['native_pre_FP'] for r in sub for t in r['tasks'])/sum(t['target_area'] for r in sub for t in r['tasks'])),
            positive_affine_pairs=len(positive),mean_pearson=float(np.mean([r['affine']['pearson'] for r in sub])),
            affine_null_component_energy=base.bootstrap([r['affine']['difference_energy_explained'] for r in positive]))
    out=root/'readout_audit';out.mkdir(exist_ok=True)
    base.write(out/'diagnosis.json',dict(state='POST_SEAL_ORACLE_AND_AFFINE_AUDIT',new_encodings=0,
        source_sha256=base.sha(Path(__file__)),source_trace_seal_sha256=base.sha(root/'sealed.json'),
        followup_seal_sha256=base.sha(follow.OUT/'sealed.json'),datasets=summary,records=records,
        warning='Both oracle thresholds use query GT. Task oracle ignores other background; full-IoU oracle is a ceiling. Neither is an algorithm or a deployable threshold.',
        theory='Positive affine score changes disappear under per-mask minmax normalization. Projecting actual mask-induced score differences onto span{1,scoreA} describes this null component, not proof that it encodes the desired task.'))
    lines=['# 读出上界与任务变化的仿射分量','',
        '所有轨迹与干预掩码先封存。以下只读取保存场和评分GT，不调用DINO、FoRIS前端或CRF。两个oracle均不可部署。',
        '任务oracle只优化A/B独占区二值AUC，完整IoU检验其背景代价；full-IoU oracle在全部像素上选最优阈值，估计固定场的读出上界。','',
        '| 数据集 | 原pre两任务均值IoU | 任务oracle完整IoU | full-IoU oracle | 原FP/GT | 任务oracle FP/GT |','|---|---:|---:|---:|---:|---:|']
    for d,x in summary.items():lines.append(f"| {d} | {x['native_pre_task_mean_iou']:.3f} | {x['task_oracle_pre_task_mean_iou']:.3f} | {x['full_iou_oracle_pre_task_mean_iou']:.3f} | {x['native_pre_FP_per_GT']:.3f} | {x['task_oracle_FP_per_GT']:.3f} |")
    lines+=['','对minmax，N(as+b)=N(s)（a>0）；局部微分为',
            '`δN=(δs−δs_min−N(s)*(δs_max−δs_min))/(s_max−s_min)`。',
            '因此仿射方向位于该算子的零空间，但这是数学性质，单独不构成原创算法理论，也不证明真实任务变化主要在这些方向。',
            '逐例mask变化的正仿射拟合、余差和完整oracle结果见`diagnosis.json`。不能将“参考阈值存在oracle空间”偷换为“参考阈值能泛化”；此前PACO参考定阈值失败仍保留。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
