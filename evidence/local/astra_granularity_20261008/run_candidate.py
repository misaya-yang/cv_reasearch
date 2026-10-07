"""Freeze one candidate, finish dev + once-only held folds + COCO, report edits."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from candidate import predict

ROOT=Path(__file__).resolve().parent
ARMS=['joint_mask','center_control','permuted_control']
COHORTS={'dev':['paco_part_f0','paco_part_f1'],
         'held':['paco_part_f2','paco_part_f3'], 'coco':['coco_fresh600']}


def miou(iu,classes,weights=None):
    n=int(classes.max())+1
    if weights is None:
        weights=np.ones(len(classes))
    ii=np.bincount(classes,weights=iu[:,0]*weights,minlength=n)
    uu=np.bincount(classes,weights=iu[:,1]*weights,minlength=n)
    return float(np.mean(ii[uu>0]/uu[uu>0])*100)


def summarize(records):
    raw=np.array([r['class'] for r in records]);_,cl=np.unique(raw,return_inverse=True)
    counts={k:np.array([r['iu'][k] for r in records]) for k in ['B']+ARMS}
    rng=np.random.RandomState(0)
    draws=[np.bincount(rng.randint(0,len(records),len(records)),minlength=len(records)) for _ in range(1000)]
    base=np.array([miou(counts['B'],cl,w) for w in draws])
    out={'n':len(records),'classes':len(np.unique(cl)),'B':miou(counts['B'],cl),'arms':{}}
    for k in ARMS:
        boots=np.array([miou(counts[k],cl,w) for w in draws])
        changes=np.array([r['edits'][k] for r in records])
        out['arms'][k]={'miou':miou(counts[k],cl),'delta':miou(counts[k],cl)-out['B'],
            'delta_ci95':np.percentile(boots-base,[2.5,97.5]).tolist(),
            'edits_addTP_addFP_delTP_delFP':changes.sum(0).tolist()}
    center_boot=np.array([miou(counts['center_control'],cl,w) for w in draws])
    joint_boot=np.array([miou(counts['joint_mask'],cl,w) for w in draws])
    out['joint_vs_center']={'delta':out['arms']['joint_mask']['miou']-out['arms']['center_control']['miou'],
                            'ci95':np.percentile(joint_boot-center_boot,[2.5,97.5]).tolist()}
    return out


def report(summary):
    lines=['# 参考掩码决定范围：固定C1实验',
      '', '协议：单参考、冻结DINOv3-L/1024；B为原存RCG场直接切0.5。位置级class mIoU，1000次episode配对95%区间。PACO f0/f1开发，f2/f3单次评测；均为已有数据再用。',
      '', '**C1：3×3联合外观—掩码模板转移。** 预期：中心特征混入整物体语义时，邻域外观与完整参考掩码的对应关系能删对附着错圈；只在B后加一道模板投票保留门。新量读取九位置的排列及标签排列，区别于中心前景相似度/最近邻标签。8种固定方格朝向、4模板均匀票、半数平票保留；不调参。',
      '', '否定条件：f2+f3增益<1.0或差区间下界≤0，或COCO下降>0.3。',
      '', '| 集合 | B | C1 | Δ [95%区间] | 补真/补假/删真/删假 |',
      '|---|---:|---:|---|---|']
    names={'dev':'PACO开发299','held':'PACO留出300','coco':'COCO600'}
    for key in ['dev','held','coco']:
        if key not in summary:continue
        r=summary[key];m=r['arms']['joint_mask'];ci=m['delta_ci95']
        edits='/'.join(str(v) for v in m['edits_addTP_addFP_delTP_delFP'])
        lines.append(f"| {names[key]} | {r['B']:.2f} | {m['miou']:.2f} | {m['delta']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] | {edits} |")
    if len(summary)==3:
        h,c=summary['held']['arms']['joint_mask'],summary['coco']['arms']['joint_mask']
        passed=h['delta']>=1 and h['delta_ci95'][0]>0 and c['delta']>=-.3
        lines+=['', '**结论：'+('位置级过线，需原图完整评测；不声称超过FoRIS。' if passed else '未过线，关闭此固定候选，不做变体。')+'**']
        ctr=summary['held']['joint_vs_center'];hp=summary['held']['arms']['permuted_control']
        lines.append(f"同匹配中心标签控制：C1相对它 {ctr['delta']:+.2f} [{ctr['ci95'][0]:+.2f},{ctr['ci95'][1]:+.2f}]；非中心标签置换控制相对B {hp['delta']:+.2f}。完整控制/逐例账见summary.json与episodes.jsonl。")
        if not passed:
            lines.append('推断：此读出未把局部外观—标注布局迁移转为足够干净的范围证据；不能据此断言DINO无所需信息。PASCAL/SUIM/LVIS按门槛未运行。只试1个候选，未用满3个上限。')
    else:
        lines+=['','状态：候选尚无全部完整读数，不能判断过线。']
    lines+=['','## 上一轮泛化研究（本轮候选前）',
      '原图输出再核算RCG−FoRIS：COCO B+C+D +1.281、PASCAL +1.460、SUIM −0.078、PACO +0.300、LVIS +0.313。SUIM真覆盖项+0.548/假前景项−0.625；LVIS −2.840/+3.153（逐类精确净分解，非因果）。另SUIM5例消融实现复现，FoRIS/RCG的I/U全同旧记录，未扩全量；该轮新guide未完成评估。原文件在previous_*，不计本轮候选。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'candidate_C1'
    out.mkdir(exist_ok=False)
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['candidate.py','run_candidate.py','PROTOCOL.md','cache_features.py']}
    (out/'frozen.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (out/'expectation.txt').write_text('C1：预期修正可匹配局部布局下的整物体错圈；PACO留出Δ<1或CI下界≤0，或COCOΔ<−0.3即否定；不调常数。\n')
    print((out/'expectation.txt').read_text(),flush=True)
    summary={}
    report(summary)
    start=time.monotonic()
    for cohort,packs in COHORTS.items():
        records=[]
        for pack in packs:
            cache=ROOT/'cache'/pack
            assert (cache/'complete.json').exists(),cache
            rows=json.loads((cache/'rows.json').read_text())
            feats=np.load(cache/'features.npy',mmap_mode='r')
            inf=np.load(cache/'inference.npz')
            # Truth is never supplied to the method, only read after each fixed mask.
            truth=np.load(cache/'evaluation.npz')['truth']
            with (out/(pack+'.jsonl')).open('x') as stream:
                masks=[]
                for i,row in enumerate(rows):
                    q=torch.tensor(feats[i,0].astype(np.float32),device='cuda')
                    r=torch.tensor(feats[i,1].astype(np.float32),device='cuda')
                    cov=torch.tensor(inf['cov'][i],device='cuda')
                    b=inf['field'][i]>.5
                    t0=time.monotonic()
                    outputs=predict(q,r,cov,torch.tensor(b,device='cuda'))
                    outputs={k:v.cpu().numpy() for k,v in outputs.items()}
                    seconds=time.monotonic()-t0
                    tt=truth[i]
                    iu={k:[int((m&tt).sum()),int((m|tt).sum())] for k,m in dict(B=b,**outputs).items()}
                    edits={}
                    for k,m in outputs.items():
                        add,delete=m&~b,b&~m
                        edits[k]=[int((add&tt).sum()),int((add&~tt).sum()),int((delete&tt).sum()),int((delete&~tt).sum())]
                        assert not add.any()
                        assert iu[k][0]==iu['B'][0]+edits[k][0]-edits[k][2]
                        assert iu[k][1]==iu['B'][1]+edits[k][1]-edits[k][3]
                    label=str(row['c']) if cohort=='coco' else pack+':'+str(row['c'])
                    rec=dict(key=row['key'],pack=pack,cohort=cohort,**{'class':label},iu=iu,edits=edits,seconds=seconds)
                    records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush()
                    masks.append(np.stack([np.packbits(outputs[k]) for k in ARMS]))
                    if (i+1)%25==0 or i+1==len(rows):
                        print(json.dumps(dict(cohort=cohort,pack=pack,n=i+1,total=len(rows),elapsed_s=time.monotonic()-start,episode_s=seconds)),flush=True)
                np.savez_compressed(out/(pack+'_masks.npz'),names=np.array(ARMS),masks=np.stack(masks))
            inf.close();del feats
        summary[cohort]=summarize(records)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('COHORT_DONE',cohort,json.dumps(summary[cohort]),flush=True)
        report(summary)
    h,c=summary['held']['arms']['joint_mask'],summary['coco']['arms']['joint_mask']
    passed=h['delta']>=1 and h['delta_ci95'][0]>0 and c['delta']>=-.3
    (out/'receipt.json').write_text(json.dumps(dict(state='COMPLETE',passed=passed,seconds=time.monotonic()-start,
        hashes=hashes,n=1199),indent=2)+'\n')
    print('CANDIDATE_COMPLETE',passed,flush=True)


if __name__=='__main__':
    main()
