"""Analyze sealed I/U and actions only; no features, model, candidate, or new masks."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
PACKS={'dev':['paco_part_f0','paco_part_f1'],'held':['paco_part_f2','paco_part_f3'],'coco':['coco_fresh600']}


def main():
    folder=ROOT/'candidate_C1'
    receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['state']=='COMPLETE'
    summary=json.loads((folder/'summary.json').read_text())
    out={}
    for group,packs in PACKS.items():
        rows=[json.loads(line) for p in packs for line in (folder/(p+'.jsonl')).read_text().splitlines()]
        assert len(rows)==summary[group]['n']
        classes=sorted(set(r['class'] for r in rows))
        values=[]
        for c in classes:
            sel=[r for r in rows if r['class']==c]
            b=np.array([r['iu']['B'] for r in sel],dtype=np.int64).sum(0)
            m=np.array([r['iu']['joint_mask'] for r in sel],dtype=np.int64).sum(0)
            edits=np.array([r['edits']['joint_mask'] for r in sel],dtype=np.int64).sum(0)
            at,af,dt,df=edits
            assert m[0]==b[0]+at-dt and m[1]==b[1]+af-df
            j=b[0]/max(b[1],1)
            terms=100*np.array([at,-j*af,-dt,j*df])/max(m[1],1)
            delta=100*(m[0]/max(m[1],1)-j)
            assert abs(terms.sum()-delta)<1e-9
            values.append(dict(c=c,n=len(sel),delta=float(delta),terms_addTP_addFP_delTP_delFP=terms.tolist()))
        means=np.mean([v['terms_addTP_addFP_delTP_delFP'] for v in values],axis=0)
        assert abs(means.sum()-summary[group]['arms']['joint_mask']['delta'])<1e-9
        ed=np.array([r['edits']['joint_mask'] for r in rows]).sum(0)
        out[group]={'classes':len(classes),'classes_gain':sum(v['delta']>1e-9 for v in values),
            'classes_loss':sum(v['delta']<-1e-9 for v in values),
            'class_macro_terms_pp':means.tolist(),'delta_pp':float(means.sum()),
            'deletion_background_fraction':float(ed[3]/max(ed[2]+ed[3],1)),
            'per_class':values}
    (ROOT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='per_class'} for k,v in out.items()},indent=2))
    text=(ROOT/'REPORT.md').read_text()
    marker='推断：此读出未把局部外观—标注布局迁移转为足够干净的范围证据；不能据此断言DINO无所需信息。'
    h=out['held'];t=h['class_macro_terms_pp']
    replacement=(f"实测留出集：删除背景占{h['deletion_background_fraction']*100:.1f}%；按各类IoU精确定价，删假贡献{t[3]:+.2f}点、删真代价{t[2]:+.2f}点，合为{h['delta_pp']:+.2f}点。"
        '推断：可能缺的是跨姿态/尺度仍保持参考标注粒度的语义对应；局部外观排列相似，并未保证标注边界可迁移。此构造未达到所需删除纯度，也未建立超过中心标签控制的增量；不是DINO信息上限证明。')
    if not receipt['passed']:
        text=text.replace(marker,replacement)
    # Preserve the actual five-case diagnostic numbers, explicitly non-representative.
    prior=ROOT/'previous_ablation_smoke/suim/episodes.jsonl'
    if prior.exists():
        rr=[json.loads(x) for x in prior.read_text().splitlines()]
        cs=sorted(set(r['c'] for r in rr))
        mi={}
        for arm in ['native','pre','rerank','smooth','rcg','signed','mean']:
            cnt=[np.array([r['iu'][arm] for r in rr if r['c']==c]).sum(0) for c in cs]
            mi[arm]=float(np.mean([a[0]/max(a[1],1) for a in cnt])*100)
        (ROOT/'previous_ablation_smoke_scores.json').write_text(json.dumps({'n':len(rr),'warning':'5-case implementation smoke only; not an estimate of dataset performance','miou':mi},indent=2)+'\n')
        text=text.replace('另SUIM5例消融实现复现，FoRIS/RCG的I/U全同旧记录，未扩全量；该轮新guide未完成评估。',
            f"另SUIM5例实现烟测（不代表数据集）：FoRIS {mi['native']:.2f}、pre {mi['pre']:.2f}、排序 {mi['rerank']:.2f}、平滑 {mi['smooth']:.2f}、RCG {mi['rcg']:.2f}、signed {mi['signed']:.2f}、MEAN {mi['mean']:.2f}；FoRIS/RCG I/U与旧记录全同，未扩全量。")
    text=text.replace('完整控制/逐例账见summary.json与episodes.jsonl。',
        '完整控制见[summary.json](candidate_C1/summary.json)，逐例账为`candidate_C1/<pack>.jsonl`。')
    (ROOT/'REPORT.md').write_text(text)


if __name__=='__main__':
    main()
