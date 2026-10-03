#!/usr/bin/env python3
"""CPU evaluation of saved final masks; no encoder, fitting, threshold or selector.

GT component/pixel deletions are diagnostics only. Largest-component retention
is a same-input naive control, not a proposed method for multi-instance masks.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from analyze_native_angular import unpack_mask
from analyze_rice_core import digest, class_table, aggregate, paired


def best_component_subset(tp, fp, gold_area):
    """Maximize sumTP/(G+sumFP) over disjoint fixed components.

    At an optimum J, adding component j helps iff TP_j>J*FP_j.
    Therefore one optimal subset is a prefix in descending purity order;
    zero-background components come first. Exhaustive testing covers ties,
    empty masks and a largest component that is entirely background.
    """
    tp, fp = np.asarray(tp,dtype=np.int64),np.asarray(fp,dtype=np.int64)
    if tp.shape != fp.shape or np.any(tp<0) or np.any(fp<0) or gold_area<0:
        raise ValueError('Valid disjoint component counts required')
    purity = tp/np.maximum(tp+fp,1)
    order = np.argsort(-purity,kind='stable')
    i, u = np.r_[0,np.cumsum(tp[order])], gold_area+np.r_[0,np.cumsum(fp[order])]
    j = i/np.maximum(u,1)
    choice = int(j.argmax())
    return [int(i[choice]),int(u[choice])],order[:choice]


def inspect(mask, truth, connectivity):
    labels, n = ndimage.label(mask,structure=ndimage.generate_binary_structure(2,connectivity))
    size = np.bincount(labels.reshape(-1),minlength=n+1)[1:]
    tp = np.bincount(labels[truth],minlength=n+1)[1:]
    fp = size-tp; gold=int(truth.sum())
    best, chosen=best_component_subset(tp,fp,gold)
    if n:
        large=int(size.argmax());naive=[int(tp[large]),gold+int(fp[large])]
    else:naive=[0,gold]
    return dict(iu=dict(native=[int(tp.sum()),gold+int(fp.sum())],largest=naive,
                           component_oracle=best,pixel_noFP_oracle=[int(tp.sum()),gold]),
                component_count=int(n),oracle_kept_count=len(chosen),
                false_positive_pixels=int(fp.sum()),
                FP_in_target_containing_components=int(fp[tp>0].sum()),
                FP_in_pure_background_components=int(fp[tp==0].sum()),
                components=[dict(area=int(s),TP=int(t),FP=int(f)) for s,t,f in zip(size,tp,fp)])


def analyze(report):
    if report.get('state')!='COMPLETED' or len(report.get('records',[]))!=40:
        raise ValueError('Completed frozen40 report required')
    output={}
    for conn in [1,2]:
        rows=[]
        for source in report['records']:
            mask=unpack_mask(source['prediction_bits']['foris_crf']).astype(bool)
            truth=unpack_mask(source['evaluator_bits']).astype(bool)
            if mask.shape!=truth.shape:raise ValueError('Saved prediction/evaluator shape differs')
            values=inspect(mask,truth,conn)
            if values['iu']['native']!=source['iu']['foris_crf']:
                raise ValueError('Saved native mask counts do not match report')
            rows.append(dict(fold=source['fold'],e=source['e'],c=source['c'],
                             support=source['support'],query=source['query'],**values))
        score={arm:aggregate(class_table(rows,arm,'iu')[0]) for arm in ['native','largest','component_oracle','pixel_noFP_oracle']}
        fp=sum(r['false_positive_pixels'] for r in rows)
        attached=sum(r['FP_in_target_containing_components'] for r in rows)
        output[str(4 if conn==1 else 8)]=dict(records=rows,scores=score,
            comparisons={arm+' vs native':paired(rows,arm,'native','iu') for arm in score if arm!='native'},
            total_FP=fp,FP_in_target_containing_components=attached,
            fraction_FP_in_target_containing_components=attached/max(fp,1))
    return dict(state='CPU_SAVED_MASK_SPATIAL_ERRORS',tasks=40,folds=4,seed=0,
        scope='Model1024 final masks on reused development tasks. Oracle uses GT and is never a deployable selector or evidence that labels are observable.',
        query_GT_used_for_diagnostic_only=True,GPU_or_encoder_or_new_features=False,connectivity=output)


def self_check():
    rng=np.random.default_rng(2056)
    for count in range(8):
        for _ in range(10):
            t=rng.integers(0,8,count);f=rng.integers(0,8,count);g=int(t.sum()+rng.integers(1,9))
            best,_=best_component_subset(t,f,g)
            value=best[0]/max(best[1],1)
            expected=max(sum(t[i] for i in s)/(g+sum(f[i] for i in s))
                         for k in range(count+1) for s in itertools.combinations(range(count),k))
            assert abs(value-expected)<1e-12
    pred=np.zeros((7,7),bool);gold=np.zeros_like(pred)
    pred[0,0]=True;pred[1,1]=True;pred[4:7,4:7]=True;gold[0,0]=True;gold[1,1]=True
    a,b=inspect(pred,gold,1),inspect(pred,gold,2)
    assert a['component_count']==3 and b['component_count']==2
    assert a['iu']['native']==[2,11] and a['iu']['largest']==[0,11]
    assert a['iu']['component_oracle']==[2,2]
    assert inspect(np.zeros_like(pred),gold,1)['iu']['native']==[0,2]
    print(json.dumps(dict(state='CPU_SPATIAL_ERROR_AUDIT_CHECKED',exhaustive_subset_fixtures=80,connectivity_and_wrong_largest=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path);p.add_argument('--out',type=Path);p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:self_check();return
    if not a.report or not a.out:p.error('--report and --out required')
    if a.out.exists():raise ValueError('Preserve previous analysis')
    result=analyze(json.loads(a.report.read_text()));result['report_sha256']=digest(a.report)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for conn,data in result['connectivity'].items():
        print(json.dumps(dict(connectivity=conn,scores={k:v['fold_mean_class_miou'] for k,v in data['scores'].items()},fraction_FP_attached=data['fraction_FP_in_target_containing_components'])))


if __name__=='__main__':main()
