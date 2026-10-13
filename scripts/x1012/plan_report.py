"""Scoring and complete report format for plan tasks, separate from inference."""
from pathlib import Path
import numpy as np
import lab
import plan_common as pc


def metric(pred,gt,valid):
    m=pc.pixel_metrics(pred,gt,valid)
    if m['GT']==0:
        value=0. if m['predicted']==0 else None
    else:
        value=float(abs(np.log2(m['predicted']/m['GT']))) if m['predicted'] else None
    m['abs_log2_area_error']=value
    m['infinite_area_error']=value is None
    return m


def score_record(binding,predictions):
    gt,valid=pc.truth(binding)
    return {'index':pc.key(binding),'episode_id':binding['row']['episode_id'],'dataset':binding['dataset'],
            'fold':binding['row']['fold'],'class_id':binding['row']['global_class_id'],
            'query_photo':str((lab.ASSETS/binding['row']['query_path']).resolve()),
            'metrics':{a:metric(p,gt,valid) for a,p in predictions.items()}}


def summary(records,arms):
    finite=[]
    for r in records:
        ms={a:dict(v,abs_log2_area_error=v['abs_log2_area_error'] or 0.) for a,v in r['metrics'].items()}
        finite.append(dict(r,metrics=ms))
    result=pc.summarize(finite,arms)
    for d,table in result.items():
        rows=[r for r in records if r['dataset']==d]
        for a,s in table.items():
            v=np.median([np.inf if r['metrics'][a]['abs_log2_area_error'] is None else r['metrics'][a]['abs_log2_area_error'] for r in rows])
            s['median_abs_log2_area_error']=float(v) if np.isfinite(v) else 'Infinity'
    return result


def fmt(value):
    if value is None:return '—'
    if isinstance(value,str):return '∞' if value=='Infinity' else value
    return f'{value:.3f}'


def report(task,out,records,arms,notes,extra=None):
    s=summary(records,arms)
    result={'state':'COMPLETE','task':task,'n':len(records),'arms':list(arms),'summary':s,
            'records':records,'new_encoder_calls':0,'notes':notes,'extra':extra or {}}
    pc.write(Path(out)/'report.json',result)
    lines=[f'# {task} 实验报告','',*notes,'',
           '主指标沿用计划lab.Meter：每个(fold,global_class)池化I/U，再对观察槽等权平均；CLI1024。不同于先平均fold的指标。',
           'GT/预测忽略iSAID ignore像素。FN/GT、FP/GT用像素总量之比；面积误差为空预测时为∞。95%区间按query照片配对bootstrap，seed20261012，1000次，每次重新聚合。全部为开发/诊断结果，未评分confirm。','',
           '|域|方法|n|class mIoU|逐例均值|FN/GT|FP/GT|面积误差中位|对FoRIS pp|配对95%区间|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for d,table in s.items():
        for a,v in table.items():
            ci=v['paired_query_photo_bootstrap95_pp']
            lines.append(f"|{d}|{a}|{v['n']}|{fmt(v['class_miou'])}|{fmt(v['episode_mean_iou'])}|{fmt(v['FN_GT'])}|{fmt(v['FP_GT'])}|{fmt(v['median_abs_log2_area_error'])}|{v['delta_to_foris_pp']:+.3f}|[{ci[0]:+.3f},{ci[1]:+.3f}]|")
    lines+=['','|域|方法|<2% n/均值|2–20% n/均值|>20% n/均值|','|---|---|---|---|---|']
    for d,table in s.items():
        for a,v in table.items():
            cells=[f"{v['size_bands'][b]['n']} / {fmt(v['size_bands'][b]['episode_mean_iou'])}" for b in ('<2%','2-20%','>20%')]
            lines.append('|'+d+'|'+a+'|'+'|'.join(cells)+'|')
    lines+=['',f'完整逐例整数、照片分组和诊断产物：`{Path(out)/"report.json"}`。']
    path=lab.ROOT/'docs/research/x1012_tasks'/f'{task}.md';path.parent.mkdir(exist_ok=True)
    path.write_text('\n'.join(lines)+'\n')
    print({'task':task,'state':'COMPLETE','n':len(records),'report':str(path)},flush=True)
    return result
