"""A3 false-positive anatomy and A8 class/size breakdown on all exposed originals."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import lab
import plan_common as pc
import plan_report as pr
FIELDS=lab.ASSETS/'a/x1012_P0/fields'


def tasks(domains):
    # All exposed originals in the fixed P0 exploration list, never reserved pairs.
    return [b for b in pc.read(lab.ASSETS/'a/x1012_P0/explore.json')
            if b['dataset'] in domains and b['cohort']!='x1012_P0_new_pair']


def field_source(b):
    r=pc.read(FIELDS/'records'/(pc.key(b)+'.json'));assert pc.sha(r['path'])==r['sha256'];return r


def predictions(b):
    r=field_source(b)
    with np.load(r['path'],allow_pickle=False) as z:
        pred=pc.unpack(z['mask/IND0']);c=lab.up(torch.from_numpy(z['c32']).reshape(64,64)).numpy()
        e=lab.up(torch.from_numpy(z['log_e'].astype(np.float32)).reshape(64,64)).numpy()
    return pred,c,e


def semantic_map(b):
    row=b['row'];p=lab.ASSETS/'demo4_cache/data/COCO2014/annotations/val2014'/(Path(row['query_path']).stem+'.png')
    if not p.exists():return None
    with Image.open(p) as im:
        raw=np.asarray(im).copy()
    assert raw.ndim==2
    crop=row.get('query_crop')
    if crop:
        x0,y0,x1,y1=crop;raw=raw[y0:y1,x0:x1]
    yy=np.arange(1024)*raw.shape[0]//1024;xx=np.arange(1024)*raw.shape[1]//1024
    return raw[yy[:,None],xx[None,:]]


def a3():
    pc.setup('cpu');out=lab.ASSETS/'a/x1012_A3';out.mkdir(exist_ok=True)
    selected=tasks(('coco','lvis','suim','isic'))
    sources=[field_source(b) for b in selected]
    pc.write(out/'sources_sealed_before_anatomy.json',{'n':len(selected),'sources':sources,'new_encoder_calls':0,'confirm_scored':False})
    records=[];components=[];totals=defaultdict(Counter);semantic=Counter();start=time.monotonic()
    for i,b in enumerate(selected,1):
        p,c,e=predictions(b);g,v=pc.truth(b);f=lab.foris_cli(b).numpy()
        records.append(pr.score_record(b,{'FoRIS':f,'IND0':p}))
        labels,n=ndimage.label(p,structure=np.ones((3,3),np.uint8));area=np.bincount(labels[v],minlength=n+1)
        tp=np.bincount(labels[g],minlength=n+1);fp=area-tp;fg_peak=float(c[g].max()) if g.any() else None
        for k in range(1,n+1):
            false=(labels==k)&v&~g
            if not false.any():continue
            kind='connected_to_true_prediction' if tp[k]>0 else 'separate_false_component'
            value=int(fp[k]);totals[b['dataset']][kind]+=value;totals[b['dataset']]['FP']+=value
            peak=float(c[false].max());ratio=peak/max(fg_peak,1e-12) if fg_peak is not None else None
            components.append({'index':pc.key(b),'episode_id':b['row']['episode_id'],'dataset':b['dataset'],
                               'component':k,'TP':int(tp[k]),'FP':value,'type':kind,'c_max':peak,
                               'c_peak_to_GT_peak':ratio,'log_E_quantiles':[float(x) for x in np.quantile(e[false],[.1,.5,.9])],
                               'area_to_GT':float(area[k]/max(int(g.sum()),1))})
        if b['dataset']=='coco':
            sem=semantic_map(b)
            if sem is not None:
                counts=np.bincount(sem[p&v&~g],minlength=256)
                semantic.update({int(k):int(x) for k,x in enumerate(counts) if x})
        if i%250==0:print({'task':'A3','done':i,'n':len(selected),'seconds':round(time.monotonic()-start,1)},flush=True)
    fractions={d:{'FP':x['FP'],'connected_fraction':x['connected_to_true_prediction']/max(x['FP'],1),
                  'separate_fraction':x['separate_false_component']/max(x['FP'],1)} for d,x in totals.items()}
    pc.write(out/'anatomy.json',{'components':components,'fractions':fractions,
             'coco_encoded_semantic_FP_counts':dict(semantic),'unlabelled_codes':[0,255],
             'semantic_interpretation':'Encoded source labels; unlabelled is not proven physical background',
             'new_encoder_calls':0,'seconds':time.monotonic()-start})
    pr.report('A3',out,records,('FoRIS','IND0'),['只分析固定explore中全部已曝光原任务，未评分confirm。','8连通分量按是否含实际预测TP区分外溢与分离干扰，份额按FP像素面积计算。','峰值比及logE分位数是诊断，语义混淆不构成冻结特征不可解定理。'],extra=fractions)
    path=lab.ROOT/'docs/research/x1012_tasks/A3.md'
    with path.open('a') as f:
        f.write('\n## 误报面积归因\n\n|域|FP像素|与预测TP相连|分离干扰|\n|---|---:|---:|---:|\n')
        for d,x in fractions.items():f.write(f"|{d}|{x['FP']}|{x['connected_fraction']:.3%}|{x['separate_fraction']:.3%}|\n")
        f.write('\nCOCO误报的源语义编码面积排名：'+str(semantic.most_common(12))+'。0/255为未标注编码，不视为已证明真实背景。\n')


def picture(b,gt,foris,ind,c,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ics.official_data import decoded_rgb
    r=b['row'];ref=decoded_rgb(lab.ASSETS/r['reference_path'],r.get('reference_crop'))
    query=decoded_rgb(lab.ASSETS/r['query_path'],r.get('query_crop'))
    refmask=lab.mask1024(r['reference_mask_path']).numpy()
    ref=np.asarray(ref.resize((1024,1024),Image.Resampling.BILINEAR)).copy()
    overlay=ref.astype(np.float32);overlay[refmask]=.5*overlay[refmask]+.5*np.array([0,255,80]);overlay=np.uint8(overlay)
    fig,axes=plt.subplots(1,6,figsize=(18,3.3))
    for ax,data,title in zip(axes,(overlay,query,gt,foris,ind,c),('Reference + mask','Query','GT','FoRIS','IND-0','c')):
        ax.imshow(data,cmap='viridis' if title=='c' else 'gray',vmin=0,vmax=1 if title in ('GT','FoRIS','IND-0','c') else 255);ax.set_title(title);ax.axis('off')
    fig.suptitle(r['episode_id']);fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def a8():
    pc.setup('cpu');out=lab.ASSETS/'a/x1012_A8';out.mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    selected=tasks(('suim','isic','coco'));sources=[field_source(b) for b in selected]
    pc.write(out/'sources_sealed_before_scoring.json',{'n':len(selected),'sources':sources,'new_encoder_calls':0,'confirm_scored':False})
    records=[];by_key={pc.key(b):b for b in selected}
    for b in selected:
        p,c,e=predictions(b);f=lab.foris_cli(b).numpy();records.append(pr.score_record(b,{'FoRIS':f,'IND0':p}))
    result=pr.report('A8',out,records,('FoRIS','IND0'),['逐类/面积诊断使用固定explore中全部已曝光原任务；未使用confirm。','差距最大12例/域是事后解释图，不能作为泛化证据或方法选择集。'])
    classes=defaultdict(list)
    for r in records:classes[(r['dataset'],str(r['fold']),str(r['class_id']))].append(r)
    table=[]
    for (d,fold,cls),rs in classes.items():
        item={'dataset':d,'fold':fold,'class_id':cls,'n':len(rs),'metrics':{}}
        for a in ('FoRIS','IND0'):
            vals=[r['metrics'][a] for r in rs];gt=sum(v['GT'] for v in vals)
            item['metrics'][a]={'IoU':100*sum(v['I'] for v in vals)/max(sum(v['U'] for v in vals),1),
                                 'episode_mean':float(np.mean([v['IoU'] for v in vals])),
                                 'FN_GT':sum(v['FN'] for v in vals)/max(gt,1),'FP_GT':sum(v['FP'] for v in vals)/max(gt,1)}
        item['delta']=item['metrics']['IND0']['IoU']-item['metrics']['FoRIS']['IoU'];table.append(item)
    isic=sorted([r for r in records if r['dataset']=='isic'],key=lambda r:r['metrics']['FoRIS']['target_fraction']);deciles=[]
    for j,indices in enumerate(np.array_split(np.arange(len(isic)),10),1):
        rs=[isic[k] for k in indices];item={'decile':j,'n':len(rs),'mean_area_fraction':float(np.mean([r['metrics']['FoRIS']['target_fraction'] for r in rs])),'arms':{}}
        for a in ('FoRIS','IND0'):
            vals=[r['metrics'][a] for r in rs];gt=sum(v['GT'] for v in vals);item['arms'][a]={'episode_mean_iou':float(np.mean([v['IoU'] for v in vals])),'FN_GT':sum(v['FN'] for v in vals)/max(gt,1),'FP_GT':sum(v['FP'] for v in vals)/max(gt,1)}
        deciles.append(item)
    pc.write(out/'class_size_breakdown.json',{'classes':table,'isic_area_deciles':deciles})
    path=lab.ROOT/'docs/research/x1012_tasks/A8.md'
    with path.open('a') as f:
        f.write('\n## 逐类差距\n\n|域|fold/class|n|FoRIS IoU|IND0 IoU|差值|IND0 FN/GT|IND0 FP/GT|\n|---|---|---:|---:|---:|---:|---:|---:|\n')
        for t in sorted(table,key=lambda t:(t['dataset'],t['delta'])):
            a,b=t['metrics']['FoRIS'],t['metrics']['IND0'];f.write(f"|{t['dataset']}|{t['fold']}/{t['class_id']}|{t['n']}|{a['IoU']:.3f}|{b['IoU']:.3f}|{t['delta']:+.3f}|{b['FN_GT']:.3f}|{b['FP_GT']:.3f}|\n")
        f.write('\n## ISIC 面积十分位\n\n|十分位|n|平均面积份额|FoRIS均值|IND0均值|IND0 FN/GT|IND0 FP/GT|\n|---|---:|---:|---:|---:|---:|---:|\n')
        for t in deciles:
            a,b=t['arms']['FoRIS'],t['arms']['IND0'];f.write(f"|{t['decile']}|{t['n']}|{t['mean_area_fraction']:.4f}|{a['episode_mean_iou']:.3f}|{b['episode_mean_iou']:.3f}|{b['FN_GT']:.3f}|{b['FP_GT']:.3f}|\n")
        f.write('\n## 每域差距最大12例\n')
        for d in ('suim','isic','coco'):
            worst=sorted([r for r in records if r['dataset']==d],key=lambda r:r['metrics']['IND0']['IoU']-r['metrics']['FoRIS']['IoU'])[:12]
            for r in worst:
                b=by_key[r['index']];p,c,e=predictions(b);g,v=pc.truth(b);foris=lab.foris_cli(b).numpy();image=out/'figures'/(r['index']+'.png')
                picture(b,g,foris&v,p&v,c,image);f.write(f"\n![{r['episode_id']}]({image})\n")
    pc.write(out/'completion.json',{'state':'COMPLETE','n':len(records),'figures':36,'new_encoder_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=('A3','A8'));a=p.parse_args();{'A3':a3,'A8':a8}[a.task]()
