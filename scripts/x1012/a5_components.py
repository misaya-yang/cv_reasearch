"""A5: exact component-union oracle first, then fixed hysteresis/deletion rules."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
from scipy import ndimage
import lab
import plan_common as pc
import plan_report as pr
OUT=lab.ASSETS/'a/x1012_A5'
FIELDS=lab.ASSETS/'a/x1012_P0/fields'
CONNECT=np.ones((3,3),np.uint8)


def load_field(b):
    rec=pc.read(FIELDS/'records'/(pc.key(b)+'.json'))
    assert pc.sha(rec['path'])==rec['sha256']
    with np.load(rec['path'],allow_pickle=False) as f:
        field=lab.minmax(lab.up(__import__('torch').from_numpy(f['ind0_32']).reshape(64,64))).numpy()
        nn=f['nn_query_reference'];back=f['nn_reference_query']
        mutual=back[nn]==np.arange(4096)
        baseline=pc.unpack(f['mask/IND0'])
    return field,mutual,baseline


def selection(tp,fp,gt_area):
    tp=np.asarray(tp,np.int64);fp=np.asarray(fp,np.int64)
    order=np.argsort(-tp/np.maximum(tp+fp,1),kind='stable')
    inter=np.r_[0,np.cumsum(tp[order])];union=gt_area+np.r_[0,np.cumsum(fp[order])]
    cut=int(np.argmax(inter/np.maximum(union,1)));keep=np.zeros(len(tp),bool);keep[order[:cut]]=True
    return keep


def upper():
    pc.setup('cpu');OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'upper.json').exists():print('A5 upper already complete',flush=True);return
    tasks=lab.subset(200,domains=('coco',))
    # All input fields and natural masks exist before any oracle GT read.
    sources=[]
    for b in tasks:
        r=pc.read(FIELDS/'records'/(pc.key(b)+'.json'));assert pc.sha(r['path'])==r['sha256'];sources.append(r)
    pc.write(OUT/'upper_sources_sealed.json',{'n':len(tasks),'sources':sources,'levels':[.3,.4,.5],'new_encoder_calls':0})
    details=[];groups=defaultdict(list)
    for b in tasks:
        field,mutual,baseline=load_field(b);gt,valid=pc.truth(b);pools={}
        for level in (.3,.4,.5):
            labels,n=ndimage.label(field>level,structure=CONNECT)
            area=np.bincount(labels[valid],minlength=n+1)[1:];tp=np.bincount(labels[gt],minlength=n+1)[1:]
            pools[str(level)]={'labels':labels,'tp':tp,'fp':area-tp}
        item={'b':b,'gt':gt,'valid':valid,'baseline':baseline,'pools':pools};details.append(item)
        groups[(str(b['row']['fold']),str(b['row']['global_class_id']))].append(item)
    for slot,items in groups.items():
        gt_area=sum(int(x['gt'].sum()) for x in items)
        for level in ('.3','.4','.5'):
            level='0'+level
            tp=np.concatenate([x['pools'][level]['tp'] for x in items]);fp=np.concatenate([x['pools'][level]['fp'] for x in items])
            keep=selection(tp,fp,gt_area);offset=0
            for x in items:
                p=x['pools'][level];k=len(p['tp']);p['class_keep']=keep[offset:offset+k];offset+=k
    records=[];arms=('FoRIS','IND0','oracle_class_l0.3','oracle_class_l0.4','oracle_class_l0.5','oracle_episode_l0.5')
    for x in details:
        predictions={'FoRIS':lab.foris_cli(x['b']).numpy(),'IND0':x['baseline']}
        for level,p in x['pools'].items():
            predictions['oracle_class_l'+level]=np.r_[False,p['class_keep']][p['labels']]
        p=x['pools']['0.5'];keep=selection(p['tp'],p['fp'],int(x['gt'].sum()))
        predictions['oracle_episode_l0.5']=np.r_[False,keep][p['labels']]
        records.append(pr.score_record(x['b'],predictions))
    report=pr.report('A5-upper',OUT,records,arms,['COCO200上界先行。按8连通分量的TP/FP纯度排序，扫描前缀求精确IoU最大集合。',
        'oracle_class按同(fold,class)所有episode联合优化，才对应class-pooled mIoU上界；oracle_episode另列。',
        '这些oracle读取GT，仅作诊断，不是部署规则。l=.3/.4/.5分别覆盖删除及低阈值生长的分量几何。',
        '区间是固定GT选择掩码的描述性重采样，不解释为无监督候选泛化区间。'])
    table=report['summary']['coco'];gain=table['oracle_class_l0.5']['class_miou']-table['IND0']['class_miou']
    best=max(table[a]['class_miou']-table['IND0']['class_miou'] for a in arms if a.startswith('oracle_class'))
    gate={'state':'UPPER_COMPLETE','coco_component_deletion_gain_pp':gain,'coco_all_level_oracle_gain_pp':best,
          'natural_rules_enabled':gain>=3,'plan_gate':'class oracle at .5 minus IND0 >=3 pp','n':200,'new_encoder_calls':0}
    pc.write(OUT/'upper.json',gate);print(gate,flush=True)


def produce():
    pc.setup('cpu');gate=pc.read(OUT/'upper.json')
    if not gate['natural_rules_enabled']:
        pc.write(OUT/'completion.json',dict(gate,state='COMPLETE_UPPER_GATE_CLOSED'));return
    tasks=pc.seed_rows(150);config=OUT/'config.json'
    for name in ('predictions','records'):(OUT/name).mkdir(exist_ok=True)
    if not config.exists():
        pc.write(config,{'state':'FROZEN','n':len(tasks),'h':[.6,.7,.8,.9],'l':[.3,.4,.5],'gamma':[.7,.8,.9],
                         'mnn_rule':'literal MNN token centers; no reference FG restriction','connectivity':8,
                         'source_sha256':{str(p):pc.sha(p) for p in (Path(__file__),Path(pc.__file__),Path(lab.__file__))},'new_encoder_calls':0})
        pc.write(OUT/'manifest.json',tasks)
    start=time.monotonic();receipts=[]
    for i,b in enumerate(tasks,1):
        name=pc.key(b);rec=OUT/'records'/(name+'.json');path=OUT/'predictions'/(name+'.npz')
        if rec.exists():
            r=pc.read(rec);assert pc.sha(r['path'])==r['sha256'];receipts.append(r);continue
        assert not path.exists()
        with pc.inference_only(b):
            field,mutual,baseline=load_field(b);output={'IND0':pc.pack(baseline)}
            for low in (.3,.4,.5):
                labels,n=ndimage.label(field>low,structure=CONNECT);peaks=ndimage.maximum(field,labels,index=np.arange(1,n+1))
                for high in (.6,.7,.8,.9):
                    output[f'hysteresis_h{high}_l{low}']=pc.pack(np.r_[False,peaks>high][labels])
                if low==.5:
                    for gamma in (.7,.8,.9):output[f'component_gamma{gamma}']=pc.pack(np.r_[False,peaks>=gamma*field.max()][labels])
                    centers=labels[8::16,8::16].flatten();keep=np.zeros(n+1,bool);keep[centers[mutual]]=True;keep[0]=False
                    output['component_MNN']=pc.pack(keep[labels])
        np.savez_compressed(path,**output);r={'index':name,'path':str(path),'sha256':pc.sha(path),'new_encoder_calls':0,'query_GT_reads':0}
        pc.write(rec,r);receipts.append(r)
        if i<=3 or i%100==0:print({'completed':i,'n':len(tasks),'seconds':round(time.monotonic()-start,1)},flush=True)
    pc.write(OUT/'sealed.json',{'state':'ALL_NATURAL_RULES_SEALED','n':len(tasks),'records':receipts,'seconds':time.monotonic()-start,'new_encoder_calls':0})


def score():
    seal=pc.read(OUT/'sealed.json');assert seal['state']=='ALL_NATURAL_RULES_SEALED';indexed={r['index']:r for r in seal['records']};records=[]
    for b in pc.read(OUT/'manifest.json'):
        r=indexed[pc.key(b)];assert pc.sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False) as f:pred={a:pc.unpack(f[a]) for a in f.files}
        pred['FoRIS']=lab.foris_cli(b).numpy();records.append(pr.score_record(b,pred))
    arms=tuple(records[0]['metrics']);result=pr.report('A5',OUT,records,arms,['COCO分量上界达标后运行全部固定自然规则；10域同一(h,l)/gamma，未按域分支。','MNN按query/reference互为最近邻，native token中心映射到分量；未默加FG限制。','先封存全部自然规则再读取GT，confirm未评分。'])
    candidates=[a for a in arms if a not in ('FoRIS','IND0')]
    robust=[a for a in candidates if all(result['summary'][d][a]['class_miou']>=result['summary'][d]['IND0']['class_miou'] for d in result['summary'])]
    pc.write(OUT/'completion.json',{'state':'COMPLETE','uniform_nondecreasing_vs_IND0':robust,'new_encoder_calls':0})
    print({'uniform_nondecreasing_vs_IND0':robust},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('upper','produce','score'));args=parser.parse_args()
    {'upper':upper,'produce':produce,'score':score}[args.mode]()
