"""Matched observation readouts, class/image-disjoint development CV.

Train directly for the episode choices' class-aggregated intersection/union,
with native mask as a legal no-change action. All observations and candidates
are frozen before these labels are used. These old episodes are not a fresh test.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,'/root/demo4_cache/env')
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor


def save(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


class Readout(nn.Module):
    def __init__(self,dense,ns):
        super().__init__();self.dense=dense
        if dense:
            self.conv=nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.GELU(),
                                    nn.Conv2d(16,16,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(2))
            self.head=nn.Sequential(nn.Linear(64+ns,64),nn.GELU(),nn.Linear(64,2))
        else:
            # Approximately match the dense head's parameter count despite the
            # simple-statistics control having more input summaries.
            hidden=max(16,round((10210-146)/(ns+49)))
            self.head=nn.Sequential(nn.Linear(ns,hidden),nn.GELU(),nn.Linear(hidden,48),nn.GELU(),nn.Linear(48,2))
    def forward(self,maps,stats):
        n,k=stats.shape[:2];s=stats.reshape(n*k,-1)
        if self.dense:s=torch.cat([self.conv(maps.reshape(n*k,8,28,28)).flatten(1),s],1)
        return self.head(s).reshape(n,k,2)


def summary(chosen,counts,classes,added=None):
    out={}
    for mode,name in enumerate(['replace','add_missing']):
        cc=counts[np.arange(len(classes)),chosen[:,mode],mode]
        d={int(c):cc[classes==c].sum(0) for c in np.unique(classes)}
        out[name]={'class_miou':100*float(np.mean([i/max(u,1) for i,u in d.values()])),
                   'mean_episode_iou':100*float(np.mean(cc[:,0]/np.maximum(cc[:,1],1)))}
        if mode==1 and added is not None:
            a=added[np.arange(len(classes)),chosen[:,mode]]
            out[name].update(true_foreground_added=int(a[:,0].sum()),false_foreground_added=int(a[:,1].sum()),
                             unchanged_fraction=float(np.mean(chosen[:,mode]==0)))
    return out


def split_rows(meta,fold):
    test=np.array([j for j,r in enumerate(meta) if r['fold']==fold])
    blocked={x for j in test for x in (meta[j]['query'],meta[j]['reference'])}
    pool=[j for j,r in enumerate(meta) if r['fold']!=fold and not blocked.intersection([r['query'],r['reference']])]
    # Split connected image components, keeping BOTH support and query roles together.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for j in pool:
        a,b=find(meta[j]['query']),find(meta[j]['reference'])
        if a!=b:parent[max(a,b)]=min(a,b)
    tr=[];val=[]
    for j in pool:
        key=find(meta[j]['query']);v=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%5==0
        (val if v else tr).append(j)
    sets=[{x for j in ix for x in (meta[j]['query'],meta[j]['reference'])} for ix in [tr,val,test]]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert len(tr)>300 and len(val)>30
    return np.array(tr),np.array(val),test


def observation_statistics(maps,common):
    """Simple readout gets exactly the same dense evidence as the CNN.

    Per-channel mean/std/10th/50th/90th percentiles inside candidate, in its
    missing part, and outside candidate. No label or class ID enters features.
    """
    out=np.zeros((*common.shape[:2],common.shape[-1]+90),np.float32)
    out[:,:,:common.shape[-1]]=common
    for i in range(len(maps)):
        for j in range(maps.shape[1]):
            x=maps[i,j].astype(np.float32).reshape(8,-1);c=x[6]>.5;b=x[7]>.5
            vals=[]
            for region in [c,c&~b,~c]:
                z=x[:6,region]
                if not z.shape[1]:vals.extend([0.]*30)
                else:vals.extend(np.r_[z.mean(1),z.std(1),np.quantile(z,[.1,.5,.9],axis=1).ravel()].tolist())
            out[i,j,common.shape[-1]:]=vals
    return out


def run(args):
    inp=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    status={'state':'WAITING_FOR_EXTRACTION','pid':os.getpid(),'models_completed':0}
    save(out/'status.json',status)
    while True:
        if (inp/'error.json').exists():raise RuntimeError((inp/'error.json').read_text())
        if (inp/'status.json').exists() and json.loads((inp/'status.json').read_text())['state']=='COMPLETED':break
        time.sleep(10)
    files=sorted(inp.glob('f*_e*.npz'));assert len(files)==1200
    arrays=[];meta=[]
    for f in files:
        with np.load(f) as d:
            arrays.append({k:d[k].copy() for k in ['maps','stats','counts','basic','added_counts','purity']})
            meta.append({k:d[k].item() for k in ['fold','e','c','query','reference']})
    maps=np.stack([a['maps'] for a in arrays]);stats=np.stack([a['stats'] for a in arrays])
    counts=np.stack([a['counts'] for a in arrays]);basic=np.stack([a['basic'] for a in arrays])
    added=np.stack([a['added_counts'] for a in arrays]);classes=np.array([r['c'] for r in meta])
    del arrays
    # Whole-query evidence is shared by every readout. A local crop alone cannot
    # determine external foreground mass and hence its complete-mask IoU.
    global_info=np.zeros((len(meta),15),np.float32)
    for fold in range(4):
        trees=torch.load(f'/root/demo4_cache/results/l3b_f{fold}.l3.pt',map_location='cpu',weights_only=False)
        for i,r in enumerate(meta):
            if r['fold']!=fold:continue
            t=trees[r['e']];assert t['c']==r['c']
            fields=[t['sim']]+[t['ev'][k] for k in ['q30','fgmax','bgmax','dF30']]
            global_info[i]=[a for field in fields for a in [np.asarray(field,np.float32).mean(),np.asarray(field,np.float32).std(),float(np.quantile(field,.9))]]
        del trees
    # Append before the six legacy crop similarities, whose indices stay fixed.
    stats=np.concatenate([stats[:,:,:-6],np.repeat(global_info[:,None],13,axis=1),stats[:,:,-6:]],axis=2)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    count_gpu=torch.from_numpy(counts.astype(np.float32)).cuda()
    class_gpu=torch.from_numpy(classes).cuda()
    view_stats=[stats]+[observation_statistics(maps[:,j],stats) for j in range(3)]
    no_crop=view_stats[1].copy();no_crop[:,:,stats.shape[-1]-6:stats.shape[-1]]=0
    view_stats.append(no_crop)
    views=[('scalar_task_head',None,0),('global_dense',0,0),('query_crop_dense',1,0),('paired_crop_dense',2,0),
           ('global_simple_stats',None,1),('query_crop_simple_stats',None,2),('paired_crop_simple_stats',None,3),
           ('global_no_crop_stats',None,4)]
    predictions={};foldinfo=[];start=time.monotonic()
    baseline={}
    for j,name in enumerate(['native','complete_support_kde']):
        b=basic[:,j];baseline[name]={'class_miou':100*float(np.mean([b[classes==c].sum(0)[0]/max(b[classes==c].sum(0)[1],1) for c in np.unique(classes)]))}
    f1pick=np.ones((len(meta),2),int)
    baseline['F1_top_node']=summary(f1pick,counts,classes,added)
    old_choices={}
    for k,name in enumerate(['plain_cls','plain_pool','plain_poold','grey_cls','grey_pool','grey_poold']):
        score=stats[:,:,0]*stats[:,:,-6+k];score[:,0]=-np.inf
        pick=score.argmax(1);old_choices[name]=np.repeat(pick[:,None],2,1)
        baseline['old_second_look_'+name]=summary(old_choices[name],counts,classes,added)
    oracle=counts[:,:,:,0]/np.maximum(counts[:,:,:,1],1)
    baseline['candidate_oracle_GT']=summary(oracle.argmax(1),counts,classes,added)
    save(out/'baselines.json',baseline)
    for fold in range(4):
        tr,val,test=split_rows(meta,fold)
        foldinfo.append({'fold':fold,'train':tr.tolist(),'validation':val.tolist(),'test':test.tolist(),
                         'train_classes':sorted(set(classes[tr].tolist())),
                         'test_classes':sorted(set(classes[test].tolist()))})
        assert not set(classes[tr])&set(classes[test])
        tri=torch.from_numpy(tr).cuda();vali=torch.from_numpy(val).cuda()
        for view,mapindex,statindex in views:
            vs=view_stats[statindex]
            mean=vs[tr].reshape(-1,vs.shape[-1]).mean(0);std=vs[tr].reshape(-1,vs.shape[-1]).std(0).clip(.01)
            sg=torch.from_numpy(((vs-mean)/std).astype(np.float32)).cuda()
            dense=mapindex is not None;mg=None if not dense else torch.from_numpy(maps[:,mapindex].astype(np.float32)).cuda()
            for seed in [2037,2038]:
                tag=f'{view}_seed{seed}';receipt=out/f'f{fold}_{tag}.json'
                if receipt.exists():
                    r=json.loads(receipt.read_text());predictions.setdefault(tag,np.zeros((len(meta),2),int))[test]=np.asarray(r['choices']);status['models_completed']+=1;continue
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model=Readout(dense,vs.shape[-1]).cuda()
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                curves=[];best=-np.inf;bestepoch=-1;beststate=None
                present=torch.unique(class_gpu[tri]);target=count_gpu[tri]
                for epoch in range(400):
                    model.train();opt.zero_grad(set_to_none=True)
                    logits=model(None if mg is None else mg[tri],sg[tri])
                    prob=torch.softmax(logits/.1,dim=1)
                    expected=(prob[:,:,:,None]*target).sum(1)
                    totals=torch.zeros((80,2,2),device='cuda').index_add(0,class_gpu[tri],expected)
                    loss=-(totals[present,:,0]/totals[present,:,1].clamp_min(1)).mean()
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    if epoch%5==0:
                        model.eval()
                        with torch.no_grad():choice=model(None if mg is None else mg[vali],sg[vali]).argmax(1).cpu().numpy()
                        metric=summary(choice,counts[val],classes[val])
                        score=np.mean([metric[m]['class_miou'] for m in ['replace','add_missing']])
                        curves.append({'epoch':epoch,'train_soft_class_iou':float(-loss.detach()),'validation':metric})
                        if score>best+1e-5:best=score;bestepoch=epoch;beststate={k:z.detach().cpu().clone() for k,z in model.state_dict().items()}
                        if epoch>=50 and epoch-bestepoch>=50:break
                model.load_state_dict(beststate);model.eval()
                with torch.no_grad():pick=model(None if mg is None else mg[test],sg[test]).argmax(1).cpu().numpy()
                predictions.setdefault(tag,np.zeros((len(meta),2),int))[test]=pick
                r={'fold':fold,'view':view,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
                   'epoch_selected_on_internal_validation':bestepoch,'last_epoch':epoch,'train_episodes':len(tr),
                   'validation_episodes':len(val),'test_episodes':len(test),'choices':pick.tolist(),
                   'metrics':summary(pick,counts[test],classes[test],added[test]),'curves':curves}
                save(receipt,r);torch.save({'state':beststate,'mean':mean,'std':std,'view':view},out/f'f{fold}_{tag}.pt')
                status.update(state='TRAINING',fold=fold,view=view,seed=seed,models_completed=status['models_completed']+1,elapsed_seconds=time.monotonic()-start)
                save(out/'status.json',status);print(json.dumps({k:z for k,z in r.items() if k not in ['choices','curves']}),flush=True)
                del model,opt,beststate
            del mg,sg
        # Strong conventional learned scalar control, same image/class split.
        for vi,vs in enumerate(view_stats):
            tag=['scalar_boost_iou','global_stats_boost','query_crop_stats_boost','paired_crop_stats_boost','global_no_crop_boost'][vi]
            receipt=out/f'f{fold}_{tag}.json'
            if receipt.exists():pick=np.asarray(json.loads(receipt.read_text())['choices'])
            else:
                xx=vs[tr].reshape(-1,vs.shape[-1]);yy=oracle[tr];pred=[]
                for mode in range(2):
                    gb=HistGradientBoostingRegressor(max_iter=400,max_leaf_nodes=31,learning_rate=.06,l2_regularization=1.,random_state=0,early_stopping=False)
                    gb.fit(xx,yy[:,:,mode].reshape(-1));pred.append(gb.predict(vs[test].reshape(-1,vs.shape[-1])).reshape(len(test),13).argmax(1))
                pick=np.stack(pred,1)
                save(receipt,{'choices':pick.tolist(),'metrics':summary(pick,counts[test],classes[test],added[test]),'max_iter':400})
            predictions.setdefault(tag,np.zeros((len(meta),2),int))[test]=pick
    save(out/'split_manifest.json',{'rows':meta,'folds':foldinfo,'contract':'class-disjoint; all query/support IDs disjoint among train/val/test per fold; development CV'})
    report={'state':'COMPLETED','n':len(meta),'main_endpoints':['class-mIoU of replacement','class-mIoU of add-missing action'],
            'scoring':'original-resolution binary mask; identical bilinear64-to-original masks across all readouts',
            'objective':'equal mean over training classes of differentiable expected intersection/expected union; two actions equally weighted',
            'split':'old development cohort, NOT untouched test; grouped internal validation selects epoch',
            'baselines':baseline,'learned':{k:summary(p,counts,classes,added) for k,p in predictions.items()},
            'predictions':{k:p.tolist() for k,p in predictions.items()},'seconds':time.monotonic()-start}
    save(out/'report.json',report);status.update(state='COMPLETED');save(out/'status.json',status)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    try:run(args)
    except Exception:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        save(out/'error.json',{'state':'ERROR','traceback':traceback.format_exc()});raise
