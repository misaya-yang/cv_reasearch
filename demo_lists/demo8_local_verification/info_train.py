"""Same verifier, candidates, supervision and training budget; vary evidence.

R0=final+final, R1=final+block12, R2=final+raw16x16 RGB patches.
R0 duplicates its stream to match capacity exactly. Raw patches use zero padding
to the common input width; no new pretrained encoder and no query-GT crops.
This probes usable evidence under a specified readout, NOT mutual information.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.path[:0]=['/root/demo4_cache/env','/root/autodl-tmp/demo4/INSID3']
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.data import build_transform
from train import save, split_rows, summary


class Tokens(Dataset):
    def __init__(self, root, rows, indices, arm, counts, masks, cache=None):
        self.root=root; self.rows=rows; self.indices=indices; self.arm=arm
        self.counts=counts; self.masks=masks; self.transform=build_transform(1024)
        self.cache=cache
    def __len__(self): return len(self.indices)
    def raw(self,name):
        x=self.transform(Image.open(f'/root/demo8_dependency_pins/data/COCO2014/{name}').convert('RGB'))
        p=F.unfold(x[None],16,stride=16)[0].T
        assert p.shape==(4096,768)
        return F.pad(p,(0,256))
    def __getitem__(self,j):
        i=int(self.indices[j]);r=self.rows[i]
        if self.cache is None:
            with np.load(self.root/'information_v1'/f"f{r['fold']}_e{r['e']:04d}.npz") as d:
                a={k:d[k].copy() for k in ['query','support','support_indices','foreground_target']}
        else: a=self.cache[i]
        q=torch.from_numpy(a['query'].copy()).float()
        s=torch.from_numpy(a['support'].copy()).float()
        bank=a['support_indices'].copy(); y=torch.from_numpy(a['foreground_target'].copy())
        # Saved layer order: intermediate, final. R0 and R1 see identical pixels.
        if self.arm=='R0_final': q=torch.stack([q[1],q[1]]);s=torch.stack([s[1],s[1]])
        elif self.arm=='R1_middle': q=q.flip(0);s=s.flip(0)
        elif self.arm=='R2_pixels':
            q=torch.stack([q[1],self.raw(r['query'])]); s=torch.stack([s[1],self.raw(r['reference'])[bank]])
        else: raise ValueError(self.arm)
        return i,q,s,y,torch.from_numpy(self.masks[i].astype(np.float32)),torch.from_numpy(self.counts[i].astype(np.float32))


class Interaction(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Linear(1024,64,bias=False)
        self.logtemp=nn.Parameter(torch.tensor(math.log(.07)))
    def forward(self,q,s):
        # Normalization is common to all input types; no fixed channel compression.
        q=F.normalize(self.project(F.normalize(q,dim=-1)),dim=-1)
        s=F.normalize(self.project(F.normalize(s,dim=-1)),dim=-1)
        a=(q@s.transpose(-1,-2))/self.logtemp.exp().clamp(.01,1.)
        fg,bg=a[:,:,:128],a[:,:,128:]
        fv=fg.softmax(-1)@s[:,:128];bv=bg.softmax(-1)@s[:,128:]
        ev=torch.stack([torch.logsumexp(fg,-1)-math.log(128),torch.logsumexp(bg,-1)-math.log(128)],-1)
        return torch.cat([q,fv,bv,ev],-1)


class Verifier(nn.Module):
    def __init__(self):
        super().__init__();self.streams=nn.ModuleList([Interaction(),Interaction()])
        self.head=nn.Sequential(nn.Linear(388,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,q,s):
        x=torch.cat([m(q[:,j],s[:,j]) for j,m in enumerate(self.streams)],-1)
        return self.head(x).squeeze(-1)


def scores(p,masks):
    """Identical fixed foreground-to-candidate readout; no learned ranking rules."""
    native=masks[:,:1]
    both=torch.stack([masks,torch.maximum(masks,native)],2)
    inter=(both*p[:,None,None]).sum(-1)
    union=both.sum(-1)+p.sum(-1)[:,None,None]-inter
    return inter/union.clamp_min(1e-6)


def loader(ds,shuffle,seed,batch):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=4,
                      persistent_workers=True,prefetch_factor=1,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model,dl,counts,classes,added):
    model.eval();choices=[];ids=[]
    for i,q,s,y,m,c in dl:
        p=model(q.cuda(),s.cuda()).sigmoid();pick=scores(p,m.cuda()).argmax(1).cpu().numpy()
        choices.extend(pick);ids.extend(i.tolist())
    ix=np.array(ids);pick=np.asarray(choices)
    return summary(pick,counts[ix],classes[ix],added[ix]),ix,pick


def run(args):
    root=Path(args.root);out=root/'information_readout_v1';out.mkdir(exist_ok=True)
    status=dict(state='WAITING_FOR_INPUT',pid=os.getpid(),completed_models=0,total_models=6,
                arms=['R0_final','R1_middle','R2_pixels'],fold=args.fold,seeds=[2042,2043],
                readout='same support-query interaction, BCE+Dice+candidate expected IoU',
                stop='first matched held-class/image fold; decide before any expansion',
                limits='usable evidence for this reader; not proof of irreversible information loss',
                fresh='seed2040 fresh800 unopened; no reuse for design')
    save(out/'status.json',status)
    while True:
        f=root/'information_v1/status.json'
        if f.exists() and json.loads(f.read_text())['state']=='COMPLETED': break
        if (root/'information_v1/error.json').exists():raise RuntimeError('Input extraction ERROR; preserve and repair failed stage')
        time.sleep(10)
    rows=json.loads((root/'information_v1/metadata.json').read_text())['rows']
    with np.load(root/'pixel_data_v1/labels.npz') as d:
        masks=np.unpackbits(d['masks'],axis=1).reshape(1200,13,4096).astype(bool)
    counts=[];added=[]
    for r in rows:
        with np.load(root/'dense_observation_v1'/f"f{r['fold']}_e{r['e']:04d}.npz") as d:
            counts.append(d['counts']);added.append(d['added_counts'])
    counts=np.array(counts);added=np.array(added);classes=np.array([r['c'] for r in rows])
    tr,val,test=split_rows(rows,args.fold)
    assert not set(classes[tr])&set(classes[test])
    save(out/'split.json',{'fold':args.fold,'train':tr.tolist(),'validation':val.tolist(),'test':test.tolist(),
                        'rows':rows,'class_and_both_image_roles_disjoint':True})
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.14)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=None
    if args.cache_ram:
        needed=sorted(set(tr.tolist()+val.tolist()+test.tolist()))
        estimate=len(needed)*2*(4096+256)*1024*2
        cg=Path('/sys/fs/cgroup/memory')
        available=int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())
        # Inactive file pages are reclaimable, unlike other jobs' anonymous RAM.
        memstat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        available+=int(memstat.get('total_inactive_file',0))
        if available>=estimate+8*1024**3:
            cache={};status.update(state='LOADING_CPU_CACHE',cache_bytes_estimate=estimate)
            save(out/'status.json',status)
            for j,i in enumerate(needed):
                r=rows[i]
                with np.load(root/'information_v1'/f"f{r['fold']}_e{r['e']:04d}.npz") as d:
                    cache[i]={k:d[k].copy() for k in ['query','support','support_indices','foreground_target']}
                if j%100==0:print('CPU cache',j,len(needed),flush=True)
            status['cache']='existing full FP16 arrays in RAM, shared read-only across forked workers; no disk copy'
        else: status['cache']='stream disk; insufficient cgroup RAM for cache plus 8GiB reserve'
    start=time.monotonic();results={}
    for seed in [2042,2043]:
        for arm in status['arms']:
            tag=f'f{args.fold}_{arm}_s{seed}';receipt=out/f'{tag}.json'
            if receipt.exists(): results[tag]=json.loads(receipt.read_text());status['completed_models']+=1;continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=Verifier().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
            train_dl=loader(Tokens(root,rows,tr,arm,counts,masks,cache),True,seed,args.batch)
            val_dl=loader(Tokens(root,rows,val,arm,counts,masks,cache),False,seed,args.batch)
            test_dl=loader(Tokens(root,rows,test,arm,counts,masks,cache),False,seed,args.batch)
            best=-np.inf;bestepoch=-1;beststate=None;curves=[]
            firstepoch=0
            active=out/'active_last.pt'
            if args.resume and active.exists():
                ck=torch.load(active,map_location='cuda',weights_only=False)
                if ck['arm']==arm and ck['seed']==seed:
                    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
                    beststate=ck['best'];best=ck['best_score'];bestepoch=ck['best_epoch']
                    curves=ck['curves'];firstepoch=ck['epoch']+1
                    torch.set_rng_state(ck['cpu_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
                    train_dl.generator.set_state(ck['loader_rng'].cpu())
            nc={int(c):int((classes[tr]==c).sum()) for c in np.unique(classes[tr])}
            status.update(state='TRAINING',arm=arm,seed=seed)
            save(out/'status.json',status)
            for epoch in range(firstepoch,200):
                model.train();losses=[]; trainsoft=[]
                for i,q,s,y,m,c in train_dl:
                    q=q.cuda();s=s.cuda();y=y.cuda();m=m.cuda();c=c.cuda()
                    opt.zero_grad(set_to_none=True);logits=model(q,s);p=logits.sigmoid()
                    bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none').mean(-1)
                    dice=1-(2*(p*y).sum(-1)+1e-6)/((p+y).sum(-1)+1e-6)
                    sc=scores(p,m);dist=(sc/.1).softmax(1)
                    expected=(dist[:,:,:,None]*c).sum(1)
                    task=(expected[:,:,0]/expected[:,:,1].clamp_min(1)).mean(-1)
                    w=torch.tensor([1/nc[int(classes[k])] for k in i],device='cuda')
                    loss=((bce+dice-task)*w).sum()/w.sum()
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    losses.append(float(loss.detach()));trainsoft.extend(task.detach().cpu().tolist())
                if epoch%5==0:
                    metric,_,_=evaluate(model,val_dl,counts,classes,added)
                    value=np.mean([metric[x]['class_miou'] for x in ['replace','add_missing']])
                    curves.append({'epoch':epoch,'train_loss':float(np.mean(losses)),
                                   'train_expected_episode_iou':100*float(np.mean(trainsoft)), 'validation':metric})
                    if value>best+1e-5:best=value;bestepoch=epoch;beststate=copy.deepcopy(model.state_dict())
                    progress=dict(arm=arm,seed=seed,epoch=epoch,selected_epoch=bestepoch,validation=metric,
                                  elapsed_seconds=time.monotonic()-start)
                    save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                    torch.save({'model':model.state_dict(),'best':beststate,'optimizer':opt.state_dict(),
                                'epoch':epoch,'arm':arm,'seed':seed,'curves':curves,
                                'best_score':best,'best_epoch':bestepoch,
                                'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                                'loader_rng':train_dl.generator.get_state()},out/'active_last.pt')
                    if epoch>=50 and epoch-bestepoch>=40:break
            model.load_state_dict(beststate)
            metric,ix,pick=evaluate(model,test_dl,counts,classes,added)
            training_metric,_,_=evaluate(model,train_dl,counts,classes,added)
            r=dict(arm=arm,seed=seed,fold=args.fold,parameters=sum(p.numel() for p in model.parameters()),
                   selected_epoch=bestepoch,last_epoch=epoch,metrics=metric,train_metrics=training_metric,
                   rows=ix.tolist(),choices=pick.tolist(),curves=curves)
            save(receipt,r);torch.save({'model':beststate,'arm':arm,'seed':seed},out/f'{tag}.pt')
            results[tag]=r;status.update(state='TRAINING',arm=arm,seed=seed,
                         completed_models=status['completed_models']+1,elapsed_seconds=time.monotonic()-start)
            save(out/'status.json',status)
            del model,opt,beststate,train_dl,val_dl,test_dl
    save(out/'report.json',{'state':'COMPLETED','contract':status,'results':results})
    status.update(state='COMPLETED');save(out/'status.json',status)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--cache-ram',action='store_true');args=ap.parse_args()
    try:run(args)
    except Exception:
        out=Path(args.root)/'information_readout_v1';out.mkdir(parents=True,exist_ok=True)
        save(out/'error.json',{'state':'ERROR','traceback':traceback.format_exc()});raise
