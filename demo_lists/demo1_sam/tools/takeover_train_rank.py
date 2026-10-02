"""Matched small rankers: teacher imitation versus GT supervision, with ablations."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

SEEDS=(2027,2028,2029)
VARIANTS=('geometry_distilled','latent_distilled','geometry_supervised','latent_supervised')


class Ranker(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def get_arrays(path):
    with np.load(path,allow_pickle=False) as d:out={k:d[k].copy() for k in d.files}
    # Idempotent schema normalization also handles the initial cached export.
    out['features'][:,:,30:33]=np.sort(out['features'][:,:,30:33],axis=-1)
    return out


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    train=get_arrays(a.train);val=get_arrays(a.validation)
    assert not set(train['image_ids'])&set(val['image_ids'])
    geometry=json.loads(a.train.with_suffix('.json').read_text())['geometry_dim']
    protocol={'seeds':SEEDS,'variants':VARIANTS,'max_epochs':500,'minimum_epochs':100,'patience':50,'batch_size':128,'lr':1e-3,'weight_decay':.01,'dropout':.1,
              'training_images':len(set(train['image_ids'])),'validation_images':len(set(val['image_ids'])),'inference':'one original frozen SAM decode plus candidate ranker, tokens1..3 only',
              'teacher':getattr(a,'teacher_description','fixed four-prompt consistency choice')+'; no GT quality labels used in distilled loss or epoch selection; benchmark prompts are annotation-derived, not annotation-free sampling',
              'supervised_control':'same architecture, training oracle choice; epoch chosen by validation meanIoU across original three regimes',
              'candidate_permutation':'shared pointwise scorer; sorted pairwise-IoU features; no candidate index feature','test_data_read_during_training':False}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    for variant in VARIANTS:
        dim=geometry if variant.startswith('geometry') else train['features'].shape[-1]
        tx=torch.from_numpy(train['features'][...,:dim]).cuda();vx=torch.from_numpy(val['features'][...,:dim]).cuda()
        mean=tx.reshape(-1,dim).mean(0);std=tx.reshape(-1,dim).std(0).clamp_min(1e-3)
        tx=(tx-mean)/std;vx=(vx-mean)/std
        distilled=variant.endswith('distilled')
        ty=torch.from_numpy(train['teacher_choice'] if distilled else train['quality'].argmax(-1)).long().cuda()
        vy=torch.from_numpy(val['teacher_choice'] if distilled else val['quality'].argmax(-1)).long().cuda()
        quality=torch.from_numpy(val['quality']).cuda()
        for seed in SEEDS:
            folder=a.output/f'{variant}_seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'report.json').exists() and json.loads((folder/'report.json').read_text()).get('status')=='COMPLETED':
                records.append(json.loads((folder/'report.json').read_text()));continue
            torch.manual_seed(seed);model=Ranker(dim).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
            best=float('inf');best_epoch=0;bad=0;curves=[];start_epoch=1
            if (folder/'last.pt').exists():
                resume=torch.load(folder/'last.pt',weights_only=True);model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer']);best=resume['best'];best_epoch=resume['best_epoch'];bad=resume['bad'];start_epoch=resume['epoch']+1;curves=resume['curves'];torch.cuda.set_rng_state(resume['CUDA_rng'])
            start=time.monotonic()
            for epoch in range(start_epoch,501):
                model.train();perm=torch.randperm(len(tx),device='cuda:0');total=0.
                for indices in perm.split(128):
                    optimizer.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(tx[indices]),ty[indices]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();total+=float(loss)*len(indices)
                model.eval()
                with torch.no_grad():
                    scores=model(vx);ce=float(nn.functional.cross_entropy(scores,vy));choices=scores.argmax(-1);iou=float(quality[torch.arange(len(quality),device='cuda:0'),choices].mean());accuracy=float((choices==vy).float().mean())
                criterion=ce if distilled else -iou
                row={'epoch':epoch,'train_cross_entropy':total/len(tx),'validation_cross_entropy':ce,'validation_target_agreement':accuracy,'validation_mean_iou':iou}
                curves.append(row)
                checkpoint={'model':model.state_dict(),'dim':dim,'geometry_dim':geometry,'mean':mean,'std':std,'variant':variant,'seed':seed,'epoch':epoch,'sort_pair_columns':[30,31,32]}
                if criterion<best-1e-5:
                    best=criterion;best_epoch=epoch;bad=0;torch.save(checkpoint,folder/'best.pt')
                else:bad+=1
                if epoch%25==0:
                    torch.save({**checkpoint,'optimizer':optimizer.state_dict(),'best':best,'best_epoch':best_epoch,'bad':bad,'curves':curves,'CUDA_rng':torch.cuda.get_rng_state()},folder/'last.pt')
                    (folder/'curves.json').write_text(json.dumps(curves,indent=2)+'\n');print(json.dumps({'variant':variant,'seed':seed,'best_epoch':best_epoch,**row}),flush=True)
                if epoch>=100 and bad>=50:break
            summary={'status':'COMPLETED','variant':variant,'seed':seed,'epochs':epoch,'selected_epoch':best_epoch,'selection_criterion':'teacher CE' if distilled else 'validation meanIoU','parameter_count':sum(p.numel() for p in model.parameters()),'wall_seconds':time.monotonic()-start,'curves':curves}
            (folder/'report.json').write_text(json.dumps(summary,indent=2)+'\n');records.append(summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='curves'}),flush=True)
    (a.output/'report.json').write_text(json.dumps({'status':'COMPLETED','records':records},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('train','validation','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--teacher-description',default='fixed four-prompt consistency choice')
    main(p.parse_args())
