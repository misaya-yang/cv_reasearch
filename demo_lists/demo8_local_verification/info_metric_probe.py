"""A frozen readout intervention: restore full-dimensional matching geometry.

Keep learned query/value projections and final head unchanged; substitute only
support attention affinities. Per-query mean/std match the trained affinities,
so this checks relative geometry rather than merely changing logit temperature.
It is an out-of-training-distribution diagnosis, NOT a validated new method.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['DEMO4_CACHE']='/root/demo8_dependency_pins'
sys.path[:0]=['/root/demo4_cache/env','/root/autodl-tmp/demo4']
import numpy as np
import torch
import torch.nn.functional as F
from info_train import Tokens,Verifier,loader,scores
from train import save,summary


def intervention(model,q,s,mode,basis):
    fields=[];agreement=[]
    for j,m in enumerate(model.streams):
        qn=F.normalize(q[:,j],dim=-1);sn=F.normalize(s[:,j],dim=-1)
        qv=F.normalize(m.project(qn),dim=-1);sv=F.normalize(m.project(sn),dim=-1)
        original=(qv@sv.transpose(-1,-2))/m.logtemp.exp().clamp(.01,1.)
        if mode=='native_debiased':
            qn=F.normalize(qn-(qn@basis)@basis.T,dim=-1)
            sn=F.normalize(sn-(sn@basis)@basis.T,dim=-1)
        new=qn@sn.transpose(-1,-2)
        new=(new-new.mean(-1,keepdim=True))/new.std(-1,keepdim=True,unbiased=False).clamp_min(1e-6)
        new=new*original.std(-1,keepdim=True,unbiased=False)+original.mean(-1,keepdim=True)
        agreement.append((new.argmax(-1)==original.argmax(-1)).float().mean(-1))
        fg,bg=new[:,:,:128],new[:,:,128:]
        fv=fg.softmax(-1)@sv[:,:128];bv=bg.softmax(-1)@sv[:,128:]
        ev=torch.stack([torch.logsumexp(fg,-1)-math.log(128),torch.logsumexp(bg,-1)-math.log(128)],-1)
        fields.append(torch.cat([qv,fv,bv,ev],-1))
    return model.head(torch.cat(fields,-1)).squeeze(-1).sigmoid(),torch.stack(agreement).mean(0)


@torch.inference_mode()
def run(args):
    root=Path(args.root);out=root/'information_readout_v1'
    receipt=json.loads((out/f'{args.tag}.json').read_text());assert receipt['arm']=='R0_final'
    split=json.loads((out/'split.json').read_text());rows=split['rows'];ix=np.asarray(receipt['rows'])
    classes=np.array([rows[i]['c'] for i in ix]);counts=np.zeros((1200,13,2,2),np.int64);added=np.zeros((1200,13,2),np.int64)
    with np.load(root/'pixel_data_v1/labels.npz') as d:
        masks=np.unpackbits(d['masks'],axis=1).reshape(1200,13,4096).astype(bool)
    for i in ix:
        r=rows[i];name='f%d_e%04d.npz'%(r['fold'],r['e'])
        with np.load(root/'dense_observation_v1'/name) as d:counts[i]=d['counts'];added[i]=d['added_counts']
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.16)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bp=root/'information_v1/native_position_basis.pt'
    if bp.exists():basis=torch.load(bp,map_location='cuda',weights_only=True)
    else:
        # Existing official INSID3 preprocessing; blank image only, no task GT.
        from icx.common import build_model
        original=build_model();basis=original.positional_basis.float().clone()
        torch.save(basis.cpu(),bp);del original;torch.cuda.empty_cache()
    assert basis.shape==(1024,500)
    model=Verifier().cuda().eval()
    ck=torch.load(out/f'{args.tag}.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
    ds=Tokens(root,rows,ix,'R0_final',counts,masks);dl=loader(ds,False,2046,4)
    pred={k:[] for k in ['learned','native_full','native_debiased']}
    l1={k:[] for k in ['native_full','native_debiased']};neighbors={k:[] for k in l1};ids=[]
    for i,q,s,y,m,c in dl:
        q=q.cuda();s=s.cuda();m=m.cuda();proper=model(q,s).sigmoid();ids.extend(i.tolist())
        pred['learned'].extend(scores(proper,m).argmax(1).cpu().tolist())
        for mode in l1:
            p,a=intervention(model,q,s,mode,basis)
            pred[mode].extend(scores(p,m).argmax(1).cpu().tolist())
            l1[mode].extend((p-proper).abs().mean(-1).cpu().tolist());neighbors[mode].extend(a.cpu().tolist())
    assert ids==ix.tolist()
    choices={k:np.asarray(v) for k,v in pred.items()}
    assert np.array_equal(choices['learned'],np.asarray(receipt['choices'])),'Original frozen choices differ'
    result=dict(state='COMPLETED',tag=args.tag,episodes=len(ix),
                intervention='only attention affinities replaced; learned value/query channels/head unchanged; per-query logit mean/std matched without GT',
                metrics={k:summary(v,counts[ix],classes,added[ix]) for k,v in choices.items()},
                sensitivity={k:{'probability_l1':float(np.mean(v)),
                                'attention_top1_agreement':float(np.mean(neighbors[k])),
                                'replace_choice_agreement':float(np.mean(choices[k][:,0]==choices['learned'][:,0]))} for k,v in l1.items()},
                rows=ix.tolist(),choices={k:v.tolist() for k,v in choices.items()},
                limits='out-of-training-distribution intervention, single fitted seed/old held fold; cannot certify information loss or establish a trained-method advantage')
    save(out/f'{args.tag}_metric_probe.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','choices']},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--tag',required=True)
    run(ap.parse_args())
