"""Check whether a frozen verifier actually responds to the supplied support.

Keep query/candidates/checkpoint fixed. Swap foreground/background bank roles,
or substitute another held class's support. These are input interventions, NOT
new methods or correctly specified segmentation tasks. No model selection here.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,'/root/demo4_cache/env')
import numpy as np
import torch
from info_train import Tokens,Verifier,loader,scores
from train import save,summary


@torch.inference_mode()
def run(args):
    root=Path(args.root);out=root/'information_readout_v1'
    receipt=json.loads((out/f'{args.tag}.json').read_text())
    split=json.loads((out/'split.json').read_text());rows=split['rows']
    ix=np.asarray(receipt['rows']);classes=np.array([rows[i]['c'] for i in ix])
    with np.load(root/'pixel_data_v1/labels.npz') as d:
        masks=np.unpackbits(d['masks'],axis=1).reshape(1200,13,4096).astype(bool)
    counts=np.zeros((1200,13,2,2),np.int64);added=np.zeros((1200,13,2),np.int64)
    for i in ix:
        r=rows[i];name='f%d_e%04d.npz'%(r['fold'],r['e'])
        with np.load(root/'dense_observation_v1'/name) as d:counts[i]=d['counts'];added[i]=d['added_counts']
    # Donors depend only on class metadata; chosen before any output inspection.
    donors={int(i):int(next(j for j in np.roll(ix,-n-1) if rows[j]['c']!=rows[i]['c'])) for n,i in enumerate(ix)}
    ds=Tokens(root,rows,ix,receipt['arm'],counts,masks)
    dl=loader(ds,False,2045,4)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.08)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Verifier().cuda().eval()
    ck=torch.load(out/f'{args.tag}.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(ck['model'])
    pred={k:[] for k in ['proper','swap_fg_bg','other_class_support']}
    l1={k:[] for k in ['swap_fg_bg','other_class_support']};ids=[]
    for i,q,s,y,m,c in dl:
        q=q.cuda();s=s.cuda();m=m.cuda()
        proper=model(q,s).sigmoid();ids.extend(i.tolist())
        pred['proper'].extend(scores(proper,m).argmax(1).cpu().tolist())
        wrong=[]
        for a in i.tolist():
            donor=rows[donors[a]];name='f%d_e%04d.npz'%(donor['fold'],donor['e'])
            with np.load(root/'information_v1'/name) as d:
                bank=d['support_indices'].copy();z=torch.from_numpy(d['support'].copy()).float()
            if receipt['arm']=='R0_final':z=torch.stack([z[1],z[1]])
            elif receipt['arm']=='R1_middle':z=z.flip(0)
            else:z=torch.stack([z[1],ds.raw(donor['reference'])[bank]])
            wrong.append(z)
        inputs={'swap_fg_bg':s.roll(128,dims=2),'other_class_support':torch.stack(wrong).cuda()}
        for name,inp in inputs.items():
            p=model(q,inp).sigmoid();pred[name].extend(scores(p,m).argmax(1).cpu().tolist())
            l1[name].extend((p-proper).abs().mean(-1).cpu().tolist())
    ids=np.asarray(ids);assert ids.tolist()==ix.tolist()
    choices={k:np.asarray(v) for k,v in pred.items()}
    assert np.array_equal(choices['proper'],np.asarray(receipt['choices'])),'Frozen proper-support choices drift'
    result=dict(state='COMPLETED',tag=args.tag,episodes=len(ix),arm=receipt['arm'],
                contract='fixed frozen checkpoint/query/candidates; incorrect support is an intervention, not a benchmark method',
                metrics={k:summary(v,counts[ix],classes,added[ix]) for k,v in choices.items()},
                sensitivity={k:{'mean_probability_l1':float(np.mean(v)),
                              'replace_choice_agreement':float(np.mean(choices[k][:,0]==choices['proper'][:,0])),
                              'add_choice_agreement':float(np.mean(choices[k][:,1]==choices['proper'][:,1]))} for k,v in l1.items()},
                choices={k:v.tolist() for k,v in choices.items()},rows=ix.tolist(),
                inference_scope='conditioning and transferability diagnostic; does not establish irreversible information loss')
    save(out/f'{args.tag}_conditioning.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['choices','rows']},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--tag',required=True)
    run(ap.parse_args())
