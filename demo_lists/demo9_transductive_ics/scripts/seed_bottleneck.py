#!/usr/bin/env python3
"""Fixed-evidence causal diagnostic of INSID3's single-seed intra-image multiplier.

Only removes intra similarity. References, masks, candidates, seed choice/boost and threshold stay fixed.
GT-positive-pixel unblock is a generous diagnostic ceiling, NOT a deployable mask or a region oracle.
No FoRIS result is implied: its seed prior has normalization and downstream disagreement correction.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def scores(s, t, refs, masks, backward='majority', k=5):
    keep = [i for i, m in enumerate(masks) if m.any()]
    if not keep:
        z = torch.zeros(s.P, device=s.fd.device)
        return z, z, []
    refs = [refs[i] for i in keep]; masks = [masks[i] for i in keep]
    w = torch.ones(len(refs), device=s.fd.device)
    proto = F.normalize(sum(wi*s.fd[j][m].mean(0) for wi,j,m in zip(w,refs,masks))/w.sum(), dim=0)
    fwd_score = s.fd[t] @ proto
    fwd = fwd_score > 0
    if not fwd.any(): fwd = fwd_score > float(torch.quantile(fwd_score, .9))
    if backward == 'majority':
        import math
        back = sum(m[s.nnv(t,j)[1]].float() for j,m in zip(refs,masks)) >= math.ceil(len(refs)/2)
    else:
        v = torch.stack([s.nnv(t,j)[0] for j in refs])
        y = torch.stack([m[s.nnv(t,j)[1]].float() for j,m in zip(refs,masks)])
        kk = min(k,len(refs)); top = v.topk(kk,dim=0).indices
        # Exact unweighted reader path, including its all-one weight division.
        ww = w[:,None].expand_as(v).gather(0,top)
        back = (y.gather(0,top)*ww).sum(0)/ww.sum(0) > (.5 if kk>1 else 0.)
    cand = fwd & back
    if not cand.any():
        z = torch.zeros(s.P, device=s.fd.device)
        return z, z, []
    lab, K, Pd, Po, area = s.lab[t], s.K[t], s.Pd[t], s.Po[t], s.area[t]
    count = torch.bincount(lab[cand], minlength=K).float(); aw = count/area
    cs = Pd@proto; seed = int(torch.where(count>0,cs,torch.full_like(cs,-9)).argmax())
    aw[seed] = 1.
    cross = (F.one_hot(lab,K).float().T@fwd_score)/area
    intra = Po@Po[seed]
    base = cross*intra*aw; relaxed = cross*aw
    rows = [dict(cluster=i, seed=i==seed, area=float(area[i]), candidate_fraction=float(count[i]/area[i]),
                 cross=float(cross[i]), intra=float(intra[i]), base=float(base[i]), relaxed=float(relaxed[i])) for i in range(K)]
    return base[lab], relaxed[lab], rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--file',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--prepared-root',default='/root/autodl-tmp/demo9'); ap.add_argument('--limit',type=int,default=30)
    a = ap.parse_args(); sys.path.insert(0,str(Path(a.prepared_root)/'scripts')); import _paths
    from tics import ImageSet, one_shot
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.3')))
    d = torch.load(a.file,weights_only=False); cls=np.array(d['cls']); names=d['names']; n=len(names)//2
    rng=np.random.default_rng(0); recs=[]; t0=time.time(); path=Path(a.out); path.parent.mkdir(parents=True,exist_ok=True)
    report=dict(state='RUNNING',file=a.file,args=vars(a),fold=d['fold'],records=recs,
                scope='Development causal intervention; retains seed area boost; fixed masks/candidates/threshold. GT pixel unblock is only a generous ceiling. No FoRIS experiment.')
    def save():
        acc={}
        for r in recs:
            for key,(i,u) in r['iu'].items():
                row=acc.setdefault(key,{}).setdefault(r['c'],[0.,0.]); row[0]+=i; row[1]+=u
        report['miou']={key:100*float(np.mean([i/max(u,1) for i,u in rows.values()])) for key,rows in acc.items()}
        report['episodes']=len(recs); report['elapsed_s']=time.time()-t0
        tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(report)); tmp.replace(path)
    save()
    try:
        for e in range(min(a.limit,n)):
            c=int(cls[2*e]); ref,q=2*e,2*e+1
            cand=[2*o+1 for o in range(n) if o!=e and cls[2*o]==c and names[2*o+1] not in (names[ref],names[q])]
            seen=set(); cand=[x for x in cand if not(names[x] in seen or seen.add(names[x]))]
            pool=[int(x) for x in rng.permutation(cand)[:15]]; idx=[ref,q]+pool
            gt64=torch.zeros_like(d['gt64'][idx]); gt64[0]=d['gt64'][ref]
            bits=np.zeros_like(d['gt_bits'][idx]); bits[0]=d['gt_bits'][ref]
            s=ImageSet(dict(c=c,names=[names[x] for x in idx],fq=d['fq'][idx],lab=d['lab'][idx],
                            Po=[d['Po'][x] for x in idx],gt64=gt64,gt_bits=bits,S=d['S']))
            J=list(range(1,s.n)); P=one_shot(s,J); donors=list(range(2,s.n)); g0=s.gt64[0]
            predictions={}; info={}
            settings=[('1shot',[0],[g0],'majority'),('naive',[0]+donors,[g0]+[P[j] for j in donors],'majority'),
                      ('pooled',[0]+donors,[g0]+[P[j] for j in donors],'pooled')]
            for tag,refs,masks,mode in settings:
                p=s.predict(1,refs,masks,backward=mode,k=5)
                base,relaxed,rows=scores(s,1,refs,masks,mode)
                if not torch.equal(p,base>s.merge): raise RuntimeError(f'Nonexact baseline reconstruction e={e} tag={tag}')
                predictions[tag]=p; predictions[tag+'_no_intra']=relaxed>s.merge; info[tag]=rows
            # Deployable intervention outputs frozen above. Query/donor truth used only below.
            truth=torch.from_numpy(np.unpackbits(d['gt_bits'][q])).to(s.fd.device).bool().reshape(s.S,s.S)
            true_patch=d['gt64'][idx].to(s.fd.device).reshape(s.n,-1)
            predictions['true_pool']=s.predict(1,[0]+donors,[g0]+[true_patch[j] for j in donors],backward='pooled',k=5)
            metrics={}; added={}
            for tag in ('1shot','naive','pooled'):
                b=s.up(predictions[tag]); alt=s.up(predictions[tag+'_no_intra']); delta=alt&~b
                ceiling=b|(delta&truth)
                for key,full in ((tag,b),(tag+'_no_intra',alt),(tag+'_GT_pixel_unblock',ceiling)):
                    metrics[key]=[float((full&truth).sum()),float((full|truth).sum())]
                added[tag]=dict(tp=float((delta&truth).sum()),fp=float((delta&~truth).sum()),
                                removed_tp=float((b&~alt&truth).sum()),removed_fp=float((b&~alt&~truth).sum()))
                for row in info[tag]:
                    pix=s.lab[1]==row['cluster']; row['gt_fraction_diagnostic']=float(true_patch[1][pix].float().mean())
            full=s.up(predictions['true_pool']); metrics['true_pool']=[float((full&truth).sum()),float((full|truth).sum())]
            recs.append(dict(e=e,c=c,query=names[q],support=names[ref],donor_ids=[names[x] for x in pool],pool=len(pool),
                             iu=metrics,regions=info,added_pixels=added)); del s
            save()
            if (e+1)%10==0: print(e+1,report['miou'],flush=True)
        report['state']='COMPLETED'; save()
    except BaseException as ex:
        report.update(state='ERROR',error=repr(ex)); save(); raise

if __name__=='__main__': main()
