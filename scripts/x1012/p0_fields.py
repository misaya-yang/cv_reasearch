"""P0-3: one MPS job, standard independent fields and GT-free baseline masks."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='4'
import numpy as np
import torch
import lab
import plan_common as pc

OUT=lab.ASSETS/'a/x1012_P0/fields'


def regions(q,k):
    centroids=lab.kmeans(lab.unit(q),k,iters=6)
    labels=(lab.unit(q)@centroids.T).argmax(1)
    onehot=torch.nn.functional.one_hot(labels,k).float()
    return labels,onehot


def compute(b):
    with pc.inference_only(b):
        mask=lab.mask1024(b['row']['reference_mask_path']).to(lab.DEV)
        y=lab.coverage64(mask).flatten()
        r,q=lab.debias(lab.feats(b,'reference')),lab.debias(lab.feats(b,'query'))
        mu=.5*(r.mean(0)+q.mean(0)); rc,qc=lab.unit(r-mu),lab.unit(q-mu)
        sim=qc@rc.T; maximum=sim.max()
        kernel=torch.exp((sim-maximum)/.05)
        df,db=kernel@y,kernel@(1-y); explanation=df+db
        c=df/explanation.clamp_min(1e-30); novelty=lab.minmax(sim.max(1).values)
        labels64,oh=regions(q,64)
        region=lambda v:oh@((oh.T@v)/oh.sum(0).clamp_min(1))
        ind0=region(c); indn=region(c*novelty)
        labels256,_=regions(q,256)
        qr=sim.argmax(1);rq=sim.argmax(0)
        reverse=torch.softmax(sim/.05,dim=0)
        back_f=reverse@y/y.sum().clamp_min(1e-6)
        back_b=reverse@(1-y)/(1-y).sum().clamp_min(1e-6)
        nearest=lab.unit(q)@lab.unit(q).T-torch.eye(4096,device=lab.DEV)*1e9
        qnn=nearest.topk(20,dim=1).indices
        fg=(r*y[:,None]).sum(0)/y.sum().clamp_min(1e-6)
        prototype=qc@lab.unit(fg-mu)
        fmax=sim.masked_fill((y<.5)[None,:],-1e9).max(1).values
        bmax=sim.masked_fill((y>=.5)[None,:],-1e9).max(1).values
        rho=1-(y*(y>=.9)).sum()/y.sum().clamp_min(1e-9)
        floats={'c':c,'log_df':df.clamp_min(1e-30).log()+maximum/.05,
                'log_db':db.clamp_min(1e-30).log()+maximum/.05,
                'log_e':explanation.clamp_min(1e-30).log()+maximum/.05,
                'sim_max':sim.max(1).values,'sim_fg_max':fmax,'sim_bg_max':bmax,
                'prototype':prototype,'reverse_mass':reverse.sum(1),'back_f':back_f,'back_b':back_b,
                'reference_coverage':y}
        out={k:v.cpu().numpy().astype(np.float16) for k,v in floats.items()}
        # FP32 natural fields ensure baseline rendering does not depend on FP16 storage.
        out.update(c32=c.cpu().numpy(),ind0_32=ind0.cpu().numpy(),indn_32=indn.cpu().numpy(),
                   regions64=labels64.cpu().numpy().astype(np.uint16),regions256=labels256.cpu().numpy().astype(np.uint16),
                   nn_query_reference=qr.cpu().numpy().astype(np.uint16),nn_reference_query=rq.cpu().numpy().astype(np.uint16),
                   query_neighbors20=qnn.cpu().numpy().astype(np.uint16),
                   ref_stats=np.array([float(y.mean()),float((y>=.9).sum()),float(rho)],np.float32))
        for name,field in (('T',c),('IND0',ind0),('INDN',indn)):
            pred=lab.minmax(lab.up(field.reshape(64,64)))>.5
            out['mask/'+name]=pc.pack(pred.cpu().numpy())
    return out


def main(probe=False):
    pc.setup('mps')
    assert torch.backends.mps.is_available()
    root=OUT;root.mkdir(parents=True,exist_ok=True)
    for name in ('records','data'): (root/name).mkdir(exist_ok=True)
    config=root/'config.json'
    if not config.exists():
        source=(Path(__file__),Path(pc.__file__),Path(lab.__file__))
        (root/'frozen').mkdir()
        for i,p in enumerate(source):shutil.copyfile(p,root/'frozen'/f'{i}_{p.name}')
        pc.write(config,{'state':'FROZEN','device':'MPS','mps_jobs':1,'new_encoder_calls':0,
                         'formula':'plan IND-0/IND-N/T; full shared mean; temperature .05; K64/K256 iters6',
                         'storage':'standard floats FP16; baseline fields and masks computed FP32',
                         'source_sha256':{str(p):pc.sha(p) for p in source},'query_GT_in_inference':False})
    assert all(pc.sha(p)==h for p,h in pc.read(config)['source_sha256'].items())
    rows=pc.read(lab.ASSETS/'a/x1012_P0/explore.json')
    by_key={pc.key(b):b for b in rows}
    # Original task cohorts first, so A1-A8 can proceed without waiting for every new pair.
    priority=[pc.key(b) for b in lab.subset(300)+lab.ext_subset(None)]
    rows=[by_key[k] for k in priority if k in by_key]+[b for b in rows if pc.key(b) not in set(priority)]
    if probe:rows=[next(b for b in rows if b['dataset']==d) for d in lab.ALL_DOMAINS]
    else:pc.write(root/'manifest.json',rows)
    start=time.monotonic(); completed=0
    for i,b in enumerate(rows,1):
        name=pc.key(b); path=root/'data'/(name+'.npz'); rec=root/'records'/(name+'.json')
        if rec.exists():
            old=pc.read(rec);assert pc.sha(old['path'])==old['sha256'];completed+=1;continue
        assert not path.exists(),'Preserve partial field '+name
        t0=time.monotonic(); result=compute(b); np.savez_compressed(path,**result)
        pc.write(rec,{'index':name,'dataset':b['dataset'],'episode_id':b['row']['episode_id'],
                      'path':str(path),'sha256':pc.sha(path),'seconds':time.monotonic()-t0,
                      'new_encoder_calls':0,'query_GT_reads':0,'raw_inputs':[b['fit'][k]['payload_path'] for k in ('reference_raw','query_raw_identity')]})
        del result;lab.free();completed+=1
        if i<=10 or i%50==0:print({'completed':completed,'total':len(rows),'dataset':b['dataset'],'seconds':round(time.monotonic()-start,1)},flush=True)
    summary={'state':'STANDARD_FIELDS_PROBE' if probe else 'STANDARD_FIELDS_SEALED','n':len(rows),
             'seconds':time.monotonic()-start,'new_encoder_calls':0,'query_GT_reads':0}
    if probe:pc.write(root/'probe.json',summary)
    else:
        pc.write(root/'sealed.json',dict(summary,manifest_sha256=pc.sha(root/'manifest.json'),config_sha256=pc.sha(config),
                 records=[pc.read(root/'records'/(pc.key(b)+'.json')) for b in rows]))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('probe','produce'));a=p.parse_args();main(a.mode=='probe')
