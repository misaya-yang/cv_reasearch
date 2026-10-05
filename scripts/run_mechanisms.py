#!/usr/bin/env python3
"""Finite, single-process cache comparison; prediction is sealed before GT scoring."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['infer','evaluate','all'])
    p.add_argument('--root',type=Path,required=True); p.add_argument('--manifest',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--methods',nargs='+',default=['transport','structure','latent','reconstruction'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--threads',type=int,default=1)
    p.add_argument('--limit',type=int); p.add_argument('--rcg-masks',type=Path)
    a=p.parse_args()
    import numpy as np
    import torch
    from ics.experiment import load_rows,load_inputs,render,unpack,sha,evaluate
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.stage in ('infer','all'):
        if a.manifest is None: p.error('--manifest required for infer')
        if a.out.exists(): raise FileExistsError('Use a fresh output directory')
        rows=load_rows(a.manifest)
        if a.limit is not None:
            if a.limit<1: raise ValueError('limit must be positive')
            # Round robin across folds for smoke; full cohorts preserve manifest order.
            ordered=[]; folds=sorted(set(r['fold'] for r in rows)); bins={f:[r for r in rows if r['fold']==f] for f in folds}
            while any(bins.values()):
                for f in folds:
                    if bins[f]: ordered.append(bins[f].pop(0))
            rows=ordered[:a.limit]
        modules={name:importlib.import_module('ics.methods.'+name) for name in a.methods
                 if name!='rcg' or a.rcg_masks is None}
        a.out.mkdir(parents=True); (a.out/'predictions').mkdir(); (a.out/'fields').mkdir()
        (a.out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
        config={name:dict(config=m.CONFIG,source_sha256=sha(m.__file__)) for name,m in modules.items()}
        (a.out/'config.json').write_text(json.dumps(dict(arms=config,device=a.device,threads=a.threads,parameter_selection='fixed before query scoring'),indent=2)+'\n')
        seal=dict(manifest_sha256=sha(a.out/'manifest.json'),config_sha256=sha(a.out/'config.json'),predictions={},inputs={})
        audits=[]
        for row in rows:
            started=time.perf_counter(); inputs, receipt=load_inputs(a.root,row); seal['inputs'][row['key']]=receipt
            fields={}; masks={}
            for name,m in modules.items():
                functions = [('',m.predict)] if name=='rcg' else [('',m.predict),('.control',m.control)]
                functions += [('.'+key+'.control',fn) for key,fn in getattr(m,'additional_controls',{}).items()]
                for suffix,fn in functions:
                    if a.device=='cuda': torch.cuda.synchronize()
                    begin=time.perf_counter()
                    z,info=fn(*inputs,device=a.device)
                    if a.device=='cuda': torch.cuda.synchronize()
                    fields[name+suffix]=z; masks[name+suffix]=np.packbits(render(z))
                    audits.append(dict(key=row['key'],arm=name+suffix,seconds=time.perf_counter()-begin,info=info))
            if a.rcg_masks:
                with np.load(a.rcg_masks/f"{row['key']}.npz",allow_pickle=False) as f:
                    masks['rcg']=np.packbits(unpack(f['mask'] if 'mask' in f else f['rcg']))
            dest=a.out/'predictions'/f"{row['key']}.npz"
            np.savez_compressed(dest,**masks); np.savez_compressed(a.out/'fields'/dest.name,**fields)
            seal['predictions'][row['key']]=sha(dest)
            (a.out/'audits.json').write_text(json.dumps(audits,indent=2)+'\n')
            print(json.dumps(dict(key=row['key'],seconds=time.perf_counter()-started)),flush=True)
        (a.out/'sealed.json').write_text(json.dumps(seal,indent=2)+'\n')
    if a.stage in ('evaluate','all'):
        result=evaluate(a.root,a.out); print(json.dumps(result['scores'],indent=2))

if __name__=='__main__': main()
