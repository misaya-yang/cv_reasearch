#!/usr/bin/env python3
"""Read every DEV241 identity from the sealed1200 composed output.

Assembly reads masks only. Separate scoring preserves the sealed raw-DINO origin,
complete controls, released INSID3 and both native/RCG replay definitions.
"""
import argparse
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def assemble(a):
    import numpy as np
    from ics.experiment import sha
    if a.out.exists():raise FileExistsError('Fresh output required')
    manifests={};seals={}
    for name,path in [('family',a.family),('candidate',a.candidate),('insid3',a.insid3)]:
        seals[name]=json.loads((path/'sealed.json').read_text())
        if seals[name]['state']!='ALL_PREDICTIONS_SEALED' or sha(path/'manifest.json')!=seals[name]['manifest_sha256']:
            raise ValueError('Unsealed or changed '+name)
        manifests[name]=json.loads((path/'manifest.json').read_text())
    rows=manifests['family']
    if len(rows)!=241 or len(manifests['candidate'])!=1200 or len(manifests['insid3'])!=241:
        raise ValueError('Require complete source cohorts')
    identity=lambda r:(r['c'],Path(r['support']).name,Path(r['query']).name)
    matches={identity(r):r for r in manifests['candidate']}
    released={identity(r):r for r in manifests['insid3']}
    if len(matches)!=1200:raise ValueError('Ambiguous candidate identity')
    a.out.mkdir();(a.out/'predictions').mkdir()
    protocol=dict(n=241,exposure='Previously inspected DEV241; subset projection of sealed1200, not confirmation',
        query_gt_during_assembly=False,new_inference=False,origin='sealed raw unprojected DINOv3 l24 nearest-reference-token mask',
        sources={name:dict(path=str(path),seal_sha256=sha(path/'sealed.json')) for name,path in [('family',a.family),('candidate',a.candidate),('insid3',a.insid3)]},
        source_code={str(Path(__file__).resolve()):sha(__file__),str(REPO/'src/ics/experiment.py'):sha(REPO/'src/ics/experiment.py')},
        input_receipts={},replay_mask_differences={})
    hashes={}
    for row in rows:
        key=row['key'];match=matches[identity(row)];released_row=released[identity(row)]
        source_paths={'family':a.family/'predictions'/(key+'.npz'),
            'candidate':a.candidate/'predictions'/(match['key']+'.npz'),
            'insid3':a.insid3/'predictions'/(released_row['key']+'.npz')}
        source_keys={'family':key,'candidate':match['key'],'insid3':released_row['key']}
        for name,path in source_paths.items():
            if sha(path)!=seals[name]['predictions'][source_keys[name]]:raise ValueError('Changed mask '+name)
        with np.load(source_paths['family'],allow_pickle=False) as z:
            masks={name:z[name].copy() for name in ['native','origin','rcg','astra.control','mean.control','insid3.control']}
        with np.load(source_paths['candidate'],allow_pickle=False) as z:
            for name in z.files:
                if name not in ['native','rcg','astra.control','mean.control']:masks[name]=z[name].copy()
            masks['native.public_replay.control']=z['native'].copy()
            masks['rcg.public_replay.control']=z['rcg'].copy()
        with np.load(source_paths['insid3'],allow_pickle=False) as z:
            for name in ['insid3.release_bilinear.control','insid3.release_crf.control']:masks[name]=z[name].copy()
        differences={name:int(np.unpackbits(masks[name]^masks[name+'.public_replay.control']).sum()) for name in ['native','rcg']}
        protocol['replay_mask_differences'][key]=differences
        protocol['input_receipts'][key]={name:dict(path=str(path),sha256=sha(path)) for name,path in source_paths.items()}
        destination=a.out/'predictions'/(key+'.npz');np.savez_compressed(destination,**masks);hashes[key]=sha(destination)
    write(a.out/'manifest.json',rows);write(a.out/'protocol.json',protocol)
    write(a.out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=241,
        manifest_sha256=sha(a.out/'manifest.json'),protocol_sha256=sha(a.out/'protocol.json'),predictions=hashes,
        query_gt_during_assembly=False,additional_encoder_forwards=0))
    print(json.dumps(dict(n=241,state='ALL_PREDICTIONS_SEALED',native_differences=sum(v['native'] for v in protocol['replay_mask_differences'].values()),rcg_differences=sum(v['rcg'] for v in protocol['replay_mask_differences'].values()))),flush=True)


def score(a):
    import numpy as np
    from ics.experiment import sha,packet,unpack,summarize
    seal=json.loads((a.out/'sealed.json').read_text());protocol=json.loads((a.out/'protocol.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED' or (a.out/'report.json').exists():raise ValueError('Incomplete or already scored')
    for field,name in [('manifest_sha256','manifest.json'),('protocol_sha256','protocol.json')]:
        if sha(a.out/name)!=seal[field]:raise ValueError('Changed scoring input')
    for path,digest in protocol['source_code'].items():
        if sha(path)!=digest:raise ValueError('Changed implementation')
    rows=json.loads((a.out/'manifest.json').read_text());arrays={};corrections={};details=[];truth_hashes={}
    for row in rows:
        key=row['key'];pp=packet(a.root,row);truth_hashes[key]=sha(pp)
        with np.load(pp,allow_pickle=False) as z:truth=unpack(z['truth'])
        path=a.out/'predictions'/(key+'.npz')
        if sha(path)!=seal['predictions'][key]:raise ValueError('Changed prediction')
        with np.load(path,allow_pickle=False) as z:masks={name:unpack(z[name]) for name in z.files}
        origin=masks['origin'];record=dict(row,iu={})
        for name,mask in masks.items():
            iu=[int((mask&truth).sum()),int((mask|truth).sum())];arrays.setdefault(name,[]).append(iu);record['iu'][name]=iu
            plus,minus=mask&~origin,origin&~mask
            corrections.setdefault(name,[]).append(dict(key=key,c=row['c'],fold=row['fold'],batch=row.get('batch','DEV'),
                add_TP=int((plus&truth).sum()),add_FP=int((plus&~truth).sum()),delete_TP=int((minus&truth).sum()),delete_FP=int((minus&~truth).sum())))
        details.append(record)
    report,_=summarize(rows,{name:np.asarray(values) for name,values in arrays.items()},corrections)
    report['corrections_vs_raw_DINO']=report.pop('corrections_vs_native')
    report['corrections_vs_raw_DINO_by_class']=report.pop('corrections_by_class')
    report['corrections_vs_raw_DINO_by_batch']=report.pop('corrections_by_batch')
    report.update(exposure=protocol['exposure'],primary='conditional.joint',origin=protocol['origin'],
        replay_differences=protocol['replay_mask_differences'],prediction_seal_sha256=sha(a.out/'sealed.json'))
    write(a.out/'report.json',report);write(a.out/'truth_receipt.json',truth_hashes)
    (a.out/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
    print(json.dumps(dict(n=241,scores=report['scores'],primary=report['contrasts']['conditional.joint'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['assemble','score']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--family',type=Path);p.add_argument('--candidate',type=Path);p.add_argument('--insid3',type=Path)
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'));a=p.parse_args()
    if a.phase=='assemble' and any(x is None for x in [a.family,a.candidate,a.insid3]):p.error('All three sealed sources required')
    {'assemble':assemble,'score':score}[a.phase](a)


if __name__=='__main__':main()
