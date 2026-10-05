#!/usr/bin/env python3
"""Complete2400 post-map of the frozen composition: keep only extent additions.

No GT during assembly. Retain original B/A+B andsame-count union-RCG addition.
Every deterministic mask is sealed before its separate CPU readout.
"""
import argparse
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def assemble(a):
    import numpy as np
    from ics.experiment import sha,unpack
    if a.out.exists():raise FileExistsError('Fresh output required')
    rows=[json.loads(line) for line in (a.combined/'episodes.jsonl').read_text().splitlines()]
    seals={str(path):json.loads((path/'sealed.json').read_text()) for path in [a.base,a.combined]}
    if len(rows)!=2400 or any(s['state']!='ALL_PREDICTIONS_SEALED' for s in seals.values()):raise ValueError('Full sealed sources required')
    a.out.mkdir();(a.out/'predictions').mkdir();hashes={};receipts={};field_seals={}
    for n,row in enumerate(rows,1):
        key=row['key'];source=a.base if key in seals[str(a.base)]['predictions'] else a.combined
        path=source/'predictions'/(key+'.npz')
        if sha(path)!=seals[str(source)]['predictions'][key]:raise ValueError('Changed source prediction')
        with np.load(path,allow_pickle=False) as z:packed={name:z[name].copy() for name in z.files}
        masks={name:unpack(value) for name,value in packed.items()}
        rcg=masks['rcg'];base=masks['frozen.delete_p.control'];selected=masks['conditional.joint']
        extent=selected&~rcg
        if (base&~selected).any():raise ValueError('Original composition unexpectedly removed pixels afterB')
        union=(masks['native']|masks['mean.control'])&~rcg
        if (extent&~union).any():raise ValueError('Extent must belong to original source union')
        run=Path(row['recheck_run']);oldkey=row.get('source_key',key)
        if str(run) not in field_seals:field_seals[str(run)]=json.loads((run/'sealed.json').read_text())
        field=run/'fields'/(oldkey+'.npz')
        if sha(field)!=field_seals[str(run)]['fields'][oldkey]:raise ValueError('Changed RCG field')
        with np.load(field,allow_pickle=False) as z:values=z['rcg'].copy().repeat(16,0).repeat(16,1)
        locations=np.flatnonzero(union.reshape(-1));locations=locations[np.argsort(-values.reshape(-1)[locations],kind='stable')[:int(extent.sum())]]
        matched=base.copy().reshape(-1);matched[locations]=True
        packed['extent_only.joint']=np.packbits(base|extent)
        packed['extent_only.same_count_union.control']=np.packbits(matched)
        destination=a.out/'predictions'/(key+'.npz');np.savez_compressed(destination,**packed);hashes[key]=sha(destination)
        receipts[key]=dict(source_prediction_sha256=sha(path),field_sha256=sha(field),extent_pixels=int(extent.sum()))
        if n%200==0:print(json.dumps(dict(n=n,total=2400)),flush=True)
    write(a.out/'manifest.json',rows)
    write(a.out/'protocol.json',dict(n=2400,exposure='new deterministic post-map after inspected2400;development only',
        primary='extent_only.joint',query_gt_in_assembly=False,additional_encoder_forwards=0,
        rule='D union (selectedA+B minusRCG);discard restoration insideRCG',
        source_code_sha256=sha(__file__),source_seal_sha256={str(path):sha(path/'sealed.json') for path in [a.base,a.combined]},inputs=receipts))
    write(a.out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=2400,manifest_sha256=sha(a.out/'manifest.json'),
        protocol_sha256=sha(a.out/'protocol.json'),predictions=hashes,query_gt_in_assembly=False))


def score(a):
    import numpy as np
    from ics.experiment import sha,unpack,summarize
    seal=json.loads((a.out/'sealed.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED' or (a.out/'report.json').exists():raise ValueError('Incomplete or already scored')
    for field,name in [('manifest_sha256','manifest.json'),('protocol_sha256','protocol.json')]:
        if sha(a.out/name)!=seal[field]:raise ValueError('Changed scoring input')
    protocol=json.loads((a.out/'protocol.json').read_text())
    if sha(__file__)!=protocol['source_code_sha256']:raise ValueError('Changed implementation')
    rows=json.loads((a.out/'manifest.json').read_text());arrays={};edits={};details=[]
    for row in rows:
        key=row['key'];path=a.out/'predictions'/(key+'.npz')
        if sha(path)!=seal['predictions'][key]:raise ValueError('Changed mask')
        with np.load(row['packet_export'],allow_pickle=False) as z:truth=unpack(z['truth'])
        with np.load(path,allow_pickle=False) as z:masks={name:unpack(z[name]) for name in z.files}
        record=dict(row,iu={})
        for name,mask in masks.items():
            iu=[int((mask&truth).sum()),int((mask|truth).sum())];record['iu'][name]=iu;arrays.setdefault(name,[]).append(iu)
            plus,minus=mask&~masks['rcg'],masks['rcg']&~mask
            values=dict(add_TP=int((plus&truth).sum()),add_FP=int((plus&~truth).sum()),delete_TP=int((minus&truth).sum()),delete_FP=int((minus&~truth).sum()))
            totals=edits.setdefault(name,{k:0 for k in values})
            for k,v in values.items():totals[k]+=v
        details.append(record)
    arrays={name:np.array(values) for name,values in arrays.items()};report,_=summarize(rows,arrays,{})
    report.update(primary='extent_only.joint',exposure=protocol['exposure'],edits_vs_RCG=edits)
    for label,blocks in [('old1200',[0,1]),('new1200',[4,5])]:
        keep=np.array([r['public_batch'] in blocks for r in rows]);sub=[r for r,k in zip(rows,keep) if k]
        result,_=summarize(sub,{name:values[keep] for name,values in arrays.items()},{})
        report[label]=result
    write(a.out/'report.json',report)
    (a.out/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
    print(json.dumps(dict(n=2400,scores=report['scores'],primary=report['contrasts']['extent_only.joint'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['assemble','score'])
    p.add_argument('--base',type=Path);p.add_argument('--combined',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.phase=='assemble' and (a.base is None or a.combined is None):p.error('Both sealed sources required')
    {'assemble':assemble,'score':score}[a.phase](a)


if __name__=='__main__':main()
