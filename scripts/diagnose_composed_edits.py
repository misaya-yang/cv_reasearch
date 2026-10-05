#!/usr/bin/env python3
"""GT diagnostics after sealing; never inference inputs or routing rules.

Separate connected-region recovery, existing-region extent and semantic false
pixels. Audit the frozen background bank against a reference-NN filtered bank.
"""
import argparse
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    from ics.experiment import sha,unpack,packet
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--annotations',type=Path,default=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations'))
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'));a=p.parse_args()
    if a.out.exists():raise FileExistsError('Fresh output required')
    seal=json.loads((a.run/'sealed.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED' or sha(a.run/'manifest.json')!=seal['manifest_sha256']:raise ValueError('Unsealed source')
    rows=json.loads((a.run/'manifest.json').read_text());providers=json.loads(a.source_manifest.read_text())
    identity=lambda r:(r['c'],Path(r['support']).name,Path(r['query']).name)
    provider={identity(r):r for r in providers};details=[];class_counts={}
    covers=lambda mask:mask.reshape(64,16,64,16).mean((1,3)).reshape(-1)
    for row in rows:
        key=row['key'];path=a.run/'predictions'/(key+'.npz')
        if sha(path)!=seal['predictions'][key]:raise ValueError('Changed prediction')
        with np.load(path,allow_pickle=False) as z:m={name:unpack(z[name]) for name in z.files}
        with np.load(packet(a.root,row),allow_pickle=False) as z:truth=unpack(z['truth'])
        ann=a.annotations/Path(row['query']).with_suffix('.png')
        with Image.open(ann) as image:labels=np.asarray(image).copy()
        yy=np.floor(np.arange(1024)*labels.shape[0]/1024).astype(int)
        xx=np.floor(np.arange(1024)*labels.shape[1]/1024).astype(int);labels=labels[yy[:,None],xx[None,:]]
        if not np.array_equal(labels==row['c']+1,truth):raise ValueError('Annotation/packet renderer mismatch')
        regions,nregions=ndimage.label(truth,structure=np.ones((3,3)))
        sizes=np.bincount(regions.reshape(-1),minlength=nregions+1)
        distance=ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape,np.inf)
        other=(labels>0)&(labels<=80)&(labels!=row['c']+1)
        unknown=(labels!=0)&~other&~truth
        selected=m['conditional.joint'];d=m['frozen.delete_p.control'];rcg=m['rcg']
        if (d&~selected).any():raise ValueError('Composition removed additional pixels after frozen B')
        restore=selected&rcg&~d;extent=selected&~rcg
        row_counts=dict(B_i=int((d&truth).sum()),B_u=int((d|truth).sum()),
            restore_TP=int((restore&truth).sum()),restore_FP=int((restore&~truth).sum()),
            extent_TP=int((extent&truth).sum()),extent_FP=int((extent&~truth).sum()))
        total=class_counts.setdefault(str(row['c']),{k:0 for k in row_counts})
        for k,v in row_counts.items():total[k]+=v
        semantic={}
        for origin in ['origin','rcg','frozen.delete_p.control']:
            old=m[origin];addition=selected&~old;deletion=old&~selected
            before=np.bincount(regions[old&truth],minlength=nregions+1)
            after=np.bincount(regions[selected&truth],minlength=nregions+1)
            missed=(before==0)&(sizes>0);missed[0]=False
            recovered=missed&(after>=.9*sizes)
            semantic[origin]=dict(delete_other_class=int((deletion&other).sum()),
                delete_background_near_target=int((deletion&(labels==0)&(distance<=16)).sum()),
                delete_background_far=int((deletion&(labels==0)&(distance>16)).sum()),
                delete_unknown=int((deletion&unknown).sum()),erroneous_true_deletion=int((deletion&truth).sum()),
                add_true_previously_missed_region=int((addition&truth&missed[regions]).sum()),
                add_true_existing_region=int((addition&truth&~missed[regions]).sum()),
                missed_regions=int(missed.sum()),recovered_regions_at90percent=int(recovered.sum()),
                new_false_pixels=int((addition&~truth).sum()))
        source=provider[identity(row)];oldkey=source['key'];run=Path(source['recheck_run'])
        with np.load(run/'fields'/(oldkey+'.npz'),allow_pickle=False) as z:zscore=z['rcg'].reshape(-1).copy()
        pp=Path(source['packet_export'])
        with np.load(pp,allow_pickle=False) as z:fg=z['fg_max'].reshape(-1).copy();bg=z['bg_max'].reshape(-1).copy()
        outside=covers(rcg)==0;k=int((covers(m['conservative.control'])>.5).sum())
        ids=np.flatnonzero(outside);ids=ids[np.argsort(-zscore[ids],kind='stable')];bank=ids[:k]
        eligible=ids[bg[ids]>fg[ids]];filtered=eligible[:k];same_size=bank[:len(filtered)]
        coverage=covers(truth)
        bank_values={name:dict(n=len(ids),target_mass=float(coverage[ids].sum()),
            target_dominated=int((coverage[ids]>.5).sum()),reference_NN_FG=int((fg[ids]>=bg[ids]).sum()))
            for name,ids in [('original',bank),('ref_BG_filtered',filtered),('unfiltered_same_size',same_size)]}
        details.append(dict(row,key=key,semantic=semantic,components=row_counts,banks=bank_values))
    class_values={}
    for c,v in class_counts.items():
        j=v['B_i']/max(v['B_u'],1)
        def gain(tp,fp):return 100*(tp-j*fp)/max(v['B_u']+fp,1)
        restored=gain(v['restore_TP'],v['restore_FP']);extent=gain(v['extent_TP'],v['extent_FP'])
        combined=gain(v['restore_TP']+v['extent_TP'],v['restore_FP']+v['extent_FP'])
        class_values[c]=dict(v,B_iou=j,restore_gain=restored,extent_gain=extent,
            combined_gain=combined,interaction=combined-restored-extent)
    aggregate={origin:{k:sum(r['semantic'][origin][k] for r in details) for k in details[0]['semantic'][origin]} for origin in details[0]['semantic']}
    banks={name:{k:sum(r['banks'][name][k] for r in details) for k in details[0]['banks'][name]} for name in details[0]['banks']}
    report=dict(n=len(rows),role='post-seal GT diagnostics; no method selection or inference inputs',
        region_definition='8-connected semantic GT regions; touching instances may merge',
        recovery_definition='zero prior overlap andatleast90percent region covered after composition',
        boundary_definition='GT label0 within16pixels oftarget GT',
        bank_NN_space='debiased cached q/r fg_max/bg_max; distinct from raw l24 accounting origin',
        semantic=aggregate,banks=banks,per_class=class_values,
        mean_component_gains={k:float(np.mean([v[k] for v in class_values.values()])) for k in ['restore_gain','extent_gain','combined_gain','interaction']},
        source_seal_sha256=sha(a.run/'sealed.json'),source_code_sha256=sha(__file__))
    a.out.mkdir();(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
    print(json.dumps({k:report[k] for k in ['n','semantic','banks','mean_component_gains']}),flush=True)


if __name__=='__main__':main()
