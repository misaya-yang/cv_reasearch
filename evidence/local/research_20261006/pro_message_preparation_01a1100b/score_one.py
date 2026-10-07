"""Separate 1CPU scoring; reads query GT only after validating M5 seal/hashes."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,json,hashlib,time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.perf_counter()
    seal=json.loads((args.out/'sealed.json').read_text())
    if seal['state']!='PREDICTIONS_SEALED':raise ValueError('Prediction seal required')
    if sha(args.out/'masks.npz')!=seal['masks_sha256'] or sha(args.out/'fields.npz')!=seal['fields_sha256']:
        raise ValueError('Sealed outputs changed')
    binding=json.loads(Path(seal['binding_path']).read_text());row=binding['row']
    ann=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')/Path(row['query']).with_suffix('.png')
    # The FIRST query-label read occurs here, after validation of prediction seal.
    with Image.open(ann) as im:original_truth=np.asarray(im)==row['c']+1
    work_truth=np.asarray(Image.fromarray(original_truth.astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST))>0
    with np.load(args.out/'masks.npz',allow_pickle=False) as z:masks={k:z[k].copy() for k in z.files}
    if original_truth.shape!=tuple(seal['query_HW']):raise ValueError('Original query GT geometry mismatch')
    records={}
    for name in seal['arms']:
        records[name]={}
        for geometry,truth in (('work',work_truth),('original',original_truth)):
            prediction=masks[name+'.'+geometry];base=masks['zero.'+geometry]
            if prediction.shape!=truth.shape or prediction.dtype!=bool:raise ValueError('Complete bool mask contract failed')
            inter=int((prediction&truth).sum());union=int((prediction|truth).sum())
            added=prediction&~base;deleted=~prediction&base
            records[name][geometry]={'intersection':inter,'union':union,'IoU_percent':100*inter/max(union,1),
                                     'truth_pixels':int(truth.sum()),'prediction_pixels':int(prediction.sum()),
                                     'add_TP':int((added&truth).sum()),'add_FP':int((added&~truth).sum()),
                                     'delete_TP':int((deleted&truth).sum()),'delete_FP':int((deleted&~truth).sum())}
    for name,row_metrics in records.items():
        for geometry,record in row_metrics.items():
            record['delta_vs_same_renderer_MEAN_pp']=record['IoU_percent']-records['zero'][geometry]['IoU_percent']
    report={'state':'ONE_EPISODE_SCORED_AFTER_SEAL','episode':seal['episode'],'episodes':1,
            'data_exposure':'existing public/dev row; not independent confirmation','threads':1,
            'sealed_json_sha256':sha(args.out/'sealed.json'),'query_annotation_sha256':sha(ann),
            'host_variant':seal['host_variant'],'metrics':records,'scoring_seconds':time.perf_counter()-started,
            'native_all_ROI_layer_maxabs':seal['native_all_ROI_layer_maxabs'],
            'inference_seconds':seal['inference_seconds'],'complete_cached_entry_seconds':seal['complete_cached_entry_seconds'],
            'peak_RSS_bytes':seal['peak_RSS_bytes'],'not_claimed':['cohort gain','stable class-summed mIoU gain','novelty','independent confirmation']}
    (args.out/'score_one.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
