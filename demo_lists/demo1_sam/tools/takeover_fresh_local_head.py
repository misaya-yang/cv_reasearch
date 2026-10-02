"""Frozen compact head cross-domain check, native encoder once per image."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import numpy as np
from PIL import Image
import torch
from takeover_extract_fresh_rank import ROOT,sam_model_registry,ResizeLongestSide,Sam,encode_prompt_rows
from takeover_compact_head_infer import load_student,decode
from takeover_rank_features import quality
from takeover_flip_score import bootstrap_image_groups


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    source=json.loads((a.subset/'manifest.json').read_text())
    report={'status':'WAITING_VRAM','protocol':'Frozen SAM1 real rank64 head trained on previous12 COCO images, epoch teacher-MSE-selected on12 other development images. DAVIS30 val first annotated frames, valid excludes void255, no temporal memory or video claim. Native SAM ViT-B encoder once per image, same clicks/boxes; budgets0/5/10% fixed, originalhead choice held fixed. Cross-domain quality, no latency evidence under concurrency. No full masks or encoded state persisted.','image_encoder_executions':0,'rows':[],'summary':{}}
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
    save()
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<4000:time.sleep(10)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=sam_model_registry['vit_b'](checkpoint=str(ROOT/'assets/checkpoints/sam_vit_b_01ec64.pth')).cuda().eval().requires_grad_(False)
    student,center,basis=load_student(ROOT/'results/takeover_20261001_v1/local_student_head_v1/deployment.pt','cuda:0')
    transform=ResizeLongestSide(1024)
    try:
        report['status']='RUNNING_FROZEN_CROSS_DOMAIN';save()
        with torch.inference_mode():
            for n,info in enumerate(source['images']):
                rgb=np.asarray(Image.open(a.subset/info['image_file']).convert('RGB'));size=rgb.shape[:2]
                resized=transform.apply_image(rgb);input_size=resized.shape[:2]
                tensor=torch.as_tensor(resized,device='cuda:0').permute(2,0,1).contiguous()[None]
                image=model.image_encoder(model.preprocess(tensor));pe=model.prompt_encoder.get_dense_pe()
                report['image_encoder_executions']+=1
                with np.load(a.subset/info['gt_file'],allow_pickle=False) as gt:
                    truth=torch.from_numpy(gt['gt']).cuda();ids=gt['annotation_ids'].tolist();valid=torch.from_numpy(gt['valid']).cuda() if 'valid' in gt else None
                for regime in ('central','near_boundary','box'):
                    sparse,dense=encode_prompt_rows(model.prompt_encoder,info['objects'],regime,size,1024,'cuda:0')
                    low,iq=model.mask_decoder.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense)
                    original=Sam.postprocess_masks(model,low,input_size,size)>0
                    q,bq=quality(original,truth,valid);selected=iq[:,1:].argmax(-1)
                    for budget in (0,.05,.1):
                        candidate,scores=decode(model.mask_decoder,student,image,pe,dense,sparse,center,basis,budget)
                        assert candidate.shape==low.shape and scores.shape==iq.shape
                        masks=Sam.postprocess_masks(model,candidate,input_size,size)>0
                        pq,pb=quality(masks,truth,valid)
                        differing=masks!=original
                        if valid is not None:differing=differing&valid[None,None]
                        flips=differing.sum((-2,-1));denom=int(valid.sum()) if valid is not None else masks.shape[-1]*masks.shape[-2]
                        for j,aid in enumerate(ids):
                            c=int(selected[j]);report['rows'].append({'image_id':info['image_id'],'sequence':info.get('sequence'),'annotation_id':aid,'regime':regime,'budget':budget,'original_iou':float(q[j,c]),'predicted_iou':float(pq[j,c]),'delta_iou':float(pq[j,c]-q[j,c]),'delta_boundary_iou':float(pb[j,c]-bq[j,c]),'valid_pixel_flip_fraction':float(flips[j,c+1]/max(1,denom)),'max_original_IQ_drift':float((scores-iq).abs().max())})
                save();print(json.dumps({'images':n+1,'rows':len(report['rows'])}),flush=True)
        for regime in ('central','near_boundary','box'):
            report['summary'][regime]={}
            for budget in (0,.05,.1):
                rows=[r for r in report['rows'] if r['regime']==regime and r['budget']==budget]
                report['summary'][regime][str(budget)]={'mean_delta_iou':float(np.mean([r['delta_iou'] for r in rows])),'image_sequence_cluster95':bootstrap_image_groups(rows,'delta_iou'),'mean_valid_flip_fraction':float(np.mean([r['valid_pixel_flip_fraction'] for r in rows]))}
        report['status']='COMPLETED_FROZEN_CROSS_DOMAIN';print(json.dumps(report['summary']),flush=True)
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('subset','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
