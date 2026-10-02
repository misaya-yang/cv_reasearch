"""Actual pretrained SAM2.1 quality controls, with native image preprocessing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'assets/source/sam2'))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from takeover_prompt_stability import perturb_objects,OFFSETS
from takeover_gpu_consistency import gpu_consistency
from takeover_rank_features import quality,candidate_features
from takeover_flip_score import bootstrap_image_groups


def encode(predictor,objects,regime,size):
    if regime=='box':
        boxes=torch.tensor([x['tight_box_xyxy'] for x in objects],device='cuda:0',dtype=torch.float32)
        boxes=predictor._transforms.transform_boxes(boxes,normalize=True,orig_hw=size)
        labels=torch.tensor([[2,3]],device='cuda:0',dtype=torch.int).expand(len(objects),-1)
        points=(boxes,labels)
    else:
        key='central_xy' if regime=='central' else 'near_boundary_xy'
        coords=torch.tensor([x[key] for x in objects],device='cuda:0',dtype=torch.float32)[:,None]
        coords=predictor._transforms.transform_coords(coords,normalize=True,orig_hw=size)
        points=(coords,torch.ones(coords.shape[:2],device='cuda:0',dtype=torch.int))
    return predictor.model.sam_prompt_encoder(points=points,boxes=None,masks=None)


def decode(predictor,sparse,dense):
    model=predictor.model
    low,scores,_,_=model.sam_mask_decoder.predict_masks(image_embeddings=predictor._features['image_embed'],image_pe=model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,repeat_image=len(sparse)>1,high_res_features=predictor._features['high_res_feats'])
    full=predictor._transforms.postprocess_masks(low,predictor._orig_hw[0])>0
    return low,scores,full


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=json.loads((a.subset/'manifest.json').read_text())
    model=build_sam2(a.config,str(a.checkpoint),device='cuda:0');predictor=SAM2ImagePredictor(model)
    capture={};export={k:[] for k in ('features','quality','boundary_quality','teacher','head','largest','teacher_choice','image_ids','regimes')};metadata=[]
    def hook(_m,_i,output):
        start=1 if model.sam_mask_decoder.pred_obj_scores else 0
        capture['tokens']=output[0][:,start:start+5].detach()
    handle=model.sam_mask_decoder.transformer.register_forward_hook(hook) if a.feature_output is not None else None
    report={'status':'RUNNING','model':a.config,'checkpoint':str(a.checkpoint),'subset':str(a.subset),'precision':'FP32 TF32 off','protocol':'native official SAM2.1 encoder/preprocess/prompt encoder/four raw mask outputs; same original clicks and GT boxes; tokens1..3 selectable; teacher offsets measured in longest-edge1024 reference coordinates; no GT selection','concurrent_run_not_latency_evidence':True,'rows':[],'summary':{}}
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    with torch.inference_mode():
        for n,info in enumerate(source['images']):
            rgb=np.asarray(Image.open(a.subset/info['image_file']).convert('RGB'));size=rgb.shape[:2]
            predictor.set_image(rgb)
            h,w=size;reference=(round(h*1024/max(size)),round(w*1024/max(size)))
            for regime in ('central','near_boundary','box'):
                sparse,dense=encode(predictor,info['objects'],regime,size);original_low,scores,full=decode(predictor,sparse,dense)
                if a.feature_output is not None:
                    features,geometry_dim=candidate_features(original_low,scores,capture['tokens'],sparse,full,regime)
                parts=[encode(predictor,perturb_objects(info['objects'],regime,size,reference,o),regime,size)[0] for o in OFFSETS]
                ps=torch.cat(parts);_,_,views=decode(predictor,ps,dense[:1].expand(len(ps),-1,-1,-1))
                choice,consistency=gpu_consistency(full,scores,views.reshape(4,len(sparse),4,*size))
                selected={'original_iou_head':scores[:,1:].argmax(-1),'largest_original_mask':full[:,1:].sum((-2,-1)).argmax(-1),'perturb_consistency':choice-1}
                with np.load(a.subset/info['gt_file'],allow_pickle=False) as gt:
                    truth=torch.from_numpy(gt['gt']).cuda();ids=gt['annotation_ids'].copy();valid=torch.from_numpy(gt['valid']).cuda() if 'valid' in gt else None
                values,boundaries=quality(full,truth,valid);values=values.cpu().numpy();boundaries=boundaries.cpu().numpy();selected={k:v.cpu().numpy() for k,v in selected.items()}
                if a.feature_output is not None:
                    for key,value in (('features',features.cpu().numpy()),('quality',values),('boundary_quality',boundaries),('teacher',consistency.cpu().numpy()),('head',selected['original_iou_head']),('largest',selected['largest_original_mask']),('teacher_choice',selected['perturb_consistency'])):export[key].extend(value)
                    for aid in ids:
                        export['image_ids'].append(info['image_id']);export['regimes'].append(('central','near_boundary','box').index(regime));metadata.append({'image_id':info['image_id'],'annotation_id':int(aid),'regime':regime})
                for j,aid in enumerate(ids):
                    row={'image_id':info['image_id'],'annotation_id':int(aid),'regime':regime,'iou_per_mask':values[j].tolist(),'boundary_iou_per_mask':boundaries[j].tolist(),'selections':{k:int(v[j]) for k,v in selected.items()}}
                    for name,indices in selected.items():
                        row[name+'_iou']=float(values[j,indices[j]]);row[name+'_boundary_iou']=float(boundaries[j,indices[j]])
                        row[name+'_delta_iou']=float(values[j,indices[j]]-values[j,selected['original_iou_head'][j]])
                    report['rows'].append(row)
            if (n+1)%16==0:save();print(json.dumps({'images':n+1,'rows':len(report['rows'])}),flush=True)
    for regime in ('central','near_boundary','box'):
        rows=[r for r in report['rows'] if r['regime']==regime];report['summary'][regime]={}
        for name in ('original_iou_head','largest_original_mask','perturb_consistency'):
            report['summary'][regime][name]={'mean_iou':float(np.mean([r[name+'_iou'] for r in rows])),'mean_boundary_iou':float(np.mean([r[name+'_boundary_iou'] for r in rows])),'mean_delta_iou':float(np.mean([r[name+'_delta_iou'] for r in rows])),'image_cluster_delta95':bootstrap_image_groups(rows,name+'_delta_iou')}
    report['status']='COMPLETED_ACTUAL_PRETRAINED_MODEL';save()
    if a.feature_output is not None:
        if a.feature_output.exists():raise FileExistsError(a.feature_output)
        np.savez_compressed(a.feature_output,**{k:np.array(v) for k,v in export.items()});a.feature_output.with_suffix('.json').write_text(json.dumps({'status':'COMPLETED','geometry_dim':geometry_dim,'feature_dim':features.shape[-1],'samples':len(metadata),'rows':metadata,'GT_in_features_or_teacher':False,'model':a.config},indent=2)+'\n');handle.remove()
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('subset','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--config',required=True);p.add_argument('--feature-output',type=Path);main(p.parse_args())
