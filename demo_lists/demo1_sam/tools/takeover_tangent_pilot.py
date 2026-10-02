"""Development pilot: reconstruct four prompt views from two coordinate tangents.

No training or GT inputs. Math-SDPA is required for forward AD; quality and cost
must later face the native optimized decoder. Concurrent elapsed time is diagnostic.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'assets/source/sam2'))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from takeover_gpu_consistency import gpu_consistency
from takeover_rank_features import quality,candidate_features
from takeover_flip_score import bootstrap_image_groups


def main(args):
    out=args.output
    if out.exists():raise FileExistsError(out)
    report={'status':'RUNNING','scope':args.scope,'method':'two coordinate JVPs predict four clipped +/-8 longest-edge1024 prompt responses; rank original candidates1..3 by the same matched consistency rule','training':False,'GT_for_decisions':False,'attention':'MATH for both tangent and exact-view control','rows':[],'summary':{}}
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    save();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
    predictor=SAM2ImagePredictor(model);subset=args.subset
    feature_path=getattr(args,'feature_output',None)
    if feature_path is not None:report['native_baseline_protocol']='Default native SDPA decode on same cached encoder/prompt; student candidate and feature decode uses MATH SDPA; compare task quality, no numerical-equivalence claim'
    if feature_path is not None and feature_path.exists():raise FileExistsError(feature_path)
    export={k:[] for k in ('features','quality','boundary_quality','teacher','head','largest','teacher_choice','image_ids','regimes')}
    capture={'enabled':False}
    def hook(_m,_i,result):
        if capture['enabled']:
            start=1 if model.sam_mask_decoder.pred_obj_scores else 0
            capture['tokens']=result[0][:,start:start+5].detach()
    handle=model.sam_mask_decoder.transformer.register_forward_hook(hook) if feature_path is not None else None
    manifest=json.loads((subset/'manifest.json').read_text())
    try:
        for info in manifest['images']:
            rgb=np.asarray(Image.open(subset/info['image_file']).convert('RGB'));h,w=rgb.shape[:2]
            # Ordinary no-grad tensors can participate as constants in forward AD.
            with torch.no_grad():predictor.set_image(rgb)
            reference=(round(h*1024/max(h,w)),round(w*1024/max(h,w)))
            dx=8*w/reference[1];dy=8*h/reference[0]
            for regime in ('central','near_boundary'):
                key='central_xy' if regime=='central' else 'near_boundary_xy'
                coords=torch.tensor([x[key] for x in info['objects']],device='cuda:0',dtype=torch.float32)[:,None]
                labels=torch.ones(coords.shape[:2],device='cuda:0',dtype=torch.int)
                def function(xy):
                    pos=predictor._transforms.transform_coords(xy,normalize=True,orig_hw=(h,w))
                    sparse,dense=model.sam_prompt_encoder(points=(pos,labels),boxes=None,masks=None)
                    low,scores,_,_=model.sam_mask_decoder.predict_masks(image_embeddings=predictor._features['image_embed'],image_pe=model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,repeat_image=len(xy)>1,high_res_features=predictor._features['high_res_feats'])
                    return low,scores
                xdir=torch.zeros_like(coords);xdir[...,0]=1;ydir=torch.zeros_like(coords);ydir[...,1]=1
                started=time.monotonic()
                if feature_path is not None:
                    with torch.no_grad():
                        native_low,native_scores=function(coords)
                        native_masks=predictor._transforms.postprocess_masks(native_low,(h,w))>0
                        native_head=native_scores[:,1:].argmax(-1)+1
                        stable=model.sam_mask_decoder._get_stability_scores(native_low[:,0:1])[:,0]>=.98
                        native_dynamic=torch.where(stable,torch.zeros_like(native_head),native_head)
                with sdpa_kernel(SDPBackend.MATH):
                    if feature_path is not None:
                        capture['enabled']=True
                        with torch.no_grad():
                            feature_low,feature_scores=function(coords)
                            feature_masks=predictor._transforms.postprocess_masks(feature_low,(h,w))>0
                            pos=predictor._transforms.transform_coords(coords,normalize=True,orig_hw=(h,w))
                            feature_sparse,_=model.sam_prompt_encoder(points=(pos,labels),boxes=None,masks=None)
                            features,geometry_dim=candidate_features(feature_low,feature_scores,capture['tokens'],feature_sparse,feature_masks,regime)
                        capture['enabled']=False
                    (base,scores),(jx,_)=torch.func.jvp(function,(coords,),(xdir,))
                    _,(jy,_)=torch.func.jvp(function,(coords,),(ydir,))
                    if feature_path is not None:
                        primal_difference=float((base-feature_low).abs().max())
                        # The deployment candidate set comes from the ordinary
                        # original forward. Anchor the derivative views there,
                        # recording AD-primal drift rather than asserting a
                        # numerical-equivalence claim for forward AD dispatch.
                        base,scores=feature_low,feature_scores
                    approximate=[];exact=[];errors=[]
                    for axis,delta in ((0,-dx),(0,dx),(1,-dy),(1,dy)):
                        moved=coords.clone();limit=w-1 if axis==0 else h-1;moved[...,axis]=(moved[...,axis]+delta).clamp(0,limit)
                        step=(moved[...,axis]-coords[...,axis]).reshape(len(coords),1,1,1)
                        pred=base+step*(jx if axis==0 else jy)
                        with torch.no_grad():truth,_=function(moved)
                        approximate.append(predictor._transforms.postprocess_masks(pred,(h,w))>0)
                        exact.append(predictor._transforms.postprocess_masks(truth,(h,w))>0)
                        errors.append(float((pred-truth).abs().max()))
                with torch.no_grad():
                    masks=predictor._transforms.postprocess_masks(base,(h,w))>0
                    tangent_choice,tangent_consistency=gpu_consistency(masks,scores,torch.stack(approximate));exact_choice,_=gpu_consistency(masks,scores,torch.stack(exact));head=scores[:,1:].argmax(-1)+1
                    with np.load(subset/info['gt_file'],allow_pickle=False) as gt:
                        valid=torch.from_numpy(gt['valid']).cuda() if 'valid' in gt else None
                        truth=torch.from_numpy(gt['gt']).cuda()
                        values,boundaries=quality(masks,truth,valid);ids=gt['annotation_ids'].copy()
                        if feature_path is not None:
                            native_values,_=quality(torch.cat([native_masks[:,:1],native_masks],1),truth,valid)
                            all_original_values,_=quality(torch.cat([masks[:,:1],masks],1),truth,valid)
                    if feature_path is not None:
                        for key,val in (('features',features),('quality',values),('boundary_quality',boundaries),('teacher',tangent_consistency),('head',head-1),('largest',masks[:,1:].sum((-2,-1)).argmax(-1)),('teacher_choice',tangent_choice-1)):
                            export[key].extend(val.cpu().numpy())
                        export['image_ids'].extend([info['image_id']]*len(ids));export['regimes'].extend([('central','near_boundary').index(regime)]*len(ids))
                    values=values.cpu().numpy();choices={k:v.cpu().numpy()-1 for k,v in (('head',head),('tangent',tangent_choice),('exact_views',exact_choice))}
                    for j,aid in enumerate(ids):
                        row={'image_id':info['image_id'],'annotation_id':int(aid),'regime':regime,'choices':{k:int(v[j]) for k,v in choices.items()},'tangent_choice_matches_exact':bool(choices['tangent'][j]==choices['exact_views'][j]),'max_view_logit_error':max(errors)}
                        if feature_path is not None:row['JVP_primal_vs_original_max_logit_error']=primal_difference
                        if feature_path is not None:
                            row['quality_all4']=all_original_values[j].cpu().tolist()
                            row['native_dynamic_iou']=float(native_values[j,native_dynamic[j]])
                            row['native_multimask_head_iou']=float(native_values[j,native_head[j]])
                        for name,c in choices.items():row[name+'_iou']=float(values[j,c[j]]);row[name+'_delta']=float(values[j,c[j]]-values[j,choices['head'][j]])
                        report['rows'].append(row)
                print(json.dumps({'image_id':info['image_id'],'regime':regime,'diagnostic_wall_seconds':time.monotonic()-started,'max_view_logit_error':max(errors)}),flush=True)
            save()
        for regime in ('central','near_boundary'):
            rows=[r for r in report['rows'] if r['regime']==regime];report['summary'][regime]={}
            for name in ('head','tangent','exact_views'):report['summary'][regime][name]={'mean_iou':float(np.mean([r[name+'_iou'] for r in rows])),'mean_delta':float(np.mean([r[name+'_delta'] for r in rows])),'image_cluster_delta95':bootstrap_image_groups(rows,name+'_delta')}
            report['summary'][regime]['choice_agreement']=float(np.mean([r['tangent_choice_matches_exact'] for r in rows]))
        report['status']='COMPLETED_DEVELOPMENT_PILOT'
        if feature_path is not None:
            np.savez_compressed(feature_path,**{k:np.array(v) for k,v in export.items()})
            feature_path.with_suffix('.json').write_text(json.dumps({'geometry_dim':geometry_dim,'feature_dim':features.shape[-1],'teacher':'Two coordinate tangents with fixed radius8 reconstruction anchored to conventional original forward; point regimes only','attention':'MATH; AD-primal versus original drift recorded; no numerical-equivalence claim','GT_in_features_or_teacher':False,'samples':len(report['rows'])},indent=2)+'\n')
            report['status']='COMPLETED_RESPONSE_FEATURE_EXPORT'
    except Exception as e:report.update(status='ERROR',error=repr(e));raise
    finally:
        if handle is not None:handle.remove()
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset',type=Path,default=ROOT/'assets/coco_quality_seed2027_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'results/takeover_20261001_v1/distill_v1/tangent_pilot24.json')
    parser.add_argument('--scope',default='24 development images; two point regimes; no heldout or optimized latency claim')
    parser.add_argument('--feature-output',type=Path)
    main(parser.parse_args())
