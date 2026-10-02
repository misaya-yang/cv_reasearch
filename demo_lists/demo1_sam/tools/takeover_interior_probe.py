"""Frozen mask-interior prompt probes, fixed radius controls, and native SAM2 baseline.

All candidates face the same views, constructed from the original head mask.
GT is loaded only after the probe and original-view features are constructed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from takeover_sam2_quality import ROOT, build_sam2, SAM2ImagePredictor, encode, decode
from takeover_gpu_consistency import gpu_consistency
from takeover_rank_features import candidate_features, quality
from takeover_flip_score import bootstrap_image_groups


def probe_points(coords, mask, size, radius, interior=False):
    """Four axis probes in longest-edge1024 units; mask ray uses 8 fixed samples."""
    h,w=size
    ref=(round(h*1024/max(size)),round(w*1024/max(size)))
    result=[]
    for axis,sign in ((0,-1),(0,1),(1,-1),(1,1)):
        move=coords.clone()
        delta=sign*radius*(w/ref[1] if axis==0 else h/ref[0])
        move[:,axis]=(move[:,axis]+delta).clamp(0,(w-1 if axis==0 else h-1))
        if interior:
            # First failed foreground sample stops the ray. Starting outside the
            # reference mask produces a zero move; consistency ties retain head.
            ts=torch.linspace(0,1,9,device=coords.device)
            ray=coords[:,None]+(move-coords)[:,None]*ts[None,:,None]
            xy=ray.round().long();xy[...,0].clamp_(0,w-1);xy[...,1].clamp_(0,h-1)
            inside=mask[torch.arange(len(coords),device=coords.device)[:,None],xy[...,1],xy[...,0]]
            prefix=inside.long().cumprod(1).sum(1)
            fraction=(prefix-1).clamp_min(0).float()/8
            move=coords+(move-coords)*fraction[:,None]
        result.append(move)
    return result


def main(a):
    if a.output.exists() or a.feature_output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=json.loads((a.subset/'manifest.json').read_text())
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0')
    predictor=SAM2ImagePredictor(model);capture={}
    def hook(_m,_i,out):
        start=1 if model.sam_mask_decoder.pred_obj_scores else 0
        capture['tokens']=out[0][:,start:start+5].detach()
    handle=model.sam_mask_decoder.transformer.register_forward_hook(hook)
    export={k:[] for k in ('features','quality','boundary_quality','teacher','head','largest','teacher_choice','image_ids','regimes')}
    report={'status':'RUNNING','scope':a.scope,'protocol':'Original candidates1..3; fixed radius1/4/8 controls; interior radius8 rays of 8 segments constrained to original head mask. Boxes retain original head for interior policy. Native dynamic singlemask includes mask0, stability delta .05 threshold .98. No GT in decisions.','rows':[],'summary':{}}
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
    save()
    try:
        with torch.inference_mode():
            for n,info in enumerate(source['images']):
                rgb=np.asarray(Image.open(a.subset/info['image_file']).convert('RGB'));size=rgb.shape[:2];predictor.set_image(rgb)
                for regime in ('central','near_boundary','box'):
                    sparse,dense=encode(predictor,info['objects'],regime,size);low,scores,full=decode(predictor,sparse,dense)
                    features,geomdim=candidate_features(low,scores,capture['tokens'],sparse,full,regime)
                    head=scores[:,1:].argmax(-1)+1
                    decoder=model.sam_mask_decoder
                    assert decoder.dynamic_multimask_stability_delta==.05 and decoder.dynamic_multimask_stability_thresh==.98
                    stable=decoder._get_stability_scores(low[:,0:1])[:,0]>=.98
                    native=torch.where(stable,torch.zeros_like(head),head)
                    # Check the independently formed indices against the actual
                    # official dynamic API on every object before recording it.
                    native_low,_=decoder._dynamic_multimask_via_stability(low,scores)
                    assert torch.equal(native_low[:,0],low[torch.arange(len(head),device='cuda:0'),native])
                    choices={'head':head,'largest':full[:,1:].sum((-2,-1)).argmax(-1)+1,'native_dynamic':native,'interior8':head}
                    teacher=torch.zeros_like(scores[:,1:],dtype=torch.float64)
                    motion={}
                    if regime!='box':
                        key='central_xy' if regime=='central' else 'near_boundary_xy'
                        coords=torch.tensor([o[key] for o in info['objects']],device='cuda:0',dtype=torch.float32)
                        refmask=full[torch.arange(len(head),device='cuda:0'),head]
                        for name,radius,interior in (('radius1',1,False),('radius4',4,False),('radius8',8,False),('interior8',8,True)):
                            moves=probe_points(coords,refmask,size,radius,interior)
                            transformed=[predictor._transforms.transform_coords(x[:,None],normalize=True,orig_hw=size) for x in moves]
                            pp=torch.cat(transformed);labels=torch.ones(pp.shape[:2],device='cuda:0',dtype=torch.int)
                            ss,dd=model.sam_prompt_encoder(points=(pp,labels),boxes=None,masks=None)
                            _,_,views=decode(predictor,ss,dd)
                            selected,consistency=gpu_consistency(full,scores,views.reshape(4,len(coords),4,*size))
                            choices[name]=selected
                            motion[name]=torch.stack([(x-coords).norm(dim=-1) for x in moves]).mean(0).cpu().numpy()
                            if interior:teacher=consistency
                    # Only now load evaluation labels, including DAVIS void mask.
                    with np.load(a.subset/info['gt_file'],allow_pickle=False) as gt:
                        truth=torch.from_numpy(gt['gt']).cuda();ids=gt['annotation_ids'].copy();valid=torch.from_numpy(gt['valid']).cuda() if 'valid' in gt else None
                    allq,allbq=quality(torch.cat([full[:,:1],full],1),truth,valid)
                    q=allq[:,1:];bq=allbq[:,1:]
                    for key,val in (('features',features),('quality',q),('boundary_quality',bq),('teacher',teacher),('head',head-1),('largest',choices['largest']-1),('teacher_choice',choices['interior8']-1)):
                        export[key].extend(val.cpu().numpy())
                    c={k:v.cpu().numpy() for k,v in choices.items()};aq=allq.cpu().numpy();ab=allbq.cpu().numpy()
                    for j,aid in enumerate(ids):
                        export['image_ids'].append(info['image_id']);export['regimes'].append(('central','near_boundary','box').index(regime))
                        row={'image_id':info['image_id'],'annotation_id':int(aid),'regime':regime,'quality_all4':aq[j].tolist(),'selections_raw4':{k:int(v[j]) for k,v in c.items()},'mean_original_pixel_move':{k:float(v[j]) for k,v in motion.items()}}
                        for name,idx in c.items():
                            row[name+'_iou']=float(aq[j,idx[j]]);row[name+'_boundary_iou']=float(ab[j,idx[j]])
                            row[name+'_delta_head']=float(aq[j,idx[j]]-aq[j,c['head'][j]])
                            row[name+'_delta_native']=float(aq[j,idx[j]]-aq[j,c['native_dynamic'][j]])
                        report['rows'].append(row)
                if (n+1)%8==0:save();print(json.dumps({'images':n+1,'rows':len(report['rows'])}),flush=True)
        for regime in ('central','near_boundary','box'):
            rows=[r for r in report['rows'] if r['regime']==regime];report['summary'][regime]={}
            names=('head','largest','native_dynamic','interior8') if regime=='box' else ('head','largest','native_dynamic','radius1','radius4','radius8','interior8')
            for name in names:
                report['summary'][regime][name]={'mean_iou':float(np.mean([r[name+'_iou'] for r in rows])),'mean_boundary_iou':float(np.mean([r[name+'_boundary_iou'] for r in rows])),'delta_head95':bootstrap_image_groups(rows,name+'_delta_head'),'delta_native95':bootstrap_image_groups(rows,name+'_delta_native')}
        np.savez_compressed(a.feature_output,**{k:np.array(v) for k,v in export.items()})
        a.feature_output.with_suffix('.json').write_text(json.dumps({'geometry_dim':geomdim,'feature_dim':features.shape[-1],'teacher':'interior8 point probes; original head on boxes','GT_in_features_or_teacher':False,'samples':len(report['rows'])},indent=2)+'\n')
        report['status']='COMPLETED';print(json.dumps(report['summary']),flush=True)
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:handle.remove();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('subset','output','feature-output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--scope',required=True);main(p.parse_args())
