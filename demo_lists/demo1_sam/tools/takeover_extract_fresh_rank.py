"""Compact original features, GT-free teacher, and evaluation labels; no large outputs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'assets/source/segment-anything'))
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from takeover_virtual_flip import encode_prompt_rows,Sam
from takeover_prompt_stability import perturb_objects,OFFSETS
from takeover_gpu_consistency import gpu_consistency
from takeover_rank_features import candidate_features,quality


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=sam_model_registry['vit_b'](checkpoint=str(a.checkpoint)).cuda().eval().requires_grad_(False)
    source=json.loads((a.subset/'manifest.json').read_text());transform=ResizeLongestSide(1024)
    capture={}
    def hook(_m,_i,outputs):capture['tokens']=outputs[0][:,:5].detach()
    handle=model.mask_decoder.transformer.register_forward_hook(hook)
    arrays={k:[] for k in ('features','quality','boundary_quality','teacher','head','largest','teacher_choice','image_ids','regimes')};metadata=[]
    status={'status':'RUNNING','subset':str(a.subset),'image_encoder_executions':0,'GT_used_for_features_or_teacher':False,'images_completed':0,'rows':[]}
    def save_status():a.output.with_suffix('.json').write_text(json.dumps(status,indent=2)+'\n')
    save_status()
    with torch.inference_mode():
        for info in source['images']:
            rgb=np.asarray(Image.open(a.subset/info['image_file']).convert('RGB'));size=rgb.shape[:2]
            resized=transform.apply_image(rgb);isize=resized.shape[:2]
            tensor=torch.as_tensor(resized,device='cuda:0').permute(2,0,1).contiguous()[None]
            image=model.image_encoder(model.preprocess(tensor));pe=model.prompt_encoder.get_dense_pe()
            status['image_encoder_executions']+=1
            for regime in ('central','near_boundary','box'):
                sparse,dense=encode_prompt_rows(model.prompt_encoder,info['objects'],regime,size,1024,'cuda:0')
                low,scores=model.mask_decoder.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense)
                full=Sam.postprocess_masks(model,low,isize,size)>0
                features,geometry_dim=candidate_features(low,scores,capture['tokens'],sparse,full,regime)
                parts=[encode_prompt_rows(model.prompt_encoder,perturb_objects(info['objects'],regime,size,isize,o),regime,size,1024,'cuda:0')[0] for o in OFFSETS]
                ps=torch.cat(parts);vl,_=model.mask_decoder.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=ps,dense_prompt_embeddings=dense[:1].expand(len(ps),-1,-1,-1))
                views=(Sam.postprocess_masks(model,vl,isize,size)>0).reshape(4,len(sparse),4,*size)
                choice,consistency=gpu_consistency(full,scores,views)
                # Evaluation GT is accessed after original features and all teacher decisions.
                with np.load(a.subset/info['gt_file'],allow_pickle=False) as truth:
                    gt=torch.from_numpy(truth['gt']).cuda();ids=truth['annotation_ids'].copy()
                    valid=torch.from_numpy(truth['valid']).cuda() if 'valid' in truth else None
                values,boundaries=quality(full,gt,valid)
                for key,value in (('features',features),('quality',values),('boundary_quality',boundaries),('teacher',consistency),('head',scores[:,1:].argmax(-1)),('largest',full[:,1:].sum((-2,-1)).argmax(-1)),('teacher_choice',choice-1)):
                    arrays[key].extend(value.cpu().numpy())
                for aid in ids:
                    arrays['image_ids'].append(info['image_id']);arrays['regimes'].append(('central','near_boundary','box').index(regime))
                    metadata.append({'image_id':info['image_id'],'annotation_id':int(aid),'regime':regime,'sequence':info.get('sequence')})
            status.update(images_completed=status['images_completed']+1,geometry_dim=geometry_dim,feature_dim=features.shape[-1],samples=len(metadata))
            if status['images_completed']%16==0:save_status();print(json.dumps({k:v for k,v in status.items() if k!='rows'}),flush=True)
    handle.remove()
    np.savez_compressed(a.output,**{k:np.array(v) for k,v in arrays.items()})
    status.update(status='COMPLETED',rows=metadata);save_status();print(json.dumps({'status':'COMPLETED','images':status['images_completed'],'samples':len(metadata)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('subset','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
