"""Compact frozen-decoder features from existing train/development cohorts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from takeover_virtual_flip import load_models
from takeover_rank_features import candidate_features


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=json.loads((a.source_run/'report.json').read_text());teacher=json.loads(a.teacher.read_text())
    decoder,_,_=load_models(a.source_run/'mask_decoder_state.pt',a.checkpoint,'cuda:0')
    capture={}
    def hook(_module,_inputs,outputs):capture['tokens']=outputs[0][:,:5].detach()
    handle=decoder.transformer.register_forward_hook(hook)
    x=[];q=[];b=[];t=[];head=[];largest=[];metadata=[];numeric=[]
    with torch.inference_mode():
        for n,info in enumerate(source['images']):
            with np.load(a.source_run/info['encoded_inputs']['npz'],allow_pickle=False) as e:
                image,pe,dense=[torch.from_numpy(e[k]).cuda() for k in ('image_embeddings','image_pe','dense_nomask')]
                for regime in ('central','near_boundary','box'):
                    rec=next(r for r in source['records'] if r['image_id']==info['image_id'] and r['method']=='official' and r['regime']==regime)
                    sparse=torch.from_numpy(e['sparse_'+regime]).cuda()
                    low,scores=decoder.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense.expand(len(sparse),-1,-1,-1))
                    with np.load(a.source_run/rec['npz'],allow_pickle=False) as old:
                        # Saved masks/scores are the established official outputs.
                        error=float(np.max(np.abs(low.cpu().numpy()-old['low_resolution_logits'])))
                        numeric.append({'image_id':info['image_id'],'regime':regime,'max_logit_error':error,'fixed_5e5_status':'PASS' if error<=5e-5 else 'FAIL'})
                        # Preserve saved original-mask geometry and its original
                        # quality labels. Replayed latent tokens are feature probes;
                        # this export is not an exact-output equivalence claim.
                        features,geometry_dim=candidate_features(torch.from_numpy(old['low_resolution_logits']).cuda(),torch.from_numpy(old['iou_prediction']).cuda(),capture['tokens'],sparse,torch.from_numpy(old['masks']).cuda(),regime)
                        for j,aid in enumerate(old['annotation_ids']):
                            tr=next(r for r in teacher['rows'] if r['image_id']==info['image_id'] and r['regime']==regime and r['annotation_id']==int(aid))
                            x.append(features[j].cpu().numpy());q.append(tr['iou_per_mask'][1:])
                            b.append([tr[k+'_boundary_iou'] for k in ('original_iou_head',)])
                            t.append(tr['consistency']);head.append(tr['selections']['original_iou_head']-1)
                            largest.append(tr['selections']['largest_original_mask']-1)
                            metadata.append({'image_id':info['image_id'],'annotation_id':int(aid),'regime':regime,'teacher_choice':tr['selections']['perturb_consistency']-1})
            if (n+1)%16==0:print(json.dumps({'completed_images':n+1,'samples':len(x)}),flush=True)
    handle.remove()
    np.savez_compressed(a.output,features=np.stack(x),quality=np.array(q),teacher=np.array(t),head=np.array(head),largest=np.array(largest),teacher_choice=np.array([r['teacher_choice'] for r in metadata]),image_ids=np.array([r['image_id'] for r in metadata]),regimes=np.array([('central','near_boundary','box').index(r['regime']) for r in metadata]))
    a.output.with_suffix('.json').write_text(json.dumps({'status':'COMPLETED','source_run':str(a.source_run),'teacher':str(a.teacher),'geometry_dim':geometry_dim,'feature_dim':x[0].shape[-1],'samples':len(x),'rows':metadata,'GT_used_for_features':False,'GT_quality_labels_for_supervised_control':True,'official_replay_numeric_checks':numeric,'fixed_5e5_failures':sum(r['fixed_5e5_status']=='FAIL' for r in numeric),'scope':'saved original masks/logits/scores with replayed latent probes; no FP32 equivalence pass inferred'},indent=2)+'\n')
    print(json.dumps({'status':'COMPLETED','samples':len(x),'features':x[0].shape[-1]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-run','teacher','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
