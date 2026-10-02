"""Train a real compact local head and execute real selected-parent fallback."""
import json
import argparse
import os
from pathlib import Path
import sys
import subprocess
import time
import math
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/compute_structure'))
from pretrained_decoder_round import load_decoder,load_inputs,full_masks
from takeover_rank_features import quality
from takeover_flip_score import bootstrap_image_groups
from takeover_head_fallback import phases,grid
OUT=ROOT/'results/takeover_20261001_v1/local_student_head_v1'


def main(resume=False):
    OUT.mkdir(exist_ok=resume)
    report={'status':'WAITING_VRAM','protocol':'Real head surrogate: final256-channel parent state -> Linear64/GELU/Linear64 PCA coefficients; fixed rank64, seed2027. Teacher-feature MSE only; first12 old development images train, last12 validate and diagnose. Min100/max300 epochs, patience50, AdamW lr.001. Epoch selection teacher MSE, never GT. Deployment uses coefficients directly contracted with mask hypervectors, avoids reconstructing512 feature channels. Selected parent patches run actual original upscaling on gathered256x1x1 states; no full original head at inference. 0/5/10% margin budgets fixed. Original full head executed only to measure task quality. Development study, no heldout/SOTA/exclusive timing claim.','curves':[],'rows':[],'summary':{}}
    if resume:
        old=json.loads((OUT/'report.json').read_text())
        if old['status']!='ERROR':raise RuntimeError('Verify the previous worker before recovery')
        attempts=len(list(OUT.glob('report.failed_attempt*.json')))+1
        (OUT/f'report.failed_attempt{attempts}.json').write_text(json.dumps(old,indent=2)+'\n')
        report['curves']=old.get('curves',[])
    report['pid']=os.getpid()
    def save():
        tmp=OUT/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'report.json')
    save()
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<2000:time.sleep(10)
    torch.set_num_threads(2);torch.manual_seed(2027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=OUT.parent/'real_original';paths=sorted(source.glob('image_*/encoded_inputs.npz'));assert len(paths)==24
    teacher,_,_=load_decoder(source/'mask_decoder_state.pt',torch.device('cuda:0'))
    basis=torch.load(OUT.parent/'head_compress_v1/basis.pt',weights_only=True,map_location='cuda:0');center=basis['mean'];v=basis['basis'][:,:64]
    capture={};handles=[teacher.transformer.register_forward_hook(lambda m,i,o:capture.update(tokens=o[0],src=o[1])),teacher.output_upscaling.register_forward_hook(lambda m,i,o:capture.update(upscaled=o))]
    generator=torch.Generator(device='cuda:0').manual_seed(2027)
    def forwards(path):
        image,pe,dense,sparse,info=load_inputs(path,torch.device('cuda:0'))
        for regime,sp in sparse.items():
            low,scores=teacher.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sp,dense_prompt_embeddings=dense)
            x=capture['src'];p=len(x)
            phi=capture['upscaled'].reshape(p,32,64,4,64,4).permute(0,2,4,3,5,1).reshape(-1,512)
            hyper=torch.stack([mlp(capture['tokens'][:,i+1]) for i,mlp in enumerate(teacher.output_hypernetworks_mlps)],1)
            yield regime,x,phi,hyper,low,scores,info
    try:
        report['status']='EXTRACT_COMPACT_TRAINING_PAIRS';save();xx=[];yy=[]
        with torch.no_grad():
            for path in paths:
                sx=[];sy=[]
                for regime,x,phi,hyper,low,scores,info in forwards(path):
                    idx=torch.randperm(len(phi),device='cuda:0',generator=generator)[:2048]
                    sx.append(x.reshape(-1,256)[idx]);sy.append((phi[idx]-center)@v)
                xx.append(torch.cat(sx));yy.append(torch.cat(sy))
        trainx=torch.cat(xx[:12]);trainy=torch.cat(yy[:12]);valx=torch.cat(xx[12:]);valy=torch.cat(yy[12:]);del xx,yy
        mean=trainx.mean(0);std=trainx.std(0).clamp_min(1e-3)
        trainx=(trainx-mean)/std;valx=(valx-mean)/std
        student=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,64)).cuda()
        optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=.01)
        best=float('inf');bad=0;best_epoch=0;start_epoch=1
        if resume and (OUT/'last.pt').exists():
            last=torch.load(OUT/'last.pt',weights_only=True);student.load_state_dict(last['state']);optimizer.load_state_dict(last['optimizer']);torch.cuda.set_rng_state(last['CUDA_rng']);start_epoch=last['epoch']+1;bad=last['bad']
            selected=torch.load(OUT/'best.pt',weights_only=True);best=selected['teacher_mse'];best_epoch=selected['epoch'];report['curves']=[r for r in report['curves'] if r['epoch']<start_epoch]
        report['status']='TRAINING';save()
        for epoch in range(start_epoch,301):
            student.train();order=torch.randperm(len(trainx),device='cuda:0');total=0
            for idx in order.split(4096):
                optimizer.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(student(trainx[idx]),trainy[idx]);loss.backward();optimizer.step();total+=float(loss)*len(idx)
            student.eval()
            with torch.no_grad():mse=float(nn.functional.mse_loss(student(valx),valy))
            if mse<best-1e-7:
                best=mse;bad=0;best_epoch=epoch;torch.save({'state':student.state_dict(),'input_mean':mean,'input_std':std,'feature_center':center,'basis':v,'epoch':epoch,'teacher_mse':mse},OUT/'best.pt')
            else:bad+=1
            report['curves'].append({'epoch':epoch,'train_mse':total/len(trainx),'validation_teacher_mse':mse,'best_epoch':best_epoch})
            if epoch%25==0:
                torch.save({'state':student.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'bad':bad,'CUDA_rng':torch.cuda.get_rng_state()},OUT/'last.pt');save();print(json.dumps(report['curves'][-1]),flush=True)
            if epoch>=100 and bad>=50:break
        checkpoint=torch.load(OUT/'best.pt',weights_only=True);student.load_state_dict(checkpoint['state']);student.eval()
        # Fold input normalization into the first affine map for deployment.
        with torch.no_grad():
            student[0].weight.div_(std);student[0].bias.sub_(student[0].weight@mean)
        torch.save({'state':student.state_dict(),'feature_center':center,'basis':v,'normalization_folded':True,'epoch':best_epoch},OUT/'deployment.pt')
        del trainx,trainy,valx,valy
        subset=ROOT/'assets/coco_quality_seed2027_v1';manifest=json.loads((subset/'manifest.json').read_text());byid={x['image_id']:x for x in manifest['images']}
        report['status']='EVALUATING_REAL_SURROGATE';save()
        with torch.no_grad():
            for n,path in enumerate(paths[12:]):
                for regime,x,phi,hyper,low,scores,info in forwards(path):
                    with np.load(subset/byid[info['image_id']]['gt_file'],allow_pickle=False) as gt:
                        truth=torch.from_numpy(gt['gt']).cuda();ids=gt['annotation_ids'].tolist()
                    assert info['annotation_ids']==ids
                    p=len(x);coeff=student(x)
                    weights=torch.einsum('pmc,ucr->pmur',hyper,v.reshape(16,32,64)).reshape(p,64,64)
                    bias=torch.einsum('pmc,uc->pmu',hyper,center.reshape(16,32)).reshape(p,64)
                    ap=(torch.bmm(coeff,weights.transpose(1,2))+bias[:,None]).reshape(p,4096,4,16)
                    margin=(ap.abs()/hyper.norm(dim=-1)[:,None,:,None].clamp_min(1e-6)).amin((-2,-1))
                    priorities=margin.argsort(-1);original=full_masks(low,info);oq,ob=quality(original,truth);chosen=scores[:,1:].argmax(-1)
                    for budget in (0,.05,.1):
                        mixed=ap.clone();patch_error=0.
                        if budget:
                            indices=priorities[:,:math.ceil(4096*budget)];batch=torch.arange(p,device='cuda:0')[:,None]
                            states=x[batch,indices].reshape(-1,256,1,1)
                            patches=teacher.output_upscaling(states).flatten(2)
                            h=hyper[:,None].expand(-1,indices.shape[1],-1,-1).reshape(-1,4,32)
                            exact=torch.bmm(h,patches).reshape(p,indices.shape[1],4,16)
                            # Layout/FP32-batch drift is diagnostic, not an
                            # equivalence pass or a relaxed old numeric gate.
                            patch_error=float((exact-phases(low)[batch,indices]).abs().max())
                            mixed[batch,indices]=exact
                        masks=full_masks(grid(mixed),info);q,bq=quality(masks,truth);flips=(masks!=original).sum((-2,-1))
                        for j,aid in enumerate(ids):
                            c=int(chosen[j]);report['rows'].append({'image_id':info['image_id'],'annotation_id':aid,'regime':regime,'budget':budget,'original_iou':float(oq[j,c]),'delta_iou':float(q[j,c]-oq[j,c]),'delta_boundary_iou':float(bq[j,c]-ob[j,c]),'flip_fraction':float(flips[j,c+1]/masks.shape[-1]/masks.shape[-2]),'actual_patch_max_logit_error':patch_error})
                save();print(json.dumps({'evaluation_images':n+1}),flush=True)
        for regime in ('central','near_boundary','box'):
            report['summary'][regime]={}
            for budget in (0,.05,.1):
                rows=[r for r in report['rows'] if r['regime']==regime and r['budget']==budget]
                report['summary'][regime][str(budget)]={'mean_delta_iou':float(np.mean([r['delta_iou'] for r in rows])),'image_cluster_delta95':bootstrap_image_groups(rows,'delta_iou'),'mean_flip_fraction':float(np.mean([r['flip_fraction'] for r in rows]))}
        report.update(status='COMPLETED_REAL_SURROGATE_DEVELOPMENT',selected_epoch=best_epoch,parameters=sum(p.numel() for p in student.parameters()));print(json.dumps(report['summary']),flush=True)
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:
        for h in handles:h.remove()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');main(p.parse_args().resume)
