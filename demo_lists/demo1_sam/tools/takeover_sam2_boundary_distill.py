"""Matched teacher-only objective controls for a frozen rank64 SAM2 surrogate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from takeover_sam2_quality import ROOT,build_sam2,SAM2ImagePredictor,encode
from takeover_head_fallback import phases

BASE=ROOT/'results/takeover_20261001_v1'
SOURCE=BASE/'sam2_local_student_v1'
INIT=BASE/'sam2_local_student_v2_converge'


def main(a):
    a.output.mkdir(exist_ok=a.resume)
    report={'status':'WAITING_VRAM','pid':os.getpid(),'stages':[],'curves':{},
      'protocol':'Objective alignment intervention on old24 only: sorted first12 train/PCA, last12 teacher-only validation and development. Same original FLOAT32 indexed dynamic/static inputs, rank64 PCA,69760-parameter Linear1024->64/GELU/Linear64 and exact10% fallback deployment unchanged. All three objectives initialize identical converged v2 best raw model, reset matched AdamW lr0.0003 wd0.01 and seed2035; min100/max1200 patience100 per-arm teacher validation objective, no GT earlystop. Controls coefficientMSE, actual native four-mask logits divided by original hyper norms MSE, boundary-weighted same response MSE. Fixed boundary weight1+9*exp(-abs(normalized teacher logit)/sigma); sigma is train-only10th percentile abs normalized teacher logit; one train-only global weight normalization preserves parent importance. No teacher/GT accessed at deployment. Actual native teacher state replay must match all indexed original inputs before augmentation. Save only small prompt hyper/projection bank, indexed FLOAT32 teacher responses and small checkpoints; reuse original456MB input archive and model weights. Not independent test, latency, policy preservation or publication evidence.'}
    if a.resume:
        prior=json.loads((a.output/'report.json').read_text())
        if prior['status']!='ERROR':raise RuntimeError('Only recover an ERROR experiment')
        (a.output/('failure_'+str(time.time_ns())+'.json')).write_text(json.dumps(prior,indent=2)+'\n')
        report=prior;report.update(pid=os.getpid(),status='RECOVERING')
    def save():
        p=a.output/'report.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(a.output/'report.json')
    save()
    try:
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6500:time.sleep(5)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with np.load(SOURCE/'training_pairs.npz',allow_pickle=False) as f:
            banks=torch.from_numpy(f['static_bank']).cuda();dynamic=torch.from_numpy(f['dynamic']).cuda();targets=torch.from_numpy(f['targets']).cuda()
            image_idx=torch.from_numpy(f['image_index'].astype(np.int64)).cuda();parent_idx=torch.from_numpy(f['parent_index'].astype(np.int64)).cuda()
        response_file=a.output/'teacher_response_pairs.npz'
        if not response_file.exists():
            report['status']='REPLAY_AND_ALIGN_NATIVE_TEACHER';save()
            model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
            predictor=SAM2ImagePredictor(model);d=model.sam_mask_decoder;capture={}
            handle=d.transformer.register_forward_hook(lambda m,i,o:capture.update(x=o[1]))
            infos=sorted(json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text())['images'],key=lambda r:r['image_id'])
            generator=torch.Generator(device='cuda:0').manual_seed(2027)
            logits=[];hyperbank=[];prompt_ids=[];metadata=[];offset=0;max_state_error=0.
            try:
                with torch.inference_mode():
                    for n,info in enumerate(infos):
                        rgb=np.asarray(Image.open(ROOT/'assets/coco_quality_seed2027_v1'/info['image_file']).convert('RGB'));predictor.set_image(rgb)
                        for ri,regime in enumerate(('central','near_boundary','box')):
                            sparse,dense=encode(predictor,info['objects'],regime,rgb.shape[:2])
                            low,iq,tokens,obj=d.predict_masks(image_embeddings=predictor._features['image_embed'],image_pe=model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,repeat_image=len(sparse)>1,high_res_features=predictor._features['high_res_feats'])
                            x=capture['x'];idx=torch.randperm(len(x)*4096,device='cuda:0',generator=generator)[:2048];pair_start=n*6144+ri*2048;sl=slice(pair_start,pair_start+2048)
                            assert torch.equal(parent_idx[sl],idx%4096) and bool((image_idx[sl]==n).all())
                            delta=float((x.reshape(-1,256)[idx]-dynamic[sl]).abs().max());max_state_error=max(max_state_error,delta)
                            if delta>5e-5:raise RuntimeError('Teacher replay states do not align: '+str(delta))
                            h=torch.stack([mlp(tokens[:,j]) for j,mlp in enumerate(d.output_hypernetworks_mlps)],1)
                            values=phases(low).reshape(-1,4,16)[idx]
                            logits.append(values.cpu().numpy());hyperbank.append(h.cpu().numpy());prompt_ids.append((idx//4096+offset).cpu().numpy().astype(np.uint16))
                            metadata.extend({'image_id':info['image_id'],'annotation_id':v['annotation_id'],'regime':regime} for v in info['objects']);offset+=len(x)
                        report.update(aligned_images=n+1,max_state_replay_error=max_state_error);save();print(json.dumps({'aligned_images':n+1,'max_state_error':max_state_error}),flush=True)
                with (a.output/'teacher_response_pairs.tmp').open('xb') as f:np.savez_compressed(f,teacher_logits=np.concatenate(logits),hyper_bank=np.concatenate(hyperbank),prompt_index=np.concatenate(prompt_ids))
                (a.output/'teacher_response_pairs.tmp').replace(response_file)
                (a.output/'teacher_response_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            finally:handle.remove()
            del model,predictor,d,capture,logits,hyperbank,prompt_ids,x,h,low,tokens
            torch.cuda.empty_cache()
        with np.load(response_file,allow_pickle=False) as f:
            teacher=torch.from_numpy(f['teacher_logits']).cuda();hyper=torch.from_numpy(f['hyper_bank']).cuda();prompt_idx=torch.from_numpy(f['prompt_index'].astype(np.int64)).cuda()
        original=torch.load(INIT/'best.pt',map_location='cuda:0',weights_only=True)
        mean,std=original['input_mean'],original['input_std']
        inputs=(torch.cat((dynamic,banks[image_idx,parent_idx]),-1)-mean)/std
        del banks,dynamic,parent_idx
        pca=torch.load(SOURCE/'basis.pt',map_location='cuda:0',weights_only=True);basis,center=pca['basis'],pca['mean']
        norm=hyper.norm(dim=-1).clamp_min(1e-6)
        projection=torch.einsum('pmc,ucr->pmur',hyper,basis.reshape(16,32,64))/norm[:,:,None,None]
        projection=projection.reshape(len(hyper),64,64).transpose(1,2).contiguous()
        bias=(torch.einsum('pmc,uc->pmu',hyper,center.reshape(16,32))/norm[:,:,None]).reshape(len(hyper),64)
        teacher=(teacher/norm[prompt_idx,:,None]).reshape(len(teacher),64)
        train=(image_idx<12).nonzero()[:,0];val=(image_idx>=12).nonzero()[:,0]
        sigma=float(torch.quantile(teacher[train].abs().flatten(),.1).clamp_min(1e-6))
        boundary_weight=1+9*torch.exp(-teacher.abs()/sigma);boundary_weight=boundary_weight/boundary_weight[train].mean()
        report.update(boundary_sigma_train_only=sigma,train_samples=len(train),validation_samples=len(val),response_archive_bytes=response_file.stat().st_size);save()
        for arm in ('coefficient','response_uniform','response_boundary'):
            out=a.output/arm
            if out.exists() and (out/'deployment.pt').exists():continue
            out.mkdir(exist_ok=True);torch.manual_seed(2035)
            s=nn.Sequential(nn.Linear(1024,64),nn.GELU(),nn.Linear(64,64)).cuda();s.load_state_dict(original['state'])
            optimizer=torch.optim.AdamW(s.parameters(),lr=.0003,weight_decay=.01)
            best=float('inf');best_epoch=bad=0;start=1
            if a.resume and (out/'last.pt').exists():
                last=torch.load(out/'last.pt',map_location='cuda:0',weights_only=True);s.load_state_dict(last['state']);optimizer.load_state_dict(last['optimizer']);torch.cuda.set_rng_state(last['CUDA_rng']);start=last['epoch']+1;bad=last['bad'];best,last_best=last['best'],last['best_epoch'];best_epoch=last_best
            def loss(ids):
                coeff=s(inputs[ids])
                if arm=='coefficient':return nn.functional.mse_loss(coeff,targets[ids])
                predicted=torch.bmm(coeff[:,None],projection[prompt_idx[ids]]).squeeze(1)+bias[prompt_idx[ids]]
                error=(predicted-teacher[ids]).square()
                if arm=='response_boundary':error=error*boundary_weight[ids]
                return error.mean()
            curves=report['curves'].setdefault(arm,[]);curves[:]=[v for v in curves if v['epoch']<start]
            report.update(status='TRAINING_MATCHED_OBJECTIVES',current_arm=arm);save()
            for epoch in range(start,1201):
                s.train();order=train[torch.randperm(len(train),device='cuda:0')];total=0.
                for ids in order.split(4096):
                    optimizer.zero_grad(set_to_none=True);value=loss(ids);value.backward();optimizer.step();total+=float(value)*len(ids)
                s.eval()
                with torch.no_grad():vl=sum(float(loss(ids))*len(ids) for ids in val.split(4096))/len(val)
                if vl<best-1e-9:
                    best,bad,best_epoch=vl,0,epoch
                    torch.save({'state':s.state_dict(),'input_mean':mean,'input_std':std,'epoch':epoch,'teacher_validation_objective':vl,'arm':arm},out/'best.pt')
                else:bad+=1
                curves.append({'epoch':epoch,'train_objective':total/len(train),'validation_objective':vl,'best_epoch':best_epoch})
                if epoch%25==0:
                    torch.save({'state':s.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'bad':bad,'best':best,'best_epoch':best_epoch,'CUDA_rng':torch.cuda.get_rng_state()},out/'last.pt');save();print(json.dumps({'arm':arm,**curves[-1]}),flush=True)
                if epoch>=100 and bad>=100:break
            state=torch.load(out/'best.pt',map_location='cuda:0',weights_only=True);s.load_state_dict(state['state']);s.eval().requires_grad_(False)
            with torch.no_grad():s[0].weight.div_(std);s[0].bias.sub_(s[0].weight@mean)
            torch.save({'state':{k:v.cpu() for k,v in s.state_dict().items()},'feature_center':center.cpu(),'basis':basis.cpu(),'normalization_folded':True,'epoch':best_epoch,'architecture':'SAM2 static768+dynamic256 hidden64/rank64'},out/'deployment.pt')
            report.setdefault('selected',{})[arm]={'epoch':best_epoch,'last_epoch':epoch,'plateau_before_cap':epoch<1200,'validation_objective':best};save()
            del s,optimizer
        report['status']='COMPLETED_MATCHED_OBJECTIVE_TRAINING';save()
        # Release training features before loading actual full decoder validation.
        del inputs,targets,teacher,hyper,projection,bias,boundary_weight
        torch.cuda.empty_cache()
        for arm in ('coefficient','response_uniform','response_boundary'):
            out=a.output/arm;quality=out/'development_compiled_fp16.json'
            if quality.exists() and json.loads(quality.read_text())['status'].startswith('COMPLETED'):continue
            report.update(status='COMPILED_DEVELOPMENT_TASK_VALIDATION',current_arm=arm);save()
            with (out/'development_compiled_fp16.log').open('x') as log:
                cmd=[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(ROOT/'assets/coco_quality_seed2027_v1'),'--old-development','--student-path',str(out/'deployment.pt'),'--compiled-decoder-fp16','--output',str(quality),'--scope','Old24 sorted-ID last12 development only; matched objective intervention, no new test or GT epoch selection']
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);report['stages'].append({'arm':arm,'pid':p.pid,'command':cmd});save();code=p.wait()
            if code:raise RuntimeError(arm+' development compiled evaluation failed')
        report.update(status='COMPLETED_MATCHED_OBJECTIVE_DEVELOPMENT',current_arm=None)
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');main(p.parse_args())
