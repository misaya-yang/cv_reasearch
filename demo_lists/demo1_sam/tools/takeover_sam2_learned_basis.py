"""Matched fixed versus learned rank64 response basis, unchanged deployment shapes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from takeover_sam2_quality import ROOT
BASE=ROOT/'results/takeover_20261001_v1'
OLD=BASE/'sam2_local_student_v1'
RESP=BASE/'sam2_boundary_objective_v1'


def main(a):
    a.output.mkdir(exist_ok=a.resume)
    r={'status':'WAITING_FORMAL_CPU_ONLY','pid':os.getpid(),'curves':{},'stages':[],
       'protocol':'Representation intervention on old24 only: sorted first12 train/PCA and last12 teacher-response validation/development; reuse original indexed inputs and native four-mask responses, no new extraction. Both controls initialize exact same boundary-response epoch735 raw student and original PCA basis; identical teacher boundary response objective/sigma/normalization, seed2036 AdamW lr0.0001 wd0.01 min100/max1000 patience100. Fixed basis student continuation vs jointly learned rank64 basis+student; center fixed. Joint arm has32768 extra TRAINABLE basis scalars, but deployment already stores the same512x64 basis, same69760-parameter student shapes, static768 per-image affine/dynamic256 and no extra inference operators or teacher access. No orthogonality/equivalence claim for learned basis. Loss fitted to original hyper-normalized four-mask responses, original10% local phase fallback/native0.98 policy untouched, all4 interfaces and0/5/10 budgets preserved. Teacher validation only selects epoch; actual compiled old-development task diagnosis required, not independent test or reused speed evidence. Formal predecessor completes before CUDA allocation, no mutual waiting/contamination.'}
    if a.resume:
        prior=json.loads((a.output/'report.json').read_text())
        if prior['status']!='ERROR':raise RuntimeError('Recovery requires ERROR')
        (a.output/('failed_'+str(time.time_ns())+'.json')).write_text(json.dumps(prior,indent=2)+'\n');r=prior;r.update(status='WAITING_FORMAL_CPU_ONLY',pid=os.getpid())
    def save():
        p=a.output/'report.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(a.output/'report.json')
    save()
    try:
        while True:
            if a.predecessor.exists():
                q=json.loads(a.predecessor.read_text())
                if q['status']=='ERROR':raise RuntimeError('Formal predecessor failed; preserve trace first')
                if q['status']=='COMPLETED_EXCLUSIVE_MATCHED_CHECKPOINT_DECODER_TIMING':break
            time.sleep(5)
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6500:time.sleep(5)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        initial=torch.load(RESP/'response_boundary/best.pt',map_location='cuda:0',weights_only=True)
        pca=torch.load(OLD/'basis.pt',map_location='cuda:0',weights_only=True);center=pca['mean'];original_basis=pca['basis']
        with np.load(OLD/'training_pairs.npz',allow_pickle=False) as f:
            bank=torch.from_numpy(f['static_bank']).cuda();dyn=torch.from_numpy(f['dynamic']).cuda();ii=torch.from_numpy(f['image_index'].astype(np.int64)).cuda();pi=torch.from_numpy(f['parent_index'].astype(np.int64)).cuda()
        mean,std=initial['input_mean'],initial['input_std'];x=(torch.cat((dyn,bank[ii,pi]),-1)-mean)/std;del bank,dyn,pi
        with np.load(RESP/'teacher_response_pairs.npz',allow_pickle=False) as f:
            teacher=torch.from_numpy(f['teacher_logits']).cuda();hyper=torch.from_numpy(f['hyper_bank']).cuda();prompt=torch.from_numpy(f['prompt_index'].astype(np.int64)).cuda()
        norm=hyper.norm(dim=-1).clamp_min(1e-6);teacher=(teacher/norm[prompt,:,None]).reshape(len(teacher),64)
        bias=(torch.einsum('pmc,uc->pmu',hyper,center.reshape(16,32))/norm[:,:,None]).reshape(len(hyper),64)
        train=(ii<12).nonzero()[:,0];val=(ii>=12).nonzero()[:,0]
        sigma=json.loads((RESP/'report.json').read_text())['boundary_sigma_train_only'];weight=1+9*torch.exp(-teacher.abs()/sigma);weight=weight/weight[train].mean()
        r.update(train_samples=len(train),validation_samples=len(val),boundary_sigma=sigma,initialization_epoch=initial['epoch']);save()
        for arm in ('fixed_basis','learned_basis'):
            out=a.output/arm
            if (out/'deployment.pt').exists():continue
            out.mkdir(exist_ok=True);torch.manual_seed(2036)
            student=nn.Sequential(nn.Linear(1024,64),nn.GELU(),nn.Linear(64,64)).cuda();student.load_state_dict(initial['state'])
            basis=nn.Parameter(original_basis.clone(),requires_grad=arm=='learned_basis')
            parameters=list(student.parameters())+([basis] if basis.requires_grad else [])
            optimizer=torch.optim.AdamW(parameters,lr=.0001,weight_decay=.01);best=float('inf');bad=best_epoch=0;start=1
            if a.resume and (out/'last.pt').exists():
                last=torch.load(out/'last.pt',map_location='cuda:0',weights_only=True);student.load_state_dict(last['state']);basis.data.copy_(last['basis']);optimizer.load_state_dict(last['optimizer']);torch.cuda.set_rng_state(last['CUDA_rng']);start=last['epoch']+1;best,best_epoch,bad=last['best'],last['best_epoch'],last['bad']
            def projection():return (torch.einsum('pmc,ucr->pmur',hyper,basis.reshape(16,32,64))/norm[:,:,None,None]).reshape(len(hyper),64,64).transpose(1,2).contiguous()
            fixed=projection().detach() if arm=='fixed_basis' else None
            def loss(ids):
                matrix=fixed if fixed is not None else projection()
                logits=torch.bmm(student(x[ids])[:,None],matrix[prompt[ids]]).squeeze(1)+bias[prompt[ids]]
                return ((logits-teacher[ids]).square()*weight[ids]).mean()
            r.update(status='TRAINING_MATCHED_BASIS_CONTROLS',current_arm=arm);r.setdefault('trainable_parameters',{})[arm]=sum(p.numel() for p in parameters);curves=r['curves'].setdefault(arm,[]);curves[:]=[v for v in curves if v['epoch']<start];save()
            for epoch in range(start,1001):
                student.train();order=train[torch.randperm(len(train),device='cuda:0')];total=0.
                for ids in order.split(4096):
                    optimizer.zero_grad(set_to_none=True);value=loss(ids);value.backward();optimizer.step();total+=float(value)*len(ids)
                student.eval()
                with torch.no_grad():vl=sum(float(loss(ids))*len(ids) for ids in val.split(4096))/len(val)
                if vl<best-1e-9:
                    best,bad,best_epoch=vl,0,epoch
                    torch.save({'state':student.state_dict(),'basis':basis.detach(),'input_mean':mean,'input_std':std,'epoch':epoch,'teacher_objective':vl},out/'best.pt')
                else:bad+=1
                curves.append({'epoch':epoch,'train_objective':total/len(train),'validation_objective':vl,'best_epoch':best_epoch})
                if epoch%25==0:
                    torch.save({'state':student.state_dict(),'basis':basis.detach(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'bad':bad,'CUDA_rng':torch.cuda.get_rng_state()},out/'last.pt');save();print(json.dumps({'arm':arm,**curves[-1]}),flush=True)
                if epoch>=100 and bad>=100:break
            chosen=torch.load(out/'best.pt',map_location='cuda:0',weights_only=True);student.load_state_dict(chosen['state']);student.eval().requires_grad_(False)
            with torch.no_grad():student[0].weight.div_(std);student[0].bias.sub_(student[0].weight@mean)
            torch.save({'state':{k:v.cpu() for k,v in student.state_dict().items()},'basis':chosen['basis'].cpu(),'feature_center':center.cpu(),'normalization_folded':True,'epoch':best_epoch,'architecture':'SAM2 static768+dynamic256 hidden64/rank64','basis_learned':arm=='learned_basis'},out/'deployment.pt')
            r.setdefault('selected',{})[arm]={'epoch':best_epoch,'last_epoch':epoch,'plateau_before_cap':epoch<1000,'teacher_objective':best};save();del student,basis,optimizer,fixed
        del x,teacher,hyper,bias,weight;torch.cuda.empty_cache();jobs=[]
        r.update(status='CONCURRENT_COMPILED_OLD_DEVELOPMENT',current_arm=None);save()
        for arm in ('fixed_basis','learned_basis'):
            out=a.output/arm;quality=out/'development_compiled_fp16.json';logpath=out/'development_compiled_fp16.log'
            if quality.exists() and json.loads(quality.read_text())['status'].startswith('COMPLETED'):continue
            stamp=str(time.time_ns())
            if quality.exists():quality.rename(out/('development_failed_'+stamp+'.json'))
            if logpath.exists():logpath.rename(out/('development_failed_'+stamp+'.log'))
            cmd=[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(ROOT/'assets/coco_quality_seed2027_v1'),'--old-development','--student-path',str(out/'deployment.pt'),'--compiled-decoder-fp16','--output',str(quality),'--scope','Old24 last12 development only; matched fixed vs learned rank64 response basis, teacher-objective epochs frozen, no GT fitting or speed claim']
            log=logpath.open('x');p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log,arm));r['stages'].append({'arm':arm,'pid':p.pid,'command':cmd});save()
        codes=[]
        for p,log,arm in jobs:
            code=p.wait();log.close();codes.append(code);r['stages'].append({'arm':arm,'returncode':code});save()
        if any(codes):raise RuntimeError('Compiled development worker failed; preserve other completed control')
        r['status']='COMPLETED_MATCHED_BASIS_DEVELOPMENT'
    except BaseException as e:r.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--predecessor',type=Path,required=True);p.add_argument('--resume',action='store_true');main(p.parse_args())
