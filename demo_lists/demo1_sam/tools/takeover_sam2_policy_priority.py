"""Development-only policy-aware allocation with unchanged local fallback budget."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from takeover_sam2_quality import ROOT,build_sam2,SAM2ImagePredictor,encode
from takeover_sam2_compact_head import load_student,build_cache,head,dense_phase_head,phase_patches,dynamic_choice,quality_all
from takeover_head_fallback import grid
from takeover_flip_score import bootstrap_image_groups


def prioritized(d,s,cache,x,hyper,center,basis,factor,window):
    p=len(x)
    coeff=s[2](s[1](torch.nn.functional.linear(x,s[0].weight[:,:256])+cache.static_hidden[None]))
    weights=torch.einsum('pmc,ucr->pmur',hyper,basis.reshape(16,32,64)).reshape(p,64,64)
    bias=torch.einsum('pmc,uc->pmu',hyper,center.reshape(16,32)).reshape(p,64)
    ph=(torch.bmm(coeff,weights.transpose(1,2))+bias[:,None]).reshape(p,4096,4,16)
    coarse=grid(ph)
    stability=d._get_stability_scores(coarse[:,:1])[:,0]
    # This gate and allocation depend only on the candidate, never teacher or GT.
    near=(stability-d.dynamic_multimask_stability_thresh).abs()<window
    normalized=ph.abs()/hyper.norm(dim=-1)[:,None,:,None].clamp_min(1e-6)
    token0=(normalized[:,:,0]/torch.where(near,factor,1.)[:,None,None]).amin(-1)
    margin=torch.minimum(normalized[:,:,1:].amin((-2,-1)),token0)
    idx=margin.topk(math.ceil(4096*.1),dim=-1,largest=False).indices
    batch=torch.arange(p,device=x.device)[:,None]
    patches=phase_patches(d,cache.phase,x[batch,idx].reshape(-1,256),cache.t0[idx].reshape(-1,32,4,4),cache.t1[idx].reshape(-1,64,2,2))
    h=hyper[:,None].expand(-1,idx.shape[1],-1,-1).reshape(-1,4,32)
    ph[batch,idx]=torch.bmm(h,patches).reshape(p,idx.shape[1],4,16)
    return grid(ph),stability,near


@torch.inference_mode()
def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    r={'status':'WAITING_PIPELINE_CPU_ONLY','pid':os.getpid(),'rows':[],'summary':{},
       'protocol':'Old24 sorted-ID last12 development images only; frozen SAM2.1-L teacher-MSE student, FP32 encoder outputs cast to decoder FP16. Native transformer once/regime, actual shared states and original hyper/IQ; candidate computes student coefficients and true original local patches, never full teacher patches. All4 outputs preserved. Budget10% unchanged, original all4 normalized margin control vs token0 priority factor2/4 gated by candidate coarse native stability within fixed0.02 of original0.98 threshold; factor2 unconditional control. Native threshold unchanged, no teacher error/GT in selector. Full token0 recomputation guard at candidate coarse distance0.005/0.02 is a strong alternative with explicitly extra full head cost/trigger fraction, not a same-budget variant. Native FP16 and full original phase head controls; reference full head calculated only for evaluation. Eager head mechanism development diagnosis, not compiled deployment/latency or independent test. GT only original prompts and scoring, no epoch/budget fitting.'}
    def save():
        p=a.output.with_suffix('.tmp');p.write_text(json.dumps(r,indent=2)+'\n');p.replace(a.output)
    save();handle=None
    try:
        while True:
            prior=json.loads(a.predecessor.read_text())
            if prior['status']=='ERROR':raise RuntimeError('Pipeline failed; retain diagnosis before next stage')
            if prior['status']=='COMPLETED_EXCLUSIVE_PIPELINE_TIMING':break
            time.sleep(5)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
        predictor=SAM2ImagePredictor(model);d=copy.deepcopy(model.sam_mask_decoder).half()
        s,center,basis=load_student(ROOT/'results/takeover_20261001_v1/sam2_local_student_v1/deployment.pt','cuda:0');s.requires_grad_(False);s=s.half();center=center.half();basis=basis.half()
        capture={};handle=d.transformer.register_forward_hook(lambda m,i,o:capture.update(x=o[1]))
        subset=ROOT/'assets/coco_quality_seed2027_v1';infos=sorted(json.loads((subset/'manifest.json').read_text())['images'],key=lambda v:v['image_id'])[12:]
        r['status']='RUNNING_DEVELOPMENT_POLICY_ALLOCATION';save()
        for n,info in enumerate(infos):
            rgb=np.asarray(Image.open(subset/info['image_file']).convert('RGB'));predictor.set_image(rgb)
            image=predictor._features['image_embed'].half();hi=[v.half() for v in predictor._features['high_res_feats']];pe=model.sam_prompt_encoder.get_dense_pe().half();cache=build_cache(d,hi,s)
            with np.load(subset/info['gt_file'],allow_pickle=False) as f:truth=torch.from_numpy(f['gt']).cuda();ids=f['annotation_ids'].tolist()
            for regime in ('central','near_boundary','box'):
                sp,dd=encode(predictor,info['objects'],regime,rgb.shape[:2]);hd=dd[:1,:,0:1,0:1].half().expand(len(sp),-1,64,64)
                native,iq,tokens,obj=d.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sp.half(),dense_prompt_embeddings=hd,repeat_image=True,high_res_features=hi)
                x=capture['x'];hyper=torch.stack([mlp(tokens[:,j]) for j,mlp in enumerate(d.output_hypernetworks_mlps)],1)
                reference=dense_phase_head(d,cache,x,hyper)
                phase_choice=dynamic_choice(d,reference,iq);head_choice=iq[:,1:].argmax(-1)+1
                ref_full=predictor._transforms.postprocess_masks(reference,rgb.shape[:2])>0;rq,rb=quality_all(ref_full,truth)
                base=head(d,s,cache,x,hyper,center,basis,.1)
                variants=[('native_fp16',native,torch.zeros(len(sp),device=x.device,dtype=torch.bool)),('margin10',base,torch.zeros(len(sp),device=x.device,dtype=torch.bool))]
                coarse=head(d,s,cache,x,hyper,center,basis,0);coarse_st=d._get_stability_scores(coarse[:,:1])[:,0]
                for name,factor,window in [('token0_x2_near002',2.,.02),('token0_x4_near002',4.,.02),('token0_x2_always',2.,2.)]:
                    low,st,flag=prioritized(d,s,cache,x,hyper,center,basis,factor,window);variants.append((name,low,flag))
                for window in (.005,.02):
                    flag=(coarse_st-d.dynamic_multimask_stability_thresh).abs()<window
                    low=base.clone()
                    # Actually evaluate full original head for flagged prompts;
                    # do not use the already computed evaluation reference.
                    if flag.any():low[flag,0]=dense_phase_head(d,cache,x[flag],hyper[flag])[:,0]
                    variants.append(('full_token0_guard'+str(window),low,flag))
                for name,low,flag in variants:
                    choice=dynamic_choice(d,low,iq);full=predictor._transforms.postprocess_masks(low,rgb.shape[:2])>0;q,b=quality_all(full,truth)
                    stability=d._get_stability_scores(low[:,:1])[:,0]
                    for j,aid in enumerate(ids):
                        hc=int(head_choice[j]);rc=int(phase_choice[j]);cc=int(choice[j])
                        r['rows'].append({'image_id':info['image_id'],'annotation_id':aid,'regime':regime,'variant':name,'delta_head_iou':float(q[j,hc]-rq[j,hc]),'delta_dynamic_iou':float(q[j,cc]-rq[j,rc]),'delta_dynamic_boundary_iou':float(b[j,cc]-rb[j,rc]),'reference_dynamic_choice':rc,'candidate_dynamic_choice':cc,'dynamic_choice_changed':rc!=cc,'intervened':bool(flag[j]),'coarse_stability':float(coarse_st[j]),'candidate_stability':float(stability[j]),'reference_stability':float(d._get_stability_scores(reference[j:j+1,:1])[0,0])})
            save();print(json.dumps({'images':n+1,'rows':len(r['rows'])}),flush=True)
        for regime in ('central','near_boundary','box'):
            r['summary'][regime]={}
            for name in sorted({v['variant'] for v in r['rows']}):
                rows=[v for v in r['rows'] if v['regime']==regime and v['variant']==name]
                r['summary'][regime][name]={'head_delta_iou':float(np.mean([v['delta_head_iou'] for v in rows])),'dynamic_delta_iou':float(np.mean([v['delta_dynamic_iou'] for v in rows])),'dynamic_image_cluster95':bootstrap_image_groups(rows,'delta_dynamic_iou'),'choice_changes':sum(v['dynamic_choice_changed'] for v in rows),'intervention_fraction':float(np.mean([v['intervened'] for v in rows]))}
        r['status']='COMPLETED_DEVELOPMENT_POLICY_ALLOCATION';print(json.dumps(r['summary']),flush=True)
    except BaseException as e:r.update(status='ERROR',error=repr(e));raise
    finally:
        if handle is not None:handle.remove()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--predecessor',type=Path,required=True);main(p.parse_args())
