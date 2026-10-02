"""Frozen real compact decoder on a disjoint image cohort, all4 masks retained."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/compute_structure'))
from pretrained_decoder_round import load_decoder,load_inputs,full_masks
from takeover_compact_head_infer import load_student,decode
from takeover_rank_features import quality
from takeover_flip_score import bootstrap_image_groups
from baselines import PositionCache
from prototype import DensePhaseCache,shape_plan
from takeover_compact_phase import predict as cached_predict


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    source=json.loads((a.source_run/'report.json').read_text());manifest=json.loads((a.subset/'manifest.json').read_text());byid={x['image_id']:x for x in manifest['images']}
    fitted=torch.load(ROOT/'results/takeover_20261001_v1/head_compress_v1/basis.pt',weights_only=True,map_location='cpu')['fit_images']
    old=json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text());assert not set(byid)&{x['image_id'] for x in old['images']}
    assert not set(byid)&set(fitted)
    report={'status':'WAITING_VRAM','protocol':'Frozen real rank64 surrogate from old12 training/12 teacher-MSE-validation images; disjoint128 original images, original annotated clicks and boxes; original transformer/IoU/hypernetworks; fullhead execution only in reference path. Budgets0/5/10% predefined. Approximation quality, not original fixed5e-5 equality or latency evidence.','rows':[],'summary':{}}
    if a.phase_fallback:report['protocol']+=' Same frozen learned head and margin/budgets, local original head recomputed through cached phase matrices; implementation verification on the same previously evaluated128 images, not a new independent test cohort.'
    if a.compiled_phase:
        if not a.phase_fallback:raise ValueError('--compiled-phase requires --phase-fallback')
        report['protocol']+=' Actual cached dense-associated transformer plus phase local head, Inductor fullgraph/dynamicFalse without fallback compiler; official native reference. No latency claim. Fixed reference IQ choice and candidate own-IQ choice both recorded to expose compiled scheduling drift.'
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
    save()
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<2500:time.sleep(10)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,_=load_decoder(a.source_run/'mask_decoder_state.pt',torch.device('cuda:0'))
    student,center,basis=load_student(ROOT/'results/takeover_20261001_v1/local_student_head_v1/deployment.pt','cuda:0')
    phase_cache=None
    if a.phase_fallback:
        from takeover_compact_phase_fallback import build_patch_cache
        phase_cache=build_patch_cache(model)
    compiled={};plans=None;first_shapes=set()
    try:
        report['status']='RUNNING_FROZEN_DISJOINT_IMAGES';save()
        with torch.inference_mode():
            for n,item in enumerate(source['images']):
                image,pe,dense,sparse,info=load_inputs(a.source_run/item['encoded_inputs']['npz'],torch.device('cuda:0'))
                with np.load(a.subset/byid[info['image_id']]['gt_file'],allow_pickle=False) as gt:
                    truth=torch.from_numpy(gt['gt']).cuda();ids=gt['annotation_ids'].tolist()
                assert info['annotation_ids']==ids
                for regime,sp in sparse.items():
                    low,scores=model.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sp,dense_prompt_embeddings=dense)
                    ref=full_masks(low,info);q,bq=quality(ref,truth);selected=scores[:,1:].argmax(-1)
                    cache=None
                    if a.compiled_phase:
                        if plans is None:
                            plans=shape_plan(model,4096,sp.shape[1]+5,'dense_assoc','auto','auto')
                            def pred0(ca,ss):return cached_predict(model,ca,ss,student,center,basis,0,plans,phase_fallback=True)
                            def pred5(ca,ss):return cached_predict(model,ca,ss,student,center,basis,.05,plans,phase_fallback=True)
                            def pred10(ca,ss):return cached_predict(model,ca,ss,student,center,basis,.1,plans,phase_fallback=True)
                            compiled={budget:torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False) for budget,fn in ((0,pred0),(.05,pred5),(.1,pred10))}
                        cache=DensePhaseCache.build(model,image,dense,PositionCache.build(model,pe,plans))
                    for budget in (0,.05,.1):
                        if a.compiled_phase:
                            shape=(len(sp),sp.shape[1],budget)
                            start=time.monotonic()
                            predicted,iq=compiled[budget](cache,sp)
                            if shape not in first_shapes:
                                torch.cuda.synchronize();first_shapes.add(shape)
                                report.setdefault('first_shape_call_seconds',[]).append({'prompts':len(sp),'tokens':sp.shape[1],'budget':budget,'seconds':time.monotonic()-start})
                        else:predicted,iq=decode(model,student,image,pe,dense,sp,center,basis,budget,phase_cache=phase_cache)
                        assert predicted.shape==low.shape and iq.shape==scores.shape
                        iqerr=float((iq-scores).abs().max())
                        masks=full_masks(predicted,info);pq,pb=quality(masks,truth);flip=(masks!=ref).sum((-2,-1))
                        own_selected=iq[:,1:].argmax(-1)
                        for j,aid in enumerate(ids):
                            c=int(selected[j]);own=int(own_selected[j]);report['rows'].append({'image_id':info['image_id'],'annotation_id':aid,'regime':regime,'budget':budget,'original_iou':float(q[j,c]),'predicted_iou':float(pq[j,c]),'delta_iou':float(pq[j,c]-q[j,c]),'delta_boundary_iou':float(pb[j,c]-bq[j,c]),'flip_fraction':float(flip[j,c+1]/masks.shape[-1]/masks.shape[-2]),'original_IQ_head_max_drift':iqerr,'candidate_choice_differs':own!=c,'predicted_iou_with_own_IQ_choice':float(pq[j,own]),'delta_iou_with_own_IQ_choice':float(pq[j,own]-q[j,c])})
                if (n+1)%8==0:save();print(json.dumps({'images':n+1}),flush=True)
        for regime in ('central','near_boundary','box'):
            report['summary'][regime]={}
            for budget in (0,.05,.1):
                rows=[r for r in report['rows'] if r['regime']==regime and r['budget']==budget]
                report['summary'][regime][str(budget)]={'mean_delta_iou':float(np.mean([r['delta_iou'] for r in rows])),'image_cluster_delta95':bootstrap_image_groups(rows,'delta_iou'),'mean_flip_fraction':float(np.mean([r['flip_fraction'] for r in rows])),'mean_own_choice_delta_iou':float(np.mean([r['delta_iou_with_own_IQ_choice'] for r in rows])),'own_choice_image_cluster_delta95':bootstrap_image_groups(rows,'delta_iou_with_own_IQ_choice'),'choice_changes':sum(r['candidate_choice_differs'] for r in rows)}
        report['status']='COMPLETED_FROZEN_DISJOINT_IMAGES';print(json.dumps(report['summary']),flush=True)
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase-fallback',action='store_true')
    p.add_argument('--compiled-phase',action='store_true')
    for k in ('source-run','subset','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
