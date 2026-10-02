"""Frozen SAM2 compact head on fresh images and cross-domain first frames."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
import torch
from takeover_sam2_quality import ROOT,build_sam2,SAM2ImagePredictor,encode
from takeover_sam2_compact_head import load_student,build_cache,decode,dynamic_choice,quality_all
from takeover_flip_score import bootstrap_image_groups


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    report={'status':'WAITING_VRAM','pid':os.getpid(),'scope':a.scope,'rows':[],'summary':{},
            'protocol':'Frozen SAM2.1-L rank64 static-skip student, teacher-MSE-selected before these outputs, budgets0/5/10% fixed. Native encoder once/image, actual compact decoder without full reference head in candidate path; all4mask/IQ/masktoken/objscore preserved. GT only prompts and evaluation; original multi1..3 head and official dynamic token0 policies scored separately, candidate uses own predictions. Void255 excluded when present. No video memory, exclusive latency or SOTA claim.'}
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
    save()
    try:
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<a.minimum_free_mib:time.sleep(10)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
        predictor=SAM2ImagePredictor(model);d=model.sam_mask_decoder
        student,center,basis=load_student(a.student_path,'cuda:0');student.requires_grad_(False)
        report['student_path']=str(a.student_path)
        compiled={};first_shapes=set()
        if a.compiled_decoder_fp16:
            # Independent point/box shapes and larger DAVIS object batches all
            # remain fullgraph specializations, rather than an eager fallback.
            torch._dynamo.config.recompile_limit=32
            report['protocol']+=' Decoder-only FP16 plus actual Inductor fullgraph/dynamicFalse, shared FP32 encoder/highres outputs cast tohalf; FP32 native reference and SAME-precision native/complete-phase controls. Cohort identity and test/development status are specified in scope; no refitting during evaluation. Full encoder remainsFP32.'
            d16=copy.deepcopy(d).half();student=student.half();center=center.half();basis=basis.half()
            def native16(ca,ss,dd):return d16.predict_masks(image_embeddings=ca.image,image_pe=ca.pe,sparse_prompt_embeddings=ss,dense_prompt_embeddings=dd,repeat_image=True,high_res_features=ca.high_res)
            def phase16(ca,ss,dd):return decode(d16,student,ca,ca.image,ca.pe,ss,dd,center,basis,0,mode='dense_phase')
            def pred0(ca,ss,dd):return decode(d16,student,ca,ca.image,ca.pe,ss,dd,center,basis,0)
            def pred5(ca,ss,dd):return decode(d16,student,ca,ca.image,ca.pe,ss,dd,center,basis,.05)
            def pred10(ca,ss,dd):return decode(d16,student,ca,ca.image,ca.pe,ss,dd,center,basis,.1)
            compiled={key:torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False) for key,fn in (('native',native16),('phase',phase16),(0,pred0),(.05,pred5),(.1,pred10))}
        def compiled_call(key,ca,ss,dd):
            shape=(str(key),len(ss),ss.shape[1]);start=time.monotonic()
            value=compiled[key](ca,ss,dd)
            if shape not in first_shapes:
                torch.cuda.synchronize();first_shapes.add(shape)
                assert all(bool(torch.isfinite(v).all()) for v in value)
                report.setdefault('first_compiled_shape_seconds',[]).append({'family':str(key),'prompts':len(ss),'tokens':ss.shape[1],'seconds':time.monotonic()-start});save()
            return value
        manifest=json.loads((a.subset/'manifest.json').read_text())
        if a.old_development:
            assert a.subset.name=='coco_quality_seed2027_v1' and len(manifest['images'])==24
            manifest['images']=sorted(manifest['images'],key=lambda r:r['image_id'])[12:]
            report['protocol']+=' Explicit old sorted-ID last12 DEVELOPMENT split; GT scoring only, not independent test.'
        train_ids=set(torch.load(ROOT/'results/takeover_20261001_v1/sam2_local_student_v1/basis.pt',weights_only=True,map_location='cpu')['fit_images'])
        old_ids={r['image_id'] for r in json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text())['images']}
        if a.subset.name.startswith('coco') and not a.old_development:
            assert not {r['image_id'] for r in manifest['images']}&(train_ids|old_ids)
        report['status']='RUNNING_FROZEN_REAL_SAM2_HEAD';save()
        with torch.inference_mode():
            for n,info in enumerate(manifest['images']):
                rgb=np.asarray(Image.open(a.subset/info['image_file']).convert('RGB'));predictor.set_image(rgb)
                if a.compiled_decoder_fp16:
                    high_res=[v.half() for v in predictor._features['high_res_feats']]
                    cache=build_cache(d16,high_res,student);cache.image=predictor._features['image_embed'].half();cache.pe=model.sam_prompt_encoder.get_dense_pe().half();cache.high_res=high_res
                else:cache=build_cache(d,predictor._features['high_res_feats'],student)
                with np.load(a.subset/info['gt_file'],allow_pickle=False) as data:
                    truth=torch.from_numpy(data['gt']).cuda();ids=data['annotation_ids'].tolist();valid=torch.from_numpy(data['valid']).cuda() if 'valid' in data else None
                assert ids==[o['annotation_id'] for o in info['objects']]
                for regime in ('central','near_boundary','box'):
                    sp,dense=encode(predictor,info['objects'],regime,rgb.shape[:2]);image,pe=predictor._features['image_embed'],model.sam_prompt_encoder.get_dense_pe()
                    low,iq,_,obj=d.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sp,dense_prompt_embeddings=dense,repeat_image=len(sp)>1,high_res_features=predictor._features['high_res_feats'])
                    full=predictor._transforms.postprocess_masks(low,predictor._orig_hw[0])>0
                    oq,ob=quality_all(full,truth,valid);choice=iq[:,1:].argmax(-1)+1;dynamic=dynamic_choice(d,low,iq)
                    if a.compiled_decoder_fp16:
                        hs=sp.half();hd=dense[:,:,0:1,0:1].half().expand_as(dense)
                        half_low,half_iq,_,_=compiled_call('native',cache,hs,hd)
                        phase_low,phase_iq,_,_=compiled_call('phase',cache,hs,hd)
                        half_full=predictor._transforms.postprocess_masks(half_low,predictor._orig_hw[0])>0
                        phase_full=predictor._transforms.postprocess_masks(phase_low,predictor._orig_hw[0])>0
                        hq,_=quality_all(half_full,truth,valid);eq,_=quality_all(phase_full,truth,valid)
                        hc=half_iq[:,1:].argmax(-1)+1;hdc=dynamic_choice(d16,half_low,half_iq)
                        ec=phase_iq[:,1:].argmax(-1)+1;edc=dynamic_choice(d16,phase_low,phase_iq)
                    for budget in (0,.05,.1):
                        if a.compiled_decoder_fp16:predicted,scores,tokens,objs=compiled_call(budget,cache,hs,hd)
                        else:predicted,scores,tokens,objs=decode(d,student,cache,image,pe,sp,dense,center,basis,budget)
                        assert predicted.shape==low.shape and scores.shape==iq.shape and tokens.shape==(len(sp),4,256) and objs.shape==obj.shape
                        masks=predictor._transforms.postprocess_masks(predicted,predictor._orig_hw[0])>0;pq,pb=quality_all(masks,truth,valid)
                        own=scores[:,1:].argmax(-1)+1;own_dynamic=dynamic_choice(d16 if a.compiled_decoder_fp16 else d,predicted,scores)
                        difference=masks!=full
                        if valid is not None:difference=difference&valid[None,None]
                        denominator=int(valid.sum()) if valid is not None else masks.shape[-2]*masks.shape[-1]
                        flips=difference.sum((-2,-1))/max(1,denominator)
                        for j,aid in enumerate(ids):
                            c,dc,cc,cd=int(choice[j]),int(dynamic[j]),int(own[j]),int(own_dynamic[j])
                            report['rows'].append({'image_id':info['image_id'],'sequence':info.get('sequence'),'annotation_id':aid,'regime':regime,'budget':budget,'original_head_iou':float(oq[j,c]),'delta_iou':float(pq[j,cc]-oq[j,c]),'delta_boundary_iou':float(pb[j,cc]-ob[j,c]),'native_dynamic_iou':float(oq[j,dc]),'delta_dynamic_iou':float(pq[j,cd]-oq[j,dc]),'head_choice_changed':c!=cc,'dynamic_choice_changed':dc!=cd,'valid_head_flip_fraction':float(flips[j,c]),'max_IQ_drift':float((scores-iq).abs().max()),'max_objscore_drift':float((objs-obj).abs().max())})
                            if a.compiled_decoder_fp16:
                                report['rows'][-1].update(native_fp16_head_delta_vs_fp32=float(hq[j,int(hc[j])]-oq[j,c]),native_fp16_dynamic_delta_vs_fp32=float(hq[j,int(hdc[j])]-oq[j,dc]),phase_fp16_head_delta_vs_fp32=float(eq[j,int(ec[j])]-oq[j,c]),phase_fp16_dynamic_delta_vs_fp32=float(eq[j,int(edc[j])]-oq[j,dc]),student_head_delta_vs_phase_fp16=float(pq[j,cc]-eq[j,int(ec[j])]),student_dynamic_delta_vs_phase_fp16=float(pq[j,cd]-eq[j,int(edc[j])]),max_IQ_difference_vs_phase_fp16=float((scores-phase_iq).abs().max()))
                if (n+1)%8==0:save();print(json.dumps({'images':n+1,'rows':len(report['rows'])}),flush=True)
        for regime in ('central','near_boundary','box'):
            report['summary'][regime]={}
            for budget in (0,.05,.1):
                rows=[r for r in report['rows'] if r['regime']==regime and r['budget']==budget]
                report['summary'][regime][str(budget)]={'mean_head_delta_iou':float(np.mean([r['delta_iou'] for r in rows])),'head_image_sequence_cluster95':bootstrap_image_groups(rows,'delta_iou'),'mean_native_dynamic_delta_iou':float(np.mean([r['delta_dynamic_iou'] for r in rows])),'dynamic_image_sequence_cluster95':bootstrap_image_groups(rows,'delta_dynamic_iou'),'head_choice_changes':sum(r['head_choice_changed'] for r in rows),'dynamic_choice_changes':sum(r['dynamic_choice_changed'] for r in rows)}
                if a.compiled_decoder_fp16:
                    for key in ('native_fp16_head_delta_vs_fp32','native_fp16_dynamic_delta_vs_fp32','phase_fp16_head_delta_vs_fp32','phase_fp16_dynamic_delta_vs_fp32','student_head_delta_vs_phase_fp16','student_dynamic_delta_vs_phase_fp16'):
                        report['summary'][regime][str(budget)][key]={'mean':float(np.mean([r[key] for r in rows])),'image_sequence_cluster95':bootstrap_image_groups(rows,key)}
        report['status']='COMPLETED_FROZEN_REAL_SAM2_HEAD';print(json.dumps(report['summary']),flush=True)
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('subset','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--compiled-decoder-fp16',action='store_true')
    p.add_argument('--old-development',action='store_true')
    p.add_argument('--student-path',type=Path,default=ROOT/'results/takeover_20261001_v1/sam2_local_student_v1/deployment.pt')
    p.add_argument('--minimum-free-mib',type=int,default=6500)
    p.add_argument('--scope',required=True);main(p.parse_args())
