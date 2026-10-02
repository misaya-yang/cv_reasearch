"""Actual SAM2 decoder screening with native/phase controls and decoder FP16."""
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
from takeover_sam2_quality import ROOT,build_sam2,SAM2ImagePredictor
from takeover_sam2_compact_head import load_student,build_cache,decode
from benchmark import measure,profile_forward


def snapshot():
    return {'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.free,power.draw,clocks.sm','--format=csv,noheader'],text=True).strip(),
            'compute_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()}


def other_pids():
    return [int(x) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split() if int(x)!=os.getpid()]


def main(a):
    a.output.mkdir(exist_ok=False)
    report={'status':'WAITING_VRAM','pid':os.getpid(),'microbatch':a.microbatch,'prompts':a.prompts,'records':[],
            'protocol':'Shared-GPU SAM2.1-L complete lowresolution decoder diagnostic,128 distinct actual original-image grid point prompts, common MB; native original SDPA/deconvolution and full exact-function phase head controls vs real static-skip student5/10%. All4masks/4IQ/masktokens/objscore retained. True Inductor fullgraph/dynamicFalse, no compiler fallback. FP32 TF32off; separate decoder-FP16 precision arms ALL use the SAME FP32 encoder/highres features cast toFP16, not an FP16 full encoder or end-to-end claim. Warm timing includes entire decoder across all128 prompts, excludes encoder/promptencoder/image-static cache/model load/postprocess. Seven cyclic orders plus reversed orders, reps10 perblock. Contention means no exclusive latency/publication speed claim; numerical drift is diagnostic, not old5e-5 equivalence pass. Cold static cache setup measured separately and cannot be free in end-to-end claims.'}
    if a.exclusive:
        report['protocol']='FORMAL EXCLUSIVE TIMING OVERRIDE: all other CUDA contexts must exit, check before/after each arm; seven arms forward/reverse rounds, warmup10/reps30. '+report['protocol'].replace('Shared-GPU','').replace('Seven cyclic orders plus reversed orders, reps10 perblock. Contention means no exclusive latency/publication speed claim;','')
    if a.prompts!=128:report['protocol']=report['protocol'].replace('128 distinct',str(a.prompts)+' distinct').replace('all128 prompts','all'+str(a.prompts)+' prompts')
    def save():
        tmp=a.output/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output/'report.json')
    save();telemetry=None;log=None
    try:
        assert a.prompts>=1 and a.prompts%a.microbatch==0
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6000:time.sleep(10)
        log=(a.output/'gpu_telemetry.csv').open('x');telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv','--loop-ms=1000'],stdout=log,stderr=subprocess.STDOUT)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
        predictor=SAM2ImagePredictor(model);d32=model.sam_mask_decoder
        student,center,basis=load_student(ROOT/'results/takeover_20261001_v1/sam2_local_student_v1/deployment.pt','cuda:0');student.requires_grad_(False)
        info=sorted(json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text())['images'],key=lambda x:x['image_id'])[0]
        rgb=np.asarray(Image.open(ROOT/'assets/coco_quality_seed2027_v1'/info['image_file']).convert('RGB'))
        with torch.inference_mode():
            predictor.set_image(rgb);image=predictor._features['image_embed'];hi=predictor._features['high_res_feats'];pe=model.sam_prompt_encoder.get_dense_pe()
            h,w=rgb.shape[:2]
            if a.prompts==128:columns,rows=16,8
            else:
                import math
                columns=math.ceil(math.sqrt(a.prompts));rows=math.ceil(a.prompts/columns)
            points=torch.tensor([[(i+.5)*w/columns,(j+.5)*h/rows] for j in range(rows) for i in range(columns)][:a.prompts],device='cuda:0')[:,None]
            transformed=predictor._transforms.transform_coords(points,normalize=True,orig_hw=rgb.shape[:2])
            sparse,dense=model.sam_prompt_encoder(points=(transformed,torch.ones(a.prompts,1,device='cuda:0',dtype=torch.int)),boxes=None,masks=None)
            assert len(torch.unique(sparse,dim=0))==a.prompts
            report.update(image_id=info['image_id'],sparse_shape=list(sparse.shape))
            start=time.monotonic();cache32=build_cache(d32,hi,student);torch.cuda.synchronize();report['static_cache_fp32_seconds']=time.monotonic()-start
            d16=copy.deepcopy(d32).half();s16=copy.deepcopy(student).half();i16,p16=image.half(),pe.half();hi16=[x.half() for x in hi];c16,b16=center.half(),basis.half()
            start=time.monotonic();cache16=build_cache(d16,hi16,s16);torch.cuda.synchronize();report['static_cache_fp16_seconds']=time.monotonic()-start
            ss32=list(sparse.split(a.microbatch));dd32=list(dense.split(a.microbatch));ss16=[s.half() for s in ss32]
            # No mask prompt: preserve the encoder's broadcast embedding instead
            # of materializing128 copies merely to change its dtype.
            assert dense.stride(-1)==dense.stride(-2)==0
            dense16_base=dense[:1,:,0:1,0:1].half()
            dd16=[dense16_base.expand(len(s),-1,64,64) for s in ss16]
            def native32(ss,dd):return d32.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=ss,dense_prompt_embeddings=dd,repeat_image=True,high_res_features=hi)
            def phase32(ss,dd):return decode(d32,student,cache32,image,pe,ss,dd,center,basis,0,mode='dense_phase')
            def student5(ss,dd):return decode(d32,student,cache32,image,pe,ss,dd,center,basis,.05)
            def student10(ss,dd):return decode(d32,student,cache32,image,pe,ss,dd,center,basis,.1)
            def native16(ss,dd):return d16.predict_masks(image_embeddings=i16,image_pe=p16,sparse_prompt_embeddings=ss,dense_prompt_embeddings=dd,repeat_image=True,high_res_features=hi16)
            def phase16(ss,dd):return decode(d16,s16,cache16,i16,p16,ss,dd,c16,b16,0,mode='dense_phase')
            def student16(ss,dd):return decode(d16,s16,cache16,i16,p16,ss,dd,c16,b16,.1)
            functions=[('native_fp32',native32),('phase_exact_fp32',phase32),('student5_fp32',student5),('student10_fp32',student10),('native_fp16',native16),('phase_exact_fp16',phase16),('student10_fp16',student16)]
            compiled={};report['status']='COMPILING';save()
            def inputs(name):return (ss16,dd16) if name.endswith('fp16') else (ss32,dd32)
            for name,fn in functions:
                report['current_arm']=name;save();start=time.monotonic()
                compiled[name]=torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False)
                ss,dd=inputs(name);value=compiled[name](ss[0],dd[0]);torch.cuda.synchronize()
                assert [tuple(v.shape) for v in value]==[(a.microbatch,4,256,256),(a.microbatch,4),(a.microbatch,4,256),(a.microbatch,1)]
                assert all(bool(torch.isfinite(v).all()) for v in value), name+' produced nonfinite outputs'
                report.setdefault('output_dtypes',{})[name]=[str(v.dtype) for v in value]
                del value;report.setdefault('compile_seconds',{})[name]=time.monotonic()-start;save()
            reference=compiled['native_fp32'](ss32[0],dd32[0])
            for name in compiled:
                ss,dd=inputs(name);value=compiled[name](ss[0],dd[0])
                report.setdefault('first_microbatch_numeric_diagnostics',{})[name]={'max_logit_difference_vs_native_fp32':float((value[0].float()-reference[0]).abs().max()),'lowres_flip_fraction':float(((value[0]>0)!=(reference[0]>0)).float().mean()),'max_IQ_difference':float((value[1].float()-reference[1]).abs().max()),'max_masktoken_difference':float((value[2].float()-reference[2]).abs().max()),'max_objscore_difference':float((value[3].float()-reference[3]).abs().max())}
                del value
            del reference;torch.cuda.empty_cache();save()
            def full_run(name):
                ss,dd=inputs(name)
                for s,v in zip(ss,dd):
                    result=compiled[name](s,v)
                    del result
            names=tuple(compiled)
            if a.exclusive:
                report['status']='WAITING_EXCLUSIVE_GPU';save()
                while other_pids():time.sleep(5)
                orders=[names,tuple(reversed(names))];report['status']='EXCLUSIVE_TIMING'
            else:
                orders=[names[i:]+names[:i] for i in range(len(names))];orders+=[tuple(reversed(order)) for order in orders];report['status']='SHARED_GPU_DIAGNOSTIC_TIMING'
            save()
            for r,order in enumerate(orders,1):
                for name in order:
                    before=snapshot()
                    if a.exclusive and other_pids():raise RuntimeError('Other CUDA process appeared; formal timing invalid')
                    for _ in range(10 if a.exclusive else 3):full_run(name)
                    timing=measure(lambda:full_run(name),torch.device('cuda:0'),30 if a.exclusive else 10)
                    report['records'].append({'round':r,'arm':name,'timing':timing,'before':before,'after':snapshot()});save();print(json.dumps({'round':r,'arm':name,'median_ms':timing['median_ms']}),flush=True)
                    if a.exclusive and other_pids():raise RuntimeError('Other CUDA process appeared during measurement; formal samples invalid')
            report['status']='UNTIMED_KERNEL_PROFILE';save()
            for name in ('native_fp32','phase_exact_fp32','student10_fp32','native_fp16','student10_fp16'):
                ss,dd=inputs(name);report.setdefault('profiles',{})[name]=profile_forward(lambda:compiled[name](ss[0],dd[0]),torch.device('cuda:0'),a.output/(name+'.trace.json'));save()
        report['status']='COMPLETED_EXCLUSIVE_SAM2_DECODER_TIMING' if a.exclusive else 'COMPLETED_SHARED_SAM2_DECODER_DIAGNOSTIC_NOT_EXCLUSIVE'
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:
        if telemetry is not None:telemetry.terminate();telemetry.wait(timeout=5)
        if log is not None:log.close()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--microbatch',type=int,default=64);p.add_argument('--prompts',type=int,default=128);p.add_argument('--exclusive',action='store_true');main(p.parse_args())
