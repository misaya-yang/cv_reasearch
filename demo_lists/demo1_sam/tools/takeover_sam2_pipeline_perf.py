"""Exclusive cold-image and cached-image SAM2 deployment cost comparison."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from takeover_sam2_quality import ROOT, build_sam2, SAM2ImagePredictor
from takeover_sam2_compact_head import load_student, static_features, decode, dynamic_choice
from takeover_sam2_patch_check import tiles
from takeover_compact_phase_fallback import build_patch_cache


def others():
    return [int(v) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split() if int(v)!=os.getpid()]


@torch.inference_mode()
def main(a):
    a.output.mkdir(exist_ok=False)
    report={'status':'WAITING_PREDECESSOR_CPU_ONLY','pid':os.getpid(),'records':[],
            'protocol':'SAM2.1-L, shared actual fullgraph Inductor FP32 image encoder, decoder-only FP16 native/full original-function phase/student10. Two old development images; real distinct grid point queries; MB=min(P,128). Native dynamic token0 policy unchanged. Cold-image latency includes native host RGB transform/upload, encoder, prompt encoding, feature casts, each-image cache, full decoder, dynamic selection, original-resolution interpolation/threshold and binary masks plus selected IQ download to CPU. Cached-image requests include prompt encoding/decoder/policy/postprocess/download, exclude image encoding and cache. RGB disk read/model load/compiler/weight conversion excluded from both; cold means new image with already loaded/compiled model. All four lowres masks/IQ/tokens/object-score computed, one dynamic selected fullres mask returned. No hole/sprinkle filling (native predictor defaults). Same output boundary/microbatch for all arms. No GT or model fitting. Exclusive forward/reverse rounds, warmup3/reps10, preserve error if any competing CUDA process appears. Fixed phase weights are model-only cached once for both optimized arms; full phase baseline does not pay unused student static projection.'}
    def save():
        p=a.output/'report.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(a.output/'report.json')
    save()
    try:
        while True:
            prior=json.loads(a.predecessor.read_text())
            if prior['status']=='ERROR':raise RuntimeError('Predecessor failed: '+str(prior.get('error')))
            if prior['status']=='COMPLETED':break
            time.sleep(5)
        while others():time.sleep(5)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
        student,center,basis=load_student(ROOT/'results/takeover_20261001_v1/sam2_local_student_v1/deployment.pt','cuda:0')
        student.requires_grad_(False);student=student.half();center=center.half();basis=basis.half()
        decoder=copy.deepcopy(model.sam_mask_decoder).half();phase_weights=build_patch_cache(decoder)
        report['status']='COMPILING_SHARED_ENCODER';save()
        model.forward_image=torch.compile(model.forward_image,backend='inductor',fullgraph=True,dynamic=False)
        predictor=SAM2ImagePredictor(model)
        pe=model.sam_prompt_encoder.get_dense_pe().half()
        def native(im,ss,dd,h0,h1,cache):
            return decoder.predict_masks(image_embeddings=im,image_pe=pe,sparse_prompt_embeddings=ss,dense_prompt_embeddings=dd,repeat_image=True,high_res_features=[h0,h1])
        def phase(im,ss,dd,h0,h1,cache):
            return decode(decoder,student,cache,im,pe,ss,dd,center,basis,0,mode='dense_phase')
        def compact(im,ss,dd,h0,h1,cache):
            return decode(decoder,student,cache,im,pe,ss,dd,center,basis,.1)
        functions={k:torch.compile(v,backend='inductor',fullgraph=True,dynamic=False) for k,v in [('native_fp16',native),('phase_fp16',phase),('student10_fp16',compact)]}
        infos=sorted(json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text())['images'],key=lambda x:x['image_id'])[:2]
        for info in infos:
            rgb=np.asarray(Image.open(ROOT/'assets/coco_quality_seed2027_v1'/info['image_file']).convert('RGB'))
            size=rgb.shape[:2]
            def image_state(arm):
                predictor.set_image(rgb)
                im=predictor._features['image_embed'].half();hi=[v.half() for v in predictor._features['high_res_feats']]
                if arm=='native_fp16':cache=None
                else:
                    if arm=='student10_fp16':
                        static,t0,t1=static_features(hi)
                        cache=SimpleNamespace(t0=t0,t1=t1,phase=phase_weights,static_hidden=torch.nn.functional.linear(static,student[0].weight[:,256:],student[0].bias))
                    else:cache=SimpleNamespace(t0=tiles(hi[0],4),t1=tiles(hi[1],2),phase=phase_weights)
                return im,hi,cache
            # Synchronized isolated cache cost, free from preceding model-copy/cast work.
            im,hi,_=image_state('native_fp16');torch.cuda.synchronize()
            for arm in ('phase_fp16','student10_fp16'):
                samples=[]
                for _ in range(12):
                    torch.cuda.synchronize();t=time.perf_counter()
                    if arm=='student10_fp16':
                        static,t0,t1=static_features(hi);hidden=torch.nn.functional.linear(static,student[0].weight[:,256:],student[0].bias)
                    else:t0,t1=tiles(hi[0],4),tiles(hi[1],2)
                    torch.cuda.synchronize();samples.append((time.perf_counter()-t)*1000)
                report.setdefault('isolated_image_cache_ms',[]).append({'image_id':info['image_id'],'arm':arm,'samples_after_warmup':samples[2:],'median_ms':statistics.median(samples[2:]),'phase_weight_cache':'model-only, excluded equally'})
            save()
            for prompts in (1,32,128,1024):
                mb=min(prompts,128);columns=math.ceil(math.sqrt(prompts));rows=math.ceil(prompts/columns)
                points=torch.tensor([[(i+.5)*size[1]/columns,(j+.5)*size[0]/rows] for j in range(rows) for i in range(columns)][:prompts],device='cuda:0')[:,None]
                labels=torch.ones(prompts,1,device='cuda:0',dtype=torch.int)
                def request(arm,state):
                    im,hi,cache=state
                    coords=predictor._transforms.transform_coords(points,normalize=True,orig_hw=size)
                    sparse,dense=model.sam_prompt_encoder(points=(coords,labels),boxes=None,masks=None)
                    dense_base=dense[:1,:,0:1,0:1].half()
                    payload=[]
                    for ss in sparse.split(mb):
                        dd=dense_base.expand(len(ss),-1,64,64)
                        masks,iq,tokens,obj=functions[arm](im,ss.half(),dd,hi[0],hi[1],cache)
                        choice=dynamic_choice(decoder,masks,iq);ix=torch.arange(len(ss),device=im.device)
                        selected=masks[ix,choice][:,None]
                        full=predictor._transforms.postprocess_masks(selected,size)>predictor.mask_threshold
                        payload.append((full.cpu().numpy(),iq[ix,choice].cpu().numpy()))
                    return payload
                report.update(status='COMPILING_DECODERS',current_image=info['image_id'],current_prompts=prompts);save()
                states={arm:image_state(arm) for arm in functions}
                for arm in functions:
                    t=time.monotonic();value=request(arm,states[arm]);torch.cuda.synchronize()
                    assert sum(len(v[0]) for v in value)==prompts
                    report.setdefault('compile_first_request_seconds',[]).append({'image_id':info['image_id'],'prompts':prompts,'arm':arm,'seconds':time.monotonic()-t});del value;save()
                for boundary in ('cached_image_request','cold_image_pipeline'):
                    for round_id,order in enumerate((tuple(functions),tuple(reversed(functions))),1):
                        for arm in order:
                            if others():raise RuntimeError('Competing CUDA process before timing')
                            def run():return request(arm,image_state(arm) if boundary=='cold_image_pipeline' else states[arm])
                            report.update(status='EXCLUSIVE_PIPELINE_TIMING',current_arm=arm,current_boundary=boundary);save()
                            for _ in range(3):value=run();del value
                            samples=[];torch.cuda.reset_peak_memory_stats()
                            for _ in range(10):
                                torch.cuda.synchronize();t=time.perf_counter();value=run();torch.cuda.synchronize()
                                samples.append((time.perf_counter()-t)*1000);del value
                            report['records'].append({'image_id':info['image_id'],'original_hw':list(size),'prompts':prompts,'microbatch':mb,'boundary':boundary,'round':round_id,'arm':arm,'milliseconds':samples,'median_ms':statistics.median(samples),'peak_allocated_bytes':torch.cuda.max_memory_allocated()});save()
                            print(json.dumps(report['records'][-1]),flush=True)
                            if others():raise RuntimeError('Competing CUDA process during timing; samples invalid')
                del states
        report['status']='COMPLETED_EXCLUSIVE_PIPELINE_TIMING'
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--predecessor',type=Path,required=True);main(p.parse_args())
