"""Exclusive matched checkpoint comparison after independent task evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from takeover_sam2_quality import ROOT,build_sam2,SAM2ImagePredictor
from takeover_sam2_compact_head import load_student,static_features,decode
from takeover_compact_phase_fallback import build_patch_cache
from benchmark import measure,profile_forward


def others():
    return [int(v) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split() if int(v)!=os.getpid()]


def snapshot():
    return {'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader'],text=True).strip(),
            'compute_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()}


@torch.inference_mode()
def main(a):
    a.output.mkdir(exist_ok=False)
    r={'status':'WAITING_QUALITY_PREDECESSOR_CPU_ONLY','pid':os.getpid(),'records':[],
       'protocol':'All four frozen actual checkpoints: original v1, matched coefficientMSE, uniform-response, boundary-response; identical69760-parameter/rank64 structure and10% original local phase recovery, native/complete original phase strong controls. SAMEFP32 encoder/highres outputs cast to FP16, all4 masks/4IQ/masktokens/objectscore, actual fullgraph Inductor dynamicFalse. Exclusive six arms forward/reverse warmup10/reps30, P128 andP1024 distinct real grid prompts, commonMB128, all provided prompts decoded. Warm complete lowresolution decoder only: excludes encoder/promptencoder/static cache/model load/compile/native dynamic policy/fullres postprocess/CPU output. Not whole-encoderFP16 or end-to-end. Cold image tile/static projection setup independently synchronized, model-only phase matrices shared once. Actual checkpoint speed required, no reuse of original v1 latency as new evidence. Any other CUDA context before/after a block invalidates formal stage, preserve ERROR and samples without stopping other jobs.'}
    def save():
        p=a.output/'report.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(a.output/'report.json')
    save()
    try:
        while True:
            q=json.loads(a.predecessor.read_text())
            if q['status']=='ERROR':raise RuntimeError('Quality predecessor failed: '+str(q.get('error')))
            if q['status']=='COMPLETED':break
            time.sleep(5)
        while others():time.sleep(5)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch._dynamo.config.recompile_limit=32
        model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
        predictor=SAM2ImagePredictor(model);d=copy.deepcopy(model.sam_mask_decoder).half()
        info=sorted(json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text())['images'],key=lambda v:v['image_id'])[0]
        rgb=np.asarray(Image.open(ROOT/'assets/coco_quality_seed2027_v1'/info['image_file']).convert('RGB'));predictor.set_image(rgb)
        image=predictor._features['image_embed'].half();hi=[v.half() for v in predictor._features['high_res_feats']];pe=model.sam_prompt_encoder.get_dense_pe().half()
        phase=build_patch_cache(d);torch.cuda.synchronize();t=time.perf_counter();static,t0,t1=static_features(hi);torch.cuda.synchronize();r['image_tile_setup_ms']=(time.perf_counter()-t)*1000
        base=ROOT/'results/takeover_20261001_v1';paths={'original_v1':base/'sam2_local_student_v1/deployment.pt'}
        paths.update({k:base/'sam2_boundary_objective_v1'/k/'deployment.pt' for k in ('coefficient','response_uniform','response_boundary')})
        candidates={}
        for name,path in paths.items():
            s,center,basis=load_student(path,'cuda:0');s.requires_grad_(False);s=s.half();center=center.half();basis=basis.half();torch.cuda.synchronize();t=time.perf_counter()
            cache=SimpleNamespace(t0=t0,t1=t1,phase=phase,static_hidden=torch.nn.functional.linear(static,s[0].weight[:,256:],s[0].bias));torch.cuda.synchronize()
            r.setdefault('static_projection_ms',{})[name]=(time.perf_counter()-t)*1000
            candidates[name]=(s,center,basis,cache);r.setdefault('checkpoint_paths',{})[name]=str(path)
        s,c,b,ca=candidates['original_v1']
        def native(ss,dd):return d.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=ss,dense_prompt_embeddings=dd,repeat_image=True,high_res_features=hi)
        def complete(ss,dd):return decode(d,s,ca,image,pe,ss,dd,c,b,0,mode='dense_phase')
        def make_student(state):
            student,center,basis,cache=state
            def forward(ss,dd):return decode(d,student,cache,image,pe,ss,dd,center,basis,.1)
            return forward
        fns={'native_fp16':native,'phase_fp16':complete};fns.update({k:make_student(v) for k,v in candidates.items()})
        compiled={k:torch.compile(v,backend='inductor',fullgraph=True,dynamic=False) for k,v in fns.items()}
        for prompts in (128,1024):
            columns,rows=(16,8) if prompts==128 else (32,32)
            points=torch.tensor([[(i+.5)*rgb.shape[1]/columns,(j+.5)*rgb.shape[0]/rows] for j in range(rows) for i in range(columns)],device='cuda:0')[:,None]
            coords=predictor._transforms.transform_coords(points,normalize=True,orig_hw=rgb.shape[:2]);ss,dd=model.sam_prompt_encoder(points=(coords,torch.ones(prompts,1,device='cuda:0',dtype=torch.int)),boxes=None,masks=None)
            assert len(torch.unique(ss,dim=0))==prompts
            sparse=[v.half() for v in ss.split(128)];dense_base=dd[:1,:,0:1,0:1].half();dense=[dense_base.expand(128,-1,64,64) for v in sparse]
            r.update(status='COMPILE_AND_VALIDATE_INTERFACES',current_prompts=prompts,image_id=info['image_id']);save()
            for name in compiled:
                r['current_arm']=name;save();t=time.monotonic();value=compiled[name](sparse[0],dense[0]);torch.cuda.synchronize()
                assert [tuple(v.shape) for v in value]==[(128,4,256,256),(128,4),(128,4,256),(128,1)]
                assert all(bool(torch.isfinite(v).all()) for v in value)
                r.setdefault('first_shape_call_seconds',[]).append({'prompts':prompts,'arm':name,'seconds':time.monotonic()-t});del value;save()
            def run(name):
                for sp,de in zip(sparse,dense):value=compiled[name](sp,de);del value
            while others():r['status']='WAITING_EXCLUSIVE_GPU';save();time.sleep(5)
            r['status']='EXCLUSIVE_COMPLETE_DECODER_TIMING';save()
            for round_id,order in enumerate((tuple(compiled),tuple(reversed(compiled))),1):
                for name in order:
                    if others():raise RuntimeError('Competing CUDA process before formal block')
                    before=snapshot()
                    for _ in range(10):run(name)
                    result=measure(lambda:run(name),torch.device('cuda:0'),30)
                    r['records'].append({'prompts':prompts,'microbatch':128,'round':round_id,'arm':name,'timing':result,'before':before,'after':snapshot()});save();print(json.dumps({'prompts':prompts,'round':round_id,'arm':name,'median_ms':result['median_ms']}),flush=True)
                    if others():raise RuntimeError('Competing CUDA process during formal block; samples invalid')
            if prompts==128:
                r['status']='UNTIMED_KERNEL_PROFILE';save()
                for name in ('native_fp16','phase_fp16','response_boundary'):
                    r.setdefault('profiles',{})[name]=profile_forward(lambda:compiled[name](sparse[0],dense[0]),torch.device('cuda:0'),a.output/(name+'.trace.json'));save()
        r['status']='COMPLETED_EXCLUSIVE_MATCHED_CHECKPOINT_DECODER_TIMING'
    except BaseException as e:r.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--predecessor',type=Path,required=True);main(p.parse_args())
