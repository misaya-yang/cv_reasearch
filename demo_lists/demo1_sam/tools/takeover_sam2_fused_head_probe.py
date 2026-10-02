"""Actual head-only fusion decision against compiled native and phase controls."""
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
from triton.runtime.errors import OutOfResources
from takeover_sam2_quality import ROOT,build_sam2,SAM2ImagePredictor
from takeover_sam2_patch_check import tiles
from takeover_compact_phase_fallback import build_patch_cache
from takeover_sam2_compact_head import dense_phase_head,dynamic_choice
from takeover_sam2_fused_head import fused_head
from benchmark import measure,profile_forward


def others():
    return [int(v) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split() if int(v)!=os.getpid()]


@torch.inference_mode()
def main(a):
    a.output.mkdir(exist_ok=False)
    r={'status':'WAITING_VRAM','pid':os.getpid(),'records':[],'numeric':[],
       'protocol':'New original-function execution mechanism, no student/oracle/GT or reduced spatial support. Actual frozenSAM2.1-L FP32 encoder outputs cast to native decoderFP16; original transformer/hyper once per genuine distinct grid workload P1/32/128, cache two skips and phase matrices as strong phase control. Triton local dot/LN/GELU/dot/skip/GELU/all4hyper contraction streams intermediate features inside one kernel. Two rounding variants are explicitly distinguished: eager manual FP16 operation-rounding vs FP32 LN intermediate, neither automatically bit-exact. IEEE input precision, no TF32 inference from Torch flags, FP fusion disabled. Compare native eager numeric function plus actual fullgraph compiled native and full-phase head controls. Head-only timing excludes transformer/encoder/promptencoder/cache/policy/postprocess; forward/reverse warmup5/reps20, exclusive if no competing CUDA context, otherwise defer without stopping others. Output all4 lowres masks; old5e-5 failures retained, no whole-model/equivalence or publication speed claim.'}
    def save():
        p=a.output/'report.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(a.output/'report.json')
    save();handle=None
    try:
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6500:time.sleep(5)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch._dynamo.config.recompile_limit=32
        model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
        pred=SAM2ImagePredictor(model);d=copy.deepcopy(model.sam_mask_decoder).half();capture={};handle=d.transformer.register_forward_hook(lambda m,i,o:capture.update(x=o[1]))
        info=sorted(json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text())['images'],key=lambda v:v['image_id'])[0]
        rgb=np.asarray(Image.open(ROOT/'assets/coco_quality_seed2027_v1'/info['image_file']).convert('RGB'));pred.set_image(rgb)
        hi=[v.half() for v in pred._features['high_res_feats']];im=pred._features['image_embed'].half();pe=model.sam_prompt_encoder.get_dense_pe().half()
        ca=SimpleNamespace(t0=tiles(hi[0],4),t1=tiles(hi[1],2),phase=build_patch_cache(d))
        def native(x,h):
            src=x.transpose(1,2).reshape(len(x),256,64,64);dc1,ln,act1,dc2,act2=d.output_upscaling
            feature=act2(dc2(act1(ln(dc1(src)+hi[1])))+hi[0])
            return torch.bmm(h,feature.flatten(2)).reshape(len(x),4,256,256)
        def phase(x,h):return dense_phase_head(d,ca,x,h)
        native_c=torch.compile(native,backend='inductor',fullgraph=True,dynamic=False);phase_c=torch.compile(phase,backend='inductor',fullgraph=True,dynamic=False)
        for prompts in (1,32,128):
            columns=int(np.ceil(np.sqrt(prompts)));rows=int(np.ceil(prompts/columns))
            if prompts==128:columns,rows=16,8
            points=torch.tensor([[(i+.5)*rgb.shape[1]/columns,(j+.5)*rgb.shape[0]/rows] for j in range(rows) for i in range(columns)][:prompts],device='cuda:0')[:,None]
            co=pred._transforms.transform_coords(points,normalize=True,orig_hw=rgb.shape[:2]);sp,de=model.sam_prompt_encoder(points=(co,torch.ones(prompts,1,device='cuda:0',dtype=torch.int)),boxes=None,masks=None)
            de=de[:1,:,0:1,0:1].half().expand(prompts,-1,64,64)
            reference,iq,tokens,obj=d.predict_masks(image_embeddings=im,image_pe=pe,sparse_prompt_embeddings=sp.half(),dense_prompt_embeddings=de,repeat_image=True,high_res_features=hi)
            x=capture['x'];hyper=torch.stack([mlp(tokens[:,j]) for j,mlp in enumerate(d.output_hypernetworks_mlps)],1)
            assert x.is_contiguous() and hyper.is_contiguous()
            functions={'native_compiled':lambda:native_c(x,hyper),'phase_compiled':lambda:phase_c(x,hyper)}
            for bn,warps in ((16,4),(16,8),(32,4)):
                for rounding in (True,False):
                    name='fused_n'+str(bn)+'_w'+str(warps)+'_'+('eagerround' if rounding else 'fp32ln')
                    functions[name]=lambda bn=bn,warps=warps,rounding=rounding:fused_head(d,ca,x,hyper,bn,warps,rounding)
            r.update(status='COMPILE_VALIDATE_NEW_FUSION',current_prompts=prompts);save()
            supported={}
            for name,fn in functions.items():
                r['current_arm']=name;save();t=time.monotonic()
                try:low=fn()
                except OutOfResources as e:
                    r.setdefault('unsupported_kernel_configurations',[]).append({'prompts':prompts,'arm':name,'error':repr(e)});save();continue
                torch.cuda.synchronize();assert low.shape==reference.shape and bool(torch.isfinite(low).all());supported[name]=fn
                r['numeric'].append({'prompts':prompts,'arm':name,'first_call_seconds':time.monotonic()-t,'max_logit_difference_vs_native_eager':float((low-reference).abs().max()),'lowres_flip_fraction':float(((low>0)!=(reference>0)).float().mean()),'native_dynamic_choices_changed':int((dynamic_choice(d,low,iq)!=dynamic_choice(d,reference,iq)).sum())});del low;save()
            if not any(name.startswith('fused_') for name in supported):raise RuntimeError('All fused configurations unsupported; preserve failures rather than measure only controls')
            functions=supported
            while others():r['status']='WAITING_EXCLUSIVE_FOR_HEAD_DIAGNOSTIC';save();time.sleep(5)
            r['status']='EXCLUSIVE_HEAD_ONLY_DIAGNOSTIC';save()
            for round_id,order in enumerate((tuple(functions),tuple(reversed(functions))),1):
                for name in order:
                    if others():raise RuntimeError('Competing CUDA context before head measurement')
                    for _ in range(5):functions[name]()
                    result=measure(functions[name],torch.device('cuda:0'),20)
                    r['records'].append({'prompts':prompts,'round':round_id,'arm':name,'timing':result});save();print(json.dumps({'prompts':prompts,'round':round_id,'arm':name,'median_ms':result['median_ms']}),flush=True)
                    if others():raise RuntimeError('Competing CUDA context during head measurement')
            if prompts==128:
                for name in ('phase_compiled','fused_n16_w4_eagerround'):
                    if name not in functions:continue
                    r.setdefault('profiles',{})[name]=profile_forward(functions[name],torch.device('cuda:0'),a.output/(name+'.trace.json'));save()
        r['status']='COMPLETED_EXCLUSIVE_ORIGINAL_HEAD_FUSION_DIAGNOSTIC'
    except BaseException as e:r.update(status='ERROR',error=repr(e));raise
    finally:
        if handle is not None:handle.remove()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
