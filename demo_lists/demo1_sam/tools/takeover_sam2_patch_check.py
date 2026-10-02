"""Modern SAM2 local-patch feasibility including both high-resolution skips."""
import json
from pathlib import Path
import subprocess
import time
import torch
import numpy as np
from PIL import Image
from takeover_sam2_quality import ROOT,build_sam2,SAM2ImagePredictor,encode,decode


def tiles(feature,scale):
    assert feature.shape[0]==1 and feature.shape[-2:]==(64*scale,64*scale)
    return feature[0].reshape(feature.shape[1],64,scale,64,scale).permute(1,3,0,2,4).reshape(4096,feature.shape[1],scale,scale)


def main():
    out=ROOT/'results/takeover_20261001_v1/local_student_head_v1/sam2_patch_check.json'
    if out.exists():raise FileExistsError(out)
    report={'status':'WAITING_VRAM','scope':'First3 previous24 development images, native SAM2.1-L. Test local head computation with actual stride8/stride4 shared skips, random64 parents per prompt and regime; FP32 and decoder-head-only FP64 diagnostics. Not a trained SAM2 surrogate, speed result, or old FP32 numeric pass.','rows':[]}
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<5000:time.sleep(10)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'),device='cuda:0').eval().requires_grad_(False)
    predictor=SAM2ImagePredictor(model);d=model.sam_mask_decoder;capture={}
    handle=d.transformer.register_forward_hook(lambda m,i,o:capture.update(tokens=o[0],state=o[1]))
    subset=ROOT/'assets/coco_quality_seed2027_v1';manifest=json.loads((subset/'manifest.json').read_text())
    generator=torch.Generator(device='cuda:0').manual_seed(2027)
    try:
        report['status']='RUNNING_NATIVE_HIGHRES_LOCALITY';save()
        with torch.inference_mode():
            for info in manifest['images'][:3]:
                rgb=np.asarray(Image.open(subset/info['image_file']).convert('RGB'));predictor.set_image(rgb)
                s0,s1=predictor._features['high_res_feats'];t0=tiles(s0,4);t1=tiles(s1,2)
                assert t0.shape[1]==32 and t1.shape[1]==64
                for regime in ('central','near_boundary','box'):
                    sparse,dense=encode(predictor,info['objects'],regime,rgb.shape[:2]);low,scores,_=decode(predictor,sparse,dense)
                    x=capture['state'];start=1 if d.pred_obj_scores else 0
                    tokens=capture['tokens'][:,start+1:start+5]
                    hyper=torch.stack([mlp(tokens[:,i]) for i,mlp in enumerate(d.output_hypernetworks_mlps)],1)
                    idx=torch.stack([torch.randperm(4096,device='cuda:0',generator=generator)[:64] for _ in range(len(sparse))]);batch=torch.arange(len(sparse),device='cuda:0')[:,None]
                    states=x[batch,idx].reshape(-1,256,1,1)
                    skip0=t0[idx].reshape(-1,32,4,4);skip1=t1[idx].reshape(-1,64,2,2)
                    dc1,ln,act1,dc2,act2=d.output_upscaling
                    patch=act2(dc2(act1(ln(dc1(states)+skip1)))+skip0)
                    repeated=hyper[:,None].expand(-1,64,-1,-1).reshape(-1,4,32)
                    calculated=torch.bmm(repeated,patch.flatten(2)).reshape(len(sparse),64,4,16)
                    reference=low.reshape(len(sparse),4,64,4,64,4).permute(0,2,4,1,3,5).reshape(len(sparse),4096,4,16)[batch,idx]
                    report['rows'].append({'image_id':info['image_id'],'regime':regime,'precision':'FP32','parents_per_prompt':64,'max_logit_error':float((calculated-reference).abs().max())})
                    # Cast only head/inputs, preserving actual native FP32 latent
                    # values exactly. Full-encoder double precision is not tested.
                    d.output_upscaling.double()
                    all_states=x.transpose(1,2).reshape(len(sparse),256,64,64).double()
                    full=act2(dc2(act1(ln(dc1(all_states)+s1.double())))+s0.double())
                    low64=torch.bmm(hyper.double(),full.flatten(2)).reshape(len(sparse),4,256,256)
                    patch64=act2(dc2(act1(ln(dc1(states.double())+skip1.double())))+skip0.double())
                    predicted64=torch.bmm(repeated.double(),patch64.flatten(2)).reshape(len(sparse),64,4,16)
                    ref64=low64.reshape(len(sparse),4,64,4,64,4).permute(0,2,4,1,3,5).reshape(len(sparse),4096,4,16)[batch,idx]
                    error=float((predicted64-ref64).abs().max());report['rows'].append({'image_id':info['image_id'],'regime':regime,'precision':'FP64 head-only','parents_per_prompt':64,'max_logit_error':error,'passes_fixed_1e10':error<=1e-10})
                    d.output_upscaling.float();save();print(json.dumps(report['rows'][-2:]),flush=True)
        report['status']='COMPLETED_NATIVE_HIGHRES_LOCALITY'
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:d.output_upscaling.float();handle.remove();save()


if __name__=='__main__':main()
