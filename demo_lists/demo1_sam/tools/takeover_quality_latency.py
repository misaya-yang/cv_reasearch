"""Repeated inference latency including mask export and GT-free CPU matching.

Prepared image tensors exclude JPEG read, resize and host-to-device transfer.
One encoder per cold image, or cached original embedding. Three fixed images;
FP32 eager official decoder, no claim versus compiled deployment baselines.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from takeover_virtual_flip import decode_rows, encode_prompt_rows, mirror_objects, virtual_feature_flip
from takeover_prompt_stability import perturb_objects, select, OFFSETS
from takeover_flip_score import match_and_select
from takeover_gpu_consistency import gpu_consistency
from takeover_virtual_flip import Sam
from types import SimpleNamespace

ARMS = ("original_head", "perturb_one", "perturb_two", "perturb_four", "perturb_four_gpu", "virtual_flip", "RGB_flip")


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    while subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip():time.sleep(15)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    report={"status":"RUNNING","scope":"three fixed development images, original central prompts; FP32 official eager complete decoder; CPU matching or GPU-equivalent full-resolution matching; all four raw original masks exported, all auxiliary masks computed; JPEG/resize/H2D/model load excluded; not a fully optimized compiled deployment benchmark",
            "warmup":2,"repetitions":5,"rounds":2,"records":[]}
    def save():args.output.write_text(json.dumps(report,indent=2)+"\n")
    save()
    model=sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to("cuda:0").eval().requires_grad_(False)
    manifest=json.loads((args.subset/"manifest.json").read_text())
    transform=ResizeLongestSide(1024)
    with torch.inference_mode():
        pe=model.prompt_encoder.get_dense_pe()
        for info in manifest["images"][:3]:
            rgb=np.asarray(Image.open(args.subset/info["image_file"]).convert("RGB"))
            orig_size=tuple(rgb.shape[:2]);input_size=transform.apply_image(rgb).shape[:2]
            tensors=[model.preprocess(torch.as_tensor(transform.apply_image(x),device="cuda:0").permute(2,0,1).contiguous()[None]) for x in (rgb,rgb[:,::-1].copy())]
            cached=model.image_encoder(tensors[0]);objects=info["objects"]
            def inference(arm,cold):
                feature=model.image_encoder(tensors[0]) if cold else cached
                sparse,dense=encode_prompt_rows(model.prompt_encoder,objects,"central",orig_size,1024,"cuda:0")
                if arm=="perturb_four_gpu":
                    low,scores=model.mask_decoder.predict_masks(image_embeddings=feature,image_pe=pe,sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense)
                    proxy=SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024))
                    full=Sam.postprocess_masks(proxy,low,input_size,orig_size)>0
                    parts=[encode_prompt_rows(model.prompt_encoder,perturb_objects(objects,"central",orig_size,input_size,o),"central",orig_size,1024,"cuda:0")[0] for o in OFFSETS]
                    ps=torch.cat(parts)
                    vl,_=model.mask_decoder.predict_masks(image_embeddings=feature,image_pe=pe,sparse_prompt_embeddings=ps,dense_prompt_embeddings=dense[:1].expand(len(ps),-1,-1,-1))
                    views=(Sam.postprocess_masks(proxy,vl,input_size,orig_size)>0).reshape(4,len(objects),4,*orig_size)
                    chosen,_=gpu_consistency(full,scores,views)
                    # Retain/export the same complete original outputs. Auxiliary
                    # views are computed fully but need only remain on the device.
                    full.cpu().numpy();scores.cpu().numpy();low.cpu().numpy()
                    return chosen.cpu().numpy()
                original=decode_rows(model.mask_decoder,feature,pe,dense[:1],sparse,input_size,orig_size,0,len(objects))
                if arm=="original_head":return original["iou_prediction"][:,1:].argmax(-1)+1
                if arm.startswith("perturb"):
                    count={"perturb_one":1,"perturb_two":2,"perturb_four":4}[arm]
                    parts=[encode_prompt_rows(model.prompt_encoder,perturb_objects(objects,"central",orig_size,input_size,o),"central",orig_size,1024,"cuda:0")[0] for o in OFFSETS[:count]]
                    ps=torch.cat(parts)
                    view=decode_rows(model.mask_decoder,feature,pe,dense[:1],ps,input_size,orig_size,0,len(ps))
                    masks=view["masks"].reshape(count,len(objects),4,*orig_size)
                    scores=view["iou_prediction"].reshape(count,len(objects),4)
                    return [select(original["masks"][j],original["iou_prediction"][j],original["low_resolution_logits"][j],masks[:,j],scores[:,j])[0]["perturb_consistency"] for j in range(len(objects))]
                vf=virtual_feature_flip(feature,input_size)[0] if arm=="virtual_flip" else model.image_encoder(tensors[1])
                ps,vd=encode_prompt_rows(model.prompt_encoder,mirror_objects(objects,orig_size[1]),"central",orig_size,1024,"cuda:0")
                view=decode_rows(model.mask_decoder,vf,pe,vd[:1],ps,input_size,orig_size,0,len(objects))
                return [match_and_select(original["masks"][j],original["iou_prediction"][j],view["masks"][j,:,:,::-1],view["iou_prediction"][j])[0]["matched_consistency_only"] for j in range(len(objects))]
            for cold in (False,True):
                for rnd in (1,2):
                    for arm in (ARMS if rnd==1 else tuple(reversed(ARMS))):
                        others={int(x) for x in subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).split()}-{os.getpid()}
                        if others:raise RuntimeError("other GPU processes appeared: "+str(others))
                        for _ in range(2):inference(arm,cold)
                        samples=[]
                        for _ in range(5):
                            torch.cuda.synchronize();start=time.perf_counter();inference(arm,cold);torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
                        row={"image_id":info["image_id"],"prompts":len(objects),"arm":arm,"prepared_image_with_encoder":cold,"round":rnd,"samples_ms":samples,"median_ms":float(np.median(samples))}
                        report["records"].append(row);save();print(json.dumps(row),flush=True)
    report["status"]="COMPLETED";save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--subset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
