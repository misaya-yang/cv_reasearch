"""Standard SAM low-resolution mask feedback: frozen model, no GT input.

This is a refinement baseline and diagnosis, not a new mechanism. Original points
or boxes are preserved. Feedback is seeded by original head or largest mask.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from takeover_virtual_flip import load_models, Sam
from takeover_flip_score import bootstrap_image_groups, iou, inner_boundary

POLICIES = ("original_iou_head", "largest_original_mask", "head_feedback_token0",
            "head_feedback_multimask_head", "largest_feedback_token0", "largest_feedback_multimask_head")


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    source=json.loads((args.source_run/"report.json").read_text())
    decoder,encoder,_=load_models(args.source_run/"mask_decoder_state.pt",args.checkpoint,"cuda:0")
    proxy=SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024))
    report={"status":"RUNNING","cohort":args.cohort,
            "protocol":"standard official SAM low-resolution mask-logit feedback, original sparse point or box retained; two separate GT-free initialization policies (original IoU head or largest full-resolution original mask); complete four masks and scores per feedback prompt, token0 or new multimask head evaluated separately",
            "new_image_encoder_calls":0,"extra_decoder_calls_per_initialization":1,
            "GT_input":False,"novelty":"baseline mechanism, not claimed new","rows":[],"summary":{}}
    def save():
        tmp=args.output.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2)+"\n");tmp.replace(args.output)
    save()
    with torch.inference_mode():
        for info in source["images"]:
            with np.load(args.source_run/info["encoded_inputs"]["npz"],allow_pickle=False) as encoded:
                image=torch.from_numpy(encoded["image_embeddings"]).cuda()
                pe=torch.from_numpy(encoded["image_pe"]).cuda()
                original_size=tuple(int(x) for x in encoded["original_size"])
                input_size=tuple(int(x) for x in encoded["input_size"])
                sparse_by_regime={r:torch.from_numpy(encoded["sparse_"+r]).cuda() for r in ("central","near_boundary","box")}
            for regime in ("central","near_boundary","box"):
                rec=next(r for r in source["records"] if r["image_id"]==info["image_id"] and r["regime"]==regime and r["method"]=="official")
                with np.load(args.source_run/rec["npz"],allow_pickle=False) as base:
                    original_masks=base["masks"]
                    p=len(original_masks);rows=np.arange(p)
                    head=base["iou_prediction"][:,1:].argmax(-1)+1
                    largest=original_masks[:,1:].sum(axis=(-2,-1)).argmax(-1)+1
                    feedback=np.concatenate([base["low_resolution_logits"][rows,head,None],base["low_resolution_logits"][rows,largest,None]])
                    # Dense embedding differs by prompt; never use the no-mask cache.
                    dense=encoder._embed_masks(torch.from_numpy(feedback).cuda())
                    sparse=torch.cat([sparse_by_regime[regime]]*2)
                    low,scores=decoder.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense)
                    if low.shape[:2]!=(2*p,4):raise ValueError("complete four masks required")
                    full=(Sam.postprocess_masks(proxy,low,input_size,original_size)>0).cpu().numpy().reshape(2,p,4,*original_size)
                    scores=scores.cpu().numpy().reshape(2,p,4)
                    for j in range(p):
                        candidates={"original_iou_head":original_masks[j,head[j]],"largest_original_mask":original_masks[j,largest[j]],
                                    "head_feedback_token0":full[0,j,0],"head_feedback_multimask_head":full[0,j,int(scores[0,j,1:].argmax())+1],
                                    "largest_feedback_token0":full[1,j,0],"largest_feedback_multimask_head":full[1,j,int(scores[1,j,1:].argmax())+1]}
                        gt=base["gt"][j];radius=max(1,int(round(.02*np.hypot(*original_size))));gb=inner_boundary(gt,radius)
                        values={k:iou(v,gt) for k,v in candidates.items()};bounds={k:iou(inner_boundary(v,radius),gb) for k,v in candidates.items()}
                        row={"image_id":info["image_id"],"annotation_id":int(base["annotation_ids"][j]),"regime":regime}
                        for name in POLICIES:
                            row[name+"_iou"]=values[name];row[name+"_boundary_iou"]=bounds[name]
                            row[name+"_delta_iou"]=values[name]-values["original_iou_head"]
                            row[name+"_delta_boundary_iou"]=bounds[name]-bounds["original_iou_head"]
                            row[name+"_delta_vs_largest"]=values[name]-values["largest_original_mask"]
                        report["rows"].append(row)
            save();print(json.dumps({"completed_images":len({r['image_id'] for r in report['rows']}),"image_id":info["image_id"]}),flush=True)
    for regime in ("central","near_boundary","box"):
        rows=[r for r in report["rows"] if r["regime"]==regime]
        report["summary"][regime]={"prompts":len(rows),"policies":{}}
        for name in POLICIES:
            report["summary"][regime]["policies"][name]={"mean_iou":float(np.mean([r[name+"_iou"] for r in rows])),
                "mean_boundary_iou":float(np.mean([r[name+"_boundary_iou"] for r in rows])),
                "mean_delta_iou":float(np.mean([r[name+"_delta_iou"] for r in rows])),
                "image_cluster_delta_iou_95":bootstrap_image_groups(rows,name+"_delta_iou"),
                "mean_delta_vs_largest":float(np.mean([r[name+"_delta_vs_largest"] for r in rows])),
                "image_cluster_delta_vs_largest_95":bootstrap_image_groups(rows,name+"_delta_vs_largest"),
                "image_cluster_boundary_delta_95":bootstrap_image_groups(rows,name+"_delta_boundary_iou")}
    report["status"]="COMPLETED_EXPLORATORY_BASELINE";save();print(json.dumps(report["summary"],indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cohort",required=True)
    run(p.parse_args())
