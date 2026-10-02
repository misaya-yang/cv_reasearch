"""Strong causal video control from DINOv3's nonparametric VOS protocol.

Frozen features, first-frame labels, seven previous predictions, top5/.2.
Square radius12, with local versus global first-frame anchor controls.
Fixed development settings; not a claim to exactly reproduce author-selected settings.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import time
import traceback

from probe import Features, DATA, mask_weights, save, summarize
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def run(args):
    prior=Path(args.prior)
    manifest=json.loads((prior/"manifest.json").read_text())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    status=dict(state="STARTING",pid=os.getpid(),input_short_side=args.size,
                downloads=0,offline_training_updates=0,query_gt_role="score only",
                protocol="first frame + seven previous soft labels; top5; temperature .2; square radius12",
                precision="FP32 TF32off",split="DAVIS2017 train/development",
                first_anchor_controls=["local", "global"],author_exact_best_claim=False)
    save(out/"status.json",status);save(out/"manifest.json",manifest)
    try:
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        feats=Features(shortest=True)
        records=[];start=time.monotonic();total_frames=0
        for ns,s in enumerate(manifest):
            seq=s["sequence"];ip=DATA/"JPEGImages/480p"/seq;gp=DATA/"Annotations/480p"/seq
            paths=sorted(ip.glob("*.jpg"));first=Image.open(paths[0]).convert("RGB")
            gt0=np.asarray(Image.open(gp/(paths[0].stem+".png")))
            m0=feats(first,args.size)[-1];hw=m0.shape[-2:];N=hw[0]*hw[1]
            x0=F.normalize(m0.flatten(1).T,dim=1)
            fg=[mask_weights(gt0==oid,hw) for oid in s["objects"]]
            bg=mask_weights((~np.isin(gt0,s["objects"]))&(gt0!=255),hw)
            y0=torch.stack([bg]+fg,1);valid0=y0.sum(1)>0
            y0=y0/y0.sum(1,keepdim=True).clamp(min=1e-8)
            yy,xx=torch.meshgrid(torch.arange(hw[0],device="cuda"),torch.arange(hw[1],device="cuda"),indexing="ij")
            coords=torch.stack([yy.flatten(),xx.flatten()],1)
            neighborhood=((coords[:,None]-coords[None]).abs().amax(-1)<=12)
            memory=deque(maxlen=7);labelmem={name:deque(maxlen=7) for name in ("local_anchor","global_anchor")}
            selected=set(s["frames"])
            for path in paths[1:]:
                image=Image.open(path).convert("RGB")
                fm=feats(image,args.size)[-1]
                assert fm.shape[-2:]==hw,(seq,path.name,hw,fm.shape)
                x=F.normalize(fm.flatten(1).T,dim=1)
                keys=torch.cat([x0]+list(memory),0)
                raw=x@keys.T
                p_all={}
                for name in labelmem:
                    logits=raw.clone()
                    if name=="local_anchor":logits[:,:N].masked_fill_(~neighborhood,float("-inf"))
                    logits[:,:N].masked_fill_(~valid0[None],float("-inf"))
                    for k in range(len(memory)):
                        logits[:,(k+1)*N:(k+2)*N].masked_fill_(~neighborhood,float("-inf"))
                    empty=~torch.isfinite(logits).any(1)
                    if empty.any():logits[empty,:N]=raw[empty,:N].masked_fill(~valid0[None],float("-inf"))
                    values,indices=logits.topk(5,dim=1)
                    weights=(values/.2).softmax(1)
                    labels=torch.cat([y0]+list(labelmem[name]),0)
                    p=(labels[indices]*weights[:,:,None]).sum(1)
                    assert torch.isfinite(p).all(),(seq,path.name,name)
                    p_all[name]=p
                # Memory update is causal and unconditional: an existing baseline, not our new policy.
                memory.append(x)
                for name in labelmem:labelmem[name].append(p_all[name])
                total_frames+=1
                if path.stem not in selected:continue
                predictions={}
                rendered={name:F.interpolate(p.T.reshape(1,-1,*hw),size=(image.height,image.width),
                          mode="bilinear",align_corners=False)[0].argmax(0).cpu().numpy()
                          for name,p in p_all.items()}
                for j,oid in enumerate(s["objects"],1):
                    pp={}
                    for name,p in p_all.items():
                        score=p[:,j];index=int(score.argmax())
                        px=int(min((index%hw[1]+.5)*image.width/hw[1],image.width-1))
                        py=int(min((index//hw[1]+.5)*image.height/hw[0],image.height-1))
                        # Full mask argmax control; original-size rendering is part of scoring.
                        cls=rendered[name]
                        pp[name]=dict(point=[px,py],score=float(score.max()),predicted_positive_pixels=int((cls==j).sum()))
                    predictions[oid]=pp
                # Query labels are loaded only after all predictions and memory updates are finalized.
                gt=np.asarray(Image.open(gp/(path.stem+".png")))
                cy=np.minimum(((np.arange(hw[0])+.5)*image.height/hw[0]).astype(int),image.height-1)
                cx=np.minimum(((np.arange(hw[1])+.5)*image.width/hw[1]).astype(int),image.width-1)
                centers=gt[cy[:,None],cx[None,:]]
                for oid,pp in predictions.items():
                    area=int((gt==oid).sum());presence=0 if area==0 else (1 if area>=32 else -1)
                    j=s["objects"].index(oid)+1
                    for name,pred in pp.items():
                        px,py=pred["point"];label=int(gt[py,px])
                        pred.update(point_hit=int(label==oid),other_annotated_object_hit=int(label not in (0,255,oid)),void_hit=int(label==255))
                        pm=(rendered[name]==j)&(gt!=255);gm=gt==oid
                        union=int((pm|gm).sum())
                        pred["region_iou"]=float((pm&gm).sum()/union) if union else 1.0
                    records.append(dict(sequence=seq,frame=path.stem,object=oid,area=area,presence=presence,
                                        grid_center_target_available=bool((centers==oid).any()),methods=pp))
            status.update(state="RUNNING",sequences_completed=ns+1,total_frames=total_frames,object_frames=len(records),
                          elapsed_s=time.monotonic()-start,running_results=summarize(records))
            save(out/"status.json",status);save(out/"records.json",records)
            print(json.dumps({k:status[k] for k in ("sequences_completed","total_frames","elapsed_s","running_results")}),flush=True)
        report=dict(methods=summarize(records),sequences=len(manifest),object_frames=len(records),total_frames=total_frames,
                    real_absent_sampled=sum(r["presence"]==0 for r in records),split="development",
                    no_query_gt_in_inference=True,no_new_method_claim=True,
                    input_short_side=args.size,notes="Continuous causal predictions for all intermediate frames. Partial paper-inspired controls, not exact official tuned replication.")
        save(out/"report.json",report)
        status.update(state="COMPLETED",allocated_peak_gb=torch.cuda.max_memory_allocated()/2**30,
                      reserved_peak_gb=torch.cuda.max_memory_reserved()/2**30)
        save(out/"status.json",status);print(json.dumps(report),flush=True)
    except Exception:
        status.update(state="ERROR",error=traceback.format_exc());save(out/"status.json",status);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prior",required=True);p.add_argument("--out",required=True)
    p.add_argument("--size",type=int,default=480);run(p.parse_args())
