"""Modern frozen-feature controls for real cross-frame identity and absence.

Given only a first-frame binary object mask; query annotations are scorer-only.
DAVIS2017 TRAIN split = development evidence. No SAM, downloads, or offline training.
Cosine, cropped-template, dual-softmax, contrast and ridge are existing controls.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "/root/demo4_cache/env")
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from safetensors.torch import load_file

DATA = Path("/root/autodl-tmp/demo1_sam/assets/datasets/DAVIS")
WEIGHT = Path("/root/demo4_cache/models/dinov3-vitl16-timm")


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def tensor(image, size=512, shortest=False):
    w,h=image.size
    scale=size/(min(w,h) if shortest else max(w,h))
    nw,nh=max(16,round(w*scale/16)*16),max(16,round(h*scale/16)*16)
    x=torch.from_numpy(np.asarray(image.resize((nw,nh),Image.BICUBIC)).copy()).permute(2,0,1).float()/255
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    return ((x-mean)/std)[None].cuda()


def sample_manifest():
    allseq=DATA.joinpath("ImageSets/2017/train.txt").read_text().split()
    rows=[]
    for seq in allseq:
        fs=sorted((DATA/"Annotations/480p"/seq).glob("*.png"))
        first=np.asarray(Image.open(fs[0]))
        ids=[int(i) for i in np.unique(first) if i not in (0,255) and int((first==i).sum())>=32]
        if len(ids)<2:continue
        counts=[]
        for f in fs:
            c=np.bincount(np.asarray(Image.open(f)).flatten(),minlength=256)
            counts.append([int(c[i]) for i in ids])
        counts=np.asarray(counts)
        selected=set(np.linspace(1,len(fs)-1,8).round().astype(int).tolist())
        # Additional real disappearance strata, not synthetic crops or pixel edits.
        for k in range(len(ids)):
            absent=np.where(counts[:,k]==0)[0]
            if len(absent):selected.update([int(absent[0]),int(absent[len(absent)//2]),int(absent[-1])])
        selected.discard(0)
        rows.append(dict(sequence=seq,objects=ids,frames=[fs[i].stem for i in sorted(selected)],
                         raw_absent_object_frames=int((counts==0).sum()),
                         sample_rule="8 uniform noninitial frames plus first/middle/last real absent per object"))
    return rows


def mask_weights(binary, hw):
    return F.adaptive_avg_pool2d(torch.from_numpy(binary.copy()).float()[None,None].cuda(),hw).flatten()


def auc(scores, labels):
    p=np.asarray(scores)[np.asarray(labels)==1]
    n=np.asarray(scores)[np.asarray(labels)==0]
    if not len(p) or not len(n):return None
    return float(((p[:,None]>n[None]).sum()+.5*(p[:,None]==n[None]).sum())/(len(p)*len(n)))


def summarize(records):
    methods=sorted(records[0]["methods"])
    result={}
    for method in methods:
        groups=defaultdict(list)
        for r in records:groups[(r["sequence"],r["object"])].append(r)
        aucs=[]
        p=0; hits=0;wrong_id=0
        for rs in groups.values():
            valid=[r for r in rs if r["presence"]!=-1]
            aa=auc([r["methods"][method]["score"] for r in valid],[r["presence"] for r in valid])
            if aa is not None:aucs.append(aa)
            for r in rs:
                if r["presence"]==1:
                    p+=1;hits+=r["methods"][method]["point_hit"]
                    wrong_id+=r["methods"][method]["other_annotated_object_hit"]
        result[method]=dict(present_frames=p,point_hit_fraction=hits/max(p,1),
                            wrong_object_fraction=wrong_id/max(p,1),
                            macro_object_absence_auc=float(np.mean(aucs)) if aucs else None,
                            objects_with_real_absence=len(aucs))
    return result


class Features:
    def __init__(self, shortest=False):
        self.shortest=shortest
        cfg=json.loads((WEIGHT/"config.json").read_text())
        assert cfg["architecture"]=="vit_large_patch16_dinov3"
        self.m=timm.create_model(cfg["architecture"],pretrained=False,num_classes=0)
        result=self.m.load_state_dict(load_file(str(WEIGHT/"model.safetensors")),strict=True)
        self.m=self.m.cuda().eval()
        for p in self.m.parameters():p.requires_grad=False
        print("MODEL",cfg["architecture"],str(result),"prefix",self.m.num_prefix_tokens,flush=True)
        assert self.m.num_prefix_tokens==5

    @torch.inference_mode()
    def __call__(self,image,size=512,shortest=None):
        shortest=self.shortest if shortest is None else shortest
        maps=self.m.forward_intermediates(tensor(image,size,shortest),indices=[11,23],norm=True,
                                         output_fmt="NCHW",intermediates_only=True)
        assert len(maps)==2
        return [m[0].float() for m in maps]


def bank(flat,w,maximum=64):
    ids=torch.where(w>.5)[0]
    if len(ids)<1:ids=w.topk(min(4,len(w))).indices
    if len(ids)>maximum:ids=ids[torch.linspace(0,len(ids)-1,maximum,device=ids.device).long()]
    return F.normalize(flat[ids],dim=1)


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    status=dict(state="STARTING",pid=os.getpid(),downloads=0,training_updates=0,
                split="DAVIS2017 train, development only",query_gt_role="scoring only",
                input="first-frame image and binary target mask; later RGB",
                task="reference localization and real target disappearance, not full mask segmentation",
                timing="shared GPU, no speed claim", input_side="shortest" if args.shortest else "longest",
                input_resolution=args.size, ridge_control="analytic fit on given first-frame prompt, no offline training")
    save(out/"status.json",status)
    try:
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        manifest=sample_manifest()
        assert len(manifest)>=30
        save(out/"manifest.json",manifest)
        features=Features(args.shortest)
        status.update(state="RUNNING",sequences_total=len(manifest),real_absent_source_frames=sum(s["raw_absent_object_frames"] for s in manifest))
        save(out/"status.json",status)
        records=[];start=time.monotonic()
        for ns,s in enumerate(manifest):
            seq=s["sequence"];images=DATA/"JPEGImages/480p"/seq;labels=DATA/"Annotations/480p"/seq
            im0=Image.open(images/"00000.jpg").convert("RGB")
            mask0=np.asarray(Image.open(labels/"00000.png"))
            reference=features(im0,args.size)
            hw=reference[-1].shape[-2:]
            flats=[m.flatten(1).T for m in reference]
            refs={}
            # Multioutput first-frame ridge is a strong same-feature supervised control.
            x=F.normalize(flats[-1],dim=1)
            mu=x.mean(0);xc=x-mu
            ys=[]
            for oid in s["objects"]:
                w=mask_weights(mask0==oid,hw)
                a=w.mean().clamp(.001,.999)
                ys.append(w/a-(1-w)/(1-a))
            target=torch.stack(ys,1)
            coeff=torch.linalg.solve(xc.T@xc/len(x)+.01*torch.eye(x.shape[1],device="cuda"),xc.T@target/len(x))
            for oi,oid in enumerate(s["objects"]):
                w=mask_weights(mask0==oid,hw)
                bg=mask_weights((mask0!=oid)&(mask0!=255),hw)
                prototypes=[F.normalize((f*w[:,None]).sum(0)/w.sum(),dim=0) for f in flats]
                background=F.normalize((flats[-1]*bg[:,None]).sum(0)/bg.sum(),dim=0)
                yy,xx=np.where(mask0==oid)
                box=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
                crop=im0.crop(box)
                cm=features(crop,256,shortest=False)[-1]
                cw=mask_weights((mask0==oid)[box[1]:box[3],box[0]:box[2]],cm.shape[-2:])
                cf=cm.flatten(1).T
                cropped=F.normalize((cf*cw[:,None]).sum(0)/cw.sum(),dim=0)
                refs[oid]=dict(prototypes=prototypes,background=background,bank=bank(flats[-1],w),
                               cropped=cropped,cropped_bank=bank(cf,cw),ridge=coeff[:,oi],ridge_center=mu)
            for frame in s["frames"]:
                image=Image.open(images/(frame+".jpg")).convert("RGB")
                query=features(image,args.size)
                q=[F.normalize(m.flatten(1).T,dim=1) for m in query]
                gh,gw=query[-1].shape[-2:]
                # No query label is used to compute any score or position.
                predictions={}
                for oid,ref in refs.items():
                    sim=ref["bank"]@q[-1].T
                    csim=ref["cropped_bank"]@q[-1].T
                    score=dict(final_proto=q[-1]@ref["prototypes"][-1],
                               middle_proto=q[0]@ref["prototypes"][0],
                               cropped_proto=q[-1]@ref["cropped"],
                               background_contrast=q[-1]@(ref["prototypes"][-1]-ref["background"]),
                               first_frame_ridge=(q[-1]-ref["ridge_center"])@ref["ridge"],
                               patch_max=sim.max(0).values,
                               dual_softmax=(sim.div(.1).softmax(1)*sim.div(.1).softmax(0)).sum(0),
                               cropped_dual_softmax=(csim.div(.1).softmax(1)*csim.div(.1).softmax(0)).sum(0))
                    score["multilayer_proto"]=.5*(score["final_proto"]+score["middle_proto"])
                    predictions[oid]={}
                    for name,v in score.items():
                        assert torch.isfinite(v).all(),(seq,frame,name)
                        idx=int(v.argmax());y=(idx//gw+.5)*image.height/gh;xpos=(idx%gw+.5)*image.width/gw
                        predictions[oid][name]=dict(score=float(v.max()),point=[int(min(xpos,image.width-1)),int(min(y,image.height-1))])
                # Explicit scorer boundary; predictions completed before opening query annotations.
                gt=np.asarray(Image.open(labels/(frame+".png")))
                for oid,pp in predictions.items():
                    area=int((gt==oid).sum());presence=0 if area==0 else (1 if area>=32 else -1)
                    for pred in pp.values():
                        px,py=pred["point"];label=int(gt[py,px])
                        pred.update(point_hit=int(label==oid),other_annotated_object_hit=int(label not in (0,255,oid)),
                                    void_hit=int(label==255))
                    cy=np.minimum(((np.arange(gh)+.5)*image.height/gh).astype(int),image.height-1)
                    cx=np.minimum(((np.arange(gw)+.5)*image.width/gw).astype(int),image.width-1)
                    centers=gt[cy[:,None],cx[None,:]]
                    records.append(dict(sequence=seq,frame=frame,object=oid,area=area,presence=presence,methods=pp,
                                        grid_center_target_available=bool((centers==oid).any())))
            status.update(sequences_completed=ns+1,object_frames=len(records),elapsed_s=time.monotonic()-start,
                          running_results=summarize(records))
            save(out/"status.json",status);save(out/"records.json",records)
            print(json.dumps({k:status[k] for k in ("sequences_completed","object_frames","elapsed_s","running_results")}),flush=True)
        report=dict(methods=summarize(records),sequences=len(manifest),object_frames=len(records),
                    real_absent_sampled=sum(r["presence"]==0 for r in records),
                    tiny_visible_excluded=sum(r["presence"]==-1 for r in records),
                    new_method_claim=False,split="development",query_gt_used_for_inference=False,
                    notes="No identity memory updates, no segmentation model, no calibrated deployment thresholds. ROC AUC is diagnostic.")
        save(out/"report.json",report)
        status.update(state="COMPLETED",allocated_peak_gb=torch.cuda.max_memory_allocated()/2**30,
                      reserved_peak_gb=torch.cuda.max_memory_reserved()/2**30)
        save(out/"status.json",status);print(json.dumps(report),flush=True)
    except Exception:
        status.update(state="ERROR",error=traceback.format_exc());save(out/"status.json",status);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True)
    p.add_argument("--size",type=int,default=512)
    p.add_argument("--shortest",action="store_true")
    run(p.parse_args())
