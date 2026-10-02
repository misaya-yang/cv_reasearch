"""Can cheap margin-selected exact parent patches repair an oracle coarse head?"""
import json
from pathlib import Path
import sys
import subprocess
import time
import math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/compute_structure'))
from pretrained_decoder_round import load_decoder,load_inputs,full_masks
from takeover_rank_features import quality
from takeover_flip_score import bootstrap_image_groups
OUT=ROOT/'results/takeover_20261001_v1/head_fallback_v1'


def phases(low):
    return low.reshape(len(low),4,64,4,64,4).permute(0,2,4,1,3,5).reshape(len(low),4096,4,16)


def grid(ph):
    return ph.reshape(len(ph),64,64,4,4,4).permute(0,3,1,4,2,5).reshape(len(ph),4,256,256)


def main():
    OUT.mkdir(exist_ok=False)
    report={'status':'WAITING_VRAM','protocol':'Previous development-only PCA basis and same last12 development images. Rank32/64; original full head always computed, so oracle feasibility rather than deployment or speed evidence. Replace exact original 4x4 child-logit patches in all4 masks at 1/5/10% parent budgets. Selector uses minimum absolute projected logit divided by mask hypernetwork norm, across all4 masks and16 children, never GT or observed teacher residual. Random same-budget parent patches are strong controls. Original IoU head selection held fixed.','rows':[],'summary':{}}
    def save():
        tmp=OUT/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'report.json')
    save()
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<2000:time.sleep(10)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    basis=torch.load(OUT.parent/'head_compress_v1/basis.pt',weights_only=True,map_location='cuda:0');mu=basis['mean'];vectors=basis['basis']
    source=OUT.parent/'real_original';paths=sorted(source.glob('image_*/encoded_inputs.npz'));assert len(paths)==24
    model,_,_=load_decoder(source/'mask_decoder_state.pt',torch.device('cuda:0'))
    capture={};handles=[model.transformer.register_forward_hook(lambda m,i,o:capture.update(tokens=o[0])),model.output_upscaling.register_forward_hook(lambda m,i,o:capture.update(upscaled=o))]
    subset=ROOT/'assets/coco_quality_seed2027_v1';manifest=json.loads((subset/'manifest.json').read_text());byid={x['image_id']:x for x in manifest['images']}
    rng=torch.Generator(device='cuda:0').manual_seed(2028)
    try:
        report['status']='RUNNING_ORACLE_FEASIBILITY';save()
        with torch.inference_mode():
            for n,path in enumerate(paths[12:]):
                image,pe,dense,sparse,info=load_inputs(path,torch.device('cuda:0'))
                with np.load(subset/byid[info['image_id']]['gt_file'],allow_pickle=False) as gt:
                    truth=torch.from_numpy(gt['gt']).cuda();ids=gt['annotation_ids'].tolist()
                assert info['annotation_ids']==ids
                for regime,sp in sparse.items():
                    low,scores=model.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sp,dense_prompt_embeddings=dense)
                    original=full_masks(low,info);oq,ob=quality(original,truth);choices=scores[:,1:].argmax(-1)
                    up=capture['upscaled'];p=len(up)
                    phi=up.reshape(p,32,64,4,64,4).permute(0,2,4,3,5,1).reshape(-1,512)
                    hyper=torch.stack([mlp(capture['tokens'][:,i+1]) for i,mlp in enumerate(model.output_hypernetworks_mlps)],1)
                    reference=phases(low)
                    assert torch.equal(grid(reference),low)
                    for rank in (32,64):
                        v=vectors[:,:rank];z=((phi-mu)@v@v.T+mu).reshape(p,64,64,4,4,32)
                        approximate=torch.einsum('pmc,pijuvc->pmijuv',hyper,z).permute(0,1,2,4,3,5).reshape(p,4,256,256)
                        ap=phases(approximate)
                        margin=(ap.abs()/hyper.norm(dim=-1)[:,None,:,None].clamp_min(1e-6)).amin((-2,-1))
                        priority=margin.argsort(-1);random=torch.rand(p,4096,device='cuda:0',generator=rng).argsort(-1)
                        configurations=[('projection',0,None)]
                        for fraction in (.01,.05,.1):
                            k=math.ceil(4096*fraction)
                            configurations.extend([('margin',fraction,priority[:,:k]),('random',fraction,random[:,:k])])
                        for selector,budget,indices in configurations:
                            mixed=ap.clone()
                            if indices is not None:
                                batch=torch.arange(p,device='cuda:0')[:,None];mixed[batch,indices]=reference[batch,indices]
                            masks=full_masks(grid(mixed),info);q,bq=quality(masks,truth);flips=(masks!=original).sum((-2,-1))
                            for j,aid in enumerate(ids):
                                c=int(choices[j]);report['rows'].append({'image_id':info['image_id'],'annotation_id':aid,'regime':regime,'rank':rank,'selector':selector,'budget':budget,'actual_parent_fraction':0 if indices is None else indices.shape[1]/4096,'delta_iou':float(q[j,c]-oq[j,c]),'delta_boundary_iou':float(bq[j,c]-ob[j,c]),'flip_fraction':float(flips[j,c+1]/masks.shape[-1]/masks.shape[-2])})
                save();print(json.dumps({'images':n+1}),flush=True)
        for regime in ('central','near_boundary','box'):
            report['summary'][regime]={}
            for rank in (32,64):
                for selector,budget in [('projection',0)]+[(s,b) for b in (.01,.05,.1) for s in ('margin','random')]:
                    rows=[r for r in report['rows'] if r['regime']==regime and r['rank']==rank and r['selector']==selector and r['budget']==budget]
                    report['summary'][regime][f'rank{rank}_{selector}_{budget}']={'mean_delta_iou':float(np.mean([r['delta_iou'] for r in rows])),'image_cluster_delta95':bootstrap_image_groups(rows,'delta_iou'),'mean_flip_fraction':float(np.mean([r['flip_fraction'] for r in rows]))}
        report['status']='COMPLETED_ORACLE_FEASIBILITY';print(json.dumps(report['summary']),flush=True)
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:
        for h in handles:h.remove()
        save()


if __name__=='__main__':main()
