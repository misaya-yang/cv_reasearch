"""Oracle local-head compressibility; not a fast deployment implementation."""
import json
import argparse
from pathlib import Path
import sys
import time
import subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/compute_structure'))
from pretrained_decoder_round import load_decoder,load_inputs,full_masks
from takeover_rank_features import quality
from takeover_flip_score import bootstrap_image_groups

OUT=ROOT/'results/takeover_20261001_v1/head_compress_v1'


def main(residual=False):
    global OUT
    if residual:OUT=ROOT/'results/takeover_20261001_v1/head_residual_v1'
    OUT.mkdir(exist_ok=False)
    report={'status':'WAITING_VRAM','protocol':'SAM1 pretrained full decoder; 24 previously used development images, first12 by image-ID fit PCA, last12 diagnose only. All original masks computed; projections are oracle and have no speed claim. Ranks16/32/64/128 fixed beforehand. Quality uses original IoU-head selection, full resolution and GT; no candidate reselection.','rows':[],'summary':{}}
    if residual:report['protocol']+=' Residual variant: cache one head per image from a reference with two not-a-point padding embeddings; compress only prompt-conditioned head differences from that shared reference. Padding embedding comes from the actual exported central point padding token, with no GT or new encoder. Reference decoder cost must later be amortized and measured.'
    def save():
        tmp=OUT/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'report.json')
    save()
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<2000:time.sleep(10)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'results/takeover_20261001_v1/real_original'
    paths=sorted(source.glob('image_*/encoded_inputs.npz'))
    assert len(paths)==24
    model,_,_=load_decoder(source/'mask_decoder_state.pt',torch.device('cuda:0'))
    capture={}
    hooks=[model.transformer.register_forward_hook(lambda m,i,o:capture.update(tokens=o[0],src=o[1])),model.output_upscaling.register_forward_hook(lambda m,i,o:capture.update(upscaled=o))]
    manifest=json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text())
    byid={x['image_id']:x for x in manifest['images']}
    total=torch.zeros(512,device='cuda:0',dtype=torch.float64);cross=torch.zeros(512,512,device='cuda:0',dtype=torch.float64);count=0
    generator=torch.Generator(device='cuda:0').manual_seed(2027)
    def forward(path):
        image,pe,dense,sparse,info=load_inputs(path,torch.device('cuda:0'))
        base_phi=None
        if residual:
            padding=sparse['central'][:1,1:2].expand(-1,2,-1)
            model.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=padding,dense_prompt_embeddings=dense)
            base_phi=capture['upscaled'].reshape(1,32,64,4,64,4).permute(0,2,4,3,5,1).reshape(4096,512)
        for regime,sp in sparse.items():
            low,scores=model.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=sp,dense_prompt_embeddings=dense)
            z=capture['upscaled'];p,c,h,w=z.shape
            assert (c,h,w)==(32,256,256)
            phi=z.reshape(p,32,64,4,64,4).permute(0,2,4,3,5,1).reshape(-1,512)
            base=None if base_phi is None else base_phi.repeat(p,1)
            if base is not None:phi=phi-base
            hyper=torch.stack([mlp(capture['tokens'][:,i+1]) for i,mlp in enumerate(model.output_hypernetworks_mlps)],1)
            yield regime,low,scores,phi,hyper,info,base
    def reconstruct(phi,hyper):
        p=len(hyper)
        y=phi.reshape(p,64,64,4,4,32)
        return torch.einsum('pmc,pijuvc->pmijuv',hyper,y).permute(0,1,2,4,3,5).reshape(p,4,256,256)
    try:
        report['status']='FIT_DEVELOPMENT_PCA';save()
        with torch.inference_mode():
            for n,path in enumerate(paths[:12]):
                for regime,low,scores,phi,hyper,info,base in forward(path):
                    indices=torch.randperm(len(phi),device='cuda:0',generator=generator)[:2048]
                    x=phi[indices].double();total+=x.sum(0);cross+=x.T@x;count+=len(x)
                print(json.dumps({'phase':'fit','images':n+1}),flush=True)
            mean=total/count;cov=cross/count-mean[:,None]*mean[None,:]
            eigen,basis=torch.linalg.eigh((cov+cov.T)/2);eigen=eigen.flip(0).clamp_min(0);basis=basis.flip(1)
            mean=mean.float();basis=basis.float()
            torch.save({'mean':mean.cpu(),'basis':basis.cpu(),'eigenvalues':eigen.cpu(),'fit_images':[int(p.parent.name.removeprefix('image_')) for p in paths[:12]],'oracle_only':True},OUT/'basis.pt')
            report['retained_variance']={str(r):float(eigen[:r].sum()/eigen.sum()) for r in (16,32,64,128)}
            report['status']='DIAGNOSE_DEVELOPMENT_IMAGES';save()
            for n,path in enumerate(paths[12:]):
                for regime,low,scores,phi,hyper,info,base in forward(path):
                    image_info=byid[info['image_id']]
                    with np.load(ROOT/'assets/coco_quality_seed2027_v1'/image_info['gt_file'],allow_pickle=False) as gt:
                        ids=gt['annotation_ids'].tolist();truth=torch.from_numpy(gt['gt']).cuda()
                    assert info['annotation_ids']==ids
                    original=full_masks(low,info);oq,ob=quality(original,truth)
                    choice=scores[:,1:].argmax(-1)
                    identity=reconstruct(phi if base is None else phi+base,hyper)
                    identity_error=float((identity-low).abs().max())
                    for rank in (16,32,64,128):
                        v=basis[:,:rank];approx=(phi-mean)@v@v.T+mean
                        if base is not None:approx=approx+base
                        candidate=reconstruct(approx,hyper);mask=full_masks(candidate,info)
                        q,bq=quality(mask,truth)
                        flips=(mask!=original).sum((-2,-1))
                        for j,aid in enumerate(ids):
                            c=int(choice[j]);delta=float(q[j,c]-oq[j,c])
                            report['rows'].append({'image_id':info['image_id'],'annotation_id':aid,'regime':regime,'rank':rank,'original_head_iou':float(oq[j,c]),'compressed_head_iou':float(q[j,c]),'delta_iou':delta,'delta_boundary_iou':float(bq[j,c]-ob[j,c]),'selected_mask_flip_fraction':float(flips[j,c+1]/mask.shape[-1]/mask.shape[-2]),'max_logit_error':float((candidate-low).abs().max()),'identity_layout_max_logit_error':identity_error})
                save();print(json.dumps({'phase':'diagnose','images':n+1}),flush=True)
            for regime in ('central','near_boundary','box'):
                report['summary'][regime]={}
                for rank in (16,32,64,128):
                    rows=[r for r in report['rows'] if r['regime']==regime and r['rank']==rank]
                    report['summary'][regime][str(rank)]={'mean_delta_iou':float(np.mean([r['delta_iou'] for r in rows])),'image_cluster_delta95':bootstrap_image_groups(rows,'delta_iou'),'mean_flip_fraction':float(np.mean([r['selected_mask_flip_fraction'] for r in rows]))}
        report['status']='COMPLETED_ORACLE_DIAGNOSTIC';print(json.dumps(report['summary']),flush=True)
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:
        for hook in hooks:hook.remove()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--residual',action='store_true');main(p.parse_args().residual)
