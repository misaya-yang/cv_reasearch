"""One prepared mechanism probe: native raw / local PG readout / full rewrite.

RSRM SAFR + adapted FROST-style dense readout is a borrowed matched control,
not our innovation/full paper reproduction. No pools/training/query-GT choices.
Attention/raw tensors live in RAM for one episode; no feature cache is written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from native_attention_readout import native_attention_readouts
from global_content_attention import global_content_attention, cuda_flash_probe
from global_representation_probe import fixed_selection, summary, pixel_ledger
from tics.readout_baselines import safr_support_weights, fuse_raw_layers, frost_style_dense_readout
from tics.native_assets import reuse_native_basis


def safr_mask(mask,grid):
    return F.interpolate(mask.reshape(1,1,*mask.shape[-2:]).float(),grid,
                         mode='bilinear',align_corners=True)[0,0]>.5


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root',default='/root/autodl-tmp/demo9')
    p.add_argument('--out');p.add_argument('--fold',type=int,default=0)
    p.add_argument('--prepared-manifest',help='CPU-prepared fixed episode manifest; verify before encoder construction')
    p.add_argument('--projection-basis',help='Reuse the source-faithful native BLACK-image basis')
    p.add_argument('--limit',type=int,default=10);p.add_argument('--self-check',action='store_true')
    p.add_argument('--arms',default='native_raw,local_pg,kernel_pad,full_rewrite,full_kernel_pad',
                   help='Select independent representation comparisons, always including native_raw')
    a=p.parse_args()
    if a.self_check:self_check();return
    if os.environ.get('DEMO9_CUDA_GUARD')!='1':raise RuntimeError('Prepared resource guard required')
    arms=set(a.arms.split(','))
    if not arms or not arms <= {'native_raw','local_pg','kernel_pad','full_rewrite','full_kernel_pad'} or 'native_raw' not in arms:
        p.error('Known arms and the matched native_raw control are required')
    if not a.out:p.error('--out is required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Fresh output required; preserve old failures/results')
    source_files=[Path(__file__),HERE/'native_attention_readout.py',HERE/'global_content_attention.py',
                  HERE.parent/'tics/readout_baselines.py']
    report=dict(state='PREPARING',args=vars(a),records=[],seed=0,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        contract='Native SQ pair + native flipped-support; fixed support-native SAFR weights and per-image native raw scales. SAME decoder recipe/native geometry for all arms. No last-layer APD/U500 in raw-fusion space.',
        decoder_fit='Same support-only rule; each arm fits its own W and LOO sigma. Not numerically identical W/sigma.',
        scope='Matched same-L24 component adaptation, not full RSRM/FROST reproduction. Development pilot, not method gain claim.')
    def save():
        report['class_miou']=summary(report['records'])
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(out/'report.json')
    save()
    if not torch.cuda.is_available():report.update(state='NO_GPU');save();return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.3')))
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    sys.path.insert(0,str(Path(a.prepared_root)/'scripts'));import _paths
    sys.path.insert(0,_paths.DEMO4)
    from icx.common import TimmDINOv3,coco_episodes,DEV
    from utils.data import build_transform,load_image,load_mask,downsample_mask
    from PIL import Image
    started=time.monotonic();host=None
    try:
        episodes,_,base=coco_episodes(a.fold,400,shot=1,seed=0)
        selected=fixed_selection(episodes,a.limit)
        report['frozen_episodes']=[dict(e=e,c=c,support=r[0],query=q) for e,(c,q,r) in selected]
        if a.prepared_manifest:
            prepared_path=Path(a.prepared_manifest)
            prepared=json.loads(prepared_path.read_text())
            if report['frozen_episodes']!=prepared['frozen_episodes']:
                raise RuntimeError('CPU-prepared episode manifest differs; no encoder was constructed')
            report['prepared_manifest']=dict(path=str(prepared_path),
                sha256=hashlib.sha256(prepared_path.read_bytes()).hexdigest(),episodes_exact=True)
        save()
        report['flash_lift_backend']=cuda_flash_probe();save()
        encoder=TimmDINOv3().to(DEV).eval()
        for param in encoder.parameters():param.requires_grad=False
        backbone=encoder.m;transform=build_transform(1024)
        report['state']='RUNNING';save()
        def encode(images):
            # Same native BF16 encoding context, no quantized feature replay.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return encoder.get_intermediate_layers(images,n=1,reshape=True)[0].float()
        def host_prediction(support_image,support_mask,query_image):
            nonlocal host
            if host is None:
                from models.insid3 import INSID3
                with reuse_native_basis(INSID3,a.projection_basis) as basis_receipt:
                    host=INSID3(encoder=encoder,image_size=1024,svd_components=500,tau=.6,
                        merge_threshold=.2,mask_refiner='bilinear',resize_to_orig_size=False,device=DEV).eval()
                report['projection_basis']=basis_receipt
            if not support_mask.any():return torch.zeros((1024,1024),dtype=torch.bool,device=DEV)
            return host.predict_mask(support_image,support_mask,query_image).reshape(1024,1024).bool()
        with torch.inference_mode():
            for e,(c,q,refs) in selected:
                si=Image.open(Path(base)/refs[0]).convert('RGB');qi=Image.open(Path(base)/q).convert('RGB')
                smask=torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN)/str(Path(refs[0]).with_suffix('.png'))))==c+1).copy())
                s=load_image(si,transform,DEV)[0];t=load_image(qi,transform,DEV)[0]
                sq=torch.cat([s,t]);sf=s.flip(-1)
                sm=load_mask(smask,1024,DEV);smf=sm.flip(-1)
                # No query annotation has been opened at this point.
                plain=encode(sq)
                capture_mode='local_pg' if arms & {'local_pg','kernel_pad'} else 'native_only'
                with native_attention_readouts(backbone,mode=capture_mode,output_dtype=torch.float32) as capture:
                    native=encode(sq);flip=encode(sf)
                if not torch.equal(plain,native):raise RuntimeError('Native trajectory changed under read-only capture')
                grid=tuple(native.shape[-2:])
                if len(backbone.blocks)!=24 or native.shape[1]!=1024:raise RuntimeError('Expected the existing ViT-L/24/1024 channels')
                # CPU capture retains all arms; CUDA holds only one SQ+FlipS raw pair.
                native_raw=capture.raw_maps('R0',grid,0,device=DEV,dtype=torch.float32)
                native_flip_raw=capture.raw_maps('R0',grid,1,device=DEV,dtype=torch.float32)
                scales=capture.native_scales(0).to(DEV);flip_scales=capture.native_scales(1).to(DEV)
                weights=safr_support_weights([native_raw[:1],native_flip_raw],
                    [native[:1],flip],[safr_mask(sm,grid)[None],safr_mask(smf,grid)[None]])
                fmasks=[downsample_mask(m.unsqueeze(1),*grid).reshape(*grid).bool() if m.any()
                        else torch.zeros(grid,dtype=torch.bool,device=DEV) for m in (sm,smf)]
                predictions={};infos={}
                if not bool(weights.valid.all()) or any(not x.any() or x.all() for x in fmasks):
                    del native_raw,native_flip_raw
                    fallback=host_prediction(s,sm,t)
                    predictions={name:fallback for name in sorted(arms)}
                    report_fallback='Degenerate support-only SAFR/anchor set: unchanged INSID3 for all arms'
                    global_audit=None
                else:
                    report_fallback=None
                    rgb=torch.from_numpy(np.asarray(qi).copy()).to(DEV).permute(2,0,1).float()/255
                    rgb_grid=F.interpolate(rgb[None],grid,mode='bilinear',align_corners=False)[0].permute(1,2,0)
                    def decode_raw_pair(name,raw_sq,raw_flip):
                        fused=fuse_raw_layers(raw_sq,weights.weights.expand(2,-1),scales)
                        fused_flip=fuse_raw_layers(raw_flip,weights.weights,flip_scales)
                        prediction,info=frost_style_dense_readout([fused[0],fused_flip[0]],fmasks,
                            fused[1],native[1],rgb_grid,float(smask.float().mean()),(1024,1024))
                        predictions[name]=prediction
                        infos[name]=dict(sigma=info['sigma'],loo_margin=info['loo_margin'],
                            candidate_patches=int(info['candidate_grid'].sum()),
                            W_source='arm_support_only',sigma_source='arm_support_LOO',
                            smooth_margin=info['ell_smooth'])
                        del fused,fused_flip
                    decode_raw_pair('native_raw',native_raw,native_flip_raw)
                    del native_raw,native_flip_raw
                    for name,key in [('local_pg','Rcf'),('kernel_pad','Rpad')]:
                        if name not in arms:continue
                        raw_sq=capture.raw_maps(key,grid,0,device=DEV,dtype=torch.float32)
                        raw_flip=capture.raw_maps(key,grid,1,device=DEV,dtype=torch.float32)
                        decode_raw_pair(name,raw_sq,raw_flip)
                        del raw_sq,raw_flip
                    # Full rewrite runs after all native-trajectory CUDA raw maps are freed.
                    global_audit=None
                    for whole_name,whole_mode in [('full_rewrite','global_content'),('full_kernel_pad','padded_native')]:
                        if whole_name not in arms:continue
                        with global_content_attention(backbone,enabled=True,mode=whole_mode) as whole_audit:
                            with native_attention_readouts(backbone,mode='observe_modified',output_dtype=torch.float32) as modified:
                                encode(sq);encode(sf)
                        raw_sq=modified.raw_maps('R0',grid,0,device=DEV,dtype=torch.float32)
                        raw_flip=modified.raw_maps('R0',grid,1,device=DEV,dtype=torch.float32)
                        decode_raw_pair(whole_name,raw_sq,raw_flip)
                        if global_audit is None:global_audit={}
                        global_audit[whole_name]=whole_audit
                        del raw_sq,raw_flip,modified
                    del rgb,rgb_grid,decode_raw_pair
                # Real unchanged host reference; no transplanted raw positional basis.
                predictions['insid3_native']=host_prediction(s,sm,t)
                # All masks are now frozen. Only evaluator opens query annotations.
                qgt=torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN)/str(Path(q).with_suffix('.png'))))==c+1).copy())
                truth=load_mask(qgt,1024,DEV)[0].bool()
                small=F.interpolate(truth.reshape(1,1,1024,1024).float(),grid,mode='nearest')[0,0].bool()
                row=dict(e=e,c=c,support=refs[0],query=q,iu={},original_iu={},pixels={},reader_info={},
                    native_trajectory_exact=True,fallback=report_fallback,
                    weights_native_support=weights.weights.cpu().tolist(),
                    readout_audit=capture.audit,full_rewrite_audit=global_audit)
                base_pred=predictions['native_raw']
                for name,prediction in predictions.items():
                    row['iu'][name]=[int((prediction&truth).sum()),int((prediction|truth).sum())]
                    original=F.interpolate(prediction.reshape(1,1,1024,1024).float(),tuple(qgt.shape),mode='bilinear',align_corners=False)[0,0]>.5
                    g=qgt.to(DEV);row['original_iu'][name]=[int((original&g).sum()),int((original|g).sum())]
                    row['pixels'][name]=pixel_ledger(base_pred,prediction,truth)
                    if name in infos:
                        info=infos[name];margin=info.pop('smooth_margin')
                        if margin is not None:
                            info['query_fg_margin']=float(margin[small].mean()) if small.any() else None
                            info['query_bg_margin']=float(margin[~small].mean()) if (~small).any() else None
                        row['reader_info'][name]=info
                report['records'].append(row);report['elapsed_concurrent_s']=time.monotonic()-started
                report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                report['peak_reserved_bytes']=torch.cuda.max_memory_reserved();save()
                print(json.dumps(dict(count=len(report['records']),e=e,c=c,scores=report['class_miou'],elapsed_s=report['elapsed_concurrent_s'])),flush=True)
                del capture,plain,native,flip,predictions,infos,sq,sf,s,t,sm,smf,fmasks,scales,flip_scales,weights
        report.update(state='COMPLETED',native_trajectory_exact=True);save()
    except BaseException as exc:
        report.update(state='ERROR',error=repr(exc),elapsed_concurrent_s=time.monotonic()-started);save();raise


def self_check():
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(901)
    raw=torch.randn(1,4,256,4,4,generator=generator);w=torch.full((1,4),.25)
    scales=raw.std((2,3,4),correction=1,keepdim=True)
    native=fuse_raw_layers(raw,w,scales)
    altered=fuse_raw_layers(raw+1e-3,w,scales)
    assert torch.isfinite(native).all() and not torch.equal(native,altered)
    masks=[torch.zeros(4,4,dtype=torch.bool) for _ in range(2)]
    masks[0][:2]=True;masks[1]=masks[0].flip(-1)
    refs=[native[0],native[0].flip(-1)]
    result,info=frost_style_dense_readout(refs,masks,native[0],native[0],torch.rand(4,4,3,generator=generator),.5,(16,16))
    assert result.shape==(16,16) and result.dtype==torch.bool
    assert info['sigma'] is None or info['sigma'] in (.05,.1,.2,.5,1)
    assert pixel_ledger(result,result,torch.ones_like(result))==dict(recovered_fn=0,lost_tp=0,added_fp=0,removed_fp=0)
    print('CPU main composition passed: fixed scales, independent readers, dense gate/KDE/smoothing/finalization; no CUDA or dataset access.')

if __name__=='__main__':main()
