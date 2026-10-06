"""Read-only M5 ROI6 component audit: 2 threads/4GiB, no image encoding/GT."""
import argparse, hashlib, json, math, os, sys, threading, time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('code','cache','binding','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    start=time.perf_counter();peak=[0]
    def guard():
        while True:
            rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
            peak[0]=max(peak[0],rss)
            if rss>4<<30 or time.perf_counter()-start>180:
                (a.out/'guard_failure.json').write_text(json.dumps({'rss':rss,'elapsed':time.perf_counter()-start}))
                os._exit(89)
            time.sleep(.2)
    threading.Thread(target=guard,daemon=True).start()
    sys.path.insert(0,str(a.code/'src'))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    from safetensors import safe_open
    from PIL import Image
    from ics.methods import pro_message_extrapolation as m5
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_sha=sha(a.cache)
    pair=torch.load(a.cache,map_location='cpu',weights_only=True,mmap=True)
    binding=json.loads(a.binding.read_text());row=binding['row'];r=pair['reference']
    with Image.open(Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')/Path(row['support']).with_suffix('.png')) as im:
        mask=np.asarray(im)==row['c']+1
    mask=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST)).copy()
    cov=F.interpolate(torch.from_numpy(mask.astype(np.float32))[None,None],(64,64),mode='area')[0,0].numpy()
    fg=cov.ravel()>=.5
    if not fg.any():fg[np.argmax(cov)]=True
    parts,_=m5.ward_partitions(r['projected_native'].numpy(),(64,64),(8,),~fg)
    regions=[np.flatnonzero(fg)]+parts[8]
    roi=regions[5];ids=r['patch_ids'][torch.as_tensor(roi)]
    print(json.dumps({'stage':'ROI_SELECTED','reference_region_sizes':list(map(len,regions)),'selected_size':len(ids)}),flush=True)
    config=json.loads(Path('/root/demo4_cache/models/dinov3-vitl16-timm/config.json').read_text())
    # Only two suffix blocks acquire CPU storage/weights. No encoder is executed.
    meta=timm.create_model(config['architecture'],pretrained=False,num_classes=0,device='meta')
    blocks=[]
    with safe_open('/root/demo4_cache/models/dinov3-vitl16-timm/model.safetensors',framework='pt',device='cpu') as state:
        for index in (20,21):
            block=meta.blocks[index].to_empty(device='cpu').float().eval().requires_grad_(False)
            prefix=f'blocks.{index}.'
            subset={key[len(prefix):]:state.get_tensor(key) for key in state.keys() if key.startswith(prefix)}
            block.load_state_dict(subset,strict=True);blocks.append(block)
    def error(actual,expected):
        difference=(actual-expected).abs();threshold=5e-5+5e-5*expected.abs()
        ratio=difference/threshold
        return {'maxabs':float(difference.max()),'allclose':bool(torch.allclose(actual,expected,atol=5e-5,rtol=5e-5)),
                'violating_elements':int((difference>threshold).sum()),'max_tolerance_ratio':float(ratio.max())}
    def padded_values(block,hidden,mode):
        n=len(hidden);size=n if mode=='base64' and n>=64 else (64 if mode=='base64' else max(256,math.ceil(n/64)*64))
        hidden=F.pad(hidden,(0,0,0,size-n))
        projection=block.attn.qkv(block.norm1(hidden[None]))
        return projection.reshape(1,size,3,16,64).permute(2,0,3,1,4)[2,0,:,:n]
    def padded_finish(block,hidden,message,mode):
        n=len(hidden);size=n if mode=='base64' and n>=64 else (64 if mode=='base64' else max(256,math.ceil(n/64)*64))
        hidden=F.pad(hidden,(0,0,0,size-n));message=F.pad(message,(0,0,0,size-n))
        x=message.transpose(0,1).reshape(1,size,1024)
        x=block.attn.proj_drop(block.attn.proj(block.attn.norm(x)))
        x=block.gamma_1*x if block.gamma_1 is not None else x
        h=hidden[None]+block.drop_path1(x)
        update=block.mlp(block.norm2(h));update=block.gamma_2*update if block.gamma_2 is not None else update
        return (h+block.drop_path2(update))[0,:n]
    def attention(q,k,v,query_ids):
        out=[]
        for pos in range(0,len(query_ids),64):
            out.append(F.scaled_dot_product_attention(q[:,query_ids[pos:pos+64]][None],k[None],v[None],dropout_p=0.,is_causal=False,scale=.125)[0])
        return torch.cat(out,dim=1)
    report={'case':row['key'],'ROI_index_one_based':6,'reference_region_sizes':list(map(len,regions)),
            'ROI_size':len(ids),'ROI_patch_ids_sha256':hashlib.sha256(np.asarray(roi,dtype=np.int64).tobytes()).hexdigest(),
            'threads':2,'RSS_limit_bytes':4<<30,'new_encoder_forwards':0,'query_GT_opened':False,'layers':[]}
    propagated={mode:r['h20'][ids].clone() for mode in ('base64','ceil64_min256')}
    with torch.inference_mode():
        for li,block in enumerate(blocks):
            hidden_all=r['h20'] if li==0 else r['native_states'][li-1]
            hidden=hidden_all[ids];expected=r['native_states'][li][ids]
            paired_hidden=torch.stack((hidden_all,pair['query']['h20'] if li==0 else pair['query']['native_states'][li-1]))
            captured={};original_sdpa=F.scaled_dot_product_attention
            def native_sdpa(q,k,v,**kwargs):
                captured.update(q=q.detach(),k=k.detach(),v=v.detach())
                output=original_sdpa(q,k,v,**kwargs);captured['message']=output.detach()
                return output
            try:
                F.scaled_dot_product_attention=native_sdpa
                actual_full=block(paired_hidden,rope=r['rope'][li])
            finally:F.scaled_dot_product_attention=original_sdpa
            q,k,v=r['qkv'][li]
            native_message=captured['message'][0,:,ids]
            record={'block':21+li,'fresh_actual_block_vs_cache':error(actual_full[0,ids],expected),
                    'strides':{'saved_q':q.stride(),'saved_k':k.stride(),'saved_v':v.stride(),
                               'actual_v':captured['v'][0].stride(),'actual_message':captured['message'][0].stride()},
                    'saved_V_vs_fresh_actual_V':error(v,captured['v'][0]),'values':{},'finish_true_native_message':{},
                    'SDPA_message':{},'propagated_beta1':{}}
            for mode in propagated:
                record['values'][mode]=error(padded_values(block,hidden,mode),v[:,ids])
                record['finish_true_native_message'][mode]=error(padded_finish(block,hidden,native_message,mode),expected)
                branch_v=padded_values(block,propagated[mode],mode)
                mixed=v.clone();mixed[:,ids]=branch_v
                message=attention(q,k,mixed,ids)
                propagated[mode]=padded_finish(block,propagated[mode],message,mode)
                record['propagated_beta1'][mode]=error(propagated[mode],expected)
            for label,kernel_v in [('saved_stride',v),('saved_contiguous',v.contiguous()),('actual_native_stride',captured['v'][0])]:
                record['SDPA_message'][label]=error(attention(q,k,kernel_v,ids),native_message)
            sample=ids[-min(64,len(ids)):]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                F.scaled_dot_product_attention(q[:,sample][None],k[None],v[None],dropout_p=0.,is_causal=False,scale=.125)
            record['SDPA_dispatch']=[x.key for x in prof.key_averages() if 'scaled_dot' in x.key or 'flash' in x.key]
            report['layers'].append(record)
            print(json.dumps(record),flush=True)
    report.update(native_pair_sha256=original_sha,native_pair_unchanged=sha(a.cache)==original_sha,
                  method_source_sha256=sha(a.code/'src/ics/methods/pro_message_extrapolation.py'),
                  wall_seconds=time.perf_counter()-start,peak_RSS_bytes=peak[0],read_only_scope='two suffix blocks on saved native inputs; no M5 code/state writes')
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'state':'BOUNDED_COMPONENT_DIAG_COMPLETE','wall':report['wall_seconds'],'RSS':peak[0]}),flush=True)


if __name__=='__main__':main()
