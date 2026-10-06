"""Finite component diagnosis from actual cached native suffix; no encoder/GT."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_MAX_ACTIVE_LEVELS='1')
import argparse,json,time,threading,sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.code/'src'))
    import torch,numpy as np
    from PIL import Image
    import torch.nn.functional as F
    from ics.methods import pro_message_extrapolation as m
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();out=args.run/'component_diagnosis.json';state={'state':'LOADING','new_encoder_calls':0}
    def write():out.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    def guard():
        while True:
            rss=next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
            state['peak_RSS_bytes']=max(state.get('peak_RSS_bytes',0),rss)
            if rss>8<<30 or time.perf_counter()-started>300:
                state.update(state='BUDGET_STOP',elapsed_seconds=time.perf_counter()-started);write();os._exit(137)
            time.sleep(.5)
    threading.Thread(target=guard,daemon=True).start()
    assets=json.loads((args.run/'outputs_v2/assets_first.json').read_text())
    pipe=m.load_local_eva_pipeline(assets,torch.device('cpu'));model=pipe['model']
    ep=args.run/'outputs_v2/episode_00_0_0_72'
    data=torch.load(ep/'native_pair.pt',map_location='cpu',weights_only=True)
    binding=json.loads((ep/'binding.json').read_text());row=binding['row']
    ann=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')/Path(row['support']).with_suffix('.png')
    with Image.open(ann) as im:mask=np.asarray(im)==row['c']+1
    a=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST)).copy()
    cov=F.interpolate(torch.from_numpy(a.astype(np.float32))[None,None],(64,64),mode='area')[0,0].numpy()
    fg=cov.ravel()>=.5
    partitions,_=m.ward_partitions(data['reference']['projected_native'].numpy(),(64,64),(8,),~fg)
    regions=[np.flatnonzero(fg)]+partitions[8]
    region=regions[5];ids=data['reference']['patch_ids'][region]
    state.update(state='DIAGNOSING',reference_region_sizes=list(map(len,regions)),failed_region_index=5,
                 failed_region_tokens=len(region),layers=[]);write()
    print(json.dumps({'regions':state['reference_region_sizes'],'failed_region_tokens':len(region)}),flush=True)
    def err(a,b):return float((a-b).abs().max())
    def pad_hidden(h,minimum,multiple):
        target=max(minimum,((len(h)+multiple-1)//multiple)*multiple)
        return F.pad(h,(0,0,0,target-len(h)))
    with torch.inference_mode():
        for layer,block in enumerate(model.blocks[20:24]):
            full=torch.stack([data[role]['h20'] if layer==0 else data[role]['native_states'][layer-1]
                              for role in ('reference','query')])
            native={};handles=[]
            for label,module in [('norm1',block.norm1),('qkv',block.attn.qkv),('proj',block.attn.proj),
                                 ('norm2',block.norm2),('mlp',block.mlp)]:
                def hook(module,inp,output,label=label):native[label]=output.detach().clone()
                handles.append(module.register_forward_hook(hook))
            original=F.scaled_dot_product_attention
            def sdpa(q,k,v,**kwargs):
                value=original(q,k,v,**kwargs);native['sdpa']=value.detach().clone();return value
            F.scaled_dot_product_attention=sdpa
            try:native_output=block(full,rope=data['reference']['rope'][layer])
            finally:
                F.scaled_dot_product_attention=original
                for h in handles:h.remove()
            q,k,v=data['reference']['qkv'][layer]
            expected_output=data['reference']['native_states'][layer]
            nnative=full.shape[1];head=int(block.attn.num_heads)
            native_v=native['qkv'].reshape(2,nnative,3,head,-1).permute(2,0,3,1,4)[2,0]
            report={'block':21+layer,'full_native_state_error':err(native_output[0],expected_output),
                    'full_native_V_vs_cache_error':err(native_v,v),'pointwise':{},'query_tiles':{}}
            hidden=full[0,ids]
            native_message=native['sdpa'][0,:,ids]
            for name,minimum,multiple in [('old_min64_no_round',64,1),('multiple64_min64',64,64),
                                         ('multiple64_min256',256,64),('multiple256_min256',256,256)]:
                padded=pad_hidden(hidden,minimum,multiple);length=len(hidden)
                norm1=block.norm1(padded[None]);qkv=block.attn.qkv(norm1)
                values=qkv.reshape(1,len(padded),3,head,-1).permute(2,0,3,1,4)[2,0,:,:length]
                joined=F.pad(native_message,(0,0,0,len(padded)-length)).transpose(0,1).reshape(1,len(padded),-1)
                projected=block.attn.proj_drop(block.attn.proj(block.attn.norm(joined)))
                residual=padded[None]+block.drop_path1(projected*block.gamma_1)
                norm2=block.norm2(residual);mlp=block.mlp(norm2)
                finish=residual+block.drop_path2(mlp*block.gamma_2)
                report['pointwise'][name]={'rows':len(padded),'norm1':err(norm1[0,:length],native['norm1'][0,ids]),
                     'V':err(values,v[:,ids]),'proj':err(projected[0,:length],native['proj'][0,ids]),
                     'norm2':err(norm2[0,:length],native['norm2'][0,ids]),'MLP':err(mlp[0,:length],native['mlp'][0,ids]),
                     'finish_with_exact_native_SDPA_message':err(finish[0,:length],expected_output[ids])}
            for chunk in (64,256):
                for pad_queries in (False,True):
                    pieces=[]
                    for start in range(0,len(ids),chunk):
                        query=q[:,ids[start:start+chunk]];actual=query.shape[1]
                        if pad_queries and actual<chunk:query=F.pad(query,(0,0,0,chunk-actual))
                        value=original(query[None],k[None],v[None],dropout_p=0.,scale=block.attn.scale)[0,:,:actual]
                        pieces.append(value)
                    report['query_tiles'][f'chunk{chunk}_pad{pad_queries}']=err(torch.cat(pieces,1),native_message)
            state['layers'].append(report);state['elapsed_seconds']=time.perf_counter()-started;write()
            print(json.dumps(report),flush=True)
    state.update(state='COMPONENT_DIAGNOSIS_COMPLETE',elapsed_seconds=time.perf_counter()-started);write()


if __name__=='__main__':main()
