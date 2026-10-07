"""Exactly five-case CPU timing of the two PLAN-authorized zero-encoding repairs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

A=Path('/root/autodl-tmp/evidence_bench_step3_20261007_25142/A_six_levels_v6')
sys.path.insert(0,str(A/'source/scripts'))
import bench_evidence as bench
from ics.methods import rcg
torch.set_num_threads(2);torch.set_num_interop_threads(1)
OUT=Path('/root/autodl-tmp/evidence_bench_step3_20261007_25142/C_first5_v1')
ANN=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')
MAN=Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/confirm1200_conditional_v1/inputs/manifest.json')
OUT.mkdir(exist_ok=False)
fresh=json.loads((A/'ordered_rows.json').read_text());inputs={r['key']:r for r in json.loads(MAN.read_text())}
report=dict(n=5,workers_for_projected600=16,threads_per_worker=2,
    only_arms=['card1','card1_global_control','card2_pure_BG_control'],
    additional_reference_encodings=0,full_attention_card2_executed=False,query_GT_read=False,
    source='same frozen A evidence/RCG helpers, original annotations and cached native Part2 reference-only arrays',
    cases=[])

@torch.inference_mode()
def fields_for(row,q,r,cov,aux):
    x=F.normalize(torch.from_numpy(q),dim=1);ref=F.normalize(torch.from_numpy(r),dim=1)
    p=torch.from_numpy(aux['fg_prototypes']);b=torch.from_numpy(aux['mu_bg_hard'])
    fields={};info={}
    c=p@b
    for arm,value in (('card1',c),('card1_global_control',torch.full_like(c,c.mean()))):
        if len(p)==1:field=aux['s2'].copy()
        else:
            changed=p-.55*(b[None,:]-value[:,None]*p)
            field=(.07*torch.logsumexp((x@changed.T)/.07,dim=1)).reshape(64,64).numpy()
        fields[arm]=field
    fg,mask_info=bench.native_foreground(row,ANN,cov)
    bg=torch.nonzero(~fg,as_tuple=False).flatten()
    mean_fg=torch.from_numpy(aux['mu_fg'])
    hard=bg[(ref[bg]@mean_fg).topk(max(1,int(.2*len(bg)))).indices] if len(bg) else bg
    path=ANN/Path(row['support']).with_suffix('.png')
    original=(np.asarray(Image.open(path))==row['c']+1).astype(np.float32)
    support=np.asarray(Image.fromarray(original,mode='F').resize((1024,1024),Image.Resampling.BILINEAR))
    safe=F.max_pool2d(torch.from_numpy(support.copy())[None,None],16)[0,0].reshape(-1)==0
    pure=(torch.from_numpy(cov.reshape(-1))==0)&safe
    chosen=hard[pure[hard]]
    if len(chosen):
        mean_bg=F.normalize(ref[chosen].mean(0),dim=0)
        residual=mean_bg-(mean_bg@mean_fg)*mean_fg
        negative=residual/residual.norm().clamp_min(1e-8)
        field=(.07*torch.logsumexp((x@p.T)/.07,dim=1)-.55*(x@negative)).reshape(64,64).numpy()
    else:field=aux['s2'].copy()
    fields['card2_pure_BG_control']=field
    info.update(K=len(p),c_mean=float(c.mean()),c_std=float(c.std(unbiased=False)),
        native_hard_BG_count=len(hard),pure_RGB_supported_BG_count=int(pure.sum()),J_count=len(chosen),
        pure_support='Original MR float PIL BILINEAR resize, tile max==0 and native cov==0; no added dilation/threshold',
        fallback_card1=len(p)==1,fallback_card2=len(chosen)==0,reference_mask=mask_info)
    return fields,info

for entry in fresh[:5]:
    row=inputs[entry['key']];start=time.monotonic();cpu=time.process_time()
    q,r,provenance=bench.features(Path(row['feature_export']));packet,packet_hash=bench.packet(Path(row['packet_export']))
    with np.load(A/'acceptance2_fields'/(row['key']+'.npz'),allow_pickle=False) as z:aux={k:z[k].copy() for k in z.files}
    fields,info=fields_for(row,q,r,packet['cov'],aux)
    output={};timings={}
    for name,field in fields.items():
        wall=time.monotonic();proc=time.process_time()
        graph,solver=rcg.predict(q,r,packet['cov'],field,device='cpu')
        output[name+'.evidence']=field;output[name+'.rcg']=graph
        timings[name]=dict(graph_wall_seconds=time.monotonic()-wall,graph_CPU_seconds=time.process_time()-proc,solver=solver)
    np.savez_compressed(OUT/(row['key']+'.npz'),**output)
    report['cases'].append(dict(key=row['key'],total_wall_seconds=time.monotonic()-start,
        total_CPU_seconds=time.process_time()-cpu,info=info,features=provenance,
        packet_sha256=packet_hash,timings=timings))
report['mean_all3arms_wall_seconds_per_case']=float(np.mean([r['total_wall_seconds'] for r in report['cases']]))
report['projected_all3arms600_seconds_at16x2']=report['mean_all3arms_wall_seconds_per_case']*600/16
report['cost_gate_pass']=report['projected_all3arms600_seconds_at16x2']<=1200
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)
