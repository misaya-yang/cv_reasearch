"""Small same-information FoRIS control, one frozen query per class.

Existing weights/images only; FP32 native FoRIS encoder. Cache only current
episode features in RAM. No feature dataset, GT donor masks or method tuning.
"""
import os,sys,json,hashlib,time,argparse
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
FORIS='/root/autodl-tmp/demo8_local_verification/foris_source'
sys.path[:0]=['/root/demo4_cache/env',FORIS,'/root/autodl-tmp/demo4']
import torch,numpy as np,torch.nn.functional as F
import models.foris as foris_mod
from models.foris import FoRIS
from icx.common import TimmDINOv3,coco_load,iu
from PIL import Image

def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)

def main(args):
 source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 doc=json.load(open(source/'manifest.json'));selected=[];seen=set()
 for r in doc['episodes']:
  if r['class_id'] not in seen:selected.append(r);seen.add(r['class_id'])
 atomic(out/'manifest.json',dict(episodes=selected,source=str(source),scope='First frozen query per class, identical supports and clean/mixed pools to open_pool_v1; 20-query strong-control pilot only'))
 # A 16-reference native head exceeded the original artificial .3 quota
 # despite >21GiB physical free memory. Keep a reserve for other sessions.
 free,total=torch.cuda.mem_get_info();fraction=min(.65,max(.3,free/total-.15))
 torch.cuda.set_per_process_memory_fraction(fraction);torch.backends.cuda.matmul.allow_tf32=False
 original_cluster=foris_mod.agglomerative_clustering;memo={}
 def cluster(x,tau=.6):
  key=(tuple(x.shape),str(x.dtype),tau,hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
  if key not in memo:memo[key]=original_cluster(x,tau=tau)
  return memo[key]
 foris_mod.agglomerative_clustering=cluster
 class Cached(FoRIS):
  def _extract_features(self,imgs):return torch.stack([self.feats[i] for i in self.order])[None]
 model=Cached(TimmDINOv3().eval(),image_size=1024,mask_refiner='bilinear',resize_to_orig_size=False).cuda().eval()
 for p in model.parameters():p.requires_grad=False
 model_transform=model._transform
 state=dict(state='RUNNING',pid=os.getpid(),completed=0,total=len(selected));atomic(out/'status.json',state);rows=[]
 for item in selected:
  path=out/f"episode{item['episode']:04d}.json"
  if path.exists():rows.append(json.load(open(path)));state['completed']+=1;continue
  names=list(dict.fromkeys([item['support'],item['query']]+item['clean']+item['mixed']));lookup={n:i for i,n in enumerate(names)}
  X=torch.stack([model_transform(Image.open(Path(doc['data'])/n).convert('RGB')) for n in names]).cuda()
  start=time.perf_counter()
  with torch.inference_mode():model.feats=torch.cat([model.encoder.get_intermediate_layers(X[i:i+1],n=1,reshape=True)[0] for i in range(len(X))])
  encoder_seconds=time.perf_counter()-start
  _,support_mask=coco_load(doc['data'],item['support'],item['class_id'])
  g0=F.interpolate(support_mask[None,None].float().cuda(),size=(1024,1024),mode='nearest')[0,0]>.5
  def pred(t,refs,masks):
   keep=[i for i,m in enumerate(masks) if m.any()]
   if not keep:return torch.zeros(1024,1024,dtype=torch.bool,device='cuda')
   refs=[refs[i] for i in keep];masks=[masks[i] for i in keep];model.order=refs+[t]
   try:
    with torch.inference_mode():return model.predict(X[refs],torch.stack(masks),X[t])
   except RuntimeError as e:
    if 'No foreground' in str(e):return torch.zeros(1024,1024,dtype=torch.bool,device='cuda')
    raise
  memo.clear();q=lookup[item['query']];P={j:pred(j,[0],[g0]) for j in range(1,len(names))}
  predictions={'foris_1shot':P[q]}
  for condition in ['clean','mixed']:
   donors=[lookup[n] for n in item[condition]];predictions['foris_'+condition+'_naive']=pred(q,[0]+donors,[g0]+[P[j] for j in donors])
  # All predictions completed before query annotation is loaded for scoring.
  _,qm=coco_load(doc['data'],item['query'],item['class_id']);truth=F.interpolate(qm[None,None].float().cuda(),size=(1024,1024),mode='nearest')[0,0]>.5
  metrics={}
  for name,prediction in predictions.items():
   I,U=iu(prediction,truth);metrics[name]=dict(intersection=I,union=U)
  old=json.load(open(source/f"episode{item['episode']:04d}.json"))
  for name in ['native_bf16','1shot','clean_agree','mixed_agree']:metrics['insid3_'+name]=old['metrics'][name]
  row=dict(episode=item['episode'],class_id=item['class_id'],metrics=metrics,encoder_calls=len(names),encoder_seconds=encoder_seconds,
   contract='Native FoRIS FP32 encoder+unchanged head, existing weights. Reuses current episode features only. No donor GT, query GT only post-inference score; same M and identities as INSID3 controls.')
  atomic(path,row);rows.append(row);state['completed']+=1;atomic(out/'status.json',state);print(json.dumps(state),flush=True)
  del X,P,predictions;model.feats=None;memo.clear();torch.cuda.empty_cache()
 summary={name:100*float(np.mean([r['metrics'][name]['intersection']/max(r['metrics'][name]['union'],1) for r in rows])) for name in rows[0]['metrics']}
 atomic(out/'report.json',dict(state='COMPLETED',episodes=len(rows),classes=len(rows),summary=summary,records=rows,scope='One episode/class pilot, not full fold mIoU/SOTA or independent test; native precision differences explicit, matched image identities and masks.'))
 state['state']='COMPLETED';atomic(out/'status.json',state);print(json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 try:main(a)
 except Exception:
  import traceback
  pp=Path(a.out);pp.mkdir(parents=True,exist_ok=True);atomic(pp/'status.json',dict(state='ERROR',pid=os.getpid(),error=traceback.format_exc()));raise
