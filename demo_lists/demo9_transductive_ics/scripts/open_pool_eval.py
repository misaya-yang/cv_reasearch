"""Fixed clean/mixed pools; target/donor GT never enters propagation."""
import os,sys,json,time,hashlib,argparse,math
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
sys.path[:0]=['/root/demo4_cache/env','/root/autodl-tmp/demo4',str(Path(__file__).resolve().parents[1])]
from icx.common import *
from icx.fast import ClassSet
from tics.imageset import ImageSet
from tics.propagate import one_shot,propagate
from utils.data import load_image,load_mask,downsample_mask

def atomic(path,d):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(path)

def image_data(cache,name):return torch.load(cache/(hashlib.sha256(name.encode()).hexdigest()[:20]+'.pt'),weights_only=False)

def prepare(row,pool,doc,cache):
 donors=[n for n in row['mixed'] if n not in row['negative']] if pool=='present_matched' else row[pool]
 names=[row['support'],row['query']]+donors;features=[image_data(cache,n) for n in names];S=features[0]['S'];h=int(math.sqrt(len(features[0]['fq'])))
 _,mask=coco_load(doc['data'],row['support'],row['class_id']);g0=load_mask(mask,S,DEV)[0]
 support64=downsample_mask(g0[None,None],h,h) if g0.any() else torch.zeros(h,h,dtype=torch.bool,device=DEV)
 # Build an inference object containing ONLY the provided support annotation.
 # The query/donor GT slots are zeros, even though ImageSet has scoring helpers.
 g64=torch.zeros(len(names),h,h,dtype=torch.bool);g64[0]=support64.cpu()
 bits=np.zeros((len(names),S*S//8),dtype=np.uint8);bits[0]=np.packbits(g0.cpu().numpy().reshape(-1))
 d=dict(c=row['class_id'],names=names,fq=torch.stack([f['fq'] for f in features]),lab=torch.stack([f['lab'] for f in features]),Po=[f['Po'] for f in features],gt64=g64,gt_bits=bits,S=S)
 s=ImageSet(d);assert not s.gt64[1:].any() and not s.gt[1:].any()
 return s,d,g0

def score(pred,truth,s):
 pp=s.up(pred);I,U=iu(pp,truth)
 return dict(intersection=I,union=U,predicted_pixels=float(pp.sum()),gt_pixels=float(truth.sum()),patch_mask_bits=np.packbits(pred.cpu().numpy()).tolist())

def main(args):
 out=Path(args.out);doc=json.load(open(out/'manifest.json'));cache=out/'cache';torch.backends.cuda.matmul.allow_tf32=False
 # Frozen before later-fold scoring; old live controller need not be restarted.
 args.matched_positives=args.matched_positives or doc['fold']>=1
 atomic(out/'evaluation_contract.json',dict(matched_positive_control=args.matched_positives,reason='Fewer positives versus harmful negative evidence; added after fold0, before this fold outputs; no method parameter tuning'))
 state=dict(state='RUNNING',pid=os.getpid(),fold=doc['fold'],completed=0,total=len(doc['episodes']));atomic(out/'eval_status.json',state)
 # Compare both compact readers exactly, then compact single-shot vs released.
 checks=[];model=build_model();native_extract=model._extract_features
 def common_precision_extract(imgs):
  # Historical compact demo9 uses FP32 post-encoder math; match this declared
  # precision and batch1 export, while scoring original BF16 separately.
  return torch.cat([native_extract(imgs[:,i:i+1]).float() for i in range(imgs.shape[1])],dim=1)
 if not (out/'interface.json').exists():
  for row in doc['episodes'][:3]:
   s,d,g0=prepare(row,'clean',doc,cache);old=ClassSet(d);a=s.predict(1,[0],[s.gt64[0]]);b=old.predict(1,[0],[old.gt64[0]])
   if not torch.equal(a,b):raise RuntimeError('New ImageSet and old ClassSet differ on fixed single-shot')
   if g0.any():
    xs=[load_image(Image.open(Path(doc['data'])/n).convert('RGB'),model._transform,DEV)[0] for n in [row['support'],row['query']]]
    model._extract_features=native_extract
    with torch.inference_mode():native=model.predict_mask(xs[0],g0[None],xs[1]).reshape(s.S,s.S)>0.5
    model._extract_features=common_precision_extract
    with torch.inference_mode():ref=model.predict_mask(xs[0],g0[None],xs[1]).reshape(s.S,s.S)>0.5
    comp=s.up(a);I,U=iu(ref,comp);agreement=I/max(U,1.)
    ni,nu=iu(native,comp);native_agreement=ni/max(nu,1.)
    if agreement<.99:
     atomic(out/'interface_error.json',dict(episode=row['episode'],released_compact_mask_iou=agreement,reader_old_new_bitwise=True,stage='before any task scoring'))
     raise RuntimeError(f'Compressed reader disagrees with released masks by >1%: IoU={agreement}; preserve before science')
   else:agreement=None;native_agreement=None
   checks.append(dict(episode=row['episode'],cached_old_new_bitwise=True,official_fp32_decoder_compact_mask_iou=agreement,original_bf16_decoder_compact_mask_iou=native_agreement,source='Native encoder BF16, batch1 export; common FP32 normalization/debias/decoder with FP16 complement coordinates. Original BF16 decoder separately scored, not claimed identical.'))
  atomic(out/'interface.json',dict(state='PASSED_COMMON_FP32_DECODER',checks=checks,gt_inference='Only labelled support mask; all other GT slots zero',tf32=False,scoring='Square1024 like earlier demo9 controls; not original-resolution paper reproduction'))
 model._extract_features=native_extract
 records=[]
 for row in doc['episodes']:
  path=out/f"episode{row['episode']:04d}.json"
  if path.exists():records.append(json.load(open(path)));state['completed']+=1;continue
  # Query ground truth is loaded ONLY for this separate scoring function.
  _,qmask=coco_load(doc['data'],row['query'],row['class_id']);truth=load_mask(qmask,1024,DEV)[0]
  record=dict(episode=row['episode'],class_id=row['class_id'],support=row['support'],query=row['query'],metrics={},trust={},source_presence_hidden=True)
  # Preserve the genuine original BF16 decoder as a separate strong control.
  _,mask=coco_load(doc['data'],row['support'],row['class_id']);g0=load_mask(mask,1024,DEV)[0]
  if g0.any():
   xs=[load_image(Image.open(Path(doc['data'])/n).convert('RGB'),model._transform,DEV)[0] for n in [row['support'],row['query']]]
   with torch.inference_mode():native=model.predict_mask(xs[0],g0[None],xs[1]).reshape(1024,1024)>0.5
  else:native=torch.zeros_like(truth)
  I,U=iu(native,truth);record['metrics']['native_bf16']=dict(intersection=I,union=U,predicted_pixels=float(native.sum()),gt_pixels=float(truth.sum()))
  conditions=['clean','mixed']+(['present_matched'] if args.matched_positives else [])
  for condition in conditions:
   s,d,g0=prepare(row,condition,doc,cache);J=list(range(1,s.n));P1=one_shot(s,J)
   if condition=='clean':record['metrics']['1shot']=score(P1[1],truth,s)
   others=[j for j in J if j!=1];start=time.perf_counter()
   pred=s.predict(1,[0]+others,[P1[0]]+[P1[j] for j in others])
   record['metrics'][condition+'_naive']=dict(score(pred,truth,s),cached_seconds=time.perf_counter()-start)
   policies=[('none','all'),('agree','tophalf')]+([] if condition=='present_matched' else [('roundtrip','thr:0.5')])
   for trust,rule in policies:
    track={};start=time.perf_counter();rounds=propagate(s,J,rounds=4,k=5,trust=trust,rule=rule,weighted=False,P1=P1,track=track)
    key=condition+'_'+trust;record['metrics'][key]=dict(score(rounds[-1][1],truth,s),cached_seconds=time.perf_counter()-start)
    # Dataset absence metadata is used here AFTER inference for diagnostics.
    record['trust'][key]=[{s.names[j]:dict(score=float(rel[j]),chosen=bool(ok[j]),target_absent=s.names[j] in row['negative']) for j in J} for rel,ok in zip(track.get('rel',[]),track.get('ok',[]))]
   record[condition+'_donors']=[n for n in row['mixed'] if n not in row['negative']] if condition=='present_matched' else row[condition]
   record['negative_donors']=row['negative']
   del s,d;torch.cuda.empty_cache()
  atomic(path,record);records.append(record);state['completed']+=1;atomic(out/'eval_status.json',state)
  if state['completed']%10==0:print(json.dumps(state),flush=True)
 sums={}
 for r in records:
  for name,v in r['metrics'].items():
   s=sums.setdefault(name,{}).setdefault(str(r['class_id']),[0.,0.]);s[0]+=v['intersection'];s[1]+=v['union']
 summary={name:100*float(np.mean([i/max(u,1) for i,u in classes.values()])) for name,classes in sums.items()}
 report=dict(state='COMPLETED',fold=doc['fold'],episodes=len(records),class_sums=sums,miou=summary,records=records,matched_positive_control=args.matched_positives,
  boundaries='Development robustness study; first5 standard episodes per class, fixed M up to15; mixed replaces half donors, no target absence input to algorithm, no new learned method. Cached timings exclude encoder/cache/CPU image load.')
 atomic(out/'report.json',report);state['state']='COMPLETED';atomic(out/'eval_status.json',state);print(json.dumps(summary),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--matched-positives',action='store_true');a=ap.parse_args()
 try:main(a)
 except Exception:
  import traceback
  p=Path(a.out);atomic(p/'eval_status.json',dict(state='ERROR',pid=os.getpid(),error=traceback.format_exc()));raise
