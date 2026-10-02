"""Own, resumable image cache; uses existing INSID3/DINO/COCO read-only."""
import os,sys,json,hashlib,time,shutil,argparse
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
sys.path[:0]=['/root/demo4_cache/env','/root/autodl-tmp/demo4']
from icx.common import *
from icx.fast import cluster_protos
from utils.data import load_image
from utils.clustering import agglomerative_clustering

def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def plan(fold,out):
 eps,classes,base=coco_episodes(fold,400,shot=1,seed=0)
 meta=pickle.load(open(f'{base}/splits/val/fold{fold}.pkl','rb'))
 positive={int(c):set(map(str,meta[c])) for c in classes}
 rng=np.random.default_rng(0);negative_rng=np.random.default_rng(2050)
 seen_class={};selected=[]
 for e,(c,query,refs) in enumerate(eps):
  # Match original clean-pool RNG consumption for every standard episode.
  support=refs[0];cand=[];seen=set()
  for cc,q,_ in eps:
   if cc==c and q not in (support,query) and q not in seen:seen.add(q);cand.append(q)
  pool=list(rng.permutation(cand)[:15]);count=seen_class.get(c,0);seen_class[c]=count+1
  if count>=5:continue
  neg=sorted({q for _,q,_ in eps if q not in positive[c] and q not in (support,query) and q not in pool})
  k=len(pool)//2
  if len(neg)<k:raise ValueError('Insufficient target-absent donor pool; no silent pool-size change')
  negative=list(negative_rng.permutation(neg)[:k]);mixed=pool[:len(pool)-k]+negative
  selected.append(dict(episode=e,class_id=c,support=support,query=query,clean=pool,mixed=mixed,negative=negative))
 if len(selected)!=100:raise ValueError('Need five fixed standard episodes for each of twenty classes')
 names=sorted({n for row in selected for n in [row['support'],row['query']]+row['clean']+row['mixed']})
 doc=dict(fold=fold,standard_episode_seed=0,negative_seed=2050,episodes=selected,names=names,
  protocol='First five standard episodes per class among first400; same support/query/actual donor count; replace floor(M/2) target-present donors by target-absent donors. Absence from COCO split membership only for sampling, hidden from inference. This cohort is development evidence.',
  data=base,feature_contract='Existing INSID31024 native BF16 encoder, FP32 normalization/debias like historical compact demo9, orthogonal-complement coordinates FP16, original cluster prototypes FP16; original native BF16 decoder separately scored, not claimed identical; not PCA learned from query GT')
 atomic(out/'manifest.json',doc);return doc

def main(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
 doc=json.load(open(out/'manifest.json')) if (out/'manifest.json').exists() else plan(args.fold,out)
 state=dict(state='EXTRACTING',pid=os.getpid(),fold=args.fold,total=len(doc['names']),completed=0)
 atomic(out/'cache_status.json',state)
 torch.backends.cuda.matmul.allow_tf32=False;model=build_model();U=model.positional_basis.float()
 Q=torch.linalg.svd(U,full_matrices=True)[0][:,U.shape[1]:].contiguous()
 for name in doc['names']:
  dest=cache/(hashlib.sha256(name.encode()).hexdigest()[:20]+'.pt')
  if not dest.exists():
   if shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('8GiB free-space protection; no automatic deletion of others')
   x=load_image(Image.open(Path(doc['data'])/name).convert('RGB'),model._transform,DEV)[0]
   with torch.inference_mode():
    f=F.normalize(model._extract_features(x[None]).float(),p=2,dim=2)[0,0]
    C,h,w=f.shape;f=f.reshape(C,-1).T;l=agglomerative_clustering(f,model.tau)
    fd=F.normalize(f-(f@U)@U.T,dim=-1)
    d=dict(name=name,fq=(fd@Q).half().cpu(),lab=l.short().cpu(),Po=cluster_protos(f,l,int(l.max())+1)[0].half().cpu(),S=model.image_size)
   tmp=dest.with_suffix('.tmp');torch.save(d,tmp);tmp.replace(dest)
  state['completed']+=1;atomic(out/'cache_status.json',state)
  if state['completed']%25==0:print(json.dumps(state),flush=True)
 state.update(state='COMPLETED',stored_bytes=sum(p.stat().st_size for p in cache.glob('*.pt')));atomic(out/'cache_status.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 try:main(a)
 except Exception:
  import traceback
  p=Path(a.out);p.mkdir(parents=True,exist_ok=True);atomic(p/'cache_status.json',dict(state='ERROR',pid=os.getpid(),error=traceback.format_exc()));raise
