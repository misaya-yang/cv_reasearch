"""One synthetic-color composition/caching check, CPU; no real encoder or task score."""
import argparse,json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tics.ten_observation import run_m05,naive_m05,run_m10,naive_m10,METHODS


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
 budget=SimpleNamespace(max_extra_b2=8,extra_b2_calls=0,actual_b2_calls=0,native_b2_calls=1,cache={})
 class Ctx:
  def __init__(self,S,M,Q):
   self.budget=budget;self.support_rgb=S;self.query_rgb=Q;self.reference_mask_original=M.bool();self.work_hw=(128,128)
   self.reference_mask=F.interpolate(M[None,None].float(),self.work_hw,mode='nearest')[0,0].bool()
   self.reference_coverage=F.avg_pool2d(self.reference_mask[None,None].float(),16,16)[0,0]
   def read(im):
    x=torch.as_tensor(np.asarray(im.resize((128,128),Image.Resampling.BILINEAR)).copy()).float()/255
    red=F.avg_pool2d(x.permute(2,0,1)[None],16,16)[0,0]
    yy,xx=torch.meshgrid(torch.arange(8).float()/8,torch.arange(8).float()/8,indexing='ij')
    features=torch.stack((red,1-red,.01*yy,.01*xx),-1).reshape(64,4)
    return F.normalize(features,dim=1),x[:,:,0]>.5,red
   self.ref_tokens,_,_=read(S);self.query_tokens,mask,score=read(Q)
   self.native=dict(mask=mask,score=score,stages={'pre_refinement_mask':mask})
  def finalize_work_mask(self,m):return m.clone().bool()
  def new_view(self,S,M,Q,key,metadata=None):
   ident=hashlib.sha256(S.tobytes()+Q.tobytes()+M.cpu().numpy().tobytes()).hexdigest()
   if key in budget.cache:
    old,child=budget.cache[key];assert old==ident;return child
   assert budget.extra_b2_calls<budget.max_extra_b2;budget.extra_b2_calls+=1
   child=Ctx(S,M,Q);budget.cache[key]=(ident,child);return child
 s=np.zeros((128,128,3),np.uint8);q=s.copy()
 for y in (0,104):
  for x in (0,104):s[y:y+24,x:x+24,0]=255
 for y in (32,72):
  for x in (32,72):q[y:y+24,x:x+24,0]=255
 ctx=Ctx(Image.fromarray(s),torch.from_numpy(s[:,:,0]>0),Image.fromarray(q))
 main5=run_m05(ctx);n5=budget.extra_b2_calls;control5=naive_m05(ctx);assert budget.extra_b2_calls==n5
 main10=run_m10(ctx);n10=budget.extra_b2_calls;control10=naive_m10(ctx);assert budget.extra_b2_calls==n10
 assert all(z['mask'].shape==(128,128) and z['mask'].dtype==torch.bool for z in (main5,control5,main10,control10))
 assert n5<=2 and n10-n5<=3 and n10<=5
 assert all('function'in m and 'naive'in m and len(m['card'])==4 for m in METHODS)
 out=dict(state='PASSED_CPU_COMPOSITION',composition_cases=1,m05_extra_view_requests=n5,m10_extra_view_requests=n10-n5,total_extra_view_requests=n10,global_cap=8,naive_repeated_calls_added_views=0,actual_b2_counter_reported=budget.actual_b2_calls,actual_counter_note='tiny synthetic factory does not instrument a native encoder; actual0 is not a production count',production_encoder_calls=0,CUDA_initialized=torch.cuda.is_initialized(),task_IoU_measured=False,m05_audit=main5['audit'],m10_audit=main10['audit'],elapsed_seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).with_name('ten_observation.py').read_bytes()).hexdigest())
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=1));print(json.dumps(out))
if __name__=='__main__':main()
