"""Bounded CPU-only geometry/operator checks, never a task accuracy test."""
import argparse,json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from tics.ten_observation import robust_similarity,shift_work,valid_phase,patch_mean,patch_adjoint,inverse_coverage,fit_iso,compose_ROI


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1);begin=time.monotonic();checks=[]
 def check(n,fn):fn();checks.append(n)
 def geo():
  s=np.array([[0,0],[40,0],[0,40],[40,40],[20,70],[60,70],[90,20],[70,10]],float);A=np.array([[1.5,-.3],[.3,1.5]]);q=s@A.T+[80,110];q[-2:]+=np.array([[200,300],[-250,350]])
  x=robust_similarity(s,q,patch=8);assert x is not None and x['inliers'].sum()==6;assert np.max(np.abs(x['A']-A))<1e-10
 check('robust_similarity6inliers2outliers',geo)
 check('under3_correspondences_abstain',lambda:exec("assert robust_similarity([[0,0],[1,1]],[[0,0],[2,2]]) is None",globals(),locals()))
 def translation():
  s=np.array([[0,0],[32,0],[0,32],[32,32]],float);q=.35*s+[200,300];x=robust_similarity(s,q,patch=4);assert abs(x['scale']-.35)<1e-12 and np.max(np.abs(x['t']-[200,300]))<1e-10
 check('scale_is_from_correspondences_not_mask_area',translation)
 def shift():
  img=np.tile(np.arange(64,dtype=np.uint8),(64,1));z=shift_work(img,8,0);assert z[0,0]==8 and z[0,-1]==63
  v=valid_phase((64,64),(4,4),8,0,'cpu');assert int(v.sum())==12 and not v[:,-1].any()
 check('phase_physical_origin_and_invalid_padding',shift)
 def roi():
  b=torch.ones(64,64,dtype=torch.bool);pre=b.clone();ctx=SimpleNamespace(work_hw=(64,64),native={'mask':b,'stages':{'pre_refinement_mask':pre}},finalize_work_mask=lambda m:m)
  nv=SimpleNamespace(native={'stages':{'pre_refinement_mask':torch.zeros(64,64,dtype=torch.bool)}})
  got=compose_ROI(ctx,nv,[16,16,32,32]);assert not got[16:32,16:32].any() and torch.equal(got[:16],b[:16]) and int(got.sum())==64*64-16*16
 check('crop_outside_native_preserved',roi)
 def const():
  m=torch.full((16,16),.7,dtype=torch.float64)
  for phase in ((0,0),(1,0),(0,1),(1,1)):
   z=patch_mean(m,phase);h=(16-phase[0])//2;w=(16-phase[1])//2;assert torch.max(torch.abs(z[:h,:w]-.7))<1e-12
 check('fourphase_valid_patch_mean_constant',const)
 def adjoint():
  g=torch.Generator().manual_seed(0);m=torch.randn(16,16,generator=g,dtype=torch.float64);v=torch.randn(8,8,generator=g,dtype=torch.float64)
  for phase in ((0,0),(1,0),(0,1),(1,1)):
   assert abs(float((patch_mean(m,phase)*v).sum()-(m*patch_adjoint(v,m.shape,phase)).sum()))<1e-10
 check('fourphase_exact_adjoint_identity',adjoint)
 def null():
  x=.5+.25*((torch.arange(16)%2)*2-1).double()[None,:].repeat(16,1)
  for phase in ((0,0),(1,0),(0,1),(1,1)):
   z=patch_mean(x,phase);h=(16-phase[0])//2;w=(16-phase[1])//2;assert torch.max(torch.abs(z[:h,:w]-.5))<1e-12
 check('counterexample_Nyquist_not_unique',null)
 def inverse():
  truth=torch.zeros(16,16);truth[4:12,5:11]=1;phases=[(0,0),(1,0),(0,1),(1,1)];obs=[patch_mean(truth,p) for p in phases];weights=[]
  for p in phases:
   w=torch.zeros(8,8);w[:(16-p[0])//2,:(16-p[1])//2]=1;weights.append(w)
  m,avg=inverse_coverage(obs,weights,phases,torch.zeros(16,16,3),torch.zeros(16,16),torch.zeros(16,16),lam=0,eta=0,steps=600)
  assert ((m-truth)**2).mean()<((avg-truth)**2).mean();assert m.min()>=0 and m.max()<=1
 check('known_box_inverse_improves_over_same_observation_average',inverse)
 def iso():
  x,y=fit_iso([0,1,2,3],[0,.8,.4,1]);assert np.all(np.diff(y)>=0) and abs(y[1]-.6)<1e-12
 check('reference_only_monotone_PAVA_contract',iso)
 code=Path(__file__).with_name('ten_observation.py')
 out=dict(state='PASSED_CPU_GEOMETRY_OPERATOR_CHECKS',checks=checks,count=len(checks),CUDA_initialized=torch.cuda.is_initialized(),encoder_calls=0,task_IoU_measured=False,source_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-begin)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=1));print(json.dumps(out))
if __name__=='__main__':main()
