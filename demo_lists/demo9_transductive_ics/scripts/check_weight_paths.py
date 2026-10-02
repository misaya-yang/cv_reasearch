"""Independent runtime references for the supplied opt-in vote-weight API."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,importlib.util,json
from pathlib import Path
sys.path.insert(0,'/root/demo4_cache/env')
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('original_fast','/root/autodl-tmp/demo4/icx/fast.py')
new=load('patched_fast',ROOT/'isolated_copy/demo_lists/demo4_incontext_seg/icx/fast.py')
torch.manual_seed(2052);n,P,C=4,16,8
f=torch.nn.functional.normalize(torch.randn(n,P,C),dim=-1);labels=torch.tensor([[i//4 for i in range(P)]]*n)
po=[torch.nn.functional.normalize(torch.stack([f[j,k*4:(k+1)*4].mean(0) for k in range(4)]),dim=-1) for j in range(n)]
d=dict(c=0,names=list('abcd'),fq=f,lab=labels,Po=po,gt64=torch.zeros(n,4,4,dtype=torch.bool),gt_bits=np.zeros((n,8),dtype=np.uint8),S=8)
old=base.ClassSet(d);cs=new.ClassSet(d)
masks=[torch.arange(P)%2==0,torch.zeros(P,dtype=torch.bool),torch.arange(P)%3==0];refs=[0,1,2]
checks={}
for branch in ['majority','pooled','pu','tri']:
 kw=dict(backward=branch,k=2)
 if branch=='tri':kw['known']=[torch.ones(P,dtype=torch.bool)]*3
 a=old.predict(3,refs,masks,**kw);b=cs.predict(3,refs,masks,vote_weights=None,**kw)
 checks['none_path_'+branch]=bool(torch.equal(a,b));assert checks['none_path_'+branch]
def votes(w,k=2):
 result=cs.predict(3,refs,masks,backward='pooled',k=k,vote_weights=w,return_parts=True)
 assert isinstance(result,tuple),'Fixture must actually trigger candidate/aggregation path'
 return result[1]['votes']
retained=[0,2];V=torch.stack([cs.nnv(3,j)[0] for j in retained]);L=torch.stack([masks[j][cs.nn(3,j)] for j in retained]).numpy().astype(float)
top=V.topk(2,dim=0).indices.numpy();selected=np.take_along_axis(L,top,axis=0)
for label,w in [('normal',[9.,100.,1.]),('zero',[0.,0.,0.]),('empty_index',[1.,1000.,3.]),('extreme',[1e300,2.,1e300])]:
 ww=np.asarray(w,dtype=np.float64)[retained];sw=ww[top];scale=sw.max(0);sw=sw/np.where(scale>0,scale,1);den=sw.sum(0)
 expected=np.divide((sw*selected).sum(0),den,out=selected.mean(0),where=den>0)
 error=float(np.max(np.abs(votes(w).numpy()-expected)));checks[label]=error;assert error<1e-12
assert torch.equal(votes([9.,0.,1.],k=1),votes([0.,0.,0.],k=1));checks['k1_vote_only_invariance']=True
assert torch.equal(votes([9.,0.,1.]),votes([90.,0.,10.]));checks['positive_scaling']=True
bad=[[1.,-1.,2.],[1.,float('nan'),2.],[1.,float('inf'),2.],[1.,2.],[[1.,2.,3.]]]
for w in bad:
 try:cs.predict(3,refs,masks,backward='pooled',vote_weights=w)
 except ValueError:pass
 else:raise AssertionError('Invalid weights not rejected before empty-reference filtering')
checks['invalid_weights_rejected']=len(bad)
try:cs.predict(3,refs,masks,backward='majority',vote_weights=[1.,2.,3.])
except ValueError:checks['nonpooled_rejected']=True
else:raise AssertionError('Nonpooled opt-in ignored')
out=ROOT/'results';out.mkdir(exist_ok=True);report=dict(state='PASSED_RUNTIME_NUMERICAL',checks=checks,scope='API/numerical causal paths only, not a real segmentation gain')
(out/'weight_paths_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
