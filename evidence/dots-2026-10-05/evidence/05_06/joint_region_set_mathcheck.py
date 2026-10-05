#!/usr/bin/env python3
"""Tiny exhaustive checks for the fixed joint region-set objective.
Synthetic algebra/correctness only: no claim of segmentation gain.
"""
import json
from pathlib import Path
import numpy as np

def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def make_problem(q,r,fg,labels):
 cosine=np.clip(q@r.T,-1,1);K=int(labels.max())+1;N=len(q)
 A0=float(cosine[:,~fg].max(1).mean());token_delta=cosine[:,fg].max(1)-cosine[:,~fg].max(1)
 a=np.bincount(labels,weights=token_delta,minlength=K)/N
 v=np.stack([cosine[labels==k].max(0)for k in range(K)])
 def facility(selected,role):
  return float(v[np.array(selected,dtype=bool)][:,role].max(0).mean())if any(selected)else -1.
 def objective(mask):
  sel=np.array([(mask>>k)&1 for k in range(K)],bool)
  return A0+float(a@sel)+.5*facility(sel,fg)+.5*facility(~sel,~fg)
 def additive(mask):
  sel=np.array([(mask>>k)&1 for k in range(K)],bool)
  forward=A0+float(a@sel)
  reverseF=-1+float((v[sel][:,fg]+1).sum(0).mean())/K
  reverseB=-1+float((v[~sel][:,~fg]+1).sum(0).mean())/K
  return forward+.5*(reverseF+reverseB)
 unary=a+.5/K*(v[:,fg].mean(1)-v[:,~fg].mean(1))
 return K,objective,additive,unary,{'A0':A0,'a':a,'v':v,'fg':fg}
def double_greedy(K,values):
 X=0;Y=(1<<K)-1;steps=[]
 for k in range(K):
  e=1<<k;a=values[X|e]-values[X];b=values[Y^e]-values[Y];assert a+b>=-1e-12
  if a>=b:X|=e;choice='add'
  else:Y^=e;choice='drop'
  steps.append({'k':k,'a':float(a),'b':float(b),'choice':choice})
 assert X==Y
 return X,steps

def check(K,F,H,unary):
 n=1<<K;vals=np.array([F(m)for m in range(n)]);hs=np.array([H(m)for m in range(n)])
 A=np.arange(n)[:,None];B=np.arange(n)[None,:];gap=vals[A]+vals[B]-vals[A|B]-vals[A&B]
 assert gap.min()>-1e-12
 modular=np.array([hs[0]+sum(unary[k]for k in range(K)if m>>k&1)for m in range(n)])
 assert np.max(abs(modular-hs))<1e-12
 direct=sum((1<<k)for k in range(K)if unary[k]>=0);assert H(direct)>=hs.max()-1e-12
 dg,steps=double_greedy(K,vals);bound=3*vals[dg]-vals.max()-vals[0]-vals[-1];assert bound>=-1e-12
 assert vals.min()>=-2-1e-12 and vals.max()<=2+1e-12
 return {'sets_exhaustively_evaluated':n,'submodularity_min_slack':float(gap.min()),'additive_modular_max_error':float(abs(modular-hs).max()),'deterministic_endpoint_bound_slack':float(bound),'selected':dg,'objective':float(vals[dg]),'exact_optimum':float(vals.max()),'min_F_plus2':float(vals.min()+2),'steps':steps}

def main():
 rng=np.random.default_rng(0);checks=[]
 for K in range(2,7):
  for repeat in range(8):
   sizes=rng.integers(1,4,K);labels=np.repeat(np.arange(K),sizes);q=unit(rng.normal(size=(len(labels),7)));r=unit(rng.normal(size=(7,7)));fg=np.array([1,1,1,0,0,0,0],bool)
   k,F,H,u,_=make_problem(q,r,fg,labels);result=check(k,F,H,u);checks.append({key:value for key,value in result.items()if key!='steps'})
 # Valid unit-cosine witness: marginal changes sign when a duplicate facility
 # has already covered the same reference foreground token.
 q=np.array([[.4,.6,np.sqrt(.48)],[.4,.6,np.sqrt(.48)],[0,1,0.]])
 r=np.array([[1.,0,0],[0,1.,0]]);fg=np.array([True,False]);labels=np.arange(3)
 K,F,H,u,p=make_problem(q,r,fg,labels);w=check(K,F,H,u)
 witness={'unit_norms':np.linalg.norm(q,axis=1).tolist(),'Fempty':F(0),'F0':F(1),'F1':F(2),'F01':F(3),'add_region1_to_empty':F(2)-F(0),'add_region1_after_region0':F(3)-F(1),'additive_unary':u.tolist(),'double_greedy':w,'pointwise_collapse':False,'interpretation':'Duplicate regions have identical appearances and unary scores but compete for one reference facility; joint marginal flips sign. This also exposes redundant-witness selection, not object binding.'}
 assert witness['add_region1_to_empty']>0 and witness['add_region1_after_region0']<0
 # Complementary worker's semantic counterexample: source-part coverage can
 # relabel a background duplicate even when the forward unary is correct.
 b=np.array([0.,.8,.6]);q2=np.stack([np.array([1.,0,0]),b,b]);r2=np.stack([np.array([1.,0,0]),np.array([0.,1,0]),b]);f2=np.array([True,True,False])
 k2,F2,H2,u2,p2=make_problem(q2,r2,f2,np.arange(3));values2=[F2(m)for m in range(8)];truth_mask=1;best_masks=[m for m,v in enumerate(values2)if abs(v-max(values2))<1e-12]
 semantic={'provided_by':'test_complementary_host_inference','query_vectors':q2.tolist(),'reference_vectors':r2.tolist(),'reference_FG':[True,True,False],'query_GT_SYNTHETIC':[True,False,False],'true_mask_F':F2(1),'false_positive_mask_F':F2(3),'allFG_F':F2(7),'global_optimal_masks':best_masks,'forward_unary_mask':sum(1<<k for k,a in enumerate(p2['a'])if a>=0),'additive_unary_mask':sum(1<<k for k,a in enumerate(u2)if a>=0),'all_values':values2,'interpretation':'Missing foreground part plus duplicate background witness makes the true mask suboptimal; exact optimization cannot fix this target-identity error.'}
 assert semantic['forward_unary_mask']==truth_mask and semantic['additive_unary_mask']==truth_mask and truth_mask not in best_masks
 semantic['double_greedy_mask'],semantic['double_greedy_steps']=double_greedy(3,np.array(values2))
 report={'state':'PASS_SYNTHETIC_EXHAUSTIVE_ALGEBRA_ONLY','random_unit_cosine_problems':len(checks),'maximum_regions':6,'tests':'allS,T submodularity; additive exactmodularity and exactthresholdoptimum;deterministicdoublegreedyendpointbound;rawrange[-2,2]','maximum_additive_error':max(x['additive_modular_max_error']for x in checks),'minimum_submodularity_slack':min(x['submodularity_min_slack']for x in checks),'minimum_endpoint_bound_slack':min(x['deterministic_endpoint_bound_slack']for x in checks),'nonpointwise_unit_cosine_witness':witness,'problems':checks,'semantic_counterexample':semantic}
 Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='problems'},indent=2))
if __name__=='__main__':main()
