#!/usr/bin/env python3
"""Portable cache-to-complete-mask replay. All predictions freeze before scoring.
Prediction inputs are q/r/reference cov/FoRIS score and explicit source maxima mode. CPU, one thread.
"""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
import scipy,sklearn
import components
import adaptive_deletion as deletion
from rcg_readout import load_rows,unpack
from statistics_randomstate import analyze,episode_key
torch.set_num_threads(1)
SOURCES=['rerun.py','components.py','rcg_readout.py','hypothesis_source_contrast.py','statistics_randomstate.py','adaptive_deletion.py']
MASK_ARMS=['RCG','MEAN_CONTROL','insid3_default','source_contrast_mean','scalar_contrast','RCG_or_source','RCG_and_source','RCG_or_scalar','RCG_and_scalar','RCG_or_default','RCG_and_default','majority3','conservative_delete','double_support_add','all_intersection','all_union','native_pre_from_score'] + deletion.ARMS + [deletion.FIXED_ARM]

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def locate(roots,relative):
 candidates=[r/relative for r in roots if (r/relative).is_file()]
 if not candidates:return None
 if len(candidates)>1 and len({sha(p)for p in candidates})>1:raise ValueError('Conflicting duplicate cache files: '+relative)
 return candidates[0]
def resolve(rows,roots,need_features):
 missing=[];paths={}
 for r in rows:
  k=episode_key(r);pp=locate(roots,f'results/extent_v1/run/packets/{k}.npz');fp=locate(roots,f'cache/evidence_v1/feat/{k}.pt')if need_features else None
  if pp is None:missing.append(k+': packet')
  if need_features and fp is None:missing.append(k+': feature')
  paths[k]=(fp,pp)
 if missing:raise FileNotFoundError(f'{len(missing)} required input files missing; no episodes dropped:\n'+'\n'.join(missing))
 return paths

def predict(rows,roots,out,smoke,bg_evidence,audit_feature_evidence):
 paths=resolve(rows,roots,True)
 out.mkdir(parents=True,exist_ok=False);(out/'frozen').mkdir();records={};audits={};start=time.perf_counter()
 for row in rows:
  k=episode_key(row);fp,pp=paths[k];begin=time.perf_counter()
  try:
   feat=torch.load(fp,map_location='cpu',weights_only=True)
   q=feat['q'].float().numpy();r=feat['r'].float().numpy()
   with np.load(pp,allow_pickle=False)as pkt:
    # Prediction-only field access: query truth/pre/native are never indexed.
    cov=pkt['cov'].copy();score=pkt['score'].copy()
    supplied_maxima = ('fg_max' in pkt and 'bg_max' in pkt)
    if bg_evidence=='packet' and not supplied_maxima:raise ValueError('Canonical candidate requires packet fg_max/bg_max; use explicit --bg-evidence recompute for quantized-feature approximation')
    fg_packet=pkt['fg_max'].copy() if supplied_maxima else None
    bg_packet=pkt['bg_max'].copy() if supplied_maxima else None
   if q.shape!=(4096,1024)or r.shape!=(4096,1024)or cov.shape!=(64,64)or score.shape!=(64,64):raise ValueError('Expected4096x1024 q/r and64x64 cov/score')
   masks,fields,info=components.predict_all(q,r,cov,score)
   fg_re=bg_re=None
   if bg_evidence=='recompute' or audit_feature_evidence:
    fg_re,bg_re,normerr=deletion.feature_maxima(q,r,cov)
   fg,bg=(fg_packet,bg_packet) if bg_evidence=='packet' else (fg_re,bg_re)
   added,budget=deletion.predict(masks,fields,fg,bg)
   masks.update(added);info['deletion_budget']=budget;info['maxima_source']=bg_evidence
   fields['fg_max_used']=np.asarray(fg,np.float64).reshape(64,64);fields['bg_max_used']=np.asarray(bg,np.float64).reshape(64,64)
   if fg_re is not None and supplied_maxima:
    alternative,ab=deletion.predict(masks,fields,fg_re,bg_re)
    canonical,cb=deletion.predict(masks,fields,fg_packet,bg_packet)
    info['quantized_feature_evidence_audit']=dict(feature_unit_norm_maxerr=normerr,fg_max_abs_error=float(np.abs(np.asarray(fg_packet).ravel()-fg_re).max()),bg_max_abs_error=float(np.abs(np.asarray(bg_packet).ravel()-bg_re).max()),packet_K=cb['explicit_deletions'],recomputed_K=ab['explicit_deletions'],K_delta=ab['explicit_deletions']-cb['explicit_deletions'],mask_mismatch_pixels={n:int(np.unpackbits(canonical[n]^alternative[n]).sum()) for n in deletion.ARMS})
   assert list(masks)==MASK_ARMS[:-1]
   for n,m in masks.items():assert m.dtype==np.uint8 and m.shape==(131072,),n
   op=out/'frozen'/f'{k}.npz';np.savez_compressed(op,**masks,**fields)
   records[k]=dict(path=str(op.relative_to(out)),sha256=sha(op),feature_sha256=sha(fp),packet_sha256=sha(pp));info['debiased_metadata']=bool(feat['debiased'])if 'debiased'in feat else None;info['seconds']=time.perf_counter()-begin;audits[k]=info
   print(json.dumps(dict(key=k,complete=len(records),hypotheses=len(info['hypotheses']),seconds=info['seconds'])),flush=True)
  except Exception as e:
   dump(out/'failure_before_GT.json',dict(key=k,error=repr(e),completed=len(records),GT_accessed=False));raise
 # Final label-free cohort pass supplies the cross-fitted fixed-fraction control.
 fractions=deletion.crossfit_fractions(rows,{k:v['deletion_budget'] for k,v in audits.items()},smoke=smoke)
 for row in rows:
  k=episode_key(row);op=out/records[k]['path']
  with np.load(op,allow_pickle=False) as z: data={n:z[n].copy() for n in z.files}
  fixed,K=deletion.fixed_fraction_mask(data,data,fractions[str(row['fold'])]['fraction']);data[deletion.FIXED_ARM]=fixed
  np.savez_compressed(op,**data);records[k]['sha256']=sha(op);audits[k]['fixed_fraction_K']=K
 dump(out/'crossfit_parameters.json',fractions)
 freeze=dict(state='ALL_REQUESTED_EPISODE_PREDICTIONS_FROZEN_NO_QUERY_GT',episodes=len(rows),smoke=smoke,scope='Execution/parity smoke only; not a benchmark result'if smoke else'Fixed cached inference replay; split status inherited from supplied manifest',manifest=rows,arms=MASK_ARMS,records=records,audit=audits,GT_accessed=False,bg_evidence=bg_evidence,crossfit_parameters=fractions,prediction_inputs=['packet.fg_max/bg_max only in packet mode or optional numerical audit','feature.q','feature.r','optional feature.debiased metadata','packet.cov','packet.score'],source_sha256={n:sha(Path(__file__).parent/n)for n in SOURCES},seconds=time.perf_counter()-start,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,threads=1,device='CPU'))
 dump(out/'freeze.json',freeze);print('ALL_PREDICTIONS_FROZEN',freeze['seconds'],flush=True)

def score(rows,roots,out,baseline):
 freeze=json.load(open(out/'freeze.json'));assert freeze['state']=='ALL_REQUESTED_EPISODE_PREDICTIONS_FROZEN_NO_QUERY_GT'
 keys=[episode_key(r)for r in rows];assert set(keys)==set(freeze['records'])and len(rows)==freeze['episodes']
 for n,h in freeze['source_sha256'].items():
  if sha(Path(__file__).parent/n)!=h:raise ValueError('Source changed since prediction freeze: '+n)
 for r,s in zip(rows,freeze['manifest']):
  if any(r[k]!=s[k]for k in ['fold','e','c','support','query']):raise ValueError('Manifest changed since freeze')
 paths=resolve(rows,roots,False);scored=[];parity={};start=time.perf_counter()
 if (out/'episodes.jsonl').exists()or(out/'report.json').exists():raise FileExistsError('Scored results already exist; no overwrite')
 for row in rows:
  k=episode_key(row);rec=freeze['records'][k];op=out/rec['path'];pp=paths[k][1]
  if sha(op)!=rec['sha256']or sha(pp)!=rec['packet_sha256']:raise ValueError('Prediction or packet changed since freeze: '+k)
  with np.load(op,allow_pickle=False)as z,np.load(pp,allow_pickle=False)as p:
   if 'truth'not in p:raise ValueError('Score mode requires packed query truth: '+k)
   gt=unpack(p['truth']);pred={n:unpack(z[n])for n in freeze['arms']}
   if 'native'in p:pred['native_cached_CRF']=unpack(p['native'])
   if 'pre'in p:pred['native_pre']=unpack(p['pre']);parity[k]=int(np.count_nonzero(pred['native_pre']!=pred['native_pre_from_score']))
   iu={};confusion={};actions={};host=pred['RCG']
   for n,m in pred.items():
    tp=int((m&gt).sum());fp=int((m&~gt).sum());fn=int((~m&gt).sum());tn=int((~m&~gt).sum());iu[n]=[tp,tp+fp+fn];confusion[n]=dict(TP=tp,FP=fp,FN=fn,TN=tn)
    add=m&~host;rem=host&~m;actions[n]=dict(add_TP=int((add&gt).sum()),add_FP=int((add&~gt).sum()),remove_TP=int((rem&gt).sum()),remove_FP=int((rem&~gt).sum()))
   if baseline not in iu:raise ValueError('Requested scoring baseline unavailable: '+baseline+'; provide native comparator or explicitly choose --baseline RCG')
   scored.append({**row,'iu':iu,'confusion':confusion,'edits_vs_RCG':actions})
 contrasts=[('RCG_count_matched_delete',n) for n in ['MEAN_CONTROL','global_RCG_same_budget','crossfit_fraction_RCG_rank']]+[(n,'RCG')for n in MASK_ARMS if n!='RCG']+[(n,'RCG_and_scalar')for n in ['majority3','conservative_delete','double_support_add','all_intersection','all_union']]
 report=dict(scope=freeze['scope'],episodes=len(rows),metric='Class macro after sumI/sumU within each class, same1024 raster',statistics_protocol='2000 RandomState(0) photograph-connected episode-group draws; numeric fold/e/c order; resampled multiplicities reaggregate classI/U; absent classes omitted',baseline=baseline,statistics=analyze(scored,baseline=baseline,contrasts=contrasts),native_pre_replay_mismatch_pixels=parity,predict_seconds=freeze['seconds'],score_seconds=time.perf_counter()-start)
 report['cohorts']={}
 if all('batch' in r for r in scored):
  for batch in sorted({r['batch'] for r in scored}):
   subset=[r for r in scored if r['batch']==batch];report['cohorts'][batch]=analyze(subset,baseline=baseline,contrasts=contrasts)
 report['actions_vs_RCG']={n:{a:sum(r['edits_vs_RCG'][n][a] for r in scored) for a in ['add_TP','add_FP','remove_TP','remove_FP']} for n in freeze['arms']}
 (out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in scored));dump(out/'report.json',report);print('SCORED',len(scored),'episodes',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['predict','score','all']);p.add_argument('--root',action='append',required=True,type=Path,help='Extracted package/cache root; repeat for partitioned packages');p.add_argument('--manifest',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--expected',type=int,default=241,help='Exact required unique episode count; default241');p.add_argument('--smoke',action='store_true',help='Explicitly label an execution-only small smoke');p.add_argument('--baseline',default='native_cached_CRF');p.add_argument('--bg-evidence',choices=['packet','recompute'],default='packet',help='packet reproduces original BG budget; recompute explicitly uses quantized q/r dot products');p.add_argument('--audit-feature-evidence',action='store_true',help='Measure packet versus full quantized-feature maxima/K/mask drift without changing chosen evidence mode');a=p.parse_args()
 rows=load_rows(a.manifest)
 if len(rows)!=a.expected:raise ValueError(f'Manifest contains {len(rows)} unique episodes, expected {a.expected}; no silent subset')
 if a.expected<20 and not a.smoke:raise ValueError('A tiny run requires explicit --smoke')
 roots=[x.resolve()for x in a.root]
 if any(not x.is_dir()for x in roots):raise FileNotFoundError('Every --root must be an existing extracted root')
 if a.mode in ['predict','all']:predict(rows,roots,a.out,a.smoke,a.bg_evidence,a.audit_feature_evidence)
 if a.mode in ['score','all']:score(rows,roots,a.out,a.baseline)
if __name__=='__main__':main()
