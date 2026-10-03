#!/usr/bin/env python3
"""10 design-DEV causal finalizer replay; no encoder, training, search or download."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from full_cached_replay import source_finalizer,unpack_mask
from conditional_full_fg_finalize import finalize_pair_from_audit_rows
from relational_exclusion_cpu import finalize_deletion

CARD=[
 'Assumption: more than half of old flips have a lower-cost legal FG explanation in the fixed full-FG star library (already 215/262), so truncation may cause TP loss.',
 'Prediction: fullFG rescission lowers TP loss versus v1; evidence of useful selection requires original IoU strictly above random thinning and low-score deletion at the same retained patch count, not merely fewer actions.',
 'Match: fixed-star truncation causally damages this design subset; asymmetric allFG/B8 remains diagnosis, not a fair final method.',
 'Mismatch: no advantage over matched-count controls refutes useful targeting; worse TP/FP tradeoff rejects this rescission; CRF cross-arm deletion monotonicity is not assumed.'
]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def name(r):return '%d_%d_%d'%(r['fold'],r['e'],r['c'])
def json_witness(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def freeze_one(row,man,v1,costs,host,out,device):
 n=name(row);start=time.monotonic()
 qp=Image.open(Path(man['data_root'])/row['query']).convert('RGB')
 if [qp.height,qp.width]!=row['query_original_hw']:raise RuntimeError('RGB metadata drift')
 tgt=host._transform(qp)[None].to(device)
 if tuple(tgt.shape)!=(1,3,1024,1024):raise RuntimeError('unverified square working frame')
 with np.load(Path(man['_packets'])/(n+'.npz'),allow_pickle=False) as z:
  score=z['score'].copy();saved=unpack_mask(z['native'],(1024,1024));pre=unpack_mask(z['pre'],(1024,1024))
 actual_pre=host._binarize_response(torch.from_numpy(score).to(device),target_hw=(1024,1024))
 if not np.array_equal(actual_pre.cpu().numpy(),pre):raise RuntimeError('pre-mask not exact')
 check=host._finalize_mask(actual_pre,tgt).cpu().numpy().astype(bool)
 if not np.array_equal(check,saved):raise RuntimeError('complete native mask not exact before GT')
 old=json_witness(v1/'witnesses'/(n+'.json.gz'))['records']
 accepted={int(r['query_root']) for r in old if r.get('status')=='ok' and r['joint_flip_predicate']['accepted']}
 def finalizer(x):return host._finalize_mask(torch.from_numpy(x).to(device),tgt).cpu().numpy().astype(bool)
 result=finalize_pair_from_audit_rows(pair_id=n,original_records=old,audited_rows=costs[n],
           native_pre_final=pre,native_final=saved,finalizer=finalizer,mapping_contract='full_grid_nearest')
 k=result['audit']['candidate_patch_deletions'];original_field=np.zeros((64,64),bool)
 original_field.reshape(-1)[list(accepted)]=True
 random_field=np.zeros((64,64),bool)
 if k:random_field.reshape(-1)[np.random.default_rng(0).choice(sorted(accepted),size=k,replace=False)]=True
 shifted=score-score.min();sn=shifted/max(float(shifted.max()),1e-6)
 eligible=np.flatnonzero((sn>.5).reshape(-1));low_field=np.zeros((64,64),bool)
 order=eligible[np.lexsort((eligible,score.reshape(-1)[eligible]))]
 if k>len(order):raise RuntimeError('matched-count low-score budget impossible')
 low_field.reshape(-1)[order[:k]]=True
 with np.load(v1/'predictions'/(n+'.npz'),allow_pickle=False) as z:
  old_final=unpack_mask(z['joint_flip'],(1024,1024))
 masks=dict(native=saved,v1=old_final,fullFG=result['final_mask']);details=dict(fullFG=result['metadata'])
 for arm,field in (('random_thinning',random_field),('score_same_count',low_field)):
  z=finalize_deletion(field,native_pre_final=pre,native_final=saved,finalizer=finalizer,mapping_contract='full_grid_nearest')
  masks[arm]=z['final_mask'];details[arm]=z['metadata']
 original={arm:(F.interpolate(torch.from_numpy(m)[None,None].float(),tuple(row['query_original_hw']),mode='bilinear',align_corners=False)[0,0].numpy()>.5) for arm,m in masks.items()}
 p=out/'predictions'/(n+'.npz');np.savez_compressed(p,**{**{arm:np.packbits(m) for arm,m in masks.items()},**{'original__'+arm:np.packbits(m) for arm,m in original.items()}})
 info=dict(row=row,predictions_sha256=sha(p),native_exact=True,query_GT_read=False,
           original_flip_patches=len(accepted),retained_flip_patches=k,random_seed=0,
           fullFG_costs_not_recomputed=True,scope='asymmetric allFG-fixed-star versus old top8-BG causal control',
           conditional_audit=result['audit'],finalizer=details,freeze_seconds=time.monotonic()-start)
 (out/'frozen'/(n+'.json')).write_text(json.dumps(info,indent=1));return info

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--v1',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
 p.add_argument('--packets',required=True);p.add_argument('--out',required=True,type=Path);p.add_argument('--preflight-first',action='store_true');p.add_argument('--reuse-preflight',action='store_true')
 p.add_argument('--budget-seconds',type=float,default=225);a=p.parse_args();start=time.monotonic()
 torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not torch.cuda.is_available():raise RuntimeError('CUDA required for original CRF only')
 free,total=torch.cuda.mem_get_info()
 if free<4*1024**3:raise RuntimeError('WAIT_FOR_MEMORY: fewer than4GiB; protect concurrent jobs')
 torch.cuda.set_per_process_memory_fraction(.2)
 man=json.loads(a.manifest.read_text());man['_packets']=a.packets
 meta=json.loads((a.v1/'metadata_preflight.json').read_text())
 if meta['manifest_sha256']!=sha(a.manifest):raise RuntimeError('old frozen manifest drift')
 for f,h in meta['source_sha256'].items():
  if sha(Path(man['foris_root'])/f)!=h:raise RuntimeError('public source drift')
 summary=json.loads((a.v1/'source_truncation_audit.json').read_text())
 if summary['state']!='COMPLETED_CPU_SOURCE_TRUNCATION_AUDIT' or summary['completed_roots']!=262 or summary['query_GT_read']:raise RuntimeError('incomplete or contaminated cost audit')
 costs={}
 for line in (a.v1/'source_truncation_audit.roots.jsonl').read_text().splitlines():
  r=json.loads(line);costs.setdefault(r['pair_id'],[]).append(r)
 old_records={name(r['row']):r for r in map(json.loads,(a.v1/'episodes.jsonl').read_text().splitlines())}
 a.out.mkdir(parents=True,exist_ok=True)
 for d in ('predictions','frozen'):(a.out/d).mkdir(exist_ok=True)
 if (a.out/'report.json').exists():raise RuntimeError('fresh causal report required')
 host=source_finalizer(man,'cuda');records=[]
 report=dict(state='RUNNING',card=CARD,episodes=0,native_exact=0,query_GT_reads=0,encoder_calls=0,training=False,
             memory_fraction=.2,design_DEV=True,CPU_fullFG_search_recomputed=False,scientific_scope='asymmetric causal diagnosis, not fair main method',speed_claim=False)
 def save():
  report.update(episodes=len(records),elapsed_seconds=time.monotonic()-start)
  if records:
   report['episode_mean_original_IoU_pp']={arm:100*np.mean([r['original_iu'][arm][0]/max(r['original_iu'][arm][1],1) for r in records]) for arm in records[0]['original_iu']}
   by={}
   for arm in records[0]['original_iu']:
    cs={}
    for r in records:cs.setdefault(r['row']['c'],[0,0]);cs[r['row']['c']]=[x+y for x,y in zip(cs[r['row']['c']],r['original_iu'][arm])]
    by[arm]=100*np.mean([i/max(u,1) for i,u in cs.values()])
   report['class_sum_original_IoU_pp']=by
  (a.out/'report.json').write_text(json.dumps(report,indent=1))
 try:
  with torch.inference_mode():
   for idx,row in enumerate(meta['episodes']):
    n=name(row);frozen=a.out/'frozen'/(n+'.json')
    if a.reuse_preflight and idx==0:
     info=json.loads(frozen.read_text())
     if not info['native_exact'] or info['query_GT_read'] or info['predictions_sha256']!=sha(a.out/'predictions'/(n+'.npz')):raise RuntimeError('preflight frozen artifacts invalid')
    else:info=freeze_one(row,man,a.v1,costs,host,a.out,'cuda')
    if a.preflight_first:
     (a.out/'preflight_first.json').write_text(json.dumps(dict(state='PREFLIGHT_FIRST_PASS',pair=n,native_exact=True,cost_interface_valid=True,query_GT_read=False,predictions_sha256=info['predictions_sha256']),indent=1));print('PREFLIGHT_FIRST_PASS',n,flush=True);return
    # All method/control model and original predictions are frozen before query GT.
    with np.load(a.out/'predictions'/(n+'.npz'),allow_pickle=False) as z:
     arms=('native','v1','fullFG','random_thinning','score_same_count')
     models={arm:unpack_mask(z[arm],(1024,1024)) for arm in arms};originals={arm:unpack_mask(z['original__'+arm],tuple(row['query_original_hw'])) for arm in arms}
    truth_o=np.asarray(Image.open(Path(man['annotation_root'])/Path(row['query']).with_suffix('.png')))==row['c']+1
    if list(truth_o.shape)!=row['query_original_hw']:raise RuntimeError('RGB/annotation shape mismatch')
    truth_m=F.interpolate(torch.from_numpy(truth_o.copy())[None,None].float(),(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
    iu=lambda m,t:[int((m&t).sum()),int((m|t).sum())]
    model_iu={arm:iu(m,truth_m) for arm,m in models.items()};original_iu={arm:iu(m,truth_o) for arm,m in originals.items()}
    if original_iu['v1']!=old_records[n]['original_iu']['joint_flip'] or original_iu['native']!=old_records[n]['original_iu']['native']:raise RuntimeError('old IU no longer matches frozen v1/native')
    counts={arm:dict(deleted_FP=int((models['native']&~m&~truth_m).sum()),lost_TP=int((models['native']&~m&truth_m).sum())) for arm,m in models.items() if arm!='native'}
    rec=dict(row=row,model_iu=model_iu,original_iu=original_iu,model_deletions=counts,original_deletions={arm:dict(deleted_FP=int((originals['native']&~m&~truth_o).sum()),lost_TP=int((originals['native']&~m&truth_o).sum())) for arm,m in originals.items() if arm!='native'},
             frozen_prediction_info=info,new_fullFG_final_deletion_outside_v1=int((models['v1']&~models['fullFG']).sum()),recovered_final_pixels_vs_v1=int((models['fullFG']&~models['v1']).sum()),query_GT_used_for_prediction=False)
    records.append(rec);report['native_exact']+=1;report['query_GT_reads']+=1
    with open(a.out/'episodes.jsonl','a') as stream:stream.write(json.dumps(rec)+'\n');stream.flush()
    save();print(json.dumps(dict(pair=n,episodes=len(records),original_IU=original_iu,remaining_patch_count=info['retained_flip_patches'])),flush=True)
    if time.monotonic()-start>=a.budget_seconds:report['state']='COMPLETED_FINITE_BUDGET';save();return
  report['state']='COMPLETED';save()
 except BaseException as e:report.update(state='ERROR',error=repr(e));save();raise
if __name__=='__main__':main()
