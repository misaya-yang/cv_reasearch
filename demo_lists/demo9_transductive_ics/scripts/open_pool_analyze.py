"""Task-level paired class summaries, with explicit dependency limitations."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/root/demo4_cache/env')
import numpy as np

def main(root):
 root=Path(root);reports=[json.load(open(root/f'f{f}/report.json')) for f in range(4)]
 names=list(reports[0]['class_sums']);classes=[]
 for report in reports:
  for c in sorted(report['class_sums']['1shot'],key=int):
   classes.append({m:100*report['class_sums'][m][c][0]/max(report['class_sums'][m][c][1],1) for m in names})
 V=np.array([[r[m] for m in names] for r in classes]);rng=np.random.default_rng(2051);ix=rng.integers(len(V),size=(5000,len(V)))
 contrasts={}
 for a,b in [('clean_agree','1shot'),('mixed_agree','1shot'),('mixed_agree','native_bf16'),('1shot','native_bf16'),('mixed_agree','clean_agree'),('mixed_agree','mixed_naive'),('mixed_agree','mixed_none'),('mixed_roundtrip','mixed_agree')]:
  delta=V[:,names.index(a)]-V[:,names.index(b)]
  contrasts[a+' minus '+b]=dict(mean_pp=float(delta.mean()),paired_class_bootstrap95=np.percentile(delta[ix].mean(1),[2.5,97.5]).tolist(),classes_better_over2=int((delta>2).sum()),classes_worse_over2=int((delta< -2).sum()))
 trusts={}
 for key in ['mixed_agree','mixed_roundtrip','mixed_none']:
  chosen_neg,total_neg,chosen_pos,total_pos=0,0,0,0
  for report in reports:
   for r in report['records']:
    for name,s in r['trust'][key][-1].items():
     if name==r['query']:continue
     if s['target_absent']:total_neg+=1;chosen_neg+=s['chosen']
     else:total_pos+=1;chosen_pos+=s['chosen']
  trusts[key]=dict(absent_donor_acceptance=chosen_neg/max(total_neg,1),present_donor_acceptance=chosen_pos/max(total_pos,1),absent_denominator=total_neg,present_denominator=total_pos)
 result=dict(state='COMPLETED',episodes=sum(r['episodes'] for r in reports),classes=len(classes),miou={m:float(V[:,i].mean()) for i,m in enumerate(names)},per_fold={str(r['fold']):r['miou'] for r in reports},contrasts=contrasts,trust_diagnostics=trusts,
  scope='Development robustness comparison, not a new method or paper success. Five standard episodes/class, same reference/query/donor count, approximately half donors replaced. Same labelled support annotation, no class-presence label supplied to inference.',
  inference_limits='Class bootstrap conditions on current episodes and seed. Photos can occur across classes and donor graphs couple queries; this is not a proof of image-independent generalization. Subsequent method design needs frozen new episodes/seeds and FoRIS same-information strong control.',
  protocol='INSID3 frozen DINOv3-L native BF16 encoder; historical compact reader uses FP32 normalization/debias/decoder plus FP16 projected cache. Genuine original BF16 decoder separately scored. TF32off. Square1024 mask I/U per class then class average, matching previous demo9, not original-resolution official-mask reproduction.')
 matched=[]
 for r in reports:
  if r.get('matched_positive_control'):matched.append(r)
  else:
   side=root/f"f{r['fold']}/matched_control.json"
   if side.exists():
    ss=json.load(open(side))
    if ss.get('state')!='COMPLETED':continue
    assert {x['episode'] for x in ss['records']}=={x['episode'] for x in r['records']},'Side control query mismatch'
    rr=dict(r);rr['class_sums']=dict(r['class_sums'],**ss['class_sums']);matched.append(rr)
 result['matched_positive_control']=dict(folds=[r['fold'] for r in matched],scope='Added before folds1-3 scoring to separate fewer target-present donors from harmful added absent donors; same positive donor identities, fewer total inputs. Not an oracle filtering deployment method.')
 for kind in ['naive','none','agree']:
  ds=[]
  for report in matched:
   for c in report['class_sums']['1shot']:
    im,um=report['class_sums']['mixed_'+kind][c];ip,up=report['class_sums']['present_matched_'+kind][c]
    ds.append(100*(im/max(um,1)-ip/max(up,1)))
  if ds:
   ds=np.array(ds);bi=rng.integers(len(ds),size=(5000,len(ds)))
   result['matched_positive_control']['mixed_minus_present_matched_'+kind]=dict(mean_pp=float(ds.mean()),paired_class_bootstrap95=np.percentile(ds[bi].mean(1),[2.5,97.5]).tolist(),classes=len(ds))
 (root/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
