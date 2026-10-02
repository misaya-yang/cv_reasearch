"""Keep native heldout and development replay scores separate."""
import argparse,hashlib,json,pathlib
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('results/native_adapt_v1'));p.add_argument('--manifest',type=pathlib.Path,default=pathlib.Path('assets/data/native_adapt_v1/manifest.json'));a=p.parse_args()
 manifest=json.loads(a.manifest.read_text());sha=hashlib.sha256(a.manifest.read_bytes()).hexdigest();groups={r['image_id']:r['dataset'] for r in manifest['rows'] if r['split']=='val'}
 reports={}
 for f in a.root.glob('*/report.json'):
  d=json.loads(f.read_text())
  if d['data_sha256']!=sha:raise ValueError('Manifest mismatch')
  reports[d['mode']]=d
 b=reports['neighborhood'];rows=[]
 for name,d in reports.items():
  if d['initial_state']!=b['initial_state'] or d['training_times']!=b['training_times'] or d['losses'][0]!=b['losses'][0]:raise ValueError('Common-start/time mismatch')
  for group in sorted(set(groups.values())):
   for t in (.2,.5,.8):
    ref={v['image_id']:v['velocity_mse'] for v in b['heldout_paired_scores'] if v['t']==t and groups[v['image_id']]==group};cur={v['image_id']:v['velocity_mse'] for v in d['heldout_paired_scores'] if v['t']==t and groups[v['image_id']]==group}
    if ref.keys()!=cur.keys():raise ValueError('Image pair mismatch')
    ids=sorted(ref);diff=np.array([cur[i]-ref[i] for i in ids]);boot=diff[np.random.default_rng(2027).integers(len(ids),size=(2000,len(ids)))].mean(1)
    rows.append({'mode':name,'dataset':group,'t':t,'images':len(ids),'mean_velocity_mse':float(np.mean(list(cur.values()))),'relative_percent_vs_static':float(100*diff.mean()/np.mean(list(ref.values()))),'mse_delta':float(diff.mean()),'image_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()})
 result={'status':'MATCHED_NATIVE_ADAPTATION_SHORT_TRIAL','rows':rows,'training_seed':2029,'data_sha256':sha,'limits':'Native32 new at selection; COCO16 development sentinel. Shared pretrained+512-stage weights; one continuation seed. MSE not perceptual T2I quality/SOTA.'}
 (a.root/'domain_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
