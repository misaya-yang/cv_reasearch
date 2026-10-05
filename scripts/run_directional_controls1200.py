#!/usr/bin/env python3
"""Frozen conditional selector on two public-labelled600blocks; CPU1200score.

This is benchmark re-evaluation containing prior DEV exposure. The pure predictor
is identical to DEV241. Supplemental edit accounting uses RCG/FoRIS; no raw DINO
mask is fabricated where it was not exported. Source masks and features are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import run_directional_controls as method
from ics.experiment import sha


def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def manifest(path):
 rows=json.loads(path.read_text())
 if len(rows)!=1200 or len({r['key'] for r in rows})!=1200:raise ValueError('Require1200unique draw IDs; natural repeated photographs are retained')
 return rows


def infer(a):
 import numpy as np
 import torch
 from ics.experiment import load_inputs
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.out.exists():raise FileExistsError('Fresh output required')
 rows=manifest(a.manifest)
 freeze=json.loads(a.freeze.read_text())
 for name,digest in freeze['method_source_sha256'].items():
  if sha(REPO/'scripts'/name)!=digest:raise ValueError('Frozen method changed')
 sources={}
 for row in rows:
  run=Path(row['recheck_run'])
  if str(run) not in sources:
   sources[str(run)]=dict(captured={})
 a.out.mkdir();(a.out/'predictions').mkdir();write(a.out/'manifest.json',rows)
 protocol=dict(n=1200,freeze_sha256=sha(a.freeze),freeze_path=str(a.freeze),config=method.CONFIG,
  selection='directional predictor plus fixed strong controls; previously inspected1200 development re-evaluation',
  runtime_accounting_origin='RCG; raw DINO bookkeeping remains reported onDEV241',resolution=1024,seed=0,
  exposure=freeze['exposure'],sources={},input_receipts={},
  source_code={str(REPO/'scripts'/name):sha(REPO/'scripts'/name) for name in ['run_directional_controls1200.py','run_directional_controls.py','pixel_budget_control.py','run_calibrated_operator.py']},
  extra_encoder_forwards_in_selector=0,query_gt_in_inference=False)
 write(a.out/'protocol.json',protocol);seals,choices={},{};begin=time.monotonic()
 def source_live():
  root=Path('/proc')/str(freeze['source_chain']['pid'])
  if not root.exists():return False
  stat=(root/'stat').read_text().rsplit(')',1)[1].split()
  return stat[0] not in ('Z','X') and int(stat[19])==freeze['source_chain']['start_ticks']
 for n,row in enumerate(rows,1):
  key=row['key'];(q,r,cov,score),receipt=load_inputs(Path('/'),row)
  run=Path(row['recheck_run'])
  pp=run/'predictions'/(key+'.npz');fp=run/'fields'/(key+'.npz')
  while True:
   try:
    with np.load(pp,allow_pickle=False) as z:comparisons={'rcg':z['RCG'].copy(),'astra.control':z['external_mean__delete'].copy(),'mean.control':z['MEAN_CONTROL'].copy()}
    with np.load(fp,allow_pickle=False) as z:field=z['rcg'].copy()
    break
   except (OSError,EOFError,zipfile.BadZipFile,ValueError):
    if not source_live():raise RuntimeError('Source producer stopped before required episode: '+key)
    time.sleep(1)
  with np.load(row['packet_export'],allow_pickle=False) as z:comparisons['native']=z['native'].copy()
  sources[str(run)]['captured'][key]=dict(prediction=sha(pp),field=sha(fp))
  output,prob,info=method.predict(q,r,cov,comparisons,field)
  info.pop('raw_origin_unchanged',None);choices[key]=info
  path=a.out/'predictions'/(key+'.npz');np.savez_compressed(path,**comparisons,**output);seals[key]=sha(path)
  protocol['input_receipts'][key]=dict(receipt,source_mask_sha256=sha(pp),source_field_sha256=sha(fp))
  if n%25==0:print(json.dumps(dict(n=n,total=1200,seconds=round(time.monotonic()-begin,1))),flush=True)
 # Stream without GT, then bind every consumed episode to the producer's final seal.
 for name,record in sources.items():
  run=Path(name)
  while not (run/'sealed.json').exists():
   if not source_live():raise RuntimeError('Source producer stopped before sealing '+name)
   time.sleep(1)
  parent=json.loads((run/'sealed.json').read_text())
  if parent['state']!='ALL_PREDICTIONS_SEALED' or sha(run/'manifest.json')!=parent['manifest_sha256']:raise ValueError('Parent seal invalid')
  prior={r['key']:r for r in json.loads((run/'manifest.json').read_text())}
  for row in [r for r in rows if r['recheck_run']==name]:
   key=row['key'];old=prior[key];captured=record['captured'][key]
   if any(Path(str(old[k])).name!=Path(str(row[k])).name for k in ['support','query']) or old['c']!=row['c']:raise ValueError('Photo/class mismatch')
   if captured['prediction']!=parent['predictions'][key] or captured['field']!=parent['fields'][key]:raise ValueError('Streamed source changed before sealing')
  protocol['sources'][name]=dict(seal_sha256=sha(run/'sealed.json'))
 write(a.out/'choices.json',choices);write(a.out/'protocol.json',protocol)
 write(a.out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=1200,manifest_sha256=sha(a.out/'manifest.json'),protocol_sha256=sha(a.out/'protocol.json'),choices_sha256=sha(a.out/'choices.json'),predictions=seals,query_gt_used_for_fitting=False,seconds=time.monotonic()-begin,cuda_peak_bytes=torch.cuda.max_memory_allocated()))
 print(json.dumps(dict(state='ALL_PREDICTIONS_SEALED',n=1200)),flush=True)


def score(a):
 import numpy as np
 from ics.experiment import unpack,summarize
 seal=json.loads((a.out/'sealed.json').read_text())
 if seal['state']!='ALL_PREDICTIONS_SEALED' or (a.out/'report.json').exists():raise ValueError('Incomplete/already scored')
 for field,name in [('manifest_sha256','manifest.json'),('protocol_sha256','protocol.json'),('choices_sha256','choices.json')]:
  if sha(a.out/name)!=seal[field]:raise ValueError('Changed scoring input')
 rows=manifest(a.out/'manifest.json');protocol=json.loads((a.out/'protocol.json').read_text())
 for path,digest in protocol['source_code'].items():
  if sha(path)!=digest:raise ValueError('Changed scoring code')
 arrays,corrections,relative_rcg,details={},{},{},[]
 for row in rows:
  key=row['key'];receipt=protocol['input_receipts'][key]
  if sha(row['packet_export'])!=receipt['packet_sha256']:raise ValueError('Changed packet')
  with np.load(row['packet_export'],allow_pickle=False) as z:truth=unpack(z['truth'])
  path=a.out/'predictions'/(key+'.npz')
  if sha(path)!=seal['predictions'][key]:raise ValueError('Changed prediction')
  with np.load(path,allow_pickle=False) as z:masks={name:unpack(z[name]) for name in z.files}
  episode=dict(row,iu={})
  for name,mask in masks.items():
   iu=[int((mask&truth).sum()),int((mask|truth).sum())];arrays.setdefault(name,[]).append(iu);episode['iu'][name]=iu
   for origin,target in [(masks['native'],corrections),(masks['rcg'],relative_rcg)]:
    plus,minus=mask&~origin,origin&~mask
    target.setdefault(name,[]).append(dict(key=key,c=row['c'],fold=row['fold'],batch=row['batch'],add_TP=int((plus&truth).sum()),add_FP=int((plus&~truth).sum()),delete_TP=int((minus&truth).sum()),delete_FP=int((minus&~truth).sum())))
  details.append(episode)
 arrays={name:np.array(values) for name,values in arrays.items()}
 report,_=summarize(rows,arrays,corrections)
 report.update(exposure=protocol['exposure'],parameter_updates=False,source_prediction_seal_sha256=sha(a.out/'sealed.json'),primary='conditional.joint',
  edits_vs_RCG={name:{k:sum(r[k] for r in values) for k in ['add_TP','add_FP','delete_TP','delete_FP']} for name,values in relative_rcg.items()},
  raw_DINO_1200='not exported; shared raw DINO accounting remainsDEV241',INSID3_1200='official replay absent; not silently substituted by cached variant')
 report['each_batch']={}
 for batch in [0,1]:
  keep=np.array([r['public_batch']==batch for r in rows]);sub=[r for r,k in zip(rows,keep) if k]
  selected_corrections={name:[v for v,k in zip(values,keep) if k] for name,values in corrections.items()}
  result,_=summarize(sub,{name:values[keep] for name,values in arrays.items()},selected_corrections)
  result['exposure']=protocol['exposure'];report['each_batch'][str(batch)]=result
 write(a.out/'report.json',report)
 (a.out/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
 print(json.dumps(dict(n=1200,scores=report['scores'],primary=report['contrasts']['conditional.joint'])),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['infer','score']);p.add_argument('--manifest',type=Path);p.add_argument('--freeze',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.phase=='infer' and (a.manifest is None or a.freeze is None):p.error('Frozen manifest/rule required')
 {'infer':infer,'score':score}[a.phase](a)

if __name__=='__main__':main()
