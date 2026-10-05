#!/usr/bin/env python3
"""Compare an already-run two-episode smoke to frozen reference masks, no GT load."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--logic-reference',required=True,type=Path);p.add_argument('--compatibility-reference',required=True,type=Path);p.add_argument('--adapter-reference',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
f=json.load(open(a.run/'freeze.json'));assert f['episodes']==2 and f['smoke'];audit={}
for key in f['records']:
 with np.load(a.run/'frozen'/f'{key}.npz',allow_pickle=False)as z,np.load(a.logic_reference/'frozen'/f'{key}.npz',allow_pickle=False)as old,np.load(a.compatibility_reference/'frozen'/f'{key}.npz',allow_pickle=False)as comp:
  rec={'pixel_mismatch_vs_prior_logic':{n:int(np.unpackbits(z[n]^old[n]).sum())for n in ['RCG','source_contrast_mean','scalar_contrast','insid3_default','RCG_or_source','RCG_and_source','RCG_or_scalar','RCG_and_scalar','RCG_or_default','RCG_and_default']},'pixel_mismatch_vs_prior_compatibility':{n:int(np.unpackbits(z[n]^comp[n]).sum())for n in ['majority3','conservative_delete','double_support_add','all_intersection','all_union']}}
  if a.adapter_reference:
   with np.load(a.adapter_reference/'frozen'/f'{key}.npz',allow_pickle=False)as adapter:
    rec['pixel_mismatch_vs_earlier_adapter_MEAN_CONTROL']=int(np.unpackbits(z['MEAN_CONTROL']^adapter['MEAN_CONTROL']).sum());rec['RCG_field_max_abs_difference_vs_adapter']=float(np.max(abs(z['RCG_field']-adapter['RCG_field'])))
  audit[key]=rec
report={'status':'Measured parity, not universal bitwise guarantee','scope':'Only two old development episodes, no new labels/features accessed by comparison. RCG references are the original frozen mask inputs used by the existing logical controls. MEAN reference, if present, is our earlier original-formula adapter, not an independently uploaded mean mask.','audit':audit,'freeze_sha256':hashlib.sha256((a.run/'freeze.json').read_bytes()).hexdigest(),'predict_seconds':f['seconds'],'environment':f['environment'],'delivered_source_sha256':f['source_sha256']}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
