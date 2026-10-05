#!/usr/bin/env python3
"""Preserve exact-episode feature reuse and prepare two public-labelled batches.

No per-image composition: FoRIS positional-debias gating depends on the labeled
reference. Query truth is never indexed. Existing feature contents are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from ics.experiment import load_rows,packet,sha


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 root=a.project;out=a.out;inputs=out/'inputs';inputs.mkdir(parents=True,exist_ok=True)
 def identity(row):return int(row['c']),Path(row['support']).name,Path(row['query']).name
 pools={}
 for label,manifest,data_root in [('dev241',root/'code_resume_v1/evidence/local/research_20261005/dev241.json',Path('/root/autodl-tmp/demo9_extent')),('historical600',root/'launch/fresh600_v1/episodes.json',root/'outputs/fresh600_root')]:
  for row in load_rows(manifest):
   feature=data_root/row.get('feature_export','cache/evidence_v1/feat/'+row['key']+'.pt')
   if not feature.exists() and feature.suffix=='.pt':feature=feature.with_suffix('.npz')
   if feature.exists():pools[identity(row)]=(label,feature,packet(data_root,row))
 rows=[];gaps=[];reuse={};metadata=json.loads((root/'outputs/claude_official/batch0.json').read_text())
 for batch in (0,1):
  manifest=root/('outputs/claude_official/batch%d.json'%batch)
  for original in load_rows(manifest):
   row=dict(original,public_batch=batch,batch='public%d'%batch)
   target_packet=root/('outputs/claude_official/root%d/results/extent_v1/run/packets'%batch)/(row['key']+'.npz')
   row['packet_export']=str(target_packet)
   row['recheck_run']=str(root/('outputs/claude_official/run%d'%batch))
   if batch==1:
    feature=inputs/'batch1_features'/(row['key']+'.pt')
    if not feature.exists():raise FileNotFoundError('Second-batch feature not preserved: '+row['key'])
    row['feature_export']=str(feature);reuse[row['key']]=dict(source='public1',feature_sha256=sha(feature))
   elif identity(row) in pools:
    label,source,source_packet=pools[identity(row)]
    with np.load(source_packet,allow_pickle=False) as z:old={name:z[name].copy() for name in ['cov','score','native']}
    with np.load(target_packet,allow_pickle=False) as z:new={name:z[name].copy() for name in ['cov','score','native']}
    parity=dict(cov_max=float(np.abs(old['cov']-new['cov']).max()),score_max=float(np.abs(old['score']-new['score']).max()),native_pixels=int(np.unpackbits(old['native']^new['native']).sum()))
    # Bind compatible cache geometry/readout; remaining tiny native replay drift stays explicit.
    if parity['cov_max'] or parity['score_max']>1e-6 or parity['native_pixels']>100:
     row['feature_export']=str(inputs/'gap_features'/(row['key']+'.pt'));gaps.append(row);reuse[row['key']]=dict(source='gap_after_parity',parity=parity)
    else:
     dest=inputs/'reused_features'/(row['key']+source.suffix);dest.parent.mkdir(exist_ok=True)
     if not dest.exists():os.link(source,dest)
     row['feature_export']=str(dest);reuse[row['key']]=dict(source=label,source_feature=str(source),feature_sha256=sha(dest),parity=parity)
   else:
    row['feature_export']=str(inputs/'gap_features'/(row['key']+'.pt'));gaps.append(row);reuse[row['key']]=dict(source='uncached_pair')
   rows.append(row)
 if len(rows)!=1200 or len({r['key'] for r in rows})!=1200:raise ValueError('Expected two complete, disjoint600draw blocks')
 (inputs/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
 metadata['episodes']=gaps;(inputs/'gap_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
 receipt=dict(n=1200,gap_pairs=len(gaps),reuse= reuse,source_manifests={str(root/('outputs/claude_official/batch%d.json'%b)):sha(root/('outputs/claude_official/batch%d.json'%b)) for b in [0,1]},
  exposure='public-labelled benchmark re-evaluation; includes DEV and previously inspected episodes',per_image_composition=False)
 (inputs/'preparation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(n=1200,gap_pairs=len(gaps),direct_reuse=1200-len(gaps),sources={label:sum(v['source']==label for v in reuse.values()) for label in set(v['source'] for v in reuse.values())})),flush=True)

if __name__=='__main__':main()
