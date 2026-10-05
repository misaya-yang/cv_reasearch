#!/usr/bin/env python3
"""Independently score sealed recheck masks, including the omitted MEAN control.

The optional fixed delete-p count arm is replayed from existing sweep statistics;
its missing full-mask edit diagnostics are explicitly retained as missing.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import itertools
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def main():
 import numpy as np
 from ics.experiment import sha,load_rows,packet,unpack,summarize
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError("Fresh output required")
 seal=json.loads((a.run/"sealed.json").read_text())
 if seal["state"]!="ALL_PREDICTIONS_SEALED" or sha(a.run/"manifest.json")!=seal["manifest_sha256"]:raise ValueError("Unsealed source")
 rows=load_rows(a.run/"manifest.json")
 if len(rows)!=600 or set(seal["predictions"])!={r["key"] for r in rows}:raise ValueError("Full600 source required")
 names={"RCG":"rcg","RCG_count_matched_delete":"c.control","conservative_delete":"conservative.control","external_mean__delete":"astra.control","external_mean_delete_sameK_RCG":"astra_sameK.control","MEAN_CONTROL":"mean.control"}
 arrays={"native":[]};corrections={};details=[];receipts={}
 for row in rows:
  key=row["key"];path=a.run/"predictions"/(key+".npz")
  if sha(path)!=seal["predictions"][key]:raise ValueError("Changed source "+key)
  pp=packet(a.root,row)
  with np.load(pp,allow_pickle=False) as z:truth,native=unpack(z["truth"]),unpack(z["native"])
  receipts[key]=dict(packet=str(pp),packet_sha256=sha(pp),prediction_sha256=sha(path))
  with np.load(path,allow_pickle=False) as z:masks={label:unpack(z[name]) for name,label in names.items()}
  masks["native"]=native;episode=dict(row,iu={})
  for name,mask in masks.items():
   iu=[int((mask&truth).sum()),int((mask|truth).sum())];arrays.setdefault(name,[]).append(iu);episode["iu"][name]=iu
   add,delete=mask&~native,native&~mask
   corrections.setdefault(name,[]).append(dict(key=key,c=row["c"],fold=row["fold"],batch=row.get("batch","unspecified"),add_TP=int((add&truth).sum()),add_FP=int((add&~truth).sum()),delete_TP=int((delete&truth).sum()),delete_FP=int((delete&~truth).sum())))
  details.append(episode)
 arrays={name:np.array(values) for name,values in arrays.items()}
 statistics=a.run/"sweep_counts_wide.npz"
 if statistics.exists():
  banks=("astra","p","all","far")+tuple(f"top{f:g}" for f in (.02,.05,.1,.2))+("split:astra","split:p","split:top0.05","split:top0.1")+("p0.5","p2","p4")
  deletions=list(itertools.product(banks,(0,5),(-.1,-.05,-.02,0.,.02,.05)))
  with np.load(statistics,allow_pickle=False) as z:
   for source,arm in [("iu_native","native"),("iu_rcg","rcg"),("iu_c","c.control"),("iu_astra","astra.control")]:
    if not np.array_equal(z[source],arrays[arm]):raise ValueError("Sweep/mask parity failed: "+arm)
   if len(z["d_rem"])!=len(deletions):raise ValueError("Unexpected deletion library")
   removed=z["d_rem"][deletions.index(("p",0,0.))]
   arrays["frozen.delete_p"]=arrays["c.control"]-removed
  for i,episode in enumerate(details):episode["iu"]["frozen.delete_p"]=arrays["frozen.delete_p"][i].tolist()
 result,_=summarize(rows,arrays,corrections)
 result.update(exposure="historical training-pool isolated600; not never-seen confirmation",source_seal_sha256=sha(a.run/"sealed.json"),
  missing_fields={"frozen.delete_p":"full-mask edits unavailable in count-only sweep; no zero counts substituted"},
  frozen_delete_p_recipe=["del[p,w0,t0]","none"],sweep_count_sha256=sha(statistics) if statistics.exists() else None)
 a.out.mkdir();(a.out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
 (a.out/"episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in details))
 (a.out/"receipt.json").write_text(json.dumps(dict(source_run=str(a.run),manifest_sha256=sha(a.run/"manifest.json"),inputs=receipts,source_code_sha256=sha(__file__)),indent=2)+"\n")
 print(json.dumps(dict(n=result["n"],scores=result["scores"],frozen=result["contrasts"].get("frozen.delete_p"))),flush=True)

if __name__=="__main__":main()
