#!/usr/bin/env python3
"""One independent CPU count/statistics check of completed fixed4000 outputs.

No mask reconstruction, new GT indexing, encoder, parameter or GPU operation.
"""
import argparse,json,os,sys,time
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
import numpy as np
ARM="mean_fine_residual_transfer_v1"
BIT="mean_fine_bit_edit_transfer.control"
BASES=("native","rcg","mean.control","rcg64.control","fine.rcg16.control","fine.rcg64")
EDITS=("add_TP","add_FP","delete_TP","delete_FP")
ENTRY_SHA="9b14b0982a10c3bf3dea3f1a494e64cdb58eb724b911ff96c26a5715d7c6627e"
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--bit",type=Path,required=True);p.add_argument("--stats-src",type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.stats_src));from ics.experiment import sha,metric,photo_groups
 started=time.monotonic();run=a.run;report,state,seal,config=[read(run/x) for x in ("report.json","score_state.json","sealed.json","config.json")]
 def check(p,h):
  if sha(p)!=h:raise ValueError("Changed sealed file: "+str(p))
 if seal["state"]!="ALL_PREDICTIONS_SEALED" or seal["n"]!=4000 or state["state"]!="CPU_GT_SCORE_COMPLETE" or state["completed"]!=4000:raise ValueError("Require completed full4000 seal and score")
 check(run/"report.json",state["report_sha256"])
 for name,key in (("manifest.json","manifest_sha256"),("config.json","config_sha256"),("inference_config.json","inference_config_sha256"),("prepared.json","prepared_sha256")):check(run/name,seal[key])
 entry=next(Path(x) for x in config["source_code_sha256"] if x.endswith("/scripts/run_mean_fine_residual_transfer4000.py"));check(entry,ENTRY_SHA)
 if config["constructor"]!="C128=bilinear(G64,128)+(Y128-bilinear(F64,128)); FP32 align_corners=False coefficient1; no clip":raise ValueError("Fixed formula changed")
 rows=read(run/"manifest.json");keys={r["key"] for r in rows}
 if len(rows)!=4000 or len(keys)!=4000 or any(sum(r["fold"]==f for r in rows)!=1000 for f in range(4)) or set(r["c"] for r in rows)!=set(range(80)):raise ValueError("Original cohort changed")
 source=Path(config["scored"]);producer=Path(config["base"]);check(producer/"sealed.json",config["base_seal_sha256"]);check(source/"counts.npz",config["scored_counts_sha256"])
 bit=a.bit;bs,bstate,breport=[read(bit/x) for x in ("sealed.json","score_state.json","report.json")]
 if bs["state"]!="ALL_PREDICTIONS_SEALED" or bs["n"]!=4000 or bstate["state"]!="CPU_GT_SCORE_COMPLETE":raise ValueError("Fixed bit-control incomplete")
 check(bit/"report.json",bstate["report_sha256"])
 for name in ("manifest","config"):check(bit/(name+".json"),bs[name+"_sha256"])
 if rows!=read(source/"manifest.json") or rows!=read(bit/"manifest.json") or rows!=read(producer/"manifest.json"):raise ValueError("Same-cohort row/order identity failed")
 if any(set(seal[x])!=keys for x in ("fields","predictions","inputs")) or set(bs["predictions"])!=keys:raise ValueError("Incomplete mask manifest")
 replay={"shared1200":0,"new2800":0}
 for row in rows:
  r=seal["inputs"][row["key"]];audit=r["source_replay"]
  if r["fine16_render_mismatched_pixels"]!=0:raise ValueError("Original fine16 CUDA reconstruction failed")
  if r["mode"]=="shared1200_G_and_inherited_exact_lambda16_replay":
   replay["shared1200"]+=1;old=audit["inherited_rcg16_audit"]
   if any(old[x]!=0 for x in ("fp32_field_mismatched_entries","stored_rcg_mask_mismatched_pixels","maximum_field_difference")) or audit["mean_mask_mismatches"]!=0:raise ValueError("Inherited exact source replay failed")
  elif r["mode"]=="remaining2800_actual_Part1_RAM_replay":
   replay["new2800"]+=1
   if any(audit[x]!=0 for x in ("rcg16_field_mismatches","rcg16_mask_mismatches","mean_mask_mismatches")):raise ValueError("New exact source replay failed")
  else:raise ValueError("Unexpected source replay mode")
 if replay!={"shared1200":1200,"new2800":2800}:raise ValueError("Source replay accounting failed")
 with np.load(run/"counts.npz",allow_pickle=False) as z:
  arrays={arm:z["iu:"+arm].copy() for arm in (*BASES,ARM)}
  edits={base:z["edits_vs_"+base].copy() for base in ("native","rcg","mean.control","fine.rcg16.control","fine.rcg64")}
 with np.load(source/"counts.npz",allow_pickle=False) as z:
  if any(not np.array_equal(arrays[arm],z["iu:"+arm]) for arm in BASES):raise ValueError("Original six-arm4000 I/U differs")
 with np.load(bit/"counts.npz",allow_pickle=False) as z:
  if any(not np.array_equal(arrays[arm],z["iu:"+arm]) for arm in BASES):raise ValueError("Bit-control source six-arm4000 I/U differs")
  arrays[BIT]=z["iu:"+BIT].copy()
 episodes=[json.loads(line) for line in (run/"episodes.jsonl").read_text().splitlines()]
 if [r["key"] for r in episodes]!=[r["key"] for r in rows]:raise ValueError("Episode counts order changed")
 for arm in (*BASES,ARM):
  if arrays[arm].shape!=(4000,2) or arrays[arm].dtype!=np.int64 or not np.array_equal(arrays[arm],np.array([r["iu"][arm] for r in episodes],np.int64)):raise ValueError("Stored per-episode I/U disagrees")
 for base,counts in edits.items():
  if not np.array_equal(counts,np.array([[r["edits_vs_controls"][base][name] for name in EDITS] for r in episodes],np.int64)):raise ValueError("Stored four-edit array disagrees")
  rebuilt=arrays[base].copy();rebuilt[:,0]+=counts[:,0]-counts[:,2];rebuilt[:,1]+=counts[:,1]-counts[:,3]
  if not np.array_equal(rebuilt,arrays[ARM]):raise ValueError("Four-edit I/U closure failed")
 cls=np.array([r["c"] for r in rows]);groups=photo_groups(rows);g=int(groups.max())+1
 draws=np.random.RandomState(0).randint(g,size=(2000,g))
 if not np.array_equal(draws,np.load(run/"bootstrap_photo_draws.npy",allow_pickle=False)):raise ValueError("Saved bootstrap draws differ")
 weights=np.stack([np.bincount(d,minlength=g) for d in draws])[:,groups]
 scores={arm:metric(value,cls) for arm,value in arrays.items()};samples={arm:np.array([metric(value,cls,w) for w in weights]) for arm,value in arrays.items()};error=0.;contrasts={}
 for arm in (*BASES,ARM):error=max(error,abs(scores[arm]-report["scores"][arm]))
 error=max(error,abs(scores[BIT]-breport["scores"][BIT]))
 for base in (*BASES,BIT):
  delta=arrays[ARM][:,0]/np.maximum(arrays[ARM][:,1],1)-arrays[base][:,0]/np.maximum(arrays[base][:,1],1)
  value=dict(gain=scores[ARM]-scores[base],ci95=np.percentile(samples[ARM]-samples[base],[2.5,97.5]).tolist(),up=int((delta>1e-12).sum()),down=int((delta< -1e-12).sum()),tie=int((np.abs(delta)<=1e-12).sum()));contrasts[base]=value
  if base in BASES:
   saved=report["contrasts"][ARM][base];error=max(error,abs(value["gain"]-saved["gain"]),float(np.abs(np.array(value["ci95"])-saved["ci95"]).max()))
   if any(value[x]!=saved[x] for x in ("up","down","tie")):raise ValueError("Saved paired episode signs differ")
 folds,batches={},{}
 for field,target,table in (("fold",folds,report["folds"]),("batch",batches,report["batchs"])):
  for label in sorted({str(r[field]) for r in rows}):
   ix=np.array([str(r[field])==label for r in rows]);point={arm:metric(value[ix],cls[ix]) for arm,value in arrays.items()}
   for arm in (*BASES,ARM):error=max(error,abs(point[arm]-table[label]["scores"][arm]))
   target[label]=dict(n=int(ix.sum()),scores=point,candidate_gains={base:point[ARM]-point[base] for base in (*BASES,BIT)})
 if error>1e-10:raise ValueError("Saved score/CI/fold differs: "+str(error))
 totals={base:{name:int(counts[:,i].sum()) for i,name in enumerate(EDITS)} for base,counts in edits.items()}
 if totals["native"]!=report["corrections_vs_native"][ARM]:raise ValueError("Native edit totals differ")
 result=dict(state="INDEPENDENT_FULL4000_NECESSARY_STATISTICS_VERIFICATION_COMPLETE",n=4000,classes=80,photo_groups=g,natural_repeated_draws=4000-len({(r["c"],Path(r["support"]).name,Path(r["query"]).name) for r in rows}),candidate=ARM,scores=scores,contrasts=contrasts,folds=folds,batches=batches,corrections_vs_controls=totals,all_original_six_arm_IU_exact=True,all_saved_counts_and_four_edit_IU_identities_exact=True,saved_score_CI_fold_max_error_pp=error,source_replay=replay,fine16_CUDA_reconstruction_mismatches=0,formula=config["constructor"],frozen_inference_entry_sha256=ENTRY_SHA,bootstrap=dict(draws=2000,rng="RandomState(0)",unit="connected support/query photographs"),provenance=dict(run=str(run.resolve()),seal_sha256=sha(run/"sealed.json"),report_sha256=sha(run/"report.json"),counts_sha256=sha(run/"counts.npz"),bit_run=str(bit.resolve()),bit_seal_sha256=sha(bit/"sealed.json"),verifier_sha256=sha(Path(__file__)),stats_sha256=sha(a.stats_src/"ics/experiment.py")),seconds=time.monotonic()-started,new_encoder_or_GT_reads=False,interpretation="Fixed coefficient1/noClip scalar graft, not directP(G), not a replacement for original fine64 primary. CI crossing0 unresolved; exposed benchmark is not fresh confirmation.")
 (run/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
 lines=["# Independent full4000 scalar-graft statistics verification","",f"Candidate macro mIoU: **{scores[ARM]:.6f}**.","","| Comparator | Candidate gain [paired95% CI], pp | Up/down/tie |","|---|---:|---:|"]
 for base,v in contrasts.items():lines.append(f"| {base} | {v['gain']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] | {v['up']}/{v['down']}/{v['tie']} |")
 lines += ["","All4000 original six-arm I/U, per-episode exports and four-edit identities match exactly. Saved scores, CIs, folds and batches reproduce with zero error. The fixed bit control shares the same original4000 source counts and manifest.","One CPU count/statistics verification only: no masks changed, encoder run, query GT indexed, parameter sweep or inference snapshot edit. Source replay remains1200 shared/2800 new, with original RCG/MEAN and all fine16 renderer mismatches0.","C is not directP(G); the historical fine64 primary is unchanged. Native edits keep their complete-native origin; full4000 raw-origin edits remain unavailable. CI crossing0 is unresolved; this reused benchmark is not independent confirmation."]
 (run/"verification.md").write_text("\n".join(lines)+"\n")
 print(json.dumps(dict(state=result["state"],scores=scores,contrasts=contrasts,saved_error=error,seconds=result["seconds"])),flush=True)
if __name__=="__main__":main()
