#!/usr/bin/env python3
"""Frozen cached DEV post-analysis only. No encoder or query-GT decoder."""
import argparse, hashlib, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for x in iter(lambda:f.read(1048576),b""):h.update(x)
 return h.hexdigest()

def uid(p):return Path(p).name.rsplit("_",1)[-1].split(".")[0]

def groups(rows):
 parent={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 for r in rows:
  s,q=uid(r["row"]["support"]),uid(r["row"]["query"]);parent[find(s)]=find(q)
 g=defaultdict(list)
 for i,r in enumerate(rows):g[find(uid(r["row"]["query"]))].append(i)
 return list(g.values())

def metric(rows,arm,ids):
 byc=defaultdict(lambda:np.zeros(2));ep=[]
 for i in ids:
  r=rows[int(i)];v=np.asarray(r["original_iu"][arm],float)
  if v.shape!=(2,) or not 0<=v[0]<=v[1] or v[1]<=0:raise ValueError("invalid I/U")
  byc[int(r["row"]["c"])]+=v;ep.append(v[0]/v[1])
 total=sum(byc.values(),np.zeros(2))
 return {"episode_mean_IoU_pp":float(np.mean(ep)*100),
  "class_summed_IU_mIoU_pp":float(np.mean([v[0]/v[1] for v in byc.values()])*100),
  "pooled_IU_IoU_pp":float(total[0]/total[1]*100),
  "class_IU":{str(c):[int(v[0]),int(v[1])] for c,v in sorted(byc.items())}}

def intervals(rows,arms,g,reps,seed):
 if len(g)<2:return {"state":"NA_FEWER_THAN_TWO_PHOTO_GROUPS"}
 rng=np.random.default_rng(seed);store={a:{b:[[],[]] for b in ("native","score_same_count")} for a in arms}
 for _ in range(reps):
  ids=[i for k in rng.integers(0,len(g),len(g)) for i in g[int(k)]]
  vals={a:metric(rows,a,ids) for a in arms}
  for a in arms:
   for b in store[a]:
    for j,k in enumerate(("episode_mean_IoU_pp","class_summed_IU_mIoU_pp")):store[a][b][j].append(vals[a][k]-vals[b][k])
 return {"state":"EXPLORATORY_TINY_DESIGN_DEV","replicates":reps,"seed":seed,"groups":len(g),
  "unit":"all-role support/query photo UID connected components",
  "class_scope":"classes observed in each bootstrap sample",
  "CI95_pp":{a:{b:{"episode_mean":np.quantile(x[0],[.025,.975]).tolist(),
   "class_summed_IU":np.quantile(x[1],[.025,.975]).tolist()} for b,x in y.items()} for a,y in store.items()}}

def maskaudit(p,r,m,arms):
 out={}
 with np.load(p,allow_pickle=False) as z:
  for space,pre,shape,key in (("original","original__",r["row"]["query_original_hw"],"original_iu"),("model","",m["query_work_hw"],"model_iu")):
   n=int(np.prod(shape));base=np.unpackbits(z[pre+"native"],count=n).astype(bool);d={}
   for a in arms:
    cand=np.unpackbits(z[pre+a],count=n).astype(bool)
    added=int(np.count_nonzero(cand&~base));removed=int(np.count_nonzero(base&~cand))
    ib,ub=r[key]["native"];ia,ua=r[key][a];b=int(ib-ia);f=int(ub-ua);j=ib/ub
    if added or b<0 or f<0 or removed!=b+f:raise ValueError("deletion/IU mismatch: "+a)
    d[a]={"deleted_FP":f,"lost_TP":b,"removed_pixels":removed,"added_pixels":added,
     "b_lt_J_a":bool(b<j*f),"b_minus_J_a":float(b-j*f)}
   out[space]=d
 return out

def witaudit(p,m):
 # Only SOURCE cov is read; no query truth/GT array materialized.
 with np.load(m["input_files"]["packet"],allow_pickle=False) as z:cov=np.asarray(z["cov"],float).reshape(-1)
 c=Counter();gmin={"F":float("inf"),"B":float("inf")};detail=[]
 doc=json.loads(p.read_text())
 for r in doc["records"]:
  c["records"]+=1
  if r.get("status")!="ok":c["abstentions"]+=1;continue
  c["cost_records"]+=1;pr=r["joint_flip_predicate"];lf,lb,uf,ub=[float(pr[k]) for k in ("LF","LB","UF","UB")]
  flip=lf<lb and ub<uf
  c["strictflip_mismatch"]+=int(flip!=bool(r["delete"]["joint_flip"]) or flip!=bool(pr["accepted"]))
  c["gap_equivalence_mismatch"]+=int(flip!=((uf-lf)-(ub-lb)>lb-lf>0))
  c["unequal_role_banks"]+=int(len(r["sampled_F_stars"])!=len(r["sampled_B_stars"]))
  for role,l,u in (("F",lf,uf),("B",lb,ub)):
   gmin[role]=min(gmin[role],u-l);c["L_gt_U_exact_"+role]+=int(l>u)
   rc=r["role_costs"][role];c["cost_ledger_mismatch_"+role]+=int(rc["independent_edges"]!=l or rc["shared_U"]!=u)
   c["sampled_center_role_mismatch_"+role]+=sum(int((cov[int(s["root"])]>0)!=(role=="F")) for s in r["sampled_"+role+"_stars"])
  if not flip:continue
  c["accepted_strict_flip"]+=1;w={"query_root":r["query_root"],"LF":lf,"LB":lb,"UF":uf,"UB":ub}
  for role in ("F","B"):
   rc=r["role_costs"][role];win=rc["shared_winner"];bank={int(x["root"]):x for x in r["sampled_"+role+"_stars"]}
   ids=bank[int(win["source_root"])]["ids"];v=[float(cov[int(x)]) for x in ids]
   roots=[int(x["source_root"]) for x in rc["independent_winners"]];comps=[int(x["source_companion_id"]) for x in rc["independent_winners"]]
   c[role+"_shared_center_partial"]+=int(0<v[0]<1)
   for name,predicate in (("zero",lambda x:x==0),("partial",lambda x:0<x<1),("full",lambda x:x==1)):
    c[role+"_shared_companions_"+name]+=sum(int(predicate(x)) for x in v[1:])
   c[role+"_independent_different_roots"]+=int(roots[0]!=roots[1]);c[role+"_independent_reused_companion"]+=int(comps[0]==comps[1])
   w[role]={"shared_root":int(win["source_root"]),"shared_ids":ids,"reference_cov":v,
    "independent_roots":roots,"independent_companion_ids":comps}
  if len(detail)<20:detail.append(w)
 return {"counts":dict(c),"minimum_U_minus_L":{k:float(v) if np.isfinite(v) else None for k,v in gmin.items()},
  "first20_accepted":detail,"scope":"star occurrences among accepted query roots; repeated stars not independent",
  "FG":"reference center cov>0, companions need not be foreground","BG":"reference center cov==0, not query semantic certificate",
  "bank_scope":"frozen equal top8 banks; no full-source FG rescue","query_GT_reread":False}

def main():
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
 p.add_argument("--expected",type=int,default=10);p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--seed",type=int,default=31097);a=p.parse_args()
 if a.out.exists():raise SystemExit("refuse overwrite")
 rp=a.run/"report.json";ep=a.run/"episodes.jsonl";rr=json.loads(rp.read_text());rows=[json.loads(x) for x in ep.read_text().splitlines() if x.strip()]
 if len(rows)!=a.expected or rr.get("episodes")!=a.expected or not str(rr.get("state","")).startswith("COMPLETED"):
  raise SystemExit("NOT_READY expected%d rows%d/report%s state%s"%(a.expected,len(rows),rr.get("episodes"),rr.get("state")))
 keys=[(int(r["row"]["fold"]),int(r["row"]["e"]),int(r["row"]["c"])) for r in rows]
 if len(set(keys))!=len(keys):raise SystemExit("duplicate episode keys")
 if any(not r["predictions_frozen_before_query_GT"] or not r["native_pre_and_final_exact"] for r in rows):raise SystemExit("freeze/native contract failed")
 arms=list(rows[0]["original_iu"]);allids=list(range(len(rows)))
 if any(set(r["original_iu"])!=set(arms) for r in rows):raise SystemExit("inconsistent arms")
 table={x:metric(rows,x,allids) for x in arms}
 for arm in arms:
  for b in ("native","score_same_count"):
   table[arm]["minus_"+b+"_pp"]={k:table[arm][k]-table[b][k] for k in ("episode_mean_IoU_pp","class_summed_IU_mIoU_pp","pooled_IU_IoU_pp")}
 totals={s:{x:Counter() for x in arms} for s in ("original","model")};wt=Counter();per=[];provenance=[]
 for i,r in enumerate(rows):
  k="%d_%d_%d"%keys[i];pp=a.run/"predictions"/(k+".npz");wp=a.run/"witnesses"/(k+".json");mp=a.run/"witnesses"/(k+"_input.json");m=json.loads(mp.read_text())
  if sha(pp)!=r["predictions_sha256"]:raise SystemExit("prediction SHA mismatch "+k)
  ma=maskaudit(pp,r,m,arms);wa=witaudit(wp,m);wt.update(wa["counts"])
  if wa["counts"].get("accepted_strict_flip",0)!=r["deletion_counts"]["joint_flip"]["changed_patch_count"]:raise SystemExit("witness action mismatch "+k)
  for s in totals:
   for arm in arms:
    for field in ("deleted_FP","lost_TP","removed_pixels","added_pixels"):totals[s][arm][field]+=ma[s][arm][field]
  vals={x:metric(rows,x,[i])["episode_mean_IoU_pp"] for x in arms}
  per.append({"key":k,"row":r["row"],"original_IoU_pp":vals,"joint_minus_native_pp":vals["joint_flip"]-vals["native"],
   "joint_minus_samecount_pp":vals["joint_flip"]-vals["score_same_count"],"deletions":ma,"witness":wa,
   "same_pre_finalizer_patch_count":r["deletion_counts"]["joint_flip"]["changed_patch_count"]==r["deletion_counts"]["score_same_count"]["changed_patch_count"]})
  provenance.append({"key":k,"predictions_sha256":sha(pp),"witness_sha256":sha(wp),"metadata_sha256":sha(mp)})
 byfold={}
 for f in sorted(set(int(r["row"]["fold"]) for r in rows)):
  ids=[i for i,r in enumerate(rows) if int(r["row"]["fold"])==f];t={x:metric(rows,x,ids) for x in arms}
  byfold[str(f)]={"episodes":len(ids),"table":t,"joint_minus_native_episode_pp":t["joint_flip"]["episode_mean_IoU_pp"]-t["native"]["episode_mean_IoU_pp"],
   "joint_minus_samecount_episode_pp":t["joint_flip"]["episode_mean_IoU_pp"]-t["score_same_count"]["episode_mean_IoU_pp"]}
 gr=groups(rows);photos=Counter(uid(r["row"][role]) for r in rows for role in ("support","query"))
 orows=[dict(r,original_iu={"oracle":r["oracle_same_final_pixel_budget_original_iu"]}) for r in rows]
 out={"state":"COMPLETED_POST_FREEZE_TEN_DEV_DIAGNOSTIC","scope":"first10 frozen old design-DEV; exploratory, not independent method score",
  "episodes":len(rows),"dataset":"official COCO-20i existing cache","encoder_calls":0,"training":False,"method_changed":False,"query_GT_reread":False,
  "query_GT_note":"FP/TP derived from frozen stored I/U and masks; only lawful SOURCE cov read",
  "source_sha256":sha(Path(__file__)),"episode_rows_sha256":sha(ep),"runner_report_sha256":sha(rp),"original_grid_table":table,
  "oracle_deletion_arithmetic_anchor":dict(metric(orows,"oracle",allids),scope="GT arithmetic at joint final removal count, not star/CRF method upper bound"),
  "exploratory_paired_interval":intervals(rows,arms,gr,a.bootstrap,a.seed),"by_fold":byfold,
  "photo_audit":{"unique_UIDs":len(photos),"repeated_UID_counts":{k:v for k,v in photos.items() if v>1},"connected_episode_groups":gr},
  "removed_FP_lost_TP":{s:{x:dict(v) for x,v in t.items()} for s,t in totals.items()},"witness_totals":dict(wt),
  "joint_b_lt_J_a_episode_count":sum(x["deletions"]["original"]["joint_flip"]["b_lt_J_a"] for x in per),
  "joint_wins_vs_native":sum(x["joint_minus_native_pp"]>0 for x in per),"joint_wins_vs_samecount":sum(x["joint_minus_samecount_pp"]>0 for x in per),
  "samecount_scope":"equal pre-finalizer patch count only; final deletion count can differ",
  "unrun_control":"random same-count deletion absent, not tested","per_episode":per,"provenance":provenance,
  "decision_scope":"descriptive only; no rescue, bank expansion, sweep or automatic next run"}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,"x") as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({"state":out["state"],"episodes":len(rows),"table":{k:[v["episode_mean_IoU_pp"],v["class_summed_IU_mIoU_pp"]] for k,v in table.items()},
  "joint_minus_native_pp":table["joint_flip"]["minus_native_pp"]["episode_mean_IoU_pp"],
  "joint_minus_samecount_pp":table["joint_flip"]["minus_score_same_count_pp"]["episode_mean_IoU_pp"],
  "original_joint_removed":out["removed_FP_lost_TP"]["original"]["joint_flip"],"witness_violations":{k:v for k,v in wt.items() if "mismatch" in k or "L_gt_U" in k},
  "out":str(a.out)},ensure_ascii=False))
if __name__=="__main__":main()
