"""One independent CPU statistic shared by final4000 and compact MEAN1200."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--run",type=Path,required=True)
p.add_argument("--counts",type=Path,required=True)
p.add_argument("--manifest",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
report=json.loads((a.run/"report.json").read_text())
doc=json.loads(a.manifest.read_text());rows=doc if isinstance(doc,list) else doc["episodes"]
z=np.load(a.counts);arrays={k[3:]:z[k] for k in z.files if k.startswith("iu:")}
photos={}
for i,r in enumerate(rows):
    for role in ("support","query"):
        photos.setdefault(Path(r[role]).name,[]).append(i)
adj=[set() for r in rows]
for ids in photos.values():
    for i in ids:adj[i].update(ids)
groups=np.full(len(rows),-1,int);g=0
for i in range(len(rows)):
    if groups[i]>=0:continue
    stack=[i];groups[i]=g
    while stack:
        current=stack.pop()
        for j in adj[current]:
            if groups[j]<0:groups[j]=g;stack.append(j)
    g+=1
draws=np.random.RandomState(0).randint(g,size=(2000,g))
if (a.run/"bootstrap_photo_draws.npy").exists():
    if not np.array_equal(draws,np.load(a.run/"bootstrap_photo_draws.npy")):raise ValueError("Source bootstrap draws differ")
classes=np.array([r["c"] for r in rows]);order=np.argsort(classes,kind="stable")
starts=np.r_[0,np.flatnonzero(np.diff(classes[order]))+1]
def metric(values,w):
    totals=np.add.reduceat(values[order]*w[order,None],starts,axis=0)
    present=np.add.reduceat(w[order],starts)>0
    return float(100*np.mean((totals[:,0]/np.maximum(totals[:,1],1))[present]))
points={k:metric(v,np.ones(len(rows),np.int64)) for k,v in arrays.items()}
samples={k:[] for k in arrays}
for draw in draws:
    w=np.bincount(draw,minlength=g)[groups]
    for k,v in arrays.items():samples[k].append(metric(v,w))
samples={k:np.array(v) for k,v in samples.items()}
errors=[]
for k,v in report["scores"].items():errors.append(abs(points[k]-v))
for arm,table in report["contrasts"].items():
    for base,r in table.items():
        errors.extend([abs(points[arm]-points[base]-r["gain"]),*abs(np.percentile(samples[arm]-samples[base],[2.5,97.5])-r["ci95"])])
if not np.isfinite(errors).all() or max(errors)>1e-10:raise ValueError("Source score or CI differs")
extras={}
primary="provided.rcg_tau15_sizecut"
if primary in arrays:
    for base in arrays:
        if base==primary:continue
        extras[base]=dict(gain=points[primary]-points[base],ci95=np.percentile(samples[primary]-samples[base],[2.5,97.5]).tolist())
result=dict(state="FINAL_COMPLETE_STATISTICS_INDEPENDENTLY_VERIFIED",n=len(rows),classes=len(set(classes)),photo_groups=g,
    exact2000_RS0_photo_draws=True,statistic="independent graph traversal,class-sorted integer reduceat",all_scores_CI_max_error=float(max(errors)),
    scores=points,provided_primary_vs_all_complete_controls=extras,
    source_report_sha256=hashlib.sha256((a.run/"report.json").read_bytes()).hexdigest())
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
