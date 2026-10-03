#!/usr/bin/env python3
"""Finite DEVELOPMENT-only support mean-margin deletion. No training, sweep or full-feature cache.
--prepare reads metadata/source identities only. Root owns all numerical CPU/GPU execution.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
OWN = HERE.parent
sys.path.insert(0, str(OWN))
sys.path.insert(0, str(HERE))
ARMS = ("native", "middle12_remove", "finalraw_remove", "querycore_remove")
RULE = {
    "threshold": 0.0,
    "reference_readout": "unit tokens; unit FG mean minus unit BG mean; query dot difference",
    "middle": "actual block12 output after residual/MLP, without applying final norm",
    "final": "actual public _extract_features output BEFORE positional debias",
    "anchors": "exact original FoRIS bilinear>.5, nearest and tiny-target fallback",
    "primary": "full public native FINAL mask (one CRF) intersect bilinear scalar margin>0; NO second CRF",
    "original_resize": "bilinear binary mask then >.5, exact earlier241 evaluator",
    "patch_diagnostic": "area-downsample full native mask>.5; truth coverage>.5; no CRF at patch resolution",
    "querycore_control": "finalraw query unit mean inside native patch foreground minus outside; same zero rule",
    "degenerate": "missing either anchor class -> native no-op with explicit per-arm state",
    "scope": "ALL241 already-observed DEVELOPMENT; no independent confirmation claim",
    "forbidden": "no layer/threshold/keep-count sweep; no QKV/NxN export or regenerated fulltensor cache",
}
CARD = [
    "Assumption: the previously inspected block12 mean readout can REMOVE false foreground, beyond the IDENTICAL finalraw readout, without losing comparable true foreground.",
    "Prediction/required outcome (not an AUC-to-IoU forecast): native59.1218395 reproduces exactly; middle removal must reach at least62.1218395 (+3pp), paired lower bound>0, all4fold gains>0 AND beat finalraw with paired lower bound>0.",
    "Match: only this whole-mask gain supports expanding a frozen rule to a genuinely unseen confirmation cohort. Scalar AUC is not the method outcome.",
    "Mismatch: exact-baseline/source failure invalidates run; otherwise stop this fixed zero-margin removal, retain the TP/FP deletion ledger; no threshold/layer/grid/selection tuning.",
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def old_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def identity(row):
    return (int(row["fold"]), int(row["e"]), int(row["c"]), row["support"], row["query"])

def prepare(a):
    """No torch import, no model load, no numerical tests, no query labels."""
    m = json.loads(Path(a.manifest).read_text()); rows = m["episodes"]
    old = old_rows(a.old_records)
    if len(rows) != 241 or len(old) != 241 or [identity(r) for r in rows] != [identity(r) for r in old]:
        raise ValueError("exact completed241 rows/order required")
    report_path = Path(a.old_records).parent / "report.json"
    report = json.loads(report_path.read_text())
    if report["state"] != "COMPLETED" or report["episodes"] != 241:
        raise ValueError("old public241 terminal source required")
    dependencies = [Path(__file__), HERE/"extent_experiment.py", HERE/"evidence_pair_audit_cpu.py",
                    OWN/"tics/native_assets.py", Path(m["foris_root"])/"models/foris.py",
                    Path(m["foris_root"])/"utils/data.py", Path(m["foris_root"])/"utils/clustering.py",
                    Path(m["foris_root"])/"utils/refinement.py",
                    Path(a.demo4_root)/"icx/common.py"]
    inputs = []
    for r in rows:
        packet = Path(a.old_packets)/("%d_%d_%d.npz" % (r["fold"],r["e"],r["c"]))
        if not packet.is_file(): raise FileNotFoundError(packet)
        inputs.append(dict(row=list(identity(r)), packet=str(packet), packet_sha256=sha(packet)))
        for role in ("support", "query"):
            if not (Path(m["data_root"])/r[role]).is_file(): raise FileNotFoundError(r[role])
            if not (Path(m["annotation_root"])/Path(r[role]).with_suffix(".png")).is_file():
                raise FileNotFoundError(r[role]+" annotation")
    prefix = Path(a.prefix)
    plan = dict(state="PREPARED_NOT_NUMERICALLY_TESTED", manifest=str(Path(a.manifest).resolve()),
                manifest_sha256=sha(a.manifest), old_records=str(Path(a.old_records).resolve()),
                old_records_sha256=sha(a.old_records), old_packets=str(Path(a.old_packets).resolve()),
                old_native_class_miou=report["class_miou"]["native"],
                episodes=241, source_hashes={str(p.resolve()):sha(p) for p in dependencies},
                inputs=inputs, rule=RULE, card=CARD, max_seconds=1800, max_output_bytes=200*1024**2,
                reserve_bytes=5*1024**3, GPU_fraction=.4,
                CPU_check_required=str(prefix)+".cpu.json", demo4_root=a.demo4_root,
                no_downloads=True, no_full_tensor_cache=True, no_query_labels_opened=True)
    save(str(prefix)+".plan.json", plan)
    env = dict(DEMO9_CUDA_GUARD="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
               PYTHONPATH="/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions")
    out = str(prefix)+".run"
    gpu = dict(name="reference_removal241",kind="gpu",cwd=str(OWN),timeout_seconds=1850,
        argv=[a.python,str(Path(__file__)),"--run","--plan",str(prefix)+".plan.json","--out",out],
        code_files=list(plan["source_hashes"]),
        requires=[dict(path=str(prefix)+".plan.json"),
                  dict(path=str(prefix)+".cpu.json",json_equals=dict(state="CPU_REFERENCE_REMOVAL_CHECKED"))],
        produces=[dict(path=out+"/report.json",json_equals=dict(state="COMPLETED"))],
        success_checks=[dict(path=out+"/report.json",json_equals=dict(state="COMPLETED"))],env=env)
    cpu = dict(name="reference_removal241_analysis",kind="cpu",role="handoff",cwd=str(OWN),timeout_seconds=60,
        argv=[a.python,str(Path(__file__)),"--analyze",out+"/report.json","--out",out+"/analysis.json"],
        code_files=[str(Path(__file__))],
        requires=[dict(path=out+"/report.json",json_equals=dict(state="COMPLETED"))],
        produces=[dict(path=out+"/analysis.json",json_equals=dict(state="FROZEN_DEVELOPMENT_REMOVAL_ANALYZED"))],
        success_checks=[dict(path=out+"/analysis.json",json_equals=dict(state="FROZEN_DEVELOPMENT_REMOVAL_ANALYZED"))],
        env={**env,"CUDA_VISIBLE_DEVICES":""})
    save(str(prefix)+".queue.json",dict(platform="autodl",cuda_python=a.python,
         stages=[gpu,cpu],shutdown_authorized=False,
         scope="SECOND line only; root merges finite valuable T1+T2 batch and power policy"))
    save(str(prefix)+".card.json",dict(rule=RULE,card=CARD,CPU_numerical_checks="UNRUN_BY_PREPARER",
         required_gate="middle-native>=3;lower95>0;4fold positive;middle-finalraw lower95>0",
         old241_native=report["class_miou"]["native"]))
    print(json.dumps(dict(state=plan["state"],plan=str(prefix)+".plan.json")))

def mean_margin(t, fg):
    import torch
    import torch.nn.functional as F
    z = F.normalize(t.float(), p=2, dim=-1)
    if z.ndim != 3 or z.shape[0] != 2 or fg.shape != (z.shape[1],):
        raise ValueError("same-pair token/anchor contract")
    if not bool(fg.any()) or bool(fg.all()):
        return torch.zeros(z.shape[1],device=z.device), "DEGENERATE_NATIVE_NOOP"
    if not bool(torch.isfinite(z).all()): raise ValueError("nonfinite source token")
    mu = F.normalize(z[0,fg].mean(0),dim=0)-F.normalize(z[0,~fg].mean(0),dim=0)
    return z[1]@mu, "ACTIVE"

def remove(native, margin, state="ACTIVE"):
    import torch.nn.functional as F
    if state != "ACTIVE": return native.clone()
    value = F.interpolate(margin[None,None].float(),size=native.shape,
                          mode="bilinear",align_corners=False)[0,0]
    return native & (value > 0)  # strict zero; no fixed keep count

@contextmanager
def middle_hook(backbone):
    trace = {}
    def tap(module, args, output):
        if "tokens" in trace: raise RuntimeError("block12 must run once")
        if output.ndim != 3 or output.shape[0] != 2: raise RuntimeError("native S,Q B2 required")
        trace["tokens"] = output.detach().clone()
    handle = backbone.blocks[11].register_forward_hook(tap)
    try:
        yield trace
        if "tokens" not in trace: raise RuntimeError("block12 not captured")
    finally: handle.remove()

def cpu_check(out):
    """Root invokes on existing no-card CPU; numerical checks never run implicitly."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from evidence_pair_audit_cpu import source_ref_grid, readouts
    torch.set_num_threads(1)
    if torch.cuda.is_initialized(): raise RuntimeError("CPU-only check")
    torch.manual_seed(31064)
    t=torch.randn(2,8,5);fg=torch.tensor([1,1,1,0,0,0,0,0],dtype=torch.bool)
    got,state=mean_margin(t,fg);old=readouts(t,fg,2)["mean_margin"]
    assert state=="ACTIVE" and np.allclose(got.numpy(),old,rtol=0,atol=0)
    swapped,_=mean_margin(t,~fg);assert torch.equal(got,-swapped)
    native=torch.ones(4,4,dtype=torch.bool);margin=torch.tensor([[1.,0.],[-1.,1.]])
    mask=remove(native,margin);assert torch.equal(mask,remove(native,margin))
    assert not bool((mask & ~native).any())
    zeros=remove(native,torch.zeros(2,2));assert not bool(zeros.any())
    assert torch.equal(remove(native,torch.ones(2,2)),native)
    no,state=mean_margin(t,torch.ones(8,dtype=torch.bool));assert state=="DEGENERATE_NATIVE_NOOP"
    assert torch.equal(remove(native,no.reshape(2,4),state),native)
    ref=torch.zeros(8,8,dtype=torch.bool);ref[:,:4]=True
    assert source_ref_grid(ref,(2,2)).tolist()==[True,False,True,False]
    # Known maps: delete one FP while losing one TP, never adds FN back.
    truth=torch.tensor([[1,0],[1,0]],dtype=torch.bool)
    pred=torch.ones(2,2,dtype=torch.bool)
    edit=remove(pred,torch.tensor([[1.,-1.],[-1.,1.]]))
    assert int((pred&~edit&truth).sum())==1 and int((pred&~edit&~truth).sum())==1
    save(out,dict(state="CPU_REFERENCE_REMOVAL_CHECKED",tests=9,
         source_sha256=sha(__file__),CUDA_initialized=False,no_model_loaded=True,
         actual_DINO_parity="UNTESTED_FIRST_REAL_PAIRS_GATE",mean_matches_old_readout_exact=True,
         strict_zero_removal_and_noop_checked=True,known_maps_ledger_checked=True))

def run(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from types import SimpleNamespace
    from extent_experiment import build_host,run_foris
    from evidence_pair_audit_cpu import source_ref_grid,tokens
    if os.environ.get("DEMO9_CUDA_GUARD")!="1" or not torch.cuda.is_available():
        raise RuntimeError("real CUDA finite root guard required")
    p=json.loads(Path(a.plan).read_text())
    for file,expected in p["source_hashes"].items():
        if sha(file)!=expected: raise RuntimeError("source changed: "+file)
    if sha(p["manifest"])!=p["manifest_sha256"] or sha(p["old_records"])!=p["old_records_sha256"]:
        raise RuntimeError("frozen source/row manifest changed")
    check=json.loads(Path(p["CPU_check_required"]).read_text())
    if check.get("state")!="CPU_REFERENCE_REMOVAL_CHECKED" or check.get("source_sha256")!=sha(__file__):
        raise RuntimeError("matching current CPU numerical receipt required")
    man=json.loads(Path(p["manifest"]).read_text());old=old_rows(p["old_records"])
    root=Path(a.out)
    if root.exists() and any(root.iterdir()): raise RuntimeError("fresh own output required")
    (root/"packets").mkdir(parents=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(state="RUNNING",records=[],rule=RULE,plan_sha256=sha(a.plan),
         no_query_GT_in_predictions=True,no_full_tokens_saved=True,no_QKV_or_NxN_saved=True,
         baseline_exact_count=0,first2_unobserved_parity=[],output_bytes=0)
    start=time.monotonic()
    def status():
        report["elapsed_s"]=time.monotonic()-start;save(root/"report.json",report)
    status()
    try:
        with torch.inference_mode():
            host=build_host(SimpleNamespace(foris_root=None,demo4_root=p["demo4_root"],fixture=False),man,"cuda")
            for i,(row,oldrow) in enumerate(zip(man["episodes"],old)):
                if time.monotonic()-start>p["max_seconds"]: raise RuntimeError("finite1800s cap")
                if shutil.disk_usage(root).free<p["reserve_bytes"]: raise RuntimeError("5GiB reserve")
                sp=Image.open(Path(man["data_root"])/row["support"]).convert("RGB")
                qp=Image.open(Path(man["data_root"])/row["query"]).convert("RGB")
                ref=np.asarray(Image.open(Path(man["annotation_root"])/Path(row["support"]).with_suffix(".png")))==row["c"]+1
                ref_tensor=torch.from_numpy(ref.copy())
                plain=None
                if i<2:
                    torch.manual_seed(0);np.random.seed(0)
                    plain=run_foris(host,sp,ref_tensor,qp)[0]
                torch.manual_seed(0);np.random.seed(0)
                with middle_hook(host.encoder.m) as capture:
                    native,got,ref_model,tgt=run_foris(host,sp,ref_tensor,qp)
                if plain is not None:
                    exact=torch.equal(native,plain);report["first2_unobserved_parity"].append(exact)
                    if not exact: raise RuntimeError("observer changes source native mask")
                expected=p["inputs"][i]
                if list(identity(row))!=expected["row"] or sha(expected["packet"])!=expected["packet_sha256"]:
                    raise RuntimeError("old matched packet changed")
                with np.load(expected["packet"],allow_pickle=False) as z:
                    old_native=np.unpackbits(z["native"]).reshape(tuple(native.shape)).astype(bool)
                if not np.array_equal(native.cpu().numpy(),old_native):
                    raise RuntimeError("native final mask not BITEXACT to old public241")
                report["baseline_exact_count"]+=1
                grid=tuple(got["raw"].shape[-2:])
                fg=source_ref_grid(ref_model,grid)
                mid=tokens(capture["tokens"],int(host.encoder.m.num_prefix_tokens),grid)
                final=tokens(got["raw"],int(host.encoder.m.num_prefix_tokens),grid)
                mm,ms=mean_margin(mid,fg);fm,fs=mean_margin(final,fg)
                native_grid=F.interpolate(native[None,None].float(),size=grid,mode="area")[0,0]>.5
                # Same cheap cosine-mean readout; no support mask for this negative control.
                z=F.normalize(final[1].float(),p=2,dim=-1);core=native_grid.flatten()
                if bool(core.any()) and not bool(core.all()):
                    qm=z@(F.normalize(z[core].mean(0),dim=0)-F.normalize(z[~core].mean(0),dim=0));qs="ACTIVE"
                else: qm=torch.zeros(len(z),device=z.device);qs="DEGENERATE_NATIVE_NOOP"
                margins=dict(middle12=mm.reshape(grid),finalraw=fm.reshape(grid),querycore=qm.reshape(grid))
                states=dict(middle12=ms,finalraw=fs,querycore=qs)
                predictions={"native":native}
                for name in margins: predictions[name+"_remove"]=remove(native,margins[name],states[name])
                # Persist ALL label-free outputs before query GT is opened for this row.
                packet=root/"packets"/("%d_%d_%d.npz"%(row["fold"],row["e"],row["c"]))
                arrays={name:np.packbits(value.cpu().numpy()) for name,value in predictions.items()}
                arrays.update({name+"_margin":v.cpu().numpy() for name,v in margins.items()})
                arrays.update(source_score=got["score"].float().cpu().numpy(),shape=np.asarray(native.shape),grid=np.asarray(grid))
                np.savez_compressed(packet,**arrays)
                with packet.open("rb") as f:os.fsync(f.fileno())
                report["output_bytes"]+=packet.stat().st_size
                if report["output_bytes"]>p["max_output_bytes"]: raise RuntimeError("200MB scalar-only budget")
                # FIRST query-label access follows immutable output packet.
                truth_o=np.asarray(Image.open(Path(man["annotation_root"])/Path(row["query"]).with_suffix(".png")))==row["c"]+1
                truth=torch.from_numpy(truth_o.copy()).cuda()
                model_truth=F.interpolate(truth[None,None].float(),native.shape,mode="nearest")[0,0].bool()
                iu=lambda v,y:[int((v&y).sum()),int((v|y).sum())]
                orig=lambda v:F.interpolate(v[None,None].float(),truth_o.shape,mode="bilinear",align_corners=False)[0,0]>.5
                original={name:iu(orig(v),truth) for name,v in predictions.items()}
                if original["native"]!=oldrow["original_iu"]["native"]:
                    raise RuntimeError("old original-resolution I/U mismatch")
                ledger={}
                for name,v in predictions.items():
                    ledger[name]=dict(removed_fp=int((native&~v&~model_truth).sum()),
                                      lost_tp=int((native&~v&model_truth).sum()),
                                      added_fp=int((~native&v&~model_truth).sum()),
                                      recovered_fn=int((~native&v&model_truth).sum()))
                coverage=F.interpolate(model_truth[None,None].float(),grid,mode="area")[0,0]>.5
                patch={"native":native_grid}
                for name,v in margins.items():patch[name+"_remove"]=native_grid if states[name]!="ACTIVE" else native_grid&(v>0)
                # Oracle deletion is labels-only diagnostic; never persisted as a deployment arm.
                oracle=native&model_truth
                record={**{k:row[k] for k in ("fold","e","c","support","query")},
                    "original_iu":original,"model_iu":{name:iu(v,model_truth) for name,v in predictions.items()},
                    "patch_diagnostic_iu":{name:iu(v,coverage) for name,v in patch.items()},
                    "oracle_delete_original_iu":iu(orig(oracle),truth),
                    "ledger_model1024":ledger,"states":states,"outputs_frozen_before_query_GT":True,
                    "packet":str(packet),"packet_sha256":sha(packet)}
                report["records"].append(record);status()
                print(json.dumps(dict(completed=len(report["records"]),fold=row["fold"],native=original["native"],
                     middle=original["middle12_remove"],final=original["finalraw_remove"])),flush=True)
                del capture,got,native,plain,mid,final,predictions,arrays,tgt,ref_model
        if len(report["records"])!=241:raise RuntimeError("incomplete full241")
        report.update(state="COMPLETED",peak_allocated_bytes=torch.cuda.max_memory_allocated());status()
    except BaseException as error:
        report.update(state="ERROR",error=repr(error));status();raise

def analyze(a):
    import numpy as np
    report=json.loads(Path(a.analyze).read_text());rows=report["records"]
    if report["state"]!="COMPLETED" or len(rows)!=241:raise ValueError("complete241 required")
    def score(arm,subset):
        by={}
        for r in subset:
            v=r["oracle_delete_original_iu"] if arm=="oracle_delete" else r["original_iu"][arm]
            z=by.setdefault(r["c"],[0,0]);z[0]+=v[0];z[1]+=v[1]
        return 100*sum(v[0]/max(v[1],1) for v in by.values())/len(by)
    # Cluster influence CI: connect every reused photograph across BOTH support/query roles.
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        for role in ("support","query"):
            if r[role] in seen:parent[find(i)]=find(seen[r[role]])
            else:seen[r[role]]=i
    groups={}
    for i in range(len(rows)):groups.setdefault(find(i),[]).append(i)
    def comparison(arm,base):
        classes=sorted({r["c"] for r in rows});tot={}
        for name in (arm,base):
            tot[name]={c:[0,0] for c in classes}
            for r in rows:
                v=r["original_iu"][name];z=tot[name][r["c"]];z[0]+=v[0];z[1]+=v[1]
        influences=[]
        for indices in groups.values():
            value=0.
            for name,sign in ((arm,1),(base,-1)):
                for i in indices:
                    r=rows[i];I,U=tot[name][r["c"]];ii,uu=r["original_iu"][name]
                    value+=sign*(ii-(I/max(U,1))*uu)/max(U,1)
            influences.append(100*value/len(classes))
        g=len(influences)
        se=float(np.linalg.norm(influences)*math.sqrt(g/(g-1))) if g>1 else None
        d=score(arm,rows)-score(base,rows)
        interval=None if se is None else [d-1.96*se,d+1.96*se]
        folds={str(f):score(arm,[r for r in rows if r["fold"]==f])-score(base,[r for r in rows if r["fold"]==f]) for f in range(4)}
        return dict(delta_pp=d,interval95=interval,all_fold_deltas_pp=folds,
                    interval_method="asymptotic photo-connected-cluster influence sandwich; NOT episode-independent bootstrap",
                    photo_groups=g,development_only=True)
    comparisons={name:comparison(name,"native") for name in ARMS if name!="native"}
    comparisons["middle_vs_final"]=comparison("middle12_remove","finalraw_remove")
    m=comparisons["middle12_remove"];f=comparisons["middle_vs_final"]
    passed=(m["delta_pp"]>=3 and m["interval95"][0]>0 and
            all(v>0 for v in m["all_fold_deltas_pp"].values()) and f["interval95"][0]>0)
    save(a.out,dict(state="FROZEN_DEVELOPMENT_REMOVAL_ANALYZED",
         scores_original_class_miou={name:score(name,rows) for name in (*ARMS,"oracle_delete")},
         comparisons=comparisons,gate_passed=passed,records=241,rule=RULE,
         decision="prepare NEW frozen confirmation only" if passed else "stop fixed zero-margin removal",
         no_AUC_method_claim=True,patch_and_model_results_never_substitute_primary=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepare",action="store_true");p.add_argument("--cpu-check")
    p.add_argument("--run",action="store_true");p.add_argument("--analyze")
    p.add_argument("--manifest",default="/root/autodl-tmp/demo9_extent/results/extent_v1/episodes.json")
    p.add_argument("--old-records",default="/root/autodl-tmp/demo9_extent/results/extent_v1/run/episodes.jsonl")
    p.add_argument("--old-packets",default="/root/autodl-tmp/demo9_extent/results/extent_v1/run/packets")
    p.add_argument("--demo4-root",default="/root/autodl-tmp/demo4")
    p.add_argument("--python",default="/root/miniconda3/bin/python")
    p.add_argument("--prefix",default=str(OWN/"results/reference_removal_241_preparation"))
    p.add_argument("--plan");p.add_argument("--out")
    a=p.parse_args()
    if a.prepare:prepare(a)
    elif a.cpu_check:cpu_check(a.cpu_check)
    elif a.run:
        if not a.plan or not a.out:p.error("--run --plan --out")
        run(a)
    elif a.analyze:
        if not a.out:p.error("--analyze --out")
        analyze(a)
    else:p.error("choose --prepare, --cpu-check, --run or --analyze")
if __name__=="__main__":main()

