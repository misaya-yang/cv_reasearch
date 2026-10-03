#!/usr/bin/env python3
"""Fixed 17-constant FoRIS reading: CPU prepare -> GPU freeze -> CPU score.

No fitting, checkpoint rewriting, download, automatic retry, or shutdown. The
outer prepared ResourceGuard owns GPU lifecycle. Prediction never opens a
query-label PNG; scoring is unavailable until the ENTIRE manifest is frozen.
The constants pool supervised calibration across classes; this is NOT a
held-class/training-free FSS claim. Existing decision_infer science is unchanged.
"""
import argparse
import base64
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
ARMS = ("native", "readout", "removal")
CARD = dict(
    assumption="The SAME label-fitted 17-constant rule retains its patch-level benefit after original-resolution enlargement and original FoRIS refinement.",
    prediction="PLAN handover predicts CONFIRM600 +1.6..+2.4pp over its paired native row, all folds positive, fewer than25 episodes losing>10pp; this is a prediction, not an observed workers1 runtime or gain.",
    match="At least+1.5pp with all folds positive retains the fixed supervised component; unseen-host validation is still separate.",
    mismatch="Below+1.5pp or any negative fold drops this fixed component without retuning constants; errors/OOM preserve completed predictions and never count as completion.")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False)); tmp.replace(path)


def own_output(path):
    path = Path(path).resolve()
    path.relative_to(ROOT.resolve())
    return path


def photo(path):
    match = re.search(r"(?:^|_)(\d{12})\.(?:jpg|png)$", Path(path).name, re.I)
    return "coco:"+str(int(match.group(1))) if match else "path:"+str(Path(path))


def case_uid(row):
    return canonical({k: row[k] for k in ("fold", "e", "c", "support", "query")})


def formula_audit(path, fit_path, train_manifest=None):
    """Read small EXISTING trusted owned exports on CPU; do not alter either."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("arm") != "formula:relations" or saved.get("channels") != "relations":
        raise ValueError("Expected global formula:relations export")
    models = saved.get("models", {})
    if set(models) != set(range(4)):
        raise ValueError("Exactly four declared fold entries required")
    weights, bias = None, None
    for fold, entry in models.items():
        if (entry.get("rung") != "pixel" or entry.get("channels") != 16 or entry.get("host") != 0
                or len(entry.get("states", [])) != 1):
            raise ValueError("A single affine 16-map state required in each fold")
        state = entry["states"][0]
        if set(state) != {"body.weight", "body.bias", "scale", "shift"}:
            raise ValueError("Unexpected learned state beyond the 17 constants")
        w, b = state["body.weight"], state["body.bias"]
        if tuple(w.shape) != (1,16,1,1) or b.numel() != 1:
            raise ValueError("Not exactly 16 coefficients plus one bias")
        if (float(state["scale"]) != 0 or float(state["shift"]) != 0
                or not torch.equal(entry["mu"], torch.zeros_like(entry["mu"]))
                or not torch.equal(entry["sd"], torch.ones_like(entry["sd"]))):
            raise ValueError("Hidden scale/shift/standardization alters the fixed formula")
        if not torch.isfinite(w).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite fixed coefficients")
        if weights is None:
            weights, bias = w.detach().float().clone(), b.detach().float().clone()
        elif not torch.equal(weights, w.float()) or not torch.equal(bias, b.float()):
            raise ValueError("Formula changes across folds")
    fitted = torch.load(fit_path, map_location="cpu", weights_only=False)
    if fitted.get("arm") != "pixel:relations" or set(fitted.get("models", {})) != set(range(4)):
        raise ValueError("The source four-fold pixel fit is required to verify provenance")
    raw, const, seeds = [], [], []
    for fold in range(4):
        e = fitted["models"][fold]; states = e["states"]
        mu, sd = e["mu"].flatten().numpy(), e["sd"].flatten().numpy()
        w = np.mean([s["body.weight"].flatten().numpy() for s in states], 0)
        b = np.mean([float(s["body.bias"])+float(s["shift"]) for s in states])
        scale = np.mean([float(s["scale"]) for s in states])
        coefficient = w/sd; coefficient[0] += scale
        raw.append(coefficient)
        const.append(b-.5*scale-float((w*mu/sd).sum())); seeds.append(len(states))
    wd = float(np.max(np.abs(np.mean(raw,0)-weights.flatten().numpy())))
    bd = abs(float(np.mean(const))-float(bias))
    if wd > 1e-6 or bd > 1e-6:
        raise ValueError("Export does not reconstruct the average of the four supervised fits")
    from decision_fit import RELATIONS
    provenance = dict(type="pooled_class_supervised_calibration", training_free=False,
        strict_held_class_fewshot=False, fitted_with_query_labels_on_training_episodes=True,
        coefficient_aggregation="four-fold mean of per-fold mean-seed affine logits, NOT mean probabilities",
        source_models_sha256=sha(fit_path), source_seeds_per_fold=seeds,
        source_fit_rule="decision_fit.py: training collection_fold != model_holdfold; the average subsequently pools all folds",
        coefficient_reconstruction_maxabs=wd, bias_reconstruction_abs=bd,
        confirmation_previously_examined_at_patch_level=True,
        independent_unexamined_confirmation_claimed=False)
    if train_manifest:
        train = json.loads(Path(train_manifest).read_text())
        rows = train["episodes"]
        provenance.update(training_manifest_sha256=sha(train_manifest), training_episodes=len(rows),
            pooled_training_classes=sorted({int(r["c"]) for r in rows}),
            eligible_training_classes_by_source_model={str(f): sorted({int(r["c"]) for r in rows if int(r["fold"]) != f}) for f in range(4)},
            training_image_splits=sorted({Path(r["query"]).parts[0] for r in rows}),
            standard_train2014_fit_claimed=False)
    return dict(arm="formula:relations", channels=list(RELATIONS), constants=17,
                coefficients=weights.flatten().tolist(), bias=float(bias),
                same_formula_all_four_folds=True, model_sha256=sha(path), provenance=provenance)


def prepare(a):
    out = own_output(a.out)
    manifest = json.loads(Path(a.manifest).read_text())
    rows = manifest["episodes"]
    if manifest.get("state") != "PREPARED" or not rows:
        raise ValueError("Prepared nonempty episode manifest required")
    if a.expected_count is not None and len(rows) != a.expected_count:
        raise ValueError("Unexpected cohort size")
    if len({case_uid(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate scientific episode UID")
    for r in rows:
        if not (0<=int(r["fold"])<4 and 0<=int(r["c"])<80):
            raise ValueError("Invalid fold/class")
        if photo(r["support"]) == photo(r["query"]):
            raise ValueError("Reference equals query photo")
        for key in ("support","query"):
            rel = Path(r[key])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Dataset paths must be relative")
            for root, suffix in ((manifest["data_root"], ".jpg"),(manifest["annotation_root"], ".png")):
                path = Path(root)/rel.with_suffix(suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
                # Target PNG is ONLY stat-checked. No target pixels or hashes.
    audit = formula_audit(a.models, a.fit_models, a.train_manifest)
    local_names = ("formula_pipeline.py","decision_infer.py","decision_cache.py","decision_fit.py",
                   "extent_experiment.py","extent_train_cache.py")
    files = [HERE/name for name in local_names]+[ROOT/"tics"/name for name in ("relations.py","decision_heads.py","native_assets.py")]
    source_root = Path(a.foris_root or manifest["foris_root"])
    files += [source_root/name for name in ("models/foris.py","utils/clustering.py","utils/refinement.py","utils/data.py")]
    files.append(Path(manifest["projection_basis"]))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".py":
            compile(path.read_text(), str(path), "exec")
    baseline = Path(a.baseline_records).resolve() if a.baseline_records else None
    if baseline and not baseline.is_file():
        raise FileNotFoundError(baseline)
    payload = dict(schema="global_formula_pipeline_v1", state="PREPARED", manifest=str(Path(a.manifest).resolve()),
        manifest_sha256=sha(a.manifest), episodes=len(rows), case_uids=[case_uid(r) for r in rows],
        models=str(Path(a.models).resolve()), fit_models=str(Path(a.fit_models).resolve()), formula=audit,
        source_hashes={str(p.resolve()):sha(p) for p in files}, cache=str(Path(a.cache).resolve()),
        packets=str(Path(a.packets).resolve()) if a.packets else None,
        baseline_records=str(baseline) if baseline else None,
        baseline_records_sha256=sha(baseline) if baseline else None,
        foris_root=str(source_root.resolve()), demo4_root=a.demo4_root, fixture=bool(a.fixture),
        renderer=dict(dtype="FP32", TF32=False, formula="direct_affine_logit_then_bilinear_zero_cut",
            legacy_probability_clipping_reused=False, refiner="bilinear_fixture" if a.fixture else "original_public_FoRIS_CRF",
            original_resize="bilinear align_cornersFalse then >.5, same as decision_infer", maps="original FP16 cache rounding preserved"),
        expected_workers=a.workers, memory_fraction=min(.35,.8/a.workers), reserve_mib=512,
        query_GT_pixels_opened=False, card=CARD)
    payload["contract_sha256"] = canonical({k:v for k,v in payload.items() if k not in ("state","expected_workers","memory_fraction")})
    if out.exists() and json.loads(out.read_text()) != payload:
        raise ValueError("Existing prepared contract differs; preserve it and use a fresh output")
    write_json(out,payload)
    print(json.dumps(dict(state="CPU_PREPARATION_COMPLETE",out=str(out),episodes=len(rows),
                         same_formula_all_folds=True,query_GT_pixels_opened=False)))


def validated(a):
    p = json.loads(Path(a.prepared).read_text())
    if p.get("state") != "PREPARED" or p.get("schema") != "global_formula_pipeline_v1":
        raise ValueError("Frozen prepared formula contract required")
    if sha(p["manifest"]) != p["manifest_sha256"] or sha(p["models"]) != p["formula"]["model_sha256"]:
        raise ValueError("Manifest/formula drift")
    for path, expected in p["source_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Prepared scientific source drift: "+path)
    if p.get("baseline_records") and sha(p["baseline_records"]) != p["baseline_records_sha256"]:
        raise ValueError("Stored native baseline records drift")
    manifest = json.loads(Path(p["manifest"]).read_text())
    if [case_uid(r) for r in manifest["episodes"]] != p["case_uids"]:
        raise ValueError("Manifest scientific UID order differs")
    return p,manifest


def packed(mask):
    import numpy as np
    value = mask.detach().bool().cpu().numpy() if hasattr(mask,"detach") else np.asarray(mask,dtype=bool)
    payload = np.packbits(value.reshape(-1)).tobytes()
    return dict(shape=list(value.shape),codec="zlib_np_packbits_big",data=base64.b64encode(zlib.compress(payload,6)).decode(),
                bits_sha256=hashlib.sha256(payload).hexdigest())


def unpack(item):
    import numpy as np
    raw = zlib.decompress(base64.b64decode(item["data"]))
    if hashlib.sha256(raw).hexdigest()!=item["bits_sha256"]:
        raise ValueError("Frozen mask corruption")
    n = int(np.prod(item["shape"]))
    if len(raw)!=(n+7)//8:
        raise ValueError("Frozen mask shape/byte count differs")
    return np.unpackbits(np.frombuffer(raw,dtype=np.uint8))[:n].reshape(item["shape"]).astype(bool)


def case_path(out,index,uid):
    return Path(out)/"predictions"/("%04d_%s.json.gz"%(index,uid[:16]))


def read_case(path,p,row,index):
    with gzip.open(path,"rt") as stream:
        value=json.load(stream)
    if (value.get("contract_sha256")!=p["contract_sha256"] or value.get("case_uid")!=case_uid(row)
            or value.get("index")!=index or value.get("query_GT_pixels_opened") is not False):
        raise ValueError("Frozen case identity/protocol differs: "+str(path))
    for key in ("original_masks","model_masks","before_masks"):
        if set(value[key])!=set(ARMS):raise ValueError("Incomplete frozen prediction arms")
        for item in value[key].values():unpack(item)
    return value


def inventory(p,manifest,out):
    result=[]
    for i,row in enumerate(manifest["episodes"]):
        path=case_path(out,i,case_uid(row))
        if path.exists():
            rec=read_case(path,p,row,i)
            result.append(dict(index=i,case_uid=case_uid(row),path=str(path.resolve()),sha256=sha(path),seconds=rec["seconds"]))
    return result


@contextmanager
def forbid_query_png(path):
    from PIL import Image
    original=Image.open; resolved=Path(path).resolve()
    def guarded(fp,*args,**kwargs):
        if isinstance(fp,(str,os.PathLike)) and Path(fp).resolve()==resolved:
            raise RuntimeError("Query-label PNG opened inside prediction scope")
        return original(fp,*args,**kwargs)
    Image.open=guarded
    try:yield
    finally:Image.open=original


def resource_precheck(workers):
    """Real visible memory, before CUDA/model allocation; no invented peak estimate."""
    if not 1<=workers<=6:raise ValueError("workers must be1..6")
    query=subprocess.run(["nvidia-smi","--query-gpu=memory.total,memory.free","--format=csv,noheader,nounits"],
                         capture_output=True,text=True,timeout=15)
    values=[line.strip().split(",") for line in query.stdout.splitlines() if line.strip()]
    if query.returncode or len(values)!=1:raise RuntimeError("One real visible GPU inventory required")
    total,free=map(float,values[0]);fraction=min(.35,.8/workers)
    required=total*fraction*workers+512
    if free<required:raise RuntimeError("Insufficient free VRAM for declared worker reservations; no scientific precision fallback")
    return dict(total_mib=total,free_mib=free,workers=workers,per_worker_fraction=fraction,
                declared_reservation_mib=required,not_a_measured_peak=True)


def freeze_worker(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from decision_infer import evidence,symmetric
    from decision_cache import taps
    from extent_experiment import build_host,run_foris,finalise
    from tics.relations import near_mask
    p,manifest=validated(a);out=own_output(a.out);out.mkdir(parents=True,exist_ok=True)
    dev="cpu" if p["fixture"] else "cuda"
    if dev=="cuda" and os.environ.get("DEMO9_CUDA_GUARD")!="1":raise RuntimeError("Outer ResourceGuard required")
    report_path=out/("freeze_worker%d.json"%a.worker_index)
    report=dict(state="RUNNING",contract_sha256=p["contract_sha256"],new_episodes=0,reused_episodes=0,
                worker_index=a.worker_index,workers=a.workers,query_GT_pixels_opened=False)
    start=time.monotonic();write_json(report_path,report)
    try:
        if dev=="cuda":
            torch.cuda.set_per_process_memory_fraction(min(.35,.8/a.workers))
        args=argparse.Namespace(fixture=p["fixture"],foris_root=p["foris_root"],demo4_root=p["demo4_root"])
        loaded=time.monotonic();host=build_host(args,manifest,dev)
        report["model_load_s"]=time.monotonic()-loaded
        w=torch.tensor(p["formula"]["coefficients"],device=dev,dtype=torch.float32).view(1,16,1,1)
        b=torch.tensor([p["formula"]["bias"]],device=dev,dtype=torch.float32)
        near=None;rows=manifest["episodes"][:a.limit]
        with torch.inference_mode():
            for i in range(a.worker_index,len(rows),a.workers):
                row=rows[i];path=case_path(out,i,case_uid(row))
                if path.exists():
                    read_case(path,p,row,i);report["reused_episodes"]+=1;continue
                c=row["c"];data=Path(manifest["data_root"]);ann=Path(manifest["annotation_root"])
                begin=time.monotonic()
                with forbid_query_png(ann/Path(row["query"]).with_suffix(".png")):
                    sp=Image.open(data/row["support"]).convert("RGB")
                    qp=Image.open(data/row["query"]).convert("RGB")
                    gold=torch.from_numpy((np.asarray(Image.open(ann/Path(row["support"]).with_suffix(".png")))==c+1).copy())
                    with taps(host) as mid:native,got,ref_mask,tgt=run_foris(host,sp,gold,qp)
                    hw=tuple(native.shape);h,ww=got["score"].shape
                    if near is None or near.shape[0]!=h*ww:near=near_mask(h*ww,dev)
                    ev=evidence(got,mid,ref_mask,near)
                    x=symmetric(ev["maps"])
                    # The actual 17-constant AFFINE LOGIT, not inverse clipped sigmoid.
                    logit=F.conv2d(x[None].float(),w,b)[0,0]
                    cut=F.interpolate(logit[None,None],hw,mode="bilinear",align_corners=False)[0,0]>0
                    pre=got["pre"].reshape(hw).bool()
                    masks=dict(native=native,readout=finalise(host,cut,tgt),removal=finalise(host,cut&pre,tgt))
                    before=dict(native=pre,readout=cut,removal=cut&pre)
                    native_check=None;cache_check=None
                    key="%d_%d_%d"%(row["fold"],row["e"],c)
                    if p["packets"] and i<8:
                        stored=Path(p["packets"])/(key+".npz")
                        if not stored.is_file():raise FileNotFoundError("Required stored native packet: "+str(stored))
                        with np.load(stored,allow_pickle=False) as packet:expected=np.asarray(packet["native"],dtype=np.uint8).reshape(-1)
                        actual=np.packbits(native.cpu().numpy()).reshape(-1)
                        native_check=bool(actual.shape==expected.shape and np.array_equal(actual,expected))
                        if not native_check:raise RuntimeError("Complete native mask differs from frozen baseline")
                    if i<16:
                        for kind in ("confirm","test","train"):
                            stored=Path(p["cache"])/kind/(key+".npz")
                            if stored.is_file():
                                with np.load(stored,allow_pickle=False) as z:
                                    difference=max(float(np.abs(z["maps"].astype(np.float32)-ev["maps"].cpu().numpy()).max()),
                                                   float(np.abs(z["layers"].astype(np.float32)-ev["layers"].cpu().numpy()).max()))
                                cache_check=dict(maxabs=difference,legacy_tolerance=.03)
                                if difference>.03:raise RuntimeError("Evidence differs from fitted cache")
                                break
                    if dev=="cuda":torch.cuda.synchronize()
                    original_masks={k:packed(F.interpolate(v[None,None].float(),(qp.height,qp.width),mode="bilinear",align_corners=False)[0,0]>.5)
                                    for k,v in masks.items()}
                    rec=dict(index=i,case_uid=case_uid(row),contract_sha256=p["contract_sha256"],
                        row={k:row[k] for k in ("fold","e","c","support","query")},original_hw=[qp.height,qp.width],
                        model_hw=list(hw),original_masks=original_masks,model_masks={k:packed(v) for k,v in masks.items()},
                        before_masks={k:packed(v) for k,v in before.items()},seconds=time.monotonic()-begin,
                        query_GT_pixels_opened=False,native_packet_exact=native_check,cache_check=cache_check,
                        actual_public_FoRIS=True,actual_CRF=not p["fixture"],same_formula_all_folds=True)
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix(path.suffix+".tmp")
                with gzip.open(tmp,"wt") as stream:json.dump(rec,stream,allow_nan=False)
                tmp.replace(path)
                report["new_episodes"]+=1;report["elapsed_s"]=time.monotonic()-start
                write_json(report_path,report)
                print(json.dumps(dict(event="CASE_FROZEN",index=i,case_uid=rec["case_uid"],seconds=rec["seconds"],query_GT_pixels_opened=False)),flush=True)
        report.update(state="WORKER_COMPLETED",elapsed_s=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if dev=="cuda" else0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if dev=="cuda" else0)
        write_json(report_path,report)
    except BaseException as error:
        report.update(state="ERROR",error=repr(error),elapsed_s=time.monotonic()-start,
                      failure_kind="CUDA_OOM" if "out of memory" in str(error).lower() else "RUNTIME")
        write_json(report_path,report);raise


def freeze(a):
    p,manifest=validated(a);out=own_output(a.out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();target=len(manifest["episodes"][:a.limit]);report_path=out/"freeze_report.json"
    if not 1<=a.workers<=6:raise ValueError("workers must be1..6")
    previous=inventory(p,manifest,out)
    if previous and not a.resume:raise ValueError("Existing predictions require explicit --resume")
    if a.reuse_from:
        old=Path(a.reuse_from)
        for i,row in enumerate(manifest["episodes"][:a.limit]):
            src=case_path(old,i,case_uid(row));dst=case_path(out,i,case_uid(row))
            if src.exists() and not dst.exists():
                read_case(src,p,row,i);dst.parent.mkdir(parents=True,exist_ok=True)
                os.link(src,dst)  # same owned filesystem; never copy/re-encode assets
    report=dict(state="RUNNING",contract_sha256=p["contract_sha256"],cohort_episodes=p["episodes"],
                target_this_stage=target,workers=a.workers,query_GT_pixels_opened=False)
    write_json(report_path,report);jobs=[]
    try:
        complete=inventory(p,manifest,out)
        needed=set(range(target))-{r["index"] for r in complete}
        if needed:
            if not p["fixture"]:
                if os.environ.get("DEMO9_CUDA_GUARD")!="1":raise RuntimeError("Outer prepared ResourceGuard required")
                report["resource_precheck"]=resource_precheck(a.workers)
            argv=[sys.executable,str(Path(__file__).resolve()),"_worker","--prepared",str(Path(a.prepared).resolve()),
                  "--out",str(out),"--workers",str(a.workers)]
            if a.limit is not None:argv += ["--limit",str(a.limit)]
            for i in range(a.workers):
                if not any(index%a.workers==i for index in needed):continue
                log=(out/("freeze_worker%d.log"%i)).open("a")
                process=subprocess.Popen(argv+["--worker-index",str(i)],stdout=log,stderr=subprocess.STDOUT)
                log.close();jobs.append(process)
            while any(j.poll() is None for j in jobs):
                if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError("Formula worker failed; no retry")
                if time.monotonic()-start>a.timeout:raise TimeoutError("Finite formula freeze deadline exceeded")
                time.sleep(.5)
            if any(j.returncode for j in jobs):raise RuntimeError("Formula worker failed")
        frozen=inventory(p,manifest,out)
        if not set(range(target)).issubset({r["index"] for r in frozen}):raise RuntimeError("Incomplete/different frozen UID set")
        all_done=len(frozen)==p["episodes"]
        samples=[r["seconds"] for r in frozen if r["index"]<target]
        workers=[json.loads(path.read_text()) for path in out.glob("freeze_worker*.json")]
        report.update(state="PREDICTIONS_FROZEN" if all_done else "SMOKE_PREDICTIONS_FROZEN",
            elapsed_s=time.monotonic()-start,frozen_episodes=len(frozen),frozen_cases=frozen,
            all_case_uids=[r["case_uid"] for r in frozen],all_prediction_sha256=canonical([(r["case_uid"],r["sha256"]) for r in frozen]),
            reused_before_run=len(previous),new_episodes=len(needed),
            model_load_s=sum(float(w.get("model_load_s",0)) for w in workers),
            new_episode_seconds=samples,median_episode_seconds=sorted(samples)[len(samples)//2] if samples else None,
            peak_allocated_bytes=max([int(w.get("peak_allocated_bytes",0)) for w in workers]+[0]),
            peak_reserved_bytes=max([int(w.get("peak_reserved_bytes",0)) for w in workers]+[0]),
            formula=p["formula"],renderer=p["renderer"],query_GT_pixels_opened=False)
        write_json(report_path,report);print(json.dumps({k:v for k,v in report.items() if k not in ("frozen_cases","formula","all_case_uids","new_episode_seconds")}),flush=True)
    except BaseException as error:
        for job in jobs:
            if job.poll() is None:job.terminate()
        for job in jobs:
            try:job.wait(timeout=10)
            except subprocess.TimeoutExpired:job.kill();job.wait()
        report.update(state="ERROR",error=repr(error),elapsed_s=time.monotonic()-start)
        write_json(report_path,report);raise


def groups(rows):
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    seen={}
    for i,row in enumerate(rows):
        for key in ("support","query"):
            uid=photo(row[key])
            if uid in seen:parents[find(i)]=find(seen[uid])
            else:seen[uid]=i
    ids={};labels=[]
    for i in range(len(rows)):
        root=find(i);ids.setdefault(root,len(ids));labels.append(ids[root])
    return labels


def compare(recs,arm,key="original_iu",draws=2000):
    import numpy as np
    from analyze_extent import miou
    cls=np.array([r["c"] for r in recs]);fold=np.array([r["fold"] for r in recs])
    array=lambda k,j:np.array([r[key][k][j] for r in recs],float)
    ia,ua,ib,ub=array(arm,0),array(arm,1),array("native",0),array("native",1)
    gain=float(miou(ia,ua,cls)[0]-miou(ib,ub,cls)[0]);labels=np.array(groups(recs))
    count=int(labels.max())+1 if len(labels) else0
    ci=None
    if count>=2:
        multiplicity=np.random.default_rng(0).multinomial(count,np.ones(count)/count,size=draws)
        w=multiplicity[:,labels]
        delta=miou(ia,ua,cls,w)-miou(ib,ub,cls,w)
        ci=[float(v) for v in np.percentile(delta,[2.5,97.5])]
    per=[]
    for f in range(4):
        selected=fold==f
        per.append(float(miou(ia[selected],ua[selected],cls[selected])[0]-miou(ib[selected],ub[selected],cls[selected])[0]) if selected.any() else None)
    ep=100*(ia/np.maximum(ua,1)-ib/np.maximum(ub,1))
    return dict(miou=float(miou(ia,ua,cls)[0]),gain=gain,ci95=ci,per_fold=per,
        up=int((ep>1).sum()),down=int((ep<-1).sum()),lose_more_than_10=int((ep<-10).sum()),
        paired_unit="connected component of all-role support/query COCO photo IDs",
        independent_photo_groups=count,ci_available=count>=2,draws=draws,bootstrap_seed=0)


def score(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1)
    p,manifest=validated(a);out=own_output(a.out);report_path=out/"report.json";start=time.monotonic()
    try:
        frozen=json.loads((out/"freeze_report.json").read_text())
        if frozen.get("state")!="PREDICTIONS_FROZEN" or frozen.get("contract_sha256")!=p["contract_sha256"]:
            raise ValueError("ENTIRE cohort must be frozen before any query-label scoring")
        current=inventory(p,manifest,out)
        if len(current)!=p["episodes"] or [v["case_uid"] for v in current]!=p["case_uids"]:
            raise ValueError("Frozen scientific UID set differs")
        digest=canonical([(r["case_uid"],r["sha256"]) for r in current])
        if digest!=frozen["all_prediction_sha256"]:raise ValueError("Predictions changed before scoring")
        baseline={}
        if p["baseline_records"]:
            for line in Path(p["baseline_records"]).read_text().splitlines():
                if line.strip():
                    row=json.loads(line);uid=case_uid(row)
                    if uid in baseline:raise ValueError("Duplicate stored baseline UID")
                    baseline[uid]=row
        recs=[];native_matches=0
        tmp=out/"episodes.jsonl.tmp"
        with tmp.open("w") as stream:
            for i,row in enumerate(manifest["episodes"]):
                saved=read_case(case_path(out,i,case_uid(row)),p,row,i)
                # FIRST query-label PNG access in this backend; all cohort masks are frozen.
                truth=np.asarray(Image.open(Path(manifest["annotation_root"])/Path(row["query"]).with_suffix(".png")))==row["c"]+1
                if list(truth.shape)!=saved["original_hw"]:raise ValueError("Original query annotation/image geometry differs")
                tm=F.interpolate(torch.from_numpy(truth)[None,None].float(),saved["model_hw"],mode="nearest")[0,0].numpy().astype(bool)
                original={k:unpack(v) for k,v in saved["original_masks"].items()}
                model={k:unpack(v) for k,v in saved["model_masks"].items()}
                before={k:unpack(v) for k,v in saved["before_masks"].items()}
                iu=lambda m,t:[int((m&t).sum()),int((m|t).sum())]
                native=original["native"]
                ledger={k:dict(recovered_fn=int((~native&v&truth).sum()),lost_tp=int((native&~v&truth).sum()),
                               added_fp=int((~native&v&~truth).sum()),removed_fp=int((native&~v&~truth).sum())) for k,v in original.items()}
                rec=dict(**saved["row"],case_uid=saved["case_uid"],original_iu={k:iu(v,truth) for k,v in original.items()},
                    iu={**{k:iu(v,tm) for k,v in model.items()},**{"pre_"+k:iu(v,tm) for k,v in before.items()}},
                    ledger=ledger,area=float(tm.mean()),seconds=saved["seconds"],predictions_sha256=sha(case_path(out,i,case_uid(row))))
                if baseline:
                    expected=baseline.get(saved["case_uid"])
                    if expected is None or rec["original_iu"]["native"]!=expected["original_iu"]["native"]:
                        raise ValueError("Complete native original I/U differs from stored baseline")
                    native_matches+=1
                recs.append(rec);stream.write(json.dumps(rec)+"\n");stream.flush()
        tmp.replace(out/"episodes.jsonl")
        from extent_experiment import class_miou
        report=dict(state="COMPLETED",episodes=len(recs),expected=p["episodes"],arm="formula:relations",
            case_uids=p["case_uids"],all_prediction_sha256=digest,formula=p["formula"],renderer=p["renderer"],card=p["card"],
            scope="original-resolution class-sum I/U; SAME pooled-class supervised formula for every fold; previously examined image cohort",
            class_miou={k:class_miou(recs,k) for k in ARMS},rows={k:compare(recs,k,draws=a.draws) for k in ARMS[1:]},
            before_refinement={k:compare([{**r,"iu":{"native":r["iu"]["pre_native"],k:r["iu"]["pre_"+k]}} for r in recs],k,key="iu",draws=a.draws) for k in ARMS[1:]},
            native_original_IU_checks=native_matches,native_original_IU_exact=native_matches,
            native_packet_checks=sum(r.get("native_packet_exact") is not None for r in [read_case(case_path(out,i,case_uid(row)),p,row,i) for i,row in enumerate(manifest["episodes"])]),
            query_GT_pixels_opened_only_after_entire_cohort_frozen=True,
            ledger={k:{field:sum(r["ledger"][k][field] for r in recs) for field in ("added_fp","removed_fp","lost_tp","recovered_fn")} for k in ARMS[1:]},
            elapsed_s=time.monotonic()-start,fixture=p["fixture"])
        write_json(report_path,report);print(json.dumps(dict(state="COMPLETED",episodes=len(recs),rows=report["rows"])),flush=True)
    except BaseException as error:
        write_json(report_path,dict(state="ERROR",error=repr(error),elapsed_s=time.monotonic()-start,
                                    query_labels_may_have_been_scored=True));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","formula-freeze","score-frozen","_worker"))
    parser.add_argument("--manifest");parser.add_argument("--models");parser.add_argument("--fit-models")
    parser.add_argument("--train-manifest");parser.add_argument("--cache");parser.add_argument("--packets")
    parser.add_argument("--baseline-records");parser.add_argument("--prepared");parser.add_argument("--out",required=True)
    parser.add_argument("--expected-count",type=int);parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--worker-index",type=int,default=0);parser.add_argument("--limit",type=int)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--reuse-from")
    parser.add_argument("--timeout",type=int,default=2400);parser.add_argument("--draws",type=int,default=2000)
    parser.add_argument("--fixture",action="store_true");parser.add_argument("--foris-root")
    parser.add_argument("--demo4-root",default="/root/autodl-tmp/demo4")
    a=parser.parse_args()
    if not 1<=a.workers<=6:parser.error("workers must be1..6")
    if a.limit is not None and a.limit<=0:parser.error("limit must be positive")
    if not 0<a.timeout<=43200:parser.error("finite positive timeout<=43200 required")
    {"prepare":prepare,"formula-freeze":freeze,"score-frozen":score,"_worker":freeze_worker}[a.command](a)


if __name__=="__main__":main()
