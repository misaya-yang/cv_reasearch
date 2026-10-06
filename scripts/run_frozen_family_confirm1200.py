#!/usr/bin/env python3
"""Locked photo-disjoint CONFIRM1200: full FoRIS + one shared fine reader.

preflight is CPU metadata/header only. wait-inputs observes the predecessor.
smoke replays three OLD golden cases without query truth. infer seals ALL1200
complete arms and reusable FP32 cosine/rowden before score opens any query GT.
No retries, shutdown, q/r archive, parameter selection, or confirmation resampling.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import metric, photo_groups, render, sha, summarize, unpack
from run_frozen_subtoken1200 import FINE16, FINE64, verify_code, write
from run_frozen_subtoken4000 import BASIS_SHA, fine_grid, make_host

ROOT = Path("/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9")
MANIFEST_SHA = "21475d39973d448a81140b87db875dc70a47c4f60a53d436ccea8bc8432d3ff6"
FREEZE_SHA = "5fec4eca46825876d40617e8f7c53a38abfd4fc0716305014767cda0477f42c7"
RECEIPT_SHA = "028e378c2c9eaa519c052395cbcaeccd4a2328b520a5f14276fc572520ec64f9"
VERIFY_SHA = "235784280ad285462afb4a5efdfdda628883d16bfe16d8c1505493b1e409602a"
EXPOSURE_SHA = "fb3c6132258310e8be3ea86fca46bc251b45f8f2c0325d6e79104cf1372c9714"
HOST_SHA = "14e99f553d93a3234d78551e67727fc3478ca7042b161d9b465dae5cc9769f2b"
HOST_MANIFEST_SHA = "4dd6078054e91c7fc90df4443120624b763539e83c1c7752cd031bfb4c859d22"
BASE_CONFIG_SHA = "12ebd209820910d5d3b02357750cc58856279c8051a9760b1776697357479c07"
MEAN_FINE = "fine.mean16.control"
GRAFT = "mean_fine_residual_transfer_v1"
PRIMARY = "strict.family12.frozen"
ARMS = ("native", "rcg", "mean.control", "rcg64.control", FINE16, FINE64, GRAFT, MEAN_FINE, PRIMARY)
FROZEN_PARAMS = {str(f): dict(arm="feat[s1.25,t%s]" % ("0.07" if f in (0, 3) else "0.15"),
                            sigma=1.25, tau=.07 if f in (0, 3) else .15) for f in range(4)}


def checked_json(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError("Frozen metadata/source changed: " + str(path))
    return json.loads(Path(path).read_text())


def cohort(a):
    man = checked_json(a.manifest, MANIFEST_SHA)
    freeze = checked_json(a.freeze, FREEZE_SHA)
    receipt = checked_json(a.manifest.parent / "receipt.json", RECEIPT_SHA)
    verification = checked_json(a.manifest.parent / "verification.json", VERIFY_SHA)
    if sha(a.manifest.parent / "exposure.json") != EXPOSURE_SHA:
        raise ValueError("Registered exposure changed")
    rows = man["episodes"]
    if (len(rows) != 1200 or len({r["key"] for r in rows}) != 1200
            or any(sum(r["fold"] == f for r in rows) != 300 for f in range(4))
            or len({r["c"] for r in rows}) != 74
            or sorted(set(range(80)) - {r["c"] for r in rows}) != [32, 35, 68, 70, 78, 79]):
        raise ValueError("Locked CONFIRM1200 cohort changed; no resampling/rebalancing")
    if (receipt["overlap_count"] != 0 or receipt["blocked_photo_count"] != 17741
            or receipt["confirmation_unique_photos"] != 2249
            or verification["manifest_sha256"] != MANIFEST_SHA
            or verification["state"] != "METADATA_CONFIRMATION_INDEX_VERIFIED"):
        raise ValueError("Photo-disjoint metadata verification changed")
    if freeze["parameters"] != FROZEN_PARAMS or freeze["controls"] != list(ARMS[:-1]):
        raise ValueError("Frozen complete controls/parameters changed")
    if freeze["fixed_recipe"] != "C=(fine16 & scalar_graft) | (~fine16 & fine64)":
        raise ValueError("Frozen global recipe changed")
    if checked_json(a.base4000 / "config.json", BASE_CONFIG_SHA)["parameters"] != FROZEN_PARAMS:
        raise ValueError("Original full4000 policy changed")
    for name, digest in freeze["source_code_sha256"].items():
        if sha(REPO / name) != digest:
            raise ValueError("Frozen numerical source changed: " + name)
    for path, digest in freeze["source_producer_bindings"].items():
        if sha(path) != digest:
            raise ValueError("Frozen original producer changed: " + path)
    for name in ("components.py", "rcg_readout.py", "hypothesis_source_contrast.py"):
        p = a.mean_source / name
        digest = freeze["source_producer_bindings"].get(str(p.resolve()))
        if digest is None or sha(p) != digest:
            raise ValueError("Original MEAN source binding changed: " + str(p))
    if (sha(a.host_root / "scripts/extent_experiment.py") != HOST_SHA
            or sha(a.host_manifest) != HOST_MANIFEST_SHA or sha(a.basis) != BASIS_SHA):
        raise ValueError("Original full FoRIS host/basis changed")
    return rows, man, freeze


def storage(n=1200):
    # File header/ZIP sizes are exact; fields/masks are compressed at runtime.
    stream = io.BytesIO()
    np.savez(stream, cos=np.zeros((16384, 25), np.float32), denominator=np.zeros(16384, np.float32))
    return dict(cos_rowden_payload_bytes=n * (16384 * 25 + 16384) * 4,
                cos_rowden_archive_bytes=n * len(stream.getvalue()),
                scalar_fields_payload_bytes=n * (5 * 4096 + 4 * 16384) * 4,
                packed_masks_payload_bytes=n * len(ARMS) * 131072,
                compressed_fields_masks_actual_bytes="measured only after inference; raw payload is conservative sizing",
                minimum_free_bytes=5 << 30, q_r_archives_created=0, fine_feature_archives_created=0)


def old_golden(a):
    rows = checked_json(a.base1200 / "manifest.json")
    if isinstance(rows, dict): rows = rows["episodes"]
    seal = checked_json(a.base1200 / "sealed.json")
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 1200:
        raise ValueError("Old golden1200 source is incomplete")
    if sha(a.base1200 / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("Old golden manifest changed")
    config = checked_json(a.base1200 / "config.json", "b641e7c64e8d76c9886c00779c794b30390b263e5c40b8d03e5fa629362d4a51")
    if config["parameters"] != FROZEN_PARAMS or sha(a.base1200 / "config.json") != seal["config_sha256"]:
        raise ValueError("Old golden configuration changed")
    # Fixed first draw in three folds covers tau .07 and .15; no score-based choice.
    selected = [next(r for r in rows if r["fold"] == f) for f in (0, 1, 3)]
    return selected, seal


def preflight(a):
    from PIL import Image
    rows, man, freeze = cohort(a)
    headers = []
    for row in rows:
        values = {}
        for role in ("support", "query"):
            p = Path(man["data_root"]) / row[role]
            with Image.open(p) as im:
                values[role] = dict(path=str(p), size=list(im.size), mode=im.mode, bytes=p.stat().st_size)
        # Annotation existence/header only. No query annotation bytes or pixels read.
        for role in ("support", "query"):
            p = Path(man["annotation_root"]) / Path(row[role]).with_suffix(".png")
            if not p.is_file(): raise FileNotFoundError(p)
            values[role + "_annotation"] = dict(path=str(p), bytes=p.stat().st_size, pixels_read=False)
        headers.append(dict(key=row["key"], **values))
    golden, seal = old_golden(a)
    import torch
    golden_headers = []
    for row in golden:
        key = row["key"]
        cached = torch.load(row["feature_export"], map_location="cpu", weights_only=True)
        if any(cached[k].shape != (4096, 1024) or cached[k].dtype != torch.float16 for k in ("q", "r")):
            raise ValueError("Old golden q/r header changed")
        if sha(row["feature_export"]) != seal["inputs"][key]["feature_sha256"]:
            raise ValueError("Old golden q/r changed")
        for folder in ("fields", "predictions"):
            if sha(a.base1200 / folder / (key + ".npz")) != seal[folder][key]:
                raise ValueError("Old golden output changed")
        with np.load(a.base1200 / "fields" / (key + ".npz"), allow_pickle=False) as z:
            fields = {k: dict(shape=list(z[k].shape), dtype=str(z[k].dtype)) for k in (FINE16, FINE64)}
        golden_headers.append(dict(key=key, fold=row["fold"], fields=fields, q_r_dtype="float16", query_truth_opened=False))
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "input_headers.json", headers)
    result = dict(state="CONFIRM1200_CPU_METADATA_HEADERS_VERIFIED_NO_GPU", n=1200,
        manifest_sha256=MANIFEST_SHA, freeze_sha256=FREEZE_SHA, receipt_sha256=RECEIPT_SHA,
        verification_sha256=VERIFY_SHA, exposure_sha256=EXPOSURE_SHA, classes=74,
        absent_classes=[32,35,68,70,78,79], unique_photos=2249, blocked_photos=17741, photo_overlap=0,
        golden=golden_headers, golden_seal_sha256=sha(a.base1200 / "sealed.json"),
        storage=storage(), free_bytes=shutil.disk_usage(a.out).free,
        input_headers_sha256=sha(a.out / "input_headers.json"), GPU_executed=False, query_truth_opened=False,
        predecessor_sealed=(a.predecessor / "sealed.json").is_file(),
        pending_GPU="old3 exact operation smoke, full1200 cold inference, post-seal CPU GT score")
    write(a.out / "preflight.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("golden",)}), flush=True)


def wait_inputs(a):
    cohort(a)
    while True:
        for p in (a.predecessor / "state.json", a.predecessor_pipeline):
            if p.is_file():
                state = checked_json(p)
                if any(token in state.get("state", "") for token in ("FAILED", "ERROR", "INTERRUPTED")):
                    raise RuntimeError("Declared DirectMEAN4000 predecessor failed; no retry: " + str(p))
        p = a.predecessor / "sealed.json"
        if p.is_file():
            seal = checked_json(p)
            if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 4000:
                raise ValueError("Declared predecessor seal is incomplete")
            a.out.parent.mkdir(parents=True, exist_ok=True)
            write(a.out, dict(state="CONFIRM1200_PREDECESSOR_SEALED_READY", predecessor=str(a.predecessor),
                            predecessor_seal_sha256=sha(p), query_truth_opened=False))
            return
        time.sleep(10)


def solve_coarse(job):
    """Source-identical CPU equations; q/r are actual full-host FP16 tokens."""
    import torch
    import torch.nn.functional as F
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.methods.rcg import minmax, rank
    torch.set_num_threads(1)
    q16, r16, cov, score, source = job
    q, r = F.normalize(torch.from_numpy(q16).float(), dim=1), F.normalize(torch.from_numpy(r16).float(), dim=1)
    pure = np.flatnonzero(cov.ravel() >= .9)
    if not len(pure): pure = np.flatnonzero(cov.ravel() == cov.max())
    sim = q @ r.T
    dq, dr = sim.topk(10, dim=1).values.mean(1), sim.topk(10, dim=0).values.mean(0)
    guide = ((2 * sim[:, pure] - dr[pure][None]).max(1).values - dq).numpy()
    del sim, dq, dr, r
    s = minmax(score).ravel()
    y = (s + .5 * (rank(guide) - rank(s))).astype(np.float64)
    sim = q @ q.T
    sim.fill_diagonal_(-2)
    values, ids = sim.topk(20, dim=1)
    del sim, q
    distance = (1 - values).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), ids.numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel()
    w = w / max(float(degree.mean()), 1e-8)
    lap = sparse.diags(np.asarray(w.sum(1)).ravel()) - w
    conf = .1 + np.abs(2 * s - 1)
    conf = (conf / conf.mean()).astype(np.float64)
    rhs, anchor = conf * y, sparse.diags(conf)
    fields, audit = {}, {}
    for lam in (16, 64):
        matrix = anchor + lam * lap
        solved, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300)
        residual = float(np.linalg.norm(matrix @ solved - rhs))
        tolerance = max(1e-9, 1e-7 * float(np.linalg.norm(rhs)))
        if status != 0 or not np.isfinite(solved).all() or residual > 1.01 * tolerance:
            raise RuntimeError("Frozen lambda%d CG failed" % lam)
        name = "rcg" if lam == 16 else "rcg64.control"
        fields[name] = solved.reshape(64, 64).astype(np.float32)
        audit[str(lam)] = dict(cg_status=int(status), absolute_residual=residual, residual_tolerance=tolerance)
    sys.path.insert(0, source)
    import components
    # Call the exact frozen source function; no alternate mean or graft prior.
    fields["mean.control"], audit["mean"] = components.mean_control(q16.astype(np.float32), r16.astype(np.float32), cov, score, components.rcg)
    masks = {name: np.packbits(render(field)) for name, field in fields.items()}
    return fields, masks, audit


def load_rgb(row, man):
    from PIL import Image
    import torch
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    paths = {k: data / row[k] for k in ("support", "query")}
    with Image.open(paths["support"]) as im: support = im.convert("RGB")
    with Image.open(paths["query"]) as im: query = im.convert("RGB")
    reference_path = ann / Path(row["support"]).with_suffix(".png")
    with Image.open(reference_path) as im:
        reference = torch.from_numpy((np.asarray(im) == row["c"] + 1).copy())
    receipt = {k + "_image_sha256": sha(p) for k, p in paths.items()}
    receipt.update(reference_mask_sha256=sha(reference_path), query_truth_opened=False)
    return support, query, reference, receipt


def full_pair(host, producer, inputs):
    import torch.nn.functional as F
    support, query, reference, receipt = inputs
    native, got, mask, target = producer.run_foris(host, support, reference, query)
    raw, deb = F.normalize(got["raw"][0], dim=1), got["deb"][0]
    if str(raw.dtype) != "torch.float32" or str(deb.dtype) != "torch.float32":
        raise ValueError("Actual paired forward must remain FP32")
    q16, r16 = deb[-1].flatten(1).T.half().cpu().numpy(), deb[0].flatten(1).T.half().cpu().numpy()
    cov = F.interpolate(mask[None,None].float(), (64,64), mode="area")[0,0].cpu().numpy()
    score = got["score"].float().cpu().numpy()
    if q16.shape != (4096,1024) or r16.shape != q16.shape or score.shape != (64,64) or native.shape != (1024,1024):
        raise ValueError("Full FoRIS producer shape changed")
    debiased = bool((raw - deb).abs().max() > 1e-4)
    receipt.update(debiased=debiased, host_score="actual complete full FoRIS final score, no cached Part1 substitute")
    return q16, r16, cov, score, np.packbits(native.cpu().numpy()), target, receipt


def neighbor_geometry(device="cuda"):
    import torch
    fi = torch.arange(128, device=device); off = torch.arange(-2,3,device=device)
    base = ((fi*8+4-8).float()/16).round().long()
    ti = base[:,None] + off[None]
    distance = ((ti*16+8)-(fi*8+4)[:,None]).float()/16
    ok = (ti>=0)&(ti<64); ti=ti.clamp(0,63)
    ny=ti[:,None,:,None].expand(128,128,5,5); nx=ti[None,:,None,:].expand(128,128,5,5)
    nb=(ny*64+nx).reshape(16384,25)
    d2=(distance[:,None,:,None].square()+distance[None,:,None,:].square()).reshape(16384,25)
    valid=(ok[:,None,:,None]&ok[None,:,None,:]).reshape(16384,25)
    return nb,d2,valid


def shared_cosine(fine_array, q16, nb, chunk):
    import torch
    import torch.nn.functional as F
    fine=torch.from_numpy(fine_array).to("cuda")
    q=F.normalize(torch.from_numpy(q16).to("cuda").float(),dim=1)
    pieces=[]
    for start in range(0,16384,chunk):
        stop=min(start+chunk,16384)
        pieces.append(torch.einsum("pc,pkc->pk",fine[start:stop],q[nb[start:stop]]))
    return torch.cat(pieces)


def read_fields(cos, coarse, params, nb, d2, valid, chunk):
    import torch
    import torch.nn.functional as F
    scalar={FINE16:torch.from_numpy(coarse["rcg"]).to("cuda").flatten(),
            FINE64:torch.from_numpy(coarse["rcg64.control"]).to("cuda").flatten(),
            MEAN_FINE:torch.from_numpy(coarse["mean.control"]).to("cuda").flatten()}
    pieces={name:[] for name in scalar}; denominators=[]
    for start in range(0,16384,chunk):
        stop=min(start+chunk,16384); ids=nb[start:stop]
        weights=torch.exp(-d2[start:stop]/(2*params["sigma"]**2))*torch.exp((cos[start:stop]-1)/params["tau"])*valid[start:stop]
        denominator=weights.sum(1).clamp_min(1e-12);denominators.append(denominator)
        for name,field in scalar.items():
            pieces[name].append((weights*field[ids]).sum(1)/denominator)
    gpu={name:torch.cat(parts).reshape(128,128) for name,parts in pieces.items()}
    up=lambda field:F.interpolate(torch.from_numpy(field).to("cuda")[None,None],(128,128),mode="bilinear",align_corners=False)[0,0]
    # Preserve original graft parentheses and original CUDA FP32 renderer.
    gpu[GRAFT]=up(coarse["mean.control"])+(gpu[FINE16]-up(coarse["rcg"]))
    fields={name:field.cpu().numpy() for name,field in gpu.items()}
    masks={name:np.packbits((F.interpolate(field[None,None],(1024,1024),mode="bilinear",align_corners=False)[0,0]>.5).cpu().numpy()) for name,field in gpu.items()}
    masks[PRIMARY]=(masks[FINE16]&masks[GRAFT])|(~masks[FINE16]&masks[FINE64])
    return fields,masks,torch.cat(denominators).cpu().numpy()


def exact_smoke(a,row,q16,r16,cov,score,fields,masks,seal,debiased):
    import torch
    key=row["key"]; failures={}
    if sha(row["feature_export"]) != seal["inputs"][key]["feature_sha256"]:
        raise ValueError("Old golden q/r changed after preflight")
    if sha(row["packet_export"]) != seal["inputs"][key]["packet_sha256"]:
        raise ValueError("Old golden cov/score/native packet changed after preflight")
    for folder in ("fields", "predictions"):
        if sha(a.base1200/folder/(key+".npz")) != seal[folder][key]:
            raise ValueError("Old golden field/mask changed after preflight")
    cached=torch.load(row["feature_export"],map_location="cpu",weights_only=True)
    for role,actual in (("q",q16),("r",r16)):
        failures[role]=int(np.count_nonzero(actual!=cached[role].numpy()))
    with np.load(row["packet_export"],allow_pickle=False) as z:
        failures.update(cov=int(np.count_nonzero(cov!=z["cov"])),score=int(np.count_nonzero(score!=z["score"])),
                        native=int(np.unpackbits(masks["native"]^z["native"]).sum()))
    with np.load(a.base1200/"fields"/(key+".npz"),allow_pickle=False) as z:
        for name in (FINE16,FINE64):failures[name+"_field"]=int(np.count_nonzero(fields[name]!=z[name]))
    with np.load(a.base1200/"predictions"/(key+".npz"),allow_pickle=False) as z:
        for name in ARMS[:6]:failures[name+"_mask"]=int(np.unpackbits(masks[name]^z[name]).sum())
    failures["debiased_gate"] = int(bool(cached["debiased"]) != debiased)
    if any(failures.values()):raise RuntimeError("Old golden exact operations failed: "+key+" "+json.dumps(failures))
    return dict(key=key,fold=row["fold"],mismatches=failures,passed=True,query_truth_opened=False,
                purpose="operational numerical parity only, no efficacy selection")


def infer(a,smoke=False):
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows,man,freeze=cohort(a)
    golden_seal=None
    if smoke:rows,golden_seal=old_golden(a)
    if shutil.disk_usage(a.out.parent).free<storage()["minimum_free_bytes"]:
        raise OSError("Need5GiB free; retain existing assets and all reusable cosine/rowden")
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ("fields","predictions","kernels"):(a.out/name).mkdir()
    write(a.out/"manifest.json",rows)
    files=[Path(__file__).resolve(),REPO/"scripts/run_frozen_subtoken4000.py",REPO/"scripts/run_frozen_subtoken1200.py",REPO/"src/ics/experiment.py"]
    code={str(p):sha(p) for p in files}
    code.update({str(REPO/p):h for p,h in freeze["source_code_sha256"].items()})
    code.update(freeze["source_producer_bindings"])
    config=dict(n=len(rows),arms=list(ARMS),primary=PRIMARY,parameters=FROZEN_PARAMS,manifest_sha256=MANIFEST_SHA,
        freeze_path=str(a.freeze),freeze_sha256=FREEZE_SHA,source_code_sha256=code,
        annotation_root=man["annotation_root"],host_manifest_sha256=sha(a.host_manifest),host_builder_sha256=HOST_SHA,
        exposure="old operational smoke only" if smoke else "independent photo-disjoint CONFIRM1200;74 observed classes; not public80class leaderboard",
        fixed_recipe=freeze["fixed_recipe"],query_truth_opened=False,cohort_recipe_parameter_updates=False,
        renderer="coarse CPU64->1024; fine/graft CUDA128->1024 bilinear align_corners=False >.5",
        producer="one actual full FoRIS paired FP32 forward including native CRF + FP32->FP16 q/r; four query-only shifts",
        cache="raw FP32 cos[16384,25] + original rowden[16384]; shared neighbors/d2/valid; no q/r archive",
        resource=dict(workers=a.workers,readers=a.readers,writers=a.writers,prefetch=a.prefetch,inflight=a.inflight,chunk=a.chunk),storage=storage(len(rows)))
    write(a.out/"config.json",config)
    nb,d2,valid=neighbor_geometry()
    np.save(a.out/"neighbors.npy",nb.cpu().numpy().astype(np.uint16));np.save(a.out/"distance2.npy",d2.cpu().numpy());np.save(a.out/"valid_neighbors.npy",valid.cpu().numpy())
    def save(row,fields,masks,cos,den,receipt):
        key=row["key"];fp=a.out/"fields"/(key+".npz");pp=a.out/"predictions"/(key+".npz");kp=a.out/"kernels"/(key+".npz")
        np.savez_compressed(fp,**fields);np.savez_compressed(pp,**masks);np.savez(kp,cos=cos,denominator=den)
        return dict(key=key,fields_sha256=sha(fp),predictions_sha256=sha(pp),kernels_sha256=sha(kp),inputs=receipt)
    start=time.monotonic();records=[];pending=[];writes=[];parities=[];gpu_seconds=0.
    with torch.inference_mode(),ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn")) as pool,ThreadPoolExecutor(a.readers) as readers,ThreadPoolExecutor(a.writers) as writers:
        host,producer,host_man=make_host(a)
        if any(host_man.get(k)!=man.get(k) for k in ("data_root","annotation_root","foris_root","projection_basis")):
            raise ValueError("Complete source host/data/basis contract differs")
        loads={i:readers.submit(load_rgb,rows[i],man) for i in range(min(a.prefetch,len(rows)))}
        def finish(item):
            row,q16,r16,cov,score,native,cos,future,receipt=item
            coarse,masks,audit=future.result()
            torch.cuda.synchronize();tick=time.monotonic()
            fine,m,den=read_fields(cos,coarse,FROZEN_PARAMS[str(row["fold"])],nb,d2,valid,a.chunk)
            torch.cuda.synchronize()
            fields=dict(coarse,**fine,**{"host.score":score,"reference.cov":cov});masks.update(m);masks["native"]=native
            receipt.update(solver=audit,fine_seconds=time.monotonic()-tick)
            if smoke:parities.append(exact_smoke(a,row,q16,r16,cov,score,fields,masks,golden_seal,receipt["debiased"]))
            return fields,masks,cos.cpu().numpy(),den,receipt
        for i,row in enumerate(rows):
            inputs=loads.pop(i).result();later=i+a.prefetch
            if later<len(rows):loads[later]=readers.submit(load_rgb,rows[later],man)
            torch.cuda.synchronize();tick=time.monotonic()
            q16,r16,cov,score,native,target,receipt=full_pair(host,producer,inputs)
            future=pool.submit(solve_coarse,(q16,r16,cov,score,str(a.mean_source)))
            fine=fine_grid(host,target,receipt["debiased"])
            cos=shared_cosine(fine,q16,nb,a.chunk)
            torch.cuda.synchronize();seconds=time.monotonic()-tick;gpu_seconds+=seconds
            receipt.update(producer_fourshift_seconds=seconds,paired_forward_calls=1,shifted_forward_calls=4)
            pending.append((row,q16,r16,cov,score,native,cos,future,receipt))
            del fine,target,inputs
            if len(pending)>=a.inflight:
                item=pending.pop(0);writes.append(writers.submit(save,item[0],*finish(item)))
            if len(writes)>=2*a.writers:records.append(writes.pop(0).result())
            if (i+1)%10==0 or i==0:
                state=dict(state="RUNNING_OLD_GOLDEN_SMOKE" if smoke else "RUNNING_CONFIRM1200_INFERENCE",n=len(rows),completed=i+1,
                           written=len(records),seconds=time.monotonic()-start,query_truth_opened=False,cuda_peak_bytes=torch.cuda.max_memory_allocated())
                write(a.out/"state.json",state);print(json.dumps(state),flush=True)
        for item in pending:writes.append(writers.submit(save,item[0],*finish(item)))
        records.extend(job.result() for job in writes)
    verify_code(config)
    seal=dict(state="SMOKE_PREDICTIONS_SEALED" if smoke else "ALL_PREDICTIONS_SEALED",n=len(rows),arms=list(ARMS),primary=PRIMARY,
        manifest_sha256=sha(a.out/"manifest.json"),config_sha256=sha(a.out/"config.json"),query_truth_opened=False,
        fields={r["key"]:r["fields_sha256"] for r in records},predictions={r["key"]:r["predictions_sha256"] for r in records},
        kernels={r["key"]:r["kernels_sha256"] for r in records},inputs={r["key"]:r["inputs"] for r in records},
        geometry={p:sha(a.out/p) for p in ("neighbors.npy","distance2.npy","valid_neighbors.npy")},
        seconds=time.monotonic()-start,gpu_producer_fourshift_seconds=gpu_seconds,cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        full_paired_forward_calls=len(rows),shifted_forward_calls=4*len(rows),q_r_archives_created=0,fine_feature_archives_created=0,
        retained_bytes={d:sum(p.stat().st_size for p in (a.out/d).iterdir()) for d in ("fields","predictions","kernels")})
    if len(records)!=len(rows) or len(seal["predictions"])!=len(rows):raise RuntimeError("Incomplete output set; no seal")
    if smoke:write(a.out/"parity.json",dict(state="OLD3_OPERATION_PARITY_EXACT",n=3,episodes=parities,efficacy_claim=False))
    write(a.out/"sealed.json",seal)
    write(a.out/"state.json",dict(state="SMOKE_COMPLETE" if smoke else "ALL_PREDICTIONS_SEALED",n=len(rows),completed=len(rows),query_truth_opened=False))
    print(seal["state"],flush=True)


def score_one(job):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1)
    row,out,annotation_root=job
    with Image.open(Path(annotation_root)/Path(row["query"]).with_suffix(".png")) as im:
        original=torch.from_numpy((np.asarray(im)==row["c"]+1).copy())
    truth=F.interpolate(original[None,None].float(),(1024,1024),mode="nearest")[0,0].numpy().astype(bool)
    with np.load(Path(out)/"predictions"/(row["key"]+".npz"),allow_pickle=False) as z:
        masks={name:unpack(z[name]) for name in ARMS}
    details=dict(row,iu={},edits_vs_native={})
    for name,mask in masks.items():
        details["iu"][name]=[int((mask&truth).sum()),int((mask|truth).sum())]
        add,delete=mask&~masks["native"],masks["native"]&~mask
        details["edits_vs_native"][name]=dict(key=row["key"],c=row["c"],fold=row["fold"],batch=row["batch"],
            add_TP=int((add&truth).sum()),add_FP=int((add&~truth).sum()),delete_TP=int((delete&truth).sum()),delete_FP=int((delete&~truth).sum()))
        edits=details["edits_vs_native"][name]
        native_iu=details["iu"]["native"]
        if (details["iu"][name][0]-native_iu[0] != edits["add_TP"]-edits["delete_TP"]
                or details["iu"][name][1]-native_iu[1] != edits["add_FP"]-edits["delete_FP"]):
            raise AssertionError("Native I/U edit closure failed")
    return details


def score(a):
    rows,man,freeze=cohort(a)
    seal=checked_json(a.out/"sealed.json");config=checked_json(a.out/"config.json")
    if seal["state"]!="ALL_PREDICTIONS_SEALED" or seal["n"]!=1200 or seal["arms"]!=list(ARMS):
        raise ValueError("ALL1200 complete arms must seal before ANY query truth")
    if sha(a.out/"manifest.json")!=seal["manifest_sha256"] or sha(a.out/"config.json")!=seal["config_sha256"]:
        raise ValueError("Sealed metadata changed")
    if checked_json(a.out/"manifest.json")!=rows:raise ValueError("Frozen cohort changed")
    verify_code(config)
    # Every output/geometry and RGB/reference source is verified before first GT read.
    for name,digest in seal["geometry"].items():
        if sha(a.out/name)!=digest:raise ValueError("Geometry changed")
    for row in rows:
        key=row["key"]
        for folder in ("fields","predictions","kernels"):
            if sha(a.out/folder/(key+".npz"))!=seal[folder][key]:raise ValueError("Sealed output changed: "+key)
        for role in ("support","query"):
            if sha(Path(man["data_root"])/row[role])!=seal["inputs"][key][role+"_image_sha256"]:raise ValueError("RGB source changed")
        reference=Path(man["annotation_root"])/Path(row["support"]).with_suffix(".png")
        if sha(reference)!=seal["inputs"][key]["reference_mask_sha256"]:raise ValueError("Reference source changed")
    started=time.monotonic()
    write(a.out/"score_state.json",dict(state="RUNNING_CPU_GT_SCORE",n=1200))
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn")) as pool:
        details=list(pool.map(score_one,[(r,str(a.out),man["annotation_root"]) for r in rows],chunksize=4))
    arrays={name:np.asarray([d["iu"][name] for d in details],np.int64) for name in ARMS}
    corrections={name:[d["edits_vs_native"][name] for d in details] for name in ARMS}
    report,draws=summarize(rows,arrays,corrections)
    # Include every strong control, even names without the legacy .control suffix.
    groups=photo_groups(rows);g=int(groups.max())+1;classes=np.asarray([r["c"] for r in rows])
    weights=np.stack([np.bincount(d,minlength=g) for d in draws])[:,groups]
    samples={name:np.asarray([metric(v,classes,w) for w in weights]) for name,v in arrays.items()}
    for name in ARMS:
        for base in ARMS:
            if name==base:continue
            delta=arrays[name][:,0]/np.maximum(arrays[name][:,1],1)-arrays[base][:,0]/np.maximum(arrays[base][:,1],1)
            report["contrasts"][name][base]=dict(gain=report["scores"][name]-report["scores"][base],
                ci95=np.percentile(samples[name]-samples[base],[2.5,97.5]).tolist(),
                up=int((delta>1e-12).sum()),down=int((delta < -1e-12).sum()),tie=int((np.abs(delta)<=1e-12).sum()))
    report.update(primary=PRIMARY,config=config,exposure=config["exposure"],freeze_sha256=FREEZE_SHA,manifest_sha256=MANIFEST_SHA,
        classes=74,absent_classes=[32,35,68,70,78,79],official_public80class_leaderboard=False,
        source_prediction_seal_sha256=sha(a.out/"sealed.json"),parameter_updates=False,recipe_updates=False,
        query_truth_first_opened_after_all1200_complete_masks_sealed=True,INSID3="complete1200 unavailable; not claimed",
        runtime=dict(inference_seconds=seal["seconds"],cpu_score_seconds=time.monotonic()-started),retained_bytes=seal["retained_bytes"])
    for label in ("folds","batchs"):
        for row in report[label].values():
            row["primary_gain_vs_controls"]={name:row["scores"][PRIMARY]-row["scores"][name] for name in ARMS[:-1]}
    np.savez_compressed(a.out/"counts.npz",**{"iu:"+name:v for name,v in arrays.items()});np.save(a.out/"bootstrap_photo_draws.npy",draws)
    (a.out/"episodes.jsonl").write_text("".join(json.dumps(d)+"\n" for d in details));write(a.out/"report.json",report)
    lines=["# Frozen photo-disjoint CONFIRM1200","","74 observed classes; missing32,35,68,70,78,79; not the public80class leaderboard.","","| arm | class-summed mIoU1024 |","|---|---:|"]
    lines += ["| %s | %.6f |"%(name,report["scores"][name]) for name in ARMS]
    lines += ["","| fixed primary versus | paired gain | connected-photo95% CI |","|---|---:|---|"]
    for name in ARMS[:-1]:
        c=report["contrasts"][PRIMARY][name];lines.append("| %s | %+.6f | [%+.6f,%+.6f] |"%(name,c["gain"],*c["ci95"]))
    (a.out/"report.md").write_text("\n".join(lines)+"\n")
    write(a.out/"score_state.json",dict(state="CPU_SCORE_COMPLETE",n=1200,primary=PRIMARY))
    print(json.dumps(report["scores"]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("preflight","wait-inputs","smoke","infer","score"));p.add_argument("--out",type=Path,required=True)
    for name,default in (("manifest","outputs/photo_disjoint_confirm1200_v1/manifest.json"),("freeze","outputs/strict_family_confirm1200_v1_prepared/freeze.json"),
        ("base1200","outputs/frozen_subtoken1200_v1"),("base4000","outputs/frozen_subtoken4000_v1"),("host-manifest","outputs/claude_official/batch0.json"),
        ("predecessor","outputs/direct_mean_fine4000_v1"),("predecessor-pipeline","launch/direct_mean_fine4000_v1/state.json"),
        ("mean-source","external/astra_emd/external_mean_delete600_modules")):
        p.add_argument("--"+name,type=Path,default=ROOT/default)
    p.add_argument("--host-root",type=Path,default=Path("/root/autodl-tmp/demo9_extent"));p.add_argument("--demo4-root",default="/root/autodl-tmp/demo4")
    p.add_argument("--basis",type=Path,default=Path("/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt"))
    for name,value in (("workers",6),("readers",6),("writers",2),("prefetch",12),("inflight",6),("chunk",512)):
        p.add_argument("--"+name,type=int,default=value)
    a=p.parse_args()
    if any(getattr(a,k)<1 for k in ("workers","readers","writers","prefetch","inflight","chunk")):p.error("Positive execution sizes required")
    try:
        if a.stage=="preflight":preflight(a)
        elif a.stage=="wait-inputs":wait_inputs(a)
        elif a.stage=="score":score(a)
        else:infer(a,a.stage=="smoke")
    except Exception as error:
        if a.stage in ("smoke","infer") and a.out.exists():
            write(a.out/"state.json",dict(state="FAILED",stage=a.stage,error=repr(error),query_truth_opened=False))
        raise


if __name__=="__main__":main()
