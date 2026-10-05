#!/usr/bin/env python3
"""FoRIS-independent complete raw-DINO graph readout; seal before CPU scoring.

Fixed DEV construction, not a novelty or SOTA claim. Reference/query joint
centering and query-neighborhood agreement are explicit, removable priors.
CPU readers and writers overlap batched CUDA work. No encoder is rerun.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from ics.experiment import load_rows, packet, render, sha, summarize, unpack

CONFIG = dict(reference_labels="coverage>=0.5", reference_neighbors_per_role=16,
              temperature=.07, query_neighbors=32, restart=.5, iterations=40,
              centering="subtract joint mean of unit query/reference tokens, renormalize",
              renderer="bilinear1024 align_corners=False then >0.5",
              origin="model.raw_nn", query_gt_in_inference=False,
              FoRIS_inputs_in_candidate=False, extra_encoder_forwards=0,
              exposure="exposed DEV241; fixed construction, not confirmation")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.inference_mode()
def predict(q, r, cov):
    q, r = F.normalize(q, dim=-1), F.normalize(r, dim=-1)
    fg = cov.flatten(1) >= .5
    if not bool(fg.any(1).all()) or bool(fg.all(1).any()):
        raise ValueError("Reference must contain both roles")
    output = {}
    for centered in (False, True):
        if centered:
            center = (q.mean(1, keepdim=True) + r.mean(1, keepdim=True)) / 2
            x, y = F.normalize(q-center, dim=-1), F.normalize(r-center, dim=-1)
        else:
            x, y = q, r
        sim = torch.bmm(x, y.transpose(1, 2))
        nn = torch.gather(fg, 1, sim.argmax(-1)).float()
        unary = []
        for b in range(len(q)):
            evidence = []
            for role in (fg[b], ~fg[b]):
                k = min(CONFIG["reference_neighbors_per_role"], int(role.sum()))
                values = sim[b, :, role].topk(k, dim=-1).values / CONFIG["temperature"]
                evidence.append(torch.logsumexp(values, -1) - math.log(k))
            unary.append(torch.sigmoid(evidence[0]-evidence[1]))
        unary = torch.stack(unary)
        del sim
        graph = torch.bmm(x, x.transpose(1, 2))
        ids = torch.arange(graph.shape[-1], device=graph.device)
        graph[:, ids, ids] = -2
        values, neighbors = graph.topk(CONFIG["query_neighbors"], dim=-1)
        weights = torch.softmax((values-values[..., :1])/CONFIG["temperature"], -1)
        del graph, values
        z = unary.clone()
        batch = torch.arange(len(q), device=q.device)[:, None, None]
        for _ in range(CONFIG["iterations"]):
            z = CONFIG["restart"]*unary + (1-CONFIG["restart"])*(weights*z[batch, neighbors]).sum(-1)
        if centered:
            output.update({"raw_graph.joint": z, "raw_graph.unary.control": unary,
                           "raw_graph.centered_nn.control": nn})
        else:
            output.update({"model.raw_nn": nn, "raw_graph.no_center.control": z,
                           "raw_graph.raw_unary.control": unary})
    means = []
    for b in range(len(q)):
        margin = q[b] @ (F.normalize(r[b, fg[b]].mean(0), dim=0)
                         - F.normalize(r[b, ~fg[b]].mean(0), dim=0))
        means.append((margin > 0).float())
    output["model.raw_mean.control"] = torch.stack(means)
    return {k: v.reshape(-1, 64, 64).cpu().numpy() for k, v in output.items()}


def infer(a):
    rows = load_rows(a.layers / "manifest.json")
    source = json.loads((a.layers / "sealed.json").read_text())
    if len(rows) != 241 or sha(a.layers/"manifest.json") != source["manifest_sha256"]:
        raise ValueError("Require original sealed full DEV241")
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True); (a.out/"predictions").mkdir()
    write(a.out/"manifest.json", rows)
    write(a.out/"config.json", dict(CONFIG, source_code_sha256=sha(Path(__file__)),
        experiment_sha256=sha(REPO/"src/ics/experiment.py"), dtype="FP32", tf32=False,
        cpu_readers=a.readers, cpu_writers=a.writers, batch=a.batch,
        device=a.device, torch_threads=a.threads,
        input_seal_sha256=sha(a.layers/"sealed.json")))

    def load(row):
        lp = a.layers/"layers"/(row["key"]+".npz")
        digest = sha(lp)
        if digest != source["layers"][row["key"]]:
            raise ValueError("Changed raw feature archive")
        with np.load(lp, allow_pickle=False) as z:
            q, r = z["q24"], z["r24"]
        if q.shape != (4096,1024) or r.shape != q.shape or q.dtype != np.float32 or r.dtype != q.dtype:
            raise ValueError("Require original FP32 l24 tokens")
        pp = packet(a.root, row)
        with np.load(pp, allow_pickle=False) as z:
            cov = z["cov"].copy()
        return q, r, cov, dict(layer_sha256=digest, packet_sha256=sha(pp))

    def save(row, fields, receipt):
        destination = a.out/"predictions"/(row["key"]+".npz")
        np.savez_compressed(destination, **{k:np.packbits(render(v)) for k,v in fields.items()})
        return row["key"], sha(destination), receipt

    begin=time.monotonic(); gpu_seconds=0.; completed=[]
    with ThreadPoolExecutor(a.readers) as readers, ThreadPoolExecutor(a.writers) as writers:
        pending = {}; depth = max(3*a.batch, a.readers)
        for i in range(min(depth,len(rows))):
            pending[i] = readers.submit(load, rows[i])
        for start in range(0,len(rows),a.batch):
            ix=list(range(start,min(start+a.batch,len(rows))))
            data=[pending.pop(i).result() for i in ix]
            for i in ix:
                j=i+depth
                if j<len(rows): pending[j]=readers.submit(load, rows[j])
            tensors = [torch.from_numpy(np.stack([d[j] for d in data])) for j in range(3)]
            if a.device == "cuda":
                tensors=[v.pin_memory().to("cuda",non_blocking=True) for v in tensors]
            t=time.monotonic(); fields=predict(*tensors)
            if a.device == "cuda": torch.cuda.synchronize()
            gpu_seconds += time.monotonic()-t
            for b,i in enumerate(ix):
                completed.append(writers.submit(save, rows[i], {k:v[b] for k,v in fields.items()}, data[b][3]))
            if start==0 or (start//a.batch)%5==0:
                print(json.dumps(dict(inferred=ix[-1]+1,total=len(rows),elapsed=time.monotonic()-begin,
                    compute_seconds=gpu_seconds,device=a.device)),flush=True)
        results=[f.result() for f in completed]
    write(a.out/"sealed.json",dict(state="ALL_PREDICTIONS_SEALED",n=len(rows),
        predictions={k:h for k,h,_ in results}, inputs={k:v for k,_,v in results},
        manifest_sha256=sha(a.out/"manifest.json"),config_sha256=sha(a.out/"config.json"),
        query_labels_opened=False, seconds=time.monotonic()-begin,compute_seconds=gpu_seconds,
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if a.device=="cuda" else 0))


def score(a):
    seal=json.loads((a.out/"sealed.json").read_text())
    if seal["state"]!="ALL_PREDICTIONS_SEALED": raise ValueError("Predictions not sealed")
    rows=load_rows(a.out/"manifest.json"); arrays={};corrections={};details=[]
    config=json.loads((a.out/"config.json").read_text())
    if sha(a.out/"config.json")!=seal["config_sha256"] or sha(a.out/"manifest.json")!=seal["manifest_sha256"]:
        raise ValueError("Changed sealed configuration")
    if sha(Path(__file__))!=config["source_code_sha256"] or sha(REPO/"src/ics/experiment.py")!=config["experiment_sha256"]:
        raise ValueError("Changed implementation")
    comparisons=json.loads((a.comparisons/"sealed.json").read_text())
    for row in rows:
        key=row["key"]; pp=packet(a.root,row); pred=a.out/"predictions"/(key+".npz")
        cp=a.comparisons/"predictions"/(key+".npz")
        if sha(pred)!=seal["predictions"][key] or sha(pp)!=seal["inputs"][key]["packet_sha256"]:
            raise ValueError("Changed sealed prediction/evaluation packet")
        if sha(cp)!=comparisons["predictions"][key]: raise ValueError("Changed comparison")
        with np.load(pp,allow_pickle=False) as z: truth,native=unpack(z["truth"]),unpack(z["native"])
        with np.load(pred,allow_pickle=False) as z: masks={k:unpack(z[k]) for k in z.files}
        with np.load(cp,allow_pickle=False) as z:
            for name in ("rcg","mean.control"):
                masks[name]=unpack(z[name])
        masks["native"]=native; origin=masks["model.raw_nn"]
        for arm,mask in masks.items():
            add,delete=mask&~origin,origin&~mask
            rec=dict(key=key,c=row["c"],fold=row["fold"],batch=str(row.get("batch","unspecified")),
                     add_TP=int((add&truth).sum()),add_FP=int((add&~truth).sum()),
                     delete_TP=int((delete&truth).sum()),delete_FP=int((delete&~truth).sum()))
            iu=[int((mask&truth).sum()),int((mask|truth).sum())]
            arrays.setdefault(arm,[]).append(iu);corrections.setdefault(arm,[]).append(rec)
            details.append(dict(rec,arm=arm,intersection=iu[0],union=iu[1]))
    report,draws=summarize(rows,{k:np.array(v) for k,v in arrays.items()},corrections)
    report["corrections_vs_origin"]=report.pop("corrections_vs_native")
    report.update(config=config,edit_origin="model.raw_nn",prediction_seal_sha256=sha(a.out/"sealed.json"))
    write(a.out/"report.json",report);write(a.out/"episode_metrics.json",details)
    np.save(a.out/"bootstrap_photo_draws.npy",draws)
    print(json.dumps(report["scores"]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("infer","score"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--layers",type=Path);p.add_argument("--comparisons",type=Path)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--batch",type=int,default=4)
    p.add_argument("--readers",type=int,default=6);p.add_argument("--writers",type=int,default=2)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--threads",type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if a.stage=="infer": infer(a)
    else: score(a)


if __name__=="__main__": main()
