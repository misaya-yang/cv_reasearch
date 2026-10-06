#!/usr/bin/env python3
"""One frozen boundary mechanism, complete masks, then isolated scoring."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ics.experiment import evaluate, load_rows, packet, render, sha, unpack
from ics.methods.reference_boundary import CONFIG, predict, selfcheck


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def one(job):
    row, root, out, controls = job
    import torch
    torch.set_num_threads(1)
    begin = time.monotonic()
    fp = Path(root) / "cache/evidence_v1/feat" / (row["key"] + ".pt")
    feature = torch.load(fp, map_location="cpu", weights_only=True)
    q, r = feature["q"].float().numpy(), feature["r"].float().numpy()
    pp = packet(root, row)
    with np.load(pp, allow_pickle=False) as z:
        coverage, contrast = z["cov"].copy(), z["s2"].copy()
    masks, info, edge_fields = predict(q, r, coverage, contrast)
    result = {k: np.packbits(render(v.astype(np.float32))) for k, v in masks.items()}
    with np.load(Path(controls) / "predictions" / (row["key"] + ".npz"), allow_pickle=False) as z:
        for k, source in (("rcg", "RCG"), ("mean.control", "MEAN_CONTROL")):
            result[k] = z[source].copy()
    np.savez_compressed(Path(out) / "predictions" / (row["key"] + ".npz"), **result)
    np.savez_compressed(Path(out) / "edge_fields" / (row["key"] + ".npz"), **edge_fields)
    return row["key"], dict(info, seconds=time.monotonic() - begin), dict(
        features=str(fp), feature_sha256=sha(fp), packet_sha256=sha(pp))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("infer", "score", "selfcheck"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--root", type=Path)
    p.add_argument("--controls", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.phase == "selfcheck":
        print(selfcheck()); return
    if a.phase == "score":
        report = evaluate(a.root, a.out)
        print(json.dumps({"scores": report["scores"], "primary": report["contrasts"]["boundary_joint"]}, indent=2))
        return
    rows = load_rows(a.manifest)
    if a.smoke:
        rows = [next(r for r in rows if r["fold"] == f) for f in range(4)]
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "predictions").mkdir(); (a.out / "edge_fields").mkdir()
    write(a.out / "manifest.json", rows)
    write(a.out / "config.json", dict(CONFIG, workers=a.workers,
        source_sha256=sha(Path(__file__)),
        method_sha256=sha(Path(__file__).resolve().parents[1] / "src/ics/methods/reference_boundary.py"),
        exposure="reused DEV241; no independent confirmation", smoke=a.smoke))
    info, inputs = {}, {}
    jobs = [(r, str(a.root), str(a.out), str(a.controls)) for r in rows]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for n, (key, receipt, source) in enumerate(pool.imap_unordered(one, jobs), 1):
            info[key], inputs[key] = receipt, source
            if n % 20 == 0 or n == len(rows):
                print(json.dumps({"completed": n, "total": len(rows), "seconds_last": round(receipt["seconds"], 2)}), flush=True)
    write(a.out / "audit.json", info)
    write(a.out / "sealed.json", dict(state="ALL_PREDICTIONS_SEALED", query_gt_in_inference=False,
        manifest_sha256=sha(a.out / "manifest.json"), inputs=inputs,
        predictions={r["key"]: sha(a.out / "predictions" / (r["key"] + ".npz")) for r in rows}))
    print("ALL_PREDICTIONS_SEALED", flush=True)


if __name__ == "__main__":
    main()
