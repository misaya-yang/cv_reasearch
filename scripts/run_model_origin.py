#!/usr/bin/env python3
"""Seal direct raw-DINO readouts from existing l24 archives, then score separately.

No encoder, new images, positional projection or CRF is loaded. CPU is an explicit
execution option, not permission to run in the restricted no-card environment.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import torch
from ics.experiment import load_rows, packet, render, sha, unpack, summarize
from ics.methods import raw_reference_origin as method


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def infer(a):
    rows = load_rows(a.manifest)
    if len(rows) != a.expected: raise ValueError("Unexpected manifest size; no silent subset")
    if a.limit: rows = rows[:a.limit]
    if not rows or a.out.exists(): raise ValueError("Require nonempty rows and a new output directory")
    if a.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("No visible GPU")
    src = json.loads((a.layers_run / "sealed.json").read_text())
    if sha(a.layers_run / "manifest.json") != src["manifest_sha256"]: raise ValueError("Layer manifest changed")
    if sha(a.layers_run / "protocol.json") != src["protocol_sha256"]: raise ValueError("Layer protocol changed")
    original = {r["key"]: r for r in load_rows(a.layers_run / "manifest.json")}
    for row in rows:
        if row["key"] not in original or any(original[row["key"]][k] != row[k] for k in ("support", "query", "fold", "e", "c")):
            raise ValueError("Raw layer/photo identity mismatch: " + row["key"])
        path = a.layers_run / "layers" / f"{row['key']}.npz"
        if not path.exists() or row["key"] not in src.get("layers", {}): raise ValueError("Missing sealed raw layers: " + row["key"])
    a.out.mkdir(parents=True); (a.out / "predictions").mkdir(); (a.out / "fields").mkdir()
    dump(a.out / "manifest.json", rows)
    sources = [Path(__file__), Path(method.__file__), REPO / "src/ics/experiment.py"]
    config = dict(method=method.CONFIG, source_sha256={str(p.resolve()): sha(p) for p in sources},
                  source_layer_seal_sha256=sha(a.layers_run / "sealed.json"), layer_keys=["q24", "r24"],
                  device=a.device, dtype="float32", tf32=False, query_gt_in_inference=False,
                  origin_arm="model.raw_nn", native_arm="complete cached FoRIS comparison only",
                  scope="execution smoke only" if a.limit else "exposed DEV241, not independent confirmation")
    dump(a.out / "config.json", config)
    seal = dict(state="INFERRING", predictions={}, fields={}, inputs={}, manifest_sha256=sha(a.out / "manifest.json"),
                config_sha256=sha(a.out / "config.json"), origin_arm="model.raw_nn", query_labels_opened=False)
    audits = {}
    for row in rows:
        key = row["key"]; lp = a.layers_run / "layers" / f"{key}.npz"; pp = packet(a.root, row)
        lh = sha(lp)
        if lh != src["layers"][key]: raise ValueError("Raw layer archive changed: " + key)
        with np.load(lp, allow_pickle=False) as z:
            q, r = z["q24"], z["r24"]
        if q.shape != (4096, 1024) or r.shape != q.shape or q.dtype != np.float32 or r.dtype != np.float32:
            raise ValueError("Expected the original float32 raw l24 archives")
        with np.load(pp, allow_pickle=False) as z:
            cov, native, foris_score = z["cov"].copy(), z["native"].copy(), z["score"].copy()
        unpack(native)  # comparison prediction only; never access query truth in this phase
        masks, fields, audits[key] = method.predict(q, r, cov, device=a.device)
        prediction = {k: np.packbits(render(v.astype(np.float32))) for k, v in masks.items()}
        prediction["native"] = native; fields["foris.score"] = np.asarray(foris_score, np.float32)
        pred_path, field_path = a.out / "predictions" / f"{key}.npz", a.out / "fields" / f"{key}.npz"
        np.savez_compressed(pred_path, **prediction); np.savez_compressed(field_path, **fields)
        seal["predictions"][key], seal["fields"][key] = sha(pred_path), sha(field_path)
        seal["inputs"][key] = dict(layer_sha256=lh, packet_sha256=sha(pp))
        print(json.dumps(dict(key=key, complete=len(seal["predictions"]), episodes=len(rows))), flush=True)
    dump(a.out / "audits.json", audits)
    seal.update(state="ALL_PREDICTIONS_SEALED", n=len(rows), audits_sha256=sha(a.out / "audits.json"))
    dump(a.out / "sealed.json", seal)


def score(a):
    seal = json.loads((a.out / "sealed.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED": raise ValueError("Origin cohort is not sealed")
    for file, key in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256")):
        if sha(a.out / file) != seal[key]: raise ValueError(file + " changed after sealing")
    config = json.loads((a.out / "config.json").read_text())
    for path, digest in config["source_sha256"].items():
        if sha(Path(path)) != digest: raise ValueError("Source changed after sealing: " + path)
    if (a.out / "report.json").exists(): raise ValueError("Refuse to overwrite scored output")
    rows = load_rows(a.out / "manifest.json"); arrays, corrections, episodes = {}, {}, []
    for row in rows:
        key = row["key"]; pred_path = a.out / "predictions" / f"{key}.npz"; pp = packet(a.root, row)
        if sha(pred_path) != seal["predictions"][key] or sha(pp) != seal["inputs"][key]["packet_sha256"]:
            raise ValueError("Prediction or evaluation packet changed: " + key)
        with np.load(pp, allow_pickle=False) as p: truth = unpack(p["truth"])
        with np.load(pred_path, allow_pickle=False) as z: masks = {k: unpack(z[k]) for k in z.files}
        origin = masks["model.raw_nn"]
        for arm, mask in masks.items():
            iu = [int((mask & truth).sum()), int((mask | truth).sum())]
            add, delete = mask & ~origin, origin & ~mask
            rec = dict(key=key, c=row["c"], fold=row["fold"], batch=str(row.get("batch", "unspecified")),
                       add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                       delete_TP=int((delete & truth).sum()), delete_FP=int((delete & ~truth).sum()))
            arrays.setdefault(arm, []).append(iu); corrections.setdefault(arm, []).append(rec)
            episodes.append(dict(rec, arm=arm, intersection=iu[0], union=iu[1]))
    report, draws = summarize(rows, {k: np.asarray(v) for k, v in arrays.items()}, corrections)
    report["corrections_vs_origin"] = report.pop("corrections_vs_native")
    report.update(edit_origin_arm="model.raw_nn", native_arm="complete cached FoRIS", config=config)
    dump(a.out / "episode_metrics.json", episodes); dump(a.out / "report.json", report)
    np.save(a.out / "bootstrap_photo_draws.npy", draws)
    print(json.dumps(report["scores"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("infer", "score", "all")); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--layers-run", type=Path); p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--expected", type=int, default=241)
    p.add_argument("--limit", type=int); p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    if a.stage != "score" and (a.layers_run is None or a.manifest is None): p.error("infer requires --layers-run and --manifest")
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if a.stage in ("infer", "all"): infer(a)
    if a.stage in ("score", "all"): score(a)


if __name__ == "__main__": main()
