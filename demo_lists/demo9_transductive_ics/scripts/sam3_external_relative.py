#!/usr/bin/env python3
"""External-pack evaluation of frozen SAM3 proposal rules (visual prompt only).

GPU inference freezes all predictions for all requested packs without opening
query annotations. CPU scoring is a separate invocation and validates that
global freeze before reading any query annotation.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sam3_stitch as S

DATASETS = ("pascal_part", "paco_part", "suim")
ARMS = ("absolute_0.5_top20", "absolute_0.3_top20", "relative_0.7_top20", "fallback_top1_top20", "full_visual_0.5")


def sha(path):
    return S.sha256(Path(path))


def resolved(value, manifest, pack_root):
    p = Path(value)
    if p.is_absolute():
        return p
    candidates = (Path.cwd() / p, manifest.parent / p, manifest.parent.parent / p, pack_root.parent / p)
    return next((x.resolve() for x in candidates if x.is_dir()), (pack_root.parent / p).resolve())


def photo_hash(image):
    rgb = image.convert("RGB")
    payload = rgb.width.to_bytes(4, "big") + rgb.height.to_bytes(4, "big") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def selected(scores, areas, rule):
    valid = [i for i, n in enumerate(areas) if int(n) > 0]
    if not valid:
        return []
    if rule.startswith("absolute_"):
        t = float(rule.split("_")[1])
        return [i for i in valid if float(scores[i]) > t]
    top = max(valid, key=lambda i: float(scores[i]))
    if rule == "fallback_top1":
        keep = [i for i in valid if float(scores[i]) > .5]
        return keep or [top]
    frac = float(rule.split("_")[1])
    return [i for i in valid if float(scores[i]) >= frac * float(scores[top])]


def unpack(bits, count, height, width):
    data = np.unpackbits(bits, axis=1)[:, :height * width]
    if data.shape[0] != count:
        raise ValueError("packed candidate count mismatch")
    return data.reshape(count, height, width).astype(bool)


def infer(a):
    import torch

    pack_root, out_root = Path(a.pack_root), Path(a.out_root)
    manifests = {}
    for dataset in a.datasets:
        manifest_path = pack_root / dataset / "episodes.json"
        man = json.loads(manifest_path.read_text())
        if man.get("state") != "PREPARED" or man.get("dataset") != dataset:
            raise ValueError("external pack is not a prepared matching manifest: " + str(manifest_path))
        episode_rows = man["episodes"][:a.limit_per_dataset]
        if not episode_rows:
            raise ValueError("empty external episode pack: " + dataset)
        keys = [(int(r["fold"]), int(r["e"]), int(r["c"])) for r in episode_rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate external episode key: " + dataset)
        data_root = resolved(man["data_root"], manifest_path, pack_root)
        ann_root = resolved(man["annotation_root"], manifest_path, pack_root)
        out = out_root / dataset
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("preserve old external result; fresh output required: " + str(out))
        (out / "candidates").mkdir(parents=True, exist_ok=True)
        manifests[dataset] = dict(path=manifest_path, man=man, rows=episode_rows, data=data_root, ann=ann_root, out=out)

    status_path = Path(a.checkpoint_status)
    status = json.loads(status_path.read_text())
    if (status.get("state") != "WEIGHT_VERIFIED" or status.get("actual_sha256") != S.CHECKPOINT_SHA256 or
        status.get("actual_bytes") != S.CHECKPOINT_BYTES or a.checkpoint.stat().st_size != S.CHECKPOINT_BYTES):
        raise ValueError("official checkpoint identity does not match the verified receipt")
    if sha(a.checkpoint) != S.CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA device required; external inference has no CPU fallback")

    torch.cuda.set_per_process_memory_fraction(.3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    sys.path.insert(0, str(a.sam3))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(bpe_path=str(a.sam3 / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"), device="cuda",
                                   checkpoint_path=str(a.checkpoint), load_from_HF=False, eval_mode=True,
                                   enable_inst_interactivity=False, compile=False).float().eval()
    S.configure_fp32_mlp_backend(model)
    run = S.Runner(Sam3Processor(model, device="cuda", confidence_threshold=.5))
    global_report = dict(state="RUNNING", checkpoint_sha256=S.CHECKPOINT_SHA256,
                         source_sha256=sha(__file__), query_annotation_opened=False, datasets={})
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "external_infer_status.json").write_text(json.dumps(global_report, indent=2))
    for dataset, item in manifests.items():
        man, data_root, ann_root, out = item["man"], item["data"], item["ann"], item["out"]
        manifest_path, episode_rows = item["path"], item["rows"]
        pred_path = out / "predictions.jsonl"
        stage = dict(state="RUNNING", episodes=0, expected=len(episode_rows), manifest_sha256=sha(manifest_path),
                     query_annotation_opened=False, protocol="SAM3 visual exemplar; 1008-square stitched canvas; reference lower60/query upper40; top20 candidate rules")
        (out / "infer_report.json").write_text(json.dumps(stage, indent=2))
        start = time.monotonic()
        try:
            with pred_path.open("x") as stream, torch.inference_mode():
                for i, row in enumerate(episode_rows):
                    ref_path, query_path = data_root / row["support"], data_root / row["query"]
                    ref, query = Image.open(ref_path).convert("RGB"), Image.open(query_path).convert("RGB")
                    c = int(row["c"])
                    ref_mask = np.asarray(Image.open(ann_root / Path(row["support"]).with_suffix(".png"))) == c + 1
                    if not ref_mask.any():
                        raise ValueError("empty external support mask: " + str((dataset, row["e"], c)))
                    box = S.component_box(ref_mask, random.Random(i))
                    canvas, norm_box = S.stitch(ref, query, box)
                    union, scores, raw, kept, presence = run(canvas, norm_box, None)
                    x, y, w, h = S.rectangles()[1]
                    candidate = raw[:, y:y+h, x:x+w].cpu().numpy().astype(bool)
                    k = len(scores)
                    if candidate.shape != (k, h, w):
                        raise ValueError("candidate/query canvas shape mismatch")
                    visual = S.to_original(union.cpu().numpy().astype(np.uint8), query.size)
                    stem = "%05d_%06d_%03d.npz" % (int(row["fold"]), int(row["e"]), c)
                    asset = out / "candidates" / stem
                    np.savez_compressed(asset,
                                        proposal_query=np.packbits(candidate.reshape(k, h*w), axis=1),
                                        scores=np.asarray(scores, np.float32),
                                        visual_original=np.packbits(visual.reshape(-1)))
                    rec = dict(fold=int(row["fold"]), e=int(row["e"]), c=c, support=row["support"], query=row["query"],
                               support_rgb_sha256=photo_hash(ref), query_rgb_sha256=photo_hash(query),
                               query_shape=[query.height, query.width], proposal_shape=[k, h, w], candidate_file="candidates/"+stem,
                               candidate_bytes=asset.stat().st_size, candidate_sha256=sha(asset), scores=[float(v) for v in scores],
                               area=[int(v) for v in candidate.reshape(k, -1).sum(1)], kept=int(kept), presence=float(presence),
                               box=box, normalized_box=norm_box, query_annotation_opened=False)
                    stream.write(json.dumps(rec, allow_nan=False) + "\n"); stream.flush()
                    stage["episodes"] += 1
                    if stage["episodes"] == 10 or stage["episodes"] % 50 == 0:
                        stage["elapsed_s"] = time.monotonic() - start
                        (out / "infer_report.json").write_text(json.dumps(stage, indent=2))
                        print(f"{dataset}: {stage['episodes']}/{len(episode_rows)}; {(time.monotonic()-start)/stage['episodes']:.2f}s/episode", flush=True)
            freeze = dict(state="PREDICTIONS_FROZEN", dataset=dataset, episodes=len(episode_rows),
                          manifest_sha256=stage["manifest_sha256"], predictions_sha256=sha(pred_path),
                          source_sha256=sha(__file__), checkpoint_sha256=S.CHECKPOINT_SHA256,
                          query_annotation_opened=False,
                          assets=[dict(path="candidates/"+p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted((out/"candidates").glob("*.npz"))])
            (out / "prediction_freeze.json").write_text(json.dumps(freeze, indent=2))
            stage.update(state="PREDICTIONS_FROZEN", elapsed_s=time.monotonic()-start, freeze_sha256=sha(out/"prediction_freeze.json"))
            (out / "infer_report.json").write_text(json.dumps(stage, indent=2))
            global_report["datasets"][dataset] = dict(state="PREDICTIONS_FROZEN", episodes=len(episode_rows),
                manifest_sha256=freeze["manifest_sha256"], prediction_freeze_sha256=sha(out/"prediction_freeze.json"))
            (out_root / "external_infer_status.json").write_text(json.dumps(global_report, indent=2))
        except BaseException as error:
            stage.update(state="ERROR", error=repr(error), elapsed_s=time.monotonic()-start)
            (out / "infer_report.json").write_text(json.dumps(stage, indent=2))
            global_report["state"] = "ERROR"
            global_report["datasets"][dataset] = dict(state="ERROR", error=repr(error))
            (out_root / "external_infer_status.json").write_text(json.dumps(global_report, indent=2))
            raise
    global_report["state"] = "PREDICTIONS_FROZEN"
    global_report["query_annotation_opened"] = False
    (out_root / "external_infer_status.json").write_text(json.dumps(global_report, indent=2))
    print(json.dumps(global_report, indent=2))


def score(a):
    out_root, pack_root = Path(a.out_root), Path(a.pack_root)
    freeze_all = json.loads((out_root / "external_infer_status.json").read_text())
    if freeze_all.get("state") != "PREDICTIONS_FROZEN" or freeze_all.get("query_annotation_opened") is not False:
        raise ValueError("all external packs must be frozen before any query annotation is opened")
    if freeze_all.get("source_sha256") != sha(__file__) or freeze_all.get("checkpoint_sha256") != S.CHECKPOINT_SHA256:
        raise ValueError("global source/checkpoint identity mismatch")
    if set(freeze_all["datasets"]) != set(a.datasets):
        raise ValueError("score dataset set differs from the frozen inference set")
    # Validate every manifest, prediction ledger, candidate file and per-pack
    # freeze receipt for the complete external cohort before opening any query GT.
    prepared = {}
    for dataset in a.datasets:
        manifest_path = pack_root / dataset / "episodes.json"
        man = json.loads(manifest_path.read_text())
        if man.get("state") != "PREPARED" or man.get("dataset") != dataset:
            raise ValueError("external manifest state mismatch: " + dataset)
        data_root = resolved(man["data_root"], manifest_path, pack_root)
        ann_root = resolved(man["annotation_root"], manifest_path, pack_root)
        out = out_root / dataset
        freeze_path, pred_path = out / "prediction_freeze.json", out / "predictions.jsonl"
        freeze = json.loads(freeze_path.read_text())
        report = json.loads((out / "infer_report.json").read_text())
        if (freeze.get("state") != "PREDICTIONS_FROZEN" or freeze.get("query_annotation_opened") is not False or
            freeze.get("manifest_sha256") != sha(manifest_path) or freeze.get("predictions_sha256") != sha(pred_path) or
            freeze.get("source_sha256") != sha(__file__) or freeze.get("checkpoint_sha256") != S.CHECKPOINT_SHA256 or
            report.get("state") != "PREDICTIONS_FROZEN"):
            raise ValueError("prediction freeze identity/state failed for " + dataset)
        receipt = freeze_all["datasets"].get(dataset, {})
        if receipt.get("state") != "PREDICTIONS_FROZEN" or receipt.get("prediction_freeze_sha256") != sha(freeze_path):
            raise ValueError("global freeze does not bind the per-pack receipt: " + dataset)
        for asset in freeze["assets"]:
            p = out / asset["path"]
            if p.stat().st_size != asset["bytes"] or sha(p) != asset["sha256"]:
                raise ValueError("candidate asset failed freeze check: " + str(p))
        preds = rows(pred_path)
        expected = [(int(r["fold"]), int(r["e"]), int(r["c"])) for r in man["episodes"]]
        observed = [(int(r["fold"]), int(r["e"]), int(r["c"])) for r in preds]
        if observed != expected or len(preds) != freeze["episodes"] or len(preds) != len(man["episodes"]):
            raise ValueError("prediction rows do not exactly cover external manifest for " + dataset)
        prepared[dataset] = (man, data_root, ann_root, out, preds)
    results = {"state": "COMPLETED", "protocol": "COCO-adapted SAM3 visual-exemplar canvas; external episode packs; class mIoU at original query resolution",
               "bootstrap": "2,000 paired draws, seed 0, connected groups of exact decoded RGB support/query hashes; transformed duplicate photos may not be joined",
               "labels_exposure": "query annotations opened only after all requested pack manifests, prediction ledgers, candidate assets and freeze receipts passed validation",
               "frozen_source_sha256": freeze_all["source_sha256"], "frozen_checkpoint_sha256": freeze_all["checkpoint_sha256"], "datasets": {}}
    for dataset in a.datasets:
        man, data_root, ann_root, out, preds = prepared[dataset]
        recs = []
        rect = S.rectangles()[1]
        for rec in preds:
            qshape = tuple(rec["query_shape"]); c = int(rec["c"])
            truth = np.asarray(Image.open(ann_root / Path(rec["query"]).with_suffix(".png"))) == c + 1
            if truth.shape != qshape:
                raise ValueError("query mask shape mismatch: " + str((dataset, rec["e"], c)))
            k, h, w = map(int, rec["proposal_shape"])
            with np.load(out / rec["candidate_file"], allow_pickle=False) as z:
                candidates = unpack(z["proposal_query"], k, h, w)
                scores = z["scores"].astype(float).tolist()
                full = np.unpackbits(z["visual_original"])[:truth.size].reshape(truth.shape).astype(bool)
            area = candidates.reshape(k, -1).sum(1).tolist()
            canvas_arms = {}
            x, y, rw, rh = rect
            for rule, key in (("absolute_0.5_top20", "absolute_0.5"), ("absolute_0.3_top20", "absolute_0.3"),
                              ("relative_0.7_top20", "relative_0.7"), ("fallback_top1_top20", "fallback_top1")):
                idx = selected(scores, area, key)
                canvas_arms[rule] = candidates[idx].any(0) if idx else np.zeros((h, w), bool)
            iu = {}
            for arm, mask in canvas_arms.items():
                full_canvas = np.zeros((S.CANVAS, S.CANVAS), bool)
                full_canvas[y:y+rh, x:x+rw] = mask
                pred = S.to_original(full_canvas.astype(np.uint8), truth.shape[::-1])
                iu[arm] = [int((pred & truth).sum()), int((pred | truth).sum())]
            iu["full_visual_0.5"] = [int((full & truth).sum()), int((full | truth).sum())]
            recs.append(dict(fold=int(rec["fold"]), e=int(rec["e"]), c=c,
                             support=rec["support_rgb_sha256"], query=rec["query_rgb_sha256"], original_iu=iu))
        base = lambda r: r["original_iu"]["absolute_0.5_top20"]
        rows_out = {}
        for arm in ARMS:
            if arm == "absolute_0.5_top20":
                paired = S.paired(recs, lambda r:r["original_iu"][arm], base)
            else:
                paired = S.paired(recs, lambda r, a=arm:r["original_iu"][a], base)
            rows_out[arm] = dict(miou=paired["miou"], gain_vs_fixed_0p5=paired["gain"], ci95_vs_fixed_0p5=paired["ci95"],
                                 photo_groups=paired["groups"])
        rel_get = lambda r:r["original_iu"]["relative_0.7_top20"]
        direct = {}
        for control in ("absolute_0.3_top20", "fallback_top1_top20", "full_visual_0.5"):
            value = S.paired(recs, rel_get, lambda r, c=control:r["original_iu"][c])
            direct[control] = dict(gain=value["gain"], ci95=value["ci95"])
        results["datasets"][dataset] = dict(episodes=len(recs), classes=len({r["c"] for r in recs}), rows=rows_out,
            relative_0p7_direct_controls=direct,
            per_fold_gain_relative_0p7={arm:{str(f):S.paired([r for r in recs if r["fold"]==f],lambda r,a=arm:r["original_iu"][a],
                lambda r:r["original_iu"]["relative_0.7_top20"],draws=1)["gain"] for f in sorted({r["fold"] for r in recs})} for arm in ARMS})
        (out / "scored_episodes.jsonl").write_text("".join(json.dumps(dict(fold=r["fold"],e=r["e"],c=r["c"],iu=r["original_iu"],support_rgb=r["support"],query_rgb=r["query"]))+"\n" for r in recs))
    path = out_root / "external_score_report.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps({"state": results["state"], "datasets": {d:{a:{"miou":v["miou"],"vs_relative":v["gain_vs_relative_0p7"],"ci95":v["ci95_vs_relative_0p7"]} for a,v in x["rows"].items()} for d,x in results["datasets"].items()}}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("infer", "score"), required=True)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--pack-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent/transfer_v1"))
    p.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/demo9_transductive_ics/results/sam3_external_v1"))
    p.add_argument("--sam3", type=Path, default=Path("/root/autodl-tmp/sam3_preparation/code"))
    p.add_argument("--checkpoint", type=Path, default=Path("/root/autodl-tmp/sam3_preparation/checkpoints/sam3.pt"))
    p.add_argument("--checkpoint-status", type=Path, default=Path("/root/autodl-tmp/sam3_preparation/download_status.json"))
    p.add_argument("--limit-per-dataset", type=int)
    a = p.parse_args()
    infer(a) if a.mode == "infer" else score(a)


if __name__ == "__main__":
    main()
